# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA
# Modified from Fast-dLLM: https://github.com/NVlabs/Fast-dLLM

"""
LLaDA evaluation harness with Fast-dLLM v1 KV cache acceleration.

Registers a "llada_dist" model with lm-eval. Supports optional MASE
quantization via the `quant_config` model arg.

Usage:
    python -m quant_eval.eval.eval_llada --tasks gsm8k --num_fewshot 0 \
        --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True

    # With MASE quantization:
    python -m quant_eval.eval.eval_llada --tasks gsm8k --num_fewshot 0 \
        --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True,quant_config='configs/kv_only_mxint4.toml'
"""

import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import json
import time
import logging

from quant_eval.eval.dllm_v2.dllm_generation import setup_dllm_generation
from quant_eval.quantize import load_quant_config
from quant_eval.utils import get_logger, move_to_gpu, set_logging_verbosity, setup_model

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
set_logging_verbosity("debug")


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        mask_id=151665,
        max_length=4096,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        quant_config=None,
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path

        # ---------------------------------------------------------------
        # FORCE CORRECT PARAMETERS FOR FAST-DLLM V2 (QWEN-BASED)
        # ---------------------------------------------------------------
        is_fast_dllm = "fast_dllm" in model_path.lower() or "qwen" in model_path.lower()

        if is_fast_dllm:
            self.mask_id = 151665  # Official Fast-dLLM v2 mask token
            self.steps = 256  # CRITICAL: Prevents 1024-step bottleneck
            self.threshold = 0.8  # Balanced for math reasoning & parallel decoding
            self.block_length = max(int(block_length), 16)
            logger.info(
                f"🔧 Fast-dLLM v2 detected. Forcing: mask_id={self.mask_id}, steps={self.steps}, threshold={self.threshold}, block_length={self.block_length}"
            )
        else:
            self.mask_id = 126336  # LLaDA mask token
            self.steps = 256
            self.threshold = 0.0 if threshold is None else float(threshold)
            self.block_length = max(int(block_length), 32)
        # ---------------------------------------------------------------

        # `lm-eval` passes `device` and `batch_size` via kwargs (from --model_args).
        # We pop them here to explicitly control their usage and avoid conflicts.
        device = kwargs.pop("device", "cuda")
        batch_size = kwargs.pop("batch_size", 32)

        self.accelerator = None
        self.model_path = model_path

        self.tokenizer, self.model = setup_model(
            model_path,
            model_parallel=False,
            dtype=torch.bfloat16,
            device=device,
            attn_implementation="eager" if quant_config is not None else "sdpa",
        )
        self.model.eval()
        detected_mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if detected_mask_id is not None:
            self.mask_id = detected_mask_id
            logger.info(f"Auto-detected mask_token_id: {self.mask_id} from tokenizer")
        else:
            self.mask_id = mask_id  # Fallback to CLI/default
            logger.warning(
                f"Tokenizer lacks mask_token_id. Using fallback: {self.mask_id}"
            )

        if quant_config is not None:
            from chop.passes.module.transforms import quantize_module_transform_pass

            pass_args = load_quant_config(quant_config)
            if "gptq" in pass_args:
                pass_args["gptq"]["device"] = device
            if "rotation_search" in pass_args:
                pass_args["rotation_search"]["device"] = device
                pass_args["rotation_search"].setdefault("model_name", model_path)
            self.model, _ = quantize_module_transform_pass(self.model, pass_args)

        self.device = torch.device(device)
        self.model = move_to_gpu(self.model, model_parallel=False)
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)

        setup_dllm_generation(self.model)

        self._rank = 0
        self._world_size = 1

        self.mask_id = mask_id
        self.batch_size = int(batch_size)
        self.mc_num = max(self.batch_size, 1)
        self.max_length = max_length
        self.cfg = 0.0
        self.is_check_greedy = False

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.factor = None
        self.is_instruct = True if "instruct" in model_path.lower() else False
        self.dual_cache = False

    @property
    def tokenizer_name(self):
        # Required by lm-eval for chat template handling
        return getattr(self, "_tokenizer_name", None) or self.model_path

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        """
        Required by lm-eval for models that support chat formatting.
        """
        # lm-eval may pass a string directly in some versions/tasks
        if isinstance(chat_history, str):
            return chat_history

        # Use the tokenizer's native chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        # Fallback if the tokenizer doesn't support it (just concatenate)
        return "".join([msg.get("content", "") for msg in chat_history])

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
        is_mask = torch.cat(
            (
                torch.zeros(
                    b, prompt_index.sum(), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])
        logits = self.model(batch).logits
        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False
        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix
        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = [None] * len(requests)
        num_tokens = 0
        start_time = time.time()

        requests_with_indices = [(i, req) for i, req in enumerate(requests)]
        requests_with_indices.sort(key=lambda x: len(x[1].args[0]))

        batched_requests = []
        current_batch = []
        for i, req in requests_with_indices:
            current_batch.append((i, req))
            if len(current_batch) == self.batch_size:
                batched_requests.append(current_batch)
                current_batch = []
        if current_batch:
            batched_requests.append(current_batch)

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            min_len = 1e9
            seq_len = []

            for orig_idx, req in batch:
                question = req.args[0]
                logger.debug(f"DEBUG: Input question: {repr(question)}")

                if req.task_name.startswith("minerva_math"):
                    question = question.replace(
                        "Solution:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )
                elif req.task_name.startswith("gsm8k"):
                    question = question.replace(
                        "Answer:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:",
                    )

                model_inputs = self.tokenizer([question], return_tensors="pt").to(
                    self.device
                )
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])

            batched_input_ids = [
                torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (1, max_len - input_ids.shape[1]),
                            self.mask_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)

            with torch.no_grad():
                generated_ids = self.model.mdm_sample(
                    batched_input_ids,
                    tokenizer=self.tokenizer,
                    block_size=self.block_length,
                    small_block_size=self.block_length // 4,
                    max_new_tokens=self.gen_length,
                    mask_id=self.mask_id,
                    min_len=int(min_len),
                    seq_len=torch.tensor(seq_len, device=self.device),
                    use_block_cache=self.use_cache,
                    threshold=self.threshold,
                )

            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_answer = self.tokenizer.decode(
                    generated_ids[batch_pos][seq_len[batch_pos] :],
                    skip_special_tokens=False,
                )

                logger.info(f"Q: {req.args[0][:60].strip()}...")
                logger.info(f"A: {generated_answer}\n" + "-" * 50)

                if self.show_speed:
                    num_tokens += (
                        generated_ids[batch_pos][seq_len[batch_pos] :] != self.mask_id
                    ).sum()

                output[orig_idx] = generated_answer

        if self.show_speed:
            elapsed = time.time() - start_time
            print(
                f"Total tokens: {num_tokens}, Time: {elapsed:.2f}s, Tokens/s: {num_tokens / elapsed:.2f}"
            )

        return output


if __name__ == "__main__":
    cli_evaluate()
