"""
Fast-dLLM v2 lm-eval harness wrapper.

Wraps a Fast-dLLM model as an lm-eval compatible LM for benchmark evaluation.
Uses block diffusion sampling instead of autoregressive generation.

Ported from Coprocessor_for_Llama/acc_simulator/cli/dllm_sim.py
"""

import time
from typing import Union, List, Dict

import torch
import torch.nn.functional as F
from datasets import Dataset
from lm_eval import evaluator
from lm_eval.api.model import LM
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .dllm_generation import FAST_DLLM_MASK_ID, FAST_DLLM_STOP_TOKEN


class FastDLLMEvalHarness(LM):
    """
    lm-eval harness wrapper for Fast-dLLM v2.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device,
        model_name: str,
        show_speed: bool = False,
        max_new_tokens: int = 2048,
        batch_size: int = 32,
        mask_id: int = FAST_DLLM_MASK_ID,
        use_block_cache: bool = False,
        small_block_size: int = 8,
        bd_size: int = 32,
        threshold: float = 0.9,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.show_speed = show_speed
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.use_block_cache = use_block_cache
        self.small_block_size = small_block_size
        self.bd_size = bd_size
        self.threshold = threshold

        self._rank = 0
        self._world_size = 1

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.model_name

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, add_generation_prompt=add_generation_prompt, tokenize=False
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        batch[:, prompt_index.sum()] = self.mask_id
        batch = torch.cat([
            batch.to(self.device),
            torch.full(
                (b, self.bd_size - batch.shape[1] % self.bd_size),
                self.mask_id, dtype=torch.long, device=self.device,
            )
        ], dim=1)
        if batch.shape[1] > l:
            batch[:, l] = self.tokenizer.eos_token_id
        return batch

    @torch.no_grad()
    def get_logits(self, batch):
        logits = self.model(batch).logits
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        perturbed_seq = self._forward_process(seq.clone(), prompt_index)
        mask_indices = perturbed_seq == self.mask_id
        logits = self.get_logits(perturbed_seq)
        seq = torch.cat([
            seq.to(self.device),
            torch.full(
                (seq.shape[0], self.bd_size - seq.shape[1] % self.bd_size),
                -100, dtype=torch.long, device=self.device,
            )
        ], dim=1)
        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none')
        return -loss.sum().item()

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {"prefix": prefix, "target": target}

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                ll = self.get_loglikelihood(elem["prefix"], elem["target"])
                out.append((ll, 0.0))
        torch.cuda.empty_cache()
        return out

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

                if req.task_name.startswith('minerva_math'):
                    question = question.replace(
                        "Solution:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )
                elif req.task_name.startswith('gsm8k'):
                    question = question.replace(
                        "Answer:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )

                model_inputs = self.tokenizer([question], return_tensors="pt").to(self.device)
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])

            batched_input_ids = [
                torch.cat([
                    input_ids,
                    torch.full(
                        (1, max_len - input_ids.shape[1]),
                        self.mask_id, dtype=torch.long, device=self.device,
                    )
                ], dim=1)
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)

            with torch.no_grad():
                generated_ids = self.model.mdm_sample(
                    batched_input_ids,
                    tokenizer=self.tokenizer,
                    block_size=self.bd_size,
                    small_block_size=self.small_block_size,
                    max_new_tokens=self.max_new_tokens,
                    mask_id=self.mask_id,
                    min_len=int(min_len),
                    seq_len=torch.tensor(seq_len, device=self.device),
                    use_block_cache=self.use_block_cache,
                    threshold=self.threshold,
                )

            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_answer = self.tokenizer.decode(
                    generated_ids[batch_pos][seq_len[batch_pos]:],
                    skip_special_tokens=True,
                )

                if self.show_speed:
                    num_tokens += (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum()

                output[orig_idx] = generated_answer

        if self.show_speed:
            elapsed = time.time() - start_time
            print(f"Total tokens: {num_tokens}, Time: {elapsed:.2f}s, Tokens/s: {num_tokens / elapsed:.2f}")

        return output


def evaluate_dllm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tasks: Union[str, List[str]],
    device,
    model_name: str,
    batch_size: int = 32,
    max_new_tokens: int = 2048,
    num_fewshot: int = 0,
    mask_id: int = FAST_DLLM_MASK_ID,
    bd_size: int = 32,
    small_block_size: int = 8,
    threshold: float = 1.0,
    show_speed: bool = True,
) -> Dict:
    """
    Evaluate a Fast-dLLM model using lm-eval harness.

    Args:
        model: Pre-loaded (optionally quantized) HuggingFace model with mdm_sample attached.
        tokenizer: Corresponding tokenizer.
        tasks: lm-eval task(s) to run.
        device: CUDA device.
        model_name: Model name for lm-eval.
        batch_size: Batch size for evaluation.
        max_new_tokens: Max tokens to generate.
        num_fewshot: Number of few-shot examples.
        mask_id: Mask token ID for dLLM.
        bd_size: Block diffusion size.
        small_block_size: Sub-block size.
        threshold: Unmasking threshold.
        show_speed: Show throughput metrics.

    Returns:
        Dictionary of evaluation results.
    """
    lm = FastDLLMEvalHarness(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        show_speed=show_speed,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        mask_id=mask_id,
        use_block_cache=False,
        small_block_size=small_block_size,
        bd_size=bd_size,
        threshold=threshold,
    )

    task_list = [tasks] if isinstance(tasks, str) else tasks
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        apply_chat_template=True,
    )

    return results
