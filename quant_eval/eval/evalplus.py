"""Evaluate code generation with evalplus (HumanEval / MBPP).

Wraps a pre-loaded HuggingFace model so that evalplus can use it without
re-loading from the Hub. Ported from ``Plena-Acc-Sim/eval/eval_evalplus.py``
so quant_eval can run HumanEval+/MBPP+ against MX-quantized models the
same way ``lm_eval.py`` runs lm-eval-harness tasks.

Usage::

    from quant_eval.eval.evalplus import evaluate_with_evalplus

    results = evaluate_with_evalplus(
        model=model,
        tokenizer=tokenizer,
        dataset="humaneval",
        batch_size=1,
    )

Requires ``evalplus`` and ``stop_sequencer`` to be installed
(``uv pip install evalplus stop-sequencer``).
"""

import gc
import json
import os
import tempfile
from typing import Dict, List, Optional

import torch
from stop_sequencer import StopSequencer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from evalplus.codegen import codegen
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.evaluate import evaluate as evalplus_evaluate
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)
from evalplus.sanitize import sanitize


class _PerSequenceTokenStoppingCriteria(StoppingCriteria):
    """Stop each batch row independently once a tokenized delimiter appears."""

    def __init__(self, stop_sequences: List[List[int]], input_length: int):
        self.stop_sequences = [sequence for sequence in stop_sequences if sequence]
        self.input_length = int(input_length)

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        del scores, kwargs
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        generated_length = input_ids.shape[1] - self.input_length
        for sequence in self.stop_sequences:
            if generated_length < len(sequence):
                continue
            suffix = torch.as_tensor(sequence, device=input_ids.device)
            done |= torch.all(input_ids[:, -len(sequence) :] == suffix, dim=1)
        return done


class PreloadedHFDecoder(DecoderBase):
    """evalplus decoder that wraps an already-loaded HF model."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: str,
        force_base_prompt: bool = False,
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(name="preloaded-hf", **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.skip_special_tokens = True
        self.force_base_prompt = force_base_prompt
        self.enable_thinking = enable_thinking
        # DecoderBase points at EvalPlus's module-level EOS list.
        self.eos = list(self.eos)

        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    def _format_prompt(self, prompt: str) -> str:
        if self.is_direct_completion():
            return prompt
        if self.enable_thinking:
            user_prompt = (
                f"{self.instruction_prefix}\n"
                f"```\n{prompt.strip()}\n```\n"
            )
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        return make_raw_chat_prompt(
            prompt,
            self.instruction_prefix,
            self.response_prefix,
            self.tokenizer,
        )

    def _trim_outputs(self, outputs: List[str]) -> List[str]:
        results = []
        for output in outputs:
            min_index = len(output)
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            results.append(output[:min_index].replace("\t", "    "))
        return results

    def prompt_token_length(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt)))

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = self._format_prompt(prompt)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["top_k"] = 20
            kwargs["temperature"] = self.temperature

        stop_sequencer = StopSequencer(
            self.model, model_type="causal", tokenizer=self.tokenizer
        )
        orig_get_stopping_criteria = self.model._get_stopping_criteria
        model = stop_sequencer.register_stop_texts(
            stop_texts=self.eos,
            input_length=input_tokens.size(-1),
        )

        try:
            outputs = model.generate(
                input_tokens,
                attention_mask=torch.ones_like(input_tokens),
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=min(self.batch_size, num_samples),
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        finally:
            # Prevent nested wrapping accumulation, including after an OOM.
            self.model._get_stopping_criteria = orig_get_stopping_criteria

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )

        return self._trim_outputs(gen_strs)

    @torch.inference_mode()
    def codegen_batch(self, prompts: List[str]) -> List[str]:
        """Greedily generate one completion for each independent prompt."""
        if not prompts:
            return []

        formatted = [self._format_prompt(prompt) for prompt in prompts]
        encoded = [self.tokenizer.encode(prompt) for prompt in formatted]
        input_length = max(len(tokens) for tokens in encoded)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id for batching.")

        input_ids = torch.full(
            (len(encoded), input_length),
            int(pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, tokens in enumerate(encoded):
            width = len(tokens)
            input_ids[row, -width:] = torch.as_tensor(tokens, device=self.device)
            attention_mask[row, -width:] = 1

        stop_sequences = [
            self.tokenizer.encode(text, add_special_tokens=False)
            for text in self.eos
        ]
        stopping_criteria = StoppingCriteriaList(
            [_PerSequenceTokenStoppingCriteria(stop_sequences, input_length)]
        )
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=int(pad_token_id),
            stopping_criteria=stopping_criteria,
        )
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=self.skip_special_tokens,
        )
        return self._trim_outputs(decoded)


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
    )


def _batched_codegen(
    *,
    target_path: str,
    decoder: PreloadedHFDecoder,
    dataset: str,
    batch_size: int,
    version: str,
    limit: Optional[int],
    resume: bool,
    length_bucket: bool,
    adaptive_batch: bool,
) -> Dict:
    """EvalPlus-compatible greedy generation across multiple task prompts."""
    if dataset == "humaneval":
        tasks = get_human_eval_plus(version=version)
    elif dataset == "mbpp":
        tasks = get_mbpp_plus(version=version)
    else:
        raise ValueError(f"Batched generation does not support dataset {dataset!r}.")

    raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    if not resume:
        for path in (target_path, raw_target_path):
            if os.path.isfile(path):
                os.unlink(path)

    completed = set()
    if resume and os.path.isfile(target_path):
        with open(target_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["task_id"])

    selected = []
    for task_id, task in tasks.items():
        if limit is not None and int(task_id.split("/")[1]) >= int(limit):
            continue
        if task_id not in completed:
            prompt = task["prompt"].strip() + "\n"
            selected.append((task_id, task, prompt))

    if length_bucket:
        selected.sort(key=lambda item: decoder.prompt_token_length(item[2]))

    print(f"Sanitized code outputs will be saved to {target_path}", flush=True)
    print(f"Raw outputs will be saved to {raw_target_path}", flush=True)
    print(
        f"Batched EvalPlus generation: pending={len(selected)}, "
        f"batch_size={batch_size}, length_bucket={length_bucket}",
        flush=True,
    )

    current_batch_size = max(1, int(batch_size))
    minimum_batch_size = current_batch_size
    generated = 0
    cursor = 0
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "a", encoding="utf-8") as clean_handle, open(
        raw_target_path, "a", encoding="utf-8"
    ) as raw_handle:
        while cursor < len(selected):
            batch = selected[cursor : cursor + current_batch_size]
            task_ids = [item[0] for item in batch]
            prompts = [item[2] for item in batch]
            print(
                f"Codegen batch {cursor + 1}-{cursor + len(batch)}/{len(selected)} "
                f"({task_ids[0]}..{task_ids[-1]}, size={len(batch)})",
                flush=True,
            )
            try:
                outputs = decoder.codegen_batch(prompts)
            except Exception as error:
                if not (_is_cuda_oom(error) and adaptive_batch and len(batch) > 1):
                    raise
                current_batch_size = max(1, len(batch) // 2)
                minimum_batch_size = min(minimum_batch_size, current_batch_size)
                print(
                    f"CUDA OOM at batch size {len(batch)}; retrying with "
                    f"batch size {current_batch_size}.",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
                continue

            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Batched decoder returned {len(outputs)} outputs for "
                    f"{len(batch)} prompts."
                )
            for (task_id, task, prompt), output in zip(batch, outputs):
                solution = prompt + output if decoder.is_direct_completion() else output
                sanitized = sanitize(solution, entrypoint=task["entry_point"])
                clean_handle.write(
                    json.dumps({"task_id": task_id, "solution": sanitized}) + "\n"
                )
                raw_handle.write(
                    json.dumps({"task_id": task_id, "solution": solution}) + "\n"
                )
            clean_handle.flush()
            raw_handle.flush()
            minimum_batch_size = min(minimum_batch_size, len(batch))
            generated += len(batch)
            cursor += len(batch)

    return {
        "resumed_count": len(completed),
        "new_generated_count": generated,
        "requested_batch_size": int(batch_size),
        "minimum_batch_size": minimum_batch_size,
        "length_bucket": bool(length_bucket),
    }


def evaluate_with_evalplus(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: str = "humaneval",
    batch_size: int = 1,
    greedy: bool = False,
    n_samples: int = 1,
    max_new_tokens: int = 32768,
    output_dir: Optional[str] = None,
    parallel: Optional[int] = None,
    base_only: bool = False,
    version: str = "default",
    overwrite: bool = False,
    limit: Optional[int] = None,
    enable_thinking: bool = True,
    batch_length_bucket: bool = True,
    adaptive_batch: bool = True,
) -> Dict:
    """Generate code and evaluate with evalplus.

    Args:
        model: Pre-loaded HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        dataset: "humaneval" or "mbpp".
        batch_size: Number of independent tasks generated together in greedy
            mode. Sampling mode retains EvalPlus's samples-per-prompt behavior.
        greedy: Greedy decoding (forces temperature=0, n_samples=1).
        n_samples: Samples per task (ignored if greedy=True).
        max_new_tokens: Max tokens to generate per sample.
        output_dir: Directory to save generated code. Uses a temp dir if None.
        parallel: Number of workers for code evaluation.
        base_only: Only run base tests (skip plus tests).
        version: Dataset version.
        overwrite: If True, regenerate even if a previous jsonl exists.
        limit: Generate only the first N tasks. Partial generation deliberately
            skips EvalPlus execution because its scorer requires a complete
            dataset; this is intended for smoke tests only.
        enable_thinking: Let chat-template models generate their native thinking
            prefix instead of pre-filling an empty thinking block.
        batch_length_bucket: Group similarly sized prompts before greedy generation.
        adaptive_batch: Halve the greedy task batch and retry after CUDA OOM.

    Returns:
        Dictionary with evaluation results.
    """
    if greedy:
        temperature = 0.0
        n_samples = 1
    else:
        temperature = 0.6

    instruction_prefix = (
        "Please provide a self-contained Python script that solves the following "
        "problem in a markdown code block:"
    )
    response_prefix = (
        "Below is a Python script with a self-contained function that solves the "
        "problem and passes corresponding tests:"
    )

    decoder = PreloadedHFDecoder(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=1 if greedy else batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        enable_thinking=enable_thinking,
    )

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="evalplus_")
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    target_path = os.path.join(
        output_dir, dataset, f"preloaded-hf_temp_{temperature}.jsonl"
    )

    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive when set.")

    batch_stats = None
    if greedy and int(batch_size) > 1:
        batch_stats = _batched_codegen(
            target_path=target_path,
            decoder=decoder,
            dataset=dataset,
            batch_size=int(batch_size),
            version=version,
            limit=limit,
            resume=not overwrite,
            length_bucket=batch_length_bucket,
            adaptive_batch=adaptive_batch,
        )
    else:
        codegen(
            target_path=target_path,
            model=decoder,
            dataset=dataset,
            greedy=greedy,
            n_samples=n_samples,
            id_range=(0, int(limit)) if limit is not None else None,
            version=version,
            resume=not overwrite,
        )

    generated_task_ids = set()
    if os.path.isfile(target_path):
        with open(target_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    generated_task_ids.add(json.loads(line)["task_id"])

    if limit is not None:
        partial_results = {
            "samples_path": target_path,
            "generated_count": len(generated_task_ids),
            "expected_count": int(limit),
            "partial": True,
            "scored": False,
        }
        if batch_stats is not None:
            partial_results["batch_stats"] = batch_stats
        return partial_results

    evalplus_evaluate(
        dataset=dataset,
        samples=target_path,
        parallel=parallel,
        base_only=base_only,
        version=version,
    )

    result_path = target_path.replace(".jsonl", "_eval_results.json")
    if os.path.isfile(result_path):
        with open(result_path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {"samples_path": target_path}

    results.setdefault("samples_path", target_path)
    results["generated_count"] = len(generated_task_ids)
    results["partial"] = False
    results["scored"] = True
    if batch_stats is not None:
        results["batch_stats"] = batch_stats
    return results


def summarize_evalplus_results(results: Dict) -> Dict:
    """Return deterministic pass@1 counts from EvalPlus per-task records."""
    evaluations = results.get("eval", {})
    if not isinstance(evaluations, dict):
        evaluations = {}
    base_correct = 0
    plus_correct = 0
    for records in evaluations.values():
        record = records[0] if isinstance(records, list) and records else records
        if not isinstance(record, dict):
            continue
        base_pass = record.get("base_status") == "pass"
        plus_pass = record.get("plus_status") == "pass"
        base_correct += int(base_pass)
        plus_correct += int(base_pass and plus_pass)
    total = len(evaluations)
    return {
        "total": total,
        "base_correct": base_correct,
        "plus_correct": plus_correct,
        "humaneval_pass_at_1": base_correct / total if total else None,
        "humaneval_plus_pass_at_1": plus_correct / total if total else None,
    }
