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

import os
import tempfile
from typing import Dict, List, Optional

import torch
from stop_sequencer import StopSequencer
from transformers import PreTrainedModel, PreTrainedTokenizer

from evalplus.codegen import codegen
from evalplus.evaluate import evaluate as evalplus_evaluate
from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class PreloadedHFDecoder(DecoderBase):
    """evalplus decoder that wraps an already-loaded HF model."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: str,
        force_base_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(name="preloaded-hf", **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.skip_special_tokens = True
        self.force_base_prompt = force_base_prompt

        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt,
                self.instruction_prefix,
                self.response_prefix,
                self.tokenizer,
            )
        )
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

        outputs = model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        # Restore original method to prevent nested wrapping accumulation.
        self.model._get_stopping_criteria = orig_get_stopping_criteria

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )

        results = []
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            results.append(output[:min_index].replace("\t", "    "))
        return results


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
) -> Dict:
    """Generate code and evaluate with evalplus.

    Args:
        model: Pre-loaded HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        dataset: "humaneval" or "mbpp".
        batch_size: Batch size for generation.
        greedy: Greedy decoding (forces temperature=0, n_samples=1).
        n_samples: Samples per task (ignored if greedy=True).
        max_new_tokens: Max tokens to generate per sample.
        output_dir: Directory to save generated code. Uses a temp dir if None.
        parallel: Number of workers for code evaluation.
        base_only: Only run base tests (skip plus tests).
        version: Dataset version.
        overwrite: If True, regenerate even if a previous jsonl exists.

    Returns:
        Dictionary with evaluation results.
    """
    if greedy:
        temperature = 0.0
        batch_size = 1
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
        batch_size=batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="evalplus_")
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    target_path = os.path.join(
        output_dir, dataset, f"preloaded-hf_temp_{temperature}.jsonl"
    )

    codegen(
        target_path=target_path,
        model=decoder,
        dataset=dataset,
        greedy=greedy,
        n_samples=n_samples,
        version=version,
        resume=not overwrite,
    )

    evalplus_evaluate(
        dataset=dataset,
        samples=target_path,
        parallel=parallel,
        base_only=base_only,
        version=version,
    )

    result_path = target_path.replace(".jsonl", "_eval_results.json")
    if os.path.isfile(result_path):
        import json
        with open(result_path) as f:
            results = json.load(f)
    else:
        results = {"samples_path": target_path}

    return results
