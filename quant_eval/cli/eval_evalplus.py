"""
Evalplus (HumanEval+/MBPP+) driver for an MX-quantized HuggingFace model.

Mirrors ``eval_lm.py`` but routes the evaluation through
``quant_eval.eval.evalplus.evaluate_with_evalplus`` instead of lm-eval-harness.
Use this to score code-generation pass@1 under a single fixed-precision
quantization profile.

Example:
    python -m quant_eval.cli.eval_evalplus \\
        --model_name Qwen/Qwen3-8B \\
        --quant_config quant_eval/configs/qwen3_mxint16_rotate.toml \\
        --dataset humaneval \\
        --evalplus_output_dir logs/evalplus/qwen3_mxint16_rotate

Requires ``evalplus`` and ``stop_sequencer`` to be installed
(``uv pip install evalplus stop-sequencer``).
"""

from typing import Union
import time

import torch
import transformers

from quant_eval.utils import (
    get_logger,
    set_logging_verbosity,
    setup_model,
    move_to_gpu,
    create_experiment_log_dir,
    save_args,
    save_results,
)
from quant_eval.eval.evalplus import evaluate_with_evalplus
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")


def main(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    dataset: str = "humaneval",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: Union[str, None] = "quant_eval/configs/llama_mxint4.toml",
    model_parallel: bool = False,
    batch_size: int = 1,
    greedy: bool = False,
    n_samples: int = 1,
    max_new_tokens: int = 4096,
    evalplus_output_dir: Union[str, None] = None,
    overwrite: bool = False,
    base_only: bool = False,
    parallel: Union[int, None] = None,
    version: str = "default",
    log_dir: Union[str, None] = None,
):
    """
    Run evalplus (HumanEval+/MBPP+) against an MX-quantized HF model with a
    single fixed activation precision profile.

    Args:
        model_name:    HuggingFace model ID.
        dataset:       "humaneval" or "mbpp".
        device_id:     CUDA device string.
        dtype:         Model dtype (float16 / bfloat16 / float32).
        quant_config:  TOML config for module-level quantization. Set the
                       per-pattern ``name`` to ``"mxint_rotate"`` to enable
                       online Hadamard rotation. ``None`` = no quantization.
        model_parallel: Use HF device_map="auto" pipeline parallel.
        batch_size:    Generation batch size (samples per task per call).
        greedy:        Greedy decoding (forces temperature=0, n_samples=1).
        n_samples:     Samples per task (ignored if greedy=True).
        max_new_tokens: Max tokens to generate per sample.
        evalplus_output_dir: Where to dump generated jsonl + eval results.
        overwrite:     Regenerate even if a previous jsonl exists.
        base_only:     Only run base tests (skip plus tests).
        parallel:      Workers for evalplus's code-execution stage.
        version:       evalplus dataset version.
        log_dir:       Directory for experiment logs and results.
    """
    if dataset not in ("humaneval", "mbpp"):
        raise ValueError(f"dataset must be 'humaneval' or 'mbpp', got {dataset!r}")

    print("=" * 64)
    print("evalplus — fixed activation precision (no phase switch)")
    print("=" * 64)
    print(f"  Model  : {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Weights: {quant_config or 'none (fp)'}")
    print(f"  Greedy : {greedy}  (n_samples={n_samples}, batch_size={batch_size})")
    print("=" * 64)

    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        save_args(log_dir, locals().copy())
        if quant_config:
            import shutil
            shutil.copy(quant_config, log_dir / "quant_config.toml")

    transformers.set_seed(0)

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Quantized attention modules (MXInt / *Rotate) replace the eager forward
    # path and assert _attn_implementation == "eager". Force eager whenever a
    # quant_config is supplied.
    attn_impl = "eager" if quant_config else "sdpa"

    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
        attn_implementation=attn_impl,
    )
    model.eval()

    # ``token_collector`` is a side-effect pass that attaches a hook; if it's
    # the *only* thing in the TOML this run is calibration mode and we skip
    # module quantization entirely.
    collector_info = None
    pass_args = load_quant_config(quant_config) if quant_config else None

    if pass_args and "token_collector" in pass_args:
        from chop.passes.module.transforms import attach_token_collector_pass

        tc_cfg = pass_args.pop("token_collector")
        logger.info("Attaching TokenCollector: %s", tc_cfg)
        model.to(device_id)
        model, collector_info = attach_token_collector_pass(model, tc_cfg)

    has_quant = pass_args is not None and (
        "gptq" in pass_args
        or any(k != "by" for k in pass_args.keys())
    )
    if has_quant:
        from chop.passes.module.transforms import quantize_module_transform_pass

        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id

        n_linear = sum(
            1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)
        )
        logger.info("Quantizing %d linear layers...", n_linear)
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

        from collections import Counter
        cls_count = Counter(
            type(m).__name__ for _, m in model.named_modules()
            if "MX" in type(m).__name__
        )
        logger.info(
            "Post-quant module classes:\n%s",
            "\n".join(f"  {c}: {n}" for c, n in cls_count.most_common()),
        )

    if model_parallel:
        model = move_to_gpu(model, model_parallel)
    else:
        model.to(device_id)

    # In calibration-only mode, the hook raises CollectorFull from inside
    # forward once the buffer is full — catch it so eval exits cleanly with
    # the calibration file already saved.
    from chop.passes.module.transforms.gptq import CollectorFull
    try:
        results = evaluate_with_evalplus(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            greedy=greedy,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            output_dir=evalplus_output_dir,
            parallel=parallel,
            base_only=base_only,
            version=version,
            overwrite=overwrite,
        )
    except CollectorFull as e:
        logger.info("[calibration mode] aborted evalplus as planned: %s", e)
        results = {"calibration_only": True}

    if collector_info is not None and not collector_info["collector"].complete:
        collector_info["collector"].finalize()

    print("\n" + "=" * 64)
    print("Results:")
    print("=" * 64)
    if isinstance(results, dict):
        # evalplus stores per-task pass@k under "pass_at_k"; surface anything
        # numeric so the log is useful even if the schema shifts.
        if "pass_at_k" in results:
            for split, metrics in results["pass_at_k"].items():
                print(f"  {split}:")
                for k, v in metrics.items():
                    print(f"    {k}: {v}")
        else:
            for k, v in results.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")

    if log_dir:
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
