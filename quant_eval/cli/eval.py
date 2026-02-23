"""
Evaluation entry point for quantized LLaMA models.

Quantization is handled by mase's quantize_module_transform_pass.
Config is loaded from a single TOML file.

Usage:
    python -m quant_eval.cli.eval --help

Example (baseline, no quantization):
    python -m quant_eval.cli.eval \
        --model_name meta-llama/Meta-Llama-3-8B \
        --tasks wikitext

Example (quantized):
    python -m quant_eval.cli.eval \
        --model_name meta-llama/Meta-Llama-3-8B \
        --quant_config configs/full_mxint.toml \
        --tasks wikitext
"""

from typing import Union
import time

import torch
import transformers

from quant_eval.eval.eval_utils import (
    create_experiment_log_dir,
    save_args,
    save_results,
    setup_model,
    move_to_gpu,
    print_all_layers,
)
from quant_eval.eval import evaluate_with_lm_eval, evaluate_perplexity
from quant_eval.quantize import load_quant_config
from quant_eval.utils import get_logger, set_logging_verbosity

logger = get_logger(__name__)
set_logging_verbosity("debug")


def eval_main(
    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    tasks: Union[str, list[str]] = "wikitext",
    device_id: str = "cuda:0",

    # Quantization config: single TOML file path
    quant_config: Union[str, None] = None,

    # Evaluation settings
    model_parallel: bool = False,
    enable_eval_harness: bool = False,
    seqlen: int = 2048,

    # Logging
    log_dir: Union[str, None] = None,
):
    """
    Evaluate a LLaMA model with optional MX quantization.

    Args:
        model_name: HuggingFace model ID
        tasks: lm-eval task(s) to run
        device_id: CUDA device

        quant_config: Path to a TOML config file for quantize_module_transform_pass.
                      If None, run unquantized baseline.

        model_parallel: Auto-dispatch across GPUs
        enable_eval_harness: Use lm-eval harness instead of manual PPL
        seqlen: Sequence length

        log_dir: Directory to save logs
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Tasks: {tasks}")

    quantize = quant_config is not None
    if quantize:
        print(f"Quantization config: {quant_config}")
    else:
        print("Quantization: None (baseline)")
    print("=" * 60)

    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        full_args = locals().copy()
        save_args(log_dir, full_args)
        if quant_config:
            import shutil
            shutil.copy(quant_config, log_dir / "quant_config.toml")

    transformers.set_seed(0)

    # Load model
    tokenizer, model = setup_model(
        model_name, model_parallel, dtype=torch.float16,
        device=device_id if not model_parallel else None,
    )
    model.eval()

    if quantize:
        from chop.passes.module.transforms import quantize_module_transform_pass

        pass_args = load_quant_config(quant_config)
        has_gptq = "gptq" in pass_args

        if has_gptq:
            pass_args["gptq"]["device"] = device_id

        n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
        logger.info("Quantizing %d linear layers...", n_linear)
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

    # Move to device
    if model_parallel:
        model = move_to_gpu(model, model_parallel)
    else:
        model.to(device_id)

    if quantize:
        print_all_layers(model)

    # Evaluate
    if enable_eval_harness:
        results = evaluate_with_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_length=seqlen,
            batch_size="auto",
            log_samples=False,
        )
    else:
        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_name=tasks,
            max_length=seqlen,
            verbose=True,
        )

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    if "results" in results:
        for task_name, task_results in results.get("results", {}).items():
            print(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
    else:
        for k, v in results.items():
            print(f"  {k}: {v}")

    if log_dir:
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(eval_main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
