"""
lm-eval harness evaluation with optional MASE quantization.

Usage:
    python -m quant_eval.cli.eval_lm --help

Example (baseline):
    python -m quant_eval.cli.eval_lm \
        --model_name Qwen/Qwen3-30B-A3B \
        --tasks wikitext

Example (quantized):
    python -m quant_eval.cli.eval_lm \
        --model_name Qwen/Qwen3-30B-A3B \
        --quant_config quant_eval/configs/qwen3_moe_mxint4.toml \
        --tasks wikitext,mmlu
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
    print_all_layers,
    create_experiment_log_dir,
    save_args,
    save_results,
)
from quant_eval.eval import evaluate_with_lm_eval
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")


def main(
    model_name: str = "Qwen/Qwen3-30B-A3B",
    tasks: Union[str, list[str]] = "wikitext",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: Union[str, None] = None,
    model_parallel: bool = False,
    seqlen: int = 2048,
    batch_size: Union[int, str] = "auto",
    log_dir: Union[str, None] = None,
):
    """
    Evaluate a model using lm-eval harness with optional MX quantization.

    Args:
        model_name: HuggingFace model ID.
        tasks: lm-eval task(s) to run (comma-separated or list).
        device_id: CUDA device.
        dtype: Model dtype (float16, bfloat16, float32).
        quant_config: Path to TOML config for quantize_module_transform_pass.
                      If None, run unquantized baseline.
        model_parallel: Auto-dispatch across GPUs.
        seqlen: Maximum sequence length.
        batch_size: Evaluation batch size.
        log_dir: Directory to save logs.
    """
    print("=" * 60)
    print("lm-eval Harness Evaluation")
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
        save_args(log_dir, locals().copy())
        if quant_config:
            import shutil

            shutil.copy(quant_config, log_dir / "quant_config.toml")

    transformers.set_seed(0)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
    )
    model.eval()

    if quantize:
        from chop.passes.module.transforms import quantize_module_transform_pass

        pass_args = load_quant_config(quant_config)
        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id

        n_linear = sum(
            1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)
        )
        logger.info("Quantizing %d linear layers...", n_linear)
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

    if model_parallel:
        model = move_to_gpu(model, model_parallel)
    else:
        model.to(device_id)

    if quantize:
        print_all_layers(model)

    results = evaluate_with_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        max_length=seqlen,
        batch_size=batch_size,
        log_samples=False,
    )

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
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
