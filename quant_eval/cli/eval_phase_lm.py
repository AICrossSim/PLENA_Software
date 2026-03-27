"""
lm-eval harness with phase-dependent MX quantization.

Uses PhaseAutoSwitch hook to automatically detect prefill vs decode
from input sequence length and swap activation precision accordingly.
Transparent to lm-eval — no modifications to the harness needed.

Usage:
    python -m quant_eval.cli.eval_phase_lm --help

Example:
    python -m quant_eval.cli.eval_phase_lm \
        --model_name Qwen/Qwen2.5-1.5B \
        --quant_config quant_eval/configs/llama_mxint4.toml \
        --prefill_data_in_width 4 \
        --decode_data_in_width 8 \
        --tasks wikitext
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
from quant_eval.eval.eval_harness import evaluate_with_lm_eval
from quant_eval.eval.phase_quant import PhaseAutoSwitch
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")


def main(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    tasks: Union[str, list[str]] = "wikitext",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: str = "quant_eval/configs/llama_mxint4.toml",
    model_parallel: bool = False,
    seqlen: int = 2048,
    batch_size: Union[int, str] = "auto",
    # Phase-dependent activation precision
    prefill_data_in_width: int = 4,
    decode_data_in_width: int = 8,
    prefill_data_in_block_size: int = 32,
    decode_data_in_block_size: int = 32,
    log_dir: Union[str, None] = None,
):
    """
    Run lm-eval harness with phase-dependent MX quantization.

    A forward pre-hook on the model automatically detects prefill
    (seq_len > 1) vs decode (seq_len == 1) and swaps activation
    quantization precision accordingly.

    Args:
        model_name: HuggingFace model ID.
        tasks: lm-eval task(s) to run.
        device_id: CUDA device.
        dtype: Model dtype.
        quant_config: TOML config for base quantization (sets weight precision).
        model_parallel: Auto-dispatch across GPUs.
        seqlen: Maximum sequence length.
        batch_size: Evaluation batch size.
        prefill_data_in_width: Activation bit-width during prefill.
        decode_data_in_width: Activation bit-width during decode.
        prefill_data_in_block_size: Activation block size during prefill.
        decode_data_in_block_size: Activation block size during decode.
        log_dir: Directory to save logs.
    """
    phase_configs = {
        "prefill": {
            "data_in_width": prefill_data_in_width,
            "data_in_block_size": prefill_data_in_block_size,
        },
        "decode": {
            "data_in_width": decode_data_in_width,
            "data_in_block_size": decode_data_in_block_size,
        },
    }

    print("=" * 60)
    print("lm-eval Harness — Phase-Dependent Quantization")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"Weight quantization: {quant_config}")
    print(f"Prefill activations: MXInt{prefill_data_in_width} (block_size={prefill_data_in_block_size})")
    print(f"Decode activations:  MXInt{decode_data_in_width} (block_size={decode_data_in_block_size})")
    print("=" * 60)

    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        save_args(log_dir, locals().copy())
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

    # Apply base quantization
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

    # Enable auto phase switching — this is the key part.
    # The hook detects seq_len > 1 → prefill, seq_len == 1 → decode
    # and swaps MX layer configs transparently before each forward pass.
    switch = PhaseAutoSwitch(model, phase_configs)
    switch.enable()
    logger.info("Phase auto-switch enabled: prefill=MXInt%d, decode=MXInt%d",
                prefill_data_in_width, decode_data_in_width)

    # Run lm-eval — it knows nothing about phases, the hook handles it
    results = evaluate_with_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        max_length=seqlen,
        batch_size=batch_size,
        log_samples=False,
    )

    switch.disable()

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Prefill: MXInt{prefill_data_in_width}")
    print(f"  Decode:  MXInt{decode_data_in_width}")
    if "results" in results:
        for task_name, task_results in results.get("results", {}).items():
            print(f"\n  {task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
    else:
        for k, v in results.items():
            print(f"  {k}: {v}")

    if log_dir:
        # Add phase config to results
        results["phase_configs"] = phase_configs
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
