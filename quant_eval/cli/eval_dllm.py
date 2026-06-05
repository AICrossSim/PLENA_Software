"""
Fast-dLLM v2 (block-diffusion language model) evaluation with optional MX
quantization.

Evaluates diffusion-based language models via lm-eval-harness, using
block-diffusion sampling instead of standard autoregressive decoding.
Quantization is applied via the same TOML-config interface as the rest of
the toolkit.

Example — baseline:

    python -m quant_eval.cli.eval_dllm \\
        --model_name Efficient-Large-Model/Fast_dLLM_v2_1.5B \\
        --tasks gsm8k

Example — quantized:

    python -m quant_eval.cli.eval_dllm \\
        --model_name Efficient-Large-Model/Fast_dLLM_v2_1.5B \\
        --quant_config quant_eval/configs/llama_mxint4.toml \\
        --tasks gsm8k
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
from quant_eval.eval.dllm_v2.dllm_generation import setup_dllm_generation
from quant_eval.eval.dllm_v2.eval_dllm import evaluate_dllm
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("info")


def main(
    model_name: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
    tasks: Union[str, list[str]] = "gsm8k",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: Union[str, None] = None,
    model_parallel: bool = False,
    # dLLM specific
    batch_size: int = 32,
    max_new_tokens: int = 2048,
    num_fewshot: int = 0,
    mask_id: int = 151665,
    bd_size: int = 32,
    small_block_size: int = 8,
    threshold: float = 0.9,
    use_block_cache: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.95,
    show_speed: bool = True,
    log_dir: Union[str, None] = None,
):
    """
    Evaluate a Fast-dLLM v2 model with optional MX quantization.

    Decoding is block-diffusion: ``bd_size`` tokens are generated per outer
    block, then refined through ``small_block_size`` sub-blocks of iterative
    unmasking.

    Args:
        model_name: HuggingFace model ID (must be a Fast-dLLM v2 checkpoint).
        tasks: lm-eval task name(s) — comma-separated string or list
            (e.g. ``"gsm8k,minerva_math"``).
        device_id: CUDA device string.
        dtype: Model dtype — ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        quant_config: Path to a TOML quantization recipe. ``None`` runs the
            unquantized baseline.
        model_parallel: Distribute across GPUs with ``device_map="auto"``.
        batch_size: lm-eval batch size.
        max_new_tokens: Maximum tokens generated per sample.
        num_fewshot: Few-shot examples prepended to each task prompt.
        mask_id: Token ID used as the diffusion mask. Default ``151665``
            (matches Qwen-based Fast-dLLM checkpoints).
        bd_size: Outer block-diffusion block size — tokens generated per
            outer sampling step.
        small_block_size: Inner block size for iterative unmasking within
            each outer block.
        threshold: Confidence threshold for committing unmasked tokens.
        use_block_cache: Cache intermediate block KV states for faster decoding.
        temperature: Sampling temperature.
        top_p: Top-p sampling probability.
        show_speed: Log throughput metrics (tokens/second).
        log_dir: Directory for ``args.json`` and ``results.json``.

    Returns:
        lm-eval results dict — per-task metrics plus aggregate scores.
    """
    print("=" * 60)
    print("Fast-dLLM Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"Block size: {bd_size}, Sub-block: {small_block_size}, Threshold: {threshold}")
    print(f"Block cache: {use_block_cache}, Temperature: {temperature}, Top-p: {top_p}")

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
        # Plumb device (+ model_name) into the GPTQ / rotation-search passes the
        # same way eval_lm does — the MASE passes need these but they don't
        # belong in the TOML schema. Required for configs 04-06 (GPTQ) and 06
        # (rotation_search), so all six table9 rows run through this CLI.
        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id
        if "rotation_search" in pass_args:
            pass_args["rotation_search"]["device"] = device_id
            pass_args["rotation_search"].setdefault("model_name", model_name)

        n_linear = sum(
            1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)
        )
        logger.info("Quantizing %d linear layers...", n_linear)
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

        # Surface which classes the dispatch landed on, so you can confirm the
        # rotate / GPTQ variants are wired in when the TOML asks for them.
        from collections import Counter

        cls_count = Counter(
            type(m).__name__ for _, m in model.named_modules()
            if "MX" in type(m).__name__
        )
        if cls_count:
            logger.info(
                "Post-quant module classes:\n%s",
                "\n".join(f"  {c}: {n}" for c, n in cls_count.most_common()),
            )

    if model_parallel:
        model = move_to_gpu(model, model_parallel)
    else:
        model.to(device_id)

    if quantize:
        print_all_layers(model)

    # Attach block diffusion sampling method
    setup_dllm_generation(model)

    device = torch.device(device_id)
    results = evaluate_dllm(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        device=device,
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_fewshot=num_fewshot,
        mask_id=mask_id,
        bd_size=bd_size,
        small_block_size=small_block_size,
        threshold=threshold,
        use_block_cache=use_block_cache,
        temperature=temperature,
        show_speed=show_speed,
    )


    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

    if log_dir:
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
