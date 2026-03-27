"""
OSWorld agentic evaluation with optional MASE quantization.

Uses text-only mode (a11y_tree) so quantized language models can serve
as OSWorld agents without vision capabilities.

Usage:
    python -m quant_eval.cli.eval_osworld --help

Example (baseline):
    python -m quant_eval.cli.eval_osworld \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --osworld_path /path/to/OSWorld

Example (quantized):
    python -m quant_eval.cli.eval_osworld \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --quant_config quant_eval/configs/llama_mxint4.toml \
        --osworld_path /path/to/OSWorld
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
from quant_eval.eval.osworld import evaluate_osworld
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")


def main(
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    osworld_path: str = "quant_eval/benchmarks/OSWorld",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: Union[str, None] = None,
    model_parallel: bool = False,
    # OSWorld environment settings
    provider_name: str = "docker",
    path_to_vm: Union[str, None] = None,
    domain: str = "all",
    max_steps: int = 15,
    max_tokens: int = 1500,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_trajectory_length: int = 3,
    a11y_tree_max_tokens: int = 10000,
    result_dir: str = "./results",
    client_password: str = "password",
    screen_width: int = 1920,
    screen_height: int = 1080,
    headless: bool = True,
    sleep_after_execution: float = 0.0,
    test_all_meta_path: Union[str, None] = None,
    log_dir: Union[str, None] = None,
):
    """
    Run OSWorld agentic evaluation with optional MX quantization.

    The model operates in text-only mode using accessibility tree
    observations. It generates pyautogui code to interact with the
    desktop environment.

    Args:
        model_name: HuggingFace model ID (should be an instruct/chat model).
        osworld_path: Path to the OSWorld repository.
        device_id: CUDA device.
        dtype: Model dtype (float16, bfloat16, float32).
        quant_config: Path to TOML config for quantize_module_transform_pass.
                      If None, run unquantized baseline.
        model_parallel: Auto-dispatch across GPUs.
        provider_name: VM provider (docker, vmware, virtualbox, aws).
        path_to_vm: Path to VM image (for vmware/virtualbox).
        domain: Task domain (all, chrome, libreoffice_calc, etc.).
        max_steps: Maximum steps per task.
        max_tokens: Max generation tokens per step.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_trajectory_length: Steps of history in context.
        a11y_tree_max_tokens: Max tokens for accessibility tree text.
        result_dir: Directory for per-task results.
        client_password: VM client password.
        screen_width: VM screen width.
        screen_height: VM screen height.
        headless: Run VM without GUI.
        sleep_after_execution: Pause after each action execution.
        test_all_meta_path: Path to test_all.json task list.
        log_dir: Directory to save experiment logs.
    """
    print("=" * 60)
    print("OSWorld Agentic Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"OSWorld path: {osworld_path}")
    print(f"Domain: {domain}")
    print(f"Observation: a11y_tree (text-only)")
    print(f"Action space: pyautogui")

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

    results = evaluate_osworld(
        model=model,
        tokenizer=tokenizer,
        osworld_path=osworld_path,
        provider_name=provider_name,
        path_to_vm=path_to_vm,
        domain=domain,
        max_steps=max_steps,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_trajectory_length=max_trajectory_length,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
        result_dir=result_dir,
        client_password=client_password,
        screen_width=screen_width,
        screen_height=screen_height,
        headless=headless,
        sleep_after_execution=sleep_after_execution,
        test_all_meta_path=test_all_meta_path,
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Average score: {results['avg_score']:.4f}")
    print(f"  Tasks completed: {results['total_success']}/{results['total_tasks']}")
    print()
    if results.get("per_domain"):
        print("Per-domain breakdown:")
        for domain_name, domain_results in results["per_domain"].items():
            print(
                f"  {domain_name}: "
                f"avg={domain_results['avg_score']:.4f}, "
                f"success={domain_results['num_success']}/{domain_results['num_tasks']}"
            )

    if log_dir:
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
