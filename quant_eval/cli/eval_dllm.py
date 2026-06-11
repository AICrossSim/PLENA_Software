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
    _fix_rotary_buffers,
)
from quant_eval.eval.dllm_v2.dllm_generation import setup_dllm_generation
from quant_eval.eval.dllm_v2.eval_dllm import evaluate_dllm
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("info")


def _normalize_tied_weights_keys(model: torch.nn.Module) -> None:
    """Coerce any list-valued ``_tied_weights_keys`` to the dict form that
    newer transformers expects.

    Older models (e.g. Fast-dLLM's remote ``modeling.py``) declare
    ``_tied_weights_keys = ["lm_head.weight"]``. The new ``save_pretrained``
    path iterates ``_tied_weights_keys.keys()``, so a list raises
    ``AttributeError: 'list' object has no attribute 'keys'``. The dict maps
    each tied (alias) weight to the source weight it shares storage with —
    here the input-embedding weight.
    """
    for submodule in model.modules():
        tied = getattr(submodule, "_tied_weights_keys", None)
        if not isinstance(tied, (list, tuple)):
            continue
        source = None
        get_input_embeddings = getattr(submodule, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                emb = get_input_embeddings()
            except (AttributeError, NotImplementedError):
                emb = None
            if emb is not None:
                # Find the dotted name of the embedding's weight within submodule.
                for name, param in submodule.named_parameters():
                    if param is getattr(emb, "weight", None):
                        source = name
                        break
        # Fall back to self-mapping if we can't resolve the source; this still
        # satisfies the `.keys()` access and tying is re-applied from config.
        submodule._tied_weights_keys = {k: (source or k) for k in tied}


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

    from pathlib import Path
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

    # Place the model on GPU *before* quantization so the loaded weights and the
    # quant passes (GPTQ / rotation-search calibration) all run on-device rather
    # than on CPU — setup_model loads onto CPU. For model-parallel we defer to
    # the dispatch_model call after quantization, since accelerate's offload
    # hooks don't compose with the in-place module-replacement passes.
    if not model_parallel:
        logger.info("Moving model to %s", device_id)
        model.to(device_id)

    if quantize:
        pass_args = load_quant_config(quant_config)
        
        # Check if we're running Table 9 configuration 06. Config 06 is the most
        # expensive because it adds rotation search on top of GPTQ + erryclip,
        # so we optionally cache it to a separate filesystem (/tmp).
        is_config6 = "table9" in str(quant_config) and "06_" in Path(quant_config).name
        cache_dir = Path("/tmp/plena_quant_cache") / f"{model_name.replace('/', '--')}_table9_config6" if is_config6 else None
        
        loaded_from_cache = False
        if cache_dir and cache_dir.exists():
            logger.info("Reloading pre-quantized model from %s", cache_dir)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                cache_dir, trust_remote_code=True, torch_dtype=torch_dtype
            )
            _fix_rotary_buffers(model, logger)
            if not model_parallel:
                model.to(device_id)
            loaded_from_cache = True

        if not loaded_from_cache:
            from chop.passes.module.transforms import quantize_module_transform_pass

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

            if cache_dir:
                logger.info("Saving quantized model to %s", cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Fast-dLLM's remote modeling.py declares `_tied_weights_keys`
                # as a list (old transformers convention), but the installed
                # transformers' save path calls `.keys()` on it (new dict
                # convention). Normalize any list-valued attr to a dict mapping
                # each tied weight -> the input-embedding weight it aliases.
                _normalize_tied_weights_keys(model)
                model.save_pretrained(cache_dir)

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

    # Final device placement. For model-parallel this is the real dispatch step;
    # for single-GPU it's a safety net that catches any modules the quant passes
    # created on CPU (the model is already on device_id from before quantization).
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
