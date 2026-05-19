"""
lm-eval-harness driver with optional MX quantization.

Applies a TOML quantization recipe once before evaluation; activation
precision stays fixed for the whole run.

Example:

    python -m quant_eval.cli.eval_lm \\
        --model_name unsloth/Llama-3.2-1B \\
        --quant_config quant_eval/configs/llama_mxint4.toml \\
        --tasks arc_easy,hellaswag,winogrande \\
        --limit 500
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
from quant_eval.eval.lm_eval import evaluate_with_lm_eval
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")


def main(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    tasks: Union[str, list[str]] = "wikitext",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: Union[str, None] = "quant_eval/configs/llama_mxint4.toml",
    model_parallel: bool = False,
    seqlen: int = 2048,
    batch_size: Union[int, str] = 64,
    limit: Union[int, float, None] = None,
    log_dir: Union[str, None] = None,
):
    """
    Run lm-eval-harness on an optionally MX-quantized HF model.

    Args:
        model_name: HuggingFace model ID.
        tasks: lm-eval task name(s) — comma-separated string or list
            (e.g. ``"arc_easy,hellaswag"``).
        device_id: CUDA device string.
        dtype: Model dtype — ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        quant_config: Path to a TOML quantization recipe. ``None`` runs the
            unquantized baseline.
        model_parallel: Distribute across GPUs with ``device_map="auto"``.
        seqlen: Maximum context length passed to lm-eval.
        batch_size: Eval batch size. Pass an int for a fixed size, or the
            string ``"auto"`` for lm-eval's auto-batching.
        limit: Cap samples per task. Int = absolute count; float in
            ``(0, 1)`` = fraction of the full dataset; ``None`` = full.
        log_dir: Directory for ``args.json`` and ``results.json``. ``None``
            disables logging.

    Returns:
        lm-eval results dict — per-task metrics plus aggregate scores.
    """
    print("=" * 64)
    print("lm-eval — fixed activation precision (no phase switch)")
    print("=" * 64)
    print(f"  Model  : {model_name}")
    print(f"  Tasks  : {tasks}")
    print(f"  Weights: {quant_config or 'none (fp)'}")
    print(f"  Seqlen : {seqlen}")
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

    # Quantized attention modules (MXInt / MXFP / *Rotate) replace the
    # eager forward path and assert _attn_implementation == "eager". Force
    # eager whenever a quant_config is supplied, regardless of TOML pattern.
    attn_impl = "eager" if quant_config else "sdpa"

    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
        attn_implementation=attn_impl,
    )
    model.eval()

    # ``token_collector`` is a side-effect pass that just attaches a hook;
    # if it's the *only* thing in the TOML this run is "calibration mode"
    # and we'll skip module quantization entirely.
    collector_info = None
    pass_args = load_quant_config(quant_config) if quant_config else None

    if pass_args and "token_collector" in pass_args:
        from chop.passes.module.transforms import attach_token_collector_pass

        tc_cfg = pass_args.pop("token_collector")
        logger.info("Attaching TokenCollector: %s", tc_cfg)
        model.to(device_id)
        model, collector_info = attach_token_collector_pass(model, tc_cfg)

    # Run quant pass only if there are real quant blocks left (selectors or
    # gptq); if pass_args is just {"by": ...} after popping token_collector,
    # we're in pure calibration mode and skip quantization.
    has_quant = pass_args is not None and (
        "gptq" in pass_args
        or any(k != "by" for k in pass_args.keys())
    )
    if has_quant:
        from chop.passes.module.transforms import quantize_module_transform_pass

        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id
        # Plumb device + model_name into rotation_search the same way; the
        # MASE pass needs them but they don't belong in the TOML schema.
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

        # Surface which classes the dispatch landed on (so you can confirm
        # rotate variants are wired in when the TOML asks for them).
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

    # In calibration-only mode, the TokenCollector hook will raise
    # ``CollectorFull`` from inside model.forward once enough tokens have
    # been buffered — we catch it here so the eval pass exits cleanly with
    # the calibration file already on disk.
    from chop.passes.module.transforms.gptq import CollectorFull
    try:
        results = evaluate_with_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_length=seqlen,
            batch_size=batch_size,
            log_samples=False,
            limit=limit,
        )
    except CollectorFull as e:
        logger.info("[calibration mode] aborted lm-eval as planned: %s", e)
        results = {"calibration_only": True}

    if collector_info is not None and not collector_info["collector"].complete:
        # lm-eval finished its limit without filling the buffer — flush whatever
        # we have to disk so downstream GPTQ has *something* to work with.
        collector_info["collector"].finalize()

    print("\n" + "=" * 64)
    print("Results:")
    print("=" * 64)
    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"  {task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
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
