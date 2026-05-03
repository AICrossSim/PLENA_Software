"""
Plain lm-eval driver for an MX-quantized HuggingFace model.

Same shape as ``eval_phase_lm.py`` but **without** PhaseLayerAutoSwitch —
activation precision is whatever the TOML quant config dictates and stays
fixed for the whole run. Use this when you want to evaluate a single
quantization profile (e.g. mxint_rotate) end-to-end without prefill/decode
disaggregation.

Example:
    python -m quant_eval.cli.eval_lm \\
        --model_name Qwen/Qwen3-8B \\
        --quant_config quant_eval/configs/qwen3_mxint16_rotate.toml \\
        --tasks wikitext \\
        --seqlen 2048
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
    num_fewshot: Union[int, None] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Union[str, None] = None,
):
    """
    Run lm-eval against an MX-quantized HF model with a single fixed
    activation precision profile (no phase-aware switching).

    Args:
        model_name:    HuggingFace model ID.
        tasks:         lm-eval task(s) to run (comma-separated string OK).
        device_id:     CUDA device string.
        dtype:         Model dtype (float16 / bfloat16 / float32).
        quant_config:  TOML config for module-level quantization. Set the
                       per-pattern ``name`` to ``"mxint_rotate"`` to enable
                       online Hadamard rotation. ``None`` = no quantization.
        model_parallel: Use HF device_map="auto" pipeline parallel.
        seqlen:        Context window for lm-eval.
        batch_size:    Eval batch size (int or "auto").
        limit:         Cap samples per task (int absolute, float in (0,1)
                       fraction, None = full dataset).
        log_dir:       Directory for experiment logs and results.
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
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
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
