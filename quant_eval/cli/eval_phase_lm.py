"""
lm-eval harness with phase- and layer-type-dependent MX quantization.

Supports four independent activation precision settings:

    prefill × attn   prefill × ffn
    decode  × attn   decode  × ffn

Uses PhaseLayerAutoSwitch from phase_quant.py, which combines:
  - a single lightweight top-level hook for prefill/decode detection
  - per-submodule hooks that apply the right config to each attention
    or FFN block immediately before its forward pass

Transparent to lm-eval — no modifications to the harness needed.

Usage:
    python -m quant_eval.cli.eval_phase_lm --help

Example — fully disaggregated:
    python -m quant_eval.cli.eval_phase_lm \
        --model_name Qwen/Qwen2.5-1.5B \
        --quant_config quant_eval/configs/llama_mxint4.toml \
        --prefill_attn_width 4 \
        --prefill_ffn_width  4 \
        --decode_attn_width  8 \
        --decode_ffn_width   6 \
        --tasks wikitext

Example — phase-only (attn == ffn per phase, reproduces old behaviour):
    python -m quant_eval.cli.eval_phase_lm \
        --model_name Qwen/Qwen2.5-1.5B \
        --quant_config quant_eval/configs/llama_mxint4.toml \
        --prefill_attn_width 4 --prefill_ffn_width 4 \
        --decode_attn_width  8 --decode_ffn_width  8 \
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
from quant_eval.eval.lm_eval import evaluate_with_lm_eval
from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
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
    batch_size: Union[int, str] = 64,
    # ── Activation precision: prefill ──────────────────────────────
    prefill_attn_width:      int = 4,
    prefill_ffn_width:       int = 4,
    prefill_attn_block_size: int = 32,
    prefill_ffn_block_size:  int = 32,
    # ── Activation precision: decode ───────────────────────────────
    decode_attn_width:       int = 8,
    decode_ffn_width:        int = 8,
    decode_attn_block_size:  int = 32,
    decode_ffn_block_size:   int = 32,
    # ── Optional keyword overrides (for non-standard architectures) ─
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords:  Union[list[str], None] = None,
    limit: Union[int, float, None] = None,
    log_dir: Union[str, None] = None,
):
    """
    Run lm-eval with independent MX activation precision per
    (phase, layer_type) pair.

    Phase is detected automatically from input sequence length;
    layer type is detected from module names. lm-eval is unchanged.

    Args:
        model_name:             HuggingFace model ID.
        tasks:                  lm-eval task(s) to run.
        device_id:              CUDA device string.
        dtype:                  Model dtype (float16 / bfloat16 / float32).
        quant_config:           TOML config for weight quantization.
        model_parallel:         Auto-dispatch across all visible GPUs.
        seqlen:                 Maximum context length passed to lm-eval.
        batch_size:             lm-eval batch size or "auto".
        prefill_attn_width:     Activation bit-width for attention during prefill.
        prefill_ffn_width:      Activation bit-width for FFN during prefill.
        prefill_attn_block_size: MX block size for attention during prefill.
        prefill_ffn_block_size:  MX block size for FFN during prefill.
        decode_attn_width:      Activation bit-width for attention during decode.
        decode_ffn_width:       Activation bit-width for FFN during decode.
        decode_attn_block_size:  MX block size for attention during decode.
        decode_ffn_block_size:   MX block size for FFN during decode.
        attn_keywords:          Override default attention module name keywords.
        ffn_keywords:           Override default FFN module name keywords.
        limit:                  Cap the number of samples per task. Int = absolute
                                count, float in (0, 1) = fraction. None = full dataset.
        log_dir:                Directory for experiment logs and results.
    """
    # ------------------------------------------------------------------
    # Build the nested phase × layer config
    # ------------------------------------------------------------------
    phase_configs = {
        "prefill": {
            "attn": {
                "data_in_width":      prefill_attn_width,
                "data_in_block_size": prefill_attn_block_size,
            },
            "ffn": {
                "data_in_width":      prefill_ffn_width,
                "data_in_block_size": prefill_ffn_block_size,
            },
        },
        "decode": {
            "attn": {
                "data_in_width":      decode_attn_width,
                "data_in_block_size": decode_attn_block_size,
            },
            "ffn": {
                "data_in_width":      decode_ffn_width,
                "data_in_block_size": decode_ffn_block_size,
            },
        },
    }

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    _pa = f"MXInt{prefill_attn_width}(bs={prefill_attn_block_size})"
    _pf = f"MXInt{prefill_ffn_width}(bs={prefill_ffn_block_size})"
    _da = f"MXInt{decode_attn_width}(bs={decode_attn_block_size})"
    _df = f"MXInt{decode_ffn_width}(bs={decode_ffn_block_size})"

    print("=" * 64)
    print("lm-eval — Phase × Layer-Type Disaggregated Quantization")
    print("=" * 64)
    print(f"  Model  : {model_name}")
    print(f"  Tasks  : {tasks}")
    print(f"  Weights: {quant_config}")
    print()
    print(f"  {'':10s}  {'attn':>24s}  {'ffn':>24s}")
    print(f"  {'prefill':10s}  {_pa:>24s}  {_pf:>24s}")
    print(f"  {'decode':10s}  {_da:>24s}  {_df:>24s}")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Logging / experiment directory
    # ------------------------------------------------------------------
    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        save_args(log_dir, locals().copy())
        import shutil
        shutil.copy(quant_config, log_dir / "quant_config.toml")

    transformers.set_seed(0)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Apply weight quantization (activation configs are set by the hook)
    # ------------------------------------------------------------------
    from chop.passes.module.transforms import quantize_module_transform_pass

    pass_args = load_quant_config(quant_config)
    if "gptq" in pass_args:
        pass_args["gptq"]["device"] = device_id

    n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    logger.info("Quantizing %d linear layers...", n_linear)
    t0 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    logger.info("Quantization complete in %.1fs", time.time() - t0)

    if model_parallel:
        model = move_to_gpu(model, model_parallel)
    else:
        model.to(device_id)

    # ------------------------------------------------------------------
    # Enable disaggregated quantization hook
    # ------------------------------------------------------------------
    switch_kwargs = {}
    if attn_keywords:
        switch_kwargs["attn_keywords"] = tuple(attn_keywords)
    if ffn_keywords:
        switch_kwargs["ffn_keywords"] = tuple(ffn_keywords)

    switch = PhaseLayerAutoSwitch(model, phase_configs, **switch_kwargs)
    switch.enable()
    logger.info("\n%s", switch.summary())

    # ------------------------------------------------------------------
    # Run lm-eval (hook fires transparently on every forward pass)
    # ------------------------------------------------------------------
    results = evaluate_with_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        max_length=seqlen,
        batch_size=128,
        log_samples=False,
        limit=limit,
    )

    switch.disable()

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Results:")
    print("=" * 64)
    print(f"\n  {'':10s}  {'attn':>24s}  {'ffn':>24s}")
    print(f"  {'prefill':10s}  {_pa:>24s}  {_pf:>24s}")
    print(f"  {'decode':10s}  {_da:>24s}  {_df:>24s}")
    print()

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
        results["phase_layer_configs"] = phase_configs
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")