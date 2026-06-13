"""WikiText2 perplexity evaluation for prefill-only DSE precision points.

This entrypoint mirrors the BFCL DSE quantization setup but evaluates
teacher-forcing PPL.  It is intentionally separate from eval_phase_bfcl.py so
PPL diagnostics cannot alter the BFCL/GPTQ execution path.
"""

from __future__ import annotations

from typing import Union
import atexit
import shutil
import time

import torch
import transformers

from quant_eval.cli.eval_phase_bfcl import (
    _FixedCudaMemoryReserve,
    _GptqWeightCache,
    _inject_gptq_config,
    _mark_gptq_projection_configs,
    _normalize_device_id,
)
from quant_eval.eval import evaluate_perplexity
from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.eval.unified_mx import apply_unified_mx_wrappers
from quant_eval.precision import (
    apply_dse_quant_config,
    fp_data_config,
    mx_data_config,
    parse_fp_setting,
    parse_mx_precision,
)
from quant_eval.quantize import load_quant_config
from quant_eval.utils import (
    create_experiment_log_dir,
    get_logger,
    move_to_gpu,
    save_args,
    save_results,
    set_logging_verbosity,
    setup_model,
)

logger = get_logger(__name__)
set_logging_verbosity("debug")


def _legacy_mx_config(width: int, block_size: int) -> dict:
    return {"data_in_width": width, "data_in_block_size": block_size}


def _resolve_phase_precision(
    *,
    phase: str,
    act: str | None,
    kv: str | None,
    fp: str | None,
    legacy_attn: dict,
    legacy_ffn: dict,
    dse_mx_block_size: int,
    model_family: str = "llama",
    enable_qwen_default_precision: bool = False,
) -> dict:
    provided = {"ACT_ELEMENT_WIDTH": act, "KV_ELEMENT_WIDTH": kv, "FP_SETTING": fp}
    if any(v is not None for v in provided.values()) and not all(v is not None for v in provided.values()):
        missing = [name for name, value in provided.items() if value is None]
        raise ValueError(
            f"{phase} DSE precision requires ACT_ELEMENT_WIDTH, KV_ELEMENT_WIDTH, "
            f"and FP_SETTING together; missing {missing}."
        )

    if act is None and kv is None and fp is None:
        if model_family in {"qwen3", "qwen3_moe"} and enable_qwen_default_precision:
            act, kv, fp = "MXINT_8", "MXINT_8", "FP_E8M5"
        else:
            attn_cfg = dict(legacy_attn)
            return {
                "attn": attn_cfg,
                "ffn": dict(legacy_ffn),
                "mlp": {},
                "rms_norm": {},
                "display": f"MXInt{legacy_attn['data_in_width']}(bs={legacy_attn['data_in_block_size']})",
                "ffn_display": f"MXInt{legacy_ffn['data_in_width']}(bs={legacy_ffn['data_in_block_size']})",
                "metadata": {
                    "ACT_ELEMENT_WIDTH": f"MXINT_{legacy_attn['data_in_width']}",
                    "KV_ELEMENT_WIDTH": f"MXINT_{legacy_attn['data_in_width']}",
                    "FP_SETTING": None,
                },
            }

    act_spec = parse_mx_precision(act or "MXINT_4")
    kv_spec = parse_mx_precision(kv or act_spec.canonical)
    fp_spec = parse_fp_setting(fp or "FP_E3M2")
    act_cfg = mx_data_config(act_spec, dse_mx_block_size)
    kv_cfg = mx_data_config(kv_spec, dse_mx_block_size)
    fp_cfg = fp_data_config(fp_spec)
    attn_cfg = {**act_cfg, "kv_cache": dict(kv_cfg), "softmax": dict(fp_cfg), "rope": dict(fp_cfg)}
    rms_cfg = {
        **fp_cfg,
        "weight_exponent_width": fp_spec.exp,
        "weight_frac_width": fp_spec.frac,
        "weight_is_finite": True,
        "weight_round_mode": "rn",
    }
    return {
        "attn": attn_cfg,
        "ffn": dict(act_cfg),
        "mlp": dict(fp_cfg),
        "rms_norm": rms_cfg,
        "display": f"{act_spec.canonical}/KV={kv_spec.canonical}/NL={fp_spec.canonical}(B{dse_mx_block_size})",
        "ffn_display": f"{act_spec.canonical}/NL={fp_spec.canonical}(B{dse_mx_block_size})",
        "metadata": {
            "ACT_ELEMENT_WIDTH": act_spec.canonical,
            "KV_ELEMENT_WIDTH": kv_spec.canonical,
            "FP_SETTING": fp_spec.canonical,
        },
    }


def main(
    model_name: str = "Qwen/Qwen3-8B",
    dataset: str = "wikitext",
    subset: str | None = "wikitext-2-raw-v1",
    split: str = "test",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    quant_config: str | None = "quant_eval/configs/qwen3_mxint16.toml",
    model_parallel: bool = False,
    model_family: str = "qwen3",
    seqlen: int = 1024,
    max_samples: int | None = 64,
    # GPU reservation guard. Held during cache/wrapper setup and released before PPL forward.
    gpu_memory_reserve_mb: int = 0,
    gpu_memory_reserve_wait_sec: int = 600,
    gpu_memory_reserve_poll_sec: float = 5.0,
    gpu_memory_reserve_chunk_mb: int = 512,
    gpu_memory_reserve_disable: bool = False,
    # GPTQ cache/config. Defaults are load-only to avoid rerunning GPTQ in PPL diagnostics.
    gptq_dataset: str | None = None,
    gptq_nsamples: int = 32,
    gptq_seqlen: int = 1024,
    gptq_format: str = "mxint",
    gptq_weight_width: int = 8,
    gptq_weight_block_size: int = 32,
    gptq_cali_batch_size: int = 1,
    gptq_max_layers: int | None = None,
    gptq_cache_dir: str | None = None,
    gptq_cache_mode: str = "require",
    # Legacy phase widths.
    prefill_attn_width: int = 4,
    prefill_ffn_width: int = 4,
    prefill_attn_block_size: int = 32,
    prefill_ffn_block_size: int = 32,
    decode_attn_width: int = 8,
    decode_ffn_width: int = 8,
    decode_attn_block_size: int = 32,
    decode_ffn_block_size: int = 32,
    # Codesign precision controls.
    act_element_width_prefill: str | None = None,
    act_element_width_decode: str | None = None,
    kv_element_width_prefill: str | None = None,
    kv_element_width_decode: str | None = None,
    fp_setting_prefill: str | None = None,
    fp_setting_decode: str | None = None,
    dse_mx_block_size: int = 16,
    dse_weight_precision: str | None = None,
    dse_weight_block_size: int | None = None,
    decode_weight_mode: str = "quantized",
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords: Union[list[str], None] = None,
    log_dir: Union[str, None] = None,
):
    device_id = _normalize_device_id(device_id)
    model_family = model_family.lower()
    qwen_model_family = model_family in {"qwen3", "qwen3_moe"}
    gpu_memory_reserve_enabled = (
        not gpu_memory_reserve_disable
        and gpu_memory_reserve_mb is not None
        and int(gpu_memory_reserve_mb) > 0
    )
    if gpu_memory_reserve_enabled and model_parallel:
        raise ValueError(
            "GPU memory reservation currently supports only single-GPU PPL eval; "
            "disable it with --gpu_memory_reserve_disable true for model_parallel runs."
        )

    quant_config_is_none = quant_config is None or str(quant_config).strip().lower() in {"", "none", "fp", "false"}
    if quant_config_is_none:
        quant_config = "none"
        active_quant = [
            name for name, value in {
                "gptq_dataset": gptq_dataset,
                "act_element_width_prefill": act_element_width_prefill,
                "kv_element_width_prefill": kv_element_width_prefill,
                "fp_setting_prefill": fp_setting_prefill,
                "dse_weight_precision": dse_weight_precision,
            }.items() if value is not None
        ]
        if active_quant:
            raise ValueError(f"--quant_config none cannot be combined with quantization options: {active_quant}")
        if decode_weight_mode != "quantized":
            logger.info("Ignoring decode_weight_mode=%s for FP PPL baseline.", decode_weight_mode)
            decode_weight_mode = "quantized"

    decode_weight_policy = {"weight_mode": "fp", "bypass": True} if decode_weight_mode == "fp" else {}
    decode_nonlinear_policy = {"bypass": True} if decode_weight_mode == "fp" else {}

    prefill = _resolve_phase_precision(
        phase="prefill",
        act=act_element_width_prefill,
        kv=kv_element_width_prefill,
        fp=fp_setting_prefill,
        legacy_attn=_legacy_mx_config(prefill_attn_width, prefill_attn_block_size),
        legacy_ffn=_legacy_mx_config(prefill_ffn_width, prefill_ffn_block_size),
        dse_mx_block_size=dse_mx_block_size,
        model_family=model_family,
        enable_qwen_default_precision=(qwen_model_family and not quant_config_is_none),
    )
    decode = _resolve_phase_precision(
        phase="decode",
        act=act_element_width_decode,
        kv=kv_element_width_decode,
        fp=fp_setting_decode,
        legacy_attn=_legacy_mx_config(decode_attn_width, decode_attn_block_size),
        legacy_ffn=_legacy_mx_config(decode_ffn_width, decode_ffn_block_size),
        dse_mx_block_size=dse_mx_block_size,
        model_family=model_family,
        enable_qwen_default_precision=(qwen_model_family and not quant_config_is_none),
    )
    phase_configs = {
        "prefill": {
            "attn": prefill["attn"],
            "ffn": prefill["ffn"],
            "mlp": prefill["mlp"],
            "rms_norm": prefill["rms_norm"],
        },
        "decode": {
            "attn": {**decode["attn"], **decode_weight_policy},
            "ffn": {**decode["ffn"], **decode_weight_policy},
            "mlp": {**decode["mlp"], **decode_nonlinear_policy},
            "rms_norm": {**decode["rms_norm"], **decode_nonlinear_policy},
        },
    }
    precision_metadata = {
        "prefill": prefill["metadata"],
        "decode": decode["metadata"],
        "dse_mx_block_size": dse_mx_block_size,
        "dse_weight_precision": dse_weight_precision or (f"MXINT_{gptq_weight_width}" if gptq_dataset else "MXINT_8"),
        "dse_weight_block_size": dse_weight_block_size if dse_weight_block_size is not None else (gptq_weight_block_size if gptq_dataset else None),
    }
    qwen3_default_precision_enabled = qwen_model_family and not quant_config_is_none
    codesign_tokens_enabled = qwen3_default_precision_enabled or any(v is not None for v in (
        act_element_width_prefill, act_element_width_decode,
        kv_element_width_prefill, kv_element_width_decode,
        fp_setting_prefill, fp_setting_decode,
    ))

    print("=" * 64)
    print("WikiText2 PPL — Prefill DSE Quantization")
    print("=" * 64)
    print(f"  Model     : {model_name}")
    print(f"  Dataset   : {dataset}/{subset or '-'} split={split}")
    print(f"  Family    : {model_family}")
    print(f"  Seqlen    : {seqlen}, max_samples={max_samples if max_samples is not None else 'all'}")
    print(f"  Weights   : {'FP baseline (no quantization)' if quant_config_is_none else quant_config}")
    if gptq_dataset:
        print(f"  GPTQ      : dataset={gptq_dataset}, cache={gptq_cache_mode}@{gptq_cache_dir}")
    print(f"  Prefill   : attn={prefill['display']} ffn={prefill['ffn_display']}")
    print(f"  Decode    : {decode_weight_mode} (unused for teacher-forcing PPL unless seq_len==1)")
    if gpu_memory_reserve_enabled:
        print(
            "  GPU reserve: "
            f"reserve={int(gpu_memory_reserve_mb)}MB, "
            f"wait={gpu_memory_reserve_wait_sec}s, "
            f"chunk={gpu_memory_reserve_chunk_mb}MB"
        )
    else:
        print("  GPU reserve: disabled")
    print("=" * 64)

    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        save_args(log_dir, locals().copy())
        if not quant_config_is_none:
            shutil.copy(str(quant_config), log_dir / "quant_config.toml")

    transformers.set_seed(0)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
        attn_implementation="eager" if not quant_config_is_none else "sdpa",
    )
    model.eval()

    gpu_memory_reserve = _FixedCudaMemoryReserve(
        device=device_id,
        reserve_mb=int(gpu_memory_reserve_mb or 0),
        wait_sec=gpu_memory_reserve_wait_sec,
        poll_sec=gpu_memory_reserve_poll_sec,
        chunk_mb=gpu_memory_reserve_chunk_mb,
        enabled=gpu_memory_reserve_enabled,
        release_label="before PPL forward",
    )
    gpu_memory_reserve.acquire()
    atexit.register(gpu_memory_reserve.release)

    gptq_cache_info = {"mode": str(gptq_cache_mode or "off").lower(), "hit": False}
    switch = None
    if quant_config_is_none:
        # No low-memory quantization/setup phase remains for FP baseline.
        gpu_memory_reserve.release()
        if model_parallel:
            model = move_to_gpu(model, model_parallel)
        else:
            model.to(device_id)
    else:
        from chop.passes.module.transforms import quantize_module_transform_pass

        pass_args = load_quant_config(str(quant_config))
        if codesign_tokens_enabled:
            apply_dse_quant_config(
                pass_args,
                act_precision=precision_metadata["prefill"]["ACT_ELEMENT_WIDTH"],
                kv_precision=precision_metadata["prefill"]["KV_ELEMENT_WIDTH"],
                fp_setting=precision_metadata["prefill"]["FP_SETTING"],
                mx_block_size=dse_mx_block_size,
                weight_precision=precision_metadata["dse_weight_precision"],
                weight_block_size=precision_metadata["dse_weight_block_size"],
                model_family=model_family,
            )
        resolved_gptq_config = _inject_gptq_config(
            pass_args,
            model_name=model_name,
            device_id=device_id,
            dataset=gptq_dataset,
            nsamples=gptq_nsamples,
            seqlen=gptq_seqlen,
            fmt=gptq_format,
            weight_width=gptq_weight_width,
            weight_block_size=gptq_weight_block_size,
            cali_batch_size=gptq_cali_batch_size,
            max_layers=gptq_max_layers,
        )
        gptq_cache = None
        try:
            if resolved_gptq_config:
                marked = _mark_gptq_projection_configs(pass_args)
                logger.info("GPTQ config present: marked_weight_configs=%s", marked)
                if str(gptq_cache_mode or "off").lower() != "off":
                    if not gptq_cache_dir:
                        raise ValueError("gptq_cache_dir must be set when gptq_cache_mode is not 'off'.")
                    gptq_cache = _GptqWeightCache(
                        cache_dir=gptq_cache_dir,
                        mode=gptq_cache_mode,
                        gptq_config=resolved_gptq_config,
                        total_layers=len(model.model.layers),
                    )
                    cache_hit = gptq_cache.prepare(model)
                    gptq_cache_info = gptq_cache.summary()
                    if cache_hit:
                        pass_args.pop("gptq", None)
                    else:
                        resolved_gptq_config["checkpoint_dir"] = str(gptq_cache.cache_path)

            t0 = time.time()
            model, _ = quantize_module_transform_pass(model, pass_args)
            if gptq_cache is not None and not gptq_cache.hit:
                gptq_cache.finalize()
                gptq_cache_info = gptq_cache.summary()
            if codesign_tokens_enabled or qwen_model_family:
                qwen3_moe_experts_config = None
                if model_family == "qwen3_moe" and prefill["mlp"]:
                    qwen3_moe_experts_config = {**prefill["ffn"], **prefill["mlp"]}
                counts = apply_unified_mx_wrappers(
                    model,
                    qwen3_attention_config=prefill["attn"] if model_family == "qwen3" else None,
                    qwen3_mlp_config=prefill["mlp"] if model_family == "qwen3" and prefill["mlp"] else None,
                    qwen3_rms_norm_config=prefill["rms_norm"] if model_family == "qwen3" and prefill["rms_norm"] else None,
                    qwen3_moe_attention_config=prefill["attn"] if model_family == "qwen3_moe" else None,
                    qwen3_moe_experts_config=qwen3_moe_experts_config,
                    qwen3_moe_rms_norm_config=prefill["rms_norm"] if model_family == "qwen3_moe" and prefill["rms_norm"] else None,
                )
                logger.info("Installed unified MX wrappers: %s", counts)
            logger.info("Quantization setup complete in %.1fs", time.time() - t0)
        finally:
            if gptq_cache is not None:
                gptq_cache.release()

        # Release before any final placement/eval allocations so the reserve does
        # not overlap with PPL forward memory.
        gpu_memory_reserve.release()
        if model_parallel:
            model = move_to_gpu(model, model_parallel)
        else:
            model.to(device_id)

        switch_kwargs = {}
        if attn_keywords:
            switch_kwargs["attn_keywords"] = tuple(attn_keywords)
        if ffn_keywords:
            switch_kwargs["ffn_keywords"] = tuple(ffn_keywords)
        if decode_weight_mode == "fp":
            switch_kwargs["model_name"] = model_name
        switch = PhaseLayerAutoSwitch(model, phase_configs, **switch_kwargs)
        switch.enable()
        logger.info("\n%s", switch.summary())

    try:
        results = evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset,
            subset=subset,
            split=split,
            max_length=seqlen,
            max_samples=max_samples,
            verbose=True,
        )
    finally:
        if switch is not None:
            switch.disable()
        gpu_memory_reserve.release(log=False)

    results.update({
        "dataset": dataset,
        "subset": subset,
        "split": split,
        "seqlen": seqlen,
        "max_samples": max_samples,
        "phase_layer_configs": phase_configs,
        "precision_metadata": precision_metadata,
        "model_family": model_family,
        "decode_weight_mode": decode_weight_mode,
        "gptq_cache": gptq_cache_info,
        "gpu_memory_reserve": gpu_memory_reserve.summary(),
    })
    if log_dir:
        save_results(log_dir, results)
    print("\nResults:")
    for key in ("ppl", "nll", "num_tokens", "nsamples"):
        print(f"  {key}: {results[key]}")
    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start = time.time()
    CLI(main)
    print(f"\n[INFO] Total workload time: {time.time() - start:.2f} seconds")
