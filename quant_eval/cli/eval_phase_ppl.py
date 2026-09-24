"""WikiText2 perplexity evaluation for prefill-only DSE precision points.

This entrypoint mirrors the BFCL DSE quantization setup but evaluates
teacher-forcing PPL.  It is intentionally separate from eval_phase_bfcl.py so
PPL diagnostics cannot alter the BFCL/GPTQ execution path.
"""

from __future__ import annotations

from typing import Union
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
import transformers

from quant_eval.cli.eval_phase_bfcl import (
    _GptqWeightCache,
    _inject_gptq_config,
    _mark_gptq_projection_configs,
    _normalize_device_id,
)
from quant_eval.eval.eval_ppl import evaluate_perplexity_with_kl
from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.eval.unified_mx import apply_qwen3_gptq_cache_unified_wrappers, apply_unified_mx_wrappers
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


def _stable_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_reference_fingerprint(model_name: str, model) -> str:
    config = model.config
    fields = (
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
    )
    return _stable_fingerprint({
        "model_name": str(model_name),
        "config": {name: getattr(config, name, None) for name in fields},
    })


def _tokenizer_reference_fingerprint(tokenizer) -> str:
    return _stable_fingerprint({
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": len(tokenizer),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    })


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
    ppl_reference_mode: str = "none",
    ppl_reference_dir: str | None = None,
    ppl_kl_samples: int = 64,
    ppl_profile_id: str = "default",
    ppl_progress_path: str | None = None,
    ppl_kl_position_block_size: int = 8,
    ppl_profiles: list[dict] | None = None,
    ppl_results_root: str | None = None,
    ppl_output_path: str | None = None,
    # GPTQ cache/config. Defaults are load-only to avoid rerunning GPTQ in PPL diagnostics.
    gptq_dataset: str | None = None,
    gptq_nsamples: int = 32,
    gptq_seqlen: int = 1024,
    gptq_format: str = "mxint",
    gptq_weight_width: int = 8,
    gptq_weight_exponent_width: int | None = None,
    gptq_weight_frac_width: int | None = None,
    gptq_weight_block_size: int = 32,
    gptq_cali_batch_size: int = 1,
    gptq_max_layers: int | None = None,
    gptq_device_map_aware: bool = False,
    gptq_cpu_staged: bool = False,
    gptq_cache_only: bool = False,
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
    decode_weight_residency: str = "disk_reload",
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords: Union[list[str], None] = None,
    log_dir: Union[str, None] = None,
):
    device_id = _normalize_device_id(device_id)
    model_family = model_family.lower()
    qwen_model_family = model_family in {"qwen3", "qwen3_moe"}

    if gptq_cpu_staged:
        if model_parallel:
            raise ValueError("gptq_cpu_staged requires model_parallel=False.")
        if gptq_device_map_aware:
            raise ValueError("gptq_cpu_staged is incompatible with gptq_device_map_aware.")
        if not gptq_cache_only:
            raise ValueError("gptq_cpu_staged is supported only with gptq_cache_only=True.")
    if gptq_cache_only:
        normalized_cache_mode = str(gptq_cache_mode or "off").lower()
        if normalized_cache_mode not in {"auto", "refresh", "require"}:
            raise ValueError(
                "gptq_cache_only requires gptq_cache_mode auto, refresh, or require."
            )
        if not gptq_cache_dir:
            raise ValueError("gptq_cache_only requires gptq_cache_dir.")
        if not gptq_dataset:
            raise ValueError("gptq_cache_only requires gptq_dataset.")

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
        if decode_weight_residency != "disk_reload":
            raise ValueError("--decode_weight_residency is only meaningful for quantized runs.")
    decode_weight_residency = str(decode_weight_residency or "disk_reload").lower()
    if decode_weight_residency not in {"disk_reload", "host_cached", "gpu_dual"}:
        raise ValueError(
            "decode_weight_residency must be 'disk_reload', 'host_cached', "
            "or 'gpu_dual', "
            f"got {decode_weight_residency!r}."
        )
    if (
        decode_weight_residency in {"host_cached", "gpu_dual"}
        and decode_weight_mode != "fp"
    ):
        raise ValueError(
            f"decode_weight_residency={decode_weight_residency!r} requires "
            "decode_weight_mode='fp'."
        )

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
        "dse_weight_precision": dse_weight_precision or (f"MXINT_{gptq_weight_width}" if gptq_dataset and gptq_format.lower() == "mxint" else "MXINT_8"),
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
    if gptq_cache_only:
        print(
            "  Cache only: enabled"
            + (f" (CPU staged on {device_id})" if gptq_cpu_staged else "")
        )
    print(f"  Prefill   : attn={prefill['display']} ffn={prefill['ffn_display']}")
    print(f"  Decode    : {decode_weight_mode} (unused for teacher-forcing PPL unless seq_len==1)")
    print(f"  Weight res: {decode_weight_residency}")
    print("=" * 64)

    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        save_args(log_dir, locals().copy())
        if not quant_config_is_none:
            shutil.copy(str(quant_config), log_dir / "quant_config.toml")

    transformers.set_seed(0)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    reference_mode = str(ppl_reference_mode or "none").lower()
    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
        attn_implementation=(
            "eager"
            if (not quant_config_is_none or reference_mode in {"write", "compare"})
            else "sdpa"
        ),
        layer_balanced_device_map=bool(
            model_parallel
            and qwen_model_family
            and (
                decode_weight_residency in {"host_cached", "gpu_dual"}
                or gptq_device_map_aware
            )
        ),
        cpu_staged=bool(gptq_cpu_staged),
    )
    model.eval()

    gptq_cache_info = {"mode": str(gptq_cache_mode or "off").lower(), "hit": False}
    switch = None
    if quant_config_is_none:
        t_move = time.time()
        if model_parallel:
            model = move_to_gpu(model, model_parallel)
        else:
            model.to(device_id)
        logger.info("Model device placement complete in %.1fs", time.time() - t_move)
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
            weight_exponent_width=gptq_weight_exponent_width,
            weight_frac_width=gptq_weight_frac_width,
            weight_block_size=gptq_weight_block_size,
            cali_batch_size=gptq_cali_batch_size,
            max_layers=gptq_max_layers,
            device_map_aware=bool(gptq_device_map_aware),
            cpu_staged=bool(gptq_cpu_staged),
        )
        gptq_cache = None
        try:
            if resolved_gptq_config:
                marked = _mark_gptq_projection_configs(pass_args)
                logger.info("GPTQ config present: marked_weight_configs=%s", marked)
                normalized_cache_mode = str(gptq_cache_mode or "off").lower()
                if normalized_cache_mode == "memory":
                    gptq_cache_info = {
                        "mode": "memory",
                        "hit": False,
                        "path": "",
                        "loaded_layers": 0,
                        "partial_layers": 0,
                        "resuming": False,
                    }
                elif normalized_cache_mode != "off":
                    if not gptq_cache_dir:
                        raise ValueError("gptq_cache_dir must be set when gptq_cache_mode is not 'off'.")
                    gptq_cache = _GptqWeightCache(
                        cache_dir=gptq_cache_dir,
                        mode=normalized_cache_mode,
                        gptq_config=resolved_gptq_config,
                        total_layers=len(model.model.layers),
                    )
                    cache_hit = gptq_cache.prepare(
                        model,
                        load_on_hit=not gptq_cache_only,
                    )
                    gptq_cache_info = gptq_cache.summary()
                    if cache_hit:
                        pass_args.pop("gptq", None)
                    else:
                        resolved_gptq_config["checkpoint_dir"] = str(gptq_cache.cache_path)

            if gptq_cache_only:
                if resolved_gptq_config is None or gptq_cache is None:
                    raise RuntimeError(
                        "Cache-only GPTQ requires a resolved GPTQ config and disk cache."
                    )
                started = time.time()
                if not gptq_cache.hit:
                    from chop.passes.module.transforms.gptq import run_gptq

                    run_gptq(model, resolved_gptq_config)
                    gptq_cache.finalize()
                    gptq_cache_info = gptq_cache.summary()
                results = {
                    "gptq_cache_only": True,
                    "gptq_cpu_staged": bool(gptq_cpu_staged),
                    "gptq_cache": gptq_cache_info,
                    "elapsed_sec": time.time() - started,
                    "model_name": model_name,
                    "model_family": model_family,
                }
                if ppl_output_path:
                    output_path = Path(ppl_output_path).expanduser().resolve()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
                    )
                if log_dir:
                    save_results(log_dir, results)
                logger.info(
                    "GPTQ cache-only stage complete in %.1fs: %s",
                    results["elapsed_sec"],
                    gptq_cache_info.get("path"),
                )
                return results

            t0 = time.time()
            qwen_cache_fast_path = bool(
                qwen_model_family
                and gptq_cache is not None
                and gptq_cache.hit
                and (codesign_tokens_enabled or qwen_model_family)
            )
            if qwen_cache_fast_path:
                logger.info("GPTQ cache hit Qwen fast path: skipping Chop transform pass.")
            else:
                t_chop = time.time()
                model, _ = quantize_module_transform_pass(model, pass_args)
                logger.info("Chop quantize_module_transform_pass complete in %.1fs", time.time() - t_chop)
                if gptq_cache is not None and not gptq_cache.hit:
                    gptq_cache.finalize()
                    gptq_cache_info = gptq_cache.summary()
            if codesign_tokens_enabled or qwen_model_family:
                qwen3_moe_experts_config = None
                if model_family == "qwen3_moe" and prefill["mlp"]:
                    qwen3_moe_experts_config = {
                        "act": prefill["ffn"],
                        "nonlinear": prefill["mlp"],
                    }
                t_unified = time.time()
                wrapper_kwargs = {
                    "qwen3_attention_config": prefill["attn"] if model_family == "qwen3" else None,
                    "qwen3_mlp_config": prefill["mlp"] if model_family == "qwen3" and prefill["mlp"] else None,
                    "qwen3_rms_norm_config": prefill["rms_norm"] if model_family == "qwen3" and prefill["rms_norm"] else None,
                    "qwen3_moe_attention_config": prefill["attn"] if model_family == "qwen3_moe" else None,
                    "qwen3_moe_experts_config": qwen3_moe_experts_config,
                    "qwen3_moe_rms_norm_config": prefill["rms_norm"] if model_family == "qwen3_moe" and prefill["rms_norm"] else None,
                }
                if qwen_cache_fast_path:
                    counts = apply_qwen3_gptq_cache_unified_wrappers(
                        model,
                        attn_linear_config=prefill["attn"],
                        ffn_linear_config=prefill["ffn"],
                        **wrapper_kwargs,
                    )
                else:
                    counts = apply_unified_mx_wrappers(model, **wrapper_kwargs)
                logger.info("Installed unified MX wrappers: %s", counts)
                logger.info("Unified wrapper install complete in %.1fs", time.time() - t_unified)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("Quantization setup complete in %.1fs", time.time() - t0)
        finally:
            if gptq_cache is not None:
                gptq_cache.release()

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
            switch_kwargs["weight_residency"] = decode_weight_residency
            if decode_weight_residency in {"disk_reload", "host_cached"}:
                quant_checkpoint_dir = gptq_cache_info.get("path")
                if quant_checkpoint_dir:
                    switch_kwargs["quant_checkpoint_dir"] = quant_checkpoint_dir
        t_switch = time.time()
        switch = PhaseLayerAutoSwitch(model, phase_configs, **switch_kwargs)
        switch.enable()
        logger.info("Phase switch setup complete in %.1fs", time.time() - t_switch)
        logger.info("\n%s", switch.summary())

    model_fingerprint = _model_reference_fingerprint(model_name, model)
    tokenizer_fingerprint = _tokenizer_reference_fingerprint(tokenizer)

    def _evaluate_profile(profile_name: str, progress: str | None) -> dict:
        return evaluate_perplexity_with_kl(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset,
            subset=subset,
            split=split,
            max_length=seqlen,
            max_samples=max_samples,
            verbose=True,
            reference_mode=reference_mode,
            reference_dir=ppl_reference_dir,
            kl_samples=ppl_kl_samples,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            profile_id=profile_name,
            progress_path=progress,
            kl_position_block_size=ppl_kl_position_block_size,
        )

    try:
        if ppl_profiles:
            if switch is None:
                raise ValueError("ppl_profiles requires a quantized run with phase switching enabled.")
            if not ppl_results_root:
                raise ValueError("ppl_results_root is required when ppl_profiles is set.")
            results_root = Path(ppl_results_root).expanduser().resolve()
            results_root.mkdir(parents=True, exist_ok=True)
            profile_results = []
            for profile in ppl_profiles:
                profile_name = str(profile["profile_id"])
                profile_prefill = _resolve_phase_precision(
                    phase=f"profile {profile_name}",
                    act=str(profile["act"]),
                    kv=str(profile["kv"]),
                    fp=str(profile["fp_setting"]),
                    legacy_attn=_legacy_mx_config(prefill_attn_width, prefill_attn_block_size),
                    legacy_ffn=_legacy_mx_config(prefill_ffn_width, prefill_ffn_block_size),
                    dse_mx_block_size=dse_mx_block_size,
                    model_family=model_family,
                    enable_qwen_default_precision=True,
                )
                profile_phase = {
                    "attn": profile_prefill["attn"],
                    "ffn": profile_prefill["ffn"],
                    "mlp": profile_prefill["mlp"],
                    "rms_norm": profile_prefill["rms_norm"],
                }
                switch.apply_phase("prefill", profile_phase)
                profile_dir = results_root / profile_name
                profile_dir.mkdir(parents=True, exist_ok=True)
                profile_result = _evaluate_profile(
                    profile_name,
                    str(profile_dir / "chunks.jsonl"),
                )
                profile_result.update({
                    "profile_id": profile_name,
                    "act": profile["act"],
                    "kv": profile["kv"],
                    "fp_setting": profile["fp_setting"],
                    "phase_config": profile_phase,
                })
                (profile_dir / "results.json").write_text(
                    json.dumps(profile_result, indent=2, sort_keys=True), encoding="utf-8"
                )
                profile_results.append(profile_result)
            results = {"profiles": profile_results}
        else:
            results = _evaluate_profile(ppl_profile_id, ppl_progress_path)
    finally:
        if switch is not None:
            switch.disable()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        "decode_weight_residency": decode_weight_residency,
        "gptq_device_map_aware": bool(gptq_device_map_aware),
        "gptq_cpu_staged": bool(gptq_cpu_staged),
        "gptq_cache_only": bool(gptq_cache_only),
        "gptq_cache": gptq_cache_info,
        "ppl_reference_mode": reference_mode,
        "ppl_reference_dir": ppl_reference_dir,
        "ppl_kl_samples": int(ppl_kl_samples),
        "ppl_profile_id": ppl_profile_id,
        "ppl_progress_path": ppl_progress_path,
        "ppl_profiles": ppl_profiles,
        "ppl_results_root": ppl_results_root,
        "ppl_output_path": ppl_output_path,
    })
    if ppl_output_path:
        output_path = Path(ppl_output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )
    if log_dir:
        save_results(log_dir, results)
    print("\nResults:")
    if "profiles" in results:
        for profile in results["profiles"]:
            print(
                f"  {profile['profile_id']}: ppl={profile['ppl']:.4f}, "
                f"kl={profile['kl_nats_per_token']}, top1={profile['top1_agreement']}"
            )
    else:
        for key in (
            "ppl", "nll", "num_tokens", "nsamples",
            "kl_nats_per_token", "top1_agreement", "kl_tokens",
        ):
            print(f"  {key}: {results[key]}")
    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start = time.time()
    CLI(main)
    print(f"\n[INFO] Total workload time: {time.time() - start:.2f} seconds")
