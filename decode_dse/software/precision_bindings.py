"""Decode-only quantisation configs for the MASE phase-split quantize pass.

Produces the ``pass_args`` that :func:`quantize_module_transform_pass` consumes,
so that:

* prefill runs unquantised (the separate prefill chip): each module uses the
  ``{"decode": {...}}`` shorthand, which the normaliser expands to
  ``prefill={"bypass": True}, decode_policy="quantized"``;
* decode runs MXINT/MXFP weights + KV cache, activations computed low-precision
  (stored bf16 on-chip);
* KV handoff defaults to ``decode_format`` — prefill quantises its KV writes into
  the decode chip's KV format.

attn and ffn share the weight format but keep independent widths (mixed
precision); KV has its own format. In GPTQ mode all linears share one weight
width (GPTQ calibrates one width per pass), so ``attn_w`` must equal ``ffn_w``;
RTN mode lets them differ.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Attention vs FFN projections, split so each can carry an independent weight
# width. Regexes match Llama and Qwen3 dense naming.
def parse_prec_token(tok: Any) -> tuple[str, Any]:
    """Parse a precision token into (format, width).

    ``4`` or ``"MXINT4"`` -> ("mxint", 4); ``"E2M1"`` -> ("mxfp", (2, 1)).
    Widths >= 16 (or ``None``) mean unquantised -> ("mxint", None).
    """
    if tok is None:
        return ("mxint", None)
    if isinstance(tok, int):
        return ("mxint", None if tok >= 16 else tok)
    s = str(tok).upper()
    if s.startswith("MXINT"):
        w = int(s[5:])
        return ("mxint", None if w >= 16 else w)
    if s.startswith("E") and "M" in s:
        e, m = s[1:].split("M")
        return ("mxfp", (int(e), int(m)))
    if s.isdigit():
        return parse_prec_token(int(s))
    raise ValueError(f"unrecognised precision token {tok!r} (use e.g. 4, 'MXINT4', 'E2M1').")


_ATTN_LINEAR_RE = r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
_FFN_LINEAR_RE = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"
_SELF_ATTN_RE = r"model\.layers\.\d+\.self_attn$"
_MLP_RE = r"model\.layers\.\d+\.mlp$"
_RMSNORM_RE = r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)$"


@dataclass(frozen=True)
class DecodeQuantSpec:
    """One decode precision point for the software eval.

    Widths are ints for MXINT and ``"E{e}M{m}"`` tokens (or (exp, frac) tuples)
    for MXFP. ``act_w`` is the decode activation compute width (stored bf16 —
    accuracy-only, no HBM cost). ``fp_setting`` optionally quantises the vector
    unit (SiLU + RMSNorm) to a plain-minifloat ``(exp, frac)`` width; leaving it
    ``None`` keeps those ops unquantised so SDPA remains usable in both phases.
    """

    attn_w: Any = 4
    ffn_w: Any = 4
    kv: Any = 4
    w_fmt: str = "mxint"
    kv_fmt: str = "mxint"
    weight_block: int = 32
    kv_block: int = 32
    act_w: Any = 8
    act_fmt: str = "mxint"
    act_block: int = 32
    use_gptq: bool = False
    use_rotation: bool = False  # selective phase-aware rotation (decode-only) — subsumes GPTQ
    fp_setting: tuple[int, int] | None = None  # vector-unit minifloat (SiLU/RMSNorm, +softmax/rope if below)
    fp_setting_attention: bool = False  # extend FP_SETTING to softmax/rope (needs eager); else SDPA-safe
    quant_attn_internals: bool = False  # qk/av GEMM quantisation — needs eager attention

    def __post_init__(self):
        if self.gptq_weights and self.attn_w != self.ffn_w:
            raise ValueError(
                "Calibrated weights (GPTQ/rotation) use a single weight width per "
                f"pass, so attn_w must equal ffn_w (got {self.attn_w} vs {self.ffn_w}). "
                "Use RTN (use_gptq=use_rotation=False) for mixed attn/FFN weight precision."
            )

    @property
    def gptq_weights(self) -> bool:
        """Whether the decode weight bank comes from a calibrated pass (GPTQ or
        rotation, which runs GPTQ internally) rather than plain RTN."""
        return self.use_gptq or self.use_rotation

    @property
    def needs_eager(self) -> bool:
        """Quantising softmax/rope (FP_SETTING on attention) or the qk/av matmuls
        requires eager attention (SDPA hides those internals). FP_SETTING on just
        MLP+RMSNorm stays SDPA-friendly."""
        return (self.fp_setting is not None and self.fp_setting_attention) or self.quant_attn_internals

    @property
    def tag(self) -> str:
        method = "rot" if self.use_rotation else "gptq" if self.use_gptq else "rtn"
        fp = f"_fp{self.fp_setting[0]}-{self.fp_setting[1]}" if self.fp_setting else ""
        act = _wtok(self.act_fmt, self.act_w) if self.act_w is not None else "i16"
        return (
            f"{method}__aw-{_wtok(self.w_fmt, self.attn_w)}"
            f"__fw-{_wtok(self.w_fmt, self.ffn_w)}"
            f"__kv-{_wtok(self.kv_fmt, self.kv)}__a-{act}__b{self.weight_block}{fp}"
        )


# One 8-bit shared scale per MX block (E8M0)
_SCALE_BITS = 8


def eff_bits(fmt: str, width: Any, block: int) -> float:
    """Average stored bits per element including the per-block shared scale."""
    if fmt == "mxint":
        elem = int(width)
    else:
        e, m = _mxfp_exp_frac(width)
        elem = 1 + e + m
    return elem + _SCALE_BITS / block


def _wtok(fmt: str, width: Any) -> str:
    if fmt == "mxint":
        return f"i{int(width)}"
    if isinstance(width, str):
        return width
    return f"E{width[0]}M{width[1]}"


def _mxfp_exp_frac(width: Any) -> tuple[int, int]:
    """Parse an MXFP width token/tuple into (exponent_bits, frac_bits)."""
    if isinstance(width, str):
        e, m = width.upper().lstrip("E").split("M")
        return int(e), int(m)
    return int(width[0]), int(width[1])


def _weight_keys(fmt: str, width: Any, block: int) -> dict[str, Any]:
    """Weight quantiser keys for the given format."""
    if fmt == "mxint":
        return {"weight_block_size": block, "weight_width": int(width)}
    e, m = _mxfp_exp_frac(width)
    return {"weight_block_size": block, "weight_exponent_width": e, "weight_frac_width": m}


def _act_keys(fmt: str, width: Any, block: int) -> dict[str, Any]:
    """Activation (data_in) quantiser keys for the given format."""
    if fmt == "mxint":
        return {"data_in_block_size": block, "data_in_width": int(width)}
    e, m = _mxfp_exp_frac(width)
    return {"data_in_block_size": block, "data_in_exponent_width": e, "data_in_frac_width": m}


def _linear_decode(spec: DecodeQuantSpec, weight_width: Any) -> dict[str, Any]:
    """Decode config for one linear group (weights + optional activation quant)."""
    cfg: dict[str, Any] = dict(_weight_keys(spec.w_fmt, weight_width, spec.weight_block))
    if spec.act_w is not None:
        cfg.update(_act_keys(spec.act_fmt, spec.act_w, spec.act_block))
    if spec.gptq_weights:
        # The decode weight bank comes from GPTQ/rotation; don't RTN it again.
        cfg["gptq"] = True
    return cfg


def _fp_stage_keys(fp_setting: tuple[int, int]) -> dict[str, Any]:
    """Vector-unit minifloat width for one attention stage (softmax / rope)."""
    e, m = fp_setting
    return {"data_in_exponent_width": e, "data_in_frac_width": m}


def _attn_decode(spec: DecodeQuantSpec) -> dict[str, Any]:
    """Decode config for a self-attention block.

    - kv_cache: always quantised at the KV precision (streamed from HBM).
    - qk/av matmuls: MXINT at the KV width when ``quant_attn_internals`` (the
      GEMM-side attention quantisation); else bypassed (SDPA-friendly).
    - softmax/rope: the FP_SETTING vector-unit minifloat when ``fp_setting`` is
      set; else bypassed. Quantising these needs eager attention
    """
    qk = _act_keys(spec.kv_fmt, spec.kv, spec.kv_block) if spec.quant_attn_internals else {"bypass": True}
    use_fp_attn = spec.fp_setting is not None and spec.fp_setting_attention
    fp = _fp_stage_keys(spec.fp_setting) if use_fp_attn else {"bypass": True}
    return {
        "qk_matmul": dict(qk),
        "av_matmul": dict(qk),
        "softmax": dict(fp),
        "rope": dict(fp),
        "kv_cache": _act_keys(spec.kv_fmt, spec.kv, spec.kv_block),
    }


def _fp_setting_keys(fp_setting: tuple[int, int]) -> dict[str, Any]:
    """Plain-minifloat vector-unit widths (SiLU / RMSNorm), symmetric weight+act."""
    e, m = fp_setting
    return {
        "weight_exponent_width": e, "weight_frac_width": m,
        "data_in_exponent_width": e, "data_in_frac_width": m,
    }


def build_decode_pass_args(
    model_name: str,
    device: str,
    spec: DecodeQuantSpec,
    gptq_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the phase-split quantize ``pass_args`` for one decode precision.

    Each module carries only a ``decode`` bucket, so the normaliser bypasses
    prefill unquantised and sets ``decode_policy="quantized"`` automatically.
    """
    wname = spec.w_fmt  # "mxint" | "mxfp"
    kname = spec.kv_fmt

    pass_args: dict[str, Any] = {
        "by": "regex_name",
        _SELF_ATTN_RE: {
            "config": {"name": kname, "decode": _attn_decode(spec)},
        },
        _ATTN_LINEAR_RE: {
            "config": {"name": wname, "decode": _linear_decode(spec, spec.attn_w)},
        },
        _FFN_LINEAR_RE: {
            "config": {"name": wname, "decode": _linear_decode(spec, spec.ffn_w)},
        },
    }

    if spec.fp_setting is not None:
        fp = _fp_setting_keys(spec.fp_setting)
        pass_args[_MLP_RE] = {"config": {"name": "minifloat", "decode": dict(fp)}}
        pass_args[_RMSNORM_RE] = {"config": {"name": "minifloat", "decode": dict(fp)}}

    # Calibrated weights (GPTQ, or rotation which runs GPTQ internally) need the
    # gptq block: it carries the calibration setup and the checkpoint dir.
    if spec.gptq_weights:
        base_gptq = {
            "model_name": model_name,
            "device": device,
            "dataset": "wikitext2",
            "nsamples": 128,
            "seqlen": 2048,
            "format": wname,
            "weight_config": _weight_keys(wname, spec.attn_w, spec.weight_block),
            # GPTQ output becomes the DECODE weight bank; FP weights are
            # restored so prefill stays unquantised.
            "phase": "decode",
            "quantile_search": True,
            "clip_search_y": False,
        }
        if gptq_cfg:
            base_gptq.update(gptq_cfg)
        pass_args["gptq"] = base_gptq

    if spec.use_rotation:
        # Route the whole quantize step through phase-aware rotation search:
        # it does GPTQ + module replacement + per-matmul rotate tuning, scoring
        # every candidate in the decode phase. Prefill stays unquantised
        rot: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "calib_data": base_gptq.get("dataset", "wikitext2"),
            "calib_nsamples": 32,
            "calib_seqlen": 1024,
            "score_phase": "decode",
        }
        ckpt = base_gptq.get("checkpoint_dir")
        if ckpt:
            rot["cache_path"] = f"{ckpt}/rotation_decisions.json"
        if gptq_cfg and gptq_cfg.get("rotation"):
            rot.update(gptq_cfg["rotation"])
        pass_args["rotation_search"] = rot

    return pass_args


def gptq_cache_key(model_name: str, spec: DecodeQuantSpec, gptq_cfg: dict[str, Any] | None) -> str:
    """Stable key for the GPTQ decode weight bank.

    GPTQ depends only on the model, the weight format/width/block and the
    calibration setup — NOT on KV width, activation width or FP_SETTING. So one
    checkpoint serves every KV/activation variant at the same weight width,
    which is what makes a decode precision sweep cheap.
    """
    cfg = gptq_cfg or {}
    payload = "|".join(
        str(x)
        for x in (
            model_name,
            spec.w_fmt,
            spec.attn_w,  # == ffn_w in calibrated mode
            spec.weight_block,
            "rot" if spec.use_rotation else "gptq",  # rotation banks differ from plain GPTQ
            cfg.get("dataset", "wikitext2"),          # task-aligned calib -> distinct bank
            cfg.get("nsamples", 128),
            cfg.get("seqlen", 2048),
            "erry" if cfg.get("clip_search_y") else "noclip",  # Erry clip changes weights
        )
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
