"""Decode-only precision bindings for the MASE phase-split quantize pass."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# Attention vs FFN projections, split so each can carry an independent weight
# width. Regexes match Llama and Qwen3 dense naming.
def parse_prec_token(tok: Any) -> tuple[str, Any]:
    """Parse a precision token into (format, width).

    ``4`` or ``"MXINT4"`` -> ("mxint", 4); ``"E2M1"`` -> ("mxfp", (2, 1)).
    Width 16 (or ``None``) means unquantised -> ("mxint", None).
    """
    def parse_integer_width(value: int) -> tuple[str, int | None]:
        if value == 16:
            return ("mxint", None)
        if value not in (2, 4, 8):
            raise ValueError("MXINT width must be 2, 4, 8, or unquantized 16")
        return ("mxint", value)

    if tok is None:
        return ("mxint", None)
    if isinstance(tok, int):
        return parse_integer_width(tok)
    s = str(tok).upper()
    if s.startswith("MXINT"):
        return parse_integer_width(int(s[5:]))
    if s.startswith("E") and "M" in s:
        e, m = s[1:].split("M")
        return ("mxfp", (int(e), int(m)))
    if s.isdigit():
        return parse_prec_token(int(s))
    raise ValueError(
        f"unrecognised precision token {tok!r} "
        "(use e.g. 4, 'MXINT4', 'E2M1')."
    )


_ATTN_LINEAR_RE = r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
_FFN_LINEAR_RE = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"
_SELF_ATTN_RE = r"model\.layers\.\d+\.self_attn$"
_MLP_RE = r"model\.layers\.\d+\.mlp$"
_DECODER_LAYER_RE = r"model\.layers\.\d+$"
_RMSNORM_RE = (
    r"(model\.layers\.\d+\.(input_layernorm|post_attention_layernorm|"
    r"self_attn\.(q_norm|k_norm))|model\.norm)$"
)


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
    key_kv: Any | None = None
    value_kv: Any | None = None
    key_kv_fmt: str | None = None
    value_kv_fmt: str | None = None
    weight_block: int = 8
    kv_block: int = 8
    act_w: Any = 8
    act_fmt: str = "mxint"
    act_block: int = 8
    use_gptq: bool = False
    use_rotation: bool = False  # selective phase-aware rotation (decode-only) — subsumes GPTQ
    fp_setting: str | tuple[int, int] | None = "BF16"
    fp_setting_attention: bool = True
    quant_attn_internals: bool = True
    matrix_mlen: int = 1024

    def __post_init__(self):
        def validate_integer(role: str, fmt: str | None, width: Any) -> None:
            if fmt != "mxint" or width is None:
                return
            value = int(width)
            if value not in (2, 4, 8) and value < 16:
                raise ValueError(
                    f"{role} MXINT width must be 2, 4, 8, or unquantized"
                )

        if self.attn_w != self.ffn_w:
            raise ValueError(
                "attention and FFN must share one global weight precision"
            )
        validate_integer("weight", self.w_fmt, self.attn_w)
        validate_integer("activation", self.act_fmt, self.act_w)
        validate_integer("KV", self.kv_fmt, self.kv)
        for name, block in (
            ("weight_block", self.weight_block),
            ("act_block", self.act_block),
            ("kv_block", self.kv_block),
        ):
            if block not in (8, 16, 32):
                raise ValueError(f"{name} must be 8, 16, or 32")
        split_values = (
            self.key_kv,
            self.value_kv,
            self.key_kv_fmt,
            self.value_kv_fmt,
        )
        if any(value is not None for value in split_values) and not all(
            value is not None for value in split_values
        ):
            raise ValueError(
                "split KV precision requires key/value formats and widths"
            )
        if all(value is not None for value in split_values):
            validate_integer("key KV", self.key_kv_fmt, self.key_kv)
            validate_integer("value KV", self.value_kv_fmt, self.value_kv)
        if self.fp_setting is not None:
            _vector_token(self.fp_setting)
        if self.fp_setting is not None and not self.fp_setting_attention:
            raise ValueError("vector precision must cover qk_norm, RoPE, and softmax")
        if self.act_w is not None and not self.quant_attn_internals:
            raise ValueError("activation precision must cover QK and PV inputs")
        if (
            isinstance(self.matrix_mlen, bool)
            or not isinstance(self.matrix_mlen, int)
            or self.matrix_mlen <= 0
            or self.matrix_mlen % 8
        ):
            raise ValueError("matrix_mlen must be a positive multiple of 8")

    @property
    def gptq_weights(self) -> bool:
        """Whether the decode weight bank comes from a calibrated pass (GPTQ or
        rotation, which runs GPTQ internally) rather than plain RTN."""
        return self.use_gptq or self.use_rotation

    @property
    def needs_eager(self) -> bool:
        """Return whether the configured decode semantics require eager attention."""
        return self.act_w is not None or self.fp_setting is not None

    @property
    def split_kv(self) -> bool:
        """Return whether K and V use explicit role-specific formats."""

        return self.key_kv is not None

    @property
    def key_kv_precision(self) -> tuple[str, Any]:
        """Return the K-cache format and width."""

        if self.split_kv:
            return str(self.key_kv_fmt), self.key_kv
        return self.kv_fmt, self.kv

    @property
    def value_kv_precision(self) -> tuple[str, Any]:
        """Return the V-cache format and width."""

        if self.split_kv:
            return str(self.value_kv_fmt), self.value_kv
        return self.kv_fmt, self.kv

    @property
    def tag(self) -> str:
        method = "rot" if self.use_rotation else "gptq" if self.use_gptq else "rtn"
        if isinstance(self.fp_setting, tuple):
            fp_token = f"E{self.fp_setting[0]}M{self.fp_setting[1]}"
        else:
            fp_token = self.fp_setting
        fp = f"_fp-{fp_token}" if fp_token else ""
        act = _wtok(self.act_fmt, self.act_w) if self.act_w is not None else "i16"
        key_fmt, key_width = self.key_kv_precision
        value_fmt, value_width = self.value_kv_precision
        kv = (
            f"__k-{_wtok(key_fmt, key_width)}__v-{_wtok(value_fmt, value_width)}"
            if self.split_kv
            else f"__kv-{_wtok(self.kv_fmt, self.kv)}"
        )
        return (
            f"{method}__aw-{_wtok(self.w_fmt, self.attn_w)}"
            f"__fw-{_wtok(self.w_fmt, self.ffn_w)}"
            f"{kv}__a-{act}__b{self.weight_block}{fp}"
        )


# One 8-bit shared scale per MX block (E8M0)
_SCALE_BITS = 8
_MATRIX_SEMANTICS = {
    "schema_version": "plena-matrix-semantics",
    "block_size": 8,
    "mxint_rule": "block8_range_safe_scale_widened_mac_max_shift16_rne_vector",
    "mxint_max_shift": 16,
    "mxint_vector_rounding": "round_to_nearest_even",
    "mxint_partial_conversion": (
        "per_mm_ic_integer_reduction_to_vector_storage_fp"
    ),
    "mxint_cross_instruction_accumulation": (
        "signed_fixed16_16_wraparound"
    ),
    "mxfp_rule": "product_cast_to_m_fp_then_fixed16_16_bank",
    "m_fp_format_binding": "profile.vector_format",
    "matrix_storage_fp_binding": "profile.vector_format",
    "matrix_instruction_k_partition": "MLEN",
    "qk_logical_k_partition": "HLEN",
    "fixed_accumulator_integer_bits": 16,
    "fixed_accumulator_fraction_bits": 16,
    "accumulator_rule": "plena_fixed16_16_accumulate_truncate",
    "output_rule": "truncate_to_vector_format",
    "mixed_family_rule": "deployment_unsupported_without_trace_evidence",
    "mixed_family_deployment_supported": False,
}


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
    cfg.update(
        _matrix_contract(
            spec.act_fmt if spec.act_w is not None else "bf16",
            spec.w_fmt,
            spec.fp_setting or "BF16",
            block_sizes=(
                spec.weight_block,
                spec.act_block if spec.act_w is not None else spec.weight_block,
            ),
            matrix_mlen=spec.matrix_mlen,
        )
    )
    if spec.gptq_weights:
        # The decode weight bank comes from GPTQ/rotation; don't RTN it again.
        cfg["gptq"] = True
    return cfg


def _vector_token(fp_setting: str | tuple[int, int]) -> str:
    """Return the canonical vector-format token."""
    if isinstance(fp_setting, tuple):
        return f"FP_E{int(fp_setting[0])}M{int(fp_setting[1])}"
    token = str(fp_setting).upper()
    if token == "BF16":
        return token
    if token.startswith("FP_E") and "M" in token:
        return token
    if token.startswith("E") and "M" in token:
        return f"FP_{token}"
    raise ValueError(f"unsupported vector format {fp_setting!r}")


def _fp_stage_keys(fp_setting: str | tuple[int, int]) -> dict[str, Any]:
    """Return one vector-stage precision config."""
    return {
        "format": _vector_token(fp_setting),
        "data_in_is_finite": False,
        "data_in_round_mode": "rn",
    }


def _matrix_contract(
    left_family: str,
    right_family: str,
    output_format: str | tuple[int, int],
    *,
    block_sizes: tuple[int, ...],
    matrix_mlen: int,
) -> dict[str, Any]:
    """Return the strict matrix-output oracle contract for one operand pair."""
    unique_blocks = set(block_sizes)
    if len(unique_blocks) != 1:
        raise ValueError("matrix operands must use one shared MX block size")
    quantization_block_size = unique_blocks.pop()
    if quantization_block_size not in (8, 16, 32):
        raise ValueError("matrix quantization block must be 8, 16, or 32")
    family_pair = (left_family, right_family)
    if family_pair == ("mxint", "mxint"):
        family_binding = _MATRIX_SEMANTICS["mxint_rule"]
    elif family_pair == ("mxfp", "mxfp"):
        family_binding = _MATRIX_SEMANTICS["mxfp_rule"]
    else:
        family_binding = _MATRIX_SEMANTICS["mixed_family_rule"]
    return {
        "accumulator_rule": _MATRIX_SEMANTICS["accumulator_rule"],
        "output_rule": _MATRIX_SEMANTICS["output_rule"],
        "output_format": _vector_token(output_format),
        "operand_family_binding": family_binding,
        "matrix_semantics": dict(_MATRIX_SEMANTICS),
        "quantization_block_size": quantization_block_size,
        "native_datapath": quantization_block_size == 8,
        "numerical_trace_conformance": "not_run",
        "matrix_oracle_scope": "instruction_partitioned_numerical_oracle",
        "matrix_mlen": matrix_mlen,
    }


def _attn_decode(spec: DecodeQuantSpec) -> dict[str, Any]:
    """Bind A to attention inputs and KV only to cached matrix operands."""
    matrix_input = (
        _act_keys(spec.act_fmt, spec.act_w, spec.act_block)
        if spec.act_w is not None
        else {"bypass": True}
    )
    fp = (
        _fp_stage_keys(spec.fp_setting)
        if spec.fp_setting is not None
        else {"bypass": True}
    )
    key_fmt, key_width = spec.key_kv_precision
    value_fmt, value_width = spec.value_kv_precision

    def matrix_config(cache_format: str) -> dict[str, Any]:
        config = dict(matrix_input)
        config.update(
            _matrix_contract(
                spec.act_fmt,
                cache_format,
                spec.fp_setting or "BF16",
                block_sizes=(spec.act_block, spec.kv_block),
                matrix_mlen=spec.matrix_mlen,
            )
        )
        return config

    if spec.split_kv:
        kv_cache = {
            "key": _act_keys(key_fmt, key_width, spec.kv_block),
            "value": _act_keys(value_fmt, value_width, spec.kv_block),
        }
    else:
        kv_cache = _act_keys(spec.kv_fmt, spec.kv, spec.kv_block)
    return {
        "qk_norm": dict(fp),
        "qk_matmul": matrix_config(key_fmt),
        "av_matmul": matrix_config(value_fmt),
        "softmax": dict(fp),
        "rope": dict(fp),
        "kv_cache": kv_cache,
    }


def _fp_setting_keys(
    fp_setting: str | tuple[int, int],
) -> dict[str, Any]:
    """Return vector config shared by norms, gates, and residuals."""
    config = _fp_stage_keys(fp_setting)
    config.update(
        {
            "weight_is_finite": False,
            "weight_round_mode": "rn",
        }
    )
    return config


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
    pass_args: dict[str, Any] = {
        "by": "regex_name",
        _SELF_ATTN_RE: {
            "config": {
                "name": spec.act_fmt,
                "kv_cache_handoff": "fp",
                "decode": _attn_decode(spec),
            },
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
        pass_args[_DECODER_LAYER_RE] = {
            "config": {"name": "minifloat", "decode": dict(fp)}
        }
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
