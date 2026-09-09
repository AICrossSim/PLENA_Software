"""Decode-only precision bindings for the MASE phase-split quantize pass."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from decode_dse.profiles import (
    LOCAL_HEAD_ARGMAX_RULE,
    LOCAL_HEAD_LOCATION,
    LOCAL_HEAD_LOGITS_FORMAT,
    LOCAL_HEAD_OUTPUT_RULE,
    LOCAL_HEAD_SCHEMA,
    LOCAL_HEAD_WEIGHT_METHOD,
    local_head_matrix_family_contract,
)

# Attention vs FFN projections, split so each can carry an independent weight
# width.  The FFN selector covers both dense projection modules and the fused
# Qwen3-MoE expert container; the router remains outside quantization.
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
_FFN_LINEAR_RE = (
    r"model\.layers\.\d+\.mlp\."
    r"(gate_proj|up_proj|down_proj|experts)$"
)
_SELF_ATTN_RE = r"model\.layers\.\d+\.self_attn$"
_MLP_RE = r"model\.layers\.\d+\.mlp$"
_DECODER_LAYER_RE = r"model\.layers\.\d+$"
_LM_HEAD_RE = r"lm_head$"
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

    @property
    def local_head_contract(self) -> dict[str, Any]:
        """Return the decode-chip LM-head binding derived from global W/A."""

        weight_format = _format_token(self.w_fmt, self.attn_w)
        activation_format = _format_token(self.act_fmt, self.act_w)
        matrix_storage_format = _vector_token(self.fp_setting or "BF16")
        family_contract = local_head_matrix_family_contract(
            weight_format,
            activation_format,
            matrix_storage_format,
        )
        return {
            "schema_version": LOCAL_HEAD_SCHEMA,
            "location": LOCAL_HEAD_LOCATION,
            "target_module": "lm_head",
            "tie_word_embeddings_required": False,
            "weight_format": weight_format,
            "activation_format": activation_format,
            "weight_format_source": "profile.weight_format",
            "activation_format_source": "profile.activation_format",
            "weight_method": LOCAL_HEAD_WEIGHT_METHOD,
            # The sealed rotation ablation covers attention projections and
            # attention stages only.  The local head remains the source
            # profile's independently packed RTN bank for every refinement.
            "weight_preconditioning": "none",
            "block_size": self.weight_block,
            "scale_format": "E8M0",
            "scale_bits": _SCALE_BITS,
            "accumulator_rule": _MATRIX_SEMANTICS["accumulator_rule"],
            "operand_family_binding": family_contract[
                "operand_family_binding"
            ],
            "operand_family_deployment_supported": family_contract[
                "operand_family_deployment_supported"
            ],
            "hardware_bit_parity_verified": family_contract[
                "hardware_bit_parity_verified"
            ],
            "numerical_oracle_rule": family_contract[
                "numerical_oracle_rule"
            ],
            "partial_conversion_rule": family_contract[
                "partial_conversion_rule"
            ],
            "partial_conversion_format": matrix_storage_format,
            "mlen_partial_conversion_rounding": "round_to_nearest_even",
            "cross_instruction_accumulation": _MATRIX_SEMANTICS[
                "mxint_cross_instruction_accumulation"
            ],
            "matrix_instruction_k_partition": "MLEN",
            "matrix_mlen": self.matrix_mlen,
            "output_rule": LOCAL_HEAD_OUTPUT_RULE,
            "matrix_semantics_output_rule": _MATRIX_SEMANTICS["output_rule"],
            "arithmetic_chain": family_contract["arithmetic_chain"],
            "matrix_storage_format": matrix_storage_format,
            "matrix_output_format": matrix_storage_format,
            "logit_container_format": LOCAL_HEAD_LOGITS_FORMAT,
            "final_logits_format": LOCAL_HEAD_LOGITS_FORMAT,
            "bf16_container_precision_recovery": False,
            "greedy_selection_rule": LOCAL_HEAD_ARGMAX_RULE,
            "offline_evaluation": {
                "materialization": "full_bf16_logits",
                "purpose": "teacher_forced_nll_and_accuracy",
            },
            "serving_selection": {
                "materialization": "tiled_bf16_logits",
                "full_batch_vocab_vsram_required": False,
                "running_state": "top_k20_and_argmax_lowest_token_id",
                "sample_probability_dtype": "FP32",
                "top_k": 20,
                "top_p": 0.95,
                "min_p": 0.0,
                "sampling_parameters_source": "publication_protocol",
            },
            "phase_ownership": {
                "first_token_owner": "prefill",
                "prefill_head_policy": "bf16",
                "decode_query_length": 1,
                "decode_head_policy": "profile_mx_matrix",
            },
        }


@dataclass(frozen=True)
class DecodeBindingExpectations:
    """Architecture-derived module counts for a sealed decode weight bank."""

    pattern_counts: tuple[int, int, int, int, int, int, int]
    binding_targets: int
    sealed_weight_modules: int
    dense_layers: int
    moe_layers: int


def decode_binding_expectations(
    model_architecture: dict[str, Any],
) -> DecodeBindingExpectations:
    """Return dense/MoE binding counts without assuming seven linears/layer."""

    layers = int(model_architecture["num_hidden_layers"])
    experts = int(model_architecture.get("num_experts", 1))
    if experts > 1:
        sparse_step = int(model_architecture.get("decoder_sparse_step", 1))
        dense_only = {
            int(index) for index in model_architecture.get("mlp_only_layers", ())
        }
        if sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if any(index < 0 or index >= layers for index in dense_only):
            raise ValueError("mlp_only_layers contains an invalid layer index")
        moe_layers = sum(
            index not in dense_only and (index + 1) % sparse_step == 0
            for index in range(layers)
        )
    else:
        moe_layers = 0
    dense_layers = layers - moe_layers
    ffn_targets = 3 * dense_layers + moe_layers
    qk_norm = bool(
        model_architecture.get(
            "use_qk_norm",
            str(model_architecture.get("model_type", "")).startswith("qwen3"),
        )
    )
    rmsnorms = layers * (4 if qk_norm else 2) + 1
    pattern_counts = (
        layers,
        4 * layers,
        ffn_targets,
        layers,
        layers,
        rmsnorms,
        1,
    )
    return DecodeBindingExpectations(
        pattern_counts=pattern_counts,
        binding_targets=sum(pattern_counts),
        sealed_weight_modules=4 * layers + ffn_targets + 1,
        dense_layers=dense_layers,
        moe_layers=moe_layers,
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


def _format_token(fmt: str, width: Any) -> str:
    """Return the canonical profile token for one matrix operand."""

    if width is None:
        return "BF16"
    if fmt == "mxint":
        return f"MXINT{int(width)}"
    e, m = _mxfp_exp_frac(width)
    return f"E{e}M{m}"


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


def _local_head_decode(spec: DecodeQuantSpec) -> dict[str, Any]:
    """Bind the untied local head to W/A while preserving BF16 logits."""

    if spec.act_w is None:
        raise ValueError("the local decode head requires an MX activation format")
    if spec.weight_block != spec.act_block:
        raise ValueError("local-head W/A operands require one shared MX block size")
    cfg: dict[str, Any] = dict(
        _weight_keys(spec.w_fmt, spec.attn_w, spec.weight_block)
    )
    cfg.update(_act_keys(spec.act_fmt, spec.act_w, spec.act_block))
    cfg.update(
        _matrix_contract(
            spec.act_fmt,
            spec.w_fmt,
            spec.fp_setting or "BF16",
            block_sizes=(spec.weight_block, spec.act_block),
            matrix_mlen=spec.matrix_mlen,
        )
    )
    cfg["local_head_contract"] = spec.local_head_contract
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

    # The untied head follows the body W/A/vector formats and stores its
    # rounded matrix outputs in the BF16 logit container.
    pass_args[_LM_HEAD_RE] = {
        "config": {"name": wname, "decode": _local_head_decode(spec)},
    }

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
