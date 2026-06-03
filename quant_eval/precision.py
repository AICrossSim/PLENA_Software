"""Precision-token helpers for PLENA DSE/BFCL evaluation.

The old Coprocessor-for-Llama codesign stack used string precision tokens such
as ``MXINT_4`` and ``FP_E3M2``.  This module keeps the same public vocabulary and
maps it to the config dictionaries consumed by MASE/Chop quantized modules.

``MXFP_E3M4`` is intentionally rejected for now: the legacy simulator supports
it, but the current PLENA/Chop runtime support needs separate validation before
we include it in automated DSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


SUPPORTED_MXINT_BITS = {2, 3, 4, 8}
SUPPORTED_MXFP_FORMATS = {(1, 2), (2, 1), (4, 3), (5, 2)}
SUPPORTED_FP_SETTINGS = {(3, 2), (2, 3), (6, 5), (5, 6), (4, 7), (8, 5)}


@dataclass(frozen=True)
class PrecisionSpec:
    family: str
    bits: int | None = None
    exp: int | None = None
    frac: int | None = None

    @property
    def canonical(self) -> str:
        if self.family == "mxint":
            return f"MXINT_{self.bits}"
        if self.family == "mxfp":
            return f"MXFP_E{self.exp}M{self.frac}"
        if self.family == "fp":
            return f"FP_E{self.exp}M{self.frac}"
        raise ValueError(f"Unsupported precision family: {self.family!r}")


def _strip_suffix(token: str) -> str:
    # Accept legacy full preset names like MXINT_4_B16_S8 when a caller passes
    # them through directly; the block/scale are controlled separately here.
    return re.sub(r"_B\d+_S\d+$", "", token.strip().upper())


def parse_mx_precision(token: str) -> PrecisionSpec:
    value = _strip_suffix(token)
    m = re.fullmatch(r"MXINT_?(\d+)", value)
    if m:
        bits = int(m.group(1))
        if bits not in SUPPORTED_MXINT_BITS:
            raise ValueError(
                f"Unsupported MXINT precision {token!r}; supported bits are "
                f"{sorted(SUPPORTED_MXINT_BITS)}."
            )
        return PrecisionSpec(family="mxint", bits=bits)

    m = re.fullmatch(r"MXFP_E(\d+)M(\d+)", value)
    if m:
        exp, frac = int(m.group(1)), int(m.group(2))
        if (exp, frac) == (3, 4):
            raise ValueError("MXFP_E3M4 is intentionally unsupported in PLENA runtime DSE for now.")
        if (exp, frac) not in SUPPORTED_MXFP_FORMATS:
            allowed = ", ".join(f"MXFP_E{e}M{f}" for e, f in sorted(SUPPORTED_MXFP_FORMATS))
            raise ValueError(f"Unsupported MXFP precision {token!r}; supported formats are: {allowed}.")
        return PrecisionSpec(family="mxfp", exp=exp, frac=frac)

    raise ValueError(f"Unsupported MX precision token: {token!r}")


def parse_fp_setting(token: str) -> PrecisionSpec:
    value = _strip_suffix(token).replace("MINIFLOAT_", "FP_", 1)
    m = re.fullmatch(r"FP_E(\d+)M(\d+)", value)
    if not m:
        raise ValueError(f"Unsupported FP_SETTING token: {token!r}")
    exp, frac = int(m.group(1)), int(m.group(2))
    if (exp, frac) not in SUPPORTED_FP_SETTINGS:
        allowed = ", ".join(f"FP_E{e}M{f}" for e, f in sorted(SUPPORTED_FP_SETTINGS))
        raise ValueError(f"Unsupported FP_SETTING {token!r}; supported settings are: {allowed}.")
    return PrecisionSpec(family="fp", exp=exp, frac=frac)


def mx_module_name(spec: PrecisionSpec) -> str:
    if spec.family == "mxint":
        return "mxint"
    if spec.family == "mxfp":
        return "mxfp"
    raise ValueError(f"MX module precision required, got {spec.family!r}")


def _mx_common(spec: PrecisionSpec, prefix: str | None = None) -> dict:
    values = {"family": spec.family, "canonical": spec.canonical}
    if prefix is not None:
        values[f"{prefix}_family"] = spec.family
        values[f"{prefix}_canonical"] = spec.canonical
    return values


def mx_data_config(spec: PrecisionSpec, block_size: int) -> dict:
    if spec.family == "mxint":
        return {**_mx_common(spec, "data_in"), "data_in_width": spec.bits, "data_in_block_size": block_size}
    if spec.family == "mxfp":
        return {
            **_mx_common(spec, "data_in"),
            "data_in_exponent_width": spec.exp,
            "data_in_frac_width": spec.frac,
            "data_in_block_size": block_size,
        }
    raise ValueError(f"MX precision required, got {spec.family!r}")


def mx_weight_config(spec: PrecisionSpec, block_size: int) -> dict:
    if spec.family == "mxint":
        return {**_mx_common(spec, "weight"), "weight_width": spec.bits, "weight_block_size": block_size}
    if spec.family == "mxfp":
        return {
            **_mx_common(spec, "weight"),
            "weight_exponent_width": spec.exp,
            "weight_frac_width": spec.frac,
            "weight_block_size": block_size,
        }
    raise ValueError(f"MX precision required, got {spec.family!r}")


def mx_bias_config(spec: PrecisionSpec, block_size: int) -> dict:
    if spec.family == "mxint":
        return {**_mx_common(spec, "bias"), "bias_width": spec.bits, "bias_block_size": block_size}
    if spec.family == "mxfp":
        return {
            **_mx_common(spec, "bias"),
            "bias_exponent_width": spec.exp,
            "bias_frac_width": spec.frac,
            "bias_block_size": block_size,
        }
    raise ValueError(f"MX precision required, got {spec.family!r}")


def fp_data_config(spec: PrecisionSpec) -> dict:
    if spec.family != "fp":
        raise ValueError(f"FP_SETTING precision required, got {spec.family!r}")
    return {
        "data_in_exponent_width": spec.exp,
        "data_in_frac_width": spec.frac,
        "data_in_is_finite": True,
        "data_in_round_mode": "rn",
    }


def fp_weight_config(spec: PrecisionSpec) -> dict:
    if spec.family != "fp":
        raise ValueError(f"FP_SETTING precision required, got {spec.family!r}")
    return {
        "weight_exponent_width": spec.exp,
        "weight_frac_width": spec.frac,
        "weight_is_finite": True,
        "weight_round_mode": "rn",
    }


SUPPORTED_MODEL_FAMILIES = {"llama", "qwen3"}


def apply_dse_quant_config(
    pass_args: dict,
    *,
    act_precision: str,
    kv_precision: str,
    fp_setting: str,
    mx_block_size: int = 16,
    weight_precision: str = "MXINT_4",
    weight_block_size: int | None = None,
    model_family: str = "llama",
) -> dict[str, str]:
    """Patch pass_args with model-family DSE configs for one precision point.

    Llama and Qwen3 use the same HF submodule names for the projection, MLP,
    and RMSNorm paths that this eval stack quantizes.  Keep the model family
    explicit so future architectures can fail fast instead of silently applying
    Llama assumptions.
    """
    model_family = model_family.lower()
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_family={model_family!r}; expected one of {sorted(SUPPORTED_MODEL_FAMILIES)}.")

    weight_block_size = mx_block_size if weight_block_size is None else weight_block_size
    act = parse_mx_precision(act_precision)
    kv = parse_mx_precision(kv_precision)
    fp = parse_fp_setting(fp_setting)
    weight = parse_mx_precision(weight_precision)

    # Runtime mixed-family support is provided by quant_eval's unified MX
    # wrappers. During the Chop replacement pass, choose the attention wrapper
    # family from ACT and the Linear wrapper family from weight so initial
    # weight quantization still uses the requested weight format.

    attn_selector = r"model\.layers\.\d+\.self_attn$"
    attn_proj_selector = r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
    mlp_selector = r"model\.layers\.\d+\.mlp$"
    mlp_proj_selector = r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj"
    rms_selector = r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)$|model\.norm$"

    # Plain Qwen3 block instantiation is currently missing in Chop's
    # module_modify_helper.  Remove any block-level selectors from TOML and let
    # quant_eval install eval-side attention wrappers after Linear replacement.
    # Llama keeps using Chop block wrappers here for backward compatibility.
    pass_args.pop(attn_selector, None)
    pass_args.pop(mlp_selector, None)
    pass_args.pop(rms_selector, None)
    if model_family == "llama":
        pass_args[attn_selector] = {
            "config": {
                "name": mx_module_name(act),
                "qk_matmul": mx_data_config(act, mx_block_size),
                "av_matmul": mx_data_config(act, mx_block_size),
                "kv_cache": mx_data_config(kv, mx_block_size),
                "softmax": fp_data_config(fp),
                "rope": fp_data_config(fp),
            }
        }

    linear_cfg = {
        "name": mx_module_name(weight),
        **mx_weight_config(weight, weight_block_size),
        **mx_data_config(act, mx_block_size),
    }
    linear_with_bias_cfg = {**linear_cfg, **mx_bias_config(weight, weight_block_size)}
    pass_args[attn_proj_selector] = {"config": linear_with_bias_cfg}
    pass_args[mlp_proj_selector] = {"config": linear_cfg}

    if model_family == "llama":
        pass_args[mlp_selector] = {
            "config": {
                "name": mx_module_name(act),
                **fp_data_config(fp),
            }
        }

        pass_args[rms_selector] = {
            "config": {
                "name": "minifloat",
                **fp_data_config(fp),
                **fp_weight_config(fp),
            }
        }

    return {
        "ACT_ELEMENT_WIDTH": act.canonical,
        "KV_ELEMENT_WIDTH": kv.canonical,
        "FP_SETTING": fp.canonical,
        "MX_BLOCK_SIZE": str(mx_block_size),
        "WEIGHT_PRECISION": weight.canonical,
        "WEIGHT_BLOCK_SIZE": str(weight_block_size),
        "MODEL_FAMILY": model_family,
    }


def apply_llama_dse_quant_config(pass_args: dict, **kwargs) -> dict[str, str]:
    """Backward-compatible alias for legacy callers."""
    kwargs.setdefault("model_family", "llama")
    return apply_dse_quant_config(pass_args, **kwargs)
