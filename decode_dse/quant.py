"""Working Model, quantization-recipe to MASE, co-design search space and Memory bandwidth estimation

Construct MXINT/MXFP quantization pass-args dicts for per-axis (weight, activation, kv) precision

Per-axis precision in MASE:
  * A linear (q/k/v/o/gate/up/down_proj) picks ONE format via its
    ``name`` (mxint or mxfp); within it, weight width and input-activation width
    are independent -> WEIGHT and the linear's ACT share a format family.
  * The attention block picks ONE format via its ``name``; within it qk/av-matmul
    (ACT) and kv_cache (KV) widths are independent -> attention ACT and KV
    share a format family.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path


MODEL_NAME = "unsloth/Llama-3.2-1B"

# The rotation/GPTQ loads any saved decisions without checking the width/block,
# so each distinct precision MUST get its own directory or results leak across
# configs. Paths derive from this package dir, so they survive folder renames.
_CKPT = Path(__file__).resolve().parent / "checkpoints"
GPTQ_VALIDATED_DIR    = str(_CKPT / "w4_mxint_gptq_rot")
GPTQ_SEARCH_CKPT_ROOT = str(_CKPT / "search")

# Architecture of Llama-3.2-1B
DIMS = {
    "hidden_size": 2048,        # d_model — size of each token's embedding
    "num_layers": 16,           # number of transformer blocks
    "num_attention_heads": 32,  # Q heads in MHA
    "num_kv_heads": 8,          # K/V heads (Grouped Query Attention, 32/8 = 4 Q per KV)
    "head_dim": 64,             # dimension per head (hidden_size / num_heads = 2048/32)
    "intermediate_size": 8192,  # MLP hidden dimension (4× hidden_size)
    "vocab_size": 128256,       # output vocabulary size
}

# Regex selectors
RE_ATTN_BLOCK = r"model\.layers\.\d+\.self_attn$" # Whole attention block (KV cache + matmuls)
RE_ATTN_PROJ  = r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj" # The Q/K/V/O projection linears inside attention
RE_MLP_PROJ   = r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj" # The gate/up/down projection linears in MLP


def build_static_pass_args(w: int, a: int, kv: int, block: int = 32) -> dict:
    """MXINT pass-args dict for uniform (weight, activation, kv) widths.

    Softmax and RoPE are left in full precision (bypass=True)
    """
    attn_block = {
        "name": "mxint",
        "qk_matmul": {"data_in_block_size": block, "data_in_width": a},
        "av_matmul": {"data_in_block_size": block, "data_in_width": a},
        "kv_cache":  {"data_in_block_size": block, "data_in_width": kv},
        "softmax":   {"bypass": True},
        "rope":      {"bypass": True},
    }
    linear = lambda: {
        "name": "mxint",
        "weight_block_size":  block, "weight_width":  w,
        "data_in_block_size": block, "data_in_width": a,
    }
    return {
        "by": "regex_name",
        RE_ATTN_BLOCK: {"config": attn_block},
        RE_ATTN_PROJ:  {"config": linear()},
        RE_MLP_PROJ:   {"config": linear()},
    }


def build_mxfp_pass_args(w, a, kv, block: int = 32) -> dict:
    """MXFP pass-args dict for (weight, activation, kv) formats.

    Each of w/a/kv is an (exponent_bits, fraction_bits) tuple. MXFP shares a block exponent like
    with each element as a tiny float (more dynamic range, better for
    outliers). Softmax and RoPE bypassed
    """
    (we, wf), (ae, af), (ke, kf) = w, a, kv
    act = {"data_in_block_size": block, "data_in_exponent_width": ae, "data_in_frac_width": af}
    attn_block = {
        "name": "mxfp",
        "qk_matmul": dict(act),
        "av_matmul": dict(act),
        "kv_cache":  {"data_in_block_size": block, "data_in_exponent_width": ke, "data_in_frac_width": kf},
        "softmax":   {"bypass": True},
        "rope":      {"bypass": True},
    }
    linear = lambda: {
        "name": "mxfp",
        "weight_block_size": block, "weight_exponent_width": we, "weight_frac_width": wf,
        **act,
    }
    return {
        "by": "regex_name",
        RE_ATTN_BLOCK: {"config": attn_block},
        RE_ATTN_PROJ:  {"config": linear()},
        RE_MLP_PROJ:   {"config": linear()},
    }


# =============================================================================
# Co-design search space  (Decode chip)
# =============================================================================
#
# Element bit budget:  MXINT = width;  MXFP = 1 sign + exp + frac
SCALE_BITS = 8 # All elements share one 8-bit scale

MXINT_WIDTHS = [2, 3, 4, 8]

# OCP - Open Compute Project
MXFP_FORMATS = {
    "E1M2": (1, 2),   # 4-bit
    "E2M1": (2, 1),   # 4-bit  (OCP MXFP4) - Industry standard MXFP4
    "E4M3": (4, 3),   # 8-bit  (OCP MXFP8) - Industry standard MXFP8
    "E5M2": (5, 2),   # 8-bit
}

BLOCK_SIZES = [16, 32, 64]


def width_label(fmt: str, width) -> str:
    """Human tag for one axis precision"""
    if fmt == "mxint":
        return f"MXINT{width}"
    e, f = width
    return f"MXFP_E{e}M{f}"


def element_bits(fmt: str, width) -> int:
    """Raw stored bits per element (no scale overhead)"""
    if fmt == "mxint":
        return int(width)
    e, f = width
    return 1 + int(e) + int(f)


def effective_bits(fmt: str, width, block: int = 32, scale_bits: int = SCALE_BITS) -> float:
    """Bits/element INCLUDING the amortised shared block-scale (block-format cost)
    
    It includes the overhead of sharing the shared scale used to calculate TPOT"""
    return element_bits(fmt, width) + scale_bits / block


def build_pass_args(fmt: str, w, a, kv, block: int = 32) -> dict:
    """Dispatch to the MXINT or MXFP builder for one (w, a, kv) precision.
    """
    if fmt == "mxint":
        return build_static_pass_args(int(w), int(a), int(kv), block)
    if fmt == "mxfp":
        return build_mxfp_pass_args(tuple(w), tuple(a), tuple(kv), block)
    raise ValueError(f"unknown fmt {fmt!r} (expected 'mxint' or 'mxfp')")


GPTQ_VALIDATED_SPEC = (4, 8, 8, 32)


def gptq_checkpoint_dir(w: int, a: int, kv: int, block: int) -> str:
    """W4/A8/KV8 block-32 was already validated so we can skip this, and for every
    other precision combination gets its own folder"""
    if (int(w), int(a), int(kv), int(block)) == GPTQ_VALIDATED_SPEC:
        return GPTQ_VALIDATED_DIR
    return f"{GPTQ_SEARCH_CKPT_ROOT}/mxint_w{w}_a{a}_kv{kv}_b{block}"


def build_gptq_pass_args(base_recipe: dict, w: int, a: int, kv: int, block: int = 32,
                         with_rotation: bool = True) -> dict:
    """MXINT + GPTQ(+rotation) pass-args at an arbitrary (w, a, kv) width.

    Takes the parsed `w4_mxint_gptq_rot.toml` recipe (via
    quant_eval.quantize.load_quant_config) for its [gptq] / [rotation_search]
    calibration blocks, then overlays the per-axis widths so GPTQ can be applied
    at any weight width -- not just the hard-coded 4-bit in the recipe file.
    GPTQ is MXINT-only (the calibration math assumes integer levels).
    """
    pa = build_static_pass_args(int(w), int(a), int(kv), block)
    pa["by"] = "regex_name"
    ckpt = gptq_checkpoint_dir(w, a, kv, block)
    # Carry over the calibration sections, overriding width/block + cache path.
    gptq = deepcopy(base_recipe.get("gptq", {}))
    gptq["checkpoint_dir"] = ckpt
    gptq.setdefault("weight_config", {})
    gptq["weight_config"]["weight_width"] = int(w)
    gptq["weight_config"]["weight_block_size"] = int(block)
    pa["gptq"] = gptq
    if with_rotation and "rotation_search" in base_recipe:
        rot = deepcopy(base_recipe["rotation_search"])
        rot["cache_path"] = str(Path(ckpt) / "rotation_decisions.json")
        pa["rotation_search"] = rot
    # Flag the linears for GPTQ calibration (attention block is not GPTQ'd).
    for re_key in (RE_ATTN_PROJ, RE_MLP_PROJ):
        pa[re_key]["config"]["gptq"] = True
    return pa


# =============================================================================
# Decode cost proxy
# =============================================================================
# Decode is dominated by streaming weights from HBM every step and reading the
# growing KV cache, "how expensive is this precision" by the bytes a
# single decode token moves: all linear weights once + the KV cache at the
# average context length

REF_BATCH = 16
REF_S_IN = 4800 # "Summarise" workload
REF_S_OUT = 8000


def _weight_elements() -> int:
    """Total decode-streamed weight elements per step (all linears + lm_head)"""
    d = DIMS
    h, kvh, hd = d["hidden_size"], d["num_kv_heads"], d["head_dim"]
    inter, vocab, L = d["intermediate_size"], d["vocab_size"], d["num_layers"]
    q = h * (d["num_attention_heads"] * hd)
    k = v = h * (kvh * hd)
    o = (d["num_attention_heads"] * hd) * h
    gate = up = h * inter
    down = inter * h
    per_layer = q + k + v + o + gate + up + down
    return per_layer * L + h * vocab # + lm_head (streamed each step)


def _kv_elements(context: int, batch: int) -> int:
    """KV-cache elements (K and V, all layers) at a given context length."""
    d = DIMS
    return 2 * d["num_kv_heads"] * d["head_dim"] * d["num_layers"] * context * batch


def decode_cost(fmt: str, w, a, kv, block: int = 32,
                batch: int = REF_BATCH, s_in: int = REF_S_IN, s_out: int = REF_S_OUT) -> dict:
    """Per-axis effective bits + a single decode byte-traffic proxy (MB/token).

    Returns a dict with eff bits for each axis, the weight/KV byte split, and
    `cost_mb_per_token` -- the scalar the Optuna search minimises against PPL.
    """
    w_bits = effective_bits(fmt, w, block)
    a_bits = effective_bits(fmt, a, block)
    kv_bits = effective_bits(fmt, kv, block)

    avg_context = s_in + s_out // 2 # KV grows over the decode loop
    weight_bytes = _weight_elements() * w_bits / 8
    kv_bytes = _kv_elements(avg_context, batch) * kv_bits / 8
    cost_mb = (weight_bytes + kv_bytes) / 1e6

    return {
        "w_eff_bits": round(w_bits, 4),
        "a_eff_bits": round(a_bits, 4),
        "kv_eff_bits": round(kv_bits, 4),
        "weight_MB_per_token": round(weight_bytes / 1e6, 3),
        "kv_MB_per_token": round(kv_bytes / 1e6, 3),
        "cost_mb_per_token": round(cost_mb, 3),
    }
