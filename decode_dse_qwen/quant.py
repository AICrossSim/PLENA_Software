"""Precision language for the Qwen3-32B decode chip: the search space, the MASE pass-args that
realise a precision, and the decode-cost proxy the search minimises.

Per PLENA's datapath, only the HBM-resident tensors are quantised -- WEIGHTS and the KV CACHE;
ACTIVATIONS stay high-precision FP (bf16) on-chip (they are the most quantisation-sensitive). So a
precision point is per-component over weights + KV:

    spec = {block,            # group size for the shared scale
            w_fmt,            # weight format (mxint|mxfp), shared by attn + ffn weights
            attn_w,           # q/k/v/o projection weight width
            ffn_w,            # gate/up/down projection weight width
            kv_fmt, kv}       # KV-cache format + width -- INDEPENDENT of the weights (mixed precision)
    widths are int (MXINT) or (exp,frac) tuples (MXFP). Activations are NOT quantised. The weight and
    KV formats are decoupled so the search can mix them (e.g. MXINT weights + MXFP KV), as PLENA allows.

The attention is quantised in two pieces because chop can't replace a plain Qwen3 attention block
(its dispatch has no `is_qwen3` branch): the LINEARS (q/k/v/o/gate/up/down) are weight-quantised by the
chop regex pass below, and the KV cache is wrapped in place by `disagg_serve.install_attention_quant`
using the q-config from `attn_qconfig` here (qk/av matmuls bypassed -> high-precision FP).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
GPTQ_RECIPE = _HERE / "configs" / "qwen3_mxint_gptq_rot.toml"
_CKPT_ROOT = str(_HERE / "checkpoints" / "search")

# chop regex selectors (standard HF Qwen3 layer naming). No self_attn-block selector exists
# chop cannot instantiate a Qwen3 attention block; the attention is wrapped separately.
RE_ATTN_PROJ = r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
RE_FFN_PROJ = r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj"

SCALE_BITS = 8  # every micro-scaling block carries one 8-bit shared scale (E8M0)

# search space
MXINT_WIDTHS = [2, 3, 4, 8, 16]
MXFP_FORMATS = {            # OCP micro-scaling float element formats
    "E2M1": (2, 1),         # 4-bit (OCP MXFP4)
    "E1M2": (1, 2),         # 4-bit
    "E4M3": (4, 3),         # 8-bit (OCP MXFP8)
    "E5M2": (5, 2),         # 8-bit
}
BLOCK_SIZES = [16, 32, 64]


def width_label(fmt: str, width) -> str:
    if fmt == "mxint":
        return f"MXINT{int(width)}"
    if fmt == "mxfp":
        return "MXFP_E{}M{}".format(*width)
    return "bf16"   # unquantised gold (fmt='bf16' = prefill precision, 16-bit)


def spec_tag(spec: dict) -> str:
    """One-line human label, e.g. 'attnW:MXINT4 ffnW:MXINT3 kv:MXFP_E4M3 b32 +gptq (act bf16)'."""
    g = "+gptq" if spec.get("gptq") else ""
    g += "+erry" if spec.get("clip_search_y") else ""
    g += "+rot" if spec.get("rotation") else ""
    wf, kf = spec["w_fmt"], spec["kv_fmt"]
    return (f"attnW:{width_label(wf, spec['attn_w'])} ffnW:{width_label(wf, spec['ffn_w'])} "
            f"kv:{width_label(kf, spec['kv'])} b{spec['block']}{g} (act bf16)")


def element_bits(fmt: str, width) -> int:
    """Stored bits per element, ignoring the shared block scale."""
    if fmt == "mxint":
        return int(width)
    if fmt == "mxfp":
        return 1 + int(width[0]) + int(width[1])
    return 16   # unquantised bf16 gold


def effective_bits(fmt: str, width, block: int) -> float:
    """Bits/element including the amortised 8-bit block scale (true block-format storage cost)."""
    if fmt not in ("mxint", "mxfp"):
        return 16.0   # unquantised bf16 gold -- no block scale
    return element_bits(fmt, width) + SCALE_BITS / block


# MASE pass-args
def _linear_cfg(fmt: str, w, block: int, gptq: bool = False) -> dict:
    """chop WEIGHT-ONLY quantised-linear config. Per PLENA (datapath characteristic i), activations
    stay high-precision FP on-chip, so we OMIT the `data_in_*` keys -- chop's linear forward then skips
    activation quantisation and runs F.linear on the bf16 (high-precision FP) activation. Weight only."""
    if fmt == "mxint":
        cfg = {"name": "mxint", "weight_block_size": block, "weight_width": int(w)}
    else:
        we, wf = w
        cfg = {"name": "mxfp", "weight_block_size": block,
               "weight_exponent_width": we, "weight_frac_width": wf}
    if gptq:
        cfg["gptq"] = True
    return cfg


def build_pass_args(spec: dict, gptq: bool = False) -> dict:
    """chop linear pass-args: attention and FFN weights share the weight format `w_fmt` but keep their
    OWN width (per-component). KV format is independent (`kv_fmt`, handled in attn_qconfig)."""
    wf = spec["w_fmt"]
    return {
        "by": "regex_name",
        RE_ATTN_PROJ: {"config": _linear_cfg(wf, spec["attn_w"], spec["block"], gptq)},
        RE_FFN_PROJ:  {"config": _linear_cfg(wf, spec["ffn_w"], spec["block"], gptq)},
    }


def attn_qconfig(spec: dict) -> dict:
    """q-config for the in-place attention wrapper. Per PLENA, only the KV CACHE (HBM-resident) is
    quantised -- in its OWN format `kv_fmt` (independent of the weights). The qk/av matmul activations
    stay high-precision FP -> bypassed; softmax/RoPE bypassed."""
    fmt, block, kv = spec["kv_fmt"], spec["block"], spec["kv"]
    if fmt == "mxint":
        kvc = {"data_in_block_size": block, "data_in_width": int(kv)}
    else:
        ke, kf = kv
        kvc = {"data_in_block_size": block, "data_in_exponent_width": ke, "data_in_frac_width": kf}
    return {"qk_matmul": {"bypass": True}, "av_matmul": {"bypass": True}, "kv_cache": kvc,
            "softmax": {"bypass": True}, "rope": {"bypass": True}}


def build_gptq_pass_args(recipe: dict, spec: dict) -> dict:
    """MXINT + GPTQ (optionally +Erry clip via clip_search_y, +rotation) pass-args for `spec`
    """
    assert spec["w_fmt"] == "mxint", "GPTQ is MXINT-only (the paper's de-facto for weight quant)"
    assert spec["attn_w"] == spec["ffn_w"], \
        "GPTQ applies one global weight width to all layers, so attn_w must equal ffn_w"
    pa = build_pass_args(spec, gptq=True)
    ckpt = f"{_CKPT_ROOT}/{spec_tag(spec).replace(' ', '_').replace('/', '-')}"

    gptq = deepcopy(recipe.get("gptq", {}))
    gptq["checkpoint_dir"] = ckpt
    gptq.setdefault("weight_config", {})
    gptq["weight_config"]["weight_width"] = int(spec["attn_w"])   # uniform width (== ffn_w)
    gptq["weight_config"]["weight_block_size"] = int(spec["block"])
    if spec.get("clip_search_y") is not None:
        gptq["clip_search_y"] = bool(spec["clip_search_y"])
    if spec.get("calib_file"):
        cf = spec["calib_file"]
        gptq["dataset"] = cf if cf.startswith("file:") else f"file:{cf}"
    pa["gptq"] = gptq

    if spec.get("rotation") and "rotation_search" in recipe:
        rot = deepcopy(recipe["rotation_search"])
        rot["cache_path"] = str(Path(ckpt) / "rotation_decisions.json")
        if spec.get("calib_file"):
            rot["dataset"] = gptq["dataset"]
        pa["rotation_search"] = rot
    return pa


# decode-cost proxy
# Decode bandwidth = stream every linear weight once per step + read the growing KV cache. Weights
# are split attention/FFN because they carry different precisions; lm_head stays unquantised (no
# selector matches it).
def _component_elements(dims: dict) -> tuple[int, int, int]:
    """(attention, FFN, lm_head) weight-element counts streamed per decode step."""
    h, ah, kvh, hd = dims["hidden"], dims["heads"], dims["kv_heads"], dims["head_dim"]
    inter, vocab, layers = dims["inter"], dims["vocab"], dims["layers"]
    attn = (h * ah * hd + 2 * h * kvh * hd + ah * hd * h) * layers          # q,k,v,o
    ffn = (2 * h * inter + inter * h) * layers                             # gate,up,down
    head = h * vocab                                                       # lm_head (unquantised)
    return attn, ffn, head


def decode_cost(dims: dict, spec: dict, *, batch: int, s_in: int, s_out: int) -> dict:
    """Per-component effective bits + the single decode byte-traffic proxy (MB/token).

    IFEval-with-thinking is short-prompt / long-output, so the KV term grows with the (large) output
    length -- the decode-heavy, capacity-pressured regime PLENA targets.
    """
    block = spec["block"]
    attn_b = effective_bits(spec["w_fmt"], spec["attn_w"], block)
    ffn_b = effective_bits(spec["w_fmt"], spec["ffn_w"], block)
    kv_b = effective_bits(spec["kv_fmt"], spec["kv"], block)

    attn_el, ffn_el, head_el = _component_elements(dims)
    weight_bytes = (attn_el * attn_b + ffn_el * ffn_b + head_el * 16) / 8   # lm_head at bf16
    avg_ctx = s_in + s_out // 2                                             # KV grows over decode
    kv_el = 2 * dims["kv_heads"] * dims["head_dim"] * dims["layers"] * avg_ctx * batch
    kv_bytes = kv_el * kv_b / 8
    return {
        "attn_w_bits": round(attn_b, 4), "ffn_w_bits": round(ffn_b, 4), "kv_bits": round(kv_b, 4),
        "weight_MB_per_token": round(weight_bytes / 1e6, 3),
        "kv_MB_per_token": round(kv_bytes / 1e6, 3),
        "cost_mb_per_token": round((weight_bytes + kv_bytes) / 1e6, 3),
    }
