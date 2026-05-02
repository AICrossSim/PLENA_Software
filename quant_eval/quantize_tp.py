"""TP-aware quant: swap nn.Linear and Attention via from_linear / from_attention."""

import re
import torch.nn as nn


def _attn_map():
    m = {}
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
        from chop.nn.quantized.modules.llama import LlamaAttentionMXInt
        m[LlamaAttention] = LlamaAttentionMXInt
    except ImportError: pass
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeAttention
        from chop.nn.quantized.modules.glm4_moe import Glm4MoeAttentionMXInt
        m[Glm4MoeAttention] = Glm4MoeAttentionMXInt
    except ImportError: pass
    return m


def quantize_tp_aware(model: nn.Module, pass_args: dict, verbose: bool = True):
    """Regex-matched swap to LinearMXInt / *AttentionMXInt, preserving DTensor shards."""
    from chop.nn.quantized.modules.linear import LinearMXInt

    cfgs = {k: v["config"] for k, v in pass_args.items()
            if isinstance(v, dict) and "config" in v}

    def match(name):
        return next((c for p, c in cfgs.items() if re.fullmatch(p, name)), None)

    def replace(name, new):
        p, _, a = name.rpartition(".")
        setattr(model.get_submodule(p) if p else model, a, new)

    attn_map = _attn_map()
    n_attn = n_lin = 0

    # Attention pass first (children are still original Linears)
    for name, m in list(model.named_modules()):
        for orig, mx in attn_map.items():
            if isinstance(m, orig) and not isinstance(m, mx) and (cfg := match(name)):
                replace(name, mx.from_attention(m, cfg))
                n_attn += 1
                break

    # Linear pass (includes children of the new MX attentions — same objects)
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and not isinstance(m, LinearMXInt) and (cfg := match(name)):
            replace(name, LinearMXInt.from_linear(m, cfg))
            n_lin += 1

    if verbose:
        print(f"[quantize_tp_aware] {n_attn} attn + {n_lin} linear swapped")
    return model, n_attn + n_lin
