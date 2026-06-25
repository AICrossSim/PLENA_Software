"""Unified MX runtime wrappers for Llama DSE evaluation.

These wrappers intentionally live in ``quant_eval`` instead of patching Chop.
They mirror the old Coprocessor-for-Llama simulator semantics: each path gets
its own metadata/config and dispatches to MXInt or MXFP at runtime.  This lets
ACT and KV use different MX families while preserving the existing Chop GPTQ and
module replacement flow for weight preparation.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor, nn
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward as _hf_eager_attention_forward,
    repeat_kv,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    eager_attention_forward as _qwen3_eager_attention_forward,
    repeat_kv as qwen3_repeat_kv,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeExperts,
    Qwen3MoeRMSNorm,
    apply_rotary_pos_emb as qwen3_moe_apply_rotary_pos_emb,
    eager_attention_forward as _qwen3_moe_eager_attention_forward,
    repeat_kv as qwen3_moe_repeat_kv,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from chop.nn.quantizers import mxint_quantizer, mxfp_quantizer
from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim
from chop.nn.quantized.functional.rope import rope_minifloat
from chop.nn.quantized.functional.silu import silu_minifloat


def _minifloat_quantize(x: Tensor, config: dict) -> Tensor:
    x_quantizer = partial(
        minifloat_quantizer_sim,
        minifloat_meta=MinifloatMeta(
            exp_bits=config["data_in_exponent_width"],
            frac_bits=config["data_in_frac_width"],
            is_finite=config.get("data_in_is_finite", True),
            round_mode=config.get("data_in_round_mode", "rn"),
        ),
    )
    return x_quantizer(x)


def _softmax_minifloat_preserve_mask(
    logits: Tensor,
    attention_mask: Optional[Tensor],
    config: dict,
    *,
    dim: int = -1,
) -> Tensor:
    """Quantize attention logits without quantizing the additive mask.

    BF16/FP attention masks often contain very large negative values.  If those
    masked logits are sent through the minifloat quantizer, the mask can be
    clipped to a finite value that is no longer effectively -inf.  Quantize the
    real QK scores first, then apply the mask immediately before softmax.
    """
    logits = _minifloat_quantize(logits, config)
    if attention_mask is not None:
        logits = logits + attention_mask.to(logits.dtype)
    return F.softmax(logits.to(torch.float32), dim=dim).to(logits.dtype)


def _infer_mx_family(config: dict | None, prefix: str = "data_in") -> str | None:
    if not config:
        return None
    family = config.get(f"{prefix}_family")
    if family in ("mxint", "mxfp"):
        return family
    # Backward-compatible inference for configs produced before prefix-specific
    # family metadata existed. Prefer prefix-local keys over the generic family
    # because Linear configs may intentionally mix activation and weight family.
    if f"{prefix}_width" in config:
        return "mxint"
    if f"{prefix}_exponent_width" in config:
        return "mxfp"
    family = config.get("family")
    if family in ("mxint", "mxfp"):
        return family
    return None


def quantize_mx(x: Tensor, config: dict | None, *, block_dim: int = -1, prefix: str = "data_in") -> Tensor:
    """Quantize ``x`` using either MXInt or MXFP according to ``config``.

    ``prefix`` selects config keys: ``data_in_*``, ``weight_*`` or ``bias_*``.
    Missing config returns the input unchanged, matching Chop's skip behavior.
    """
    family = _infer_mx_family(config, prefix)
    if family is None:
        return x
    block_size = config.get(f"{prefix}_block_size")
    if block_size is None:
        return x
    if family == "mxint":
        width = config.get(f"{prefix}_width")
        if width is None:
            return x
        y = mxint_quantizer(
            x,
            block_size=block_size,
            element_bits=width,
            block_dim=block_dim,
            quantile_search=bool(config.get("clip_search", False)),
        )
        return y.to(dtype=x.dtype)
    if family == "mxfp":
        exp = config.get(f"{prefix}_exponent_width")
        frac = config.get(f"{prefix}_frac_width")
        if exp is None or frac is None:
            return x
        y = mxfp_quantizer(
            x,
            block_size=block_size,
            element_exp_bits=exp,
            element_frac_bits=frac,
            block_dim=block_dim,
            quantile_search=bool(config.get("clip_search", False)),
        )
        return y.to(dtype=x.dtype)
    raise ValueError(f"Unsupported MX family {family!r} in config {config!r}")


def _hf_attention_dispatch(module, query_states, key_states, value_states, attention_mask, **kwargs):
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation,
        _hf_eager_attention_forward,
    )
    return attention_interface(module, query_states, key_states, value_states, attention_mask, **kwargs)


def _eager_attention_forward_unified(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict | None = None,
    av_bypass: bool = False,
    av_config: dict | None = None,
    softmax_bypass: bool = False,
    softmax_config: dict | None = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        query = quantize_mx(query, qk_config, block_dim=-1, prefix="data_in")

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    causal_mask = None
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    if not softmax_bypass:
        attn_weights = _softmax_minifloat_preserve_mask(attn_weights, causal_mask, softmax_config, dim=-1)
    else:
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask.to(attn_weights.dtype)
        attn_weights = F.softmax(attn_weights.to(torch.float32), dim=-1).to(attn_weights.dtype)

    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    if not av_bypass:
        attn_weights = quantize_mx(attn_weights, av_config, block_dim=-1, prefix="data_in")

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _eager_qwen3_attention_forward_unified(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict | None = None,
    av_bypass: bool = False,
    av_config: dict | None = None,
    softmax_bypass: bool = False,
    softmax_config: dict | None = None,
    **kwargs,
):
    key_states = qwen3_repeat_kv(key, module.num_key_value_groups)
    value_states = qwen3_repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        query = quantize_mx(query, qk_config, block_dim=-1, prefix="data_in")

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if not softmax_bypass:
        attn_weights = _softmax_minifloat_preserve_mask(attn_weights, attention_mask, softmax_config, dim=-1)
    else:
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
        attn_weights = F.softmax(attn_weights.to(torch.float32), dim=-1).to(attn_weights.dtype)

    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    if not av_bypass:
        attn_weights = quantize_mx(attn_weights, av_config, block_dim=-1, prefix="data_in")

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _eager_qwen3_moe_attention_forward_unified(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_bypass: bool = False,
    qk_config: dict | None = None,
    av_bypass: bool = False,
    av_config: dict | None = None,
    softmax_bypass: bool = False,
    softmax_config: dict | None = None,
    **kwargs,
):
    key_states = qwen3_moe_repeat_kv(key, module.num_key_value_groups)
    value_states = qwen3_moe_repeat_kv(value, module.num_key_value_groups)

    if not qk_bypass:
        query = quantize_mx(query, qk_config, block_dim=-1, prefix="data_in")

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if not softmax_bypass:
        attn_weights = _softmax_minifloat_preserve_mask(attn_weights, attention_mask, softmax_config, dim=-1)
    else:
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
        attn_weights = F.softmax(attn_weights.to(torch.float32), dim=-1).to(attn_weights.dtype)

    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    if not av_bypass:
        attn_weights = quantize_mx(attn_weights, av_config, block_dim=-1, prefix="data_in")

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _phase_attn_config_to_q_config(config: dict | None) -> dict:
    cfg = deepcopy(config or {})
    kv = deepcopy(cfg.pop("kv_cache", {}))
    softmax = deepcopy(cfg.pop("softmax", {}))
    rope = deepcopy(cfg.pop("rope", {}))
    qk = deepcopy(cfg.pop("qk_matmul", {}))
    av = deepcopy(cfg.pop("av_matmul", {}))
    cfg.pop("linear_bypass", None)
    act_cfg = deepcopy(cfg)
    return {
        "qk_matmul": {**deepcopy(act_cfg), **qk},
        "av_matmul": {**deepcopy(act_cfg), **av},
        "kv_cache": kv,
        "softmax": softmax,
        "rope": rope,
    }


class LinearMXUnified(nn.Linear):
    """Linear wrapper with independent runtime dispatch for activation/weight MX family."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, config=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = bool(config.get("bypass", False))
        self.gptq = bool(config.get("gptq", False))
        self.clip_search = bool(config.get("clip_search", False))
        self.use_fp_weight = False
        self.register_buffer("fp_weight", None, persistent=False)
        self.register_buffer("fp_bias", None, persistent=False)

    @classmethod
    def from_existing(cls, module: nn.Linear, config: dict | None = None) -> "LinearMXUnified":
        cfg = deepcopy(config if config is not None else getattr(module, "config", {}))
        new = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device="meta",
            dtype=module.weight.dtype,
            config=cfg,
        )
        # Reuse the already GPTQ/fake-quantized Parameter instead of allocating
        # and copying a second full-sized weight tensor. This is critical for
        # very large MoE models where wrapper installation can otherwise double
        # resident GPU memory.
        new.weight = module.weight
        new.bias = module.bias
        new.training = module.training
        return new

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: dict) -> "LinearMXUnified":
        new = cls.from_existing(linear, config)
        new.load_state_dict(linear.state_dict(), strict=False)
        return new

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        self._materialize_on_input_device(x.device)
        weight = self.fp_weight if self.use_fp_weight and self.fp_weight is not None else self.weight
        bias = self.fp_bias if self.use_fp_weight and self.fp_bias is not None else self.bias
        if self.bypass:
            return F.linear(x.to(dtype=weight.dtype), weight, bias)
        x = quantize_mx(x, self.config, block_dim=-1, prefix="data_in")
        return F.linear(x.to(dtype=weight.dtype), weight, bias)

    @torch.no_grad()
    def _materialize_on_input_device(self, device: torch.device) -> None:
        """Keep replacement wrappers compatible with HF Accelerate device maps.

        ``device_map="auto"`` attaches hooks before we install some unified
        wrappers, so the replacement module can retain CPU tensors while its
        parent layer is executed on a GPU.  Move each tensor once to the device
        that actually feeds the layer.
        """
        if self.weight.device != device:
            self.weight.data = self.weight.data.to(device=device)
        if self.bias is not None and self.bias.device != device:
            self.bias.data = self.bias.data.to(device=device)
        if self.fp_weight is not None and self.fp_weight.device != device:
            self.fp_weight = self.fp_weight.to(device=device)
        if self.fp_bias is not None and self.fp_bias.device != device:
            self.fp_bias = self.fp_bias.to(device=device)

    @torch.no_grad()
    def set_fp_weight_backup(self, weight: Tensor, bias: Tensor | None = None) -> None:
        self.fp_weight = weight.to(device=self.weight.device, dtype=self.weight.dtype).contiguous()
        if bias is not None:
            self.fp_bias = bias.to(
                device=self.bias.device if self.bias is not None else self.weight.device,
                dtype=self.bias.dtype if self.bias is not None else self.weight.dtype,
            ).contiguous()
        elif self.bias is not None:
            self.fp_bias = None

    def set_use_fp_weight(self, enabled: bool) -> None:
        if enabled and self.fp_weight is None:
            raise RuntimeError("FP weight residency requested before fp_weight backup was materialized.")
        self.use_fp_weight = bool(enabled)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.bypass or self.gptq:
            return result
        self.weight.data.copy_(quantize_mx(self.weight.data, self.config, block_dim=1, prefix="weight"))
        if self.bias is not None:
            self.bias.data.copy_(quantize_mx(self.bias.data, self.config, block_dim=0, prefix="bias"))
        return result


class LlamaAttentionMXUnified(LlamaAttention):
    """Llama attention wrapper with independent ACT/KV MX families."""

    def __init__(self, config, layer_idx, q_config: dict | None = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = bool(self.qk_config.get("bypass", False))
        self.av_bypass = bool(self.av_config.get("bypass", False))
        self.rope_bypass = bool(self.rope_config.get("bypass", False))
        self.softmax_bypass = bool(self.softmax_config.get("bypass", False))
        self.kv_cache_bypass = bool(self.kv_cache_config.get("bypass", False))

    @classmethod
    def from_attention(cls, attention: LlamaAttention, q_config: dict | None = None) -> "LlamaAttentionMXUnified":
        cfg = q_config or {
            "qk_matmul": deepcopy(getattr(attention, "qk_config", {})),
            "av_matmul": deepcopy(getattr(attention, "av_config", {})),
            "kv_cache": deepcopy(getattr(attention, "kv_cache_config", {})),
            "softmax": deepcopy(getattr(attention, "softmax_config", {})),
            "rope": deepcopy(getattr(attention, "rope_config", {})),
        }
        with torch.device("meta"):
            new = cls(attention.config, attention.layer_idx, cfg)
        # Preserve already-quantized projection modules instead of rebuilding
        # them as plain nn.Linear modules.
        new.q_proj = attention.q_proj
        new.k_proj = attention.k_proj
        new.v_proj = attention.v_proj
        new.o_proj = attention.o_proj
        new.training = attention.training
        return new

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(query_states, key_states, cos, sin, self.rope_config)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if not self.kv_cache_bypass:
                key_states = quantize_mx(key_states, self.kv_cache_config, block_dim=-1, prefix="data_in")
                value_states = quantize_mx(value_states, self.kv_cache_config, block_dim=-1, prefix="data_in")
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attn_output, attn_weights = _hf_attention_dispatch(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            # Quantized QK/AV/softmax paths need the explicit eager attention
            # implementation regardless of the HF model-level backend setting.
            # This keeps mixed-family DSE compatible with models loaded using
            # sdpa/flash defaults while only overriding the quantized path.
            attn_output, attn_weights = _eager_attention_forward_unified(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3AttentionMXUnified(Qwen3Attention):
    """Qwen3 attention wrapper with independent ACT/KV MX families."""

    def __init__(self, config, layer_idx, q_config: dict | None = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = bool(self.qk_config.get("bypass", False))
        self.av_bypass = bool(self.av_config.get("bypass", False))
        self.rope_bypass = bool(self.rope_config.get("bypass", False))
        self.softmax_bypass = bool(self.softmax_config.get("bypass", False))
        self.kv_cache_bypass = bool(self.kv_cache_config.get("bypass", False))

    @classmethod
    def from_attention(cls, attention: Qwen3Attention, q_config: dict | None = None) -> "Qwen3AttentionMXUnified":
        cfg = q_config or {
            "qk_matmul": deepcopy(getattr(attention, "qk_config", {})),
            "av_matmul": deepcopy(getattr(attention, "av_config", {})),
            "kv_cache": deepcopy(getattr(attention, "kv_cache_config", {})),
            "softmax": deepcopy(getattr(attention, "softmax_config", {})),
            "rope": deepcopy(getattr(attention, "rope_config", {})),
        }
        with torch.device("meta"):
            new = cls(attention.config, attention.layer_idx, cfg)
        new.q_proj = attention.q_proj
        new.k_proj = attention.k_proj
        new.v_proj = attention.v_proj
        new.o_proj = attention.o_proj
        new.q_norm = attention.q_norm
        new.k_norm = attention.k_norm
        new.training = attention.training
        return new

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(query_states, key_states, cos, sin, self.rope_config)
        else:
            query_states, key_states = qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if not self.kv_cache_bypass:
                key_states = quantize_mx(key_states, self.kv_cache_config, block_dim=-1, prefix="data_in")
                value_states = quantize_mx(value_states, self.kv_cache_config, block_dim=-1, prefix="data_in")
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                _qwen3_eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        else:
            attn_output, attn_weights = _eager_qwen3_attention_forward_unified(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3MoeAttentionMXUnified(Qwen3MoeAttention):
    """Qwen3-MoE attention wrapper with independent ACT/KV MX families."""

    def __init__(self, config, layer_idx, q_config: dict | None = None):
        super().__init__(config, layer_idx)
        q_config = q_config or {}
        self.qk_config = q_config.get("qk_matmul", {})
        self.av_config = q_config.get("av_matmul", {})
        self.rope_config = q_config.get("rope", {})
        self.softmax_config = q_config.get("softmax", {})
        self.kv_cache_config = q_config.get("kv_cache", {})
        self.qk_bypass = bool(self.qk_config.get("bypass", False))
        self.av_bypass = bool(self.av_config.get("bypass", False))
        self.rope_bypass = bool(self.rope_config.get("bypass", False))
        self.softmax_bypass = bool(self.softmax_config.get("bypass", False))
        self.kv_cache_bypass = bool(self.kv_cache_config.get("bypass", False))

    @classmethod
    def from_attention(cls, attention: Qwen3MoeAttention, q_config: dict | None = None) -> "Qwen3MoeAttentionMXUnified":
        cfg = q_config or {
            "qk_matmul": deepcopy(getattr(attention, "qk_config", {})),
            "av_matmul": deepcopy(getattr(attention, "av_config", {})),
            "kv_cache": deepcopy(getattr(attention, "kv_cache_config", {})),
            "softmax": deepcopy(getattr(attention, "softmax_config", {})),
            "rope": deepcopy(getattr(attention, "rope_config", {})),
        }
        with torch.device("meta"):
            new = cls(attention.config, attention.layer_idx, cfg)
        new.q_proj = attention.q_proj
        new.k_proj = attention.k_proj
        new.v_proj = attention.v_proj
        new.o_proj = attention.o_proj
        new.q_norm = attention.q_norm
        new.k_norm = attention.k_norm
        new.training = attention.training
        return new

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if not self.rope_bypass:
            query_states, key_states = rope_minifloat(query_states, key_states, cos, sin, self.rope_config)
        else:
            query_states, key_states = qwen3_moe_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if not self.kv_cache_bypass:
                key_states = quantize_mx(key_states, self.kv_cache_config, block_dim=-1, prefix="data_in")
                value_states = quantize_mx(value_states, self.kv_cache_config, block_dim=-1, prefix="data_in")
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.qk_bypass and self.av_bypass and self.softmax_bypass:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                _qwen3_moe_eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        else:
            attn_output, attn_weights = _eager_qwen3_moe_attention_forward_unified(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                qk_bypass=self.qk_bypass,
                qk_config=self.qk_config,
                av_bypass=self.av_bypass,
                av_config=self.av_config,
                softmax_bypass=self.softmax_bypass,
                softmax_config=self.softmax_config,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3MLPMXUnified(Qwen3MLP):
    """Qwen3 MLP wrapper whose SiLU precision is driven by FP_SETTING.

    Projection Linears are already replaced by ``LinearMXUnified`` before this
    wrapper is installed, so this class only owns the nonlinear SiLU path.  This
    mirrors Chop's Qwen3MLPMXInt/MXFP behavior while keeping the implementation
    in quant_eval instead of patching the installed Chop package.
    """

    def __init__(self, config, layer_idx=None, q_config: dict | None = None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self.bypass = bool(self.q_config.get("bypass", False))

    @classmethod
    def from_mlp(cls, mlp: Qwen3MLP, q_config: dict | None = None) -> "Qwen3MLPMXUnified":
        cfg = deepcopy(q_config if q_config is not None else getattr(mlp, "q_config", {}))
        layer_idx = getattr(mlp, "layer_idx", None)
        with torch.device("meta"):
            new = cls(mlp.config, layer_idx=layer_idx, q_config=cfg)
        # Preserve already-replaced projection modules and the configured HF act_fn.
        new.gate_proj = mlp.gate_proj
        new.up_proj = mlp.up_proj
        new.down_proj = mlp.down_proj
        new.act_fn = mlp.act_fn
        new.training = mlp.training
        return new

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(x)
        x = silu_minifloat(self.gate_proj(x), self.q_config) * self.up_proj(x)
        return self.down_proj(x)


class Qwen3MoeExpertsMXUnified(Qwen3MoeExperts):
    """Qwen3-MoE expert wrapper for routed ACT and SiLU precision.

    Router, top-k selection, and routing weights live outside
    ``Qwen3MoeExperts`` and intentionally remain FP.
    """

    def __init__(self, config, q_config: dict | None = None):
        super().__init__(config)
        self.config = config
        self.q_config = q_config or {}
        self.bypass = bool(self.q_config.get("bypass", False))

    @classmethod
    def from_experts(cls, experts: Qwen3MoeExperts, q_config: dict | None = None) -> "Qwen3MoeExpertsMXUnified":
        cfg = deepcopy(q_config if q_config is not None else getattr(experts, "q_config", {}))
        config = getattr(experts, "config", None)
        if config is None:
            class _Config:
                pass
            config = _Config()
            config.num_experts = experts.num_experts
            config.hidden_size = experts.hidden_dim
            config.moe_intermediate_size = experts.intermediate_dim
            config.hidden_act = getattr(getattr(experts, "act_fn", None), "__name__", "silu")
        with torch.device("meta"):
            new = cls(config, q_config=cfg)
        new.gate_up_proj = experts.gate_up_proj
        new.down_proj = experts.down_proj
        new.act_fn = experts.act_fn
        new.training = experts.training
        return new

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        self._materialize_on_input_device(hidden_states.device)
        if self.bypass:
            return super().forward(hidden_states, top_k_index, top_k_weights)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_state = quantize_mx(current_state, self.q_config, block_dim=-1, prefix="data_in")
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = silu_minifloat(gate, self.q_config) * up
            current_hidden_states = quantize_mx(current_hidden_states, self.q_config, block_dim=-1, prefix="data_in")
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    @torch.no_grad()
    def _materialize_on_input_device(self, device: torch.device) -> None:
        if self.gate_up_proj.device != device:
            self.gate_up_proj.data = self.gate_up_proj.data.to(device=device)
        if self.down_proj.device != device:
            self.down_proj.data = self.down_proj.data.to(device=device)


def _build_minifloat_quantizer(q_config: dict, *, prefix: str):
    return partial(
        minifloat_quantizer_sim,
        minifloat_meta=MinifloatMeta(
            exp_bits=q_config[f"{prefix}_exponent_width"],
            frac_bits=q_config[f"{prefix}_frac_width"],
            is_finite=q_config.get(f"{prefix}_is_finite", True),
            round_mode=q_config.get(f"{prefix}_round_mode", "rn"),
        ),
    )


class Qwen3RMSNormMinifloatUnified(Qwen3RMSNorm):
    """Qwen3 RMSNorm wrapper controlled by FP_SETTING.

    Covers ordinary decoder RMSNorms, final model.norm, and Qwen3 attention
    q_norm/k_norm.  The forward path intentionally mirrors Chop's
    Qwen3RMSNormMinifloat while preserving the input dtype at the output.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, layer_idx=None, q_config: dict | None = None):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self._sync_quantizers_from_config()

    @classmethod
    def from_rms_norm(
        cls,
        rms_norm: Qwen3RMSNorm,
        q_config: dict | None = None,
    ) -> "Qwen3RMSNormMinifloatUnified":
        cfg = deepcopy(q_config if q_config is not None else getattr(rms_norm, "q_config", {}))
        with torch.device("meta"):
            new = cls(
                hidden_size=rms_norm.weight.numel(),
                eps=float(getattr(rms_norm, "variance_epsilon", 1e-6)),
                layer_idx=getattr(rms_norm, "layer_idx", None),
                q_config=cfg,
            )
        new.weight = rms_norm.weight
        new.training = rms_norm.training
        return new

    def _sync_quantizers_from_config(self) -> None:
        cfg = self.q_config or {}
        self.bypass = bool(cfg.get("bypass", False))
        self.weight_bypass = bool(cfg.get("weight_bypass", False))
        self.data_in_bypass = bool(cfg.get("data_in_bypass", False))
        self.w_quantizer = None if self.bypass or self.weight_bypass else _build_minifloat_quantizer(cfg, prefix="weight")
        self.x_quantizer = None if self.bypass or self.data_in_bypass else _build_minifloat_quantizer(cfg, prefix="data_in")

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        if self.x_quantizer is not None:
            hidden_states = self.x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        base_weight = self.weight.to(device=hidden_states.device)
        weight = self.w_quantizer(base_weight) if self.w_quantizer is not None else base_weight
        return weight * hidden_states.to(input_dtype)


class Qwen3MoeRMSNormMinifloatUnified(Qwen3MoeRMSNorm):
    """Qwen3-MoE RMSNorm wrapper controlled by FP_SETTING."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, layer_idx=None, q_config: dict | None = None):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.layer_idx = layer_idx
        self.q_config = q_config or {}
        self._sync_quantizers_from_config()

    @classmethod
    def from_rms_norm(
        cls,
        rms_norm: Qwen3MoeRMSNorm,
        q_config: dict | None = None,
    ) -> "Qwen3MoeRMSNormMinifloatUnified":
        cfg = deepcopy(q_config if q_config is not None else getattr(rms_norm, "q_config", {}))
        with torch.device("meta"):
            new = cls(
                hidden_size=rms_norm.weight.numel(),
                eps=float(getattr(rms_norm, "variance_epsilon", 1e-6)),
                layer_idx=getattr(rms_norm, "layer_idx", None),
                q_config=cfg,
            )
        new.weight = rms_norm.weight
        new.training = rms_norm.training
        return new

    def _sync_quantizers_from_config(self) -> None:
        cfg = self.q_config or {}
        self.bypass = bool(cfg.get("bypass", False))
        self.weight_bypass = bool(cfg.get("weight_bypass", False))
        self.data_in_bypass = bool(cfg.get("data_in_bypass", False))
        self.w_quantizer = None if self.bypass or self.weight_bypass else _build_minifloat_quantizer(cfg, prefix="weight")
        self.x_quantizer = None if self.bypass or self.data_in_bypass else _build_minifloat_quantizer(cfg, prefix="data_in")

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        if self.x_quantizer is not None:
            hidden_states = self.x_quantizer(hidden_states)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        base_weight = self.weight.to(device=hidden_states.device)
        weight = self.w_quantizer(base_weight) if self.w_quantizer is not None else base_weight
        return weight * hidden_states.to(input_dtype)


def _patch_rms_norm_dtype_preservation() -> bool:
    """Keep Chop RMSNorm minifloat outputs in the input dtype.

    Chop's minifloat quantizer may return fp32 tensors for bf16 models. Most
    quantized Linear paths cast before F.linear, but the final lm_head remains a
    plain Linear. This eval-side monkey patch avoids fp32/bf16 matmul failures
    without editing the installed Chop package.
    """
    try:
        from chop.nn.quantized.modules.llama.rms_norm import LlamaRMSNormMinifloat
    except Exception:  # pragma: no cover - optional in non-Llama tests
        return False
    if getattr(LlamaRMSNormMinifloat, "_quant_eval_dtype_preserving", False):
        return False

    original_forward = LlamaRMSNormMinifloat.forward

    def forward_dtype_preserving(self, hidden_states, *args, **kwargs):
        out = original_forward(self, hidden_states, *args, **kwargs)
        return out.to(dtype=hidden_states.dtype)

    LlamaRMSNormMinifloat.forward = forward_dtype_preserving
    LlamaRMSNormMinifloat._quant_eval_dtype_preserving = True
    return True


def _replace_child(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _is_qwen3_gptq_linear_target(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    if ".self_attn." in name and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return True
    if ".mlp." in name and leaf in {"gate_proj", "up_proj", "down_proj"}:
        return True
    return False


def _qwen3_gptq_linear_config(name: str, attn_config: dict, ffn_config: dict) -> dict:
    config = attn_config if ".self_attn." in name else ffn_config
    cfg = deepcopy(config or {})
    cfg["gptq"] = True
    return cfg


def apply_qwen3_gptq_cache_unified_wrappers(
    model: nn.Module,
    *,
    attn_linear_config: dict,
    ffn_linear_config: dict,
    qwen3_attention_config: dict | None = None,
    qwen3_mlp_config: dict | None = None,
    qwen3_rms_norm_config: dict | None = None,
    qwen3_moe_attention_config: dict | None = None,
    qwen3_moe_experts_config: dict | None = None,
    qwen3_moe_rms_norm_config: dict | None = None,
) -> dict[str, int]:
    """Install unified wrappers directly on GPTQ-cache-loaded Qwen modules.

    GPTQ cache loading writes fake-quantized weights back into the original HF
    modules.  On a full cache hit we can skip Chop's replacement pass and wrap
    only the affected projections, preserving cached weights while leaving
    non-target modules such as ``lm_head`` and MoE routers untouched.
    """
    counts = {"direct_gptq_linear": 0}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LinearMXUnified):
            continue
        if not _is_qwen3_gptq_linear_target(name):
            continue
        cfg = _qwen3_gptq_linear_config(name, attn_linear_config, ffn_linear_config)
        _replace_child(model, name, LinearMXUnified.from_existing(module, cfg))
        counts["direct_gptq_linear"] += 1

    wrapper_counts = apply_unified_mx_wrappers(
        model,
        qwen3_attention_config=qwen3_attention_config,
        qwen3_mlp_config=qwen3_mlp_config,
        qwen3_rms_norm_config=qwen3_rms_norm_config,
        qwen3_moe_attention_config=qwen3_moe_attention_config,
        qwen3_moe_experts_config=qwen3_moe_experts_config,
        qwen3_moe_rms_norm_config=qwen3_moe_rms_norm_config,
    )
    counts.update(wrapper_counts)
    return counts


def _install_qwen3_moe_router_device_hooks(model: nn.Module) -> int:
    """Keep FP router weights colocated with routed hidden states."""

    def _move_router_to_input_device(module: nn.Module, args) -> None:
        if not args:
            return None
        hidden_states = args[0]
        if not torch.is_tensor(hidden_states) or not hasattr(module, "weight"):
            return None
        weight = module.weight
        if weight.device != hidden_states.device:
            weight.data = weight.data.to(device=hidden_states.device)
        return None

    count = 0
    for name, module in model.named_modules():
        if not name.endswith(".mlp.gate") or not hasattr(module, "weight"):
            continue
        if getattr(module, "_plena_qwen3_moe_router_device_hook", False):
            continue
        module.register_forward_pre_hook(_move_router_to_input_device)
        module._plena_qwen3_moe_router_device_hook = True
        count += 1
    return count


def apply_unified_mx_wrappers(
    model: nn.Module,
    qwen3_attention_config: dict | None = None,
    qwen3_mlp_config: dict | None = None,
    qwen3_rms_norm_config: dict | None = None,
    qwen3_moe_attention_config: dict | None = None,
    qwen3_moe_experts_config: dict | None = None,
    qwen3_moe_rms_norm_config: dict | None = None,
) -> dict[str, int]:
    """Replace Chop/Transformers MX modules with unified runtime wrappers."""
    from chop.nn.quantized.modules.linear import LinearMXInt, LinearMXFP
    from chop.nn.quantized.modules.llama.attention import LlamaAttentionMXInt, LlamaAttentionMXFP

    try:
        from chop.nn.quantized.modules.qwen3.attention import Qwen3AttentionMXInt, Qwen3AttentionMXFP
        qwen3_chop_attention_types = (Qwen3AttentionMXInt, Qwen3AttentionMXFP)
    except Exception:  # pragma: no cover - optional package shape
        qwen3_chop_attention_types = ()

    counts = {
        "linear": 0,
        "llama_attention": 0,
        "qwen3_attention": 0,
        "qwen3_mlp": 0,
        "qwen3_rms_norm": 0,
        "qwen3_moe_attention": 0,
        "qwen3_moe_experts": 0,
        "qwen3_moe_rms_norm": 0,
        "qwen3_moe_router_device_hook": 0,
        "rms_norm_dtype_patch": 0,
    }
    counts["rms_norm_dtype_patch"] = int(_patch_rms_norm_dtype_preservation())

    # Replace leaves first so attention replacement preserves unified projections.
    for name, module in list(model.named_modules()):
        if isinstance(module, (LinearMXInt, LinearMXFP)):
            _replace_child(model, name, LinearMXUnified.from_existing(module))
            counts["linear"] += 1

    for name, module in list(model.named_modules()):
        if isinstance(module, (LlamaAttentionMXInt, LlamaAttentionMXFP)):
            _replace_child(model, name, LlamaAttentionMXUnified.from_attention(module))
            counts["llama_attention"] += 1
        elif qwen3_chop_attention_types and isinstance(module, qwen3_chop_attention_types):
            _replace_child(model, name, Qwen3AttentionMXUnified.from_attention(module))
            counts["qwen3_attention"] += 1
        elif qwen3_attention_config is not None and isinstance(module, Qwen3Attention):
            q_config = _phase_attn_config_to_q_config(qwen3_attention_config)
            _replace_child(model, name, Qwen3AttentionMXUnified.from_attention(module, q_config))
            counts["qwen3_attention"] += 1
        elif qwen3_moe_attention_config is not None and isinstance(module, Qwen3MoeAttention):
            q_config = _phase_attn_config_to_q_config(qwen3_moe_attention_config)
            _replace_child(model, name, Qwen3MoeAttentionMXUnified.from_attention(module, q_config))
            counts["qwen3_moe_attention"] += 1

    if qwen3_mlp_config is not None:
        for name, module in list(model.named_modules()):
            if isinstance(module, Qwen3MLP) and not isinstance(module, Qwen3MLPMXUnified):
                _replace_child(model, name, Qwen3MLPMXUnified.from_mlp(module, qwen3_mlp_config))
                counts["qwen3_mlp"] += 1

    if qwen3_moe_experts_config is not None:
        for name, module in list(model.named_modules()):
            if isinstance(module, Qwen3MoeExperts) and not isinstance(module, Qwen3MoeExpertsMXUnified):
                _replace_child(model, name, Qwen3MoeExpertsMXUnified.from_experts(module, qwen3_moe_experts_config))
                counts["qwen3_moe_experts"] += 1

    if qwen3_rms_norm_config is not None:
        for name, module in list(model.named_modules()):
            if isinstance(module, Qwen3RMSNorm) and not isinstance(module, Qwen3RMSNormMinifloatUnified):
                _replace_child(model, name, Qwen3RMSNormMinifloatUnified.from_rms_norm(module, qwen3_rms_norm_config))
                counts["qwen3_rms_norm"] += 1

    if qwen3_moe_rms_norm_config is not None:
        for name, module in list(model.named_modules()):
            if isinstance(module, Qwen3MoeRMSNorm) and not isinstance(module, Qwen3MoeRMSNormMinifloatUnified):
                _replace_child(model, name, Qwen3MoeRMSNormMinifloatUnified.from_rms_norm(module, qwen3_moe_rms_norm_config))
                counts["qwen3_moe_rms_norm"] += 1

    if qwen3_moe_attention_config is not None or qwen3_moe_experts_config is not None:
        counts["qwen3_moe_router_device_hook"] = _install_qwen3_moe_router_device_hooks(model)

    counts["attention"] = counts["llama_attention"] + counts["qwen3_attention"] + counts["qwen3_moe_attention"]
    return counts
