import copy

import pytest
import torch

import quant_eval.eval.unified_mx as unified_mx
from quant_eval.eval.unified_mx import Qwen3AttentionMXUnified
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding


def _tiny_qwen3_attention():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
        torch_dtype="float32",
    )
    config._attn_implementation = "eager"
    attention = Qwen3Attention(config, layer_idx=0).eval()
    return config, attention


def _inputs(config: Qwen3Config):
    torch.manual_seed(17)
    batch, seq_len = 2, 7
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    rotary = Qwen3RotaryEmbedding(config=config)
    position_embeddings = rotary(hidden_states, position_ids)

    # Match the additive mask shape consumed by HF Qwen3 eager attention.
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len)
    attention_mask = attention_mask.masked_fill(
        torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1).view(1, 1, seq_len, seq_len),
        torch.finfo(hidden_states.dtype).min,
    )
    return hidden_states, position_embeddings, attention_mask


def _assert_close(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_qwen3_unified_attention_all_bypass_matches_hf_eager():
    config, hf_attention = _tiny_qwen3_attention()
    wrapped = Qwen3AttentionMXUnified.from_attention(
        copy.deepcopy(hf_attention),
        {
            "qk_matmul": {"bypass": True},
            "av_matmul": {"bypass": True},
            "softmax": {"bypass": True},
            "rope": {"bypass": True},
            "kv_cache": {"bypass": True},
        },
    ).eval()

    hidden_states, position_embeddings, attention_mask = _inputs(config)

    with torch.no_grad():
        expected, _ = hf_attention(hidden_states, position_embeddings, attention_mask)
        actual, _ = wrapped(hidden_states, position_embeddings, attention_mask)

    _assert_close(actual, expected)


def test_qwen3_unified_manual_attention_identity_quant_matches_hf_eager(monkeypatch: pytest.MonkeyPatch):
    config, hf_attention = _tiny_qwen3_attention()
    wrapped = Qwen3AttentionMXUnified.from_attention(
        copy.deepcopy(hf_attention),
        {
            "qk_matmul": {"data_in_family": "mxint", "data_in_width": 8, "data_in_block_size": 16},
            "av_matmul": {"data_in_family": "mxint", "data_in_width": 8, "data_in_block_size": 16},
            "softmax": {"data_in_exponent_width": 8, "data_in_frac_width": 5},
            "rope": {"bypass": True},
            "kv_cache": {"bypass": True},
        },
    ).eval()

    monkeypatch.setattr(unified_mx, "quantize_mx", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(unified_mx, "_minifloat_quantize", lambda x, *args, **kwargs: x)

    hidden_states, position_embeddings, attention_mask = _inputs(config)

    with torch.no_grad():
        expected, _ = hf_attention(hidden_states, position_embeddings, attention_mask)
        actual, _ = wrapped(hidden_states, position_embeddings, attention_mask)

    _assert_close(actual, expected)
