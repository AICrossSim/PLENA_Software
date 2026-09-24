import copy
from types import SimpleNamespace

import pytest
import torch

import quant_eval.eval.unified_mx as unified_mx
from quant_eval.eval.unified_mx import (
    AttentionQueryChunkController,
    Qwen3AttentionMXUnified,
    Qwen3MoeAttentionMXUnified,
)
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRotaryEmbedding,
)


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


def _all_bypass_config() -> dict:
    return {
        "qk_matmul": {"bypass": True},
        "av_matmul": {"bypass": True},
        "softmax": {"bypass": True},
        "rope": {"bypass": True},
        "kv_cache": {"bypass": True},
    }


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


def test_qwen3_query_chunked_identity_matches_full_and_bounds_scores(
    monkeypatch: pytest.MonkeyPatch,
):
    config, hf_attention = _tiny_qwen3_attention()
    controller = AttentionQueryChunkController(
        mode="query_chunked", chunk_size=3, min_chunk_size=1
    )
    wrapped = Qwen3AttentionMXUnified.from_attention(
        copy.deepcopy(hf_attention),
        {
            "qk_matmul": {"data_in_family": "mxint", "data_in_width": 8, "data_in_block_size": 16},
            "av_matmul": {"data_in_family": "mxint", "data_in_width": 8, "data_in_block_size": 16},
            "softmax": {"data_in_exponent_width": 8, "data_in_frac_width": 5},
            "rope": {"bypass": True},
            "kv_cache": {"bypass": True},
        },
        attention_chunk_controller=controller,
    ).eval()
    monkeypatch.setattr(unified_mx, "quantize_mx", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(unified_mx, "_minifloat_quantize", lambda x, *args, **kwargs: x)

    materialized_query_lengths = []
    original_chunk = unified_mx._qwen3_attention_chunk

    def recording_chunk(module, query, *args, **kwargs):
        materialized_query_lengths.append(int(query.shape[-2]))
        return original_chunk(module, query, *args, **kwargs)

    monkeypatch.setattr(unified_mx, "_qwen3_attention_chunk", recording_chunk)
    hidden_states, position_embeddings, attention_mask = _inputs(config)
    with torch.no_grad():
        expected, _ = hf_attention(hidden_states, position_embeddings, attention_mask)
        actual, weights = wrapped(hidden_states, position_embeddings, attention_mask)

    _assert_close(actual, expected)
    assert weights is None
    assert max(materialized_query_lengths) <= 3
    assert controller.snapshot()["chunked_calls"] == 1
    assert controller.snapshot()["max_materialized_query_chunk"] == 3


def test_qwen3_query_chunked_real_quant_matches_full_quant():
    config, attention = _tiny_qwen3_attention()
    q_config = {
        "qk_matmul": {"data_in_family": "mxint", "data_in_width": 8, "data_in_block_size": 16},
        "av_matmul": {"data_in_family": "mxint", "data_in_width": 8, "data_in_block_size": 16},
        "softmax": {"data_in_exponent_width": 8, "data_in_frac_width": 5},
        "rope": {"bypass": True},
        "kv_cache": {"bypass": True},
    }
    full = Qwen3AttentionMXUnified.from_attention(
        copy.deepcopy(attention), q_config
    ).eval()
    chunked = Qwen3AttentionMXUnified.from_attention(
        copy.deepcopy(attention),
        q_config,
        attention_chunk_controller=AttentionQueryChunkController(
            mode="query_chunked", chunk_size=3, min_chunk_size=1
        ),
    ).eval()
    hidden_states, position_embeddings, attention_mask = _inputs(config)
    with torch.no_grad():
        expected, _ = full(hidden_states, position_embeddings, attention_mask)
        actual, _ = chunked(hidden_states, position_embeddings, attention_mask)
    # Changing the GEMM query extent can change floating-point accumulation
    # order and move values across a minifloat rounding boundary. The identity
    # test above is strict; real quantization is expected to remain numerically
    # close rather than bit-identical.
    torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.01)


def test_qwen3_query_chunked_supports_cached_q_less_than_k_mask():
    torch.manual_seed(23)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 7, 8)
    value = torch.randn(2, 2, 7, 8)
    mask = torch.zeros(2, 1, 3, 7)
    mask[..., -2:] = torch.finfo(query.dtype).min

    common = {
        "scaling": 8**-0.5,
        "qk_bypass": True,
        "av_bypass": True,
        "softmax_bypass": True,
    }
    full, _ = unified_mx._eager_qwen3_attention_forward_unified(
        module, query, key, value, mask, **common
    )
    chunked, weights = unified_mx._eager_qwen3_attention_forward_unified(
        module,
        query,
        key,
        value,
        mask,
        attention_chunk_controller=AttentionQueryChunkController(
            mode="query_chunked", chunk_size=2, min_chunk_size=1
        ),
        **common,
    )
    _assert_close(chunked, full)
    assert weights is None


def test_qwen3_moe_query_chunked_all_bypass_matches_hf_eager():
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
        dtype="float32",
    )
    config._attn_implementation = "eager"
    hf_attention = Qwen3MoeAttention(config, layer_idx=0).eval()
    wrapped = Qwen3MoeAttentionMXUnified.from_attention(
        copy.deepcopy(hf_attention),
        _all_bypass_config(),
        attention_chunk_controller=AttentionQueryChunkController(
            mode="query_chunked", chunk_size=3, min_chunk_size=1
        ),
    ).eval()
    torch.manual_seed(29)
    batch, seq_len = 2, 7
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    position_embeddings = Qwen3MoeRotaryEmbedding(config=config)(
        hidden_states, position_ids
    )
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len).masked_fill(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
        .triu(1)
        .view(1, 1, seq_len, seq_len),
        torch.finfo(hidden_states.dtype).min,
    )
    with torch.no_grad():
        expected, _ = hf_attention(
            hidden_states, position_embeddings, attention_mask
        )
        actual, weights = wrapped(
            hidden_states, position_embeddings, attention_mask
        )
    _assert_close(actual, expected)
    assert weights is None


def test_qwen3_query_chunked_retries_locally_after_oom(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(31)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(1, 4, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn(1, 2, 7, 8)
    mask = torch.zeros(1, 1, 7, 7)
    controller = AttentionQueryChunkController(
        mode="query_chunked", chunk_size=4, min_chunk_size=2
    )
    original = unified_mx._qwen3_attention_with_query_chunks

    def simulated_oom(*args, chunk_size, **kwargs):
        if chunk_size is not None and chunk_size > 2:
            raise torch.cuda.OutOfMemoryError("simulated CUDA out of memory")
        return original(*args, chunk_size=chunk_size, **kwargs)

    monkeypatch.setattr(
        unified_mx, "_qwen3_attention_with_query_chunks", simulated_oom
    )
    output, weights = unified_mx._eager_qwen3_attention_forward_unified(
        module,
        query,
        key,
        value,
        mask,
        scaling=8**-0.5,
        qk_bypass=True,
        av_bypass=True,
        softmax_bypass=True,
        attention_chunk_controller=controller,
    )
    assert output.shape == (1, 7, 4, 8)
    assert weights is None
    stats = controller.snapshot()
    assert stats["effective_chunk_sizes"] == {"cpu": 2}
    assert stats["oom_count"] == 1
    assert stats["fallback_count"] == 1

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
