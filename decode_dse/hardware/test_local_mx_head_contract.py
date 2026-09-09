"""Fail-closed contracts for the canonical decode-local MX output head."""

from __future__ import annotations

from decode_dse.hardware.lm_head_service import (
    local_mx_head_boundary_status,
    local_mx_head_status_valid,
)
from decode_dse.hardware.workload_events import (
    DenseDecoderShape,
    local_output_head_padding_event_counts,
)
from decode_dse.profiles import DecodePrecisionProfile


def test_local_head_status_is_bound_to_the_exact_profile_oracle() -> None:
    profile = DecodePrecisionProfile.quantized(
        "E2M1",
        "E2M1",
        "MXINT4",
        "FP_E3M2",
    )
    status = local_mx_head_boundary_status(
        profile_id=profile.profile_id,
        weight_format=profile.weight_format,
        activation_format=profile.activation_format,
        vector_format=profile.vector_format,
        matrix_mlen=profile.matrix_mlen,
    )
    head = profile.local_head_contract

    assert status["accumulation_chain"] == head["arithmetic_chain"]
    assert status["operand_family_binding"] == head["operand_family_binding"]
    assert status["numerical_oracle_rule"] == head["numerical_oracle_rule"]
    assert status["partial_conversion_rule"] == head["partial_conversion_rule"]
    assert status["hardware_bit_parity_verified"] is False
    assert local_mx_head_status_valid(status)

    rebound = dict(status)
    rebound["numerical_matrix_mlen"] = 2048
    assert not local_mx_head_status_valid(
        rebound,
        profile_id=profile.profile_id,
        weight_format=profile.weight_format,
        activation_format=profile.activation_format,
        vector_format=profile.vector_format,
        matrix_mlen=profile.matrix_mlen,
    )


def test_head_padding_events_charge_batch_hidden_and_vocab_tails() -> None:
    shape = DenseDecoderShape(
        hidden_size=2048,
        intermediate_size=768,
        layers=48,
        attention_heads=32,
        kv_heads=4,
        head_dim=128,
        vocab_size=151_936,
    )
    events = local_output_head_padding_event_counts(
        shape,
        batch=3,
        mlen=1024,
        blen=8,
        vlen=1024,
    )

    # Five BLEN-padding rows each need 2,048 zero values. The hidden axis is
    # already MLEN-aligned, while 640 padded vocab values are masked per row.
    assert events["activation_zero_fill_vector_events_per_rank"] == 10
    assert events["activation_zero_fill_vector_events_system"] == 10
    assert events["padded_vocab_mask_vector_events_slowest_rank"] == 2
    assert events["padded_vocab_mask_vector_events_system"] == 2


def test_head_padding_rounds_each_tp_rank_before_system_aggregation() -> None:
    shape = DenseDecoderShape(
        hidden_size=2048,
        intermediate_size=768,
        layers=48,
        attention_heads=32,
        kv_heads=4,
        head_dim=128,
        vocab_size=151_936,
    )
    events = local_output_head_padding_event_counts(
        shape,
        batch=1,
        mlen=1024,
        blen=8,
        vlen=128,
        tp=4,
        kvp=2,
    )

    # Each 37,984-token shard has a 928-token tail: ceil(928/128)=8
    # vector events independently on every rank and every KVP replica.
    assert events["padded_vocab_mask_vector_events_slowest_rank"] == 8
    assert events["padded_vocab_mask_vector_events_system"] == 64
    assert events["padded_vocab_mask_vector_events_system"] != (
        (4 * 928 * 2 + 127) // 128
    )
