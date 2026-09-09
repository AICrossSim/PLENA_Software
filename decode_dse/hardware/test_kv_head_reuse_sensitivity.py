"""Focused contracts for the isolated rank-local GQA reuse sensitivity."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from decode_dse.hardware.design_space import (
    KV_HEAD_REUSE_NOOP_REASON,
    HardwareCandidate,
)
from decode_dse.hardware.evaluation import precision_request
from decode_dse.hardware.kv_head_reuse_sensitivity import (
    CLASSIFICATION,
    KVHeadReuseWorkload,
    build_qwen3_kv_head_reuse_sensitivity,
    load_qwen3_kv_head_reuse_sensitivity,
    materialize_qwen3_kv_head_reuse_sensitivity,
    validate_qwen3_kv_head_reuse_sensitivity,
)
from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.simulator_bridge import DecodeSimulator


_SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _SOFTWARE_ROOT.parent
_SIMULATOR_ROOT = _WORKSPACE_ROOT / "PLENA_Simulator"
_MODEL = (
    _SIMULATOR_ROOT
    / "compiler"
    / "doc"
    / "Model_Lib"
    / "qwen3-30b-a3b-thinking-2507.json"
)
_HARDWARE = _SIMULATOR_ROOT / "plena_settings.toml"
_ISA = _SIMULATOR_ROOT / "analytic_models" / "performance" / "customISA_lib.json"


def _precision() -> dict:
    simulator = DecodeSimulator(
        str(_MODEL), settings_toml=_HARDWARE, isa_path=_ISA
    )
    profile = DecodePrecisionProfile.quantized(
        "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
    )
    request = precision_request(profile)
    precision = simulator.make_precision(
        attn_w=request.weight,
        ffn_w=request.weight,
        key=request.key,
        value=request.value,
        w_fmt=request.weight_family,
        key_fmt=request.key_family,
        value_fmt=request.value_family,
        block=request.block_size,
        act_w=request.activation,
        act_fmt=request.activation_family,
    )
    head = profile.local_head_contract
    return {
        **precision.spec,
        "profile_id": profile.profile_id,
        "head_vector_format": profile.vector_format,
        "head_matrix_storage_format": profile.vector_format,
        "head_logit_container_format": "BF16",
        "head_bf16_container_precision_recovery": False,
        "head_operand_family_supported": bool(
            head["operand_family_deployment_supported"]
        ),
        "head_operand_family_binding": head["operand_family_binding"],
        "head_numerical_oracle_rule": head["numerical_oracle_rule"],
        "head_partial_conversion_rule": head["partial_conversion_rule"],
        "head_hardware_bit_parity_verified": bool(
            head["hardware_bit_parity_verified"]
        ),
        "head_accumulation_chain": list(head["arithmetic_chain"]),
        "head_numerical_matrix_mlen": profile.matrix_mlen,
    }


def _candidates(*, hlen: int = 128) -> tuple[HardwareCandidate, ...]:
    return tuple(
        HardwareCandidate(
            mlen=1024,
            blen=4,
            vlen=1024,
            hlen=hlen,
            batch=1,
            hbm_channels=8,
            hbm_generation="HBM2",
            chip_count=tp,
            tp=tp,
            kvp=1,
            link_ports=0 if tp == 1 else 1,
            sram_policy="streaming",
            kv_head_reuse=False,
            drain_overlapped=False,
            expert_parallel_mode="tensor_parallel",
        )
        for tp in (1, 2, 4)
    )


def _build_sensitivity() -> dict:
    return build_qwen3_kv_head_reuse_sensitivity(
        precision=_precision(),
        baseline_candidates=_candidates(),
        workload=KVHeadReuseWorkload(
            input_sequence_tokens=128,
            output_sequence_tokens=1,
            stride=1,
            runtime_hbm_reserve_bytes_per_chip=0,
        ),
        model_config_path=_MODEL,
        hardware_config_path=_HARDWARE,
        custom_isa_path=_ISA,
    )


@pytest.fixture(scope="module")
def sensitivity() -> dict:
    return _build_sensitivity()


def test_full_loop_artifact_is_deterministic(sensitivity: dict) -> None:
    assert _build_sensitivity() == sensitivity


def test_full_loop_pairs_exact_rank_local_kv_traffic(sensitivity: dict) -> None:
    validate_qwen3_kv_head_reuse_sensitivity(sensitivity)
    for case in sensitivity["cases"][:2]:
        local_heads = case["rank_local_kv_heads"]
        assert local_heads in (4, 2)
        baseline = case["baseline"]["metrics"]
        variant = case["reuse_variant"]["metrics"]
        for scope in ("per_batch_step", "per_generated_token"):
            false_kv = baseline["kv_traffic"][scope]
            true_kv = variant["kv_traffic"][scope]
            assert false_kv["total_read_bytes"] == pytest.approx(
                true_kv["total_read_bytes"] * local_heads
            )
            assert false_kv["total_write_bytes"] == pytest.approx(
                true_kv["total_write_bytes"]
            )
        assert baseline["capacity"] == variant["capacity"]
        assert baseline["kv_head_reuse_control_area_mm2_per_chip"] == 0.0
        assert variant["kv_head_reuse_control_area_mm2_per_chip"] > 0.0
        assert case["delta"]["measured_delta_applied_to_full_model_projection"] is False
        assert case["multi_chip_rank_local_packed_q1_timing_matched"] is False
        assert case["classification"] == CLASSIFICATION


def test_reuse_pair_preserves_body_head_and_moe_ledgers(sensitivity: dict) -> None:
    for case in sensitivity["cases"][:2]:
        false_hashes = case["baseline"]["metrics"]["component_hashes"]
        true_hashes = case["reuse_variant"]["metrics"]["component_hashes"]
        assert false_hashes["body_physical_layout"] == true_hashes[
            "body_physical_layout"
        ]
        assert false_hashes["moe_workload"] == true_hashes["moe_workload"]
        assert false_hashes["local_output_head_stable_boundary"] == true_hashes[
            "local_output_head_stable_boundary"
        ]
        assert "power" in case["baseline"]["metrics"]["components"]
        assert "power" in case["reuse_variant"]["metrics"]["components"]


def test_tp4_hkv1_is_authenticated_prune_not_evaluated_true(
    sensitivity: dict,
) -> None:
    case = sensitivity["cases"][2]
    assert case["tp"] == 4
    assert case["rank_local_kv_heads"] == 1
    assert case["reuse_variant"] is None
    assert case["structural_prune"]["reason"] == KV_HEAD_REUSE_NOOP_REASON
    assert case["structural_prune"]["true_candidate_constructed"] is False
    assert case["structural_prune"]["true_candidate_evaluated"] is False
    assert case["structural_prune"]["control_area_mm2_per_chip"] == 0.0


def test_materialization_round_trip_and_tamper_fail_closed(
    sensitivity: dict, tmp_path: Path
) -> None:
    receipt = materialize_qwen3_kv_head_reuse_sensitivity(
        sensitivity, tmp_path
    )
    loaded = load_qwen3_kv_head_reuse_sensitivity(receipt["artifact_path"])
    assert loaded == sensitivity

    tampered = copy.deepcopy(sensitivity)
    tampered["cases"][0]["delta"]["tpot_s"] += 1.0
    with pytest.raises(ValueError, match="content_hash"):
        validate_qwen3_kv_head_reuse_sensitivity(tampered)


def test_illegal_tp1_reuse_geometry_fails_before_true_evaluation() -> None:
    with pytest.raises(ValueError, match="not a legal measured"):
        build_qwen3_kv_head_reuse_sensitivity(
            precision=_precision(),
            baseline_candidates=_candidates(hlen=512),
            workload=KVHeadReuseWorkload(128, 1, runtime_hbm_reserve_bytes_per_chip=0),
            model_config_path=_MODEL,
            hardware_config_path=_HARDWARE,
            custom_isa_path=_ISA,
        )
