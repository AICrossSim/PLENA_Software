"""Contracts for native routed-MoE power-event accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from decode_dse.hardware import power_model
from decode_dse.hardware import evaluation
from decode_dse.hardware.moe_power_events import (
    BF16_ROUTER_CALIBRATION_BLOCKER,
    EXPERT_ID_TRACE_BLOCKER,
    TARGET_MODEL,
    TARGET_REVISION,
    build_moe_power_event_input_binding,
    build_moe_power_event_ledger,
    generic_calibration_event_counts,
    validate_moe_power_event_input_binding,
    validate_moe_power_event_ledger,
    validate_moe_power_event_receipt_for_engine,
)
from decode_dse.hardware.workload_events import (
    DenseDecoderShape,
    count_decode_events,
    merge_decode_event_counts,
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _dims() -> dict[str, object]:
    return {
        "model_type": "qwen3_moe",
        "hidden": 16,
        "inter": 6,
        "layers": 2,
        "num_experts": 4,
        "experts_per_token": 2,
        "router_weight_bits": 16,
    }


def _body(mode: str) -> dict[str, object]:
    exact = mode == "tensor_parallel"
    return {
        "schema_version": "plena-body-weight-physical-layout/v1",
        "provenance": {
            "analytic_layout_valid": True,
            "compiler_layout_valid": False,
            "rtl_layout_valid": False,
            "tp": 2,
            "kvp": 3,
            "mlen": 8,
            "expert_parallel_mode": mode,
            "expert_route_assignment_exact": exact,
            "padding_order": (
                "partition_rank_local_then_pad_each_matrix_to_mlen"
            ),
            "rank_shapes": {
                "experts": [
                    {
                        "rank": rank,
                        "gate_up": {
                            "logical_rows": 3 if exact else 6,
                        },
                        "down": {
                            "logical_columns": 3 if exact else 6,
                        },
                    }
                    for rank in range(2)
                ]
            },
        },
    }


def _moe_workload() -> dict[str, object]:
    return {
        "schema": "plena-routed-moe-decode-workload/v1",
        "tokens_per_step": 2,
        "num_experts": 4,
        "experts_per_token": 2,
        "expert_batch_ledger": {
            "route_assignments": 4,
            "active_experts": 2,
            "expert_row_tiles": 2,
            "expert_token_count_histogram": {"2": 2},
        },
    }


def _ledger(mode: str = "tensor_parallel") -> dict[str, object]:
    return build_moe_power_event_ledger(
        dims=_dims(),
        profile_id="dqp-test",
        candidate_id="hw-test",
        expert_linear_signature="LINEAR:MXINT4xMXINT4",
        vector_signature="VECTOR:FP_E3M2",
        body_layout=_body(mode),
        moe_workload=_moe_workload(),
        batch=2,
        decode_steps=5,
        mlen=8,
        blen=2,
        vlen=8,
        tp=2,
        kvp=3,
        expert_parallel_mode=mode,
    )


def test_tensor_moe_events_conserve_assignment_equivalents_and_kvp() -> None:
    ledger = _ledger()
    conservation = ledger["assignment_conservation"]
    operations = ledger["event_counts"]["per_operation"]

    assert validate_moe_power_event_ledger(ledger)
    assert conservation["expected_logical_assignments"] == 40
    assert conservation["expected_physical_executed_assignments"] == 120
    assert conservation["physical_executed_assignments_per_rank"] == [
        20,
        20,
        20,
        20,
        20,
        20,
    ]
    assert sum(conservation["physical_executed_assignments_per_rank"]) == 120
    for operation in operations.values():
        assert operation["per_rank"][:2] * 3 == operation["per_rank"]
        assert sum(operation["per_rank"]) == operation["aggregate_system"]


def test_tensor_moe_semantics_cover_ragged_stages_without_dense_fallback() -> None:
    ledger = _ledger()
    operations = ledger["event_counts"]["per_operation"]

    assert ledger["dense_ffn_fallback_used"] is False
    assert operations["moe_router"]["per_rank"] == [40] * 6
    assert operations["moe_topk"]["per_rank"] == [80] * 6
    assert operations["moe_route_filter"]["per_rank"] == [40] * 6
    assert operations["moe_expert_gate_up"]["per_rank"] == [160] * 6
    assert operations["moe_expert_swiglu"]["per_rank"] == [240] * 6
    assert operations["moe_expert_down"]["per_rank"] == [160] * 6
    assert operations["moe_combine"]["per_rank"] == [160] * 6
    assert operations["moe_hidden_dispatch"]["aggregate_system"] == 0
    assert operations["moe_expert_output_collective"]["per_rank"] == [10] * 6


def test_tp1_has_zero_dispatch_and_no_expert_output_collective() -> None:
    body = _body("tensor_parallel")
    provenance = body["provenance"]
    provenance["tp"] = 1
    provenance["kvp"] = 2
    provenance["rank_shapes"]["experts"] = [
        {
            "rank": 0,
            "gate_up": {"logical_rows": 6},
            "down": {"logical_columns": 6},
        }
    ]
    ledger = build_moe_power_event_ledger(
        dims=_dims(),
        profile_id="dqp-test",
        candidate_id="hw-test",
        expert_linear_signature="LINEAR:MXINT4xMXINT4",
        vector_signature="VECTOR:FP_E3M2",
        body_layout=body,
        moe_workload=_moe_workload(),
        batch=2,
        decode_steps=5,
        mlen=8,
        blen=2,
        vlen=8,
        tp=1,
        kvp=2,
        expert_parallel_mode="tensor_parallel",
    )
    operations = ledger["event_counts"]["per_operation"]
    conservation = ledger["assignment_conservation"]
    assert validate_moe_power_event_ledger(ledger)
    assert operations["moe_hidden_dispatch"]["aggregate_system"] == 0
    assert operations["moe_expert_output_collective"]["aggregate_system"] == 0
    assert conservation["physical_executed_assignments_per_rank"] == [40, 40]


def test_router_and_control_are_not_proxied_to_calibrated_signatures() -> None:
    ledger = _ledger()
    translated = generic_calibration_event_counts(ledger)
    translated_operations = {
        operation
        for event in ledger["generic_calibration_events"]
        for operation in event["semantic_operations"]
    }

    assert BF16_ROUTER_CALIBRATION_BLOCKER in ledger["blockers"]
    assert ledger["provenance"]["router_calibration_signature"] is None
    assert "moe_router" not in translated_operations
    assert "moe_route_filter" not in translated_operations
    assert {signature for signature, _ in translated} == {
        "LINEAR:MXINT4xMXINT4",
        "VECTOR:FP_E3M2",
    }
    assert ledger["power_engine_eligible"] is False
    assert ledger["receipt_emitted"] is False


def test_expert_id_without_per_rank_trace_is_costed_but_not_conserved_receipt() -> None:
    ledger = _ledger("expert_id_parallel")
    conservation = ledger["assignment_conservation"]

    assert validate_moe_power_event_ledger(ledger)
    assert conservation["exact"] is False
    assert conservation["physical_executed_assignments_per_rank"] is None
    assert conservation["observed_physical_executed_assignments"] is None
    assert EXPERT_ID_TRACE_BLOCKER in ledger["blockers"]
    assert ledger["event_counts"]["per_operation"][
        "moe_expert_gate_up"
    ]["aggregate_system"] > 0


def test_power_event_hash_and_structural_tampering_fail_closed() -> None:
    ledger = _ledger()
    tampered = json.loads(json.dumps(ledger))
    tampered["event_counts"]["per_operation"]["moe_hidden_dispatch"][
        "per_rank"
    ][0] = 1
    assert not validate_moe_power_event_ledger(tampered)

    rehashed = json.loads(json.dumps(tampered))
    body = dict(rehashed)
    body.pop("content_hash")
    rehashed["content_hash"] = _hash(body)
    assert not validate_moe_power_event_ledger(rehashed)


def test_dense_event_path_is_unchanged_and_native_merge_removes_proxy() -> None:
    shape = DenseDecoderShape(16, 24, 4, 2, 4, 2, 32)
    arguments = dict(
        input_seq=8,
        output_seq=3,
        batch=2,
        mlen=8,
        blen=2,
        hlen=4,
        vlen=8,
        linear_signature="LINEAR:MXINT4xMXINT4",
        qk_signature="QK:MXINT4xMXINT4",
        pv_signature="PV:MXINT4xMXINT4",
        vector_signature="VECTOR:FP_E3M2",
    )
    implicit_dense = count_decode_events(shape, **arguments)
    explicit_dense = count_decode_events(
        shape,
        include_dense_ffn=True,
        **arguments,
    )
    no_dense_ffn = count_decode_events(
        shape,
        include_dense_ffn=False,
        **arguments,
    )
    assert implicit_dense == explicit_dense
    dense = {event.signature: event.count for event in implicit_dense}
    native_base = {event.signature: event.count for event in no_dense_ffn}
    assert native_base["LINEAR:MXINT4xMXINT4"] < dense[
        "LINEAR:MXINT4xMXINT4"
    ]
    assert native_base["VECTOR:FP_E3M2"] < dense["VECTOR:FP_E3M2"]

    merged = merge_decode_event_counts(
        no_dense_ffn,
        (
            ("LINEAR:MXINT4xMXINT4", 17),
            ("LINEAR:MXINT8xMXINT8", 19),
        ),
        mlen=8,
        blen=2,
    )
    merged_counts = {event.signature: event.count for event in merged}
    assert merged_counts["LINEAR:MXINT4xMXINT4"] == (
        native_base["LINEAR:MXINT4xMXINT4"] + 17
    )
    assert merged_counts["LINEAR:MXINT8xMXINT8"] == 19


def test_ragged_rows_change_matrix_events_without_losing_assignments() -> None:
    workload = _moe_workload()
    workload["expert_batch_ledger"] = {
        "route_assignments": 4,
        "active_experts": 2,
        "expert_row_tiles": 3,
        "expert_token_count_histogram": {"1": 1, "3": 1},
    }
    ragged = build_moe_power_event_ledger(
        dims=_dims(),
        profile_id="dqp-test",
        candidate_id="hw-test",
        expert_linear_signature="LINEAR:MXINT4xMXINT4",
        vector_signature="VECTOR:FP_E3M2",
        body_layout=_body("tensor_parallel"),
        moe_workload=workload,
        batch=2,
        decode_steps=5,
        mlen=8,
        blen=2,
        vlen=8,
        tp=2,
        kvp=3,
        expert_parallel_mode="tensor_parallel",
    )
    uniform = _ledger()
    assert ragged["assignment_conservation"] == uniform[
        "assignment_conservation"
    ]
    assert ragged["event_counts"]["per_operation"][
        "moe_expert_gate_up"
    ]["aggregate_system"] > uniform["event_counts"]["per_operation"][
        "moe_expert_gate_up"
    ]["aggregate_system"]


def _target_receipt_inputs() -> tuple[dict[str, object], dict[str, str]]:
    dims = {
        **_dims(),
        "layers": 48,
        "num_experts": 128,
        "experts_per_token": 8,
    }
    workload = {
        "schema": "plena-routed-moe-decode-workload/v1",
        "tokens_per_step": 2,
        "num_experts": 128,
        "experts_per_token": 8,
        "expert_batch_ledger": {
            "route_assignments": 16,
            "active_experts": 8,
            "expert_row_tiles": 8,
            "expert_token_count_histogram": {"2": 8},
        },
    }
    ledger = build_moe_power_event_ledger(
        dims=dims,
        profile_id="dqp-target",
        candidate_id="hw-target",
        expert_linear_signature="LINEAR:MXINT4xMXINT4",
        vector_signature="VECTOR:FP_E3M2",
        body_layout=_body("tensor_parallel"),
        moe_workload=workload,
        batch=2,
        decode_steps=5,
        mlen=8,
        blen=2,
        vlen=8,
        tp=2,
        kvp=3,
        expert_parallel_mode="tensor_parallel",
    )
    binding = build_moe_power_event_input_binding(
        config_sha256="1" * 64,
        workload={"input_seq": 16, "output_seq": 5, "stride": 1},
        power_calibration_sha256="2" * 64,
    )
    return ledger, binding


def _target_receipt(
    ledger: dict[str, object],
    binding: dict[str, str],
) -> dict[str, object]:
    conservation = dict(ledger["assignment_conservation"])
    conservation.pop("exact")
    body: dict[str, object] = {
        "schema_version": "decode-moe-power-event-receipt/v1",
        "publication_valid": True,
        "profile_id": ledger["profile_id"],
        "candidate_id": ledger["candidate_id"],
        "model": {
            "model_name": TARGET_MODEL,
            "model_revision": TARGET_REVISION,
            "tokenizer_revision": TARGET_REVISION,
            "model_type": "qwen3_moe",
            "layers": 48,
            "experts": 128,
            "experts_per_token": 8,
        },
        "config_sha256": binding["config_sha256"],
        "artifact_provenance_sha256": "3" * 64,
        "workload": {
            "sha256": binding["workload_sha256"],
            **ledger["workload"],
        },
        "topology": {
            key: ledger["topology"][key]
            for key in (
                "tp",
                "kvp",
                "chip_count",
                "expert_parallel_mode",
            )
        },
        "dispatch_policy": ledger["dispatch_policy"],
        "assignment_conservation": conservation,
        "event_counts": ledger["event_counts"],
        "body_layout_sha256": ledger["provenance"]["body_layout_sha256"],
        "route_provenance": {
            "source": "exact_router_trace",
            "sha256": ledger["provenance"]["moe_workload_sha256"],
            "publication_valid": True,
        },
        "power_calibration": {
            "calibration_id": "power-test",
            "sha256": binding["power_calibration_sha256"],
        },
        "dense_ffn_fallback_used": False,
        "power_engine_consumed_exact_receipt": True,
    }
    return {**body, "content_hash": _hash(body)}


def test_results_v1_input_and_receipt_core_are_hash_authenticated() -> None:
    ledger, binding = _target_receipt_inputs()
    receipt = _target_receipt(ledger, binding)
    assert validate_moe_power_event_input_binding(binding)
    assert validate_moe_power_event_receipt_for_engine(
        receipt,
        ledger=ledger,
        input_binding=binding,
        calibration_id="power-test",
        calibration_sha256="2" * 64,
    )

    tampered = json.loads(json.dumps(receipt))
    tampered["event_counts"]["per_operation"]["moe_router"][
        "aggregate_system"
    ] += 1
    assert not validate_moe_power_event_receipt_for_engine(
        tampered,
        ledger=ledger,
        input_binding=binding,
        calibration_id="power-test",
        calibration_sha256="2" * 64,
    )


def test_power_bridge_prices_each_linear_signature_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signatures = (
        "LINEAR:MXINT4xMXINT4",
        "LINEAR:MXINT8xMXINT8",
        "QK:MXINT4xMXINT4",
        "PV:MXINT4xMXINT4",
        "VECTOR:FP_E3M2",
        "SELECTOR:PACKED_KV",
    )
    models = {
        signature: [float(index + 1), 0.0, 0.0]
        for index, signature in enumerate(signatures)
    }
    status = power_model.SimulatorPowerStatus(
        source_path=Path("calibration.json"),
        source_sha256="1" * 64,
        model_version="test",
        provenance_hash="2" * 64,
        required_signatures=signatures,
        available_signatures=signatures,
        missing_signatures=(),
        failures=(),
        raw={
            "event_energy_models": models,
            "leakage_power_model": [1.0, 0.0, 0.0],
            "hbm_energy_j_per_byte": 0.5,
        },
    )
    monkeypatch.setattr(power_model, "_estimate_matches_status", lambda *_: True)
    monkeypatch.setattr(
        power_model,
        "_estimate_within_calibration_domain",
        lambda *_: True,
    )
    events = [
        {"signature": signature, "count": 1, "MLEN": 8, "BLEN": 2}
        for signature in signatures
    ]
    compute = 1.0 + 2.0 + 3.0 + 4.0
    vector = 5.0
    selector = 6.0
    dynamic = compute + vector + selector
    duration = 2.0
    leakage = duration
    hbm = 2.0
    estimate = {
        "calibrated": True,
        "rankable": True,
        "missing_signatures": [],
        "events": events,
        "vector_fp": "FP_E3M2",
        "selector_enabled": True,
        "compute_dynamic_energy_j": compute,
        "vector_dynamic_energy_j": vector,
        "selector_dynamic_energy_j": selector,
        "dynamic_energy_j": dynamic,
        "leakage_energy_j": leakage,
        "leakage_power_w": 1.0,
        "hbm_energy_j": hbm,
        "hbm_bytes": 4.0,
        "total_energy_j": dynamic + leakage + hbm,
        "average_power_w": (dynamic + leakage + hbm) / duration,
        "MLEN": 8,
        "BLEN": 2,
    }
    energy = power_model.calibrated_energy_from_simulator(
        status,
        estimate,
        duration_s=duration,
    )
    assert energy is not None
    assert energy.compute_j == compute

    missing_one = power_model.SimulatorPowerStatus(
        **{
            **status.__dict__,
            "raw": {
                **status.raw,
                "event_energy_models": {
                    key: value
                    for key, value in models.items()
                    if key != "LINEAR:MXINT8xMXINT8"
                },
            },
        }
    )
    with pytest.raises(ValueError, match="uncalibrated simulator event"):
        power_model.calibrated_energy_from_simulator(
            missing_one,
            estimate,
            duration_s=duration,
        )


def test_evaluator_provenance_uses_results_v1_input_binding() -> None:
    class Backend:
        sim = type("Simulator", (), {"dims": {"num_experts": 128}})()
        output_head_location = evaluation.DECODE_MX_HEAD
        provenance = {
            "backend": "test",
            "output_head_location": evaluation.DECODE_MX_HEAD,
        }

    class Power:
        status = type(
            "Status",
            (),
            {"source_sha256": "2" * 64},
        )()
        provenance = {"engine": "test"}

    workload = evaluation.HardwareWorkload(
        input_seq=16,
        output_seq=5,
        stride=1,
        runtime_hbm_reserve_bytes=0,
    )
    evaluator = evaluation.ProductionHardwareEvaluator(
        Backend(),
        workload,
        power_engine=Power(),
        study_config_sha256="1" * 64,
    )
    binding = evaluator.provenance["moe_power_event_inputs"]
    assert validate_moe_power_event_input_binding(binding)
    assert binding == build_moe_power_event_input_binding(
        config_sha256="1" * 64,
        workload=workload.to_dict(),
        power_calibration_sha256="2" * 64,
    )
