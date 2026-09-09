"""Contracts that must hold before a hardware study is launched remotely.

Every source the evaluator hashes into its identity has to exist and resolve on
the host that will run the sweep. A rename that leaves one of these dangling
raises `FileNotFoundError` inside `ExactHardwareEvaluator.__init__`, which
otherwise surfaces only once the study starts.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from decode_dse.hardware import evaluation
from decode_dse.hardware import admission_cost
from decode_dse.hardware.admission_cost import (
    ADMISSION_CORRECTNESS_SCHEMA,
    ADMISSION_CORRECTNESS_SCOPE,
    ADMISSION_PERSISTENCE_CONTRACT,
    DECODE_FORMATS,
    MX_BLOCK_SIZE,
    RECOMPUTABLE_ADMISSION_POLICY,
    admission_correctness_status_valid,
)
from decode_dse.hardware.design_space import (
    COMPILER_TRACE_EXECUTION_MODE,
    COMPILER_TRACE_TIMING_SET_SCHEMA,
    FULL_MODEL_DECODE_SCOPE,
    LEGACY_AGGREGATE_BANDWIDTH_MODE,
    KV_HEAD_REUSE_NOOP_REASON,
    PHYSICAL_TRAFFIC_KEYS,
    STAGE_CALIBRATED_ANALYTIC_TIMING_TIER,
    TIMING_TIER_REQUIRED_VALIDITY,
    CapacityBreakdown,
    ExactHardwareSpace,
    ExactHardwareStudy,
    HardwareCandidate,
    HardwareEvaluation,
    HardwareMetrics,
    PhysicalTraffic,
    ResourceBudget,
    ResourceBudgetStatus,
    _HardwareFactorEvaluation,
    _HardwareFactorGroup,
    _FactorizedHardwareReducer,
    _content_hash,
    _factor_join_class_id,
    evaluate_publication_admission,
    kv_head_reuse_candidate_status,
    load_hardware_artifact,
    select_admitted_rows,
)
from decode_dse.hardware.lm_head_service import (
    DECODE_BF16_HEAD,
    EXTERNAL_BF16_HEAD,
    HEAD_SERVICE_MODE,
    LOCAL_HEAD_COMPUTE_IDEALIZATION,
    LOCAL_HEAD_MODE,
    composite_system_calibration_id,
    load_bf16_head_service_artifact,
    local_head_boundary_status,
    local_mx_head_breakdown_valid,
    local_head_system_calibration_id,
)
from decode_dse.hardware.packedkv_claims import AttentionTopology
from decode_dse.hardware.power_bridge import analytic_energy_from_simulator
from decode_dse.hardware.moe_power_events import (
    BF16_ROUTER_CALIBRATION_BLOCKER,
    generic_calibration_event_counts,
    validate_moe_power_event_ledger,
)
from decode_dse.hardware.selection import (
    PublicationCandidate,
    individually_validated_candidates,
    individually_validated_points,
    select_refinement_sources,
    select_final_deployment,
)
from decode_dse.legality import (
    INDIVIDUAL_VALIDATION_STAGES,
    MODEL_REQUIRED_VALIDITY_STAGES,
    PRICING_BLOCKING_STAGES,
    PRICING_RECORDED_STAGES,
    ADMISSION_BASIS,
    PackedKVRuntimeTarget,
    StackValidity,
    evaluate_stack_capability,
    scope_stack_validity,
)
from decode_dse.hardware.workload_events import (
    DecodeEvent,
    DenseDecoderShape,
    count_decode_events,
)
from decode_dse.manifest import (
    QuantizerProvenance,
    QuantizerSource,
    ResolvedImportOrigin,
    build_exhaustive_manifest,
)
from decode_dse.profiles import (
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    DecodePrecisionProfile,
)
from decode_dse.software.refinement_schedule import _hardware_point
from decode_dse.simulator_bridge import DecodeSimulator

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _serialized_measured_head_boundary(batch: int = 1) -> dict[str, object]:
    """Minimal serialized form of a valid measured-head test fixture."""

    calibration_id = "bf16-head-service-" + "c" * 64
    provenance_id = "bf16-head-provenance-" + "d" * 64
    status = {
        "schema_version": "bf16-output-head-service",
        "artifact_sha256": "a" * 64,
        "head_weight_sha256": "b" * 64,
        "cost_scope": {
            "dynamic_energy": "endpoint_only",
            "leakage": "endpoint_only",
            "link_dynamic_energy": "endpoint_receive_transmit_incremental_only",
            "measurement_link_timing": "instrumentation_driver_to_endpoint_not_deployment",
            "measurement_driver_dynamic_included": False,
            "measurement_driver_leakage_included": False,
        },
        "passed": True,
        "failures": [],
        "calibration_id": calibration_id,
        "provenance_id": provenance_id,
        "service_mode": HEAD_SERVICE_MODE,
        "service_location": "prefill_chip",
        "required_batches": [1, 4, 8],
        "numerical_policy": {
            "mac_input_dtype": "BF16",
            "accumulator_dtype": "FP32",
            "logit_dtype": "BF16",
            "selection_policy": "argmax_lowest_token_id_on_tie",
            "validation_topk": 10,
            "logit_max_abs_error_limit": 0.25,
            "logit_mean_abs_error_limit": 0.02,
            "topk_set_agreement_min": 0.9,
        },
        "numerical_validation": {
            "measurement_count": 12,
            "numerical_sample_count": 12,
            "holdout_count": 3,
            "selected_token_exact_match_count": 12,
            "sampled_logit_max_abs_error": 0.0,
            "sampled_logit_mean_abs_error_max": 0.0,
            "sampled_topk_set_agreement_min": 1.0,
        },
    }
    resource_status = {
        "schema_version": "bf16-output-head-endpoint-resources/v1",
        "artifact_sha256": "e" * 64,
        "content_hash": "f" * 64,
        "receipt_id": "bf16-head-endpoint-resources-" + "1" * 64,
        "passed": True,
        "failures": [],
        "head_service_artifact_sha256": "a" * 64,
        "head_service_calibration_id": calibration_id,
        "head_service_provenance_id": provenance_id,
        "service_mode": HEAD_SERVICE_MODE,
        "service_location": "prefill_chip",
        "deployment_scope": (
            "prefill_endpoint_with_bf16_head_service_fully_accounted"
        ),
        "service_instances": 1,
        "endpoint_instances": 1,
        "endpoint_resources_included_once": True,
        "endpoint_shared_with_decoder": False,
        "endpoint_shared_with_prefill": True,
        "decoder_resources_included": False,
        "prefill_resources_included": True,
        "measurement_driver_role": "instrumentation_only_not_deployed",
        "measurement_driver_resources_included": False,
        "endpoint": {
            "device_name": "Synthetic Accelerator",
            "device_uuid": "GPU-SYNTHETIC",
            "aggregate_compute_silicon_area_mm2": 100.0,
            "compute_die_count": 2,
            "hbm_capacity_bytes": 1_000_000,
            "hbm_bandwidth_bytes_per_s": 1_000_000_000.0,
            "prefill_resident_bytes": 100_000,
            "head_resident_bytes": 100_000,
            "runtime_reserve_bytes": 100_000,
            "resident_total_bytes": 300_000,
            "area_comparison_basis": (
                "aggregate_physical_compute_silicon_area_mm2_unscaled_excludes_hbm"
            ),
            "hbm_capacity_basis": "installed_endpoint_capacity_bytes",
            "hbm_bandwidth_basis": (
                "vendor_peak_theoretical_bytes_per_s"
            ),
        },
        "composed_link_energy": {
            "decoder_interface_energy_j_per_byte": 1e-12,
            "decoder_interface_energy_scope": (
                "decoder_request_response_interface_only_excludes_endpoint"
            ),
            "endpoint_interface_energy_scope": (
                "endpoint_receive_transmit_incremental_only"
            ),
            "measurement_driver_dynamic_included": False,
            "complete": True,
        },
        "deployment_link_timing": {
            "request_bandwidth_bytes_s": 1e9,
            "response_bandwidth_bytes_s": 1e9,
            "link_peak_bandwidth_bytes_s": 1e9,
            "request_fixed_latency_s": 1e-6,
            "response_fixed_latency_s": 1e-6,
            "scope": "plena_decoder_to_prefill_endpoint_bound_interface",
            "measurement_driver_timing_used": False,
            "complete": True,
        },
        "model_residency": {
            "precision": "BF16",
            "prefill_model_excluding_lm_head_bytes": 100_000,
            "lm_head_bytes": 100_000,
            "untied_lm_head_counted_once": True,
        },
        "evidence": {
            "input_artifact_sha256": "2" * 64,
            "specification_artifact_sha256": "3" * 64,
            "source": {
                "publisher": "Synthetic vendor",
                "title": "Synthetic specification",
                "revision": "1",
                "locator": "retained://synthetic",
                "retrieved_at_utc": "2026-08-20T00:00:00Z",
                "area_basis_statement": (
                    "two compute dies, aggregate compute silicon, HBM excluded"
                ),
                "deployment_link_basis_statement": (
                    "bound decoder-to-endpoint interface timing and energy"
                ),
            },
        },
    }
    return {
        "location": EXTERNAL_BF16_HEAD,
        "service_mode": HEAD_SERVICE_MODE,
        "scope_idealizations": [],
        "status": status,
        "estimate": {
            "calibration_id": calibration_id,
            "provenance_id": provenance_id,
            "service_mode": HEAD_SERVICE_MODE,
            "service_location": "prefill_chip",
            "batch": batch,
        },
        "comparison_estimate": None,
        "resource_status": resource_status,
    }


def test_hardware_launch_requires_mode_appropriate_timing_artifacts() -> None:
    with pytest.raises(ValueError, match="requires --compiler-trace-artifacts"):
        evaluation._validate_execution_launch(
            execution_mode=COMPILER_TRACE_EXECUTION_MODE,
            compiler_trace_artifacts=None,
            publication_timing_tier="compiler_trace_request_calibrated",
        )
    with pytest.raises(ValueError, match="cannot consume compiler-trace"):
        evaluation._validate_execution_launch(
            execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
            compiler_trace_artifacts="artifacts.json",
            publication_timing_tier="stage_calibrated_analytic",
        )
    with pytest.raises(ValueError, match="tier differs from the execution mode"):
        evaluation._validate_execution_launch(
            execution_mode=COMPILER_TRACE_EXECUTION_MODE,
            compiler_trace_artifacts="artifacts.json",
            publication_timing_tier="stage_calibrated_analytic",
        )
    with pytest.raises(ValueError, match="tier differs from the execution mode"):
        evaluation._validate_execution_launch(
            execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
            compiler_trace_artifacts=None,
            publication_timing_tier="compiler_trace_request_calibrated",
        )
    with pytest.raises(ValueError, match="unsupported publication timing tier"):
        evaluation._validate_execution_launch(
            execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
            compiler_trace_artifacts=None,
            publication_timing_tier="uncalibrated",
        )

    evaluation._validate_execution_launch(
        execution_mode=COMPILER_TRACE_EXECUTION_MODE,
        compiler_trace_artifacts="artifacts.json",
        publication_timing_tier="compiler_trace_request_calibrated",
    )
    evaluation._validate_execution_launch(
        execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
        compiler_trace_artifacts=None,
        publication_timing_tier="stage_calibrated_analytic",
    )


def test_compiler_and_emulator_validity_remain_exact_batch_scoped() -> None:
    observed = StackValidity(True, True, True, True, True)
    evidence_target = PackedKVRuntimeTarget(batch=1)
    identity = scope_stack_validity(
        observed,
        evidence_target=evidence_target,
        runtime_target=PackedKVRuntimeTarget(batch=1),
    )
    another_batch = scope_stack_validity(
        observed,
        evidence_target=evidence_target,
        runtime_target=PackedKVRuntimeTarget(batch=2),
    )
    assert identity.compiler_valid is identity.emulator_valid is True
    assert identity.rtl_valid is True
    assert another_batch.compiler_valid is None
    assert another_batch.emulator_valid is None
    assert another_batch.rtl_valid is None
    assert another_batch.dc_calibrated is True


@pytest.mark.parametrize("batch", (8, 256))
def test_exact_record_batches_have_no_frontend_capability_blocker(batch: int) -> None:
    profile = DecodePrecisionProfile.quantized(
        "MXINT4",
        "MXINT4",
        "MXINT4",
        "FP_E3M2",
    )
    capability = evaluate_stack_capability(
        profile,
        PackedKVRuntimeTarget(
            mlen=128,
            blen=2,
            hlen=16,
            batch=batch,
            kv_heads=2,
            head_dim=16,
            block_size=8,
        ),
    )

    assert all(
        issue.code != "packedkv_frontend_batch_unsupported"
        for issue in capability.issues
    )
    assert capability.stage_support["compiler"] is True


def _compiler_trace_set() -> dict[str, object]:
    return {
        "schema_version": COMPILER_TRACE_TIMING_SET_SCHEMA,
        "execution_mode": COMPILER_TRACE_EXECUTION_MODE,
        "artifact_scope": FULL_MODEL_DECODE_SCOPE,
        "request_count": 2,
        "compiler_input_descriptor_sha256": "0" * 64,
        "compiler_lowering_key_sha256": "6" * 64,
        "compiler_artifact_set_sha256": "7" * 64,
        "request_set_sha256": "1" * 64,
        "compiler_source_sha256": "2" * 64,
        "latency_library_sha256": "3" * 64,
        "request_memory_sidecar_set_sha256": "4" * 64,
        "request_memory_calibration_ids": [
            "request-latency-" + "5" * 64
        ],
        "step_composition": "max_compute_memory",
    }


def test_hardware_metrics_seal_full_model_compiler_trace_evidence() -> None:
    evidence = _compiler_trace_set()
    evidence_id = "compiler-trace-timing-" + hashlib.sha256(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metrics = HardwareMetrics(
        tpot_ms=1.0,
        tps=1_000.0,
        area_mm2=1.0,
        traffic=PhysicalTraffic(1.0, 0.0, 0.0, 0.0),
        capacity=CapacityBreakdown(1, 0, 0, 2),
        algorithmic_bottleneck="compute",
        realized_bottleneck="compute",
        frac_algorithmic_memory_bound=0.0,
        frac_realized_memory_bound=0.0,
        frac_serialization_bound=0.0,
        timing_calibrated=True,
        timing_evidence_id=evidence_id,
        timing_reason="compiler_trace_timing_validated",
        execution_mode=COMPILER_TRACE_EXECUTION_MODE,
        compiler_trace_timing=evidence,
        output_head_location=DECODE_BF16_HEAD,
        output_head_status=local_head_boundary_status(),
    )
    serialized = metrics.to_dict()
    assert serialized["execution_mode"] == COMPILER_TRACE_EXECUTION_MODE
    assert serialized["compiler_trace_timing"] == evidence
    assert serialized["bandwidth_calibration_id"] is None
    assert serialized["memory_timing_calibrated"] is True


def test_compiler_trace_request_is_immutably_bound_to_one_dse_point() -> None:
    simulator = DecodeSimulator("qwen3-32b")
    precision = simulator.make_precision(
        attn_w=4,
        ffn_w=4,
        kv=4,
        act_w=4,
    )
    overrides = {
        "MLEN": 128,
        "BLEN": 2,
        "VLEN": 128,
        "HLEN": 16,
        "TP": 1,
        "KVP": 1,
        "LINK_PORTS": 0,
        "SRAM_POLICY": "streaming",
        "KV_HEAD_REUSE": False,
        "DRAIN_OVERLAPPED": False,
        **simulator.hbm_overrides("HBM2", 8),
    }
    hardware = simulator.base_hw.model_copy(update=overrides)
    point = simulator.compiler_trace_point(
        precision,
        hardware=hardware,
        overrides=overrides,
        batch=1,
        input_seq=16,
        output_seq=2,
        stride=1,
        n_chips=1,
        hbm_gen="HBM2",
        hbm_channels=8,
        kv_layout="dense_selector",
        runtime_hbm_reserve_bytes=0,
        output_head_location="external_bf16_service",
    )
    descriptor = point.to_dict()
    assert descriptor["model"]["layer_scope"] == "all_decoder_layers"
    assert descriptor["hardware"]["array_geometry"] == {
        "mlen": 128,
        "blen": 2,
        "vlen": 128,
        "hlen": 16,
    }
    assert descriptor["hardware"]["topology"]["chip_count"] == 1
    assert descriptor["precision"]["specification"] == precision.spec

    # `to_dict` returns a copy, so caller mutation cannot change the key.
    original_digest = point.descriptor_sha256
    descriptor["serving"]["batch"] = 99
    assert point.descriptor_sha256 == original_digest

    from compiler_trace_timing import (
        ArrayGeometry,
        CompilerTraceTimingRequest,
        HBMOperatingPoint,
    )

    class Binder:
        def bind(self, bound_point):
            inputs = bound_point.to_dict()
            geometry = ArrayGeometry(
                **inputs["hardware"]["array_geometry"]
            )
            hbm = HBMOperatingPoint(
                **inputs["hardware"]["hbm_timing_geometry"]
            )

            def request(context_tokens):
                return CompilerTraceTimingRequest(
                    compiler_inputs_sha256=bound_point.descriptor_sha256,
                    compiler_source_sha256="a" * 64,
                    context_tokens=context_tokens,
                    batch=inputs["serving"]["batch"],
                    geometry=geometry,
                    hbm=hbm,
                    frequency_hz=inputs["compiler"]["frequency_hz"],
                )

            return request

    simulator.trace_request_binder = Binder()
    request_factory = simulator._bind_trace_request_factory(point)
    request = request_factory(16)
    assert request.compiler_inputs_sha256 == original_digest
    assert request.context_tokens == 16


def test_provenance_modules_import_and_hash() -> None:
    digests = evaluation._module_source_digests()
    assert set(digests) == set(evaluation._PROVENANCE_MODULES)
    for name, digest in digests.items():
        assert SHA256.fullmatch(digest), name


def test_lazy_exact_hardware_exports_resolve() -> None:
    import decode_dse.hardware as hardware

    for name in hardware._EXACT_EXPORTS:
        assert getattr(hardware, name) is getattr(evaluation, name)


def _handoff_payload() -> dict[str, object]:
    return {
        "schema_version": evaluation.PREFILL_HANDOFF_INPUT_SCHEMA,
        "model": {"name": "synthetic/model", "revision": "a" * 40},
        "workload": {"prompt_tokens": 16, "generation_tokens": 2},
        "prefill": {
            "precision": "BF16",
            "scope": evaluation.PREFILL_MEASUREMENT_SCOPE,
            "measurements": [
                {
                    "batch": batch,
                    "latency_s": 0.01 * batch,
                    "energy_j": 0.1 * batch,
                    "latency_evidence_id": (
                        "prefill-latency-" + f"{batch:064x}"
                    ),
                    "energy_evidence_id": (
                        "prefill-energy-" + f"{batch + 8:064x}"
                    ),
                    "evidence_tier": "measured",
                }
                for batch in (1, 4)
            ],
        },
        "schedule": {
            "decode_ready_delay_s": 0.02,
            "prefill_stall_power_w": 50.0,
            "decode_idle_power_w": 20.0,
        },
        "links": {
            "direct_generation": "nvlink4",
            "host_generation": "pcie5",
            "direct_energy_pj_per_bit": 1.5,
            "host_energy_pj_per_bit": 10.0,
            "evidence_id": "link-energy-" + "b" * 64,
            "evidence_tier": "declared_sensitivity",
        },
        "admission": {
            "bandwidth_bytes_per_s": 900e9,
            "quantize_energy_j_per_element": 1e-12,
            "memory_energy_j_per_byte": 2e-12,
            "calibrated": True,
            "calibration_id": "admission-" + "c" * 64,
            "evidence_tier": "emulator_measured",
        },
    }


def test_prefill_handoff_artifact_is_explicit_and_batch_complete(tmp_path) -> None:
    source = tmp_path / "prefill_handoff.json"
    source.write_text(json.dumps(_handoff_payload()), encoding="utf-8")
    workload = evaluation.HardwareWorkload(16, 2, 1, 0)

    artifact = evaluation.PrefillHandoffArtifact.load(
        source,
        model_name="synthetic/model",
        model_revision="a" * 40,
        workload=workload,
        required_batches=(1, 4),
    )
    assert artifact.publication_rankable is True
    assert artifact.measurement(4).latency_s == pytest.approx(0.04)
    assert artifact.to_status()["artifact_id"] == artifact.artifact_id
    assert "source_path" not in artifact.to_status()

    with pytest.raises(ValueError, match="lacks required batches: 8"):
        evaluation.PrefillHandoffArtifact.load(
            source,
            model_name="synthetic/model",
            model_revision="a" * 40,
            workload=workload,
            required_batches=(1, 4, 8),
        )


def test_provenance_modules_are_the_evaluator_dependencies() -> None:
    """The hashed set must cover every decode_dse module the evaluator imports.

    A new dependency that is not hashed would let a logic change alter results
    without changing the evaluator identity.
    """

    imported = {
        f"decode_dse.{name}"
        for name in ("legality", "manifest", "profiles", "simulator_bridge")
    } | {
        f"decode_dse.hardware.{name}"
        for name in (
            "admission_cost",
            "design_space",
            "evaluation",
            "lm_head_service",
            "power_model",
            "synthesis_anchor",
            "workload_events",
        )
    }
    hashed = set(evaluation._PROVENANCE_MODULES)
    # simulator_bridge is hashed in the backend provenance, not the evaluator's.
    assert imported - hashed == {"decode_dse.simulator_bridge"}


def test_simulator_sources_resolve() -> None:
    root = evaluation.simulator_root()
    missing = [
        relative
        for relative in evaluation._SIMULATOR_SOURCES.values()
        if not (root / relative).is_file()
    ]
    assert not missing, f"missing simulator sources under {root}: {missing}"

    digests = evaluation._simulator_source_digests()
    assert set(digests) == set(evaluation._SIMULATOR_SOURCES)
    for name, digest in digests.items():
        assert SHA256.fullmatch(digest), name


def test_bandwidth_calibration_sources_resolve() -> None:
    root = evaluation.simulator_root()
    missing = [
        relative
        for relative in evaluation._BANDWIDTH_CALIBRATION_SOURCES
        if not (root / relative).is_file()
    ]
    assert not missing, f"missing calibration tables under {root}: {missing}"


def test_backend_provenance_field_names_are_unique() -> None:
    """Backend provenance flattens several sources into one dict."""

    reserved = {
        "model_json_sha256",
        "settings_sha256",
        "isa_sha256",
        "timing_evidence_sha256",
        "simulator_bridge_sha256",
        "head_service_model_sha256",
    }
    assert not reserved & set(evaluation._SIMULATOR_SOURCES)


def test_multichip_space_emits_only_legal_factorizations() -> None:
    space = ExactHardwareSpace(
        mlen=(128,),
        blen=(4,),
        hlen=(128,),
        batch=(1,),
        hbm_channels=(8,),
        chip_count=(1, 4),
        link_ports=(1, 2),
        sram_policy=("streaming",),
        attention_heads=32,
        kv_heads=8,
    )
    candidates = space.candidates(4096)
    assert candidates
    assert all(point.tp * point.kvp == point.chip_count for point in candidates)
    assert all(
        point.link_ports >= int(point.tp > 1) + int(point.kvp > 1)
        for point in candidates
        if point.chip_count > 1
    )
    assert all(32 % point.tp == 0 and 8 % point.tp == 0 for point in candidates)


def test_rank_local_padding_keeps_mlen_larger_than_hidden_in_exact_grid() -> None:
    space = ExactHardwareSpace(
        mlen=(1024, 2048, 4096),
        blen=(8,),
        hlen=(128,),
        batch=(1,),
        hbm_channels=(8,),
        chip_count=(1, 4, 8),
        tp=(1, 2, 4, 8),
        kvp=(1, 2, 4),
        link_ports=(1, 2),
        sram_policy=("streaming",),
        kv_head_reuse=(False,),
        drain_overlapped=(False,),
        expert_parallel_mode=("tensor_parallel", "expert_id_parallel"),
        allow_rank_local_mlen_padding=True,
        attention_heads=32,
        kv_heads=4,
    )
    candidates = space.candidates(2048)

    assert space.candidate_count(2048) == len(candidates)
    assert {candidate.mlen for candidate in candidates} == {1024, 2048, 4096}
    assert {candidate.tp for candidate in candidates} <= {1, 2, 4}
    assert all(candidate.tp != 8 for candidate in candidates)
    assert all(
        candidate.expert_parallel_mode == "tensor_parallel"
        for candidate in candidates
        if candidate.tp == 1
    )
    assert {
        candidate.expert_parallel_mode
        for candidate in candidates
        if candidate.tp > 1
    } == {"tensor_parallel", "expert_id_parallel"}


def test_resource_budget_enforces_all_aggregate_limits() -> None:
    budget = ResourceBudget(
        aggregate_area_limit_mm2=100.0,
        aggregate_hbm_capacity_limit_bytes=1_000,
        aggregate_hbm_bandwidth_limit_bytes_per_s=2_000.0,
        reference_system="test-reference",
    )
    status = ResourceBudgetStatus(
        aggregate_area_mm2=99.0,
        aggregate_hbm_capacity_bytes=1_001,
        aggregate_hbm_bandwidth_bytes_per_s=1_999.0,
        aggregate_multiplier_count=4096,
        budget=budget,
    )
    assert status.area_feasible
    assert not status.hbm_capacity_feasible
    assert status.hbm_bandwidth_feasible
    assert not status.feasible


def test_legacy_candidate_evidence_normalises_without_rewriting_raw_identity() -> None:
    raw = {
        "MLEN": 128,
        "BLEN": 4,
        "VLEN": 128,
        "HLEN": 128,
        "BATCH": 1,
        "HBM_CHANNELS": 8,
        "HBM_GENERATION": "HBM2",
        "CHIP_COUNT": 1,
    }
    candidate = HardwareCandidate.from_dict(
        raw,
        allow_legacy_single_chip=True,
    )
    assert candidate.to_legacy_dict() == raw
    assert candidate.tp == candidate.kvp == 1
    assert candidate.link_ports == 0
    assert candidate.sram_policy == "streaming"
    assert candidate.to_dict() == {
        **raw,
        "TP": 1,
        "KVP": 1,
        "LINK_PORTS": 0,
        "SRAM_POLICY": "streaming",
    }


def test_legacy_multichip_evidence_is_not_given_new_partition_semantics() -> None:
    raw = {
        "MLEN": 128,
        "BLEN": 4,
        "VLEN": 128,
        "HLEN": 128,
        "BATCH": 1,
        "HBM_CHANNELS": 8,
        "HBM_GENERATION": "HBM2",
        "CHIP_COUNT": 2,
    }
    with pytest.raises(ValueError, match="legacy multi-chip"):
        HardwareCandidate.from_dict(raw, allow_legacy_single_chip=True)


def test_pre_e2_candidate_identity_is_stable_and_explicit_false_is_distinct() -> None:
    pre_e2 = HardwareCandidate(
        mlen=1024,
        blen=2,
        vlen=1024,
        hlen=128,
        batch=4,
        hbm_channels=8,
        hbm_generation="HBM2",
        chip_count=1,
        tp=1,
        kvp=1,
        link_ports=0,
        sram_policy="streaming",
    )
    raw = pre_e2.to_dict()
    expected_id = "hw-" + hashlib.sha256(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    restored = HardwareCandidate.from_dict(raw)
    explicit_false = HardwareCandidate(
        mlen=pre_e2.mlen,
        blen=pre_e2.blen,
        vlen=pre_e2.vlen,
        hlen=pre_e2.hlen,
        batch=pre_e2.batch,
        hbm_channels=pre_e2.hbm_channels,
        hbm_generation=pre_e2.hbm_generation,
        chip_count=pre_e2.chip_count,
        tp=pre_e2.tp,
        kvp=pre_e2.kvp,
        link_ports=pre_e2.link_ports,
        sram_policy=pre_e2.sram_policy,
        kv_head_reuse=False,
        drain_overlapped=False,
    )
    assert restored.to_dict() == raw
    assert restored.candidate_id == pre_e2.candidate_id == expected_id
    assert explicit_false.to_dict() == {
        **raw,
        "KV_HEAD_REUSE": False,
        "DRAIN_OVERLAPPED": False,
    }
    assert explicit_false.candidate_id != pre_e2.candidate_id
    invalid = dict(explicit_false.to_dict())
    invalid["KV_HEAD_REUSE"] = 1
    with pytest.raises(TypeError, match="must be boolean"):
        HardwareCandidate.from_dict(invalid)


def test_explicit_expert_mode_uses_roundtrippable_e3_candidate_schema() -> None:
    candidate = HardwareCandidate(
        mlen=4096,
        blen=8,
        vlen=4096,
        hlen=128,
        batch=1,
        hbm_channels=8,
        hbm_generation="HBM3E",
        chip_count=4,
        tp=4,
        kvp=1,
        link_ports=1,
        sram_policy="streaming",
        expert_parallel_mode="expert_id_parallel",
    )
    raw = candidate.to_dict()

    assert set(raw) == set(HardwareCandidate.E3_FIELDS)
    assert raw["KV_HEAD_REUSE"] is False
    assert raw["DRAIN_OVERLAPPED"] is False
    assert raw["EXPERT_PARALLEL_MODE"] == "expert_id_parallel"
    assert HardwareCandidate.from_dict(raw).to_dict() == raw


def test_reuse_search_enumerates_only_fp_sram_legal_geometries() -> None:
    space = ExactHardwareSpace(
        mlen=(1024,),
        blen=(2, 4),
        hlen=(128,),
        batch=(4,),
        hbm_channels=(8,),
        chip_count=(1,),
        sram_policy=("streaming",),
        kv_head_reuse=(True,),
        drain_overlapped=(False,),
        attention_heads=64,
        kv_heads=8,
        fp_sram_depth=512,
    )
    candidates = space.candidates(4096)
    assert candidates
    assert {candidate.blen for candidate in candidates} == {2}
    assert all(candidate.kv_head_reuse for candidate in candidates)
    assert all(candidate.architecture_knobs_explicit for candidate in candidates)


def test_reuse_legality_is_rank_local_and_count_matches_iterator() -> None:
    space = ExactHardwareSpace(
        mlen=(1024,),
        blen=(4,),
        hlen=(128,),
        batch=(4,),
        hbm_channels=(8,),
        chip_count=(1, 2, 4, 8),
        tp=(1, 2, 4, 8),
        link_ports=(2,),
        sram_policy=("streaming",),
        kv_head_reuse=(False, True),
        drain_overlapped=(False,),
        attention_heads=64,
        kv_heads=8,
        fp_sram_depth=512,
    )
    candidates = space.candidates(4096)
    assert space.candidate_count(4096) == len(candidates)
    by_tp = {
        tp: tuple(candidate for candidate in candidates if candidate.tp == tp)
        for tp in (1, 2, 4, 8)
    }
    assert all(by_tp.values())
    assert not any(candidate.kv_head_reuse for candidate in by_tp[1])
    assert all(
        {candidate.kv_head_reuse for candidate in by_tp[tp]} == {False, True}
        for tp in (2, 4)
    )
    assert not any(candidate.kv_head_reuse for candidate in by_tp[8])
    for candidate in candidates:
        local_kv_heads = 8 // candidate.tp
        if candidate.kv_head_reuse:
            assert local_kv_heads <= candidate.mlen // candidate.hlen
            assert (
                6
                + 3
                * candidate.blen
                * (candidate.mlen // candidate.hlen)
                * local_kv_heads
                <= space.fp_sram_depth
            )


def test_qwen3_target_prunes_tp4_single_local_kv_head_reuse_noop() -> None:
    space = ExactHardwareSpace(
        mlen=(1024,),
        blen=(2,),
        hlen=(128,),
        batch=(4,),
        hbm_channels=(8,),
        chip_count=(1, 2, 4),
        tp=(1, 2, 4),
        kvp=(1,),
        link_ports=(1,),
        sram_policy=("streaming",),
        kv_head_reuse=(False, True),
        drain_overlapped=(False,),
        attention_heads=32,
        kv_heads=4,
    )
    candidates = space.candidates(2048)
    by_tp = {
        tp: tuple(candidate for candidate in candidates if candidate.tp == tp)
        for tp in (1, 2, 4)
    }

    assert all(by_tp.values())
    assert {candidate.kv_head_reuse for candidate in by_tp[1]} == {False, True}
    assert {candidate.kv_head_reuse for candidate in by_tp[2]} == {False, True}
    assert {candidate.kv_head_reuse for candidate in by_tp[4]} == {False}
    no_op = kv_head_reuse_candidate_status(
        enabled=True,
        mlen=1024,
        blen=2,
        hlen=128,
        local_kv_heads=1,
    )
    assert no_op == {
        "enabled": True,
        "legal": False,
        "structural_no_op": True,
        "legality_reason": KV_HEAD_REUSE_NOOP_REASON,
        "local_kv_heads": 1,
        "required_fp_sram_slots": 0,
        "available_fp_sram_slots": 512,
    }


def test_rank_local_heads_bind_selector_capability_and_reuse_area() -> None:
    simulator = evaluation.simulator_root()
    if str(simulator) not in sys.path:
        sys.path.insert(0, str(simulator))
    from analytic_models.disagg_serve.packed_kv import (
        architecture_option_area_mm2,
        kv_head_reuse_status,
    )

    shape = evaluation.DenseDecoderShape(
        hidden_size=4096,
        intermediate_size=11008,
        attention_heads=64,
        kv_heads=8,
        head_dim=64,
        layers=32,
        vocab_size=32000,
    )
    profile = DecodePrecisionProfile.quantized(
        "MXINT8", "MXINT8", "MXINT8", "FP_E3M2"
    )
    areas = {}
    for tp in (1, 2, 4):
        candidate = HardwareCandidate(
            mlen=1024,
            blen=4,
            vlen=1024,
            hlen=128,
            batch=4,
            hbm_channels=8,
            hbm_generation="HBM2",
            chip_count=tp,
            tp=tp,
            kvp=1,
            link_ports=(0 if tp == 1 else 1),
            sram_policy="streaming",
            kv_head_reuse=True,
            drain_overlapped=False,
        )
        partition = evaluation.RankLocalAttentionGeometry.bind(shape, candidate)
        assert partition.local_attention_heads == 64 // tp
        assert partition.local_kv_heads == 8 // tp
        assert partition.global_kv_heads == 8
        target = evaluation._selector_runtime_target(profile, candidate, shape)
        assert target.kv_heads == 8 // tp
        reuse = kv_head_reuse_status(
            enabled=True,
            mlen=candidate.mlen,
            hlen=candidate.hlen,
            blen=candidate.blen,
            kv_heads=partition.local_kv_heads,
        )
        assert reuse["kv_heads"] == 8 // tp
        assert reuse["supported"] is (tp > 1)
        areas[tp] = architecture_option_area_mm2(
            mlen=candidate.mlen,
            hlen=candidate.hlen,
            kv_heads=partition.local_kv_heads,
            kv_head_reuse=True,
            drain_overlapped=False,
        )["area_mm2_per_chip"]
    assert areas[1] > areas[2] > areas[4]
    no_op = kv_head_reuse_status(
        enabled=True,
        mlen=1024,
        hlen=128,
        blen=4,
        kv_heads=1,
    )
    assert no_op["supported"] is False
    assert no_op["legality_reason"] == KV_HEAD_REUSE_NOOP_REASON
    with pytest.raises(ValueError, match="structural_no_op"):
        architecture_option_area_mm2(
            mlen=1024,
            hlen=128,
            kv_heads=1,
            kv_head_reuse=True,
            drain_overlapped=False,
        )


def test_exact_search_knob_choice_respects_area_and_capacity_tradeoffs() -> None:
    simulator = evaluation.simulator_root()
    if str(simulator) not in sys.path:
        sys.path.insert(0, str(simulator))
    from analytic_models.disagg_serve.packed_kv import (
        architecture_option_area_mm2,
    )

    space = ExactHardwareSpace(
        mlen=(1024,),
        blen=(2,),
        hlen=(128,),
        batch=(4, 8),
        hbm_channels=(8,),
        chip_count=(1,),
        sram_policy=("streaming",),
        kv_head_reuse=(False, True),
        drain_overlapped=(False,),
        attention_heads=64,
        kv_heads=8,
    )
    candidates = space.candidates(4096)
    base_area = 100.0

    def aggregate_area(candidate: HardwareCandidate) -> float:
        option = architecture_option_area_mm2(
            mlen=candidate.mlen,
            hlen=candidate.hlen,
            kv_heads=8,
            kv_head_reuse=candidate.kv_head_reuse,
            drain_overlapped=candidate.drain_overlapped,
        )
        return base_area + float(option["area_mm2_per_chip"])

    reuse_addition = min(
        aggregate_area(candidate) - base_area
        for candidate in candidates
        if candidate.kv_head_reuse
    )
    assert reuse_addition > 0

    def choose(area_limit: float) -> HardwareCandidate:
        feasible = []
        for candidate in candidates:
            capacity_feasible = candidate.batch <= 4
            area_status = ResourceBudgetStatus(
                aggregate_area_mm2=aggregate_area(candidate),
                aggregate_hbm_capacity_bytes=1_000,
                aggregate_hbm_bandwidth_bytes_per_s=1_000.0,
                aggregate_multiplier_count=(
                    candidate.mlen * candidate.blen
                ),
                budget=ResourceBudget(
                    aggregate_area_limit_mm2=area_limit,
                    aggregate_hbm_capacity_limit_bytes=1_000,
                    aggregate_hbm_bandwidth_limit_bytes_per_s=1_000.0,
                    reference_system="focused-search",
                ),
            )
            if capacity_feasible and area_status.feasible:
                feasible.append(candidate)
        return max(
            feasible,
            key=lambda candidate: (
                candidate.batch * (8 if candidate.kv_head_reuse else 1),
                candidate.candidate_id,
            ),
        )

    loose = choose(base_area + reuse_addition)
    tight = choose(base_area + reuse_addition / 2)
    assert loose.batch == tight.batch == 4
    assert loose.kv_head_reuse is True
    assert tight.kv_head_reuse is False
    assert all(candidate.batch == 8 for candidate in candidates if candidate.batch > 4)


def test_full_model_spaces_have_exact_bounded_structural_counts() -> None:
    repository = Path(__file__).resolve().parents[2]
    # The Qwen grid is pruned to compiler-legal geometry with searchable
    # HBM channels.  The Llama grid is now restricted the same way on the two
    # architecture-option axes: drain-overlapped execution has no anchor in
    # the timing evidence and the packed-q1 timing contracts have not been
    # generated, so KV_HEAD_REUSE / DRAIN_OVERLAPPED true candidates could
    # only ever be recorded as timing-uncalibrated.  This is a declared
    # evidence-availability restriction, not objective pruning.  Both grids
    # search the same three measured HBM2 calibration groups (8/16/32
    # interface units), so the channel axis contributes an identical factor
    # of three to each count.
    expected = {
        "qwen3_32b": 22_320,
        "llama3_1_8b": 753_984,
    }
    for name, count in expected.items():
        config = json.loads(
            (repository / "decode_dse" / "configs" / f"{name}.json")
            .read_text(encoding="utf-8")
        )
        space = ExactHardwareSpace.from_study_config(config)
        hidden = int(config["model_architecture"]["hidden_size"])
        assert space.candidate_count(hidden) == count
        assert count < 2_000_000


def _synthetic_manifest():
    provenance = QuantizerProvenance(
        sources=(
            QuantizerSource(
                component="synthetic",
                path="decode_dse/profiles.py",
                sha256="a" * 64,
            ),
        ),
        resolved_imports=(
            ResolvedImportOrigin(
                module="chop.nn.quantizers.mxint.fake",
                path="mase/src/chop/nn/quantizers/mxint/fake.py",
            ),
        ),
    )
    return build_exhaustive_manifest(
        "synthetic-model",
        "b" * 40,
        {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "use_qk_norm": False,
        },
        provenance,
        "c" * 40,
    )


def _qwen_moe_backend(tmp_path: Path):
    study_config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "qwen3_30b_a3b_thinking_2507.json"
        ).read_text(encoding="utf-8")
    )
    model_path = tmp_path / "qwen3_moe.json"
    model_path.write_text(
        json.dumps(study_config["model_architecture"]),
        encoding="utf-8",
    )
    backend = object.__new__(evaluation.DecodeSimulatorBackend)
    backend.sim = DecodeSimulator(str(model_path))
    backend.output_head_location = evaluation.DECODE_MX_HEAD
    backend.calibrated_bandwidth = False
    backend._provenance = {"backend": "synthetic-qwen-moe-test"}
    backend.capacity_model_id = "test-rank-local-capacity"
    backend.traffic_ledger_id = "test-rank-local-traffic"
    backend._resource_ledger_cache = {}
    backend._resource_area_cache = {}
    return backend


def test_qwen_moe_backend_binds_local_head_in_preflight_and_evaluate(
    tmp_path: Path,
) -> None:
    manifest = _synthetic_manifest()
    entry = next(
        item
        for item in manifest.entries
        if (
            item.profile.kind == PROFILE_KIND_QUANTIZED
            and item.legality.hardware_candidate
            and item.profile.local_head_contract[
                "operand_family_deployment_supported"
            ]
        )
    )
    candidate = HardwareCandidate(
        mlen=1024,
        blen=8,
        vlen=1024,
        hlen=128,
        batch=1,
        hbm_channels=24,
        hbm_generation="HBM3",
        chip_count=4,
        tp=4,
        kvp=1,
        link_ports=1,
        sram_policy="streaming",
        kv_head_reuse=False,
        drain_overlapped=False,
        expert_parallel_mode="tensor_parallel",
    )
    workload = evaluation.HardwareWorkload(
        input_seq=16,
        output_seq=1,
        stride=1,
        runtime_hbm_reserve_bytes=0,
    )
    backend = _qwen_moe_backend(tmp_path)
    area_preflight_evaluator = evaluation.ProductionHardwareEvaluator(
        backend,
        workload,
        publication_timing_tier=STAGE_CALIBRATED_ANALYTIC_TIMING_TIER,
    )
    with pytest.raises(ValueError, match="exact power-event receipt"):
        area_preflight_evaluator._area_events(entry.profile, candidate)

    preflight = backend.resource_preflight(entry, candidate, workload)
    observation = backend.evaluate(entry, candidate, workload)

    assert preflight.capacity.slowest_rank_required_bytes is not None
    assert observation.body_physical_layout is not None
    assert observation.local_output_head is not None
    assert observation.local_output_head["passed"] is True
    assert observation.local_output_head["failures"] == []
    assert observation.local_output_head["compiler_lowering_receipt"] is None
    assert observation.local_output_head["compiler_lowering_blocker"] == (
        "tensor_parallel_local_head_compiler_lowering_unavailable"
    )
    power_ledger = observation.moe_power_event_ledger
    assert power_ledger is not None
    assert validate_moe_power_event_ledger(power_ledger)
    assert power_ledger["dense_ffn_fallback_used"] is False
    assert BF16_ROUTER_CALIBRATION_BLOCKER in power_ledger["blockers"]
    assert observation.moe_power_event_receipt is None
    conservation = power_ledger["assignment_conservation"]
    assert conservation["expected_logical_assignments"] == 1 * 8 * 48 * 1
    assert conservation["expected_physical_executed_assignments"] == 384
    operations = power_ledger["event_counts"]["per_operation"]
    assert operations["moe_hidden_dispatch"]["aggregate_system"] == 0
    assert operations["moe_expert_output_collective"]["aggregate_system"] > 0

    shape = DenseDecoderShape.from_mapping(backend.sim.dims)
    common = dict(
        input_seq=workload.input_seq,
        output_seq=workload.output_seq,
        batch=candidate.batch,
        mlen=candidate.mlen,
        blen=candidate.blen,
        vlen=candidate.vlen,
        hlen=candidate.hlen,
        linear_signature=(
            f"LINEAR:{evaluation._simulator_token(entry.profile.weight_format)}"
            f"x{evaluation._simulator_token(entry.profile.activation_format)}"
        ),
        qk_signature=(
            f"QK:{evaluation._simulator_token(entry.profile.key_format)}"
            f"x{evaluation._simulator_token(entry.profile.activation_format)}"
        ),
        pv_signature=(
            f"PV:{evaluation._simulator_token(entry.profile.value_format)}"
            f"x{evaluation._simulator_token(entry.profile.activation_format)}"
        ),
        vector_signature=f"VECTOR:{entry.profile.vector_format}",
        include_local_output_head_padding=True,
        tp=candidate.tp,
        kvp=candidate.kvp,
        lm_head_signature=(
            f"LINEAR:{evaluation._simulator_token(entry.profile.weight_format)}"
            f"x{evaluation._simulator_token(entry.profile.activation_format)}"
        ),
    )
    dense_proxy = dict(
        (event.signature, event.count)
        for event in count_decode_events(shape, **common)
    )
    native_base = dict(
        (event.signature, event.count)
        for event in count_decode_events(
            shape,
            include_dense_ffn=False,
            **common,
        )
    )
    observed = {event.signature: event.count for event in observation.events}
    native_additions = dict(generic_calibration_event_counts(power_ledger))
    linear = common["linear_signature"]
    assert observed[linear] == native_base[linear] + native_additions[linear]
    assert observed[linear] != dense_proxy[linear] + native_additions[linear]
    power_engine = object.__new__(evaluation.SimulatorPowerEngine)
    power_engine.status = SimpleNamespace(
        calibration_id="sim-power-test",
        source_sha256="4" * 64,
    )
    with pytest.raises(ValueError, match=BF16_ROUTER_CALIBRATION_BLOCKER):
        power_engine.evaluate(entry.profile, candidate, observation)
    with pytest.raises(ValueError, match=BF16_ROUTER_CALIBRATION_BLOCKER):
        power_engine.anchor_prediction(entry.profile, candidate, observation)
    with pytest.raises(ValueError, match=BF16_ROUTER_CALIBRATION_BLOCKER):
        power_engine.hbm_energy_per_token(observation)
    with pytest.raises(ValueError, match=BF16_ROUTER_CALIBRATION_BLOCKER):
        power_engine.area_mm2(
            entry.profile,
            candidate,
            observation.events,
            moe_power_event_ledger=power_ledger,
        )
    assert local_mx_head_breakdown_valid(observation.local_output_head)
    padding = observation.local_output_head["padding_preparation"]
    assert len(padding["padded_vocab_mask_by_tp_rank"]) == 4
    assert padding["padded_vocab_mask_vector_events_system"] == sum(
        item["vector_events"]
        for item in padding["padded_vocab_mask_by_tp_rank"]
    )
    invalid = json.loads(json.dumps(observation.local_output_head))
    invalid["padding_preparation"]["padded_vocab_mask_by_tp_rank"][0][
        "vector_events"
    ] -= 1
    assert not local_mx_head_breakdown_valid(invalid)
    undercounted = json.loads(json.dumps(observation.local_output_head))
    undercounted["cycles_per_batch_step"]["serving_slowest_rank"] = 1.0
    assert not local_mx_head_breakdown_valid(undercounted)

    priced_timing = replace(
        observation,
        timing_calibrated=True,
        timing_evidence_id="timing-moe-power-contract",
        timing_reason="timing_calibrated_emulator_tier",
        bandwidth_calibration_id="bandwidth-moe-power-contract",
        realized_bottleneck=(
            "memory"
            if observation.realized_bottleneck == "unavailable"
            else observation.realized_bottleneck
        ),
    )

    class StubBackend:
        output_head_location = evaluation.DECODE_MX_HEAD
        provenance = {
            "backend": "native-moe-power-contract",
            "output_head_location": evaluation.DECODE_MX_HEAD,
        }

        @staticmethod
        def evaluate(*_args, **_kwargs):
            return priced_timing

    evaluator = evaluation.ProductionHardwareEvaluator(
        StubBackend(),
        workload,
        publication_timing_tier=STAGE_CALIBRATED_ANALYTIC_TIMING_TIER,
    )
    outcome = evaluator(
        entry,
        candidate,
        {
            "state": "succeeded",
            "result": {"mean_nll": 1.0, "token_count": 64},
        },
    )
    assert outcome.error_code == "moe_power_event_unrankable"
    assert outcome.metrics is not None
    emitted = outcome.metrics.to_dict()
    assert emitted["moe_power_event_ledger"]["dense_ffn_fallback_used"] is False
    assert emitted["moe_power_event_receipt"] is None
    assert emitted["calibrated_energy"] is None
    assert emitted["energy_per_token_j"] is None


def test_bf16_reference_is_explicitly_a_non_plena_control(
    tmp_path: Path,
) -> None:
    entry = next(
        item
        for item in _synthetic_manifest().entries
        if item.profile.kind == PROFILE_KIND_BF16_REFERENCE
    )
    candidate = HardwareCandidate(
        mlen=1024,
        blen=8,
        vlen=1024,
        hlen=128,
        batch=1,
        hbm_channels=8,
        hbm_generation="HBM2",
    )
    workload = evaluation.HardwareWorkload(16, 1, 1, 0)
    backend = _qwen_moe_backend(tmp_path)

    with pytest.raises(ValueError, match="numerical/B200 control"):
        backend.resource_preflight(entry, candidate, workload)
    with pytest.raises(ValueError, match="numerical/B200 control"):
        backend.evaluate(entry, candidate, workload)


def _numerical_row(entry, mean_nll: float) -> dict[str, object]:
    return {
        "profile_id": entry.profile_id,
        "state": "succeeded",
        "attempt": 1,
        "result": {"mean_nll": mean_nll, "token_count": 32},
    }


def test_e1_handoff_analysis_uses_explicit_artifact_and_decode_costs(
    tmp_path,
) -> None:
    source = tmp_path / "prefill_handoff.json"
    source.write_text(json.dumps(_handoff_payload()), encoding="utf-8")
    workload = evaluation.HardwareWorkload(16, 2, 1, 0)
    artifact = evaluation.PrefillHandoffArtifact.load(
        source,
        model_name="synthetic/model",
        model_revision="a" * 40,
        workload=workload,
        required_batches=(1,),
    )
    entry = next(
        row
        for row in _synthetic_manifest().entries
        if (
            row.profile.kind == PROFILE_KIND_QUANTIZED
            and row.legality.hardware_candidate
        )
    )
    candidate = HardwareCandidate(
        mlen=128,
        blen=2,
        vlen=128,
        hlen=16,
        batch=1,
        hbm_channels=8,
        hbm_generation="HBM2",
        chip_count=1,
        tp=1,
        kvp=1,
        link_ports=0,
        sram_policy="streaming",
    )

    class Simulator:
        dims = {
            "hidden": 128,
            "intermediate": 256,
            "layers": 2,
            "heads": 8,
            "kv_heads": 2,
            "head_dim": 16,
            "vocab": 256,
        }

        @staticmethod
        def make_precision(**_kwargs):
            return SimpleNamespace(spec={"kv_bits": 4.0})

    backend = object.__new__(evaluation.DecodeSimulatorBackend)
    backend.sim = Simulator()
    analysis = backend.evaluate_handoff(
        entry,
        candidate,
        workload,
        artifact,
        decode_tpot_s=0.002,
        decode_energy_per_token_j=0.08,
        decode_energy_tier="analytic_anchored",
        decode_timing_evidence_id="timing-" + "d" * 64,
        system_calibration_id="system-" + "e" * 64,
    )

    assert analysis["ordinary_decode_ranking_effect"] == "none"
    assert analysis["publication_rankable"] is True
    assert analysis["handoff"]["wire_bytes"] > (
        analysis["handoff"]["decode_cache_bytes"]
    )
    assert [row["regime"] for row in analysis["regimes"]] == [
        "fully_pipelined",
        "back_pressure",
        "host_buffered",
    ]
    assert all(row["prefill_decode_ratio"] > 0 for row in analysis["regimes"])


def test_lossless_schedule_filters_only_hard_gates_and_joins_cached_rows(
    tmp_path: Path,
) -> None:
    manifest = _synthetic_manifest()
    reference = next(
        entry
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
    )
    quantized = tuple(
        entry
        for entry in manifest.entries
        if (
            entry.profile.kind == PROFILE_KIND_QUANTIZED
            and entry.legality.hardware_candidate
        )
    )[:3]
    rows = (
        _numerical_row(reference, 1.0),
        _numerical_row(quantized[0], 1.001),
        _numerical_row(quantized[1], 1.005),
        _numerical_row(quantized[2], 1.02),
    )

    class Evaluator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def physical_cost_group_key(self, entry, numerical):
            return {"synthetic_cost": "identical"}

        def preflight_group_key(self, entry, numerical):
            return {"synthetic_cost": "identical"}

        def evaluation_group_key(self, entry, numerical):
            return {"synthetic_cost": "identical"}

        def preflight(self, entry, candidate, numerical):
            if candidate.batch > 4:
                return HardwareEvaluation.failed(
                    "runtime_capacity_exceeded",
                    "synthetic physical capacity exceeded",
                )
            return None

        def __call__(self, entry, candidate, numerical):
            self.calls.append((entry.profile_id, candidate.batch))
            return HardwareEvaluation.failed(
                "synthetic_unrankable",
                "cost invocation recorded",
            )

    evaluator = Evaluator()
    space = ExactHardwareSpace(
        mlen=(128,),
        blen=(2,),
        hlen=(16,),
        batch=(1, 4, 8),
        hbm_channels=(8,),
        chip_count=(1,),
        sram_policy=("streaming",),
        kv_head_reuse=(False,),
        drain_overlapped=(False,),
        attention_heads=8,
        kv_heads=2,
    )
    study = ExactHardwareStudy(
        manifest=manifest,
        numerical_results=rows,
        space=space,
        hidden_size=128,
        evaluator=evaluator,
        evaluator_version="synthetic-evaluator",
        require_complete=False,
        relative_perplexity_limit=1.01,
    )
    assert study.capability_target is not None
    assert study.capability_target.kv_heads == 2
    assert study.capability_target.head_dim == 16
    assert study.capability_target.hlen == 16
    results = tuple(study.iter_results())
    schedule = study.provenance.search_schedule
    assert schedule["profile_counts"] == {
        "hardware_relevant_available": 3,
        "accuracy_passing": 2,
        "accuracy_rejected": 1,
        "physical_cost_signatures": 1,
        "preflight_equivalence_groups": 1,
    }
    assert schedule["cross_product_counts"] == {
        "raw_hardware_relevant": 9,
        "after_accuracy_constraint": 9,
        "physical_signature_pairs": 3,
        "preflight_passing_equivalence_pairs": 2,
        "simulator_priced_pairs": 2,
        "joined_result_rows": 6,
    }
    assert schedule["preflight_rejections_by_code"] == {
        "runtime_capacity_exceeded": 1,
    }
    # The over-budget profile is priced and labelled, never removed.
    assert len(results) == study.expected_result_count == 6
    assert {result.profile_id for result in results} == {
        quantized[0].profile_id,
        quantized[1].profile_id,
        quantized[2].profile_id,
    }
    assert {result.candidate.batch for result in results} == {1, 4}
    assert len(evaluator.calls) == 2

    # Every hard-feasible candidate is still priced, so an arbitrary TPOT or
    # energy objective has the same optimum as an exhaustive feasible scan.
    priced_batches = {batch for _, batch in evaluator.calls}
    exhaustive_feasible = {
        candidate.batch
        for candidate in space.candidates(128)
        if candidate.batch <= 4
    }
    assert priced_batches == exhaustive_feasible
    synthetic_tpot = {1: 4.0, 4: 1.0}
    assert min(priced_batches, key=synthetic_tpot.get) == min(
        exhaustive_feasible,
        key=synthetic_tpot.get,
    )
    assert schedule["accuracy_constraint"]["maximum_mean_nll"] == pytest.approx(
        1.0 + math.log(1.01)
    )

    legacy_evaluator = Evaluator()
    legacy_study = ExactHardwareStudy(
        manifest=manifest,
        numerical_results=rows,
        space=space,
        hidden_size=128,
        evaluator=legacy_evaluator,
        evaluator_version="synthetic-evaluator",
        require_complete=False,
        relative_perplexity_limit=1.01,
    )
    legacy_rows = []
    for entries, candidate_mask, _ in legacy_study._schedule:
        for candidate_index, candidate in enumerate(
            space.iter_candidates(128)
        ):
            if not candidate_mask[candidate_index]:
                continue
            evaluation_groups = {}
            for entry in entries:
                key = legacy_study._evaluation_group_keys[entry.profile_id]
                evaluation_groups.setdefault(key, []).append(entry)
            for grouped_entries in evaluation_groups.values():
                representative = grouped_entries[0]
                outcome = legacy_evaluator(
                    representative,
                    candidate,
                    legacy_study._rows[representative.profile_id],
                )
                legacy_rows.extend(
                    legacy_study._joined_result(
                        entry,
                        candidate,
                        outcome,
                    ).to_dict()
                    for entry in grouped_entries
                )
    factorized_bytes = json.dumps(
        [result.to_dict() for result in results],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_bytes = json.dumps(
        legacy_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert factorized_bytes == legacy_bytes
    assert len(legacy_evaluator.calls) == 2

    study.iter_results = lambda: (_ for _ in ()).throw(
        AssertionError("production artifact writing cannot expand conceptual rows")
    )
    artifact = study.write(tmp_path / "factorized_hardware.jsonl")
    factorized_header, _ = load_hardware_artifact(artifact.path)
    assert factorized_header["factor_evaluation_count"] == 2
    assert factorized_header["conceptual_result_count"] == 6
    assert [
        membership["member_count"]
        for membership in factorized_header["factor_memberships"]
    ] == [3]


def _compact_publication_profiles() -> tuple[DecodePrecisionProfile, ...]:
    quantized = (
        DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT8", "MXINT8", "FP_E3M2"
        ),
        DecodePrecisionProfile.quantized(
            "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
        ),
        DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT4", "MXINT2", "FP_E3M2"
        ),
        DecodePrecisionProfile.quantized(
            "MXINT4", "MXINT8", "MXINT8", "FP_E2M3"
        ),
    )
    controls = tuple(
        DecodePrecisionProfile.vector_bf16_control(
            profile.weight_format,
            profile.activation_format,
            profile.kv_format,
        )
        for profile in quantized
    )
    return quantized + controls


def test_factor_reduction_cost_is_independent_of_alias_membership() -> None:
    manifest = _synthetic_manifest()
    aliases = tuple(
        entry
        for entry in manifest.entries
        if (
            entry.profile.kind == PROFILE_KIND_QUANTIZED
            and entry.profile.weight_format == "MXINT8"
            and entry.profile.activation_format == "MXINT8"
            and entry.profile.key_format == "MXINT8"
            and entry.profile.value_format == "MXINT8"
        )
    )
    assert len(aliases) == 6
    numerical = {
        entry.profile_id: _numerical_row(entry, 1.0)
        for entry in aliases
    }
    join_class_ids = {
        _factor_join_class_id(entry, numerical[entry.profile_id])
        for entry in aliases
    }
    assert len(join_class_ids) == 1
    join_class_id = next(iter(join_class_ids))

    provenance_body = {"fixture": "alias-factor-reduction"}
    provenance_hash = _content_hash(provenance_body)
    run_id = f"hwdse-{provenance_hash}"
    group = _HardwareFactorGroup(
        factor_id=f"hardware-factor-{'1' * 64}",
        physical_signature_id="physical-alias-group",
        preflight_group_id="preflight-alias-group",
        evaluation_group_id="evaluation-alias-group",
        schedule_ordinal=0,
        evaluation_group_ordinal=0,
        entries=aliases,
        join_classes=((join_class_id, aliases),),
        candidate_mask_sha256=hashlib.sha256(b"\x01\x01").hexdigest(),
        passing_candidate_count=2,
    )
    factors = []
    joined_rows = {}
    for candidate_ordinal, (suffix, tpot_ms, energy_j) in enumerate(
        (("fast", 1.0, 2.0), ("efficient", 2.0, 1.0))
    ):
        candidate_id = f"candidate-{suffix}"
        hardware = {
            "CHIP_COUNT": 1,
            "TP": 1,
            "KVP": 1,
            "fixture_variant": suffix,
        }
        candidate = SimpleNamespace(
            candidate_id=candidate_id,
            to_dict=lambda hardware=hardware: hardware,
        )
        common = _compact_hardware_row(
            run_id=run_id,
            profile=aliases[0].profile,
            profile_ordinal=aliases[0].ordinal,
            candidate_suffix=suffix,
            mean_nll=1.0,
            tpot_ms=tpot_ms,
            energy_j=energy_j,
        )
        common_metrics = common["metrics"]
        outcome = SimpleNamespace(
            metrics=SimpleNamespace(
                to_dict=lambda metrics=common_metrics: metrics
            ),
            validity=StackValidity(True, True, True, True, True),
            error_code=None,
            error_message=None,
        )
        factors.append(
            _HardwareFactorEvaluation(
                ordinal=candidate_ordinal,
                candidate_ordinal=candidate_ordinal,
                group=group,
                candidate=candidate,
                outcome=outcome,
            )
        )
        for entry in aliases:
            row = _compact_hardware_row(
                run_id=run_id,
                profile=entry.profile,
                profile_ordinal=entry.ordinal,
                candidate_suffix=suffix,
                mean_nll=1.0,
                tpot_ms=tpot_ms,
                energy_j=energy_j,
            )
            row.pop("record_hash")
            row["candidate_id"] = candidate_id
            row["hardware"] = hardware
            row["metrics"] = common_metrics
            joined_rows[(entry.profile_id, candidate_id)] = {
                **row,
                "record_hash": _content_hash(row),
            }

    membership = {
        "factor_id": group.factor_id,
        "physical_signature_id": group.physical_signature_id,
        "preflight_group_id": group.preflight_group_id,
        "evaluation_group_id": group.evaluation_group_id,
        "schedule_ordinal": 0,
        "evaluation_group_ordinal": 0,
        "candidate_mask_sha256": group.candidate_mask_sha256,
        "passing_candidate_count": 2,
        "member_count": len(aliases),
        "conceptual_result_count": 2 * len(aliases),
        "members": [
            {
                "member_ordinal": member_ordinal,
                "join_class_id": join_class_id,
                "profile_ordinal": entry.ordinal,
                "profile_id": entry.profile_id,
                "profile": entry.profile.to_dict(),
                "legality": entry.legality.to_dict(),
                "numerical_result_hash": _content_hash(
                    numerical[entry.profile_id]
                ),
                "numerical_summary": {
                    "state": "succeeded",
                    "attempt": 1,
                    "record_hash": None,
                    "result_path": None,
                    "scalar_metrics": {"mean_nll": 1.0},
                    "document_metrics_hash": None,
                },
            }
            for member_ordinal, entry in enumerate(aliases)
        ],
    }
    join_calls = []

    def joined_result(entry, candidate, outcome):
        join_calls.append((entry.profile_id, candidate.candidate_id))
        row = joined_rows[(entry.profile_id, candidate.candidate_id)]
        return SimpleNamespace(to_dict=lambda row=row: row)

    study = SimpleNamespace(
        provenance=SimpleNamespace(run_id=run_id),
        scatter_sample_limit=0,
        expected_factor_evaluation_count=2,
        _factor_memberships=(group,),
        factor_membership_records=lambda: (membership,),
        _joined_result=joined_result,
    )
    reducer = _FactorizedHardwareReducer(study)
    for factor in factors:
        reducer.consume(factor)
    assert len(join_calls) == 2
    summary, stored_factors, stored_bindings = reducer.finish()
    assert len(stored_factors) == 2
    assert len(stored_bindings) == 2 * len(aliases)
    assert len(join_calls) == 2 + 2 * len(aliases)
    assert all(
        aggregate["local_frontier_count"] == 2
        for aggregate in summary["profile_aggregates"]
    )


def _compact_hardware_row(
    *,
    run_id: str,
    profile: DecodePrecisionProfile,
    profile_ordinal: int,
    candidate_suffix: str,
    mean_nll: float,
    tpot_ms: float,
    energy_j: float,
) -> dict[str, object]:
    candidate_id = f"candidate-{profile_ordinal:02d}-{candidate_suffix}"
    area_mm2 = 300.0 + profile_ordinal
    body = {
        "schema_version": "decode-hardware-result",
        "run_id": run_id,
        "profile_ordinal": profile_ordinal,
        "profile_id": profile.profile_id,
        "profile": profile.to_dict(),
        "legality": {},
        "numerical_result_hash": _content_hash(
            {"profile_id": profile.profile_id, "mean_nll": mean_nll}
        ),
        "numerical_summary": {
            "state": "succeeded",
            "attempt": 1,
            "record_hash": "numerical-record",
            "result_path": "fixture.json",
            "scalar_metrics": {"mean_nll": mean_nll},
            "document_metrics_hash": None,
        },
        "candidate_id": candidate_id,
        "hardware": {
            "CHIP_COUNT": 1,
            "TP": 1,
            "KVP": 1,
        },
        "capability": {},
        "packedkv_selector_valid": True,
        "packedkv_selector_evidence": {},
        "validity": StackValidity(True, True, True, True, True).to_dict(),
        "software_valid": True,
        "compiler_valid": True,
        "emulator_valid": True,
        "rtl_valid": True,
        "dc_calibrated": True,
        "deployment_valid": True,
        "metrics": {
            "whole_model": {
                "rankable": True,
                "strict_system_resource_boundary_valid": True,
                "publication_timing_tier": "stage_calibrated_analytic",
                "tpot_ms": tpot_ms,
                "tps": 1000.0 / tpot_ms,
                "system_calibration_id": "system-calibration",
                "calibrated_energy": {
                    "total_j": energy_j,
                    "energy_tier": "dc_calibrated",
                    "energy_id": "power-calibration",
                },
            },
            "output_head_boundary": _serialized_measured_head_boundary(),
            "generated_tokens_per_step": 1,
            "capacity": {"feasible": True},
            "runtime_capacity_evidence": {"max_runtime_batch": 256},
            "timing_calibrated": True,
            "runtime_feasible": True,
            "area_mm2": area_mm2,
            "system_area_mm2": area_mm2,
            "resource_budget": {
                "aggregate_area_mm2": area_mm2,
                "aggregate_area_limit_mm2": 1000.0,
                "feasible": True,
            },
        },
        "error_code": None,
        "error_message": None,
    }
    return {**body, "record_hash": _content_hash(body)}


def _write_compact_fixture(path: Path, *, sample_limit: int = 2):
    provenance_body = {"fixture": "compact-hardware-artifact"}
    provenance_hash = _content_hash(provenance_body)
    run_id = f"hwdse-{provenance_hash}"
    profiles = _compact_publication_profiles()
    failed_profile = DecodePrecisionProfile.quantized(
        "MXINT8", "MXINT8", "MXINT4", "FP_E3M2"
    )
    rows = []
    mean_nll_by_profile = {}
    for profile_ordinal, profile in enumerate(profiles):
        mean_nll = 1.001 + profile_ordinal * 0.0005
        mean_nll_by_profile[profile.profile_id] = mean_nll
        best_tpot = 1.0 + profile_ordinal * 0.1
        best_energy = 0.2 + profile_ordinal * 0.02
        if profile_ordinal == 3:
            rows.extend(
                (
                    _compact_hardware_row(
                        run_id=run_id,
                        profile=profile,
                        profile_ordinal=profile_ordinal,
                        candidate_suffix="lowest-energy",
                        mean_nll=mean_nll,
                        tpot_ms=best_tpot + 5.0,
                        energy_j=best_energy,
                    ),
                    _compact_hardware_row(
                        run_id=run_id,
                        profile=profile,
                        profile_ordinal=profile_ordinal,
                        candidate_suffix="fastest",
                        mean_nll=mean_nll,
                        tpot_ms=best_tpot,
                        energy_j=best_energy + 1.0,
                    ),
                )
            )
        else:
            rows.extend((
                _compact_hardware_row(
                    run_id=run_id,
                    profile=profile,
                    profile_ordinal=profile_ordinal,
                    candidate_suffix="dominated",
                    mean_nll=mean_nll,
                    tpot_ms=best_tpot + 5.0,
                    energy_j=best_energy + 1.0,
                ),
                _compact_hardware_row(
                    run_id=run_id,
                    profile=profile,
                    profile_ordinal=profile_ordinal,
                    candidate_suffix="winner",
                    mean_nll=mean_nll,
                    tpot_ms=best_tpot,
                    energy_j=best_energy,
                ),
            ))
    failed_ordinal = len(profiles)
    failed_mean_nll = 1.02
    mean_nll_by_profile[failed_profile.profile_id] = failed_mean_nll
    for suffix in ("compiler", "timing"):
        failed = _compact_hardware_row(
            run_id=run_id,
            profile=failed_profile,
            profile_ordinal=failed_ordinal,
            candidate_suffix=suffix,
            mean_nll=failed_mean_nll,
            tpot_ms=10.0,
            energy_j=2.0,
        )
        failed.pop("record_hash")
        failed.update(
            {
                "validity": StackValidity(
                    True,
                    False,
                    False,
                    False,
                    False,
                ).to_dict(),
                "compiler_valid": False,
                "emulator_valid": False,
                "rtl_valid": False,
                "dc_calibrated": False,
                "deployment_valid": False,
                "metrics": None,
                "error_code": f"{suffix}_failed",
                "error_message": f"synthetic {suffix} failure",
            }
        )
        rows.append({**failed, "record_hash": _content_hash(failed)})
    provenance = SimpleNamespace(
        run_id=run_id,
        canonical_hash=provenance_hash,
        to_dict=lambda: provenance_body,
    )
    groups = []
    factors = []
    membership_records = []
    factor_ordinal = 0
    rows_by_identity = {
        (str(row["profile_id"]), str(row["candidate_id"])): row
        for row in rows
    }
    for profile_ordinal, profile in enumerate(profiles + (failed_profile,)):
        profile_rows = tuple(
            row for row in rows if row["profile_id"] == profile.profile_id
        )
        entry = SimpleNamespace(
            ordinal=profile_ordinal,
            profile_id=profile.profile_id,
        )
        factor_id = f"hardware-factor-{profile_ordinal:064x}"
        join_class_id = f"hardware-join-class-{profile_ordinal:064x}"
        group = _HardwareFactorGroup(
            factor_id=factor_id,
            physical_signature_id=f"physical-{profile_ordinal}",
            preflight_group_id=f"preflight-{profile_ordinal}",
            evaluation_group_id=f"evaluation-{profile_ordinal}",
            schedule_ordinal=profile_ordinal,
            evaluation_group_ordinal=0,
            entries=(entry,),
            join_classes=((join_class_id, (entry,)),),
            candidate_mask_sha256=hashlib.sha256(
                bytes([1] * len(profile_rows))
            ).hexdigest(),
            passing_candidate_count=len(profile_rows),
        )
        groups.append(group)
        first = profile_rows[0]
        membership_records.append(
            {
                "factor_id": factor_id,
                "physical_signature_id": group.physical_signature_id,
                "preflight_group_id": group.preflight_group_id,
                "evaluation_group_id": group.evaluation_group_id,
                "schedule_ordinal": profile_ordinal,
                "evaluation_group_ordinal": 0,
                "candidate_mask_sha256": group.candidate_mask_sha256,
                "passing_candidate_count": len(profile_rows),
                "member_count": 1,
                "conceptual_result_count": len(profile_rows),
                "members": [
                    {
                        "member_ordinal": 0,
                        "join_class_id": join_class_id,
                        "profile_ordinal": profile_ordinal,
                        "profile_id": profile.profile_id,
                        "profile": profile.to_dict(),
                        "legality": first["legality"],
                        "numerical_result_hash": first[
                            "numerical_result_hash"
                        ],
                        "numerical_summary": first["numerical_summary"],
                    }
                ],
            }
        )
        for candidate_ordinal, row in enumerate(profile_rows):
            candidate = SimpleNamespace(
                candidate_id=row["candidate_id"],
                to_dict=lambda row=row: row["hardware"],
            )
            metrics = (
                SimpleNamespace(to_dict=lambda row=row: row["metrics"])
                if row["metrics"] is not None
                else None
            )
            outcome = SimpleNamespace(
                metrics=metrics,
                validity=StackValidity.from_dict(row["validity"]),
                error_code=row["error_code"],
                error_message=row["error_message"],
            )
            factors.append(
                _HardwareFactorEvaluation(
                    ordinal=factor_ordinal,
                    candidate_ordinal=candidate_ordinal,
                    group=group,
                    candidate=candidate,
                    outcome=outcome,
                )
            )
            factor_ordinal += 1

    def joined_result(entry, candidate, outcome):
        row = rows_by_identity[(entry.profile_id, candidate.candidate_id)]
        return SimpleNamespace(to_dict=lambda row=row: row)

    study = SimpleNamespace(
        provenance=provenance,
        expected_result_count=len(rows),
        expected_factor_evaluation_count=len(factors),
        scatter_sample_limit=sample_limit,
        _factor_memberships=tuple(groups),
        factor_membership_records=lambda: tuple(membership_records),
        iter_factor_evaluations=lambda: iter(factors),
        _joined_result=joined_result,
    )
    artifact = ExactHardwareStudy.write(study, path)
    return artifact, profiles + (failed_profile,), rows, mean_nll_by_profile


def test_compact_hardware_artifact_preserves_promotion_and_streams(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact, profiles, full_rows, mean_nll = _write_compact_fixture(
        tmp_path / "hardware.jsonl",
        sample_limit=2,
    )
    header, compact_rows = load_hardware_artifact(artifact.path)
    assert header["storage_revision"] == "factorized-exact"
    assert header["expected_result_count"] == len(full_rows) == 18
    assert header["conceptual_result_count"] == 18
    assert header["factor_evaluation_count"] == 18
    assert artifact.result_count == 18
    assert artifact.stored_result_count == len(compact_rows) < 18
    assert header["retention"]["sampled_dominated_count"] <= 2
    assert header["retention"]["sample_limit"] == 2
    failed_aggregate = header["profile_aggregates"][-1]
    assert failed_aggregate["valid_count"] == 0
    assert failed_aggregate["error_count"] == 2
    assert failed_aggregate["local_frontier"] == []
    assert all(
        aggregate["valid_count"] == 2
        for aggregate in header["profile_aggregates"][:-1]
    )

    profile_by_id = {profile.profile_id: profile for profile in profiles}
    full_points = tuple(
        _hardware_point(
            row,
            profile=profile_by_id[str(row["profile_id"])],
            mean_nll=mean_nll[str(row["profile_id"])],
        )
        for row in full_rows
    )
    compact_points = tuple(
        _hardware_point(
            row,
            profile=profile_by_id[str(row["profile_id"])],
            mean_nll=mean_nll[str(row["profile_id"])],
        )
        for row in compact_rows
        if "profile_frontier" in row["retention_labels"]
    )
    assert all(point is not None for point in compact_points)
    full_selection = select_refinement_sources(
        (point for point in full_points if point is not None),
        reference_mean_nll=1.0,
    )
    compact_selection = select_refinement_sources(
        (point for point in compact_points if point is not None),
        reference_mean_nll=1.0,
    )
    assert compact_selection.to_dict() == full_selection.to_dict()
    deployment_profile_id = profiles[3].profile_id
    deployment_alternatives = tuple(
        point
        for point in compact_selection.promotion.hardware_alternatives
        if point.profile_id == deployment_profile_id
    )
    assert len(deployment_alternatives) == 2
    assert {point.candidate_id for point in deployment_alternatives} == {
        "candidate-03-lowest-energy",
        "candidate-03-fastest",
    }

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "read_text",
            lambda self, *args, **kwargs: (
                (_ for _ in ()).throw(AssertionError("data read_text is forbidden"))
                if self.resolve() == artifact.path.resolve()
                else original_read_text(self, *args, **kwargs)
            ),
        )
        patch.setattr(
            Path,
            "read_bytes",
            lambda self, *args, **kwargs: (
                (_ for _ in ()).throw(AssertionError("data read_bytes is forbidden"))
                if self.resolve() == artifact.path.resolve()
                else original_read_bytes(self, *args, **kwargs)
            ),
        )
        streamed_header, streamed_rows = load_hardware_artifact(artifact.path)
    assert streamed_header["factor_evaluation_sha256"] == header[
        "factor_evaluation_sha256"
    ]
    assert streamed_rows == compact_rows


def test_compact_hardware_artifact_detects_digest_and_count_tampering(
    tmp_path: Path,
) -> None:
    artifact, _, _, _ = _write_compact_fixture(tmp_path / "hardware.jsonl")
    original_data = artifact.path.read_bytes()
    original_metadata = artifact.metadata_path.read_bytes()
    lines = artifact.path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["factor_evaluation_sha256"] = "0" * 64
    lines[0] = json.dumps(header, sort_keys=True, separators=(",", ":"))
    artifact.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content hash|enumeration digest"):
        load_hardware_artifact(artifact.path)

    artifact.path.write_bytes(original_data)
    artifact.metadata_path.write_bytes(original_metadata)
    lines = artifact.path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["conceptual_result_count"] += 1
    lines[0] = json.dumps(header, sort_keys=True, separators=(",", ":"))
    artifact.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    metadata["content_sha256"] = hashlib.sha256(
        artifact.path.read_bytes()
    ).hexdigest()
    artifact.metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cardinality|expansion contract"):
        load_hardware_artifact(artifact.path)


def test_gqa_topology_roles_are_model_general() -> None:
    topology = AttentionTopology(
        role="gqa8",
        query_heads=32,
        kv_heads=4,
        head_dim=64,
        mlen=512,
    )
    assert topology.query_heads // topology.kv_heads == 8


def test_lossless_schedule_rejects_unsafe_equivalence_nesting() -> None:
    manifest = _synthetic_manifest()
    quantized = tuple(
        entry
        for entry in manifest.entries
        if (
            entry.profile.kind == PROFILE_KIND_QUANTIZED
            and entry.legality.hardware_candidate
        )
    )[:2]
    rows = tuple(_numerical_row(entry, 1.0) for entry in quantized)

    class UnsafeEvaluator:
        @staticmethod
        def physical_cost_group_key(entry, numerical):
            return entry.profile_id

        @staticmethod
        def preflight_group_key(entry, numerical):
            return "incorrectly_merged"

        @staticmethod
        def evaluation_group_key(entry, numerical):
            return entry.profile_id

        def __call__(self, entry, candidate, numerical):  # pragma: no cover
            raise AssertionError("unsafe schedule must fail before evaluation")

    space = ExactHardwareSpace(
        mlen=(128,),
        blen=(2,),
        hlen=(16,),
        batch=(1,),
        hbm_channels=(8,),
        chip_count=(1,),
        sram_policy=("streaming",),
        kv_head_reuse=(False,),
        drain_overlapped=(False,),
        attention_heads=8,
        kv_heads=2,
    )
    with pytest.raises(
        ValueError,
        match="preflight equivalence merges distinct physical-cost signatures",
    ):
        ExactHardwareStudy(
            manifest=manifest,
            numerical_results=rows,
            space=space,
            hidden_size=128,
            evaluator=UnsafeEvaluator(),
            evaluator_version="unsafe",
            require_complete=False,
        )


def test_analytic_power_bridge_reports_every_structural_energy_term() -> None:
    candidate = HardwareCandidate(
        mlen=128,
        blen=4,
        vlen=128,
        hlen=128,
        batch=4,
        hbm_channels=8,
        hbm_generation="HBM2",
        chip_count=2,
        tp=2,
        kvp=1,
        link_ports=1,
    )
    traffic = {
        "weight_element_read_bytes": 1_000.0,
        "weight_scale_read_bytes": 100.0,
        "bf16_weight_read_bytes": 200.0,
        "activation_read_bytes": 0.0,
        "activation_write_bytes": 0.0,
        "kv_element_read_bytes": 500.0,
        "kv_scale_read_bytes": 50.0,
        "kv_element_write_bytes": 20.0,
        "kv_scale_write_bytes": 2.0,
    }
    observation = SimpleNamespace(
        hbm_traffic_per_generated_token=tuple(traffic.items()),
        tps=1_000.0,
        vector_sram_required_bytes=64_000,
        avg_realized_compute_seconds=0.001,
        tpot_ms=4.0,
        capacity=SimpleNamespace(available_bytes=32_000_000_000),
        generated_tokens_per_step=4,
    )
    energy = analytic_energy_from_simulator(
        candidate=candidate,
        observation=observation,
        mac_bits=4,
        per_chip_logic_area_mm2=10.0,
        collective_bytes_per_generated_token=1_024.0,
    )
    assert energy.energy_tier == "analytic_anchored"
    assert energy.compute_j > 0
    assert energy.sram_j > 0
    assert energy.hbm_j > 0
    assert energy.leakage_j > 0
    assert energy.link_j > 0
    assert energy.tokens_per_joule == pytest.approx(1.0 / energy.total_j)
    assert energy.edp_j_s == pytest.approx(energy.total_j * 0.004)


@dataclass(frozen=True)
class _TimingEvidence:
    evidence_id: str
    mode: str = "rtl_serialized"
    passed: bool = True


@dataclass(frozen=True)
class _HeadEvidence:
    calibration_id: str
    provenance_id: str
    service_mode: str = HEAD_SERVICE_MODE
    passed: bool = True


@dataclass(frozen=True)
class _PowerCalibration:
    calibration_id: str
    passed: bool = True


def test_selection_accepts_analytic_energy_and_prefers_dc() -> None:
    suffix = "a" * 64
    analytic_id = f"analytic-decode-energy-{suffix}"
    dc_id = f"exact-dc-power-{'b' * 64}"
    head_id = f"bf16-head-service-{'c' * 64}"
    head_provenance = f"bf16-head-provenance-{'d' * 64}"
    timing = _TimingEvidence(f"timing-{'e' * 64}")
    head = _HeadEvidence(head_id, head_provenance)
    validity = StackValidity(
        software_valid=True,
        compiler_valid=True,
        emulator_valid=True,
        rtl_valid=True,
    )

    def candidate(role: str, index: int, tier: str, energy: float):
        power_id = dc_id if tier == "dc_calibrated" else analytic_id
        system_id = composite_system_calibration_id(
            power_id,
            head_id,
            head_provenance,
        )
        return PublicationCandidate(
            evaluation_class=role,
            profile_id=f"profile-{index}",
            candidate_id=f"candidate-{index}",
            profile_kind="quantized",
            perplexity=10.0,
            tpot_ms=2.0,
            energy_per_token_j=energy,
            energy_tier=tier,
            validity=(
                StackValidity(
                    software_valid=True,
                    compiler_valid=True,
                    emulator_valid=True,
                    rtl_valid=True,
                    dc_calibrated=True,
                )
                if tier == "dc_calibrated"
                else validity
            ),
            power_calibration_id=power_id,
            cost_scope="whole_model",
            system_calibration_id=system_id,
            head_service_calibration_id=head_id,
            output_head_location=EXTERNAL_BF16_HEAD,
            whole_model_rankable=True,
            timing_calibrated=True,
            timing_evidence_id=timing.evidence_id,
            task_delta_lower_ci=(("gsm8k", 0.0), ("ifeval", 0.0)),
            ruler_scores=tuple(
                (length, 1.0, 1.0)
                for length in (4096, 8192, 16384, 32768)
            ),
        )

    reference = PublicationCandidate(
        evaluation_class="bf16_reference",
        profile_id="reference",
        candidate_id="accuracy-only",
        profile_kind=PROFILE_KIND_BF16_REFERENCE,
        perplexity=10.0,
        tpot_ms=None,
        energy_per_token_j=None,
        validity=StackValidity(software_valid=True),
        hardware_candidate=False,
    )
    values = (
        reference,
        candidate("uniform_i8", 1, "analytic_anchored", 0.5),
        candidate("uniform_i4", 2, "analytic_anchored", 0.4),
        candidate("pareto_candidate", 3, "dc_calibrated", 0.8),
    )
    decision = select_final_deployment(
        values,
        calibration=_PowerCalibration(dc_id),
        timing_evidence=timing,
        head_service_evidence=head,
        output_head_location=EXTERNAL_BF16_HEAD,
    )
    assert decision.selected is values[-1]
    assert decision.energy_tier == "dc_calibrated"

    analytic_values = (
        reference,
        candidate("uniform_i8", 11, "analytic_anchored", 0.5),
        candidate("uniform_i4", 12, "analytic_anchored", 0.4),
        candidate("pareto_candidate", 13, "analytic_anchored", 0.3),
    )
    analytic_decision = select_final_deployment(
        analytic_values,
        calibration=None,
        timing_evidence=timing,
        head_service_evidence=head,
        output_head_location=EXTERNAL_BF16_HEAD,
    )
    assert analytic_decision.selected is analytic_values[-1]
    assert analytic_decision.energy_tier == "analytic_anchored"


@pytest.mark.parametrize("name", evaluation._PROVENANCE_MODULES)
def test_provenance_module_has_a_source_file(name: str) -> None:
    assert getattr(importlib.import_module(name), "__file__", None)


def _synthetic_head_phase_samples() -> tuple[list, list, dict]:
    """Generate phase samples from a known linear service ground truth."""

    import random

    from decode_dse.hardware.measure_bf16_head_service import (
        PhaseSample,
        closed_form_dimensions,
    )

    hidden_size, vocab_size = 64, 256
    batches = (1, 4, 8)
    truth = {
        "request_bw": 4.0e11,
        "request_fixed": 2.0e-6,
        "response_bw": 4.0e11,
        "response_fixed": 1.5e-6,
        "mac_per_s": 1.0e15,
        "memory_bw": 8.0e12,
        "selection_rate": 1.0e-9,
        "head_fixed": 3.0e-6,
        "link_energy": 5.0e-12,
        "mac_energy": 1.0e-12,
        "memory_energy": 5.0e-12,
        "selection_energy": 1.0e-11,
        "leakage_w": 80.0,
    }
    rng = random.Random(7)

    def sample(batch: int, noise: float, hidden_tag: str) -> PhaseSample:
        dims = closed_form_dimensions(
            batch=batch, hidden_size=hidden_size, vocab_size=vocab_size
        )
        jitter = lambda value: value * (1 + rng.uniform(-noise, noise))
        head_compute = truth["head_fixed"] + max(
            dims["bf16_macs"] / truth["mac_per_s"],
            dims["head_memory_bytes"] / truth["memory_bw"],
        )
        digest = lambda tag: hashlib.sha256(tag.encode()).hexdigest()
        return PhaseSample(
            batch=batch,
            request_latency_s=jitter(
                truth["request_fixed"]
                + dims["request_bytes"] / truth["request_bw"]
            ),
            head_compute_latency_s=jitter(head_compute),
            selection_latency_s=jitter(
                dims["selection_elements"] * truth["selection_rate"]
            ),
            response_latency_s=jitter(
                truth["response_fixed"]
                + dims["response_bytes"] / truth["response_bw"]
            ),
            link_energy_j=jitter(
                (dims["request_bytes"] + dims["response_bytes"])
                * truth["link_energy"]
            ),
            head_compute_energy_j=jitter(
                dims["bf16_macs"] * truth["mac_energy"]
                + dims["head_memory_bytes"] * truth["memory_energy"]
            ),
            selection_energy_j=jitter(
                dims["selection_elements"] * truth["selection_energy"]
            ),
            leakage_power_w=jitter(truth["leakage_w"]),
            hidden_sha256=digest(f"hidden-{hidden_tag}"),
            reference_logits_sha256=digest(f"ref-logits-{hidden_tag}"),
            service_logits_sha256=digest(f"svc-logits-{hidden_tag}"),
            reference_token_ids_sha256=digest(f"tokens-{hidden_tag}"),
            service_token_ids_sha256=digest(f"tokens-{hidden_tag}"),
            logit_max_abs_error=0.0,
            logit_mean_abs_error=0.0,
            topk_set_agreement=1.0,
            selected_tokens_equal=True,
        )

    repeats = []
    holdouts = []
    for batch in batches:
        for _ in range(3):
            repeats.append(sample(batch, 0.01, f"repeat-b{batch}"))
        for holdout in range(2):
            holdouts.append(
                (holdout, sample(batch, 0.02, f"holdout-b{batch}-h{holdout}"))
            )
    return repeats, holdouts, {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "batches": batches,
        "mac_per_s": truth["mac_per_s"],
    }


def _assembled_synthetic_head_artifact(tmp_path: Path) -> tuple[Path, dict]:
    from decode_dse.hardware.measure_bf16_head_service import (
        assemble_artifact,
        fit_service_coefficients,
        measurement_record,
    )

    repeats, holdouts, scope = _synthetic_head_phase_samples()
    protocol, service = fit_service_coefficients(
        repeats,
        hidden_size=scope["hidden_size"],
        vocab_size=scope["vocab_size"],
        measured_bf16_mac_per_s=scope["mac_per_s"],
    )
    service["head_weight_sha256"] = "a" * 64
    measurements = []
    counters: dict[int, int] = {}
    for sample in repeats:
        index = counters.get(sample.batch, 0)
        counters[sample.batch] = index + 1
        measurements.append(
            measurement_record(
                sample,
                measurement_id=f"repeat-b{sample.batch}-r{index}",
                split="repeat",
                repeat=index,
                hidden_size=scope["hidden_size"],
                vocab_size=scope["vocab_size"],
                protocol=protocol,
                service=service,
            )
        )
    for holdout, sample in holdouts:
        measurements.append(
            measurement_record(
                sample,
                measurement_id=f"holdout-b{sample.batch}-h{holdout}",
                split="holdout",
                repeat=holdout,
                hidden_size=scope["hidden_size"],
                vocab_size=scope["vocab_size"],
                protocol=protocol,
                service=service,
            )
        )
    model = {
        "model_name": "synthetic/decode-head",
        "model_revision": "0" * 40,
        "hidden_size": scope["hidden_size"],
        "vocab_size": scope["vocab_size"],
        "tie_embeddings": False,
    }
    provenance = {
        "repository": "PLENA_Software",
        "revision": "b" * 40,
        "source_tree_sha256": "c" * 64,
        "command": ["measure_bf16_head_service", "--config", "synthetic"],
        "toolchain": {"torch": "2.7.0", "cuda": "12.8"},
        "environment_sha256": "d" * 64,
        "link_id": "driver->endpoint",
        "head_service_id": "Synthetic Accelerator:GPU-SYNTHETIC",
        "process_corner": "measured_silicon",
        "measured_at_utc": "2026-08-03T12:00:00Z",
        "measurement_resolution": {
            "meter_methods": {
                "driver": "nvml_total_energy_counter",
                "endpoint": "nvml_total_energy_counter",
            },
            "power_plausibility_ceiling_w": 2400.0,
            "min_counter_delta_j": 0.1,
            "idle_power_w": {
                "driver": 100.0,
                "endpoint": service["leakage_power_w"],
                "total": 100.0 + service["leakage_power_w"],
            },
            "service_leakage_power_w": service["leakage_power_w"],
            "service_leakage_scope": "endpoint_only",
            "measurement_driver_idle_role": (
                "instrumentation_only_not_deployed"
            ),
            "measurement_driver_leakage_included": False,
            "phase_windows": {},
        },
    }
    document = assemble_artifact(
        model=model,
        protocol=protocol,
        service=service,
        required_batches=scope["batches"],
        measurements=measurements,
        provenance=provenance,
    )
    path = tmp_path / "bf16_output_head_service.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path, model


def test_head_service_fit_and_assembly_pass_the_exact_loader(
    tmp_path: Path,
) -> None:
    """Synthetic linear-service phases must seal into a passing artifact."""

    path, model = _assembled_synthetic_head_artifact(tmp_path)
    status = load_bf16_head_service_artifact(
        path,
        model_name=model["model_name"],
        model_revision=model["model_revision"],
        hidden_size=model["hidden_size"],
        vocab_size=model["vocab_size"],
        tie_embeddings=model["tie_embeddings"],
        required_batches=(1, 4, 8),
    )
    assert status.passed, status.failures
    assert status.calibration_id.startswith("bf16-head-service-")
    assert status.provenance_id.startswith("bf16-head-provenance-")
    estimate = status.calibration.estimate(4)
    assert estimate.queue_delay_s == 0.0
    assert estimate.dynamic_energy_j > 0


def test_head_service_loader_rejects_a_corrupted_measurement(
    tmp_path: Path,
) -> None:
    path, model = _assembled_synthetic_head_artifact(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["measurements"][0]["bf16_macs"] += 1
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(document) + "\n", encoding="utf-8")
    status = load_bf16_head_service_artifact(
        corrupted,
        model_name=model["model_name"],
        model_revision=model["model_revision"],
        hidden_size=model["hidden_size"],
        vocab_size=model["vocab_size"],
        tie_embeddings=model["tie_embeddings"],
        required_batches=(1, 4, 8),
    )
    assert not status.passed


class _FakeNVMLProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class _FakeNVML:
    """Deterministic NVML stand-in for the head-service meter tests."""

    def __init__(
        self,
        *,
        uuid: str = "GPU-aaaa",
        counter_deltas_mj: list[int] | None = None,
        counter_available: bool = True,
        power_mw: int = 400_000,
        power_available: bool = True,
        compute_pids: list[int] | None = None,
        enforced_limit_mw: int = 1_000_000,
    ) -> None:
        self.uuid = uuid
        self.counter_available = counter_available
        self.power_mw = power_mw
        self.power_available = power_available
        self.compute_pids = list(compute_pids or [])
        self.enforced_limit_mw = enforced_limit_mw
        self._counter_mj = 1_000_000
        self._pending_deltas = list(counter_deltas_mj or [])

    def nvmlDeviceGetHandleByUUID(self, uuid: str):
        if uuid.lower() != self.uuid.lower():
            raise RuntimeError("uuid not found")
        return "handle"

    def nvmlDeviceGetUUID(self, handle):
        return self.uuid

    def nvmlDeviceGetTotalEnergyConsumption(self, handle) -> int:
        if not self.counter_available:
            raise RuntimeError("counter unsupported")
        if self._pending_deltas:
            self._counter_mj += self._pending_deltas.pop(0)
        return self._counter_mj

    def nvmlDeviceGetPowerUsage(self, handle) -> int:
        if not self.power_available:
            raise RuntimeError("power unsupported")
        return self.power_mw

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [_FakeNVMLProcess(pid) for pid in self.compute_pids]

    def nvmlDeviceGetEnforcedPowerLimit(self, handle) -> int:
        return self.enforced_limit_mw


def test_head_meter_uuid_binding_rejects_mismatched_handle() -> None:
    from decode_dse.hardware import measure_bf16_head_service as measure

    nvml = _FakeNVML(uuid="GPU-aaaa")
    handle = measure._bind_nvml_handle_by_uuid(nvml, "aaaa")
    assert handle == "handle"
    nvml.nvmlDeviceGetUUID = lambda handle: "GPU-bbbb"
    with pytest.raises(RuntimeError, match="different GPU UUID"):
        measure._bind_nvml_handle_by_uuid(nvml, "aaaa")


def test_head_meter_exclusivity_gate_rejects_foreign_processes() -> None:
    import os

    from decode_dse.hardware import measure_bf16_head_service as measure

    own = _FakeNVML(compute_pids=[os.getpid()])
    measure._require_exclusive_compute(own, "handle", "driver cuda:0")
    foreign = _FakeNVML(compute_pids=[os.getpid(), 4242])
    with pytest.raises(SystemExit, match="foreign compute processes"):
        measure._require_exclusive_compute(foreign, "handle", "driver cuda:0")


def test_head_phase_energy_rejects_corrupted_and_implausible() -> None:
    from decode_dse.hardware import measure_bf16_head_service as measure

    measure._check_phase_energy(
        delta_j=1.0, wall_s=0.01, power_ceiling_w=2400.0, label="head"
    )
    # zero is a quantization outcome the adaptive loop escalates, not an error
    measure._check_phase_energy(
        delta_j=0.0, wall_s=0.01, power_ceiling_w=2400.0, label="head"
    )
    with pytest.raises(RuntimeError, match="negative"):
        measure._check_phase_energy(
            delta_j=-0.001, wall_s=0.01, power_ceiling_w=2400.0, label="head"
        )
    with pytest.raises(RuntimeError, match="plausibility ceiling"):
        measure._check_phase_energy(
            delta_j=9.243, wall_s=0.003265, power_ceiling_w=2400.0, label="head"
        )


def test_head_phase_resolution_floor_drives_adaptive_iterations() -> None:
    from decode_dse.hardware import measure_bf16_head_service as measure

    assert not measure._phase_delta_sufficient(0.0009)
    assert not measure._phase_delta_sufficient(0.05)
    assert measure._phase_delta_sufficient(0.1)
    assert measure._phase_delta_sufficient(1.0)


def test_head_meter_probes_counter_then_falls_back_to_power_trace() -> None:
    from decode_dse.hardware import measure_bf16_head_service as measure

    # deltas consumed by: constructor probe, begin() alignment reads, end()
    counter = _FakeNVML(counter_deltas_mj=[0, 0, 100, 500])
    meter = measure._BoardEnergyMeter(counter, "handle", "driver")
    assert meter.method == measure.NVML_TOTAL_ENERGY_METHOD
    meter.begin()
    delta, window = meter.end()
    assert delta == pytest.approx(0.5)
    assert window > 0

    fallback = _FakeNVML(counter_available=False, power_mw=200_000)
    meter = measure._BoardEnergyMeter(fallback, "handle", "driver")
    assert meter.method == measure.NVML_POWER_TRACE_METHOD
    meter.begin()
    time.sleep(0.05)
    energy, window = meter.end()
    assert energy > 0
    assert energy == pytest.approx(200.0 * 0.05, rel=0.5)
    assert window == pytest.approx(0.05, rel=0.5)

    dead = _FakeNVML(counter_available=False, power_available=False)
    with pytest.raises(RuntimeError, match="no usable NVML board-energy meter"):
        measure._BoardEnergyMeter(dead, "handle", "driver")


def test_head_dynamic_energy_resolution_policy() -> None:
    from decode_dse.hardware import measure_bf16_head_service as measure

    # positive measurements pass through untouched
    value, below = measure._resolved_dynamic_energy(
        dynamic_total_j=1.5, idle_energy_j=280.0, label="head"
    )
    assert value == 1.5 and below is False

    # a residual inside the idle-drift band becomes the declared bound
    value, below = measure._resolved_dynamic_energy(
        dynamic_total_j=-6.2, idle_energy_j=280.0, label="response"
    )
    assert value == pytest.approx(0.05 * 280.0)
    assert below is True

    # beyond the band is corruption
    with pytest.raises(RuntimeError, match="noise band"):
        measure._resolved_dynamic_energy(
            dynamic_total_j=-20.0, idle_energy_j=280.0, label="response"
        )


def test_parallel_block_pricing_reproduces_the_serial_artifact(
    tmp_path: Path,
) -> None:
    manifest = _synthetic_manifest()
    reference = next(
        entry
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
    )
    quantized = tuple(
        entry
        for entry in manifest.entries
        if (
            entry.profile.kind == PROFILE_KIND_QUANTIZED
            and entry.legality.hardware_candidate
        )
    )[:3]
    rows = (
        _numerical_row(reference, 1.0),
        _numerical_row(quantized[0], 1.001),
        _numerical_row(quantized[1], 1.005),
        _numerical_row(quantized[2], 1.02),
    )

    class Evaluator:
        def physical_cost_group_key(self, entry, numerical):
            return {"profile": entry.profile_id}

        def preflight_group_key(self, entry, numerical):
            return {"profile": entry.profile_id}

        def evaluation_group_key(self, entry, numerical):
            return {"profile": entry.profile_id}

        def preflight(self, entry, candidate, numerical):
            if candidate.batch > 4:
                return HardwareEvaluation.failed(
                    "runtime_capacity_exceeded",
                    "synthetic physical capacity exceeded",
                )
            return None

        def __call__(self, entry, candidate, numerical):
            return HardwareEvaluation.failed(
                "synthetic_unrankable",
                f"{entry.profile_id}:{candidate.batch}",
            )

    space = ExactHardwareSpace(
        mlen=(128,),
        blen=(2,),
        hlen=(16,),
        batch=(1, 4, 8),
        hbm_channels=(8,),
        chip_count=(1,),
        sram_policy=("streaming",),
        kv_head_reuse=(False,),
        drain_overlapped=(False,),
        attention_heads=8,
        kv_heads=2,
    )

    def build_study():
        return ExactHardwareStudy(
            manifest=manifest,
            numerical_results=rows,
            space=space,
            hidden_size=128,
            evaluator=Evaluator(),
            evaluator_version="synthetic-evaluator",
            require_complete=False,
            relative_perplexity_limit=1.01,
        )

    serial_study = build_study()
    serial_artifact = serial_study.write(tmp_path / "serial.jsonl")

    block_study = build_study()
    assert block_study.factor_block_count == serial_study.factor_block_count
    block_stream = block_study.iter_factor_evaluations_from_blocks(
        block_study.price_block(index)
        for index in range(block_study.factor_block_count)
    )
    block_artifact = block_study.write(
        tmp_path / "blocks.jsonl",
        factor_stream=block_stream,
    )
    assert (
        (tmp_path / "serial.jsonl").read_bytes()
        == (tmp_path / "blocks.jsonl").read_bytes()
    )
    assert serial_artifact.content_hash == block_artifact.content_hash

    truncated = build_study()
    with pytest.raises(RuntimeError, match="fewer blocks"):
        list(truncated.iter_factor_evaluations_from_blocks(iter(())))


def test_dual_accuracy_frontiers_label_instead_of_filtering() -> None:
    import math as _math

    from decode_dse.hardware.selection import ParetoPoint, dual_accuracy_frontiers
    from decode_dse.profiles import DecodePrecisionProfile

    reference = 2.5

    def _point(weight_format: str, mean_nll: float, tpot: float, energy: float):
        return ParetoPoint(
            profile=DecodePrecisionProfile.quantized(
                weight_format, "MXINT8", "MXINT8", "FP_E5M6"
            ),
            mean_nll=mean_nll,
            tpot_ms=tpot,
            tps=1000.0 / tpot,
            energy_per_token_j=energy,
            area_mm2=100.0,
            candidate_id=f"cand-{weight_format}",
            power_calibration_id="synthetic-power",
            cost_scope="whole_model",
            system_calibration_id="synthetic-system",
            head_service_calibration_id="synthetic-head",
            output_head_location=EXTERNAL_BF16_HEAD,
            whole_model_rankable=True,
            energy_tier="analytic_anchored",
            publication_timing_tier="stage_calibrated_analytic",
        )

    eight_bit = _point("MXINT8", reference + _math.log(1.005), 2.0, 0.10)
    four_bit = _point("MXINT4", reference + _math.log(1.04), 1.0, 0.05)

    report = dual_accuracy_frontiers(
        (eight_bit, four_bit),
        reference_mean_nll=reference,
        strict_relative_perplexity=1.01,
        relaxed_relative_perplexity=1.05,
    )
    budgets = report["budgets"]
    assert budgets["strict"]["admitted_points"] == 1
    assert budgets["relaxed"]["admitted_points"] == 2
    strict_ids = [entry["candidate_id"] for entry in budgets["strict"]["front"]]
    relaxed_ids = [entry["candidate_id"] for entry in budgets["relaxed"]["front"]]
    assert strict_ids == ["cand-MXINT8"]
    # Accuracy stays a dominance objective inside each budget, so the relaxed
    # front keeps the accurate 8-bit point and adds the cost-dominant 4-bit
    # point the strict budget refuses to admit.
    assert relaxed_ids == ["cand-MXINT8", "cand-MXINT4"]
    assert budgets["unconstrained"]["front"] == budgets["relaxed"]["front"]

    with pytest.raises(ValueError, match="tighter than the strict budget"):
        dual_accuracy_frontiers(
            (eight_bit,),
            reference_mean_nll=reference,
            strict_relative_perplexity=1.05,
            relaxed_relative_perplexity=1.01,
        )


# --- Runtime HBM reserve contracts -----------------------------------------
#
# The configured reserve is a PER-CHIP working-memory allowance.  A reserve
# sized for a large device (8 GiB against an 80 GB HBM3 part) must never be
# able to render every geometry on a small HBM2 chip infeasible by
# construction, and - critically - the cheap resource preflight and the full
# simulator evaluation must price the *same* effective reserve, or the
# preflight admits candidates the evaluation then rejects after paying for a
# complete decode walk (the failure mode that marked an entire hardware study
# runtime_infeasible).


def test_effective_runtime_reserve_is_capacity_scaled() -> None:
    effective = evaluation._effective_runtime_hbm_reserve_bytes
    # 8 GiB configured against an 8 GB HBM2 chip clamps to capacity / 8.
    assert effective(8_000_000_000, 8_589_934_592) == 1_000_000_000
    assert effective(32_000_000_000, 8_589_934_592) == 4_000_000_000
    # A reserve below the structural ceiling passes through unchanged, so a
    # correctly sized configuration is priced exactly as written.
    assert effective(8_000_000_000, 536_870_912) == 536_870_912
    assert effective(8_000_000_000, 0) == 0


def test_runtime_reserve_is_per_chip_on_both_topology_paths() -> None:
    # The simulator prices the reserve per chip whether the topology is the
    # legacy aggregate one or an explicit TP x KVP mesh; a replay without
    # TP/KVP must not report a different physical capacity verdict than the
    # study path for the identical system.
    simulator = DecodeSimulator("qwen3-32b")
    precision = simulator.make_precision(attn_w=4, ffn_w=4, kv=4, act_w=4)
    overrides = {
        "MLEN": 128,
        "BLEN": 2,
        "VLEN": 128,
        "HLEN": 128,
        **simulator.hbm_overrides("HBM2", 8),
    }
    reserve = 1 << 29
    common = dict(
        batch=1,
        input_seq=16,
        output_seq=2,
        stride=1,
        n_chips=4,
        hbm_gen="HBM2",
        hbm_channels=8,
        runtime_hbm_reserve_bytes=reserve,
    )
    legacy = simulator.evaluate(precision, hw_over=dict(overrides), **common)
    explicit = simulator.evaluate(
        precision,
        hw_over={
            **overrides,
            "TP": 4,
            "KVP": 1,
            "LINK_PORTS": 1,
            "SRAM_POLICY": "streaming",
        },
        **common,
    )
    assert legacy.runtime_hbm_reserve_bytes == reserve * 4
    assert explicit.runtime_hbm_reserve_bytes == reserve * 4


def test_resource_preflight_prices_the_capacity_scaled_reserve() -> None:
    entry = next(
        row
        for row in _synthetic_manifest().entries
        if (
            row.profile.kind == PROFILE_KIND_QUANTIZED
            and row.legality.hardware_candidate
        )
    )
    candidate = HardwareCandidate(
        mlen=128,
        blen=2,
        vlen=128,
        hlen=128,
        batch=1,
        hbm_channels=8,
        hbm_generation="HBM2",
        chip_count=4,
        tp=4,
        kvp=1,
        link_ports=1,
        sram_policy="streaming",
    )
    backend = object.__new__(evaluation.DecodeSimulatorBackend)
    backend.sim = DecodeSimulator("qwen3-32b")
    backend.output_head_location = evaluation.EXTERNAL_BF16_HEAD
    backend._resource_ledger_cache = {}
    backend._resource_area_cache = {}

    oversized = evaluation.HardwareWorkload(
        input_seq=16,
        output_seq=8,
        stride=1,
        runtime_hbm_reserve_bytes=8_589_934_592,
    )
    status = backend.resource_preflight(entry, candidate, oversized)
    # An oversized reserve is clamped to capacity / divisor on every chip
    # instead of consuming more than the chip itself.
    assert status.capacity.runtime_bytes == 4 * 1_000_000_000
    assert status.capacity.feasible

    sized = evaluation.HardwareWorkload(
        input_seq=16,
        output_seq=8,
        stride=1,
        runtime_hbm_reserve_bytes=536_870_912,
    )
    resized = backend.resource_preflight(entry, candidate, sized)
    # The ledger cache is keyed by the effective reserve, so a different
    # configured reserve can never resurrect a stale capacity verdict.
    assert resized.capacity.runtime_bytes == 4 * 536_870_912
    assert resized.capacity.feasible


def test_backend_evaluation_threads_the_same_reserve_as_preflight() -> None:
    captured: dict[str, object] = {}

    class StubDD:
        @staticmethod
        def set_area_model(*_args, **_kwargs) -> None:
            return None

    class StubSimulator:
        _dd = StubDD()

        @staticmethod
        def make_precision(**_kwargs):
            return SimpleNamespace(spec={"kv_bits": 4.0, "ffn_bits": 4.0})

        @staticmethod
        def hbm_overrides(_generation, _channels):
            return {"HBM_WIDTH": 1024, "HBM_SIZE": 8_000_000_000}

        @staticmethod
        def evaluate(*_args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("captured simulator call")

    backend = object.__new__(evaluation.DecodeSimulatorBackend)
    backend.sim = StubSimulator()
    backend.output_head_location = evaluation.EXTERNAL_BF16_HEAD
    entry = next(
        row
        for row in _synthetic_manifest().entries
        if (
            row.profile.kind == PROFILE_KIND_QUANTIZED
            and row.legality.hardware_candidate
        )
    )
    candidate = HardwareCandidate(
        mlen=128,
        blen=2,
        vlen=128,
        hlen=128,
        batch=1,
        hbm_channels=8,
        hbm_generation="HBM2",
        chip_count=2,
        tp=2,
        kvp=1,
        link_ports=1,
        sram_policy="streaming",
    )
    workload = evaluation.HardwareWorkload(
        input_seq=16,
        output_seq=8,
        stride=1,
        runtime_hbm_reserve_bytes=8_589_934_592,
    )
    with pytest.raises(RuntimeError, match="captured simulator call"):
        backend.evaluate(entry, candidate, workload)
    assert captured["runtime_hbm_reserve_bytes"] == (
        evaluation._effective_runtime_hbm_reserve_bytes(
            8_000_000_000,
            workload.runtime_hbm_reserve_bytes,
        )
    )
    assert captured["runtime_hbm_reserve_bytes"] == 1_000_000_000


# --- Stage-scoped PackedKV capability contracts ------------------------------
#
# The declared publication timing tiers require exactly
# ("compiler_valid", "emulator_valid"); RTL and DC evidence is recorded and
# disclosed, never required.  A capability limitation that lives only in the
# RTL implementation must therefore be RECORDED on the row - full issue code
# list, rtl_valid false - without deleting the row from the priced design
# space.  A limitation on a stage a tier actually rests on still fails closed.


def _packed_batched_target(**overrides) -> PackedKVRuntimeTarget:
    """A geometrically legal packed + batched selector target."""

    fields = {
        "mlen": 512,
        "blen": 2,
        "hlen": 128,
        "batch": 1,
        "kv_heads": 2,
        "head_dim": 128,
        "block_size": 8,
        "selector_bits": 4,
        "packed_kv": True,
        "batched_attention": True,
    }
    fields.update(overrides)
    return PackedKVRuntimeTarget(**fields)


def _selector_traffic_ledger() -> tuple[tuple[str, float], ...]:
    return tuple(
        (key, 1.0 if key == "weight_element_read_bytes" else 0.0)
        for key in sorted(PHYSICAL_TRAFFIC_KEYS)
    )


def _selector_observation(
    *,
    supported: bool,
    issue_codes: tuple[str, ...],
    blocking_issue_codes: tuple[str, ...],
    recorded_issue_codes: tuple[str, ...],
) -> evaluation.SimulatorObservation:
    """One priced-shape observation carrying a chosen capability record."""

    return evaluation.SimulatorObservation(
        profile_id="profile-selector-contract",
        candidate_id="candidate-selector-contract",
        tpot_ms=10.0,
        tps=100.0,
        total_time_s=0.08,
        analytical_area_mm2=50.0,
        traffic=PhysicalTraffic(
            weight_bytes=1.0,
            activation_bytes=0.0,
            kv_read_bytes=0.0,
            kv_write_bytes=0.0,
        ),
        capacity=CapacityBreakdown(
            weight_bytes=1,
            kv_cache_bytes=0,
            runtime_bytes=0,
            available_bytes=1_000,
        ),
        algorithmic_bottleneck="memory",
        realized_bottleneck="memory",
        frac_algorithmic_memory_bound=1.0,
        frac_realized_memory_bound=1.0,
        frac_serialization_bound=0.0,
        generated_tokens_per_step=1,
        decode_steps=8,
        timing_mode="rtl_serialized",
        timing_calibrated=True,
        timing_evidence_id="timing-selector-contract",
        timing_reason="timing_calibrated_emulator_tier",
        execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
        compiler_trace_timing=None,
        kv_layout=evaluation.KV_LAYOUT,
        layout_id="layout-selector-contract",
        capacity_model="capacity-selector-contract",
        runtime_feasible=True,
        max_batch=1,
        max_resident_batch=1,
        max_synchronous_batch=1,
        max_runtime_batch=1,
        fits_onchip_sram=True,
        vector_sram_capacity_bytes=1,
        vector_sram_required_bytes=1,
        matrix_sram_capacity_bytes=1,
        matrix_sram_required_bytes=1,
        hbm_traffic_per_batch_step=_selector_traffic_ledger(),
        hbm_traffic_per_generated_token=_selector_traffic_ledger(),
        traffic_ledger_id="traffic-selector-contract",
        packedkv_selector_supported=supported,
        packedkv_selector_capability_id="packedkv-selector-capability-test",
        packedkv_selector_issue_codes=issue_codes,
        packedkv_selector_blocking_issue_codes=blocking_issue_codes,
        packedkv_selector_recorded_issue_codes=recorded_issue_codes,
        bandwidth_calibration_id="bandwidth-selector-contract",
        total_hbm_bytes=8.0,
        events=(
            DecodeEvent("LINEAR:MXINT8xMXINT8", 1, 512, 2),
            DecodeEvent("UNMODELED:LM_HEAD_BF16", 1, 512, 2),
        ),
        output_head_location=evaluation.DECODE_BF16_HEAD,
        system_area_mm2=50.0,
        area_evidence_tier="analytical_uncalibrated",
        logic_area_mm2=25.0,
    )


def _selector_evaluation(
    observation: evaluation.SimulatorObservation,
) -> HardwareEvaluation:
    """Drive the fail-closed evaluator over one injected observation."""

    entry = next(
        row
        for row in _synthetic_manifest().entries
        if (
            row.profile.kind == PROFILE_KIND_QUANTIZED
            and row.legality.hardware_candidate
        )
    )
    candidate = HardwareCandidate(
        mlen=512,
        blen=2,
        vlen=512,
        hlen=128,
        batch=1,
        hbm_channels=8,
        hbm_generation="HBM2",
        chip_count=1,
        tp=1,
        kvp=1,
        link_ports=0,
        sram_policy="streaming",
    )
    bound = replace(
        observation,
        profile_id=entry.profile_id,
        candidate_id=candidate.candidate_id,
    )

    class StubBackend:
        provenance = {"backend": "selector-capability-contract"}

        @staticmethod
        def evaluate(*_args, **_kwargs):
            return bound

    evaluator = evaluation.ProductionHardwareEvaluator(
        StubBackend(),
        evaluation.HardwareWorkload(
            input_seq=16,
            output_seq=8,
            stride=1,
            runtime_hbm_reserve_bytes=536_870_912,
        ),
        publication_timing_tier=STAGE_CALIBRATED_ANALYTIC_TIMING_TIER,
    )
    return evaluator(
        entry,
        candidate,
        {
            "state": "succeeded",
            "result": {"mean_nll": 1.0, "token_count": 64},
        },
    )


def test_declared_tiers_never_require_the_recorded_stages() -> None:
    # The whole gate rests on this: no publication tier asks for RTL or DC
    # validity, so no RTL-or-DC-only limitation may withhold a price.
    assert set(PRICING_BLOCKING_STAGES) & set(PRICING_RECORDED_STAGES) == set()
    for tier, required in TIMING_TIER_REQUIRED_VALIDITY.items():
        recorded_only = {
            f"{stage}_valid" for stage in PRICING_RECORDED_STAGES
        } | {"dc_calibrated"}
        assert not set(required) & recorded_only, tier
        assert set(required) <= {
            f"{stage}_valid" for stage in PRICING_BLOCKING_STAGES
        }, tier


def test_mxfp_packed_batched_profile_is_recorded_and_still_priced() -> None:
    # E4M3 is the best-measured hardware-legal format in the numerical screen
    # and it is MXFP, so the batched selector's MXINT-only implementation is
    # precisely the limitation that must not delete it.  (E3M4 scores well too
    # but is excluded by an independent and correct legality rule: it is not a
    # HARDWARE_MXFP_FORMATS operand at all.)
    for token in ("E4M3", "E5M2"):
        profile = DecodePrecisionProfile.quantized(
            token,
            token,
            token,
            "FP_E3M2",
        )
        capability = evaluate_stack_capability(profile, _packed_batched_target())
        codes = tuple(issue.code for issue in capability.issues)
        assert codes == ("rtl_batched_mxfp_unsupported",)
        assert all(issue.stages == ("rtl",) for issue in capability.issues)
        # Recorded, never required.
        assert capability.blocking_issues == ()
        assert capability.prices_at_publication_tier is True
        assert tuple(
            issue.code for issue in capability.recorded_issues
        ) == codes
        # The row still carries the RTL failure in its validity floor.
        assert capability.validity_floor.rtl_valid is False
        assert capability.validity_floor.compiler_valid is None
        assert capability.validity_floor.emulator_valid is None
        assert capability.stage_support["rtl"] is False
        assert capability.stage_support["compiler"] is True
        assert capability.stage_support["emulator"] is True

    observation = _selector_observation(
        supported=True,
        issue_codes=("rtl_batched_mxfp_unsupported",),
        blocking_issue_codes=(),
        recorded_issue_codes=("rtl_batched_mxfp_unsupported",),
    )
    # The exhaustive issue list survives on the row; only the pricing verdict
    # is stage-scoped.
    assert observation.packedkv_selector_issue_codes == (
        "rtl_batched_mxfp_unsupported",
    )
    outcome = _selector_evaluation(observation)
    # The row travels all the way past the selector gate to the next real
    # gate; the recorded RTL limitation no longer terminates it.  The
    # decode-local output head is no longer a terminal gate either, so the
    # next unmet requirement is the stub's absent admission evidence.
    assert outcome.error_code == "admission_correctness_unverified"


def test_blocking_stage_selector_limits_still_fail_closed() -> None:
    # A misaligned packed geometry is a compiler/emulator-stage limitation:
    # the tiers rest on those stages, so the point must remain unrankable.
    profile = DecodePrecisionProfile.quantized(
        "MXINT8",
        "MXINT8",
        "MXINT8",
        "FP_E3M2",
    )
    capability = evaluate_stack_capability(
        profile,
        _packed_batched_target(head_dim=64),
    )
    blocking = tuple(issue.code for issue in capability.blocking_issues)
    assert "packedkv_selector_stride" in blocking
    assert capability.prices_at_publication_tier is False
    assert capability.validity_floor.compiler_valid is False
    assert capability.validity_floor.emulator_valid is False

    observation = _selector_observation(
        supported=False,
        issue_codes=("packedkv_selector_stride",),
        blocking_issue_codes=("packedkv_selector_stride",),
        recorded_issue_codes=(),
    )
    outcome = _selector_evaluation(observation)
    assert outcome.error_code == "packedkv_selector_unsupported"
    assert outcome.error_message == (
        "PackedKV selector capability failed: packedkv_selector_stride"
    )


def test_legacy_moe_provenance_without_native_power_ledger_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires the physical body layout"):
        replace(
            _selector_observation(
                supported=True,
                issue_codes=(),
                blocking_issue_codes=(),
                recorded_issue_codes=(),
            ),
            moe_workload={
                "schema": "plena-routed-moe-decode-workload/v1",
                "provenance": {
                    "publication_rankable": False,
                    "unrankable_reason": "missing matched routed-MoE calibration",
                },
            },
        )


def test_selector_observation_contract_rejects_inconsistent_evidence() -> None:
    # supported must remain exactly the emptiness of the BLOCKING codes, the
    # partition must stay exhaustive, and the lists must stay canonical.
    with pytest.raises(ValueError, match="capability evidence is inconsistent"):
        _selector_observation(
            supported=True,
            issue_codes=("packedkv_selector_stride",),
            blocking_issue_codes=("packedkv_selector_stride",),
            recorded_issue_codes=(),
        )
    with pytest.raises(ValueError, match="capability evidence is inconsistent"):
        _selector_observation(
            supported=False,
            issue_codes=("rtl_batched_mxfp_unsupported",),
            blocking_issue_codes=(),
            recorded_issue_codes=("rtl_batched_mxfp_unsupported",),
        )
    with pytest.raises(ValueError, match="partition exhaustively"):
        # Dropping an RTL limitation from the recorded list is exactly the
        # silent loss of evidence this contract exists to prevent.
        _selector_observation(
            supported=True,
            issue_codes=("rtl_batched_mxfp_unsupported",),
            blocking_issue_codes=(),
            recorded_issue_codes=(),
        )
    with pytest.raises(ValueError, match="blocking and recorded"):
        _selector_observation(
            supported=False,
            issue_codes=("packedkv_selector_stride",),
            blocking_issue_codes=("packedkv_selector_stride",),
            recorded_issue_codes=("packedkv_selector_stride",),
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        _selector_observation(
            supported=False,
            issue_codes=("packedkv_selector_stride", "packedkv_block_alignment"),
            blocking_issue_codes=(
                "packedkv_selector_stride",
                "packedkv_block_alignment",
            ),
            recorded_issue_codes=(),
        )


def _admission_status_dict(**overrides: object) -> dict[str, object]:
    """A serialized admission status shaped like the live workspace receipt.

    The live receipt is prepared under the content-addressed recompute policy,
    which persists nothing; the persisted contract is the other declared
    policy. Both must read as valid evidence, and nothing else may.
    """

    document_count = 48
    artifact_count = document_count * (len(DECODE_FORMATS) + 1)
    status: dict[str, object] = {
        "schema_version": ADMISSION_CORRECTNESS_SCHEMA,
        "scope": ADMISSION_CORRECTNESS_SCOPE,
        "passed": True,
        "failures": [],
        "receipt_sha256": "a" * 64,
        "evidence_id": "admission-correctness-" + "b" * 64,
        "manifest_hash": "c" * 64,
        "run_plan_hash": "d" * 64,
        "prompt_manifest_hash": "e" * 64,
        "admission_contract_id": "packedkv-admission-2c4458e60da527dd",
        "admission_index_hash": "f" * 64,
        "numerical_validation_hash": "0" * 64,
        "admission_code_revision": "1" * 64,
        "runtime_environment_fingerprint": "2" * 64,
        "sample_bundle_hash": "3" * 64,
        "layout_id": "packed-gqa-mlen1024-block8-native-encoding",
        "persistence_contract": RECOMPUTABLE_ADMISSION_POLICY,
        "formats": list(DECODE_FORMATS),
        "document_count": document_count,
        "artifact_count": artifact_count,
        "tensor_count": artifact_count * 64,
        "persisted_bytes": 0,
        "projected_cold_artifact_bytes": 36546648474,
        "projected_numerical_view_bytes": 43098611184,
        "source_dtype": "BF16",
        "block_size": MX_BLOCK_SIZE,
        "steady_state_tpot_included": False,
        "hardware_latency_calibrated": False,
        "hardware_energy_calibrated": False,
        "ttft_rankable": False,
        "admission_energy_rankable": False,
    }
    status.update(overrides)
    return status


def test_recomputable_admission_status_reads_as_valid_evidence() -> None:
    # The recompute policy rebuilds every packed plane per format and
    # deliberately persists nothing. Requiring persisted_bytes > 0 here
    # rejected every receipt the pipeline actually prepares, which failed
    # 100% of study rows with admission_correctness_unverified.
    assert admission_correctness_status_valid(_admission_status_dict())


def test_persisted_admission_status_reads_as_valid_evidence() -> None:
    assert admission_correctness_status_valid(
        _admission_status_dict(
            persistence_contract=ADMISSION_PERSISTENCE_CONTRACT,
            persisted_bytes=36546648474,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        # Each policy pins its own persisted-byte expectation; crossing them
        # is unproven evidence, not a relaxation.
        {"persisted_bytes": 1},
        {
            "persistence_contract": ADMISSION_PERSISTENCE_CONTRACT,
            "persisted_bytes": 0,
        },
        # An undeclared policy has no persisted-byte expectation at all.
        {"persistence_contract": "content_addressed_recompute"},
        {"persistence_contract": None},
        {"persistence_contract": ""},
        # A receipt that projects nothing to rebuild proves nothing; under the
        # persisted contract this was already implied by persisted_bytes > 0.
        {"projected_cold_artifact_bytes": 0},
        {"projected_numerical_view_bytes": 0},
        # Booleans are not byte counts.
        {"persisted_bytes": False},
    ],
)
def test_admission_status_validity_stays_fail_closed(
    overrides: dict[str, object],
) -> None:
    assert not admission_correctness_status_valid(
        _admission_status_dict(**overrides)
    )


def test_admission_status_validity_still_requires_full_identity() -> None:
    # Widening the persistence policy must not widen anything else.
    for field in (
        "receipt_sha256",
        "manifest_hash",
        "run_plan_hash",
        "prompt_manifest_hash",
        "admission_index_hash",
        "numerical_validation_hash",
        "admission_code_revision",
        "runtime_environment_fingerprint",
        "sample_bundle_hash",
    ):
        assert not admission_correctness_status_valid(
            _admission_status_dict(**{field: None})
        ), field
        assert not admission_correctness_status_valid(
            _admission_status_dict(**{field: "not-a-sha256"})
        ), field
    assert not admission_correctness_status_valid(
        _admission_status_dict(passed=False)
    )
    assert not admission_correctness_status_valid(
        _admission_status_dict(failures=["boom"])
    )
    assert not admission_correctness_status_valid(
        _admission_status_dict(scope="hardware_energy")
    )


def _write_admission_receipt(
    path: Path,
    *,
    manifest_hash: str,
) -> Path:
    """A receipt that is well-formed exactly up to its manifest identity.

    The identity comparison is the first check the loader makes, so a receipt
    that stops being valid immediately afterwards still proves whether that
    comparison fired and which hash it fired against.
    """

    body = {
        "schema_version": admission_cost.ADMISSION_PREPARATION_SCHEMA,
        "manifest_hash": manifest_hash,
    }
    body["content_hash"] = admission_cost._content_hash(
        {key: value for key, value in body.items() if key != "content_hash"}
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_admission_receipt_is_bound_to_the_hash_it_is_given(
    tmp_path: Path,
) -> None:
    # Admission is a workspace-level operation, so the evaluator validates the
    # receipt against the workspace manifest that sits beside it, never the
    # per-shard manifest passed as --manifest. This check is what makes that
    # binding meaningful; it must stay exact in both directions.
    workspace_hash = "e" * 64
    shard_hash = "9" * 64
    receipt = _write_admission_receipt(
        tmp_path / "admission_preparation.json",
        manifest_hash=workspace_hash,
    )

    mismatched = admission_cost.load_admission_correctness_evidence(
        receipt,
        manifest_hash=shard_hash,
    )
    assert not mismatched.passed
    assert mismatched.failures == (
        "ValueError: admission receipt manifest identity mismatch",
    )

    # The matching hash gets past the identity gate and fails later on the
    # parts this stub receipt deliberately omits, which is how we know the
    # gate is hash-specific rather than always-on.
    matched = admission_cost.load_admission_correctness_evidence(
        receipt,
        manifest_hash=workspace_hash,
    )
    assert not matched.passed
    assert "manifest identity mismatch" not in matched.failures[0]


def test_admission_receipt_rejects_a_malformed_manifest_hash(
    tmp_path: Path,
) -> None:
    receipt = _write_admission_receipt(
        tmp_path / "admission_preparation.json",
        manifest_hash="f" * 64,
    )
    for bad in ("", "not-a-sha256", "F" * 64, "f" * 63):
        status = admission_cost.load_admission_correctness_evidence(
            receipt,
            manifest_hash=bad,
        )
        assert not status.passed, bad
        assert status.failures[0].startswith("ValueError:"), bad


# --- Study configuration parity ---------------------------------------------


def test_study_configs_declare_the_same_calibrated_hbm_and_accuracy_budgets() -> None:
    """Both studies search the receipted HBM2 groups under one accuracy budget.

    The Llama grid previously searched a single HBM channel group and declared
    no accuracy budget, which made its promotion record carry a null
    `dual_accuracy_frontiers` and its Pareto figure render no envelopes - an
    absent disclosure rather than a narrower claim.
    """

    repository = Path(__file__).resolve().parents[2]
    configs = {
        name: json.loads(
            (repository / "decode_dse" / "configs" / f"{name}.json")
            .read_text(encoding="utf-8")
        )
        for name in ("llama3_1_8b", "qwen3_32b")
    }
    budgets = {
        "strict_relative_perplexity": 1.01,
        "relaxed_relative_perplexity": 1.05,
    }
    for name, config in configs.items():
        space = config["hardware_space"]
        assert space["HBM_GENERATION"] == "HBM2", name
        assert space["HBM_CHANNELS"] == [8, 16, 32], name
        assert config["accuracy_budgets"] == budgets, name
        # A per-config baseline channel count nothing reads is a claim with no
        # consumer; the searched channel set is the only declaration.
        assert "baseline_hbm_channels" not in config, name


# --- Simulator checkout resolution ------------------------------------------


def test_simulator_root_resolution_is_explicit_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A study must record which simulator checkout priced it.

    Falling back to a sibling directory named `PLENA_Simulator` lets a
    worktree price against a different checkout than the one under test, and
    without the resolved root in provenance the two outcomes are
    indistinguishable after the fact.
    """

    from decode_dse.hardware import power_bridge

    checkout = tmp_path / "simulator-checkout"
    (checkout / "analytic_models" / "disagg_serve").mkdir(parents=True)

    monkeypatch.setenv(power_bridge.SIMULATOR_ROOT_ENV_VAR, str(checkout))
    resolved = power_bridge.resolve_simulator_root()
    assert resolved.root == checkout.resolve()
    assert resolved.source == power_bridge.SIMULATOR_ROOT_FROM_ENVIRONMENT
    assert resolved.to_dict() == {
        "simulator_root": str(checkout.resolve()),
        "simulator_root_source": power_bridge.SIMULATOR_ROOT_FROM_ENVIRONMENT,
    }
    # Every reader of the simulator tree resolves through the same rule.
    assert evaluation.simulator_root() == checkout.resolve()

    monkeypatch.setenv(power_bridge.SIMULATOR_ROOT_ENV_VAR, "   ")
    with pytest.raises(ValueError, match="empty value"):
        power_bridge.resolve_simulator_root()

    monkeypatch.setenv(
        power_bridge.SIMULATOR_ROOT_ENV_VAR,
        str(tmp_path / "not-a-checkout"),
    )
    with pytest.raises(FileNotFoundError, match="not a simulator checkout"):
        power_bridge.resolve_simulator_root()

    monkeypatch.delenv(power_bridge.SIMULATOR_ROOT_ENV_VAR, raising=False)
    sibling = Path(power_bridge.__file__).resolve().parents[3] / "PLENA_Simulator"
    if (sibling / "analytic_models" / "disagg_serve").is_dir():
        implicit = power_bridge.resolve_simulator_root()
        assert implicit.root == sibling.resolve()
        assert (
            implicit.source
            == power_bridge.SIMULATOR_ROOT_FROM_SIBLING_DEFAULT
        )
    else:
        with pytest.raises(FileNotFoundError):
            power_bridge.resolve_simulator_root()


def test_analytic_power_provenance_records_the_resolved_simulator_root() -> None:
    from decode_dse.hardware import power_bridge

    provenance = power_bridge.analytic_power_provenance()
    resolved = power_bridge.resolve_simulator_root()
    assert provenance["simulator_root"] == str(resolved.root)
    assert provenance["simulator_root_source"] == resolved.source
    assert provenance["energy_tier"] == "analytic_anchored"


# --- Admission persistence policy naming ------------------------------------
#
# The admission persistence policy is one concept written under three key
# names: `persistence_policy` at the top level of the preparation receipt and
# its index, `persistence_contract` inside the persisted-contract resource
# projection, and `policy` inside the recomputable resource projection. All
# three carry values from the same two-element vocabulary, so they are read
# through one accessor that fails closed on anything else. The on-disk
# artifacts are untouched.


def test_admission_policy_reads_every_documented_key_name() -> None:
    for key in admission_cost.ADMISSION_POLICY_KEYS:
        assert (
            admission_cost.admission_persistence_policy(
                {key: RECOMPUTABLE_ADMISSION_POLICY}
            )
            == RECOMPUTABLE_ADMISSION_POLICY
        )
        assert (
            admission_cost.admission_persistence_policy(
                {key: ADMISSION_PERSISTENCE_CONTRACT}
            )
            == ADMISSION_PERSISTENCE_CONTRACT
        )
    # One document may spell the policy more than once, but only consistently.
    assert (
        admission_cost.admission_persistence_policy(
            {
                "persistence_policy": RECOMPUTABLE_ADMISSION_POLICY,
                "policy": RECOMPUTABLE_ADMISSION_POLICY,
            }
        )
        == RECOMPUTABLE_ADMISSION_POLICY
    )


def test_admission_policy_fails_closed_on_unknown_or_conflicting_values() -> None:
    with pytest.raises(ValueError, match="unknown admission persistence policy"):
        admission_cost.admission_persistence_policy(
            {"persistence_policy": "content_addressed_recompute"}
        )
    with pytest.raises(ValueError, match="declared inconsistently"):
        admission_cost.admission_persistence_policy(
            {
                "persistence_policy": RECOMPUTABLE_ADMISSION_POLICY,
                "policy": ADMISSION_PERSISTENCE_CONTRACT,
            }
        )
    with pytest.raises(ValueError, match="undeclared"):
        admission_cost.admission_persistence_policy({})
    assert (
        admission_cost.admission_persistence_policy({}, required=False) is None
    )
    assert (
        admission_cost.admission_persistence_policy(
            {"persistence_policy": None},
            required=False,
        )
        is None
    )
    with pytest.raises(TypeError, match="must be read from an object"):
        admission_cost.admission_persistence_policy(
            RECOMPUTABLE_ADMISSION_POLICY
        )


def test_admission_cost_exports_its_recompute_policy() -> None:
    assert "RECOMPUTABLE_ADMISSION_POLICY" in admission_cost.__all__
    assert "admission_persistence_policy" in admission_cost.__all__

    from decode_dse.hardware import selection

    assert "dual_accuracy_frontiers" in selection.__all__


# --- decode-local BF16 output head ------------------------------------------
#
# The decode-local BF16 head is an analytic sensitivity: its weights and its
# per-decode-step traffic are charged to the decode chip's physical ledger, but
# its compute is idealized. These tests pin its disclosure and unrankability,
# then independently validate the measured remote publication boundary.


def _head_service_calibration(tmp_path: Path):
    """A passing synthetic remote-head calibration for the publication arm."""

    path, model = _assembled_synthetic_head_artifact(tmp_path)
    status = load_bf16_head_service_artifact(
        path,
        model_name=model["model_name"],
        model_revision=model["model_revision"],
        hidden_size=model["hidden_size"],
        vocab_size=model["vocab_size"],
        tie_embeddings=model["tie_embeddings"],
        required_batches=(1, 4, 8),
    )
    assert status.passed, status.failures
    return status


def _boundary_metrics(
    *,
    location: str,
    head_estimate=None,
    comparison=None,
    head_status: Mapping[str, object] | None = None,
    head_resource_status: Mapping[str, object] | None = None,
    rankable: bool | None = None,
    energy=None,
    system_calibration_id: str | None = None,
) -> HardwareMetrics:
    """One priced row at the requested output-head boundary."""

    from decode_dse.hardware.design_space import OUTPUT_HEAD_SERVICE_MODES

    if rankable is None:
        rankable = location == EXTERNAL_BF16_HEAD
    if (
        head_resource_status is None
        and rankable
        and head_status is not None
    ):
        head_resource_status = dict(
            _serialized_measured_head_boundary()["resource_status"]
        )
        head_resource_status["head_service_artifact_sha256"] = head_status[
            "artifact_sha256"
        ]
        head_resource_status["head_service_calibration_id"] = head_status[
            "calibration_id"
        ]
        head_resource_status["head_service_provenance_id"] = head_status[
            "provenance_id"
        ]
    tpot_ms = 10.0
    batch = head_estimate.batch if head_estimate is not None else (
        comparison.batch if comparison is not None else 1
    )
    head_latency_ms = (
        head_estimate.total_latency_s * 1000.0
        if head_estimate is not None
        else 0.0
    )
    whole_tpot_ms = tpot_ms + head_latency_ms
    return HardwareMetrics(
        tpot_ms=tpot_ms,
        tps=batch * 1_000.0 / tpot_ms,
        area_mm2=1.0,
        traffic=PhysicalTraffic(1.0, 0.0, 0.0, 0.0),
        capacity=CapacityBreakdown(1, 0, 0, batch),
        algorithmic_bottleneck="memory",
        realized_bottleneck="memory",
        frac_algorithmic_memory_bound=1.0,
        frac_realized_memory_bound=1.0,
        frac_serialization_bound=0.0,
        timing_calibrated=True,
        timing_evidence_id="timing-" + "a" * 64,
        timing_reason="calibrated",
        execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
        bandwidth_calibration_id="bandwidth-operating-point-" + "b" * 64,
        service_mode=OUTPUT_HEAD_SERVICE_MODES[location],
        output_head_location=location,
        output_head_status=(
            dict(head_status)
            if head_status is not None
            else (
                local_head_boundary_status()
                if location == DECODE_BF16_HEAD
                else {}
            )
        ),
        output_head_resource_status=(head_resource_status or {}),
        output_head_service=head_estimate,
        output_head_comparison=comparison,
        whole_model_tpot_ms=whole_tpot_ms if rankable else None,
        whole_model_tps=(
            batch * 1_000.0 / whole_tpot_ms if rankable else None
        ),
        whole_model_energy=energy,
        system_calibration_id=system_calibration_id,
        whole_model_rankable=rankable,
        publication_timing_tier=(
            STAGE_CALIBRATED_ANALYTIC_TIMING_TIER if rankable else None
        ),
    )


def test_local_bf16_head_is_unrankable_and_discloses_its_idealization() -> None:
    metrics = _boundary_metrics(location=DECODE_BF16_HEAD)
    boundary = metrics.to_dict()["output_head_boundary"]

    # The decoder-stack sensitivity remains available, but missing measured
    # head compute prevents every whole-model publication metric.
    assert metrics.tpot_ms == 10.0
    assert metrics.whole_model_rankable is False
    assert metrics.publication_timing_tier is None
    assert metrics.whole_model_tpot_ms is None
    assert metrics.whole_model_tps is None

    # The idealization is on the row, named, and machine readable.
    assert boundary["location"] == DECODE_BF16_HEAD
    assert boundary["service_mode"] == LOCAL_HEAD_MODE
    assert boundary["scope_idealizations"] == [
        LOCAL_HEAD_COMPUTE_IDEALIZATION
    ]
    # Nothing here may be mistaken for a measured head.
    assert boundary["estimate"] is None
    assert boundary["status"]["passed"] is False
    assert LOCAL_HEAD_COMPUTE_IDEALIZATION in boundary["status"]["failures"]
    with pytest.raises(
        ValueError,
        match="fully priced output-head boundary",
    ):
        replace(
            metrics,
            whole_model_rankable=True,
            whole_model_tpot_ms=metrics.tpot_ms,
            whole_model_tps=metrics.tps,
            publication_timing_tier=(
                STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
            ),
        )


def test_local_head_disclosure_cannot_be_dropped_or_forged() -> None:
    # The disclosure is derived from the priced boundary, so a caller can
    # neither omit it nor attach it to the measured remote service.
    assert _boundary_metrics(
        location=DECODE_BF16_HEAD
    ).output_head_idealizations == (LOCAL_HEAD_COMPUTE_IDEALIZATION,)
    with pytest.raises(ValueError, match="idealizations must match"):
        replace(
            _boundary_metrics(location=DECODE_BF16_HEAD),
            output_head_location=EXTERNAL_BF16_HEAD,
            output_head_idealizations=(LOCAL_HEAD_COMPUTE_IDEALIZATION,),
        )


def test_local_head_charges_its_weights_and_traffic_to_the_decode_hbm() -> None:
    # The whole point of the local boundary: the decode chip pays for the
    # head's BF16 weights in capacity and re-reads them every decode step.
    from decode_dse.hardware.power_bridge import _simulator_module
    from decode_dse.simulator_bridge import _disagg

    dd = _disagg()
    ledger = _simulator_module("physical_ledger")
    dims = {
        "hidden": 4096,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "layers": 4,
        "inter": 14336,
        "vocab": 128256,
        "tie_embeddings": False,
        "model_type": "llama",
        "num_experts": 1,
        "experts_per_token": 1,
        "sliding_window": 0,
        "n_sliding": 0,
        "n_full": 4,
        "qk_norm": False,
    }
    prec = {
        "attn_bits": 8.0,
        "attn_elem": 8,
        "ffn_bits": 8.0,
        "ffn_elem": 8,
        "kv_bits": 8.0,
        "kv_elem": 8,
        "block_size": 8,
    }
    head_bytes = dims["hidden"] * dims["vocab"] * 2

    assert dd.decoder_owns_output_head(DECODE_BF16_HEAD) is True
    assert dd.decoder_owns_output_head(EXTERNAL_BF16_HEAD) is False

    with_head = ledger.weight_ledger(
        dims, prec, include_lm_head=True, mlen=128
    )
    without_head = ledger.weight_ledger(
        dims, prec, include_lm_head=False, mlen=128
    )
    capacity_delta = (
        with_head.resident.total_aligned - without_head.resident.total_aligned
    )
    assert capacity_delta >= head_bytes

    step_with = ledger.decode_step_traffic_ledger(
        dims,
        prec,
        context=128,
        batch=1,
        mlen=1024,
        kv_layout="dense_selector",
        weights=with_head,
        include_lm_head=True,
    )
    step_without = ledger.decode_step_traffic_ledger(
        dims,
        prec,
        context=128,
        batch=1,
        mlen=1024,
        kv_layout="dense_selector",
        weights=without_head,
        include_lm_head=False,
    )
    assert step_with.read_bytes - step_without.read_bytes >= head_bytes


def test_external_head_arm_still_prices_and_stays_measured(
    tmp_path: Path,
) -> None:
    status = _head_service_calibration(tmp_path)
    estimate = status.calibration.estimate(1)
    metrics = _boundary_metrics(
        location=EXTERNAL_BF16_HEAD,
        head_estimate=estimate,
        head_status=status.to_dict(),
    )
    boundary = metrics.to_dict()["output_head_boundary"]

    assert boundary["location"] == EXTERNAL_BF16_HEAD
    assert boundary["service_mode"] == HEAD_SERVICE_MODE
    # The measured arm claims no idealization at all.
    assert boundary["scope_idealizations"] == []
    assert boundary["estimate"]["calibration_id"] == estimate.calibration_id
    # The remote endpoint is a second, serialized device.
    assert metrics.whole_model_tpot_ms > metrics.tpot_ms


def test_strict_system_boundary_requires_a_feasible_composed_budget(
    tmp_path: Path,
) -> None:
    status = _head_service_calibration(tmp_path)
    estimate = status.calibration.estimate(1)
    metrics = _boundary_metrics(
        location=EXTERNAL_BF16_HEAD,
        head_estimate=estimate,
        head_status=status.to_dict(),
    )

    # A valid endpoint receipt alone is insufficient: the row must carry a
    # composed decoder-plus-endpoint budget and survive every hard limit.
    assert (
        metrics.to_dict()["whole_model"][
            "strict_system_resource_boundary_valid"
        ]
        is False
    )

    feasible_budget = ResourceBudgetStatus(
        aggregate_area_mm2=100.0,
        aggregate_hbm_capacity_bytes=1_000,
        aggregate_hbm_bandwidth_bytes_per_s=2_000.0,
        aggregate_multiplier_count=1,
        budget=ResourceBudget(
            aggregate_area_limit_mm2=100.0,
            aggregate_hbm_capacity_limit_bytes=1_000,
            aggregate_hbm_bandwidth_limit_bytes_per_s=2_000.0,
            reference_system="test-reference",
        ),
    )
    feasible = replace(
        metrics,
        system_area_mm2=100.0,
        resource_budget=feasible_budget,
    )
    assert (
        feasible.to_dict()["whole_model"][
            "strict_system_resource_boundary_valid"
        ]
        is True
    )

    over_budget = replace(
        metrics,
        system_area_mm2=100.1,
        resource_budget=replace(
            feasible_budget,
            aggregate_area_mm2=100.1,
        ),
    )
    serialized = over_budget.to_dict()
    assert serialized["resource_budget"]["area_feasible"] is False
    assert serialized["resource_budget"]["feasible"] is False
    assert (
        serialized["whole_model"][
            "strict_system_resource_boundary_valid"
        ]
        is False
    )


def test_head_service_artifact_is_recorded_beside_a_local_headline(
    tmp_path: Path,
) -> None:
    status = _head_service_calibration(tmp_path)
    estimate = status.calibration.estimate(1)
    metrics = _boundary_metrics(
        location=DECODE_BF16_HEAD,
        comparison=estimate,
    )
    boundary = metrics.to_dict()["output_head_boundary"]

    # Both placements stay reportable from one row, but the local arm remains
    # a decoder-stack sensitivity rather than a whole-model price.
    assert boundary["location"] == DECODE_BF16_HEAD
    assert boundary["estimate"] is None
    assert boundary["comparison_estimate"]["calibration_id"] == (
        estimate.calibration_id
    )
    assert metrics.whole_model_rankable is False
    assert metrics.whole_model_tpot_ms is None
    with pytest.raises(ValueError, match="belongs to the local boundary"):
        _boundary_metrics(
            location=EXTERNAL_BF16_HEAD,
            head_estimate=estimate,
            comparison=estimate,
            head_status=status.to_dict(),
        )


def test_local_head_energy_charges_no_external_idle_power(
    tmp_path: Path,
) -> None:
    from decode_dse.hardware.design_space import CalibratedEnergy

    decoder = CalibratedEnergy(
        calibration_id="analytic-decode-energy-" + "a" * 64,
        compute_j=1.0,
        vector_j=0.0,
        sram_j=0.0,
        hbm_j=1.0,
        leakage_j=0.01,
        duration_s=0.01,
        energy_tier="analytic_anchored",
        energy_id="analytic-decode-energy-" + "a" * 64,
        token_latency_s=0.01,
    )
    estimate = _head_service_calibration(tmp_path).calibration.estimate(1)

    external, external_id = evaluation._whole_model_energy(
        decoder,
        estimate,
        decoder_tpot_ms=10.0,
        batch=1,
        head_resource_receipt=SimpleNamespace(
            decoder_interface_energy_j_per_byte=1e-12
        ),
    )
    local, local_id = evaluation._whole_model_energy(
        decoder,
        None,
        decoder_tpot_ms=10.0,
        batch=1,
    )

    # Reserving a whole external endpoint charges its idle draw across the
    # decode step; a decode chip that owns its own head has no second device
    # to provision, so no idle power is charged at all.
    assert external.leakage_j > decoder.leakage_j
    assert local.leakage_j == decoder.leakage_j
    assert local.total_j == decoder.total_j
    assert local.total_j < external.total_j
    # The two systems can never share an identity.
    assert local_id != external_id
    assert local_id == local_head_system_calibration_id(
        decoder.calibration_id
    )
    assert local_id.startswith("decode-local-head-system-")


def test_legacy_configs_retain_an_unrankable_local_analytic_headline() -> None:
    for name in ("llama3_1_8b", "qwen3_32b"):
        config = json.loads(
            (
                Path(evaluation.__file__).resolve().parents[1]
                / "configs"
                / f"{name}.json"
            ).read_text(encoding="utf-8")
        )
        contract = config["output_head_contract"]
        assert contract["headline_location"] == DECODE_BF16_HEAD
        assert contract["headline_idealizations"] == [
            LOCAL_HEAD_COMPUTE_IDEALIZATION
        ]
        # The measured service is retained as the comparison arm.
        assert contract["comparison_location"] == EXTERNAL_BF16_HEAD
        assert evaluation.config_output_head_location(config) == (
            DECODE_BF16_HEAD
        )
        # The evaluator accepts the declared contract unchanged.
        evaluation._validate_system_boundary_config(config)


def test_qwen30b_config_requires_the_local_mx_headline() -> None:
    config = json.loads(
        (
            Path(evaluation.__file__).resolve().parents[1]
            / "configs"
            / "qwen3_30b_a3b_thinking_2507.json"
        ).read_text(encoding="utf-8")
    )
    contract = config["output_head_contract"]
    assert contract == dict(evaluation.OUTPUT_HEAD_CONTRACT)
    assert contract["headline_location"] == evaluation.DECODE_MX_HEAD
    assert contract["headline_idealizations"] == []
    assert contract["comparison_location"] == EXTERNAL_BF16_HEAD
    assert config["publication"]["output_head_location"] == (
        evaluation.DECODE_MX_HEAD
    )
    assert evaluation.config_output_head_location(config) == (
        evaluation.DECODE_MX_HEAD
    )
    evaluation._validate_system_boundary_config(config)


# --- publication eligibility rests on the blocking stages --------------------
#
# The declared contract is that RTL evidence is recorded, never required: both
# publication timing tiers rest on compiler-emitted programs executed under the
# calibrated emulator contract (see TIMING_TIER_REQUIRED_VALIDITY).  The
# deployment gate must therefore admit an RTL-unvalidated candidate while still
# refusing a candidate whose *blocking* stages are missing or failed.


def _publication_fixture(
    *,
    validity: StackValidity,
):
    """One complete measured-external-head publication set."""

    power_id = "analytic-decode-energy-" + "a" * 64
    head_id = "bf16-head-service-" + "c" * 64
    head_provenance = "bf16-head-provenance-" + "d" * 64
    timing = _TimingEvidence("timing-" + "e" * 64)
    head = _HeadEvidence(head_id, head_provenance)
    system_id = composite_system_calibration_id(
        power_id,
        head_id,
        head_provenance,
    )

    def candidate(role: str, index: int) -> PublicationCandidate:
        return PublicationCandidate(
            evaluation_class=role,
            profile_id=f"profile-{index}",
            candidate_id=f"candidate-{index}",
            profile_kind="quantized",
            perplexity=10.0,
            tpot_ms=2.0,
            energy_per_token_j=0.5 - index * 0.01,
            energy_tier="analytic_anchored",
            validity=validity,
            power_calibration_id=power_id,
            cost_scope="whole_model",
            system_calibration_id=system_id,
            head_service_calibration_id=head_id,
            output_head_location=EXTERNAL_BF16_HEAD,
            output_head_idealizations=(),
            whole_model_rankable=True,
            timing_calibrated=True,
            timing_evidence_id=timing.evidence_id,
            task_delta_lower_ci=(("gsm8k", 0.0), ("ifeval", 0.0)),
            ruler_scores=tuple(
                (length, 1.0, 1.0)
                for length in (4096, 8192, 16384, 32768)
            ),
        )

    reference = PublicationCandidate(
        evaluation_class="bf16_reference",
        profile_id="reference",
        candidate_id="accuracy-only",
        profile_kind=PROFILE_KIND_BF16_REFERENCE,
        perplexity=10.0,
        tpot_ms=None,
        energy_per_token_j=None,
        validity=StackValidity(software_valid=True),
        hardware_candidate=False,
    )
    values = (
        reference,
        candidate("uniform_i8", 1),
        candidate("uniform_i4", 2),
        candidate("pareto_candidate", 3),
    )
    return values, timing, head


def test_publication_admits_a_candidate_whose_rtl_is_unvalidated() -> None:
    for rtl_valid in (False, None):
        values, timing, head = _publication_fixture(
            validity=StackValidity(
                software_valid=True,
                compiler_valid=True,
                emulator_valid=True,
                rtl_valid=rtl_valid,
                dc_calibrated=None,
            ),
        )
        decision = select_final_deployment(
            values,
            calibration=None,
            timing_evidence=timing,
            head_service_evidence=head,
            output_head_location=EXTERNAL_BF16_HEAD,
        )
        assert decision.global_failures == (), decision.global_failures
        assert decision.selected is values[-1]
        failures = dict(decision.candidate_failures)
        assert failures["profile-3/candidate-3"] == ()

        selected = decision.selected
        # Admitted, and the gap is disclosed rather than hidden.
        assert selected.priced_evidence_complete is True
        assert selected.rests_on_unimplemented_rtl_path is True
        assert selected.all_stages_valid is False
        record = selected.to_dict()
        assert record["rests_on_unimplemented_rtl_path"] is True
        assert record["all_stages_valid"] is False
        assert record["recorded_validity"] == {
            "rtl_valid": rtl_valid,
            "dc_calibrated": None,
        }


def _stack(**overrides: Any) -> StackValidity:
    fields = {
        "software_valid": True,
        "compiler_valid": True,
        "emulator_valid": True,
        "rtl_valid": True,
        "dc_calibrated": True,
    }
    fields.update(overrides)
    return StackValidity(**fields)


def test_publication_still_rejects_blocking_stage_failures() -> None:
    """A measured failure, and unmeasured software validity, still exclude.

    Software validity is not geometry scoped, so its absence is a genuine gap.
    A measured ``False`` on the compiler or emulator stage is evidence about
    the point itself and excludes it regardless of the pricing model.
    """

    cases = (
        ("software_valid", False),
        ("software_valid", None),
        ("compiler_valid", False),
        ("emulator_valid", False),
    )
    for field, value in cases:
        values, timing, head = _publication_fixture(
            validity=_stack(**{field: value}),
        )
        decision = select_final_deployment(
            values,
            calibration=None,
            timing_evidence=timing,
            head_service_evidence=head,
            output_head_location=EXTERNAL_BF16_HEAD,
        )
        assert decision.selected is None
        failures = dict(decision.candidate_failures)
        assert "cross_stack_validity" in failures[
            "profile-3/candidate-3"
        ], (field, value)


def test_publication_admits_unmeasured_compiler_and_emulator_coverage() -> None:
    """Unmeasured individual validation is disclosed, never disqualifying.

    Compiler and emulator evidence is scoped to the geometry it was measured
    at, so a candidate priced away from that geometry carries ``None``.  It is
    eligible because the pricing model it was priced by is validated, and its
    record says plainly that it was not individually validated.
    """

    values, timing, head = _publication_fixture(
        validity=_stack(compiler_valid=None, emulator_valid=None),
    )
    decision = select_final_deployment(
        values,
        calibration=None,
        timing_evidence=timing,
        head_service_evidence=head,
        output_head_location=EXTERNAL_BF16_HEAD,
    )
    assert decision.global_failures == (), decision.global_failures
    selected = decision.selected
    assert selected is values[-1]
    assert dict(decision.candidate_failures)["profile-3/candidate-3"] == ()
    assert selected.priced_evidence_complete is True
    assert selected.individually_validated is False
    # The strict record never silently promotes an unmeasured stage.
    assert selected.all_stages_valid is False
    record = selected.to_dict()
    assert record["individually_validated"] is False
    assert record["individual_validation_stages"] == {
        "compiler": None,
        "emulator": None,
    }
    assert record["all_stages_valid"] is False
    assert record["admission_basis"] == ADMISSION_BASIS


def test_strict_subset_selector_returns_only_individually_validated() -> None:
    """Both framings are selectable from the same admitted population."""

    validated, timing, head = _publication_fixture(validity=_stack())
    unvalidated, _, _ = _publication_fixture(
        validity=_stack(compiler_valid=None, emulator_valid=None),
    )
    population = (*validated, *unvalidated)
    strict = individually_validated_candidates(population)
    assert strict
    assert len(strict) < len(population)
    assert all(value.individually_validated for value in strict)
    assert all(
        value.validity.compiler_valid is True
        and value.validity.emulator_valid is True
        for value in strict
    )
    # Every strict candidate is also admitted under the model-validated view.
    assert all(value.priced_evidence_complete for value in strict)


def test_dc_calibrated_energy_still_requires_measured_dc_validity() -> None:
    # RTL is never required, but a candidate that claims the DC-calibrated
    # energy tier must actually carry DC validity: there the tier is the claim.
    values, timing, head = _publication_fixture(
        validity=StackValidity(
            software_valid=True,
            compiler_valid=True,
            emulator_valid=True,
            rtl_valid=None,
            dc_calibrated=None,
        ),
    )
    claimed = replace(values[-1], energy_tier="dc_calibrated")
    assert claimed.priced_evidence_complete is False
    assert values[-1].priced_evidence_complete is True


def test_publication_rejects_the_unmeasured_local_head_boundary() -> None:
    values, timing, _ = _publication_fixture(
        validity=StackValidity(
            software_valid=True,
            compiler_valid=True,
            emulator_valid=True,
            rtl_valid=False,
            dc_calibrated=None,
        ),
    )
    decision = select_final_deployment(
        values,
        calibration=None,
        timing_evidence=timing,
        head_service_evidence=None,
        output_head_location=DECODE_BF16_HEAD,
    )
    assert decision.selected is None
    assert "output_head_boundary_unmeasured" in decision.global_failures
    record = decision.to_dict()
    assert record["output_head_location"] == DECODE_BF16_HEAD
    assert record["output_head_idealizations"] == [
        LOCAL_HEAD_COMPUTE_IDEALIZATION
    ]


def test_local_head_candidate_cannot_borrow_a_head_service_identity() -> None:
    with pytest.raises(
        ValueError,
        match="carries no head-service calibration",
    ):
        PublicationCandidate(
            evaluation_class="pareto_candidate",
            profile_id="profile-1",
            candidate_id="candidate-1",
            profile_kind="quantized",
            perplexity=10.0,
            tpot_ms=2.0,
            energy_per_token_j=0.5,
            energy_tier="analytic_anchored",
            validity=StackValidity(True, True, True),
            power_calibration_id="analytic-decode-energy-" + "a" * 64,
            cost_scope="whole_model",
            system_calibration_id="decode-local-head-system-" + "b" * 64,
            head_service_calibration_id="bf16-head-service-" + "c" * 64,
            output_head_location=DECODE_BF16_HEAD,
            output_head_idealizations=(LOCAL_HEAD_COMPUTE_IDEALIZATION,),
            whole_model_rankable=True,
            timing_calibrated=True,
            timing_evidence_id="timing-" + "e" * 64,
        )


# ---------------------------------------------------------------------------
# Publication admission: priced by a validated model, coverage disclosed
# ---------------------------------------------------------------------------

_ADMISSION_DROP = object()
"""Sentinel marking an admission input a case removes rather than falsifies."""


def _model_validated_row(**overrides: Any) -> dict[str, Any]:
    """A row priced by the validated pricing model at an unvalidated geometry.

    Compiler and emulator evidence is ``None`` because ``scope_stack_validity``
    confined the measured observation to the geometry it was taken at.  Every
    pricing-model identity the admission test requires is present.
    """

    row: dict[str, Any] = {
        "profile_id": "dqp-model-validated",
        "candidate_id": "hw-model-validated",
        "record_hash": "0" * 64,
        "deployment_valid": False,
        "error_code": None,
        "packedkv_selector_valid": False,
        "packedkv_selector_evidence": {
            "kind": "static_capability",
            "reason": "selector_is_wired_only_to_the_mxint_matrix_path",
            "evidence_id": "packedkv-selector-static-0",
        },
        "capability": {
            "issues": [
                {
                    "code": "rtl_batched_mxfp_unsupported",
                    "message": (
                        "The packed batched attention selector is implemented "
                        "only on the MXINT matrix path."
                    ),
                    "stages": ["rtl"],
                }
            ],
            "stage_support": {
                "software": True,
                "compiler": True,
                "emulator": True,
                "rtl": False,
                "dc": True,
            },
            "target": {
                "mlen": 512,
                "blen": 2,
                "hlen": 128,
                "batch": 4,
                "kv_heads": 2,
                "head_dim": 128,
                "block_size": 8,
                "selector_bits": 4,
                "packed_kv": True,
                "batched_attention": True,
            },
        },
        "validity": {
            "software_valid": True,
            "compiler_valid": None,
            "emulator_valid": None,
            "rtl_valid": False,
            "dc_calibrated": None,
        },
        "numerical_summary": {"state": "succeeded"},
        "legality": {"hardware_candidate": True},
        "metrics": {
            "timing_calibrated": True,
            "timing_evidence_id": "timing-" + "a" * 64,
            "timing_mode": "rtl_serialized",
            "memory_timing_calibrated": True,
            "bandwidth_calibration_id": "bandwidth-operating-point-" + "b" * 64,
            "area_source": "analytic_full_chip",
            "area_calibration_id": None,
            "area_scope": "decode_chip_only",
            "area_mm2": 100.0,
            "layout_id": "packed-kv-0",
            "runtime_feasible": True,
            "capacity": {"feasible": True},
            "runtime_capacity_evidence": {"max_runtime_batch": 4},
            "resource_budget": {"feasible": True},
            "output_head_boundary": _serialized_measured_head_boundary(4),
            "generated_tokens_per_step": 4,
            "whole_model": {
                "rankable": True,
                "strict_system_resource_boundary_valid": True,
                "publication_timing_tier": "stage_calibrated_analytic",
                "tpot_ms": 10.0,
                "tps": 100.0,
                "system_calibration_id": "decode-head-system-" + "d" * 64,
                "calibrated_energy": {
                    "energy_tier": "analytic_anchored",
                    "energy_id": "analytic-decode-energy-" + "e" * 64,
                    "total_j": 1.0,
                },
            },
        },
    }
    row.update(overrides)
    return row


def _individually_validated_row(**overrides: Any) -> dict[str, Any]:
    """The same row, additionally compiled and emulated at its own geometry."""

    row = _model_validated_row(**overrides)
    row["validity"] = {
        **row["validity"],
        "compiler_valid": True,
        "emulator_valid": True,
    }
    return row


def _mutate_row(row: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    target: Any = copied
    for key in path[:-1]:
        target = target[key]
    if value is _ADMISSION_DROP:
        target.pop(path[-1], None)
    else:
        target[path[-1]] = value
    return copied


def test_admission_admits_a_model_validated_row_without_individual_evidence() -> None:
    """The pricing model is validated; this point was not separately emulated."""

    admission = evaluate_publication_admission(_model_validated_row())
    assert admission.admitted is True
    assert admission.reason is None
    assert admission.individually_validated is False

    coverage = admission.coverage
    assert coverage is not None
    assert coverage.individually_validated is False
    assert coverage.validated_stages == ()
    assert coverage.unmeasured_stages == ("compiler", "emulator")
    assert coverage.failed_stages == ()
    # Nothing is promoted: the row is not claimed to sit at any evidence target.
    assert coverage.evidence_target is None
    assert coverage.runtime_target is not None
    assert coverage.required_stage_validity == {"software": True}

    record = admission.to_dict()
    assert record["individually_validated"] is False
    assert record["admission_basis"] == ADMISSION_BASIS
    assert record["individual_validation_coverage"]["unmeasured_stages"] == [
        "compiler",
        "emulator",
    ]
    # The models behind every cost are named on the row itself.
    pricing = record["pricing_model"]
    assert pricing["timing_evidence_id"].startswith("timing-")
    assert pricing["bandwidth_calibration_id"].startswith(
        "bandwidth-operating-point-"
    )
    assert pricing["energy_tier"] == "analytic_anchored"
    assert pricing["energy_id"].startswith("analytic-decode-energy-")
    assert pricing["area_source"] == "analytic_full_chip"
    assert pricing["system_calibration_id"].startswith("decode-head-system-")


def test_admission_rejects_a_forged_rankable_local_head_row() -> None:
    row = _model_validated_row()
    row["deployment_valid"] = True
    row["metrics"]["output_head_boundary"] = {
        "location": DECODE_BF16_HEAD,
        "service_mode": LOCAL_HEAD_MODE,
        "scope_idealizations": [LOCAL_HEAD_COMPUTE_IDEALIZATION],
        "status": local_head_boundary_status(),
        "estimate": None,
        "comparison_estimate": None,
    }

    admission = evaluate_publication_admission(row)

    assert admission.admitted is False
    assert admission.reason == "output_head_boundary"


def test_admission_rechecks_budget_on_a_strict_deployment_row() -> None:
    row = _model_validated_row()
    row["deployment_valid"] = True
    row["metrics"]["resource_budget"]["feasible"] = False

    admission = evaluate_publication_admission(row)

    assert admission.admitted is False
    assert admission.reason == "output_head_boundary"


def test_admission_marks_an_individually_validated_row_true() -> None:
    """A point compiled and emulated at its own geometry says so."""

    admission = evaluate_publication_admission(_individually_validated_row())
    assert admission.admitted is True
    assert admission.individually_validated is True

    coverage = admission.coverage
    assert coverage is not None
    assert coverage.validated_stages == ("compiler", "emulator")
    assert coverage.unmeasured_stages == ()
    # Measured compiler and emulator evidence survives scoping only at the
    # geometry it was taken at, so the row's own target is that target.
    assert coverage.evidence_target == coverage.runtime_target
    assert coverage.evidence_target is not None
    assert coverage.evidence_target["mlen"] == 512


def test_admission_requires_every_model_validation_input_individually() -> None:
    """Each pricing-model input is load-bearing on its own."""

    cases: tuple[tuple[tuple[str, ...], Any, str], ...] = (
        (("metrics", "timing_calibrated"), False, "timing_calibration"),
        (
            ("metrics", "timing_evidence_id"),
            _ADMISSION_DROP,
            "timing_evidence_identity",
        ),
        (("metrics", "timing_evidence_id"), "", "timing_evidence_identity"),
        (
            ("metrics", "whole_model", "publication_timing_tier"),
            "uncalibrated",
            "publication_timing_tier",
        ),
        (
            ("metrics", "memory_timing_calibrated"),
            False,
            "memory_timing_calibration",
        ),
        (
            ("metrics", "bandwidth_calibration_id"),
            _ADMISSION_DROP,
            "bandwidth_calibration_identity",
        ),
        (
            ("metrics", "whole_model", "calibrated_energy", "energy_tier"),
            "modelled",
            "energy_tier",
        ),
        (
            ("metrics", "whole_model", "calibrated_energy", "energy_id"),
            _ADMISSION_DROP,
            "energy_identity",
        ),
        (
            ("metrics", "area_source"),
            _ADMISSION_DROP,
            "area_model_identity",
        ),
        (
            ("metrics", "area_source"),
            "dc_calibrated",
            "area_calibration_identity",
        ),
        (
            ("metrics", "output_head_boundary", "estimate"),
            _ADMISSION_DROP,
            "output_head_boundary",
        ),
        (
            ("metrics", "whole_model", "system_calibration_id"),
            _ADMISSION_DROP,
            "system_calibration_identity",
        ),
        (("metrics", "layout_id"), "", "layout_identity"),
        (("metrics", "capacity", "feasible"), False, "capacity_feasibility"),
        (("metrics", "runtime_feasible"), False, "runtime_feasibility"),
        (
            ("metrics", "runtime_capacity_evidence"),
            _ADMISSION_DROP,
            "capacity_evidence",
        ),
        (
            ("metrics", "resource_budget", "feasible"),
            False,
            "resource_budget",
        ),
        (("numerical_summary", "state"), "failed", "numerical_state"),
        (
            ("legality", "hardware_candidate"),
            False,
            "static_hardware_legality",
        ),
        (("error_code",), "capacity_overflow", "error_code"),
        # Software validity is not geometry scoped, so it stays required.
        (("validity", "software_valid"), None, "blocking_stage_validity"),
        # A measured failure is evidence about the point, not a coverage gap.
        (("validity", "compiler_valid"), False, "blocking_stage_validity"),
        (("validity", "emulator_valid"), False, "blocking_stage_validity"),
    )
    for path, value, expected_reason in cases:
        row = _mutate_row(_model_validated_row(), path, value)
        admission = evaluate_publication_admission(row)
        assert admission.admitted is False, (path, value)
        assert admission.reason == expected_reason, (path, value)
        # A refusal still discloses coverage rather than going silent.
        assert admission.coverage is not None


def test_admission_strict_view_selects_only_individually_validated_rows() -> None:
    """Both framings come from one predicate over one admitted population."""

    priced = _model_validated_row()
    validated = _individually_validated_row(
        candidate_id="hw-individually-validated",
    )
    population = (priced, validated)

    admitted = select_admitted_rows(population)
    assert len(admitted) == 2

    strict = select_admitted_rows(
        population,
        require_individual_validation=True,
    )
    assert len(strict) == 1
    assert strict[0]["candidate_id"] == "hw-individually-validated"

    # The strict view refuses for the coverage reason, never by pretending the
    # pricing evidence was missing.
    refusal = evaluate_publication_admission(
        priced,
        require_individual_validation=True,
    )
    assert refusal.admitted is False
    assert refusal.reason == "individual_validation"
    assert refusal.coverage is not None
    assert refusal.coverage.unmeasured_stages == ("compiler", "emulator")

    # A row refused on the pricing evidence reports that reason even in the
    # strict view; missing coverage never masks a missing model input.
    broken = _mutate_row(priced, ("metrics", "timing_calibrated"), False)
    assert (
        evaluate_publication_admission(
            broken,
            require_individual_validation=True,
        ).reason
        == "timing_calibration"
    )


def test_downstream_promotion_accepts_a_newly_admitted_row() -> None:
    """The promotion consumer prices the row and carries its coverage."""

    profile = DecodePrecisionProfile(
        kind=PROFILE_KIND_QUANTIZED,
        weight_format="MXINT4",
        activation_format="MXINT8",
        key_format="MXINT4",
        value_format="MXINT4",
        vector_format="FP_E4M7",
        block_size=MX_BLOCK_SIZE,
    )
    point = _hardware_point(
        _model_validated_row(),
        profile=profile,
        mean_nll=1.0,
    )
    assert point is not None
    assert point.individually_validated is False
    assert point.whole_model_rankable is True
    record = point.to_dict()
    assert record["individually_validated"] is False
    assert record["individual_validation_coverage"]["unmeasured_stages"] == [
        "compiler",
        "emulator",
    ]
    assert record["pricing_model"]["timing_evidence_id"].startswith("timing-")

    validated_point = _hardware_point(
        _individually_validated_row(),
        profile=profile,
        mean_nll=1.0,
    )
    assert validated_point is not None
    assert validated_point.individually_validated is True
    assert individually_validated_points((point, validated_point)) == (
        validated_point,
    )


def test_stage_split_partitions_the_blocking_stages() -> None:
    """The two stage sets are an exhaustive, disjoint split, by construction."""

    assert set(INDIVIDUAL_VALIDATION_STAGES) | set(
        MODEL_REQUIRED_VALIDITY_STAGES
    ) == set(PRICING_BLOCKING_STAGES)
    assert not set(INDIVIDUAL_VALIDATION_STAGES) & set(
        MODEL_REQUIRED_VALIDITY_STAGES
    )
    # The split follows scope_stack_validity: exactly the stages it demotes
    # from True to None off-geometry are the individually validated ones.
    observed = StackValidity(
        software_valid=True,
        compiler_valid=True,
        emulator_valid=True,
        rtl_valid=True,
        dc_calibrated=True,
    )
    scoped = scope_stack_validity(
        observed,
        evidence_target=PackedKVRuntimeTarget(),
        runtime_target=PackedKVRuntimeTarget(mlen=2048),
    )
    demoted = {
        stage
        for stage in PRICING_BLOCKING_STAGES
        if getattr(scoped, f"{stage}_valid") is None
    }
    assert demoted == set(INDIVIDUAL_VALIDATION_STAGES)
