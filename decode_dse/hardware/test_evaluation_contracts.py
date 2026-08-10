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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from decode_dse.hardware import evaluation
from decode_dse.hardware.design_space import (
    COMPILER_TRACE_EXECUTION_MODE,
    COMPILER_TRACE_TIMING_SET_SCHEMA,
    FULL_MODEL_DECODE_SCOPE,
    LEGACY_AGGREGATE_BANDWIDTH_MODE,
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
    load_hardware_artifact,
)
from decode_dse.hardware.lm_head_service import (
    HEAD_SERVICE_MODE,
    composite_system_calibration_id,
    load_bf16_head_service_artifact,
)
from decode_dse.hardware.packedkv_claims import AttentionTopology
from decode_dse.hardware.power_bridge import analytic_energy_from_simulator
from decode_dse.hardware.selection import (
    PublicationCandidate,
    select_refinement_sources,
    select_final_deployment,
)
from decode_dse.legality import (
    PackedKVRuntimeTarget,
    StackValidity,
    evaluate_stack_capability,
    scope_stack_validity,
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
        for tp in (2, 4, 8)
    )
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
    for tp in (1, 2, 4, 8):
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
    assert areas[4] == areas[8]


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
    expected = {
        "qwen3_32b": 471_072,
        "llama3_1_8b": 616_032,
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
            "output_head_boundary": {
                "estimate": {"calibration_id": "head-calibration"}
            },
            "capacity": {"feasible": True},
            "runtime_capacity_evidence": {"max_runtime_batch": 256},
            "timing_calibrated": True,
            "runtime_feasible": True,
            "area_mm2": area_mm2,
            "system_area_mm2": area_mm2,
            "resource_budget": {
                "aggregate_area_mm2": area_mm2,
                "aggregate_area_limit_mm2": 1000.0,
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
        "head_service_id": "endpoint",
        "process_corner": "measured_silicon",
        "measured_at_utc": "2026-08-03T12:00:00Z",
        "measurement_resolution": {
            "meter_methods": {
                "driver": "nvml_total_energy_counter",
                "endpoint": "nvml_total_energy_counter",
            },
            "power_plausibility_ceiling_w": 2400.0,
            "min_counter_delta_j": 0.1,
            "idle_power_w": 270.0,
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
