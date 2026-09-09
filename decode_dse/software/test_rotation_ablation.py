"""Fail-closed contracts for the attention-only Qwen3-MoE rotation ablation."""

from __future__ import annotations

from pathlib import Path

import pytest

from decode_dse.legality import StackValidity
from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software import rotation_ablation
from decode_dse.software.refinement_runner import (
    ATTENTION_ONLY_ROTATION_MATMUL_TYPES,
    FUSED_EXPERT_ROTATION_EXCLUSIONS,
    RefinementMergedResults,
    rotation_policy,
    validate_attention_only_rotation_decision,
)
from decode_dse.software.refinement_schedule import (
    DecodeRefinementProfile,
    DoomedGateDecision,
    DoomedGatePolicy,
    RefinementSchedule,
    RefinementScheduleEntry,
    build_selective_rotation_schedule,
    write_refinement_schedule,
)
from decode_dse.software.sweep_plan import load_immutable_json, write_immutable_json


def _sources() -> dict[str, DecodePrecisionProfile]:
    return {
        "uniform_mxint8": DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT8", "MXINT8", "FP_E3M2"
        ),
        "uniform_mxint4": DecodePrecisionProfile.quantized(
            "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
        ),
        "mxint_kv2": DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT8", "MXINT2", "FP_E3M2"
        ),
        "accuracy_constrained_deployment": DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT4", "MXINT4", "FP_E2M3"
        ),
    }


def _base_schedule() -> RefinementSchedule:
    profiles = tuple(
        DecodeRefinementProfile(source, source.kv_format, source.kv_format)
        for source in _sources().values()
    )
    gate = DoomedGateDecision("scheduled", "fixture", None, None)
    validity = StackValidity(True, True, True, True, True)
    return RefinementSchedule(
        entries=tuple(
            RefinementScheduleEntry(index, profile, gate, validity)
            for index, profile in enumerate(profiles)
        ),
        source_profile_ids=tuple(
            sorted(profile.source_profile.profile_id for profile in profiles)
        ),
        reference_mean_nll=1.0,
        policy=DoomedGatePolicy(),
    )


def _merged(schedule: RefinementSchedule, *, failed: str | None = None):
    rows = tuple(
        {
            "profile_id": entry.profile_id,
            "record_hash": f"{index + 1:064x}",
            "state": "failed" if entry.profile_id == failed else "succeeded",
            "result": {"mean_token_nll": 1.0 + index / 100.0},
        }
        for index, entry in enumerate(schedule.entries)
    )
    return RefinementMergedResults(
        receipt={
            "content_hash": "a" * 64,
            "sample_bundle_hash": "b" * 64,
        },
        results_path=Path("/tmp/fixture-results.jsonl"),
        results_sha256="c" * 64,
        terminal_rows=rows,
    )


def _roles(schedule: RefinementSchedule) -> dict[str, str]:
    sources = _sources()
    return {role: source.profile_id for role, source in sources.items()}


def _rotation_schedule():
    base = _base_schedule()
    chosen = tuple(entry.profile_id for entry in base.entries)
    uniform_source = _sources()["uniform_mxint8"].profile_id
    uniform = next(
        entry.profile_id
        for entry in base.entries
        if entry.profile.source_profile.profile_id == uniform_source
    )
    return base, build_selective_rotation_schedule(
        base,
        best_supported_profile_ids=chosen,
        uniform_i8_profile_id=uniform,
    )


def test_rotation_policy_pins_only_seven_attention_operations() -> None:
    policy = rotation_policy()
    assert policy["method_id"] == rotation_ablation.ATTENTION_ONLY_ROTATION_METHOD
    assert policy["matmul_types"] == list(ATTENTION_ONLY_ROTATION_MATMUL_TYPES)
    assert policy["fused_expert_matmul_types"]["included"] == []
    assert policy["fused_expert_matmul_types"]["excluded"] == list(
        FUSED_EXPERT_ROTATION_EXCLUSIONS
    )


def test_rotation_decision_rejects_fused_expert_winner() -> None:
    decision = {
        "winners": ["q_proj"],
        "matmul_types_searched": list(ATTENTION_ONLY_ROTATION_MATMUL_TYPES),
        "rotation_scope": {
            "architecture": "qwen3_moe_fused",
            "eligible_matmul_types": list(ATTENTION_ONLY_ROTATION_MATMUL_TYPES),
            "excluded_matmul_types": list(FUSED_EXPERT_ROTATION_EXCLUSIONS),
            "excluded_reason": "fused expert tensors have no rotation lowerer",
        },
        "baseline_ppl": 2.0,
        "final_ppl": 1.9,
    }
    assert validate_attention_only_rotation_decision(decision)["winners"] == [
        "q_proj"
    ]
    decision["winners"] = ["gate_proj"]
    with pytest.raises(ValueError, match="attention-only"):
        validate_attention_only_rotation_decision(decision)


def test_measured_sources_produce_four_symmetric_unmeasured_rotation_rows() -> None:
    base = _base_schedule()
    selected = rotation_ablation.select_measured_rotation_sources(
        base, _merged(base), _roles(base)
    )
    uniform = next(
        item["base_refinement_profile_id"]
        for item in selected
        if item["source_role"] == "uniform_mxint8"
    )
    schedule = build_selective_rotation_schedule(
        base,
        best_supported_profile_ids=tuple(
            item["base_refinement_profile_id"] for item in selected
        ),
        uniform_i8_profile_id=uniform,
    )
    assert len(schedule.entries) == 4
    assert all(entry.profile.weight_method == "rotation" for entry in schedule.entries)
    assert all(not entry.profile.split_kv for entry in schedule.entries)
    assert all(
        value is None
        for entry in schedule.entries
        for value in entry.validity.to_dict().values()
    )


def test_rotation_source_selection_requires_terminal_success() -> None:
    base = _base_schedule()
    failed = base.entries[0].profile_id
    with pytest.raises(ValueError, match="no successful canonical GPTQ"):
        rotation_ablation.select_measured_rotation_sources(
            base, _merged(base, failed=failed), _roles(base)
        )


def test_rotation_builder_rejects_split_kv_parent() -> None:
    base = _base_schedule()
    first = base.entries[0]
    split = DecodeRefinementProfile(
        first.profile.source_profile,
        first.profile.key_format,
        "MXINT4",
    )
    schedule = RefinementSchedule(
        entries=(RefinementScheduleEntry(0, split, first.gate, first.validity),)
        + tuple(
            RefinementScheduleEntry(
                index,
                entry.profile,
                entry.gate,
                entry.validity,
            )
            for index, entry in enumerate(base.entries[1:], start=1)
        ),
        source_profile_ids=base.source_profile_ids,
        reference_mean_nll=base.reference_mean_nll,
        policy=base.policy,
    )
    uniform = schedule.entries[0].profile_id
    with pytest.raises(ValueError, match="equal-K/V"):
        build_selective_rotation_schedule(
            schedule,
            best_supported_profile_ids=tuple(
                entry.profile_id for entry in schedule.entries
            ),
            uniform_i8_profile_id=uniform,
        )


def test_target_config_declares_nonselectable_attention_ablation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = rotation_ablation._load_config(
        root / "configs" / "qwen3_30b_a3b_thinking_2507.json"
    )
    rotation_ablation._validate_target_config(config)


def test_materializer_binds_successful_base_rows_without_inheriting_validity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _base_schedule()
    base_schedule = write_refinement_schedule(tmp_path / "base.json", base)
    base_merge = tmp_path / "base-merge.json"
    base_merge.write_text("fixture\n", encoding="utf-8")
    source = write_immutable_json(
        tmp_path / "source.json",
        {
            "schema_version": "decode-refinement-source-selection",
            "schedule_hash": base.canonical_hash,
            "hardware_study_sha256": ["f" * 64],
            "source_selection": {"source_roles": _roles(base)},
        },
    )
    monkeypatch.setattr(
        rotation_ablation,
        "load_refinement_merged_results",
        lambda *args, **kwargs: _merged(base),
    )
    monkeypatch.setattr(
        rotation_ablation,
        "_hardware_evidence",
        lambda *args, **kwargs: (
            {"path": "/fixture", "sha256": "f" * 64, "matched_source_rows": []},
        ),
    )
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "qwen3_30b_a3b_thinking_2507.json"
    )
    schedule = tmp_path / "rotation.json"
    plan = tmp_path / "plan.json"
    value = rotation_ablation.materialize_rotation_plan(
        config_path=config,
        base_schedule_path=base_schedule,
        base_merge_path=base_merge,
        source_selection_path=source,
        hardware_paths=(tmp_path / "hardware.jsonl",),
        schedule_path=schedule,
        plan_path=plan,
    )
    stored = load_immutable_json(plan)
    assert stored["rotation_schedule"]["schedule_hash"] == value[
        "rotation_schedule"
    ]["schedule_hash"]
    assert stored["strict_stack_status"]["selection_eligible"] is False
    assert len(stored["source_ancestry"]) == 4


def test_cost_contract_requires_exact_terminal_and_profile_coverage() -> None:
    base, schedule = _rotation_schedule()
    merged = _merged(schedule)
    ancestry = []
    for base_entry, rotation_entry, row in zip(
        base.entries, schedule.entries, _merged(base).terminal_rows
    ):
        ancestry.append(
            {
                "source_profile_id": base_entry.profile.source_profile.profile_id,
                "rotation_profile_id": rotation_entry.profile_id,
                "base_refinement_record_hash": row["record_hash"],
            }
        )
    plan = {
        "source_ancestry": ancestry,
    }
    rows = []
    for item, row in zip(ancestry, merged.terminal_rows):
        rows.append(
            {
                **item,
                "rotation_refinement_record_hash": row["record_hash"],
                "device_name": "NVIDIA B200",
                "device_uuid": "GPU-fixture",
                "timing_source": "cuda_event_synchronized",
                "power_source": "nvml_total_energy_counter",
                "decode_batch_size": 4,
                "prompt_tokens": 512,
                "decode_steps": 128,
                "warmup_decode_steps": 32,
                "paired_repetitions": 10,
                "measured_decode_tokens": 5120,
                "baseline_tpot_ms": 1.0,
                "rotation_tpot_ms": 1.1,
                "baseline_tokens_per_second": 4000.0,
                "rotation_tokens_per_second": 3600.0,
                "baseline_average_power_w": 500.0,
                "rotation_average_power_w": 510.0,
                "baseline_energy_per_token_j": 0.125,
                "rotation_energy_per_token_j": 0.142,
            }
        )
    raw = {
        "schema_version": rotation_ablation.COST_RAW_SCHEMA,
        "model_name": rotation_ablation.TARGET_MODEL,
        "model_revision": rotation_ablation.TARGET_REVISION,
        "method_id": rotation_ablation.ATTENTION_ONLY_ROTATION_METHOD,
        "rotation_schedule_hash": schedule.canonical_hash,
        "rotation_merge_content_hash": merged.receipt["content_hash"],
        "sample_bundle_hash": merged.receipt["sample_bundle_hash"],
        "measurement_harness_sha256": "d" * 64,
        "runtime_environment_fingerprint": "e" * 64,
        "rows": rows,
    }
    validated = rotation_ablation._validate_cost_raw(
        raw, plan=plan, schedule=schedule, merged=merged
    )
    assert len(validated) == 4
    raw["rows"] = rows[:-1]
    with pytest.raises(ValueError, match="coverage"):
        rotation_ablation._validate_cost_raw(
            raw, plan=plan, schedule=schedule, merged=merged
        )
