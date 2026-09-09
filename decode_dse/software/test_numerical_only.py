from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from decode_dse.manifest import write_manifest
from decode_dse.software import numerical_only
from decode_dse.software.sweep import _build_manifest, create_workspace
from decode_dse.software.sweep_plan import (
    GPUBaselinePlan,
    PromptManifest,
    PromptRecord,
    build_run_plan,
    make_stage_manifest,
    write_immutable_json,
)
from decode_dse.software.sweep_runner import ResultShardStore


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "decode_dse/configs/qwen3_30b_a3b_thinking_2507.json"


def _target_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _compiler_blocker() -> dict:
    return {
        "compiler_trace_preflight_feasible": False,
        "compiler_trace_preflight_blockers": [
            "unsupported_native_lowering:"
            "mixture_of_experts_trace_evidence_not_bound:4320"
        ],
    }


def test_contract_is_exactly_target_bound_and_non_publication() -> None:
    numerical_only._require_target(_target_config())
    changed = _target_config()
    changed["model_revision"] = "0" * 40
    with pytest.raises(ValueError, match="model_revision"):
        numerical_only._require_target(changed)

    contract = numerical_only.build_contract(
        config_sha256="1" * 64,
        manifest_hash="2" * 64,
        run_plan_hash="3" * 64,
        prompt_manifest_hash="4" * 64,
        provenance_sha256="5" * 64,
        quantizer_provenance_hash="6" * 64,
        compiler_preflight=_compiler_blocker(),
    )
    assert contract["target"]["model_name"] == numerical_only.MODEL_NAME
    assert contract["target"]["model_revision"] == numerical_only.MODEL_REVISION
    assert contract["classification"]["publication_rankable"] is False
    assert contract["classification"]["hardware_rankable"] is False
    assert contract["classification"]["selection_eligible"] is False
    assert contract["execution"]["allowed_stages"] == [
        "preflight",
        "numerical-screen",
    ]
    assert contract["execution"]["required_shards"] == 4
    assert contract["strict_pipeline"]["bypassed"] is False
    assert contract["strict_pipeline"]["normal_pipeline_remains_fail_closed"] is True

    feasible = dict(_compiler_blocker())
    feasible["compiler_trace_preflight_feasible"] = True
    with pytest.raises(ValueError, match="only while the strict compiler trace is blocked"):
        numerical_only.build_contract(
            config_sha256="1" * 64,
            manifest_hash="2" * 64,
            run_plan_hash="3" * 64,
            prompt_manifest_hash="4" * 64,
            provenance_sha256="5" * 64,
            quantizer_provenance_hash="6" * 64,
            compiler_preflight=feasible,
        )


def test_omission_receipt_covers_every_disallowed_claim() -> None:
    receipt = numerical_only.build_omission_receipt(
        contract_hash="a" * 64,
        bindings={"manifest_hash": "b" * 64},
        compiler_blockers=_compiler_blocker()["compiler_trace_preflight_blockers"],
    )
    stages = {item["stage"] for item in receipt["omissions"]}
    assert stages == {
        "compiler_full_model_moe_timing",
        "transaction_emulator_validation",
        "hardware_validation",
        "analytic_hardware_search",
        "latency_throughput_power_energy_area",
        "publication_selection",
    }
    assert all(item["evidence_present"] is False for item in receipt["omissions"])
    assert all(item["artifact"] is None for item in receipt["omissions"])
    assert receipt["classification"]["may_claim_latency"] is False
    assert receipt["classification"]["may_claim_numerical_nll_perplexity"] is True
    assert receipt["classification"]["may_claim_task_accuracy"] is False


def _write_terminal_stage(
    tmp_path: Path,
    *,
    manifest,
    plan,
    contract: dict,
    retain_failure: bool = True,
    runtime_seconds: float = 0.1,
    weight_bank_seconds: float = 2.0,
) -> str | None:
    failed_id = plan.preflight_profile_ids[7] if retain_failure else None
    write_immutable_json(
        tmp_path / "numerical_only_omissions.json",
        {"schema_version": "test-omission"},
    )
    write_immutable_json(
        tmp_path / "admission_preparation.json",
        {"schema_version": "test-admission"},
    )
    omission = numerical_only.load_immutable_json(
        tmp_path / "numerical_only_omissions.json"
    )
    admission_sha256 = numerical_only._sha256_file(
        tmp_path / "admission_preparation.json"
    )
    contract_hash = contract["content_hash"]
    full_stage_manifest = make_stage_manifest(manifest, plan.preflight_profile_ids)
    write_immutable_json(
        tmp_path / "preflight/sharding.json",
        {
            "schema_version": "decode-numerical-only-sharding/v1",
            "stage": "preflight",
            "contract_hash": contract_hash,
            "master_manifest_hash": manifest.canonical_hash,
            "full_stage_manifest_hash": full_stage_manifest.canonical_hash,
            "run_plan_hash": plan.canonical_hash,
            "shard_count": numerical_only.REQUIRED_SHARDS,
            "algorithm": numerical_only.PARTITION_ALGORITHM,
            "publication_rankable": False,
        },
    )
    for index in range(numerical_only.REQUIRED_SHARDS):
        ids = numerical_only.partition_stage_profile_ids(
            manifest,
            plan.preflight_profile_ids,
            shard_index=index,
            shard_count=numerical_only.REQUIRED_SHARDS,
        )
        partition = make_stage_manifest(manifest, ids)
        root = tmp_path / "preflight" / f"part-{index:04d}-of-0004"
        write_manifest(root / "manifest.json", partition)
        write_immutable_json(
            root / "invocation.json",
            {
                "schema_version": numerical_only.INVOCATION_SCHEMA,
                "contract_hash": contract_hash,
                "omission_receipt_hash": omission["content_hash"],
                "fidelity_gate_hash": None,
                "stage": "preflight",
                "master_manifest_hash": manifest.canonical_hash,
                "stage_manifest_hash": partition.canonical_hash,
                "run_plan_hash": plan.canonical_hash,
                "prompt_manifest_hash": contract["bindings"]["prompt_manifest_hash"],
                "config_sha256": contract["bindings"]["config_sha256"],
                "provenance_sha256": contract["bindings"]["provenance_sha256"],
                "admission_preparation_sha256": admission_sha256,
                "shard_index": index,
                "shard_count": numerical_only.REQUIRED_SHARDS,
                "profile_count": len(partition.entries),
                "sample_contract": numerical_only._stage_contract(
                    plan, "preflight"
                ).to_dict(),
                "decode_microbatch_size": plan.numerical_screen_microbatch_size,
                "evidence_class": numerical_only.EVIDENCE_CLASS,
                "publication_rankable": False,
                "hardware_rankable": False,
                "selection_eligible": False,
                "failed_rows_retained": True,
            },
        )
        completed = root / "completed"
        completed.mkdir(parents=True)
        store = ResultShardStore(root, partition)
        for entry in partition.entries:
            state = "failed" if entry.profile_id == failed_id else "succeeded"
            attempt = 3 if entry.profile_id == failed_id else 1
            pointer = store.append(
                entry,
                attempt=attempt,
                state=state,
                validity=entry.validity,
                result={
                    "test_metric": 1.0,
                    "sample_contract": numerical_only._stage_contract(
                        plan,
                        "preflight",
                    ).to_dict(),
                    "gpu_memory": {
                        "microbatch_size": 16,
                        "peak_allocated_bytes": 60 << 30,
                        "peak_reserved_bytes": 64 << 30,
                        "total_device_bytes": 192 << 30,
                        "peak_reserved_fraction": 1.0 / 3.0,
                    },
                    "runtime_environment": {
                        "device_uuid": f"GPU-test-{index}",
                    },
                    "resource_observation": {
                        "schema_version": (
                            "decode-profile-resource-observation/v1"
                        ),
                        "host_before_evaluation": {
                            "host_mem_available_bytes": 512 << 30,
                            "process_rss_bytes": 64 << 30,
                            "process_peak_rss_bytes": 72 << 30,
                        },
                        "host_after_evaluation": {
                            "host_mem_available_bytes": 500 << 30,
                            "process_rss_bytes": 65 << 30,
                            "process_peak_rss_bytes": 73 << 30,
                        },
                        "decode_cache_lru": {
                            "configured_capacity_bytes": 24 << 30,
                            "resident_bytes_after_evaluation": 1 << 30,
                            "resident_entries_after_evaluation": 16,
                            "peak_resident_bytes_this_weight_bank": 2 << 30,
                            "peak_entries_this_weight_bank": 32,
                        },
                    },
                    "weight_bank": {
                        "weight_format": entry.profile.weight_format,
                        "build_seconds": weight_bank_seconds,
                        "resource_observation": {
                            "schema_version": (
                                "decode-weight-bank-resource-observation/v1"
                            ),
                            "build_serialized_across_workers": True,
                            "host_before_build": {
                                "host_mem_available_bytes": 600 << 30,
                                "process_rss_bytes": 2 << 30,
                                "process_peak_rss_bytes": 3 << 30,
                            },
                            "host_after_build": {
                                "host_mem_available_bytes": 520 << 30,
                                "process_rss_bytes": 62 << 30,
                                "process_peak_rss_bytes": 70 << 30,
                            },
                            "gpu": {
                                "total_device_bytes": 192 << 30,
                                "allocated_bytes_after_build": 58 << 30,
                                "reserved_bytes_after_build": 60 << 30,
                                "peak_allocated_bytes_during_build": 62 << 30,
                                "peak_reserved_bytes_during_build": 64 << 30,
                            },
                            "fused_expert_module_count": 48,
                            "fused_expert_parameter_bytes": 48 << 30,
                            "timing": {
                                "pre_lock_setup_seconds": (
                                    weight_bank_seconds * 0.05
                                ),
                                "lock_wait_seconds": weight_bank_seconds * 0.10,
                                "serialized_build_seconds": (
                                    weight_bank_seconds * 0.75
                                ),
                                "post_lock_validation_seconds": (
                                    weight_bank_seconds * 0.10
                                ),
                                "outer_open_seconds": weight_bank_seconds,
                            },
                        },
                    },
                },
                error=(RuntimeError("retained failure") if state == "failed" else None),
                runtime_seconds=runtime_seconds,
            )
            body = {
                "schema_version": "decode-sweep-completion",
                "manifest_hash": partition.canonical_hash,
                "profile_id": entry.profile_id,
                "ordinal": entry.ordinal,
                "state": state,
                "attempt": attempt,
                "result_path": pointer.journal_path,
            }
            marker = body | {"marker_hash": numerical_only._canonical_hash(body)}
            (completed / f"{entry.profile_id}.json").write_text(
                json.dumps(marker, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        bank_count = len({entry.profile.weight_format for entry in partition.entries})
        (root / "progress.json").write_text(
            json.dumps(
                {
                    "schema_version": "decode-sweep-progress",
                    "event": "profile-completed",
                    "stage": "preflight",
                    "work_class": "numerical",
                    "completed_profiles": len(partition.entries),
                    "succeeded_profiles": len(partition.entries)
                    - int(failed_id in ids),
                    "failed_terminal_profiles": int(failed_id in ids),
                    "total_profiles": len(partition.entries),
                    "remaining_profiles": 0,
                    "attempts_observed_this_invocation": len(partition.entries),
                    "unique_weight_banks_opened": bank_count,
                    "unique_weight_banks_required_this_invocation": bank_count,
                    "unique_weight_banks_remaining": 0,
                    "last_trial_seconds": runtime_seconds,
                    "mean_trial_seconds": runtime_seconds,
                    "last_weight_bank_open_seconds": None,
                    "mean_weight_bank_open_seconds": weight_bank_seconds,
                    "estimated_remaining_seconds": 0.0,
                    "estimated_completion_utc": "2026-07-25T00:00:00Z",
                    "updated_at": "2026-07-25T00:00:00Z",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return failed_id


def test_four_way_summary_is_deterministic_and_retains_failed_row(tmp_path: Path) -> None:
    config = _target_config()
    manifest = _build_manifest(config, ROOT)
    plan = build_run_plan(
        manifest,
        device_labels=("B200",),
        numerical_screen_workers=4,
        hardware_validation_workers=4,
        numerical_screen_microbatch_size=16,
        hardware_validation_microbatch_size=8,
        gpu_baseline=GPUBaselinePlan.from_config(config["gpu_baseline"]),
    )
    first_partition = tuple(
        numerical_only.partition_stage_profile_ids(
            manifest,
            plan.numerical_screen_profile_ids,
            shard_index=index,
            shard_count=4,
        )
        for index in range(4)
    )
    second_partition = tuple(
        numerical_only.partition_stage_profile_ids(
            manifest,
            plan.numerical_screen_profile_ids,
            shard_index=index,
            shard_count=4,
        )
        for index in range(4)
    )
    assert first_partition == second_partition
    assert set().union(*(set(part) for part in first_partition)) == set(
        plan.numerical_screen_profile_ids
    )
    assert sum(map(len, first_partition)) == len(plan.numerical_screen_profile_ids)

    contract = {
        "content_hash": "c" * 64,
        "bindings": {
            "prompt_manifest_hash": "d" * 64,
            "config_sha256": "e" * 64,
            "provenance_sha256": "f" * 64,
        },
    }
    failed_id = _write_terminal_stage(
        tmp_path,
        manifest=manifest,
        plan=plan,
        contract=contract,
    )
    summary = numerical_only.summarize_stage(
        workspace=tmp_path,
        stage="preflight",
        manifest=manifest,
        plan=plan,
        contract=contract,
    )
    assert summary["complete"] is True
    assert summary["passed"] is False
    assert summary["failed_profiles"] == 1
    retained = [
        row for row in summary["terminal_records"] if row["profile_id"] == failed_id
    ]
    assert len(retained) == 1
    assert retained[0]["state"] == "failed"
    assert retained[0]["attempt"] == 3
    assert retained[0]["result_path"].endswith(retained[0]["result_record_hash"])


def _forecast_inputs(tmp_path: Path, *, plan=None):
    config = _target_config()
    manifest = _build_manifest(config, ROOT)
    if plan is None:
        plan = build_run_plan(
            manifest,
            device_labels=("B200",),
            numerical_screen_workers=4,
            hardware_validation_workers=4,
            numerical_screen_microbatch_size=16,
            hardware_validation_microbatch_size=8,
            gpu_baseline=GPUBaselinePlan.from_config(config["gpu_baseline"]),
        )
    contract = {
        "content_hash": "c" * 64,
        "bindings": {
            "prompt_manifest_hash": "d" * 64,
            "config_sha256": "e" * 64,
            "provenance_sha256": "f" * 64,
        },
    }
    _write_terminal_stage(
        tmp_path,
        manifest=manifest,
        plan=plan,
        contract=contract,
        retain_failure=False,
    )
    summary = numerical_only.summarize_stage(
        workspace=tmp_path,
        stage="preflight",
        manifest=manifest,
        plan=plan,
        contract=contract,
    )
    omission = numerical_only.load_immutable_json(
        tmp_path / "numerical_only_omissions.json"
    )
    gate = {"content_hash": "a" * 64, "pilot": summary, "passed": True}
    return manifest, plan, contract, omission, gate


def test_exact_pilot_rows_are_reused_without_reducing_screen_manifest(
    tmp_path: Path,
) -> None:
    manifest, plan, _, _, gate = _forecast_inputs(tmp_path)
    screen_ids = numerical_only.partition_stage_profile_ids(
        manifest,
        plan.numerical_screen_profile_ids,
        shard_index=0,
        shard_count=4,
    )
    stage_manifest = make_stage_manifest(manifest, screen_ids)
    outcomes = numerical_only._pilot_reuse_outcomes(
        workspace=tmp_path,
        manifest=manifest,
        plan=plan,
        fidelity_gate=gate,
        shard_index=0,
        stage_manifest=stage_manifest,
    )

    expected = set(screen_ids).intersection(plan.preflight_profile_ids)
    assert set(outcomes) == expected
    assert len(outcomes) == 10
    assert len(stage_manifest.entries) == 897
    assert any(
        entry.profile.weight_format == "BF16" and entry.profile_id in outcomes
        for entry in stage_manifest.entries
    )
    for profile_id, outcome in outcomes.items():
        reuse = outcome.metrics["evaluation_reuse"]
        assert reuse["schema_version"] == numerical_only.PILOT_REUSE_SCHEMA
        assert reuse["profile_id"] == profile_id
        assert reuse["fidelity_gate_hash"] == gate["content_hash"]
        assert reuse["numerical_metrics_reused_without_recomputation"] is True


def test_walltime_forecast_passes_only_with_exact_full_coverage(tmp_path: Path) -> None:
    manifest, plan, contract, omission, gate = _forecast_inputs(tmp_path)
    forecast = numerical_only._forecast_body(
        workspace=tmp_path,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=gate,
        available_wall_hours=16.0,
        safety_factor=1.5,
    )
    assert forecast["passed"] is True
    assert forecast["launch_forbidden"] is False
    assert forecast["coverage"]["pilot_profile_count"] == 36
    assert forecast["coverage"]["declared_full_profile_count"] == 3585
    assert forecast["coverage"]["forecast_full_profile_count"] == 3585
    assert forecast["coverage"]["exact_pilot_reuse_profile_count"] == 36
    assert forecast["coverage"]["forecast_executed_profile_count"] == 3549
    assert forecast["coverage"]["profile_reduction_permitted"] is False
    assert [row["full_screen"]["profile_count"] for row in forecast["shards"]] == [
        897,
        896,
        896,
        896,
    ]
    assert [
        row["full_screen"]["executed_profile_count"]
        for row in forecast["shards"]
    ] == [887, 889, 889, 884]
    assert all(
        row["full_screen"]["executed_weight_bank_count"] == 2
        for row in forecast["shards"]
    )
    assert forecast["resource_admission"]["basis"] == (
        "measured_36_profile_four_worker_pilot"
    )
    assert forecast["resource_admission"]["weight_bank_build_concurrency"] == 1
    assert len(forecast["resource_admission"]["per_shard"]) == 4
    assert forecast["aggregate"]["conservative_critical_path_hours"] < 16.0
    assert forecast["aggregate"]["conservative_gpu_hours"] == pytest.approx(
        sum(
            row["projection"]["conservative_total_hours"]
            for row in forecast["shards"]
        )
    )
    assert forecast["classification"]["publication_rankable"] is False


def test_walltime_forecast_blocks_over_budget_and_missing_artifact(tmp_path: Path) -> None:
    manifest, plan, contract, omission, gate = _forecast_inputs(tmp_path)
    failed = numerical_only._forecast_body(
        workspace=tmp_path,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=gate,
        available_wall_hours=0.001,
        safety_factor=1.5,
    )
    assert failed["passed"] is False
    assert failed["launch_forbidden"] is True
    with pytest.raises(RuntimeError, match="forecast is absent"):
        numerical_only._load_passed_walltime_forecast(
            workspace=tmp_path,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=gate,
        )
    write_immutable_json(
        tmp_path / "numerical_screen_walltime_forecast.json",
        failed,
    )
    with pytest.raises(RuntimeError, match="failed admission"):
        numerical_only._load_passed_walltime_forecast(
            workspace=tmp_path,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=gate,
        )


def test_worker_resource_admission_uses_measured_pilot_floors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    manifest, plan, contract, omission, gate = _forecast_inputs(tmp_path)
    forecast = numerical_only._forecast_body(
        workspace=tmp_path,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=gate,
        available_wall_hours=16.0,
        safety_factor=1.5,
    ) | {"content_hash": "9" * 64}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            uuid="GPU-test-0",
            total_memory=192 << 30,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda index: (100 << 30, 192 << 30),
    )
    monkeypatch.setattr(
        numerical_only,
        "_host_mem_available_bytes",
        lambda: 520 << 30,
    )
    admitted = numerical_only._measure_worker_resource_admission(
        forecast=forecast,
        shard_index=0,
    )
    assert admitted["passed"] is True
    assert admitted["failures"] == []
    assert admitted["guessed_server_ram_bytes"] is None

    monkeypatch.setattr(
        numerical_only,
        "_host_mem_available_bytes",
        lambda: 1,
    )
    blocked = numerical_only._measure_worker_resource_admission(
        forecast=forecast,
        shard_index=0,
    )
    assert blocked["passed"] is False
    assert "host_mem_available_below_measured_pilot_floor" in blocked["failures"]


def test_walltime_forecast_detects_source_tamper(tmp_path: Path) -> None:
    manifest, plan, contract, omission, gate = _forecast_inputs(tmp_path)
    forecast = numerical_only._forecast_body(
        workspace=tmp_path,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=gate,
        available_wall_hours=16.0,
        safety_factor=1.5,
    )
    write_immutable_json(
        tmp_path / "numerical_screen_walltime_forecast.json",
        forecast,
    )
    progress_path = tmp_path / "preflight/part-0000-of-0004/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    original_progress = dict(progress)
    progress["updated_at"] = "2026-07-25T00:00:01Z"
    progress_path.write_text(
        json.dumps(progress, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tampered or stale"):
        numerical_only._load_passed_walltime_forecast(
            workspace=tmp_path,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=gate,
        )
    progress_path.write_text(
        json.dumps(original_progress, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    journal_path = next(
        (tmp_path / "preflight/part-0000-of-0004/shards").glob("*.jsonl")
    )
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["runtime_seconds"] = float(row["runtime_seconds"]) + 1.0
    lines[0] = json.dumps(row, sort_keys=True)
    journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result checksum mismatch"):
        numerical_only._load_passed_walltime_forecast(
            workspace=tmp_path,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=gate,
        )


def test_walltime_forecast_rejects_pilot_without_full_bank_coverage(
    tmp_path: Path,
) -> None:
    config = _target_config()
    manifest = _build_manifest(config, ROOT)
    canonical = build_run_plan(
        manifest,
        device_labels=("B200",),
        numerical_screen_workers=4,
        hardware_validation_workers=4,
        numerical_screen_microbatch_size=16,
        hardware_validation_microbatch_size=8,
        gpu_baseline=GPUBaselinePlan.from_config(config["gpu_baseline"]),
    )
    by_id = {entry.profile_id: entry for entry in manifest.entries}
    bf16_id = next(
        profile_id
        for profile_id in canonical.preflight_profile_ids
        if by_id[profile_id].profile.weight_format == "BF16"
    )
    replacement = next(
        entry.profile_id
        for entry in manifest.entries
        if entry.profile.weight_format == "E2M1"
        and entry.profile_id not in canonical.preflight_profile_ids
    )
    changed_ids = tuple(
        replacement if profile_id == bf16_id else profile_id
        for profile_id in canonical.preflight_profile_ids
    )
    changed_plan = replace(canonical, preflight_profile_ids=changed_ids)
    manifest, plan, contract, omission, gate = _forecast_inputs(
        tmp_path,
        plan=changed_plan,
    )
    with pytest.raises(RuntimeError, match="does not cover every full-screen weight bank"):
        numerical_only._forecast_body(
            workspace=tmp_path,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=gate,
            available_wall_hours=16.0,
            safety_factor=1.5,
        )


def test_worker_rejects_every_stage_outside_numerical_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the numerical-only contract"):
        numerical_only.launch_worker(
            config_path=CONFIG_PATH,
            workspace=tmp_path,
            stage="hardware-validation",
            device_label="B200",
            shard_index=0,
            shard_count=4,
        )


def test_normal_strict_workspace_remains_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError,
        match="mixture_of_experts_trace_evidence_not_bound",
    ):
        create_workspace(
            config_path=CONFIG_PATH,
            output_dir=tmp_path / "strict-workspace",
            device_labels=("B200",),
            prompt_manifest_path=None,
            numerical_screen_workers=4,
            hardware_validation_workers=4,
            dry_run=False,
        )
    assert not (tmp_path / "strict-workspace/manifest.json").exists()


def test_plan_seals_exact_workspace_and_revalidates_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PassedPreflight:
        def require_passed(self) -> None:
            return None

        def to_dict(self) -> dict:
            return {"schema_version": "test-preflight", "passed": True}

    monkeypatch.setattr(
        numerical_only,
        "run_launch_preflight",
        lambda *args, **kwargs: _PassedPreflight(),
    )
    prompts = PromptManifest(
        dataset_name="sealed/test",
        dataset_revision="1" * 40,
        numerical_screen=tuple(
            PromptRecord(
                document_id=f"numerical-{index:02d}",
                prompt_hash=f"{index + 1:064x}",
            )
            for index in range(16)
        ),
        hardware_validation=tuple(
            PromptRecord(
                document_id=f"validation-{index:02d}",
                prompt_hash=f"{index + 100:064x}",
            )
            for index in range(32)
        ),
    )
    prompt_path = tmp_path / "input-prompts.json"
    write_immutable_json(prompt_path, prompts.to_dict())
    workspace = tmp_path / "numerical-workspace"
    summary = numerical_only.create_workspace(
        config_path=CONFIG_PATH,
        output_dir=workspace,
        prompt_manifest_path=prompt_path,
        device_label="B200",
    )
    assert summary["profile_count"] == 3585
    assert summary["required_shards"] == 4
    assert summary["classification"]["publication_rankable"] is False
    assert (workspace / "numerical_only_contract.json").is_file()
    assert (workspace / "numerical_only_omissions.json").is_file()

    _, contract, manifest, plan, loaded_prompts, _ = numerical_only._load_contract(
        config_path=CONFIG_PATH,
        workspace=workspace,
    )
    assert contract["bindings"]["manifest_hash"] == manifest.canonical_hash
    assert contract["bindings"]["run_plan_hash"] == plan.canonical_hash
    assert contract["bindings"]["prompt_manifest_hash"] == prompts.canonical_hash
    assert loaded_prompts == prompts
