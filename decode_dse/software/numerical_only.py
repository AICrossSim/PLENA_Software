"""Run the fail-labelled, software-only Qwen3-MoE numerical screen.

This module deliberately does not relax the publication pipeline.  It owns a
separate workspace contract, permits only a deep-oracle numerical pilot and
the exhaustive numerical screen, and records every omitted hardware stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.manifest import load_manifest, write_manifest
from decode_dse.legality import StackValidity
from decode_dse.software.runtime_environment import run_launch_preflight
from decode_dse.software.sweep import (
    DEFAULT_EXECUTOR_FACTORY,
    _build_manifest,
    _compiler_trace_feasibility,
    _load_executor_factory,
    _load_workspace,
    _provenance,
    _sha256_file,
    _stage_contract,
    _stage_ids,
    _sweep_launcher_load_config,
    _validate_provenance,
    parse_gpu_list,
    partition_stage_profile_ids,
)
from decode_dse.software.sweep_plan import (
    ExecutorContext,
    GPUBaselinePlan,
    PromptManifest,
    SweepRunPlan,
    build_run_plan,
    load_immutable_json,
    load_prompt_manifest,
    make_stage_manifest,
    write_immutable_json,
)
from decode_dse.software.sweep_runner import (
    EvaluationOutcome,
    ExhaustiveSweepRunner,
    ResultShardStore,
    SweepRunSummary,
)


CONTRACT_SCHEMA = "decode-numerical-only-contract/v1"
OMISSION_SCHEMA = "decode-numerical-only-omission-receipt/v1"
INVOCATION_SCHEMA = "decode-numerical-only-invocation/v1"
SHARD_LAUNCH_SCHEMA = "decode-numerical-only-shard-launch/v1"
FIDELITY_GATE_SCHEMA = "decode-numerical-only-fidelity-gate/v1"
WALLTIME_FORECAST_SCHEMA = "decode-numerical-only-walltime-forecast/v1"
COMPLETION_SCHEMA = "decode-numerical-only-completion-receipt/v1"
PILOT_REUSE_SCHEMA = "decode-numerical-only-pilot-reuse/v1"

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
REQUIRED_SHARDS = 4
ALLOWED_STAGES = ("preflight", "numerical-screen")
EVIDENCE_CLASS = "measured_numerical_only_non_publication"
PARTITION_ALGORITHM = "whole_weight_bank_round_robin/v1"
EXPECTED_SCREEN_PROFILES = 3585  # canonical full census; the sealed plan is binding


def _expected_screen_profiles(plan: Any) -> int:
    """The declared census is whatever the sealed run plan enumerates.

    The config's ``search.declared_exclusions`` may trim the canonical 3,585
    census to a disclosed subspace; the manifest/plan seal validates the exact
    profile IDs against that declared space, so the forecast binds to the
    sealed count rather than to the canonical constant.
    """

    return len(plan.numerical_screen_profile_ids)


def _authorizes_only(plan: Any) -> str:
    return f"exact-{_expected_screen_profiles(plan)}-profile-numerical-screen"
DEFAULT_SCREEN_WALL_HOURS = 16.0
DEFAULT_FORECAST_SAFETY_FACTOR = 1.5

_TARGET_ARCHITECTURE = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "use_qk_norm": True,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_target(config: Mapping[str, Any]) -> None:
    """Reject every model, revision, topology, and publication-enabled config."""

    if config.get("model_name") != MODEL_NAME:
        raise ValueError(f"numerical-only lane is sealed to {MODEL_NAME}")
    for name in ("model_revision", "tokenizer_revision"):
        if config.get(name) != MODEL_REVISION:
            raise ValueError(f"{name} must equal the sealed target revision")
    architecture = config.get("model_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("model_architecture is required")
    mismatches = {
        key: {"required": expected, "observed": architecture.get(key)}
        for key, expected in _TARGET_ARCHITECTURE.items()
        if architecture.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "target architecture mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if int(config.get("max_parallel_points", 0)) != REQUIRED_SHARDS:
        raise ValueError("target numerical-only lane requires four workers")
    pipeline = config.get("publication_pipeline")
    resources = pipeline.get("resources") if isinstance(pipeline, Mapping) else None
    if not isinstance(resources, Mapping) or resources.get("publication_enabled") is not False:
        raise ValueError(
            "numerical-only lane requires publication_pipeline.resources."
            "publication_enabled=false"
        )


def _classification() -> dict[str, Any]:
    return {
        "evidence_class": EVIDENCE_CLASS,
        "publication_rankable": False,
        "hardware_rankable": False,
        "selection_eligible": False,
        "may_claim_numerical_nll_perplexity": True,
        "may_claim_task_accuracy": False,
        "may_claim_latency": False,
        "may_claim_throughput": False,
        "may_claim_power": False,
        "may_claim_energy": False,
        "may_claim_area": False,
    }


def build_contract(
    *,
    config_sha256: str,
    manifest_hash: str,
    run_plan_hash: str,
    prompt_manifest_hash: str,
    provenance_sha256: str,
    quantizer_provenance_hash: str,
    compiler_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable claim boundary for one numerical-only workspace."""

    blockers = tuple(
        str(item)
        for item in compiler_preflight.get("compiler_trace_preflight_blockers", ())
    )
    if compiler_preflight.get("compiler_trace_preflight_feasible") is not False:
        raise ValueError(
            "numerical-only lane is allowed only while the strict compiler trace is blocked"
        )
    if not any("mixture_of_experts_trace_evidence_not_bound" in item for item in blockers):
        raise ValueError("target MoE compiler-trace blocker is not present")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "target": {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "architecture": dict(_TARGET_ARCHITECTURE),
        },
        "bindings": {
            "config_sha256": config_sha256,
            "manifest_hash": manifest_hash,
            "run_plan_hash": run_plan_hash,
            "prompt_manifest_hash": prompt_manifest_hash,
            "provenance_sha256": provenance_sha256,
            "quantizer_provenance_hash": quantizer_provenance_hash,
        },
        "classification": _classification(),
        "execution": {
            "required_shards": REQUIRED_SHARDS,
            "allowed_stages": list(ALLOWED_STAGES),
            "preflight_role": "deep_oracle_numerical_fidelity_pilot",
            "full_screen_admission": "sealed_pilot_walltime_forecast_required",
            "partition_algorithm": PARTITION_ALGORITHM,
            "restart_policy": "append_only_rows_and_terminal_markers",
            "failed_rows_retained": True,
        },
        "strict_pipeline": {
            "bypassed": False,
            "invoked": False,
            "compiler_trace_preflight_feasible": False,
            "compiler_trace_preflight_blockers": list(blockers),
            "normal_pipeline_remains_fail_closed": True,
        },
        "claim_boundary": (
            "Rows measure teacher-forced cached-decode numerical quality only. "
            "They are not hardware, serving, performance, power, area, or "
            "publication-selection evidence."
        ),
    }


def build_omission_receipt(
    *,
    contract_hash: str,
    bindings: Mapping[str, Any],
    compiler_blockers: Sequence[str],
) -> dict[str, Any]:
    """Record the hardware evidence that this lane does not create."""

    common = {
        "status": "omitted_by_numerical_only_contract",
        "evidence_present": False,
        "artifact": None,
    }
    return {
        "schema_version": OMISSION_SCHEMA,
        "contract_hash": contract_hash,
        "bindings": dict(bindings),
        "classification": _classification(),
        "omissions": [
            {
                "stage": "compiler_full_model_moe_timing",
                "blockers": list(compiler_blockers),
                **common,
            },
            {
                "stage": "transaction_emulator_validation",
                "blockers": ["compiler_timing_evidence_unavailable"],
                **common,
            },
            {
                "stage": "hardware_validation",
                "blockers": ["compiler_and_emulator_evidence_unavailable"],
                **common,
            },
            {
                "stage": "analytic_hardware_search",
                "blockers": ["hardware_validation_unavailable"],
                **common,
            },
            {
                "stage": "latency_throughput_power_energy_area",
                "blockers": ["no_hardware_timing_or_calibration_evidence"],
                **common,
            },
            {
                "stage": "publication_selection",
                "blockers": ["numerical_only_rows_are_selection_ineligible"],
                **common,
            },
        ],
    }


def _contract_path(workspace: Path) -> Path:
    return workspace / "numerical_only_contract.json"


def _omission_path(workspace: Path) -> Path:
    return workspace / "numerical_only_omissions.json"


def _fidelity_path(workspace: Path) -> Path:
    return workspace / "numerical_fidelity_gate.json"


def _walltime_forecast_path(workspace: Path) -> Path:
    return workspace / "numerical_screen_walltime_forecast.json"


def _completion_path(workspace: Path) -> Path:
    return workspace / "numerical_only_completion.json"


def create_workspace(
    *,
    config_path: Path,
    output_dir: Path,
    prompt_manifest_path: Path,
    device_label: str,
) -> dict[str, Any]:
    """Create a four-way numerical-only workspace after host/GPU preflight."""

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    prompt_manifest_path = prompt_manifest_path.resolve()
    repository = Path(__file__).resolve().parents[2]
    config = _sweep_launcher_load_config(config_path)
    _require_target(config)
    manifest = _build_manifest(config, repository)
    compiler_preflight = _compiler_trace_feasibility(config, manifest)
    # Deliberately observe the blocker without calling the strict pipeline's
    # feasibility requirement.  build_contract requires that blocker.
    launch_preflight = run_launch_preflight(
        config,
        repository_root=repository,
        workspace_root=output_dir,
        device_labels=(device_label,),
    )
    launch_preflight.require_passed()
    prompts = load_prompt_manifest(prompt_manifest_path)
    executor_config = config.get("executor")
    if not isinstance(executor_config, Mapping):
        raise ValueError("config.executor is required")
    microbatches = executor_config.get("decode_microbatch_size")
    if not isinstance(microbatches, Mapping):
        raise ValueError("executor.decode_microbatch_size is required")
    gpu_baseline = config.get("gpu_baseline")
    if not isinstance(gpu_baseline, Mapping):
        raise ValueError("config.gpu_baseline is required")
    plan = build_run_plan(
        manifest,
        device_labels=(device_label,),
        numerical_screen_workers=REQUIRED_SHARDS,
        # This shared field is retained for ExecutorContext compatibility;
        # the numerical-only contract never authorizes that stage.
        hardware_validation_workers=REQUIRED_SHARDS,
        numerical_screen_microbatch_size=int(microbatches["numerical_screen"]),
        hardware_validation_microbatch_size=int(microbatches["hardware_validation"]),
        gpu_baseline=GPUBaselinePlan.from_config(gpu_baseline),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "manifest.json", manifest)
    write_immutable_json(output_dir / "run_plan.json", plan.to_dict())
    write_immutable_json(output_dir / "prompt_manifest.json", prompts.to_dict())
    provenance_path = output_dir / "provenance.json"
    existing_created_at = (
        load_immutable_json(provenance_path).get("created_at_utc")
        if provenance_path.is_file()
        else None
    )
    write_immutable_json(
        provenance_path,
        _provenance(
            repository=repository,
            config_path=config_path,
            manifest=manifest,
            plan=plan,
            prompts=prompts,
            created_at_utc=(
                str(existing_created_at) if existing_created_at is not None else None
            ),
        ),
    )
    contract = build_contract(
        config_sha256=_sha256_file(config_path),
        manifest_hash=manifest.canonical_hash,
        run_plan_hash=plan.canonical_hash,
        prompt_manifest_hash=prompts.canonical_hash,
        provenance_sha256=_sha256_file(provenance_path),
        quantizer_provenance_hash=manifest.quantizer_provenance.canonical_hash,
        compiler_preflight=compiler_preflight,
    )
    write_immutable_json(_contract_path(output_dir), contract)
    installed_contract = load_immutable_json(_contract_path(output_dir))
    omission = build_omission_receipt(
        contract_hash=str(installed_contract["content_hash"]),
        bindings=contract["bindings"],
        compiler_blockers=contract["strict_pipeline"][
            "compiler_trace_preflight_blockers"
        ],
    )
    write_immutable_json(_omission_path(output_dir), omission)
    return {
        "schema_version": "decode-numerical-only-plan/v1",
        "workspace": str(output_dir),
        "contract": str(_contract_path(output_dir)),
        "contract_hash": installed_contract["content_hash"],
        "omission_receipt": str(_omission_path(output_dir)),
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "prompt_manifest_hash": prompts.canonical_hash,
        "profile_count": len(plan.numerical_screen_profile_ids),
        "required_shards": REQUIRED_SHARDS,
        "classification": _classification(),
        "launch_preflight": launch_preflight.to_dict(),
    }


def _load_contract(
    *,
    config_path: Path,
    workspace: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    SweepRunPlan,
    PromptManifest,
    Mapping[str, Any],
]:
    """Load and revalidate every code/config/workspace binding."""

    config_path = config_path.resolve()
    workspace = workspace.resolve()
    repository = Path(__file__).resolve().parents[2]
    config = _sweep_launcher_load_config(config_path)
    _require_target(config)
    manifest, plan, prompts = _load_workspace(workspace)
    _validate_provenance(
        workspace / "provenance.json",
        repository=repository,
        config_path=config_path,
        manifest=manifest,
        plan=plan,
        prompts=prompts,
    )
    contract = load_immutable_json(_contract_path(workspace))
    omission = load_immutable_json(_omission_path(workspace))
    expected_bindings = {
        "config_sha256": _sha256_file(config_path),
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "prompt_manifest_hash": prompts.canonical_hash,
        "provenance_sha256": _sha256_file(workspace / "provenance.json"),
        "quantizer_provenance_hash": manifest.quantizer_provenance.canonical_hash,
    }
    expected_contract = build_contract(
        **expected_bindings,
        compiler_preflight=_compiler_trace_feasibility(config, manifest),
    )
    contract_body = dict(contract)
    contract_body.pop("content_hash", None)
    if contract_body != expected_contract:
        raise ValueError("numerical-only contract differs from exact current bindings")
    expected_omission = build_omission_receipt(
        contract_hash=str(contract["content_hash"]),
        bindings=expected_bindings,
        compiler_blockers=expected_contract["strict_pipeline"][
            "compiler_trace_preflight_blockers"
        ],
    )
    omission_body = dict(omission)
    omission_body.pop("content_hash", None)
    if omission_body != expected_omission:
        raise ValueError("numerical-only omission receipt mismatch")
    return config, contract, manifest, plan, prompts, omission


def _stage_partitions(stage_root: Path, shard_count: int) -> tuple[Path, ...]:
    return tuple(
        stage_root / f"part-{index:04d}-of-{shard_count:04d}"
        for index in range(shard_count)
    )


def _load_completion_marker(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    marker_hash = value.pop("marker_hash", None)
    if marker_hash != _canonical_hash(value):
        raise ValueError(f"completion marker checksum mismatch: {path}")
    return value | {"marker_hash": marker_hash}


def summarize_stage(
    *,
    workspace: Path,
    stage: str,
    manifest: Any,
    plan: SweepRunPlan,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all four partitions and retain every terminal row pointer."""

    if stage not in ALLOWED_STAGES:
        raise ValueError(f"stage {stage!r} is outside the numerical-only contract")
    contract_hash = str(contract["content_hash"])
    bindings = contract["bindings"]
    omission = load_immutable_json(_omission_path(workspace))
    fidelity_gate = (
        load_immutable_json(_fidelity_path(workspace))
        if stage == "numerical-screen"
        else None
    )
    expected_gate_hash = (
        fidelity_gate["content_hash"] if fidelity_gate is not None else None
    )
    pilot_terminal_by_id = (
        {
            str(record["profile_id"]): record
            for record in fidelity_gate.get("pilot", {}).get(
                "terminal_records", ()
            )
        }
        if fidelity_gate is not None
        else {}
    )
    expected_forecast_hash = (
        load_immutable_json(_walltime_forecast_path(workspace))["content_hash"]
        if stage == "numerical-screen"
        else None
    )
    admission_sha256 = _sha256_file(workspace / "admission_preparation.json")
    expected_ids = tuple(_stage_ids(plan, stage))
    expected = set(expected_ids)
    full_stage_manifest = make_stage_manifest(manifest, expected_ids)
    sharding = load_immutable_json(workspace / stage / "sharding.json")
    if sharding != {
        "schema_version": "decode-numerical-only-sharding/v1",
        "stage": stage,
        "contract_hash": contract_hash,
        "master_manifest_hash": manifest.canonical_hash,
        "full_stage_manifest_hash": full_stage_manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "shard_count": REQUIRED_SHARDS,
        "algorithm": PARTITION_ALGORITHM,
        "publication_rankable": False,
        "content_hash": sharding.get("content_hash"),
    }:
        raise ValueError("numerical-only sharding receipt mismatch")
    scheduled: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    partitions = []
    for index, root in enumerate(_stage_partitions(workspace / stage, REQUIRED_SHARDS)):
        manifest_path = root / "manifest.json"
        invocation_path = root / "invocation.json"
        if not manifest_path.is_file() or not invocation_path.is_file():
            raise RuntimeError(f"stage partition is incomplete: {root}")
        partition_manifest = load_manifest(manifest_path)
        invocation = load_immutable_json(invocation_path)
        if (
            invocation.get("schema_version") != INVOCATION_SCHEMA
            or invocation.get("contract_hash") != contract_hash
            or invocation.get("omission_receipt_hash") != omission["content_hash"]
            or invocation.get("fidelity_gate_hash") != expected_gate_hash
            or invocation.get("walltime_forecast_hash") != expected_forecast_hash
            or invocation.get("stage") != stage
            or invocation.get("master_manifest_hash") != manifest.canonical_hash
            or invocation.get("stage_manifest_hash")
            != partition_manifest.canonical_hash
            or invocation.get("run_plan_hash") != plan.canonical_hash
            or invocation.get("prompt_manifest_hash")
            != bindings["prompt_manifest_hash"]
            or invocation.get("config_sha256") != bindings["config_sha256"]
            or invocation.get("provenance_sha256")
            != bindings["provenance_sha256"]
            or invocation.get("admission_preparation_sha256")
            != admission_sha256
            or invocation.get("shard_index") != index
            or invocation.get("shard_count") != REQUIRED_SHARDS
            or invocation.get("profile_count") != len(partition_manifest.entries)
            or invocation.get("evidence_class") != EVIDENCE_CLASS
            or invocation.get("publication_rankable") is not False
            or invocation.get("hardware_rankable") is not False
            or invocation.get("selection_eligible") is not False
            or invocation.get("failed_rows_retained") is not True
            or invocation.get("sample_contract")
            != _stage_contract(plan, stage).to_dict()
            or invocation.get("decode_microbatch_size")
            != plan.numerical_screen_microbatch_size
        ):
            raise ValueError(f"numerical-only invocation mismatch: {invocation_path}")
        resource_receipts: list[dict[str, Any]] = []
        if stage == "numerical-screen":
            for resource_path in sorted((root / "resource_admissions").glob("*.json")):
                resource = load_immutable_json(resource_path)
                if (
                    resource.get("schema_version")
                    != "decode-numerical-only-worker-resource-admission/v1"
                    or resource.get("walltime_forecast_hash")
                    != expected_forecast_hash
                    or resource.get("shard_index") != index
                    or resource.get("weight_bank_build_concurrency") != 1
                    or resource.get("guessed_server_ram_bytes") is not None
                    or not isinstance(resource.get("failures"), list)
                    or resource.get("passed") is not (
                        not resource.get("failures")
                    )
                ):
                    raise ValueError(
                        f"invalid worker resource admission: {resource_path}"
                    )
                resource_receipts.append(
                    {
                        "relative_path": str(resource_path.relative_to(workspace)),
                        "content_hash": resource["content_hash"],
                        "passed": bool(resource["passed"]),
                    }
                )
            if not any(receipt["passed"] for receipt in resource_receipts):
                raise RuntimeError(
                    f"stage partition has no passing resource admission: {root}"
                )
        partition_ids = {entry.profile_id for entry in partition_manifest.entries}
        if scheduled & partition_ids:
            raise ValueError("a profile appears in multiple numerical-only shards")
        if not partition_ids <= expected:
            raise ValueError("numerical-only shard contains an unexpected profile")
        scheduled.update(partition_ids)
        result_store = ResultShardStore(root, partition_manifest)
        result_by_path = {
            pointer.journal_path: pointer for pointer in result_store.records
        }
        for marker_path in sorted((root / "completed").glob("*.json")):
            marker = _load_completion_marker(marker_path)
            profile_id = str(marker["profile_id"])
            result_pointer = result_by_path.get(str(marker.get("result_path")))
            if (
                profile_id not in partition_ids
                or marker.get("manifest_hash") != partition_manifest.canonical_hash
                or marker.get("state") not in {"succeeded", "failed"}
                or profile_id in records
                or result_pointer is None
                or result_pointer.record.get("profile_id") != profile_id
                or result_pointer.record.get("state") != marker.get("state")
                or result_pointer.record.get("attempt") != marker.get("attempt")
            ):
                raise ValueError(f"invalid terminal marker: {marker_path}")
            result_metrics = result_pointer.record.get("result")
            reuse = (
                result_metrics.get("evaluation_reuse")
                if isinstance(result_metrics, Mapping)
                else None
            )
            if stage == "preflight" and reuse is not None:
                raise ValueError("a pilot result cannot itself be reused")
            if reuse is not None:
                source = pilot_terminal_by_id.get(profile_id)
                if (
                    not isinstance(reuse, Mapping)
                    or not isinstance(source, Mapping)
                    or reuse.get("schema_version") != PILOT_REUSE_SCHEMA
                    or reuse.get("profile_id") != profile_id
                    or reuse.get("source_stage") != "preflight"
                    or reuse.get("source_shard_index") != index
                    or reuse.get("source_result_path") != source.get("result_path")
                    or reuse.get("source_result_record_hash")
                    != source.get("result_record_hash")
                    or reuse.get("source_completion_marker_hash")
                    != source.get("marker_hash")
                    or reuse.get("fidelity_gate_hash") != expected_gate_hash
                    or reuse.get("sample_contract_exact_match") is not True
                    or reuse.get("numerical_metrics_reused_without_recomputation")
                    is not True
                    or float(result_pointer.record.get("runtime_seconds", -1.0))
                    != 0.0
                ):
                    raise ValueError(f"invalid pilot reuse row: {profile_id}")
            records[profile_id] = {
                "profile_id": profile_id,
                "ordinal": int(marker["ordinal"]),
                "state": marker["state"],
                "attempt": int(marker["attempt"]),
                "result_path": str(marker["result_path"]),
                "result_record_hash": result_pointer.record_hash,
                "runtime_seconds": float(result_pointer.record["runtime_seconds"]),
                "error_class": result_pointer.record.get("error_class"),
                "error_message": result_pointer.record.get("error_message"),
                "marker_path": str(marker_path.relative_to(workspace)),
                "marker_hash": marker["marker_hash"],
                "execution_origin": (
                    "exact_pilot_reuse" if reuse is not None else "executed"
                ),
                "pilot_reuse": dict(reuse) if isinstance(reuse, Mapping) else None,
            }
        partitions.append(
            {
                "shard_index": index,
                "manifest_hash": partition_manifest.canonical_hash,
                "profile_count": len(partition_ids),
                "invocation_hash": invocation["content_hash"],
                "resource_admission_receipts": resource_receipts,
            }
        )
    if scheduled != expected:
        raise RuntimeError("numerical-only shard schedule is not exhaustive")
    missing = tuple(profile_id for profile_id in expected_ids if profile_id not in records)
    ordered = tuple(records[profile_id] for profile_id in expected_ids if profile_id in records)
    failures = tuple(record for record in ordered if record["state"] == "failed")
    reused = tuple(
        record for record in ordered if record["execution_origin"] == "exact_pilot_reuse"
    )
    observed_seconds = tuple(float(record["runtime_seconds"]) for record in ordered)
    return {
        "stage": stage,
        "expected_profiles": len(expected_ids),
        "terminal_profiles": len(ordered),
        "succeeded_profiles": len(ordered) - len(failures),
        "failed_profiles": len(failures),
        "executed_profiles": len(ordered) - len(reused),
        "reused_pilot_profiles": len(reused),
        "terminal_attempt_runtime_seconds_sum": sum(observed_seconds),
        "maximum_terminal_attempt_runtime_seconds": max(
            observed_seconds,
            default=None,
        ),
        "pending_profile_ids": list(missing),
        "terminal_records": list(ordered),
        "partitions": partitions,
        "complete": not missing,
        "passed": not missing and not failures,
    }


def build_fidelity_gate(*, config_path: Path, workspace: Path) -> dict[str, Any]:
    """Gate the full screen on the deep-oracle software fidelity pilot."""

    _, contract, manifest, plan, _, omission = _load_contract(
        config_path=config_path,
        workspace=workspace,
    )
    summary = summarize_stage(
        workspace=workspace.resolve(),
        stage="preflight",
        manifest=manifest,
        plan=plan,
        contract=contract,
    )
    gate = {
        "schema_version": FIDELITY_GATE_SCHEMA,
        "contract_hash": contract["content_hash"],
        "omission_receipt_hash": omission["content_hash"],
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "classification": _classification(),
        "pilot": summary,
        "passed": bool(summary["passed"]),
        "authorizes_only": "numerical-screen",
    }
    write_immutable_json(_fidelity_path(workspace), gate)
    return load_immutable_json(_fidelity_path(workspace))


def _load_passed_fidelity_gate(
    *,
    workspace: Path,
    contract: Mapping[str, Any],
    omission: Mapping[str, Any],
    manifest: Any,
    plan: SweepRunPlan,
) -> Mapping[str, Any]:
    gate = load_immutable_json(_fidelity_path(workspace))
    if (
        gate.get("schema_version") != FIDELITY_GATE_SCHEMA
        or gate.get("contract_hash") != contract.get("content_hash")
        or gate.get("omission_receipt_hash") != omission.get("content_hash")
        or gate.get("manifest_hash") != manifest.canonical_hash
        or gate.get("run_plan_hash") != plan.canonical_hash
        or gate.get("classification") != _classification()
        or gate.get("authorizes_only") != "numerical-screen"
        or gate.get("passed") is not True
    ):
        raise RuntimeError("numerical fidelity gate is absent, failed, or mismatched")
    return gate


def _pilot_reuse_outcomes(
    *,
    workspace: Path,
    manifest: Any,
    plan: SweepRunPlan,
    fidelity_gate: Mapping[str, Any],
    shard_index: int,
    stage_manifest: Any,
) -> dict[str, EvaluationOutcome]:
    """Reuse exact successful pilot rows in the identical full-screen census.

    The preflight stage intentionally uses the numerical-screen sample
    contract, microbatch, runtime, profile identity, and deterministic
    weight-bank shard.  Reinstalling those content-addressed measurements
    avoids 36 duplicate evaluations and, critically, avoids a second full BF16
    model load for the single BF16 profile.
    """

    if fidelity_gate.get("passed") is not True:
        raise RuntimeError("pilot reuse requires a passing fidelity gate")
    pilot_summary = fidelity_gate.get("pilot")
    if not isinstance(pilot_summary, Mapping) or pilot_summary.get("passed") is not True:
        raise RuntimeError("pilot reuse requires complete successful pilot rows")
    terminal_by_id = {
        str(record["profile_id"]): record
        for record in pilot_summary.get("terminal_records", ())
    }
    expected_ids = set(plan.preflight_profile_ids).intersection(
        entry.profile_id for entry in stage_manifest.entries
    )
    pilot_ids = partition_stage_profile_ids(
        manifest,
        plan.preflight_profile_ids,
        shard_index=shard_index,
        shard_count=REQUIRED_SHARDS,
    )
    if set(pilot_ids) != expected_ids:
        raise ValueError("pilot/full-screen weight-bank shard identity changed")
    root = (
        workspace
        / "preflight"
        / f"part-{shard_index:04d}-of-{REQUIRED_SHARDS:04d}"
    )
    partition_manifest = load_manifest(root / "manifest.json")
    if tuple(entry.profile_id for entry in partition_manifest.entries) != pilot_ids:
        raise ValueError("pilot partition manifest order changed before reuse")
    invocation = load_immutable_json(root / "invocation.json")
    store = ResultShardStore(root, partition_manifest)
    pointers = {pointer.record_hash: pointer for pointer in store.records}
    screen_contract = _stage_contract(plan, "numerical-screen").to_dict()
    pilot_contract = _stage_contract(plan, "preflight").to_dict()
    if pilot_contract != screen_contract:
        raise ValueError("pilot and numerical screen sample contracts differ")

    outcomes: dict[str, EvaluationOutcome] = {}
    for profile_id in pilot_ids:
        terminal = terminal_by_id.get(profile_id)
        if not isinstance(terminal, Mapping) or terminal.get("state") != "succeeded":
            raise RuntimeError(f"pilot profile cannot be reused: {profile_id}")
        pointer = pointers.get(str(terminal.get("result_record_hash")))
        if (
            pointer is None
            or pointer.record.get("profile_id") != profile_id
            or pointer.record.get("state") != "succeeded"
            or pointer.record.get("attempt") != terminal.get("attempt")
            or pointer.journal_path != terminal.get("result_path")
        ):
            raise ValueError(f"pilot reuse ancestry mismatch: {profile_id}")
        metrics = pointer.record.get("result")
        if not isinstance(metrics, Mapping):
            raise TypeError("pilot result metrics must be an object")
        if metrics.get("sample_contract") != screen_contract:
            raise ValueError("pilot result sample contract differs from full screen")
        if "evaluation_reuse" in metrics:
            raise ValueError("pilot source result is already a reused row")
        reuse = {
            "schema_version": PILOT_REUSE_SCHEMA,
            "profile_id": profile_id,
            "source_stage": "preflight",
            "source_shard_index": shard_index,
            "source_partition_manifest_hash": partition_manifest.canonical_hash,
            "source_invocation_hash": invocation["content_hash"],
            "source_result_path": pointer.journal_path,
            "source_result_record_hash": pointer.record_hash,
            "source_completion_marker_hash": terminal["marker_hash"],
            "fidelity_gate_hash": fidelity_gate["content_hash"],
            "sample_contract_exact_match": True,
            "numerical_metrics_reused_without_recomputation": True,
        }
        outcomes[profile_id] = EvaluationOutcome(
            metrics=dict(metrics) | {"evaluation_reuse": reuse},
            validity=StackValidity.from_dict(pointer.record.get("validity")),
            artifacts=tuple(str(value) for value in pointer.record.get("artifacts", ()))
            + (str((root / "manifest.json").resolve()),),
        )
    if set(outcomes) != expected_ids:
        raise RuntimeError("pilot reuse does not cover the exact shard pilot subset")
    return outcomes


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _pilot_resource_policy(
    pointers: Sequence[Any],
    *,
    shard_index: int,
) -> dict[str, Any]:
    """Derive launch floors only from the measured four-worker pilot."""

    host_available: list[int] = []
    process_peak: list[int] = []
    gpu_peak_reserved: list[int] = []
    gpu_totals: set[int] = set()
    device_uuids: set[str] = set()
    cache_peaks: list[int] = []
    cache_capacities: set[int] = set()
    fused_counts: set[int] = set()
    fused_bytes: list[int] = []

    def add_host(value: Any, label: str) -> None:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} host observation is missing")
        host_available.append(
            _positive_integer(
                value.get("host_mem_available_bytes"),
                f"{label}.host_mem_available_bytes",
            )
        )
        _positive_integer(value.get("process_rss_bytes"), f"{label}.process_rss_bytes")
        process_peak.append(
            _positive_integer(
                value.get("process_peak_rss_bytes"),
                f"{label}.process_peak_rss_bytes",
            )
        )

    for pointer in pointers:
        result = pointer.record.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("pilot result metrics are missing")
        profile_resource = result.get("resource_observation")
        if (
            not isinstance(profile_resource, Mapping)
            or profile_resource.get("schema_version")
            != "decode-profile-resource-observation/v1"
        ):
            raise ValueError("pilot profile resource observation is missing")
        add_host(
            profile_resource.get("host_before_evaluation"),
            "profile.host_before_evaluation",
        )
        add_host(
            profile_resource.get("host_after_evaluation"),
            "profile.host_after_evaluation",
        )
        cache = profile_resource.get("decode_cache_lru")
        if not isinstance(cache, Mapping):
            raise ValueError("pilot decode-cache observation is missing")
        capacity = _positive_integer(
            cache.get("configured_capacity_bytes"),
            "decode_cache_lru.configured_capacity_bytes",
        )
        peak = _nonnegative_integer(
            cache.get("peak_resident_bytes_this_weight_bank"),
            "decode_cache_lru.peak_resident_bytes_this_weight_bank",
        )
        resident = _nonnegative_integer(
            cache.get("resident_bytes_after_evaluation"),
            "decode_cache_lru.resident_bytes_after_evaluation",
        )
        if peak > capacity or resident > capacity or resident > peak:
            raise ValueError("pilot decode-cache residency exceeds its bound")
        cache_capacities.add(capacity)
        cache_peaks.append(peak)

        bank = result.get("weight_bank")
        bank_resource = (
            bank.get("resource_observation") if isinstance(bank, Mapping) else None
        )
        if (
            not isinstance(bank_resource, Mapping)
            or bank_resource.get("schema_version")
            != "decode-weight-bank-resource-observation/v1"
            or bank_resource.get("build_serialized_across_workers") is not True
        ):
            raise ValueError("pilot weight-bank resource observation is missing")
        add_host(bank_resource.get("host_before_build"), "bank.host_before_build")
        add_host(bank_resource.get("host_after_build"), "bank.host_after_build")
        fused_counts.add(
            _positive_integer(
                bank_resource.get("fused_expert_module_count"),
                "weight_bank.fused_expert_module_count",
            )
        )
        fused_bytes.append(
            _positive_integer(
                bank_resource.get("fused_expert_parameter_bytes"),
                "weight_bank.fused_expert_parameter_bytes",
            )
        )
        bank_gpu = bank_resource.get("gpu")
        if not isinstance(bank_gpu, Mapping):
            raise ValueError("pilot bank GPU observation is missing")
        bank_total = _positive_integer(
            bank_gpu.get("total_device_bytes"),
            "weight_bank.gpu.total_device_bytes",
        )
        bank_peak = _positive_integer(
            bank_gpu.get("peak_reserved_bytes_during_build"),
            "weight_bank.gpu.peak_reserved_bytes_during_build",
        )
        if bank_peak > bank_total:
            raise ValueError("pilot bank peak exceeds device capacity")
        gpu_totals.add(bank_total)
        gpu_peak_reserved.append(bank_peak)

        evaluation_gpu = result.get("gpu_memory")
        runtime = result.get("runtime_environment")
        if not isinstance(evaluation_gpu, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError("pilot evaluation GPU observation is missing")
        evaluation_total = _positive_integer(
            evaluation_gpu.get("total_device_bytes"),
            "gpu_memory.total_device_bytes",
        )
        evaluation_peak = _positive_integer(
            evaluation_gpu.get("peak_reserved_bytes"),
            "gpu_memory.peak_reserved_bytes",
        )
        if evaluation_peak > evaluation_total:
            raise ValueError("pilot evaluation peak exceeds device capacity")
        device_uuid = str(runtime.get("device_uuid", ""))
        if not device_uuid or device_uuid == "unavailable":
            raise ValueError("pilot resource policy requires a physical GPU UUID")
        gpu_totals.add(evaluation_total)
        gpu_peak_reserved.append(evaluation_peak)
        device_uuids.add(device_uuid)

    if (
        not pointers
        or len(gpu_totals) != 1
        or len(device_uuids) != 1
        or len(cache_capacities) != 1
        or fused_counts != {_TARGET_ARCHITECTURE["num_hidden_layers"]}
    ):
        raise ValueError("pilot resource observations are internally inconsistent")
    total_device_bytes = next(iter(gpu_totals))
    required_gpu_free = max(gpu_peak_reserved)
    return {
        "schema_version": "decode-numerical-only-resource-pilot/v1",
        "shard_index": shard_index,
        "observation_count": len(pointers),
        "device_uuid": next(iter(device_uuids)),
        "total_device_bytes": total_device_bytes,
        "required_gpu_free_bytes_at_worker_start": required_gpu_free,
        "minimum_gpu_headroom_bytes_at_peak": (
            total_device_bytes - required_gpu_free
        ),
        "minimum_host_mem_available_bytes_observed": min(host_available),
        "maximum_process_peak_rss_bytes_observed": max(process_peak),
        "configured_decode_cache_capacity_bytes": next(iter(cache_capacities)),
        "maximum_decode_cache_resident_bytes_observed": max(cache_peaks),
        "fused_expert_module_count": next(iter(fused_counts)),
        "maximum_fused_expert_parameter_bytes": max(fused_bytes),
        "weight_bank_build_concurrency": 1,
        "policy": (
            "same_four_workers_same_microbatch_current_host_available_not_below_"
            "pilot_floor_and_gpu_free_not_below_measured_peak"
        ),
        "guessed_server_ram_bytes": None,
        "passed": True,
    }


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("a timing percentile requires at least one observation")
    ordered = sorted(_positive_finite(value, "timing observation") for value in values)
    index = max(0, math.ceil(float(quantile) * len(ordered)) - 1)
    return ordered[index]


def _weight_formats_for_ids(manifest: Any, profile_ids: Sequence[str]) -> tuple[str, ...]:
    by_id = {entry.profile_id: entry for entry in manifest.entries}
    return tuple(
        dict.fromkeys(by_id[profile_id].profile.weight_format for profile_id in profile_ids)
    )


def _forecast_body(
    *,
    workspace: Path,
    contract: Mapping[str, Any],
    omission: Mapping[str, Any],
    manifest: Any,
    plan: SweepRunPlan,
    fidelity_gate: Mapping[str, Any],
    available_wall_hours: float,
    safety_factor: float,
) -> dict[str, Any]:
    """Project the exhaustive screen from the sealed deep-oracle pilot."""

    available_hours = _positive_finite(
        available_wall_hours, "available numerical-screen wall-hours"
    )
    safety = _positive_finite(safety_factor, "forecast safety factor")
    if safety < 1.0:
        raise ValueError("forecast safety factor must be at least 1.0")
    if len(plan.preflight_profile_ids) != 36:
        raise ValueError("wall-time forecast requires the exact 36-profile pilot")
    if len(plan.numerical_screen_profile_ids) != _expected_screen_profiles(plan):
        raise ValueError(
            f"wall-time forecast requires all {_expected_screen_profiles(plan)} screen profiles"
        )

    pilot_summary = summarize_stage(
        workspace=workspace,
        stage="preflight",
        manifest=manifest,
        plan=plan,
        contract=contract,
    )
    if pilot_summary != fidelity_gate.get("pilot"):
        raise ValueError("fidelity gate no longer matches the measured pilot")
    if pilot_summary.get("passed") is not True:
        raise RuntimeError("wall-time forecast requires a fully successful pilot")

    by_id = {entry.profile_id: entry for entry in manifest.entries}
    terminal_by_id = {
        str(record["profile_id"]): record
        for record in pilot_summary["terminal_records"]
    }
    pilot_seen: set[str] = set()
    screen_seen: set[str] = set()
    shards: list[dict[str, Any]] = []

    for shard_index in range(REQUIRED_SHARDS):
        pilot_ids = partition_stage_profile_ids(
            manifest,
            plan.preflight_profile_ids,
            shard_index=shard_index,
            shard_count=REQUIRED_SHARDS,
        )
        screen_ids = partition_stage_profile_ids(
            manifest,
            plan.numerical_screen_profile_ids,
            shard_index=shard_index,
            shard_count=REQUIRED_SHARDS,
        )
        if pilot_seen.intersection(pilot_ids) or screen_seen.intersection(screen_ids):
            raise ValueError("deterministic shard projection contains duplicate profiles")
        pilot_seen.update(pilot_ids)
        screen_seen.update(screen_ids)
        pilot_banks = _weight_formats_for_ids(manifest, pilot_ids)
        screen_banks = _weight_formats_for_ids(manifest, screen_ids)
        pilot_id_set = set(pilot_ids)
        execution_ids = tuple(
            profile_id for profile_id in screen_ids if profile_id not in pilot_id_set
        )
        execution_banks = _weight_formats_for_ids(manifest, execution_ids)
        if pilot_banks != screen_banks:
            raise RuntimeError(
                f"pilot shard {shard_index} does not cover every full-screen weight bank"
            )

        root = (
            workspace
            / "preflight"
            / f"part-{shard_index:04d}-of-{REQUIRED_SHARDS:04d}"
        )
        partition_manifest = load_manifest(root / "manifest.json")
        if tuple(entry.profile_id for entry in partition_manifest.entries) != pilot_ids:
            raise ValueError(f"pilot shard {shard_index} manifest order changed")
        invocation = load_immutable_json(root / "invocation.json")
        store = ResultShardStore(root, partition_manifest)
        pointers_by_hash = {pointer.record_hash: pointer for pointer in store.records}

        terminal_pointers = []
        per_profile_attempt_seconds: dict[str, float] = {}
        attempts_by_profile: dict[str, int] = {}
        for pointer in store.records:
            profile_id = str(pointer.record.get("profile_id"))
            if profile_id not in set(pilot_ids):
                raise ValueError(f"pilot shard {shard_index} contains an unknown result row")
            runtime = _nonnegative_finite(
                pointer.record.get("runtime_seconds"),
                f"pilot shard {shard_index} result runtime",
            )
            per_profile_attempt_seconds[profile_id] = (
                per_profile_attempt_seconds.get(profile_id, 0.0) + runtime
            )
            attempts_by_profile[profile_id] = attempts_by_profile.get(profile_id, 0) + 1
        for profile_id in pilot_ids:
            terminal = terminal_by_id.get(profile_id)
            if terminal is None:
                raise RuntimeError(f"pilot shard {shard_index} lacks terminal coverage")
            pointer = pointers_by_hash.get(str(terminal["result_record_hash"]))
            if pointer is None or pointer.record.get("state") != "succeeded":
                raise ValueError(f"pilot shard {shard_index} terminal result mismatch")
            if per_profile_attempt_seconds.get(profile_id, 0.0) <= 0.0:
                raise ValueError(f"pilot shard {shard_index} has no positive evaluation timing")
            terminal_pointers.append(pointer)

        resource_pilot = _pilot_resource_policy(
            terminal_pointers,
            shard_index=shard_index,
        )

        bank_build_seconds: dict[str, float] = {}
        bank_local_overhead_seconds: dict[str, float] = {}
        bank_outer_open_seconds: dict[str, float] = {}
        bank_lock_wait_seconds: dict[str, float] = {}
        for bank in pilot_banks:
            observed: list[float] = []
            local_observed: list[float] = []
            outer_observed: list[float] = []
            wait_observed: list[float] = []
            for pointer in terminal_pointers:
                profile_id = str(pointer.record["profile_id"])
                if by_id[profile_id].profile.weight_format != bank:
                    continue
                result = pointer.record.get("result")
                telemetry = result.get("weight_bank") if isinstance(result, Mapping) else None
                if not isinstance(telemetry, Mapping):
                    raise ValueError(
                        f"pilot shard {shard_index} result lacks weight-bank telemetry"
                    )
                if telemetry.get("weight_format") != bank:
                    raise ValueError(
                        f"pilot shard {shard_index} weight-bank telemetry mismatch"
                    )
                resource = telemetry.get("resource_observation")
                timing = resource.get("timing") if isinstance(resource, Mapping) else None
                if (
                    not isinstance(resource, Mapping)
                    or resource.get("schema_version")
                    != "decode-weight-bank-resource-observation/v1"
                    or resource.get("build_serialized_across_workers") is not True
                    or not isinstance(timing, Mapping)
                ):
                    raise ValueError(
                        f"pilot shard {shard_index} lacks serialized bank resources"
                    )
                serialized = _positive_finite(
                    timing.get("serialized_build_seconds"),
                    f"pilot shard {shard_index} {bank} serialized bank build",
                )
                outer = _positive_finite(
                    timing.get("outer_open_seconds"),
                    f"pilot shard {shard_index} {bank} outer bank open",
                )
                wait = _nonnegative_finite(
                    timing.get("lock_wait_seconds"),
                    f"pilot shard {shard_index} {bank} bank lock wait",
                )
                pre = _nonnegative_finite(
                    timing.get("pre_lock_setup_seconds"),
                    f"pilot shard {shard_index} {bank} pre-lock setup",
                )
                post = _nonnegative_finite(
                    timing.get("post_lock_validation_seconds"),
                    f"pilot shard {shard_index} {bank} post-lock validation",
                )
                if (
                    not math.isclose(
                        outer,
                        pre + wait + serialized + post,
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        outer,
                        _positive_finite(
                            telemetry.get("build_seconds"),
                            f"pilot shard {shard_index} {bank} bank build",
                        ),
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError("weight-bank timing decomposition is inconsistent")
                observed.append(serialized)
                local_observed.append(pre + post)
                outer_observed.append(outer)
                wait_observed.append(wait)
            if not observed:
                raise ValueError(
                    f"pilot shard {shard_index} has no measured build for bank {bank}"
                )
            tolerance = max(observed) * 1.0e-9
            if max(observed) - min(observed) > tolerance:
                raise ValueError(
                    f"pilot shard {shard_index} reports inconsistent build time for bank {bank}"
                )
            bank_build_seconds[bank] = max(observed)
            bank_local_overhead_seconds[bank] = max(local_observed)
            bank_outer_open_seconds[bank] = max(outer_observed)
            bank_lock_wait_seconds[bank] = max(wait_observed)

        progress_path = root / "progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError(
                f"pilot shard {shard_index} is missing its measured progress snapshot"
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_required = progress.get("unique_weight_banks_required_this_invocation")
        progress_opened = progress.get("unique_weight_banks_opened")
        progress_remaining = progress.get("unique_weight_banks_remaining")
        if (
            progress.get("schema_version") != "decode-sweep-progress"
            or progress.get("stage") != "preflight"
            or progress.get("work_class") != "numerical"
            or progress.get("completed_profiles") != len(pilot_ids)
            or progress.get("succeeded_profiles") != len(pilot_ids)
            or progress.get("failed_terminal_profiles") != 0
            or progress.get("total_profiles") != len(pilot_ids)
            or progress.get("remaining_profiles") != 0
            or not isinstance(progress_required, int)
            or not isinstance(progress_opened, int)
            or not isinstance(progress_remaining, int)
            or not 0 <= progress_required <= len(pilot_banks)
            or progress_opened != progress_required
            or progress_remaining != 0
        ):
            raise ValueError(f"pilot shard {shard_index} progress snapshot is inconsistent")
        progress_mean = progress.get("mean_weight_bank_open_seconds")
        if progress_opened:
            progress_mean_seconds = _positive_finite(
                progress_mean,
                f"pilot shard {shard_index} mean weight-bank open time",
            )
        else:
            if progress_mean is not None:
                raise ValueError(
                    f"pilot shard {shard_index} reports bank timing without an opened bank"
                )
            progress_mean_seconds = None

        evaluation_samples = tuple(
            per_profile_attempt_seconds[profile_id] for profile_id in pilot_ids
        )
        evaluation_mean = sum(evaluation_samples) / len(evaluation_samples)
        evaluation_p95 = _nearest_rank(evaluation_samples, 0.95)
        evaluation_max = max(evaluation_samples)
        bank_samples = tuple(bank_build_seconds[bank] for bank in pilot_banks)
        bank_p95 = _nearest_rank(bank_samples, 0.95)
        bank_max = max(bank_samples)
        bank_local_samples = tuple(
            bank_local_overhead_seconds[bank] for bank in pilot_banks
        )
        bank_local_p95 = _nearest_rank(bank_local_samples, 0.95)
        bank_local_max = max(bank_local_samples)

        raw_evaluation_seconds = evaluation_mean * len(execution_ids)
        raw_serialized_bank_seconds = sum(
            bank_build_seconds[bank] for bank in execution_banks
        )
        raw_local_bank_seconds = sum(
            bank_local_overhead_seconds[bank] for bank in execution_banks
        )
        raw_bank_seconds = raw_serialized_bank_seconds + raw_local_bank_seconds
        conservative_evaluation_seconds = (
            max(evaluation_p95, evaluation_max) * len(execution_ids)
        )
        conservative_serialized_bank_seconds = max(
            raw_serialized_bank_seconds,
            bank_p95 * len(execution_banks),
            bank_max * len(execution_banks),
        )
        conservative_local_bank_seconds = max(
            raw_local_bank_seconds,
            bank_local_p95 * len(execution_banks),
            bank_local_max * len(execution_banks),
        )
        conservative_bank_seconds = (
            conservative_serialized_bank_seconds
            + conservative_local_bank_seconds
        )
        raw_seconds = raw_evaluation_seconds + raw_bank_seconds
        conservative_seconds_before_safety = (
            conservative_evaluation_seconds + conservative_bank_seconds
        )
        conservative_seconds = conservative_seconds_before_safety * safety
        terminal_records = [terminal_by_id[profile_id] for profile_id in pilot_ids]
        shards.append(
            {
                "shard_index": shard_index,
                "pilot": {
                    "profile_count": len(pilot_ids),
                    "profile_ids": list(pilot_ids),
                    "weight_bank_count": len(pilot_banks),
                    "weight_banks": list(pilot_banks),
                    "attempt_count": sum(attempts_by_profile.values()),
                    "retry_count": sum(attempts_by_profile.values()) - len(pilot_ids),
                    "partition_manifest_hash": partition_manifest.canonical_hash,
                    "invocation_hash": invocation["content_hash"],
                    "attempt_result_record_hashes": [
                        pointer.record_hash for pointer in store.records
                    ],
                    "result_record_hashes": [
                        str(record["result_record_hash"]) for record in terminal_records
                    ],
                    "completion_marker_hashes": [
                        str(record["marker_hash"]) for record in terminal_records
                    ],
                    "progress_sha256": _sha256_file(progress_path),
                    "progress_snapshot": progress,
                },
                "full_screen": {
                    "profile_count": len(screen_ids),
                    "exact_pilot_reuse_profile_count": len(pilot_ids),
                    "executed_profile_count": len(execution_ids),
                    "weight_bank_count": len(screen_banks),
                    "weight_banks": list(screen_banks),
                    "executed_weight_bank_count": len(execution_banks),
                    "executed_weight_banks": list(execution_banks),
                },
                "resource_pilot": resource_pilot,
                "measured_seconds": {
                    "evaluation_per_profile_mean": evaluation_mean,
                    "evaluation_per_profile_p95_nearest_rank": evaluation_p95,
                    "evaluation_per_profile_max": evaluation_max,
                    "weight_bank_build_by_format": bank_build_seconds,
                    "weight_bank_local_overhead_by_format": (
                        bank_local_overhead_seconds
                    ),
                    "weight_bank_outer_open_by_format": bank_outer_open_seconds,
                    "weight_bank_lock_wait_by_format": bank_lock_wait_seconds,
                    "weight_bank_build_p95_nearest_rank": bank_p95,
                    "weight_bank_build_max": bank_max,
                    "weight_bank_local_overhead_p95_nearest_rank": (
                        bank_local_p95
                    ),
                    "weight_bank_local_overhead_max": bank_local_max,
                    "progress_mean_weight_bank_open": progress_mean_seconds,
                },
                "projection": {
                    "raw_evaluation_hours": raw_evaluation_seconds / 3600.0,
                    "raw_weight_bank_hours": raw_bank_seconds / 3600.0,
                    "raw_serialized_weight_bank_hours": (
                        raw_serialized_bank_seconds / 3600.0
                    ),
                    "raw_local_weight_bank_overhead_hours": (
                        raw_local_bank_seconds / 3600.0
                    ),
                    "raw_total_hours": raw_seconds / 3600.0,
                    "conservative_evaluation_hours_before_safety": (
                        conservative_evaluation_seconds / 3600.0
                    ),
                    "conservative_weight_bank_hours_before_safety": (
                        conservative_bank_seconds / 3600.0
                    ),
                    "conservative_serialized_weight_bank_hours_before_safety": (
                        conservative_serialized_bank_seconds / 3600.0
                    ),
                    "conservative_local_weight_bank_overhead_hours_before_safety": (
                        conservative_local_bank_seconds / 3600.0
                    ),
                    "conservative_total_hours_before_safety": (
                        conservative_seconds_before_safety / 3600.0
                    ),
                    "conservative_total_hours": conservative_seconds / 3600.0,
                },
            }
        )

    if pilot_seen != set(plan.preflight_profile_ids):
        raise ValueError("pilot shard union is not the exact declared pilot")
    if screen_seen != set(plan.numerical_screen_profile_ids):
        raise ValueError("full-screen shard union silently changes declared coverage")
    conservative_hours = tuple(
        float(shard["projection"]["conservative_total_hours"]) for shard in shards
    )
    raw_hours = tuple(float(shard["projection"]["raw_total_hours"]) for shard in shards)
    raw_serialized_bank_hours = sum(
        float(shard["projection"]["raw_serialized_weight_bank_hours"])
        for shard in shards
    )
    raw_evaluation_critical_hours = max(
        float(shard["projection"]["raw_evaluation_hours"])
        + float(
            shard["projection"]["raw_local_weight_bank_overhead_hours"]
        )
        for shard in shards
    )
    conservative_serialized_bank_hours_before_safety = sum(
        float(
            shard["projection"][
                "conservative_serialized_weight_bank_hours_before_safety"
            ]
        )
        for shard in shards
    )
    conservative_evaluation_critical_hours_before_safety = max(
        float(
            shard["projection"][
                "conservative_evaluation_hours_before_safety"
            ]
        )
        + float(
            shard["projection"][
                "conservative_local_weight_bank_overhead_hours_before_safety"
            ]
        )
        for shard in shards
    )
    # Model construction is deliberately serialized across all four workers
    # to bound unknown host-RAM pressure.  Treat every build as dependency-
    # bound in the forecast instead of incorrectly assuming four-way overlap.
    raw_critical_path = raw_serialized_bank_hours + raw_evaluation_critical_hours
    critical_path_before_safety = (
        conservative_serialized_bank_hours_before_safety
        + conservative_evaluation_critical_hours_before_safety
    )
    critical_path = critical_path_before_safety * safety
    gpu_hours = sum(conservative_hours)
    raw_gpu_hours = sum(raw_hours)
    resource_pilots = tuple(shard["resource_pilot"] for shard in shards)
    pilot_device_uuids = tuple(
        str(resource["device_uuid"]) for resource in resource_pilots
    )
    if not 1 <= len(set(pilot_device_uuids)) <= REQUIRED_SHARDS:
        raise ValueError("resource pilot must bind between one and four physical GPUs")
    # The four logical shards may share physical GPUs (e.g. a three- or two-GPU
    # allocation runs shard pairs back to back). The admission receipt still
    # binds every shard to the exact device its pilot ran on; the wall-clock
    # projection below remains the per-shard critical path, so a smaller
    # allocation is longer in wall time but identical in GPU-hours.
    resource_admission = {
        "schema_version": "decode-numerical-only-resource-admission/v1",
        "basis": "measured_36_profile_four_worker_pilot",
        "weight_bank_build_concurrency": 1,
        "minimum_host_mem_available_bytes_observed": min(
            int(resource["minimum_host_mem_available_bytes_observed"])
            for resource in resource_pilots
        ),
        "per_shard": [
            {
                "shard_index": int(resource["shard_index"]),
                "device_uuid": str(resource["device_uuid"]),
                "total_device_bytes": int(resource["total_device_bytes"]),
                "required_gpu_free_bytes_at_worker_start": int(
                    resource["required_gpu_free_bytes_at_worker_start"]
                ),
                "maximum_process_peak_rss_bytes_observed": int(
                    resource["maximum_process_peak_rss_bytes_observed"]
                ),
                "maximum_decode_cache_resident_bytes_observed": int(
                    resource["maximum_decode_cache_resident_bytes_observed"]
                ),
            }
            for resource in resource_pilots
        ],
        "unknown_server_ram_policy": (
            "no_capacity_guess; require current MemAvailable to be no lower "
            "than the minimum observed while the exact four-worker pilot ran"
        ),
        "profile_or_microbatch_reduction_permitted": False,
        "passed": True,
    }
    passed = critical_path <= available_hours
    return {
        "schema_version": WALLTIME_FORECAST_SCHEMA,
        "contract_hash": contract["content_hash"],
        "omission_receipt_hash": omission["content_hash"],
        "fidelity_gate_hash": fidelity_gate["content_hash"],
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "source_bindings": dict(contract["bindings"]),
        "classification": _classification(),
        "budget": {
            "available_numerical_screen_wall_hours": available_hours,
            "available_four_gpu_hours": available_hours * REQUIRED_SHARDS,
            "default_wall_hours": DEFAULT_SCREEN_WALL_HOURS,
            "safety_factor": safety,
            "default_safety_factor": DEFAULT_FORECAST_SAFETY_FACTOR,
            "override_policy": "set_once_when_forecast_is_first_sealed",
        },
        "estimator": {
            "version": "pilot-reuse-serialized-build-p95-max/v2",
            "evaluation": (
                "pilot per-profile total attempt time; only the 3,549 profiles "
                "not already measured by the exact pilot are projected, using "
                "the larger of nearest-rank p95 and maximum"
            ),
            "weight_bank": (
                "exact remaining bank census; lock wait is excluded, globally "
                "serialized build time and rank-local pre/post-lock overhead are "
                "projected separately with exact-sum, p95-times-count, and "
                "max-times-count floors"
            ),
            "safety_factor_applied_after_components": True,
            "parallelism": (
                "four independent one-GPU whole-weight-bank shards with a "
                "global serialized model/weight-bank build lock"
            ),
            "exact_pilot_reuse": (
                "36 content-addressed rows under the identical numerical-screen "
                "sample/microbatch/profile contract"
            ),
        },
        "coverage": {
            "pilot_profile_count": len(plan.preflight_profile_ids),
            "declared_full_profile_count": len(plan.numerical_screen_profile_ids),
            "forecast_full_profile_count": sum(
                int(shard["full_screen"]["profile_count"]) for shard in shards
            ),
            "exact_pilot_reuse_profile_count": sum(
                int(shard["full_screen"]["exact_pilot_reuse_profile_count"])
                for shard in shards
            ),
            "forecast_executed_profile_count": sum(
                int(shard["full_screen"]["executed_profile_count"])
                for shard in shards
            ),
            "required_shards": REQUIRED_SHARDS,
            "partition_algorithm": PARTITION_ALGORITHM,
            "pilot_bank_union": sorted(
                {bank for shard in shards for bank in shard["pilot"]["weight_banks"]}
            ),
            "full_bank_union": sorted(
                {bank for shard in shards for bank in shard["full_screen"]["weight_banks"]}
            ),
            "pilot_covers_every_full_shard_bank": True,
            "profile_reduction_permitted": False,
            "failed_profile_omission_permitted": False,
            "assumptions": [
                "pilot per-profile timing distribution bounds its same-shard full screen",
                "each exact weight bank opens once per shard invocation",
                "four shards execute concurrently on four independent B200 GPUs",
                "the global build lock serializes all eight remaining quantized bank builds",
                "pilot rows are reused only when their exact source and sample contract validate",
                "terminal failures are retained and never converted into reduced coverage",
                "queueing, unrelated tenant load, and future retries are "
                "covered only by the explicit safety factor",
            ],
        },
        "shards": shards,
        "resource_admission": resource_admission,
        "aggregate": {
            "raw_critical_path_hours": raw_critical_path,
            "raw_gpu_hours": raw_gpu_hours,
            "raw_serialized_weight_bank_hours": raw_serialized_bank_hours,
            "raw_evaluation_critical_path_hours": raw_evaluation_critical_hours,
            "conservative_serialized_weight_bank_hours_before_safety": (
                conservative_serialized_bank_hours_before_safety
            ),
            "conservative_evaluation_critical_path_hours_before_safety": (
                conservative_evaluation_critical_hours_before_safety
            ),
            "conservative_critical_path_hours_before_safety": (
                critical_path_before_safety
            ),
            "conservative_critical_path_hours": critical_path,
            "conservative_gpu_hours": gpu_hours,
            "reserved_four_gpu_hours_at_critical_path": (
                critical_path * REQUIRED_SHARDS
            ),
            "critical_path_margin_hours": available_hours - critical_path,
        },
        "passed": passed,
        "authorizes_only": _authorizes_only(plan),
        "launch_forbidden": not passed,
        "publication_rankable": False,
        "selection_eligible": False,
    }


def build_walltime_forecast(
    *,
    config_path: Path,
    workspace: Path,
    available_wall_hours: float = DEFAULT_SCREEN_WALL_HOURS,
    safety_factor: float = DEFAULT_FORECAST_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Seal the one-time pilot-derived screen forecast and admission decision."""

    _, contract, manifest, plan, _, omission = _load_contract(
        config_path=config_path,
        workspace=workspace,
    )
    fidelity_gate = _load_passed_fidelity_gate(
        workspace=workspace,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
    )
    body = _forecast_body(
        workspace=workspace.resolve(),
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=fidelity_gate,
        available_wall_hours=available_wall_hours,
        safety_factor=safety_factor,
    )
    write_immutable_json(_walltime_forecast_path(workspace), body)
    return load_immutable_json(_walltime_forecast_path(workspace))


def _load_passed_walltime_forecast(
    *,
    workspace: Path,
    contract: Mapping[str, Any],
    omission: Mapping[str, Any],
    manifest: Any,
    plan: SweepRunPlan,
    fidelity_gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = _walltime_forecast_path(workspace)
    if not path.is_file():
        raise RuntimeError("numerical-screen wall-time forecast is absent")
    forecast = load_immutable_json(path)
    budget = forecast.get("budget")
    if not isinstance(budget, Mapping):
        raise RuntimeError("numerical-screen wall-time forecast budget is missing")
    expected = _forecast_body(
        workspace=workspace,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=fidelity_gate,
        available_wall_hours=budget.get("available_numerical_screen_wall_hours"),
        safety_factor=budget.get("safety_factor"),
    )
    observed_body = dict(forecast)
    observed_body.pop("content_hash", None)
    if observed_body != expected:
        raise RuntimeError("numerical-screen wall-time forecast is tampered or stale")
    if (
        forecast.get("schema_version") != WALLTIME_FORECAST_SCHEMA
        or forecast.get("authorizes_only")
        != _authorizes_only(plan)
        or forecast.get("classification") != _classification()
        or forecast.get("publication_rankable") is not False
        or forecast.get("selection_eligible") is not False
    ):
        raise RuntimeError("numerical-screen wall-time forecast failed admission")
    if forecast.get("passed") is not True or forecast.get("launch_forbidden") is not False:
        aggregate = forecast.get("aggregate")
        required = (
            aggregate.get("conservative_critical_path_hours")
            if isinstance(aggregate, Mapping)
            else None
        )
        available = budget.get("available_numerical_screen_wall_hours")
        raise RuntimeError(
            "numerical-screen wall-time forecast failed admission: "
            f"projected_required_wall_hours={required!r}, "
            f"available_wall_hours={available!r}; declared_profiles="
            f"{_expected_screen_profiles(plan)} (no reduction permitted)"
        )
    return forecast


def _host_mem_available_bytes() -> int:
    path = Path("/proc/meminfo")
    if not path.is_file():
        raise RuntimeError("host-memory availability cannot be measured")
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, payload = line.partition(":")
        if name == "MemAvailable" and separator:
            amount, unit = payload.strip().split()
            if unit != "kB":
                break
            return int(amount) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _measure_worker_resource_admission(
    *,
    forecast: Mapping[str, Any],
    shard_index: int,
) -> dict[str, Any]:
    """Check the full-screen worker against its sealed pilot resource floor."""

    import torch

    policy = forecast.get("resource_admission")
    if (
        not isinstance(policy, Mapping)
        or policy.get("schema_version")
        != "decode-numerical-only-resource-admission/v1"
        or policy.get("basis") != "measured_36_profile_four_worker_pilot"
        or policy.get("weight_bank_build_concurrency") != 1
        or policy.get("passed") is not True
    ):
        raise RuntimeError("numerical-screen resource pilot is absent or invalid")
    rows = policy.get("per_shard")
    if not isinstance(rows, list):
        raise RuntimeError("numerical-screen per-shard resource policy is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("shard_index") == shard_index
    ]
    if len(matches) != 1:
        raise RuntimeError("numerical-screen resource shard policy is ambiguous")
    expected = matches[0]
    if not torch.cuda.is_available():
        raise RuntimeError("numerical-screen resource admission requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(properties, "uuid", "unavailable"))
    total_device_bytes = int(properties.total_memory)
    free_device_bytes, observed_total = torch.cuda.mem_get_info(0)
    host_available = _host_mem_available_bytes()
    required_host = _positive_integer(
        policy.get("minimum_host_mem_available_bytes_observed"),
        "resource admission host floor",
    )
    required_gpu = _positive_integer(
        expected.get("required_gpu_free_bytes_at_worker_start"),
        "resource admission GPU floor",
    )
    failures = []
    if device_uuid != expected.get("device_uuid"):
        failures.append("physical_gpu_uuid_changed_from_pilot")
    if (
        total_device_bytes != expected.get("total_device_bytes")
        or int(observed_total) != total_device_bytes
    ):
        failures.append("physical_gpu_capacity_changed_from_pilot")
    if int(free_device_bytes) < required_gpu:
        failures.append("gpu_free_memory_below_measured_pilot_peak")
    if host_available < required_host:
        failures.append("host_mem_available_below_measured_pilot_floor")
    return {
        "schema_version": "decode-numerical-only-worker-resource-admission/v1",
        "walltime_forecast_hash": forecast["content_hash"],
        "shard_index": shard_index,
        "weight_bank_build_concurrency": 1,
        "pilot_device_uuid": expected.get("device_uuid"),
        "observed_device_uuid": device_uuid,
        "pilot_total_device_bytes": expected.get("total_device_bytes"),
        "observed_total_device_bytes": total_device_bytes,
        "required_gpu_free_bytes": required_gpu,
        "observed_gpu_free_bytes": int(free_device_bytes),
        "required_host_mem_available_bytes": required_host,
        "observed_host_mem_available_bytes": host_available,
        "failures": failures,
        "passed": not failures,
        "guessed_server_ram_bytes": None,
    }


def _seal_worker_resource_admission(
    *,
    output_dir: Path,
    observation: Mapping[str, Any],
) -> Path:
    root = output_dir / "resource_admissions"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"{timestamp}-pid-{os.getpid()}.json"
    write_immutable_json(path, dict(observation))
    return path


def launch_worker(
    *,
    config_path: Path,
    workspace: Path,
    stage: str,
    device_label: str,
    shard_index: int,
    shard_count: int,
    executor_factory: str = DEFAULT_EXECUTOR_FACTORY,
) -> SweepRunSummary:
    """Run one deterministic shard without authorizing hardware evidence."""

    if stage not in ALLOWED_STAGES:
        raise ValueError(f"stage {stage!r} is outside the numerical-only contract")
    if shard_count != REQUIRED_SHARDS:
        raise ValueError("numerical-only execution requires exactly four shards")
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid numerical-only shard index")
    config_path = config_path.resolve()
    workspace = workspace.resolve()
    config, contract, manifest, plan, prompts, omission = _load_contract(
        config_path=config_path,
        workspace=workspace,
    )
    if plan.numerical_screen_workers != REQUIRED_SHARDS:
        raise ValueError("run plan does not bind four numerical-screen workers")
    if device_label not in plan.device_labels:
        raise ValueError("device label is outside the numerical-only plan")
    admission_path = workspace / "admission_preparation.json"
    if not admission_path.is_file():
        raise FileNotFoundError("numerical-only execution requires sealed admission preparation")
    gate_hash = None
    forecast_hash = None
    fidelity_gate: Mapping[str, Any] | None = None
    if stage == "numerical-screen":
        fidelity_gate = _load_passed_fidelity_gate(
            workspace=workspace,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
        )
        gate_hash = fidelity_gate["content_hash"]
        forecast = _load_passed_walltime_forecast(
            workspace=workspace,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=fidelity_gate,
        )
        forecast_hash = forecast["content_hash"]

    stage_ids = _stage_ids(plan, stage)
    shard_ids = partition_stage_profile_ids(
        manifest,
        stage_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    stage_manifest = make_stage_manifest(manifest, shard_ids)
    full_stage_manifest = make_stage_manifest(manifest, stage_ids)
    stage_root = workspace / stage
    write_immutable_json(
        stage_root / "sharding.json",
        {
            "schema_version": "decode-numerical-only-sharding/v1",
            "stage": stage,
            "contract_hash": contract["content_hash"],
            "master_manifest_hash": manifest.canonical_hash,
            "full_stage_manifest_hash": full_stage_manifest.canonical_hash,
            "run_plan_hash": plan.canonical_hash,
            "shard_count": REQUIRED_SHARDS,
            "algorithm": PARTITION_ALGORITHM,
            "publication_rankable": False,
        },
    )
    stage_output = stage_root / f"part-{shard_index:04d}-of-{shard_count:04d}"
    invocation = {
        "schema_version": INVOCATION_SCHEMA,
        "stage": stage,
        "contract_hash": contract["content_hash"],
        "omission_receipt_hash": omission["content_hash"],
        "fidelity_gate_hash": gate_hash,
        "walltime_forecast_hash": forecast_hash,
        "master_manifest_hash": manifest.canonical_hash,
        "stage_manifest_hash": stage_manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "prompt_manifest_hash": prompts.canonical_hash,
        "config_sha256": _sha256_file(config_path),
        "provenance_sha256": _sha256_file(workspace / "provenance.json"),
        "admission_preparation_sha256": _sha256_file(admission_path),
        "executor_factory": executor_factory,
        "device_label": device_label,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "profile_count": len(stage_manifest.entries),
        "sample_contract": _stage_contract(plan, stage).to_dict(),
        "decode_microbatch_size": plan.numerical_screen_microbatch_size,
        "max_attempts": int(config.get("runtime", {}).get("max_attempts", 3)),
        "evidence_class": EVIDENCE_CLASS,
        "publication_rankable": False,
        "hardware_rankable": False,
        "selection_eligible": False,
        "failed_rows_retained": True,
    }
    write_immutable_json(stage_output / "invocation.json", invocation)
    if stage == "numerical-screen":
        if fidelity_gate is None:
            raise AssertionError("numerical screen lost its fidelity gate")
        resource_observation = _measure_worker_resource_admission(
            forecast=forecast,
            shard_index=shard_index,
        )
        resource_path = _seal_worker_resource_admission(
            output_dir=stage_output,
            observation=resource_observation,
        )
        if resource_observation["passed"] is not True:
            raise RuntimeError(
                "numerical-screen worker resource admission failed; receipt="
                f"{resource_path}: {resource_observation['failures']}"
            )
    context = ExecutorContext(
        stage=stage,
        workspace_root=workspace,
        output_dir=stage_output,
        config=config,
        master_manifest=manifest,
        stage_manifest=stage_manifest,
        run_plan=plan,
        prompts=prompts,
        sample_contract=_stage_contract(plan, stage),
        shard_index=shard_index,
        shard_count=shard_count,
        device_label=device_label,
    )
    factory = _load_executor_factory(executor_factory)
    executor = factory(context)
    for method in ("open_weight_bank", "open_kv_admission_cache", "evaluate"):
        if not callable(getattr(executor, method, None)):
            raise TypeError(f"executor is missing callable {method}")
    precomputed = (
        _pilot_reuse_outcomes(
            workspace=workspace,
            manifest=manifest,
            plan=plan,
            fidelity_gate=fidelity_gate,
            shard_index=shard_index,
            stage_manifest=stage_manifest,
        )
        if fidelity_gate is not None
        else {}
    )
    return ExhaustiveSweepRunner(
        manifest=stage_manifest,
        output_dir=stage_output,
        executor=executor,
        max_attempts=int(config.get("runtime", {}).get("max_attempts", 3)),
        stage=stage,
        precomputed_outcomes=precomputed,
    ).run()


def _worker_command(
    *,
    config_path: Path,
    workspace: Path,
    stage: str,
    device_label: str,
    shard_index: int,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "decode_dse.software.numerical_only",
        "worker",
        "--config",
        str(config_path),
        "--workspace",
        str(workspace),
        "--stage",
        stage,
        "--device-label",
        device_label,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(REQUIRED_SHARDS),
    )


def launch_shards(
    *,
    config_path: Path,
    workspace: Path,
    stage: str,
    device_label: str,
    devices: Sequence[str],
) -> int:
    """Fan one allowed stage over exactly four deterministic GPU shards."""

    if stage not in ALLOWED_STAGES:
        raise ValueError(f"stage {stage!r} is outside the numerical-only contract")
    if len(devices) != REQUIRED_SHARDS or len(set(devices)) != REQUIRED_SHARDS:
        raise ValueError("numerical-only execution requires four distinct GPU IDs")
    _, contract, manifest, plan, _, omission = _load_contract(
        config_path=config_path,
        workspace=workspace,
    )
    if plan.device_labels != (device_label,):
        raise ValueError("device label differs from numerical-only plan")
    if stage == "numerical-screen":
        gate = _load_passed_fidelity_gate(
            workspace=workspace,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
        )
        _load_passed_walltime_forecast(
            workspace=workspace,
            contract=contract,
            omission=omission,
            manifest=manifest,
            plan=plan,
            fidelity_gate=gate,
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_root = workspace / "logs" / f"numerical-only-{stage}-{timestamp}"
    log_root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    workers = []
    try:
        for index, device in enumerate(devices):
            command = _worker_command(
                config_path=config_path.resolve(),
                workspace=workspace.resolve(),
                stage=stage,
                device_label=device_label,
                shard_index=index,
            )
            log_path = log_root / f"part-{index:04d}.log"
            handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            try:
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                handle.close()
                raise
            processes.append((process, handle, log_path))
            workers.append(
                {
                    "shard_index": index,
                    "cuda_visible_devices": device,
                    "command": list(command),
                    "log": str(log_path.resolve()),
                    "progress": str(
                        (
                            workspace
                            / stage
                            / f"part-{index:04d}-of-{REQUIRED_SHARDS:04d}"
                            / "progress.json"
                        ).resolve()
                    ),
                }
            )
    except BaseException:
        for process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            handle.close()
        raise

    reported: dict[int, str] = {}

    def report_progress() -> None:
        for index, worker in enumerate(workers):
            path = Path(worker["progress"])
            if not path.is_file():
                continue
            try:
                progress = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            identity = str(progress.get("updated_at", ""))
            if not identity or reported.get(index) == identity:
                continue
            reported[index] = identity
            eta = progress.get("estimated_remaining_seconds")
            eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
            print(
                f"numerical-only {stage} shard {index}: "
                f"profiles={progress.get('completed_profiles')}/"
                f"{progress.get('total_profiles')} remaining-eta={eta_text}",
                flush=True,
            )

    return_codes: list[int] = []
    try:
        while any(process.poll() is None for process, _, _ in processes):
            report_progress()
            time.sleep(1.0)
        report_progress()
        return_codes = [process.wait() for process, _, _ in processes]
    except BaseException:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in processes:
            process.wait()
        raise
    finally:
        for _, handle, _ in processes:
            handle.close()

    write_immutable_json(
        log_root / "summary.json",
        {
            "schema_version": SHARD_LAUNCH_SCHEMA,
            "contract_hash": contract["content_hash"],
            "stage": stage,
            "device_label": device_label,
            "shard_count": REQUIRED_SHARDS,
            "partition_algorithm": PARTITION_ALGORITHM,
            "classification": _classification(),
            "workers": [
                worker | {"return_code": return_codes[index]}
                for index, worker in enumerate(workers)
            ],
        },
    )
    failed = [index for index, value in enumerate(return_codes) if value != 0]
    if failed:
        print(f"numerical-only {stage} failed on shards {failed}", file=sys.stderr)
        return 2
    return 0


def finalize(*, config_path: Path, workspace: Path) -> dict[str, Any]:
    """Seal complete terminal coverage without promoting any hardware claim."""

    _, contract, manifest, plan, prompts, omission = _load_contract(
        config_path=config_path,
        workspace=workspace,
    )
    gate = _load_passed_fidelity_gate(
        workspace=workspace,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
    )
    forecast = _load_passed_walltime_forecast(
        workspace=workspace,
        contract=contract,
        omission=omission,
        manifest=manifest,
        plan=plan,
        fidelity_gate=gate,
    )
    summary = summarize_stage(
        workspace=workspace.resolve(),
        stage="numerical-screen",
        manifest=manifest,
        plan=plan,
        contract=contract,
    )
    receipt = {
        "schema_version": COMPLETION_SCHEMA,
        "contract_hash": contract["content_hash"],
        "omission_receipt_hash": omission["content_hash"],
        "fidelity_gate_hash": gate["content_hash"],
        "walltime_forecast_hash": forecast["content_hash"],
        "bindings": {
            "manifest_hash": manifest.canonical_hash,
            "run_plan_hash": plan.canonical_hash,
            "prompt_manifest_hash": prompts.canonical_hash,
        },
        "classification": _classification(),
        "screen": summary,
        "complete": bool(summary["complete"]),
        "successful": bool(summary["passed"]),
        "failed_rows_retained": True,
        "claim_boundary": contract["claim_boundary"],
    }
    write_immutable_json(_completion_path(workspace), receipt)
    return load_immutable_json(_completion_path(workspace))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--workspace", type=Path, required=True)
    plan.add_argument("--prompt-manifest", type=Path, required=True)
    plan.add_argument("--device-label", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--workspace", type=Path, required=True)
    worker.add_argument("--stage", choices=ALLOWED_STAGES, required=True)
    worker.add_argument("--device-label", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    shards = commands.add_parser("shards")
    shards.add_argument("--config", type=Path, required=True)
    shards.add_argument("--workspace", type=Path, required=True)
    shards.add_argument("--stage", choices=ALLOWED_STAGES, required=True)
    shards.add_argument("--device-label", required=True)
    shards.add_argument("--gpus", required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--workspace", type=Path, required=True)
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--config", type=Path, required=True)
    forecast.add_argument("--workspace", type=Path, required=True)
    forecast.add_argument(
        "--available-wall-hours",
        type=float,
        default=DEFAULT_SCREEN_WALL_HOURS,
    )
    forecast.add_argument(
        "--safety-factor",
        type=float,
        default=DEFAULT_FORECAST_SAFETY_FACTOR,
    )
    finish = commands.add_parser("finalize")
    finish.add_argument("--config", type=Path, required=True)
    finish.add_argument("--workspace", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.command == "plan":
            result = create_workspace(
                config_path=args.config,
                output_dir=args.workspace,
                prompt_manifest_path=args.prompt_manifest,
                device_label=args.device_label,
            )
            status = 0
        elif args.command == "worker":
            summary = launch_worker(
                config_path=args.config,
                workspace=args.workspace,
                stage=args.stage,
                device_label=args.device_label,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            result = {
                "schema_version": "decode-numerical-only-worker-summary/v1",
                "stage": args.stage,
                "shard_index": args.shard_index,
                "attempts_written": summary.attempts_written,
                "succeeded": summary.succeeded,
                "failed_terminal": summary.failed_terminal,
                "pending": summary.pending,
                "result_rows": summary.result_rows,
                "classification": _classification(),
            }
            status = 0
        elif args.command == "shards":
            return launch_shards(
                config_path=args.config.resolve(),
                workspace=args.workspace.resolve(),
                stage=args.stage,
                device_label=args.device_label,
                devices=parse_gpu_list(args.gpus),
            )
        elif args.command == "gate":
            result = build_fidelity_gate(
                config_path=args.config.resolve(),
                workspace=args.workspace.resolve(),
            )
            status = 0 if result["passed"] else 2
        elif args.command == "forecast":
            result = build_walltime_forecast(
                config_path=args.config.resolve(),
                workspace=args.workspace.resolve(),
                available_wall_hours=args.available_wall_hours,
                safety_factor=args.safety_factor,
            )
            status = 0 if result["passed"] else 2
        else:
            result = finalize(
                config_path=args.config.resolve(),
                workspace=args.workspace.resolve(),
            )
            status = 0 if result["successful"] else 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return status
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
