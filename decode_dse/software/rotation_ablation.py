"""Plan and verify the Qwen3-MoE attention-only Hadamard ablation.

The numerical run deliberately reuses the restartable refinement evaluator.
This module seals its ancestry and refuses to promote it into a selectable
PLENA row: fused experts are not rotated, and the compiler/emulator do not yet
provide measured lowering or timing for the added attention transforms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.software.refinement_runner import (
    ATTENTION_ONLY_ROTATION_METHOD,
    ATTENTION_ONLY_ROTATION_MATMUL_TYPES,
    FUSED_EXPERT_ROTATION_EXCLUSIONS,
    RefinementBankSpec,
    RefinementMergedResults,
    _tree_hash,
    load_refinement_merged_results,
    rotation_decision_contract,
    validate_attention_only_rotation_decision,
)
from decode_dse.software.refinement_schedule import (
    RefinementSchedule,
    RefinementScheduleEntry,
    build_selective_rotation_schedule,
    load_refinement_schedule,
    write_refinement_schedule,
)
from decode_dse.software.sweep_plan import load_immutable_json, write_immutable_json


TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
TARGET_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
TARGET_SOURCE_ROLES = frozenset(
    {
        "uniform_mxint8",
        "uniform_mxint4",
        "mxint_kv2",
        "accuracy_constrained_deployment",
    }
)
PLAN_SCHEMA = "decode-attention-only-rotation-ablation-plan/v1"
COST_RAW_SCHEMA = "decode-attention-only-rotation-cost-raw/v1"
COST_RECEIPT_SCHEMA = "decode-attention-only-rotation-cost-receipt/v1"
FINAL_SCHEMA = "decode-attention-only-rotation-ablation-final/v1"
HARDWARE_STUDY_SCHEMA = "decode-hardware-study"
TIMING_SOURCES = frozenset({"cuda_event_synchronized"})
POWER_SOURCES = frozenset(
    {"nvml_total_energy_counter", "nvml_power_trace_trapezoidal"}
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: str | Path) -> str:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    token = str(value)
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("rotation config must be a JSON object")
    return value


def _validate_target_config(config: Mapping[str, Any]) -> None:
    architecture = config.get("model_architecture")
    refinement = config.get("refinement")
    ablation = config.get("rotation_ablation")
    blocker = config.get("rotation_refinement_blocker")
    if (
        config.get("model_name") != TARGET_MODEL
        or config.get("model_revision") != TARGET_REVISION
        or config.get("tokenizer_revision") != TARGET_REVISION
        or not isinstance(architecture, Mapping)
        or architecture.get("model_type") != "qwen3_moe"
        or int(architecture.get("num_experts", -1)) != 128
        or int(architecture.get("num_experts_per_tok", -1)) != 8
    ):
        raise ValueError("rotation ablation is not bound to the sealed Qwen3-MoE target")
    if (
        not isinstance(refinement, Mapping)
        or refinement.get("require_symmetric_kv") is not True
    ):
        raise ValueError("rotation ablation requires canonical equal-K/V refinement")
    if config.get("use_rotation") is not False:
        raise ValueError("rotation must remain outside the canonical base sweep")
    if (
        not isinstance(blocker, Mapping)
        or blocker.get("selector_may_emit_rotation") is not False
    ):
        raise ValueError("the strict selector rotation gate must remain fail-closed")
    if (
        not isinstance(ablation, Mapping)
        or ablation.get("schema_version")
        != "decode-attention-only-rotation-ablation/v1"
        or ablation.get("method_id") != ATTENTION_ONLY_ROTATION_METHOD
        or ablation.get("eligible_matmul_types")
        != list(ATTENTION_ONLY_ROTATION_MATMUL_TYPES)
        or ablation.get("excluded_fused_expert_matmul_types")
        != list(FUSED_EXPERT_ROTATION_EXCLUSIONS)
        or ablation.get("publication_rankable") is not False
        or ablation.get("selector_may_emit") is not False
    ):
        raise ValueError("rotation-ablation config differs from the sealed method")


def _source_roles(value: Mapping[str, Any]) -> dict[str, str]:
    if value.get("schema_version") != "decode-refinement-source-selection":
        raise ValueError("unsupported refinement source-selection artifact")
    selection = value.get("source_selection")
    roles = selection.get("source_roles") if isinstance(selection, Mapping) else None
    if (
        not isinstance(roles, Mapping)
        or set(roles) != TARGET_SOURCE_ROLES
        or len(set(map(str, roles.values()))) != 4
    ):
        raise ValueError("rotation source roles are incomplete or duplicated")
    return {str(role): str(profile_id) for role, profile_id in roles.items()}


def select_measured_rotation_sources(
    base_schedule: RefinementSchedule,
    merged: RefinementMergedResults,
    source_roles: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Choose the lowest-NLL successful equal-K/V GPTQ row for each source."""

    if set(source_roles) != TARGET_SOURCE_ROLES:
        raise ValueError("rotation source-role set differs from the sealed contract")
    if set(source_roles.values()) != set(base_schedule.source_profile_ids):
        raise ValueError("source selection and base refinement sources differ")
    rows = {
        str(row.get("profile_id")): row for row in merged.terminal_rows
    }
    if len(rows) != len(base_schedule.entries):
        raise ValueError("base refinement merge does not have exact schedule coverage")
    role_by_source = {profile_id: role for role, profile_id in source_roles.items()}
    selected = []
    for source_id in base_schedule.source_profile_ids:
        candidates = []
        for entry in base_schedule.entries:
            profile = entry.profile
            row = rows.get(entry.profile_id)
            result = row.get("result") if isinstance(row, Mapping) else None
            if (
                profile.source_profile.profile_id != source_id
                or profile.weight_method != "gptq_erry"
                or profile.key_format != profile.value_format
                or profile.key_format != profile.source_profile.kv_format
                or not entry.gate.executable
                or not isinstance(row, Mapping)
                or row.get("state") != "succeeded"
                or not isinstance(result, Mapping)
            ):
                continue
            mean_nll = float(result.get("mean_token_nll", math.nan))
            if not math.isfinite(mean_nll) or mean_nll < 0:
                continue
            candidates.append((mean_nll, entry.profile_id, entry, row))
        if not candidates:
            raise ValueError(
                f"source {source_id} has no successful canonical GPTQ refinement"
            )
        mean_nll, _, entry, row = min(candidates)
        selected.append(
            {
                "source_role": role_by_source[source_id],
                "source_profile_id": source_id,
                "base_refinement_profile_id": entry.profile_id,
                "base_refinement_record_hash": _sha256(
                    row.get("record_hash"), "base refinement record hash"
                ),
                "base_mean_token_nll": mean_nll,
            }
        )
    by_role = {item["source_role"]: item for item in selected}
    uniform = by_role["uniform_mxint8"]
    uniform_entry = next(
        entry
        for entry in base_schedule.entries
        if entry.profile_id == uniform["base_refinement_profile_id"]
    )
    profile = uniform_entry.profile.source_profile
    if (
        profile.weight_format,
        profile.activation_format,
        profile.kv_format,
        uniform_entry.profile.key_format,
        uniform_entry.profile.value_format,
    ) != ("MXINT8", "MXINT8", "MXINT8", "MXINT8", "MXINT8"):
        raise ValueError("uniform-MXINT8 rotation control has the wrong precision")
    return tuple(selected)


def _hardware_evidence(
    paths: Sequence[str | Path],
    *,
    declared_hashes: Sequence[str],
    source_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    from decode_dse.hardware.design_space import load_hardware_artifact

    if not paths:
        raise ValueError("rotation planning requires bound source hardware artifacts")
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(resolved) != len(set(resolved)):
        raise ValueError("rotation hardware artifact paths are duplicated")
    digests = tuple(_file_hash(path) for path in resolved)
    if len(digests) != len(set(digests)) or set(digests) != set(declared_hashes):
        raise ValueError(
            "rotation hardware artifacts differ from source-selection provenance"
        )
    source_set = set(source_ids)
    covered: set[str] = set()
    receipts = []
    for path, digest in zip(resolved, digests):
        header, rows = load_hardware_artifact(path)
        provenance = header.get("provenance")
        if (
            header.get("schema_version") != HARDWARE_STUDY_SCHEMA
            or not isinstance(provenance, Mapping)
            or provenance.get("model_revision") != TARGET_REVISION
            or provenance.get("tokenizer_revision") != TARGET_REVISION
        ):
            raise ValueError("rotation hardware provenance is not the sealed target")
        matched = []
        for row in rows:
            profile_id = str(row.get("profile_id", ""))
            if profile_id in source_set:
                covered.add(profile_id)
                matched.append(
                    {
                        "profile_id": profile_id,
                        "candidate_id": str(row.get("candidate_id", "")),
                        "row_hash": _content_hash(row),
                    }
                )
        receipts.append(
            {
                "path": str(path),
                "sha256": digest,
                "provenance_hash": _content_hash(provenance),
                "matched_source_rows": matched,
            }
        )
    if covered != source_set:
        raise ValueError("hardware artifacts do not cost every rotation source")
    return tuple(receipts)


def materialize_rotation_plan(
    *,
    config_path: str | Path,
    base_schedule_path: str | Path,
    base_merge_path: str | Path,
    source_selection_path: str | Path,
    hardware_paths: Sequence[str | Path],
    schedule_path: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    base_schedule_file = Path(base_schedule_path).resolve()
    base_merge_file = Path(base_merge_path).resolve()
    source_file = Path(source_selection_path).resolve()
    output_schedule = Path(schedule_path).resolve()
    config = _load_config(config_file)
    _validate_target_config(config)
    base_schedule = load_refinement_schedule(base_schedule_file)
    base_merged = load_refinement_merged_results(base_schedule, base_merge_file)
    source = load_immutable_json(source_file)
    roles = _source_roles(source)
    if source.get("schedule_hash") != base_schedule.canonical_hash:
        raise ValueError("source selection does not bind the base refinement schedule")
    selected = select_measured_rotation_sources(base_schedule, base_merged, roles)
    hardware = _hardware_evidence(
        hardware_paths,
        declared_hashes=tuple(map(str, source.get("hardware_study_sha256", ()))),
        source_ids=base_schedule.source_profile_ids,
    )
    selected_ids = tuple(item["base_refinement_profile_id"] for item in selected)
    uniform_id = next(
        item["base_refinement_profile_id"]
        for item in selected
        if item["source_role"] == "uniform_mxint8"
    )
    schedule = build_selective_rotation_schedule(
        base_schedule,
        best_supported_profile_ids=selected_ids,
        uniform_i8_profile_id=uniform_id,
    )
    if (
        len(schedule.entries) != 4
        or any(entry.profile.split_kv for entry in schedule.entries)
        or any(entry.profile.weight_method != "rotation" for entry in schedule.entries)
        or any(
            value is not None
            for entry in schedule.entries
            for value in entry.validity.to_dict().values()
        )
    ):
        raise AssertionError("rotation schedule is not a four-row unmeasured ablation")
    write_refinement_schedule(output_schedule, schedule)
    rotation_by_source = {
        entry.profile.source_profile.profile_id: entry.profile_id
        for entry in schedule.entries
    }
    ancestry = [
        dict(item)
        | {"rotation_profile_id": rotation_by_source[item["source_profile_id"]]}
        for item in selected
    ]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "model_name": TARGET_MODEL,
        "model_revision": TARGET_REVISION,
        "tokenizer_revision": TARGET_REVISION,
        "method_id": ATTENTION_ONLY_ROTATION_METHOD,
        "scope": {
            "eligible_matmul_types": list(ATTENTION_ONLY_ROTATION_MATMUL_TYPES),
            "excluded_fused_expert_matmul_types": list(
                FUSED_EXPERT_ROTATION_EXCLUSIONS
            ),
            "kv_precision": "canonical_equal_kv",
        },
        "config": {"path": str(config_file), "sha256": _file_hash(config_file)},
        "source_selection": {
            "path": str(source_file),
            "sha256": _file_hash(source_file),
            "content_hash": source["content_hash"],
        },
        "base_refinement": {
            "schedule_path": str(base_schedule_file),
            "schedule_file_sha256": _file_hash(base_schedule_file),
            "schedule_hash": base_schedule.canonical_hash,
            "merge_path": str(base_merge_file),
            "merge_file_sha256": _file_hash(base_merge_file),
            "merge_content_hash": base_merged.receipt["content_hash"],
            "results_path": str(base_merged.results_path),
            "results_sha256": base_merged.results_sha256,
        },
        "rotation_schedule": {
            "path": str(output_schedule),
            "file_sha256": _file_hash(output_schedule),
            "schedule_hash": schedule.canonical_hash,
        },
        "source_ancestry": ancestry,
        "source_hardware_costs": list(hardware),
        "evidence_class": "planned_attention_only_rotation_ablation",
        "required_before_final": [
            "exact_terminal_success_for_all_four_rotation_profiles",
            "profile_local_rotation_decision_artifacts",
            "paired_synchronized_b200_timing_and_power_receipt",
        ],
        "strict_stack_status": {
            "compiler_valid": None,
            "emulator_valid": None,
            "rtl_valid": None,
            "plena_rotation_cost_measured": False,
            "publication_rankable": False,
            "selection_eligible": False,
        },
    }
    write_immutable_json(plan_path, plan)
    return plan


def _load_bound_plan(
    plan_path: str | Path,
    schedule_path: str | Path,
    merge_path: str | Path,
) -> tuple[dict[str, Any], RefinementSchedule, RefinementMergedResults]:
    plan = load_immutable_json(plan_path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("method_id") != ATTENTION_ONLY_ROTATION_METHOD
    ):
        raise ValueError("unsupported rotation-ablation plan")
    config_receipt = plan.get("config")
    source_receipt = plan.get("source_selection")
    base_receipt = plan.get("base_refinement")
    hardware_receipts = plan.get("source_hardware_costs")
    if not all(
        isinstance(value, Mapping)
        for value in (config_receipt, source_receipt, base_receipt)
    ) or not isinstance(hardware_receipts, list):
        raise ValueError("rotation plan ancestry receipts are malformed")
    config_file = Path(str(config_receipt["path"])).resolve()
    if config_receipt.get("sha256") != _file_hash(config_file):
        raise ValueError("rotation target config changed after planning")
    _validate_target_config(_load_config(config_file))
    source_file = Path(str(source_receipt["path"])).resolve()
    source = load_immutable_json(source_file)
    if (
        source_receipt.get("sha256") != _file_hash(source_file)
        or source_receipt.get("content_hash") != source.get("content_hash")
    ):
        raise ValueError("rotation source-selection artifact changed after planning")
    base_schedule_file = Path(str(base_receipt["schedule_path"])).resolve()
    base_merge_file = Path(str(base_receipt["merge_path"])).resolve()
    if (
        base_receipt.get("schedule_file_sha256") != _file_hash(base_schedule_file)
        or base_receipt.get("merge_file_sha256") != _file_hash(base_merge_file)
    ):
        raise ValueError("rotation base-refinement artifacts changed after planning")
    base_schedule = load_refinement_schedule(base_schedule_file)
    base_merged = load_refinement_merged_results(base_schedule, base_merge_file)
    if (
        base_receipt.get("schedule_hash") != base_schedule.canonical_hash
        or base_receipt.get("merge_content_hash")
        != base_merged.receipt.get("content_hash")
        or base_receipt.get("results_path") != str(base_merged.results_path)
        or base_receipt.get("results_sha256") != base_merged.results_sha256
    ):
        raise ValueError("rotation base-refinement ancestry no longer matches")
    for receipt in hardware_receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("rotation source-hardware receipt is malformed")
        path = Path(str(receipt.get("path", ""))).resolve()
        if receipt.get("sha256") != _file_hash(path):
            raise ValueError("rotation source-hardware artifact changed after planning")
    schedule_file = Path(schedule_path).resolve()
    declared = plan.get("rotation_schedule")
    if (
        not isinstance(declared, Mapping)
        or declared.get("path") != str(schedule_file)
        or declared.get("file_sha256") != _file_hash(schedule_file)
    ):
        raise ValueError("rotation schedule differs from the immutable plan")
    schedule = load_refinement_schedule(schedule_file)
    if declared.get("schedule_hash") != schedule.canonical_hash:
        raise ValueError("rotation schedule hash differs from the immutable plan")
    merged = load_refinement_merged_results(schedule, merge_path)
    return plan, schedule, merged


def _validate_cost_raw(
    raw: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    schedule: RefinementSchedule,
    merged: RefinementMergedResults,
) -> tuple[dict[str, Any], ...]:
    if (
        raw.get("schema_version") != COST_RAW_SCHEMA
        or raw.get("model_name") != TARGET_MODEL
        or raw.get("model_revision") != TARGET_REVISION
        or raw.get("method_id") != ATTENTION_ONLY_ROTATION_METHOD
        or raw.get("rotation_schedule_hash") != schedule.canonical_hash
        or raw.get("rotation_merge_content_hash")
        != merged.receipt.get("content_hash")
        or raw.get("sample_bundle_hash")
        != merged.receipt.get("sample_bundle_hash")
    ):
        raise ValueError("rotation cost measurement has the wrong study identity")
    _sha256(raw.get("measurement_harness_sha256"), "measurement harness SHA-256")
    _sha256(raw.get("runtime_environment_fingerprint"), "runtime fingerprint")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != len(schedule.entries):
        raise ValueError("rotation cost measurement coverage is not exact")
    ancestry = {
        str(item["rotation_profile_id"]): item
        for item in plan["source_ancestry"]
    }
    terminal = {
        str(row["profile_id"]): row for row in merged.terminal_rows
    }
    observed: dict[str, dict[str, Any]] = {}
    required = {
        "source_profile_id",
        "rotation_profile_id",
        "base_refinement_record_hash",
        "rotation_refinement_record_hash",
        "device_name",
        "device_uuid",
        "timing_source",
        "power_source",
        "decode_batch_size",
        "prompt_tokens",
        "decode_steps",
        "warmup_decode_steps",
        "paired_repetitions",
        "measured_decode_tokens",
        "baseline_tpot_ms",
        "rotation_tpot_ms",
        "baseline_tokens_per_second",
        "rotation_tokens_per_second",
        "baseline_average_power_w",
        "rotation_average_power_w",
        "baseline_energy_per_token_j",
        "rotation_energy_per_token_j",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("rotation cost row fields differ from the sealed schema")
        profile_id = str(row["rotation_profile_id"])
        parent = ancestry.get(profile_id)
        result = terminal.get(profile_id)
        if (
            profile_id in observed
            or parent is None
            or result is None
            or result.get("state") != "succeeded"
            or row["source_profile_id"] != parent["source_profile_id"]
            or row["base_refinement_record_hash"]
            != parent["base_refinement_record_hash"]
            or row["rotation_refinement_record_hash"] != result.get("record_hash")
        ):
            raise ValueError("rotation cost row ancestry or terminal binding is invalid")
        if (
            row["timing_source"] not in TIMING_SOURCES
            or row["power_source"] not in POWER_SOURCES
            or not str(row["device_name"]).strip()
            or not str(row["device_uuid"]).strip()
        ):
            raise ValueError("rotation cost measurement provenance is invalid")
        for name, minimum in (
            ("decode_batch_size", 1),
            ("prompt_tokens", 512),
            ("decode_steps", 128),
            ("warmup_decode_steps", 32),
            ("paired_repetitions", 10),
            ("measured_decode_tokens", 1280),
        ):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"rotation cost {name} is below its measurement floor")
        for name in required - {
            "source_profile_id",
            "rotation_profile_id",
            "base_refinement_record_hash",
            "rotation_refinement_record_hash",
            "device_name",
            "device_uuid",
            "timing_source",
            "power_source",
            "decode_batch_size",
            "prompt_tokens",
            "decode_steps",
            "warmup_decode_steps",
            "paired_repetitions",
            "measured_decode_tokens",
        }:
            _positive(row[name], f"rotation cost {name}")
        observed[profile_id] = dict(row)
    if set(observed) != {entry.profile_id for entry in schedule.entries}:
        raise ValueError("rotation cost measurement omits a scheduled profile")
    return tuple(observed[entry.profile_id] for entry in schedule.entries)


def bind_rotation_cost(
    *,
    plan_path: str | Path,
    schedule_path: str | Path,
    merge_path: str | Path,
    raw_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    plan, schedule, merged = _load_bound_plan(plan_path, schedule_path, merge_path)
    if any(row.get("state") != "succeeded" for row in merged.terminal_rows):
        raise ValueError("rotation cost cannot bind before every row succeeds")
    raw_file = Path(raw_path).resolve()
    raw = load_immutable_json(raw_file)
    rows = _validate_cost_raw(raw, plan=plan, schedule=schedule, merged=merged)
    receipt = {
        "schema_version": COST_RECEIPT_SCHEMA,
        "method_id": ATTENTION_ONLY_ROTATION_METHOD,
        "plan_content_hash": plan["content_hash"],
        "rotation_schedule_hash": schedule.canonical_hash,
        "rotation_merge_content_hash": merged.receipt["content_hash"],
        "raw_measurement": {
            "path": str(raw_file),
            "sha256": _file_hash(raw_file),
            "content_hash": raw["content_hash"],
        },
        "rows": list(rows),
        "evidence_class": "measured_b200_software_decode_cost_non_publication",
        "plena_hardware_cost_status": "unmeasured_rotation_lowering",
        "publication_rankable": False,
        "selection_eligible": False,
    }
    write_immutable_json(output_path, receipt)
    return receipt


def _verify_rotation_checkpoints(
    schedule: RefinementSchedule,
    merged: RefinementMergedResults,
    checkpoint_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    root = Path(checkpoint_root).resolve()
    contracts: dict[str, tuple[RefinementBankSpec, Path, Mapping[str, Any]]] = {}
    for path in sorted(root.rglob("bank_contract.json")):
        value = load_immutable_json(path)
        if value.get("schema_version") != "decode-refinement-bank-contract":
            raise ValueError(f"unsupported rotation bank contract: {path}")
        bank = RefinementBankSpec.from_dict(value["bank"])
        if bank.bank_id in contracts:
            raise ValueError("rotation checkpoint root repeats a bank ID")
        contracts[bank.bank_id] = (bank, path.resolve(), value)
    entry_by_id = {entry.profile_id: entry for entry in schedule.entries}
    evidence = []
    for row in merged.terminal_rows:
        profile_id = str(row["profile_id"])
        entry = entry_by_id[profile_id]
        item = contracts.get(str(row.get("bank_id", "")))
        if item is None:
            raise ValueError("successful rotation row has no exact bank contract")
        bank, contract_path, contract = item
        checkpoint = Path(bank.checkpoint_dir).resolve()
        if (
            bank.weight_method != "rotation"
            or bank.rotation_profile_id != profile_id
            or bank.rotation_config_hash is None
            or not checkpoint.is_relative_to(root)
            or contract.get("rotation_policy")
            != rotation_decision_contract(entry.profile)
        ):
            raise ValueError("rotation bank does not implement the sealed profile scope")
        decision_path = checkpoint / "rotation_decisions.json"
        if not decision_path.is_file():
            raise ValueError("rotation bank has no decision artifact")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        summary = validate_attention_only_rotation_decision(decision)
        execution = row.get("result", {}).get("execution_evidence")
        observed_tree = _tree_hash(checkpoint)
        if (
            not isinstance(execution, Mapping)
            or execution.get("checkpoint_tree_sha256") != observed_tree
        ):
            raise ValueError("rotation checkpoint tree changed after measurement")
        evidence.append(
            {
                "profile_id": profile_id,
                "bank_id": bank.bank_id,
                "bank_contract_path": str(contract_path),
                "bank_contract_sha256": _file_hash(contract_path),
                "checkpoint_path": str(checkpoint),
                "checkpoint_tree_sha256": observed_tree,
                "decision_path": str(decision_path),
                "decision_sha256": _file_hash(decision_path),
                "decision": summary,
            }
        )
    return tuple(evidence)


def finalize_rotation_ablation(
    *,
    plan_path: str | Path,
    schedule_path: str | Path,
    merge_path: str | Path,
    cost_receipt_path: str | Path,
    checkpoint_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    plan, schedule, merged = _load_bound_plan(plan_path, schedule_path, merge_path)
    rows = {str(row["profile_id"]): row for row in merged.terminal_rows}
    if len(rows) != 4 or any(row.get("state") != "succeeded" for row in rows.values()):
        raise ValueError("rotation ablation requires four exact terminal successes")
    cost_file = Path(cost_receipt_path).resolve()
    cost = load_immutable_json(cost_file)
    if (
        cost.get("schema_version") != COST_RECEIPT_SCHEMA
        or cost.get("plan_content_hash") != plan["content_hash"]
        or cost.get("rotation_schedule_hash") != schedule.canonical_hash
        or cost.get("rotation_merge_content_hash") != merged.receipt["content_hash"]
        or cost.get("publication_rankable") is not False
        or cost.get("selection_eligible") is not False
    ):
        raise ValueError("rotation cost receipt differs from this completed run")
    raw_receipt = cost.get("raw_measurement")
    if not isinstance(raw_receipt, Mapping):
        raise ValueError("rotation cost receipt has no raw measurement identity")
    raw_file = Path(str(raw_receipt.get("path", ""))).resolve()
    raw = load_immutable_json(raw_file)
    if (
        raw_receipt.get("sha256") != _file_hash(raw_file)
        or raw_receipt.get("content_hash") != raw.get("content_hash")
    ):
        raise ValueError("raw rotation cost measurement changed after binding")
    validated_cost_rows = _validate_cost_raw(
        raw, plan=plan, schedule=schedule, merged=merged
    )
    if cost.get("rows") != list(validated_cost_rows):
        raise ValueError("rotation cost receipt rows differ from raw measurement")
    checkpoint_evidence = _verify_rotation_checkpoints(
        schedule, merged, checkpoint_root
    )
    ancestry = {
        str(item["rotation_profile_id"]): item
        for item in plan["source_ancestry"]
    }
    numerical = []
    for entry in schedule.entries:
        row = rows[entry.profile_id]
        value = float(row["result"]["mean_token_nll"])
        base = float(ancestry[entry.profile_id]["base_mean_token_nll"])
        numerical.append(
            {
                "source_profile_id": entry.profile.source_profile.profile_id,
                "rotation_profile_id": entry.profile_id,
                "base_refinement_record_hash": ancestry[entry.profile_id][
                    "base_refinement_record_hash"
                ],
                "rotation_refinement_record_hash": row["record_hash"],
                "base_mean_token_nll": base,
                "rotation_mean_token_nll": value,
                "mean_token_nll_delta": value - base,
                "numerically_improved_on_refinement_bundle": value < base,
            }
        )
    receipt = {
        "schema_version": FINAL_SCHEMA,
        "model_name": TARGET_MODEL,
        "model_revision": TARGET_REVISION,
        "method_id": ATTENTION_ONLY_ROTATION_METHOD,
        "plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": _file_hash(plan_path),
            "content_hash": plan["content_hash"],
        },
        "rotation_schedule": {
            "path": str(Path(schedule_path).resolve()),
            "sha256": _file_hash(schedule_path),
            "schedule_hash": schedule.canonical_hash,
        },
        "rotation_merge": {
            "path": str(Path(merge_path).resolve()),
            "sha256": _file_hash(merge_path),
            "content_hash": merged.receipt["content_hash"],
            "results_path": str(merged.results_path),
            "results_sha256": merged.results_sha256,
        },
        "cost_receipt": {
            "path": str(cost_file),
            "sha256": _file_hash(cost_file),
            "content_hash": cost["content_hash"],
        },
        "numerical_comparisons": numerical,
        "checkpoint_evidence": list(checkpoint_evidence),
        "evidence_class": "measured_attention_only_rotation_ablation_non_publication",
        "publication_rankable": False,
        "selection_eligible": False,
        "remaining_blockers": [
            "no_compiler_lowering_for_attention_hadamard_rotation",
            "no_emulator_execution_for_attention_hadamard_rotation",
            "no_plena_timing_power_or_area_measurement_for_rotation",
        ],
    }
    write_immutable_json(output_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("materialize")
    plan.add_argument("--config", required=True)
    plan.add_argument("--base-schedule", required=True)
    plan.add_argument("--base-merge", required=True)
    plan.add_argument("--source-selection", required=True)
    plan.add_argument("--hardware-artifact", action="append", required=True)
    plan.add_argument("--schedule", required=True)
    plan.add_argument("--plan", required=True)
    cost = commands.add_parser("bind-cost")
    cost.add_argument("--plan", required=True)
    cost.add_argument("--schedule", required=True)
    cost.add_argument("--merge", required=True)
    cost.add_argument("--raw", required=True)
    cost.add_argument("--out", required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--plan", required=True)
    final.add_argument("--schedule", required=True)
    final.add_argument("--merge", required=True)
    final.add_argument("--cost-receipt", required=True)
    final.add_argument("--checkpoint-root", required=True)
    final.add_argument("--out", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.command == "materialize":
            result = materialize_rotation_plan(
                config_path=args.config,
                base_schedule_path=args.base_schedule,
                base_merge_path=args.base_merge,
                source_selection_path=args.source_selection,
                hardware_paths=args.hardware_artifact,
                schedule_path=args.schedule,
                plan_path=args.plan,
            )
        elif args.command == "bind-cost":
            result = bind_rotation_cost(
                plan_path=args.plan,
                schedule_path=args.schedule,
                merge_path=args.merge,
                raw_path=args.raw,
                output_path=args.out,
            )
        else:
            result = finalize_rotation_ablation(
                plan_path=args.plan,
                schedule_path=args.schedule,
                merge_path=args.merge,
                cost_receipt_path=args.cost_receipt,
                checkpoint_root=args.checkpoint_root,
                output_path=args.out,
            )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTENTION_ONLY_ROTATION_METHOD",
    "COST_RAW_SCHEMA",
    "COST_RECEIPT_SCHEMA",
    "FINAL_SCHEMA",
    "PLAN_SCHEMA",
    "bind_rotation_cost",
    "finalize_rotation_ablation",
    "main",
    "materialize_rotation_plan",
    "select_measured_rotation_sources",
]
