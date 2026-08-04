"""Build the workspace stack-validity artifact from measured stage reports.

The artifact records, per hardware-validation profile, whether the compiler
and emulator stages passed their measured validation. It binds the workspace
run plan and sweep manifest by canonical hash, so it must be produced against
the planned workspace it will gate — never generated elsewhere and copied.

Inputs are the two stage reports emitted by the simulator's
``build_stack_stage_reports`` tool (timestamped compiler and emulator runs
with per-artifact SHA-256 bindings) and the retained emulator-calibration
artifacts whose ids both reports must carry. The builder refuses to emit when
any scoped profile's structural capability floor forbids a required stage, so
a document the launch gate would reject is never written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from decode_dse.legality import evaluate_stack_capability
from decode_dse.manifest import SweepManifest
from decode_dse.software.sweep_plan import SweepRunPlan, write_immutable_json

STAGE_REPORT_SCHEMA = "plena-stack-stage-report"
CALIBRATION_SCHEMA = "plena-decode-emulator-calibration"
RTL_SERIALIZED = "rtl_serialized"
REQUIRED_STAGES = ("compiler", "emulator")


def _canonical_content_hash(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_hash"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_stage_report(path: str | Path, *, stage: str) -> dict[str, Any]:
    """Load one measured stage report and verify its identity and shape."""

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError(f"{stage} stage report must be a JSON object")
    if report.get("schema") != STAGE_REPORT_SCHEMA:
        raise ValueError(f"{path} is not a {STAGE_REPORT_SCHEMA} document")
    if report.get("stage") != stage:
        raise ValueError(
            f"{path} records stage {report.get('stage')!r}, expected {stage!r}"
        )
    recorded_hash = report.get("content_hash")
    if recorded_hash != _canonical_content_hash(report):
        raise ValueError(f"{stage} stage report failed its content-hash check")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{stage} stage report lacks provenance")
    for field in ("started_at_utc", "completed_at_utc"):
        if not provenance.get(field):
            raise ValueError(f"{stage} stage report lacks {field}")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"{stage} stage report binds no artifacts")
    calibration_ids = report.get("calibration_ids")
    if not isinstance(calibration_ids, list) or not calibration_ids:
        raise ValueError(f"{stage} stage report binds no calibration ids")
    return dict(report)


def load_calibration_artifact(path: str | Path) -> dict[str, Any]:
    """Load one retained calibration artifact and verify its gate result."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"{path} is not a {CALIBRATION_SCHEMA} artifact")
    if value.get("passed") is not True:
        raise ValueError(f"{path} did not pass its stage-validation gate")
    contract = value.get("execution_contract", {})
    if contract.get("timing_mode") != RTL_SERIALIZED:
        raise ValueError(
            f"{path} was calibrated under timing mode "
            f"{contract.get('timing_mode')!r}, not {RTL_SERIALIZED!r}"
        )
    calibration_id = value.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError(f"{path} has no calibration id")
    return value


def build_stack_validity_document(
    *,
    manifest: SweepManifest,
    plan: SweepRunPlan,
    compiler_report: Mapping[str, Any],
    emulator_report: Mapping[str, Any],
    calibration_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Assemble the per-profile validity document the launch gate loads."""

    for stage, report in (
        ("compiler", compiler_report),
        ("emulator", emulator_report),
    ):
        bound = set(report["calibration_ids"])
        missing = sorted(set(calibration_ids) - bound)
        if missing:
            raise ValueError(
                f"{stage} stage report does not bind calibration ids: {missing}"
            )

    entries = {entry.profile_id: entry for entry in manifest.entries}
    profiles: dict[str, dict[str, Any]] = {}
    blocked: list[str] = []
    for profile_id in plan.hardware_validation_profile_ids:
        entry = entries.get(profile_id)
        if entry is None:
            raise ValueError(f"plan profile {profile_id} is not in the manifest")
        capability = evaluate_stack_capability(entry.profile)
        floor = capability.validity_floor
        if any(
            getattr(floor, f"{stage}_valid") is False for stage in REQUIRED_STAGES
        ):
            blocked.append(profile_id)
            continue
        profiles[profile_id] = {
            "software_valid": floor.software_valid,
            "compiler_valid": True,
            "emulator_valid": True,
            "rtl_valid": floor.rtl_valid,
            "dc_calibrated": floor.dc_calibrated,
        }
    if blocked:
        raise ValueError(
            "structural capability forbids a required stage for "
            f"{len(blocked)} scoped profiles (first: {blocked[0]}); "
            "the launch gate would reject this document"
        )

    return {
        "schema": "decode-stack-validity",
        "run_plan_hash": plan.canonical_hash,
        "manifest_hash": manifest.canonical_hash,
        "calibration_ids": sorted(calibration_ids),
        "profiles": profiles,
        "source_reports": {
            "compiler": dict(compiler_report),
            "emulator": dict(emulator_report),
        },
    }


def build_stack_validity_artifact(
    *,
    manifest: SweepManifest,
    plan: SweepRunPlan,
    compiler_report_path: str | Path,
    emulator_report_path: str | Path,
    calibration_paths: tuple[Path, ...],
    destination: str | Path,
) -> dict[str, Any]:
    """Validate every input and write the immutable workspace artifact."""

    compiler_report = load_stage_report(compiler_report_path, stage="compiler")
    emulator_report = load_stage_report(emulator_report_path, stage="emulator")
    if not calibration_paths:
        raise ValueError("at least one retained calibration artifact is required")
    calibration_ids = tuple(
        sorted(
            load_calibration_artifact(path)["calibration_id"]
            for path in calibration_paths
        )
    )
    document = build_stack_validity_document(
        manifest=manifest,
        plan=plan,
        compiler_report=compiler_report,
        emulator_report=emulator_report,
        calibration_ids=calibration_ids,
    )
    write_immutable_json(destination, document)
    return document
