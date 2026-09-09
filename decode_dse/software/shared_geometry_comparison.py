"""One-row replay adapter for the matched shared-PLENA geometry ablation.

The exhaustive hardware grid remains unchanged.  This adapter starts from one
source-verified selected decode row, replays it, derives the 65,536-multiplier
decode-specialized arm by changing BLEN only, and constructs the paper-geometry
arm with ``BLEN=32, MLEN=VLEN=2048``.  Both arms are invoked through the same
reconstructed ``ProductionHardwareEvaluator`` before they are handed to the
simulator-owned comparison contract.

Hardware timing is rerun by the strict loader.  Exact-MLEN numerical rows are
instead normal-scientific-provenance evidence: their producer files, sample
populations, and arithmetic ledgers are revalidated, but the GPU kernels are
not independently re-executed and no adversarial/cryptographic authentication
is claimed.  That boundary is sealed in every v2 producer receipt.

The public entry point :func:`materialize_from_campaign` consumes the normal
sealed hardware-launch arguments; it does not open a second DSE grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from decode_dse.hardware.design_space import (
    HardwareCandidate,
    HardwareEvaluation,
    load_hardware_artifact,
    physical_cost_signature,
)
from decode_dse.hardware.lm_head_service import (
    DECODE_MX_HEAD,
    local_mx_head_breakdown_valid,
    local_mx_head_status_valid,
)
from decode_dse.hardware.power_bridge import resolve_simulator_root


CAMPAIGN_SCHEMA = "plena-qwen30b-moe-campaign/v1"
SELECTED_SOURCE_SCHEMA = "plena-campaign-selected-decode-source-receipt/v1"
PRODUCER_RECEIPT_SCHEMA = (
    "plena-matched-decode-comparison-producer-receipt/v2"
)
PRODUCER_EVIDENCE_CLASS = (
    "hardware_replay_validated_source_hash_bound_numerical_projection"
)
NUMERICAL_EVIDENCE_CLASS = (
    "source_hash_bound_measured_numerical_nonadversarial"
)
ANALYTIC_HANDOFF_SCHEMA = "plena-config-bound-analytic-handoff/v1"
NUMERICAL_EVIDENCE_SCHEMA = "plena-matched-mlen-evidence-receipt/v2"
EVALUATOR_REPLAY_SCHEMA = "plena-hardware-evaluator-replay-invocation/v1"
ORCHESTRATION_SCHEMA = "plena-matched-decode-comparison-orchestration/v1"

_NO_WRITE_OUTPUT = "__plena_matched_geometry_replay_no_write__"
_PATH_OPTIONS = {
    "--manifest",
    "--numerical-jsonl",
    "--refinement-schedule",
    "--refinement-merge",
    "--refinement-results",
    "--config",
    "--timing-evidence",
    "--compiler-trace-artifacts",
    "--request-memory-calibration",
    "--power-calibration",
    "--area-config",
    "--exact-dc-anchors",
    "--head-service-calibration",
    "--head-service-resource-receipt",
    "--admission-receipt",
    "--handoff-artifact",
    "--settings-toml",
    "--isa-path",
}
_FORBIDDEN_REPLAY_OPTIONS = {"--profile-id", "--allow-incomplete"}
_ORCHESTRATION_FIELDS = {
    "schema_version",
    "selection_role",
    "shared_geometry",
    "specialized_derivation_rule",
    "metric_whitelist",
    "forbidden_metrics",
    "claim_scope",
    "handoff",
    "numerical",
    "extensible_models",
    "content_hash",
}
_EVALUATOR_REPLAY_FIELDS = {
    "schema_version",
    "argv",
    "argv_sha256",
    "input_receipts",
    "config_path",
    "config_file_sha256",
    "orchestration_contract_path",
    "orchestration_contract_file_sha256",
    "orchestration_contract_content_sha256",
    "receipt_id",
}

_SELECTED_RECEIPT_FIELDS = {
    "schema_version",
    "campaign_path",
    "campaign_file_sha256",
    "campaign_content_sha256",
    "selection_role",
    "hardware_artifact_paths",
    "hardware_artifact_sha256s",
    "selected_artifact_path",
    "selected_artifact_sha256",
    "selected_artifact_provenance_sha256",
    "selected_profile_id",
    "selected_candidate_id",
    "selected_record_hash",
    "selected_row_sha256",
    "model_sha256",
    "workload_sha256",
    "phase_contract_sha256",
    "selector_script_path",
    "selector_script_sha256",
    "selector_replay_output_sha256",
    "evaluator_id",
    "evaluator_provenance_sha256",
    "receipt_id",
}
_PRODUCER_FIELDS = {
    "schema_version",
    "evidence_class",
    "publication_rankable",
    "claim_boundary",
    "source_receipt",
    "numerical_evidence_receipt",
    "evaluator_replay",
    "comparison",
    "replay",
    "producer",
    "receipt_id",
    "content_sha256",
}
_PRODUCER_METADATA_FIELDS = {
    "adapter_path",
    "adapter_sha256",
    "simulator_core_path",
    "simulator_core_sha256",
}


@dataclass(frozen=True)
class ResolvedCampaignSource:
    """Campaign row that has been re-resolved from producer files.

    Callers cannot authorize repricing with a hand-authored normalized
    mapping: :func:`reprice_matched_pair` accepts this type and resolves its
    campaign and hardware artifact paths again before executing the evaluator.
    """

    receipt: Mapping[str, Any]
    row: Mapping[str, Any]
    header: Mapping[str, Any]


@dataclass(frozen=True)
class ResolvedNumericalEvidence:
    """Source-hash-bound same-split controls from one sealed completion."""

    receipt: Mapping[str, Any]
    specialized_entry: Any
    shared_entry: Any
    specialized_result: Mapping[str, Any]
    shared_result: Mapping[str, Any]
    specialized_receipt: Mapping[str, Any]
    shared_receipt: Mapping[str, Any]
    bf16_oracle: Mapping[str, Any]


def _core():
    root = resolve_simulator_root().root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from analytic_models.performance import matched_decode_comparison

    module_path = Path(matched_decode_comparison.__file__).resolve()
    expected_path = (
        root / "analytic_models" / "performance" / "matched_decode_comparison.py"
    ).resolve()
    if module_path != expected_path:
        raise RuntimeError(
            "matched-comparison core was imported from a different Simulator "
            f"root: expected {expected_path}, observed {module_path}"
        )
    return matched_decode_comparison


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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _immutable_json(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} root must be an object")
    result = json.loads(_canonical_bytes(dict(value)))
    expected = result.pop("content_hash", None)
    if expected != _content_hash(result):
        raise ValueError(f"{label} content hash differs")
    result["content_hash"] = expected
    return result


def _path_receipt(option: str, path: str | Path) -> dict[str, Any]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"replay input cannot be a symlink: {supplied}")
    source = supplied.resolve()
    if source.is_file():
        return {
            "option": option,
            "path": str(source),
            "kind": "file",
            "entries": [],
            "content_sha256": _file_hash(source),
        }
    if not source.is_dir():
        raise ValueError(f"replay input does not exist: {source}")
    entries = []
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"replay input tree contains a symlink: {child}")
        if child.is_file():
            entries.append(
                {
                    "relative_path": child.relative_to(source).as_posix(),
                    "sha256": _file_hash(child),
                }
            )
    if not entries:
        raise ValueError(f"replay input directory is empty: {source}")
    return {
        "option": option,
        "path": str(source),
        "kind": "directory",
        "entries": entries,
        "content_sha256": _content_hash(entries),
    }


def _validate_orchestration_contract(
    raw: Any,
    *,
    expected_model: str | None = None,
) -> dict[str, Any]:
    contract = _object(raw, _ORCHESTRATION_FIELDS, "orchestration contract")
    expected_hash = contract.pop("content_hash")
    if expected_hash != _content_hash(contract):
        raise ValueError("orchestration contract content hash differs")
    contract["content_hash"] = expected_hash
    if contract["schema_version"] != ORCHESTRATION_SCHEMA:
        raise ValueError("unsupported orchestration contract schema")
    if contract["selection_role"] != "balanced_source":
        raise ValueError("v1 orchestration supports only balanced_source")
    if contract["shared_geometry"] != {
        "MLEN": 2048,
        "BLEN": 32,
        "VLEN": 2048,
    }:
        raise ValueError("orchestration shared geometry differs from PLENA")
    if contract["specialized_derivation_rule"] != (
        "preserve_all_axes_except_blen_set_blen_to_65536_div_mlen"
    ):
        raise ValueError("orchestration specialized derivation differs")
    required_metrics = {
        "decode_stage_tpot_ms",
        "one_time_handoff_ms",
        "handoff_amortized_decode_side_service_ms",
        "relative_perplexity_vs_bf16",
    }
    if set(contract["metric_whitelist"]) != required_metrics:
        raise ValueError("orchestration metric whitelist differs")
    if set(contract["forbidden_metrics"]) != {"goodput"}:
        raise ValueError("orchestration must explicitly forbid goodput")
    if contract["claim_scope"] != "controlled_decode_stage_geometry_ablation":
        raise ValueError("orchestration claim scope differs")
    models = contract["extensible_models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(value, str) or not value for value in models)
        or len(models) != len(set(models))
        or (expected_model is not None and expected_model not in models)
    ):
        raise ValueError("orchestration model extension list is invalid")
    handoff = contract["handoff"]
    if not isinstance(handoff, Mapping) or set(handoff) != {
        "mode",
        "analytic_contract",
    }:
        raise ValueError("orchestration handoff contract differs")
    mode = handoff["mode"]
    if mode == "measured_prefill_artifact":
        if handoff["analytic_contract"] is not None:
            raise ValueError("measured handoff cannot carry an analytic contract")
    elif mode == "config_bound_analytic":
        if not isinstance(handoff["analytic_contract"], Mapping):
            raise ValueError("analytic handoff contract is missing")
    else:
        raise ValueError("orchestration handoff mode is unsupported")
    numerical = contract["numerical"]
    if not isinstance(numerical, Mapping) or set(numerical) != {
        "source_mlen",
        "shared_mlen",
        "suite",
        "same_plan_bf16_required",
    }:
        raise ValueError("orchestration numerical contract differs")
    if (
        isinstance(numerical["source_mlen"], bool)
        or not isinstance(numerical["source_mlen"], int)
        or numerical["source_mlen"] <= 0
        or numerical["shared_mlen"] != 2048
        or numerical["suite"] not in {"validation", "refinement"}
        or numerical["same_plan_bf16_required"] is not True
    ):
        raise ValueError("orchestration numerical controls are invalid")
    return contract


def _canonical_launch_argv(argv: Sequence[str]) -> tuple[str, ...]:
    raw = [str(value) for value in argv]
    if not raw:
        raise ValueError("hardware replay argv is empty")
    if any("=" in value and value.startswith("--") for value in raw):
        raise ValueError("hardware replay argv requires separated option values")
    if any(value in _FORBIDDEN_REPLAY_OPTIONS for value in raw):
        raise ValueError(
            "hardware replay forbids profile filters and incomplete coverage"
        )
    result: list[str] = []
    index = 0
    output_count = 0
    while index < len(raw):
        option = raw[index]
        if option == "--output":
            if index + 1 >= len(raw):
                raise ValueError("hardware replay output value is missing")
            output_count += 1
            result.extend((option, _NO_WRITE_OUTPUT))
            index += 2
            continue
        if option == "--parallel-workers":
            if index + 1 >= len(raw) or raw[index + 1] != "1":
                raise ValueError("hardware replay requires one worker")
            result.extend((option, "1"))
            index += 2
            continue
        if option in _PATH_OPTIONS:
            if index + 1 >= len(raw) or raw[index + 1].startswith("--"):
                raise ValueError(f"hardware replay path is missing for {option}")
            result.extend((option, str(Path(raw[index + 1]).resolve())))
            index += 2
            continue
        result.append(option)
        index += 1
    if output_count != 1:
        raise ValueError("hardware replay requires exactly one inert output option")
    return tuple(result)


def evaluator_replay_receipt_id(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_id", None)
    return "hardware-evaluator-replay-" + _content_hash(body)


def build_evaluator_replay_invocation(
    hardware_launch_argv: Sequence[str],
    *,
    orchestration_contract_path: str | Path,
) -> dict[str, Any]:
    """Seal every evaluator input plus the comparison-control contract."""

    argv = _canonical_launch_argv(hardware_launch_argv)
    receipts = []
    config_paths = []
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in _PATH_OPTIONS:
            path = argv[index + 1]
            receipts.append(_path_receipt(option, path))
            if option == "--config":
                config_paths.append(path)
            index += 2
            continue
        index += 1
    if len(config_paths) != 1:
        raise ValueError("hardware replay requires exactly one config")
    contract_path = Path(orchestration_contract_path).resolve()
    contract = _validate_orchestration_contract(
        _immutable_json(contract_path, "orchestration contract")
    )
    body = {
        "schema_version": EVALUATOR_REPLAY_SCHEMA,
        "argv": list(argv),
        "argv_sha256": _content_hash(list(argv)),
        "input_receipts": receipts,
        "config_path": config_paths[0],
        "config_file_sha256": _file_hash(config_paths[0]),
        "orchestration_contract_path": str(contract_path),
        "orchestration_contract_file_sha256": _file_hash(contract_path),
        "orchestration_contract_content_sha256": contract["content_hash"],
    }
    return {**body, "receipt_id": evaluator_replay_receipt_id(body)}


def validate_evaluator_replay_invocation(raw: Any) -> dict[str, Any]:
    receipt = _object(raw, _EVALUATOR_REPLAY_FIELDS, "evaluator replay")
    if receipt["schema_version"] != EVALUATOR_REPLAY_SCHEMA:
        raise ValueError("unsupported evaluator replay schema")
    if tuple(receipt["argv"]) != _canonical_launch_argv(receipt["argv"]):
        raise ValueError("evaluator replay argv is not canonical")
    if receipt["argv_sha256"] != _content_hash(receipt["argv"]):
        raise ValueError("evaluator replay argv hash differs")
    if receipt["receipt_id"] != evaluator_replay_receipt_id(receipt):
        raise ValueError("evaluator replay identity differs")
    expected = build_evaluator_replay_invocation(
        receipt["argv"],
        orchestration_contract_path=receipt["orchestration_contract_path"],
    )
    if receipt != expected:
        raise ValueError("evaluator replay input bytes or contract changed")
    return receipt


def _sha256(value: Any, label: str) -> str:
    token = str(value)
    if (
        len(token) != 64
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields differ from the schema")
    return json.loads(_canonical_bytes(dict(value)))


def selected_source_receipt_id(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_id", None)
    return "selected-decode-source-" + _content_hash(body)


def validate_selected_source_receipt(raw: Any) -> dict[str, Any]:
    """Validate a normalized campaign-source receipt.

    This does not authenticate the producer files by itself.  Only
    :func:`resolve_campaign_source` returns the opaque object accepted by the
    repricer.
    """

    receipt = _object(raw, _SELECTED_RECEIPT_FIELDS, "selected source receipt")
    if receipt["schema_version"] != SELECTED_SOURCE_SCHEMA:
        raise ValueError("unsupported selected-source receipt schema")
    for name in (
        "campaign_path",
        "selection_role",
        "selected_artifact_path",
        "selected_profile_id",
        "selected_candidate_id",
        "selector_script_path",
    ):
        if not isinstance(receipt[name], str) or not receipt[name]:
            raise ValueError(f"selected source {name} must be explicit")
    if receipt["selection_role"] != "balanced_source":
        raise ValueError("v1 strict producer supports only balanced_source")
    paths = receipt["hardware_artifact_paths"]
    digests = receipt["hardware_artifact_sha256s"]
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(value, str) or not value for value in paths)
        or len(paths) != len(set(paths))
        or not isinstance(digests, list)
        or len(digests) != len(paths)
        or digests != sorted(set(digests))
    ):
        raise ValueError("selected source hardware-artifact set is invalid")
    for name in (
        "campaign_file_sha256",
        "campaign_content_sha256",
        "selected_artifact_sha256",
        "selected_artifact_provenance_sha256",
        "selected_record_hash",
        "selected_row_sha256",
        "model_sha256",
        "workload_sha256",
        "phase_contract_sha256",
        "selector_script_sha256",
        "selector_replay_output_sha256",
        "evaluator_provenance_sha256",
    ):
        _sha256(receipt[name], name)
    for digest in digests:
        _sha256(digest, "hardware artifact SHA-256")
    if receipt["selected_artifact_sha256"] not in digests:
        raise ValueError("selected artifact is outside the campaign artifact set")
    if not isinstance(receipt["evaluator_id"], str) or not receipt["evaluator_id"]:
        raise ValueError("selected source evaluator identity must be explicit")
    if receipt["receipt_id"] != selected_source_receipt_id(receipt):
        raise ValueError("selected-source receipt identity is inconsistent")
    return receipt


def _load_campaign(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_bytes())
    if not isinstance(value, Mapping):
        raise TypeError("campaign root must be an object")
    campaign = json.loads(_canonical_bytes(dict(value)))
    expected = campaign.pop("content_hash", None)
    if expected != _content_hash(campaign):
        raise ValueError("campaign content hash differs")
    campaign["content_hash"] = expected
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported campaign schema")
    return campaign


def _campaign_candidate(
    campaign: Mapping[str, Any], selection_role: str
) -> dict[str, Any]:
    if selection_role != "balanced_source":
        raise ValueError("v1 strict producer supports only balanced_source")
    selection = campaign.get("selection")
    variants = (
        selection.get("calibrated_variants")
        if isinstance(selection, Mapping)
        else None
    )
    geometry_revalidation = campaign.get("geometry_revalidation")
    if (
        not isinstance(selection, Mapping)
        or selection.get("evidence_mode") != "exploratory"
        or selection.get("projected_joint_source_bound") is not True
        or selection.get("whole_study_publication_candidate") is not False
        or not isinstance(variants, Mapping)
        or variants.get("emitted_methods") != []
        or variants.get("required_methods") not in (None, [])
        or not isinstance(geometry_revalidation, Mapping)
        or geometry_revalidation.get("corrected_numerical_input") is not None
    ):
        raise ValueError(
            "v1 balanced-source replay requires the projected exploratory "
            "campaign frozen before calibrated or corrected-MLEN coverage"
        )
    rows = campaign.get("top_co_design_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("campaign top-row selector population is missing")
    normalized: list[tuple[float, tuple[str, str], dict[str, Any], int]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("campaign top-row selector entry must be an object")
        profile_id = str(raw.get("profile_id", ""))
        candidate_id = str(raw.get("candidate_id", ""))
        score = raw.get("balanced_score")
        rank = raw.get("rank")
        if (
            not profile_id
            or not candidate_id
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
        ):
            raise ValueError("campaign top-row selector identity is invalid")
        payload = {
            key: item
            for key, item in raw.items()
            if key not in {"rank", "balanced_score", "roles"}
        }
        normalized.append(
            (float(score), (profile_id, candidate_id), payload, rank)
        )
    ranks = [item[3] for item in normalized]
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("campaign top-row ranks are not a permutation")
    # The selector deliberately removes extrema from the per-profile top-row
    # display when another candidate from that profile was already retained.
    # Consequently the deterministic overall choice is not necessarily the
    # minimum *displayed* top row.  The producer-owned selector seals it in
    # metric_extrema.overall, and the byte-for-byte selector replay below
    # independently recomputes that object from the source-verified hardware
    # streams.
    extrema = campaign.get("metric_extrema")
    overall = extrema.get("overall") if isinstance(extrema, Mapping) else None
    value = campaign.get("balanced_source")
    if not isinstance(overall, Mapping) or not isinstance(value, Mapping):
        raise ValueError("campaign deterministic overall selector is missing")
    if dict(value) != dict(overall):
        raise ValueError(
            "campaign balanced_source differs from metric_extrema.overall"
        )
    if not isinstance(value, Mapping):
        raise ValueError("campaign selected source is missing")
    candidate = json.loads(_canonical_bytes(dict(value)))
    for name in ("profile_id", "candidate_id", "record_hash"):
        if not isinstance(candidate.get(name), str) or not candidate[name]:
            raise ValueError(f"campaign selected source lacks {name}")
    _sha256(candidate["record_hash"], "campaign selected record")
    for name in ("profile", "hardware", "accuracy"):
        if not isinstance(candidate.get(name), Mapping):
            raise ValueError(f"campaign selected source lacks {name}")
    return candidate


def _selector_script(campaign_path: Path) -> Path:
    # Campaign data may be copied for archival or publication, but it cannot
    # bring a nearby executable and thereby redefine how ``balanced_source``
    # was selected.  Replay is owned exclusively by the canonical Results
    # checkout paired with this Software tree.
    canonical = (
        Path(__file__).resolve().parents[3]
        / "PLENA_Qwen30B_Moe_Results"
        / "scripts"
        / "select_campaign.py"
    ).resolve()
    if not canonical.is_file() or canonical.is_symlink():
        raise ValueError("canonical Results campaign selector is unavailable")
    return canonical


def _replay_campaign_selector(
    campaign_path: Path,
    *,
    hardware_artifact_paths: Sequence[Path],
    campaign: Mapping[str, Any],
) -> dict[str, str]:
    """Rerun the projected selector and require byte-identical semantics."""

    # Structural validation fixes the only v1 role before running any code.
    _campaign_candidate(campaign, "balanced_source")
    selection = campaign["selection"]
    supplied_by_sha = {
        _file_hash(path): path.resolve() for path in hardware_artifact_paths
    }
    declared = campaign.get("hardware_artifacts")
    if not isinstance(declared, list) or {
        str(value.get("sha256"))
        for value in declared
        if isinstance(value, Mapping)
    } != set(supplied_by_sha):
        raise ValueError("selector replay hardware artifacts differ")
    ordered_hardware = [
        supplied_by_sha[str(value["sha256"])] for value in declared
    ]
    reference = campaign.get("reference_artifact")
    if (
        not isinstance(reference, Mapping)
        or reference.get("schema_version")
        != "plena-qwen30b-moe-projected-joint-completion/v2"
        or not isinstance(reference.get("path"), str)
        or campaign.get("refinement_artifact") is not None
    ):
        raise ValueError(
            "v1 selector replay requires one projected-joint reference and "
            "no refinement artifact"
        )
    script = _selector_script(campaign_path)
    software_root = Path(__file__).resolve().parents[2]
    argv: list[str] = []
    for path in ordered_hardware:
        argv.extend(("--hardware-artifact", str(path.resolve())))
    argv.extend(
        (
            "--projected-joint-receipt",
            str(Path(reference["path"]).resolve()),
            "--strict-relative-ppl",
            str(selection["strict_relative_perplexity"]),
            "--relaxed-relative-ppl",
            str(selection["relaxed_relative_perplexity"]),
            "--top-k",
            str(selection["top_k"]),
            "--evidence-mode",
            "exploratory",
            "--software-root",
            str(software_root),
        )
    )
    required_methods = selection["calibrated_variants"].get(
        "required_methods", []
    )
    if required_methods:
        raise ValueError("v1 selector replay forbids calibrated method requirements")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "campaign.json"
        command = [sys.executable, str(script), *argv, "--out", str(output)]
        completed = subprocess.run(
            command,
            cwd=script.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                "campaign selector replay failed: " + completed.stderr.strip()
            )
        replayed = _load_campaign(output)
        if replayed != campaign:
            raise ValueError("campaign differs from deterministic selector replay")
        replay_sha = _file_hash(output)
        if replay_sha != _file_hash(campaign_path):
            raise ValueError("campaign bytes differ from deterministic selector output")
    return {
        "selector_script_path": str(script),
        "selector_script_sha256": _file_hash(script),
        "selector_replay_output_sha256": replay_sha,
    }


def _row_body_sha256(row: Mapping[str, Any]) -> str:
    body = dict(row)
    body.pop("record_hash", None)
    body.pop("retention_labels", None)
    return _content_hash(body)


def _evaluator_identity(evaluator: Any) -> tuple[str, str]:
    evaluator_id = str(getattr(evaluator, "evaluator_id", ""))
    provenance = getattr(evaluator, "provenance", None)
    if not evaluator_id or not isinstance(provenance, Mapping):
        raise ValueError("evaluator identity/provenance is unavailable")
    return evaluator_id, _content_hash(dict(provenance))


def _validate_model_architecture(evaluator: Any, model: Mapping[str, Any]) -> None:
    architecture = model.get("model_architecture")
    simulator = getattr(getattr(evaluator, "backend", None), "sim", None)
    dimensions = getattr(simulator, "dims", None)
    if not isinstance(architecture, Mapping) or not isinstance(dimensions, Mapping):
        raise ValueError("live evaluator/model architecture binding is unavailable")
    if model.get("architecture_sha256") != _content_hash(dict(architecture)):
        raise ValueError("model architecture content hash differs")
    mapping = {
        "model_type": "model_type",
        "hidden_size": "hidden",
        "num_hidden_layers": "layers",
        "num_attention_heads": "heads",
        "num_key_value_heads": "kv_heads",
        "head_dim": "head_dim",
        "vocab_size": "vocab",
        "tie_word_embeddings": "tie_embeddings",
        "intermediate_size": "dense_inter",
        "num_experts": "num_experts",
        "num_experts_per_tok": "experts_per_token",
        "moe_intermediate_size": "inter",
        "use_qk_norm": "qk_norm",
        "norm_topk_prob": "norm_topk_prob",
    }
    missing = [
        name
        for name, dimension in mapping.items()
        if name in architecture and dimension not in dimensions
    ]
    if missing:
        raise ValueError(
            "live evaluator omits model architecture fields: " + ",".join(missing)
        )
    mismatched = [
        name
        for name, dimension in mapping.items()
        if name in architecture
        and dimension in dimensions
        and architecture[name] != dimensions[dimension]
    ]
    if mismatched:
        raise ValueError(
            "live evaluator and model architecture differ: " + ",".join(mismatched)
        )


def _comparison_workload(evaluator: Any) -> dict[str, Any]:
    live = getattr(evaluator, "workload", None)
    if live is None:
        raise ValueError("live evaluator workload is unavailable")
    return {
        "scope": "steady_state_cached_q1",
        "query_length": 1,
        "input_seq": int(live.input_seq),
        "output_seq": int(live.output_seq),
        "stride": int(live.stride),
        "runtime_hbm_reserve_bytes": int(live.runtime_hbm_reserve_bytes),
        "kv_layout": str(live.kv_layout),
    }


def _comparison_context(
    evaluator: Any,
    replay: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(Path(replay["config_path"]).read_bytes())
    if not isinstance(config, Mapping):
        raise TypeError("hardware study config root must be an object")
    architecture = dict(config.get("model_architecture", {}))
    model = {
        "name": str(config.get("model_name", "")),
        "revision": str(config.get("model_revision", "")),
        "tokenizer_revision": str(config.get("tokenizer_revision", "")),
        "model_architecture": architecture,
        "architecture_sha256": _content_hash(architecture),
    }
    phase = config.get("phase_contract")
    if not isinstance(phase, Mapping):
        raise ValueError("hardware study config lacks its phase contract")
    workload = _comparison_workload(evaluator)
    _validate_model_architecture(evaluator, model)
    contract = _validate_orchestration_contract(
        _immutable_json(
            replay["orchestration_contract_path"],
            "orchestration contract",
        ),
        expected_model=model["name"],
    )
    return model, workload, dict(phase), contract


def resolve_campaign_source(
    campaign_path: str | Path,
    *,
    hardware_artifact_paths: tuple[str | Path, ...],
    evaluator: Any,
    model: Mapping[str, Any],
    workload: Mapping[str, Any],
    phase_contract: Mapping[str, Any],
    selection_role: str = "balanced_source",
) -> ResolvedCampaignSource:
    """Resolve one supported campaign role to its exact retained hardware row."""

    campaign_file = Path(campaign_path).resolve()
    campaign = _load_campaign(campaign_file)
    _validate_model_architecture(evaluator, model)
    if dict(workload) != _comparison_workload(evaluator):
        raise ValueError("comparison workload differs from the live evaluator")
    if (
        campaign.get("model_name") != model.get("name")
        or campaign.get("model_revision") != model.get("revision")
        or campaign.get("tokenizer_revision") != model.get("tokenizer_revision")
    ):
        raise ValueError("campaign and live model identities differ")
    selected = _campaign_candidate(campaign, selection_role)
    declared_receipts = campaign.get("hardware_artifacts")
    if not isinstance(declared_receipts, list) or not declared_receipts:
        raise ValueError("campaign has no hardware-artifact receipts")
    declared_by_sha: dict[str, Mapping[str, Any]] = {}
    for raw in declared_receipts:
        if not isinstance(raw, Mapping):
            raise TypeError("campaign hardware receipt must be an object")
        digest = _sha256(raw.get("sha256"), "campaign hardware artifact")
        if digest in declared_by_sha:
            raise ValueError("campaign repeats a hardware artifact")
        declared_by_sha[digest] = raw
    supplied = tuple(Path(value).resolve() for value in hardware_artifact_paths)
    if not supplied or len(supplied) != len(set(supplied)):
        raise ValueError("hardware artifact paths must be unique and non-empty")
    supplied_by_sha = {_file_hash(path): path for path in supplied}
    if len(supplied_by_sha) != len(supplied):
        raise ValueError("hardware artifacts repeat identical content")
    if set(supplied_by_sha) != set(declared_by_sha):
        raise ValueError("supplied hardware artifacts differ from the campaign set")
    selector_replay = _replay_campaign_selector(
        campaign_file,
        hardware_artifact_paths=tuple(
            supplied_by_sha[digest] for digest in sorted(supplied_by_sha)
        ),
        campaign=campaign,
    )

    live_evaluator_id, live_evaluator_provenance = _evaluator_identity(evaluator)
    matches: list[tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = []
    family: tuple[str, str] | None = None
    for digest, path in sorted(supplied_by_sha.items()):
        header, rows = load_hardware_artifact(path)
        provenance = header.get("provenance")
        evaluator_provenance = (
            provenance.get("evaluator_provenance")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(provenance, Mapping) or not isinstance(
            evaluator_provenance, Mapping
        ):
            raise ValueError("hardware artifact evaluator provenance is missing")
        evaluator_id = str(provenance.get("evaluator_version", ""))
        evaluator_sha = _content_hash(dict(evaluator_provenance))
        if evaluator_id != live_evaluator_id or evaluator_sha != live_evaluator_provenance:
            raise ValueError("hardware artifact and live evaluator identities differ")
        current_family = evaluator_id, evaluator_sha
        if family is None:
            family = current_family
        elif family != current_family:
            raise ValueError("campaign hardware artifacts use different evaluators")
        if (
            provenance.get("model_revision") != model.get("revision")
            or provenance.get("tokenizer_revision")
            != model.get("tokenizer_revision")
        ):
            raise ValueError("hardware artifact and live model revisions differ")
        for row in rows:
            if (
                row.get("profile_id") == selected["profile_id"]
                and row.get("candidate_id") == selected["candidate_id"]
            ):
                matches.append((path, header, row))
    if len(matches) != 1:
        raise ValueError("campaign source does not resolve to one hardware row")
    selected_path, header, row = matches[0]
    if (
        row.get("record_hash") != selected["record_hash"]
        or _row_body_sha256(row) != selected["record_hash"]
        or row.get("profile") != selected["profile"]
        or row.get("hardware") != selected["hardware"]
        or row.get("numerical_result_hash")
        != selected["accuracy"].get("numerical_result_hash")
    ):
        raise ValueError("campaign source differs from its exact hardware row")
    candidate = HardwareCandidate.from_dict(row["hardware"])
    if candidate.candidate_id != row["candidate_id"]:
        raise ValueError("campaign hardware identity is inconsistent")
    profile = row["profile"]
    if profile.get("matrix_mlen") != candidate.mlen:
        raise ValueError("campaign source is not exact-MLEN bound")
    metrics = row.get("metrics")
    boundary = metrics.get("output_head_boundary") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("runtime_feasible") is not True
        or metrics.get("timing_calibrated") is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("location") != DECODE_MX_HEAD
    ):
        raise ValueError("campaign source lacks feasible local-head decode timing")
    provenance = header["provenance"]
    body = {
        "schema_version": SELECTED_SOURCE_SCHEMA,
        "campaign_path": str(campaign_file),
        "campaign_file_sha256": _file_hash(campaign_file),
        "campaign_content_sha256": campaign["content_hash"],
        "selection_role": selection_role,
        "hardware_artifact_paths": [
            str(supplied_by_sha[digest]) for digest in sorted(supplied_by_sha)
        ],
        "hardware_artifact_sha256s": sorted(supplied_by_sha),
        "selected_artifact_path": str(selected_path),
        "selected_artifact_sha256": _file_hash(selected_path),
        "selected_artifact_provenance_sha256": _content_hash(dict(provenance)),
        "selected_profile_id": str(row["profile_id"]),
        "selected_candidate_id": str(row["candidate_id"]),
        "selected_record_hash": str(row["record_hash"]),
        "selected_row_sha256": _content_hash(dict(row)),
        "model_sha256": _content_hash(dict(model)),
        "workload_sha256": _content_hash(dict(workload)),
        "phase_contract_sha256": _content_hash(dict(phase_contract)),
        **selector_replay,
        "evaluator_id": live_evaluator_id,
        "evaluator_provenance_sha256": live_evaluator_provenance,
    }
    receipt = {**body, "receipt_id": selected_source_receipt_id(body)}
    return ResolvedCampaignSource(
        receipt=validate_selected_source_receipt(receipt),
        row=json.loads(_canonical_bytes(dict(row))),
        header=json.loads(_canonical_bytes(dict(header))),
    )


def paper_geometry_candidate(selected: HardwareCandidate) -> HardwareCandidate:
    """Change only the three published PLENA geometry axes."""

    return replace(selected, mlen=2048, blen=32, vlen=2048)


def _profile_id(entry: Any) -> str:
    value = getattr(entry, "profile_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("repricing entry lacks a profile identity")
    return value


def _matrix_mlen(entry: Any) -> int:
    value = getattr(getattr(entry, "profile", None), "matrix_mlen", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("repricing profile lacks an exact numerical matrix MLEN")
    return value


def numerical_method_contract(profile: Any) -> dict[str, Any]:
    """Return the full profile contract after removing only MLEN derivatives."""

    value = json.loads(_canonical_bytes(dict(profile.to_dict())))
    value.pop("matrix_mlen", None)
    for name in ("numerical_oracle", "local_head"):
        nested = value.get(name)
        if isinstance(nested, Mapping):
            nested = dict(nested)
            nested.pop("matrix_mlen", None)
            value[name] = nested
    return value


def _numerical_evidence_receipt_id(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_id", None)
    return "matched-mlen-evidence-" + _content_hash(body)


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _same_float(observed: Any, expected: float, label: str) -> None:
    value = _finite_nonnegative(observed, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} differs from its recomputed ledger")


def _validate_runtime_rebinding(
    value: Any,
    *,
    profile: Any,
    bank: Mapping[str, Any],
    suite: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"MLEN {suite} runtime-rebinding receipt is missing")
    is_bf16 = profile.kind == "bf16_reference"
    performed = value.get("performed")
    if performed is not (not is_bf16):
        raise ValueError(f"MLEN {suite} runtime-rebinding mode differs")
    _finite_nonnegative(value.get("seconds"), f"MLEN {suite} bind seconds")
    target_count = value.get("target_count")
    sealed_modules = value.get("sealed_weight_modules")
    before = value.get("weight_quantization_events_before")
    after = value.get("weight_quantization_events_after")
    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count < 0
        or (is_bf16 and target_count != 0)
        or (not is_bf16 and target_count <= 0)
        or isinstance(sealed_modules, bool)
        or not isinstance(sealed_modules, int)
        or sealed_modules < 0
        or (is_bf16 and sealed_modules != 0)
        or (not is_bf16 and sealed_modules <= 0)
        or isinstance(before, bool)
        or not isinstance(before, int)
        or before < 0
        or isinstance(after, bool)
        or not isinstance(after, int)
        or after != before
        or value.get("used_cached_targets") is not True
        or value.get("weight_requantizations") != 0
    ):
        raise ValueError(
            f"MLEN {suite} runtime rebinding is not a native no-requantization bind"
        )
    identity = bank["identity_fingerprint"]
    structure = bank["structure_fingerprint"]
    if (
        value.get("weight_identity_before") != identity
        or value.get("weight_identity_after") != identity
        or value.get("weight_structure_fingerprint") != structure
    ):
        raise ValueError(f"MLEN {suite} runtime binding changed its weight bank")


def _validate_document_nll_ledger(
    metrics: Mapping[str, Any],
    *,
    suite: str,
    expected_documents: int,
    expected_steps: int,
    expected_document_ids: Sequence[str],
    expected_initial_cache_length: int,
) -> tuple[float, int, tuple[tuple[str, int, int, int], ...]]:
    documents = metrics.get("documents")
    if not isinstance(documents, list) or len(documents) != expected_documents:
        raise ValueError(f"MLEN {suite} document coverage differs from its plan")
    document_ids: set[str] = set()
    ordered_population: list[tuple[str, int, int, int]] = []
    nll_values: list[float] = []
    token_count = 0
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError(f"MLEN {suite} document ledger entry must be an object")
        document_id = document.get("document_id")
        tokens = document.get("token_count")
        initial = document.get("initial_cache_length")
        final = document.get("final_cache_length")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id in document_ids
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens != expected_steps
            or isinstance(initial, bool)
            or not isinstance(initial, int)
            or initial != expected_initial_cache_length
            or isinstance(final, bool)
            or not isinstance(final, int)
            or final - initial != expected_steps
        ):
            raise ValueError(f"MLEN {suite} document/token/cache ledger differs")
        document_ids.add(document_id)
        ordered_population.append((document_id, tokens, initial, final))
        nll = _finite_nonnegative(
            document.get("nll_sum"), f"MLEN {suite} document NLL sum"
        )
        _same_float(
            document.get("mean_token_nll"),
            nll / tokens,
            f"MLEN {suite} document mean token NLL",
        )
        nll_values.append(nll)
        token_count += tokens
    nll_sum = math.fsum(nll_values)
    if tuple(value[0] for value in ordered_population) != tuple(
        expected_document_ids
    ):
        raise ValueError(f"MLEN {suite} ordered document population differs")
    expected_tokens = expected_documents * expected_steps
    if token_count != expected_tokens or metrics.get("token_count") != expected_tokens:
        raise ValueError(f"MLEN {suite} token conservation differs from its plan")
    if "document_count" in metrics and metrics.get("document_count") != expected_documents:
        raise ValueError(f"MLEN {suite} document-count receipt differs")
    _same_float(metrics.get("nll_sum"), nll_sum, f"MLEN {suite} NLL sum")
    mean = nll_sum / token_count
    _same_float(
        metrics.get("mean_token_nll", metrics.get("mean_nll")),
        mean,
        f"MLEN {suite} mean token NLL",
    )
    if "mean_nll" in metrics:
        _same_float(metrics["mean_nll"], mean, f"MLEN {suite} mean NLL")
    return mean, token_count, tuple(ordered_population)


def _resolve_mlen_sample_populations(
    plan: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from decode_dse.software.sweep_plan import (
        HARDWARE_VALIDATION_SAMPLE_CONTRACT,
        PromptManifest,
        load_immutable_json,
    )
    from decode_dse.software.token_samples import load_refinement_sample_bundle

    paths = plan.get("paths")
    bindings = plan.get("bindings")
    suites = plan.get("sample_suites")
    if not all(isinstance(value, Mapping) for value in (paths, bindings, suites)):
        raise ValueError("MLEN plan omits producer sample-source bindings")

    raw_refinement_path = Path(str(paths.get("refinement_sample_bundle", "")))
    if raw_refinement_path.is_symlink():
        raise ValueError("MLEN refinement sample source cannot be a symlink")
    refinement_path = raw_refinement_path.resolve()
    if not refinement_path.is_file():
        raise ValueError("MLEN refinement sample source is unavailable")
    refinement = load_refinement_sample_bundle(refinement_path)
    refinement_contract = suites.get("refinement")
    if (
        not isinstance(refinement_contract, Mapping)
        or _file_hash(refinement_path)
        != bindings.get("refinement_sample_file_sha256")
        or refinement.canonical_hash
        != bindings.get("refinement_sample_bundle_hash")
        or refinement_contract.get("sample_bundle_hash")
        != refinement.canonical_hash
        or refinement.model_revision != model.get("revision")
        or refinement.tokenizer_revision != model.get("tokenizer_revision")
        or len(refinement.samples) != refinement_contract.get("document_count")
        or any(
            len(sample.decode_target_ids)
            < int(refinement_contract.get("decode_steps", 0))
            for sample in refinement.samples
        )
    ):
        raise ValueError("MLEN refinement sample source differs from its plan")
    refinement_origins = {len(sample.prompt_token_ids) for sample in refinement.samples}
    if len(refinement_origins) != 1:
        raise ValueError("MLEN refinement cache origins are not uniform")

    numerical_workspace = Path(str(paths.get("numerical_workspace", ""))).resolve()
    raw_prompt_path = numerical_workspace / "prompt_manifest.json"
    if raw_prompt_path.is_symlink() or not raw_prompt_path.is_file():
        raise ValueError("MLEN validation prompt source is unavailable or a symlink")
    prompt_value = load_immutable_json(raw_prompt_path)
    prompts = PromptManifest.from_dict(prompt_value)
    validation_contract = suites.get("validation")
    if (
        not isinstance(validation_contract, Mapping)
        or prompts.canonical_hash != bindings.get("prompt_manifest_hash")
        or validation_contract.get("prompt_set") != "hardware_validation"
        or len(prompts.hardware_validation)
        != validation_contract.get("document_count")
        or validation_contract.get("decode_steps")
        != HARDWARE_VALIDATION_SAMPLE_CONTRACT.decode_steps
        or validation_contract.get("q_len") != 1
    ):
        raise ValueError("MLEN validation prompt source differs from its plan")

    populations = {
        "validation": {
            "document_ids": tuple(
                record.document_id for record in prompts.hardware_validation
            ),
            "initial_cache_length": (
                HARDWARE_VALIDATION_SAMPLE_CONTRACT.prefill_tokens
            ),
        },
        "refinement": {
            "document_ids": tuple(
                sample.document_id for sample in refinement.samples
            ),
            "initial_cache_length": refinement_origins.pop(),
        },
    }
    sources = {
        "validation_prompt_manifest_path": str(raw_prompt_path.resolve()),
        "validation_prompt_manifest_file_sha256": _file_hash(raw_prompt_path),
        "validation_prompt_manifest_content_sha256": prompts.canonical_hash,
        "refinement_sample_bundle_path": str(refinement_path),
        "refinement_sample_bundle_file_sha256": _file_hash(refinement_path),
        "refinement_sample_bundle_content_sha256": refinement.canonical_hash,
    }
    return populations, sources


def _validate_native_append(
    value: Any,
    *,
    suite: str,
    profile: Any,
    layers: int,
) -> None:
    if not isinstance(value, Mapping) or value.get("q_len") != 1:
        raise ValueError(f"MLEN {suite} native-append receipt is missing")
    calls = value.get("calls")
    tensors = value.get("tensor_checks")
    quantized = value.get("quantized_tensor_checks")
    expected_tensors = value.get("expected_tensor_checks")
    expected_quantized = value.get("expected_quantized_tensor_checks")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in (calls, tensors, quantized, expected_tensors, expected_quantized)
    ):
        raise ValueError(f"MLEN {suite} native-append counts are invalid")
    if suite == "refinement":
        required_tensors = layers * 2
        required_quantized = (
            0 if profile.kind == "bf16_reference" else required_tensors
        )
        if (
            calls != 1
            or tensors != required_tensors
            or quantized != required_quantized
            or expected_tensors != required_tensors
            or expected_quantized != required_quantized
        ):
            raise ValueError("MLEN refinement native-append coverage differs")
        return
    expected_calls = value.get("expected_calls")
    if (
        isinstance(expected_calls, bool)
        or not isinstance(expected_calls, int)
        or expected_calls < 0
        or calls != expected_calls
        or tensors != expected_tensors
        or quantized != expected_quantized
        or expected_tensors != expected_calls * layers * 2
        or expected_quantized
        != (0 if profile.kind == "bf16_reference" else expected_tensors)
    ):
        raise ValueError("MLEN validation native-append coverage differs")
    deep = value.get("deep_oracle_enabled")
    mode = value.get("mode")
    if (
        not isinstance(deep, bool)
        or mode != ("deep_oracle" if deep else "preflight_gated")
        or (deep and calls <= 0)
        or (not deep and calls != 0)
    ):
        raise ValueError("MLEN validation native-append mode differs")


def _validate_mlen_terminal_row(
    row: Mapping[str, Any],
    *,
    profile: Any,
    planned: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_index: int,
    model_architecture: Mapping[str, Any],
    sample_populations: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[float, int, tuple[tuple[str, int, int, int], ...]]]:
    if row.get("profile") != planned.get("profile") or row.get("role") != planned.get(
        "role"
    ):
        raise ValueError("MLEN terminal profile differs from its plan")
    classification = row.get("classification")
    if (
        row.get("state") != "succeeded"
        or row.get("error") is not None
        or row.get("shard_index") != shard_index
        or not isinstance(classification, Mapping)
        or classification.get("measured_numerical") is not True
        or classification.get("publication_rankable") is not False
        or classification.get("selection_eligible") is not False
        or classification.get("compiler_valid") is not False
        or classification.get("emulator_valid") is not False
        or classification.get("rtl_valid") is not False
        or classification.get("hardware_bit_parity_verified") is not False
    ):
        raise ValueError("MLEN terminal row classification/provenance differs")
    _finite_nonnegative(row.get("runtime_seconds"), "MLEN row runtime")
    bank = row.get("weight_bank")
    if not isinstance(bank, Mapping):
        raise ValueError("MLEN row lacks a sealed weight-bank receipt")
    profile_ids = bank.get("profile_ids")
    if (
        bank.get("weight_format") != profile.weight_format
        or bank.get("weight_method") != profile.method
        or not isinstance(profile_ids, list)
        or profile.profile_id not in profile_ids
        or len(profile_ids) != len(set(profile_ids))
        or bank.get("same_in_memory_bank_across_mlen_values") is not True
        or isinstance(bank.get("parameter_count"), bool)
        or not isinstance(bank.get("parameter_count"), int)
        or bank.get("parameter_count") <= 0
    ):
        raise ValueError("MLEN weight-bank identity differs from its profile")
    _finite_nonnegative(bank.get("build_seconds"), "MLEN bank build time")
    _sha256(bank.get("identity_fingerprint"), "MLEN bank identity")
    _sha256(bank.get("structure_fingerprint"), "MLEN bank structure")
    sample_suites = plan.get("sample_suites")
    if not isinstance(sample_suites, Mapping):
        raise ValueError("MLEN plan lacks sample-suite geometry")
    layers = model_architecture.get("num_hidden_layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError("comparison model lacks native-append layer coverage")
    metrics_by_suite: dict[
        str, tuple[float, int, tuple[tuple[str, int, int, int], ...]]
    ] = {}
    for suite in ("validation", "refinement"):
        contract = sample_suites.get(suite)
        metrics = row.get(suite)
        if not isinstance(contract, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError(f"MLEN {suite} plan/row metrics are missing")
        documents = contract.get("document_count")
        steps = contract.get("decode_steps")
        q_len = contract.get("q_len")
        if (
            isinstance(documents, bool)
            or not isinstance(documents, int)
            or documents <= 0
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
            or q_len != 1
        ):
            raise ValueError(f"MLEN {suite} sample-suite geometry is invalid")
        if suite == "refinement":
            bindings = plan.get("bindings")
            if (
                not isinstance(bindings, Mapping)
                or metrics.get("sample_bundle_hash")
                != bindings.get("refinement_sample_bundle_hash")
                or contract.get("sample_bundle_hash")
                != bindings.get("refinement_sample_bundle_hash")
                or metrics.get("document_count") != documents
                or metrics.get("decode_steps") != steps
                or metrics.get("q_len") != 1
            ):
                raise ValueError("MLEN refinement sample-bundle binding differs")
        else:
            sample_contract = metrics.get("sample_contract")
            validation_bank = metrics.get("weight_bank")
            runtime_environment = metrics.get("runtime_environment")
            if (
                not isinstance(sample_contract, Mapping)
                or sample_contract.get("prompt_set") != contract.get("prompt_set")
                or sample_contract.get("prompt_count") != documents
                or sample_contract.get("decode_steps") != steps
                or sample_contract.get("q_len") != 1
                or sample_contract.get("teacher_forced_cached") is not True
                or not isinstance(validation_bank, Mapping)
                or validation_bank.get("weight_format") != bank["weight_format"]
                or validation_bank.get("weight_method") != bank["weight_method"]
                or validation_bank.get("parameter_count") != bank["parameter_count"]
                or validation_bank.get("identity_fingerprint")
                != bank["identity_fingerprint"]
                or validation_bank.get("structure_fingerprint")
                != bank["structure_fingerprint"]
                or not isinstance(runtime_environment, Mapping)
            ):
                raise ValueError("MLEN validation sample contract differs")
            _sha256(
                runtime_environment.get("logical_fingerprint"),
                "MLEN runtime logical fingerprint",
            )
            _sha256(
                runtime_environment.get("mase_tree_sha256"),
                "MLEN runtime MASE source tree",
            )
        population = sample_populations.get(suite)
        if not isinstance(population, Mapping):
            raise ValueError(f"MLEN {suite} sample population is unavailable")
        mean, tokens, ordered_population = _validate_document_nll_ledger(
            metrics,
            suite=suite,
            expected_documents=documents,
            expected_steps=steps,
            expected_document_ids=population["document_ids"],
            expected_initial_cache_length=population["initial_cache_length"],
        )
        _validate_runtime_rebinding(
            metrics.get("runtime_rebinding"),
            profile=profile,
            bank=bank,
            suite=suite,
        )
        _validate_native_append(
            metrics.get("native_append_validation"),
            suite=suite,
            profile=profile,
            layers=layers,
        )
        metrics_by_suite[suite] = (mean, tokens, ordered_population)
    return metrics_by_suite


def _validate_mlen_mapping_sources(
    plan: Mapping[str, Any],
    *,
    selected_source_profile_id: str,
    required_mlens: set[int],
) -> list[dict[str, Any]]:
    mappings = [
        value
        for value in plan.get("candidate_mapping", ())
        if isinstance(value, Mapping)
        and value.get("source_profile_id") == selected_source_profile_id
        and int(value.get("candidate_matrix_mlen", 0)) in required_mlens
    ]
    observed_mlens = {int(value["candidate_matrix_mlen"]) for value in mappings}
    if observed_mlens != required_mlens or len(mappings) != len(required_mlens):
        raise ValueError("MLEN plan source mapping coverage is not exact")
    receipts = []
    for mapping in sorted(mappings, key=lambda item: int(item["candidate_matrix_mlen"])):
        target_hardware = mapping.get("hardware")
        if not isinstance(target_hardware, Mapping):
            raise ValueError("MLEN mapped target hardware is missing")
        target_candidate = HardwareCandidate.from_dict(target_hardware)
        if (
            target_candidate.candidate_id != mapping.get("candidate_id")
            or target_candidate.mlen != int(mapping["candidate_matrix_mlen"])
        ):
            raise ValueError("MLEN mapped target candidate identity differs")
        raw_source_path = Path(str(mapping.get("source_artifact_path", "")))
        if raw_source_path.is_symlink():
            raise ValueError("MLEN mapped hardware source is unavailable or a symlink")
        source_path = raw_source_path.resolve()
        if not source_path.is_file():
            raise ValueError("MLEN mapped hardware source is unavailable or a symlink")
        metadata_path = source_path.with_name(f"{source_path.name}.meta.json")
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("MLEN mapped hardware metadata is unavailable or a symlink")
        source_sha = _file_hash(source_path)
        if source_sha != mapping.get("source_artifact_sha256"):
            raise ValueError("MLEN mapped hardware source bytes changed")
        header, source_rows = load_hardware_artifact(source_path)
        matches = [
            row
            for row in source_rows
            if row.get("profile_id") == selected_source_profile_id
            and row.get("record_hash") == mapping.get("source_record_hash")
        ]
        if len(matches) != 1:
            raise ValueError("MLEN mapped hardware row membership differs")
        source_hardware = matches[0].get("hardware")
        if not isinstance(source_hardware, Mapping):
            raise ValueError("MLEN mapped source row lacks hardware")
        source_candidate = HardwareCandidate.from_dict(source_hardware)
        if source_candidate.candidate_id != matches[0].get("candidate_id"):
            raise ValueError("MLEN mapped source candidate identity differs")
        planned = next(
            value
            for value in plan["evaluation_profiles"]
            if value["profile_id"] == selected_source_profile_id
        )
        if matches[0].get("profile") != planned.get("profile"):
            raise ValueError("MLEN mapped hardware source profile differs")
        receipts.append(
            {
                "candidate_matrix_mlen": int(mapping["candidate_matrix_mlen"]),
                "candidate_id": str(mapping["candidate_id"]),
                "source_record_hash": str(mapping["source_record_hash"]),
                "source_candidate_id": str(matches[0].get("candidate_id")),
                "source_hardware_sha256": _content_hash(source_hardware),
                "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_sha,
                "source_metadata_path": str(metadata_path),
                "source_metadata_sha256": _file_hash(metadata_path),
                "hardware_identity_hash": str(mapping["hardware_identity_hash"]),
                "source_run_id": str(header.get("run_id")),
                "source_provenance_sha256": _content_hash(
                    dict(header.get("provenance", {}))
                ),
            }
        )
    return receipts


def resolve_mlen_numerical_evidence(
    completion_path: str | Path,
    *,
    selected_source_profile_id: str,
    selected_hardware_mlen: int,
    model: Mapping[str, Any],
    suite: str = "refinement",
) -> ResolvedNumericalEvidence:
    """Resolve the specialized/shared/BF16 controls from one completion.

    A completion that omitted the predeclared campaign source or its MLEN=2048
    control fails here with an actionable error.  The resolver never reuses a
    1024 result for another MLEN.
    """

    if suite not in {"validation", "refinement"}:
        raise ValueError("matched comparison suite is unsupported")
    from decode_dse.legality import StackValidity, evaluate_profile_legality
    from decode_dse.manifest import SweepManifestEntry
    from decode_dse.profiles import DecodePrecisionProfile
    from decode_dse.software import mlen_revalidation as mlen_producer
    from decode_dse.software.mlen_revalidation import (
        COMPLETION_SCHEMA,
        ROW_SCHEMA,
        _load_plan,
        finalize,
    )
    from decode_dse.software.sweep_plan import load_immutable_json

    mlen_producer_path = Path(mlen_producer.__file__).resolve()
    expected_mlen_producer_path = Path(__file__).with_name(
        "mlen_revalidation.py"
    ).resolve()
    if mlen_producer_path != expected_mlen_producer_path:
        raise RuntimeError("MLEN producer was imported from another Software tree")

    raw_completion_file = Path(completion_path)
    if raw_completion_file.is_symlink():
        raise ValueError("MLEN completion is unavailable or a symlink")
    completion_file = raw_completion_file.resolve()
    if not completion_file.is_file():
        raise ValueError("MLEN completion is unavailable or a symlink")
    completion = load_immutable_json(completion_file)
    if (
        completion.get("schema_version") != COMPLETION_SCHEMA
        or completion.get("complete") is not True
        or completion.get("successful") is not True
        or completion.get("failed_profile_count") != 0
    ):
        raise ValueError("MLEN completion is not terminal-successful")
    raw_plan_file = Path(str(completion.get("plan_path", "")))
    if raw_plan_file.is_symlink():
        raise ValueError("MLEN plan is unavailable or a symlink")
    plan_file = raw_plan_file.resolve()
    if not plan_file.is_file():
        raise ValueError("MLEN plan is unavailable or a symlink")
    plan = _load_plan(plan_file)
    expected_completion_path = (
        Path(str(plan.get("paths", {}).get("output_root", ""))).resolve()
        / "mlen_revalidation_completion.json"
    )
    if (
        completion_file != expected_completion_path
        or completion.get("plan_hash") != plan.get("content_hash")
    ):
        raise ValueError("MLEN completion and plan identities differ")
    replayed_completion = finalize(plan_path=plan_file)
    if replayed_completion != completion:
        raise ValueError("MLEN producer completion replay differs")
    plan_classification = plan.get("classification")
    if (
        not isinstance(plan_classification, Mapping)
        or plan_classification.get("measured_numerical") is not True
        or plan_classification.get("publication_rankable") is not False
        or plan_classification.get("publication_selection_eligible") is not False
        or plan_classification.get("hardware_bit_parity_verified") is not False
    ):
        raise ValueError("MLEN plan numerical classification differs")
    target = plan.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("model_name") != model.get("name")
        or target.get("model_revision") != model.get("revision")
        or target.get("tokenizer_revision") != model.get("tokenizer_revision")
    ):
        raise ValueError("MLEN plan and comparison model identities differ")
    sample_populations, sample_sources = _resolve_mlen_sample_populations(
        plan, model=model
    )
    receipts = completion.get("row_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("MLEN completion has no terminal row receipts")
    planned_profiles = {
        str(value["profile_id"]): value
        for value in plan.get("evaluation_profiles", ())
        if isinstance(value, Mapping)
    }
    shard_by_profile: dict[str, int] = {}
    for partition in plan.get("sharding", {}).get("partitions", ()):
        if not isinstance(partition, Mapping):
            raise TypeError("MLEN shard partition must be an object")
        shard_index = partition.get("shard_index")
        profile_ids = partition.get("profile_ids")
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or shard_index < 0
            or not isinstance(profile_ids, list)
        ):
            raise ValueError("MLEN shard partition identity is invalid")
        for profile_id in profile_ids:
            if not isinstance(profile_id, str) or profile_id in shard_by_profile:
                raise ValueError("MLEN shard partition coverage is not unique")
            shard_by_profile[profile_id] = shard_index
    if set(shard_by_profile) != set(planned_profiles):
        raise ValueError("MLEN shard partition coverage differs from the plan")
    rows: dict[str, dict[str, Any]] = {}
    row_sources: dict[str, dict[str, Any]] = {}
    recomputed_metrics: dict[
        str,
        dict[str, tuple[float, int, tuple[tuple[str, int, int, int], ...]]],
    ] = {}
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise TypeError("MLEN terminal row receipt must be an object")
        profile_id = str(raw.get("profile_id", ""))
        raw_row_path = Path(str(raw.get("path", "")))
        if raw_row_path.is_symlink():
            raise ValueError("MLEN terminal row is unavailable or a symlink")
        row_path = raw_row_path.resolve()
        if not row_path.is_file():
            raise ValueError("MLEN terminal row is unavailable or a symlink")
        row = load_immutable_json(row_path)
        classification = row.get("classification")
        if (
            not profile_id
            or profile_id in rows
            or raw.get("state") != "succeeded"
            or row.get("schema_version") != ROW_SCHEMA
            or row.get("state") != "succeeded"
            or row.get("profile_id") != profile_id
            or row.get("plan_hash") != plan["content_hash"]
            or raw.get("content_hash") != row.get("content_hash")
            or not isinstance(classification, Mapping)
        ):
            raise ValueError("MLEN terminal row receipt is inconsistent")
        planned = planned_profiles.get(profile_id)
        if not isinstance(planned, Mapping):
            raise ValueError("MLEN row names an unplanned profile")
        profile = DecodePrecisionProfile.from_dict(row["profile"])
        if profile.profile_id != profile_id:
            raise ValueError("MLEN terminal profile identity differs")
        recomputed_metrics[profile_id] = _validate_mlen_terminal_row(
            row,
            profile=profile,
            planned=planned,
            plan=plan,
            shard_index=shard_by_profile[profile_id],
            model_architecture=model["model_architecture"],
            sample_populations=sample_populations,
        )
        rows[profile_id] = row
        row_sources[profile_id] = {
            "path": str(row_path),
            "file_sha256": _file_hash(row_path),
            "content_hash": row["content_hash"],
        }
    if set(rows) != set(planned_profiles):
        raise ValueError("MLEN completion does not cover every planned profile")
    for profile_id, row in rows.items():
        planned = planned_profiles[profile_id]
        if (
            row.get("profile") != planned.get("profile")
            or row.get("role") != planned.get("role")
        ):
            raise ValueError("MLEN terminal profile differs from its plan")

    # Source/control variants of one weight format must retain the same sealed
    # in-memory parameter identity.  A fresh or re-quantized bank cannot be
    # substituted merely by copying the profile metadata.
    bank_identities: dict[str, tuple[Any, ...]] = {}
    for profile_id, row in rows.items():
        profile = DecodePrecisionProfile.from_dict(row["profile"])
        bank = row["weight_bank"]
        identity = (
            bank["weight_method"],
            bank["identity_fingerprint"],
            bank["structure_fingerprint"],
            bank["parameter_count"],
        )
        previous = bank_identities.setdefault(profile.weight_format, identity)
        if previous != identity:
            raise ValueError("MLEN controls did not share one sealed weight bank")
    runtime_identities = {
        (
            row["validation"]["runtime_environment"]["logical_fingerprint"],
            row["validation"]["runtime_environment"]["mase_tree_sha256"],
        )
        for row in rows.values()
    }
    if len(runtime_identities) != 1:
        raise ValueError("MLEN controls used different logical/software runtimes")
    for suite in ("validation", "refinement"):
        populations = {
            metrics[suite][2] for metrics in recomputed_metrics.values()
        }
        if len(populations) != 1:
            raise ValueError(
                f"MLEN {suite} controls used different ordered populations"
            )

    mappings = [
        value
        for value in plan.get("candidate_mapping", ())
        if isinstance(value, Mapping)
        and value.get("source_profile_id") == selected_source_profile_id
    ]
    required_mapping_mlens = {2048}
    if int(selected_hardware_mlen) != 1024:
        required_mapping_mlens.add(int(selected_hardware_mlen))
    mapping_sources = _validate_mlen_mapping_sources(
        plan,
        selected_source_profile_id=selected_source_profile_id,
        required_mlens=required_mapping_mlens,
    )

    def variant_id(mlen: int) -> str:
        if mlen == 1024:
            if selected_source_profile_id not in rows:
                raise ValueError(
                    "MLEN plan omitted the selected source profile control"
                )
            return selected_source_profile_id
        values = {
            str(value.get("revalidated_profile_id"))
            for value in mappings
            if value.get("candidate_matrix_mlen") == mlen
        }
        if len(values) != 1 or "None" in values:
            raise ValueError(
                "comparison-specific MLEN plan omitted the selected source "
                f"profile's exact MLEN={mlen} control"
            )
        profile_id = values.pop()
        if profile_id not in rows:
            raise ValueError("MLEN variant row is absent from the completion")
        return profile_id

    specialized_id = variant_id(int(selected_hardware_mlen))
    shared_id = variant_id(2048)
    bf16_ids = [
        profile_id
        for profile_id, value in planned_profiles.items()
        if value.get("role") == "same_split_bf16_reference"
    ]
    if len(bf16_ids) != 1:
        raise ValueError("MLEN plan lacks one same-split BF16 control")
    bf16_id = bf16_ids[0]
    bf16_profile = DecodePrecisionProfile.from_dict(rows[bf16_id]["profile"])
    specialized_profile = DecodePrecisionProfile.from_dict(
        rows[specialized_id]["profile"]
    )
    shared_profile = DecodePrecisionProfile.from_dict(rows[shared_id]["profile"])
    if (
        specialized_profile.profile_id != specialized_id
        or shared_profile.profile_id != shared_id
        or specialized_profile.matrix_mlen != selected_hardware_mlen
        or shared_profile.matrix_mlen != 2048
        or specialized_profile.kind == "bf16_reference"
        or shared_profile.kind == "bf16_reference"
        or numerical_method_contract(specialized_profile)
        != numerical_method_contract(shared_profile)
    ):
        raise ValueError("MLEN comparison profiles are not exact method variants")
    bf16_oracle_contract = bf16_profile.numerical_oracle_contract
    bf16_head_contract = bf16_profile.local_head_contract
    if (
        bf16_profile.profile_id != bf16_id
        or bf16_profile.kind != "bf16_reference"
        or bf16_id in {specialized_id, shared_id}
        or any(
            value != "BF16"
            for value in (
                bf16_profile.weight_format,
                bf16_profile.activation_format,
                bf16_profile.key_format,
                bf16_profile.value_format,
                bf16_profile.vector_format,
            )
        )
        or bf16_oracle_contract.get("rule") != "backend_bf16_reference"
        or bf16_oracle_contract.get("implementation")
        != "torch.nn.functional.linear_bf16_reference"
        or bf16_oracle_contract.get("matrix_mlen") is not None
        or bf16_head_contract.get("weight_method") != "bf16_reference"
        or bf16_head_contract.get("numerical_oracle_rule")
        != "backend_bf16_reference"
    ):
        raise ValueError("same-split BF16 row is not a backend BF16 reference")
    specialized_nll, specialized_tokens, _ = recomputed_metrics[specialized_id][
        suite
    ]
    shared_nll, shared_tokens, _ = recomputed_metrics[shared_id][suite]
    bf16_nll, bf16_tokens, _ = recomputed_metrics[bf16_id][suite]
    if len({specialized_tokens, shared_tokens, bf16_tokens}) != 1:
        raise ValueError("MLEN controls scored different token populations")
    bindings = plan.get("bindings")
    sample_suites = plan.get("sample_suites")
    if not isinstance(bindings, Mapping) or not isinstance(sample_suites, Mapping):
        raise ValueError("MLEN plan omits sample/protocol bindings")
    if suite == "refinement":
        sample_set_sha = _sha256(
            bindings.get("refinement_sample_bundle_hash"),
            "refinement sample bundle",
        )
        dataset_sha = _sha256(
            bindings.get("refinement_sample_file_sha256"),
            "refinement sample file",
        )
        prompt_sha = sample_set_sha
    else:
        sample_set_sha = _sha256(
            bindings.get("prompt_manifest_hash"), "validation prompt manifest"
        )
        dataset_sha = sample_set_sha
        prompt_sha = sample_set_sha
    scope = f"mlen_revalidation_{suite}_same_split_teacher_forced_cached_decode"
    method_hash = _content_hash(numerical_method_contract(specialized_profile))
    core = _core()

    def numeric_receipt(profile: Any, row_id: str, nll: float) -> dict[str, Any]:
        body = {
            "schema_version": core.NUMERICAL_RECEIPT_SCHEMA,
            "profile_id": profile.profile_id,
            "evaluated_mlen": profile.matrix_mlen,
            "nominal_precision_sha256": _content_hash(
                physical_cost_signature(profile, exact_vector_format=True)
            ),
            "method_contract_sha256": method_hash,
            "source_receipt_sha256": rows[row_id]["content_hash"],
            "accuracy_scope": scope,
            "sample_set_sha256": sample_set_sha,
            "scored_tokens": specialized_tokens,
            "candidate_mean_token_nll": nll,
            "bf16_mean_token_nll": bf16_nll,
            "state": "succeeded",
            "hardware_bit_parity_verified": False,
            "publication_rankable": False,
        }
        return {**body, "receipt_id": core.numerical_receipt_id(body)}

    specialized_receipt = numeric_receipt(
        specialized_profile, specialized_id, specialized_nll
    )
    shared_receipt = numeric_receipt(shared_profile, shared_id, shared_nll)
    protocol_sha = _content_hash(
        {
            "plan_hash": plan["content_hash"],
            "suite": suite,
            "contract": sample_suites.get(suite),
        }
    )
    seed_sha = _content_hash(
        {
            "sample_set_sha256": sample_set_sha,
            "seed_policy": "sealed_by_mlen_sample_bundle",
        }
    )
    bf16_body = {
        "schema_version": core.BF16_ORACLE_SCHEMA,
        "model": dict(model),
        "profile_id": bf16_id,
        "source_receipt_sha256": rows[bf16_id]["content_hash"],
        "accuracy_scope": scope,
        "evaluation_protocol_sha256": protocol_sha,
        "dataset_sha256": dataset_sha,
        "prompt_manifest_sha256": prompt_sha,
        "seed_receipt_sha256": seed_sha,
        "mean_nll_receipt_sha256": rows[bf16_id]["content_hash"],
        "sample_set_sha256": sample_set_sha,
        "scored_tokens": bf16_tokens,
        "mean_token_nll": bf16_nll,
        "latency_role": "accuracy_only_not_hardware_priced",
        "state": "succeeded",
    }
    bf16_oracle = {
        **bf16_body,
        "receipt_id": core.bf16_oracle_receipt_id(bf16_body),
    }
    receipt_body = {
        "schema_version": NUMERICAL_EVIDENCE_SCHEMA,
        "evidence_class": NUMERICAL_EVIDENCE_CLASS,
        "source_hashes_and_internal_ledgers_verified": True,
        "independent_numerical_execution_replayed": False,
        "adversarial_tamper_resistance_claimed": False,
        "completion_path": str(completion_file),
        "completion_file_sha256": _file_hash(completion_file),
        "completion_content_hash": completion["content_hash"],
        "plan_path": str(plan_file),
        "plan_file_sha256": _file_hash(plan_file),
        "plan_content_hash": plan["content_hash"],
        "selected_source_profile_id": selected_source_profile_id,
        "selected_hardware_mlen": int(selected_hardware_mlen),
        "specialized_profile_id": specialized_id,
        "shared_profile_id": shared_id,
        "bf16_profile_id": bf16_id,
        "suite": suite,
        "sample_set_sha256": sample_set_sha,
        "scored_tokens": specialized_tokens,
        "row_sources": {
            profile_id: row_sources[profile_id]
            for profile_id in (specialized_id, shared_id, bf16_id)
        },
        "mapping_sources": mapping_sources,
        "sample_sources": sample_sources,
        "producer_replay": {
            "producer_module_path": str(mlen_producer_path),
            "producer_module_sha256": _file_hash(mlen_producer_path),
            "plan_validator": "mlen_revalidation._load_plan",
            "completion_validator": "mlen_revalidation.finalize",
            "completion_replayed_equal": True,
            "all_terminal_row_ledgers_recomputed": True,
            "native_append_and_no_requantization_verified": True,
            "mapped_hardware_membership_verified": True,
        },
    }
    numerical_receipt = {
        **receipt_body,
        "receipt_id": _numerical_evidence_receipt_id(receipt_body),
    }

    def entry(profile: Any, ordinal: int) -> Any:
        return SweepManifestEntry(
            ordinal=ordinal,
            profile=profile,
            legality=evaluate_profile_legality(profile),
            validity=StackValidity(),
        )

    def evaluator_result(nll: float, tokens: int) -> dict[str, Any]:
        return {
            "state": "succeeded",
            "result": {
                "mean_nll": nll,
                "mean_token_nll": nll,
                "token_count": tokens,
            },
        }

    return ResolvedNumericalEvidence(
        receipt=numerical_receipt,
        specialized_entry=entry(specialized_profile, 0),
        shared_entry=entry(shared_profile, 1),
        specialized_result=evaluator_result(specialized_nll, specialized_tokens),
        shared_result=evaluator_result(shared_nll, shared_tokens),
        specialized_receipt=specialized_receipt,
        shared_receipt=shared_receipt,
        bf16_oracle=bf16_oracle,
    )


def _run_one(
    evaluator: Any,
    entry: Any,
    numerical_result: Mapping[str, Any],
    candidate: HardwareCandidate,
) -> HardwareEvaluation:
    """Apply both normal preflight gates, then execute exactly one row."""

    candidate_preflight = getattr(evaluator, "candidate_preflight", None)
    if callable(candidate_preflight):
        rejected = candidate_preflight(candidate)
        if rejected is not None:
            raise ValueError(
                "matched candidate failed resource preflight: "
                f"{rejected.error_code}: {rejected.error_message}"
            )
    preflight = getattr(evaluator, "preflight", None)
    if callable(preflight):
        rejected = preflight(entry, candidate, numerical_result)
        if rejected is not None:
            # The comparison may retain a fully timed point that exceeds the
            # common ceiling, but the core will label it resource-unmatched
            # and suppress every ratio.  Runtime/structural failures remain
            # hard blockers.
            if rejected.error_code != "resource_budget_exceeded":
                raise ValueError(
                    "matched candidate failed exact preflight: "
                    f"{rejected.error_code}: {rejected.error_message}"
                )
    evaluation = evaluator(entry, candidate, numerical_result)
    if not isinstance(evaluation, HardwareEvaluation):
        raise TypeError("matched repricing evaluator returned the wrong type")
    if evaluation.metrics is None:
        raise ValueError(
            "matched candidate produced no timing metrics: "
            f"{evaluation.error_code}: {evaluation.error_message}"
        )
    return evaluation


def _evaluation_record(evaluation: HardwareEvaluation) -> dict[str, Any]:
    return {
        "metrics": evaluation.metrics.to_dict(),
        "validity": evaluation.validity.to_dict(),
        "error_code": evaluation.error_code,
        "error_message": evaluation.error_message,
    }


def _head_projection(entry: Any, metrics: Any, *, mlen: int) -> dict[str, Any]:
    full_contract = dict(entry.profile.local_head_contract)
    if full_contract.get("matrix_mlen") != mlen:
        raise ValueError("local-head profile contract is not exact-MLEN bound")
    semantic = dict(full_contract)
    semantic.pop("matrix_mlen", None)
    breakdown = dict(metrics.output_head_cost_breakdown)
    status = dict(metrics.output_head_status)
    complete = (
        metrics.output_head_location == DECODE_MX_HEAD
        and local_mx_head_status_valid(
            status,
            profile_id=entry.profile_id,
            weight_format=entry.profile.weight_format,
            activation_format=entry.profile.activation_format,
            vector_format=entry.profile.vector_format,
            matrix_mlen=mlen,
        )
        and local_mx_head_breakdown_valid(
            breakdown,
            profile_id=entry.profile_id,
            weight_format=entry.profile.weight_format,
            activation_format=entry.profile.activation_format,
            vector_format=entry.profile.vector_format,
            matrix_mlen=mlen,
        )
    )
    if not complete:
        raise ValueError("local MX-head timing/cost receipt is incomplete")
    return {
        "location": DECODE_MX_HEAD,
        "semantic_contract_sha256": _content_hash(semantic),
        "geometry_receipt_sha256": _content_hash(
            {"contract": full_contract, "status": status, "breakdown": breakdown}
        ),
        "evaluated_mlen": mlen,
        "local_cost_complete": True,
        "idealizations": list(metrics.output_head_idealizations),
    }


def _routing_projection(
    metrics: Any,
    candidate: HardwareCandidate,
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    experts = architecture.get("num_experts")
    if experts is None:
        dense_subject = {"kind": "dense", "model": dict(architecture)}
        return {
            "kind": "dense",
            "routing_source_kind": "analytic_expected",
            "routing_source_receipt_sha256": _content_hash(dense_subject),
            "routing_semantics_sha256": _content_hash(dense_subject),
            "placement_policy_sha256": _content_hash(
                {"kind": "dense", "expert_parallel_mode": None}
            ),
            "geometry_timing_receipt_sha256": _content_hash(
                {"architecture_options": dict(metrics.architecture_options)}
            ),
            "expert_parallel_mode": None,
            "resident_expert_count": None,
            "model_expert_count": None,
            "timing_complete": True,
        }

    workload = metrics.moe_workload
    body = metrics.body_physical_layout
    if not isinstance(workload, Mapping) or not isinstance(body, Mapping):
        raise ValueError("routed-MoE timing lacks workload or body-layout evidence")
    provenance = workload.get("provenance")
    body_provenance = body.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("expert_matrix_cycles_included") is not True
        or provenance.get("router_and_vector_cycles_included") is not True
        or not isinstance(body_provenance, Mapping)
        or body_provenance.get("analytic_timing_valid") is not True
    ):
        raise ValueError("routed-MoE body timing receipt is incomplete")
    route_repricing = workload.get("route_repricing")
    if route_repricing is None:
        source_kind = "analytic_expected"
        source_subject = {
            "source": "analytic_expected_routing",
            "model_num_experts": architecture.get("num_experts"),
            "model_top_k": architecture.get("num_experts_per_tok"),
            "batch": candidate.batch,
        }
    elif isinstance(route_repricing, Mapping):
        for name in ("trace_content_hash", "summary_content_hash"):
            _sha256(route_repricing.get(name), f"routing {name}")
        source_kind = "verified_trace"
        source_subject = dict(route_repricing)
    else:
        raise TypeError("route repricing evidence must be an object or null")
    geometry_fields = {
        "expert_row_tiles_per_layer",
        "expert_padded_rows_per_layer",
        "expert_padding_rows_per_layer",
        "expert_batch_ledger",
        "provenance",
    }
    route_subject = {
        key: value
        for key, value in workload.items()
        if key not in geometry_fields
    }
    placement_subject = {
        "expert_parallel_mode": candidate.expert_parallel_mode,
        "resident_experts": int(experts),
        "model_experts": int(experts),
        "body_storage_scope": "all_model_experts_resident",
    }
    return {
        "kind": "routed_moe",
        "routing_source_kind": source_kind,
        "routing_source_receipt_sha256": _content_hash(source_subject),
        "routing_semantics_sha256": _content_hash(route_subject),
        "placement_policy_sha256": _content_hash(placement_subject),
        "geometry_timing_receipt_sha256": _content_hash(
            {"moe_workload": dict(workload), "body_physical_layout": dict(body)}
        ),
        "expert_parallel_mode": candidate.expert_parallel_mode,
        "resident_expert_count": int(experts),
        "model_expert_count": int(experts),
        "timing_complete": True,
    }


def _point(
    *,
    role: str,
    entry: Any,
    candidate: HardwareCandidate,
    evaluation: HardwareEvaluation,
    numerical_receipt: Mapping[str, Any],
    model: Mapping[str, Any],
    workload: Mapping[str, Any],
    phase_contract: Mapping[str, Any],
    evaluator: Any,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    core = _core()
    metrics = evaluation.metrics
    assert metrics is not None
    mlen = _matrix_mlen(entry)
    if mlen != candidate.mlen:
        raise ValueError("hardware MLEN differs from the numerical profile identity")
    if not metrics.runtime_feasible or not metrics.timing_calibrated:
        raise ValueError("matched point lacks feasible calibrated decode timing")
    if metrics.resource_budget is None:
        raise ValueError("matched point lacks its common resource envelope")
    if metrics.clock_hz is None or metrics.system_area_mm2 is None:
        raise ValueError("matched point lacks clock or system-area evidence")
    record = _evaluation_record(evaluation)
    record_sha = _content_hash(record)
    architecture = model["model_architecture"]
    routing = _routing_projection(metrics, candidate, architecture)
    body_complete = (
        True
        if routing["kind"] == "dense"
        else routing["timing_complete"]
    )
    evaluator_id = str(getattr(evaluator, "evaluator_id", ""))
    provenance = getattr(evaluator, "provenance", None)
    if not evaluator_id or not isinstance(provenance, Mapping):
        raise ValueError("evaluator identity/provenance is unavailable")
    method_hash = _content_hash(numerical_method_contract(entry.profile))
    if numerical_receipt.get("method_contract_sha256") != method_hash:
        raise ValueError("numerical receipt and profile method contracts differ")
    budget_status = metrics.resource_budget
    budget = budget_status.budget
    timing_tier = str(getattr(evaluator, "publication_timing_tier", ""))
    if not timing_tier:
        raise ValueError("evaluator timing tier is unavailable")
    return {
        "schema_version": core.POINT_SCHEMA,
        "role": role,
        "model": dict(model),
        "nominal_precision": physical_cost_signature(
            entry.profile, exact_vector_format=True
        ),
        "numerical_receipt": dict(numerical_receipt),
        "hardware": candidate.to_dict(),
        "workload": dict(workload),
        "phase_contract": dict(phase_contract),
        "clock_hz": float(metrics.clock_hz),
        "resource_receipt": {
            "matrix_pe_equivalents_per_chip": candidate.mlen * candidate.blen,
            "aggregate_multiplier_count": (
                metrics.resource_budget.aggregate_multiplier_count
            ),
            "system_area_mm2": float(metrics.system_area_mm2),
            "aggregate_hbm_capacity_bytes": (
                budget_status.aggregate_hbm_capacity_bytes
            ),
            "aggregate_hbm_bandwidth_bytes_per_s": float(
                budget_status.aggregate_hbm_bandwidth_bytes_per_s
            ),
            "aggregate_area_limit_mm2": float(
                budget.aggregate_area_limit_mm2
            ),
            "aggregate_hbm_capacity_limit_bytes": (
                budget.aggregate_hbm_capacity_limit_bytes
            ),
            "aggregate_hbm_bandwidth_limit_bytes_per_s": float(
                budget.aggregate_hbm_bandwidth_limit_bytes_per_s
            ),
            "resource_budget_sha256": _content_hash(
                metrics.resource_budget.budget.to_dict()
            ),
            "resource_budget_feasible": metrics.resource_budget.feasible,
            "runtime_feasible": metrics.runtime_feasible,
            "timing_complete": metrics.timing_calibrated,
            "body_timing_complete": body_complete,
            "broader_publication_rankable": metrics.whole_model_rankable,
        },
        "output_head": _head_projection(entry, metrics, mlen=mlen),
        "routing": routing,
        "timing": {
            "tpot_ms": float(metrics.tpot_ms),
            "timing_tier": timing_tier,
            "timing_evidence_id": str(metrics.timing_evidence_id),
            "execution_mode": str(metrics.execution_mode),
            "metric_scope": "whole_model_decode_step_local_mx_head",
            "timing_valid": True,
        },
        "source": {
            "artifact_sha256": source_artifact_sha256,
            "record_sha256": record_sha,
            "evaluator_id": evaluator_id,
            "evaluator_provenance_sha256": _content_hash(dict(provenance)),
        },
    }


def handoff_service_receipt(
    analysis: Mapping[str, Any],
    *,
    specialized_point: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact evaluator handoff analysis onto the core receipt."""

    core = _core()
    if analysis.get("schema_version") != "plena-prefill-handoff-analysis-v1":
        raise ValueError("unsupported evaluator handoff analysis")
    rows = analysis.get("regimes")
    if not isinstance(rows, list):
        raise ValueError("handoff analysis has no regime rows")
    source_kind = str(
        analysis.get("source_kind", "measured_prefill_handoff")
    )
    regime = (
        "back_pressure"
        if source_kind == "measured_prefill_handoff"
        else "serial_transfer_plus_admission_no_queue_wait"
    )
    back_pressure = next(
        (row for row in rows if row.get("regime") == regime), None
    )
    if not isinstance(back_pressure, Mapping):
        raise ValueError("handoff analysis lacks the back-pressure regime")
    handoff = analysis.get("handoff")
    if not isinstance(handoff, Mapping):
        raise ValueError("handoff analysis lacks its physical ledger")
    model = specialized_point["model"]
    architecture = model["model_architecture"]
    elements = (
        2
        * int(architecture["num_hidden_layers"])
        * int(architecture["num_key_value_heads"])
        * int(architecture["head_dim"])
        * int(specialized_point["workload"]["input_seq"])
        * int(specialized_point["hardware"]["BATCH"])
    )
    artifact_id = str(analysis.get("input_artifact_id", ""))
    prefix = (
        "prefill-handoff-"
        if source_kind == "measured_prefill_handoff"
        else "analytic-handoff-"
    )
    if not artifact_id.startswith(prefix):
        raise ValueError("handoff analysis input artifact identity is invalid")
    body = {
        "schema_version": core.HANDOFF_SCHEMA,
        "model": dict(model),
        "workload_sha256": _content_hash(specialized_point["workload"]),
        "phase_contract_sha256": _content_hash(
            specialized_point["phase_contract"]
        ),
        "source_point_record_sha256": specialized_point["source"][
            "record_sha256"
        ],
        "input_artifact_id": artifact_id,
        "input_artifact_sha256": artifact_id[len(prefix) :],
        "analysis_sha256": _content_hash(dict(analysis)),
        "source_kind": source_kind,
        "regime": regime,
        "transfer_mode": "bulk",
        "admission_scope": "full_bf16_read_plus_packed_write",
        "layers": int(architecture["num_hidden_layers"]),
        "kv_heads": int(architecture["num_key_value_heads"]),
        "head_dim": int(architecture["head_dim"]),
        "prompt_tokens": int(specialized_point["workload"]["input_seq"]),
        "batch": int(specialized_point["hardware"]["BATCH"]),
        "wire_bits": 16,
        "wire_bytes": float(handoff["wire_bytes"]),
        "decode_cache_bytes": float(handoff["decode_cache_bytes"]),
        "decode_cache_effective_bits_per_element": (
            float(handoff["decode_cache_bytes"]) * 8.0 / elements
        ),
        "nominal_precision_sha256": _content_hash(
            specialized_point["nominal_precision"]
        ),
        "link_generation": str(handoff["direct_link_generation"]),
        "link_bandwidth_bytes_per_s": float(
            core.HANDOFF_LINK_BANDWIDTH_BYTES_PER_S[
                str(handoff["direct_link_generation"])
            ]
        ),
        "link_ports_used": 1,
        "effective_link_bandwidth_bytes_per_s": float(
            core.HANDOFF_LINK_BANDWIDTH_BYTES_PER_S[
                str(handoff["direct_link_generation"])
            ]
        ),
        "transfer_ms": float(back_pressure["transfer_s"]) * 1000.0,
        "admission_bytes": (
            float(handoff["wire_bytes"]) + float(handoff["decode_cache_bytes"])
        ),
        "admission_bandwidth_bytes_per_s": float(
            handoff["admission_bandwidth_bytes_per_s"]
        ),
        "admission_bandwidth_policy": str(
            handoff["admission_bandwidth_policy"]
        ),
        "admission_bandwidth_source_sha256": str(
            handoff["admission_bandwidth_source_sha256"]
        ),
        "admission_calibrated": bool(handoff["admission_calibrated"]),
        "admission_calibration_id": handoff["admission_calibration_id"],
        "admission_evidence_tier": str(handoff["admission_evidence_tier"]),
        "admission_ms": float(back_pressure["admission_s"]) * 1000.0,
        "decode_ready_wait_ms": float(back_pressure["wait_s"]) * 1000.0,
        "publication_rankable": bool(analysis["publication_rankable"]),
    }
    # Prove that the schedule row really uses the bulk transfer and complete
    # admission values from the physical handoff ledger.
    if not math.isclose(
        body["transfer_ms"],
        float(handoff["transfer_bulk_s"]) * 1000.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        body["admission_ms"],
        float(handoff["admission_s"]) * 1000.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("back-pressure row differs from the physical handoff ledger")
    return {**body, "receipt_id": core.handoff_receipt_id(body)}


def physical_handoff_analysis(
    *,
    evaluator: Any,
    entry: Any,
    candidate: HardwareCandidate,
) -> dict[str, Any]:
    """Replay only the physical handoff/admission ledger.

    This remains available when MoE power/compiler/RTL evidence prevents the
    evaluator from constructing its broader request-energy schedule.  No
    dummy energy value is introduced; the comparison consumes only transfer,
    admission, and the explicitly declared decode-ready delay.
    """

    artifact = getattr(evaluator, "handoff_artifact", None)
    backend = getattr(evaluator, "backend", None)
    simulator = getattr(backend, "sim", None)
    dims = getattr(simulator, "dims", None)
    workload = getattr(evaluator, "workload", None)
    if artifact is None or not isinstance(dims, Mapping) or workload is None:
        raise ValueError("live evaluator lacks its handoff physical inputs")
    from decode_dse.hardware.evaluation import precision_request

    root = resolve_simulator_root().root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from analytic_models.disagg_serve.handoff import AdmissionModel, handoff_time

    request = precision_request(entry.profile)
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
    admission = AdmissionModel(
        bandwidth_bytes_per_s=artifact.admission_bandwidth_bytes_per_s,
        quantize_energy_j_per_element=(
            artifact.admission_quantize_energy_j_per_element
        ),
        memory_energy_j_per_byte=artifact.admission_memory_energy_j_per_byte,
        calibrated=artifact.admission_calibrated,
        calibration_id=artifact.admission_calibration_id,
    )
    handoff = handoff_time(
        dict(dims),
        precision.spec,
        int(workload.input_seq),
        candidate.batch,
        link_gen=artifact.direct_link_generation,
        admission=admission,
    )
    return {
        "schema_version": "plena-prefill-handoff-analysis-v1",
        "scope": "physical_transfer_admission_replay_without_energy_claim",
        "source_kind": "measured_prefill_handoff",
        "input_artifact_id": artifact.artifact_id,
        "publication_rankable": False,
        "ordinary_decode_ranking_effect": "none",
        "candidate_id": candidate.candidate_id,
        "batch": candidate.batch,
        "handoff": {
            "wire_bytes": handoff.wire_bytes,
            "decode_cache_bytes": handoff.decode_cache_bytes,
            "transfer_bulk_s": handoff.transfer_bulk_s,
            "admission_s": handoff.admission_s,
            "admission_bandwidth_bytes_per_s": (
                artifact.admission_bandwidth_bytes_per_s
            ),
            "admission_calibrated": artifact.admission_calibrated,
            "admission_calibration_id": artifact.admission_calibration_id,
            "admission_evidence_tier": artifact.admission_evidence_tier,
            "admission_bandwidth_policy": "measured_admission_artifact",
            "admission_bandwidth_source_sha256": _content_hash(
                artifact.to_status()
            ),
            "direct_link_generation": artifact.direct_link_generation,
        },
        "regimes": [
            {
                "regime": "back_pressure",
                "transfer_s": handoff.transfer_bulk_s,
                "admission_s": handoff.admission_s,
                "wait_s": artifact.decode_ready_delay_s,
            }
        ],
        "source_status_sha256": _content_hash(artifact.to_status()),
    }


def analytic_handoff_analysis(
    contract: Mapping[str, Any],
    *,
    evaluator: Any,
    entry: Any,
    candidate: HardwareCandidate,
    aggregate_hbm_bandwidth_bytes_per_s: float,
    resource_receipt_sha256: str,
) -> dict[str, Any]:
    """Replay a serialized BF16 transfer plus full admission with zero queue wait."""

    required = {
        "schema_version",
        "link_generation",
        "admission_bandwidth_policy",
        "admission_evidence_id",
        "admission_evidence_tier",
        "decode_ready_wait_ms",
    }
    value = _object(contract, required, "analytic handoff contract")
    if value["schema_version"] != ANALYTIC_HANDOFF_SCHEMA:
        raise ValueError("unsupported analytic handoff contract")
    if value["decode_ready_wait_ms"] != 0:
        raise ValueError("analytic handoff contract must use zero queue wait")
    if value["admission_bandwidth_policy"] != (
        "matched_candidate_aggregate_hbm_roofline"
    ):
        raise ValueError("analytic handoff bandwidth policy differs")
    _sha256(value["admission_evidence_id"], "analytic admission evidence")
    if (
        not isinstance(value["admission_evidence_tier"], str)
        or not value["admission_evidence_tier"]
    ):
        raise ValueError("analytic admission evidence tier must be explicit")
    core = _core()
    generation = str(value["link_generation"])
    if generation not in core.HANDOFF_LINK_BANDWIDTH_BYTES_PER_S:
        raise ValueError("analytic handoff link generation is unsupported")
    bandwidth = float(aggregate_hbm_bandwidth_bytes_per_s)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("analytic admission bandwidth must be positive")
    backend = getattr(evaluator, "backend", None)
    simulator = getattr(backend, "sim", None)
    dims = getattr(simulator, "dims", None)
    workload = getattr(evaluator, "workload", None)
    if not isinstance(dims, Mapping) or workload is None:
        raise ValueError("live evaluator lacks handoff dimensions/workload")
    from decode_dse.hardware.evaluation import precision_request

    root = resolve_simulator_root().root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from analytic_models.disagg_serve.handoff import AdmissionModel, handoff_time

    request = precision_request(entry.profile)
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
    handoff = handoff_time(
        dict(dims),
        precision.spec,
        int(workload.input_seq),
        candidate.batch,
        link_gen=generation,
        admission=AdmissionModel(
            bandwidth_bytes_per_s=bandwidth,
            calibrated=False,
            calibration_id=None,
        ),
    )
    contract_sha = _content_hash(value)
    return {
        "schema_version": "plena-prefill-handoff-analysis-v1",
        "scope": "serial_transfer_plus_full_admission_no_queue_wait",
        "source_kind": "config_bound_analytic_handoff",
        "input_artifact_id": "analytic-handoff-" + contract_sha,
        "publication_rankable": False,
        "ordinary_decode_ranking_effect": "none",
        "candidate_id": candidate.candidate_id,
        "batch": candidate.batch,
        "handoff": {
            "wire_bytes": handoff.wire_bytes,
            "decode_cache_bytes": handoff.decode_cache_bytes,
            "transfer_bulk_s": handoff.transfer_bulk_s,
            "admission_s": handoff.admission_s,
            "admission_bandwidth_bytes_per_s": bandwidth,
            "admission_calibrated": False,
            "admission_calibration_id": None,
            "admission_evidence_tier": value["admission_evidence_tier"],
            "admission_bandwidth_policy": value[
                "admission_bandwidth_policy"
            ],
            "admission_bandwidth_source_sha256": _sha256(
                resource_receipt_sha256,
                "analytic HBM-envelope source",
            ),
            "direct_link_generation": generation,
        },
        "regimes": [
            {
                "regime": "serial_transfer_plus_admission_no_queue_wait",
                "transfer_s": handoff.transfer_bulk_s,
                "admission_s": handoff.admission_s,
                "wait_s": 0.0,
            }
        ],
        "source_status_sha256": contract_sha,
    }


def _replay_selected_row(
    row: Mapping[str, Any], evaluation: HardwareEvaluation
) -> None:
    """Require the live selected-row replay to reproduce the campaign row."""

    metrics = evaluation.metrics
    assert metrics is not None
    stored = row.get("metrics")
    if not isinstance(stored, Mapping):
        raise ValueError("campaign selected row has no metrics")
    numeric = {
        "tpot_ms": (metrics.tpot_ms, stored.get("tpot_ms")),
        "clock_hz": (metrics.clock_hz, stored.get("clock_hz")),
        "system_area_mm2": (
            metrics.system_area_mm2,
            stored.get("system_area_mm2"),
        ),
    }
    for name, (observed, expected) in numeric.items():
        if observed is None or expected is None or not math.isclose(
            float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"selected source replay changed {name}")
    exact = {
        "timing_evidence_id": (
            metrics.timing_evidence_id,
            stored.get("timing_evidence_id"),
        ),
        "execution_mode": (metrics.execution_mode, stored.get("execution_mode")),
        "runtime_feasible": (
            metrics.runtime_feasible,
            stored.get("runtime_feasible"),
        ),
    }
    for name, (observed, expected) in exact.items():
        if observed != expected:
            raise ValueError(f"selected source replay changed {name}")
    boundary = stored.get("output_head_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("location") != metrics.output_head_location
        or metrics.output_head_location != DECODE_MX_HEAD
    ):
        raise ValueError("selected source replay changed output-head placement")


def _matched_specialized_candidate(
    selected: HardwareCandidate,
) -> HardwareCandidate:
    if 65_536 % selected.mlen:
        raise ValueError("selected MLEN cannot derive an integer matched BLEN")
    candidate = replace(selected, blen=65_536 // selected.mlen)
    if candidate.mlen * candidate.blen != 65_536:
        raise AssertionError("matched-specialized derivation changed")
    return candidate


def producer_receipt_id(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("receipt_id", None)
    body.pop("content_sha256", None)
    return "matched-decode-producer-" + _content_hash(body)


def _producer_receipt(
    *,
    source: ResolvedCampaignSource,
    numerical_evidence: ResolvedNumericalEvidence,
    evaluator_replay: Mapping[str, Any],
    comparison: Mapping[str, Any],
    selected_record_sha256: str,
    specialized_record_sha256: str,
    shared_record_sha256: str,
    evaluation_count: int,
) -> dict[str, Any]:
    core = _core()
    adapter_path = Path(__file__).resolve()
    core_path = Path(core.__file__).resolve()
    body = {
        "schema_version": PRODUCER_RECEIPT_SCHEMA,
        "evidence_class": PRODUCER_EVIDENCE_CLASS,
        "publication_rankable": False,
        "claim_boundary": {
            "hardware_evaluator_execution_replayed_on_load": True,
            "numerical_source_hashes_and_ledgers_replayed_on_load": True,
            "independent_numerical_execution_replayed_on_load": False,
            "numerical_adversarial_authentication_claimed": False,
            "accuracy_evidence_label": NUMERICAL_EVIDENCE_CLASS,
        },
        "source_receipt": dict(source.receipt),
        "numerical_evidence_receipt": dict(numerical_evidence.receipt),
        "evaluator_replay": dict(evaluator_replay),
        "comparison": dict(comparison),
        "replay": {
            "state": "succeeded",
            "selection_role": source.receipt["selection_role"],
            "selected_campaign_record_hash": source.receipt[
                "selected_record_hash"
            ],
            "selected_live_replay_sha256": selected_record_sha256,
            "derived_specialized_record_sha256": specialized_record_sha256,
            "shared_plena_record_sha256": shared_record_sha256,
            "evaluator_call_count": evaluation_count,
            "selected_source_membership_replayed": True,
            "shared_ratio_was_not_a_selection_objective": True,
            "strict_evaluator_replay_on_load": True,
            "strict_hardware_evaluator_replay_on_load": True,
            "numerical_execution_replay_on_load": False,
        },
        "producer": {
            "adapter_path": str(adapter_path),
            "adapter_sha256": _file_hash(adapter_path),
            "simulator_core_path": str(core_path),
            "simulator_core_sha256": _file_hash(core_path),
        },
    }
    receipt_id = producer_receipt_id(body)
    with_id = {**body, "receipt_id": receipt_id}
    return {**with_id, "content_sha256": _content_hash(with_id)}


def reprice_matched_pair(
    *,
    evaluator: Any,
    evaluator_replay: Mapping[str, Any],
    selected_source: ResolvedCampaignSource,
    numerical_evidence: ResolvedNumericalEvidence,
    model: Mapping[str, Any],
    workload: Mapping[str, Any],
    phase_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay campaign ancestry and price one matched pair.

    The selected campaign row is context only.  If it does not already carry
    65,536 multipliers per chip, the decode-specialized arm is the sealed
    BLEN-only derivative.  The other arm changes only MLEN/BLEN/VLEN to the
    paper geometry.
    """

    if not isinstance(selected_source, ResolvedCampaignSource):
        raise TypeError("repricing requires a producer-resolved campaign source")
    if not isinstance(numerical_evidence, ResolvedNumericalEvidence):
        raise TypeError("repricing requires resolved exact-MLEN numerical evidence")
    replay = validate_evaluator_replay_invocation(evaluator_replay)
    from decode_dse.hardware.evaluation import construct_study_from_argv

    replay_study = construct_study_from_argv(replay["argv"])
    replay_evaluator = replay_study.evaluator
    if _evaluator_identity(evaluator) != _evaluator_identity(replay_evaluator):
        raise ValueError("caller evaluator differs from the sealed replay evaluator")
    evaluator = replay_evaluator
    live_model, live_workload, live_phase, orchestration = _comparison_context(
        evaluator,
        replay,
    )
    if (
        dict(model) != live_model
        or dict(workload) != live_workload
        or dict(phase_contract) != live_phase
    ):
        raise ValueError(
            "comparison model/workload/phase differs from reconstructed study"
        )
    if selected_source.receipt["selection_role"] != orchestration[
        "selection_role"
    ]:
        raise ValueError("campaign selection role differs from orchestration")
    handoff_contract = orchestration["handoff"]
    analytic_handoff_contract = (
        handoff_contract["analytic_contract"]
        if handoff_contract["mode"] == "config_bound_analytic"
        else None
    )
    refreshed = resolve_campaign_source(
        selected_source.receipt["campaign_path"],
        hardware_artifact_paths=tuple(
            selected_source.receipt["hardware_artifact_paths"]
        ),
        evaluator=evaluator,
        model=model,
        workload=workload,
        phase_contract=phase_contract,
        selection_role=selected_source.receipt["selection_role"],
    )
    if (
        dict(refreshed.receipt) != dict(selected_source.receipt)
        or dict(refreshed.row) != dict(selected_source.row)
    ):
        raise ValueError("resolved campaign source changed before repricing")
    selected_source = refreshed
    core = _core()
    row = selected_source.row
    selected_candidate = HardwareCandidate.from_dict(row["hardware"])
    numerical_contract = orchestration["numerical"]
    if (
        selected_candidate.mlen != numerical_contract["source_mlen"]
        or numerical_evidence.receipt.get("suite")
        != numerical_contract["suite"]
    ):
        raise ValueError(
            "selected MLEN/numerical suite differs from orchestration"
        )
    refreshed_numerical = resolve_mlen_numerical_evidence(
        numerical_evidence.receipt["completion_path"],
        selected_source_profile_id=selected_source.receipt["selected_profile_id"],
        selected_hardware_mlen=selected_candidate.mlen,
        model=model,
        suite=numerical_evidence.receipt["suite"],
    )
    if dict(refreshed_numerical.receipt) != dict(numerical_evidence.receipt):
        raise ValueError("resolved MLEN numerical evidence changed before repricing")
    numerical_evidence = refreshed_numerical
    specialized_entry = numerical_evidence.specialized_entry
    shared_entry = numerical_evidence.shared_entry
    specialized_numerical_result = numerical_evidence.specialized_result
    shared_numerical_result = numerical_evidence.shared_result
    specialized_numerical_receipt = numerical_evidence.specialized_receipt
    shared_numerical_receipt = numerical_evidence.shared_receipt
    bf16_accuracy_oracle = numerical_evidence.bf16_oracle
    if _matrix_mlen(specialized_entry) != selected_candidate.mlen:
        raise ValueError("specialized entry lacks the selected exact MLEN")
    specialized_candidate = _matched_specialized_candidate(selected_candidate)
    shared_candidate = paper_geometry_candidate(selected_candidate)
    if specialized_candidate == shared_candidate:
        raise ValueError(
            "selected campaign role resolves to MLEN=2048; the sealed comparison "
            "has no non-shared contrast and will not substitute another row"
        )
    if _matrix_mlen(shared_entry) != 2048:
        raise ValueError("paper arm lacks an exact MLEN=2048 numerical identity")
    specialized_method = numerical_method_contract(specialized_entry.profile)
    shared_method = numerical_method_contract(shared_entry.profile)
    if specialized_method != shared_method:
        raise ValueError("comparison entries differ beyond exact MLEN derivatives")
    if physical_cost_signature(
        specialized_entry.profile, exact_vector_format=True
    ) != physical_cost_signature(shared_entry.profile, exact_vector_format=True):
        raise ValueError("comparison profiles differ on nominal physical precision")
    method_hash = _content_hash(specialized_method)
    for entry, numerical_receipt, candidate in (
        (specialized_entry, specialized_numerical_receipt, specialized_candidate),
        (shared_entry, shared_numerical_receipt, shared_candidate),
    ):
        if (
            numerical_receipt.get("profile_id") != _profile_id(entry)
            or numerical_receipt.get("evaluated_mlen") != candidate.mlen
            or numerical_receipt.get("method_contract_sha256") != method_hash
        ):
            raise ValueError("exact-MLEN numerical receipt and repricing entry differ")
    numerical_derivation_receipt_sha256 = _content_hash(
        numerical_evidence.receipt
    )

    selected_evaluation = _run_one(
        evaluator,
        specialized_entry,
        specialized_numerical_result,
        selected_candidate,
    )
    _replay_selected_row(row, selected_evaluation)
    selected_record_sha = _content_hash(_evaluation_record(selected_evaluation))
    evaluation_count = 1
    if specialized_candidate == selected_candidate:
        specialized_evaluation = selected_evaluation
    else:
        specialized_evaluation = _run_one(
            evaluator,
            specialized_entry,
            specialized_numerical_result,
            specialized_candidate,
        )
        evaluation_count += 1
    shared_evaluation = _run_one(
        evaluator,
        shared_entry,
        shared_numerical_result,
        shared_candidate,
    )
    evaluation_count += 1
    specialized_record = _evaluation_record(specialized_evaluation)
    shared_record = _evaluation_record(shared_evaluation)
    specialized_record_sha = _content_hash(specialized_record)
    shared_record_sha = _content_hash(shared_record)
    specialized_point = _point(
        role="decode_specialized",
        entry=specialized_entry,
        candidate=specialized_candidate,
        evaluation=specialized_evaluation,
        numerical_receipt=specialized_numerical_receipt,
        model=model,
        workload=workload,
        phase_contract=phase_contract,
        evaluator=evaluator,
        source_artifact_sha256=_content_hash(
            {
                "scope": "matched_specialized_blen_derivative",
                "selected_source_receipt_id": selected_source.receipt["receipt_id"],
                "record": specialized_record,
            }
        ),
    )
    shared_point = _point(
        role="shared_plena_geometry",
        entry=shared_entry,
        candidate=shared_candidate,
        evaluation=shared_evaluation,
        numerical_receipt=shared_numerical_receipt,
        model=model,
        workload=workload,
        phase_contract=phase_contract,
        evaluator=evaluator,
        source_artifact_sha256=_content_hash(
            {
                "scope": "single_shared_plena_geometry_reprice",
                "selected_source_receipt_id": selected_source.receipt["receipt_id"],
                "record": shared_record,
            }
        ),
    )
    handoff_analysis = (
        analytic_handoff_analysis(
            analytic_handoff_contract,
            evaluator=evaluator,
            entry=specialized_entry,
            candidate=specialized_candidate,
            aggregate_hbm_bandwidth_bytes_per_s=float(
                specialized_point["resource_receipt"][
                    "aggregate_hbm_bandwidth_bytes_per_s"
                ]
            ),
            resource_receipt_sha256=_content_hash(
                specialized_point["resource_receipt"]
            ),
        )
        if analytic_handoff_contract is not None
        else physical_handoff_analysis(
            evaluator=evaluator,
            entry=specialized_entry,
            candidate=specialized_candidate,
        )
    )
    handoff = handoff_service_receipt(
        handoff_analysis,
        specialized_point=specialized_point,
    )
    ancestry_body = {
        "schema_version": core.ANCESTRY_SCHEMA,
        "selected_source_receipt_id": selected_source.receipt["receipt_id"],
        "selected_source_profile_id": selected_source.receipt[
            "selected_profile_id"
        ],
        "selected_candidate_id": selected_candidate.candidate_id,
        "selected_hardware": selected_candidate.to_dict(),
        "selected_replay_record_sha256": selected_record_sha,
        "derived_specialized_profile_id": _profile_id(specialized_entry),
        "numerical_derivation_receipt_sha256": (
            numerical_derivation_receipt_sha256
        ),
        "derived_specialized_candidate_id": specialized_candidate.candidate_id,
        "derived_specialized_hardware": specialized_candidate.to_dict(),
        "derivation_rule": (
            "preserve_all_axes_except_blen_set_blen_to_65536_div_mlen"
        ),
        "selected_was_already_multiplier_matched": (
            selected_candidate == specialized_candidate
        ),
    }
    comparison_input = {
        "schema_version": core.INPUT_SCHEMA,
        "arms": {
            "decode_specialized": specialized_point,
            "shared_plena_geometry": shared_point,
        },
        "selection_ancestry": {
            **ancestry_body,
            "receipt_id": core.ancestry_receipt_id(ancestry_body),
        },
        "handoff": handoff,
        "bf16_accuracy_oracle": dict(bf16_accuracy_oracle),
    }
    comparison = core.build_comparison(comparison_input)
    return _producer_receipt(
        source=selected_source,
        numerical_evidence=numerical_evidence,
        evaluator_replay=replay,
        comparison=comparison,
        selected_record_sha256=selected_record_sha,
        specialized_record_sha256=specialized_record_sha,
        shared_record_sha256=shared_record_sha,
        evaluation_count=evaluation_count,
    )


def materialize_from_campaign(
    *,
    hardware_launch_argv: Sequence[str],
    orchestration_contract_path: str | Path,
    campaign_path: str | Path,
    hardware_artifact_paths: Sequence[str | Path],
    mlen_completion_path: str | Path,
    numerical_suite: str = "refinement",
) -> dict[str, Any]:
    """Reconstruct all producers and materialize one source-bound pair.

    This is the target-runner entry point.  No metric is accepted from a
    caller: the campaign role comes from the immutable orchestration contract,
    accuracy comes from the exact-MLEN completion, and timing is rerun by the
    reconstructed hardware evaluator.
    """

    replay = build_evaluator_replay_invocation(
        hardware_launch_argv,
        orchestration_contract_path=orchestration_contract_path,
    )
    from decode_dse.hardware.evaluation import construct_study_from_argv

    study = construct_study_from_argv(replay["argv"])
    evaluator = study.evaluator
    model, workload, phase, orchestration = _comparison_context(
        evaluator,
        replay,
    )
    source = resolve_campaign_source(
        campaign_path,
        hardware_artifact_paths=tuple(hardware_artifact_paths),
        evaluator=evaluator,
        model=model,
        workload=workload,
        phase_contract=phase,
        selection_role=orchestration["selection_role"],
    )
    selected_candidate = HardwareCandidate.from_dict(source.row["hardware"])
    numerical_contract = orchestration["numerical"]
    if (
        selected_candidate.mlen != numerical_contract["source_mlen"]
        or numerical_suite != numerical_contract["suite"]
    ):
        raise ValueError(
            "selected MLEN/numerical suite differs from orchestration"
        )
    numerical = resolve_mlen_numerical_evidence(
        mlen_completion_path,
        selected_source_profile_id=source.receipt["selected_profile_id"],
        selected_hardware_mlen=selected_candidate.mlen,
        model=model,
        suite=numerical_suite,
    )
    return reprice_matched_pair(
        evaluator=evaluator,
        evaluator_replay=replay,
        selected_source=source,
        numerical_evidence=numerical,
        model=model,
        workload=workload,
        phase_contract=phase,
    )


def _verify_producer_source_files(receipt: Mapping[str, Any]) -> None:
    campaign_path = Path(receipt["campaign_path"])
    if _file_hash(campaign_path) != receipt["campaign_file_sha256"]:
        raise ValueError("producer campaign file changed")
    campaign = _load_campaign(campaign_path)
    if campaign["content_hash"] != receipt["campaign_content_sha256"]:
        raise ValueError("producer campaign content identity changed")
    selected = _campaign_candidate(campaign, receipt["selection_role"])
    selector_replay = _replay_campaign_selector(
        campaign_path.resolve(),
        hardware_artifact_paths=tuple(
            Path(value).resolve() for value in receipt["hardware_artifact_paths"]
        ),
        campaign=campaign,
    )
    if any(receipt[name] != value for name, value in selector_replay.items()):
        raise ValueError("producer campaign selector replay differs")
    rows = []
    selected_header = None
    observed = []
    for raw_path in receipt["hardware_artifact_paths"]:
        path = Path(raw_path)
        digest = _file_hash(path)
        observed.append(digest)
        header, artifact_rows = load_hardware_artifact(path)
        for row in artifact_rows:
            if (
                row.get("profile_id") == receipt["selected_profile_id"]
                and row.get("candidate_id") == receipt["selected_candidate_id"]
            ):
                rows.append(row)
                selected_header = header
                if str(path.resolve()) != receipt["selected_artifact_path"]:
                    raise ValueError("producer selected artifact path changed")
                if digest != receipt["selected_artifact_sha256"]:
                    raise ValueError("producer selected artifact hash changed")
    if sorted(observed) != receipt["hardware_artifact_sha256s"]:
        raise ValueError("producer hardware artifact set changed")
    if len(rows) != 1 or selected_header is None:
        raise ValueError("producer selected source row is missing or duplicated")
    row = rows[0]
    provenance = selected_header.get("provenance")
    evaluator_provenance = (
        provenance.get("evaluator_provenance")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        row.get("record_hash") != selected["record_hash"]
        or row.get("record_hash") != receipt["selected_record_hash"]
        or _row_body_sha256(row) != receipt["selected_record_hash"]
        or _content_hash(dict(row)) != receipt["selected_row_sha256"]
        or not isinstance(provenance, Mapping)
        or not isinstance(evaluator_provenance, Mapping)
        or _content_hash(dict(provenance))
        != receipt["selected_artifact_provenance_sha256"]
        or provenance.get("evaluator_version") != receipt["evaluator_id"]
        or _content_hash(dict(evaluator_provenance))
        != receipt["evaluator_provenance_sha256"]
    ):
        raise ValueError("producer selected source membership differs")


def _load_producer_receipt_structural(path: str | Path) -> dict[str, Any]:
    """Replay schemas/arithmetic without granting producer authentication."""

    value = json.loads(Path(path).read_bytes())
    receipt = _object(value, _PRODUCER_FIELDS, "producer receipt")
    expected_claim_boundary = {
        "hardware_evaluator_execution_replayed_on_load": True,
        "numerical_source_hashes_and_ledgers_replayed_on_load": True,
        "independent_numerical_execution_replayed_on_load": False,
        "numerical_adversarial_authentication_claimed": False,
        "accuracy_evidence_label": NUMERICAL_EVIDENCE_CLASS,
    }
    if (
        receipt["schema_version"] != PRODUCER_RECEIPT_SCHEMA
        or receipt["evidence_class"] != PRODUCER_EVIDENCE_CLASS
        or receipt["publication_rankable"] is not False
        or receipt["claim_boundary"] != expected_claim_boundary
    ):
        raise ValueError("producer receipt classification differs")
    expected_content = dict(receipt)
    content_sha = expected_content.pop("content_sha256")
    if content_sha != _content_hash(expected_content):
        raise ValueError("producer receipt content hash differs")
    if receipt["receipt_id"] != producer_receipt_id(receipt):
        raise ValueError("producer receipt identity differs")
    producer = _object(
        receipt["producer"], _PRODUCER_METADATA_FIELDS, "producer metadata"
    )
    adapter_path = Path(__file__).resolve()
    core_path = Path(_core().__file__).resolve()
    if (
        producer["adapter_path"] != str(adapter_path)
        or producer["adapter_sha256"] != _file_hash(adapter_path)
        or producer["simulator_core_path"] != str(core_path)
        or producer["simulator_core_sha256"] != _file_hash(core_path)
    ):
        raise ValueError("producer adapter/Simulator core binding differs")
    source = validate_selected_source_receipt(receipt["source_receipt"])
    validate_evaluator_replay_invocation(receipt["evaluator_replay"])
    core = _core()
    comparison = receipt["comparison"]
    rebuilt = core.build_comparison(comparison["input"])
    if comparison != rebuilt:
        raise ValueError("producer core comparison replay differs")
    specialized = comparison["input"]["arms"]["decode_specialized"]
    shared = comparison["input"]["arms"]["shared_plena_geometry"]
    ancestry = comparison["input"]["selection_ancestry"]
    replay = receipt["replay"]
    numerical = resolve_mlen_numerical_evidence(
        receipt["numerical_evidence_receipt"]["completion_path"],
        selected_source_profile_id=source["selected_profile_id"],
        selected_hardware_mlen=int(ancestry["selected_hardware"]["MLEN"]),
        model=specialized["model"],
        suite=receipt["numerical_evidence_receipt"]["suite"],
    )
    if (
        dict(numerical.receipt) != dict(receipt["numerical_evidence_receipt"])
        or specialized["numerical_receipt"] != numerical.specialized_receipt
        or shared["numerical_receipt"] != numerical.shared_receipt
        or comparison["input"]["bf16_accuracy_oracle"]
        != numerical.bf16_oracle
        or source["model_sha256"] != _content_hash(specialized["model"])
        or source["workload_sha256"] != _content_hash(specialized["workload"])
        or source["phase_contract_sha256"]
        != _content_hash(specialized["phase_contract"])
        or source["evaluator_id"] != specialized["source"]["evaluator_id"]
        or source["evaluator_provenance_sha256"]
        != specialized["source"]["evaluator_provenance_sha256"]
        or ancestry["selected_source_receipt_id"] != source["receipt_id"]
        or ancestry["selected_replay_record_sha256"]
        != replay.get("selected_live_replay_sha256")
        or specialized["source"]["record_sha256"]
        != replay.get("derived_specialized_record_sha256")
        or shared["source"]["record_sha256"]
        != replay.get("shared_plena_record_sha256")
        or replay.get("state") != "succeeded"
        or replay.get("selected_source_membership_replayed") is not True
        or replay.get("shared_ratio_was_not_a_selection_objective") is not True
        or replay.get("strict_evaluator_replay_on_load") is not True
        or replay.get("strict_hardware_evaluator_replay_on_load") is not True
        or replay.get("numerical_execution_replay_on_load") is not False
        or numerical.receipt.get("evidence_class")
        != NUMERICAL_EVIDENCE_CLASS
        or numerical.receipt.get("independent_numerical_execution_replayed")
        is not False
        or numerical.receipt.get("adversarial_tamper_resistance_claimed")
        is not False
    ):
        raise ValueError("producer replay bindings differ")
    return receipt


def load_producer_receipt_strict(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Rerun hardware timing and revalidate source-bound numerical evidence.

    ``verify_sources=False`` is intentionally unsupported: a strict receipt is
    meaningful only when every launch input, campaign row, MLEN source row,
    and code producer is revalidated and the complete newly produced receipt
    is equal.  The MLEN numerical kernels are *not* independently re-executed
    here; the sealed claim boundary records that normal-scientific-provenance
    limitation explicitly.
    """

    if not verify_sources:
        raise ValueError("strict producer replay cannot skip source verification")
    receipt = _load_producer_receipt_structural(path)
    source = receipt["source_receipt"]
    replay = receipt["evaluator_replay"]
    numerical = receipt["numerical_evidence_receipt"]
    _verify_producer_source_files(source)
    rebuilt = materialize_from_campaign(
        hardware_launch_argv=replay["argv"],
        orchestration_contract_path=replay["orchestration_contract_path"],
        campaign_path=source["campaign_path"],
        hardware_artifact_paths=source["hardware_artifact_paths"],
        mlen_completion_path=numerical["completion_path"],
        numerical_suite=numerical["suite"],
    )
    if receipt != rebuilt:
        raise ValueError(
            "strict evaluator replay differs from the producer receipt"
        )
    return receipt


def load_producer_receipt(
    path: str | Path,
    *,
    verify_sources: bool = True,
) -> dict[str, Any]:
    """Load a receipt; source verification always means strict evaluator replay."""

    if verify_sources:
        return load_producer_receipt_strict(path, verify_sources=True)
    return _load_producer_receipt_structural(path)


def write_producer_receipt(
    path: str | Path, receipt: Mapping[str, Any]
) -> Path:
    """Atomically create an immutable producer receipt after full replay."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(dict(receipt)) + b"\n"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace a different producer receipt: {destination}"
            )
        load_producer_receipt(destination, verify_sources=True)
        return destination
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        load_producer_receipt_strict(temporary_name, verify_sources=True)
        os.link(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    load_producer_receipt(destination, verify_sources=True)
    return destination


__all__ = [
    "ANALYTIC_HANDOFF_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "EVALUATOR_REPLAY_SCHEMA",
    "NUMERICAL_EVIDENCE_CLASS",
    "NUMERICAL_EVIDENCE_SCHEMA",
    "ORCHESTRATION_SCHEMA",
    "PRODUCER_EVIDENCE_CLASS",
    "PRODUCER_RECEIPT_SCHEMA",
    "SELECTED_SOURCE_SCHEMA",
    "ResolvedCampaignSource",
    "ResolvedNumericalEvidence",
    "analytic_handoff_analysis",
    "build_evaluator_replay_invocation",
    "evaluator_replay_receipt_id",
    "handoff_service_receipt",
    "load_producer_receipt",
    "load_producer_receipt_strict",
    "materialize_from_campaign",
    "numerical_method_contract",
    "paper_geometry_candidate",
    "physical_handoff_analysis",
    "producer_receipt_id",
    "reprice_matched_pair",
    "resolve_campaign_source",
    "resolve_mlen_numerical_evidence",
    "selected_source_receipt_id",
    "validate_evaluator_replay_invocation",
    "validate_selected_source_receipt",
    "write_producer_receipt",
]
