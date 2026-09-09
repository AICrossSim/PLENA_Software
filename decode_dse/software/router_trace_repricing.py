"""Materialize fail-closed model/config overlays for route-trace repricing.

The native analytic model accepts two route-dependent fields.  This module
binds those fields to a verified summary without changing model architecture,
resident expert storage, precision, or hardware axes other than narrowing each
derived config to the batch for which its aggregate was computed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
SUMMARY_SCHEMA = "plena-qwen3-moe-router-trace-summary/v1"
MODEL_OVERLAY_SCHEMA = "plena-qwen3-moe-route-model-overlay/v1"
CONFIG_OVERLAY_SCHEMA = "plena-qwen3-moe-route-study-config-overlay/v1"
BUNDLE_SCHEMA = "plena-qwen3-moe-route-repricing-bundle/v1"
AGGREGATION_POLICY = (
    "consecutive_nonoverlapping_windows_conservative_observed_max/v1"
)
SUPPORTED_FIELDS = (
    "moe_unique_experts_per_step",
    "moe_routing_imbalance_factor",
)
UNSUPPORTED_TIMING_INPUTS = [
    "per-window/per-layer expert identities",
    "per-window/per-layer expert-token histograms",
    "source-chip and placement decisions",
    "expert-cache hits, misses, and reuse",
    "BLEN-specific timing derived from the measured histogram",
    "collective topology, contention, packetization, and overlap",
]
EXPECTED_REPRICING_CONTRACT = {
    "schema_version": "decode-router-trace-repricing-contract/v1",
    "batch_source": "hardware_space.BATCH",
    "windowing": "consecutive_nonoverlapping_global_trace_steps",
    "tail_policy": "drop_and_report_incomplete_final_window",
    "aggregation_policy": AGGREGATION_POLICY,
    "supported_override_fields": list(SUPPORTED_FIELDS),
    "resident_expert_storage": "all_128_experts",
    "execution_scope": "legacy_aggregate_analytic_route_sensitivity_only",
    "publication_rankable": False,
    "hardware_rankable": False,
    "selection_eligible": False,
}
EXPECTED_MODEL = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "vocab_size": 151936,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_hash", None)
    return body | {"content_hash": canonical_hash(body)}


def _encode(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def load_hashed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    body = dict(value)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError(f"content hash mismatch: {path}")
    return value


def _atomic_install(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to replace concurrently installed artifact: {path}"
                )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_model(config: Mapping[str, Any], simulator_root: Path) -> Path:
    raw = str(config.get("sim_model", ""))
    candidate = Path(raw)
    if candidate.suffix == ".json" and candidate.is_file():
        return candidate.resolve()
    path = simulator_root / "compiler" / "doc" / "Model_Lib" / (
        raw if raw.endswith(".json") else f"{raw}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"simulator model is missing: {path}")
    return path.resolve()


def _validate_base(
    config: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    if config.get("model_name") != MODEL_ID or config.get("model_revision") != MODEL_REVISION:
        raise ValueError("study config differs from the sealed model identity")
    if config.get("router_trace_repricing") != EXPECTED_REPRICING_CONTRACT:
        raise ValueError("study config route-repricing contract differs")
    for key, expected in EXPECTED_MODEL.items():
        if model.get(key) != expected:
            raise ValueError(f"base simulator model {key} differs from the target")
    for key in ("num_shared_experts", "shared_expert_intermediate_size"):
        if model.get(key) not in (None, 0):
            raise ValueError(f"base simulator model {key} enables shared experts")
    forbidden = {
        "moe_unique_experts_per_step",
        "moe_routing_imbalance_factor",
        "moe_route_repricing",
        "content_hash",
    }
    overlap = forbidden.intersection(model)
    if overlap:
        raise ValueError(
            "base simulator model already contains route overrides: "
            + ",".join(sorted(overlap))
        )
    if "route_repricing" in config or "content_hash" in config:
        raise ValueError("base study config is already a derived artifact")


def validate_summary(summary: Mapping[str, Any]) -> None:
    body = dict(summary)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError("route summary content hash mismatch")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("model_id") != MODEL_ID
        or summary.get("model_revision") != MODEL_REVISION
        or summary.get("aggregation_policy") != AGGREGATION_POLICY
    ):
        raise ValueError("route summary differs from the supported target contract")
    if summary.get("supported_override_fields") != list(SUPPORTED_FIELDS):
        raise ValueError("route summary exposes unsupported override fields")
    if summary.get("unsupported_timing_inputs") != UNSUPPORTED_TIMING_INPUTS:
        raise ValueError("route summary scope limits differ")
    for field in ("trace_content_hash", "content_hash"):
        if not _SHA256.fullmatch(str(summary.get(field, ""))):
            raise ValueError(f"route summary has an invalid {field}")
    classification = summary.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(field) is not False
        for field in (
            "publication_rankable",
            "hardware_rankable",
            "selection_eligible",
        )
    ):
        raise ValueError("route summary must remain fail-closed")
    batches = summary.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("route summary contains no batch rows")
    source = summary.get("source_binding")
    if not isinstance(source, Mapping) or source.get("collector_verified") is not True:
        raise ValueError("route summary lacks verified collector provenance")
    for path_field, hash_field in (
        ("router_index_path", "router_index_sha256"),
        ("placement_input_path", "placement_input_sha256"),
        ("router_trace_evidence_path", "router_trace_evidence_sha256"),
    ):
        path = Path(str(source.get(path_field, "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or file_hash(path) != source.get(hash_field)
        ):
            raise ValueError(f"route summary source {path_field} is missing or changed")
    configured = [row.get("batch_size") for row in batches if isinstance(row, Mapping)]
    if len(configured) != len(batches) or configured != sorted(set(configured)):
        raise ValueError("route summary batches are malformed or duplicated")


def _batch_rows(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        row
        for row in summary["batches"]
        if isinstance(row, Mapping) and row.get("status") == "materializable"
    )
    declared = [int(value) for value in summary.get("materializable_batches", [])]
    if declared != [int(row["batch_size"]) for row in rows]:
        raise ValueError("route summary materializable batch index differs")
    if not rows:
        raise ValueError("route summary has no materializable batch")
    for row in rows:
        override = row.get("supported_override")
        if not isinstance(override, Mapping) or set(override).intersection(
            SUPPORTED_FIELDS
        ) != set(SUPPORTED_FIELDS):
            raise ValueError("route summary batch has no supported override")
        if override.get("aggregation_policy") != AGGREGATION_POLICY:
            raise ValueError("route summary batch aggregation policy differs")
        unique = override.get("moe_unique_experts_per_step")
        imbalance = override.get("moe_routing_imbalance_factor")
        assignments = int(row["batch_size"]) * 8
        if (
            isinstance(unique, bool)
            or not isinstance(unique, int)
            or not 8 <= unique <= min(128, assignments)
        ):
            raise ValueError("route summary unique-expert override is invalid")
        if (
            isinstance(imbalance, bool)
            or not isinstance(imbalance, (int, float))
            or not 1.0 <= float(imbalance) <= float(unique)
        ):
            raise ValueError("route summary imbalance override is invalid")
    return rows


def _document_receipt(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "content_hash": value["content_hash"],
    }


def _materialize_documents(
    *,
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    base_config: Mapping[str, Any],
    base_config_path: Path,
    base_model: Mapping[str, Any],
    base_model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    batch = int(row["batch_size"])
    override = dict(row["supported_override"])
    injected = {field: override[field] for field in SUPPORTED_FIELDS}
    classification = {
        "evidence": "measured_trace_aggregate_analytic_sensitivity",
        "publication_rankable": False,
        "hardware_rankable": False,
        "selection_eligible": False,
        "blockers": list(summary["classification"]["blockers"]),
    }
    overlay_metadata = {
        "schema": MODEL_OVERLAY_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": summary["trace_content_hash"],
        "summary_content_hash": summary["content_hash"],
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_hash(summary_path),
        "batch_size": batch,
        "aggregation_policy": AGGREGATION_POLICY,
        "injected_fields": injected,
        "unsupported_timing_inputs": list(UNSUPPORTED_TIMING_INPUTS),
        "resident_expert_storage": {
            "num_experts": 128,
            "policy": "all_experts_remain_resident",
            "changed_by_overlay": False,
        },
        "classification": classification,
    }
    model_document = _hashed_body(
        dict(base_model)
        | injected
        | {"moe_route_repricing": overlay_metadata}
    )
    model_payload = _encode(model_document)
    model_sha = hashlib.sha256(model_payload).hexdigest()
    model_path = output_dir / f"model.batch-{batch}.{model_sha}.json"
    _atomic_install(model_path, model_payload)

    hardware_space = base_config.get("hardware_space")
    if not isinstance(hardware_space, Mapping):
        raise ValueError("base config is missing hardware_space")
    configured_batches = hardware_space.get("BATCH")
    if not isinstance(configured_batches, list) or batch not in configured_batches:
        raise ValueError(f"batch {batch} is outside the base hardware space")
    config_metadata = {
        "schema": CONFIG_OVERLAY_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": summary["trace_content_hash"],
        "summary_content_hash": summary["content_hash"],
        "batch_size": batch,
        "aggregation_policy": AGGREGATION_POLICY,
        "sim_model_overlay": {
            "path": str(model_path.resolve()),
            "sha256": model_sha,
            "content_hash": model_document["content_hash"],
        },
        "base_study_config": {
            "path": str(base_config_path.resolve()),
            "sha256": file_hash(base_config_path),
        },
        "base_simulator_model": {
            "path": str(base_model_path.resolve()),
            "sha256": file_hash(base_model_path),
        },
        "execution_scope": (
            "legacy_aggregate_analytic_route_sensitivity_only"
        ),
        "classification": classification,
    }
    derived_config = dict(base_config)
    derived_config["sim_model"] = str(model_path.resolve())
    derived_config["hardware_space"] = dict(hardware_space) | {"BATCH": [batch]}
    derived_config["route_repricing"] = config_metadata
    config_document = _hashed_body(derived_config)
    config_payload = _encode(config_document)
    config_sha = hashlib.sha256(config_payload).hexdigest()
    config_path = output_dir / f"study_config.batch-{batch}.{config_sha}.json"
    _atomic_install(config_path, config_payload)
    return {
        "batch_size": batch,
        "supported_override": injected,
        "model_overlay": _document_receipt(model_path, model_document),
        "study_config_overlay": _document_receipt(config_path, config_document),
        "publication_rankable": False,
        "hardware_rankable": False,
        "selection_eligible": False,
    }


def materialize_repricing_bundle(
    *,
    summary_path: Path,
    base_config_path: Path,
    simulator_root: Path,
    output_dir: Path,
    index_path: Path,
) -> dict[str, Any]:
    """Create or verify one immutable batch-indexed repricing bundle."""

    summary_path = summary_path.resolve()
    base_config_path = base_config_path.resolve()
    simulator_root = simulator_root.resolve()
    output_dir = output_dir.resolve()
    index_path = index_path.resolve()
    if index_path.exists():
        result = verify_repricing_bundle(
            index_path=index_path,
            simulator_root=simulator_root,
        )
        result["reused"] = True
        return result

    summary = load_hashed_json(summary_path)
    validate_summary(summary)
    base_config = _load_object(base_config_path)
    base_model_path = _resolve_model(base_config, simulator_root)
    base_model = _load_object(base_model_path)
    _validate_base(base_config, base_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        _materialize_documents(
            row=row,
            summary=summary,
            summary_path=summary_path,
            base_config=base_config,
            base_config_path=base_config_path,
            base_model=base_model,
            base_model_path=base_model_path,
            output_dir=output_dir,
        )
        for row in _batch_rows(summary)
    ]
    body = {
        "schema": BUNDLE_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": summary["trace_content_hash"],
        "summary": {
            "path": str(summary_path),
            "sha256": file_hash(summary_path),
            "content_hash": summary["content_hash"],
        },
        "base_study_config": {
            "path": str(base_config_path),
            "sha256": file_hash(base_config_path),
        },
        "base_simulator_model": {
            "path": str(base_model_path),
            "sha256": file_hash(base_model_path),
        },
        "materializable_batches": [entry["batch_size"] for entry in entries],
        "unsupported_batches": list(summary["unsupported_batches"]),
        "entries": entries,
        "execution_scope": "legacy_aggregate_analytic_route_sensitivity_only",
        "classification": dict(summary["classification"]),
        "tool_revision": file_hash(Path(__file__).resolve()),
    }
    index = _hashed_body(body)
    _atomic_install(index_path, _encode(index))
    result = dict(index)
    result.update({"index_path": str(index_path), "reused": False})
    return result


def verify_repricing_bundle(
    *,
    index_path: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    """Verify all immutable inputs and regenerate every expected document."""

    index_path = index_path.resolve()
    simulator_root = simulator_root.resolve()
    index = load_hashed_json(index_path)
    if index.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported route-repricing bundle schema")
    if index.get("tool_revision") != file_hash(Path(__file__).resolve()):
        raise ValueError("route-repricing tool revision changed")
    summary_receipt = index.get("summary")
    config_receipt = index.get("base_study_config")
    model_receipt = index.get("base_simulator_model")
    if not all(isinstance(value, Mapping) for value in (
        summary_receipt,
        config_receipt,
        model_receipt,
    )):
        raise ValueError("route-repricing bundle source receipts are malformed")
    summary_path = Path(str(summary_receipt["path"])).resolve()
    base_config_path = Path(str(config_receipt["path"])).resolve()
    base_model_path = Path(str(model_receipt["path"])).resolve()
    for path, receipt, label in (
        (summary_path, summary_receipt, "summary"),
        (base_config_path, config_receipt, "base config"),
        (base_model_path, model_receipt, "base model"),
    ):
        if not path.is_file() or file_hash(path) != receipt.get("sha256"):
            raise ValueError(f"route-repricing {label} is missing or changed")
    summary = load_hashed_json(summary_path)
    validate_summary(summary)
    if (
        summary.get("content_hash") != summary_receipt.get("content_hash")
        or summary.get("trace_content_hash") != index.get("trace_content_hash")
    ):
        raise ValueError("route-repricing summary binding differs")
    base_config = _load_object(base_config_path)
    base_model = _load_object(base_model_path)
    _validate_base(base_config, base_model)
    if _resolve_model(base_config, simulator_root) != base_model_path:
        raise ValueError("route-repricing base model resolution changed")

    expected_rows = _batch_rows(summary)
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_rows):
        raise ValueError("route-repricing bundle entry count differs")
    if index.get("materializable_batches") != [
        int(row["batch_size"]) for row in expected_rows
    ] or index.get("unsupported_batches") != summary.get("unsupported_batches"):
        raise ValueError("route-repricing batch index differs")

    for row, observed in zip(expected_rows, entries):
        if not isinstance(observed, Mapping):
            raise ValueError("route-repricing bundle entry is malformed")
        model_receipt_entry = observed.get("model_overlay")
        config_receipt_entry = observed.get("study_config_overlay")
        if not isinstance(model_receipt_entry, Mapping) or not isinstance(
            config_receipt_entry, Mapping
        ):
            raise ValueError("route-repricing artifact receipt is malformed")
        model_artifact = Path(str(model_receipt_entry.get("path", ""))).resolve()
        config_artifact = Path(str(config_receipt_entry.get("path", ""))).resolve()
        if model_artifact.parent != config_artifact.parent:
            raise ValueError("route-repricing artifacts must share one output directory")
        for role, receipt, artifact in (
            ("model_overlay", model_receipt_entry, model_artifact),
            ("study_config_overlay", config_receipt_entry, config_artifact),
        ):
            if not artifact.is_file() or file_hash(artifact) != receipt.get("sha256"):
                raise ValueError(f"route-repricing {role} is missing or changed")
            if load_hashed_json(artifact)["content_hash"] != receipt.get(
                "content_hash"
            ):
                raise ValueError(f"route-repricing {role} content binding differs")
        expected = _materialize_documents(
            row=row,
            summary=summary,
            summary_path=summary_path,
            base_config=base_config,
            base_config_path=base_config_path,
            base_model=base_model,
            base_model_path=base_model_path,
            output_dir=model_artifact.parent,
        )
        if dict(observed) != expected:
            raise ValueError("route-repricing entry metadata differs")
    classification = index.get("classification")
    if classification != summary.get("classification"):
        raise ValueError("route-repricing classification differs")
    result = dict(index)
    result.update({"index_path": str(index_path), "verified": True})
    return result


__all__ = [
    "BUNDLE_SCHEMA",
    "CONFIG_OVERLAY_SCHEMA",
    "MODEL_OVERLAY_SCHEMA",
    "materialize_repricing_bundle",
    "validate_summary",
    "verify_repricing_bundle",
]
