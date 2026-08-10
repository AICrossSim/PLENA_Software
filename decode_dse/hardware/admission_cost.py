"""Content-addressed numerical evidence for decode-cache admission."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from decode_dse.profiles import DECODE_FORMATS, MX_BLOCK_SIZE

ADMISSION_CORRECTNESS_SCHEMA = (
    "decode-admission-correctness-evidence"
)
ADMISSION_PREPARATION_SCHEMA = "decode-admission-preparation"
ADMISSION_INDEX_SCHEMA = "decode-admission-index"
ADMISSION_NUMERICAL_VALIDATION_SCHEMA = (
    "decode-admission-numerical-validation"
)
ADMISSION_CORRECTNESS_SCOPE = "bf16_to_packedkv_numerical_correctness"
ADMISSION_VALIDATION_BASIS = (
    "independent_source_conversion_exact_persisted_planes"
)
# The recompute policy rebuilds each plane from its source instead of reading
# a persisted one, so it validates against a basis of its own. Each policy
# still requires exactly its own basis string.
RECOMPUTABLE_ADMISSION_VALIDATION_BASIS = (
    "independent_source_recompute_exact_planes"
)
ADMISSION_PERSISTENCE_CONTRACT = (
    "packed_planes_plus_bf16_numerical_view"
)
# Admission may instead recompute each format on demand, persisting nothing.
# The recomputable projection reports a logical total and a runtime peak in
# place of the persisted-plane fields the contract above accounts for.
RECOMPUTABLE_ADMISSION_POLICY = "content_addressed_recompute_per_format"


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: Any) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def _load_immutable_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    body = dict(value)
    expected = body.pop("content_hash", None)
    if expected != _content_hash(body):
        raise ValueError(f"immutable JSON checksum mismatch: {path}")
    return {**body, "content_hash": expected}


@dataclass(frozen=True)
class AdmissionCorrectnessStatus:
    """Fail-closed admission correctness and provenance status."""

    source_path: Path | None
    receipt_sha256: str | None
    evidence_id: str | None
    manifest_hash: str | None
    run_plan_hash: str | None
    prompt_manifest_hash: str | None
    admission_contract_id: str | None
    admission_index_hash: str | None
    numerical_validation_hash: str | None
    admission_code_revision: str | None
    runtime_environment_fingerprint: str | None
    sample_bundle_hash: str | None
    layout_id: str | None
    persistence_contract: str | None
    formats: tuple[str, ...]
    document_count: int
    artifact_count: int
    tensor_count: int
    persisted_bytes: int
    projected_cold_artifact_bytes: int
    projected_numerical_view_bytes: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            not self.failures
            and self.source_path is not None
            and self.receipt_sha256 is not None
            and self.evidence_id is not None
            and self.manifest_hash is not None
            and self.run_plan_hash is not None
            and self.prompt_manifest_hash is not None
            and self.admission_contract_id is not None
            and self.admission_index_hash is not None
            and self.numerical_validation_hash is not None
            and self.admission_code_revision is not None
            and self.runtime_environment_fingerprint is not None
            and self.sample_bundle_hash is not None
            and self.layout_id is not None
            and (
                # The persisted contract keeps its planes on disk; the
                # recompute policy rebuilds them and persists nothing, so
                # each policy pins its own persisted-byte expectation.
                (
                    self.persistence_contract == ADMISSION_PERSISTENCE_CONTRACT
                    and self.persisted_bytes > 0
                )
                or (
                    self.persistence_contract == RECOMPUTABLE_ADMISSION_POLICY
                    and self.persisted_bytes == 0
                )
            )
            and bool(self.formats)
            and self.document_count > 0
            and self.artifact_count > 0
            and self.tensor_count > 0
            and self.projected_cold_artifact_bytes >= self.persisted_bytes
            and self.projected_numerical_view_bytes > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADMISSION_CORRECTNESS_SCHEMA,
            "scope": ADMISSION_CORRECTNESS_SCOPE,
            "passed": self.passed,
            "failures": list(self.failures),
            "receipt_sha256": self.receipt_sha256,
            "evidence_id": self.evidence_id,
            "manifest_hash": self.manifest_hash,
            "run_plan_hash": self.run_plan_hash,
            "prompt_manifest_hash": self.prompt_manifest_hash,
            "admission_contract_id": self.admission_contract_id,
            "admission_index_hash": self.admission_index_hash,
            "numerical_validation_hash": self.numerical_validation_hash,
            "admission_code_revision": self.admission_code_revision,
            "runtime_environment_fingerprint": (
                self.runtime_environment_fingerprint
            ),
            "sample_bundle_hash": self.sample_bundle_hash,
            "layout_id": self.layout_id,
            "persistence_contract": self.persistence_contract,
            "formats": list(self.formats),
            "document_count": self.document_count,
            "artifact_count": self.artifact_count,
            "tensor_count": self.tensor_count,
            "persisted_bytes": self.persisted_bytes,
            "projected_cold_artifact_bytes": (
                self.projected_cold_artifact_bytes
            ),
            "projected_numerical_view_bytes": (
                self.projected_numerical_view_bytes
            ),
            "source_dtype": "BF16",
            "block_size": MX_BLOCK_SIZE,
            "steady_state_tpot_included": False,
            "hardware_latency_calibrated": False,
            "hardware_energy_calibrated": False,
            "ttft_rankable": False,
            "admission_energy_rankable": False,
        }


def missing_admission_correctness_status(
    failure: str = "missing_admission_preparation_receipt",
) -> AdmissionCorrectnessStatus:
    return AdmissionCorrectnessStatus(
        source_path=None,
        receipt_sha256=None,
        evidence_id=None,
        manifest_hash=None,
        run_plan_hash=None,
        prompt_manifest_hash=None,
        admission_contract_id=None,
        admission_index_hash=None,
        numerical_validation_hash=None,
        admission_code_revision=None,
        runtime_environment_fingerprint=None,
        sample_bundle_hash=None,
        layout_id=None,
        persistence_contract=None,
        formats=(),
        document_count=0,
        artifact_count=0,
        tensor_count=0,
        persisted_bytes=0,
        projected_cold_artifact_bytes=0,
        projected_numerical_view_bytes=0,
        failures=(failure,),
    )


def _validate_records(
    records: Sequence[Any],
    *,
    formats: tuple[str, ...],
) -> dict[tuple[str, str], tuple[str, str]]:
    if not records:
        raise ValueError("admission index has no records")
    allowed = set(formats) | {"BF16"}
    observed: dict[tuple[str, str], tuple[str, str]] = {}
    artifact_ids: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("admission index records must be objects")
        format_id = str(raw.get("format_id", ""))
        if format_id not in allowed:
            raise ValueError("admission index contains an unsupported format")
        document_id = str(raw.get("document_id", ""))
        if not document_id:
            raise ValueError("admission record document_id is empty")
        key = (format_id, document_id)
        if key in observed:
            raise ValueError("admission record identity is duplicated")
        artifact_id = _require_sha256(
            "admission artifact_id",
            raw.get("artifact_id"),
        )
        source_artifact_id = _require_sha256(
            "admission source_artifact_id",
            raw.get("source_artifact_id"),
        )
        _require_sha256("admission prompt_hash", raw.get("prompt_hash"))
        relative_path = raw.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("admission record relative_path is empty")
        if artifact_id in artifact_ids:
            raise ValueError("admission artifact identity is duplicated")
        artifact_ids.add(artifact_id)
        observed[key] = (artifact_id, source_artifact_id)
        build_seconds = raw.get("build_seconds")
        if (
            isinstance(build_seconds, bool)
            or not isinstance(build_seconds, (int, float))
            or not math.isfinite(float(build_seconds))
            or float(build_seconds) <= 0
        ):
            raise ValueError("admission record build time is invalid")
        for name in ("payload_bytes", "persisted_bytes"):
            value = raw.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"admission record {name} is invalid")
    documents = {document_id for _, document_id in observed}
    for document_id in documents:
        document_formats = {
            format_id
            for format_id, observed_document in observed
            if observed_document == document_id
        }
        if document_formats != allowed:
            raise ValueError("admission record format coverage is incomplete")
    return observed


def _validate_numerical_records(
    records: Sequence[Any],
    *,
    indexed: Mapping[tuple[str, str], tuple[str, str]],
) -> int:
    if len(records) != len(indexed):
        raise ValueError("admission numerical record count differs")
    observed: set[tuple[str, str]] = set()
    total_tensors = 0
    layer_counts: set[int] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("admission numerical records must be objects")
        format_id = str(raw.get("format_id", ""))
        document_id = str(raw.get("document_id", ""))
        key = (format_id, document_id)
        if key not in indexed or key in observed:
            raise ValueError("admission numerical record identity differs")
        observed.add(key)
        artifact_id, source_artifact_id = indexed[key]
        if (
            raw.get("artifact_id") != artifact_id
            or raw.get("source_artifact_id") != source_artifact_id
        ):
            raise ValueError("admission numerical artifact identity differs")
        layer_count = raw.get("layer_count")
        tensor_count = raw.get("tensor_count")
        checks = raw.get("tensor_checks")
        if (
            isinstance(layer_count, bool)
            or not isinstance(layer_count, int)
            or layer_count <= 0
            or isinstance(tensor_count, bool)
            or not isinstance(tensor_count, int)
            or tensor_count != 2 * layer_count
            or not isinstance(checks, list)
            or len(checks) != tensor_count
        ):
            raise ValueError("admission numerical tensor coverage differs")
        layer_counts.add(layer_count)
        expected_keys = {
            (layer_index, role)
            for layer_index in range(layer_count)
            for role in ("key", "value")
        }
        check_keys: set[tuple[int, str]] = set()
        for check_raw in checks:
            if not isinstance(check_raw, Mapping):
                raise TypeError("admission tensor checks must be objects")
            check = dict(check_raw)
            check_hash = check.pop("check_hash", None)
            if check_hash != _content_hash(check):
                raise ValueError("admission tensor-check hash differs")
            layer_index = check.get("layer_index")
            role = check.get("role")
            if (
                isinstance(layer_index, bool)
                or not isinstance(layer_index, int)
                or not isinstance(role, str)
                or (layer_index, role) not in expected_keys
                or (layer_index, role) in check_keys
            ):
                raise ValueError("admission tensor-check coverage differs")
            check_keys.add((layer_index, role))
            if (
                check.get("format_id") != format_id
                or check.get("exact_match") is not True
            ):
                raise ValueError("admission tensor exact-match proof differs")
            for name in (
                "source_sha256",
                "element_sha256",
                "scale_sha256",
                "numerical_view_sha256",
                "descriptor_sha256",
            ):
                _require_sha256(f"admission tensor {name}", check.get(name))
        if check_keys != expected_keys:
            raise ValueError("admission tensor-check coverage is incomplete")
        total_tensors += tensor_count
    if observed != set(indexed):
        raise ValueError("admission numerical artifact coverage is incomplete")
    if len(layer_counts) != 1:
        raise ValueError("admission numerical layer coverage differs")
    return total_tensors


def _format_bytes(
    records: Sequence[Mapping[str, Any]],
    *,
    formats: Sequence[str],
) -> dict[str, int]:
    """Total rebuild cost per format, including the BF16 reference."""

    totals = {format_id: 0 for format_id in (*formats, "BF16")}
    for record in records:
        format_id = str(record.get("format_id", ""))
        if format_id not in totals:
            raise ValueError("admission record declares an unknown format")
        totals[format_id] += int(record["persisted_bytes"])
    return totals


def _validate_recomputable_resource_projection(
    value: Mapping[str, Any],
    *,
    logical_bytes: int,
    format_bytes: Mapping[str, int],
) -> dict[str, int | float | str]:
    """Validate the projection written by the recompute-per-format policy."""

    integer_fields = (
        "logical_total_bytes",
        "runtime_peak_format_bytes",
        "persisted_after_preparation_bytes",
        "required_cold_capacity_bytes",
        "observed_cold_available_bytes",
        "required_host_bytes",
        "observed_host_available_bytes",
    )
    integers: dict[str, int] = {}
    for name in integer_fields:
        field = value.get(name)
        if isinstance(field, bool) or not isinstance(field, int) or field < 0:
            raise ValueError(f"admission resource {name} is invalid")
        integers[name] = field
    if integers["logical_total_bytes"] != logical_bytes:
        raise ValueError("admission logical total differs from its records")
    if integers["runtime_peak_format_bytes"] != max(format_bytes.values()):
        raise ValueError("admission runtime peak differs from its records")
    if integers["persisted_after_preparation_bytes"] != 0:
        raise ValueError("recomputable admission must persist nothing")
    if integers["required_cold_capacity_bytes"] <= 0:
        raise ValueError("admission required cold capacity is invalid")
    if (
        integers["observed_cold_available_bytes"]
        < integers["required_cold_capacity_bytes"]
    ):
        raise ValueError("admission cold capacity was not available")
    if integers["required_host_bytes"] <= 0:
        raise ValueError("admission required host bytes is invalid")
    if integers["observed_host_available_bytes"] < integers["required_host_bytes"]:
        raise ValueError("admission host capacity was not available")
    resolved: dict[str, int | float | str] = dict(integers)
    # Present the recomputable projection through the same field names the
    # persisted contract reports, so downstream evidence stays one shape:
    # cold capacity is what preparation required, and the numerical-view
    # total is the logical cost of rebuilding every plane.
    resolved["persistence_contract"] = RECOMPUTABLE_ADMISSION_POLICY
    resolved["projected_cold_artifact_bytes"] = integers[
        "required_cold_capacity_bytes"
    ]
    resolved["projected_numerical_view_bytes"] = integers["logical_total_bytes"]
    return resolved


def _validate_resource_projection(
    value: Any,
    *,
    persisted_bytes: int,
) -> dict[str, int | float | str]:
    if not isinstance(value, Mapping):
        raise TypeError("admission resource projection must be an object")
    if value.get("persistence_contract") != ADMISSION_PERSISTENCE_CONTRACT:
        raise ValueError("admission persistence contract differs")
    integer_fields = (
        "artifact_space_reserve_bytes",
        "projected_element_plane_bytes",
        "projected_scale_plane_bytes",
        "projected_numerical_view_bytes",
        "projected_metadata_reserve_bytes",
        "projected_cold_artifact_bytes",
        "required_cold_capacity_bytes",
        "observed_cold_available_bytes",
        "projected_peak_host_bytes",
        "required_host_bytes",
        "observed_host_available_bytes",
    )
    integers: dict[str, int] = {}
    for name in integer_fields:
        field = value.get(name)
        if (
            isinstance(field, bool)
            or not isinstance(field, int)
            or field < 0
        ):
            raise ValueError(f"admission resource {name} is invalid")
        integers[name] = field
    safety_factor = value.get("artifact_space_safety_factor")
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or float(safety_factor) < 1.0
    ):
        raise ValueError("admission resource safety factor is invalid")
    projected = sum(
        integers[name]
        for name in (
            "projected_element_plane_bytes",
            "projected_scale_plane_bytes",
            "projected_numerical_view_bytes",
            "projected_metadata_reserve_bytes",
        )
    )
    minimum_capacity = (
        math.ceil(projected * float(safety_factor))
        + integers["artifact_space_reserve_bytes"]
    )
    if (
        projected <= 0
        or integers["projected_numerical_view_bytes"] <= 0
        or integers["projected_cold_artifact_bytes"] != projected
        or integers["projected_cold_artifact_bytes"] < persisted_bytes
        or integers["required_cold_capacity_bytes"] < minimum_capacity
        or integers["observed_cold_available_bytes"]
        < integers["required_cold_capacity_bytes"]
        or integers["required_host_bytes"]
        < integers["projected_peak_host_bytes"]
        or integers["observed_host_available_bytes"]
        < integers["required_host_bytes"]
    ):
        raise ValueError("admission resource projection is inconsistent")
    return {
        "persistence_contract": str(value["persistence_contract"]),
        "artifact_space_safety_factor": float(safety_factor),
        **integers,
    }


def load_admission_correctness_evidence(
    path: str | Path,
    *,
    manifest_hash: str,
    required_formats: Sequence[str] = DECODE_FORMATS,
) -> AdmissionCorrectnessStatus:
    """Validate one numerical admission receipt and its immutable index."""

    source = Path(path).resolve()
    receipt_sha256: str | None = None
    try:
        _require_sha256("manifest_hash", manifest_hash)
        receipt_sha256 = _file_sha256(source)
        receipt = _load_immutable_json(source)
        if (
            receipt.get("schema_version")
            != ADMISSION_PREPARATION_SCHEMA
        ):
            raise ValueError("unsupported admission preparation schema")
        if receipt.get("manifest_hash") != manifest_hash:
            raise ValueError("admission receipt manifest identity mismatch")
        formats = tuple(str(value) for value in required_formats)
        if formats != tuple(DECODE_FORMATS):
            raise ValueError("required admission formats are invalid")
        run_plan_hash = _require_sha256(
            "admission run_plan_hash",
            receipt.get("run_plan_hash"),
        )
        prompt_manifest_hash = _require_sha256(
            "admission prompt_manifest_hash",
            receipt.get("prompt_manifest_hash"),
        )
        runtime_fingerprint = _require_sha256(
            "admission runtime_environment_fingerprint",
            receipt.get("runtime_environment_fingerprint"),
        )
        contract_id = str(receipt.get("admission_contract_id", ""))
        index_hash = _require_sha256(
            "admission_index_hash",
            receipt.get("admission_index_hash"),
        )
        validation_hash = _require_sha256(
            "numerical_validation_hash",
            receipt.get("numerical_validation_hash"),
        )
        if not contract_id:
            raise ValueError("admission receipt identities are incomplete")
        raw_index_path = Path(
            str(receipt.get("admission_index_path", ""))
        )
        index_path = (
            raw_index_path
            if raw_index_path.is_absolute()
            else source.parent / raw_index_path
        ).resolve()
        index = _load_immutable_json(index_path)
        if index.get("content_hash") != index_hash:
            raise ValueError("admission index hash differs from its receipt")
        if index.get("schema_version") != ADMISSION_INDEX_SCHEMA:
            raise ValueError("unsupported admission index schema")
        if index.get("admission_contract_id") != contract_id:
            raise ValueError("admission contract identities differ")
        admission_code_revision = _require_sha256(
            "admission_code_revision",
            index.get("admission_code_revision"),
        )
        index_runtime_fingerprint = _require_sha256(
            "admission index runtime_environment_fingerprint",
            index.get("runtime_environment_fingerprint"),
        )
        sample_bundle_hash = _require_sha256(
            "admission sample_bundle_hash",
            index.get("sample_bundle_hash"),
        )
        if index_runtime_fingerprint != runtime_fingerprint:
            raise ValueError("admission runtime identities differ")
        if int(index.get("block_size", -1)) != MX_BLOCK_SIZE:
            raise ValueError("admission index uses a non-native block size")
        if tuple(index.get("quantized_formats", ())) != formats:
            raise ValueError("admission format coverage differs")
        if tuple(index.get("reference_formats", ())) != ("BF16",):
            raise ValueError("admission BF16 reference coverage differs")
        layout_id = str(index.get("layout_id", ""))
        if not layout_id:
            raise ValueError("admission layout identity is empty")
        records = index.get("records")
        if not isinstance(records, list):
            raise TypeError("admission index records must be a list")
        indexed = _validate_records(records, formats=formats)
        document_count = len(
            {document_id for _, document_id in indexed}
        )
        persisted_bytes = sum(
            int(record["persisted_bytes"]) for record in records
        )
        payload_bytes = sum(
            int(record["payload_bytes"]) for record in records
        )
        projection = index.get("resource_projection")
        recomputable = (
            isinstance(projection, Mapping)
            and projection.get("policy") == RECOMPUTABLE_ADMISSION_POLICY
        )
        if (
            int(receipt.get("artifact_count", -1)) != len(records)
            or int(index.get("artifact_count", -2)) != len(records)
            or int(receipt.get("document_count", -1)) != document_count
            or int(index.get("document_count", -2)) != document_count
            or int(receipt.get("quantized_format_count", -1))
            != len(formats)
            or receipt.get("resource_projection")
            != index.get("resource_projection")
        ):
            raise ValueError("admission aggregate counts differ")
        if recomputable:
            # Under the content-addressed recompute policy nothing survives
            # preparation: each per-record size is the cost of rebuilding that
            # artifact, their sum is the logical total, and the persisted
            # totals must be exactly zero.
            if (
                int(receipt.get("persisted_bytes", -1)) != 0
                or int(index.get("persisted_bytes", -2)) != 0
                or int(receipt.get("logical_artifact_bytes", -1))
                != persisted_bytes
                or int(index.get("logical_artifact_bytes", -2))
                != persisted_bytes
                or receipt.get("persistence_policy")
                != RECOMPUTABLE_ADMISSION_POLICY
                or index.get("persistence_policy")
                != RECOMPUTABLE_ADMISSION_POLICY
            ):
                raise ValueError("admission aggregate counts differ")
            resources = _validate_recomputable_resource_projection(
                projection,
                logical_bytes=persisted_bytes,
                format_bytes=_format_bytes(records, formats=formats),
            )
            # Nothing survives preparation under this policy; the summed
            # per-record size is the logical rebuild cost, reported through
            # the projection rather than as persisted bytes.
            reported_persisted_bytes = 0
        else:
            if (
                int(receipt.get("persisted_bytes", -1)) != persisted_bytes
                or int(index.get("persisted_bytes", -2)) != persisted_bytes
                or int(index.get("payload_bytes", -1)) != payload_bytes
            ):
                raise ValueError("admission aggregate counts differ")
            resources = _validate_resource_projection(
                projection,
                persisted_bytes=persisted_bytes,
            )
            reported_persisted_bytes = persisted_bytes
        raw_validation_path = Path(
            str(receipt.get("numerical_validation_path", ""))
        )
        validation_path = (
            raw_validation_path
            if raw_validation_path.is_absolute()
            else source.parent / raw_validation_path
        ).resolve()
        validation = _load_immutable_json(validation_path)
        if validation.get("content_hash") != validation_hash:
            raise ValueError(
                "admission numerical validation hash differs from its receipt"
            )
        if (
            validation.get("schema_version")
            != ADMISSION_NUMERICAL_VALIDATION_SCHEMA
            or validation.get("passed") is not True
            or validation.get("basis")
            != (
                RECOMPUTABLE_ADMISSION_VALIDATION_BASIS
                if recomputable
                else ADMISSION_VALIDATION_BASIS
            )
            or validation.get("admission_index_hash") != index_hash
            or validation.get("admission_contract_id") != contract_id
            or validation.get("admission_code_revision")
            != admission_code_revision
            or validation.get("runtime_environment_fingerprint")
            != runtime_fingerprint
            or validation.get("sample_bundle_hash") != sample_bundle_hash
            or validation.get("layout_id") != layout_id
            or validation.get("source_dtype") != "BF16"
            or int(validation.get("block_size", -1)) != MX_BLOCK_SIZE
            or tuple(validation.get("quantized_formats", ())) != formats
            or tuple(validation.get("reference_formats", ())) != ("BF16",)
            or validation.get("steady_state_tpot_included") is not False
        ):
            raise ValueError("admission numerical validation contract differs")
        numerical_records = validation.get("records")
        if not isinstance(numerical_records, list):
            raise TypeError("admission numerical records must be a list")
        tensor_count = _validate_numerical_records(
            numerical_records,
            indexed=indexed,
        )
        if (
            int(validation.get("artifact_count", -1)) != len(indexed)
            or int(validation.get("document_count", -1)) != document_count
            or int(validation.get("tensor_count", -1)) != tensor_count
        ):
            raise ValueError("admission numerical aggregate counts differ")
        evidence_id = "admission-correctness-" + _content_hash(
            {
                "receipt_sha256": receipt_sha256,
                "receipt_content_hash": receipt["content_hash"],
                "admission_index_hash": index_hash,
                "numerical_validation_hash": validation_hash,
                "manifest_hash": manifest_hash,
                "run_plan_hash": run_plan_hash,
                "prompt_manifest_hash": prompt_manifest_hash,
                "admission_contract_id": contract_id,
                "admission_code_revision": admission_code_revision,
                "runtime_environment_fingerprint": runtime_fingerprint,
                "sample_bundle_hash": sample_bundle_hash,
                "layout_id": layout_id,
                "formats": formats,
                "document_count": document_count,
                "artifact_count": len(indexed),
                "tensor_count": tensor_count,
                "persistence_contract": resources[
                    "persistence_contract"
                ],
                "persisted_bytes": reported_persisted_bytes,
                "projected_cold_artifact_bytes": resources[
                    "projected_cold_artifact_bytes"
                ],
                "projected_numerical_view_bytes": resources[
                    "projected_numerical_view_bytes"
                ],
            }
        )
        return AdmissionCorrectnessStatus(
            source_path=source,
            receipt_sha256=receipt_sha256,
            evidence_id=evidence_id,
            manifest_hash=manifest_hash,
            run_plan_hash=run_plan_hash,
            prompt_manifest_hash=prompt_manifest_hash,
            admission_contract_id=contract_id,
            admission_index_hash=index_hash,
            numerical_validation_hash=validation_hash,
            admission_code_revision=admission_code_revision,
            runtime_environment_fingerprint=runtime_fingerprint,
            sample_bundle_hash=sample_bundle_hash,
            layout_id=layout_id,
            persistence_contract=str(
                resources["persistence_contract"]
            ),
            formats=formats,
            document_count=document_count,
            artifact_count=len(indexed),
            tensor_count=tensor_count,
            persisted_bytes=reported_persisted_bytes,
            projected_cold_artifact_bytes=int(
                resources["projected_cold_artifact_bytes"]
            ),
            projected_numerical_view_bytes=int(
                resources["projected_numerical_view_bytes"]
            ),
            failures=(),
        )
    except Exception as exc:
        return AdmissionCorrectnessStatus(
            source_path=source,
            receipt_sha256=receipt_sha256,
            evidence_id=None,
            manifest_hash=None,
            run_plan_hash=None,
            prompt_manifest_hash=None,
            admission_contract_id=None,
            admission_index_hash=None,
            numerical_validation_hash=None,
            admission_code_revision=None,
            runtime_environment_fingerprint=None,
            sample_bundle_hash=None,
            layout_id=None,
            persistence_contract=None,
            formats=(),
            document_count=0,
            artifact_count=0,
            tensor_count=0,
            persisted_bytes=0,
            projected_cold_artifact_bytes=0,
            projected_numerical_view_bytes=0,
            failures=(f"{type(exc).__name__}: {exc}",),
        )


def admission_correctness_status_valid(value: Mapping[str, Any]) -> bool:
    """Return whether a serialized status proves numerical admission only."""

    evidence_id = value.get("evidence_id")
    formats = value.get("formats")
    document_count = value.get("document_count")
    artifact_count = value.get("artifact_count")
    tensor_count = value.get("tensor_count")
    persisted_bytes = value.get("persisted_bytes")
    projected_bytes = value.get("projected_cold_artifact_bytes")
    numerical_view_bytes = value.get("projected_numerical_view_bytes")
    return (
        value.get("schema_version")
        == ADMISSION_CORRECTNESS_SCHEMA
        and value.get("scope") == ADMISSION_CORRECTNESS_SCOPE
        and value.get("passed") is True
        and value.get("failures") == []
        and isinstance(evidence_id, str)
        and evidence_id.startswith("admission-correctness-")
        and _is_sha256(evidence_id.removeprefix("admission-correctness-"))
        and all(
            _is_sha256(value.get(name))
            for name in (
                "receipt_sha256",
                "manifest_hash",
                "run_plan_hash",
                "prompt_manifest_hash",
                "admission_index_hash",
                "numerical_validation_hash",
                "admission_code_revision",
                "runtime_environment_fingerprint",
                "sample_bundle_hash",
            )
        )
        and isinstance(value.get("admission_contract_id"), str)
        and bool(value.get("admission_contract_id"))
        and isinstance(value.get("layout_id"), str)
        and bool(value.get("layout_id"))
        and value.get("persistence_contract")
        == ADMISSION_PERSISTENCE_CONTRACT
        and formats == list(DECODE_FORMATS)
        and isinstance(document_count, int)
        and not isinstance(document_count, bool)
        and document_count > 0
        and isinstance(artifact_count, int)
        and not isinstance(artifact_count, bool)
        and artifact_count == document_count * (len(DECODE_FORMATS) + 1)
        and isinstance(tensor_count, int)
        and not isinstance(tensor_count, bool)
        and tensor_count >= artifact_count * 2
        and tensor_count % (artifact_count * 2) == 0
        and isinstance(persisted_bytes, int)
        and not isinstance(persisted_bytes, bool)
        and persisted_bytes > 0
        and isinstance(projected_bytes, int)
        and not isinstance(projected_bytes, bool)
        and projected_bytes >= persisted_bytes
        and isinstance(numerical_view_bytes, int)
        and not isinstance(numerical_view_bytes, bool)
        and numerical_view_bytes > 0
        and value.get("source_dtype") == "BF16"
        and value.get("block_size") == MX_BLOCK_SIZE
        and value.get("steady_state_tpot_included") is False
        and value.get("hardware_latency_calibrated") is False
        and value.get("hardware_energy_calibrated") is False
        and value.get("ttft_rankable") is False
        and value.get("admission_energy_rankable") is False
    )


__all__ = [
    "ADMISSION_CORRECTNESS_SCHEMA",
    "ADMISSION_CORRECTNESS_SCOPE",
    "ADMISSION_NUMERICAL_VALIDATION_SCHEMA",
    "ADMISSION_PERSISTENCE_CONTRACT",
    "ADMISSION_VALIDATION_BASIS",
    "AdmissionCorrectnessStatus",
    "admission_correctness_status_valid",
    "load_admission_correctness_evidence",
    "missing_admission_correctness_status",
]
