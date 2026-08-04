"""Deterministic manifests and restartable status journals for decode sweeps."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from decode_dse.legality import (
    constrain_stack_validity,
    evaluate_stack_capability,
)
from decode_dse.legality import (
    ProfileLegality,
    StackValidity,
    evaluate_profile_legality,
)
from decode_dse.profiles import (
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    PROFILE_KIND_VECTOR_BF16_CONTROL,
    DecodePrecisionProfile,
    enumerate_decode_profiles,
)

MANIFEST_SCHEMA = "decode-sweep-manifest"
STATUS_SCHEMA = "decode-sweep-status"
EXPECTED_QUANTIZED_PROFILES = 3072
EXPECTED_VECTOR_CONTROLS = 512
EXPECTED_BF16_REFERENCES = 1
EXPECTED_TOTAL_PROFILES = 3585
STATUS_STATES = ("pending", "running", "succeeded", "failed")
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class QuantizerSource:
    """One source file whose exact bytes can affect decode arithmetic."""

    component: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.component or not self.path:
            raise ValueError("quantizer sources require component and path")
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("quantizer source paths must be workspace-relative")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("quantizer source hashes must be lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantizerSource":
        return cls(
            component=str(value["component"]),
            path=str(value["path"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class ResolvedImportOrigin:
    """Resolved source origin for a numerically significant import."""

    module: str
    path: str

    def __post_init__(self) -> None:
        if not self.module or not self.path:
            raise ValueError("resolved imports require module and path")
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("resolved import paths must be workspace-relative")

    def to_dict(self) -> dict[str, str]:
        return {"module": self.module, "path": self.path}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedImportOrigin":
        return cls(module=str(value["module"]), path=str(value["path"]))


@dataclass(frozen=True)
class QuantizerProvenance:
    """Ordered, content-addressed identity of the sweep's arithmetic stack."""

    sources: tuple[QuantizerSource, ...]
    resolved_imports: tuple[ResolvedImportOrigin, ...]
    schema_version: str = "decode-quantizer-provenance"

    def __post_init__(self) -> None:
        if self.schema_version != "decode-quantizer-provenance":
            raise ValueError("unsupported quantizer-provenance schema")
        if not self.sources or not self.resolved_imports:
            raise ValueError("quantizer provenance must identify sources and imports")
        source_keys = tuple((item.component, item.path) for item in self.sources)
        if source_keys != tuple(sorted(source_keys)) or len(source_keys) != len(
            set(source_keys)
        ):
            raise ValueError("quantizer sources must be unique and canonically ordered")
        import_keys = tuple((item.module, item.path) for item in self.resolved_imports)
        if import_keys != tuple(sorted(import_keys)) or len(import_keys) != len(
            set(import_keys)
        ):
            raise ValueError("resolved imports must be unique and canonically ordered")
        origins = {item.module: item.path for item in self.resolved_imports}
        mxint_origin = origins.get("chop.nn.quantizers.mxint.fake", "")
        if not mxint_origin.endswith("/chop/nn/quantizers/mxint/fake.py"):
            raise ValueError(
                "decode MXINT must resolve to chop.nn.quantizers.mxint.fake"
            )
        if "mxint_hardware.py" in mxint_origin:
            raise ValueError(
                "decode MXINT resolved to the incompatible hardware helper"
            )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [source.to_dict() for source in self.sources],
            "resolved_imports": [origin.to_dict() for origin in self.resolved_imports],
        }

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._content_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"quantizer_provenance_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuantizerProvenance":
        provenance = cls(
            schema_version=str(value["schema_version"]),
            sources=tuple(
                QuantizerSource.from_dict(source) for source in value["sources"]
            ),
            resolved_imports=tuple(
                ResolvedImportOrigin.from_dict(origin)
                for origin in value["resolved_imports"]
            ),
        )
        if value.get("quantizer_provenance_hash") != provenance.canonical_hash:
            raise ValueError("quantizer-provenance content hash mismatch")
        return provenance


@dataclass(frozen=True)
class SweepManifestEntry:
    ordinal: int
    profile: DecodePrecisionProfile
    legality: ProfileLegality
    validity: StackValidity = StackValidity()

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("manifest ordinals must be non-negative")
        object.__setattr__(
            self,
            "validity",
            constrain_stack_validity(self.profile, self.validity),
        )

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    def to_dict(self) -> dict[str, Any]:
        capability = evaluate_stack_capability(self.profile)
        return {
            "ordinal": self.ordinal,
            "profile_id": self.profile_id,
            "profile": self.profile.to_dict(),
            "legality": self.legality.to_dict(),
            "capability": capability.to_dict(),
            **self.validity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepManifestEntry":
        profile = DecodePrecisionProfile.from_dict(value["profile"])
        if value["profile_id"] != profile.profile_id:
            raise ValueError(f"profile hash mismatch at ordinal {value['ordinal']}")
        legality = ProfileLegality.from_dict(value["legality"])
        if legality != evaluate_profile_legality(profile):
            raise ValueError(f"legality mismatch at ordinal {value['ordinal']}")
        capability = evaluate_stack_capability(profile)
        stored_capability = value.get("capability")
        if stored_capability is not None and stored_capability != capability.to_dict():
            raise ValueError(f"capability mismatch at ordinal {value['ordinal']}")
        return cls(
            ordinal=int(value["ordinal"]),
            profile=profile,
            legality=legality,
            validity=StackValidity.from_dict(value),
        )


@dataclass(frozen=True)
class SweepManifest:
    model_name: str
    model_revision: str
    model_architecture: Mapping[str, Any]
    entries: tuple[SweepManifestEntry, ...]
    quantizer_provenance: QuantizerProvenance
    tokenizer_revision: str | None = None
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported manifest schema {self.schema_version!r}")
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if not self.model_revision:
            raise ValueError("model_revision must be pinned")
        architecture = dict(self.model_architecture)
        required_architecture = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "tie_word_embeddings",
            "attention_bias",
            "use_qk_norm",
        }
        if set(architecture) != required_architecture:
            raise ValueError("manifest model architecture is incomplete")
        object.__setattr__(
            self,
            "model_architecture",
            {key: architecture[key] for key in sorted(architecture)},
        )
        if self.tokenizer_revision is None:
            object.__setattr__(self, "tokenizer_revision", self.model_revision)
        if not self.tokenizer_revision:
            raise ValueError("tokenizer_revision must be pinned")
        ordinals = tuple(entry.ordinal for entry in self.entries)
        if ordinals != tuple(range(len(self.entries))):
            raise ValueError("manifest ordinals must be contiguous and ordered")
        profile_ids = tuple(entry.profile_id for entry in self.entries)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("manifest contains duplicate profile IDs")

    @property
    def counts(self) -> dict[str, int]:
        counts = {
            PROFILE_KIND_QUANTIZED: 0,
            PROFILE_KIND_VECTOR_BF16_CONTROL: 0,
            PROFILE_KIND_BF16_REFERENCE: 0,
            "total": len(self.entries),
        }
        for entry in self.entries:
            counts[entry.profile.kind] += 1
        return counts

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "model_architecture": dict(self.model_architecture),
            "tokenizer_revision": self.tokenizer_revision,
            "quantizer_provenance": self.quantizer_provenance.to_dict(),
            "counts": self.counts,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def canonical_hash(self) -> str:
        content = _canonical_json(self._content_dict()).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "manifest_hash": self.canonical_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepManifest":
        tokenizer_revision = value.get("tokenizer_revision") or value["model_revision"]
        manifest = cls(
            schema_version=str(value["schema_version"]),
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            model_architecture=dict(value["model_architecture"]),
            tokenizer_revision=str(tokenizer_revision),
            quantizer_provenance=QuantizerProvenance.from_dict(
                value["quantizer_provenance"]
            ),
            entries=tuple(
                SweepManifestEntry.from_dict(entry) for entry in value["entries"]
            ),
        )
        expected_counts = value.get("counts")
        if expected_counts is not None and dict(expected_counts) != manifest.counts:
            raise ValueError("manifest count metadata does not match its entries")
        expected_hash = value.get("manifest_hash")
        if expected_hash is not None and expected_hash != manifest.canonical_hash:
            raise ValueError("manifest content hash mismatch")
        return manifest


def build_exhaustive_manifest(
    model_name: str,
    model_revision: str,
    model_architecture: Mapping[str, Any],
    quantizer_provenance: QuantizerProvenance,
    tokenizer_revision: str | None = None,
) -> SweepManifest:
    """Build the exact quantized, vector-control, and BF16 reference schedule."""

    profiles = enumerate_decode_profiles()
    entries = tuple(
        SweepManifestEntry(
            ordinal=ordinal,
            profile=profile,
            legality=evaluate_profile_legality(profile),
        )
        for ordinal, profile in enumerate(profiles)
    )
    manifest = SweepManifest(
        model_name=model_name,
        model_revision=model_revision,
        model_architecture=model_architecture,
        tokenizer_revision=tokenizer_revision,
        quantizer_provenance=quantizer_provenance,
        entries=entries,
    )
    expected = {
        "quantized": EXPECTED_QUANTIZED_PROFILES,
        "vector_bf16_control": EXPECTED_VECTOR_CONTROLS,
        "bf16_reference": EXPECTED_BF16_REFERENCES,
        "total": EXPECTED_TOTAL_PROFILES,
    }
    if manifest.counts != expected:
        raise AssertionError(
            f"unexpected exhaustive manifest counts: {manifest.counts}"
        )
    return manifest


def write_manifest(path: str | os.PathLike[str], manifest: SweepManifest) -> Path:
    """Atomically create an immutable manifest or verify an identical one."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_manifest(destination)
        if existing.canonical_hash != manifest.canonical_hash:
            raise FileExistsError(
                f"refusing to replace a different manifest: {destination}"
            )
        return destination

    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_manifest(path: str | os.PathLike[str]) -> SweepManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return SweepManifest.from_dict(value)


@dataclass(frozen=True)
class StatusRecord:
    profile_id: str
    state: str
    attempt: int
    updated_at: str
    validity: StackValidity = StackValidity()
    error_class: str | None = None
    error_message: str | None = None
    result_path: str | None = None
    schema_version: str = STATUS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATUS_SCHEMA:
            raise ValueError(f"unsupported status schema {self.schema_version!r}")
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if self.state not in STATUS_STATES:
            raise ValueError(f"unsupported status state {self.state!r}")
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")
        if self.state in {"running", "succeeded", "failed"} and self.attempt < 1:
            raise ValueError(f"{self.state} records require a positive attempt")
        if self.state == "failed" and not self.error_message:
            raise ValueError("failed records require an error message")
        if self.state != "failed" and (self.error_class or self.error_message):
            raise ValueError("only failed records may carry error details")

    @classmethod
    def create(
        cls,
        profile_id: str,
        state: str,
        attempt: int,
        *,
        validity: StackValidity | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
        result_path: str | None = None,
    ) -> "StatusRecord":
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return cls(
            profile_id=profile_id,
            state=state,
            attempt=attempt,
            updated_at=timestamp,
            validity=validity or StackValidity(),
            error_class=error_class,
            error_message=error_message,
            result_path=result_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "state": self.state,
            "attempt": self.attempt,
            "updated_at": self.updated_at,
            **self.validity.to_dict(),
            "error_class": self.error_class,
            "error_message": self.error_message,
            "result_path": self.result_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StatusRecord":
        return cls(
            schema_version=str(value["schema_version"]),
            profile_id=str(value["profile_id"]),
            state=str(value["state"]),
            attempt=int(value["attempt"]),
            updated_at=str(value["updated_at"]),
            validity=StackValidity.from_dict(value),
            error_class=value.get("error_class"),
            error_message=value.get("error_message"),
            result_path=value.get("result_path"),
        )


class StatusJournal:
    """Append-only status storage with deterministic resume selection."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        manifest: SweepManifest,
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._profile_ids = frozenset(entry.profile_id for entry in manifest.entries)

    def append(self, record: StatusRecord) -> None:
        if record.profile_id not in self._profile_ids:
            raise ValueError(
                f"status references an unknown profile: {record.profile_id}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (_canonical_json(record.to_dict()) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_RDWR,
            0o644,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            end = os.lseek(descriptor, 0, os.SEEK_END)
            if end and os.pread(descriptor, 1, end - 1) != b"\n":
                cursor = end
                last_newline = -1
                while cursor and last_newline < 0:
                    start = max(0, cursor - 65536)
                    chunk = os.pread(descriptor, cursor - start, start)
                    offset = chunk.rfind(b"\n")
                    if offset >= 0:
                        last_newline = start + offset
                    cursor = start
                os.ftruncate(
                    descriptor,
                    0 if last_newline < 0 else last_newline + 1,
                )
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("incomplete status journal append")
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def records(self) -> tuple[StatusRecord, ...]:
        if not self.path.exists():
            return ()
        payload = self.path.read_bytes()
        lines = payload.splitlines(keepends=True)
        records: list[StatusRecord] = []
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                if index == len(lines) - 1:
                    break
                raise ValueError(f"incomplete journal record at line {index + 1}")
            value = json.loads(line)
            record = StatusRecord.from_dict(value)
            if record.profile_id not in self._profile_ids:
                raise ValueError(
                    f"journal references an unknown profile at line {index + 1}"
                )
            records.append(record)
        return tuple(records)

    def latest(self) -> dict[str, StatusRecord]:
        latest: dict[str, StatusRecord] = {}
        for record in self.records():
            previous = latest.get(record.profile_id)
            if previous is not None and record.attempt < previous.attempt:
                raise ValueError(
                    f"attempt counter moved backward for {record.profile_id}"
                )
            latest[record.profile_id] = record
        return latest

    def next_entries(
        self,
        *,
        max_attempts: int = 3,
        limit: int | None = None,
    ) -> tuple[SweepManifestEntry, ...]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return ()
        latest = self.latest()
        pending: list[SweepManifestEntry] = []
        for entry in self.manifest.entries:
            record = latest.get(entry.profile_id)
            if record is not None and (
                record.state == "succeeded" or record.attempt >= max_attempts
            ):
                continue
            pending.append(entry)
            if limit is not None and len(pending) >= limit:
                break
        return tuple(pending)

    def begin(self, profile_id: str) -> StatusRecord:
        previous = self.latest().get(profile_id)
        if previous is not None and previous.state == "succeeded":
            raise ValueError(f"profile already succeeded: {profile_id}")
        attempt = 1 if previous is None else previous.attempt + 1
        record = StatusRecord.create(profile_id, "running", attempt)
        self.append(record)
        return record

    def complete(
        self,
        profile_id: str,
        *,
        validity: StackValidity,
        result_path: str,
    ) -> StatusRecord:
        previous = self.latest().get(profile_id)
        if previous is None or previous.state != "running":
            raise ValueError(f"profile is not running: {profile_id}")
        record = StatusRecord.create(
            profile_id,
            "succeeded",
            previous.attempt,
            validity=validity,
            result_path=result_path,
        )
        self.append(record)
        return record

    def fail(
        self,
        profile_id: str,
        *,
        error: BaseException | str,
        validity: StackValidity | None = None,
    ) -> StatusRecord:
        previous = self.latest().get(profile_id)
        if previous is None or previous.state != "running":
            raise ValueError(f"profile is not running: {profile_id}")
        if isinstance(error, BaseException):
            error_class = type(error).__name__
            error_message = str(error)
        else:
            error_class = "Error"
            error_message = str(error)
        record = StatusRecord.create(
            profile_id,
            "failed",
            previous.attempt,
            validity=validity,
            error_class=error_class,
            error_message=error_message or error_class,
        )
        self.append(record)
        return record


def _load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_sweep_config(config: Mapping[str, Any]) -> None:
    """Reject configuration drift from the canonical exhaustive schedule."""

    from decode_dse.profiles import DECODE_FORMATS, VECTOR_FP_FORMATS

    search = config.get("search", {})
    expected: tuple[tuple[str, Any, Any], ...] = (
        ("search.weight_w", tuple(search.get("weight_w", ())), DECODE_FORMATS),
        ("search.act_w", tuple(search.get("act_w", ())), DECODE_FORMATS),
        ("search.kv", tuple(search.get("kv", ())), DECODE_FORMATS),
        ("search.vector_fp", tuple(search.get("vector_fp", ())), VECTOR_FP_FORMATS),
        ("search.block", tuple(search.get("block", ())), (8,)),
        ("search.mixed_weight", search.get("mixed_weight"), False),
        (
            "search.include_vector_bf16_controls",
            search.get("include_vector_bf16_controls"),
            True,
        ),
        ("search.include_bf16_reference", search.get("include_bf16_reference"), True),
        (
            "search.expected_quantized_profiles",
            search.get("expected_quantized_profiles"),
            EXPECTED_QUANTIZED_PROFILES,
        ),
        (
            "search.expected_vector_bf16_controls",
            search.get("expected_vector_bf16_controls"),
            EXPECTED_VECTOR_CONTROLS,
        ),
        (
            "search.expected_total_profiles",
            search.get("expected_total_profiles"),
            EXPECTED_TOTAL_PROFILES,
        ),
        ("software_search", config.get("software_search"), "deterministic_exhaustive"),
        ("search_budget", config.get("search_budget"), EXPECTED_TOTAL_PROFILES),
        ("sampler", config.get("sampler"), "deterministic_grid"),
        ("use_rotation", config.get("use_rotation"), False),
    )
    mismatches = [
        f"{name}: expected {required!r}, got {actual!r}"
        for name, actual, required in expected
        if actual != required
    ]
    fp_pairs = tuple(tuple(value) for value in search.get("front_fp_setting", ()))
    required_pairs = tuple(
        (
            int(token.removeprefix("FP_E").split("M")[0]),
            int(token.split("M")[1]),
        )
        for token in VECTOR_FP_FORMATS
    )
    if fp_pairs != required_pairs:
        mismatches.append(
            f"search.front_fp_setting: expected {required_pairs!r}, got {fp_pairs!r}"
        )
    required_gpu_baseline = {
        "attention_implementation": "sdpa",
        "warmup_steps": 16,
        "measured_steps": 128,
        "repetitions": 3,
        "batch_sizes": [1, 2, 4, 8],
        "precision": "BF16",
        "q_len": 1,
        "first_gpu_only": True,
        "energy_meter_priority": [
            "nvml_total_energy_counter",
            "nvml_power_trace_trapezoidal",
        ],
        "power_trace_sample_interval_ms": 10,
    }
    gpu_baseline = config.get("gpu_baseline")
    if not isinstance(gpu_baseline, Mapping):
        mismatches.append("gpu_baseline must be an explicit object")
    else:
        actual_fields = set(gpu_baseline)
        required_fields = set(required_gpu_baseline)
        if actual_fields != required_fields:
            mismatches.append(
                "gpu_baseline fields: expected "
                f"{sorted(required_fields)!r}, got {sorted(actual_fields)!r}"
            )
        for name, required in required_gpu_baseline.items():
            actual = gpu_baseline.get(name)
            if type(actual) is not type(required) or actual != required:
                mismatches.append(
                    f"gpu_baseline.{name}: expected {required!r}, got {actual!r}"
                )
    revision_fields = (
        ("model_revision", config.get("model_revision")),
        ("tokenizer_revision", config.get("tokenizer_revision")),
        (
            "evaluation_data.dataset_revision",
            (
                config.get("evaluation_data", {}).get("dataset_revision")
                if isinstance(config.get("evaluation_data"), Mapping)
                else None
            ),
        ),
    )
    for name, revision in revision_fields:
        if not isinstance(revision, str) or not _IMMUTABLE_REVISION_RE.fullmatch(
            revision
        ):
            mismatches.append(
                f"{name} must be an immutable 40-64 digit hexadecimal revision"
            )
    if mismatches:
        raise ValueError(
            "invalid exhaustive sweep configuration:\n" + "\n".join(mismatches)
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    config = _load_config(args.config)
    validate_sweep_config(config)
    from decode_dse.software.sweep_plan import build_quantizer_provenance

    repository = Path(__file__).resolve().parents[1]
    manifest = build_exhaustive_manifest(
        str(config["model_name"]),
        str(config["model_revision"]),
        dict(config["model_architecture"]),
        build_quantizer_provenance(repository, config),
        str(config["tokenizer_revision"]),
    )
    output = write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "path": str(output),
                "manifest_hash": manifest.canonical_hash,
                "counts": manifest.counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_BF16_REFERENCES",
    "EXPECTED_QUANTIZED_PROFILES",
    "EXPECTED_TOTAL_PROFILES",
    "EXPECTED_VECTOR_CONTROLS",
    "MANIFEST_SCHEMA",
    "QuantizerProvenance",
    "QuantizerSource",
    "ResolvedImportOrigin",
    "STATUS_SCHEMA",
    "StatusJournal",
    "StatusRecord",
    "SweepManifest",
    "SweepManifestEntry",
    "build_exhaustive_manifest",
    "load_manifest",
    "validate_sweep_config",
    "write_manifest",
]
