"""Execute publication benchmarks with fail-closed paired gates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import random
import re
import time
import traceback
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from decode_dse.hardware.design_space import (
    evaluate_publication_admission,
)
from decode_dse.hardware.statistics import percentile
from decode_dse.legality import ADMISSION_BASIS, StackValidity
from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.refinement_schedule import DecodeRefinementProfile
from decode_dse.software.sweep_plan import load_immutable_json, write_immutable_json


PUBLICATION_CONTRACT_SCHEMA = "decode-publication-contract"
PUBLICATION_CONFIGURATION_MANIFEST_SCHEMA = "decode-publication-configurations"
PUBLICATION_BENCHMARK_MANIFEST_SCHEMA = "decode-publication-benchmark-manifest"
PUBLICATION_CHAT_TEMPLATE_SCHEMA = "decode-chat-template"
PUBLICATION_RESULT_SCHEMA = "decode-publication-result"
PUBLICATION_REPORT_SCHEMA = "decode-publication-report"
PUBLICATION_ROLES = ("bf16", "uniform_i8", "uniform_i4", "pareto")
PUBLICATION_BENCHMARKS = ("wikitext2", "ifeval", "gsm8k", "ruler")
RULER_LENGTHS = (4096, 8192, 16384, 32768)
STANDARD_WIKITEXT2_METRIC_ID = (
    "standard_cached_teacher_forced_wikitext2_nll/v1"
)
POST_HANDOFF_METRIC_ID = "post_handoff_greedy_conditioned_nll/v1"
TASK_METRIC_ID = "cached_greedy_official_task_score/v1"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _artifact(path_value: str) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"publication artifact is missing or empty: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@dataclass(frozen=True)
class PublicationConfiguration:
    """One precision evaluated once, independently of hardware geometry."""

    role: str
    profile: DecodePrecisionProfile | DecodeRefinementProfile
    validity: StackValidity

    def __post_init__(self) -> None:
        if self.role not in PUBLICATION_ROLES:
            raise ValueError(f"unsupported publication role {self.role!r}")
        if self.role == "bf16":
            if (
                not isinstance(self.profile, DecodePrecisionProfile)
                or self.profile.kind != "bf16_reference"
            ):
                raise ValueError("the BF16 role requires the split BF16 reference")
        else:
            expected = {
                "uniform_i8": "MXINT8",
                "uniform_i4": "MXINT4",
            }.get(self.role)
            source_profile = (
                self.profile.source_profile
                if isinstance(self.profile, DecodeRefinementProfile)
                else self.profile
            )
            if expected is not None and (
                source_profile.weight_format,
                source_profile.activation_format,
                source_profile.key_format,
                source_profile.value_format,
            ) != (expected, expected, expected, expected):
                raise ValueError(
                    f"{self.role} does not descend from its uniform source"
                )
            # Every declared timing tier requires at least compiler and
            # emulator validity; the tier-specific requirement is enforced by
            # build_publication_configuration_manifest, which knows the tier.
            if any(
                value is not True
                for value in (
                    self.validity.compiler_valid,
                    self.validity.emulator_valid,
                )
            ):
                raise ValueError(
                    f"{self.role} requires measured compiler and emulator validity"
                )

    @property
    def configuration_id(self) -> str:
        body = {
            "role": self.role,
            "profile_id": self.profile.profile_id,
            "validity": self.validity.to_dict(),
        }
        return f"pub-{_content_hash(body)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "configuration_id": self.configuration_id,
            "profile_type": (
                "refinement"
                if isinstance(self.profile, DecodeRefinementProfile)
                else "base"
            ),
            "profile": self.profile.to_dict(),
            "profile_id": self.profile.profile_id,
            "validity": self.validity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationConfiguration":
        profile_type = str(value.get("profile_type", "base"))
        if profile_type == "base":
            profile = DecodePrecisionProfile.from_dict(value["profile"])
        elif profile_type == "refinement":
            profile = DecodeRefinementProfile.from_dict(value["profile"])
        else:
            raise ValueError("publication profile type is unsupported")
        item = cls(
            role=str(value["role"]),
            profile=profile,
            validity=StackValidity.from_dict(value["validity"]),
        )
        if value.get("profile_id") != item.profile.profile_id:
            raise ValueError("publication profile identity mismatch")
        if value.get("configuration_id") != item.configuration_id:
            raise ValueError("publication configuration identity mismatch")
        return item


@dataclass(frozen=True)
class PublicationHardwareAlternative:
    """One exact repriced geometry joined only after accuracy evaluation."""

    configuration_id: str
    profile_id: str
    source_profile_id: str
    candidate_id: str
    record_hash: str
    hardware_artifact_sha256: str
    tpot_ms: float
    energy_per_token_j: float
    energy_tier: str
    #: Whether this exact geometry was additionally compiled and emulated, on
    #: top of being priced by the validated pricing model that admitted it.
    #: Disclosure only: it is implied by ``record_hash`` and so is deliberately
    #: outside ``alternative_id``, which already binds the row.
    individually_validated: bool | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("configuration_id", self.configuration_id),
            ("profile_id", self.profile_id),
            ("source_profile_id", self.source_profile_id),
            ("candidate_id", self.candidate_id),
        ):
            if not value:
                raise ValueError(f"publication hardware {label} is required")
        _sha256(self.record_hash, "record_hash")
        _sha256(self.hardware_artifact_sha256, "hardware_artifact_sha256")
        for label, value in (
            ("tpot_ms", self.tpot_ms),
            ("energy_per_token_j", self.energy_per_token_j),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"publication hardware {label} must be positive")
        if self.energy_tier not in {"analytic_anchored", "dc_calibrated"}:
            raise ValueError("publication hardware energy tier is unsupported")

    @property
    def alternative_id(self) -> str:
        return "pubhw-" + _content_hash(
            {
                "configuration_id": self.configuration_id,
                "profile_id": self.profile_id,
                "source_profile_id": self.source_profile_id,
                "candidate_id": self.candidate_id,
                "record_hash": self.record_hash,
                "hardware_artifact_sha256": self.hardware_artifact_sha256,
                "tpot_ms": self.tpot_ms,
                "energy_per_token_j": self.energy_per_token_j,
                "energy_tier": self.energy_tier,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alternative_id": self.alternative_id,
            "configuration_id": self.configuration_id,
            "profile_id": self.profile_id,
            "source_profile_id": self.source_profile_id,
            "candidate_id": self.candidate_id,
            "record_hash": self.record_hash,
            "hardware_artifact_sha256": self.hardware_artifact_sha256,
            "tpot_ms": self.tpot_ms,
            "energy_per_token_j": self.energy_per_token_j,
            "energy_tier": self.energy_tier,
            "admission_basis": ADMISSION_BASIS,
            "individually_validated": self.individually_validated,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "PublicationHardwareAlternative":
        coverage = value.get("individually_validated")
        item = cls(
            configuration_id=str(value["configuration_id"]),
            profile_id=str(value["profile_id"]),
            source_profile_id=str(value["source_profile_id"]),
            candidate_id=str(value["candidate_id"]),
            record_hash=str(value["record_hash"]),
            hardware_artifact_sha256=str(value["hardware_artifact_sha256"]),
            tpot_ms=float(value["tpot_ms"]),
            energy_per_token_j=float(value["energy_per_token_j"]),
            energy_tier=str(value["energy_tier"]),
            individually_validated=(
                coverage if isinstance(coverage, bool) else None
            ),
        )
        if value.get("alternative_id") != item.alternative_id:
            raise ValueError("publication hardware alternative identity mismatch")
        return item


@dataclass(frozen=True)
class PublicationBenchmark:
    """An immutable full-suite item manifest."""

    name: str
    dataset_name: str
    dataset_config: str
    dataset_revision: str
    split: str
    item_ids: tuple[str, ...]
    source_item_count: int
    item_order_sha256: str
    item_content_sha256: str
    dataset_tree_sha256: str
    enumeration_receipt_sha256: str
    context_length: int | None = None
    full_evaluation: bool = True

    def __post_init__(self) -> None:
        if self.name not in PUBLICATION_BENCHMARKS:
            raise ValueError(f"unsupported benchmark {self.name!r}")
        if (
            not self.dataset_name
            or not self.dataset_config
            or not self.dataset_revision
            or not self.split
        ):
            raise ValueError("publication datasets must be pinned")
        if not _IMMUTABLE_REVISION.fullmatch(self.dataset_revision):
            raise ValueError("publication dataset revision must be an immutable hash")
        if not self.full_evaluation:
            raise ValueError("publication evaluation does not permit benchmark subsets")
        ids = tuple(map(str, self.item_ids))
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("benchmark item IDs must be non-empty and unique")
        if self.source_item_count != len(ids):
            raise ValueError("benchmark IDs do not cover the sealed source count")
        _sha256(self.item_order_sha256, "item_order_sha256")
        _sha256(self.item_content_sha256, "item_content_sha256")
        _sha256(self.dataset_tree_sha256, "dataset_tree_sha256")
        _sha256(
            self.enumeration_receipt_sha256,
            "enumeration_receipt_sha256",
        )
        expected_order = _content_hash({"item_ids": list(ids)})
        if self.item_order_sha256 != expected_order:
            raise ValueError("benchmark item order differs from its receipt")
        if self.name == "ruler":
            if self.context_length not in RULER_LENGTHS:
                raise ValueError("RULER context length is outside the publication evaluation contract")
        elif self.context_length is not None:
            raise ValueError("only RULER carries a context length")
        object.__setattr__(self, "item_ids", ids)
        receipt_body = {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "context_length": self.context_length,
            "source_item_count": self.source_item_count,
            "item_order_sha256": self.item_order_sha256,
            "item_content_sha256": self.item_content_sha256,
            "dataset_tree_sha256": self.dataset_tree_sha256,
        }
        if self.enumeration_receipt_sha256 != _content_hash(receipt_body):
            raise ValueError("benchmark enumeration receipt identity mismatch")

    @property
    def benchmark_id(self) -> str:
        return f"bench-{_content_hash(self._content_dict())}"

    def _content_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "item_ids": list(self.item_ids),
            "source_item_count": self.source_item_count,
            "item_order_sha256": self.item_order_sha256,
            "item_content_sha256": self.item_content_sha256,
            "dataset_tree_sha256": self.dataset_tree_sha256,
            "enumeration_receipt_sha256": self.enumeration_receipt_sha256,
            "context_length": self.context_length,
            "full_evaluation": self.full_evaluation,
        }

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"benchmark_id": self.benchmark_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationBenchmark":
        benchmark = cls(
            name=str(value["name"]),
            dataset_name=str(value["dataset_name"]),
            dataset_config=str(value["dataset_config"]),
            dataset_revision=str(value["dataset_revision"]),
            split=str(value["split"]),
            item_ids=tuple(value["item_ids"]),
            source_item_count=int(value["source_item_count"]),
            item_order_sha256=str(value["item_order_sha256"]),
            item_content_sha256=str(value["item_content_sha256"]),
            dataset_tree_sha256=str(value["dataset_tree_sha256"]),
            enumeration_receipt_sha256=str(
                value["enumeration_receipt_sha256"]
            ),
            context_length=(
                None
                if value.get("context_length") is None
                else int(value["context_length"])
            ),
            full_evaluation=_strict_bool(
                value["full_evaluation"], "full_evaluation"
            ),
        )
        if value.get("benchmark_id") != benchmark.benchmark_id:
            raise ValueError("publication benchmark identity mismatch")
        return benchmark


@dataclass(frozen=True)
class LocalDatasetSnapshot:
    """Rows and local cache files returned by the offline dataset loader."""

    records: tuple[Mapping[str, Any], ...]
    source_files: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("publication dataset snapshot is empty")
        if not self.source_files:
            raise ValueError("publication dataset snapshot has no local source files")


@contextmanager
def _huggingface_offline_environment():
    """Prevent dataset preparation from consulting a remote Hub endpoint."""

    names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_local_huggingface_snapshot(
    specification: Mapping[str, Any],
) -> LocalDatasetSnapshot:
    """Read a prepared Arrow snapshot without invoking a Hub-aware loader."""

    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as error:
        raise RuntimeError(
            "the pinned datasets package is required to enumerate publication inputs"
        ) from error

    cache_root = Path(str(specification["cache_dir"])).resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"publication dataset cache does not exist: {cache_root}"
        )
    dataset_name = str(specification["dataset_name"])
    dataset_config = str(specification["dataset_config"])
    dataset_revision = str(specification["dataset_revision"])
    split = str(specification["split"])
    snapshot_roots = []
    for candidate in cache_root.glob(f"*/*/*/{dataset_revision}"):
        info_path = candidate / "dataset_info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"publication dataset metadata is unreadable: {info_path}"
            ) from error
        checksums = info.get("download_checksums")
        splits = info.get("splits")
        source_prefix = f"hf://datasets/{dataset_name}@{dataset_revision}/"
        if (
            info.get("config_name") != dataset_config
            or not isinstance(checksums, Mapping)
            or not any(str(value).startswith(source_prefix) for value in checksums)
            or not isinstance(splits, Mapping)
            or split not in splits
        ):
            continue
        snapshot_roots.append((candidate.resolve(), info, info_path.resolve()))
    if len(snapshot_roots) != 1:
        raise FileNotFoundError(
            "expected exactly one prepared local dataset snapshot for "
            f"{dataset_name}/{dataset_config}@{dataset_revision}:{split}; "
            f"found {len(snapshot_roots)}"
        )
    snapshot_root, info, info_path = snapshot_roots[0]
    dataset_file_prefix = f"{info.get('dataset_name')}-{split}"
    arrow_files = tuple(
        sorted(
            path.resolve()
            for path in snapshot_root.glob(f"{dataset_file_prefix}*.arrow")
            if path.name == f"{dataset_file_prefix}.arrow"
            or path.name.startswith(f"{dataset_file_prefix}-")
        )
    )
    if not arrow_files:
        raise FileNotFoundError(
            f"prepared publication split has no Arrow files: {snapshot_root}"
        )
    with _huggingface_offline_environment():
        shards = tuple(Dataset.from_file(str(path)) for path in arrow_files)
        dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    split_info = info["splits"][split]
    expected_rows = (
        split_info.get("num_examples")
        if isinstance(split_info, Mapping)
        else None
    )
    if expected_rows != len(dataset):
        raise ValueError("prepared publication split differs from dataset metadata")
    return LocalDatasetSnapshot(
        records=tuple(dict(record) for record in dataset),
        source_files=(info_path, *arrow_files),
    )


def _canonical_dataset_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("publication dataset contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_dataset_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_dataset_value(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return _canonical_dataset_value(scalar())
    raise TypeError(
        f"publication dataset contains unsupported value {type(value).__name__}"
    )


def _publication_dataset_specification(
    name: str,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"publication dataset {name} must be a mapping")
    required = {
        "dataset_name",
        "dataset_config",
        "dataset_revision",
        "split",
        "cache_dir",
        "content_columns",
        "id_column",
        "source_item_count",
    }
    optional = {"context_length"}
    if set(value) - required - optional or not required.issubset(value):
        raise ValueError(
            f"publication dataset {name} declaration has missing or unknown fields"
        )
    specification = dict(value)
    for field in ("dataset_name", "dataset_config", "split", "cache_dir"):
        if not isinstance(specification[field], str) or not specification[field]:
            raise ValueError(f"publication dataset {name}.{field} is required")
    revision = str(specification["dataset_revision"])
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError(
            f"publication dataset {name}.dataset_revision must be an immutable hash"
        )
    columns = specification["content_columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError(
            f"publication dataset {name}.content_columns must be unique strings"
        )
    id_column = specification["id_column"]
    if id_column is not None and (
        not isinstance(id_column, str) or not id_column
    ):
        raise ValueError(
            f"publication dataset {name}.id_column must be null or a string"
        )
    count = specification["source_item_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            f"publication dataset {name}.source_item_count must be positive"
        )
    context_length = specification.get("context_length")
    if name == "ruler":
        if context_length not in RULER_LENGTHS:
            raise ValueError("RULER dataset context length is not canonical")
    elif context_length is not None:
        raise ValueError("only RULER dataset declarations carry context_length")
    return specification


def _publication_dataset_tree_identity(
    *,
    cache_root: Path,
    source_files: Sequence[Path],
) -> str:
    identities = []
    for path in sorted(Path(value).resolve() for value in source_files):
        if path != cache_root and cache_root not in path.parents:
            raise ValueError("publication dataset source escapes its cache root")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"publication dataset source is missing or empty: {path}")
        identities.append(
            {
                "path": path.relative_to(cache_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not identities or len({item["path"] for item in identities}) != len(
        identities
    ):
        raise ValueError("publication dataset source-file identity is incomplete")
    return _content_hash({"files": identities})


def _build_publication_benchmark(
    name: str,
    specification: Mapping[str, Any],
    *,
    load_snapshot: Callable[[Mapping[str, Any]], LocalDatasetSnapshot],
) -> PublicationBenchmark:
    snapshot = load_snapshot(specification)
    if len(snapshot.records) != int(specification["source_item_count"]):
        raise ValueError(
            f"publication dataset {name} contains {len(snapshot.records)} rows; "
            f"expected {specification['source_item_count']}"
        )
    columns = tuple(str(value) for value in specification["content_columns"])
    id_column = specification["id_column"]
    item_ids = []
    canonical_records = []
    for ordinal, record in enumerate(snapshot.records):
        if not isinstance(record, Mapping):
            raise TypeError(f"publication dataset {name} row is not a mapping")
        required_columns = set(columns)
        if id_column is not None:
            required_columns.add(str(id_column))
        missing = required_columns - set(record)
        if missing:
            raise ValueError(
                f"publication dataset {name} row is missing {sorted(missing)}"
            )
        content = {
            column: _canonical_dataset_value(record[column])
            for column in columns
        }
        row_hash = _content_hash(content)
        if id_column is None:
            item_id = f"{ordinal:08d}-{row_hash[:16]}"
        else:
            item_id = str(_canonical_dataset_value(record[str(id_column)]))
            if not item_id:
                raise ValueError(f"publication dataset {name} has an empty item ID")
        item_ids.append(item_id)
        canonical_records.append({"item_id": item_id, "content": content})
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"publication dataset {name} item IDs are not unique")
    item_order_sha256 = _content_hash({"item_ids": item_ids})
    item_content_sha256 = _content_hash({"records": canonical_records})
    cache_root = Path(str(specification["cache_dir"])).resolve()
    dataset_tree_sha256 = _publication_dataset_tree_identity(
        cache_root=cache_root,
        source_files=snapshot.source_files,
    )
    receipt_body = {
        "name": name,
        "dataset_name": str(specification["dataset_name"]),
        "dataset_config": str(specification["dataset_config"]),
        "dataset_revision": str(specification["dataset_revision"]),
        "split": str(specification["split"]),
        "context_length": specification.get("context_length"),
        "source_item_count": len(item_ids),
        "item_order_sha256": item_order_sha256,
        "item_content_sha256": item_content_sha256,
        "dataset_tree_sha256": dataset_tree_sha256,
    }
    return PublicationBenchmark(
        name=name,
        dataset_name=str(specification["dataset_name"]),
        dataset_config=str(specification["dataset_config"]),
        dataset_revision=str(specification["dataset_revision"]),
        split=str(specification["split"]),
        item_ids=tuple(item_ids),
        source_item_count=len(item_ids),
        item_order_sha256=item_order_sha256,
        item_content_sha256=item_content_sha256,
        dataset_tree_sha256=dataset_tree_sha256,
        enumeration_receipt_sha256=_content_hash(receipt_body),
        context_length=specification.get("context_length"),
    )


def build_publication_benchmark_manifest(
    config: Mapping[str, Any],
    *,
    load_snapshot: Callable[[Mapping[str, Any]], LocalDatasetSnapshot]
    | None = None,
) -> dict[str, Any]:
    """Enumerate complete benchmark splits from immutable local snapshots."""

    publication = config.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("config.publication is required")
    declared = publication.get("benchmark_datasets")
    if not isinstance(declared, Mapping):
        raise ValueError("config.publication.benchmark_datasets is required")
    required = {"wikitext2", "ifeval", "gsm8k"}
    if not required.issubset(declared) or set(declared) - set(
        PUBLICATION_BENCHMARKS
    ):
        raise ValueError(
            "publication datasets require WikiText-2, IFEval, and GSM8K; "
            "RULER is optional"
        )
    loader = _load_local_huggingface_snapshot if load_snapshot is None else load_snapshot
    benchmarks = []
    for name in PUBLICATION_BENCHMARKS:
        raw = declared.get(name)
        if raw is None:
            continue
        values = raw if name == "ruler" and isinstance(raw, list) else [raw]
        for value in values:
            specification = _publication_dataset_specification(name, value)
            benchmarks.append(
                _build_publication_benchmark(
                    name,
                    specification,
                    load_snapshot=loader,
                )
            )
    ruler_lengths = sorted(
        benchmark.context_length
        for benchmark in benchmarks
        if benchmark.name == "ruler"
    )
    if ruler_lengths not in ([], list(RULER_LENGTHS)):
        raise ValueError("RULER must be omitted or declared at every canonical length")
    body = {
        "schema_version": PUBLICATION_BENCHMARK_MANIFEST_SCHEMA,
        "benchmarks": [benchmark.to_dict() for benchmark in benchmarks],
    }
    return body | {"manifest_hash": _content_hash(body)}


def load_publication_benchmark_manifest(
    path: str | Path,
) -> tuple[PublicationBenchmark, ...]:
    value = load_immutable_json(path)
    value.pop("content_hash", None)
    if value.get("schema_version") != PUBLICATION_BENCHMARK_MANIFEST_SCHEMA:
        raise ValueError("unsupported publication benchmark manifest")
    manifest_hash = value.pop("manifest_hash", None)
    if manifest_hash != _content_hash(value):
        raise ValueError("publication benchmark manifest identity mismatch")
    records = value.get("benchmarks")
    if not isinstance(records, list):
        raise ValueError("publication benchmark manifest records are missing")
    return tuple(PublicationBenchmark.from_dict(record) for record in records)


@dataclass(frozen=True)
class PublicationProtocol:
    model_name: str
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    thinking_mode: str
    enable_thinking: bool
    greedy: bool
    temperature: float
    token_budgets: tuple[tuple[str, int], ...]
    output_head_location: str = "decode_bf16_unmodeled"
    output_head_precision: str = "BF16"

    def __post_init__(self) -> None:
        for label, value in (
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("thinking_mode", self.thinking_mode),
        ):
            if not value:
                raise ValueError(f"{label} must be pinned")
        for label, value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if not _IMMUTABLE_REVISION.fullmatch(value):
                raise ValueError(f"{label} must be an immutable hash")
        if self.thinking_mode != "disabled" or self.enable_thinking is not False:
            raise ValueError(
                "publication evaluation requires thinking to be disabled"
            )
        _sha256(self.chat_template_sha256, "chat_template_sha256")
        if not self.greedy or self.temperature != 0.0:
            raise ValueError("publication evaluation generation must be greedy with temperature zero")
        budgets = tuple((str(name), int(value)) for name, value in self.token_budgets)
        if {name for name, _ in budgets} != set(PUBLICATION_BENCHMARKS):
            raise ValueError("token budgets must cover every publication evaluation benchmark")
        if len(budgets) != len(set(name for name, _ in budgets)):
            raise ValueError("token budgets contain duplicates")
        if any(value <= 0 for _, value in budgets):
            raise ValueError("token budgets must be positive")
        # Either placement is publishable, and both keep the head at BF16 -
        # the accuracy protocol is unchanged by where the projection runs.
        if (
            self.output_head_location
            not in {"decode_bf16_unmodeled", "external_bf16_service"}
            or self.output_head_precision != "BF16"
        ):
            raise ValueError(
                "headline publication evaluation requires a BF16 output head"
            )
        object.__setattr__(self, "token_budgets", tuple(sorted(budgets)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "thinking_mode": self.thinking_mode,
            "enable_thinking": self.enable_thinking,
            "greedy": self.greedy,
            "temperature": self.temperature,
            "token_budgets": dict(self.token_budgets),
            "output_head_location": self.output_head_location,
            "output_head_precision": self.output_head_precision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationProtocol":
        budgets = value["token_budgets"]
        if not isinstance(budgets, Mapping):
            raise TypeError("token_budgets must be a mapping")
        return cls(
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            chat_template_sha256=str(value["chat_template_sha256"]),
            thinking_mode=str(value["thinking_mode"]),
            enable_thinking=_strict_bool(
                value["enable_thinking"], "enable_thinking"
            ),
            greedy=_strict_bool(value["greedy"], "greedy"),
            temperature=float(value["temperature"]),
            token_budgets=tuple(
                (str(name), int(budget)) for name, budget in budgets.items()
            ),
            output_head_location=str(value["output_head_location"]),
            output_head_precision=str(value["output_head_precision"]),
        )


@dataclass(frozen=True)
class PublicationContract:
    configurations: tuple[PublicationConfiguration, ...]
    hardware_alternatives: tuple[PublicationHardwareAlternative, ...]
    benchmarks: tuple[PublicationBenchmark, ...]
    protocol: PublicationProtocol
    schema_version: str = PUBLICATION_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PUBLICATION_CONTRACT_SCHEMA:
            raise ValueError("unsupported publication contract schema")
        configurations = tuple(self.configurations)
        if tuple(item.role for item in configurations) != PUBLICATION_ROLES:
            raise ValueError("publication evaluation configurations must be BF16/I8/I4/Pareto in order")
        ids = tuple(item.configuration_id for item in configurations)
        profile_ids = tuple(item.profile.profile_id for item in configurations)
        if len(ids) != len(set(ids)) or len(profile_ids) != len(set(profile_ids)):
            raise ValueError("publication evaluation configurations must be distinct")
        configuration_by_id = {
            item.configuration_id: item for item in configurations
        }
        alternatives = tuple(self.hardware_alternatives)
        identities = tuple(
            (item.configuration_id, item.candidate_id) for item in alternatives
        )
        if len(identities) != len(set(identities)):
            raise ValueError("publication hardware alternatives must be distinct")
        alternatives_by_configuration: dict[
            str, list[PublicationHardwareAlternative]
        ] = {}
        for alternative in alternatives:
            configuration = configuration_by_id.get(
                alternative.configuration_id
            )
            if (
                configuration is None
                or configuration.role == "bf16"
                or alternative.profile_id != configuration.profile.profile_id
            ):
                raise ValueError(
                    "publication hardware alternative differs from its accuracy configuration"
                )
            alternatives_by_configuration.setdefault(
                alternative.configuration_id,
                [],
            ).append(alternative)
        for configuration in configurations:
            count = len(
                alternatives_by_configuration.get(
                    configuration.configuration_id,
                    (),
                )
            )
            if (configuration.role == "bf16" and count) or (
                configuration.role != "bf16" and not count
            ):
                raise ValueError(
                    "publication hardware coverage differs from accuracy configurations"
                )
        benchmarks = tuple(self.benchmarks)
        names = [item.name for item in benchmarks]
        if names.count("wikitext2") != 1 or names.count("ifeval") != 1 or names.count("gsm8k") != 1:
            raise ValueError("publication evaluation requires one full WikiText-2, IFEval, and GSM8K")
        ruler = sorted(
            item.context_length for item in benchmarks if item.name == "ruler"
        )
        if ruler not in ([], list(RULER_LENGTHS)):
            raise ValueError(
                "publication evaluation requires either no RULER study or "
                "RULER at 4K/8K/16K/32K"
            )
        benchmark_ids = tuple(item.benchmark_id for item in benchmarks)
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("publication evaluation benchmark identities must be unique")
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "hardware_alternatives", alternatives)
        object.__setattr__(self, "benchmarks", benchmarks)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "configurations": [item.to_dict() for item in self.configurations],
            "hardware_alternatives": [
                item.to_dict() for item in self.hardware_alternatives
            ],
            "benchmarks": [item.to_dict() for item in self.benchmarks],
            "protocol": self.protocol.to_dict(),
        }

    @property
    def canonical_hash(self) -> str:
        return _content_hash(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"contract_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationContract":
        contract = cls(
            configurations=tuple(
                PublicationConfiguration.from_dict(item)
                for item in value["configurations"]
            ),
            hardware_alternatives=tuple(
                PublicationHardwareAlternative.from_dict(item)
                for item in value["hardware_alternatives"]
            ),
            benchmarks=tuple(
                PublicationBenchmark.from_dict(item)
                for item in value["benchmarks"]
            ),
            protocol=PublicationProtocol.from_dict(value["protocol"]),
            schema_version=str(value["schema_version"]),
        )
        if value.get("contract_hash") != contract.canonical_hash:
            raise ValueError("publication contract identity mismatch")
        return contract


@dataclass(frozen=True)
class PublicationItemMetric:
    item_id: str
    score: float | None = None
    nll_sum: float | None = None
    token_count: int | None = None
    post_handoff_nll_sum: float | None = None
    post_handoff_token_count: int | None = None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("publication item ID is required")
        score_mode = self.score is not None
        nll_mode = self.nll_sum is not None or self.token_count is not None
        post_handoff_mode = (
            self.post_handoff_nll_sum is not None
            or self.post_handoff_token_count is not None
        )
        if score_mode == nll_mode:
            raise ValueError("item result must contain either score or NLL")
        if score_mode and post_handoff_mode:
            raise ValueError("task scores cannot carry WikiText-2 NLL")
        if nll_mode != post_handoff_mode:
            raise ValueError(
                "WikiText-2 requires standard and split-boundary NLL"
            )
        if score_mode and (
            not math.isfinite(float(self.score)) or not 0 <= float(self.score) <= 100
        ):
            raise ValueError("task score must be finite and in percentage points")
        if nll_mode and (
            self.nll_sum is None
            or self.token_count is None
            or not math.isfinite(self.nll_sum)
            or self.nll_sum < 0
            or self.token_count <= 0
        ):
            raise ValueError("NLL result is invalid")
        if post_handoff_mode and (
            self.post_handoff_nll_sum is None
            or self.post_handoff_token_count is None
            or not math.isfinite(self.post_handoff_nll_sum)
            or self.post_handoff_nll_sum < 0
            or self.post_handoff_token_count <= 0
        ):
            raise ValueError("post-handoff NLL result is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "score_unit": "percentage_points" if self.score is not None else None,
            "nll_sum": self.nll_sum,
            "token_count": self.token_count,
            "post_handoff_nll_sum": self.post_handoff_nll_sum,
            "post_handoff_token_count": self.post_handoff_token_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationItemMetric":
        if value.get("score") is not None and value.get("score_unit") not in {
            None,
            "percentage_points",
        }:
            raise ValueError("publication task-score unit mismatch")
        return cls(
            item_id=str(value["item_id"]),
            score=None if value.get("score") is None else float(value["score"]),
            nll_sum=(
                None if value.get("nll_sum") is None else float(value["nll_sum"])
            ),
            token_count=(
                None
                if value.get("token_count") is None
                else int(value["token_count"])
            ),
            post_handoff_nll_sum=(
                None
                if value.get("post_handoff_nll_sum") is None
                else float(value["post_handoff_nll_sum"])
            ),
            post_handoff_token_count=(
                None
                if value.get("post_handoff_token_count") is None
                else int(value["post_handoff_token_count"])
            ),
        )


@dataclass(frozen=True)
class PublicationSplitEvidence:
    mode: str
    metric_id: str
    handoff_token_source: str
    prefill_precision: str
    transferred_kv_precision: str
    first_token_owner: str
    decode_q_lengths: tuple[int, ...]
    exact_cache_positions: bool
    exact_one_entry_growth: bool
    independent_caches: bool
    admission_count_per_prompt: int
    cache_free_calls: int
    full_item_coverage: bool

    def __post_init__(self) -> None:
        expected = {
            "teacher_forced_cached": (
                STANDARD_WIKITEXT2_METRIC_ID,
                "ground_truth",
                "teacher_forced_reference",
            ),
            "greedy_conditioned_cached_nll": (
                POST_HANDOFF_METRIC_ID,
                "prefill_greedy",
                "prefill",
            ),
            "greedy_cached_generation": (
                TASK_METRIC_ID,
                "prefill_greedy",
                "prefill",
            ),
        }
        if self.mode not in expected:
            raise ValueError("unsupported publication evaluation decode mode")
        if (
            self.metric_id,
            self.handoff_token_source,
            self.first_token_owner,
        ) != expected[self.mode]:
            raise ValueError("publication evaluation handoff metric contract is invalid")
        if (
            self.prefill_precision != "BF16"
            or self.transferred_kv_precision != "BF16"
        ):
            raise ValueError("publication evaluation split-prefill contract is invalid")
        if self.decode_q_lengths != (1,):
            raise ValueError("publication evaluation may contain only cached q_len=1 decode")
        if not all(
            (
                self.exact_cache_positions,
                self.exact_one_entry_growth,
                self.independent_caches,
                self.full_item_coverage,
            )
        ):
            raise ValueError("publication evaluation cached-decode evidence is incomplete")
        if self.admission_count_per_prompt != 1 or self.cache_free_calls != 0:
            raise ValueError("publication evaluation admission/cache-free accounting is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metric_id": self.metric_id,
            "handoff_token_source": self.handoff_token_source,
            "prefill_precision": self.prefill_precision,
            "transferred_kv_precision": self.transferred_kv_precision,
            "first_token_owner": self.first_token_owner,
            "decode_q_lengths": list(self.decode_q_lengths),
            "exact_cache_positions": self.exact_cache_positions,
            "exact_one_entry_growth": self.exact_one_entry_growth,
            "independent_caches": self.independent_caches,
            "admission_count_per_prompt": self.admission_count_per_prompt,
            "cache_free_calls": self.cache_free_calls,
            "full_item_coverage": self.full_item_coverage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationSplitEvidence":
        return cls(
            mode=str(value["mode"]),
            metric_id=str(value["metric_id"]),
            handoff_token_source=str(value["handoff_token_source"]),
            prefill_precision=str(value["prefill_precision"]),
            transferred_kv_precision=str(value["transferred_kv_precision"]),
            first_token_owner=str(value["first_token_owner"]),
            decode_q_lengths=tuple(int(item) for item in value["decode_q_lengths"]),
            exact_cache_positions=_strict_bool(
                value["exact_cache_positions"], "exact_cache_positions"
            ),
            exact_one_entry_growth=_strict_bool(
                value["exact_one_entry_growth"], "exact_one_entry_growth"
            ),
            independent_caches=_strict_bool(
                value["independent_caches"], "independent_caches"
            ),
            admission_count_per_prompt=int(value["admission_count_per_prompt"]),
            cache_free_calls=int(value["cache_free_calls"]),
            full_item_coverage=_strict_bool(
                value["full_item_coverage"], "full_item_coverage"
            ),
        )


@dataclass(frozen=True)
class PublicationEvaluation:
    items: tuple[PublicationItemMetric, ...]
    evidence: PublicationSplitEvidence
    artifacts: tuple[str, ...]
    post_handoff_evidence: PublicationSplitEvidence | None = None

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("publication evaluation returned no items")
        if not self.artifacts:
            raise ValueError("publication evaluation returned no artifacts")
        if self.evidence.metric_id == STANDARD_WIKITEXT2_METRIC_ID:
            if (
                self.post_handoff_evidence is None
                or self.post_handoff_evidence.metric_id
                != POST_HANDOFF_METRIC_ID
            ):
                raise ValueError(
                    "WikiText-2 requires split-boundary metric evidence"
                )
        elif self.post_handoff_evidence is not None:
            raise ValueError(
                "task suites cannot carry split-boundary NLL evidence"
            )


class PublicationExecutor(Protocol):
    def open_configuration(
        self,
        configuration: PublicationConfiguration,
        protocol: PublicationProtocol,
    ) -> AbstractContextManager[Any]:
        ...

    def evaluate(
        self,
        configuration: PublicationConfiguration,
        benchmark: PublicationBenchmark,
        protocol: PublicationProtocol,
        *,
        configuration_handle: Any,
    ) -> PublicationEvaluation:
        ...


def load_publication_executor(
    factory_spec: str,
    *,
    config: Mapping[str, Any],
    contract: PublicationContract,
) -> PublicationExecutor:
    """Load an explicit external adapter; no legacy evaluator is accepted."""

    module_name, separator, function_name = factory_spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("executor factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name)
    executor = factory(config=config, contract=contract)
    for name in ("open_configuration", "evaluate"):
        if not callable(getattr(executor, name, None)):
            raise TypeError(f"publication executor lacks {name}")
    return executor


def _paired_bootstrap(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int,
    replicates: int,
    mode: str,
) -> dict[str, Any]:
    if len(reference) != len(candidate) or len(reference) < 2:
        raise ValueError("paired bootstrap requires aligned observations")
    rng = random.Random(seed)
    values = []
    count = len(reference)
    for _ in range(replicates):
        indices = [rng.randrange(count) for _ in range(count)]
        left = sum(reference[index] for index in indices) / count
        right = sum(candidate[index] for index in indices) / count
        values.append(right - left if mode == "difference" else right / left)
    left = sum(reference) / count
    right = sum(candidate) / count
    estimate = right - left if mode == "difference" else right / left
    return {
        "method": "paired_item_bootstrap",
        "mode": mode,
        "replicates": replicates,
        "seed": seed,
        "estimate": estimate,
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
    }


def _paired_document_nll_bootstrap(
    reference: Sequence[tuple[float, int]],
    candidate: Sequence[tuple[float, int]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if len(reference) != len(candidate) or len(reference) < 2:
        raise ValueError("paired NLL bootstrap requires aligned documents")
    if any(left[1] != right[1] for left, right in zip(reference, candidate)):
        raise ValueError("paired NLL documents have different token counts")
    rng = random.Random(seed)
    count = len(reference)
    ratios = []
    for _ in range(replicates):
        indices = [rng.randrange(count) for _ in range(count)]
        ref_nll = sum(reference[index][0] for index in indices)
        cand_nll = sum(candidate[index][0] for index in indices)
        tokens = sum(reference[index][1] for index in indices)
        ratios.append(math.exp((cand_nll - ref_nll) / tokens))
    reference_nll = sum(value for value, _ in reference)
    candidate_nll = sum(value for value, _ in candidate)
    tokens = sum(value for _, value in reference)
    return {
        "method": "paired_document_cluster_bootstrap",
        "mode": "perplexity_ratio",
        "replicates": replicates,
        "seed": seed,
        "estimate": math.exp((candidate_nll - reference_nll) / tokens),
        "ci95_low": percentile(ratios, 0.025),
        "ci95_high": percentile(ratios, 0.975),
    }


class _PublicationStore:
    def __init__(self, root: Path, contract: PublicationContract) -> None:
        self.root = root
        self.contract = contract
        self.path = root / "results.jsonl"
        root.mkdir(parents=True, exist_ok=True)
        self.records: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = self.path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            payload = payload[: payload.rfind(b"\n") + 1]
            descriptor = os.open(self.path, os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        configurations = {
            item.configuration_id: item for item in self.contract.configurations
        }
        benchmarks = {
            item.benchmark_id: item for item in self.contract.benchmarks
        }
        for line_number, line in enumerate(payload.splitlines(), start=1):
            value = json.loads(line)
            record_hash = value.pop("record_hash", None)
            if record_hash != _content_hash(value):
                raise ValueError(
                    f"publication result checksum mismatch at line {line_number}"
                )
            if value.get("contract_hash") != self.contract.canonical_hash:
                raise ValueError("publication result contract mismatch")
            if value.get("schema_version") != PUBLICATION_RESULT_SCHEMA:
                raise ValueError("publication result schema mismatch")
            attempt = value.get("attempt")
            runtime_seconds = value.get("runtime_seconds")
            if (
                isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or not 1 <= attempt <= 3
                or isinstance(runtime_seconds, bool)
                or not isinstance(runtime_seconds, (int, float))
                or not math.isfinite(runtime_seconds)
                or runtime_seconds < 0
                or value.get("state") not in {"succeeded", "failed"}
            ):
                raise ValueError("publication result terminal fields are invalid")
            configuration = configurations.get(value.get("configuration_id"))
            benchmark = benchmarks.get(value.get("benchmark_id"))
            if (
                configuration is None
                or benchmark is None
                or value.get("configuration") != configuration.to_dict()
                or value.get("benchmark") != benchmark.to_dict()
            ):
                raise ValueError("publication result binding mismatch")
            key = (
                configuration.configuration_id,
                benchmark.benchmark_id,
                attempt,
            )
            if key in self.records:
                raise ValueError("duplicate publication result attempt")
            if value.get("state") == "succeeded":
                if (
                    not isinstance(value.get("result"), Mapping)
                    or value.get("error_class") is not None
                    or value.get("error_message") is not None
                ):
                    raise ValueError("successful publication result is malformed")
                artifacts = value.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    raise ValueError("successful publication result has no artifacts")
                if any(_artifact(item["path"]) != item for item in artifacts):
                    raise ValueError("publication artifact identity changed")
            elif (
                value.get("result") is not None
                or not value.get("error_class")
                or value.get("error_message") is None
            ):
                raise ValueError("failed publication result is malformed")
            self.records[key] = value | {"record_hash": record_hash}

    def latest(self, configuration_id: str, benchmark_id: str) -> dict[str, Any] | None:
        rows = [
            value
            for (left, right, _), value in self.records.items()
            if left == configuration_id and right == benchmark_id
        ]
        return max(rows, key=lambda value: value["attempt"], default=None)

    def append(
        self,
        configuration: PublicationConfiguration,
        benchmark: PublicationBenchmark,
        *,
        attempt: int,
        state: str,
        result: Mapping[str, Any] | None,
        artifacts: Sequence[Mapping[str, Any]],
        runtime_seconds: float,
        error: BaseException | None = None,
    ) -> None:
        body = {
            "schema_version": PUBLICATION_RESULT_SCHEMA,
            "contract_hash": self.contract.canonical_hash,
            "configuration_id": configuration.configuration_id,
            "configuration": configuration.to_dict(),
            "benchmark_id": benchmark.benchmark_id,
            "benchmark": benchmark.to_dict(),
            "attempt": attempt,
            "state": state,
            "result": result,
            "artifacts": [dict(item) for item in artifacts],
            "error_class": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "traceback": traceback.format_exc() if error else None,
            "runtime_seconds": runtime_seconds,
            "completed_at": _timestamp(),
        }
        record = body | {"record_hash": _content_hash(body)}
        key = (
            configuration.configuration_id,
            benchmark.benchmark_id,
            attempt,
        )
        if key in self.records:
            raise ValueError("publication result attempt already exists")
        payload = (_canonical_json(record) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        self.records[key] = record


@dataclass(frozen=True)
class PublicationRunSummary:
    succeeded: int
    failed_terminal: int
    pending: int
    report_path: str | None


class PublicationRunner:
    """Run every configuration/benchmark pair and seal selection gates."""

    def __init__(
        self,
        *,
        contract: PublicationContract,
        executor: PublicationExecutor,
        output_dir: str | Path,
        bootstrap_replicates: int = 2000,
        max_attempts: int = 3,
    ) -> None:
        if bootstrap_replicates < 100:
            raise ValueError("publication bootstrap requires at least 100 replicates")
        if max_attempts != 3:
            raise ValueError("publication execution requires three attempts")
        self.contract = contract
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.bootstrap_replicates = bootstrap_replicates
        self.max_attempts = max_attempts

    @contextmanager
    def _lock(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.output_dir / ".run.lock",
            os.O_CREAT | os.O_RDWR,
            0o644,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _validate_evaluation(
        benchmark: PublicationBenchmark,
        evaluation: PublicationEvaluation,
    ) -> dict[str, Any]:
        items = tuple(evaluation.items)
        if tuple(item.item_id for item in items) != benchmark.item_ids:
            raise ValueError("publication item coverage/order differs from contract")
        if benchmark.name == "wikitext2":
            if (
                evaluation.evidence.metric_id
                != STANDARD_WIKITEXT2_METRIC_ID
                or evaluation.evidence.mode != "teacher_forced_cached"
                or evaluation.post_handoff_evidence is None
                or evaluation.post_handoff_evidence.metric_id
                != POST_HANDOFF_METRIC_ID
            ):
                raise ValueError(
                    "WikiText-2 requires standard and split-boundary evidence"
                )
            if any(
                item.nll_sum is None
                or item.post_handoff_nll_sum is None
                for item in items
            ):
                raise ValueError("WikiText-2 returned incomplete NLL metrics")
            nll_sum = sum(float(item.nll_sum) for item in items)
            token_count = sum(int(item.token_count) for item in items)
            post_handoff_nll_sum = sum(
                float(item.post_handoff_nll_sum) for item in items
            )
            post_handoff_token_count = sum(
                int(item.post_handoff_token_count) for item in items
            )
            result = {
                "metric_id": STANDARD_WIKITEXT2_METRIC_ID,
                "nll_sum": nll_sum,
                "token_count": token_count,
                "mean_token_nll": nll_sum / token_count,
                "perplexity": math.exp(nll_sum / token_count),
                "post_handoff_metric_id": POST_HANDOFF_METRIC_ID,
                "post_handoff_nll_sum": post_handoff_nll_sum,
                "post_handoff_token_count": post_handoff_token_count,
                "post_handoff_greedy_conditioned_nll": (
                    post_handoff_nll_sum / post_handoff_token_count
                ),
                "post_handoff_greedy_conditioned_exp_nll": math.exp(
                    post_handoff_nll_sum / post_handoff_token_count
                ),
            }
        else:
            if (
                evaluation.evidence.mode != "greedy_cached_generation"
                or evaluation.evidence.metric_id != TASK_METRIC_ID
            ):
                raise ValueError("task suites require cached greedy generation")
            if any(item.score is None for item in items):
                raise ValueError("task suite returned NLL instead of scores")
            result = {
                "metric_id": TASK_METRIC_ID,
                "score": sum(float(item.score) for item in items) / len(items),
                "score_unit": "percentage_points",
            }
        return {
            **result,
            "items": [item.to_dict() for item in items],
            "split_execution": evaluation.evidence.to_dict(),
            "post_handoff_split_execution": (
                evaluation.post_handoff_evidence.to_dict()
                if evaluation.post_handoff_evidence is not None
                else None
            ),
        }

    def _report(self, store: _PublicationStore) -> dict[str, Any]:
        rows: dict[tuple[str, str], Mapping[str, Any]] = {}
        for configuration in self.contract.configurations:
            for benchmark in self.contract.benchmarks:
                latest = store.latest(
                    configuration.configuration_id,
                    benchmark.benchmark_id,
                )
                if latest is None or latest["state"] != "succeeded":
                    raise ValueError("publication report requires every successful result")
                rows[(configuration.role, benchmark.benchmark_id)] = latest["result"]
        reference_role = "bf16"
        gates = {}
        split_boundary_comparisons = {}
        for configuration in self.contract.configurations[1:]:
            role = configuration.role
            role_gates: dict[str, Any] = {}
            for benchmark in self.contract.benchmarks:
                reference = rows[(reference_role, benchmark.benchmark_id)]
                candidate = rows[(role, benchmark.benchmark_id)]
                reference_items = reference["items"]
                candidate_items = candidate["items"]
                seed = int(
                    hashlib.sha256(
                        f"{configuration.configuration_id}:{benchmark.benchmark_id}".encode()
                    ).hexdigest()[:16],
                    16,
                )
                if benchmark.name == "wikitext2":
                    ref_values = [
                        (float(item["nll_sum"]), int(item["token_count"]))
                        for item in reference_items
                    ]
                    cand_values = [
                        (float(item["nll_sum"]), int(item["token_count"]))
                        for item in candidate_items
                    ]
                    comparison = _paired_document_nll_bootstrap(
                        ref_values,
                        cand_values,
                        seed=seed,
                        replicates=self.bootstrap_replicates,
                    )
                    perplexity_ratio = comparison["estimate"]
                    role_gates["wikitext2"] = {
                        "relative_perplexity_ratio": perplexity_ratio,
                        "paired_document_bootstrap": comparison,
                        "passed": perplexity_ratio <= 1.01,
                    }
                    split_reference_values = [
                        (
                            float(item["post_handoff_nll_sum"]),
                            int(item["post_handoff_token_count"]),
                        )
                        for item in reference_items
                    ]
                    split_candidate_values = [
                        (
                            float(item["post_handoff_nll_sum"]),
                            int(item["post_handoff_token_count"]),
                        )
                        for item in candidate_items
                    ]
                    split_boundary_comparisons[role] = {
                        "metric_id": POST_HANDOFF_METRIC_ID,
                        "selection_gate": False,
                        "paired_document_exp_nll_ratio": (
                            _paired_document_nll_bootstrap(
                                split_reference_values,
                                split_candidate_values,
                                seed=seed,
                                replicates=self.bootstrap_replicates,
                            )
                        ),
                    }
                else:
                    ref_scores = [float(item["score"]) for item in reference_items]
                    cand_scores = [float(item["score"]) for item in candidate_items]
                    comparison = _paired_bootstrap(
                        ref_scores,
                        cand_scores,
                        seed=seed,
                        replicates=self.bootstrap_replicates,
                        mode="difference",
                    )
                    if benchmark.name == "ruler":
                        ref_mean = sum(ref_scores) / len(ref_scores)
                        cand_mean = sum(cand_scores) / len(cand_scores)
                        retention = cand_mean / ref_mean if ref_mean > 0 else (
                            1.0 if cand_mean == 0 else math.inf
                        )
                        passed = retention >= 0.98
                        role_gates[f"ruler_{benchmark.context_length}"] = {
                            "score_retention": retention,
                            "paired_bootstrap": comparison,
                            "passed": passed,
                        }
                    else:
                        role_gates[benchmark.name] = {
                            "paired_score_difference": comparison,
                            "score_unit": "percentage_points",
                            "passed": comparison["ci95_low"] >= -2.0,
                        }
            role_gates["all_accuracy_gates_passed"] = all(
                value["passed"]
                for key, value in role_gates.items()
                if key != "all_accuracy_gates_passed"
            )
            gates[role] = role_gates
        passing = tuple(
            configuration
            for configuration in self.contract.configurations
            if configuration.role != "bf16"
            and gates[configuration.role]["all_accuracy_gates_passed"]
        )
        selected = bool(passing)
        return {
            "schema_version": PUBLICATION_REPORT_SCHEMA,
            "contract_hash": self.contract.canonical_hash,
            "bootstrap_replicates": self.bootstrap_replicates,
            "accuracy_gates": gates,
            "split_boundary_metrics": split_boundary_comparisons,
            "selection": {
                "accuracy_configuration_ids": [
                    configuration.configuration_id
                    for configuration in passing
                ],
                "selected": selected,
                "failure_action": (
                    None if selected else "report_pareto_frontier_without_near_lossless_claim"
                ),
            },
        }

    def _run_round(self, store: _PublicationStore) -> bool:
        attempted = False
        for configuration in self.contract.configurations:
            pending = [
                benchmark
                for benchmark in self.contract.benchmarks
                if (
                    (latest := store.latest(
                        configuration.configuration_id,
                        benchmark.benchmark_id,
                    ))
                    is None
                    or (
                        latest["state"] == "failed"
                        and int(latest["attempt"]) < self.max_attempts
                    )
                )
            ]
            if not pending:
                continue
            attempted = True
            try:
                manager = self.executor.open_configuration(
                    configuration,
                    self.contract.protocol,
                )
                with manager as handle:
                    for benchmark in pending:
                        latest = store.latest(
                            configuration.configuration_id,
                            benchmark.benchmark_id,
                        )
                        attempt = (
                            1 if latest is None else int(latest["attempt"]) + 1
                        )
                        started = time.monotonic()
                        try:
                            evaluation = self.executor.evaluate(
                                configuration,
                                benchmark,
                                self.contract.protocol,
                                configuration_handle=handle,
                            )
                            result = self._validate_evaluation(
                                benchmark, evaluation
                            )
                            artifacts = tuple(
                                _artifact(path) for path in evaluation.artifacts
                            )
                            store.append(
                                configuration,
                                benchmark,
                                attempt=attempt,
                                state="succeeded",
                                result=result,
                                artifacts=artifacts,
                                runtime_seconds=time.monotonic() - started,
                            )
                        except Exception as error:
                            latest_after = store.latest(
                                configuration.configuration_id,
                                benchmark.benchmark_id,
                            )
                            if (
                                latest_after is not None
                                and int(latest_after["attempt"]) == attempt
                                and latest_after["state"] == "succeeded"
                            ):
                                raise
                            store.append(
                                configuration,
                                benchmark,
                                attempt=attempt,
                                state="failed",
                                result=None,
                                artifacts=(),
                                runtime_seconds=time.monotonic() - started,
                                error=error,
                            )
            except Exception as error:
                for benchmark in pending:
                    latest = store.latest(
                        configuration.configuration_id,
                        benchmark.benchmark_id,
                    )
                    if latest is not None and latest["state"] == "succeeded":
                        continue
                    attempt = (
                        1 if latest is None else int(latest["attempt"]) + 1
                    )
                    if attempt > self.max_attempts:
                        continue
                    store.append(
                        configuration,
                        benchmark,
                        attempt=attempt,
                        state="failed",
                        result=None,
                        artifacts=(),
                        runtime_seconds=0.0,
                        error=error,
                    )
        return attempted

    def run(self) -> PublicationRunSummary:
        with self._lock():
            write_immutable_json(
                self.output_dir / "contract.json",
                self.contract.to_dict(),
            )
            store = _PublicationStore(self.output_dir, self.contract)
            while self._run_round(store):
                pass
            succeeded = 0
            failed_terminal = 0
            pending_count = 0
            for configuration in self.contract.configurations:
                for benchmark in self.contract.benchmarks:
                    latest = store.latest(
                        configuration.configuration_id,
                        benchmark.benchmark_id,
                    )
                    if latest is not None and latest["state"] == "succeeded":
                        succeeded += 1
                    elif latest is not None and int(latest["attempt"]) >= 3:
                        failed_terminal += 1
                    else:
                        pending_count += 1
            report_path = None
            expected = len(self.contract.configurations) * len(
                self.contract.benchmarks
            )
            if succeeded == expected:
                report = self._report(store)
                path = self.output_dir / "publication_report.json"
                write_immutable_json(path, report)
                report_path = str(path)
            return PublicationRunSummary(
                succeeded=succeeded,
                failed_terminal=failed_terminal,
                pending=pending_count,
                report_path=report_path,
            )


# ---------------------------------------------------------------------------
# Run command
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument(
        "--executor",
        default="decode_dse.software.benchmark_evaluator:create_executor",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    config: Any = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain a JSON object")
    value = load_immutable_json(args.contract)
    value.pop("content_hash", None)
    contract = PublicationContract.from_dict(value)
    if contract.protocol.model_name != str(config["model_name"]):
        raise ValueError("publication model differs from execution config")
    if contract.protocol.model_revision != str(config["model_revision"]):
        raise ValueError("publication model revision mismatch")
    if contract.protocol.tokenizer_revision != str(
        config["tokenizer_revision"]
    ):
        raise ValueError("publication tokenizer revision mismatch")
    executor = load_publication_executor(
        args.executor,
        config=config,
        contract=contract,
    )
    summary = PublicationRunner(
        contract=contract,
        executor=executor,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    ).run()
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0 if summary.report_path is not None else 2


# ---------------------------------------------------------------------------
# Contract construction command
# ---------------------------------------------------------------------------
def build_contract_load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_benchmark_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal full publication benchmark splits from local caches."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def build_benchmark_manifest_main(
    argv: Iterable[str] | None = None,
) -> int:
    args = build_benchmark_manifest_parser().parse_args(
        tuple(argv) if argv is not None else None
    )
    config = build_contract_load_json(args.config)
    if not isinstance(config, Mapping):
        raise ValueError("config must contain a JSON object")
    manifest = build_publication_benchmark_manifest(config)
    output = write_immutable_json(args.output, manifest)
    load_publication_benchmark_manifest(output)
    print(
        json.dumps(
            load_immutable_json(output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_source_selection(
    path: str | Path,
    *,
    manifest_hash: str,
    schedule_hash: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    value = load_immutable_json(path)
    if (
        value.get("schema_version") != "decode-refinement-source-selection"
        or value.get("manifest_hash") != manifest_hash
        or value.get("schedule_hash") != schedule_hash
    ):
        raise ValueError("refinement source selection provenance differs")
    selection = value.get("source_selection")
    roles = selection.get("source_roles") if isinstance(selection, Mapping) else None
    expected = {
        "uniform_mxint8",
        "uniform_mxint4",
        "mxint_kv2",
        "accuracy_constrained_deployment",
    }
    if (
        not isinstance(roles, Mapping)
        or set(roles) != expected
        or any(not isinstance(profile_id, str) or not profile_id for profile_id in roles.values())
        or len(set(roles.values())) != len(expected)
    ):
        raise ValueError("refinement source roles are incomplete or duplicated")
    return {str(role): str(profile_id) for role, profile_id in roles.items()}, value


def _hardware_alternatives_for_configuration(
    configuration: PublicationConfiguration,
    *,
    source_profile_id: str,
    hardware_artifacts: Sequence[str | Path],
    model_revision: str,
    tokenizer_revision: str,
) -> tuple[PublicationHardwareAlternative, ...]:
    from decode_dse.hardware.design_space import (
        HARDWARE_STORAGE_REVISION,
        load_hardware_artifact,
    )

    alternatives = []
    identities: set[tuple[str, str]] = set()
    for raw_path in hardware_artifacts:
        path = Path(raw_path).resolve()
        artifact_sha256 = _sha256_file(path)
        header, rows = load_hardware_artifact(path)
        provenance = header.get("provenance")
        if (
            header.get("storage_revision") != HARDWARE_STORAGE_REVISION
            or not isinstance(provenance, Mapping)
            or provenance.get("model_revision") != model_revision
            or provenance.get("tokenizer_revision") != tokenizer_revision
        ):
            raise ValueError("refined hardware artifact provenance differs")
        for row in rows:
            if row.get("profile_id") != configuration.profile.profile_id:
                continue
            if row.get("profile") != configuration.profile.to_dict():
                raise ValueError("refined hardware profile binding differs")
            labels = row.get("retention_labels")
            if not isinstance(labels, list) or "profile_frontier" not in labels:
                continue
            validity = row.get("validity")
            metrics = row.get("metrics")
            whole = metrics.get("whole_model") if isinstance(metrics, Mapping) else None
            energy = (
                whole.get("calibrated_energy")
                if isinstance(whole, Mapping)
                else None
            )
            # Admission is the single policy: the row must be priced by a
            # validated, identified pricing model.  Individual compiler and
            # emulator coverage is *disclosed* on the alternative rather than
            # demanded a second time here -- re-imposing it would restrict the
            # contract to the one geometry the hardware-validation stage ran
            # at.  RTL and DC evidence stays recorded, never demanded.
            admission = evaluate_publication_admission(row)
            if (
                not admission.admitted
                or not isinstance(validity, Mapping)
                or not isinstance(whole, Mapping)
                or whole.get("rankable") is not True
                or not isinstance(energy, Mapping)
                or not isinstance(energy.get("energy_tier"), str)
                or energy.get("total_j") is None
                or whole.get("tpot_ms") is None
            ):
                raise ValueError(
                    "retained refined hardware frontier row is not deployable"
                )
            alternative = PublicationHardwareAlternative(
                configuration_id=configuration.configuration_id,
                profile_id=configuration.profile.profile_id,
                source_profile_id=source_profile_id,
                candidate_id=str(row["candidate_id"]),
                record_hash=str(row["record_hash"]),
                hardware_artifact_sha256=artifact_sha256,
                tpot_ms=float(whole["tpot_ms"]),
                energy_per_token_j=float(energy["total_j"]),
                energy_tier=str(energy["energy_tier"]),
                individually_validated=admission.individually_validated,
            )
            identity = alternative.configuration_id, alternative.candidate_id
            if identity in identities:
                raise ValueError(
                    "refined hardware artifacts contain a duplicate candidate"
                )
            identities.add(identity)
            alternatives.append(alternative)
    if not alternatives:
        raise ValueError(
            f"no exact repriced hardware frontier covers {configuration.role}"
        )
    return tuple(
        sorted(
            alternatives,
            key=lambda item: (
                0 if item.energy_tier == "dc_calibrated" else 1,
                item.energy_per_token_j,
                item.tpot_ms,
                item.candidate_id,
                item.record_hash,
            ),
        )
    )


def build_publication_configuration_manifest(
    *,
    manifest_path: str | Path,
    schedule_path: str | Path,
    source_selection_path: str | Path,
    merge_receipt_path: str | Path,
    hardware_artifacts: Sequence[str | Path],
    merged_results_path: str | Path | None = None,
    publication_timing_tier: str,
) -> dict[str, Any]:
    """Select accuracy configurations once and retain every hardware option."""

    from decode_dse.hardware.design_space import TIMING_TIER_REQUIRED_VALIDITY
    from decode_dse.legality import evaluate_profile_legality
    from decode_dse.manifest import load_manifest
    from decode_dse.software.refinement_runner import (
        load_refinement_merged_results,
    )
    from decode_dse.software.refinement_schedule import load_refinement_schedule

    required_validity = TIMING_TIER_REQUIRED_VALIDITY.get(publication_timing_tier)
    if required_validity is None:
        raise ValueError(
            "no measured-validity requirement is declared for timing tier "
            f"{publication_timing_tier!r}"
        )
    manifest = load_manifest(manifest_path)
    schedule = load_refinement_schedule(schedule_path)
    roles, source_selection = _load_source_selection(
        source_selection_path,
        manifest_hash=manifest.canonical_hash,
        schedule_hash=schedule.canonical_hash,
    )
    merge = load_refinement_merged_results(
        schedule,
        merge_receipt_path,
        results_path=merged_results_path,
    )
    terminal_by_profile = {
        str(row["profile_id"]): row for row in merge.terminal_rows
    }
    schedule_by_source: dict[str, list[Any]] = {}
    for entry in schedule.entries:
        schedule_by_source.setdefault(
            entry.profile.source_profile.profile_id,
            [],
        ).append(entry)

    bf16_entries = tuple(
        entry for entry in manifest.entries if entry.profile.kind == "bf16_reference"
    )
    if len(bf16_entries) != 1:
        raise ValueError("publication accuracy requires exactly one BF16 reference")
    configurations = [
        PublicationConfiguration(
            role="bf16",
            profile=bf16_entries[0].profile,
            validity=bf16_entries[0].validity,
        )
    ]
    selected_receipts = []
    role_sources = (
        ("uniform_i8", roles["uniform_mxint8"]),
        ("uniform_i4", roles["uniform_mxint4"]),
        ("pareto", roles["accuracy_constrained_deployment"]),
    )
    for role, source_profile_id in role_sources:
        candidates = []
        for entry in schedule_by_source.get(source_profile_id, ()):
            row = terminal_by_profile[entry.profile_id]
            result = row.get("result")
            if row.get("state") != "succeeded" or not isinstance(result, Mapping):
                continue
            if any(
                getattr(entry.validity, name) is not True
                for name in required_validity
            ):
                continue
            # Mirror the refined-artifact writer's filter so every selected
            # role winner is guaranteed to exist in the repriced artifact.
            if not evaluate_profile_legality(entry.profile).hardware_candidate:
                continue
            candidates.append(
                (
                    float(result["mean_token_nll"]),
                    entry.profile_id,
                    entry,
                    row,
                )
            )
        if not candidates:
            raise ValueError(
                f"no successful fully measured refinement covers {role}"
            )
        mean_nll, _, entry, row = min(candidates)
        configuration = PublicationConfiguration(
            role=role,
            profile=entry.profile,
            validity=entry.validity,
        )
        configurations.append(configuration)
        selected_receipts.append(
            {
                "role": role,
                "configuration_id": configuration.configuration_id,
                "profile_id": entry.profile_id,
                "source_profile_id": source_profile_id,
                "mean_token_nll": mean_nll,
                "terminal_record_hash": row["record_hash"],
                "selection_rule": (
                    "minimum_successful_mean_token_nll_then_profile_id"
                ),
            }
        )

    alternatives = []
    for configuration, (_, source_profile_id) in zip(
        configurations[1:], role_sources
    ):
        alternatives.extend(
            _hardware_alternatives_for_configuration(
                configuration,
                source_profile_id=source_profile_id,
                hardware_artifacts=hardware_artifacts,
                model_revision=manifest.model_revision,
                tokenizer_revision=str(manifest.tokenizer_revision),
            )
        )
    artifact_receipts = [
        {
            "path": str(Path(path).resolve()),
            "size_bytes": Path(path).resolve().stat().st_size,
            "sha256": _sha256_file(Path(path).resolve()),
        }
        for path in hardware_artifacts
    ]
    body = {
        "schema_version": PUBLICATION_CONFIGURATION_MANIFEST_SCHEMA,
        "model_name": manifest.model_name,
        "model_revision": manifest.model_revision,
        "tokenizer_revision": manifest.tokenizer_revision,
        "base_manifest_hash": manifest.canonical_hash,
        "refinement_schedule_hash": schedule.canonical_hash,
        "source_selection_content_hash": source_selection["content_hash"],
        "refinement_merge_content_hash": merge.receipt["content_hash"],
        "merged_results_sha256": merge.results_sha256,
        "selection_semantics": {
            "accuracy_evaluations_per_configuration": 1,
            "hardware_join_stage": "after_publication_accuracy_gates",
            "hardware_frontier_policy": (
                "all_exact_profile_frontier_rows_per_energy_tier"
            ),
            "source_hardware_costs_inherited": False,
            "control_source_excluded": "mxint_kv2",
        },
        "selected_refinements": selected_receipts,
        "hardware_artifacts": artifact_receipts,
        "configurations": [item.to_dict() for item in configurations],
        "hardware_alternatives": [item.to_dict() for item in alternatives],
    }
    return body | {"manifest_hash": _content_hash(body)}


def load_publication_configuration_manifest(
    path: str | Path,
) -> dict[str, Any]:
    value = load_immutable_json(path)
    content_hash = value.pop("content_hash")
    manifest_hash = value.pop("manifest_hash", None)
    if (
        value.get("schema_version") != PUBLICATION_CONFIGURATION_MANIFEST_SCHEMA
        or manifest_hash != _content_hash(value)
        or not isinstance(value.get("configurations"), list)
        or not isinstance(value.get("hardware_alternatives"), list)
    ):
        raise ValueError("publication configuration manifest is invalid")
    configurations = tuple(
        PublicationConfiguration.from_dict(item)
        for item in value["configurations"]
    )
    alternatives = tuple(
        PublicationHardwareAlternative.from_dict(item)
        for item in value["hardware_alternatives"]
    )
    if tuple(item.role for item in configurations) != PUBLICATION_ROLES:
        raise ValueError("publication configuration roles are incomplete or reordered")
    configuration_by_id = {
        item.configuration_id: item for item in configurations
    }
    if len(configuration_by_id) != len(configurations):
        raise ValueError("publication configurations are duplicated")
    coverage = {item.configuration_id: 0 for item in configurations}
    seen = set()
    for alternative in alternatives:
        identity = alternative.configuration_id, alternative.candidate_id
        configuration = configuration_by_id.get(alternative.configuration_id)
        if (
            identity in seen
            or configuration is None
            or configuration.role == "bf16"
            or alternative.profile_id != configuration.profile.profile_id
        ):
            raise ValueError("publication hardware coverage is inconsistent")
        seen.add(identity)
        coverage[alternative.configuration_id] += 1
    if coverage[configurations[0].configuration_id] != 0 or any(
        coverage[item.configuration_id] == 0 for item in configurations[1:]
    ):
        raise ValueError("publication hardware coverage is incomplete")
    return {
        **value,
        "manifest_hash": manifest_hash,
        "content_hash": content_hash,
    }


def _resolve_config_asset(
    value: str,
    *,
    config_path: str | Path | None,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    roots = [Path.cwd()]
    if config_path is not None:
        source = Path(config_path).resolve()
        roots.extend((source.parent, *source.parents))
    for root in roots:
        candidate = (root / raw).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"configured chat-template asset is missing: {value}")


def seal_publication_chat_template(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    tokenizer_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Seal an exact pinned template without inventing or downloading one."""

    publication = config.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("config.publication is required")
    if publication.get("enable_thinking") is not False:
        raise ValueError("publication chat template requires thinking disabled")
    model_name = str(config.get("model_name", ""))
    model_revision = str(config.get("model_revision", ""))
    tokenizer_revision = str(config.get("tokenizer_revision", ""))
    if (
        not model_name
        or not _IMMUTABLE_REVISION.fullmatch(model_revision)
        or not _IMMUTABLE_REVISION.fullmatch(tokenizer_revision)
    ):
        raise ValueError("chat-template model and tokenizer must be pinned")

    configured_asset = publication.get("chat_template_asset")
    if configured_asset is not None:
        if not isinstance(configured_asset, str) or not configured_asset:
            raise ValueError("publication.chat_template_asset is invalid")
        source_path = _resolve_config_asset(
            configured_asset,
            config_path=config_path,
        )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if (
            not isinstance(source, Mapping)
            or source.get("schema_version") != PUBLICATION_CHAT_TEMPLATE_SCHEMA
            or source.get("model_name") != model_name
            or source.get("model_revision") != model_revision
            or source.get("tokenizer_revision") != tokenizer_revision
            or source.get("enable_thinking") is not False
            or not isinstance(source.get("chat_template"), str)
            or not source.get("chat_template")
        ):
            raise ValueError("configured chat-template asset contract differs")
        template = str(source["chat_template"])
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if source.get("chat_template_sha256") != template_hash:
            raise ValueError("configured chat-template content hash differs")
        source_receipt = {
            "kind": "pinned_local_asset",
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }
    else:
        if tokenizer_loader is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required to extract the pinned local tokenizer template"
                ) from exc
            tokenizer_loader = AutoTokenizer.from_pretrained
        try:
            tokenizer = tokenizer_loader(
                model_name,
                revision=tokenizer_revision,
                local_files_only=True,
                trust_remote_code=False,
            )
            template = tokenizer.get_chat_template()
        except Exception as exc:
            raise RuntimeError(
                "the pinned tokenizer is unavailable locally or has no unambiguous chat template"
            ) from exc
        if not isinstance(template, str) or not template:
            raise ValueError("the pinned tokenizer returned an empty chat template")
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        source_receipt = {
            "kind": "pinned_local_tokenizer_execution",
            "loader": "transformers.AutoTokenizer.from_pretrained",
            "local_files_only": True,
            "trust_remote_code": False,
        }
    configured_hash = publication.get("chat_template_sha256")
    if configured_hash is not None and configured_hash != template_hash:
        raise ValueError("chat template hash differs from config.publication")
    return {
        "schema_version": PUBLICATION_CHAT_TEMPLATE_SCHEMA,
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "enable_thinking": False,
        "chat_template_sha256": template_hash,
        "chat_template": template,
        "source": source_receipt,
    }


def build_configurations_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build accuracy-only publication configurations and hardware alternatives."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--refinement-schedule", required=True)
    parser.add_argument("--source-selection", required=True)
    parser.add_argument("--refinement-merge", required=True)
    parser.add_argument("--refinement-results")
    parser.add_argument(
        "--hardware-artifact",
        action="append",
        required=True,
    )
    parser.add_argument("--publication-timing-tier", required=True)
    parser.add_argument("--output", required=True)
    return parser


def build_configurations_main(argv: Iterable[str] | None = None) -> int:
    args = build_configurations_parser().parse_args(
        tuple(argv) if argv is not None else None
    )
    manifest = build_publication_configuration_manifest(
        manifest_path=args.manifest,
        schedule_path=args.refinement_schedule,
        source_selection_path=args.source_selection,
        merge_receipt_path=args.refinement_merge,
        merged_results_path=args.refinement_results,
        hardware_artifacts=tuple(args.hardware_artifact),
        publication_timing_tier=args.publication_timing_tier,
    )
    output = write_immutable_json(args.output, manifest)
    load_publication_configuration_manifest(output)
    print(json.dumps(load_immutable_json(output), indent=2, sort_keys=True))
    return 0


def build_chat_template_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal the exact publication chat template from pinned local inputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def build_chat_template_main(argv: Iterable[str] | None = None) -> int:
    args = build_chat_template_parser().parse_args(
        tuple(argv) if argv is not None else None
    )
    config = build_contract_load_json(args.config)
    if not isinstance(config, Mapping):
        raise ValueError("config must contain a JSON object")
    asset = seal_publication_chat_template(
        config,
        config_path=args.config,
    )
    output = write_immutable_json(args.output, asset)
    print(json.dumps(load_immutable_json(output), indent=2, sort_keys=True))
    return 0



def build_publication_contract(
    *,
    config: Mapping[str, Any],
    configurations: Iterable[Mapping[str, Any]],
    hardware_alternatives: Iterable[Mapping[str, Any]],
    benchmarks: Iterable[Mapping[str, Any]],
    chat_template_path: str | Path,
) -> PublicationContract:
    """Bind exact items, precisions, hardware evidence, and generation settings."""

    settings = config.get("publication")
    if not isinstance(settings, Mapping):
        raise ValueError("config.publication is required")
    template_path = Path(chat_template_path).resolve()
    if not template_path.is_file() or template_path.stat().st_size <= 0:
        raise ValueError("the pinned chat template is missing")
    template_asset = build_contract_load_json(template_path)
    if (
        not isinstance(template_asset, Mapping)
        or template_asset.get("schema_version") != PUBLICATION_CHAT_TEMPLATE_SCHEMA
        or template_asset.get("model_name") != config.get("model_name")
        or template_asset.get("model_revision") != config.get("model_revision")
        or template_asset.get("tokenizer_revision")
        != config.get("tokenizer_revision")
        or template_asset.get("enable_thinking") is not False
        or not isinstance(template_asset.get("chat_template"), str)
        or not isinstance(template_asset.get("source"), Mapping)
    ):
        raise ValueError("the pinned chat-template asset is invalid")
    template_hash = hashlib.sha256(
        template_asset["chat_template"].encode("utf-8")
    ).hexdigest()
    if template_asset.get("chat_template_sha256") != template_hash:
        raise ValueError("the chat-template asset hash is invalid")
    expected_template_hash = settings.get("chat_template_sha256")
    if (
        expected_template_hash is not None
        and expected_template_hash != template_hash
    ):
        raise ValueError("chat template hash differs from config.publication")
    token_budgets = settings.get("token_budgets")
    if not isinstance(token_budgets, Mapping):
        raise ValueError("config.publication.token_budgets is required")
    contract = PublicationContract(
        configurations=tuple(
            PublicationConfiguration.from_dict(value)
            for value in configurations
        ),
        hardware_alternatives=tuple(
            PublicationHardwareAlternative.from_dict(value)
            for value in hardware_alternatives
        ),
        benchmarks=tuple(
            PublicationBenchmark.from_dict(value) for value in benchmarks
        ),
        protocol=PublicationProtocol(
            model_name=str(config["model_name"]),
            model_revision=str(config["model_revision"]),
            tokenizer_revision=str(config["tokenizer_revision"]),
            chat_template_sha256=template_hash,
            thinking_mode=str(settings.get("thinking_mode", "")),
            enable_thinking=settings.get("enable_thinking"),
            greedy=settings.get("greedy"),
            temperature=float(settings.get("temperature", math.nan)),
            token_budgets=tuple(
                (str(name), int(value)) for name, value in token_budgets.items()
            ),
            output_head_location=str(
                settings.get("output_head_location", "")
            ),
            output_head_precision=str(
                settings.get("output_head_precision", "")
            ),
        ),
    )
    return contract


def build_contract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--configurations", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--output", required=True)
    return parser


def build_contract_main(argv: Iterable[str] | None = None) -> int:
    args = build_contract_parser().parse_args(tuple(argv) if argv is not None else None)
    config = build_contract_load_json(args.config)
    configuration_manifest = load_publication_configuration_manifest(
        args.configurations
    )
    benchmarks = load_publication_benchmark_manifest(args.benchmarks)
    if not isinstance(config, dict):
        raise ValueError("config must contain a JSON object")
    if any(
        configuration_manifest.get(name) != config.get(name)
        for name in ("model_name", "model_revision", "tokenizer_revision")
    ):
        raise ValueError("publication configuration manifest model differs")
    contract = build_publication_contract(
        config=config,
        configurations=configuration_manifest["configurations"],
        hardware_alternatives=configuration_manifest["hardware_alternatives"],
        benchmarks=(benchmark.to_dict() for benchmark in benchmarks),
        chat_template_path=args.chat_template,
    )
    write_immutable_json(args.output, contract.to_dict())
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "PUBLICATION_BENCHMARKS",
    "PUBLICATION_BENCHMARK_MANIFEST_SCHEMA",
    "PUBLICATION_CHAT_TEMPLATE_SCHEMA",
    "PUBLICATION_CONFIGURATION_MANIFEST_SCHEMA",
    "PUBLICATION_CONTRACT_SCHEMA",
    "PUBLICATION_REPORT_SCHEMA",
    "PUBLICATION_RESULT_SCHEMA",
    "PUBLICATION_ROLES",
    "POST_HANDOFF_METRIC_ID",
    "RULER_LENGTHS",
    "STANDARD_WIKITEXT2_METRIC_ID",
    "TASK_METRIC_ID",
    "PublicationBenchmark",
    "PublicationConfiguration",
    "PublicationContract",
    "PublicationHardwareAlternative",
    "PublicationEvaluation",
    "PublicationExecutor",
    "PublicationItemMetric",
    "PublicationProtocol",
    "PublicationRunSummary",
    "PublicationRunner",
    "PublicationSplitEvidence",
    "LocalDatasetSnapshot",
    "build_publication_benchmark_manifest",
    "build_publication_configuration_manifest",
    "load_publication_configuration_manifest",
    "load_publication_benchmark_manifest",
    "load_publication_executor",
    "seal_publication_chat_template",
]

def dispatch(argv: Sequence[str] | None = None) -> int:
    """Route to one of this module's commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "run": main,
        "contract": build_contract_main,
        "manifest": build_benchmark_manifest_main,
        "configurations": build_configurations_main,
        "chat-template": build_chat_template_main,
    }
    if arguments and arguments[0] in commands:
        return commands[arguments[0]](arguments[1:])
    return main(arguments)


if __name__ == "__main__":
    raise SystemExit(dispatch())
