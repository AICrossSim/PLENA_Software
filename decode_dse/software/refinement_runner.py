"""Prepare, run, and launch restartable cached-decode precision refinement.

Commands: `prepare` builds disjoint samples, BF16 caches, and GPTQ
calibration; `run` executes or resumes one shard; `launch` fans the
schedule across isolated workers and checksum-merges their shards."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Protocol, Sequence

from decode_dse.hardware.statistics import percentile
from decode_dse.software.cache_artifacts import (
    ArtifactProvenance,
    load_prefill_artifact,
    save_prefill_artifact,
)
from decode_dse.software.cached_decode import capture_bf16_prefill
from decode_dse.software.refinement_schedule import (
    DecodeRefinementProfile,
    RefinementSchedule,
    RefinementScheduleEntry,
    RefinementShardPlan,
    build_refinement_shard_plans,
    validate_refinement_shard_plan,
    write_refinement_shard_plan,
)
from decode_dse.software.runtime_environment import (
    RuntimeEnvironment,
    capture_runtime_environment,
    initialize_numerical_runtime,
)
from decode_dse.software.sweep_plan import load_immutable_json, write_immutable_json
from decode_dse.software.token_samples import (
    REFINEMENT_DECODE_STEPS,
    RefinementSampleBundle,
    TokenizedSourceDocument,
    build_refinement_bundle_from_documents,
    load_refinement_sample_bundle,
    load_sample_bundle,
    save_refinement_sample_bundle,
)

REFINEMENT_BANK_SCHEMA = "decode-refinement-bank"


REFINEMENT_RESULT_SCHEMA = "decode-refinement-result"


REFINEMENT_COMPLETION_SCHEMA = "decode-refinement-completion"


REFINEMENT_MERGE_SCHEMA = "decode-refinement-merge"


_SAFE_TOKEN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = tuple(sorted(value for value in path.rglob("*") if value.is_file()))
    if not files:
        raise ValueError(f"checkpoint contains no files: {path}")
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        payload = value.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _artifact_identity(path_value: str) -> dict[str, Any]:
    path = Path(path_value).resolve()
    if path.is_file():
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"refinement artifact is empty: {path}")
        digest = _file_hash(path)
        kind = "file"
        file_count = 1
    elif path.is_dir():
        files = tuple(sorted(value for value in path.rglob("*") if value.is_file()))
        if not files:
            raise ValueError(f"refinement artifact directory is empty: {path}")
        size = sum(value.stat().st_size for value in files)
        digest = _tree_hash(path)
        kind = "directory"
        file_count = len(files)
    else:
        raise ValueError(f"refinement artifact does not exist: {path}")
    return {
        "path": str(path),
        "kind": kind,
        "file_count": file_count,
        "size_bytes": size,
        "sha256": digest,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        isinstance(error, MemoryError)
        or "outofmemory" in name
        or ("out of memory" in message and ("cuda" in message or "host" in message))
    )


def rotation_policy() -> dict[str, Any]:
    """Return the fixed calibration-aware rotation search contract."""

    return {
        "calibration_samples": 32,
        "calibration_sequence_length": 1024,
        "improvement_epsilon": 0.0,
        "matmul_types": "all_supported",
        "cache_winners": True,
        "score_phase": "decode",
    }


def rotation_policy_hash() -> str:
    return _content_hash(rotation_policy())


def rotation_decision_contract(
    profile: DecodeRefinementProfile,
) -> dict[str, Any]:
    """Bind rotation search to every precision role that affects its score."""

    return {
        "policy": rotation_policy(),
        "profile_id": profile.profile_id,
        "profile": profile.to_dict(),
    }


def rotation_decision_contract_hash(
    profile: DecodeRefinementProfile,
) -> str:
    return _content_hash(rotation_decision_contract(profile))


def refinement_rng_policy() -> dict[str, Any]:
    """Return the deterministic calibration and search runtime contract."""

    return {
        "python_random": "seeded",
        "numpy_random": "seeded_if_available",
        "torch_cpu": "seeded",
        "torch_cuda_all": "seeded",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }


def refinement_rng_policy_hash() -> str:
    return _content_hash(refinement_rng_policy())


def refinement_bank_key(
    profile: DecodeRefinementProfile,
) -> tuple[str, str, str]:
    """Keep GPTQ banks W-shared and rotation decisions profile-local."""

    scope = profile.profile_id if profile.weight_method == "rotation" else "shared"
    return profile.weight_format, profile.weight_method, scope


@dataclass(frozen=True)
class RefinementBankSpec:
    """Immutable identity of one calibrated W/method bank."""

    model_name: str
    model_revision: str
    weight_format: str
    weight_method: str
    block_size: int
    calibration_dataset: str
    calibration_revision: str
    calibration_bundle_hash: str
    calibration_path: str
    calibration_samples: int
    calibration_sequence_length: int
    calibration_batch_size: int
    checkpoint_dir: str
    quantile_search: bool = True
    clip_search_y: bool = True
    rng_seed: int = 0
    rng_policy_hash: str = ""
    rotation_config_hash: str | None = None
    rotation_profile_id: str | None = None
    schema_version: str = REFINEMENT_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFINEMENT_BANK_SCHEMA:
            raise ValueError(f"unsupported bank schema {self.schema_version!r}")
        for label, value in (
            ("model_name", self.model_name),
            ("model_revision", self.model_revision),
            ("weight_format", self.weight_format),
            ("weight_method", self.weight_method),
            ("calibration_dataset", self.calibration_dataset),
            ("calibration_revision", self.calibration_revision),
            ("calibration_bundle_hash", self.calibration_bundle_hash),
            ("calibration_path", self.calibration_path),
            ("checkpoint_dir", self.checkpoint_dir),
        ):
            if not value:
                raise ValueError(f"{label} is required")
        if self.weight_method not in {"gptq_erry", "rotation"}:
            raise ValueError(
                "refinement banks must use GPTQ+Erry or selective rotation"
            )
        if self.block_size != 8:
            raise ValueError("refinement banks require native block size 8")
        if (
            min(
                self.calibration_samples,
                self.calibration_sequence_length,
                self.calibration_batch_size,
            )
            <= 0
        ):
            raise ValueError("calibration dimensions must be positive")
        if isinstance(self.rng_seed, bool) or not 0 <= self.rng_seed < 2**63:
            raise ValueError("rng_seed must be an integer in [0, 2^63)")
        if self.rng_policy_hash != refinement_rng_policy_hash():
            raise ValueError("refinement RNG policy identity mismatch")
        if not self.quantile_search or not self.clip_search_y:
            raise ValueError("GPTQ+Erry requires quantile and output-error clipping")
        if self.weight_method == "rotation" and not self.rotation_config_hash:
            raise ValueError("rotation banks require a pinned rotation config")
        if self.weight_method == "rotation" and not self.rotation_profile_id:
            raise ValueError("rotation banks require a profile identity")
        if self.weight_method != "rotation" and self.rotation_config_hash is not None:
            raise ValueError("non-rotation banks cannot carry rotation identity")
        if self.weight_method != "rotation" and self.rotation_profile_id is not None:
            raise ValueError("non-rotation banks cannot carry a rotation profile")
        _sha256(self.calibration_bundle_hash, "calibration_bundle_hash")
        if self.rotation_config_hash is not None:
            _sha256(self.rotation_config_hash, "rotation_config_hash")
        if not Path(self.calibration_path).is_absolute():
            raise ValueError("calibration_path must be absolute")
        if not Path(self.checkpoint_dir).is_absolute():
            raise ValueError("checkpoint_dir must be absolute")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "weight_format": self.weight_format,
            "weight_method": self.weight_method,
            "block_size": self.block_size,
            "calibration_dataset": self.calibration_dataset,
            "calibration_revision": self.calibration_revision,
            "calibration_bundle_hash": self.calibration_bundle_hash,
            "calibration_path": self.calibration_path,
            "calibration_samples": self.calibration_samples,
            "calibration_sequence_length": self.calibration_sequence_length,
            "calibration_batch_size": self.calibration_batch_size,
            "checkpoint_dir": self.checkpoint_dir,
            "quantile_search": self.quantile_search,
            "clip_search_y": self.clip_search_y,
            "rng_seed": self.rng_seed,
            "rng_policy_hash": self.rng_policy_hash,
            "rotation_config_hash": self.rotation_config_hash,
            "rotation_profile_id": self.rotation_profile_id,
        }

    @property
    def bank_id(self) -> str:
        return f"rwb-{_content_hash(self._content_dict())}"

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"bank_id": self.bank_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementBankSpec":
        spec = cls(
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            weight_format=str(value["weight_format"]),
            weight_method=str(value["weight_method"]),
            block_size=int(value["block_size"]),
            calibration_dataset=str(value["calibration_dataset"]),
            calibration_revision=str(value["calibration_revision"]),
            calibration_bundle_hash=str(value["calibration_bundle_hash"]),
            calibration_path=str(value["calibration_path"]),
            calibration_samples=int(value["calibration_samples"]),
            calibration_sequence_length=int(value["calibration_sequence_length"]),
            calibration_batch_size=int(value["calibration_batch_size"]),
            checkpoint_dir=str(value["checkpoint_dir"]),
            quantile_search=_strict_bool(value["quantile_search"], "quantile_search"),
            clip_search_y=_strict_bool(value["clip_search_y"], "clip_search_y"),
            rng_seed=int(value["rng_seed"]),
            rng_policy_hash=str(value["rng_policy_hash"]),
            rotation_config_hash=(
                None
                if value.get("rotation_config_hash") is None
                else str(value["rotation_config_hash"])
            ),
            rotation_profile_id=(
                None
                if value.get("rotation_profile_id") is None
                else str(value["rotation_profile_id"])
            ),
            schema_version=str(value["schema_version"]),
        )
        if value.get("bank_id") != spec.bank_id:
            raise ValueError("refinement bank identity mismatch")
        return spec


@dataclass(frozen=True)
class RefinementBankHandle:
    """Measured bank identity returned by an execution adapter."""

    bank_id: str
    checkpoint_tree_sha256: str
    weight_identity_before: str

    def __post_init__(self) -> None:
        for label, value in (
            ("bank_id", self.bank_id),
            ("checkpoint_tree_sha256", self.checkpoint_tree_sha256),
            ("weight_identity_before", self.weight_identity_before),
        ):
            if not value:
                raise ValueError(f"{label} is required")


@dataclass(frozen=True)
class RefinementDocumentMetric:
    document_id: str
    source_cluster_id: str
    nll_sum: float
    token_count: int
    initial_cache_length: int
    final_cache_length: int

    def __post_init__(self) -> None:
        if not self.document_id or not self.source_cluster_id:
            raise ValueError("document and source-cluster IDs are required")
        if not math.isfinite(self.nll_sum) or self.nll_sum < 0:
            raise ValueError("document NLL must be finite and non-negative")
        if self.token_count != REFINEMENT_DECODE_STEPS:
            raise ValueError(
                "each refinement document must score "
                f"{REFINEMENT_DECODE_STEPS} tokens"
            )
        if self.initial_cache_length != 512:
            raise ValueError(
                "refinement decode must start from a 512-token prompt cache"
            )
        if self.final_cache_length - self.initial_cache_length != self.token_count:
            raise ValueError(
                "refinement cache growth must equal the scored token count"
            )

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.token_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_cluster_id": self.source_cluster_id,
            "nll_sum": self.nll_sum,
            "token_count": self.token_count,
            "mean_token_nll": self.mean_nll,
            "initial_cache_length": self.initial_cache_length,
            "final_cache_length": self.final_cache_length,
        }


@dataclass(frozen=True)
class RefinementExecutionEvidence:
    """Proof that one result used cached q_len=1 decode and runtime rebinding."""

    prefill_precision: str
    prefill_kv_precision: str
    first_token_owner: str
    q_len_values: tuple[int, ...]
    exact_cache_positions: bool
    independent_batch_caches: bool
    admission_count_per_prompt: int
    direct_native_kv_append: bool
    runtime_rebinding: bool
    weight_requantizations: int
    weight_identity_before: str
    weight_identity_after: str
    checkpoint_tree_sha256: str

    def __post_init__(self) -> None:
        if self.prefill_precision != "BF16" or self.prefill_kv_precision != "BF16":
            raise ValueError(
                "refinement requires a BF16 prefill artifact and BF16 handoff"
            )
        if self.first_token_owner != "prefill":
            raise ValueError("the first generated token must come from prefill")
        if self.q_len_values != (1,):
            raise ValueError("refinement evidence must contain only q_len=1 calls")
        if not self.exact_cache_positions or not self.independent_batch_caches:
            raise ValueError(
                "refinement requires exact positions and independent caches"
            )
        if self.admission_count_per_prompt != 1 or not self.direct_native_kv_append:
            raise ValueError(
                "refinement requires one admission and native decode appends"
            )
        if not self.runtime_rebinding or self.weight_requantizations != 0:
            raise ValueError(
                "refinement profiles must rebind without weight quantization"
            )
        if self.weight_identity_before != self.weight_identity_after:
            raise ValueError("weight identity changed during runtime rebinding")
        if not self.checkpoint_tree_sha256:
            raise ValueError("checkpoint tree identity is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefill_precision": self.prefill_precision,
            "prefill_kv_precision": self.prefill_kv_precision,
            "first_token_owner": self.first_token_owner,
            "q_len_values": list(self.q_len_values),
            "exact_cache_positions": self.exact_cache_positions,
            "independent_batch_caches": self.independent_batch_caches,
            "admission_count_per_prompt": self.admission_count_per_prompt,
            "direct_native_kv_append": self.direct_native_kv_append,
            "runtime_rebinding": self.runtime_rebinding,
            "weight_requantizations": self.weight_requantizations,
            "weight_identity_before": self.weight_identity_before,
            "weight_identity_after": self.weight_identity_after,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
        }


@dataclass(frozen=True)
class RefinementEvaluation:
    documents: tuple[RefinementDocumentMetric, ...]
    evidence: RefinementExecutionEvidence
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        if len(documents) != 128:
            raise ValueError("refinement requires 128 document metrics")
        ids = tuple(document.document_id for document in documents)
        if len(ids) != len(set(ids)):
            raise ValueError("refinement document metrics contain duplicate IDs")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "artifacts", tuple(map(str, self.artifacts)))


class RefinementExecutor(Protocol):
    def open_weight_bank(
        self,
        bank: RefinementBankSpec,
        entries: tuple[RefinementScheduleEntry, ...],
    ) -> AbstractContextManager[RefinementBankHandle]: ...

    def open_split_kv_admission_cache(
        self,
        key_format: str,
        value_format: str,
        samples: RefinementSampleBundle,
    ) -> AbstractContextManager[Any]: ...

    def evaluate(
        self,
        entry: RefinementScheduleEntry,
        *,
        samples: RefinementSampleBundle,
        weight_bank: RefinementBankHandle,
        kv_admission_cache: Any,
    ) -> RefinementEvaluation: ...


def clustered_bootstrap_mean_nll(
    documents: Sequence[RefinementDocumentMetric],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, float | int]:
    """Bootstrap source documents while retaining all windows from each source."""

    clusters: dict[str, list[RefinementDocumentMetric]] = {}
    for document in documents:
        clusters.setdefault(document.source_cluster_id, []).append(document)
    if len(clusters) < 2:
        raise ValueError("clustered bootstrap requires at least two sources")
    if replicates < 100:
        raise ValueError("clustered bootstrap requires at least 100 replicates")
    rng = random.Random(int(seed))
    means = []
    ordered_clusters = tuple(tuple(clusters[key]) for key in sorted(clusters))
    count = len(ordered_clusters)
    for _ in range(replicates):
        selected_clusters = [
            ordered_clusters[rng.randrange(count)] for _ in range(count)
        ]
        selected = [document for cluster in selected_clusters for document in cluster]
        means.append(
            sum(document.nll_sum for document in selected)
            / sum(document.token_count for document in selected)
        )
    observed = sum(document.nll_sum for document in documents) / sum(
        document.token_count for document in documents
    )
    return {
        "method": "clustered_source_document_bootstrap",
        "replicates": replicates,
        "seed": int(seed),
        "source_cluster_count": count,
        "window_count": len(documents),
        "mean_nll": observed,
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def build_refinement_bank_specs(
    schedule: RefinementSchedule,
    *,
    model_name: str,
    model_revision: str,
    calibration_dataset: str,
    calibration_revision: str,
    calibration_bundle_hash: str,
    calibration_path: str | Path,
    checkpoint_root: str | Path,
    calibration_samples: int = 128,
    calibration_sequence_length: int = 2048,
    calibration_batch_size: int = 8,
    rng_seed_base: int = 0,
    rotation_config_hashes: Mapping[str, str] | None = None,
    profile_ids: Sequence[str] | None = None,
) -> dict[tuple[str, str, str], RefinementBankSpec]:
    """Share GPTQ by W while keeping every rotation decision profile-local."""

    rotation_hashes = dict(rotation_config_hashes or {})
    selected_ids = (
        {str(profile_id) for profile_id in profile_ids}
        if profile_ids is not None
        else None
    )
    if selected_ids is not None:
        known_ids = {entry.profile_id for entry in schedule.entries}
        if len(selected_ids) != len(tuple(profile_ids)):
            raise ValueError("refinement bank filter contains duplicate profiles")
        if selected_ids - known_ids:
            raise ValueError("refinement bank filter contains unknown profiles")
    entries_by_key: dict[tuple[str, str, str], RefinementScheduleEntry] = {}
    for entry in schedule.entries:
        if selected_ids is not None and entry.profile_id not in selected_ids:
            continue
        if not entry.gate.executable:
            continue
        entries_by_key.setdefault(refinement_bank_key(entry.profile), entry)
    root = Path(checkpoint_root).resolve()
    calibration = Path(calibration_path).resolve()
    specs = {}
    for key in sorted(entries_by_key):
        weight_format, weight_method, scope = key
        representative = entries_by_key[key].profile
        seed_material = (
            f"{int(rng_seed_base)}:{weight_method}:{weight_format}:{scope}"
        ).encode("utf-8")
        rng_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (
            2**63
        )
        rotation_hash = (
            rotation_hashes.get(
                representative.profile_id,
                rotation_decision_contract_hash(representative),
            )
            if weight_method == "rotation"
            else None
        )
        spec = RefinementBankSpec(
            model_name=model_name,
            model_revision=model_revision,
            weight_format=weight_format,
            weight_method=weight_method,
            block_size=8,
            calibration_dataset=calibration_dataset,
            calibration_revision=calibration_revision,
            calibration_bundle_hash=calibration_bundle_hash,
            calibration_path=str(calibration),
            calibration_samples=calibration_samples,
            calibration_sequence_length=calibration_sequence_length,
            calibration_batch_size=calibration_batch_size,
            rng_seed=rng_seed,
            rng_policy_hash=refinement_rng_policy_hash(),
            checkpoint_dir=str(root / weight_method / weight_format / scope),
            rotation_config_hash=rotation_hash,
            rotation_profile_id=(
                representative.profile_id if weight_method == "rotation" else None
            ),
        )
        specs[key] = spec
    return specs


class _RecordStore:
    def __init__(
        self,
        root: Path,
        schedule: RefinementSchedule,
        samples: RefinementSampleBundle,
        entries: Sequence[RefinementScheduleEntry] | None = None,
    ) -> None:
        self.root = root
        self.schedule = schedule
        self.samples = samples
        self.shards = root / "shards"
        self.completed = root / "completed"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.completed.mkdir(parents=True, exist_ok=True)
        selected = tuple(entries) if entries is not None else schedule.entries
        self.execution_entries = selected
        self.entries = {entry.profile_id: entry for entry in selected}
        if len(self.entries) != len(selected):
            raise ValueError("refinement execution entries contain duplicates")
        self.records: dict[tuple[str, int], dict[str, Any]] = {}
        self._load()

    def _repair_tail(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.lseek(descriptor, 0, os.SEEK_END)
            if size == 0:
                return
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = os.read(descriptor, size)
            if not payload.endswith(b"\n"):
                os.ftruncate(descriptor, payload.rfind(b"\n") + 1)
                os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load(self) -> None:
        for path in sorted(self.shards.glob("*.jsonl")):
            self._repair_tail(path)
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                value = json.loads(line)
                record_hash = value.pop("record_hash", None)
                if record_hash != _content_hash(value):
                    raise ValueError(
                        f"refinement result checksum mismatch at {path}:{line_number}"
                    )
                if value.get("schedule_hash") != self.schedule.canonical_hash:
                    raise ValueError("refinement result schedule mismatch")
                if value.get("sample_bundle_hash") != self.samples.canonical_hash:
                    raise ValueError("refinement result sample-bundle mismatch")
                profile_id = str(value["profile_id"])
                entry = self.entries.get(profile_id)
                if entry is None or value.get("profile") != entry.profile.to_dict():
                    raise ValueError("refinement result profile mismatch")
                if value.get("state") not in {
                    "succeeded",
                    "failed",
                    "oom",
                    "skipped_doomed",
                    "blocked_evidence",
                }:
                    raise ValueError("refinement result state is invalid")
                if value["state"] == "succeeded":
                    artifacts = value.get("artifacts")
                    if not isinstance(artifacts, list) or not artifacts:
                        raise ValueError(
                            "successful refinement result has no artifacts"
                        )
                    for artifact in artifacts:
                        if not isinstance(artifact, Mapping):
                            raise ValueError(
                                "refinement artifact identity is malformed"
                            )
                        current = _artifact_identity(str(artifact.get("path", "")))
                        if current != artifact:
                            raise ValueError(
                                "refinement result artifact identity changed"
                            )
                key = profile_id, int(value["attempt"])
                if key in self.records:
                    raise ValueError("duplicate refinement result attempt")
                self.records[key] = value | {"record_hash": record_hash}

    def latest(self, profile_id: str) -> dict[str, Any] | None:
        values = [
            value
            for (candidate, _), value in self.records.items()
            if candidate == profile_id
        ]
        return max(values, key=lambda value: int(value["attempt"]), default=None)

    def append(
        self,
        entry: RefinementScheduleEntry,
        *,
        attempt: int,
        state: str,
        bank_id: str | None,
        result: Mapping[str, Any] | None = None,
        artifacts: Sequence[Mapping[str, Any]] = (),
        error: BaseException | None = None,
        runtime_seconds: float = 0.0,
    ) -> dict[str, Any]:
        if (entry.profile_id, attempt) in self.records:
            raise ValueError("refinement result attempt already exists")
        body = {
            "schema_version": REFINEMENT_RESULT_SCHEMA,
            "schedule_hash": self.schedule.canonical_hash,
            "sample_bundle_hash": self.samples.canonical_hash,
            "ordinal": entry.ordinal,
            "profile_id": entry.profile_id,
            "profile": entry.profile.to_dict(),
            "gate": entry.gate.to_dict(),
            "bank_id": bank_id,
            "attempt": attempt,
            "state": state,
            "result": result,
            "artifacts": [dict(artifact) for artifact in artifacts],
            "error_class": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "traceback": traceback.format_exc() if error else None,
            "runtime_seconds": float(runtime_seconds),
            "completed_at": _timestamp(),
        }
        record = body | {"record_hash": _content_hash(body)}
        shard = "gates" if bank_id is None else bank_id
        if not _SAFE_TOKEN.fullmatch(shard):
            raise ValueError("unsafe refinement shard identity")
        path = self.shards / f"{shard}.jsonl"
        payload = _canonical_bytes(record)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("incomplete refinement journal append")
                offset += written
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        self.records[(entry.profile_id, attempt)] = record
        return record

    def mark_terminal(
        self, entry: RefinementScheduleEntry, record: Mapping[str, Any]
    ) -> None:
        body = {
            "schema_version": REFINEMENT_COMPLETION_SCHEMA,
            "schedule_hash": self.schedule.canonical_hash,
            "sample_bundle_hash": self.samples.canonical_hash,
            "profile_id": entry.profile_id,
            "ordinal": entry.ordinal,
            "state": record["state"],
            "attempt": int(record["attempt"]),
            "record_hash": record["record_hash"],
        }
        write_immutable_json(self.completed / f"{entry.profile_id}.json", body)

    def reconcile_completions(self, max_attempts: int) -> None:
        for entry in self.execution_entries:
            latest = self.latest(entry.profile_id)
            marker_path = self.completed / f"{entry.profile_id}.json"
            terminal = latest is not None and (
                latest["state"] in {"succeeded", "skipped_doomed", "blocked_evidence"}
                or (
                    latest["state"] in {"failed", "oom"}
                    and int(latest["attempt"]) >= max_attempts
                )
            )
            if marker_path.exists():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                content_hash = marker.pop("content_hash", None)
                if content_hash != _content_hash(marker):
                    raise ValueError("refinement completion checksum mismatch")
                if not terminal:
                    raise ValueError("refinement completion marker is not terminal")
                expected = {
                    "schema_version": REFINEMENT_COMPLETION_SCHEMA,
                    "schedule_hash": self.schedule.canonical_hash,
                    "sample_bundle_hash": self.samples.canonical_hash,
                    "profile_id": entry.profile_id,
                    "ordinal": entry.ordinal,
                    "state": latest["state"],
                    "attempt": int(latest["attempt"]),
                    "record_hash": latest["record_hash"],
                }
                if marker != expected:
                    raise ValueError(
                        "refinement completion marker disagrees with its result"
                    )
            elif terminal:
                self.mark_terminal(entry, latest)

    def require_complete(self, max_attempts: int) -> None:
        """Require checksummed terminal markers for every assigned profile."""

        for entry in self.execution_entries:
            latest = self.latest(entry.profile_id)
            if latest is None:
                raise ValueError("refinement shard has an unstarted profile")
            terminal = latest["state"] in {
                "succeeded",
                "skipped_doomed",
                "blocked_evidence",
            } or (
                latest["state"] in {"failed", "oom"}
                and int(latest["attempt"]) >= max_attempts
            )
            if not terminal:
                raise ValueError("refinement shard has a non-terminal profile")
            marker_path = self.completed / f"{entry.profile_id}.json"
            if not marker_path.is_file():
                raise ValueError("refinement shard lacks a completion marker")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            content_hash = marker.pop("content_hash", None)
            if content_hash != _content_hash(marker):
                raise ValueError("refinement completion checksum mismatch")
            expected = {
                "schema_version": REFINEMENT_COMPLETION_SCHEMA,
                "schedule_hash": self.schedule.canonical_hash,
                "sample_bundle_hash": self.samples.canonical_hash,
                "profile_id": entry.profile_id,
                "ordinal": entry.ordinal,
                "state": latest["state"],
                "attempt": int(latest["attempt"]),
                "record_hash": latest["record_hash"],
            }
            if marker != expected:
                raise ValueError(
                    "refinement completion marker disagrees with its result"
                )


@dataclass(frozen=True)
class RefinementRunSummary:
    succeeded: int
    failed_terminal: int
    oom_terminal: int
    skipped_doomed: int
    blocked_evidence: int
    pending: int
    result_rows: int


@dataclass(frozen=True)
class RefinementMergeSummary:
    master_schedule_hash: str
    sample_bundle_hash: str
    profile_count: int
    result_rows: int
    merged_results_sha256: str
    receipt_path: str


@dataclass(frozen=True)
class RefinementMergedResults:
    """Verified terminal rows from one immutable four-shard merge."""

    receipt: Mapping[str, Any]
    results_path: Path
    results_sha256: str
    terminal_rows: tuple[Mapping[str, Any], ...]


def load_refinement_merged_results(
    schedule: RefinementSchedule,
    merge_receipt: str | Path,
    *,
    results_path: str | Path | None = None,
    verify_result_artifacts: bool = True,
) -> RefinementMergedResults:
    """Fail closed on merge coverage, checksums, terminals, and artifacts."""

    receipt_path = Path(merge_receipt).resolve()
    receipt = load_immutable_json(receipt_path)
    if receipt.get("schema_version") != REFINEMENT_MERGE_SCHEMA:
        raise ValueError("unsupported refinement merge receipt")
    if receipt.get("master_schedule_hash") != schedule.canonical_hash:
        raise ValueError("refinement merge was produced for another schedule")
    expected_ids = tuple(entry.profile_id for entry in schedule.entries)
    if (
        int(receipt.get("profile_count", -1)) != len(expected_ids)
        or tuple(receipt.get("profile_ids", ())) != expected_ids
    ):
        raise ValueError("refinement merge profile coverage is not exact")
    declared = receipt.get("merged_results")
    if not isinstance(declared, Mapping):
        raise ValueError("refinement merge results identity is missing")
    source = Path(
        results_path if results_path is not None else str(declared.get("path", ""))
    ).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("merged refinement results are missing or empty")
    digest = _file_hash(source)
    if (
        source.stat().st_size != int(declared.get("size_bytes", -1))
        or digest != declared.get("sha256")
    ):
        raise ValueError("merged refinement results identity changed")

    entries = {entry.profile_id: entry for entry in schedule.entries}
    attempts: dict[str, dict[int, dict[str, Any]]] = {}
    observed_rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"merged refinement results contain a blank row at {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError("merged refinement result must be an object")
            row = dict(value)
            record_hash = row.pop("record_hash", None)
            if record_hash != _content_hash(row):
                raise ValueError(
                    f"merged refinement result checksum mismatch at row {line_number}"
                )
            if (
                row.get("schema_version") != REFINEMENT_RESULT_SCHEMA
                or row.get("schedule_hash") != schedule.canonical_hash
                or row.get("sample_bundle_hash")
                != receipt.get("sample_bundle_hash")
            ):
                raise ValueError("merged refinement result provenance differs")
            profile_id = str(row.get("profile_id", ""))
            entry = entries.get(profile_id)
            if (
                entry is None
                or int(row.get("ordinal", -1)) != entry.ordinal
                or row.get("profile") != entry.profile.to_dict()
                or row.get("gate") != entry.gate.to_dict()
            ):
                raise ValueError("merged refinement result profile binding differs")
            state = str(row.get("state", ""))
            attempt = int(row.get("attempt", -1))
            if entry.gate.executable:
                if attempt not in {1, 2, 3} or state not in {
                    "succeeded",
                    "failed",
                    "oom",
                }:
                    raise ValueError(
                        "executable refinement result terminal state is invalid"
                    )
            elif (
                attempt != 0
                or state != entry.gate.execution_state
                or state not in {"skipped_doomed", "blocked_evidence"}
            ):
                raise ValueError(
                    "gated refinement result terminal state is invalid"
                )
            profile_attempts = attempts.setdefault(profile_id, {})
            if attempt in profile_attempts:
                raise ValueError("merged refinement result attempt is duplicated")
            restored = {**row, "record_hash": record_hash}
            if restored.get("state") == "succeeded":
                result = restored.get("result")
                mean_nll = (
                    result.get("mean_token_nll")
                    if isinstance(result, Mapping)
                    else None
                )
                if (
                    isinstance(mean_nll, bool)
                    or not isinstance(mean_nll, (int, float))
                    or not math.isfinite(float(mean_nll))
                    or float(mean_nll) < 0
                ):
                    raise ValueError(
                        "successful refinement result has no finite mean NLL"
                    )
                artifacts = restored.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    raise ValueError(
                        "successful refinement result has no sealed artifacts"
                    )
                if verify_result_artifacts:
                    for artifact in artifacts:
                        if (
                            not isinstance(artifact, Mapping)
                            or _artifact_identity(str(artifact.get("path", "")))
                            != artifact
                        ):
                            raise ValueError(
                                "refinement result artifact identity changed"
                            )
            profile_attempts[attempt] = restored
            observed_rows += 1
    if observed_rows != int(receipt.get("result_rows", -1)):
        raise ValueError("refinement merge result-row count differs")
    if set(attempts) != set(expected_ids):
        raise ValueError("refinement merge terminal coverage is incomplete")
    terminal_rows = tuple(
        attempts[profile_id][max(attempts[profile_id])]
        for profile_id in expected_ids
    )
    terminal_receipts = receipt.get("terminal")
    expected_terminal = [
        {
            "profile_id": profile_id,
            "state": row["state"],
            "attempt": int(row["attempt"]),
            "record_hash": row["record_hash"],
        }
        for profile_id, row in zip(expected_ids, terminal_rows)
    ]
    if terminal_receipts != expected_terminal:
        raise ValueError("refinement merge terminal receipt differs from results")
    return RefinementMergedResults(
        receipt=receipt,
        results_path=source,
        results_sha256=digest,
        terminal_rows=terminal_rows,
    )


class RefinementRunner:
    """Execute an immutable schedule with at most three attempts per point."""

    def __init__(
        self,
        *,
        schedule: RefinementSchedule,
        samples: RefinementSampleBundle,
        banks: Mapping[tuple[str, str, str], RefinementBankSpec],
        executor: RefinementExecutor,
        output_dir: str | Path,
        max_attempts: int = 3,
        bootstrap_replicates: int = 2000,
        shard_plan: RefinementShardPlan | None = None,
    ) -> None:
        if max_attempts != 3:
            raise ValueError("the refinement artifact contract requires three attempts")
        if bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        self.schedule = schedule
        self.samples = samples
        self.banks = dict(banks)
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.max_attempts = max_attempts
        self.bootstrap_replicates = bootstrap_replicates
        self.shard_plan = shard_plan
        self.execution_entries = (
            validate_refinement_shard_plan(schedule, shard_plan)
            if shard_plan is not None
            else schedule.entries
        )
        expected_groups = {
            refinement_bank_key(entry.profile)
            for entry in self.execution_entries
            if entry.gate.executable
        }
        if set(self.banks) != expected_groups:
            raise ValueError(
                "refinement bank coverage differs from executable schedule groups"
            )
        for key, bank in self.banks.items():
            expected_key = (
                bank.weight_format,
                bank.weight_method,
                bank.rotation_profile_id or "shared",
            )
            if key != expected_key:
                raise ValueError("refinement bank mapping key is inconsistent")
            calibration_path = Path(bank.calibration_path)
            if not calibration_path.is_file():
                raise ValueError(
                    f"refinement calibration artifact is missing: {calibration_path}"
                )
            if _file_hash(calibration_path) != bank.calibration_bundle_hash:
                raise ValueError("refinement calibration artifact hash mismatch")

    @contextmanager
    def _run_lock(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / ".run.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _result(
        self,
        entry: RefinementScheduleEntry,
        evaluation: RefinementEvaluation,
        handle: RefinementBankHandle,
    ) -> dict[str, Any]:
        expected_ids = tuple(sample.document_id for sample in self.samples.samples)
        actual_ids = tuple(document.document_id for document in evaluation.documents)
        if actual_ids != expected_ids:
            raise ValueError(
                "refinement results differ from the immutable sample order"
            )
        expected_clusters = tuple(
            sample.source_cluster_id for sample in self.samples.samples
        )
        actual_clusters = tuple(
            document.source_cluster_id for document in evaluation.documents
        )
        if actual_clusters != expected_clusters:
            raise ValueError(
                "refinement result clusters differ from the immutable samples"
            )
        if handle.bank_id != self.banks[refinement_bank_key(entry.profile)].bank_id:
            raise ValueError("executor returned the wrong refinement weight bank")
        if evaluation.evidence.weight_identity_before != handle.weight_identity_before:
            raise ValueError("refinement evidence does not match the open weight bank")
        if evaluation.evidence.checkpoint_tree_sha256 != handle.checkpoint_tree_sha256:
            raise ValueError("refinement checkpoint evidence changed during evaluation")
        seed = int(entry.profile.canonical_hash[:16], 16)
        bootstrap = clustered_bootstrap_mean_nll(
            evaluation.documents,
            seed=seed,
            replicates=self.bootstrap_replicates,
        )
        return {
            "nll_sum": sum(document.nll_sum for document in evaluation.documents),
            "token_count": sum(
                document.token_count for document in evaluation.documents
            ),
            "mean_token_nll": bootstrap["mean_nll"],
            "post_handoff_greedy_conditioned_nll": bootstrap["mean_nll"],
            "post_handoff_greedy_conditioned_exp_nll": math.exp(
                float(bootstrap["mean_nll"])
            ),
            "metric_id": "post_handoff_greedy_conditioned_nll/v1",
            "clustered_bootstrap": bootstrap,
            "documents": [document.to_dict() for document in evaluation.documents],
            "execution_evidence": evaluation.evidence.to_dict(),
        }

    def _pending(self, store: _RecordStore) -> tuple[RefinementScheduleEntry, ...]:
        pending = []
        for entry in self.execution_entries:
            latest = store.latest(entry.profile_id)
            if latest is None:
                pending.append(entry)
            elif latest["state"] in {"failed", "oom"} and int(latest["attempt"]) < 3:
                pending.append(entry)
        return tuple(pending)

    def _summary(self, store: _RecordStore) -> RefinementRunSummary:
        states = []
        for entry in self.execution_entries:
            latest = store.latest(entry.profile_id)
            states.append(None if latest is None else latest["state"])
        return RefinementRunSummary(
            succeeded=states.count("succeeded"),
            failed_terminal=sum(
                state == "failed"
                and int(store.latest(entry.profile_id)["attempt"]) >= 3
                for state, entry in zip(states, self.execution_entries)
                if state is not None
            ),
            oom_terminal=sum(
                state == "oom" and int(store.latest(entry.profile_id)["attempt"]) >= 3
                for state, entry in zip(states, self.execution_entries)
                if state is not None
            ),
            skipped_doomed=states.count("skipped_doomed"),
            blocked_evidence=states.count("blocked_evidence"),
            pending=states.count(None)
            + sum(
                state in {"failed", "oom"}
                and int(store.latest(entry.profile_id)["attempt"]) < 3
                for state, entry in zip(states, self.execution_entries)
                if state is not None
            ),
            result_rows=len(store.records),
        )

    def run(self) -> RefinementRunSummary:
        with self._run_lock():
            return self._run_locked()

    def _run_locked(self) -> RefinementRunSummary:
        runtime_environment = getattr(
            self.executor,
            "runtime_environment",
            None,
        )
        if callable(runtime_environment):
            runtime_environment = runtime_environment()
        write_immutable_json(
            self.output_dir / "contract.json",
            {
                "schema_version": "decode-refinement-run",
                "schedule": self.schedule.to_dict(),
                "shard_plan": (
                    self.shard_plan.to_dict() if self.shard_plan is not None else None
                ),
                "execution_profile_ids": [
                    entry.profile_id for entry in self.execution_entries
                ],
                "sample_bundle": self.samples.to_dict(),
                "banks": [self.banks[key].to_dict() for key in sorted(self.banks)],
                "runtime_environment": runtime_environment,
                "max_attempts": self.max_attempts,
                "bootstrap_replicates": self.bootstrap_replicates,
            },
        )
        store = _RecordStore(
            self.output_dir,
            self.schedule,
            self.samples,
            self.execution_entries,
        )
        store.reconcile_completions(self.max_attempts)
        for entry in self.execution_entries:
            if entry.gate.executable or store.latest(entry.profile_id) is not None:
                continue
            record = store.append(
                entry,
                attempt=0,
                state=entry.gate.execution_state,
                bank_id=None,
                result={"gate": entry.gate.to_dict()},
            )
            store.mark_terminal(entry, record)

        while True:
            pending = tuple(
                entry for entry in self._pending(store) if entry.gate.executable
            )
            if not pending:
                break
            groups: dict[tuple[str, str, str], list[RefinementScheduleEntry]] = {}
            for entry in pending:
                groups.setdefault(
                    refinement_bank_key(entry.profile),
                    [],
                ).append(entry)
            for key in sorted(groups):
                entries = tuple(sorted(groups[key], key=lambda value: value.ordinal))
                bank = self.banks[key]
                try:
                    manager = self.executor.open_weight_bank(bank, entries)
                    with manager as handle:
                        if not isinstance(handle, RefinementBankHandle):
                            raise TypeError(
                                "refinement executor must return RefinementBankHandle"
                            )
                        if handle.bank_id != bank.bank_id:
                            raise ValueError(
                                "refinement executor opened an unexpected bank"
                            )
                        pair_groups: dict[
                            tuple[str, str], list[RefinementScheduleEntry]
                        ] = {}
                        for entry in entries:
                            pair_groups.setdefault(
                                (
                                    entry.profile.key_format,
                                    entry.profile.value_format,
                                ),
                                [],
                            ).append(entry)
                        for pair in sorted(pair_groups):
                            with self.executor.open_split_kv_admission_cache(
                                pair[0],
                                pair[1],
                                self.samples,
                            ) as admission:
                                for entry in pair_groups[pair]:
                                    started = time.monotonic()
                                    latest = store.latest(entry.profile_id)
                                    attempt = (
                                        1
                                        if latest is None
                                        else int(latest["attempt"]) + 1
                                    )
                                    try:
                                        evaluation = self.executor.evaluate(
                                            entry,
                                            samples=self.samples,
                                            weight_bank=handle,
                                            kv_admission_cache=admission,
                                        )
                                        result = self._result(
                                            entry,
                                            evaluation,
                                            handle,
                                        )
                                        if not evaluation.artifacts:
                                            raise ValueError(
                                                "refinement evaluation returned no bound artifacts"
                                            )
                                        record = store.append(
                                            entry,
                                            attempt=attempt,
                                            state="succeeded",
                                            bank_id=bank.bank_id,
                                            result=result,
                                            artifacts=tuple(
                                                _artifact_identity(path)
                                                for path in evaluation.artifacts
                                            ),
                                            runtime_seconds=(
                                                time.monotonic() - started
                                            ),
                                        )
                                        store.mark_terminal(entry, record)
                                    except Exception as error:
                                        current = store.latest(entry.profile_id)
                                        if (
                                            current is not None
                                            and int(current["attempt"]) == attempt
                                            and current["state"] == "succeeded"
                                        ):
                                            raise
                                        state = (
                                            "oom"
                                            if _is_out_of_memory(error)
                                            else "failed"
                                        )
                                        record = store.append(
                                            entry,
                                            attempt=attempt,
                                            state=state,
                                            bank_id=bank.bank_id,
                                            error=error,
                                            runtime_seconds=(
                                                time.monotonic() - started
                                            ),
                                        )
                                        if attempt >= self.max_attempts:
                                            store.mark_terminal(entry, record)
                except Exception as error:
                    for entry in entries:
                        latest = store.latest(entry.profile_id)
                        if latest is not None and latest["state"] == "succeeded":
                            continue
                        attempt = 1 if latest is None else int(latest["attempt"]) + 1
                        state = "oom" if _is_out_of_memory(error) else "failed"
                        record = store.append(
                            entry,
                            attempt=attempt,
                            state=state,
                            bank_id=bank.bank_id,
                            error=error,
                        )
                        if attempt >= self.max_attempts:
                            store.mark_terminal(entry, record)
        return self._summary(store)


def _write_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(_canonical_bytes(dict(row)) for row in rows)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace different merged results: {path}"
            )
        return digest
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to replace different merged results: {path}"
                )
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return digest


def merge_refinement_shards(
    *,
    schedule: RefinementSchedule,
    samples: RefinementSampleBundle,
    shard_roots: Sequence[str | Path],
    output_dir: str | Path,
    max_attempts: int = 3,
) -> RefinementMergeSummary:
    """Verify four shard journals and create one immutable coverage artifact."""

    if max_attempts != 3:
        raise ValueError("the refinement merge contract requires three attempts")
    if len(shard_roots) != 4:
        raise ValueError("the refinement merge requires exactly four shard roots")
    expected_plans = build_refinement_shard_plans(schedule)
    observed: dict[
        int,
        tuple[Path, RefinementShardPlan, Mapping[str, Any]],
    ] = {}
    logical_fingerprint: str | None = None
    for raw_root in shard_roots:
        root = Path(raw_root).resolve()
        contract_path = root / "contract.json"
        contract = load_immutable_json(contract_path)
        if contract.get("schema_version") != "decode-refinement-run":
            raise ValueError("unsupported refinement shard-run contract")
        if contract.get("schedule") != schedule.to_dict():
            raise ValueError("refinement shard master schedule mismatch")
        if contract.get("sample_bundle") != samples.to_dict():
            raise ValueError("refinement shard sample bundle mismatch")
        plan_value = contract.get("shard_plan")
        if not isinstance(plan_value, Mapping):
            raise ValueError("refinement shard contract has no shard plan")
        plan = RefinementShardPlan.from_dict(plan_value)
        validate_refinement_shard_plan(schedule, plan)
        if plan != expected_plans[plan.shard_index]:
            raise ValueError("refinement shard plan differs from deterministic plan")
        if plan.shard_index in observed:
            raise ValueError("duplicate refinement shard index")
        if contract.get("execution_profile_ids") != list(plan.profile_ids):
            raise ValueError("refinement shard execution coverage mismatch")
        if int(contract.get("max_attempts", -1)) != max_attempts:
            raise ValueError("refinement shard retry contract mismatch")
        runtime = contract.get("runtime_environment")
        if not isinstance(runtime, Mapping):
            raise ValueError("refinement shard lacks runtime-environment evidence")
        environment = RuntimeEnvironment.from_dict(runtime)
        current_fingerprint = environment.logical_fingerprint
        if logical_fingerprint is None:
            logical_fingerprint = current_fingerprint
        elif current_fingerprint != logical_fingerprint:
            raise ValueError("refinement shards used different numerical runtimes")
        observed[plan.shard_index] = (root, plan, contract)
    if set(observed) != set(range(4)):
        raise ValueError("refinement shard coverage is incomplete")
    device_observations = [
        dict(
            RuntimeEnvironment.from_dict(
                observed[index][2]["runtime_environment"]
            ).observation
        )
        for index in range(4)
    ]

    merged_rows: list[dict[str, Any]] = []
    terminal_rows: dict[str, dict[str, Any]] = {}
    source_artifacts = []
    for shard_index in range(4):
        root, plan, _ = observed[shard_index]
        entries = validate_refinement_shard_plan(schedule, plan)
        store = _RecordStore(root, schedule, samples, entries)
        store.require_complete(max_attempts)
        rows = tuple(
            sorted(
                store.records.values(),
                key=lambda row: (int(row["ordinal"]), int(row["attempt"])),
            )
        )
        merged_rows.extend(dict(row) for row in rows)
        for entry in entries:
            latest = store.latest(entry.profile_id)
            if latest is None:
                raise AssertionError("complete refinement shard has no latest row")
            if entry.profile_id in terminal_rows:
                raise ValueError("refinement profile appears in multiple shards")
            terminal_rows[entry.profile_id] = latest
        result_files = tuple(sorted((root / "shards").glob("*.jsonl")))
        completion_files = tuple(sorted((root / "completed").glob("*.json")))
        source_artifacts.append(
            {
                "shard_index": shard_index,
                "root": str(root),
                "shard_plan_hash": plan.canonical_hash,
                "contract_sha256": _file_hash(root / "contract.json"),
                "result_files": [
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": _file_hash(path),
                    }
                    for path in result_files
                ],
                "completion_files": [
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": _file_hash(path),
                    }
                    for path in completion_files
                ],
            }
        )
    expected_ids = tuple(entry.profile_id for entry in schedule.entries)
    if set(terminal_rows) != set(expected_ids):
        raise ValueError("merged refinement coverage differs from the master schedule")
    merged_rows.sort(key=lambda row: (int(row["ordinal"]), int(row["attempt"])))
    destination = Path(output_dir).resolve()
    results_path = destination / "results.jsonl"
    merged_hash = _write_immutable_jsonl(results_path, merged_rows)
    receipt_path = destination / "merge.json"
    write_immutable_json(
        receipt_path,
        {
            "schema_version": REFINEMENT_MERGE_SCHEMA,
            "master_schedule_hash": schedule.canonical_hash,
            "sample_bundle_hash": samples.canonical_hash,
            "logical_runtime_fingerprint": logical_fingerprint,
            "device_observations": device_observations,
            "profile_count": len(expected_ids),
            "profile_ids": list(expected_ids),
            "terminal": [
                {
                    "profile_id": profile_id,
                    "state": terminal_rows[profile_id]["state"],
                    "attempt": int(terminal_rows[profile_id]["attempt"]),
                    "record_hash": terminal_rows[profile_id]["record_hash"],
                }
                for profile_id in expected_ids
            ],
            "result_rows": len(merged_rows),
            "merged_results": {
                "path": str(results_path),
                "size_bytes": results_path.stat().st_size,
                "sha256": merged_hash,
            },
            "sources": source_artifacts,
        },
    )
    return RefinementMergeSummary(
        master_schedule_hash=schedule.canonical_hash,
        sample_bundle_hash=samples.canonical_hash,
        profile_count=len(expected_ids),
        result_rows=len(merged_rows),
        merged_results_sha256=merged_hash,
        receipt_path=str(receipt_path),
    )


def seal_checkpoint_identity(
    bank: RefinementBankSpec,
    checkpoint_dir: str | Path,
    weight_identity: str,
) -> RefinementBankHandle:
    """Bind an opened bank to the exact files produced or resumed."""

    path = Path(checkpoint_dir).resolve()
    if path != Path(bank.checkpoint_dir).resolve():
        raise ValueError("checkpoint path differs from the bank contract")
    return RefinementBankHandle(
        bank_id=bank.bank_id,
        checkpoint_tree_sha256=_tree_hash(path),
        weight_identity_before=weight_identity,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain a JSON object")
    return value


def _refinement_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("refinement")
    if not isinstance(value, Mapping):
        raise ValueError("config.refinement is required")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--sample-bundle", required=True)
    parser.add_argument("--prefill-root", required=True)
    parser.add_argument("--admission-root", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--calibration-receipt", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-root")
    parser.add_argument("--shard-plan")
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--decode-microbatch-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    config = _load_config(args.config)
    refinement = _refinement_config(config)
    schedule = load_refinement_schedule(args.schedule)
    shard_plan = (
        load_refinement_shard_plan(args.shard_plan) if args.shard_plan else None
    )
    execution_entries = (
        validate_refinement_shard_plan(schedule, shard_plan)
        if shard_plan is not None
        else schedule.entries
    )
    samples = load_refinement_sample_bundle(args.sample_bundle)
    receipt = load_immutable_json(args.calibration_receipt)
    if receipt.get("schema_version") != "decode-gptq-calibration":
        raise ValueError("unsupported GPTQ calibration receipt")
    calibration = Path(args.calibration).resolve()
    if str(calibration) != receipt.get("calibration_path"):
        raise ValueError("GPTQ calibration path differs from its receipt")
    calibration_hash = _file_sha256(calibration)
    if calibration_hash != receipt.get("calibration_sha256"):
        raise ValueError("GPTQ calibration hash differs from its receipt")
    if receipt.get("model_revision") != str(config["model_revision"]):
        raise ValueError("GPTQ calibration model revision mismatch")
    if receipt.get("tokenizer_revision") != str(config["tokenizer_revision"]):
        raise ValueError("GPTQ calibration tokenizer revision mismatch")
    selection = receipt.get("selection")
    if (
        receipt.get("selection_policy") != "document_round_robin_nonoverlap/v1"
        or not isinstance(selection, list)
        or len(selection) != 128
    ):
        raise ValueError("GPTQ calibration selection receipt is incomplete")
    selection_hash = hashlib.sha256(
        json.dumps(
            selection,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if selection_hash != receipt.get("selection_sha256"):
        raise ValueError("GPTQ calibration selection receipt hash mismatch")
    configured_seed = int(refinement.get("calibration_seed", 0))
    if receipt.get("selection_seed") != configured_seed:
        raise ValueError("GPTQ calibration selection seed mismatch")
    banks = build_refinement_bank_specs(
        schedule,
        model_name=str(config["model_name"]),
        model_revision=str(config["model_revision"]),
        calibration_dataset=str(receipt["dataset_name"]),
        calibration_revision=str(receipt["dataset_revision"]),
        calibration_bundle_hash=calibration_hash,
        calibration_path=calibration,
        checkpoint_root=args.checkpoint_root,
        calibration_samples=int(receipt["sample_count"]),
        calibration_sequence_length=int(receipt["sequence_length"]),
        calibration_batch_size=int(refinement.get("calibration_batch_size", 8)),
        rng_seed_base=configured_seed,
        profile_ids=tuple(entry.profile_id for entry in execution_entries),
    )
    work_root = (
        Path(args.work_root).resolve()
        if args.work_root
        else Path(args.output_dir).resolve() / "work"
    )
    # Imported here: refinement_evaluator imports this module's bank types.
    from decode_dse.software.refinement_evaluator import RefinementEvaluator

    adapter = RefinementEvaluator(
        config=config,
        sample_bundle_path=args.sample_bundle,
        prefill_root=args.prefill_root,
        admission_root=args.admission_root,
        workspace_root=work_root,
        device_label=args.device_label,
        decode_microbatch_size=args.decode_microbatch_size,
        max_cpu_cache_gib=float(refinement.get("max_cpu_cache_gib", 24)),
    )
    summary = RefinementRunner(
        schedule=schedule,
        samples=samples,
        banks=banks,
        executor=adapter,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        shard_plan=shard_plan,
    ).run()
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return (
        0
        if summary.pending == 0
        and summary.failed_terminal == 0
        and summary.oom_terminal == 0
        else 2
    )


_GPU_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def parse_gpu_pool(value: str) -> tuple[str, ...]:
    devices = tuple(token.strip() for token in value.split(",") if token.strip())
    if not 1 <= len(devices) <= 4 or len(set(devices)) != len(devices):
        raise ValueError(
            "refinement launch requires one to four unique GPU identifiers"
        )
    if any(not _GPU_TOKEN.fullmatch(device) for device in devices):
        raise ValueError("GPU identifiers contain unsupported characters")
    return devices


def refinement_worker_command(
    *,
    config: Path,
    schedule: Path,
    shard_plan: Path,
    sample_bundle: Path,
    prefill_root: Path,
    admission_root: Path,
    calibration: Path,
    calibration_receipt: Path,
    checkpoint_root: Path,
    output_dir: Path,
    work_root: Path,
    device_label: str,
    decode_microbatch_size: int,
    bootstrap_replicates: int,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "decode_dse.software.refinement_runner",
        "--config",
        str(config),
        "--schedule",
        str(schedule),
        "--shard-plan",
        str(shard_plan),
        "--sample-bundle",
        str(sample_bundle),
        "--prefill-root",
        str(prefill_root),
        "--admission-root",
        str(admission_root),
        "--calibration",
        str(calibration),
        "--calibration-receipt",
        str(calibration_receipt),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-dir",
        str(output_dir),
        "--work-root",
        str(work_root),
        "--device-label",
        device_label,
        "--decode-microbatch-size",
        str(decode_microbatch_size),
        "--bootstrap-replicates",
        str(bootstrap_replicates),
    )


def _shard_roots(output_root: Path) -> tuple[Path, ...]:
    return tuple(output_root / "shards" / f"shard-{index:02d}" for index in range(4))


def _refinement_launch_waves(
    devices: Sequence[str],
) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Map four immutable source shards onto a bounded physical GPU pool."""

    pool = tuple(devices)
    if not 1 <= len(pool) <= 4 or len(set(pool)) != len(pool):
        raise ValueError(
            "refinement execution requires one to four unique GPUs"
        )
    assignments = tuple(
        (shard_index, pool[shard_index % len(pool)])
        for shard_index in range(4)
    )
    return tuple(
        assignments[offset : offset + len(pool)]
        for offset in range(0, len(assignments), len(pool))
    )


def _require_isolated_roots(roots: Sequence[Path]) -> None:
    resolved = tuple(path.resolve() for path in roots)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("refinement mutable roots must not overlap")


def launch_refinement(
    *,
    config: Path,
    schedule_path: Path,
    sample_bundle_path: Path,
    prefill_root: Path,
    admission_root: Path,
    calibration: Path,
    calibration_receipt: Path,
    checkpoint_root: Path,
    output_root: Path,
    work_root: Path,
    device_label: str,
    devices: Sequence[str],
    decode_microbatch_size: int = 8,
    bootstrap_replicates: int = 2000,
) -> int:
    waves = _refinement_launch_waves(devices)
    _require_isolated_roots((admission_root, checkpoint_root, output_root, work_root))
    schedule = load_refinement_schedule(schedule_path)
    plans = build_refinement_shard_plans(schedule)
    shard_roots = _shard_roots(output_root)
    plan_root = output_root / "plans"
    plan_paths = []
    for plan in plans:
        path = plan_root / f"shard-{plan.shard_index:02d}.json"
        write_refinement_shard_plan(path, plan)
        plan_paths.append(path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_root = output_root / "logs" / f"refinement-{timestamp}"
    log_root.mkdir(parents=True, exist_ok=False)
    workers: list[dict[str, Any] | None] = [None] * 4
    return_codes: list[int | None] = [None] * 4
    for wave_index, wave in enumerate(waves):
        processes: list[
            tuple[int, subprocess.Popen[bytes], object]
        ] = []
        try:
            for worker_slot, (index, device) in enumerate(wave):
                shard_name = f"shard-{index:02d}"
                command = refinement_worker_command(
                    config=config,
                    schedule=schedule_path,
                    shard_plan=plan_paths[index],
                    sample_bundle=sample_bundle_path,
                    prefill_root=prefill_root,
                    admission_root=admission_root / shard_name,
                    calibration=calibration,
                    calibration_receipt=calibration_receipt,
                    checkpoint_root=checkpoint_root / shard_name,
                    output_dir=shard_roots[index],
                    work_root=work_root / shard_name,
                    device_label=device_label,
                    decode_microbatch_size=decode_microbatch_size,
                    bootstrap_replicates=bootstrap_replicates,
                )
                log_path = log_root / f"{shard_name}.log"
                handle = log_path.open("wb")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(device)
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
                processes.append((index, process, handle))
                workers[index] = {
                    "shard_index": index,
                    "source_profile_id": plans[index].source_profile_id,
                    "shard_plan_hash": plans[index].canonical_hash,
                    "wave_index": wave_index,
                    "worker_slot": worker_slot,
                    "cuda_visible_devices": str(device),
                    "command": list(command),
                    "log": str(log_path.resolve()),
                }
            for index, process, _ in processes:
                return_codes[index] = process.wait()
        except BaseException:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process, _ in processes:
                process.wait()
            raise
        finally:
            for _, _, handle in processes:
                handle.close()

    if any(worker is None for worker in workers) or any(
        return_code is None for return_code in return_codes
    ):
        raise AssertionError("refinement launch did not cover every source shard")

    samples = load_refinement_sample_bundle(sample_bundle_path)
    merge = merge_refinement_shards(
        schedule=schedule,
        samples=samples,
        shard_roots=shard_roots,
        output_dir=output_root / "merged",
    )
    write_immutable_json(
        log_root / "summary.json",
        {
            "schema_version": "decode-refinement-launch",
            "master_schedule_hash": schedule.canonical_hash,
            "sample_bundle_hash": samples.canonical_hash,
            "workers": [
                worker | {"return_code": return_codes[index]}
                for index, worker in enumerate(workers)
                if worker is not None
            ],
            "merge": merge.__dict__,
        },
    )
    if all(return_code == 0 for return_code in return_codes):
        print(json.dumps(merge.__dict__, indent=2, sort_keys=True))
        return 0
    failed = [index for index, return_code in enumerate(return_codes) if return_code]
    print(
        f"refinement produced complete terminal coverage with failed shards {failed}",
        file=sys.stderr,
    )
    return 2


def merge_existing_refinement(
    *,
    schedule_path: Path,
    sample_bundle_path: Path,
    output_root: Path,
) -> int:
    summary = merge_refinement_shards(
        schedule=load_refinement_schedule(schedule_path),
        samples=load_refinement_sample_bundle(sample_bundle_path),
        shard_roots=_shard_roots(output_root),
        output_dir=output_root / "merged",
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--schedule", type=Path, required=True)
    launch.add_argument("--sample-bundle", type=Path, required=True)
    launch.add_argument("--prefill-root", type=Path, required=True)
    launch.add_argument("--admission-root", type=Path, required=True)
    launch.add_argument("--calibration", type=Path, required=True)
    launch.add_argument("--calibration-receipt", type=Path, required=True)
    launch.add_argument("--checkpoint-root", type=Path, required=True)
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--work-root", type=Path, required=True)
    launch.add_argument("--device-label", required=True)
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--decode-microbatch-size", type=int, default=8)
    launch.add_argument("--bootstrap-replicates", type=int, default=2000)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--schedule", type=Path, required=True)
    merge.add_argument("--sample-bundle", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    return parser


def launch_main(argv: Iterable[str] | None = None) -> int:
    parser = launch_parser()
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.command == "merge":
            return merge_existing_refinement(
                schedule_path=args.schedule.resolve(),
                sample_bundle_path=args.sample_bundle.resolve(),
                output_root=args.output_root.resolve(),
            )
        return launch_refinement(
            config=args.config.resolve(),
            schedule_path=args.schedule.resolve(),
            sample_bundle_path=args.sample_bundle.resolve(),
            prefill_root=args.prefill_root.resolve(),
            admission_root=args.admission_root.resolve(),
            calibration=args.calibration.resolve(),
            calibration_receipt=args.calibration_receipt.resolve(),
            checkpoint_root=args.checkpoint_root.resolve(),
            output_root=args.output_root.resolve(),
            work_root=args.work_root.resolve(),
            device_label=args.device_label,
            devices=parse_gpu_pool(args.gpus),
            decode_microbatch_size=args.decode_microbatch_size,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


__all__ = [
    "REFINEMENT_BANK_SCHEMA",
    "REFINEMENT_COMPLETION_SCHEMA",
    "REFINEMENT_MERGE_SCHEMA",
    "REFINEMENT_RESULT_SCHEMA",
    "RefinementBankHandle",
    "RefinementBankSpec",
    "RefinementDocumentMetric",
    "RefinementEvaluation",
    "RefinementExecutionEvidence",
    "RefinementMergedResults",
    "RefinementRunSummary",
    "RefinementMergeSummary",
    "RefinementRunner",
    "build_refinement_bank_specs",
    "clustered_bootstrap_mean_nll",
    "merge_refinement_shards",
    "load_refinement_merged_results",
    "rotation_policy",
    "rotation_policy_hash",
    "rotation_decision_contract",
    "rotation_decision_contract_hash",
    "refinement_bank_key",
    "refinement_rng_policy",
    "refinement_rng_policy_hash",
    "seal_checkpoint_identity",
]


REFINEMENT_PREFILL_INDEX_SCHEMA = "decode-refinement-prefill-index"


GPTQ_CALIBRATION_SCHEMA = "decode-gptq-calibration"


GPTQ_SELECTION_POLICY = "document_round_robin_nonoverlap/v1"


def _load_token_documents(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
) -> tuple[tuple[int, str, tuple[int, ...]], ...]:
    from decode_dse.software.sweep import _group_wikitext_documents

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GPTQ calibration preparation requires datasets and transformers"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        revision=str(config["tokenizer_revision"]),
        cache_dir=config.get("hf_cache_dir"),
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    dataset = load_dataset(
        str(data["dataset_name"]),
        data.get("dataset_config"),
        split=str(data.get("split", "train")),
        revision=str(data["dataset_revision"]),
        cache_dir=data.get("cache_dir"),
    )
    text_column = str(data.get("text_column", "text"))
    if text_column not in dataset.column_names:
        raise ValueError(f"dataset lacks text column {text_column!r}")
    grouped = _group_wikitext_documents(
        dataset[text_column],
        split=str(data.get("split", "train")),
        separator=str(data.get("document_separator", "\n\n")),
    )
    documents = []
    for document_index, (_, text) in enumerate(grouped):
        token_ids = tuple(
            int(token)
            for token in tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        if token_ids:
            documents.append(
                (
                    document_index,
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    token_ids,
                )
            )
    return tuple(documents)


def _load_heldout_documents(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
) -> tuple[TokenizedSourceDocument, ...]:
    from decode_dse.software.sweep import _group_wikitext_documents

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "refinement sample preparation requires datasets and transformers"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        revision=str(config["tokenizer_revision"]),
        cache_dir=config.get("hf_cache_dir"),
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    split = str(data.get("split", "test"))
    dataset = load_dataset(
        str(data["dataset_name"]),
        data.get("dataset_config"),
        split=split,
        revision=str(data["dataset_revision"]),
        cache_dir=data.get("cache_dir"),
    )
    text_column = str(data.get("text_column", "text"))
    if text_column not in dataset.column_names:
        raise ValueError(f"dataset lacks text column {text_column!r}")
    documents = []
    for document_id, text in _group_wikitext_documents(
        dataset[text_column],
        split=split,
        separator=str(data.get("document_separator", "\n\n")),
    ):
        tokens = tuple(
            int(token)
            for token in tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        if tokens:
            documents.append(
                TokenizedSourceDocument(
                    document_id=document_id,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    token_ids=tokens,
                )
            )
    return tuple(documents)


def select_document_calibration_windows(
    documents: Iterable[tuple[int, str, tuple[int, ...]]],
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], dict[str, Any]], ...]:
    """Select non-overlapping windows while maximizing document coverage."""

    by_document = []
    for document_index, document_hash, token_ids in documents:
        windows = []
        for window_index, offset in enumerate(
            range(0, len(token_ids) - seqlen + 1, seqlen)
        ):
            window = tuple(token_ids[offset : offset + seqlen])
            rank = hashlib.sha256(
                f"{seed}:{document_hash}:{window_index}".encode("utf-8")
            ).hexdigest()
            windows.append((rank, window_index, offset, window))
        if windows:
            by_document.append(
                (
                    document_index,
                    document_hash,
                    tuple(sorted(windows)),
                )
            )
    if not by_document:
        raise ValueError("calibration dataset has no complete token windows")
    selected = []
    round_index = 0
    while len(selected) < nsamples:
        available = [item for item in by_document if round_index < len(item[2])]
        if not available:
            break
        available.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{round_index}:{item[1]}:{item[0]}".encode("utf-8")
            ).hexdigest()
        )
        for document_index, document_hash, windows in available:
            if len(selected) == nsamples:
                break
            _, window_index, offset, window = windows[round_index]
            window_hash = hashlib.sha256(
                json.dumps(
                    window,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            selected.append(
                (
                    window,
                    {
                        "ordinal": len(selected),
                        "document_index": document_index,
                        "document_hash": document_hash,
                        "window_index": window_index,
                        "token_offset": offset,
                        "token_count": seqlen,
                        "window_hash": window_hash,
                    },
                )
            )
        round_index += 1
    if len(selected) != nsamples:
        raise ValueError(
            f"calibration dataset provides {len(selected)} of {nsamples} windows"
        )
    return tuple(selected)


def build_refinement_sample_file(
    *,
    config: Mapping[str, Any],
    source_bundle_path: str | Path,
    output_path: str | Path,
) -> RefinementSampleBundle:
    """Create samples disjoint from sealed screen and validation prompts."""

    from decode_dse.software.sweep import _dataset_config

    source = load_sample_bundle(source_bundle_path)
    if source.model_revision != str(config["model_revision"]):
        raise ValueError(
            "source-bundle model revision differs from refinement configuration"
        )
    if source.tokenizer_revision != str(config["tokenizer_revision"]):
        raise ValueError(
            "source-bundle tokenizer revision differs from refinement configuration"
        )
    data = _dataset_config(config)
    documents = _load_heldout_documents(config, data)
    refinement = config.get("refinement")
    if not isinstance(refinement, Mapping):
        raise ValueError("config.refinement is required")
    bundle = build_refinement_bundle_from_documents(
        documents,
        model_revision=source.model_revision,
        tokenizer_revision=source.tokenizer_revision,
        dataset_name=str(data["dataset_name"]),
        dataset_revision=str(data["dataset_revision"]),
        excluded_source_spans=tuple(
            span
            for sample in source.numerical_screen + source.hardware_validation
            for span in sample.source_spans
        ),
        excluded_prompt_hashes=tuple(
            sample.prompt_hash
            for sample in source.numerical_screen + source.hardware_validation
        ),
        selection_seed=int(
            refinement.get("sample_selection_seed", config.get("seed", 0))
        ),
    )
    save_refinement_sample_bundle(bundle, output_path)
    return bundle


def prepare_refinement_prefill_artifacts(
    *,
    config: Mapping[str, Any],
    bundle_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Capture one immutable BF16 prefill artifact per refinement prompt."""

    from decode_dse.software.sweep import (
        _artifact_directory,
        _repository_hash,
        _verify_prefill,
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "refinement prefill preparation requires torch and transformers"
        ) from exc
    bundle = load_refinement_sample_bundle(bundle_path)
    if bundle.model_revision != str(config["model_revision"]):
        raise ValueError("refinement sample model revision differs from config")
    if bundle.tokenizer_revision != str(config["tokenizer_revision"]):
        raise ValueError("refinement sample tokenizer revision differs from config")
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    device = str(config.get("device", "cuda:0"))
    seed = int(config.get("seed", 0))
    initialize_numerical_runtime(seed)
    runtime_environment = capture_runtime_environment(device, seed=seed)
    preparation_device = {
        **dict(runtime_environment.observation),
        "device_name": runtime_environment.logical["device_name"],
        "compute_capability": runtime_environment.logical["compute_capability"],
    }
    preparation_metadata = {
        "preparation_device_index": str(preparation_device["device_index"]),
        "preparation_device_uuid": str(preparation_device["device_uuid"]),
        "preparation_total_memory_bytes": str(preparation_device["total_memory_bytes"]),
        "preparation_device_name": str(preparation_device["device_name"]),
        "preparation_compute_capability": str(preparation_device["compute_capability"]),
    }
    code_revision = _repository_hash()
    provenance = ArtifactProvenance(
        producer="packedkv-refinement-bf16-prefill",
        code_revision=code_revision,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        parameters=(
            ("model_revision", bundle.model_revision),
            ("sample_bundle_hash", bundle.canonical_hash),
            ("tokenizer_revision", bundle.tokenizer_revision),
            (
                "runtime_environment_fingerprint",
                runtime_environment.logical_fingerprint,
            ),
        ),
    )
    artifact_ids: dict[str, str] = {}
    pending = []
    for sample in bundle.samples:
        path = _artifact_directory(root, sample.document_id)
        if path.exists():
            artifact = load_prefill_artifact(path)
            _verify_prefill(
                artifact,
                sample=sample,
                bundle=bundle,
                provenance=provenance,
                runtime_environment=runtime_environment,
                model_architecture=config["model_architecture"],
            )
            artifact_ids[sample.document_id] = artifact.artifact_id
        else:
            pending.append(sample)

    model = None
    try:
        if pending:
            if str(config.get("dtype", "bfloat16")).lower() != "bfloat16":
                raise ValueError("refinement prefill capture requires bfloat16")
            model = AutoModelForCausalLM.from_pretrained(
                str(config["model_name"]),
                revision=bundle.model_revision,
                torch_dtype=torch.bfloat16,
                cache_dir=config.get("hf_cache_dir"),
                local_files_only=bool(config.get("local_files_only", True)),
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            model = model.to(device).eval()
            for sample in pending:
                input_ids = torch.tensor(
                    [sample.prompt_token_ids],
                    dtype=torch.long,
                    device=device,
                )
                artifact = capture_bf16_prefill(
                    model,
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    model_revision=bundle.model_revision,
                    tokenizer_revision=bundle.tokenizer_revision,
                    provenance=provenance,
                    metadata={
                        "document_id": sample.document_id,
                        "sample_bundle_hash": bundle.canonical_hash,
                        **preparation_metadata,
                    },
                )
                _verify_prefill(
                    artifact,
                    sample=sample,
                    bundle=bundle,
                    provenance=provenance,
                    runtime_environment=runtime_environment,
                    model_architecture=config["model_architecture"],
                )
                save_prefill_artifact(
                    artifact,
                    _artifact_directory(root, sample.document_id),
                )
                artifact_ids[sample.document_id] = artifact.artifact_id
    finally:
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = [
        {
            "document_id": sample.document_id,
            "prompt_hash": sample.prompt_hash,
            "artifact_id": artifact_ids[sample.document_id],
            "relative_path": _artifact_directory(
                Path("."), sample.document_id
            ).as_posix(),
        }
        for sample in bundle.samples
    ]
    value = {
        "schema_version": REFINEMENT_PREFILL_INDEX_SCHEMA,
        "model_revision": bundle.model_revision,
        "tokenizer_revision": bundle.tokenizer_revision,
        "sample_bundle_hash": bundle.canonical_hash,
        "code_revision": code_revision,
        "runtime_environment": runtime_environment.to_dict(),
        "records": records,
    }
    write_immutable_json(root / "index.json", value)
    return value


def prepare_gptq_calibration(
    *,
    config: Mapping[str, Any],
    output_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Seal a pinned file-backed loader consumed identically by every W bank."""

    from decode_dse.software.sweep import _dataset_config

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPTQ calibration preparation requires torch") from exc
    refinement = config.get("refinement")
    if not isinstance(refinement, Mapping):
        raise ValueError("config.refinement is required")
    data = refinement.get("calibration_data")
    if not isinstance(data, Mapping):
        raise ValueError("config.refinement.calibration_data is required")
    required = ("dataset_name", "dataset_revision")
    if any(not data.get(key) for key in required):
        raise ValueError("calibration dataset name and revision must be pinned")
    nsamples = int(refinement.get("calibration_samples", 128))
    seqlen = int(refinement.get("calibration_sequence_length", 2048))
    seed = int(refinement.get("calibration_seed", 0))
    if isinstance(refinement.get("calibration_seed", 0), bool) or not 0 <= seed < 2**63:
        raise ValueError("calibration_seed must be an integer in [0, 2^63)")
    if nsamples != 128 or seqlen != 2048:
        raise ValueError(
            "refinement GPTQ calibration is fixed at 128 samples by 2,048 tokens"
        )
    heldout = _dataset_config(config)
    calibration_identity = (
        str(data["dataset_name"]),
        str(data.get("dataset_config", "")),
        str(data["dataset_revision"]),
        str(data.get("split", "train")),
    )
    heldout_identity = (
        str(heldout["dataset_name"]),
        str(heldout.get("dataset_config", "")),
        str(heldout["dataset_revision"]),
        str(heldout.get("split", "validation")),
    )
    if calibration_identity == heldout_identity:
        raise ValueError("GPTQ calibration must not reuse the held-out selection split")
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_metadata = {
        "seqlen": seqlen,
        "nsamples": nsamples,
        "dataset_name": str(data["dataset_name"]),
        "dataset_config": str(data.get("dataset_config", "")),
        "dataset_revision": str(data["dataset_revision"]),
        "dataset_split": str(data.get("split", "train")),
        "heldout_dataset_name": heldout_identity[0],
        "heldout_dataset_config": heldout_identity[1],
        "heldout_dataset_revision": heldout_identity[2],
        "heldout_dataset_split": heldout_identity[3],
        "heldout_split_disjoint": True,
        "model_revision": str(config["model_revision"]),
        "tokenizer_revision": str(config["tokenizer_revision"]),
        "selection_policy": GPTQ_SELECTION_POLICY,
        "selection_seed": seed,
    }
    documents = _load_token_documents(config, data)
    selected = select_document_calibration_windows(
        documents,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
    )
    selection = [record for _, record in selected]
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if (
            not isinstance(existing, Mapping)
            or not isinstance(existing.get("loader"), list)
            or len(existing["loader"]) != nsamples
            or any(
                existing.get(key) != value for key, value in expected_metadata.items()
            )
            or existing.get("selection") != selection
        ):
            raise FileExistsError(
                f"existing calibration artifact differs: {destination}"
            )
        for (expected_tokens, _), item in zip(selected, existing["loader"]):
            if (
                not isinstance(item, (tuple, list))
                or len(item) != 2
                or tuple(int(token) for token in item[0][0].tolist()) != expected_tokens
            ):
                raise FileExistsError(
                    f"existing calibration tokens differ: {destination}"
                )
    else:
        loader = []
        for window, _ in selected:
            input_ids = torch.tensor(
                [window],
                dtype=torch.long,
            )
            target = input_ids.clone()
            target[:, :-1] = -100
            loader.append((input_ids, target))
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
            torch.save(
                {
                    "loader": loader,
                    "selection": selection,
                    **expected_metadata,
                },
                temporary_name,
            )
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
    receipt = {
        "schema_version": GPTQ_CALIBRATION_SCHEMA,
        "model_revision": str(config["model_revision"]),
        "tokenizer_revision": str(config["tokenizer_revision"]),
        "dataset_name": str(data["dataset_name"]),
        "dataset_config": str(data.get("dataset_config", "")),
        "dataset_revision": str(data["dataset_revision"]),
        "split": str(data.get("split", "train")),
        "heldout_dataset_name": heldout_identity[0],
        "heldout_dataset_config": heldout_identity[1],
        "heldout_dataset_revision": heldout_identity[2],
        "heldout_dataset_split": heldout_identity[3],
        "heldout_split_disjoint": True,
        "sample_count": nsamples,
        "sequence_length": seqlen,
        "selection_policy": GPTQ_SELECTION_POLICY,
        "selection_seed": seed,
        "selection": selection,
        "selection_sha256": hashlib.sha256(
            json.dumps(
                selection,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "calibration_path": str(destination),
        "calibration_sha256": _file_sha256(destination),
    }
    write_immutable_json(receipt_path, receipt)
    return receipt


def _refinement_inputs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser("samples")
    samples.add_argument("--config", required=True)
    samples.add_argument("--source-bundle", required=True)
    samples.add_argument("--output", required=True)
    prefill = commands.add_parser("prefill")
    prefill.add_argument("--config", required=True)
    prefill.add_argument("--sample-bundle", required=True)
    prefill.add_argument("--artifact-root", required=True)
    calibration = commands.add_parser("calibration")
    calibration.add_argument("--config", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--receipt", required=True)
    return parser


def refinement_inputs_main(argv: Iterable[str] | None = None) -> int:
    args = _refinement_inputs_parser().parse_args(
        tuple(argv) if argv is not None else None
    )
    config = _load_config(args.config)
    if args.command == "samples":
        result: Any = build_refinement_sample_file(
            config=config,
            source_bundle_path=args.source_bundle,
            output_path=args.output,
        ).to_dict()
    elif args.command == "prefill":
        result = prepare_refinement_prefill_artifacts(
            config=config,
            bundle_path=args.sample_bundle,
            artifact_root=args.artifact_root,
        )
    else:
        result = prepare_gptq_calibration(
            config=config,
            output_path=args.output,
            receipt_path=args.receipt,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


_refinement_inputs_all__ = [
    "GPTQ_CALIBRATION_SCHEMA",
    "GPTQ_SELECTION_POLICY",
    "REFINEMENT_PREFILL_INDEX_SCHEMA",
    "build_refinement_sample_file",
    "refinement_inputs_main",
    "prepare_gptq_calibration",
    "prepare_refinement_prefill_artifacts",
    "select_document_calibration_windows",
]


def dispatch(argv: Sequence[str] | None = None) -> int:
    """Route to one of this module's commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"launch", "merge"}:
        return launch_main(arguments)
    commands = {
        "prepare": refinement_inputs_main,
        "run": main,
    }
    if not arguments or arguments[0] not in commands:
        raise SystemExit(
            "usage: <command> [options]; commands: launch, merge, prepare, run"
        )
    return commands[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(dispatch())
