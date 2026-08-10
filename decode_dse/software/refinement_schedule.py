"""Deterministic precision refinement with independent K/V formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.hardware.design_space import (
    HARDWARE_STORAGE_REVISION,
    LEGACY_COMPACT_STORAGE_REVISION,
    load_hardware_artifact,
)
from decode_dse.hardware.evaluation import load_terminal_numerical_rows
from decode_dse.hardware.selection import (
    EpsilonPolicy,
    ParetoPoint,
    select_refinement_sources,
)
from decode_dse.legality import StackValidity
from decode_dse.manifest import SweepManifest, SweepManifestEntry, load_manifest
from decode_dse.profiles import (
    DECODE_FORMATS,
    MX_BLOCK_SIZE,
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    DecodePrecisionProfile,
    format_descriptor,
)
from decode_dse.software.sweep_plan import (
    SweepRunPlan,
    load_immutable_json,
    make_stage_manifest,
    validate_run_plan,
    write_immutable_json,
)

REFINEMENT_PROFILE_SCHEMA = "decode-refinement-profile"
REFINEMENT_SCHEDULE_SCHEMA = "decode-refinement-schedule"
REFINEMENT_SHARD_PLAN_SCHEMA = "decode-refinement-shard-plan"
REFINEMENT_VALIDITY_SCHEMA = "decode-refinement-validity"
REFINEMENT_WEIGHT_METHODS = ("rtn", "gptq_erry", "rotation")
REFINEMENT_EVIDENCE_STATES = ("succeeded", "failed", "oom")
REFINEMENT_EXECUTION_STATES = (
    "scheduled",
    "skipped_doomed",
    "blocked_evidence",
)
_VALIDITY_FIELDS = (
    "software_valid",
    "compiler_valid",
    "emulator_valid",
    "rtl_valid",
    "dc_calibrated",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class DecodeRefinementProfile:
    """Precision contract derived from one promoted hardware-validation profile."""

    source_profile: DecodePrecisionProfile
    key_format: str
    value_format: str
    weight_method: str = "gptq_erry"
    schema_version: str = REFINEMENT_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFINEMENT_PROFILE_SCHEMA:
            raise ValueError(f"unsupported refinement schema {self.schema_version!r}")
        if self.source_profile.kind != PROFILE_KIND_QUANTIZED:
            raise ValueError("refinement requires a quantized source profile")
        if self.source_profile.block_size != MX_BLOCK_SIZE:
            raise ValueError(f"refinement requires block size {MX_BLOCK_SIZE}")
        if self.key_format not in DECODE_FORMATS:
            raise ValueError(f"unsupported K refinement format {self.key_format!r}")
        if self.value_format not in DECODE_FORMATS:
            raise ValueError(f"unsupported V refinement format {self.value_format!r}")
        if self.weight_method not in REFINEMENT_WEIGHT_METHODS:
            raise ValueError(
                f"unsupported refinement weight method {self.weight_method!r}"
            )

    @property
    def weight_format(self) -> str:
        return self.source_profile.weight_format

    @property
    def activation_format(self) -> str:
        return self.source_profile.activation_format

    @property
    def vector_format(self) -> str:
        return self.source_profile.vector_format

    @property
    def block_size(self) -> int:
        return self.source_profile.block_size

    @property
    def kind(self) -> str:
        return self.source_profile.kind

    @property
    def scale_format(self) -> str:
        return self.source_profile.scale_format

    @property
    def scale_bits(self) -> int:
        return self.source_profile.scale_bits

    @property
    def accumulator_rule(self) -> str:
        return self.source_profile.accumulator_rule

    @property
    def output_rule(self) -> str:
        return self.source_profile.output_rule

    @property
    def matrix_semantics(self) -> Any:
        return self.source_profile.matrix_semantics

    @property
    def weight_operators(self) -> tuple[str, ...]:
        return self.source_profile.weight_operators

    @property
    def activation_operators(self) -> tuple[str, ...]:
        return self.source_profile.activation_operators

    @property
    def kv_operators(self) -> tuple[str, ...]:
        return self.source_profile.kv_operators

    @property
    def vector_operators(self) -> tuple[str, ...]:
        return self.source_profile.vector_operators

    @property
    def bf16_operators(self) -> tuple[str, ...]:
        return self.source_profile.bf16_operators

    @property
    def split_kv(self) -> bool:
        return self.key_format != self.value_format

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_profile_id": self.source_profile.profile_id,
            "source_profile": self.source_profile.to_dict(),
            "key_format": self.key_format,
            "value_format": self.value_format,
            "weight_method": self.weight_method,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodeRefinementProfile":
        source = DecodePrecisionProfile.from_dict(value["source_profile"])
        if value.get("source_profile_id") != source.profile_id:
            raise ValueError("refinement source-profile identity mismatch")
        return cls(
            source_profile=source,
            key_format=str(value["key_format"]),
            value_format=str(value["value_format"]),
            weight_method=str(value.get("weight_method", "gptq_erry")),
            schema_version=str(
                value.get(
                    "schema_version",
                    REFINEMENT_PROFILE_SCHEMA,
                )
            ),
        )

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def profile_id(self) -> str:
        return f"drp-{self.canonical_hash}"


def _adjacent_role_formats(format_id: str) -> tuple[str, ...]:
    descriptor = format_descriptor(format_id)
    family_formats = tuple(
        token
        for token in DECODE_FORMATS
        if format_descriptor(token).family == descriptor.family
    )
    tiers = sorted({format_descriptor(token).element_bits for token in family_formats})
    current_bits = descriptor.element_bits
    lower = max((bits for bits in tiers if bits < current_bits), default=None)
    upper = min((bits for bits in tiers if bits > current_bits), default=None)
    selected_bits = {current_bits}
    if lower is not None:
        selected_bits.add(lower)
    if upper is not None:
        selected_bits.add(upper)
    return tuple(
        token
        for token in family_formats
        if token != format_id and format_descriptor(token).element_bits in selected_bits
    )


def iter_split_kv_variants(
    source_profile: DecodePrecisionProfile,
    *,
    weight_method: str = "gptq_erry",
) -> Iterable[DecodeRefinementProfile]:
    """Yield equal-KV baseline and adjacent-tier directional controls."""

    if source_profile.kind != PROFILE_KIND_QUANTIZED:
        raise ValueError("only quantized profiles can enter refinement")
    base = source_profile.kv_format
    candidates = [(base, base)]
    for neighbor in _adjacent_role_formats(base):
        candidates.extend(((base, neighbor), (neighbor, base)))
    seen: set[tuple[str, str]] = set()
    for key_format, value_format in candidates:
        pair = key_format, value_format
        if pair in seen:
            continue
        seen.add(pair)
        yield DecodeRefinementProfile(
            source_profile=source_profile,
            key_format=key_format,
            value_format=value_format,
            weight_method=weight_method,
        )


@dataclass(frozen=True)
class RefinementAccuracyEvidence:
    """Measured hardware-validation evidence consumed by the doomed gate."""

    source_profile_id: str
    state: str
    mean_nll: float | None = None
    error_class: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.source_profile_id:
            raise ValueError("source_profile_id is required")
        if self.state not in REFINEMENT_EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence state {self.state!r}")
        if self.state == "succeeded":
            if self.mean_nll is None or not math.isfinite(self.mean_nll):
                raise ValueError("successful evidence requires finite mean_nll")
            if self.mean_nll < 0:
                raise ValueError("mean_nll must be non-negative")
            if self.error_class is not None or self.error_message is not None:
                raise ValueError("successful evidence cannot carry an error")
        else:
            if self.mean_nll is not None:
                raise ValueError("failed evidence cannot carry a numerical score")
            if not self.error_class:
                raise ValueError("failed evidence requires error_class")

    @classmethod
    def succeeded(
        cls,
        profile_id: str,
        mean_nll: float,
    ) -> "RefinementAccuracyEvidence":
        return cls(profile_id, "succeeded", float(mean_nll))

    @classmethod
    def failed(
        cls,
        profile_id: str,
        *,
        error_class: str,
        error_message: str | None = None,
        oom: bool = False,
    ) -> "RefinementAccuracyEvidence":
        return cls(
            profile_id,
            "oom" if oom else "failed",
            error_class=error_class,
            error_message=error_message,
        )


@dataclass(frozen=True)
class DoomedGatePolicy:
    """Conservative perplexity limits for skipping costly refinement."""

    perplexity_ratio: float = 3.0
    absolute_perplexity: float = 100.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.perplexity_ratio) or self.perplexity_ratio <= 1.0:
            raise ValueError("perplexity_ratio must be finite and greater than one")
        if (
            not math.isfinite(self.absolute_perplexity)
            or self.absolute_perplexity <= 1.0
        ):
            raise ValueError("absolute_perplexity must be finite and greater than one")

    def to_dict(self) -> dict[str, float]:
        return {
            "perplexity_ratio": self.perplexity_ratio,
            "absolute_perplexity": self.absolute_perplexity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoomedGatePolicy":
        return cls(
            perplexity_ratio=float(value["perplexity_ratio"]),
            absolute_perplexity=float(value["absolute_perplexity"]),
        )


@dataclass(frozen=True)
class DoomedGateDecision:
    execution_state: str
    reason: str
    threshold_mean_nll: float | None
    observed_mean_nll: float | None

    def __post_init__(self) -> None:
        if self.execution_state not in REFINEMENT_EXECUTION_STATES:
            raise ValueError(f"unsupported execution state {self.execution_state!r}")

    @property
    def executable(self) -> bool:
        return self.execution_state == "scheduled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_state": self.execution_state,
            "executable": self.executable,
            "reason": self.reason,
            "threshold_mean_nll": self.threshold_mean_nll,
            "observed_mean_nll": self.observed_mean_nll,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DoomedGateDecision":
        decision = cls(
            execution_state=str(value["execution_state"]),
            reason=str(value["reason"]),
            threshold_mean_nll=(
                None
                if value.get("threshold_mean_nll") is None
                else float(value["threshold_mean_nll"])
            ),
            observed_mean_nll=(
                None
                if value.get("observed_mean_nll") is None
                else float(value["observed_mean_nll"])
            ),
        )
        if (
            value.get("executable") is not None
            and bool(value["executable"]) != decision.executable
        ):
            raise ValueError("stored refinement executable flag is inconsistent")
        return decision


def evaluate_doomed_gate(
    evidence: RefinementAccuracyEvidence | None,
    *,
    reference_mean_nll: float,
    policy: DoomedGatePolicy = DoomedGatePolicy(),
) -> DoomedGateDecision:
    """Return an auditable decision without scoring failed evaluations."""

    if not math.isfinite(reference_mean_nll) or reference_mean_nll < 0:
        raise ValueError("reference_mean_nll must be finite and non-negative")
    if evidence is None:
        return DoomedGateDecision(
            "blocked_evidence",
            "missing_source_evidence",
            None,
            None,
        )
    if evidence.state != "succeeded":
        return DoomedGateDecision(
            "blocked_evidence",
            f"source_{evidence.state}:{evidence.error_class}",
            None,
            None,
        )
    threshold = max(
        reference_mean_nll + math.log(policy.perplexity_ratio),
        math.log(policy.absolute_perplexity),
    )
    if evidence.mean_nll > threshold:
        return DoomedGateDecision(
            "skipped_doomed",
            "source_accuracy_exceeds_gate",
            threshold,
            evidence.mean_nll,
        )
    return DoomedGateDecision(
        "scheduled",
        "source_accuracy_within_gate",
        threshold,
        evidence.mean_nll,
    )


@dataclass(frozen=True)
class RefinementScheduleEntry:
    ordinal: int
    profile: DecodeRefinementProfile
    gate: DoomedGateDecision
    validity: StackValidity = StackValidity()

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("refinement ordinals must be non-negative")

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "profile_id": self.profile_id,
            "profile": self.profile.to_dict(),
            "gate": self.gate.to_dict(),
            **self.validity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementScheduleEntry":
        profile = DecodeRefinementProfile.from_dict(value["profile"])
        if value.get("profile_id") != profile.profile_id:
            raise ValueError("refinement profile identity mismatch")
        return cls(
            ordinal=int(value["ordinal"]),
            profile=profile,
            gate=DoomedGateDecision.from_dict(value["gate"]),
            validity=StackValidity.from_dict(value),
        )


@dataclass(frozen=True)
class RefinementSchedule:
    entries: tuple[RefinementScheduleEntry, ...]
    source_profile_ids: tuple[str, ...]
    reference_mean_nll: float
    policy: DoomedGatePolicy
    schema_version: str = REFINEMENT_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFINEMENT_SCHEDULE_SCHEMA:
            raise ValueError(f"unsupported refinement schedule {self.schema_version!r}")
        if not math.isfinite(self.reference_mean_nll):
            raise ValueError("reference_mean_nll must be finite")
        if tuple(entry.ordinal for entry in self.entries) != tuple(
            range(len(self.entries))
        ):
            raise ValueError("refinement ordinals must be contiguous")
        profile_ids = tuple(entry.profile_id for entry in self.entries)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("refinement schedule contains duplicate profiles")
        if tuple(sorted(set(self.source_profile_ids))) != self.source_profile_ids:
            raise ValueError("source_profile_ids must be unique and sorted")

    @property
    def counts(self) -> dict[str, int]:
        counts = {state: 0 for state in REFINEMENT_EXECUTION_STATES}
        for entry in self.entries:
            counts[entry.gate.execution_state] += 1
        counts["total"] = len(self.entries)
        return counts

    def to_dict(self) -> dict[str, Any]:
        content = {
            "schema_version": self.schema_version,
            "source_profile_ids": list(self.source_profile_ids),
            "reference_mean_nll": self.reference_mean_nll,
            "policy": self.policy.to_dict(),
            "counts": self.counts,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        return {
            **content,
            "schedule_hash": self._hash_content(content),
        }

    @staticmethod
    def _hash_content(content: Mapping[str, Any]) -> str:
        return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()

    @property
    def canonical_hash(self) -> str:
        return str(self.to_dict()["schedule_hash"])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementSchedule":
        schedule = cls(
            entries=tuple(
                RefinementScheduleEntry.from_dict(entry) for entry in value["entries"]
            ),
            source_profile_ids=tuple(
                str(profile_id) for profile_id in value["source_profile_ids"]
            ),
            reference_mean_nll=float(value["reference_mean_nll"]),
            policy=DoomedGatePolicy.from_dict(value["policy"]),
            schema_version=str(value["schema_version"]),
        )
        if value.get("counts") != schedule.counts:
            raise ValueError("refinement schedule count mismatch")
        if value.get("schedule_hash") != schedule.canonical_hash:
            raise ValueError("refinement schedule hash mismatch")
        return schedule


@dataclass(frozen=True)
class RefinementShardPlan:
    """One source-owned partition of an immutable refinement schedule."""

    master_schedule_hash: str
    shard_index: int
    shard_count: int
    source_profile_id: str
    profile_ids: tuple[str, ...]
    schema_version: str = REFINEMENT_SHARD_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFINEMENT_SHARD_PLAN_SCHEMA:
            raise ValueError("unsupported refinement shard-plan schema")
        if len(self.master_schedule_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.master_schedule_hash
        ):
            raise ValueError("master schedule hash must be a lowercase SHA-256")
        if self.shard_count != 4:
            raise ValueError("refinement execution requires exactly four shards")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise ValueError("refinement shard index is out of range")
        if not self.source_profile_id:
            raise ValueError("refinement shard source identity is required")
        if not self.profile_ids or len(self.profile_ids) != len(set(self.profile_ids)):
            raise ValueError("refinement shard profiles must be non-empty and unique")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "master_schedule_hash": self.master_schedule_hash,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "source_profile_id": self.source_profile_id,
            "profile_ids": list(self.profile_ids),
        }

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._content_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"shard_plan_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementShardPlan":
        plan = cls(
            master_schedule_hash=str(value["master_schedule_hash"]),
            shard_index=int(value["shard_index"]),
            shard_count=int(value["shard_count"]),
            source_profile_id=str(value["source_profile_id"]),
            profile_ids=tuple(str(item) for item in value["profile_ids"]),
            schema_version=str(value["schema_version"]),
        )
        if value.get("shard_plan_hash") != plan.canonical_hash:
            raise ValueError("refinement shard-plan hash mismatch")
        return plan


def validate_refinement_shard_plan(
    schedule: RefinementSchedule,
    plan: RefinementShardPlan,
) -> tuple[RefinementScheduleEntry, ...]:
    """Bind a shard plan to its exact master entries and source."""

    if plan.master_schedule_hash != schedule.canonical_hash:
        raise ValueError("refinement shard plan targets another master schedule")
    by_id = {entry.profile_id: entry for entry in schedule.entries}
    if any(profile_id not in by_id for profile_id in plan.profile_ids):
        raise ValueError("refinement shard plan contains an unknown profile")
    entries = tuple(by_id[profile_id] for profile_id in plan.profile_ids)
    if tuple(entry.ordinal for entry in entries) != tuple(
        sorted(entry.ordinal for entry in entries)
    ):
        raise ValueError("refinement shard profiles differ from master order")
    if any(
        entry.profile.source_profile.profile_id != plan.source_profile_id
        for entry in entries
    ):
        raise ValueError("refinement shard spans multiple source profiles")
    expected = tuple(
        entry.profile_id
        for entry in schedule.entries
        if entry.profile.source_profile.profile_id == plan.source_profile_id
    )
    if plan.profile_ids != expected:
        raise ValueError("refinement shard does not cover its complete source")
    return entries


def build_refinement_shard_plans(
    schedule: RefinementSchedule,
) -> tuple[RefinementShardPlan, ...]:
    """Partition four sources into disjoint, deterministic GPU shards."""

    if len(schedule.source_profile_ids) != 4:
        raise ValueError("refinement sharding requires exactly four source profiles")
    plans = tuple(
        RefinementShardPlan(
            master_schedule_hash=schedule.canonical_hash,
            shard_index=index,
            shard_count=4,
            source_profile_id=source_profile_id,
            profile_ids=tuple(
                entry.profile_id
                for entry in schedule.entries
                if entry.profile.source_profile.profile_id == source_profile_id
            ),
        )
        for index, source_profile_id in enumerate(schedule.source_profile_ids)
    )
    covered = tuple(profile_id for plan in plans for profile_id in plan.profile_ids)
    expected = tuple(entry.profile_id for entry in schedule.entries)
    if len(covered) != len(set(covered)) or set(covered) != set(expected):
        raise AssertionError("refinement shard plans are not a disjoint cover")
    for plan in plans:
        validate_refinement_shard_plan(schedule, plan)
    return plans


def write_refinement_shard_plan(
    path: str | Path,
    plan: RefinementShardPlan,
) -> Path:
    from decode_dse.software.sweep_plan import write_immutable_json

    return write_immutable_json(path, plan.to_dict())


def load_refinement_shard_plan(path: str | Path) -> RefinementShardPlan:
    from decode_dse.software.sweep_plan import load_immutable_json

    value = load_immutable_json(path)
    value.pop("content_hash", None)
    return RefinementShardPlan.from_dict(value)


@dataclass(frozen=True)
class RefinementValidityRecord:
    """Measured validity for one exact equal- or split-K/V profile."""

    profile_id: str
    validity: StackValidity
    evidence: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        if not self.profile_id.startswith("drp-"):
            raise ValueError("refinement validity requires a refinement profile ID")
        evidence = tuple(sorted((str(name), value) for name, value in self.evidence))
        if tuple(name for name, _ in evidence) != tuple(sorted(_VALIDITY_FIELDS)):
            raise ValueError("refinement validity evidence fields are incomplete")
        for name, digest in evidence:
            observed = getattr(self.validity, name)
            if observed is None:
                if digest is not None:
                    raise ValueError("unmeasured validity cannot cite evidence")
            elif (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("measured validity requires a lowercase evidence hash")
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            **self.validity.to_dict(),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementValidityRecord":
        if set(value) != {"profile_id", *_VALIDITY_FIELDS, "evidence"}:
            raise ValueError("refinement validity record fields differ from schema")
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TypeError("refinement validity evidence must be a mapping")
        return cls(
            profile_id=str(value["profile_id"]),
            validity=StackValidity.from_dict(value),
            evidence=tuple((str(name), digest) for name, digest in evidence.items()),
        )


@dataclass(frozen=True)
class RefinementValidityManifest:
    """Immutable profile-local validity join for one base schedule."""

    source_schedule_hash: str
    records: tuple[RefinementValidityRecord, ...]
    schema_version: str = REFINEMENT_VALIDITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFINEMENT_VALIDITY_SCHEMA:
            raise ValueError("unsupported refinement validity schema")
        if len(self.source_schedule_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_schedule_hash
        ):
            raise ValueError("source schedule hash must be a lowercase SHA-256")
        profile_ids = tuple(record.profile_id for record in self.records)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("refinement validity contains duplicate profiles")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_schedule_hash": self.source_schedule_hash,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._content_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"validity_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementValidityManifest":
        manifest = cls(
            source_schedule_hash=str(value["source_schedule_hash"]),
            records=tuple(
                RefinementValidityRecord.from_dict(record)
                for record in value["records"]
            ),
            schema_version=str(value["schema_version"]),
        )
        if value.get("validity_hash") != manifest.canonical_hash:
            raise ValueError("refinement validity identity mismatch")
        return manifest


def attach_refinement_validity(
    schedule: RefinementSchedule,
    validity: RefinementValidityManifest,
) -> RefinementSchedule:
    """Join only measurements made for each exact refinement profile."""

    if validity.source_schedule_hash != schedule.canonical_hash:
        raise ValueError("refinement validity was measured for another schedule")
    expected_ids = tuple(entry.profile_id for entry in schedule.entries)
    observed_ids = tuple(record.profile_id for record in validity.records)
    if observed_ids != expected_ids:
        raise ValueError("refinement validity coverage is missing, extra, or reordered")
    entries = tuple(
        RefinementScheduleEntry(
            ordinal=entry.ordinal,
            profile=entry.profile,
            gate=entry.gate,
            validity=record.validity,
        )
        for entry, record in zip(schedule.entries, validity.records)
    )
    return RefinementSchedule(
        entries=entries,
        source_profile_ids=schedule.source_profile_ids,
        reference_mean_nll=schedule.reference_mean_nll,
        policy=schedule.policy,
    )


def write_refinement_validity(
    path: str | Path,
    validity: RefinementValidityManifest,
) -> Path:
    from decode_dse.software.sweep_plan import write_immutable_json

    return write_immutable_json(path, validity.to_dict())


def load_refinement_validity(
    path: str | Path,
) -> RefinementValidityManifest:
    from decode_dse.software.sweep_plan import load_immutable_json

    value = load_immutable_json(path)
    value.pop("content_hash", None)
    return RefinementValidityManifest.from_dict(value)


def build_refinement_schedule(
    promoted_profiles: Sequence[DecodePrecisionProfile],
    evidence: Mapping[str, RefinementAccuracyEvidence],
    *,
    reference_mean_nll: float,
    policy: DoomedGatePolicy = DoomedGatePolicy(),
    weight_method: str = "gptq_erry",
) -> RefinementSchedule:
    """Build a stable schedule without expanding the screening manifest."""

    sources = tuple(sorted(promoted_profiles, key=lambda item: item.profile_id))
    source_ids = tuple(profile.profile_id for profile in sources)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("promoted profiles must be unique")
    unknown_evidence = sorted(set(evidence) - set(source_ids))
    if unknown_evidence:
        raise ValueError(f"evidence contains unpromoted profiles: {unknown_evidence}")

    entries: list[RefinementScheduleEntry] = []
    seen: set[str] = set()
    for source in sources:
        source_evidence = evidence.get(source.profile_id)
        if (
            source_evidence is not None
            and source_evidence.source_profile_id != source.profile_id
        ):
            raise ValueError("refinement evidence key/profile mismatch")
        decision = evaluate_doomed_gate(
            source_evidence,
            reference_mean_nll=reference_mean_nll,
            policy=policy,
        )
        for profile in iter_split_kv_variants(
            source,
            weight_method=weight_method,
        ):
            if profile.profile_id in seen:
                raise AssertionError("split-KV generation produced a duplicate")
            seen.add(profile.profile_id)
            entries.append(
                RefinementScheduleEntry(
                    ordinal=len(entries),
                    profile=profile,
                    gate=decision,
                )
            )
    return RefinementSchedule(
        entries=tuple(entries),
        source_profile_ids=source_ids,
        reference_mean_nll=reference_mean_nll,
        policy=policy,
    )


def build_refinement_schedule_from_promotion(
    promotion: Any,
    evidence: Mapping[str, RefinementAccuracyEvidence],
    *,
    reference_mean_nll: float,
    policy: DoomedGatePolicy = DoomedGatePolicy(),
    weight_method: str = "gptq_erry",
) -> RefinementSchedule:
    """Build refinement directly from a complete epsilon-Pareto promotion."""

    from decode_dse.hardware.selection import PromotionResult

    if not isinstance(promotion, PromotionResult):
        raise TypeError("promotion must be a PromotionResult")
    if not promotion.controls_complete:
        raise ValueError("refinement requires a complete promotion control set")
    return build_refinement_schedule(
        tuple(point.profile for point in promotion.points),
        evidence,
        reference_mean_nll=reference_mean_nll,
        policy=policy,
        weight_method=weight_method,
    )


def build_selective_rotation_schedule(
    base_schedule: RefinementSchedule,
    *,
    best_supported_profile_ids: Sequence[str],
    uniform_i8_profile_id: str,
) -> RefinementSchedule:
    """Rotate one measured winner per source, including equal-K/V MXINT8."""

    best = tuple(str(profile_id) for profile_id in best_supported_profile_ids)
    if len(best) != 4 or len(set(best)) != 4:
        raise ValueError("selective rotation requires four distinct profiles")
    if uniform_i8_profile_id not in set(best):
        raise ValueError("the four rotation profiles must include uniform MXINT8")
    entries_by_id = {entry.profile_id: entry for entry in base_schedule.entries}
    missing = tuple(
        profile_id for profile_id in best if profile_id not in entries_by_id
    )
    if missing:
        raise ValueError(f"rotation selection contains unknown profiles: {missing}")
    selected_by_source: dict[str, RefinementScheduleEntry] = {}
    for profile_id in best:
        entry = entries_by_id[profile_id]
        source_id = entry.profile.source_profile.profile_id
        if source_id in selected_by_source:
            raise ValueError("rotation selection contains two profiles from one source")
        selected_by_source[source_id] = entry
    if set(selected_by_source) != set(base_schedule.source_profile_ids):
        raise ValueError("rotation selection must cover every refinement source")
    requested = tuple(
        selected_by_source[source_id].profile_id
        for source_id in base_schedule.source_profile_ids
    )

    rotated = []
    for profile_id in requested:
        source_entry = entries_by_id[profile_id]
        if not source_entry.gate.executable:
            raise ValueError("rotation selection contains a gated profile")
        validity = source_entry.validity
        if any(
            value is not True
            for value in (
                validity.software_valid,
                validity.compiler_valid,
                validity.emulator_valid,
                validity.rtl_valid,
            )
        ):
            raise ValueError(
                "rotation selection requires measured software/compiler/emulator/RTL support"
            )
        source = source_entry.profile.source_profile
        if profile_id == uniform_i8_profile_id and (
            source.weight_format,
            source.activation_format,
            source.kv_format,
            source_entry.profile.key_format,
            source_entry.profile.value_format,
        ) != (
            "MXINT8",
            "MXINT8",
            "MXINT8",
            "MXINT8",
            "MXINT8",
        ):
            raise ValueError("uniform-I8 rotation control has the wrong precision")
        profile = DecodeRefinementProfile(
            source_profile=source,
            key_format=source_entry.profile.key_format,
            value_format=source_entry.profile.value_format,
            weight_method="rotation",
        )
        rotated.append(
            RefinementScheduleEntry(
                ordinal=len(rotated),
                profile=profile,
                gate=source_entry.gate,
                validity=StackValidity(),
            )
        )
    source_ids = tuple(
        sorted({entry.profile.source_profile.profile_id for entry in rotated})
    )
    return RefinementSchedule(
        entries=tuple(rotated),
        source_profile_ids=source_ids,
        reference_mean_nll=base_schedule.reference_mean_nll,
        policy=base_schedule.policy,
    )


def write_refinement_schedule(
    path: str | Path,
    schedule: RefinementSchedule,
) -> Path:
    """Atomically create or verify an immutable refinement schedule."""

    from decode_dse.software.sweep_plan import write_immutable_json

    return write_immutable_json(path, schedule.to_dict())


def load_refinement_schedule(path: str | Path) -> RefinementSchedule:
    """Load and verify an immutable refinement schedule."""

    from decode_dse.software.sweep_plan import load_immutable_json

    value = load_immutable_json(path)
    value.pop("content_hash", None)
    return RefinementSchedule.from_dict(value)


def refinement_profile_to_decode_quant_spec(
    profile: DecodeRefinementProfile,
) -> Any:
    """Map one refinement profile to a role-specific MASE runtime binding."""

    from decode_dse.software.precision_bindings import DecodeQuantSpec

    def operand(token: str) -> tuple[str, Any]:
        descriptor = format_descriptor(token)
        if descriptor.family == "mxint":
            return descriptor.family, descriptor.element_bits
        return descriptor.family, (
            descriptor.exponent_bits,
            descriptor.mantissa_bits,
        )

    weight_family, weight_width = operand(profile.weight_format)
    activation_family, activation_width = operand(profile.activation_format)
    key_family, key_width = operand(profile.key_format)
    value_family, value_width = operand(profile.value_format)
    return DecodeQuantSpec(
        attn_w=weight_width,
        ffn_w=weight_width,
        kv=key_width,
        w_fmt=weight_family,
        kv_fmt=key_family,
        key_kv=key_width,
        value_kv=value_width,
        key_kv_fmt=key_family,
        value_kv_fmt=value_family,
        weight_block=profile.block_size,
        kv_block=profile.block_size,
        act_w=activation_width,
        act_fmt=activation_family,
        act_block=profile.block_size,
        use_gptq=profile.weight_method == "gptq_erry",
        use_rotation=profile.weight_method == "rotation",
        fp_setting=profile.vector_format,
        fp_setting_attention=True,
        quant_attn_internals=True,
    )


# ---------------------------------------------------------------------------
# Schedule construction command
# ---------------------------------------------------------------------------
def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _path_hash(path: str | Path) -> str:
    """Digest a receipt input that may be a file or a stage partition directory."""

    resolved = Path(path)
    if resolved.is_file():
        return _file_hash(resolved)
    if not resolved.is_dir():
        raise ValueError(f"receipt input does not exist: {resolved}")
    files = tuple(sorted(value for value in resolved.rglob("*") if value.is_file()))
    if not files:
        raise ValueError(f"receipt input directory is empty: {resolved}")
    digest = hashlib.sha256()
    for value in files:
        name = value.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(bytes.fromhex(_file_hash(value)))
    return digest.hexdigest()


def _load_sharded_stage_rows(
    paths: Sequence[str | Path],
    *,
    manifest: Any,
    plan: SweepRunPlan,
    stage: str,
    profile_ids: Sequence[str],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    dict[str, Any],
]:
    from decode_dse.software.sweep import partition_stage_profile_ids

    invocations = []
    for raw in paths:
        path = Path(raw)
        candidates = (
            tuple(path.rglob("invocation.json"))
            if path.is_dir()
            else (path.parent / "invocation.json",)
        )
        for candidate in candidates:
            if candidate.is_file():
                value = load_immutable_json(candidate)
                if value.get("stage") == stage:
                    invocations.append((candidate, value))
    unique = {str(path.resolve()): (path, value) for path, value in invocations}
    invocations = tuple(unique[key] for key in sorted(unique))
    if not invocations:
        raise ValueError(f"{stage.upper()} result shards lack invocation receipts")
    shard_counts = {int(value["shard_count"]) for _, value in invocations}
    if len(shard_counts) != 1:
        raise ValueError(f"{stage.upper()} shard counts disagree")
    shard_count = shard_counts.pop()
    if {int(value["shard_index"]) for _, value in invocations} != set(
        range(shard_count)
    ):
        raise ValueError(f"{stage.upper()} shard receipts are incomplete")
    rows_by_id = {}
    manifests = {}
    for invocation_path, invocation in invocations:
        if (
            invocation.get("master_manifest_hash") != manifest.canonical_hash
            or invocation.get("run_plan_hash") != plan.canonical_hash
        ):
            raise ValueError(f"{stage.upper()} invocation identity mismatch")
        shard_ids = partition_stage_profile_ids(
            manifest,
            profile_ids,
            shard_index=int(invocation["shard_index"]),
            shard_count=shard_count,
        )
        shard_manifest = make_stage_manifest(manifest, shard_ids)
        if invocation.get("stage_manifest_hash") != shard_manifest.canonical_hash:
            raise ValueError(f"{stage.upper()} shard manifest mismatch")
        manifests[shard_manifest.canonical_hash] = shard_manifest
        rows = load_terminal_numerical_rows(
            (invocation_path.parent,),
            shard_manifest,
            require_complete=True,
        )
        for row in rows:
            profile_id = str(row["profile_id"])
            if profile_id in rows_by_id:
                raise ValueError(f"{stage.upper()} contains duplicate profile rows")
            rows_by_id[profile_id] = row
    expected = tuple(profile_ids)
    if set(rows_by_id) != set(expected):
        raise ValueError(f"{stage.upper()} profile coverage is incomplete")
    full_manifest = make_stage_manifest(manifest, expected)
    manifests[full_manifest.canonical_hash] = full_manifest
    return tuple(rows_by_id[profile_id] for profile_id in expected), manifests


def validate_complete_numerical_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_profile_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Reject missing, duplicate, reordered, or unmeasured numerical rows."""

    expected = tuple(expected_profile_ids)
    observed = tuple(str(row.get("profile_id", "")) for row in rows)
    if observed != expected or len(observed) != len(set(observed)):
        raise ValueError("hardware-validation coverage is incomplete or reordered")
    result = {}
    for row in rows:
        state = row.get("state")
        if state not in {"succeeded", "failed"}:
            raise ValueError("hardware-validation row is not terminal")
        validity = row.get("validity")
        if not isinstance(validity, Mapping):
            raise ValueError("hardware-validation row lacks measured validity")
        observed_validity = validity.get("software_valid")
        if not isinstance(observed_validity, bool):
            raise ValueError("hardware-validation software validity is unmeasured")
        if state == "succeeded" and observed_validity is not True:
            raise ValueError("successful hardware-validation row is not software-valid")
        result[str(row["profile_id"])] = row
    return result


def _finite_positive(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("hardware promotion metric must be finite and positive")
    return float(value)


def _hardware_point(
    row: Mapping[str, Any],
    *,
    profile: Any,
    mean_nll: float,
) -> ParetoPoint | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or row.get("error_code") is not None:
        return None
    whole = metrics.get("whole_model")
    energy = whole.get("calibrated_energy") if isinstance(whole, Mapping) else None
    output_head = metrics.get("output_head_boundary")
    head_estimate = (
        output_head.get("estimate") if isinstance(output_head, Mapping) else None
    )
    capacity = metrics.get("capacity")
    energy_tier = energy.get("energy_tier") if isinstance(energy, Mapping) else None
    energy_identity = (
        energy.get("energy_id") or energy.get("calibration_id")
        if isinstance(energy, Mapping)
        else None
    )
    timing_tier = whole.get("publication_timing_tier") if isinstance(whole, Mapping) else None
    if (
        row.get("deployment_valid") is not True
        or not isinstance(whole, Mapping)
        or whole.get("rankable") is not True
        or timing_tier
        not in {
            "compiler_trace_request_calibrated",
            "stage_calibrated_analytic",
        }
        or not isinstance(energy, Mapping)
        or energy_tier not in {"analytic_anchored", "dc_calibrated"}
        or not isinstance(energy_identity, str)
        or not energy_identity
        or metrics.get("timing_calibrated") is not True
        or metrics.get("runtime_feasible") is not True
        or not isinstance(capacity, Mapping)
        or capacity.get("feasible") is not True
        or row.get("packedkv_selector_valid") is not True
        or not isinstance(head_estimate, Mapping)
    ):
        return None
    return ParetoPoint(
        profile=profile,
        mean_nll=mean_nll,
        tpot_ms=_finite_positive(whole.get("tpot_ms")),
        tps=_finite_positive(whole.get("tps")),
        energy_per_token_j=_finite_positive(energy.get("total_j")),
        area_mm2=_finite_positive(metrics.get("area_mm2")),
        candidate_id=str(row["candidate_id"]),
        power_calibration_id=energy_identity,
        cost_scope="whole_model",
        system_calibration_id=str(whole["system_calibration_id"]),
        head_service_calibration_id=str(head_estimate["calibration_id"]),
        whole_model_rankable=True,
        energy_tier=energy_tier,
        publication_timing_tier=str(timing_tier),
    )


def _error_evidence(row: Mapping[str, Any]) -> RefinementAccuracyEvidence:
    if row["state"] == "succeeded":
        return RefinementAccuracyEvidence.succeeded(
            str(row["profile_id"]),
            float(row["result"]["mean_nll"]),
        )
    error_class = str(row.get("error_class") or "NumericalEvaluationFailed")
    return RefinementAccuracyEvidence.failed(
        str(row["profile_id"]),
        error_class=error_class,
        error_message=(
            None if row.get("error_message") is None else str(row["error_message"])
        ),
        oom="outofmemory" in error_class.casefold()
        or "out of memory" in str(row.get("error_message", "")).casefold(),
    )


def derive_refinement_validity(
    schedule: RefinementSchedule,
    stack_validity_document: Mapping[str, Any],
) -> RefinementValidityManifest:
    """Carry measured stack validity onto exactly-matching refinement profiles.

    A refinement profile inherits its source profile's measured validity only
    when its physical formats are identical to the source's (the equal-K/V
    baseline). Split-K/V variants were never validated as whole profiles by
    the stack reports, so their validity is recorded as unmeasured rather
    than synthesized; they remain accuracy-only downstream.
    """

    profiles = stack_validity_document.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("stack validity lacks per-profile records")
    content_hash = stack_validity_document.get("content_hash")
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        raise ValueError("stack validity lacks a content hash")
    records = []
    for entry in schedule.entries:
        profile = entry.profile
        source_id = profile.source_profile.profile_id
        source_validity = profiles.get(source_id)
        if source_validity is None:
            raise ValueError(
                f"stack validity does not cover refinement source {source_id}"
            )
        if (
            profile.key_format
            == profile.value_format
            == profile.source_profile.kv_format
        ):
            validity = StackValidity.from_dict(source_validity)
        else:
            validity = StackValidity()
        evidence = tuple(
            (name, content_hash if getattr(validity, name) is not None else None)
            for name in _VALIDITY_FIELDS
        )
        records.append(
            RefinementValidityRecord(
                profile_id=entry.profile_id,
                validity=validity,
                evidence=evidence,
            )
        )
    return RefinementValidityManifest(
        source_schedule_hash=schedule.canonical_hash,
        records=tuple(records),
    )


def prepare_refinement_schedule(
    *,
    manifest_path: str | Path,
    run_plan_path: str | Path,
    numerical_screen_paths: Sequence[str | Path],
    hardware_validation_paths: Sequence[str | Path],
    hardware_study_paths: Sequence[str | Path],
    schedule_path: str | Path,
    promotion_path: str | Path,
    epsilon: EpsilonPolicy = EpsilonPolicy(),
    validity_path: str | Path | None = None,
    stack_validity_path: str | Path | None = None,
    validity_output_path: str | Path | None = None,
) -> None:
    """Build four measured refinement sources from complete upstream results."""

    manifest = load_manifest(manifest_path)
    plan_value = load_immutable_json(run_plan_path)
    plan_value.pop("content_hash", None)
    plan = SweepRunPlan.from_dict(plan_value)
    validate_run_plan(plan, manifest)
    hardware_validation_manifest = make_stage_manifest(
        manifest, plan.hardware_validation_profile_ids
    )
    hardware_validation_rows, hardware_validation_manifests = _load_sharded_stage_rows(
        hardware_validation_paths,
        manifest=manifest,
        plan=plan,
        stage="hardware-validation",
        profile_ids=plan.hardware_validation_profile_ids,
    )
    numerical = validate_complete_numerical_rows(
        hardware_validation_rows,
        expected_profile_ids=plan.hardware_validation_profile_ids,
    )
    numerical_screen_rows, numerical_screen_manifests = _load_sharded_stage_rows(
        numerical_screen_paths,
        manifest=manifest,
        plan=plan,
        stage="numerical-screen",
        profile_ids=plan.numerical_screen_profile_ids,
    )
    screen_rows_by_profile = {
        profile_id: row
        for profile_id, row in zip(
            plan.numerical_screen_profile_ids, numerical_screen_rows
        )
    }
    workspace_references = tuple(
        entry
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
    )
    if len(workspace_references) != 1:
        raise ValueError("the workspace manifest must declare exactly one BF16 reference")
    reference_entry = workspace_references[0]
    # The hardware studies are produced per numerical-screen shard; shards that
    # lack the BF16 reference are priced under an in-memory reference-augmented
    # manifest, so its hash must be recomputed here to validate provenance.
    study_manifest_hashes = set(numerical_screen_manifests)
    for shard_manifest in numerical_screen_manifests.values():
        if any(
            entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
            for entry in shard_manifest.entries
        ):
            continue
        augmented = SweepManifest(
            model_name=shard_manifest.model_name,
            model_revision=shard_manifest.model_revision,
            model_architecture=shard_manifest.model_architecture,
            tokenizer_revision=shard_manifest.tokenizer_revision,
            quantizer_provenance=shard_manifest.quantizer_provenance,
            entries=shard_manifest.entries
            + (
                SweepManifestEntry(
                    ordinal=len(shard_manifest.entries),
                    profile=reference_entry.profile,
                    legality=reference_entry.legality,
                    validity=reference_entry.validity,
                ),
            ),
        )
        study_manifest_hashes.add(augmented.canonical_hash)
    reference_rows = [
        row
        for entry, row in zip(manifest.entries, numerical_screen_rows)
        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
    ]
    if len(reference_rows) != 1:
        raise ValueError("numerical screen must contain one BF16 reference")
    reference = reference_rows[0]
    if (
        reference.get("state") != "succeeded"
        or reference.get("validity", {}).get("software_valid") is not True
    ):
        raise ValueError("BF16 reference is incomplete or unmeasured")
    reference_mean_nll = float(reference["result"]["mean_nll"])
    if not math.isfinite(reference_mean_nll) or reference_mean_nll < 0:
        raise ValueError("BF16 reference NLL is invalid")

    hardware_rows = []
    compact_profiles: dict[str, Mapping[str, Any]] = {}
    if not hardware_study_paths:
        raise ValueError("at least one hardware-validation study is required")
    for hardware_path in hardware_study_paths:
        header, artifact_rows = load_hardware_artifact(hardware_path)
        provenance = header.get("provenance", {})
        if provenance.get("manifest_hash") not in study_manifest_hashes:
            raise ValueError(
                "hardware study was produced for another manifest"
            )
        if header.get("storage_revision") in {
            HARDWARE_STORAGE_REVISION,
            LEGACY_COMPACT_STORAGE_REVISION,
        }:
            aggregates = header.get("profile_aggregates")
            if not isinstance(aggregates, list):
                raise ValueError("compact hardware profile aggregates are missing")
            for aggregate in aggregates:
                if not isinstance(aggregate, Mapping):
                    raise TypeError("compact hardware aggregate must be a mapping")
                profile_id = str(aggregate.get("profile_id", ""))
                if not profile_id or profile_id in compact_profiles:
                    raise ValueError(
                        "compact hardware profile aggregates overlap partitions"
                    )
                compact_profiles[profile_id] = aggregate
        hardware_rows.extend(artifact_rows)
    by_profile: dict[str, list[Mapping[str, Any]]] = {}
    for row in hardware_rows:
        profile_id = str(row["profile_id"])
        if profile_id not in numerical:
            raise ValueError("hardware-validation row references an unknown profile")
        if row.get("profile") != next(
            entry.profile.to_dict()
            for entry in hardware_validation_manifest.entries
            if entry.profile_id == profile_id
        ):
            raise ValueError("hardware-validation profile identity mismatch")
        if row.get("numerical_result_hash") != _canonical_hash(
            screen_rows_by_profile[profile_id]
        ):
            raise ValueError(
                "hardware study joined a numerical row this screen did not measure"
            )
        retention_labels = tuple(row.get("retention_labels", ()))
        if (
            retention_labels
            and retention_labels != ("legacy_full_row",)
            and "profile_frontier" not in retention_labels
        ):
            continue
        by_profile.setdefault(profile_id, []).append(row)

    entries = {
        entry.profile_id: entry for entry in hardware_validation_manifest.entries
    }
    for profile_id, aggregate in compact_profiles.items():
        entry = entries.get(profile_id)
        if (
            entry is None
            or aggregate.get("profile") != entry.profile.to_dict()
            or aggregate.get("numerical_result_hash")
            != _canonical_hash(screen_rows_by_profile[profile_id])
        ):
            raise ValueError(
                "compact hardware aggregate differs from its numerical profile"
            )
    points = []
    evidence = {}
    for profile_id in plan.hardware_validation_profile_ids:
        entry = entries[profile_id]
        row = numerical[profile_id]
        evidence[profile_id] = _error_evidence(row)
        if row["state"] != "succeeded":
            continue
        mean_nll = float(row["result"]["mean_nll"])
        hardware_points = tuple(
            point
            for candidate in by_profile.get(profile_id, ())
            if (
                point := _hardware_point(
                    candidate,
                    profile=entry.profile,
                    mean_nll=mean_nll,
                )
            )
            is not None
        )
        aggregate = compact_profiles.get(profile_id)
        if aggregate is not None:
            valid_count = int(aggregate.get("valid_count", -1))
            frontier_count = int(aggregate.get("local_frontier_count", -1))
            if valid_count < 0 or frontier_count < 0:
                raise ValueError("compact hardware valid count is invalid")
            if (
                (valid_count > 0) != bool(hardware_points)
                or frontier_count != len(hardware_points)
            ):
                raise ValueError(
                    "compact hardware frontier differs from its profile aggregate"
                )
        if entry.legality.hardware_candidate and not hardware_points:
            raise ValueError(
                "hardware-validation costs are unmeasured for " f"{profile_id}"
            )
        points.extend(
            hardware_points
            or (
                ParetoPoint(
                    profile=entry.profile,
                    mean_nll=mean_nll,
                    tpot_ms=None,
                    tps=None,
                    energy_per_token_j=None,
                    area_mm2=None,
                ),
            )
        )
    source_selection = select_refinement_sources(
        points,
        reference_mean_nll=reference_mean_nll,
        epsilon=epsilon,
    )
    promotion = source_selection.promotion
    promoted_evidence = {
        point.profile_id: evidence[point.profile_id] for point in promotion.points
    }
    schedule = build_refinement_schedule_from_promotion(
        promotion,
        promoted_evidence,
        reference_mean_nll=reference_mean_nll,
    )
    if stack_validity_path is not None and validity_path is not None:
        raise ValueError(
            "pass either a stack-validity manifest to derive validity from or "
            "a prebuilt refinement validity manifest, not both"
        )
    validity_manifest = None
    if stack_validity_path is not None:
        if validity_output_path is None:
            raise ValueError(
                "deriving refinement validity requires an output path"
            )
        document = load_immutable_json(stack_validity_path)
        if (
            document.get("manifest_hash") != manifest.canonical_hash
            or document.get("run_plan_hash") != plan.canonical_hash
        ):
            raise ValueError("stack validity is bound to another workspace")
        validity_manifest = derive_refinement_validity(schedule, document)
        schedule = attach_refinement_validity(schedule, validity_manifest)
        write_refinement_validity(validity_output_path, validity_manifest)
    elif validity_path is not None:
        schedule = attach_refinement_validity(
            schedule,
            load_refinement_validity(validity_path),
        )
    write_refinement_schedule(schedule_path, schedule)
    write_immutable_json(
        promotion_path,
        {
            "schema_version": "decode-refinement-source-selection",
            "manifest_hash": manifest.canonical_hash,
            "run_plan_hash": plan.canonical_hash,
            "hardware_validation_manifest_hash": (
                hardware_validation_manifest.canonical_hash
            ),
            "numerical_screen_sha256": [
                _path_hash(path) for path in numerical_screen_paths
            ],
            "hardware_validation_sha256": [
                _path_hash(path) for path in hardware_validation_paths
            ],
            "hardware_study_sha256": [
                _path_hash(path) for path in hardware_study_paths
            ],
            "reference_mean_nll": reference_mean_nll,
            "source_selection": source_selection.to_dict(),
            "schedule_hash": schedule.canonical_hash,
            "validity_hash": (
                validity_manifest.canonical_hash
                if validity_manifest is not None
                else None
            ),
        },
    )


def build_schedule_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-plan", required=True)
    parser.add_argument(
        "--numerical-screen-results",
        dest="numerical_screen_results",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--hardware-validation-results",
        dest="hardware_validation_results",
        action="append",
        required=True,
    )
    parser.add_argument("--hardware-study", action="append", required=True)
    parser.add_argument("--validity")
    parser.add_argument("--stack-validity")
    parser.add_argument("--validity-output")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--promotion", required=True)
    for name in ("mean-nll", "tpot-ms", "tps", "energy-per-token-j", "area-mm2"):
        parser.add_argument(f"--epsilon-{name}", type=float, default=0.0)
    return parser


def build_schedule_main(argv: Iterable[str] | None = None) -> int:
    args = build_schedule_parser().parse_args(tuple(argv) if argv is not None else None)
    prepare_refinement_schedule(
        manifest_path=args.manifest,
        run_plan_path=args.run_plan,
        numerical_screen_paths=args.numerical_screen_results,
        hardware_validation_paths=args.hardware_validation_results,
        hardware_study_paths=args.hardware_study,
        schedule_path=args.schedule,
        promotion_path=args.promotion,
        validity_path=args.validity,
        stack_validity_path=args.stack_validity,
        validity_output_path=args.validity_output,
        epsilon=EpsilonPolicy(
            mean_nll=args.epsilon_mean_nll,
            tpot_ms=args.epsilon_tpot_ms,
            tps=args.epsilon_tps,
            energy_per_token_j=args.epsilon_energy_per_token_j,
            area_mm2=args.epsilon_area_mm2,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(build_schedule_main())


__all__ = [
    "DecodeRefinementProfile",
    "DoomedGateDecision",
    "DoomedGatePolicy",
    "REFINEMENT_EVIDENCE_STATES",
    "REFINEMENT_EXECUTION_STATES",
    "REFINEMENT_PROFILE_SCHEMA",
    "REFINEMENT_SCHEDULE_SCHEMA",
    "REFINEMENT_SHARD_PLAN_SCHEMA",
    "REFINEMENT_VALIDITY_SCHEMA",
    "RefinementAccuracyEvidence",
    "RefinementSchedule",
    "RefinementScheduleEntry",
    "RefinementShardPlan",
    "RefinementValidityManifest",
    "RefinementValidityRecord",
    "attach_refinement_validity",
    "build_refinement_schedule_from_promotion",
    "build_refinement_schedule",
    "build_refinement_shard_plans",
    "build_selective_rotation_schedule",
    "derive_refinement_validity",
    "evaluate_doomed_gate",
    "iter_split_kv_variants",
    "load_refinement_schedule",
    "load_refinement_shard_plan",
    "load_refinement_validity",
    "refinement_profile_to_decode_quant_spec",
    "write_refinement_schedule",
    "write_refinement_shard_plan",
    "write_refinement_validity",
    "validate_refinement_shard_plan",
]
