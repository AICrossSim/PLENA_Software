"""Immutable stage plans and fail-closed preflight gates for decode sweeps."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.manifest import (
    EXPECTED_BF16_REFERENCES,
    EXPECTED_QUANTIZED_PROFILES,
    EXPECTED_TOTAL_PROFILES,
    EXPECTED_VECTOR_CONTROLS,
    QuantizerProvenance,
    QuantizerSource,
    ResolvedImportOrigin,
    SweepManifest,
    SweepManifestEntry,
)
from decode_dse.profiles import (
    DECODE_FORMATS,
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    PROFILE_KIND_VECTOR_BF16_CONTROL,
    VECTOR_FP_FORMATS,
    DeclaredSearchSpace,
    DecodePrecisionProfile,
    enumerate_decode_profiles,
    format_descriptor,
)
from types import MappingProxyType

RUN_PLAN_SCHEMA = "decode-sweep-run-plan"
PROMPT_MANIFEST_SCHEMA = "decode-prompt-manifest"
PREFLIGHT_EVIDENCE_SCHEMA = "decode-preflight-evidence"
GATE_REPORT_SCHEMA = "decode-preflight-gates"

PREFLIGHT_PROFILE_COUNT = 36
NUMERICAL_SCREEN_PROFILE_COUNT = 3585
HARDWARE_VALIDATION_QUANTIZED_PROFILE_COUNT = 858
HARDWARE_VALIDATION_VECTOR_CONTROL_COUNT = 143
HARDWARE_VALIDATION_PROFILE_COUNT = (
    HARDWARE_VALIDATION_QUANTIZED_PROFILE_COUNT
    + HARDWARE_VALIDATION_VECTOR_CONTROL_COUNT
)
PARITY_TOLERANCE = 1e-5
MAX_PROJECTED_HOURS = 168.0
WORKSPACE_PATH_PREFIX = "workspace://"


def resolve_bound_path(
    value: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve repository-relative, absolute, or workspace-bound paths."""

    token = os.fspath(value)
    repository = Path(repository_root).resolve()
    if token.startswith(WORKSPACE_PATH_PREFIX):
        if workspace_root is None:
            raise ValueError("workspace_root is required for workspace-bound paths")
        suffix = token[len(WORKSPACE_PATH_PREFIX) :]
        relative = Path(suffix)
        if not suffix or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid workspace-bound path: {token!r}")
        workspace = Path(workspace_root).resolve()
        resolved = (workspace / relative).resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                f"workspace-bound path escapes its root: {token!r}"
            ) from exc
        return resolved
    path = Path(token)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def write_immutable_json(
    path: str | os.PathLike[str],
    value: Mapping[str, Any],
) -> Path:
    """Atomically create a checksummed JSON file or verify an identical file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = dict(value)
    body.pop("content_hash", None)
    payload = body | {"content_hash": _content_hash(body)}
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if load_immutable_json(destination) != payload:
            raise FileExistsError(
                f"refusing to replace a different immutable file: {destination}"
            )
        return destination

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
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if load_immutable_json(destination) != payload:
                raise FileExistsError(
                    f"refusing to replace a different immutable file: {destination}"
                )
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_immutable_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.pop("content_hash", None)
    if expected != _content_hash(value):
        raise ValueError(f"immutable JSON checksum mismatch: {path}")
    return value | {"content_hash": expected}


def manifest_declared_space(manifest: SweepManifest) -> DeclaredSearchSpace:
    """Derive the declared precision space from the manifest's own entries.

    The distinct formats on each axis, ordered canonically, define the space;
    the cross-product completeness check in validate_exhaustive_manifest then
    proves the manifest enumerates that space exhaustively. Exclusion
    rationales live in the sweep configuration, not the manifest, so the
    derived space carries none.
    """

    quantized = tuple(
        entry.profile
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_QUANTIZED
    )
    weight = {profile.weight_format for profile in quantized}
    activation = {profile.activation_format for profile in quantized}
    kv = {profile.kv_format for profile in quantized}
    vector = {profile.vector_format for profile in quantized}
    return DeclaredSearchSpace(
        weight_formats=tuple(f for f in DECODE_FORMATS if f in weight),
        activation_formats=tuple(f for f in DECODE_FORMATS if f in activation),
        kv_formats=tuple(f for f in DECODE_FORMATS if f in kv),
        vector_formats=tuple(f for f in VECTOR_FP_FORMATS if f in vector),
        exclusions=MappingProxyType({}),
    )


def validate_exhaustive_manifest(manifest: SweepManifest) -> None:
    """Require exhaustive canonical-order coverage of the declared space.

    A manifest over the canonical space reproduces the historical 3,585
    profiles exactly; a declared-subspace manifest must be the complete
    cross product of the formats it contains, in the same nested order.
    """

    space = manifest_declared_space(manifest)
    expected_counts = {
        PROFILE_KIND_QUANTIZED: space.expected_quantized_profiles,
        PROFILE_KIND_VECTOR_BF16_CONTROL: space.expected_vector_bf16_controls,
        PROFILE_KIND_BF16_REFERENCE: EXPECTED_BF16_REFERENCES,
        "total": space.expected_total_profiles,
    }
    if manifest.counts != expected_counts:
        raise ValueError(
            f"manifest counts differ from the exhaustive contract: {manifest.counts}"
        )
    expected_profiles = enumerate_decode_profiles(space)
    actual_ids = tuple(entry.profile_id for entry in manifest.entries)
    expected_ids = tuple(profile.profile_id for profile in expected_profiles)
    if actual_ids != expected_ids:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_ids, expected_ids)
                )
                if actual != expected
            ),
            min(len(actual_ids), len(expected_ids)),
        )
        raise ValueError(f"manifest profile order differs at ordinal {mismatch}")
    for entry, profile in zip(manifest.entries, expected_profiles):
        if entry.profile != profile:
            raise ValueError(f"manifest profile differs at ordinal {entry.ordinal}")
    if len(set(actual_ids)) != space.expected_total_profiles:
        raise ValueError("manifest profile IDs are not unique")


def _entry_features(entry: SweepManifestEntry) -> frozenset[str]:
    profile = entry.profile
    features = {
        f"kind:{profile.kind}",
        f"weight:{profile.weight_format}",
        f"activation:{profile.activation_format}",
        f"kv:{profile.kv_format}",
        f"vector:{profile.vector_format}",
        (
            "legality:hardware_candidate"
            if entry.legality.hardware_candidate
            else "legality:numerical_only"
        ),
    }
    if profile.kind != PROFILE_KIND_BF16_REFERENCE:
        features.add(
            "families:"
            + "/".join(
                (
                    format_descriptor(profile.weight_format).family,
                    format_descriptor(profile.activation_format).family,
                    format_descriptor(profile.kv_format).family,
                )
            )
        )
    features.update(f"issue:{issue.code}" for issue in entry.legality.issues)
    return frozenset(features)


def preflight_required_features(
    manifest: SweepManifest,
) -> frozenset[str]:
    """Return all format, family, vector, kind, and legality strata."""

    return frozenset(
        feature for entry in manifest.entries for feature in _entry_features(entry)
    )


def _evenly_spaced(
    entries: Sequence[SweepManifestEntry],
    count: int,
) -> tuple[SweepManifestEntry, ...]:
    if count <= 0 or not entries:
        return ()
    if count >= len(entries):
        return tuple(entries)
    if count == 1:
        return (entries[len(entries) // 2],)
    indices = tuple(
        round(index * (len(entries) - 1) / (count - 1)) for index in range(count)
    )
    return tuple(entries[index] for index in indices)


def select_preflight_entries(
    manifest: SweepManifest,
) -> tuple[SweepManifestEntry, ...]:
    """Select 36 deterministic profiles covering every required stratum."""

    validate_exhaustive_manifest(manifest)
    features = {entry.profile_id: _entry_features(entry) for entry in manifest.entries}
    uncovered = set(preflight_required_features(manifest))
    selected: dict[str, SweepManifestEntry] = {}

    while uncovered:
        candidates = (
            entry for entry in manifest.entries if entry.profile_id not in selected
        )
        best = max(
            candidates,
            key=lambda entry: (
                len(features[entry.profile_id] & uncovered),
                -entry.ordinal,
            ),
        )
        gain = features[best.profile_id] & uncovered
        if not gain:
            raise AssertionError(
                f"uncovered preflight strata are unreachable: {sorted(uncovered)}"
            )
        selected[best.profile_id] = best
        uncovered.difference_update(gain)

    legal_quantized = tuple(
        entry
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_QUANTIZED
        and entry.legality.hardware_candidate
    )
    for entry in _evenly_spaced(legal_quantized, 12):
        selected.setdefault(entry.profile_id, entry)

    fill_buckets = (
        legal_quantized,
        tuple(
            entry
            for entry in manifest.entries
            if entry.profile.kind == PROFILE_KIND_QUANTIZED
            and not entry.legality.hardware_candidate
        ),
        tuple(
            entry
            for entry in manifest.entries
            if entry.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
        ),
    )
    fill_width = PREFLIGHT_PROFILE_COUNT
    fill_candidates = tuple(
        entry for bucket in fill_buckets for entry in _evenly_spaced(bucket, fill_width)
    )
    for entry in fill_candidates:
        if len(selected) == PREFLIGHT_PROFILE_COUNT:
            break
        selected.setdefault(entry.profile_id, entry)
    if len(selected) < PREFLIGHT_PROFILE_COUNT:
        for entry in manifest.entries:
            if len(selected) == PREFLIGHT_PROFILE_COUNT:
                break
            selected.setdefault(entry.profile_id, entry)
    if len(selected) != PREFLIGHT_PROFILE_COUNT:
        raise AssertionError(
            f"expected {PREFLIGHT_PROFILE_COUNT} preflight profiles, got {len(selected)}"
        )

    ordered = tuple(sorted(selected.values(), key=lambda entry: entry.ordinal))
    covered = frozenset(
        feature for entry in ordered for feature in features[entry.profile_id]
    )
    missing = preflight_required_features(manifest) - covered
    if missing:
        raise AssertionError(f"preflight selection misses strata: {sorted(missing)}")
    return ordered


@dataclass(frozen=True)
class StageSampleContract:
    """Pinned numerical and validation workload for one sweep stage."""

    name: str
    prompt_set: str
    prompt_count: int
    prefill_tokens: int
    decode_steps: int
    q_len: int
    teacher_forced_cached: bool
    compiler_required: bool
    emulator_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_set": self.prompt_set,
            "prompt_count": self.prompt_count,
            "prefill_tokens": self.prefill_tokens,
            "decode_steps": self.decode_steps,
            "q_len": self.q_len,
            "teacher_forced_cached": self.teacher_forced_cached,
            "compiler_required": self.compiler_required,
            "emulator_required": self.emulator_required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageSampleContract":
        return cls(
            name=str(value["name"]),
            prompt_set=str(value["prompt_set"]),
            prompt_count=int(value["prompt_count"]),
            prefill_tokens=int(value["prefill_tokens"]),
            decode_steps=int(value["decode_steps"]),
            q_len=int(value["q_len"]),
            teacher_forced_cached=_strict_bool(
                value["teacher_forced_cached"], "teacher_forced_cached"
            ),
            compiler_required=_strict_bool(
                value["compiler_required"], "compiler_required"
            ),
            emulator_required=_strict_bool(
                value["emulator_required"], "emulator_required"
            ),
        )


NUMERICAL_SCREEN_SAMPLE_CONTRACT = StageSampleContract(
    name="numerical-screen",
    prompt_set="numerical_screen",
    prompt_count=16,
    prefill_tokens=512,
    decode_steps=16,
    q_len=1,
    teacher_forced_cached=True,
    compiler_required=False,
    emulator_required=False,
)
HARDWARE_VALIDATION_SAMPLE_CONTRACT = StageSampleContract(
    name="hardware-validation",
    prompt_set="hardware_validation",
    prompt_count=32,
    prefill_tokens=512,
    decode_steps=16,
    q_len=1,
    teacher_forced_cached=True,
    compiler_required=True,
    emulator_required=True,
)


@dataclass(frozen=True)
class GPUBaselinePlan:
    """CUDA-free plan for measured cached-one-token GPU baseline work units."""

    attention_implementation: str = "sdpa"
    warmup_steps: int = 16
    measured_steps: int = 128
    repetitions: int = 3
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    precision: str = "BF16"
    q_len: int = 1
    first_gpu_only: bool = True
    energy_meter_priority: tuple[str, ...] = (
        "nvml_total_energy_counter",
        "nvml_power_trace_trapezoidal",
    )
    power_trace_sample_interval_ms: int = 10

    def __post_init__(self) -> None:
        if self.attention_implementation != "sdpa":
            raise ValueError("GPU baseline attention implementation must be sdpa")
        for name in ("warmup_steps", "measured_steps", "repetitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"GPU baseline {name} must be a positive integer")
        if self.repetitions < 3:
            raise ValueError("GPU baseline requires at least three repetitions")
        if (
            not self.batch_sizes
            or self.batch_sizes != tuple(sorted(set(self.batch_sizes)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
                for value in self.batch_sizes
            )
        ):
            raise ValueError(
                "GPU baseline batch sizes must be unique, increasing, and within the prompt count"
            )
        if self.precision != "BF16" or self.q_len != 1:
            raise ValueError("GPU baseline must use BF16 cached q_len=1 decode")
        if self.first_gpu_only is not True:
            raise ValueError("GPU baseline must run on only the first physical GPU")
        if self.energy_meter_priority != (
            "nvml_total_energy_counter",
            "nvml_power_trace_trapezoidal",
        ):
            raise ValueError("GPU baseline energy-meter priority is not canonical")
        if (
            isinstance(self.power_trace_sample_interval_ms, bool)
            or not isinstance(self.power_trace_sample_interval_ms, int)
            or not 1 <= self.power_trace_sample_interval_ms <= 1000
        ):
            raise ValueError(
                "GPU baseline power-trace interval must be an integer in [1, 1000] ms"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_implementation": self.attention_implementation,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "repetitions": self.repetitions,
            "batch_sizes": list(self.batch_sizes),
            "precision": self.precision,
            "q_len": self.q_len,
            "first_gpu_only": self.first_gpu_only,
            "energy_meter_priority": list(self.energy_meter_priority),
            "power_trace_sample_interval_ms": self.power_trace_sample_interval_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUBaselinePlan":
        return cls(
            attention_implementation=str(value["attention_implementation"]),
            warmup_steps=int(value["warmup_steps"]),
            measured_steps=int(value["measured_steps"]),
            repetitions=int(value["repetitions"]),
            batch_sizes=tuple(int(item) for item in value["batch_sizes"]),
            precision=str(value["precision"]),
            q_len=int(value["q_len"]),
            first_gpu_only=_strict_bool(
                value["first_gpu_only"],
                "gpu_baseline.first_gpu_only",
            ),
            energy_meter_priority=tuple(
                str(item) for item in value["energy_meter_priority"]
            ),
            power_trace_sample_interval_ms=int(
                value["power_trace_sample_interval_ms"]
            ),
        )

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "GPUBaselinePlan":
        raw = dict(value or {})
        defaults = cls()
        return cls(
            attention_implementation=str(
                raw.get(
                    "attention_implementation",
                    defaults.attention_implementation,
                )
            ),
            warmup_steps=int(raw.get("warmup_steps", defaults.warmup_steps)),
            measured_steps=int(raw.get("measured_steps", defaults.measured_steps)),
            repetitions=int(raw.get("repetitions", defaults.repetitions)),
            batch_sizes=tuple(
                int(item) for item in raw.get("batch_sizes", defaults.batch_sizes)
            ),
            precision=str(raw.get("precision", defaults.precision)),
            q_len=int(raw.get("q_len", defaults.q_len)),
            first_gpu_only=raw.get("first_gpu_only", defaults.first_gpu_only),
            energy_meter_priority=tuple(
                str(item)
                for item in raw.get(
                    "energy_meter_priority",
                    defaults.energy_meter_priority,
                )
            ),
            power_trace_sample_interval_ms=int(
                raw.get(
                    "power_trace_sample_interval_ms",
                    defaults.power_trace_sample_interval_ms,
                )
            ),
        )


@dataclass(frozen=True)
class SweepRunPlan:
    """Content-addressed precision sweep and measured GPU-baseline schedule."""

    manifest_hash: str
    quantizer_provenance: QuantizerProvenance
    preflight_profile_ids: tuple[str, ...]
    numerical_screen_profile_ids: tuple[str, ...]
    hardware_validation_profile_ids: tuple[str, ...]
    device_labels: tuple[str, ...]
    gpu_baseline: GPUBaselinePlan = GPUBaselinePlan()
    numerical_screen_workers: int = 4
    hardware_validation_workers: int = 4
    numerical_screen_microbatch_size: int = 16
    hardware_validation_microbatch_size: int = 8
    sample_contracts: tuple[StageSampleContract, ...] = (
        NUMERICAL_SCREEN_SAMPLE_CONTRACT,
        HARDWARE_VALIDATION_SAMPLE_CONTRACT,
    )
    parity_tolerance: float = PARITY_TOLERANCE
    max_projected_hours: float = MAX_PROJECTED_HOURS
    schema_version: str = RUN_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUN_PLAN_SCHEMA:
            raise ValueError(f"unsupported run-plan schema {self.schema_version!r}")
        if not isinstance(self.gpu_baseline, GPUBaselinePlan):
            raise TypeError("run plan requires a GPU baseline plan")
        if len(self.preflight_profile_ids) != PREFLIGHT_PROFILE_COUNT:
            raise ValueError("run plan requires exactly 36 preflight profiles")
        if not self.numerical_screen_profile_ids:
            raise ValueError("run plan requires numerical-screen profiles")
        if not self.hardware_validation_profile_ids:
            raise ValueError("run plan requires hardware-validation profiles")
        if not set(self.hardware_validation_profile_ids) <= set(
            self.numerical_screen_profile_ids
        ):
            raise ValueError(
                "hardware-validation profiles must come from the numerical screen"
            )
        # Exhaustiveness over the declared search space is enforced against the
        # manifest in validate_run_plan; the canonical space yields the
        # historical 3,585-profile screen.
        for label, values in (
            ("preflight", self.preflight_profile_ids),
            ("numerical_screen", self.numerical_screen_profile_ids),
            ("hardware_validation", self.hardware_validation_profile_ids),
            ("device", self.device_labels),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"run plan contains duplicate {label} identifiers")
        if not self.device_labels or any(not value for value in self.device_labels):
            raise ValueError("run plan requires non-empty device labels")
        if self.device_labels != tuple(sorted(self.device_labels)):
            raise ValueError("run-plan device labels must be in canonical order")
        if self.parity_tolerance <= 0 or not math.isfinite(self.parity_tolerance):
            raise ValueError("parity tolerance must be finite and positive")
        if self.max_projected_hours <= 0 or not math.isfinite(self.max_projected_hours):
            raise ValueError("runtime limit must be finite and positive")
        for name in ("numerical_screen_workers", "hardware_validation_workers"):
            if getattr(self, name) <= 0 or getattr(self, name) > len(DECODE_FORMATS):
                raise ValueError(f"{name} must be in [1, {len(DECODE_FORMATS)}]")
        if (
            not 1
            <= self.numerical_screen_microbatch_size
            <= NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count
        ):
            raise ValueError(
                "numerical-screen microbatch size is outside the prompt count"
            )
        if (
            not 1
            <= self.hardware_validation_microbatch_size
            <= HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
        ):
            raise ValueError(
                "hardware-validation microbatch size is outside the prompt count"
            )

    @property
    def canonical_hash(self) -> str:
        return _content_hash(self._content_dict())

    @property
    def gpu_baseline_work_units(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "device_label": device_label,
                "batch_size": batch_size,
                "first_gpu_only": self.gpu_baseline.first_gpu_only,
            }
            for device_label in self.device_labels
            for batch_size in self.gpu_baseline.batch_sizes
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "quantizer_provenance": self.quantizer_provenance.to_dict(),
            "preflight_profile_ids": list(self.preflight_profile_ids),
            "numerical_screen_profile_ids": list(self.numerical_screen_profile_ids),
            "hardware_validation_profile_ids": list(
                self.hardware_validation_profile_ids
            ),
            "device_labels": list(self.device_labels),
            "gpu_baseline": self.gpu_baseline.to_dict(),
            "parallel_workers": {
                "numerical_screen": self.numerical_screen_workers,
                "hardware_validation": self.hardware_validation_workers,
            },
            "decode_microbatch_size": {
                "numerical_screen": self.numerical_screen_microbatch_size,
                "hardware_validation": self.hardware_validation_microbatch_size,
            },
            "sample_contracts": [
                contract.to_dict() for contract in self.sample_contracts
            ],
            "parity_tolerance": self.parity_tolerance,
            "max_projected_hours": self.max_projected_hours,
        }

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"run_plan_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepRunPlan":
        plan = cls(
            schema_version=str(value["schema_version"]),
            manifest_hash=str(value["manifest_hash"]),
            quantizer_provenance=QuantizerProvenance.from_dict(
                value["quantizer_provenance"]
            ),
            preflight_profile_ids=tuple(value["preflight_profile_ids"]),
            numerical_screen_profile_ids=tuple(value["numerical_screen_profile_ids"]),
            hardware_validation_profile_ids=tuple(
                value["hardware_validation_profile_ids"]
            ),
            device_labels=tuple(value["device_labels"]),
            gpu_baseline=GPUBaselinePlan.from_dict(value["gpu_baseline"]),
            numerical_screen_workers=int(value["parallel_workers"]["numerical_screen"]),
            hardware_validation_workers=int(
                value["parallel_workers"]["hardware_validation"]
            ),
            numerical_screen_microbatch_size=int(
                value["decode_microbatch_size"]["numerical_screen"]
            ),
            hardware_validation_microbatch_size=int(
                value["decode_microbatch_size"]["hardware_validation"]
            ),
            sample_contracts=tuple(
                StageSampleContract.from_dict(contract)
                for contract in value["sample_contracts"]
            ),
            parity_tolerance=float(value["parity_tolerance"]),
            max_projected_hours=float(value["max_projected_hours"]),
        )
        if value.get("run_plan_hash") != plan.canonical_hash:
            raise ValueError("run-plan content hash mismatch")
        return plan


def build_run_plan(
    manifest: SweepManifest,
    *,
    device_labels: Sequence[str],
    numerical_screen_workers: int = 4,
    hardware_validation_workers: int = 4,
    numerical_screen_microbatch_size: int = 16,
    hardware_validation_microbatch_size: int = 8,
    gpu_baseline: GPUBaselinePlan = GPUBaselinePlan(),
) -> SweepRunPlan:
    validate_exhaustive_manifest(manifest)
    preflight = select_preflight_entries(manifest)
    legal_triples = {
        (
            entry.profile.weight_format,
            entry.profile.activation_format,
            entry.profile.kv_format,
        )
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_QUANTIZED
        and entry.legality.hardware_candidate
    }
    hardware_validation_entries = tuple(
        entry
        for entry in manifest.entries
        if (
            entry.profile.kind == PROFILE_KIND_QUANTIZED
            and entry.legality.hardware_candidate
        )
        or (
            entry.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
            and (
                entry.profile.weight_format,
                entry.profile.activation_format,
                entry.profile.kv_format,
            )
            in legal_triples
        )
    )
    quantized_count = sum(
        entry.profile.kind == PROFILE_KIND_QUANTIZED
        for entry in hardware_validation_entries
    )
    control_count = sum(
        entry.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
        for entry in hardware_validation_entries
    )
    if manifest_declared_space(manifest).is_canonical and (
        quantized_count != HARDWARE_VALIDATION_QUANTIZED_PROFILE_COUNT
        or control_count != HARDWARE_VALIDATION_VECTOR_CONTROL_COUNT
    ):
        raise AssertionError(
            f"unexpected hardware-validation composition: {quantized_count} quantized, "
            f"{control_count} controls"
        )
    if quantized_count == 0 or control_count == 0:
        raise AssertionError(
            "hardware validation requires quantized profiles and vector controls"
        )
    plan = SweepRunPlan(
        manifest_hash=manifest.canonical_hash,
        quantizer_provenance=manifest.quantizer_provenance,
        preflight_profile_ids=tuple(entry.profile_id for entry in preflight),
        numerical_screen_profile_ids=tuple(
            entry.profile_id for entry in manifest.entries
        ),
        hardware_validation_profile_ids=tuple(
            entry.profile_id for entry in hardware_validation_entries
        ),
        device_labels=tuple(sorted(str(label) for label in device_labels)),
        gpu_baseline=gpu_baseline,
        numerical_screen_workers=numerical_screen_workers,
        hardware_validation_workers=hardware_validation_workers,
        numerical_screen_microbatch_size=numerical_screen_microbatch_size,
        hardware_validation_microbatch_size=hardware_validation_microbatch_size,
    )
    validate_run_plan(plan, manifest)
    return plan


def validate_run_plan(plan: SweepRunPlan, manifest: SweepManifest) -> None:
    validate_exhaustive_manifest(manifest)
    if plan.manifest_hash != manifest.canonical_hash:
        raise ValueError("run plan references a different manifest")
    if plan.quantizer_provenance != manifest.quantizer_provenance:
        raise ValueError("run plan references different quantizer arithmetic")
    expected_preflight = tuple(
        entry.profile_id for entry in select_preflight_entries(manifest)
    )
    if plan.preflight_profile_ids != expected_preflight:
        raise ValueError("run-plan preflight selection is not canonical")
    expected_numerical_screen = tuple(entry.profile_id for entry in manifest.entries)
    if plan.numerical_screen_profile_ids != expected_numerical_screen:
        raise ValueError("run-plan numerical-screen schedule is not exhaustive")
    expected_hardware_validation = {
        entry.profile_id
        for entry in manifest.entries
        if (
            entry.profile.kind == PROFILE_KIND_QUANTIZED
            and entry.legality.hardware_candidate
        )
    }
    legal_triples = {
        (
            entry.profile.weight_format,
            entry.profile.activation_format,
            entry.profile.kv_format,
        )
        for entry in manifest.entries
        if entry.profile_id in expected_hardware_validation
    }
    expected_hardware_validation.update(
        entry.profile_id
        for entry in manifest.entries
        if entry.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
        and (
            entry.profile.weight_format,
            entry.profile.activation_format,
            entry.profile.kv_format,
        )
        in legal_triples
    )
    canonical_hardware_validation = tuple(
        entry.profile_id
        for entry in manifest.entries
        if entry.profile_id in expected_hardware_validation
    )
    if plan.hardware_validation_profile_ids != canonical_hardware_validation:
        raise ValueError(
            "run-plan hardware-validation schedule differs from the legality contract"
        )
    if plan.sample_contracts != (
        NUMERICAL_SCREEN_SAMPLE_CONTRACT,
        HARDWARE_VALIDATION_SAMPLE_CONTRACT,
    ):
        raise ValueError(
            "run-plan sample contracts differ from numerical-screen/hardware-validation"
        )
    profile_by_id = {entry.profile_id: entry.profile for entry in manifest.entries}
    for stage, profile_ids, workers in (
        (
            "numerical-screen",
            plan.numerical_screen_profile_ids,
            plan.numerical_screen_workers,
        ),
        (
            "hardware-validation",
            plan.hardware_validation_profile_ids,
            plan.hardware_validation_workers,
        ),
    ):
        weight_banks = {
            profile_by_id[profile_id].weight_format for profile_id in profile_ids
        }
        if workers > len(weight_banks):
            raise ValueError(
                f"{stage} workers exceed its {len(weight_banks)} weight banks"
            )


@dataclass(frozen=True)
class PromptRecord:
    document_id: str
    prompt_hash: str

    def __post_init__(self) -> None:
        if not self.document_id or not self.prompt_hash:
            raise ValueError("prompt records require document and content hashes")

    def to_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "prompt_hash": self.prompt_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptRecord":
        return cls(
            document_id=str(value["document_id"]),
            prompt_hash=str(value["prompt_hash"]),
        )


@dataclass(frozen=True)
class PromptManifest:
    dataset_name: str
    dataset_revision: str
    numerical_screen: tuple[PromptRecord, ...]
    hardware_validation: tuple[PromptRecord, ...]
    schema_version: str = PROMPT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROMPT_MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported prompt-manifest schema {self.schema_version!r}"
            )
        if not self.dataset_name or not self.dataset_revision:
            raise ValueError("prompt dataset name and revision must be pinned")
        if len(self.numerical_screen) != NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count:
            raise ValueError(
                "numerical screen requires exactly "
                f"{NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count} prompt records"
            )
        if (
            len(self.hardware_validation)
            != HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
        ):
            raise ValueError(
                "hardware validation requires exactly "
                f"{HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count} prompt records"
            )
        all_records = self.numerical_screen + self.hardware_validation
        document_ids = tuple(record.document_id for record in all_records)
        prompt_hashes = tuple(record.prompt_hash for record in all_records)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                "numerical-screen and hardware-validation documents must be disjoint"
            )
        if len(prompt_hashes) != len(set(prompt_hashes)):
            raise ValueError(
                "numerical-screen and hardware-validation prompts must be disjoint"
            )

    @property
    def canonical_hash(self) -> str:
        return _content_hash(self._content_dict())

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "numerical_screen": [record.to_dict() for record in self.numerical_screen],
            "hardware_validation": [
                record.to_dict() for record in self.hardware_validation
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"prompt_manifest_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptManifest":
        prompt_manifest = cls(
            schema_version=str(value["schema_version"]),
            dataset_name=str(value["dataset_name"]),
            dataset_revision=str(value["dataset_revision"]),
            numerical_screen=tuple(
                PromptRecord.from_dict(record) for record in value["numerical_screen"]
            ),
            hardware_validation=tuple(
                PromptRecord.from_dict(record)
                for record in value["hardware_validation"]
            ),
        )
        if value.get("prompt_manifest_hash") != prompt_manifest.canonical_hash:
            raise ValueError("prompt-manifest content hash mismatch")
        return prompt_manifest


def load_prompt_manifest(path: str | os.PathLike[str]) -> PromptManifest:
    return PromptManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        condition = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {condition}")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _require_source(value: Mapping[str, Any]) -> str:
    source = str(value.get("source_artifact", ""))
    if not source:
        raise ValueError("every preflight observation requires a source artifact")
    return source


@dataclass(frozen=True)
class RuntimeSample:
    stage: str
    profile_id: str
    device_label: str
    runtime_seconds: float
    basis: str
    source_artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeSample":
        stage = str(value["stage"])
        if stage not in {"numerical-screen", "hardware-validation"}:
            raise ValueError(f"unsupported runtime stage {stage!r}")
        basis = str(value["basis"])
        if basis != (
            "evaluation_per_profile_excluding_weight_bank_build_"
            "and_deep_append_oracle"
        ):
            raise ValueError(f"unsupported runtime basis {basis!r}")
        return cls(
            stage=stage,
            profile_id=str(value["profile_id"]),
            device_label=str(value["device_label"]),
            runtime_seconds=_finite_float(
                value["runtime_seconds"], "runtime_seconds", positive=True
            ),
            basis=basis,
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class WeightBankBuildSample:
    weight_format: str
    device_label: str
    build_seconds: float
    source_artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WeightBankBuildSample":
        return cls(
            weight_format=str(value["weight_format"]),
            device_label=str(value["device_label"]),
            build_seconds=_finite_float(
                value["build_seconds"],
                "build_seconds",
                positive=True,
            ),
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class RuntimeRebindingSample:
    stage: str
    profile_id: str
    device_label: str
    binding_seconds: float
    performed: bool
    target_count: int
    used_cached_targets: bool
    weight_requantizations: int
    sealed_weight_modules: int
    weight_quantization_events_before: int
    weight_quantization_events_after: int
    weight_identity_before: str
    weight_identity_after: str
    weight_structure_fingerprint: str
    source_artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeRebindingSample":
        stage = str(value["stage"])
        if stage not in {"numerical-screen", "hardware-validation"}:
            raise ValueError(f"unsupported rebinding stage {stage!r}")
        target_count = int(value["target_count"])
        requantizations = int(value["weight_requantizations"])
        sealed_modules = int(value["sealed_weight_modules"])
        events_before = int(value["weight_quantization_events_before"])
        events_after = int(value["weight_quantization_events_after"])
        if any(
            number < 0
            for number in (
                target_count,
                requantizations,
                sealed_modules,
                events_before,
                events_after,
            )
        ):
            raise ValueError("rebinding counts must be non-negative")
        fingerprints = tuple(
            str(value[key])
            for key in (
                "weight_identity_before",
                "weight_identity_after",
                "weight_structure_fingerprint",
            )
        )
        if any(not fingerprint for fingerprint in fingerprints):
            raise ValueError("rebinding fingerprints must be non-empty")
        return cls(
            stage=stage,
            profile_id=str(value["profile_id"]),
            device_label=str(value["device_label"]),
            binding_seconds=_nonnegative_float(
                value["binding_seconds"],
                "binding_seconds",
            ),
            performed=_strict_bool(value["performed"], "performed"),
            target_count=target_count,
            used_cached_targets=_strict_bool(
                value["used_cached_targets"],
                "used_cached_targets",
            ),
            weight_requantizations=requantizations,
            sealed_weight_modules=sealed_modules,
            weight_quantization_events_before=events_before,
            weight_quantization_events_after=events_after,
            weight_identity_before=fingerprints[0],
            weight_identity_after=fingerprints[1],
            weight_structure_fingerprint=fingerprints[2],
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class NativeAppendValidationSample:
    stage: str
    profile_id: str
    device_label: str
    mode: str
    deep_oracle_enabled: bool
    calls: int
    expected_calls: int
    tensor_checks: int
    expected_tensor_checks: int
    quantized_tensor_checks: int
    expected_quantized_tensor_checks: int
    oracle_seconds: float
    source_artifact: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "NativeAppendValidationSample":
        stage = str(value["stage"])
        if stage not in {"numerical-screen", "hardware-validation"}:
            raise ValueError(f"unsupported append-validation stage {stage!r}")
        counts = {
            key: int(value[key])
            for key in (
                "calls",
                "expected_calls",
                "tensor_checks",
                "expected_tensor_checks",
                "quantized_tensor_checks",
                "expected_quantized_tensor_checks",
            )
        }
        if any(count < 0 for count in counts.values()):
            raise ValueError("append-validation counts must be non-negative")
        return cls(
            stage=stage,
            profile_id=str(value["profile_id"]),
            device_label=str(value["device_label"]),
            mode=str(value["mode"]),
            deep_oracle_enabled=_strict_bool(
                value["deep_oracle_enabled"],
                "deep_oracle_enabled",
            ),
            calls=counts["calls"],
            expected_calls=counts["expected_calls"],
            tensor_checks=counts["tensor_checks"],
            expected_tensor_checks=counts["expected_tensor_checks"],
            quantized_tensor_checks=counts["quantized_tensor_checks"],
            expected_quantized_tensor_checks=counts["expected_quantized_tensor_checks"],
            oracle_seconds=_nonnegative_float(
                value["oracle_seconds"],
                "oracle_seconds",
            ),
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class NLLParityCheck:
    profile_id: str
    left_label: str
    right_label: str
    left_mean_token_nll: float
    right_mean_token_nll: float
    source_artifact: str

    @property
    def absolute_error(self) -> float:
        return abs(self.left_mean_token_nll - self.right_mean_token_nll)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NLLParityCheck":
        return cls(
            profile_id=str(value["profile_id"]),
            left_label=str(value["left_label"]),
            right_label=str(value["right_label"]),
            left_mean_token_nll=_nonnegative_float(
                value["left_mean_token_nll"], "left_mean_token_nll"
            ),
            right_mean_token_nll=_nonnegative_float(
                value["right_mean_token_nll"], "right_mean_token_nll"
            ),
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class BF16SplitCheck:
    document_id: str
    token_ids_equal: bool
    max_abs_logit_error: float
    mean_token_nll_abs_error: float
    source_artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BF16SplitCheck":
        return cls(
            document_id=str(value["document_id"]),
            token_ids_equal=_strict_bool(value["token_ids_equal"], "token_ids_equal"),
            max_abs_logit_error=_nonnegative_float(
                value["max_abs_logit_error"], "max_abs_logit_error"
            ),
            mean_token_nll_abs_error=_nonnegative_float(
                value["mean_token_nll_abs_error"],
                "mean_token_nll_abs_error",
            ),
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class CacheReuseCheck:
    document_id: str
    cache_content_equal: bool
    mean_token_nll_abs_error: float
    source_artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheReuseCheck":
        return cls(
            document_id=str(value["document_id"]),
            cache_content_equal=_strict_bool(
                value["cache_content_equal"], "cache_content_equal"
            ),
            mean_token_nll_abs_error=_nonnegative_float(
                value["mean_token_nll_abs_error"],
                "mean_token_nll_abs_error",
            ),
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class MicrobatchParityCheck:
    profile_id: str
    microbatch_size: int
    max_abs_logit_error: float
    max_abs_token_nll_error: float
    max_abs_permutation_nll_error: float
    cache_growth_exact: bool
    lane_isolation_checked: bool
    source_artifact: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MicrobatchParityCheck":
        microbatch_size = int(value["microbatch_size"])
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        return cls(
            profile_id=str(value["profile_id"]),
            microbatch_size=microbatch_size,
            max_abs_logit_error=_nonnegative_float(
                value["max_abs_logit_error"],
                "max_abs_logit_error",
            ),
            max_abs_token_nll_error=_nonnegative_float(
                value["max_abs_token_nll_error"],
                "max_abs_token_nll_error",
            ),
            max_abs_permutation_nll_error=_nonnegative_float(
                value["max_abs_permutation_nll_error"],
                "max_abs_permutation_nll_error",
            ),
            cache_growth_exact=_strict_bool(
                value["cache_growth_exact"],
                "cache_growth_exact",
            ),
            lane_isolation_checked=_strict_bool(
                value["lane_isolation_checked"],
                "lane_isolation_checked",
            ),
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class GPUMemorySample:
    stage: str
    profile_id: str
    device_label: str
    microbatch_size: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    total_device_bytes: int
    source_artifact: str

    @property
    def peak_reserved_fraction(self) -> float:
        return self.peak_reserved_bytes / self.total_device_bytes

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUMemorySample":
        stage = str(value["stage"])
        if stage not in {"numerical-screen", "hardware-validation"}:
            raise ValueError(f"unsupported GPU-memory stage {stage!r}")
        fields = {
            name: int(value[name])
            for name in (
                "microbatch_size",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "total_device_bytes",
            )
        }
        if any(number <= 0 for number in fields.values()):
            raise ValueError("GPU memory fields must be positive")
        if (
            fields["peak_allocated_bytes"] > fields["peak_reserved_bytes"]
            or fields["peak_reserved_bytes"] > fields["total_device_bytes"]
        ):
            raise ValueError("GPU memory accounting is inconsistent")
        return cls(
            stage=stage,
            profile_id=str(value["profile_id"]),
            device_label=str(value["device_label"]),
            source_artifact=_require_source(value),
            **fields,
        )


@dataclass(frozen=True)
class AdmissionPreparationSample:
    manifest_hash: str
    run_plan_hash: str
    prompt_manifest_hash: str
    admission_index_hash: str
    admission_contract_id: str
    quantized_format_count: int
    document_count: int
    artifact_count: int
    cold_build_seconds: float
    persistence_policy: str
    logical_artifact_bytes: int
    persisted_bytes: int
    required_cold_capacity_bytes: int
    observed_cold_available_bytes: int
    required_host_bytes: int
    observed_host_available_bytes: int
    source_artifact: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "AdmissionPreparationSample":
        if value.get("schema_version") != "decode-admission-preparation":
            raise ValueError("unsupported admission preparation schema")
        resources = value["resource_projection"]
        persistence_policy = str(value.get("persistence_policy", ""))
        if persistence_policy != "content_addressed_recompute_per_format":
            raise ValueError("unsupported admission persistence policy")
        counts = {
            name: int(value[name])
            for name in (
                "quantized_format_count",
                "document_count",
                "artifact_count",
                "persisted_bytes",
                "logical_artifact_bytes",
            )
        }
        counts.update(
            {
                name: int(resources[name])
                for name in (
                    "required_cold_capacity_bytes",
                    "observed_cold_available_bytes",
                    "required_host_bytes",
                    "observed_host_available_bytes",
                )
            }
        )
        positive = {
            name: number for name, number in counts.items() if name != "persisted_bytes"
        }
        if (
            any(number <= 0 for number in positive.values())
            or counts["persisted_bytes"] != 0
        ):
            raise ValueError("admission preparation counts must be positive")
        identities = {
            name: str(value[name])
            for name in (
                "manifest_hash",
                "run_plan_hash",
                "prompt_manifest_hash",
                "admission_index_hash",
                "admission_contract_id",
            )
        }
        if any(not identity for identity in identities.values()):
            raise ValueError("admission preparation identities cannot be empty")
        return cls(
            **identities,
            quantized_format_count=counts["quantized_format_count"],
            document_count=counts["document_count"],
            artifact_count=counts["artifact_count"],
            cold_build_seconds=_finite_float(
                value["cold_build_seconds"],
                "cold_build_seconds",
                positive=True,
            ),
            persistence_policy=persistence_policy,
            logical_artifact_bytes=counts["logical_artifact_bytes"],
            persisted_bytes=counts["persisted_bytes"],
            required_cold_capacity_bytes=counts["required_cold_capacity_bytes"],
            observed_cold_available_bytes=counts["observed_cold_available_bytes"],
            required_host_bytes=counts["required_host_bytes"],
            observed_host_available_bytes=counts["observed_host_available_bytes"],
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class StackEvidencePreparationSample:
    """Measured compiler/emulator evidence-production time for hardware validation."""

    manifest_hash: str
    run_plan_hash: str
    required_stages: tuple[str, ...]
    compiler_seconds: float
    emulator_seconds: float
    critical_path_seconds: float
    timing_basis: str
    stage_report_hashes: Mapping[str, str]
    stack_validity_hash: str
    source_artifact: str

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "StackEvidencePreparationSample":
        required_stages = tuple(map(str, value["required_stages"]))
        if required_stages != ("compiler", "emulator"):
            raise ValueError("stack preparation must cover compiler and emulator")
        timing_basis = str(value["timing_basis"])
        if timing_basis != "union_of_measured_stage_wall_intervals":
            raise ValueError("unsupported stack preparation timing basis")
        compiler_seconds = _finite_float(
            value["compiler_seconds"],
            "compiler_seconds",
            positive=True,
        )
        emulator_seconds = _finite_float(
            value["emulator_seconds"],
            "emulator_seconds",
            positive=True,
        )
        critical_path_seconds = _finite_float(
            value["critical_path_seconds"],
            "critical_path_seconds",
            positive=True,
        )
        if not (
            max(compiler_seconds, emulator_seconds)
            <= critical_path_seconds
            <= compiler_seconds + emulator_seconds
        ):
            raise ValueError("stack critical path is inconsistent with stage intervals")
        report_hashes = value["stage_report_hashes"]
        if (
            not isinstance(report_hashes, Mapping)
            or set(report_hashes) != {"compiler", "emulator"}
            or any(not str(item) for item in report_hashes.values())
        ):
            raise ValueError("stack stage-report hashes are incomplete")
        identities = {
            key: str(value[key])
            for key in (
                "manifest_hash",
                "run_plan_hash",
                "stack_validity_hash",
            )
        }
        if any(not item for item in identities.values()):
            raise ValueError("stack preparation identities cannot be empty")
        return cls(
            **identities,
            required_stages=required_stages,
            compiler_seconds=compiler_seconds,
            emulator_seconds=emulator_seconds,
            critical_path_seconds=critical_path_seconds,
            timing_basis=timing_basis,
            stage_report_hashes={
                str(key): str(item) for key, item in report_hashes.items()
            },
            source_artifact=_require_source(value),
        )


@dataclass(frozen=True)
class PreflightEvidence:
    manifest_hash: str
    run_plan_hash: str
    prompt_manifest_hash: str
    completed_profile_ids: tuple[str, ...]
    runtime_samples: tuple[RuntimeSample, ...]
    weight_bank_build_samples: tuple[WeightBankBuildSample, ...]
    runtime_rebinding_samples: tuple[RuntimeRebindingSample, ...]
    native_append_validation_samples: tuple[
        NativeAppendValidationSample,
        ...,
    ]
    numerical_screen_workers: int
    hardware_validation_workers: int
    weight_bank_build_serialized: bool
    weight_bank_checks: tuple[NLLParityCheck, ...]
    cross_device_checks: tuple[NLLParityCheck, ...]
    bf16_split_checks: tuple[BF16SplitCheck, ...]
    cache_reuse_checks: tuple[CacheReuseCheck, ...]
    microbatch_checks: tuple[MicrobatchParityCheck, ...]
    gpu_memory_samples: tuple[GPUMemorySample, ...]
    admission_preparation: AdmissionPreparationSample
    stack_evidence_preparation: StackEvidencePreparationSample
    schema_version: str = PREFLIGHT_EVIDENCE_SCHEMA

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreflightEvidence":
        if value["schema_version"] != PREFLIGHT_EVIDENCE_SCHEMA:
            raise ValueError(
                f"unsupported preflight evidence schema {value['schema_version']!r}"
            )
        workers = value["parallel_workers"]
        evidence = cls(
            schema_version=str(value["schema_version"]),
            manifest_hash=str(value["manifest_hash"]),
            run_plan_hash=str(value["run_plan_hash"]),
            prompt_manifest_hash=str(value["prompt_manifest_hash"]),
            completed_profile_ids=tuple(value["completed_profile_ids"]),
            runtime_samples=tuple(
                RuntimeSample.from_dict(sample) for sample in value["runtime_samples"]
            ),
            weight_bank_build_samples=tuple(
                WeightBankBuildSample.from_dict(sample)
                for sample in value["weight_bank_build_samples"]
            ),
            runtime_rebinding_samples=tuple(
                RuntimeRebindingSample.from_dict(sample)
                for sample in value["runtime_rebinding_samples"]
            ),
            native_append_validation_samples=tuple(
                NativeAppendValidationSample.from_dict(sample)
                for sample in value["native_append_validation_samples"]
            ),
            numerical_screen_workers=int(workers["numerical_screen"]),
            hardware_validation_workers=int(workers["hardware_validation"]),
            weight_bank_build_serialized=_strict_bool(
                value["weight_bank_build_serialized"],
                "weight_bank_build_serialized",
            ),
            weight_bank_checks=tuple(
                NLLParityCheck.from_dict(check) for check in value["weight_bank_checks"]
            ),
            cross_device_checks=tuple(
                NLLParityCheck.from_dict(check)
                for check in value["cross_device_checks"]
            ),
            bf16_split_checks=tuple(
                BF16SplitCheck.from_dict(check) for check in value["bf16_split_checks"]
            ),
            cache_reuse_checks=tuple(
                CacheReuseCheck.from_dict(check)
                for check in value["cache_reuse_checks"]
            ),
            microbatch_checks=tuple(
                MicrobatchParityCheck.from_dict(check)
                for check in value["microbatch_checks"]
            ),
            gpu_memory_samples=tuple(
                GPUMemorySample.from_dict(sample)
                for sample in value["gpu_memory_samples"]
            ),
            admission_preparation=AdmissionPreparationSample.from_dict(
                value["admission_preparation"]
            ),
            stack_evidence_preparation=(
                StackEvidencePreparationSample.from_dict(
                    value["stack_evidence_preparation"]
                )
            ),
        )
        if (
            evidence.numerical_screen_workers <= 0
            or evidence.hardware_validation_workers <= 0
        ):
            raise ValueError("parallel worker counts must be positive")
        return evidence


def load_preflight_evidence(path: str | os.PathLike[str]) -> PreflightEvidence:
    return PreflightEvidence.from_dict(load_immutable_json(path))


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    measured: float | int | None = None
    limit: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "measured": self.measured,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class PreflightGateReport:
    manifest_hash: str
    run_plan_hash: str
    gates: tuple[GateResult, ...]
    schema_version: str = GATE_REPORT_SCHEMA

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "run_plan_hash": self.run_plan_hash,
            "passed": self.passed,
            "gates": [gate.to_dict() for gate in self.gates],
        }
        return body | {"gate_report_hash": _content_hash(body)}

    def require_passed(self) -> None:
        failed = [gate for gate in self.gates if not gate.passed]
        if failed:
            detail = "; ".join(f"{gate.name}: {gate.detail}" for gate in failed)
            raise RuntimeError(f"preflight gates failed: {detail}")


def _percentile_nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile without samples")
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _set_gate(
    name: str,
    actual: Iterable[str],
    expected: Iterable[str],
) -> GateResult:
    actual_values = tuple(actual)
    actual_set = set(actual_values)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    duplicate_count = len(actual_values) - len(actual_set)
    passed = not missing and not unexpected and duplicate_count == 0
    detail = (
        "exact coverage"
        if passed
        else (
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"duplicates={duplicate_count}"
        )
    )
    return GateResult(name, passed, detail, len(actual_set), len(expected_set))


def _runtime_observation_key(
    stage: str,
    profile_id: str,
    device_label: str,
) -> str:
    return "\x1f".join((stage, profile_id, device_label))


def _max_stage_profiles_per_worker(
    profile_ids: Sequence[str],
    *,
    workers: int,
    profile_by_id: Mapping[str, DecodePrecisionProfile],
) -> int:
    weight_order: list[str] = []
    counts: dict[str, int] = {}
    for profile_id in profile_ids:
        weight = profile_by_id[profile_id].weight_format
        counts[weight] = counts.get(weight, 0) + 1
        if weight not in weight_order:
            weight_order.append(weight)
    if workers <= 0 or workers > len(weight_order):
        raise ValueError("worker count exceeds the stage weight-bank count")
    loads = [0] * workers
    for index, weight in enumerate(weight_order):
        loads[index % workers] += counts[weight]
    return max(loads)


def _max_stage_build_seconds_per_worker(
    profile_ids: Sequence[str],
    *,
    workers: int,
    profile_by_id: Mapping[str, DecodePrecisionProfile],
    builds_by_format: Mapping[str, float],
) -> float:
    weight_order: list[str] = []
    for profile_id in profile_ids:
        weight = profile_by_id[profile_id].weight_format
        if weight not in weight_order:
            weight_order.append(weight)
    if workers <= 0 or workers > len(weight_order):
        raise ValueError("worker count exceeds the stage weight-bank count")
    missing = set(weight_order) - set(builds_by_format)
    if missing:
        raise ValueError(f"missing weight-bank build samples: {sorted(missing)}")
    loads = [0.0] * workers
    for index, weight in enumerate(weight_order):
        loads[index % workers] += builds_by_format[weight]
    return max(loads)


def _project_homogeneous_runtime_hours(
    *,
    plan: SweepRunPlan,
    profile_by_id: Mapping[str, DecodePrecisionProfile],
    builds_by_format: Mapping[str, float],
    p95_numerical_screen: float,
    p95_hardware_validation: float,
    admission_hours: float,
    stack_evidence_hours: float,
    workers: int,
    serialized_builds: bool,
) -> float:
    numerical_screen_evaluations = _max_stage_profiles_per_worker(
        plan.numerical_screen_profile_ids,
        workers=workers,
        profile_by_id=profile_by_id,
    )
    hardware_validation_evaluations = _max_stage_profiles_per_worker(
        plan.hardware_validation_profile_ids,
        workers=workers,
        profile_by_id=profile_by_id,
    )
    if serialized_builds:
        stage_weights = tuple(
            tuple(
                dict.fromkeys(
                    profile_by_id[profile_id].weight_format
                    for profile_id in profile_ids
                )
            )
            for profile_ids in (
                plan.numerical_screen_profile_ids,
                plan.hardware_validation_profile_ids,
            )
        )
        build_seconds = sum(
            builds_by_format[weight] for weights in stage_weights for weight in weights
        )
    else:
        build_seconds = _max_stage_build_seconds_per_worker(
            plan.numerical_screen_profile_ids,
            workers=workers,
            profile_by_id=profile_by_id,
            builds_by_format=builds_by_format,
        ) + _max_stage_build_seconds_per_worker(
            plan.hardware_validation_profile_ids,
            workers=workers,
            profile_by_id=profile_by_id,
            builds_by_format=builds_by_format,
        )
    return (
        stack_evidence_hours
        + admission_hours
        + (
            p95_numerical_screen * numerical_screen_evaluations
            + p95_hardware_validation * hardware_validation_evaluations
            + build_seconds
        )
        / 3600.0
    )


def evaluate_preflight_gates(
    manifest: SweepManifest,
    plan: SweepRunPlan,
    prompts: PromptManifest,
    evidence: PreflightEvidence,
) -> PreflightGateReport:
    """Evaluate every launch gate without substituting missing measurements."""

    validate_run_plan(plan, manifest)
    gates: list[GateResult] = []
    gates.append(
        GateResult(
            "manifest_integrity",
            evidence.manifest_hash == manifest.canonical_hash,
            "evidence is bound to the exhaustive manifest",
            len(manifest.entries),
            manifest_declared_space(manifest).expected_total_profiles,
        )
    )
    gates.append(
        GateResult(
            "run_plan_integrity",
            evidence.run_plan_hash == plan.canonical_hash,
            "evidence is bound to the run plan",
        )
    )
    gates.append(
        GateResult(
            "prompt_manifest_integrity",
            evidence.prompt_manifest_hash == prompts.canonical_hash,
            "evidence is bound to the fixed prompt sets",
        )
    )
    gates.append(
        GateResult(
            "execution_topology",
            (
                evidence.numerical_screen_workers == plan.numerical_screen_workers
                and evidence.hardware_validation_workers
                == plan.hardware_validation_workers
            ),
            "worker counts are bound to the immutable run plan",
        )
    )
    admission = evidence.admission_preparation
    admission_identity_valid = (
        admission.manifest_hash == manifest.canonical_hash
        and admission.run_plan_hash == plan.canonical_hash
        and admission.prompt_manifest_hash == prompts.canonical_hash
    )
    admission_coverage_valid = (
        admission.quantized_format_count == len(DECODE_FORMATS)
        and admission.document_count
        == (
            NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count
            + HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
        )
        and admission.artifact_count
        == admission.document_count * (len(DECODE_FORMATS) + 1)
    )
    admission_resources_valid = (
        admission.persistence_policy == "content_addressed_recompute_per_format"
        and admission.logical_artifact_bytes > 0
        and admission.persisted_bytes == 0
        and admission.observed_cold_available_bytes
        >= admission.required_cold_capacity_bytes
        and admission.observed_host_available_bytes >= admission.required_host_bytes
    )
    gates.append(
        GateResult(
            "admission_preparation",
            (
                admission_identity_valid
                and admission_coverage_valid
                and admission_resources_valid
            ),
            (
                f"formats={admission.quantized_format_count}, "
                f"documents={admission.document_count}, "
                f"artifacts={admission.artifact_count}, "
                f"cold_build_seconds={admission.cold_build_seconds:.6g}"
            ),
            admission.artifact_count,
            admission.document_count * (len(DECODE_FORMATS) + 1),
        )
    )
    stack_preparation = evidence.stack_evidence_preparation
    stack_identity_valid = (
        stack_preparation.manifest_hash == manifest.canonical_hash
        and stack_preparation.run_plan_hash == plan.canonical_hash
        and stack_preparation.required_stages == ("compiler", "emulator")
    )
    gates.append(
        GateResult(
            "stack_evidence_preparation",
            stack_identity_valid,
            (
                f"compiler_seconds={stack_preparation.compiler_seconds:.6g}, "
                f"emulator_seconds={stack_preparation.emulator_seconds:.6g}, "
                "critical_path_seconds="
                f"{stack_preparation.critical_path_seconds:.6g}"
            ),
            stack_preparation.critical_path_seconds,
        )
    )
    expected_microbatches = {
        plan.numerical_screen_microbatch_size,
        plan.hardware_validation_microbatch_size,
    }
    observed_microbatches = {
        check.microbatch_size for check in evidence.microbatch_checks
    }
    max_microbatch_error = max(
        (
            max(
                check.max_abs_logit_error,
                check.max_abs_token_nll_error,
                check.max_abs_permutation_nll_error,
            )
            for check in evidence.microbatch_checks
        ),
        default=None,
    )
    invalid_microbatch = any(
        (
            check.profile_id
            not in {
                entry.profile_id
                for entry in manifest.entries
                if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
            }
            or not check.cache_growth_exact
            or not check.lane_isolation_checked
        )
        for check in evidence.microbatch_checks
    )
    gates.append(
        GateResult(
            "microbatch_parity",
            (
                observed_microbatches == expected_microbatches
                and not invalid_microbatch
                and max_microbatch_error is not None
                and max_microbatch_error <= plan.parity_tolerance
            ),
            (
                f"sizes={sorted(observed_microbatches)}, "
                f"max_error={max_microbatch_error}"
            ),
            max_microbatch_error,
            plan.parity_tolerance,
        )
    )
    memory_keys = tuple(
        _runtime_observation_key(
            sample.stage,
            sample.profile_id,
            sample.device_label,
        )
        for sample in evidence.gpu_memory_samples
    )
    runtime_keys_for_memory = tuple(
        _runtime_observation_key(
            sample.stage,
            sample.profile_id,
            sample.device_label,
        )
        for sample in evidence.runtime_samples
    )
    expected_microbatch_by_stage = {
        "numerical-screen": plan.numerical_screen_microbatch_size,
        "hardware-validation": plan.hardware_validation_microbatch_size,
    }
    invalid_memory = tuple(
        sample.profile_id
        for sample in evidence.gpu_memory_samples
        if (
            sample.microbatch_size != expected_microbatch_by_stage[sample.stage]
            or sample.peak_reserved_fraction >= 0.90
        )
    )
    max_reserved_fraction = max(
        (sample.peak_reserved_fraction for sample in evidence.gpu_memory_samples),
        default=None,
    )
    gates.append(
        GateResult(
            "gpu_memory_headroom",
            (
                not invalid_memory
                and sorted(memory_keys) == sorted(runtime_keys_for_memory)
                and max_reserved_fraction is not None
            ),
            (
                f"max_peak_reserved_fraction={max_reserved_fraction}, "
                f"invalid={list(invalid_memory[:3])}"
            ),
            max_reserved_fraction,
            0.90,
        )
    )
    gates.append(
        _set_gate(
            "preflight_completion",
            evidence.completed_profile_ids,
            plan.preflight_profile_ids,
        )
    )

    pilot_ids = set(plan.preflight_profile_ids)
    numerical_screen_samples = tuple(
        sample
        for sample in evidence.runtime_samples
        if sample.stage == "numerical-screen"
    )
    hardware_validation_samples = tuple(
        sample
        for sample in evidence.runtime_samples
        if sample.stage == "hardware-validation"
    )
    gates.append(
        _set_gate(
            "numerical_screen_runtime_coverage",
            (sample.profile_id for sample in numerical_screen_samples),
            plan.preflight_profile_ids,
        )
    )
    legal_pilot_ids = {
        entry.profile_id
        for entry in manifest.entries
        if entry.profile_id in pilot_ids and entry.legality.hardware_candidate
    }
    gates.append(
        _set_gate(
            "hardware_validation_runtime_coverage",
            (sample.profile_id for sample in hardware_validation_samples),
            legal_pilot_ids,
        )
    )
    profile_by_id = {entry.profile_id: entry.profile for entry in manifest.entries}
    preflight_weight_formats = {
        profile_by_id[profile_id].weight_format
        for profile_id in plan.preflight_profile_ids
    }
    gates.append(
        _set_gate(
            "weight_bank_build_coverage",
            (sample.weight_format for sample in evidence.weight_bank_build_samples),
            preflight_weight_formats,
        )
    )

    runtime_keys = tuple(
        _runtime_observation_key(
            sample.stage,
            sample.profile_id,
            sample.device_label,
        )
        for sample in evidence.runtime_samples
    )
    rebinding_keys = tuple(
        _runtime_observation_key(
            sample.stage,
            sample.profile_id,
            sample.device_label,
        )
        for sample in evidence.runtime_rebinding_samples
    )
    gates.append(
        _set_gate(
            "runtime_rebinding_coverage",
            rebinding_keys,
            runtime_keys,
        )
    )
    append_validation_keys = tuple(
        _runtime_observation_key(
            sample.stage,
            sample.profile_id,
            sample.device_label,
        )
        for sample in evidence.native_append_validation_samples
    )
    gates.append(
        _set_gate(
            "native_append_oracle_coverage",
            append_validation_keys,
            runtime_keys,
        )
    )

    invalid_append_validation: list[str] = []
    decoder_layers = int(manifest.model_architecture["num_hidden_layers"])
    for sample in evidence.native_append_validation_samples:
        if not sample.deep_oracle_enabled or sample.mode != "deep_oracle":
            invalid_append_validation.append(f"{sample.profile_id}:oracle-disabled")
            continue
        if sample.expected_calls <= 0:
            invalid_append_validation.append(f"{sample.profile_id}:no-calls")
        if (
            sample.calls != sample.expected_calls
            or sample.tensor_checks != sample.expected_tensor_checks
            or sample.quantized_tensor_checks != sample.expected_quantized_tensor_checks
        ):
            invalid_append_validation.append(f"{sample.profile_id}:coverage-mismatch")
        if sample.expected_tensor_checks != sample.expected_calls * decoder_layers * 2:
            invalid_append_validation.append(f"{sample.profile_id}:layer-role-coverage")
        profile = profile_by_id.get(sample.profile_id)
        if profile is None:
            invalid_append_validation.append(f"{sample.profile_id}:unknown-profile")
        elif profile.kind == PROFILE_KIND_BF16_REFERENCE:
            if sample.expected_quantized_tensor_checks != 0:
                invalid_append_validation.append(
                    f"{sample.profile_id}:bf16-quantized-checks"
                )
        elif sample.expected_quantized_tensor_checks != sample.expected_tensor_checks:
            invalid_append_validation.append(
                f"{sample.profile_id}:quantized-check-coverage"
            )
    gates.append(
        GateResult(
            "native_append_oracle_integrity",
            not invalid_append_validation,
            (
                f"every pilot append passed the full {decoder_layers}-layer K/V oracle"
                if not invalid_append_validation
                else f"invalid={invalid_append_validation[:3]}"
            ),
            sum(
                sample.tensor_checks
                for sample in evidence.native_append_validation_samples
            ),
        )
    )

    from decode_dse.software.precision_bindings import (
        decode_binding_expectations,
    )

    invalid_rebinding: list[str] = []
    binding_expectations = decode_binding_expectations(
        dict(manifest.model_architecture)
    )
    expected_sealed_modules = binding_expectations.sealed_weight_modules
    expected_binding_targets = binding_expectations.binding_targets
    for sample in evidence.runtime_rebinding_samples:
        profile = profile_by_id.get(sample.profile_id)
        is_bf16 = profile is not None and profile.kind == PROFILE_KIND_BF16_REFERENCE
        if profile is None:
            invalid_rebinding.append(f"{sample.profile_id}:unknown-profile")
            continue
        if sample.weight_identity_before != sample.weight_identity_after:
            invalid_rebinding.append(f"{sample.profile_id}:weight-mutated")
        if not sample.used_cached_targets:
            invalid_rebinding.append(f"{sample.profile_id}:uncached-targets")
        if sample.weight_requantizations != 0:
            invalid_rebinding.append(f"{sample.profile_id}:requantized")
        if (
            sample.weight_quantization_events_after
            - sample.weight_quantization_events_before
            != sample.weight_requantizations
        ):
            invalid_rebinding.append(f"{sample.profile_id}:counter-disagreement")
        if is_bf16:
            if (
                sample.performed
                or sample.target_count != 0
                or sample.sealed_weight_modules != 0
                or sample.weight_quantization_events_before != 0
                or sample.weight_quantization_events_after != 0
            ):
                invalid_rebinding.append(f"{sample.profile_id}:bf16-binding")
        elif (
            not sample.performed
            or sample.target_count != expected_binding_targets
            or sample.sealed_weight_modules != expected_sealed_modules
            or sample.weight_quantization_events_before != expected_sealed_modules
            or sample.weight_quantization_events_after != expected_sealed_modules
        ):
            invalid_rebinding.append(f"{sample.profile_id}:binding-or-seal-coverage")
    total_requantizations = sum(
        sample.weight_requantizations for sample in evidence.runtime_rebinding_samples
    )
    gates.append(
        GateResult(
            "runtime_rebinding_integrity",
            not invalid_rebinding,
            (
                "cached targets preserved every decode weight bank"
                if not invalid_rebinding
                else f"invalid={invalid_rebinding[:3]}"
            ),
            total_requantizations,
            0,
        )
    )

    runtime_by_key = {
        _runtime_observation_key(
            sample.stage,
            sample.profile_id,
            sample.device_label,
        ): sample.runtime_seconds
        for sample in evidence.runtime_samples
    }
    binding_exceeds_runtime = tuple(
        sample.profile_id
        for sample in evidence.runtime_rebinding_samples
        if sample.binding_seconds
        > runtime_by_key.get(
            _runtime_observation_key(
                sample.stage,
                sample.profile_id,
                sample.device_label,
            ),
            -1.0,
        )
    )
    gates.append(
        GateResult(
            "runtime_rebinding_accounting",
            not binding_exceeds_runtime,
            (
                "binding time is contained in profile evaluation time"
                if not binding_exceeds_runtime
                else f"inconsistent={list(binding_exceeds_runtime[:3])}"
            ),
        )
    )

    allowed_devices = set(plan.device_labels)
    observed_devices = {sample.device_label for sample in evidence.runtime_samples}
    observed_devices.update(
        sample.device_label for sample in evidence.runtime_rebinding_samples
    )
    observed_devices.update(
        sample.device_label for sample in evidence.weight_bank_build_samples
    )
    unknown_devices = sorted(observed_devices - allowed_devices)
    gates.append(
        GateResult(
            "runtime_device_labels",
            not unknown_devices,
            (
                "runtime samples use planned devices"
                if not unknown_devices
                else f"unplanned device labels: {unknown_devices}"
            ),
        )
    )

    projected_hours: float | None = None
    evaluation_hours: float | None = None
    bank_build_hours: float | None = None
    rebinding_hours: float | None = None
    homogeneous_fallback: str | None = None
    admission_hours = admission.cold_build_seconds / 3600.0
    stack_evidence_hours = stack_preparation.critical_path_seconds / 3600.0
    projection_error: str | None = None
    try:
        p95_numerical_screen = _percentile_nearest_rank(
            (sample.runtime_seconds for sample in numerical_screen_samples), 0.95
        )
        p95_hardware_validation = _percentile_nearest_rank(
            (sample.runtime_seconds for sample in hardware_validation_samples), 0.95
        )
        numerical_screen_shard_profiles = _max_stage_profiles_per_worker(
            plan.numerical_screen_profile_ids,
            workers=evidence.numerical_screen_workers,
            profile_by_id=profile_by_id,
        )
        hardware_validation_shard_profiles = _max_stage_profiles_per_worker(
            plan.hardware_validation_profile_ids,
            workers=evidence.hardware_validation_workers,
            profile_by_id=profile_by_id,
        )
        evaluation_hours = (
            p95_numerical_screen * numerical_screen_shard_profiles
            + p95_hardware_validation * hardware_validation_shard_profiles
        ) / 3600.0
        builds_by_format = {
            sample.weight_format: sample.build_seconds
            for sample in evidence.weight_bank_build_samples
        }
        if len(builds_by_format) != len(evidence.weight_bank_build_samples):
            raise ValueError("duplicate weight-bank build samples")
        numerical_screen_weight_formats = {
            profile_by_id[profile_id].weight_format
            for profile_id in plan.numerical_screen_profile_ids
        }
        hardware_validation_weight_formats = {
            profile_by_id[profile_id].weight_format
            for profile_id in plan.hardware_validation_profile_ids
        }
        required_builds = (
            numerical_screen_weight_formats | hardware_validation_weight_formats
        )
        missing_builds = required_builds - set(builds_by_format)
        if missing_builds:
            raise ValueError(
                f"missing weight-bank build samples: {sorted(missing_builds)}"
            )
        if evidence.weight_bank_build_serialized:
            bank_build_seconds = sum(
                builds_by_format[weight]
                for weights in (
                    numerical_screen_weight_formats,
                    hardware_validation_weight_formats,
                )
                for weight in weights
            )
        else:
            bank_build_seconds = _max_stage_build_seconds_per_worker(
                plan.numerical_screen_profile_ids,
                workers=evidence.numerical_screen_workers,
                profile_by_id=profile_by_id,
                builds_by_format=builds_by_format,
            ) + _max_stage_build_seconds_per_worker(
                plan.hardware_validation_profile_ids,
                workers=evidence.hardware_validation_workers,
                profile_by_id=profile_by_id,
                builds_by_format=builds_by_format,
            )
        bank_build_hours = bank_build_seconds / 3600.0

        numerical_screen_bindings = tuple(
            sample.binding_seconds
            for sample in evidence.runtime_rebinding_samples
            if sample.stage == "numerical-screen"
        )
        hardware_validation_bindings = tuple(
            sample.binding_seconds
            for sample in evidence.runtime_rebinding_samples
            if sample.stage == "hardware-validation"
        )
        p95_numerical_screen_binding = _percentile_nearest_rank(
            numerical_screen_bindings, 0.95
        )
        p95_hardware_validation_binding = _percentile_nearest_rank(
            hardware_validation_bindings, 0.95
        )
        rebinding_hours = (
            p95_numerical_screen_binding * numerical_screen_shard_profiles
            + p95_hardware_validation_binding * hardware_validation_shard_profiles
        ) / 3600.0
        projected_hours = (
            stack_evidence_hours + admission_hours + evaluation_hours + bank_build_hours
        )
        if len(plan.device_labels) == 1:
            max_homogeneous_workers = min(
                len(numerical_screen_weight_formats),
                len(hardware_validation_weight_formats),
            )
            alternatives = tuple(
                (
                    workers,
                    _project_homogeneous_runtime_hours(
                        plan=plan,
                        profile_by_id=profile_by_id,
                        builds_by_format=builds_by_format,
                        p95_numerical_screen=p95_numerical_screen,
                        p95_hardware_validation=p95_hardware_validation,
                        admission_hours=admission_hours,
                        stack_evidence_hours=stack_evidence_hours,
                        workers=workers,
                        serialized_builds=(evidence.weight_bank_build_serialized),
                    ),
                )
                for workers in range(1, max_homogeneous_workers + 1)
            )
            feasible = tuple(
                item for item in alternatives if item[1] <= plan.max_projected_hours
            )
            if feasible:
                workers, hours = feasible[0]
                homogeneous_fallback = (
                    f"minimum homogeneous whole-bank topology is {workers} "
                    f"workers at {hours:.6g} hours"
                )
            else:
                workers, hours = min(
                    alternatives,
                    key=lambda item: item[1],
                )
                homogeneous_fallback = (
                    f"whole-bank sharding reaches {hours:.6g} hours at its "
                    f"{workers}-worker limit; extend the budget or use a "
                    "separately validated replicated-bank plan"
                )
    except ValueError as error:
        projection_error = str(error)
    gates.append(
        GateResult(
            "projected_rebinding_runtime",
            (
                rebinding_hours is not None
                and rebinding_hours <= plan.max_projected_hours
            ),
            (
                f"p95 rebinding projection is {rebinding_hours:.6g} hours"
                if rebinding_hours is not None
                else f"rebinding measurements are incomplete: {projection_error}"
            ),
            rebinding_hours,
            plan.max_projected_hours,
        )
    )
    gates.append(
        GateResult(
            "projected_runtime",
            (
                projected_hours is not None
                and projected_hours <= plan.max_projected_hours
            ),
            (
                (
                    f"p95 projection is {projected_hours:.6g} hours "
                    f"(stack_evidence={stack_evidence_hours:.6g}, "
                    f"admission={admission_hours:.6g}, "
                    f"evaluation={evaluation_hours:.6g}, "
                    f"weight_banks={bank_build_hours:.6g}, "
                    f"serialized_weight_builds="
                    f"{evidence.weight_bank_build_serialized})"
                    + (
                        f"; {homogeneous_fallback}"
                        if homogeneous_fallback is not None
                        else ""
                    )
                )
                if projected_hours is not None
                else f"runtime measurements are incomplete: {projection_error}"
            ),
            projected_hours,
            plan.max_projected_hours,
        )
    )

    bank_by_weight: dict[str, list[NLLParityCheck]] = {}
    invalid_bank_profile = False
    for check in evidence.weight_bank_checks:
        profile = profile_by_id.get(check.profile_id)
        if (
            profile is None
            or check.profile_id not in pilot_ids
            or {check.left_label, check.right_label} != {"fresh_process", "reused_bank"}
        ):
            invalid_bank_profile = True
            continue
        bank_by_weight.setdefault(profile.weight_format, []).append(check)
    bank_errors = tuple(
        check.absolute_error for checks in bank_by_weight.values() for check in checks
    )
    bank_coverage = set(bank_by_weight) == set(DECODE_FORMATS)
    max_bank_error = max(bank_errors, default=None)
    gates.append(
        GateResult(
            "reused_weight_bank_parity",
            (
                bank_coverage
                and not invalid_bank_profile
                and max_bank_error is not None
                and max_bank_error <= plan.parity_tolerance
            ),
            (
                f"covered={sorted(bank_by_weight)}, "
                + (
                    f"max_abs_nll_error={max_bank_error:.6g}"
                    if max_bank_error is not None
                    else "max_abs_nll_error=missing"
                )
            ),
            max_bank_error,
            plan.parity_tolerance,
        )
    )

    required_pairs = {
        frozenset(pair) for pair in itertools.combinations(plan.device_labels, 2)
    }
    observed_pairs: set[frozenset[str]] = set()
    profiles_by_pair: dict[frozenset[str], set[str]] = {}
    device_errors: list[float] = []
    invalid_device_check = False
    for check in evidence.cross_device_checks:
        pair = frozenset((check.left_label, check.right_label))
        if (
            len(pair) != 2
            or not pair.issubset(allowed_devices)
            or check.profile_id not in pilot_ids
        ):
            invalid_device_check = True
            continue
        observed_pairs.add(pair)
        profiles_by_pair.setdefault(pair, set()).add(check.profile_id)
        device_errors.append(check.absolute_error)
    max_device_error = max(device_errors, default=0.0)
    common_anchor = (
        set.intersection(*profiles_by_pair.values()) if profiles_by_pair else set()
    )
    device_passed = (
        observed_pairs == required_pairs
        and not invalid_device_check
        and max_device_error <= plan.parity_tolerance
        and (not required_pairs or bool(common_anchor))
    )
    gates.append(
        GateResult(
            "cross_device_parity",
            device_passed,
            (
                f"pairs={len(observed_pairs)}/{len(required_pairs)}, "
                f"common_anchors={len(common_anchor)}, "
                f"max_abs_nll_error={max_device_error:.6g}"
            ),
            max_device_error,
            plan.parity_tolerance,
        )
    )

    numerical_screen_documents = {
        record.document_id for record in prompts.numerical_screen
    }
    split_documents = {check.document_id for check in evidence.bf16_split_checks}
    split_max = max(
        (
            max(check.max_abs_logit_error, check.mean_token_nll_abs_error)
            for check in evidence.bf16_split_checks
        ),
        default=None,
    )
    split_passed = (
        split_documents == numerical_screen_documents
        and len(evidence.bf16_split_checks) == len(numerical_screen_documents)
        and all(
            isinstance(check.token_ids_equal, bool) and check.token_ids_equal
            for check in evidence.bf16_split_checks
        )
        and split_max is not None
        and split_max <= plan.parity_tolerance
    )
    gates.append(
        GateResult(
            "bf16_split_parity",
            split_passed,
            (
                f"documents={len(split_documents)}/{len(numerical_screen_documents)}, "
                + (
                    f"max_error={split_max:.6g}"
                    if split_max is not None
                    else "max_error=missing"
                )
            ),
            split_max,
            plan.parity_tolerance,
        )
    )

    cache_documents = {check.document_id for check in evidence.cache_reuse_checks}
    cache_max = max(
        (check.mean_token_nll_abs_error for check in evidence.cache_reuse_checks),
        default=None,
    )
    cache_passed = (
        cache_documents == numerical_screen_documents
        and len(evidence.cache_reuse_checks) == len(numerical_screen_documents)
        and all(
            isinstance(check.cache_content_equal, bool) and check.cache_content_equal
            for check in evidence.cache_reuse_checks
        )
        and cache_max is not None
        and cache_max <= plan.parity_tolerance
    )
    gates.append(
        GateResult(
            "cache_snapshot_reuse",
            cache_passed,
            (
                f"documents={len(cache_documents)}/{len(numerical_screen_documents)}, "
                + (
                    f"max_abs_nll_error={cache_max:.6g}"
                    if cache_max is not None
                    else "max_abs_nll_error=missing"
                )
            ),
            cache_max,
            plan.parity_tolerance,
        )
    )
    return PreflightGateReport(
        manifest_hash=manifest.canonical_hash,
        run_plan_hash=plan.canonical_hash,
        gates=tuple(gates),
    )


def make_stage_manifest(
    manifest: SweepManifest,
    profile_ids: Sequence[str],
) -> SweepManifest:
    """Build a contiguous stage manifest while preserving canonical profiles."""

    by_id = {entry.profile_id: entry for entry in manifest.entries}
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("stage schedule contains duplicate profile IDs")
    unknown = set(profile_ids) - set(by_id)
    if unknown:
        raise ValueError(f"stage schedule contains unknown profiles: {sorted(unknown)}")
    entries = tuple(
        SweepManifestEntry(
            ordinal=ordinal,
            profile=by_id[profile_id].profile,
            legality=by_id[profile_id].legality,
            validity=by_id[profile_id].validity,
        )
        for ordinal, profile_id in enumerate(profile_ids)
    )
    return SweepManifest(
        model_name=manifest.model_name,
        model_revision=manifest.model_revision,
        model_architecture=manifest.model_architecture,
        tokenizer_revision=manifest.tokenizer_revision,
        quantizer_provenance=manifest.quantizer_provenance,
        entries=entries,
    )


def _sha256_source(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_quantizer_provenance(
    repository: Path,
    config: Mapping[str, Any],
) -> QuantizerProvenance:
    """Capture every arithmetic source and the imports selected by decode."""

    repository = repository.resolve()
    workspace = repository.parent
    executor_config = config.get("executor")
    if not isinstance(executor_config, Mapping):
        raise ValueError("config.executor is required")
    configured_mase = executor_config.get("mase_src")
    if not configured_mase:
        raise ValueError("executor.mase_src is required")
    mase_root = Path(str(configured_mase))
    if not mase_root.is_absolute():
        mase_root = repository / mase_root
    mase_root = mase_root.resolve()

    roots: tuple[tuple[str, Path], ...] = (
        (
            "mase_quantizers",
            mase_root / "chop" / "nn" / "quantizers",
        ),
        (
            "mase_decode_arithmetic",
            mase_root / "chop" / "nn" / "quantized",
        ),
        (
            "mase_quantize_wiring",
            mase_root / "chop" / "passes" / "module" / "transforms" / "quantize",
        ),
        (
            "plena_rtl_quantizers",
            workspace / "PLENA_RTL" / "PLENA_Tools" / "plena_quant",
        ),
        (
            "plena_simulator_quantizers",
            workspace / "PLENA_Simulator" / "PLENA_Tools" / "plena_quant",
        ),
    )
    selected: list[tuple[str, Path]] = []
    for component, root in roots:
        if not root.is_dir():
            raise ValueError(f"quantizer source root does not exist: {root}")
        files = tuple(
            path
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
        if not files:
            raise ValueError(f"quantizer source root contains no Python files: {root}")
        selected.extend((component, path) for path in files)
    for path in (
        mase_root / "chop" / "passes" / "module" / "module_modify_helper.py",
        mase_root / "chop" / "passes" / "module" / "state_dict_map.py",
        repository / "decode_dse" / "software" / "precision_bindings.py",
    ):
        if not path.is_file():
            raise ValueError(f"quantizer wiring source does not exist: {path}")
        component = (
            "plena_software_binding"
            if repository in path.parents
            else "mase_quantize_wiring"
        )
        selected.append((component, path))

    sources = tuple(
        sorted(
            (
                QuantizerSource(
                    component=component,
                    path=path.resolve().relative_to(workspace).as_posix(),
                    sha256=_sha256_source(path),
                )
                for component, path in selected
            ),
            key=lambda source: (source.component, source.path),
        )
    )
    import_paths = {
        "chop.nn.quantized.functional.vector": (
            mase_root / "chop/nn/quantized/functional/vector.py"
        ),
        "chop.nn.quantized.modules.qwen3": (
            mase_root / "chop/nn/quantized/modules/qwen3/__init__.py"
        ),
        "chop.nn.quantizers._minifloat_mx.fake": (
            mase_root / "chop/nn/quantizers/_minifloat_mx/fake.py"
        ),
        "chop.nn.quantizers.mxfp.fake": (mase_root / "chop/nn/quantizers/mxfp/fake.py"),
        "chop.nn.quantizers.mxint.fake": (
            mase_root / "chop/nn/quantizers/mxint/fake.py"
        ),
        "chop.passes.module.transforms.quantize.quantize": (
            mase_root / "chop/passes/module/transforms/quantize/quantize.py"
        ),
    }
    missing = tuple(path for path in import_paths.values() if not path.is_file())
    if missing:
        raise ValueError(f"resolved quantizer imports do not exist: {missing}")
    resolved_imports = tuple(
        ResolvedImportOrigin(
            module=module,
            path=path.resolve().relative_to(workspace).as_posix(),
        )
        for module, path in sorted(import_paths.items())
    )
    source_paths = {source.path for source in sources}
    if any(origin.path not in source_paths for origin in resolved_imports):
        raise ValueError("resolved quantizer imports must be hashed sources")
    return QuantizerProvenance(
        sources=sources,
        resolved_imports=resolved_imports,
    )


def _software_tree_hash(repository: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in (repository / "decode_dse").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".sh"}
        and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _mase_tree_hash(
    repository: Path,
    config: Mapping[str, Any],
) -> str:
    executor_config = config.get("executor")
    if not isinstance(executor_config, Mapping):
        raise ValueError("config.executor is required")
    configured = executor_config.get("mase_src")
    if not configured:
        raise ValueError("executor.mase_src is required")
    root = Path(str(configured))
    if not root.is_absolute():
        root = repository / root
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"MASE source tree does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json"}
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"MASE source tree contains no tracked source files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def profile_to_decode_quant_spec(
    profile: DecodePrecisionProfile,
) -> Any | None:
    """Map a canonical numerical profile to the decode quantizer binding."""

    if profile.kind == PROFILE_KIND_BF16_REFERENCE:
        return None
    from decode_dse.software.precision_bindings import DecodeQuantSpec

    def operand(token: str) -> tuple[str, Any]:
        descriptor = format_descriptor(token)
        if descriptor.family == "mxint":
            return "mxint", descriptor.element_bits
        if descriptor.family == "mxfp":
            return "mxfp", (descriptor.exponent_bits, descriptor.mantissa_bits)
        raise ValueError(f"{token!r} is not an MX operand format")

    weight_family, weight_width = operand(profile.weight_format)
    activation_family, activation_width = operand(profile.activation_format)
    kv_family, kv_width = operand(profile.kv_format)
    return DecodeQuantSpec(
        attn_w=weight_width,
        ffn_w=weight_width,
        kv=kv_width,
        w_fmt=weight_family,
        kv_fmt=kv_family,
        weight_block=profile.block_size,
        kv_block=profile.block_size,
        act_w=activation_width,
        act_fmt=activation_family,
        act_block=profile.block_size,
        use_gptq=False,
        use_rotation=False,
        fp_setting=profile.vector_format,
        fp_setting_attention=True,
        quant_attn_internals=True,
        matrix_mlen=profile.matrix_mlen,
    )


@dataclass(frozen=True)
class ExecutorContext:
    """Pinned resources provided to an injected evaluation executor."""

    stage: str
    workspace_root: Path
    output_dir: Path
    config: Mapping[str, Any]
    master_manifest: SweepManifest
    stage_manifest: SweepManifest
    run_plan: SweepRunPlan
    prompts: PromptManifest
    sample_contract: StageSampleContract
    shard_index: int
    shard_count: int
    device_label: str


__all__ = [
    "ExecutorContext",
    "profile_to_decode_quant_spec",
    "BF16SplitCheck",
    "CacheReuseCheck",
    "GateResult",
    "GPUBaselinePlan",
    "MAX_PROJECTED_HOURS",
    "NativeAppendValidationSample",
    "NLLParityCheck",
    "PARITY_TOLERANCE",
    "PREFLIGHT_PROFILE_COUNT",
    "PreflightEvidence",
    "PreflightGateReport",
    "PromptManifest",
    "PromptRecord",
    "RuntimeRebindingSample",
    "RuntimeSample",
    "NUMERICAL_SCREEN_PROFILE_COUNT",
    "NUMERICAL_SCREEN_SAMPLE_CONTRACT",
    "HARDWARE_VALIDATION_PROFILE_COUNT",
    "HARDWARE_VALIDATION_SAMPLE_CONTRACT",
    "StageSampleContract",
    "SweepRunPlan",
    "WeightBankBuildSample",
    "build_run_plan",
    "build_quantizer_provenance",
    "evaluate_preflight_gates",
    "load_immutable_json",
    "load_preflight_evidence",
    "load_prompt_manifest",
    "make_stage_manifest",
    "preflight_required_features",
    "select_preflight_entries",
    "manifest_declared_space",
    "validate_exhaustive_manifest",
    "validate_run_plan",
    "write_immutable_json",
]
