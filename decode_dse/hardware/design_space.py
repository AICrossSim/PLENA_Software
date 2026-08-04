"""Exact, provenance-bound hardware enumeration for decode results."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Protocol, Sequence

from decode_dse.legality import (
    DEFAULT_PACKED_KV_TARGET,
    PackedKVRuntimeTarget,
    evaluate_stack_capability,
    scope_stack_validity,
)
from decode_dse.hardware.admission_cost import (
    admission_correctness_status_valid,
)
from decode_dse.hardware.lm_head_service import (
    BF16HeadServiceEstimate,
    HEAD_SERVICE_MODE,
    head_service_status_valid,
)
from decode_dse.legality import StackValidity
from decode_dse.manifest import SweepManifest, SweepManifestEntry
from decode_dse.profiles import (
    PROFILE_KIND_BF16_REFERENCE,
    VECTOR_FORMATS,
    format_descriptor,
)

HARDWARE_STUDY_SCHEMA = "decode-hardware-study"
HARDWARE_RESULT_SCHEMA = "decode-hardware-result"
HARDWARE_FACTOR_RESULT_SCHEMA = "decode-hardware-factor-result"
HARDWARE_FACTOR_BINDING_SCHEMA = "decode-hardware-factor-binding"
HARDWARE_ARTIFACT_SCHEMA = "decode-hardware-artifact"
HARDWARE_STORAGE_REVISION = "factorized-exact"
LEGACY_COMPACT_STORAGE_REVISION = "compact-exact"
HARDWARE_SCATTER_SAMPLE_LIMIT = 4096
HARDWARE_RETENTION_LABELS = frozenset(
    {
        "profile_frontier",
        "profile_fastest",
        "profile_lowest_energy",
        "profile_best_edp",
        "exact_frontier",
        "exact_fastest",
        "exact_lowest_energy",
        "exact_best_edp",
        "sampled_dominated",
        "sampled_unrankable",
        "legacy_full_row",
    }
)
PHYSICAL_TRAFFIC_KEYS = frozenset(
    {
        "weight_element_read_bytes",
        "weight_scale_read_bytes",
        "bf16_weight_read_bytes",
        "activation_read_bytes",
        "activation_write_bytes",
        "kv_element_read_bytes",
        "kv_scale_read_bytes",
        "kv_element_write_bytes",
        "kv_scale_write_bytes",
    }
)
_VALIDITY_FIELDS = (
    "software_valid",
    "compiler_valid",
    "emulator_valid",
    "rtl_valid",
    "dc_calibrated",
)
STEP_COMPOSITION = "max_compute_memory"
COMPILER_TRACE_EXECUTION_MODE = "compiler_trace"
LEGACY_AGGREGATE_BANDWIDTH_MODE = "legacy_aggregate_bandwidth"
COMPILER_TRACE_TIMING_SET_SCHEMA = "plena-compiler-trace-timing-set-v1"
FULL_MODEL_DECODE_SCOPE = (
    "full_model_decode_step_independent_request_batch"
)
DECODE_EXECUTION_MODES = frozenset(
    {
        COMPILER_TRACE_EXECUTION_MODE,
        LEGACY_AGGREGATE_BANDWIDTH_MODE,
    }
)

CHIP_COUNTS = (1, 2, 4, 8, 16)
SRAM_POLICIES = (
    "streaming",
    "projection_resident",
    "kv_resident_25",
    "kv_resident_50",
    "kv_resident_75",
    "kv_resident_100",
)
DEFAULT_REFERENCE_CHIP_COUNT = 4
DEFAULT_REFERENCE_DIE_AREA_MM2 = 826.0
DEFAULT_REFERENCE_HBM_CAPACITY_BYTES = 80_000_000_000
DEFAULT_REFERENCE_HBM_BANDWIDTH_BYTES_PER_S = 2.039e12
DEFAULT_AREA_MARGIN = 1.10
DEFAULT_FP_SRAM_DEPTH = 512
SOFTMAX_CONSTANT_SLOTS = 6
SOFTMAX_STATE_VALUES_PER_ROW = 3

def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + suffix
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_finite(name: str, value: float, *, positive: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _canonical_trace_timing_set(
    execution_mode: str,
    evidence: Mapping[str, Any] | None,
    timing_evidence_id: str | None,
) -> dict[str, Any] | None:
    """Validate the sealed request set carried by compiler-trace timing."""

    if execution_mode not in DECODE_EXECUTION_MODES:
        raise ValueError("unsupported decode execution mode")
    if execution_mode == LEGACY_AGGREGATE_BANDWIDTH_MODE:
        if evidence is not None:
            raise ValueError("legacy timing cannot carry compiler-trace evidence")
        return None
    if not isinstance(evidence, Mapping):
        raise ValueError("compiler timing requires request-set evidence")
    canonical = json.loads(_canonical_bytes(dict(evidence)))
    required = {
        "schema_version",
        "execution_mode",
        "artifact_scope",
        "request_count",
        "compiler_input_descriptor_sha256",
        "compiler_lowering_key_sha256",
        "compiler_artifact_set_sha256",
        "request_set_sha256",
        "compiler_source_sha256",
        "latency_library_sha256",
        "request_memory_sidecar_set_sha256",
        "request_memory_calibration_ids",
        "step_composition",
    }
    if set(canonical) != required:
        raise ValueError("compiler timing request-set fields differ from schema")
    if canonical["schema_version"] != COMPILER_TRACE_TIMING_SET_SCHEMA:
        raise ValueError("unsupported compiler timing request-set schema")
    if canonical["execution_mode"] != COMPILER_TRACE_EXECUTION_MODE:
        raise ValueError("compiler timing execution mode is inconsistent")
    if canonical["artifact_scope"] != FULL_MODEL_DECODE_SCOPE:
        raise ValueError("compiler timing lacks full-model decode scope")
    request_count = canonical["request_count"]
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count <= 0
    ):
        raise ValueError("compiler timing request count must be positive")
    for name in (
        "compiler_input_descriptor_sha256",
        "compiler_lowering_key_sha256",
        "compiler_artifact_set_sha256",
        "request_set_sha256",
        "compiler_source_sha256",
        "latency_library_sha256",
        "request_memory_sidecar_set_sha256",
    ):
        value = canonical[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    calibration_ids = canonical["request_memory_calibration_ids"]
    if (
        not isinstance(calibration_ids, list)
        or not calibration_ids
        or calibration_ids != sorted(set(calibration_ids))
        or any(
            not isinstance(value, str)
            or not value.startswith("request-latency-")
            or len(value) != len("request-latency-") + 64
            or any(
                character not in "0123456789abcdef"
                for character in value[len("request-latency-") :]
            )
            for value in calibration_ids
        )
    ):
        raise ValueError("request-memory calibration identities are invalid")
    if canonical["step_composition"] != STEP_COMPOSITION:
        raise ValueError("compiler timing step composition is unsupported")
    expected_id = "compiler-trace-timing-" + _content_hash(canonical)
    if timing_evidence_id != expected_id:
        raise ValueError("compiler timing evidence identity is inconsistent")
    return canonical


def _ordered_positive(values: Sequence[int], name: str) -> tuple[int, ...]:
    ordered = tuple(sorted({int(value) for value in values}))
    if not ordered or any(value <= 0 for value in ordered):
        raise ValueError(f"{name} must contain positive integers")
    return ordered


def _positive_sequence(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (int(value),)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an integer or integer sequence")
    return _ordered_positive(value, name)


def _boolean_sequence(value: Any, name: str) -> tuple[bool, ...]:
    if isinstance(value, bool):
        value = (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a boolean or boolean sequence")
    if not value or any(not isinstance(item, bool) for item in value):
        raise TypeError(f"{name} must contain booleans")
    return tuple(item for item in (False, True) if item in value)


def _normalise_sram_policy(value: str) -> str:
    token = str(value).strip().casefold().replace("-", "_").replace("%", "")
    aliases = {
        "projection": "projection_resident",
        "25": "kv_resident_25",
        "50": "kv_resident_50",
        "75": "kv_resident_75",
        "100": "kv_resident_100",
        "kv_25": "kv_resident_25",
        "kv_50": "kv_resident_50",
        "kv_75": "kv_resident_75",
        "kv_100": "kv_resident_100",
    }
    token = aliases.get(token, token)
    if token not in SRAM_POLICIES:
        raise ValueError(f"unsupported SRAM policy {value!r}")
    return token


@dataclass(frozen=True)
class ResourceBudget:
    """Matched aggregate limits used by the hardware study."""

    aggregate_area_limit_mm2: float = (
        DEFAULT_REFERENCE_CHIP_COUNT
        * DEFAULT_AREA_MARGIN
        * DEFAULT_REFERENCE_DIE_AREA_MM2
    )
    aggregate_hbm_capacity_limit_bytes: int = (
        DEFAULT_REFERENCE_CHIP_COUNT
        * DEFAULT_REFERENCE_HBM_CAPACITY_BYTES
    )
    aggregate_hbm_bandwidth_limit_bytes_per_s: float = (
        DEFAULT_REFERENCE_CHIP_COUNT
        * DEFAULT_REFERENCE_HBM_BANDWIDTH_BYTES_PER_S
    )
    reference_system: str = "A100x4"

    def __post_init__(self) -> None:
        _require_finite(
            "aggregate_area_limit_mm2",
            float(self.aggregate_area_limit_mm2),
            positive=True,
        )
        if (
            isinstance(self.aggregate_hbm_capacity_limit_bytes, bool)
            or not isinstance(self.aggregate_hbm_capacity_limit_bytes, int)
            or self.aggregate_hbm_capacity_limit_bytes <= 0
        ):
            raise ValueError(
                "aggregate_hbm_capacity_limit_bytes must be positive"
            )
        _require_finite(
            "aggregate_hbm_bandwidth_limit_bytes_per_s",
            float(self.aggregate_hbm_bandwidth_limit_bytes_per_s),
            positive=True,
        )
        if not self.reference_system:
            raise ValueError("reference_system must be non-empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ResourceBudget":
        raw = dict(value or {})
        return cls(
            aggregate_area_limit_mm2=float(
                raw.get(
                    "aggregate_area_limit_mm2",
                    cls.aggregate_area_limit_mm2,
                )
            ),
            aggregate_hbm_capacity_limit_bytes=int(
                raw.get(
                    "aggregate_hbm_capacity_limit_bytes",
                    cls.aggregate_hbm_capacity_limit_bytes,
                )
            ),
            aggregate_hbm_bandwidth_limit_bytes_per_s=float(
                raw.get(
                    "aggregate_hbm_bandwidth_limit_bytes_per_s",
                    cls.aggregate_hbm_bandwidth_limit_bytes_per_s,
                )
            ),
            reference_system=str(
                raw.get("reference_system", cls.reference_system)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_system": self.reference_system,
            "aggregate_area_limit_mm2": self.aggregate_area_limit_mm2,
            "aggregate_hbm_capacity_limit_bytes": (
                self.aggregate_hbm_capacity_limit_bytes
            ),
            "aggregate_hbm_bandwidth_limit_bytes_per_s": (
                self.aggregate_hbm_bandwidth_limit_bytes_per_s
            ),
        }


@dataclass(frozen=True)
class ResourceBudgetStatus:
    """Aggregate resource use and hard-constraint outcomes."""

    aggregate_area_mm2: float
    aggregate_hbm_capacity_bytes: int
    aggregate_hbm_bandwidth_bytes_per_s: float
    aggregate_multiplier_count: int
    budget: ResourceBudget = ResourceBudget()

    def __post_init__(self) -> None:
        _require_finite(
            "aggregate_area_mm2",
            float(self.aggregate_area_mm2),
            positive=True,
        )
        if (
            isinstance(self.aggregate_hbm_capacity_bytes, bool)
            or not isinstance(self.aggregate_hbm_capacity_bytes, int)
            or self.aggregate_hbm_capacity_bytes <= 0
        ):
            raise ValueError("aggregate_hbm_capacity_bytes must be positive")
        _require_finite(
            "aggregate_hbm_bandwidth_bytes_per_s",
            float(self.aggregate_hbm_bandwidth_bytes_per_s),
            positive=True,
        )
        if (
            isinstance(self.aggregate_multiplier_count, bool)
            or not isinstance(self.aggregate_multiplier_count, int)
            or self.aggregate_multiplier_count <= 0
        ):
            raise ValueError("aggregate_multiplier_count must be positive")

    @property
    def area_feasible(self) -> bool:
        return self.aggregate_area_mm2 <= self.budget.aggregate_area_limit_mm2

    @property
    def hbm_capacity_feasible(self) -> bool:
        return (
            self.aggregate_hbm_capacity_bytes
            <= self.budget.aggregate_hbm_capacity_limit_bytes
        )

    @property
    def hbm_bandwidth_feasible(self) -> bool:
        return (
            self.aggregate_hbm_bandwidth_bytes_per_s
            <= self.budget.aggregate_hbm_bandwidth_limit_bytes_per_s
        )

    @property
    def feasible(self) -> bool:
        return (
            self.area_feasible
            and self.hbm_capacity_feasible
            and self.hbm_bandwidth_feasible
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.budget.to_dict(),
            "aggregate_area_mm2": self.aggregate_area_mm2,
            "aggregate_hbm_capacity_bytes": self.aggregate_hbm_capacity_bytes,
            "aggregate_hbm_bandwidth_bytes_per_s": (
                self.aggregate_hbm_bandwidth_bytes_per_s
            ),
            "aggregate_multiplier_count": self.aggregate_multiplier_count,
            "area_feasible": self.area_feasible,
            "hbm_capacity_feasible": self.hbm_capacity_feasible,
            "hbm_bandwidth_feasible": self.hbm_bandwidth_feasible,
            "feasible": self.feasible,
        }


@dataclass(frozen=True)
class HardwareCandidate:
    """One structurally legal decode-chip geometry and serving point."""

    LEGACY_FIELDS: ClassVar[tuple[str, ...]] = (
        "MLEN",
        "BLEN",
        "VLEN",
        "HLEN",
        "BATCH",
        "HBM_CHANNELS",
        "HBM_GENERATION",
        "CHIP_COUNT",
    )
    PRE_E2_FIELDS: ClassVar[tuple[str, ...]] = LEGACY_FIELDS + (
        "TP",
        "KVP",
        "LINK_PORTS",
        "SRAM_POLICY",
    )
    E2_FIELDS: ClassVar[tuple[str, ...]] = PRE_E2_FIELDS + (
        "KV_HEAD_REUSE",
        "DRAIN_OVERLAPPED",
    )

    mlen: int
    blen: int
    vlen: int
    hlen: int
    batch: int
    hbm_channels: int
    hbm_generation: str
    chip_count: int = 1
    tp: int | None = None
    kvp: int | None = None
    link_ports: int | None = None
    sram_policy: str = "streaming"
    kv_head_reuse: bool | None = None
    drain_overlapped: bool | None = None
    _architecture_knobs_explicit: bool = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "mlen",
            "blen",
            "vlen",
            "hlen",
            "batch",
            "hbm_channels",
            "chip_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        tp = self.tp
        kvp = self.kvp
        if tp is None and kvp is None:
            tp, kvp = self.chip_count, 1
        elif tp is None:
            if self.chip_count % int(kvp):
                raise ValueError("KVP must divide CHIP_COUNT")
            tp = self.chip_count // int(kvp)
        elif kvp is None:
            if self.chip_count % int(tp):
                raise ValueError("TP must divide CHIP_COUNT")
            kvp = self.chip_count // int(tp)
        for name, value in (("tp", tp), ("kvp", kvp)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if tp * kvp != self.chip_count:
            raise ValueError("CHIP_COUNT must equal TP * KVP")
        object.__setattr__(self, "tp", tp)
        object.__setattr__(self, "kvp", kvp)
        ports = self.link_ports
        if ports is None:
            ports = 0 if self.chip_count == 1 else (int(tp > 1) + int(kvp > 1))
        if isinstance(ports, bool) or not isinstance(ports, int) or ports < 0:
            raise ValueError("link_ports must be non-negative")
        required_ports = int(tp > 1) + int(kvp > 1)
        if self.chip_count == 1 and ports != 0:
            raise ValueError("a single-chip candidate cannot have link ports")
        if self.chip_count > 1 and ports < required_ports:
            raise ValueError(
                "link_ports cannot serve every active parallel dimension"
            )
        object.__setattr__(self, "link_ports", ports)
        object.__setattr__(
            self,
            "sram_policy",
            _normalise_sram_policy(self.sram_policy),
        )
        knobs_explicit = (
            self.kv_head_reuse is not None
            or self.drain_overlapped is not None
        )
        for name in ("kv_head_reuse", "drain_overlapped"):
            value = getattr(self, name)
            if value is None:
                value = False
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "_architecture_knobs_explicit",
            knobs_explicit,
        )
        if self.vlen != self.mlen:
            raise ValueError("the decode search requires VLEN == MLEN")
        if not self.hbm_generation:
            raise ValueError("hbm_generation must be non-empty")

    @property
    def architecture_knobs_explicit(self) -> bool:
        return self._architecture_knobs_explicit

    def to_pre_e2_dict(self) -> dict[str, Any]:
        """Return the topology-aware representation before E2 knobs."""

        return {
            "MLEN": self.mlen,
            "BLEN": self.blen,
            "VLEN": self.vlen,
            "HLEN": self.hlen,
            "BATCH": self.batch,
            "HBM_CHANNELS": self.hbm_channels,
            "HBM_GENERATION": self.hbm_generation,
            "CHIP_COUNT": self.chip_count,
            "TP": self.tp,
            "KVP": self.kvp,
            "LINK_PORTS": self.link_ports,
            "SRAM_POLICY": self.sram_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_pre_e2_dict()
        if self.architecture_knobs_explicit:
            value.update(
                {
                    "KV_HEAD_REUSE": self.kv_head_reuse,
                    "DRAIN_OVERLAPPED": self.drain_overlapped,
                }
            )
        return value

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the pre-topology single-chip representation.

        This exists only to validate immutable historical evidence.  New
        candidates and result records always use :meth:`to_dict`.
        """

        if self.chip_count != 1:
            raise ValueError(
                "legacy multi-chip candidates used ideal, not explicit, partitioning"
            )
        return {
            name: self.to_pre_e2_dict()[name]
            for name in self.LEGACY_FIELDS
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        allow_legacy_single_chip: bool = False,
    ) -> "HardwareCandidate":
        """Parse a canonical candidate, optionally normalising old evidence."""

        raw = dict(value)
        keys = set(raw)
        legacy = keys == set(cls.LEGACY_FIELDS)
        pre_e2 = keys == set(cls.PRE_E2_FIELDS)
        canonical = keys == set(cls.E2_FIELDS)
        if not any((legacy, pre_e2, canonical)):
            raise ValueError("hardware candidate fields differ from the schema")
        if legacy and not allow_legacy_single_chip:
            raise ValueError("hardware candidate is missing explicit topology")
        if legacy and int(raw["CHIP_COUNT"]) != 1:
            raise ValueError(
                "legacy multi-chip evidence cannot be assigned explicit semantics"
            )
        if canonical and any(
            not isinstance(raw[name], bool)
            for name in ("KV_HEAD_REUSE", "DRAIN_OVERLAPPED")
        ):
            raise TypeError("architecture option fields must be boolean")
        candidate = cls(
            mlen=int(raw["MLEN"]),
            blen=int(raw["BLEN"]),
            vlen=int(raw["VLEN"]),
            hlen=int(raw["HLEN"]),
            batch=int(raw["BATCH"]),
            hbm_channels=int(raw["HBM_CHANNELS"]),
            hbm_generation=str(raw["HBM_GENERATION"]),
            chip_count=int(raw["CHIP_COUNT"]),
            tp=(1 if legacy else int(raw["TP"])),
            kvp=(1 if legacy else int(raw["KVP"])),
            link_ports=(0 if legacy else int(raw["LINK_PORTS"])),
            sram_policy=(
                "streaming" if legacy else str(raw["SRAM_POLICY"])
            ),
            kv_head_reuse=(
                raw["KV_HEAD_REUSE"] if canonical else None
            ),
            drain_overlapped=(
                raw["DRAIN_OVERLAPPED"] if canonical else None
            ),
        )
        expected = candidate.to_legacy_dict() if legacy else candidate.to_dict()
        if raw != expected:
            raise ValueError("hardware candidate fields are not canonical")
        return candidate

    @property
    def candidate_id(self) -> str:
        return f"hw-{_content_hash(self.to_dict())}"


@dataclass(frozen=True)
class ExactHardwareSpace:
    """Finite hardware grid enumerated in canonical lexical order."""

    mlen: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    blen: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    hlen: tuple[int, ...] = (16, 32, 64, 128)
    batch: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)
    hbm_channels: tuple[int, ...] = (8, 16, 32)
    hbm_generation: str = "HBM2"
    chip_count: tuple[int, ...] | int = CHIP_COUNTS
    tp: tuple[int, ...] = ()
    kvp: tuple[int, ...] = ()
    link_ports: tuple[int, ...] = (1, 2, 4)
    sram_policy: tuple[str, ...] = SRAM_POLICIES
    kv_head_reuse: tuple[bool, ...] = (False, True)
    drain_overlapped: tuple[bool, ...] = (False, True)
    resource_budget: ResourceBudget = ResourceBudget()
    attention_heads: int | None = None
    kv_heads: int | None = None
    fp_sram_depth: int = DEFAULT_FP_SRAM_DEPTH

    def __post_init__(self) -> None:
        for name in ("mlen", "blen", "hlen", "batch", "hbm_channels"):
            object.__setattr__(
                self,
                name,
                _ordered_positive(getattr(self, name), name),
            )
        if not self.hbm_generation:
            raise ValueError("hbm_generation must be non-empty")
        object.__setattr__(
            self,
            "chip_count",
            _positive_sequence(self.chip_count, "chip_count"),
        )
        for name in ("tp", "kvp"):
            values = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _ordered_positive(values, name) if values else (),
            )
        object.__setattr__(
            self,
            "link_ports",
            _ordered_positive(self.link_ports, "link_ports"),
        )
        policies = tuple(
            sorted({_normalise_sram_policy(value) for value in self.sram_policy})
        )
        if not policies:
            raise ValueError("sram_policy must not be empty")
        object.__setattr__(self, "sram_policy", policies)
        for name in ("kv_head_reuse", "drain_overlapped"):
            object.__setattr__(
                self,
                name,
                _boolean_sequence(getattr(self, name), name),
            )
        if not isinstance(self.resource_budget, ResourceBudget):
            raise TypeError("resource_budget must be ResourceBudget")
        if (self.attention_heads is None) != (self.kv_heads is None):
            raise ValueError("attention_heads and kv_heads must be paired")
        for name in ("attention_heads", "kv_heads"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.fp_sram_depth, bool)
            or not isinstance(self.fp_sram_depth, int)
            or self.fp_sram_depth <= 0
        ):
            raise ValueError("fp_sram_depth must be positive")
        if self.fp_sram_depth != DEFAULT_FP_SRAM_DEPTH:
            raise ValueError(
                "the search is bound to the current 512-slot FP SRAM"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ExactHardwareSpace":
        data = dict(value or {})
        tp_raw = data.get("TP", data.get("tp", ()))
        kvp_raw = data.get("KVP", data.get("kvp", ()))
        ports_raw = data.get(
            "LINK_PORTS",
            data.get("link_ports", (1, 2, 4)),
        )
        policies_raw = data.get(
            "SRAM_POLICY",
            data.get("sram_policy", SRAM_POLICIES),
        )
        reuse_raw = data.get(
            "KV_HEAD_REUSE",
            data.get("kv_head_reuse", (False, True)),
        )
        drain_raw = data.get(
            "DRAIN_OVERLAPPED",
            data.get("drain_overlapped", (False, True)),
        )
        return cls(
            mlen=tuple(data.get("MLEN", data.get("mlen", cls.mlen))),
            blen=tuple(data.get("BLEN", data.get("blen", cls.blen))),
            hlen=tuple(data.get("HLEN", data.get("hlen", cls.hlen))),
            batch=tuple(data.get("BATCH", data.get("batch", cls.batch))),
            hbm_channels=tuple(
                data.get("HBM_CHANNELS", data.get("hbm_channels", cls.hbm_channels))
            ),
            hbm_generation=str(
                data.get(
                    "HBM_GENERATION",
                    data.get("hbm_generation", cls.hbm_generation),
                )
            ),
            chip_count=data.get(
                "CHIP_COUNT",
                data.get("chip_count", CHIP_COUNTS),
            ),
            tp=(() if not tp_raw else _positive_sequence(tp_raw, "tp")),
            kvp=(() if not kvp_raw else _positive_sequence(kvp_raw, "kvp")),
            link_ports=_positive_sequence(ports_raw, "link_ports"),
            sram_policy=(
                (str(policies_raw),)
                if isinstance(policies_raw, str)
                else tuple(policies_raw)
            ),
            kv_head_reuse=_boolean_sequence(
                reuse_raw,
                "kv_head_reuse",
            ),
            drain_overlapped=_boolean_sequence(
                drain_raw,
                "drain_overlapped",
            ),
            resource_budget=ResourceBudget.from_dict(
                data.get("RESOURCE_BUDGET", data.get("resource_budget"))
            ),
            attention_heads=(
                int(data["ATTENTION_HEADS"])
                if "ATTENTION_HEADS" in data
                else (
                    int(data["attention_heads"])
                    if "attention_heads" in data
                    else None
                )
            ),
            kv_heads=(
                int(data["KV_HEADS"])
                if "KV_HEADS" in data
                else (
                    int(data["kv_heads"])
                    if "kv_heads" in data
                    else None
                )
            ),
            fp_sram_depth=int(
                data.get(
                    "FP_SRAM_DEPTH",
                    data.get("fp_sram_depth", DEFAULT_FP_SRAM_DEPTH),
                )
            ),
        )

    @classmethod
    def from_study_config(
        cls,
        config: Mapping[str, Any],
    ) -> "ExactHardwareSpace":
        """Load the complete hardware grid and bind model head geometry.

        The model architecture is deliberately injected here instead of being
        copied into each study configuration.  This keeps the searched PackedKV
        legality constraints tied to the same immutable architecture used by
        the numerical manifest.
        """

        configured = config.get("hardware_space")
        if not isinstance(configured, Mapping):
            raise ValueError("config is missing hardware_space")
        raw = dict(configured)
        aliases = {
            "MLEN": "mlen",
            "BLEN": "blen",
            "HLEN": "hlen",
            "BATCH": "batch",
            "HBM_CHANNELS": "hbm_channels",
        }
        for canonical, alias in aliases.items():
            if canonical not in raw and alias not in raw:
                raise ValueError(f"hardware space is missing {canonical}")
        if "hbm_gen" in raw:
            if "HBM_GENERATION" in raw or "hbm_generation" in raw:
                raise ValueError(
                    "hardware space has conflicting HBM generation keys"
                )
            raw["HBM_GENERATION"] = raw.pop("hbm_gen")
        if "HBM_GENERATION" not in raw and "hbm_generation" not in raw:
            raise ValueError("hardware space is missing HBM_GENERATION")
        if "CHIP_COUNT" not in raw and "chip_count" not in raw:
            raw["CHIP_COUNT"] = list(CHIP_COUNTS)
        if "RESOURCE_BUDGET" not in raw and "resource_budget" not in raw:
            configured_budget = config.get("resource_budget")
            if configured_budget is not None:
                raw["RESOURCE_BUDGET"] = configured_budget
        architecture = config.get("model_architecture")
        if isinstance(architecture, Mapping):
            attention_heads = architecture.get("num_attention_heads")
            kv_heads = architecture.get("num_key_value_heads")
            if (attention_heads is None) != (kv_heads is None):
                raise ValueError(
                    "model architecture must pair attention and KV heads"
                )
            if attention_heads is not None:
                raw.setdefault("ATTENTION_HEADS", attention_heads)
                raw.setdefault("KV_HEADS", kv_heads)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "MLEN": list(self.mlen),
            "BLEN": list(self.blen),
            "HLEN": list(self.hlen),
            "BATCH": list(self.batch),
            "HBM_CHANNELS": list(self.hbm_channels),
            "HBM_GENERATION": self.hbm_generation,
            "CHIP_COUNT": list(self.chip_count),
            "TP": list(self.tp),
            "KVP": list(self.kvp),
            "LINK_PORTS": list(self.link_ports),
            "SRAM_POLICY": list(self.sram_policy),
            "KV_HEAD_REUSE": list(self.kv_head_reuse),
            "DRAIN_OVERLAPPED": list(self.drain_overlapped),
            "RESOURCE_BUDGET": self.resource_budget.to_dict(),
            "ATTENTION_HEADS": self.attention_heads,
            "KV_HEADS": self.kv_heads,
            "FP_SRAM_DEPTH": self.fp_sram_depth,
        }

    def _local_kv_heads(self, tp: int) -> int | None:
        """Return the exact KV-head count owned by one tensor-parallel rank."""

        if self.attention_heads is None:
            return None
        assert self.kv_heads is not None
        if self.attention_heads % tp or self.kv_heads % tp:
            return None
        return self.kv_heads // tp

    def _legal_reuse_values(
        self,
        *,
        mlen: int,
        blen: int,
        hlen: int,
        local_kv_heads: int | None,
    ) -> tuple[bool, ...]:
        """Filter the reuse knob against the storage owned by one TP rank."""

        legal: list[bool] = []
        for reuse in self.kv_head_reuse:
            if reuse:
                if local_kv_heads is None:
                    continue
                broadcast_heads = mlen // hlen
                required_slots = (
                    SOFTMAX_CONSTANT_SLOTS
                    + SOFTMAX_STATE_VALUES_PER_ROW
                    * blen
                    * broadcast_heads
                    * local_kv_heads
                )
                if (
                    local_kv_heads > broadcast_heads
                    or required_slots > self.fp_sram_depth
                ):
                    continue
            legal.append(reuse)
        return tuple(legal)

    def iter_candidates(self, hidden_size: int) -> Iterator[HardwareCandidate]:
        """Yield every structural candidate without sampling."""

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        for mlen, blen, hlen, batch, channels in itertools.product(
            self.mlen,
            self.blen,
            self.hlen,
            self.batch,
            self.hbm_channels,
        ):
            if mlen % blen or mlen % hlen:
                continue
            if not blen <= hlen <= mlen:
                continue
            if hidden_size % mlen:
                continue
            for chip_count in self.chip_count:
                for tp in range(1, chip_count + 1):
                    if chip_count % tp:
                        continue
                    kvp = chip_count // tp
                    if self.tp and tp not in self.tp:
                        continue
                    if self.kvp and kvp not in self.kvp:
                        continue
                    if hidden_size % tp:
                        continue
                    local_kv_heads = self._local_kv_heads(tp)
                    if self.attention_heads is not None and local_kv_heads is None:
                        continue
                    reuse_values = self._legal_reuse_values(
                        mlen=mlen,
                        blen=blen,
                        hlen=hlen,
                        local_kv_heads=local_kv_heads,
                    )
                    if not reuse_values:
                        continue
                    if chip_count == 1:
                        port_values = (0,)
                    else:
                        minimum_ports = int(tp > 1) + int(kvp > 1)
                        port_values = tuple(
                            value
                            for value in self.link_ports
                            if value >= minimum_ports
                        )
                    for ports, policy, reuse, drain in itertools.product(
                        port_values,
                        self.sram_policy,
                        reuse_values,
                        self.drain_overlapped,
                    ):
                        yield HardwareCandidate(
                            mlen=mlen,
                            blen=blen,
                            vlen=mlen,
                            hlen=hlen,
                            batch=batch,
                            hbm_channels=channels,
                            hbm_generation=self.hbm_generation,
                            chip_count=chip_count,
                            tp=tp,
                            kvp=kvp,
                            link_ports=ports,
                            sram_policy=policy,
                            kv_head_reuse=reuse,
                            drain_overlapped=drain,
                        )

    def candidate_count(self, hidden_size: int) -> int:
        """Return the exact structural-grid size without materialising it."""

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        total = 0
        for mlen, blen, hlen, batch, channels in itertools.product(
            self.mlen,
            self.blen,
            self.hlen,
            self.batch,
            self.hbm_channels,
        ):
            if mlen % blen or mlen % hlen or not blen <= hlen <= mlen:
                continue
            if hidden_size % mlen:
                continue
            for chip_count in self.chip_count:
                for tp in range(1, chip_count + 1):
                    if chip_count % tp:
                        continue
                    kvp = chip_count // tp
                    if self.tp and tp not in self.tp:
                        continue
                    if self.kvp and kvp not in self.kvp:
                        continue
                    if hidden_size % tp:
                        continue
                    local_kv_heads = self._local_kv_heads(tp)
                    if self.attention_heads is not None and local_kv_heads is None:
                        continue
                    reuse_values = self._legal_reuse_values(
                        mlen=mlen,
                        blen=blen,
                        hlen=hlen,
                        local_kv_heads=local_kv_heads,
                    )
                    if not reuse_values:
                        continue
                    if chip_count == 1:
                        port_count = 1
                    else:
                        minimum_ports = int(tp > 1) + int(kvp > 1)
                        port_count = sum(
                            value >= minimum_ports
                            for value in self.link_ports
                        )
                    total += (
                        port_count
                        * len(self.sram_policy)
                        * len(reuse_values)
                        * len(self.drain_overlapped)
                    )
        return total

    def candidates(self, hidden_size: int) -> tuple[HardwareCandidate, ...]:
        """Return every structural candidate without sampling."""

        return tuple(self.iter_candidates(hidden_size))


@dataclass(frozen=True)
class PhysicalTraffic:
    """Average physical off-chip bytes per generated token."""

    weight_bytes: float
    activation_bytes: float
    kv_read_bytes: float
    kv_write_bytes: float
    scale_bytes: float = 0.0
    other_bytes: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "weight_bytes",
            "activation_bytes",
            "kv_read_bytes",
            "kv_write_bytes",
            "scale_bytes",
            "other_bytes",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def total_bytes(self) -> float:
        return sum(
            (
                self.weight_bytes,
                self.activation_bytes,
                self.kv_read_bytes,
                self.kv_write_bytes,
                self.scale_bytes,
                self.other_bytes,
            )
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "unit": "bytes_per_generated_token",
            "weight_bytes": self.weight_bytes,
            "activation_bytes": self.activation_bytes,
            "kv_read_bytes": self.kv_read_bytes,
            "kv_write_bytes": self.kv_write_bytes,
            "scale_bytes": self.scale_bytes,
            "other_bytes": self.other_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class CapacityBreakdown:
    """Resident bytes and available physical capacity."""

    weight_bytes: int
    kv_cache_bytes: int
    runtime_bytes: int
    available_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "weight_bytes",
            "kv_cache_bytes",
            "runtime_bytes",
            "available_bytes",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative")

    @property
    def required_bytes(self) -> int:
        return self.weight_bytes + self.kv_cache_bytes + self.runtime_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.available_bytes - self.required_bytes

    @property
    def feasible(self) -> bool:
        return self.headroom_bytes >= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": "bytes",
            "weight_bytes": self.weight_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "runtime_bytes": self.runtime_bytes,
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "headroom_bytes": self.headroom_bytes,
            "feasible": self.feasible,
        }


@dataclass(frozen=True)
class CalibratedEnergy:
    """Identity-bound energy components for one decoded token."""

    calibration_id: str
    compute_j: float
    vector_j: float
    sram_j: float
    hbm_j: float
    leakage_j: float
    duration_s: float
    unattributed_dynamic_j: float = 0.0
    link_j: float = 0.0
    energy_tier: str = "dc_calibrated"
    energy_id: str | None = None
    token_latency_s: float | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")
        for name in (
            "compute_j",
            "vector_j",
            "sram_j",
            "hbm_j",
            "leakage_j",
            "unattributed_dynamic_j",
            "link_j",
        ):
            value = float(getattr(self, name))
            _require_finite(name, value)
            object.__setattr__(self, name, value)
        duration = float(self.duration_s)
        _require_finite("duration_s", duration, positive=True)
        object.__setattr__(self, "duration_s", duration)
        if self.token_latency_s is not None:
            latency = float(self.token_latency_s)
            _require_finite("token_latency_s", latency, positive=True)
            object.__setattr__(self, "token_latency_s", latency)
        if self.total_j <= 0:
            raise ValueError("total energy must be positive")
        if self.energy_tier not in {
            "analytic_anchored",
            "dc_calibrated",
        }:
            raise ValueError("energy_tier is unsupported")
        identity = self.energy_id or self.calibration_id
        if not identity:
            raise ValueError("energy_id must be non-empty")
        object.__setattr__(self, "energy_id", identity)

    @property
    def total_j(self) -> float:
        return (
            self.compute_j
            + self.vector_j
            + self.sram_j
            + self.hbm_j
            + self.leakage_j
            + self.unattributed_dynamic_j
            + self.link_j
        )

    @property
    def average_power_w(self) -> float:
        return self.total_j / self.duration_s

    @property
    def energy_per_token_j(self) -> float:
        return self.total_j

    @property
    def tokens_per_joule(self) -> float:
        return 1.0 / self.total_j

    @property
    def edp_j_s(self) -> float:
        latency = (
            self.token_latency_s
            if self.token_latency_s is not None
            else self.duration_s
        )
        return self.total_j * latency

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "energy_id": self.energy_id,
            "energy_tier": self.energy_tier,
            "compute_j": self.compute_j,
            "vector_j": self.vector_j,
            "sram_j": self.sram_j,
            "hbm_j": self.hbm_j,
            "leakage_j": self.leakage_j,
            "unattributed_dynamic_j": self.unattributed_dynamic_j,
            "link_j": self.link_j,
            "total_j": self.total_j,
            "energy_per_token_j": self.energy_per_token_j,
            "tokens_per_joule": self.tokens_per_joule,
            "edp_j_s": self.edp_j_s,
            "duration_s": self.duration_s,
            "token_latency_s": self.token_latency_s,
            "average_power_w": self.average_power_w,
        }


@dataclass(frozen=True)
class HardwareMetrics:
    """Measured or analytical outputs for one profile and hardware candidate."""

    tpot_ms: float
    tps: float
    area_mm2: float
    traffic: PhysicalTraffic
    capacity: CapacityBreakdown
    algorithmic_bottleneck: str
    realized_bottleneck: str
    frac_algorithmic_memory_bound: float
    frac_realized_memory_bound: float
    frac_serialization_bound: float
    generated_tokens_per_step: int = 1
    kv_layout: str = "dense_selector"
    layout_id: str | None = None
    capacity_model: str = "resident_weights_kv_runtime"
    runtime_feasible: bool = True
    max_batch: int | None = None
    max_resident_batch: int | None = None
    max_synchronous_batch: int | None = None
    max_runtime_batch: int | None = None
    fits_onchip_sram: bool | None = None
    vector_sram_capacity_bytes: int | None = None
    vector_sram_required_bytes: int | None = None
    matrix_sram_capacity_bytes: int | None = None
    matrix_sram_required_bytes: int | None = None
    hbm_traffic_per_batch_step: tuple[tuple[str, float], ...] = ()
    hbm_traffic_per_generated_token: tuple[tuple[str, float], ...] = ()
    traffic_ledger_id: str | None = None
    area_source: str = "analytical_uncalibrated"
    energy: CalibratedEnergy | None = None
    area_calibration_id: str | None = None
    dc_anchor_id: str | None = None
    dc_anchor_status: Mapping[str, Any] = field(default_factory=dict)
    cycles: int | None = None
    clock_hz: float | None = None
    timing_mode: str = "rtl_serialized"
    timing_calibrated: bool = False
    timing_evidence_id: str | None = None
    timing_reason: str | None = None
    execution_mode: str = LEGACY_AGGREGATE_BANDWIDTH_MODE
    compiler_trace_timing: Mapping[str, Any] | None = None
    bandwidth_calibration_id: str | None = None
    admission_correctness_status: Mapping[str, Any] = field(
        default_factory=dict
    )
    service_mode: str = "unmodeled"
    output_head_status: Mapping[str, Any] = field(default_factory=dict)
    output_head_service: BF16HeadServiceEstimate | None = None
    whole_model_tpot_ms: float | None = None
    whole_model_tps: float | None = None
    whole_model_energy: CalibratedEnergy | None = None
    system_calibration_id: str | None = None
    whole_model_rankable: bool = False
    avg_ideal_compute_seconds: float | None = None
    avg_realized_compute_seconds: float | None = None
    avg_memory_seconds: float | None = None
    step_composition: str = STEP_COMPOSITION
    classical_roofline_bottleneck: str | None = None
    architecture_issue_bottleneck: str | None = None
    frac_classical_memory_bound: float | None = None
    frac_architecture_issue_memory_bound: float | None = None
    avg_peak_compute_seconds: float | None = None
    system_area_mm2: float | None = None
    resource_budget: ResourceBudgetStatus | None = None
    architecture_options: Mapping[str, Any] = field(default_factory=dict)
    capacity_throughput_chain: Mapping[str, Any] = field(default_factory=dict)
    handoff_analysis: Mapping[str, Any] | None = None

    @property
    def capacity_evidence_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.max_resident_batch,
                self.max_batch,
                self.max_synchronous_batch,
                self.max_runtime_batch,
                self.fits_onchip_sram,
                self.vector_sram_capacity_bytes,
                self.vector_sram_required_bytes,
                self.matrix_sram_capacity_bytes,
                self.matrix_sram_required_bytes,
            )
        )

    @property
    def traffic_evidence_complete(self) -> bool:
        return (
            bool(self.hbm_traffic_per_batch_step)
            and bool(self.hbm_traffic_per_generated_token)
            and self.traffic_ledger_id is not None
        )

    @property
    def admission_correctness_valid(self) -> bool:
        return admission_correctness_status_valid(
            self.admission_correctness_status
        )

    @property
    def memory_timing_calibrated(self) -> bool:
        """Return whether the active execution mode has memory-time evidence."""

        return (
            self.execution_mode == COMPILER_TRACE_EXECUTION_MODE
            or bool(self.bandwidth_calibration_id)
        )

    def __post_init__(self) -> None:
        for name in ("tpot_ms", "tps", "area_mm2"):
            value = float(getattr(self, name))
            _require_finite(name, value, positive=True)
            object.__setattr__(self, name, value)
        if self.algorithmic_bottleneck not in {"memory", "compute"}:
            raise ValueError("algorithmic bottleneck must be memory or compute")
        if self.realized_bottleneck not in {
            "memory",
            "serialization",
            "compute",
        }:
            raise ValueError(
                "realized bottleneck must be memory, serialization, or compute"
            )
        for name in (
            "frac_algorithmic_memory_bound",
            "frac_realized_memory_bound",
            "frac_serialization_bound",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "frac_classical_memory_bound",
            "frac_architecture_issue_memory_bound",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "classical_roofline_bottleneck",
            "architecture_issue_bottleneck",
        ):
            value = getattr(self, name)
            if value is not None and value not in {
                "memory",
                "compute",
                "unavailable",
            }:
                raise ValueError(f"{name} is invalid")
        if (
            self.architecture_issue_bottleneck
            not in {None, "unavailable", self.algorithmic_bottleneck}
        ):
            raise ValueError(
                "architecture-issue and compatibility labels disagree"
            )
        if (
            self.frac_architecture_issue_memory_bound is not None
            and abs(
                self.frac_architecture_issue_memory_bound
                - self.frac_algorithmic_memory_bound
            )
            > 1e-12
        ):
            raise ValueError(
                "architecture-issue and compatibility fractions disagree"
            )
        if (
            isinstance(self.generated_tokens_per_step, bool)
            or not isinstance(self.generated_tokens_per_step, int)
            or self.generated_tokens_per_step <= 0
        ):
            raise ValueError("generated_tokens_per_step must be positive")
        if not isinstance(self.runtime_feasible, bool):
            raise TypeError("runtime_feasible must be boolean")
        if not self.kv_layout:
            raise ValueError("kv_layout must be non-empty")
        if self.layout_id is not None and not self.layout_id:
            raise ValueError("layout_id must be non-empty")
        if not self.capacity_model:
            raise ValueError("capacity_model must be non-empty")
        capacity_fields = (
            "max_batch",
            "max_resident_batch",
            "max_synchronous_batch",
            "max_runtime_batch",
            "fits_onchip_sram",
            "vector_sram_capacity_bytes",
            "vector_sram_required_bytes",
            "matrix_sram_capacity_bytes",
            "matrix_sram_required_bytes",
        )
        capacity_values = tuple(getattr(self, name) for name in capacity_fields)
        if any(value is not None for value in capacity_values):
            if any(value is None for value in capacity_values):
                raise ValueError("runtime capacity evidence must be complete")
            for name in capacity_fields:
                if name == "fits_onchip_sram":
                    if not isinstance(self.fits_onchip_sram, bool):
                        raise TypeError(
                            "fits_onchip_sram must be boolean"
                        )
                    continue
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be non-negative")
            expected_sram_fit = (
                self.vector_sram_required_bytes
                <= self.vector_sram_capacity_bytes
                and self.matrix_sram_required_bytes
                <= self.matrix_sram_capacity_bytes
            )
            if self.fits_onchip_sram != expected_sram_fit:
                raise ValueError("on-chip SRAM capacity evidence is inconsistent")
            if self.runtime_feasible != (
                self.capacity.feasible and self.fits_onchip_sram
            ):
                raise ValueError("runtime capacity evidence is inconsistent")
            if self.max_runtime_batch > self.max_resident_batch:
                raise ValueError(
                    "runtime batch limit cannot exceed resident batch limit"
                )
            if self.max_batch != self.max_runtime_batch:
                raise ValueError("legacy and runtime batch limits disagree")
            if self.runtime_feasible != (
                self.generated_tokens_per_step <= self.max_runtime_batch
            ):
                raise ValueError("runtime batch limit disagrees with feasibility")
        step_traffic = tuple(self.hbm_traffic_per_batch_step)
        token_traffic = tuple(self.hbm_traffic_per_generated_token)
        traffic_evidence = (
            bool(step_traffic),
            bool(token_traffic),
            self.traffic_ledger_id is not None,
        )
        if any(traffic_evidence) and not all(traffic_evidence):
            raise ValueError("physical traffic evidence must be complete")
        if all(traffic_evidence):
            if not self.traffic_ledger_id:
                raise ValueError(
                    "physical traffic producer identity must be non-empty"
                )
            step_map = dict(step_traffic)
            token_map = dict(token_traffic)
            if (
                len(step_map) != len(step_traffic)
                or len(token_map) != len(token_traffic)
            ):
                raise ValueError("physical traffic keys must be unique")
            if (
                set(step_map) != PHYSICAL_TRAFFIC_KEYS
                or set(token_map) != PHYSICAL_TRAFFIC_KEYS
            ):
                raise ValueError("physical traffic schema mismatch")
            for name in PHYSICAL_TRAFFIC_KEYS:
                step_value = float(step_map[name])
                token_value = float(token_map[name])
                _require_finite(name, step_value)
                _require_finite(name, token_value)
                expected = token_value * self.generated_tokens_per_step
                if abs(step_value - expected) > max(
                    1e-6,
                    abs(expected) * 1e-9,
                ):
                    raise ValueError(
                        f"physical traffic units disagree for {name}"
                    )
            if abs(sum(token_map.values()) - self.traffic.total_bytes) > max(
                1e-6,
                abs(self.traffic.total_bytes) * 1e-9,
            ):
                raise ValueError("physical traffic roles do not conserve bytes")
            object.__setattr__(
                self,
                "hbm_traffic_per_batch_step",
                tuple(sorted((str(key), float(value)) for key, value in step_map.items())),
            )
            object.__setattr__(
                self,
                "hbm_traffic_per_generated_token",
                tuple(
                    sorted(
                        (str(key), float(value))
                        for key, value in token_map.items()
                    )
                ),
            )
        if not self.area_source:
            raise ValueError("area_source must be non-empty")
        canonical_dc_anchor = json.loads(
            _canonical_bytes(dict(self.dc_anchor_status))
        )
        object.__setattr__(self, "dc_anchor_status", canonical_dc_anchor)
        if self.dc_anchor_id is not None and not self.dc_anchor_id:
            raise ValueError("DC anchor identity must be non-empty")
        if self.area_source == "dc_calibrated":
            if (
                not self.dc_anchor_id
                or canonical_dc_anchor.get("anchor_id")
                != self.dc_anchor_id
            ):
                raise ValueError(
                    "exact DC area requires matching anchor evidence"
                )
        elif self.dc_anchor_id is not None or canonical_dc_anchor:
            raise ValueError(
                "DC anchor evidence requires exact calibrated area"
            )
        if self.cycles is not None and self.cycles <= 0:
            raise ValueError("cycles must be positive")
        if self.clock_hz is not None:
            _require_finite("clock_hz", float(self.clock_hz), positive=True)
        if not self.timing_mode:
            raise ValueError("timing_mode must be non-empty")
        if self.timing_calibrated != bool(self.timing_evidence_id):
            raise ValueError(
                "timing calibration requires a timing evidence identity"
            )
        if self.timing_reason is not None and not self.timing_reason:
            raise ValueError("timing_reason must be non-empty")
        trace_timing = _canonical_trace_timing_set(
            self.execution_mode,
            self.compiler_trace_timing,
            self.timing_evidence_id,
        )
        object.__setattr__(self, "compiler_trace_timing", trace_timing)
        if self.execution_mode == COMPILER_TRACE_EXECUTION_MODE:
            if not self.timing_calibrated:
                raise ValueError("compiler trace timing must be calibrated")
            if self.bandwidth_calibration_id is not None:
                raise ValueError(
                    "aggregate-bandwidth evidence is inapplicable to compiler timing"
                )
        canonical_admission = json.loads(
            _canonical_bytes(dict(self.admission_correctness_status))
        )
        if (
            canonical_admission.get("passed") is True
            and not admission_correctness_status_valid(canonical_admission)
        ):
            raise ValueError(
                "passing admission correctness evidence is inconsistent"
            )
        object.__setattr__(
            self,
            "admission_correctness_status",
            canonical_admission,
        )
        if self.energy is not None:
            if self.energy.energy_tier == "dc_calibrated":
                if self.area_calibration_id != self.energy.calibration_id:
                    raise ValueError(
                        "DC area and energy must use one calibration identity"
                    )
                if self.area_source not in {
                    "dc_calibrated",
                    "dc_calibrated_model",
                }:
                    raise ValueError(
                        "DC-calibrated energy requires DC-calibrated area"
                    )
            elif self.area_calibration_id is not None:
                raise ValueError(
                    "analytic energy cannot claim a DC area calibration"
                )
            expected_duration = (
                self.tpot_ms
                / 1000.0
                / self.generated_tokens_per_step
            )
            tolerance = max(1e-12, expected_duration * 1e-6)
            if abs(self.energy.duration_s - expected_duration) > tolerance:
                raise ValueError("energy duration must match TPOT")
            expected_latency = self.tpot_ms / 1000.0
            if (
                self.energy.token_latency_s is not None
                and abs(
                    self.energy.token_latency_s - expected_latency
                )
                > max(1e-12, expected_latency * 1e-6)
            ):
                raise ValueError("energy token latency must match TPOT")
        elif self.area_calibration_id is not None:
            raise ValueError(
                "calibrated area identity requires calibrated energy"
            )
        if not self.service_mode:
            raise ValueError("service_mode must be non-empty")
        if self.step_composition != STEP_COMPOSITION:
            raise ValueError("decode step composition is unsupported")
        timing_components = (
            self.avg_ideal_compute_seconds,
            self.avg_realized_compute_seconds,
            self.avg_memory_seconds,
        )
        if any(value is not None for value in timing_components):
            if any(value is None for value in timing_components):
                raise ValueError("timing decomposition must be complete")
            for name, value in zip(
                (
                    "avg_ideal_compute_seconds",
                    "avg_realized_compute_seconds",
                    "avg_memory_seconds",
                ),
                timing_components,
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{name} must be finite and positive")
            if (
                self.avg_realized_compute_seconds
                < self.avg_ideal_compute_seconds
            ):
                raise ValueError(
                    "realized compute time cannot beat the ideal issue time"
                )
        if self.avg_peak_compute_seconds is not None:
            peak = self.avg_peak_compute_seconds
            if (
                isinstance(peak, bool)
                or not isinstance(peak, (int, float))
                or not math.isfinite(peak)
                or peak <= 0
            ):
                raise ValueError(
                    "avg_peak_compute_seconds must be finite and positive"
                )
            if self.avg_ideal_compute_seconds is None:
                raise ValueError(
                    "peak compute requires the complete timing decomposition"
                )
            if peak > self.avg_ideal_compute_seconds:
                raise ValueError(
                    "peak compute cannot be slower than ideal issue"
                )
        if not isinstance(self.whole_model_rankable, bool):
            raise TypeError("whole_model_rankable must be boolean")
        canonical_head_status = json.loads(
            _canonical_bytes(dict(self.output_head_status))
        )
        object.__setattr__(
            self,
            "output_head_status",
            canonical_head_status,
        )
        whole_timing = (
            self.whole_model_tpot_ms,
            self.whole_model_tps,
        )
        if self.output_head_service is not None:
            if self.service_mode != HEAD_SERVICE_MODE:
                raise ValueError(
                    "head-service costs require the remote service mode"
                )
            if not head_service_status_valid(canonical_head_status):
                raise ValueError(
                    "head-service costs require passing v2 numerical evidence"
                )
            if canonical_head_status.get(
                "calibration_id"
            ) != self.output_head_service.calibration_id:
                raise ValueError(
                    "head-service calibration identities differ"
                )
            if canonical_head_status.get(
                "provenance_id"
            ) != self.output_head_service.provenance_id:
                raise ValueError(
                    "head-service provenance identities differ"
                )
            if canonical_head_status.get(
                "service_location"
            ) != self.output_head_service.service_location:
                raise ValueError(
                    "head-service locations differ"
                )
            if (
                self.output_head_service.batch
                != self.generated_tokens_per_step
            ):
                raise ValueError(
                    "head-service and decoder batch sizes differ"
                )
        if self.whole_model_rankable:
            if self.service_mode != HEAD_SERVICE_MODE:
                raise ValueError(
                    "rankable whole-model metrics require the remote service"
                )
            if self.output_head_service is None:
                raise ValueError(
                    "rankable whole-model metrics require head-service costs"
                )
            if any(value is None for value in whole_timing):
                raise ValueError(
                    "rankable whole-model timing must be complete"
                )
            expected_tpot = (
                self.tpot_ms
                + self.output_head_service.total_latency_s * 1000.0
            )
            if abs(float(self.whole_model_tpot_ms) - expected_tpot) > max(
                1e-9,
                expected_tpot * 1e-9,
            ):
                raise ValueError(
                    "whole-model TPOT does not conserve service latency"
                )
            expected_tps = (
                self.generated_tokens_per_step
                * 1000.0
                / expected_tpot
            )
            if abs(float(self.whole_model_tps) - expected_tps) > max(
                1e-9,
                expected_tps * 1e-9,
            ):
                raise ValueError(
                    "whole-model TPOT and TPS are inconsistent"
                )
        else:
            if any(value is not None for value in whole_timing):
                raise ValueError(
                    "unrankable whole-model timing must remain absent"
                )
        if self.whole_model_energy is not None:
            if not self.whole_model_rankable or self.energy is None:
                raise ValueError(
                    "whole-model energy requires rankable calibrated components"
                )
            if (
                not self.system_calibration_id
                or self.whole_model_energy.calibration_id
                != self.system_calibration_id
            ):
                raise ValueError(
                    "whole-model energy calibration identity is inconsistent"
                )
            expected_duration = (
                float(self.whole_model_tpot_ms)
                / 1000.0
                / self.generated_tokens_per_step
            )
            tolerance = max(1e-12, expected_duration * 1e-6)
            if (
                abs(
                    self.whole_model_energy.duration_s
                    - expected_duration
                )
                > tolerance
            ):
                raise ValueError(
                    "whole-model energy duration must match whole TPOT"
                )
            expected_latency = float(self.whole_model_tpot_ms) / 1000.0
            if (
                self.whole_model_energy.token_latency_s is not None
                and abs(
                    self.whole_model_energy.token_latency_s
                    - expected_latency
                )
                > max(1e-12, expected_latency * 1e-6)
            ):
                raise ValueError(
                    "whole-model energy token latency must match whole TPOT"
                )
        elif self.system_calibration_id is not None:
            raise ValueError(
                "system calibration identity requires whole-model energy"
            )
        if (
            self.whole_model_rankable
            and self.energy is not None
            and self.whole_model_energy is None
        ):
            raise ValueError(
                "calibrated rankable components require whole-model energy"
            )
        if self.system_area_mm2 is not None:
            _require_finite(
                "system_area_mm2",
                float(self.system_area_mm2),
                positive=True,
            )
        if self.resource_budget is not None:
            if not isinstance(self.resource_budget, ResourceBudgetStatus):
                raise TypeError(
                    "resource_budget must be ResourceBudgetStatus"
                )
            if self.system_area_mm2 is None:
                raise ValueError(
                    "resource-budget evidence requires system area"
                )
            if abs(
                self.resource_budget.aggregate_area_mm2
                - float(self.system_area_mm2)
            ) > max(1e-9, float(self.system_area_mm2) * 1e-9):
                raise ValueError(
                    "system area and resource-budget evidence disagree"
                )
        for name in (
            "architecture_options",
            "capacity_throughput_chain",
        ):
            raw_evidence = getattr(self, name)
            if not isinstance(raw_evidence, Mapping):
                raise TypeError(f"{name} must be an object")
            canonical_evidence = json.loads(
                _canonical_bytes(dict(raw_evidence))
            )
            object.__setattr__(self, name, canonical_evidence)
        if self.architecture_options:
            required = {
                "schema",
                "explicit",
                "kv_head_reuse",
                "drain_overlapped",
                "area",
            }
            if set(self.architecture_options) != required:
                raise ValueError(
                    "architecture-option evidence fields differ from the schema"
                )
            if not isinstance(self.architecture_options["explicit"], bool):
                raise TypeError("architecture-option explicit flag must be boolean")
        if self.capacity_throughput_chain:
            chain = self.capacity_throughput_chain
            if int(chain.get("evaluated_batch", -1)) != self.generated_tokens_per_step:
                raise ValueError("capacity-throughput evaluated batch disagrees")
            if (
                self.max_runtime_batch is not None
                and int(chain.get("max_feasible_batch", -1))
                != self.max_runtime_batch
            ):
                raise ValueError("capacity-throughput batch ceiling disagrees")
            if chain.get("runtime_feasible") is not self.runtime_feasible:
                raise ValueError("capacity-throughput feasibility disagrees")
            expected_throughput = self.tps if self.runtime_feasible else None
            observed_throughput = chain.get(
                "evaluated_throughput_tokens_per_second"
            )
            if expected_throughput is None:
                if observed_throughput is not None:
                    raise ValueError(
                        "infeasible capacity cannot report evaluated throughput"
                    )
            elif not math.isclose(
                float(observed_throughput),
                expected_throughput,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("capacity-throughput TPS disagrees")
        if self.handoff_analysis is not None:
            if not isinstance(self.handoff_analysis, Mapping):
                raise TypeError("handoff_analysis must be an object")
            handoff = json.loads(
                _canonical_bytes(dict(self.handoff_analysis))
            )
            if handoff.get("schema_version") != (
                "plena-prefill-handoff-analysis-v1"
            ):
                raise ValueError("unsupported prefill handoff analysis schema")
            if handoff.get("scope") != (
                "request_level_prefill_decode_schedule"
            ):
                raise ValueError("prefill handoff analysis scope is invalid")
            if handoff.get("ordinary_decode_ranking_effect") != "none":
                raise ValueError("handoff analysis cannot alter decode ranking")
            if not isinstance(handoff.get("publication_rankable"), bool):
                raise TypeError("handoff rankability must be boolean")
            artifact_id = handoff.get("input_artifact_id")
            if (
                not isinstance(artifact_id, str)
                or not artifact_id.startswith("prefill-handoff-")
                or len(artifact_id) != len("prefill-handoff-") + 64
                or any(
                    character not in "0123456789abcdef"
                    for character in artifact_id[-64:]
                )
            ):
                raise ValueError("handoff input artifact identity is invalid")
            regimes = handoff.get("regimes")
            if not isinstance(regimes, list) or len(regimes) not in {0, 3}:
                raise ValueError("handoff analysis must contain zero or three regimes")
            if regimes:
                names = [regime.get("regime") for regime in regimes]
                if names != [
                    "fully_pipelined",
                    "back_pressure",
                    "host_buffered",
                ]:
                    raise ValueError("handoff regime ordering is invalid")
            if handoff["publication_rankable"]:
                if handoff.get("unrankable_reason") is not None or not regimes:
                    raise ValueError("rankable handoff evidence is incomplete")
            elif not handoff.get("unrankable_reason"):
                raise ValueError("unrankable handoff evidence requires a reason")
            object.__setattr__(self, "handoff_analysis", handoff)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_scope": "decoder_stack_only",
            "tpot_ms": self.tpot_ms,
            "tps": self.tps,
            "area_mm2": self.area_mm2,
            "area_scope": "decode_chip_only",
            "physical_traffic": self.traffic.to_dict(),
            "capacity": self.capacity.to_dict(),
            "algorithmic_bottleneck": self.algorithmic_bottleneck,
            "classical_roofline_bottleneck": (
                self.classical_roofline_bottleneck
            ),
            "architecture_issue_bottleneck": (
                self.architecture_issue_bottleneck
                if self.architecture_issue_bottleneck is not None
                else self.algorithmic_bottleneck
            ),
            "realized_bottleneck": self.realized_bottleneck,
            "frac_classical_memory_bound": (
                self.frac_classical_memory_bound
            ),
            "frac_architecture_issue_memory_bound": (
                self.frac_architecture_issue_memory_bound
                if self.frac_architecture_issue_memory_bound is not None
                else self.frac_algorithmic_memory_bound
            ),
            "frac_algorithmic_memory_bound": (
                self.frac_algorithmic_memory_bound
            ),
            "frac_realized_memory_bound": self.frac_realized_memory_bound,
            "frac_serialization_bound": self.frac_serialization_bound,
            "generated_tokens_per_step": self.generated_tokens_per_step,
            "kv_layout": self.kv_layout,
            "layout_id": self.layout_id,
            "capacity_model": self.capacity_model,
            "runtime_feasible": self.runtime_feasible,
            "max_batch": self.max_batch,
            "max_resident_batch": self.max_resident_batch,
            "max_synchronous_batch": self.max_synchronous_batch,
            "max_runtime_batch": self.max_runtime_batch,
            "fits_onchip_sram": self.fits_onchip_sram,
            "vector_sram_capacity_bytes": self.vector_sram_capacity_bytes,
            "vector_sram_required_bytes": self.vector_sram_required_bytes,
            "matrix_sram_capacity_bytes": self.matrix_sram_capacity_bytes,
            "matrix_sram_required_bytes": self.matrix_sram_required_bytes,
            "runtime_capacity_evidence": (
                {
                    "producer_id": self.capacity_model,
                    "batch_unit": "active_sequences",
                    "byte_unit": "bytes",
                    "evaluated_batch": self.generated_tokens_per_step,
                    "max_batch": self.max_batch,
                    "max_resident_batch": self.max_resident_batch,
                    "max_synchronous_batch": self.max_synchronous_batch,
                    "max_runtime_batch": self.max_runtime_batch,
                    "fits_hbm": self.capacity.feasible,
                    "fits_onchip_sram": self.fits_onchip_sram,
                    "fits_runtime": self.runtime_feasible,
                    "vector_sram_capacity_bytes": (
                        self.vector_sram_capacity_bytes
                    ),
                    "vector_sram_required_bytes": (
                        self.vector_sram_required_bytes
                    ),
                    "matrix_sram_capacity_bytes": (
                        self.matrix_sram_capacity_bytes
                    ),
                    "matrix_sram_required_bytes": (
                        self.matrix_sram_required_bytes
                    ),
                }
                if self.capacity_evidence_complete
                else None
            ),
            "hbm_traffic_ledger": (
                {
                    "producer_id": self.traffic_ledger_id,
                    "per_batch_step": {
                        "unit": "bytes_per_batch_step",
                        "values": dict(self.hbm_traffic_per_batch_step),
                    },
                    "per_generated_token": {
                        "unit": "bytes_per_generated_token",
                        "values": dict(
                            self.hbm_traffic_per_generated_token
                        ),
                    },
                }
                if self.traffic_ledger_id is not None
                else None
            ),
            "area_source": self.area_source,
            "calibrated_energy": self.energy.to_dict() if self.energy else None,
            "energy_per_token_j": (
                self.energy.energy_per_token_j if self.energy else None
            ),
            "energy_tier": self.energy.energy_tier if self.energy else None,
            "tokens_per_joule": (
                self.energy.tokens_per_joule if self.energy else None
            ),
            "edp_j_s": self.energy.edp_j_s if self.energy else None,
            "area_calibration_id": self.area_calibration_id,
            "system_area_mm2": self.system_area_mm2,
            "resource_budget": (
                self.resource_budget.to_dict()
                if self.resource_budget is not None
                else None
            ),
            "architecture_options": dict(self.architecture_options),
            "capacity_throughput_chain": dict(
                self.capacity_throughput_chain
            ),
            "handoff_analysis": self.handoff_analysis,
            "dc_anchor_id": self.dc_anchor_id,
            "dc_anchor_status": dict(self.dc_anchor_status),
            "cycles": self.cycles,
            "clock_hz": self.clock_hz,
            "timing_mode": self.timing_mode,
            "timing_calibrated": self.timing_calibrated,
            "timing_evidence_id": self.timing_evidence_id,
            "timing_reason": self.timing_reason,
            "execution_mode": self.execution_mode,
            "compiler_trace_timing": self.compiler_trace_timing,
            "bandwidth_calibration_id": self.bandwidth_calibration_id,
            "memory_timing_calibrated": self.memory_timing_calibrated,
            "admission_boundary": dict(
                self.admission_correctness_status
            ),
            "timing_decomposition": (
                {
                    "unit": "seconds_per_batch_step",
                    "composition": self.step_composition,
                    **(
                        {
                            "peak_compute_seconds": (
                                self.avg_peak_compute_seconds
                            )
                        }
                        if self.avg_peak_compute_seconds is not None
                        else {}
                    ),
                    "ideal_compute_seconds": (
                        self.avg_ideal_compute_seconds
                    ),
                    "realized_compute_seconds": (
                        self.avg_realized_compute_seconds
                    ),
                    "memory_seconds": self.avg_memory_seconds,
                }
                if self.avg_ideal_compute_seconds is not None
                else None
            ),
            "decoder_stack": {
                "scope": "decode_chip_through_final_rmsnorm",
                "tpot_ms": self.tpot_ms,
                "tps": self.tps,
                "calibrated_energy": (
                    self.energy.to_dict()
                    if self.energy is not None
                    else None
                ),
                "energy_per_token_j": (
                    self.energy.energy_per_token_j
                    if self.energy is not None
                    else None
                ),
                "energy_tier": (
                    self.energy.energy_tier
                    if self.energy is not None
                    else None
                ),
                "tokens_per_joule": (
                    self.energy.tokens_per_joule
                    if self.energy is not None
                    else None
                ),
                "edp_j_s": (
                    self.energy.edp_j_s
                    if self.energy is not None
                    else None
                ),
                "area_mm2": self.area_mm2,
                "area_scope": "decode_chip_only",
            },
            "output_head_boundary": {
                "service_mode": self.service_mode,
                "status": dict(self.output_head_status),
                "estimate": (
                    self.output_head_service.to_dict()
                    if self.output_head_service is not None
                    else None
                ),
            },
            "whole_model": {
                "rankable": self.whole_model_rankable,
                "tpot_ms": self.whole_model_tpot_ms,
                "tps": self.whole_model_tps,
                "calibrated_energy": (
                    self.whole_model_energy.to_dict()
                    if self.whole_model_energy is not None
                    else None
                ),
                "energy_per_token_j": (
                    self.whole_model_energy.energy_per_token_j
                    if self.whole_model_energy is not None
                    else None
                ),
                "energy_tier": (
                    self.whole_model_energy.energy_tier
                    if self.whole_model_energy is not None
                    else None
                ),
                "tokens_per_joule": (
                    self.whole_model_energy.tokens_per_joule
                    if self.whole_model_energy is not None
                    else None
                ),
                "edp_j_s": (
                    self.whole_model_energy.edp_j_s
                    if self.whole_model_energy is not None
                    else None
                ),
                "system_calibration_id": self.system_calibration_id,
            },
        }


@dataclass(frozen=True)
class HardwareEvaluation:
    """Evaluator output with independent cross-stack validity."""

    metrics: HardwareMetrics | None
    validity: StackValidity
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if bool(self.error_code) != bool(self.error_message):
            raise ValueError("error_code and error_message must be set together")
        if self.validity.dc_calibrated is True:
            if (
                self.metrics is None
                or self.metrics.energy is None
                or self.metrics.area_source != "dc_calibrated"
                or not self.metrics.dc_anchor_id
            ):
                raise ValueError(
                    "dc_calibrated=True requires an exact DC/SAIF anchor"
                )

    @classmethod
    def failed(
        cls,
        code: str,
        message: str,
        *,
        validity: StackValidity = StackValidity(),
    ) -> "HardwareEvaluation":
        return cls(
            metrics=None,
            validity=validity,
            error_code=code,
            error_message=message,
        )


class HardwareEvaluator(Protocol):
    def __call__(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        numerical_result: Mapping[str, Any],
    ) -> HardwareEvaluation:
        ...


def physical_cost_signature(
    profile: Any,
    *,
    exact_vector_format: bool = True,
) -> dict[str, Any]:
    """Return every profile field that can change hardware cost.

    Numerical quality and profile identity are intentionally absent.  Exact
    format tokens are retained for matrix roles because the calibrated area
    model distinguishes exponent and mantissa allocation.  An evaluator that
    uses the width-only analytical vector model may explicitly collapse vector
    formats with the same storage width.
    """

    vector = format_descriptor(profile.vector_format)
    return {
        "schema": "plena-physical-cost-signature",
        "weight_format": profile.weight_format,
        "activation_format": profile.activation_format,
        "key_format": profile.key_format,
        "value_format": profile.value_format,
        "vector_format": (
            profile.vector_format
            if exact_vector_format
            else {
                "family": vector.family,
                "element_bits": vector.element_bits,
            }
        ),
        "block_size": profile.block_size,
        "scale_format": profile.scale_format,
        "scale_bits": profile.scale_bits,
        "accumulator_rule": profile.accumulator_rule,
        "output_rule": profile.output_rule,
        "matrix_semantics": profile.matrix_semantics.to_dict(),
        "operator_bindings": {
            "weight": list(profile.weight_operators),
            "activation": list(profile.activation_operators),
            "kv": list(profile.kv_operators),
            "vector": list(profile.vector_operators),
            "bf16": list(profile.bf16_operators),
        },
    }


def physical_cost_signature_id(
    profile: Any,
    *,
    exact_vector_format: bool = True,
) -> str:
    """Return the content identity used for lossless profile grouping."""

    return "physical-cost-" + _content_hash(
        physical_cost_signature(
            profile,
            exact_vector_format=exact_vector_format,
        )
    )


def _evaluator_group_key(
    evaluator: HardwareEvaluator,
    method_name: str,
    entry: SweepManifestEntry,
    numerical: Mapping[str, Any],
) -> str:
    method = getattr(evaluator, method_name, None)
    if method is None:
        return f"profile:{entry.profile_id}"
    value = method(entry, numerical)
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{method_name} must return a string or mapping")
    return f"{method_name}:" + _content_hash(dict(value))


def _mean_nll(row: Mapping[str, Any]) -> float | None:
    if row.get("state") != "succeeded":
        return None
    metrics = row.get("result", row.get("metrics"))
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get("mean_nll")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        return None
    return float(value)


def _validity_from_result(value: Mapping[str, Any]) -> StackValidity:
    nested = value.get("validity")
    if isinstance(nested, Mapping):
        return StackValidity.from_dict(nested)
    return StackValidity.from_dict(value)


def merge_validity(*values: StackValidity) -> StackValidity:
    """Merge observations conservatively so a measured failure dominates."""

    merged: dict[str, bool | None] = {}
    for name in _VALIDITY_FIELDS:
        observations = [getattr(value, name) for value in values]
        if False in observations:
            merged[name] = False
        elif True in observations:
            merged[name] = True
        else:
            merged[name] = None
    return StackValidity(**merged)


def _index_numerical_results(
    manifest: SweepManifest,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    entries = {entry.profile_id: entry for entry in manifest.entries}
    attempts: dict[str, dict[int, Mapping[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        profile_id = str(row.get("profile_id", ""))
        if profile_id not in entries:
            raise ValueError(f"numerical result references unknown profile {profile_id!r}")
        embedded_profile = row.get("profile")
        if embedded_profile is not None:
            if embedded_profile != entries[profile_id].profile.to_dict():
                raise ValueError(f"numerical profile mismatch for {profile_id}")
        attempt = int(row.get("attempt", 1))
        profile_attempts = attempts.setdefault(profile_id, {})
        if attempt in profile_attempts:
            raise ValueError(f"duplicate numerical attempt {attempt} for {profile_id}")
        profile_attempts[attempt] = row
    return {
        profile_id: profile_attempts[max(profile_attempts)]
        for profile_id, profile_attempts in attempts.items()
    }


def _numerical_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row.get("result", row.get("metrics", {}))
    scalar_metrics: dict[str, Any] = {}
    document_hash: str | None = None
    if isinstance(metrics, Mapping):
        for key, value in sorted(metrics.items()):
            if key == "documents":
                document_hash = _content_hash(value)
            elif value is None or isinstance(value, (bool, int, float, str)):
                scalar_metrics[str(key)] = value
    return {
        "state": row.get("state"),
        "attempt": row.get("attempt"),
        "record_hash": row.get("record_hash"),
        "result_path": row.get("result_path"),
        "scalar_metrics": scalar_metrics,
        "document_metrics_hash": document_hash,
    }


def _factor_join_class_id(
    entry: SweepManifestEntry,
    numerical: Mapping[str, Any],
) -> str:
    """Group members whose deployment eligibility is candidate-equivalent.

    Numerical quality values deliberately do not enter this identity.  Hardware
    retention uses only the presence of a finite NLL; the value itself remains
    bound to each member and is consumed later by the joint precision search.
    """

    base_validity = merge_validity(
        entry.validity,
        _validity_from_result(numerical),
    )
    profile = entry.profile
    format_families = {
        role: format_descriptor(token).family
        for role, token in (
            ("weight", profile.weight_format),
            ("activation", profile.activation_format),
            ("key", profile.key_format),
            ("value", profile.value_format),
        )
    }
    legality_stage_signature = sorted(
        {
            (issue.code, tuple(sorted(issue.stages)))
            for issue in entry.legality.issues
        }
    )
    body = {
        "profile_kind": profile.kind,
        "hardware_candidate": entry.legality.hardware_candidate,
        "software_supported": entry.legality.software_supported,
        "legality_stage_signature": legality_stage_signature,
        "block_size": profile.block_size,
        "vector_format_supported": profile.vector_format in VECTOR_FORMATS,
        "format_families": format_families,
        "rtl_mxint_activation_supported": (
            profile.activation_format in {"MXINT4", "MXINT8"}
        ),
        "base_validity": base_validity.to_dict(),
        "numerical_succeeded": numerical.get("state") == "succeeded",
        "finite_mean_nll": _mean_nll(numerical) is not None,
    }
    return "hardware-join-class-" + _content_hash(body)


def _packedkv_selector_status(
    entry: SweepManifestEntry,
    candidate: HardwareCandidate,
    numerical: Mapping[str, Any],
    validity: StackValidity,
    capability: Mapping[str, Any] | None,
) -> tuple[bool | None, dict[str, Any]]:
    """Bind PackedKV selector eligibility to format, target, and RTL evidence."""

    profile = entry.profile
    if profile.kind == PROFILE_KIND_BF16_REFERENCE:
        return False, {
            "kind": "static_capability",
            "evidence_id": None,
            "reason": "bf16_reference_is_not_a_packedkv_hardware_profile",
        }
    families = {
        role: format_descriptor(token).family
        for role, token in (
            ("weight", profile.weight_format),
            ("activation", profile.activation_format),
            ("key", profile.key_format),
            ("value", profile.value_format),
        )
    }
    if any(family != "mxint" for family in families.values()):
        return False, {
            "kind": "static_capability",
            "evidence_id": (
                "packedkv-selector-static-"
                + _content_hash(
                    {
                        "constraint": "mxint_matrix_path_only",
                        "families": families,
                        "profile_id": entry.profile_id,
                        "candidate_id": candidate.candidate_id,
                    }
                )
            ),
            "reason": "selector_is_wired_only_to_the_mxint_matrix_path",
            "format_families": families,
        }
    issue_codes = tuple(
        sorted(
            issue["code"]
            for issue in (capability or {}).get("issues", ())
            if "rtl" in issue.get("stages", ())
        )
    )
    if issue_codes or validity.rtl_valid is False:
        return False, {
            "kind": "capability_or_measured_failure",
            "evidence_id": (
                "packedkv-selector-failure-"
                + _content_hash(
                    {
                        "profile_id": entry.profile_id,
                        "candidate_id": candidate.candidate_id,
                        "issue_codes": issue_codes,
                        "rtl_valid": validity.rtl_valid,
                        "numerical_record_hash": numerical.get("record_hash"),
                    }
                )
            ),
            "reason": "selector_capability_or_validation_failed",
            "issue_codes": list(issue_codes),
        }
    if validity.rtl_valid is not True:
        return None, {
            "kind": "unmeasured",
            "evidence_id": None,
            "reason": "candidate_scoped_rtl_selector_evidence_is_missing",
        }
    evidence_payload = {
        "profile_id": entry.profile_id,
        "candidate_id": candidate.candidate_id,
        "numerical_record_hash": numerical.get("record_hash"),
        "rtl_valid": True,
        "capability_schema": (capability or {}).get("schema_version"),
    }
    return True, {
        "kind": "candidate_scoped_rtl_validation",
        "evidence_id": (
            "packedkv-selector-rtl-" + _content_hash(evidence_payload)
        ),
        **evidence_payload,
    }


@dataclass(frozen=True)
class StudyProvenance:
    """Content hashes that bind a hardware study to all of its inputs."""

    manifest_hash: str
    numerical_results_hash: str
    hardware_space_hash: str
    evaluator_version: str
    model_revision: str
    tokenizer_revision: str
    code_revisions: tuple[tuple[str, str], ...] = ()
    evaluator_provenance: Mapping[str, Any] = field(default_factory=dict)
    search_schedule: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = HARDWARE_STUDY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_hash",
            "numerical_results_hash",
            "hardware_space_hash",
            "evaluator_version",
            "model_revision",
            "tokenizer_revision",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(
            self,
            "code_revisions",
            tuple(sorted((str(key), str(value)) for key, value in self.code_revisions)),
        )
        canonical_evaluator = json.loads(
            _canonical_bytes(dict(self.evaluator_provenance))
        )
        object.__setattr__(
            self,
            "evaluator_provenance",
            canonical_evaluator,
        )
        object.__setattr__(
            self,
            "search_schedule",
            json.loads(_canonical_bytes(dict(self.search_schedule))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "numerical_results_hash": self.numerical_results_hash,
            "hardware_space_hash": self.hardware_space_hash,
            "evaluator_version": self.evaluator_version,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "code_revisions": dict(self.code_revisions),
            "evaluator_provenance": dict(self.evaluator_provenance),
            "search_schedule": dict(self.search_schedule),
        }

    @property
    def canonical_hash(self) -> str:
        return _content_hash(self.to_dict())

    @property
    def run_id(self) -> str:
        return f"hwdse-{self.canonical_hash}"


@dataclass(frozen=True)
class JoinedHardwareResult:
    """One exact join of numerical identity, hardware metrics, and validity."""

    run_id: str
    profile_ordinal: int
    profile_id: str
    profile: Mapping[str, Any]
    legality: Mapping[str, Any]
    numerical_result_hash: str
    numerical_summary: Mapping[str, Any]
    candidate: HardwareCandidate
    validity: StackValidity
    metrics: HardwareMetrics | None
    capability: Mapping[str, Any] | None = None
    packedkv_selector_valid: bool | None = None
    packedkv_selector_evidence: Mapping[str, Any] = field(
        default_factory=dict
    )
    error_code: str | None = None
    error_message: str | None = None
    schema_version: str = HARDWARE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.packedkv_selector_valid is not None
            and not isinstance(self.packedkv_selector_valid, bool)
        ):
            raise TypeError(
                "packedkv_selector_valid must be boolean or null"
            )
        evidence = dict(self.packedkv_selector_evidence)
        if not evidence:
            raise ValueError("PackedKV selector evidence must be explicit")
        if self.packedkv_selector_valid is True and not evidence.get(
            "evidence_id"
        ):
            raise ValueError(
                "a valid PackedKV selector requires an evidence identity"
            )
        object.__setattr__(
            self,
            "packedkv_selector_evidence",
            json.loads(_canonical_bytes(evidence)),
        )

    @property
    def deployment_valid(self) -> bool:
        return (
            self.numerical_summary.get("state") == "succeeded"
            and
            self.legality.get("hardware_candidate") is True
            and all(
                getattr(self.validity, name) is True
                for name in _VALIDITY_FIELDS
                if name != "dc_calibrated"
            )
            and self.metrics is not None
            and self.metrics.capacity.feasible
            and self.metrics.runtime_feasible
            and self.metrics.capacity_evidence_complete
            and self.metrics.traffic_evidence_complete
            and self.packedkv_selector_valid is True
            and self.metrics.energy is not None
            and self.metrics.energy.energy_tier in {
                "analytic_anchored",
                "dc_calibrated",
            }
            and (
                self.metrics.resource_budget is not None
                and self.metrics.resource_budget.feasible
            )
            and bool(self.metrics.layout_id)
            and self.metrics.memory_timing_calibrated
            and self.metrics.admission_correctness_valid
            and self.metrics.timing_calibrated
            and self.metrics.whole_model_rankable
            and self.metrics.whole_model_energy is not None
            and bool(self.metrics.system_calibration_id)
            and not self.error_code
        )

    def _body(self) -> dict[str, Any]:
        validity = self.validity.to_dict()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "profile_ordinal": self.profile_ordinal,
            "profile_id": self.profile_id,
            "profile": dict(self.profile),
            "legality": dict(self.legality),
            "numerical_result_hash": self.numerical_result_hash,
            "numerical_summary": dict(self.numerical_summary),
            "candidate_id": self.candidate.candidate_id,
            "hardware": self.candidate.to_dict(),
            "capability": dict(self.capability) if self.capability else None,
            "packedkv_selector_valid": self.packedkv_selector_valid,
            "packedkv_selector_evidence": dict(
                self.packedkv_selector_evidence
            ),
            "validity": validity,
            **validity,
            "deployment_valid": self.deployment_valid,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "record_hash": _content_hash(body)}


@dataclass(frozen=True)
class HardwareStudyArtifact:
    path: Path
    metadata_path: Path
    run_id: str
    result_count: int
    stored_result_count: int
    content_hash: str


@dataclass(frozen=True)
class _HardwareFactorGroup:
    """One evaluator equivalence class and its ordered profile membership."""

    factor_id: str
    physical_signature_id: str
    preflight_group_id: str
    evaluation_group_id: str
    schedule_ordinal: int
    evaluation_group_ordinal: int
    entries: tuple[SweepManifestEntry, ...]
    join_classes: tuple[tuple[str, tuple[SweepManifestEntry, ...]], ...]
    candidate_mask_sha256: str
    passing_candidate_count: int


@dataclass(frozen=True)
class _HardwareFactorEvaluation:
    """One physical-signature by hardware-candidate evaluation."""

    ordinal: int
    candidate_ordinal: int
    group: _HardwareFactorGroup
    candidate: HardwareCandidate
    outcome: HardwareEvaluation

    def to_dict(self, run_id: str) -> dict[str, Any]:
        body = {
            "schema_version": HARDWARE_FACTOR_RESULT_SCHEMA,
            "run_id": run_id,
            "factor_id": self.group.factor_id,
            "factor_ordinal": self.ordinal,
            "schedule_ordinal": self.group.schedule_ordinal,
            "evaluation_group_ordinal": self.group.evaluation_group_ordinal,
            "candidate_ordinal": self.candidate_ordinal,
            "candidate_id": self.candidate.candidate_id,
            "hardware": self.candidate.to_dict(),
            "evaluation_validity": self.outcome.validity.to_dict(),
            "metrics": (
                self.outcome.metrics.to_dict()
                if self.outcome.metrics is not None
                else None
            ),
            "error_code": self.outcome.error_code,
            "error_message": self.outcome.error_message,
        }
        return {**body, "factor_record_hash": _content_hash(body)}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_compact_metric(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def _compact_mean_nll(row: Mapping[str, Any]) -> float | None:
    summary = row.get("numerical_summary")
    scalars = summary.get("scalar_metrics") if isinstance(summary, Mapping) else None
    if not isinstance(scalars, Mapping):
        return None
    value = scalars.get("mean_nll")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        return None
    return float(value)


def _compact_energy_tier_rank(value: Any) -> int:
    if value == "dc_calibrated":
        return 0
    if value == "analytic_anchored":
        return 1
    return 2


def _promotion_retention_key(
    row: Mapping[str, Any],
) -> tuple[Any, ...] | None:
    """Mirror the exact profile-level ordering used by refinement selection."""

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
    mean_nll = _compact_mean_nll(row)
    tpot_ms = (
        _positive_compact_metric(whole.get("tpot_ms"))
        if isinstance(whole, Mapping)
        else None
    )
    tps = (
        _positive_compact_metric(whole.get("tps"))
        if isinstance(whole, Mapping)
        else None
    )
    energy_j = (
        _positive_compact_metric(energy.get("total_j"))
        if isinstance(energy, Mapping)
        else None
    )
    area_mm2 = _positive_compact_metric(metrics.get("area_mm2"))
    if (
        row.get("deployment_valid") is not True
        or not isinstance(whole, Mapping)
        or whole.get("rankable") is not True
        or not isinstance(energy, Mapping)
        or energy_tier not in {"analytic_anchored", "dc_calibrated"}
        or not isinstance(energy_identity, str)
        or not energy_identity
        or not isinstance(whole.get("system_calibration_id"), str)
        or not whole.get("system_calibration_id")
        or metrics.get("timing_calibrated") is not True
        or metrics.get("runtime_feasible") is not True
        or not isinstance(capacity, Mapping)
        or capacity.get("feasible") is not True
        or row.get("packedkv_selector_valid") is not True
        or not isinstance(head_estimate, Mapping)
        or not isinstance(head_estimate.get("calibration_id"), str)
        or not head_estimate.get("calibration_id")
        or None in (mean_nll, tpot_ms, tps, energy_j, area_mm2)
    ):
        return None
    return (
        mean_nll,
        _compact_energy_tier_rank(energy_tier),
        energy_j,
        tpot_ms,
        -tps,
        area_mm2,
        str(row.get("profile_id", "")),
        str(row.get("candidate_id", "")),
    )


def _plot_retention_values(
    row: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if row.get("deployment_valid") is not True:
        return None
    metrics = row.get("metrics")
    whole = metrics.get("whole_model") if isinstance(metrics, Mapping) else None
    capacity = (
        metrics.get("runtime_capacity_evidence")
        if isinstance(metrics, Mapping)
        else None
    )
    hardware = row.get("hardware")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(whole, Mapping)
        or whole.get("rankable") is not True
        or not isinstance(capacity, Mapping)
        or not isinstance(hardware, Mapping)
    ):
        raise ValueError("deployment-valid row has incomplete plot metrics")
    energy = whole.get("calibrated_energy")
    if isinstance(energy, Mapping):
        energy_j = _positive_compact_metric(energy.get("total_j"))
        energy_tier = energy.get("energy_tier", whole.get("energy_tier"))
    else:
        energy_j = _positive_compact_metric(whole.get("energy_per_token_j"))
        energy_tier = whole.get("energy_tier")
    resource_budget = metrics.get("resource_budget")
    area_value = metrics.get(
        "system_area_mm2",
        resource_budget.get("aggregate_area_mm2")
        if isinstance(resource_budget, Mapping)
        else metrics.get("area_mm2"),
    )
    tpot_ms = _positive_compact_metric(whole.get("tpot_ms"))
    tps = _positive_compact_metric(whole.get("tps"))
    area_mm2 = _positive_compact_metric(area_value)
    if None in (tpot_ms, tps, energy_j, area_mm2):
        raise ValueError("deployment-valid row has non-positive plot metrics")
    return {
        "tpot_ms": tpot_ms,
        "tps": tps,
        "energy_j": energy_j,
        "area_mm2": area_mm2,
        "edp_j_s": energy_j * tpot_ms / 1000.0,
        "energy_tier": (
            str(energy_tier) if energy_tier not in (None, "") else None
        ),
    }


def _latency_energy_dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return (
        float(left["tpot_ms"]) <= float(right["tpot_ms"])
        and float(left["energy_j"]) <= float(right["energy_j"])
        and (
            float(left["tpot_ms"]) < float(right["tpot_ms"])
            or float(left["energy_j"]) < float(right["energy_j"])
        )
    )


def _frontier_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("profile_id", "")),
        str(row.get("candidate_id", "")),
        str(row.get("record_hash", "")),
    )


def _frontier_retention_dominates(
    left_row: Mapping[str, Any],
    left_values: Mapping[str, Any],
    right_row: Mapping[str, Any],
    right_values: Mapping[str, Any],
) -> bool:
    """Pareto dominance with one canonical representative for exact ties."""

    if _latency_energy_dominates(left_values, right_values):
        return True
    return (
        float(left_values["tpot_ms"]) == float(right_values["tpot_ms"])
        and float(left_values["energy_j"]) == float(right_values["energy_j"])
        and _frontier_identity(left_row) < _frontier_identity(right_row)
    )


def _extrema_retention_key(
    row: Mapping[str, Any],
    values: Mapping[str, Any],
    metric: str,
) -> tuple[Any, ...]:
    return (float(values[metric]), *_frontier_identity(row))


class _CompactHardwareReducer:
    """Retain exact decision rows while hashing the complete joined stream."""

    def __init__(
        self,
        *,
        run_id: str,
        expected_count: int,
        sample_limit: int,
    ) -> None:
        self.run_id = run_id
        self.expected_count = expected_count
        self.sample_limit = sample_limit
        self.sample_seed = f"{run_id}:dominated-scatter"
        self.enumeration_digest = hashlib.sha256()
        self.observed_count = 0
        self.profile_aggregates: dict[str, dict[str, Any]] = {}
        self.frontier: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
        self.extrema: dict[
            str,
            tuple[tuple[Any, ...], int, Mapping[str, Any], Mapping[str, Any]],
        ] = {}
        self.sample_heap: list[tuple[int, int, str]] = []
        self.sample_rows: dict[
            str, tuple[int, Mapping[str, Any], Mapping[str, Any], str]
        ] = {}
        self.scatter_population_count = 0
        self.declared_scatter_population_count = 0
        self.dominated_population_count = 0
        self.unrankable_population_count = 0

    @staticmethod
    def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row.get("profile_id", "")), str(row.get("candidate_id", ""))

    def _sample(
        self,
        *,
        ordinal: int,
        row: Mapping[str, Any],
        values: Mapping[str, Any],
        classification: str,
    ) -> None:
        if self.sample_limit == 0:
            return
        record_hash = str(row["record_hash"])
        score = int(
            hashlib.sha256(
                f"{self.sample_seed}:{record_hash}".encode("utf-8")
            ).hexdigest(),
            16,
        )
        entry = (-score, -ordinal, record_hash)
        if len(self.sample_heap) < self.sample_limit:
            heapq.heappush(self.sample_heap, entry)
            self.sample_rows[record_hash] = (
                ordinal,
                row,
                values,
                classification,
            )
            return
        if entry <= self.sample_heap[0]:
            return
        removed = heapq.heapreplace(self.sample_heap, entry)
        self.sample_rows.pop(removed[2], None)
        self.sample_rows[record_hash] = (
            ordinal,
            row,
            values,
            classification,
        )

    @staticmethod
    def _update_frontier(
        frontier: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
        *,
        ordinal: int,
        row: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> tuple[
        bool,
        tuple[tuple[int, Mapping[str, Any], Mapping[str, Any]], ...],
    ]:
        if any(
            _frontier_retention_dominates(
                existing_row,
                existing_values,
                row,
                values,
            )
            for _, existing_row, existing_values in frontier
        ):
            return False, ()
        evicted = tuple(
            item
            for item in frontier
            if _frontier_retention_dominates(
                row,
                values,
                item[1],
                item[2],
            )
        )
        frontier[:] = [item for item in frontier if item not in evicted]
        frontier.append((ordinal, row, values))
        return True, evicted

    @staticmethod
    def _update_extrema(
        extrema: dict[
            str, tuple[tuple[Any, ...], int, Mapping[str, Any], Mapping[str, Any]]
        ],
        *,
        ordinal: int,
        row: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> None:
        for label, metric in (
            ("fastest", "tpot_ms"),
            ("lowest_energy", "energy_j"),
            ("best_edp", "edp_j_s"),
        ):
            key = _extrema_retention_key(row, values, metric)
            current = extrema.get(label)
            if current is None or key < current[0]:
                extrema[label] = (key, ordinal, row, values)

    def consume(self, row_value: Mapping[str, Any]) -> None:
        row = dict(row_value)
        record_hash = row.get("record_hash")
        body = dict(row)
        body.pop("record_hash", None)
        if record_hash != _content_hash(body):
            raise ValueError("hardware result checksum changed before compaction")
        if row.get("run_id") != self.run_id:
            raise ValueError("hardware result run differs from compact artifact")
        payload = _canonical_bytes(
            {"record_type": "result", **row},
            newline=True,
        )
        self.enumeration_digest.update(payload)
        ordinal = self.observed_count
        self.observed_count += 1
        profile_id, candidate_id = self._row_identity(row)
        if not profile_id or not candidate_id:
            raise ValueError("hardware result identity is incomplete")
        aggregate = self.profile_aggregates.get(profile_id)
        if aggregate is None:
            aggregate = {
                "profile_id": profile_id,
                "profile_ordinal": int(row.get("profile_ordinal", -1)),
                "profile": row.get("profile"),
                "numerical_result_hash": row.get("numerical_result_hash"),
                "total_count": 0,
                "deployment_valid_count": 0,
                "valid_count": 0,
                "error_count": 0,
                "error_code_counts": {},
                "_frontiers": {},
                "_extrema": {},
            }
            self.profile_aggregates[profile_id] = aggregate
        elif (
            aggregate["profile"] != row.get("profile")
            or aggregate["numerical_result_hash"]
            != row.get("numerical_result_hash")
            or aggregate["profile_ordinal"] != int(row.get("profile_ordinal", -1))
        ):
            raise ValueError("hardware profile identity changed during compaction")
        aggregate["total_count"] += 1
        if row.get("deployment_valid") is True:
            aggregate["deployment_valid_count"] += 1
        error_code = row.get("error_code")
        if error_code is not None:
            code = str(error_code)
            aggregate["error_count"] += 1
            counts = aggregate["error_code_counts"]
            counts[code] = counts.get(code, 0) + 1
        promotion_key = _promotion_retention_key(row)
        if promotion_key is not None:
            aggregate["valid_count"] += 1
        plot_values = _plot_retention_values(row)
        if plot_values is None:
            return
        self.scatter_population_count += 1
        if plot_values["energy_tier"] is None:
            self.unrankable_population_count += 1
            self._sample(
                ordinal=ordinal,
                row=row,
                values=plot_values,
                classification="sampled_unrankable",
            )
            return
        self.declared_scatter_population_count += 1
        retained_globally, evicted = self._update_frontier(
            self.frontier,
            ordinal=ordinal,
            row=row,
            values=plot_values,
        )
        newly_dominated = evicted if retained_globally else ((ordinal, row, plot_values),)
        for dominated_ordinal, dominated_row, dominated_values in newly_dominated:
            self.dominated_population_count += 1
            self._sample(
                ordinal=dominated_ordinal,
                row=dominated_row,
                values=dominated_values,
                classification="sampled_dominated",
            )
        self._update_extrema(
            self.extrema,
            ordinal=ordinal,
            row=row,
            values=plot_values,
        )
        if promotion_key is not None:
            tier = str(plot_values["energy_tier"])
            local_frontiers = aggregate["_frontiers"]
            local_frontier = local_frontiers.setdefault(tier, [])
            self._update_frontier(
                local_frontier,
                ordinal=ordinal,
                row=row,
                values=plot_values,
            )
            local_extrema = aggregate["_extrema"].setdefault(tier, {})
            self._update_extrema(
                local_extrema,
                ordinal=ordinal,
                row=row,
                values=plot_values,
            )

    def finish(
        self,
    ) -> tuple[dict[str, Any], tuple[tuple[Mapping[str, Any], tuple[str, ...]], ...]]:
        if self.observed_count != self.expected_count:
            raise RuntimeError("hardware result count changed during enumeration")
        retained: dict[str, tuple[int, Mapping[str, Any]]] = {}
        labels: dict[str, set[str]] = {}

        def retain(
            ordinal: int,
            row: Mapping[str, Any],
            label: str,
        ) -> None:
            record_hash = str(row["record_hash"])
            retained.setdefault(record_hash, (ordinal, row))
            labels.setdefault(record_hash, set()).add(label)

        aggregate_rows = []
        for aggregate in sorted(
            self.profile_aggregates.values(),
            key=lambda item: (int(item["profile_ordinal"]), str(item["profile_id"])),
        ):
            local_frontiers = aggregate.pop("_frontiers")
            local_extrema = aggregate.pop("_extrema")
            frontier_records = []
            for tier, frontier in sorted(local_frontiers.items()):
                for ordinal, row, values in sorted(
                    frontier,
                    key=lambda item: (
                        float(item[2]["tpot_ms"]),
                        float(item[2]["energy_j"]),
                        _frontier_identity(item[1]),
                    ),
                ):
                    retain(ordinal, row, "profile_frontier")
                    frontier_records.append(
                        {
                            "energy_tier": tier,
                            "record_hash": row["record_hash"],
                            "candidate_id": row["candidate_id"],
                        }
                    )
            extrema_records: dict[str, list[dict[str, Any]]] = {
                "fastest": [],
                "lowest_energy": [],
                "best_edp": [],
            }
            for tier, extrema in sorted(local_extrema.items()):
                for name, (_, ordinal, row, _) in sorted(extrema.items()):
                    retain(ordinal, row, f"profile_{name}")
                    extrema_records[name].append(
                        {
                            "energy_tier": tier,
                            "record_hash": row["record_hash"],
                            "candidate_id": row["candidate_id"],
                        }
                    )
            aggregate_rows.append(
                {
                    **aggregate,
                    "error_code_counts": dict(
                        sorted(aggregate["error_code_counts"].items())
                    ),
                    "local_frontier": frontier_records,
                    "local_frontier_count": len(frontier_records),
                    "local_extrema": extrema_records,
                }
            )
        for ordinal, row, _ in self.frontier:
            retain(ordinal, row, "exact_frontier")
        for label, (_, ordinal, row, _) in self.extrema.items():
            retain(ordinal, row, f"exact_{label}")
        sampled_dominated_count = 0
        sampled_unrankable_count = 0
        for ordinal, row, _, label in self.sample_rows.values():
            if label == "sampled_unrankable":
                sampled_unrankable_count += 1
            else:
                sampled_dominated_count += 1
            retain(ordinal, row, label)
        stored = tuple(
            (row, tuple(sorted(labels[record_hash])))
            for record_hash, (ordinal, row) in sorted(
                retained.items(),
                key=lambda item: (item[1][0], item[0]),
            )
        )
        label_counts: dict[str, int] = {}
        for _, row_labels in stored:
            for label in row_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        retention = {
            "sample_seed": self.sample_seed,
            "sample_limit": self.sample_limit,
            "scatter_population_count": self.scatter_population_count,
            "declared_scatter_population_count": (
                self.declared_scatter_population_count
            ),
            "dominated_population_count": self.dominated_population_count,
            "unrankable_population_count": self.unrankable_population_count,
            "sampled_dominated_count": sampled_dominated_count,
            "sampled_unrankable_count": sampled_unrankable_count,
            "exact_frontier_count": len(self.frontier),
            "stored_result_count": len(stored),
            "label_counts": dict(sorted(label_counts.items())),
            "sampling_policy": (
                "smallest_sha256_over_exact_dominated_and_unrankable_population"
            ),
        }
        summary = {
            "observed_result_count": self.observed_count,
            "enumeration_sha256": self.enumeration_digest.hexdigest(),
            "profile_aggregates": aggregate_rows,
            "retention": retention,
        }
        return summary, stored


class _FactorizedHardwareReducer:
    """Reduce physical evaluations without constructing the conceptual join."""

    def __init__(self, study: "ExactHardwareStudy") -> None:
        self.study = study
        self.run_id = study.provenance.run_id
        self.sample_limit = study.scatter_sample_limit
        self.sample_seed = f"{self.run_id}:factor-dominated-scatter"
        self.factor_digest = hashlib.sha256()
        self.observed_factor_count = 0
        self.rankable_factor_population_count = 0
        self.dominated_factor_population_count = 0
        self.unrankable_factor_population_count = 0
        self.global_frontier: list[tuple[Any, ...]] = []
        self.global_extrema: dict[str, tuple[Any, ...]] = {}
        self.sample_heap: list[tuple[int, int, str]] = []
        self.sample_factors: dict[str, tuple[Any, ...]] = {}
        self.class_states: dict[tuple[str, str], dict[str, Any]] = {}
        self.profile_state: dict[str, dict[str, Any]] = {}
        for group in study._factor_memberships:
            for class_id, entries in group.join_classes:
                state = {
                    "group": group,
                    "class_id": class_id,
                    "entries": entries,
                    "total_count": group.passing_candidate_count,
                    "deployment_valid_count": 0,
                    "valid_count": 0,
                    "error_count": 0,
                    "error_code_counts": {},
                    "frontiers": {},
                    "extrema": {},
                }
                self.class_states[(group.factor_id, class_id)] = state
                for entry in entries:
                    if entry.profile_id in self.profile_state:
                        raise ValueError(
                            "a profile occurs in more than one factor membership"
                        )
                    self.profile_state[entry.profile_id] = state

    @staticmethod
    def _update_frontier(
        frontier: list[tuple[Any, ...]],
        item: tuple[Any, ...],
    ) -> tuple[bool, tuple[tuple[Any, ...], ...]]:
        row = item[2]
        values = item[3]
        if any(
            _frontier_retention_dominates(
                existing[2],
                existing[3],
                row,
                values,
            )
            for existing in frontier
        ):
            return False, ()
        evicted = tuple(
            existing
            for existing in frontier
            if _frontier_retention_dominates(
                row,
                values,
                existing[2],
                existing[3],
            )
        )
        frontier[:] = [existing for existing in frontier if existing not in evicted]
        frontier.append(item)
        return True, evicted

    @staticmethod
    def _update_extrema(
        extrema: dict[str, tuple[Any, ...]],
        item: tuple[Any, ...],
    ) -> None:
        row = item[2]
        values = item[3]
        for label, metric in (
            ("fastest", "tpot_ms"),
            ("lowest_energy", "energy_j"),
            ("best_edp", "edp_j_s"),
        ):
            key = _extrema_retention_key(row, values, metric)
            current = extrema.get(label)
            if current is None or key < current[0]:
                extrema[label] = (key, *item)

    def _sample(self, item: tuple[Any, ...], classification: str) -> None:
        if self.sample_limit == 0:
            return
        factor = item[1]
        factor_row = factor.to_dict(self.run_id)
        factor_hash = str(factor_row["factor_record_hash"])
        score = int(
            hashlib.sha256(
                f"{self.sample_seed}:{factor_hash}".encode("utf-8")
            ).hexdigest(),
            16,
        )
        entry = (-score, -int(factor.ordinal), factor_hash)
        if len(self.sample_heap) < self.sample_limit:
            heapq.heappush(self.sample_heap, entry)
            self.sample_factors[factor_hash] = (*item, classification)
            return
        if entry <= self.sample_heap[0]:
            return
        removed = heapq.heapreplace(self.sample_heap, entry)
        self.sample_factors.pop(removed[2], None)
        self.sample_factors[factor_hash] = (*item, classification)

    def consume(self, factor: _HardwareFactorEvaluation) -> None:
        if factor.ordinal != self.observed_factor_count:
            raise ValueError("factor evaluation order changed during enumeration")
        factor_row = factor.to_dict(self.run_id)
        self.factor_digest.update(
            _canonical_bytes(
                {"record_type": "factor_result", **factor_row},
                newline=True,
            )
        )
        self.observed_factor_count += 1
        class_rows = []
        for class_id, entries in factor.group.join_classes:
            state = self.class_states[(factor.group.factor_id, class_id)]
            representative = min(
                entries,
                key=lambda entry: (entry.profile_id, entry.ordinal),
            )
            joined = self.study._joined_result(
                representative,
                factor.candidate,
                factor.outcome,
            ).to_dict()
            if joined["candidate_id"] != factor_row["candidate_id"]:
                raise ValueError("factor candidate changed during profile binding")
            if joined["metrics"] != factor_row["metrics"]:
                raise ValueError("factor metrics changed during profile binding")
            if joined.get("deployment_valid") is True:
                state["deployment_valid_count"] += 1
            if joined.get("error_code") is not None:
                code = str(joined["error_code"])
                state["error_count"] += 1
                counts = state["error_code_counts"]
                counts[code] = counts.get(code, 0) + 1
            promotion_key = _promotion_retention_key(joined)
            if promotion_key is not None:
                state["valid_count"] += 1
            values = _plot_retention_values(joined)
            if values is not None and promotion_key is not None:
                tier = str(values["energy_tier"])
                item = (
                    factor.ordinal,
                    factor,
                    joined,
                    values,
                    representative,
                )
                local_frontier = state["frontiers"].setdefault(tier, [])
                self._update_frontier(local_frontier, item)
                local_extrema = state["extrema"].setdefault(tier, {})
                self._update_extrema(local_extrema, item)
            class_rows.append((entries, representative, joined, values))

        rankable = [item for item in class_rows if item[3] is not None]
        if not rankable:
            self.unrankable_factor_population_count += 1
            _, canonical_entry, joined, _ = min(
                class_rows,
                key=lambda item: (item[1].profile_id, item[1].ordinal),
            )
            self._sample(
                (
                    factor.ordinal,
                    factor,
                    joined,
                    {},
                    canonical_entry,
                ),
                "sampled_unrankable",
            )
            return
        self.rankable_factor_population_count += 1
        _, canonical_entry, canonical_joined, values = min(
            rankable,
            key=lambda item: _frontier_identity(item[2]),
        )
        global_item = (
            factor.ordinal,
            factor,
            canonical_joined,
            values,
            canonical_entry,
        )
        retained, evicted = self._update_frontier(
            self.global_frontier,
            global_item,
        )
        newly_dominated = evicted if retained else (global_item,)
        for item in newly_dominated:
            self.dominated_factor_population_count += 1
            self._sample(item, "sampled_dominated")
        self._update_extrema(self.global_extrema, global_item)

    def finish(
        self,
    ) -> tuple[
        dict[str, Any],
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
    ]:
        if self.observed_factor_count != self.study.expected_factor_evaluation_count:
            raise RuntimeError("hardware factor count changed during enumeration")
        retained_factors: dict[str, tuple[int, Mapping[str, Any]]] = {}
        binding_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
        binding_labels: dict[tuple[str, str], set[str]] = {}

        def retain(
            factor: _HardwareFactorEvaluation,
            entry: SweepManifestEntry,
            label: str,
        ) -> Mapping[str, Any]:
            factor_row = factor.to_dict(self.run_id)
            factor_hash = str(factor_row["factor_record_hash"])
            retained_factors.setdefault(
                factor_hash,
                (factor.ordinal, factor_row),
            )
            key = (factor_hash, entry.profile_id)
            joined = binding_rows.get(key)
            if joined is None:
                joined = self.study._joined_result(
                    entry,
                    factor.candidate,
                    factor.outcome,
                ).to_dict()
                if (
                    joined["candidate_id"] != factor_row["candidate_id"]
                    or joined["hardware"] != factor_row["hardware"]
                    or joined["metrics"] != factor_row["metrics"]
                    or joined["error_code"] != factor_row["error_code"]
                    or joined["error_message"] != factor_row["error_message"]
                ):
                    raise ValueError("retained factor does not reconstruct its join")
                binding_rows[key] = joined
            binding_labels.setdefault(key, set()).add(label)
            return joined

        frontier_by_profile: dict[str, list[dict[str, Any]]] = {}
        extrema_by_profile: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for state in self.class_states.values():
            for entry in state["entries"]:
                frontier_by_profile[entry.profile_id] = []
                extrema_by_profile[entry.profile_id] = {
                    "fastest": [],
                    "lowest_energy": [],
                    "best_edp": [],
                }
            for tier, frontier in sorted(state["frontiers"].items()):
                ordered = sorted(
                    frontier,
                    key=lambda item: (
                        float(item[3]["tpot_ms"]),
                        float(item[3]["energy_j"]),
                        str(item[1].candidate.candidate_id),
                    ),
                )
                for item in ordered:
                    factor = item[1]
                    for entry in state["entries"]:
                        joined = retain(factor, entry, "profile_frontier")
                        frontier_by_profile[entry.profile_id].append(
                            {
                                "energy_tier": tier,
                                "factor_record_hash": factor.to_dict(self.run_id)[
                                    "factor_record_hash"
                                ],
                                "record_hash": joined["record_hash"],
                                "candidate_id": joined["candidate_id"],
                            }
                        )
            for tier, extrema in sorted(state["extrema"].items()):
                for name, extremum in sorted(extrema.items()):
                    factor = extremum[2]
                    for entry in state["entries"]:
                        joined = retain(factor, entry, f"profile_{name}")
                        extrema_by_profile[entry.profile_id][name].append(
                            {
                                "energy_tier": tier,
                                "factor_record_hash": factor.to_dict(self.run_id)[
                                    "factor_record_hash"
                                ],
                                "record_hash": joined["record_hash"],
                                "candidate_id": joined["candidate_id"],
                            }
                        )

        for item in self.global_frontier:
            retain(item[1], item[4], "exact_frontier")
        for label, extremum in self.global_extrema.items():
            retain(extremum[2], extremum[5], f"exact_{label}")
        sampled_dominated_count = 0
        sampled_unrankable_count = 0
        for item in self.sample_factors.values():
            label = str(item[-1])
            if label == "sampled_unrankable":
                sampled_unrankable_count += 1
            else:
                sampled_dominated_count += 1
            retain(item[1], item[4], label)

        aggregate_rows = []
        membership_by_profile = {
            str(member["profile_id"]): member
            for membership in self.study.factor_membership_records()
            for member in membership["members"]
        }
        for profile_id, member in sorted(
            membership_by_profile.items(),
            key=lambda item: (
                int(item[1]["profile_ordinal"]),
                item[0],
            ),
        ):
            state = self.profile_state[profile_id]
            local_frontier = frontier_by_profile[profile_id]
            aggregate_rows.append(
                {
                    "profile_id": profile_id,
                    "profile_ordinal": member["profile_ordinal"],
                    "profile": member["profile"],
                    "numerical_result_hash": member["numerical_result_hash"],
                    "total_count": state["total_count"],
                    "deployment_valid_count": state[
                        "deployment_valid_count"
                    ],
                    "valid_count": state["valid_count"],
                    "error_count": state["error_count"],
                    "error_code_counts": dict(
                        sorted(state["error_code_counts"].items())
                    ),
                    "local_frontier": local_frontier,
                    "local_frontier_count": len(local_frontier),
                    "local_extrema": extrema_by_profile[profile_id],
                }
            )

        stored_bindings = []
        factor_ordinal_by_hash = {
            factor_hash: ordinal
            for factor_hash, (ordinal, _) in retained_factors.items()
        }
        for key, joined in sorted(
            binding_rows.items(),
            key=lambda item: (
                factor_ordinal_by_hash[item[0][0]],
                int(item[1]["profile_ordinal"]),
                item[0][1],
            ),
        ):
            factor_hash, profile_id = key
            labels = tuple(sorted(binding_labels[key]))
            join = {
                "capability": joined["capability"],
                "packedkv_selector_valid": joined[
                    "packedkv_selector_valid"
                ],
                "packedkv_selector_evidence": joined[
                    "packedkv_selector_evidence"
                ],
                "validity": joined["validity"],
                "deployment_valid": joined["deployment_valid"],
                "joined_record_hash": joined["record_hash"],
            }
            body = {
                "schema_version": HARDWARE_FACTOR_BINDING_SCHEMA,
                "factor_record_hash": factor_hash,
                "profile_id": profile_id,
                "retention_labels": list(labels),
                "join": join,
            }
            stored_bindings.append(
                {**body, "binding_hash": _content_hash(body)}
            )

        stored_factor_rows = tuple(
            row
            for _, row in sorted(
                retained_factors.values(),
                key=lambda item: (item[0], item[1]["factor_record_hash"]),
            )
        )
        label_counts: dict[str, int] = {}
        for labels in binding_labels.values():
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        retention = {
            "sample_seed": self.sample_seed,
            "sample_limit": self.sample_limit,
            "factor_population_count": self.observed_factor_count,
            "rankable_factor_population_count": (
                self.rankable_factor_population_count
            ),
            "dominated_factor_population_count": (
                self.dominated_factor_population_count
            ),
            "unrankable_factor_population_count": (
                self.unrankable_factor_population_count
            ),
            "sampled_dominated_count": sampled_dominated_count,
            "sampled_unrankable_count": sampled_unrankable_count,
            "exact_frontier_count": len(self.global_frontier),
            "stored_factor_result_count": len(stored_factor_rows),
            "stored_result_count": len(stored_bindings),
            "label_counts": dict(sorted(label_counts.items())),
            "sampling_policy": (
                "smallest_sha256_over_exact_factor_dominated_and_unrankable_population"
            ),
        }
        summary = {
            "factor_evaluation_count": self.observed_factor_count,
            "factor_evaluation_sha256": self.factor_digest.hexdigest(),
            "profile_aggregates": aggregate_rows,
            "retention": retention,
        }
        return summary, stored_factor_rows, tuple(stored_bindings)


class ExactHardwareStudy:
    """Join the complete feasible precision and hardware cross-product."""

    def __init__(
        self,
        *,
        manifest: SweepManifest,
        numerical_results: Iterable[Mapping[str, Any]],
        space: ExactHardwareSpace,
        hidden_size: int,
        evaluator: HardwareEvaluator,
        evaluator_version: str,
        evaluator_provenance: Mapping[str, Any] | None = None,
        code_revisions: Mapping[str, str] | None = None,
        require_complete: bool = True,
        include_numerical_only: bool = False,
        capability_target: PackedKVRuntimeTarget | None = None,
        relative_perplexity_limit: float | None = None,
        scatter_sample_limit: int = HARDWARE_SCATTER_SAMPLE_LIMIT,
    ) -> None:
        self.manifest = manifest
        self.space = space
        self.hidden_size = int(hidden_size)
        self.evaluator = evaluator
        if (
            isinstance(scatter_sample_limit, bool)
            or not isinstance(scatter_sample_limit, int)
            or scatter_sample_limit < 0
        ):
            raise ValueError("scatter_sample_limit must be a non-negative integer")
        self.scatter_sample_limit = scatter_sample_limit
        if capability_target is None:
            architecture = manifest.model_architecture
            capability_target = replace(
                DEFAULT_PACKED_KV_TARGET,
                hlen=int(architecture["head_dim"]),
                kv_heads=int(architecture["num_key_value_heads"]),
                head_dim=int(architecture["head_dim"]),
            )
        self.capability_target = capability_target
        self._rows = _index_numerical_results(manifest, numerical_results)
        source_entries = tuple(
            entry
            for entry in manifest.entries
            if include_numerical_only or entry.legality.hardware_candidate
        )
        required_entries = list(source_entries)
        reference_entry = None
        if relative_perplexity_limit is not None:
            if (
                isinstance(relative_perplexity_limit, bool)
                or not isinstance(relative_perplexity_limit, (int, float))
                or not math.isfinite(float(relative_perplexity_limit))
                or float(relative_perplexity_limit) <= 1.0
            ):
                raise ValueError(
                    "relative_perplexity_limit must be finite and greater than one"
                )
            references = tuple(
                entry
                for entry in manifest.entries
                if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
            )
            if len(references) != 1:
                raise ValueError(
                    "the accuracy gate requires exactly one BF16 reference"
                )
            reference_entry = references[0]
            if reference_entry not in required_entries:
                required_entries.append(reference_entry)
        if require_complete:
            missing = [
                entry.profile_id
                for entry in required_entries
                if entry.profile_id not in self._rows
            ]
            if missing:
                raise ValueError(
                    f"missing {len(missing)} hardware-relevant numerical results"
                )
        available_entries = tuple(
            entry
            for entry in source_entries
            if entry.profile_id in self._rows
        )
        accuracy_threshold = None
        reference_mean_nll = None
        if relative_perplexity_limit is not None:
            reference_row = self._rows.get(reference_entry.profile_id)
            reference_mean_nll = (
                _mean_nll(reference_row)
                if reference_row is not None
                else None
            )
            if reference_mean_nll is None:
                raise ValueError(
                    "the BF16 accuracy reference is missing a successful finite NLL"
                )
            accuracy_threshold = reference_mean_nll + math.log(
                float(relative_perplexity_limit)
            )
            self._entries = tuple(
                entry
                for entry in available_entries
                if (
                    (mean_nll := _mean_nll(self._rows[entry.profile_id]))
                    is not None
                    and mean_nll <= accuracy_threshold
                )
            )
        else:
            self._entries = available_entries

        rows_bound_to_schedule = {
            entry.profile_id: self._rows[entry.profile_id]
            for entry in required_entries
            if entry.profile_id in self._rows
        }
        selected_rows = [
            rows_bound_to_schedule[entry.profile_id]
            for entry in manifest.entries
            if entry.profile_id in rows_bound_to_schedule
        ]
        structural_candidate_count = space.candidate_count(self.hidden_size)
        candidate_preflight = getattr(evaluator, "candidate_preflight", None)
        base_candidate_mask = bytearray(structural_candidate_count)
        base_candidate_count = 0
        candidate_rejection_counts: dict[str, int] = {}
        candidate_preflight_errors: dict[str, int] = {}
        for candidate_index, candidate in enumerate(
            space.iter_candidates(self.hidden_size)
        ):
            gate = None
            if candidate_preflight is not None:
                try:
                    gate = candidate_preflight(candidate)
                    if gate is not None and not isinstance(
                        gate,
                        HardwareEvaluation,
                    ):
                        raise TypeError(
                            "candidate preflight must return HardwareEvaluation or None"
                        )
                    if gate is not None and (
                        gate.metrics is not None or gate.error_code is None
                    ):
                        raise ValueError(
                            "a candidate preflight rejection must be a failed evaluation"
                        )
                except Exception as exc:
                    error_name = type(exc).__name__
                    candidate_preflight_errors[error_name] = (
                        candidate_preflight_errors.get(error_name, 0) + 1
                    )
                    gate = None
            if gate is None:
                base_candidate_mask[candidate_index] = 1
                base_candidate_count += 1
            else:
                code = str(gate.error_code)
                candidate_rejection_counts[code] = (
                    candidate_rejection_counts.get(code, 0) + 1
                )

        physical_groups: dict[str, list[SweepManifestEntry]] = {}
        preflight_groups: dict[str, list[SweepManifestEntry]] = {}
        group_keys: dict[str, tuple[str, str, str]] = {}
        for entry in self._entries:
            numerical = self._rows[entry.profile_id]
            physical_key = _evaluator_group_key(
                evaluator,
                "physical_cost_group_key",
                entry,
                numerical,
            )
            physical_groups.setdefault(physical_key, []).append(entry)
            preflight_key = _evaluator_group_key(
                evaluator,
                "preflight_group_key",
                entry,
                numerical,
            )
            evaluation_key = _evaluator_group_key(
                evaluator,
                "evaluation_group_key",
                entry,
                numerical,
            )
            group_keys[entry.profile_id] = (
                physical_key,
                preflight_key,
                evaluation_key,
            )
            preflight_groups.setdefault(preflight_key, []).append(entry)

        for label, key_index in (("preflight", 1), ("evaluation", 2)):
            physical_by_group: dict[str, set[str]] = {}
            for keys in group_keys.values():
                physical_key = keys[0]
                group_key = keys[key_index]
                physical_by_group.setdefault(group_key, set()).add(physical_key)
            if any(len(keys) != 1 for keys in physical_by_group.values()):
                raise ValueError(
                    f"{label} equivalence merges distinct physical-cost signatures"
                )
        preflight_by_evaluation: dict[str, set[str]] = {}
        for _, preflight_key, evaluation_key in group_keys.values():
            preflight_by_evaluation.setdefault(evaluation_key, set()).add(
                preflight_key
            )
        if any(len(keys) != 1 for keys in preflight_by_evaluation.values()):
            raise ValueError(
                "one evaluation equivalence class spans different preflight proofs"
            )

        preflight = getattr(evaluator, "preflight", None)
        scheduled: list[
            tuple[tuple[SweepManifestEntry, ...], bytes, int]
        ] = []
        rejection_counts: dict[str, int] = {}
        preflight_error_counts: dict[str, int] = {}
        preflight_calls = 0
        simulator_pricing_count = 0
        for entries in preflight_groups.values():
            ordered_entries = tuple(sorted(entries, key=lambda item: item.ordinal))
            representative = ordered_entries[0]
            numerical = self._rows[representative.profile_id]
            passing_mask = bytearray(structural_candidate_count)
            passing_count = 0
            for candidate_index, candidate in enumerate(
                space.iter_candidates(self.hidden_size)
            ):
                if not base_candidate_mask[candidate_index]:
                    continue
                gate = None
                if preflight is not None:
                    preflight_calls += 1
                    try:
                        gate = preflight(representative, candidate, numerical)
                        if gate is not None and not isinstance(
                            gate,
                            HardwareEvaluation,
                        ):
                            raise TypeError(
                                "hardware preflight must return HardwareEvaluation or None"
                            )
                        if gate is not None and gate.metrics is not None:
                            raise ValueError(
                                "a rejected hardware preflight cannot carry metrics"
                            )
                        if gate is not None and gate.error_code is None:
                            raise ValueError(
                                "a rejected hardware preflight requires an error code"
                            )
                    except Exception as exc:
                        error_name = type(exc).__name__
                        preflight_error_counts[error_name] = (
                            preflight_error_counts.get(error_name, 0) + 1
                        )
                        # A failed proof is not proof of infeasibility.  Keep
                        # the point for the full evaluator to fail closed.
                        gate = None
                if gate is None:
                    passing_mask[candidate_index] = 1
                    passing_count += 1
                else:
                    code = str(gate.error_code)
                    rejection_counts[code] = rejection_counts.get(code, 0) + 1
            scheduled.append(
                (ordered_entries, bytes(passing_mask), passing_count)
            )
            evaluation_keys = {
                group_keys[entry.profile_id][2]
                for entry in ordered_entries
            }
            simulator_pricing_count += len(evaluation_keys) * passing_count

        self._schedule = tuple(
            sorted(scheduled, key=lambda item: item[0][0].ordinal)
        )
        self._evaluation_group_keys = {
            profile_id: keys[2] for profile_id, keys in group_keys.items()
        }
        factor_schedule = []
        factor_memberships = []
        for schedule_ordinal, (entries, candidate_mask, passing_count) in enumerate(
            self._schedule
        ):
            evaluation_groups: dict[str, list[SweepManifestEntry]] = {}
            for entry in entries:
                evaluation_key = self._evaluation_group_keys[entry.profile_id]
                evaluation_groups.setdefault(evaluation_key, []).append(entry)
            groups = []
            for evaluation_group_ordinal, (
                evaluation_key,
                grouped_entries_value,
            ) in enumerate(evaluation_groups.items()):
                grouped_entries = tuple(grouped_entries_value)
                representative = grouped_entries[0]
                physical_key, preflight_key, _ = group_keys[
                    representative.profile_id
                ]
                factor_id = "hardware-factor-" + _content_hash(
                    {
                        "physical_signature_id": physical_key,
                        "preflight_group_id": preflight_key,
                        "evaluation_group_id": evaluation_key,
                        "member_profile_ids": [
                            entry.profile_id for entry in grouped_entries
                        ],
                    }
                )
                class_members: dict[str, list[SweepManifestEntry]] = {}
                for entry in grouped_entries:
                    class_id = _factor_join_class_id(
                        entry,
                        self._rows[entry.profile_id],
                    )
                    class_members.setdefault(class_id, []).append(entry)
                join_classes = tuple(
                    (class_id, tuple(members))
                    for class_id, members in class_members.items()
                )
                group = _HardwareFactorGroup(
                    factor_id=factor_id,
                    physical_signature_id=physical_key,
                    preflight_group_id=preflight_key,
                    evaluation_group_id=evaluation_key,
                    schedule_ordinal=schedule_ordinal,
                    evaluation_group_ordinal=evaluation_group_ordinal,
                    entries=grouped_entries,
                    join_classes=join_classes,
                    candidate_mask_sha256=hashlib.sha256(candidate_mask).hexdigest(),
                    passing_candidate_count=passing_count,
                )
                groups.append(group)
                factor_memberships.append(group)
            factor_schedule.append((tuple(groups), candidate_mask))
        self._factor_schedule = tuple(factor_schedule)
        self._factor_memberships = tuple(factor_memberships)
        self._expected_result_count = sum(
            len(entries) * passing_count
            for entries, _, passing_count in self._schedule
        )
        self._expected_factor_evaluation_count = sum(
            group.passing_candidate_count
            for group in self._factor_memberships
        )
        raw_cross_product = len(available_entries) * structural_candidate_count
        accuracy_cross_product = len(self._entries) * structural_candidate_count
        signature_cross_product = len(physical_groups) * base_candidate_count
        preflight_equivalence_pairs = sum(
            passing_count for _, _, passing_count in self._schedule
        )
        search_schedule = {
            "schema": "plena-lossless-joint-search-schedule",
            "profile_counts": {
                "hardware_relevant_available": len(available_entries),
                "accuracy_passing": len(self._entries),
                "accuracy_rejected": (
                    len(available_entries) - len(self._entries)
                ),
                "physical_cost_signatures": len(physical_groups),
                "preflight_equivalence_groups": len(preflight_groups),
            },
            "candidate_counts": {
                "structurally_legal": structural_candidate_count,
                "candidate_resource_passing": base_candidate_count,
            },
            "cross_product_counts": {
                "raw_hardware_relevant": raw_cross_product,
                "after_accuracy_constraint": accuracy_cross_product,
                "physical_signature_pairs": signature_cross_product,
                "preflight_passing_equivalence_pairs": (
                    preflight_equivalence_pairs
                ),
                "simulator_priced_pairs": simulator_pricing_count,
                "joined_result_rows": self._expected_result_count,
            },
            "compact_artifact_projection": {
                "full_joined_result_rows": self._expected_result_count,
                "factor_evaluation_rows": self._expected_factor_evaluation_count,
                "simulator_pricing_calls": simulator_pricing_count,
                "profile_frontier_rows": "data_dependent",
                "maximum_dominated_sample_rows": self.scatter_sample_limit,
                "exact_frontier_rows": "data_dependent",
                "materializes_dominated_rows": False,
                "materializes_conceptual_join": False,
            },
            "preflight_calls": preflight_calls,
            "candidate_preflight_rejections_by_code": dict(
                sorted(candidate_rejection_counts.items())
            ),
            "candidate_preflight_errors_promoted_to_full_evaluation": dict(
                sorted(candidate_preflight_errors.items())
            ),
            "preflight_rejections_by_code": dict(sorted(rejection_counts.items())),
            "preflight_errors_promoted_to_full_evaluation": dict(
                sorted(preflight_error_counts.items())
            ),
            "accuracy_constraint": {
                "enabled": relative_perplexity_limit is not None,
                "reference_profile_id": (
                    reference_entry.profile_id
                    if reference_entry is not None
                    else None
                ),
                "reference_mean_nll": reference_mean_nll,
                "relative_perplexity_limit": (
                    float(relative_perplexity_limit)
                    if relative_perplexity_limit is not None
                    else None
                ),
                "maximum_mean_nll": accuracy_threshold,
                "formula": (
                    "candidate_mean_nll <= reference_mean_nll + "
                    "log(relative_perplexity_limit)"
                ),
            },
            "losslessness_proof": {
                "accuracy": (
                    "Only profiles violating the declared hard accuracy "
                    "constraint are removed."
                ),
                "signature_cache": (
                    "Each evaluator-declared equivalence class has identical "
                    "physical precision, cost dependencies, and evidence scope; "
                    "group nesting is validated once, keys are immutable for the "
                    "run, and one result is joined back to every member."
                ),
                "preflight": (
                    "Only structural legality, matched area, physical HBM/SRAM "
                    "capacity, and aggregate resource limits may reject a pair."
                ),
                "performance": (
                    "No latency estimate, bottleneck label, bandwidth demand, "
                    "Pareto heuristic, sampling rule, or promotion limit removes "
                    "a feasible pair before simulator pricing."
                ),
            },
        }
        self.provenance = StudyProvenance(
            manifest_hash=manifest.canonical_hash,
            numerical_results_hash=_content_hash(selected_rows),
            hardware_space_hash=_content_hash(
                {
                    "hidden_size": self.hidden_size,
                    "space": space.to_dict(),
                    "include_numerical_only": include_numerical_only,
                    "capability_target": (
                        self.capability_target.to_dict()
                        if self.capability_target is not None
                        else None
                    ),
                }
            ),
            evaluator_version=evaluator_version,
            model_revision=manifest.model_revision,
            tokenizer_revision=str(manifest.tokenizer_revision),
            code_revisions=tuple((code_revisions or {}).items()),
            evaluator_provenance=dict(evaluator_provenance or {}),
            search_schedule=search_schedule,
        )

    @property
    def expected_result_count(self) -> int:
        return self._expected_result_count

    @property
    def expected_factor_evaluation_count(self) -> int:
        return self._expected_factor_evaluation_count

    def factor_membership_records(self) -> tuple[dict[str, Any], ...]:
        """Return the complete ordered map used by conceptual expansion."""

        records = []
        for group in self._factor_memberships:
            class_by_profile = {
                entry.profile_id: class_id
                for class_id, entries in group.join_classes
                for entry in entries
            }
            members = []
            for member_ordinal, entry in enumerate(group.entries):
                numerical = self._rows[entry.profile_id]
                members.append(
                    {
                        "member_ordinal": member_ordinal,
                        "join_class_id": class_by_profile[entry.profile_id],
                        "profile_ordinal": entry.ordinal,
                        "profile_id": entry.profile_id,
                        "profile": entry.profile.to_dict(),
                        "legality": entry.legality.to_dict(),
                        "numerical_result_hash": _content_hash(numerical),
                        "numerical_summary": _numerical_summary(numerical),
                    }
                )
            records.append(
                {
                    "factor_id": group.factor_id,
                    "physical_signature_id": group.physical_signature_id,
                    "preflight_group_id": group.preflight_group_id,
                    "evaluation_group_id": group.evaluation_group_id,
                    "schedule_ordinal": group.schedule_ordinal,
                    "evaluation_group_ordinal": (
                        group.evaluation_group_ordinal
                    ),
                    "candidate_mask_sha256": group.candidate_mask_sha256,
                    "passing_candidate_count": group.passing_candidate_count,
                    "member_count": len(members),
                    "conceptual_result_count": (
                        group.passing_candidate_count * len(members)
                    ),
                    "members": members,
                }
            )
        return tuple(records)

    def iter_factor_evaluations(self) -> Iterator[_HardwareFactorEvaluation]:
        """Price each evaluator-equivalent factor exactly once."""

        factor_ordinal = 0
        for groups, candidate_mask in self._factor_schedule:
            for candidate_ordinal, candidate in enumerate(
                self.space.iter_candidates(self.hidden_size)
            ):
                if not candidate_mask[candidate_ordinal]:
                    continue
                for group in groups:
                    representative = group.entries[0]
                    representative_numerical = self._rows[
                        representative.profile_id
                    ]
                    try:
                        outcome = self.evaluator(
                            representative,
                            candidate,
                            representative_numerical,
                        )
                        if not isinstance(outcome, HardwareEvaluation):
                            raise TypeError(
                                "hardware evaluator must return HardwareEvaluation"
                            )
                    except Exception as exc:
                        outcome = HardwareEvaluation.failed(
                            "evaluator_exception",
                            f"{type(exc).__name__}: {exc}",
                        )
                    yield _HardwareFactorEvaluation(
                        ordinal=factor_ordinal,
                        candidate_ordinal=candidate_ordinal,
                        group=group,
                        candidate=candidate,
                        outcome=outcome,
                    )
                    factor_ordinal += 1

    def iter_results(self) -> Iterator[JoinedHardwareResult]:
        """Expand the factor stream in the historical canonical row order.

        Production artifact writing intentionally does not call this method.  It
        remains available for small equivalence fixtures and compatibility checks.
        """

        for factor in self.iter_factor_evaluations():
            for entry in factor.group.entries:
                yield self._joined_result(
                    entry,
                    factor.candidate,
                    factor.outcome,
                )

    def _joined_result(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        outcome: HardwareEvaluation,
    ) -> JoinedHardwareResult:
        numerical = self._rows[entry.profile_id]
        numerical_hash = _content_hash(numerical)
        base_validity = merge_validity(
            entry.validity,
            _validity_from_result(numerical),
        )
        capability = None
        capability_validity = StackValidity()
        scoped_base_validity = base_validity
        if self.capability_target is not None:
            if self.capability_target.kv_heads % candidate.tp:
                raise ValueError(
                    "tensor parallelism must divide capability KV heads"
                )
            runtime_target = replace(
                self.capability_target,
                mlen=candidate.mlen,
                blen=candidate.blen,
                hlen=candidate.hlen,
                batch=candidate.batch,
                kv_heads=self.capability_target.kv_heads // candidate.tp,
            )
            scoped_base_validity = scope_stack_validity(
                base_validity,
                evidence_target=self.capability_target,
                runtime_target=runtime_target,
            )
            capability_report = evaluate_stack_capability(
                entry.profile,
                runtime_target,
            )
            capability = capability_report.to_dict()
            capability_validity = capability_report.validity_floor
        validity = merge_validity(
            scoped_base_validity,
            capability_validity,
            outcome.validity,
        )
        validity = replace(
            validity,
            dc_calibrated=outcome.validity.dc_calibrated,
        )
        selector_valid, selector_evidence = _packedkv_selector_status(
            entry,
            candidate,
            numerical,
            validity,
            capability,
        )
        return JoinedHardwareResult(
            run_id=self.provenance.run_id,
            profile_ordinal=entry.ordinal,
            profile_id=entry.profile_id,
            profile=entry.profile.to_dict(),
            legality=entry.legality.to_dict(),
            numerical_result_hash=numerical_hash,
            numerical_summary=_numerical_summary(numerical),
            candidate=candidate,
            validity=validity,
            metrics=outcome.metrics,
            capability=capability,
            packedkv_selector_valid=selector_valid,
            packedkv_selector_evidence=selector_evidence,
            error_code=outcome.error_code,
            error_message=outcome.error_message,
        )

    def write(self, path: str | os.PathLike[str]) -> HardwareStudyArtifact:
        """Atomically seal factor evaluations and retained profile bindings."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reducer = _FactorizedHardwareReducer(self)
        for factor in self.iter_factor_evaluations():
            reducer.consume(factor)
        compact_summary, stored_factors, stored_bindings = reducer.finish()
        memberships = self.factor_membership_records()
        membership_digest = hashlib.sha256()
        for membership in memberships:
            membership_digest.update(
                _canonical_bytes(
                    {"record_type": "factor_membership", **membership},
                    newline=True,
                )
            )
        conceptual_result_count = sum(
            int(membership["conceptual_result_count"])
            for membership in memberships
        )
        factor_evaluation_count = sum(
            int(membership["passing_candidate_count"])
            for membership in memberships
        )
        if conceptual_result_count != self.expected_result_count:
            raise RuntimeError("factor memberships do not prove the conceptual count")
        if factor_evaluation_count != self.expected_factor_evaluation_count:
            raise RuntimeError("factor memberships do not prove the evaluation count")
        expansion_contract = {
            "schema": "decode-hardware-factor-expansion",
            "factor_stream_order": [
                "schedule_ordinal",
                "candidate_ordinal",
                "evaluation_group_ordinal",
            ],
            "conceptual_row_order": [
                "schedule_ordinal",
                "candidate_ordinal",
                "evaluation_group_ordinal",
                "member_ordinal",
            ],
            "expansion_rule": (
                "For each factor result, join its candidate, evaluator validity, "
                "metrics, and error fields to every ordered member using the "
                "profile-scoped capability and PackedKV selector contracts."
            ),
            "factor_count_formula": (
                "sum(factor_membership.passing_candidate_count)"
            ),
            "conceptual_count_formula": (
                "sum(factor_membership.passing_candidate_count * "
                "factor_membership.member_count)"
            ),
            "factor_evaluation_count": factor_evaluation_count,
            "conceptual_result_count": conceptual_result_count,
        }
        expansion_contract_sha256 = _content_hash(expansion_contract)
        header = {
            "record_type": "study",
            "schema_version": HARDWARE_STUDY_SCHEMA,
            "storage_revision": HARDWARE_STORAGE_REVISION,
            "run_id": self.provenance.run_id,
            "provenance": self.provenance.to_dict(),
            "expected_result_count": self.expected_result_count,
            "conceptual_result_count": conceptual_result_count,
            "factor_memberships": list(memberships),
            "ordered_membership_map_sha256": membership_digest.hexdigest(),
            "expansion_contract": expansion_contract,
            "expansion_contract_sha256": expansion_contract_sha256,
            **compact_summary,
        }
        temporary_name: str | None = None
        digest = hashlib.sha256()
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                header_payload = _canonical_bytes(header, newline=True)
                temporary.write(header_payload)
                digest.update(header_payload)
                for factor in stored_factors:
                    payload = _canonical_bytes(
                        {"record_type": "factor_result", **factor},
                        newline=True,
                    )
                    temporary.write(payload)
                    digest.update(payload)
                for binding in stored_bindings:
                    payload = _canonical_bytes(
                        {"record_type": "factor_binding", **binding},
                        newline=True,
                    )
                    temporary.write(payload)
                    digest.update(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            content_hash = digest.hexdigest()
            if destination.exists():
                existing_hash = _sha256_path(destination)
                if existing_hash != content_hash:
                    raise FileExistsError(
                        f"refusing to replace a different hardware study: {destination}"
                    )
            else:
                os.chmod(temporary_name, 0o644)
                os.link(temporary_name, destination)
            metadata_path = destination.with_name(f"{destination.name}.meta.json")
            metadata = {
                "schema_version": HARDWARE_ARTIFACT_SCHEMA,
                "storage_revision": HARDWARE_STORAGE_REVISION,
                "run_id": self.provenance.run_id,
                "result_count": self.expected_result_count,
                "factor_evaluation_count": factor_evaluation_count,
                "stored_factor_result_count": len(stored_factors),
                "stored_result_count": len(stored_bindings),
                "factor_evaluation_sha256": compact_summary[
                    "factor_evaluation_sha256"
                ],
                "ordered_membership_map_sha256": (
                    membership_digest.hexdigest()
                ),
                "expansion_contract_sha256": expansion_contract_sha256,
                "content_sha256": content_hash,
                "data_file": destination.name,
                "provenance_hash": self.provenance.canonical_hash,
            }
            _atomic_create_or_verify(metadata_path, _canonical_bytes(metadata, newline=True))
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return HardwareStudyArtifact(
            path=destination,
            metadata_path=metadata_path,
            run_id=self.provenance.run_id,
            result_count=self.expected_result_count,
            stored_result_count=len(stored_bindings),
            content_hash=content_hash,
        )


def _atomic_create_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace a different artifact: {path}")
        return
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
        os.link(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_hardware_artifact(
    path: str | os.PathLike[str],
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Stream and verify a legacy or compact hardware JSONL artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    results: list[Mapping[str, Any]] = []
    factor_rows: list[Mapping[str, Any]] = []
    binding_rows: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError("hardware artifact is empty")
        header = json.loads(first_line)
        if header.get("record_type") != "study":
            raise ValueError("hardware artifact is missing its study header")
        run_id = header.get("run_id")
        provenance = header.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("hardware artifact is missing provenance")
        if run_id != f"hwdse-{_content_hash(provenance)}":
            raise ValueError("hardware artifact provenance hash mismatch")
        storage_revision = header.get("storage_revision")
        factorized = storage_revision == HARDWARE_STORAGE_REVISION
        legacy_compact = storage_revision == LEGACY_COMPACT_STORAGE_REVISION
        if storage_revision not in (
            None,
            HARDWARE_STORAGE_REVISION,
            LEGACY_COMPACT_STORAGE_REVISION,
        ):
            raise ValueError("unsupported hardware artifact storage revision")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                raise ValueError(f"blank hardware record at line {line_number}")
            row = json.loads(line)
            record_type = row.pop("record_type", None)
            if factorized:
                if record_type == "factor_result":
                    factor_hash = row.get("factor_record_hash")
                    body = dict(row)
                    body.pop("factor_record_hash", None)
                    if factor_hash != _content_hash(body):
                        raise ValueError(
                            f"factor checksum mismatch at line {line_number}"
                        )
                    if row.get("run_id") != run_id:
                        raise ValueError(f"run mismatch at line {line_number}")
                    factor_rows.append(row)
                    continue
                if record_type == "factor_binding":
                    binding_hash = row.get("binding_hash")
                    body = dict(row)
                    body.pop("binding_hash", None)
                    if binding_hash != _content_hash(body):
                        raise ValueError(
                            f"factor binding checksum mismatch at line {line_number}"
                        )
                    binding_rows.append(row)
                    continue
                raise ValueError(f"unexpected record type at line {line_number}")
            if record_type != "result":
                raise ValueError(f"unexpected record type at line {line_number}")
            raw_labels = row.pop("retention_labels", None)
            if legacy_compact:
                if (
                    not isinstance(raw_labels, list)
                    or not raw_labels
                    or len(raw_labels) != len(set(raw_labels))
                    or any(label not in HARDWARE_RETENTION_LABELS for label in raw_labels)
                    or "legacy_full_row" in raw_labels
                ):
                    raise ValueError(
                        f"invalid compact retention labels at line {line_number}"
                    )
                retention_labels = tuple(sorted(str(label) for label in raw_labels))
            else:
                if raw_labels is not None:
                    raise ValueError("legacy hardware row carries compact labels")
                retention_labels = ("legacy_full_row",)
            record_hash = row.pop("record_hash", None)
            if record_hash != _content_hash(row):
                raise ValueError(f"hardware checksum mismatch at line {line_number}")
            if row.get("run_id") != run_id:
                raise ValueError(f"run mismatch at line {line_number}")
            results.append(
                {
                    **row,
                    "record_hash": record_hash,
                    "retention_labels": retention_labels,
                }
            )
    metadata_path = source.with_name(f"{source.name}.meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("content_sha256") != _sha256_path(source):
        raise ValueError("hardware artifact content hash mismatch")
    if metadata.get("run_id") != run_id:
        raise ValueError("hardware artifact metadata run mismatch")
    if metadata.get("provenance_hash") != _content_hash(provenance):
        raise ValueError("hardware artifact metadata provenance mismatch")
    if metadata.get("data_file") != source.name:
        raise ValueError("hardware artifact metadata file binding mismatch")
    if factorized:
        results = list(
            _expand_and_validate_factorized_hardware_artifact(
                header=header,
                factors=factor_rows,
                bindings=binding_rows,
                metadata=metadata,
            )
        )
    elif legacy_compact:
        _validate_compact_hardware_artifact(
            header=header,
            results=results,
            metadata=metadata,
        )
    else:
        if len(results) != int(header["expected_result_count"]):
            raise ValueError("hardware artifact result count mismatch")
        if int(metadata["result_count"]) != len(results):
            raise ValueError("hardware artifact metadata count mismatch")
    return header, tuple(results)


def _expand_and_validate_factorized_hardware_artifact(
    *,
    header: Mapping[str, Any],
    factors: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate factor commitments and expand only retained bindings."""

    def require_digest(name: str, value: Any) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
        return value

    run_id = str(header.get("run_id", ""))
    memberships = header.get("factor_memberships")
    if not isinstance(memberships, list):
        raise ValueError("factorized hardware membership map is missing")
    membership_digest = hashlib.sha256()
    membership_by_factor: dict[str, Mapping[str, Any]] = {}
    member_by_factor_profile: dict[tuple[str, str], Mapping[str, Any]] = {}
    membership_order = []
    conceptual_count = 0
    factor_count = 0
    profile_ids = set()
    for membership in memberships:
        if not isinstance(membership, Mapping):
            raise TypeError("factor membership must be a mapping")
        membership_digest.update(
            _canonical_bytes(
                {"record_type": "factor_membership", **membership},
                newline=True,
            )
        )
        factor_id = str(membership.get("factor_id", ""))
        members = membership.get("members")
        schedule_ordinal = int(membership.get("schedule_ordinal", -1))
        group_ordinal = int(
            membership.get("evaluation_group_ordinal", -1)
        )
        passing_count = int(membership.get("passing_candidate_count", -1))
        member_count = int(membership.get("member_count", -1))
        declared_conceptual = int(
            membership.get("conceptual_result_count", -1)
        )
        if (
            not factor_id.startswith("hardware-factor-")
            or factor_id in membership_by_factor
            or not isinstance(members, list)
            or member_count <= 0
            or len(members) != member_count
            or passing_count < 0
            or schedule_ordinal < 0
            or group_ordinal < 0
            or declared_conceptual != passing_count * member_count
        ):
            raise ValueError("factor membership counts or identity are invalid")
        for name in (
            "physical_signature_id",
            "preflight_group_id",
            "evaluation_group_id",
        ):
            if not isinstance(membership.get(name), str) or not membership[name]:
                raise ValueError(f"factor membership {name} is missing")
        require_digest(
            "candidate_mask_sha256",
            membership.get("candidate_mask_sha256"),
        )
        for member_ordinal, member in enumerate(members):
            if not isinstance(member, Mapping):
                raise TypeError("factor member must be a mapping")
            profile_id = str(member.get("profile_id", ""))
            if (
                int(member.get("member_ordinal", -1)) != member_ordinal
                or not profile_id
                or profile_id in profile_ids
                or not isinstance(member.get("profile"), Mapping)
                or not isinstance(member.get("legality"), Mapping)
                or not isinstance(member.get("numerical_summary"), Mapping)
                or not isinstance(member.get("join_class_id"), str)
                or not member.get("join_class_id")
            ):
                raise ValueError("factor member ordering or identity is invalid")
            profile_ids.add(profile_id)
            member_by_factor_profile[(factor_id, profile_id)] = member
        membership_by_factor[factor_id] = membership
        membership_order.append((schedule_ordinal, group_ordinal))
        factor_count += passing_count
        conceptual_count += declared_conceptual
    if membership_order != sorted(membership_order):
        raise ValueError("factor membership ordering is not canonical")
    sealed_membership_digest = require_digest(
        "ordered_membership_map_sha256",
        header.get("ordered_membership_map_sha256"),
    )
    if (
        membership_digest.hexdigest() != sealed_membership_digest
        or metadata.get("ordered_membership_map_sha256")
        != sealed_membership_digest
    ):
        raise ValueError("ordered factor membership digest differs")

    expansion = header.get("expansion_contract")
    if not isinstance(expansion, Mapping):
        raise ValueError("factor expansion contract is missing")
    expansion_digest = require_digest(
        "expansion_contract_sha256",
        header.get("expansion_contract_sha256"),
    )
    if (
        _content_hash(expansion) != expansion_digest
        or metadata.get("expansion_contract_sha256") != expansion_digest
        or expansion.get("schema") != "decode-hardware-factor-expansion"
        or expansion.get("factor_stream_order")
        != [
            "schedule_ordinal",
            "candidate_ordinal",
            "evaluation_group_ordinal",
        ]
        or expansion.get("conceptual_row_order")
        != [
            "schedule_ordinal",
            "candidate_ordinal",
            "evaluation_group_ordinal",
            "member_ordinal",
        ]
        or int(expansion.get("factor_evaluation_count", -1)) != factor_count
        or int(expansion.get("conceptual_result_count", -1))
        != conceptual_count
    ):
        raise ValueError("factor expansion contract differs from the schema")
    if (
        int(header.get("factor_evaluation_count", -1)) != factor_count
        or int(header.get("conceptual_result_count", -1)) != conceptual_count
        or int(header.get("expected_result_count", -1)) != conceptual_count
        or int(metadata.get("factor_evaluation_count", -1)) != factor_count
        or int(metadata.get("result_count", -1)) != conceptual_count
    ):
        raise ValueError("factorized hardware cardinality proof differs")
    factor_digest = require_digest(
        "factor_evaluation_sha256",
        header.get("factor_evaluation_sha256"),
    )
    if metadata.get("factor_evaluation_sha256") != factor_digest:
        raise ValueError("factor evaluation digest differs from metadata")

    factor_by_hash: dict[str, Mapping[str, Any]] = {}
    factor_ordinals = []
    for factor in factors:
        factor_hash = str(factor.get("factor_record_hash", ""))
        factor_id = str(factor.get("factor_id", ""))
        membership = membership_by_factor.get(factor_id)
        if (
            factor.get("schema_version") != HARDWARE_FACTOR_RESULT_SCHEMA
            or factor.get("run_id") != run_id
            or not factor_hash
            or factor_hash in factor_by_hash
            or membership is None
            or factor.get("schedule_ordinal")
            != membership.get("schedule_ordinal")
            or factor.get("evaluation_group_ordinal")
            != membership.get("evaluation_group_ordinal")
            or not isinstance(factor.get("hardware"), Mapping)
            or not isinstance(factor.get("evaluation_validity"), Mapping)
        ):
            raise ValueError("retained factor result is invalid")
        factor_by_hash[factor_hash] = factor
        factor_ordinals.append(int(factor.get("factor_ordinal", -1)))
    if factor_ordinals != sorted(factor_ordinals) or len(set(factor_ordinals)) != len(
        factor_ordinals
    ):
        raise ValueError("retained factor ordering is invalid")

    expanded = []
    factor_hash_by_record_hash: dict[str, str] = {}
    binding_keys = set()
    for binding in bindings:
        factor_hash = str(binding.get("factor_record_hash", ""))
        profile_id = str(binding.get("profile_id", ""))
        factor = factor_by_hash.get(factor_hash)
        member = member_by_factor_profile.get(
            (str(factor.get("factor_id", "")), profile_id)
        ) if factor is not None else None
        labels = binding.get("retention_labels")
        join = binding.get("join")
        if (
            binding.get("schema_version") != HARDWARE_FACTOR_BINDING_SCHEMA
            or factor is None
            or member is None
            or (factor_hash, profile_id) in binding_keys
            or not isinstance(labels, list)
            or not labels
            or labels != sorted(set(labels))
            or any(
                label not in HARDWARE_RETENTION_LABELS
                or label == "legacy_full_row"
                for label in labels
            )
            or not isinstance(join, Mapping)
            or set(join)
            != {
                "capability",
                "packedkv_selector_valid",
                "packedkv_selector_evidence",
                "validity",
                "deployment_valid",
                "joined_record_hash",
            }
            or not isinstance(join.get("validity"), Mapping)
            or not isinstance(join.get("packedkv_selector_evidence"), Mapping)
            or not isinstance(join.get("deployment_valid"), bool)
        ):
            raise ValueError("retained factor binding is invalid")
        binding_keys.add((factor_hash, profile_id))
        validity = dict(join["validity"])
        if set(validity) != set(_VALIDITY_FIELDS):
            raise ValueError("retained binding validity fields differ")
        body = {
            "schema_version": HARDWARE_RESULT_SCHEMA,
            "run_id": run_id,
            "profile_ordinal": member["profile_ordinal"],
            "profile_id": profile_id,
            "profile": member["profile"],
            "legality": member["legality"],
            "numerical_result_hash": member["numerical_result_hash"],
            "numerical_summary": member["numerical_summary"],
            "candidate_id": factor["candidate_id"],
            "hardware": factor["hardware"],
            "capability": join["capability"],
            "packedkv_selector_valid": join["packedkv_selector_valid"],
            "packedkv_selector_evidence": join[
                "packedkv_selector_evidence"
            ],
            "validity": validity,
            **validity,
            "deployment_valid": join["deployment_valid"],
            "metrics": factor["metrics"],
            "error_code": factor["error_code"],
            "error_message": factor["error_message"],
        }
        record_hash = str(join.get("joined_record_hash", ""))
        if not record_hash or record_hash != _content_hash(body):
            raise ValueError("factor binding does not reconstruct its joined row")
        if record_hash in factor_hash_by_record_hash:
            raise ValueError("factor bindings reconstruct duplicate joined rows")
        factor_hash_by_record_hash[record_hash] = factor_hash
        expanded.append(
            {
                **body,
                "record_hash": record_hash,
                "retention_labels": tuple(labels),
            }
        )
    if set(factor_by_hash) != {key[0] for key in binding_keys}:
        raise ValueError("retained factors and bindings do not have exact coverage")

    retention = header.get("retention")
    aggregates = header.get("profile_aggregates")
    if not isinstance(retention, Mapping) or not isinstance(aggregates, list):
        raise ValueError("factorized hardware summary is incomplete")
    if (
        int(retention.get("stored_factor_result_count", -1)) != len(factors)
        or int(retention.get("stored_result_count", -1)) != len(expanded)
        or int(metadata.get("stored_factor_result_count", -1)) != len(factors)
        or int(metadata.get("stored_result_count", -1)) != len(expanded)
        or metadata.get("storage_revision") != HARDWARE_STORAGE_REVISION
    ):
        raise ValueError("factorized hardware stored counts differ")
    labels: dict[str, int] = {}
    result_by_hash = {}
    for row in expanded:
        result_by_hash[row["record_hash"]] = row
        for label in row["retention_labels"]:
            labels[label] = labels.get(label, 0) + 1
    if dict(sorted(labels.items())) != retention.get("label_counts"):
        raise ValueError("factorized hardware retention labels differ")
    factor_population = int(retention.get("factor_population_count", -1))
    rankable_population = int(
        retention.get("rankable_factor_population_count", -1)
    )
    dominated_population = int(
        retention.get("dominated_factor_population_count", -1)
    )
    unrankable_population = int(
        retention.get("unrankable_factor_population_count", -1)
    )
    exact_frontier_count = int(retention.get("exact_frontier_count", -1))
    sample_limit = int(retention.get("sample_limit", -1))
    sampled_count = labels.get("sampled_dominated", 0) + labels.get(
        "sampled_unrankable", 0
    )
    if (
        factor_population != factor_count
        or min(
            rankable_population,
            dominated_population,
            unrankable_population,
            exact_frontier_count,
            sample_limit,
        )
        < 0
        or rankable_population + unrankable_population != factor_population
        or dominated_population + exact_frontier_count != rankable_population
        or sampled_count
        != min(
            sample_limit,
            dominated_population + unrankable_population,
        )
        or retention.get("sampling_policy")
        != "smallest_sha256_over_exact_factor_dominated_and_unrankable_population"
        or labels.get("exact_frontier", 0) != exact_frontier_count
        or labels.get("sampled_dominated", 0)
        != int(retention.get("sampled_dominated_count", -1))
        or labels.get("sampled_unrankable", 0)
        != int(retention.get("sampled_unrankable_count", -1))
    ):
        raise ValueError("factorized hardware retention populations differ")

    aggregate_ids = set()
    total_count = 0
    frontier_hashes = set()
    for aggregate in aggregates:
        if not isinstance(aggregate, Mapping):
            raise TypeError("hardware profile aggregate must be a mapping")
        profile_id = str(aggregate.get("profile_id", ""))
        matching_member = next(
            (
                (factor_id, member)
                for (factor_id, member_profile_id), member
                in member_by_factor_profile.items()
                if member_profile_id == profile_id
            ),
            None,
        )
        if (
            not profile_id
            or profile_id in aggregate_ids
            or matching_member is None
        ):
            raise ValueError("hardware profile aggregates are duplicated or unknown")
        aggregate_ids.add(profile_id)
        factor_id, member = matching_member
        expected_total = int(
            membership_by_factor[factor_id]["passing_candidate_count"]
        )
        total = int(aggregate.get("total_count", -1))
        deployment_valid = int(aggregate.get("deployment_valid_count", -1))
        valid = int(aggregate.get("valid_count", -1))
        errors = int(aggregate.get("error_count", -1))
        error_counts = aggregate.get("error_code_counts")
        local_frontier = aggregate.get("local_frontier")
        local_extrema = aggregate.get("local_extrema")
        if (
            total != expected_total
            or not 0 <= valid <= deployment_valid <= total
            or not 0 <= errors <= total
            or not isinstance(error_counts, Mapping)
            or sum(int(value) for value in error_counts.values()) != errors
            or aggregate.get("profile") != member.get("profile")
            or aggregate.get("numerical_result_hash")
            != member.get("numerical_result_hash")
            or not isinstance(local_frontier, list)
            or int(aggregate.get("local_frontier_count", -1))
            != len(local_frontier)
            or not isinstance(local_extrema, Mapping)
            or set(local_extrema)
            != {"fastest", "lowest_energy", "best_edp"}
        ):
            raise ValueError("hardware profile aggregate counts are invalid")
        total_count += total
        local_hashes = set()
        rows_by_tier: dict[str, list[Mapping[str, Any]]] = {}
        for reference in local_frontier:
            if not isinstance(reference, Mapping):
                raise TypeError("hardware frontier reference must be a mapping")
            record_hash = str(reference.get("record_hash", ""))
            row = result_by_hash.get(record_hash)
            tier = reference.get("energy_tier")
            if (
                row is None
                or row.get("profile_id") != profile_id
                or row.get("candidate_id") != reference.get("candidate_id")
                or factor_hash_by_record_hash.get(record_hash)
                != reference.get("factor_record_hash")
                or tier not in {"analytic_anchored", "dc_calibrated"}
                or record_hash in local_hashes
                or "profile_frontier" not in row["retention_labels"]
                or _promotion_retention_key(row) is None
            ):
                raise ValueError("hardware local frontier reference is invalid")
            values = _plot_retention_values(row)
            if values is None or values.get("energy_tier") != tier:
                raise ValueError("hardware local frontier metrics are invalid")
            local_hashes.add(record_hash)
            frontier_hashes.add(record_hash)
            rows_by_tier.setdefault(str(tier), []).append(row)
        for tier_rows in rows_by_tier.values():
            for index, left in enumerate(tier_rows):
                left_values = _plot_retention_values(left)
                if any(
                    _frontier_retention_dominates(
                        right,
                        _plot_retention_values(right),
                        left,
                        left_values,
                    )
                    for right in tier_rows[:index] + tier_rows[index + 1 :]
                ):
                    raise ValueError("hardware local frontier is dominated")
        if valid == 0:
            if local_frontier or any(local_extrema.values()):
                raise ValueError("zero-valid profile cannot carry a frontier")
            continue
        if not local_frontier:
            raise ValueError("valid hardware profile has no local frontier")
        for name, label in (
            ("fastest", "profile_fastest"),
            ("lowest_energy", "profile_lowest_energy"),
            ("best_edp", "profile_best_edp"),
        ):
            references = local_extrema[name]
            if not isinstance(references, list) or len(references) != len(
                rows_by_tier
            ):
                raise ValueError("hardware local extrema are incomplete")
            seen_tiers = set()
            for reference in references:
                if not isinstance(reference, Mapping):
                    raise TypeError("hardware extremum reference must be a mapping")
                tier = str(reference.get("energy_tier", ""))
                record_hash = str(reference.get("record_hash", ""))
                row = result_by_hash.get(record_hash)
                if (
                    tier in seen_tiers
                    or tier not in rows_by_tier
                    or record_hash not in local_hashes
                    or row is None
                    or label not in row["retention_labels"]
                    or factor_hash_by_record_hash.get(record_hash)
                    != reference.get("factor_record_hash")
                ):
                    raise ValueError("hardware local extremum is invalid")
                seen_tiers.add(tier)
    if aggregate_ids != profile_ids or total_count != conceptual_count:
        raise ValueError("hardware profile aggregates do not prove cardinality")
    if labels.get("profile_frontier", 0) != len(frontier_hashes):
        raise ValueError("hardware profile frontier labels are inconsistent")
    return tuple(expanded)


def _validate_compact_hardware_artifact(
    *,
    header: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    expected = int(header.get("expected_result_count", -1))
    observed = int(header.get("observed_result_count", -1))
    if expected < 0 or observed != expected:
        raise ValueError("compact hardware enumeration count mismatch")
    enumeration_sha256 = header.get("enumeration_sha256")
    if (
        not isinstance(enumeration_sha256, str)
        or len(enumeration_sha256) != 64
        or any(value not in "0123456789abcdef" for value in enumeration_sha256)
        or metadata.get("enumeration_sha256") != enumeration_sha256
    ):
        raise ValueError("compact hardware enumeration digest is invalid")
    if (
        metadata.get("storage_revision") != LEGACY_COMPACT_STORAGE_REVISION
        or int(metadata.get("result_count", -1)) != observed
        or int(metadata.get("stored_result_count", -1)) != len(results)
    ):
        raise ValueError("compact hardware metadata count mismatch")
    retention = header.get("retention")
    aggregates = header.get("profile_aggregates")
    if not isinstance(retention, Mapping) or not isinstance(aggregates, list):
        raise ValueError("compact hardware summary is incomplete")
    if int(retention.get("stored_result_count", -1)) != len(results):
        raise ValueError("compact hardware stored-row count mismatch")
    if retention.get("sampling_policy") != (
        "smallest_sha256_over_exact_dominated_and_unrankable_population"
    ):
        raise ValueError("compact hardware sampling policy is unsupported")
    sample_seed = retention.get("sample_seed")
    if not isinstance(sample_seed, str) or not sample_seed:
        raise ValueError("compact hardware sample seed is missing")
    labels: dict[str, int] = {}
    result_by_hash = {}
    for row in results:
        record_hash = str(row["record_hash"])
        if record_hash in result_by_hash:
            raise ValueError("compact hardware rows contain duplicate identities")
        result_by_hash[record_hash] = row
        for label in row["retention_labels"]:
            labels[label] = labels.get(label, 0) + 1
    if dict(sorted(labels.items())) != retention.get("label_counts"):
        raise ValueError("compact hardware retention-label counts differ")
    sample_limit = int(retention.get("sample_limit", -1))
    sampled_count = labels.get("sampled_dominated", 0) + labels.get(
        "sampled_unrankable", 0
    )
    if sample_limit < 0 or sampled_count > sample_limit:
        raise ValueError("compact hardware dominated sample exceeds its bound")
    dominated_population = int(retention.get("dominated_population_count", -1))
    unrankable_population = int(retention.get("unrankable_population_count", -1))
    declared_population = int(
        retention.get("declared_scatter_population_count", -1)
    )
    scatter_population = int(retention.get("scatter_population_count", -1))
    if (
        min(
            dominated_population,
            unrankable_population,
            declared_population,
            scatter_population,
        )
        < 0
        or declared_population + unrankable_population != scatter_population
        or sampled_count
        != min(sample_limit, dominated_population + unrankable_population)
    ):
        raise ValueError("compact hardware sampled population is inconsistent")
    for field, actual in (
        ("sampled_dominated_count", labels.get("sampled_dominated", 0)),
        ("sampled_unrankable_count", labels.get("sampled_unrankable", 0)),
        ("exact_frontier_count", labels.get("exact_frontier", 0)),
    ):
        if int(retention.get(field, -1)) != actual:
            raise ValueError(f"compact hardware {field} differs from stored labels")
    if dominated_population < labels.get("sampled_dominated", 0):
        raise ValueError("compact hardware dominated population is inconsistent")
    profile_ids = set()
    total_count = 0
    frontier_hashes = set()
    for aggregate in aggregates:
        if not isinstance(aggregate, Mapping):
            raise TypeError("compact hardware profile aggregate must be a mapping")
        profile_id = str(aggregate.get("profile_id", ""))
        if not profile_id or profile_id in profile_ids:
            raise ValueError("compact hardware profile aggregates are duplicated")
        profile_ids.add(profile_id)
        total = int(aggregate.get("total_count", -1))
        deployment_valid = int(aggregate.get("deployment_valid_count", -1))
        valid = int(aggregate.get("valid_count", -1))
        errors = int(aggregate.get("error_count", -1))
        error_counts = aggregate.get("error_code_counts")
        profile = aggregate.get("profile")
        local_frontier = aggregate.get("local_frontier")
        local_extrema = aggregate.get("local_extrema")
        if (
            total <= 0
            or not 0 <= valid <= deployment_valid <= total
            or not 0 <= errors <= total
            or not isinstance(error_counts, Mapping)
            or sum(int(value) for value in error_counts.values()) != errors
            or not isinstance(profile, Mapping)
            or not isinstance(local_frontier, list)
            or int(aggregate.get("local_frontier_count", -1))
            != len(local_frontier)
            or not isinstance(local_extrema, Mapping)
            or set(local_extrema)
            != {"fastest", "lowest_energy", "best_edp"}
        ):
            raise ValueError("compact hardware profile aggregate counts are invalid")
        total_count += total
        local_hashes = set()
        local_rows_by_tier: dict[str, list[Mapping[str, Any]]] = {}
        for reference in local_frontier:
            if not isinstance(reference, Mapping):
                raise TypeError("compact hardware frontier reference must be a mapping")
            tier = reference.get("energy_tier")
            record_hash = str(reference.get("record_hash", ""))
            candidate_id = reference.get("candidate_id")
            row = result_by_hash.get(record_hash)
            values = _plot_retention_values(row) if row is not None else None
            if (
                tier not in {"analytic_anchored", "dc_calibrated"}
                or not record_hash
                or record_hash in local_hashes
                or row is None
                or row.get("profile_id") != profile_id
                or row.get("candidate_id") != candidate_id
                or "profile_frontier" not in row["retention_labels"]
                or _promotion_retention_key(row) is None
                or values is None
                or values.get("energy_tier") != tier
            ):
                raise ValueError("compact hardware local frontier is invalid")
            local_hashes.add(record_hash)
            frontier_hashes.add(record_hash)
            local_rows_by_tier.setdefault(str(tier), []).append(row)
        for tier_rows in local_rows_by_tier.values():
            for index, left in enumerate(tier_rows):
                left_values = _plot_retention_values(left)
                if any(
                    _frontier_retention_dominates(
                        right,
                        _plot_retention_values(right),
                        left,
                        left_values,
                    )
                    for right in tier_rows[:index] + tier_rows[index + 1 :]
                ):
                    raise ValueError("compact hardware local frontier is dominated")
        if valid == 0:
            if local_frontier or any(local_extrema.values()):
                raise ValueError("zero-valid profile cannot carry a frontier")
            continue
        if not local_frontier:
            raise ValueError("valid compact hardware profile has no local frontier")
        for name, label in (
            ("fastest", "profile_fastest"),
            ("lowest_energy", "profile_lowest_energy"),
            ("best_edp", "profile_best_edp"),
        ):
            references = local_extrema[name]
            if (
                not isinstance(references, list)
                or len(references) != len(local_rows_by_tier)
            ):
                raise ValueError("compact hardware local extrema are incomplete")
            seen_tiers = set()
            for reference in references:
                if not isinstance(reference, Mapping):
                    raise TypeError(
                        "compact hardware extremum reference must be a mapping"
                    )
                tier = str(reference.get("energy_tier", ""))
                record_hash = str(reference.get("record_hash", ""))
                row = result_by_hash.get(record_hash)
                if (
                    tier in seen_tiers
                    or tier not in local_rows_by_tier
                    or record_hash not in local_hashes
                    or row is None
                    or row.get("candidate_id") != reference.get("candidate_id")
                    or label not in row["retention_labels"]
                ):
                    raise ValueError("compact hardware local extremum is invalid")
                seen_tiers.add(tier)
    if total_count != observed:
        raise ValueError("compact hardware per-profile totals are incomplete")
    if labels.get("profile_frontier", 0) != len(frontier_hashes):
        raise ValueError("compact hardware frontier labels are inconsistent")
    if declared_population - int(retention.get("exact_frontier_count", -1)) != (
        dominated_population
    ):
        raise ValueError("compact hardware global frontier population differs")


__all__ = [
    "CHIP_COUNTS",
    "COMPILER_TRACE_EXECUTION_MODE",
    "COMPILER_TRACE_TIMING_SET_SCHEMA",
    "CalibratedEnergy",
    "CapacityBreakdown",
    "DECODE_EXECUTION_MODES",
    "ExactHardwareSpace",
    "ExactHardwareStudy",
    "HardwareCandidate",
    "HardwareEvaluation",
    "HardwareMetrics",
    "HardwareStudyArtifact",
    "FULL_MODEL_DECODE_SCOPE",
    "HARDWARE_FACTOR_BINDING_SCHEMA",
    "HARDWARE_FACTOR_RESULT_SCHEMA",
    "HARDWARE_STORAGE_REVISION",
    "JoinedHardwareResult",
    "LEGACY_COMPACT_STORAGE_REVISION",
    "LEGACY_AGGREGATE_BANDWIDTH_MODE",
    "PhysicalTraffic",
    "ResourceBudget",
    "ResourceBudgetStatus",
    "SRAM_POLICIES",
    "StudyProvenance",
    "load_hardware_artifact",
    "merge_validity",
    "physical_cost_signature",
    "physical_cost_signature_id",
]
