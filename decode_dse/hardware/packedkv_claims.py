"""Fail-closed publication gate for the PackedKV causal ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.hardware.lm_head_service import (
    HEAD_SERVICE_MODE,
    require_content_addressed_id,
)

PADDED_PER_HEAD = "padded_per_head"
DENSE_COMPILER = "dense_compiler"
DENSE_SELECTOR = "dense_selector"
IDEAL_TRAFFIC = "ideal_traffic"
PACKEDKV_MODES = (
    PADDED_PER_HEAD,
    DENSE_COMPILER,
    DENSE_SELECTOR,
    IDEAL_TRAFFIC,
)
REALIZED_BOTTLENECKS = ("memory", "serialization", "compute")
ALGORITHMIC_BOTTLENECKS = ("memory", "compute")
PRECISION_ROLES = ("i8", "i4", "selected")
TOPOLOGY_ROLES = ("gqa1", "gqa4", "gqa8", "mqa")
REQUIRED_PROVENANCE = (
    "software",
    "compiler",
    "emulator",
    "rtl",
    "analytical",
    "synthesis",
)
STEP_COMPOSITION = "max_compute_memory"
TPOT_SCOPE = "whole_model_remote_bf16_head"
CAPACITY_LIMITERS = ("hbm", "matrix_sram", "vector_sram", "isa_batch")
CAPACITY_THROUGHPUT_SCOPES = (
    "cross_stack_measured",
    "analytical_projection",
)
EVIDENCE_SCHEMA = "packedkv-publication-evidence"
REPORT_SCHEMA = "packedkv-publication-report"


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _require_fields(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields differ")


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _aligned_bytes(bits: int, alignment_bytes: int) -> int:
    raw_bytes = math.ceil(bits / 8)
    return math.ceil(raw_bytes / alignment_bytes) * alignment_bytes


def _unique_pairs(
    values: Iterable[Sequence[str]],
    *,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    pairs = tuple(sorted((str(pair[0]), str(pair[1])) for pair in values))
    if any(not key or not value for key, value in pairs):
        raise ValueError(f"{field_name} keys and values must be non-empty")
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(f"{field_name} contains duplicate keys")
    return pairs


@dataclass(frozen=True)
class PackedKVPrecision:
    """One precision role and its physical element/scale representation."""

    role: str
    format_id: str
    element_bits: int
    scale_bits: int = 8
    block_size: int = 8

    def __post_init__(self) -> None:
        if self.role not in PRECISION_ROLES:
            raise ValueError(f"unknown precision role {self.role!r}")
        if not self.format_id:
            raise ValueError("format_id must be non-empty")
        if self.element_bits <= 0 or self.scale_bits < 0 or self.block_size <= 0:
            raise ValueError("precision widths and block size are invalid")
        if self.block_size != 8:
            raise ValueError(
                "deployment evidence requires the native block size of eight"
            )
        if (
            self.format_id.upper().startswith("MXINT")
            and self.element_bits not in {2, 4, 8}
        ):
            raise ValueError("MXINT deployment evidence requires 2, 4, or 8 bits")
        if self.role == "i8" and self.element_bits != 8:
            raise ValueError("the i8 control must use eight element bits")
        if self.role == "i4" and self.element_bits != 4:
            raise ValueError("the i4 control must use four element bits")

    @property
    def effective_bits(self) -> float:
        return self.element_bits + self.scale_bits / self.block_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "format_id": self.format_id,
            "element_bits": self.element_bits,
            "scale_bits": self.scale_bits,
            "block_size": self.block_size,
            "effective_bits": self.effective_bits,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackedKVPrecision":
        _require_fields(
            value,
            {
                "role",
                "format_id",
                "element_bits",
                "scale_bits",
                "block_size",
                "effective_bits",
            },
            "PackedKV precision",
        )
        result = cls(
            role=str(value["role"]),
            format_id=str(value["format_id"]),
            element_bits=int(value["element_bits"]),
            scale_bits=int(value.get("scale_bits", 8)),
            block_size=int(value.get("block_size", 8)),
        )
        if not _close(result.effective_bits, float(value["effective_bits"])):
            raise ValueError("PackedKV effective bits are inconsistent")
        return result


@dataclass(frozen=True)
class AttentionTopology:
    """One attention topology in the fixed PackedKV geometry."""

    role: str
    query_heads: int
    kv_heads: int
    head_dim: int
    mlen: int = 1024

    def __post_init__(self) -> None:
        if self.role not in TOPOLOGY_ROLES:
            raise ValueError(f"unknown topology role {self.role!r}")
        for name in ("query_heads", "kv_heads", "head_dim", "mlen"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must contain complete KV groups")
        if self.head_dim > self.mlen or self.mlen % self.head_dim:
            raise ValueError("MLEN must contain complete KV heads")
        ratio = self.query_heads // self.kv_heads
        expected_ratio = {"gqa1": 1, "gqa4": 4, "gqa8": 8}
        if self.role in expected_ratio and ratio != expected_ratio[self.role]:
            raise ValueError(f"{self.role} requires a {expected_ratio[self.role]}:1 ratio")
        if self.role == "mqa" and (self.kv_heads != 1 or self.query_heads <= 1):
            raise ValueError("mqa requires one KV head and multiple query heads")

    @property
    def active_elements(self) -> int:
        return self.kv_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "mlen": self.mlen,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttentionTopology":
        _require_fields(
            value,
            {"role", "query_heads", "kv_heads", "head_dim", "mlen"},
            "attention topology",
        )
        return cls(
            role=str(value["role"]),
            query_heads=int(value["query_heads"]),
            kv_heads=int(value["kv_heads"]),
            head_dim=int(value["head_dim"]),
            mlen=int(value.get("mlen", 1024)),
        )


@dataclass(frozen=True)
class AblationEnvironment:
    """Fields held constant across the causal layout comparison."""

    model_id: str
    model_revision: str
    workload_id: str
    geometry_id: str
    numerical_reference_id: str
    capacity_model_id: str
    capacity_model_validated: bool
    output_head_artifact_sha256: str
    output_head_service_id: str
    output_head_provenance_id: str
    output_head_mode: str
    tpot_scope: str
    serving_execution_evidence_id: str
    executable_batches: tuple[int, ...]
    sequence_length: int
    latency_batch: int
    clock_hz: float
    hbm_bandwidth_bytes_per_s: float
    hbm_capacity_bytes: int
    alignment_bytes: int = 64

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "workload_id",
            "geometry_id",
            "numerical_reference_id",
            "capacity_model_id",
            "output_head_artifact_sha256",
            "output_head_service_id",
            "output_head_provenance_id",
            "output_head_mode",
            "tpot_scope",
            "serving_execution_evidence_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        require_content_addressed_id(
            "numerical_reference_id",
            self.numerical_reference_id,
        )
        require_content_addressed_id(
            "capacity_model_id",
            self.capacity_model_id,
        )
        _require_sha256(
            "output_head_artifact_sha256",
            self.output_head_artifact_sha256,
        )
        require_content_addressed_id(
            "output_head_service_id",
            self.output_head_service_id,
            prefix="bf16-head-service-",
        )
        require_content_addressed_id(
            "output_head_provenance_id",
            self.output_head_provenance_id,
            prefix="bf16-head-provenance-",
        )
        if self.output_head_mode != HEAD_SERVICE_MODE:
            raise ValueError("PackedKV evidence requires the remote BF16 head")
        if self.tpot_scope != TPOT_SCOPE:
            raise ValueError("PackedKV TPOT must use the whole-model boundary")
        if not isinstance(self.capacity_model_validated, bool):
            raise TypeError("capacity_model_validated must be boolean")
        require_content_addressed_id(
            "serving_execution_evidence_id",
            self.serving_execution_evidence_id,
        )
        batches = tuple(sorted(set(self.executable_batches)))
        if (
            not batches
            or any(
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch <= 0
                for batch in batches
            )
        ):
            raise ValueError("executable_batches must be positive integers")
        if self.latency_batch not in batches:
            raise ValueError(
                "fixed-batch TPOT requires cross-stack execution evidence"
            )
        object.__setattr__(self, "executable_batches", batches)
        for name in (
            "sequence_length",
            "latency_batch",
            "hbm_capacity_bytes",
            "alignment_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        _require_positive("clock_hz", self.clock_hz)
        _require_positive(
            "hbm_bandwidth_bytes_per_s",
            self.hbm_bandwidth_bytes_per_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "workload_id": self.workload_id,
            "geometry_id": self.geometry_id,
            "numerical_reference_id": self.numerical_reference_id,
            "capacity_model_id": self.capacity_model_id,
            "capacity_model_validated": self.capacity_model_validated,
            "output_head_artifact_sha256": (
                self.output_head_artifact_sha256
            ),
            "output_head_service_id": self.output_head_service_id,
            "output_head_provenance_id": self.output_head_provenance_id,
            "output_head_mode": self.output_head_mode,
            "tpot_scope": self.tpot_scope,
            "serving_execution_evidence_id": (
                self.serving_execution_evidence_id
            ),
            "executable_batches": list(self.executable_batches),
            "sequence_length": self.sequence_length,
            "latency_batch": self.latency_batch,
            "clock_hz": self.clock_hz,
            "hbm_bandwidth_bytes_per_s": self.hbm_bandwidth_bytes_per_s,
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AblationEnvironment":
        _require_fields(
            value,
            {
                "model_id",
                "model_revision",
                "workload_id",
                "geometry_id",
                "numerical_reference_id",
                "capacity_model_id",
                "capacity_model_validated",
                "output_head_artifact_sha256",
                "output_head_service_id",
                "output_head_provenance_id",
                "output_head_mode",
                "tpot_scope",
                "serving_execution_evidence_id",
                "executable_batches",
                "sequence_length",
                "latency_batch",
                "clock_hz",
                "hbm_bandwidth_bytes_per_s",
                "hbm_capacity_bytes",
                "alignment_bytes",
            },
            "ablation environment",
        )
        return cls(
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            workload_id=str(value["workload_id"]),
            geometry_id=str(value["geometry_id"]),
            numerical_reference_id=str(value["numerical_reference_id"]),
            capacity_model_id=str(value["capacity_model_id"]),
            capacity_model_validated=value["capacity_model_validated"],
            output_head_artifact_sha256=str(
                value["output_head_artifact_sha256"]
            ),
            output_head_service_id=str(value["output_head_service_id"]),
            output_head_provenance_id=str(
                value["output_head_provenance_id"]
            ),
            output_head_mode=str(value["output_head_mode"]),
            tpot_scope=str(value["tpot_scope"]),
            serving_execution_evidence_id=str(
                value["serving_execution_evidence_id"]
            ),
            executable_batches=tuple(
                int(batch) for batch in value["executable_batches"]
            ),
            sequence_length=int(value["sequence_length"]),
            latency_batch=int(value["latency_batch"]),
            clock_hz=float(value["clock_hz"]),
            hbm_bandwidth_bytes_per_s=float(
                value["hbm_bandwidth_bytes_per_s"]
            ),
            hbm_capacity_bytes=int(value["hbm_capacity_bytes"]),
            alignment_bytes=int(value.get("alignment_bytes", 64)),
        )


@dataclass(frozen=True)
class PackedKVModeMeasurement:
    """Physical, numerical, and serving observations for one layout mode."""

    mode: str
    storage_bytes_per_sequence_token: int
    read_bytes_per_sequence_token: int
    write_bytes_per_appended_token: int
    feasible_batch: int
    tpot_ms: float
    decoder_tpot_ms: float
    head_service_tpot_ms: float
    peak_compute_ms: float
    ideal_compute_ms: float
    realized_compute_ms: float
    memory_ms: float
    capacity_limited_tpot_ms: float
    capacity_limited_tokens_per_s: float
    capacity_limiter: str
    capacity_throughput_scope: str
    max_abs_error: float
    max_rel_error: float
    compared_values: int
    numerical_evidence_id: str
    timing_evidence_id: str
    capacity_evidence_id: str
    bottleneck: str
    classical_roofline_bottleneck: str
    architecture_issue_bottleneck: str
    algorithmic_bottleneck: str
    step_composition: str = STEP_COMPOSITION

    def __post_init__(self) -> None:
        if self.mode not in PACKEDKV_MODES:
            raise ValueError(f"unknown PackedKV mode {self.mode!r}")
        for name in (
            "storage_bytes_per_sequence_token",
            "read_bytes_per_sequence_token",
            "write_bytes_per_appended_token",
            "feasible_batch",
            "compared_values",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        _require_positive("tpot_ms", self.tpot_ms)
        _require_positive("decoder_tpot_ms", self.decoder_tpot_ms)
        _require_positive("head_service_tpot_ms", self.head_service_tpot_ms)
        _require_positive("peak_compute_ms", self.peak_compute_ms)
        _require_positive("ideal_compute_ms", self.ideal_compute_ms)
        _require_positive("realized_compute_ms", self.realized_compute_ms)
        _require_positive("memory_ms", self.memory_ms)
        _require_positive(
            "capacity_limited_tpot_ms",
            self.capacity_limited_tpot_ms,
        )
        _require_positive(
            "capacity_limited_tokens_per_s",
            self.capacity_limited_tokens_per_s,
        )
        _require_nonnegative("max_abs_error", self.max_abs_error)
        _require_nonnegative("max_rel_error", self.max_rel_error)
        require_content_addressed_id(
            "numerical_evidence_id",
            self.numerical_evidence_id,
        )
        require_content_addressed_id(
            "timing_evidence_id",
            self.timing_evidence_id,
            prefix="timing-",
        )
        require_content_addressed_id(
            "capacity_evidence_id",
            self.capacity_evidence_id,
        )
        if self.step_composition != STEP_COMPOSITION:
            raise ValueError("PackedKV timing composition is unsupported")
        if self.ideal_compute_ms < self.peak_compute_ms:
            raise ValueError("ideal issue cannot beat the compute ceiling")
        if self.realized_compute_ms < self.ideal_compute_ms:
            raise ValueError("realized compute cannot beat the ideal issue time")
        expected_decoder_tpot = max(
            self.realized_compute_ms,
            self.memory_ms,
        )
        if not _close(self.decoder_tpot_ms, expected_decoder_tpot):
            raise ValueError("decoder TPOT differs from max(compute, memory)")
        if not _close(
            self.tpot_ms,
            self.decoder_tpot_ms + self.head_service_tpot_ms,
        ):
            raise ValueError("whole-model TPOT omits the remote-head boundary")
        expected_capacity_tps = (
            self.feasible_batch * 1000.0 / self.capacity_limited_tpot_ms
        )
        if not _close(
            self.capacity_limited_tokens_per_s,
            expected_capacity_tps,
        ):
            raise ValueError("capacity throughput differs from batch/TPOT")
        if self.capacity_limiter not in CAPACITY_LIMITERS:
            raise ValueError("capacity_limiter is unsupported")
        if self.capacity_throughput_scope not in CAPACITY_THROUGHPUT_SCOPES:
            raise ValueError("capacity_throughput_scope is unsupported")
        if self.bottleneck not in REALIZED_BOTTLENECKS:
            raise ValueError(
                f"unknown realized bottleneck {self.bottleneck!r}"
            )
        if self.algorithmic_bottleneck not in ALGORITHMIC_BOTTLENECKS:
            raise ValueError(
                "algorithmic bottleneck must be memory or compute"
            )
        if self.classical_roofline_bottleneck not in ALGORITHMIC_BOTTLENECKS:
            raise ValueError(
                "classical roofline bottleneck must be memory or compute"
            )
        if self.architecture_issue_bottleneck not in ALGORITHMIC_BOTTLENECKS:
            raise ValueError(
                "architecture issue bottleneck must be memory or compute"
            )
        expected_classical = (
            "memory"
            if self.memory_ms >= self.peak_compute_ms
            else "compute"
        )
        expected_architecture = (
            "memory"
            if self.memory_ms >= self.ideal_compute_ms
            else "compute"
        )
        expected_realized = (
            "memory"
            if self.memory_ms >= self.realized_compute_ms
            else "serialization"
            if (
                expected_architecture == "memory"
                and self.realized_compute_ms > self.ideal_compute_ms
            )
            else "compute"
        )
        if self.classical_roofline_bottleneck != expected_classical:
            raise ValueError("classical roofline bottleneck differs from timing")
        if self.architecture_issue_bottleneck != expected_architecture:
            raise ValueError("architecture issue bottleneck differs from timing")
        if self.algorithmic_bottleneck != expected_architecture:
            raise ValueError("algorithmic bottleneck differs from timing")
        if self.bottleneck != expected_realized:
            raise ValueError("realized bottleneck differs from timing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "storage_bytes_per_sequence_token": (
                self.storage_bytes_per_sequence_token
            ),
            "read_bytes_per_sequence_token": self.read_bytes_per_sequence_token,
            "write_bytes_per_appended_token": self.write_bytes_per_appended_token,
            "feasible_batch": self.feasible_batch,
            "tpot_ms": self.tpot_ms,
            "decoder_tpot_ms": self.decoder_tpot_ms,
            "head_service_tpot_ms": self.head_service_tpot_ms,
            "peak_compute_ms": self.peak_compute_ms,
            "ideal_compute_ms": self.ideal_compute_ms,
            "realized_compute_ms": self.realized_compute_ms,
            "memory_ms": self.memory_ms,
            "step_composition": self.step_composition,
            "capacity_limited_tpot_ms": self.capacity_limited_tpot_ms,
            "capacity_limited_tokens_per_s": self.capacity_limited_tokens_per_s,
            "capacity_limiter": self.capacity_limiter,
            "capacity_throughput_scope": self.capacity_throughput_scope,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "compared_values": self.compared_values,
            "numerical_evidence_id": self.numerical_evidence_id,
            "timing_evidence_id": self.timing_evidence_id,
            "capacity_evidence_id": self.capacity_evidence_id,
            "bottleneck": self.bottleneck,
            "classical_roofline_bottleneck": (
                self.classical_roofline_bottleneck
            ),
            "architecture_issue_bottleneck": (
                self.architecture_issue_bottleneck
            ),
            "algorithmic_bottleneck": self.algorithmic_bottleneck,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackedKVModeMeasurement":
        _require_fields(
            value,
            {
                "mode",
                "storage_bytes_per_sequence_token",
                "read_bytes_per_sequence_token",
                "write_bytes_per_appended_token",
                "feasible_batch",
                "tpot_ms",
                "decoder_tpot_ms",
                "head_service_tpot_ms",
                "peak_compute_ms",
                "ideal_compute_ms",
                "realized_compute_ms",
                "memory_ms",
                "step_composition",
                "capacity_limited_tpot_ms",
                "capacity_limited_tokens_per_s",
                "capacity_limiter",
                "capacity_throughput_scope",
                "max_abs_error",
                "max_rel_error",
                "compared_values",
                "numerical_evidence_id",
                "timing_evidence_id",
                "capacity_evidence_id",
                "bottleneck",
                "classical_roofline_bottleneck",
                "architecture_issue_bottleneck",
                "algorithmic_bottleneck",
            },
            "PackedKV measurement",
        )
        return cls(
            mode=str(value["mode"]),
            storage_bytes_per_sequence_token=int(
                value["storage_bytes_per_sequence_token"]
            ),
            read_bytes_per_sequence_token=int(
                value["read_bytes_per_sequence_token"]
            ),
            write_bytes_per_appended_token=int(
                value["write_bytes_per_appended_token"]
            ),
            feasible_batch=int(value["feasible_batch"]),
            tpot_ms=float(value["tpot_ms"]),
            decoder_tpot_ms=float(value["decoder_tpot_ms"]),
            head_service_tpot_ms=float(value["head_service_tpot_ms"]),
            peak_compute_ms=float(value["peak_compute_ms"]),
            ideal_compute_ms=float(value["ideal_compute_ms"]),
            realized_compute_ms=float(value["realized_compute_ms"]),
            memory_ms=float(value["memory_ms"]),
            step_composition=str(value["step_composition"]),
            capacity_limited_tpot_ms=float(
                value["capacity_limited_tpot_ms"]
            ),
            capacity_limited_tokens_per_s=float(
                value["capacity_limited_tokens_per_s"]
            ),
            capacity_limiter=str(value["capacity_limiter"]),
            capacity_throughput_scope=str(
                value["capacity_throughput_scope"]
            ),
            max_abs_error=float(value["max_abs_error"]),
            max_rel_error=float(value["max_rel_error"]),
            compared_values=int(value["compared_values"]),
            numerical_evidence_id=str(value["numerical_evidence_id"]),
            timing_evidence_id=str(value["timing_evidence_id"]),
            capacity_evidence_id=str(value["capacity_evidence_id"]),
            bottleneck=str(value["bottleneck"]),
            classical_roofline_bottleneck=str(
                value["classical_roofline_bottleneck"]
            ),
            architecture_issue_bottleneck=str(
                value["architecture_issue_bottleneck"]
            ),
            algorithmic_bottleneck=str(value["algorithmic_bottleneck"]),
        )


@dataclass(frozen=True)
class PackedKVAblationGroup:
    """The four causal modes for one precision and attention topology."""

    precision: PackedKVPrecision
    topology: AttentionTopology
    measurements: tuple[PackedKVModeMeasurement, ...]

    def __post_init__(self) -> None:
        modes = [measurement.mode for measurement in self.measurements]
        if len(modes) != len(set(modes)):
            raise ValueError("an ablation group contains duplicate modes")

    @property
    def key(self) -> tuple[str, str]:
        return self.precision.role, self.topology.role

    def by_mode(self) -> dict[str, PackedKVModeMeasurement]:
        return {measurement.mode: measurement for measurement in self.measurements}

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision.to_dict(),
            "topology": self.topology.to_dict(),
            "measurements": [
                measurement.to_dict() for measurement in self.measurements
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackedKVAblationGroup":
        _require_fields(
            value,
            {"precision", "topology", "measurements"},
            "PackedKV ablation group",
        )
        return cls(
            precision=PackedKVPrecision.from_dict(value["precision"]),
            topology=AttentionTopology.from_dict(value["topology"]),
            measurements=tuple(
                PackedKVModeMeasurement.from_dict(measurement)
                for measurement in value["measurements"]
            ),
        )


@dataclass(frozen=True)
class SelectorSynthesisEvidence:
    """Common-corner selector-off/on synthesis comparison."""

    baseline_area_mm2: float
    selector_area_mm2: float
    baseline_fmax_hz: float
    selector_fmax_hz: float
    constraint_ns: float
    tool_id: str
    library_id: str
    process_corner: str
    selector_off_report_hash: str
    selector_on_report_hash: str
    calibrated: bool

    def __post_init__(self) -> None:
        for name in (
            "baseline_area_mm2",
            "selector_area_mm2",
            "baseline_fmax_hz",
            "selector_fmax_hz",
            "constraint_ns",
        ):
            _require_positive(name, float(getattr(self, name)))
        for name in (
            "tool_id",
            "library_id",
            "process_corner",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        _require_sha256(
            "selector_off_report_hash",
            self.selector_off_report_hash,
        )
        _require_sha256(
            "selector_on_report_hash",
            self.selector_on_report_hash,
        )
        if self.selector_off_report_hash == self.selector_on_report_hash:
            raise ValueError("selector-off/on synthesis reports must be distinct")
        if not isinstance(self.calibrated, bool):
            raise TypeError("selector calibration state must be boolean")

    @property
    def area_overhead(self) -> float:
        return (self.selector_area_mm2 - self.baseline_area_mm2) / self.baseline_area_mm2

    @property
    def fmax_loss(self) -> float:
        return (self.baseline_fmax_hz - self.selector_fmax_hz) / self.baseline_fmax_hz

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_area_mm2": self.baseline_area_mm2,
            "selector_area_mm2": self.selector_area_mm2,
            "baseline_fmax_hz": self.baseline_fmax_hz,
            "selector_fmax_hz": self.selector_fmax_hz,
            "constraint_ns": self.constraint_ns,
            "tool_id": self.tool_id,
            "library_id": self.library_id,
            "process_corner": self.process_corner,
            "selector_off_report_hash": self.selector_off_report_hash,
            "selector_on_report_hash": self.selector_on_report_hash,
            "calibrated": self.calibrated,
            "area_overhead": self.area_overhead,
            "fmax_loss": self.fmax_loss,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectorSynthesisEvidence":
        _require_fields(
            value,
            {
                "baseline_area_mm2",
                "selector_area_mm2",
                "baseline_fmax_hz",
                "selector_fmax_hz",
                "constraint_ns",
                "tool_id",
                "library_id",
                "process_corner",
                "selector_off_report_hash",
                "selector_on_report_hash",
                "calibrated",
                "area_overhead",
                "fmax_loss",
            },
            "selector synthesis",
        )
        result = cls(
            baseline_area_mm2=float(value["baseline_area_mm2"]),
            selector_area_mm2=float(value["selector_area_mm2"]),
            baseline_fmax_hz=float(value["baseline_fmax_hz"]),
            selector_fmax_hz=float(value["selector_fmax_hz"]),
            constraint_ns=float(value["constraint_ns"]),
            tool_id=str(value["tool_id"]),
            library_id=str(value["library_id"]),
            process_corner=str(value["process_corner"]),
            selector_off_report_hash=str(value["selector_off_report_hash"]),
            selector_on_report_hash=str(value["selector_on_report_hash"]),
            calibrated=value["calibrated"],
        )
        if not _close(result.area_overhead, float(value["area_overhead"])):
            raise ValueError("selector area overhead is inconsistent")
        if not _close(result.fmax_loss, float(value["fmax_loss"])):
            raise ValueError("selector Fmax loss is inconsistent")
        return result


@dataclass(frozen=True)
class PipelineValidationEvidence:
    """Trace evidence required before TPOT is a headline outcome."""

    overlap_validated: bool
    compiler_trace_hash: str = ""
    emulator_trace_hash: str = ""
    rtl_trace_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.overlap_validated, bool):
            raise TypeError("overlap_validated must be boolean")
        traces = (
            self.compiler_trace_hash,
            self.emulator_trace_hash,
            self.rtl_trace_hash,
        )
        if self.overlap_validated:
            for name, value in zip(
                (
                    "compiler_trace_hash",
                    "emulator_trace_hash",
                    "rtl_trace_hash",
                ),
                traces,
            ):
                _require_sha256(name, value)
        elif any(traces):
            raise ValueError("unvalidated overlap cannot retain trace evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlap_validated": self.overlap_validated,
            "compiler_trace_hash": self.compiler_trace_hash,
            "emulator_trace_hash": self.emulator_trace_hash,
            "rtl_trace_hash": self.rtl_trace_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineValidationEvidence":
        _require_fields(
            value,
            {
                "overlap_validated",
                "compiler_trace_hash",
                "emulator_trace_hash",
                "rtl_trace_hash",
            },
            "pipeline validation",
        )
        return cls(
            overlap_validated=value["overlap_validated"],
            compiler_trace_hash=str(value.get("compiler_trace_hash", "")),
            emulator_trace_hash=str(value.get("emulator_trace_hash", "")),
            rtl_trace_hash=str(value.get("rtl_trace_hash", "")),
        )


@dataclass(frozen=True)
class PackedKVPublicationEvidence:
    """Complete evidence bundle for one fixed causal ablation."""

    environment: AblationEnvironment
    groups: tuple[PackedKVAblationGroup, ...]
    selector_synthesis: SelectorSynthesisEvidence
    pipeline_validation: PipelineValidationEvidence
    provenance: tuple[tuple[str, str], ...]
    schema_version: str = EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported evidence schema {self.schema_version!r}")
        object.__setattr__(
            self,
            "provenance",
            _unique_pairs(self.provenance, field_name="provenance"),
        )
        keys = [group.key for group in self.groups]
        if len(keys) != len(set(keys)):
            raise ValueError("evidence contains duplicate precision/topology groups")
        for key, digest in self.provenance:
            if key in REQUIRED_PROVENANCE:
                _require_sha256(f"provenance.{key}", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "selector_synthesis": self.selector_synthesis.to_dict(),
            "pipeline_validation": self.pipeline_validation.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackedKVPublicationEvidence":
        _require_fields(
            value,
            {
                "schema_version",
                "environment",
                "groups",
                "selector_synthesis",
                "pipeline_validation",
                "provenance",
            },
            "PackedKV publication evidence",
        )
        provenance = value.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            environment=AblationEnvironment.from_dict(value["environment"]),
            groups=tuple(
                PackedKVAblationGroup.from_dict(group)
                for group in value["groups"]
            ),
            selector_synthesis=SelectorSynthesisEvidence.from_dict(
                value["selector_synthesis"]
            ),
            pipeline_validation=PipelineValidationEvidence.from_dict(
                value.get("pipeline_validation", {})
            ),
            provenance=tuple(
                (str(key), str(item)) for key, item in provenance.items()
            ),
        )


@dataclass(frozen=True)
class PackedKVGatePolicy:
    """Thresholds for accepting a PackedKV publication claim."""

    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-5
    byte_reduction: float = 7.5
    selector_area_overhead: float = 0.01
    selector_fmax_loss: float = 0.03
    capacity_improvement: float = 2.0
    tpot_improvement: float = 0.15
    max_workload_regression: float = 0.02

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            _require_nonnegative(name, value)
        if self.byte_reduction < 1 or self.capacity_improvement < 1:
            raise ValueError("reduction and capacity thresholds must be at least one")

    def to_dict(self) -> dict[str, float]:
        return {
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "byte_reduction": self.byte_reduction,
            "selector_area_overhead": self.selector_area_overhead,
            "selector_fmax_loss": self.selector_fmax_loss,
            "capacity_improvement": self.capacity_improvement,
            "tpot_improvement": self.tpot_improvement,
            "max_workload_regression": self.max_workload_regression,
        }


@dataclass(frozen=True)
class PackedKVGateCheck:
    name: str
    passed: bool
    observed: Any
    requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
        }


@dataclass(frozen=True)
class PackedKVPublicationReport:
    """Derived headline outcome and every fail-closed gate."""

    passed: bool
    headline_outcome: str | None
    checks: tuple[PackedKVGateCheck, ...]
    evidence_hash: str
    policy: PackedKVGatePolicy
    reduction_by_precision: tuple[tuple[str, float], ...]
    selected_capacity_improvement: float | None
    selected_capacity_throughput_improvement: float | None
    selected_capacity_throughput_scope: str | None
    selected_tpot_improvement: float | None
    maximum_regression: float | None
    bottleneck_counts: tuple[tuple[str, int], ...]
    classical_roofline_bottleneck_counts: tuple[tuple[str, int], ...]
    architecture_issue_bottleneck_counts: tuple[tuple[str, int], ...]
    algorithmic_bottleneck_counts: tuple[tuple[str, int], ...]
    schema_version: str = REPORT_SCHEMA

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "headline_outcome": self.headline_outcome,
            "checks": [check.to_dict() for check in self.checks],
            "failures": list(self.failures),
            "evidence_hash": self.evidence_hash,
            "policy": self.policy.to_dict(),
            "reduction_by_precision": dict(
                self.reduction_by_precision
            ),
            "selected_capacity_improvement": self.selected_capacity_improvement,
            "selected_capacity_throughput_improvement": (
                self.selected_capacity_throughput_improvement
            ),
            "selected_capacity_throughput_scope": (
                self.selected_capacity_throughput_scope
            ),
            "selected_tpot_improvement": self.selected_tpot_improvement,
            "maximum_regression": self.maximum_regression,
            "bottleneck_counts": dict(self.bottleneck_counts),
            "classical_roofline_bottleneck_counts": dict(
                self.classical_roofline_bottleneck_counts
            ),
            "architecture_issue_bottleneck_counts": dict(
                self.architecture_issue_bottleneck_counts
            ),
            "algorithmic_bottleneck_counts": dict(
                self.algorithmic_bottleneck_counts
            ),
            "report_id": self.report_id,
        }

    @property
    def report_id(self) -> str:
        content = {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "headline_outcome": self.headline_outcome,
            "checks": [check.to_dict() for check in self.checks],
            "evidence_hash": self.evidence_hash,
            "policy": self.policy.to_dict(),
        }
        return f"packedkv-report-{_canonical_hash(content)}"


def expected_packedkv_bytes(
    precision: PackedKVPrecision,
    topology: AttentionTopology,
    alignment_bytes: int,
    mode: str,
) -> tuple[int, int, int]:
    if topology.head_dim % precision.block_size:
        raise ValueError("KV head dimension must contain complete precision blocks")
    element_bytes = _aligned_bytes(
        topology.mlen * precision.element_bits,
        alignment_bytes,
    )
    scale_bytes = (
        _aligned_bytes(
            topology.mlen // precision.block_size * precision.scale_bits,
            alignment_bytes,
        )
        if precision.scale_bits
        else 0
    )
    row_bytes = element_bytes + scale_bytes
    dense_rows = math.ceil(topology.active_elements / topology.mlen)
    ideal_one_plane = math.ceil(
        topology.active_elements * precision.effective_bits / 8
    )
    if mode == PADDED_PER_HEAD:
        storage = topology.kv_heads * row_bytes
        read = storage
    elif mode == DENSE_COMPILER:
        storage = dense_rows * row_bytes
        read = topology.kv_heads * row_bytes
    elif mode == DENSE_SELECTOR:
        storage = dense_rows * row_bytes
        read = storage
    elif mode == IDEAL_TRAFFIC:
        storage = ideal_one_plane
        read = storage
    else:
        raise ValueError(f"unknown PackedKV mode {mode!r}")
    # K and V occupy separate planes with identical layout.
    return 2 * storage, 2 * read, 2 * storage


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def evaluate_packedkv_publication(
    evidence: PackedKVPublicationEvidence,
    policy: PackedKVGatePolicy = PackedKVGatePolicy(),
) -> PackedKVPublicationReport:
    """Validate the complete causal grid without inferring missing evidence."""

    expected_keys = {
        (precision, topology)
        for precision in PRECISION_ROLES
        for topology in TOPOLOGY_ROLES
    }
    groups = {group.key: group for group in evidence.groups}
    grid_complete = set(groups) == expected_keys and all(
        set(group.by_mode()) == set(PACKEDKV_MODES)
        for group in evidence.groups
    )

    precision_definitions: dict[str, set[tuple[Any, ...]]] = {
        role: set() for role in PRECISION_ROLES
    }
    topology_definitions: dict[str, set[tuple[Any, ...]]] = {
        role: set() for role in TOPOLOGY_ROLES
    }
    for group in evidence.groups:
        precision_definitions[group.precision.role].add(
            (
                group.precision.format_id,
                group.precision.element_bits,
                group.precision.scale_bits,
                group.precision.block_size,
            )
        )
        topology_definitions[group.topology.role].add(
            (
                group.topology.query_heads,
                group.topology.kv_heads,
                group.topology.head_dim,
                group.topology.mlen,
            )
        )
    definitions_consistent = (
        all(len(values) == 1 for values in precision_definitions.values())
        and all(len(values) == 1 for values in topology_definitions.values())
    )
    topology_values = [
        next(iter(values))
        for values in topology_definitions.values()
        if values
    ]
    topology_sweep_consistent = bool(topology_values) and len(
        {(value[0], value[2], value[3]) for value in topology_values}
    ) == 1

    byte_failures: list[str] = []
    numerical_failures: list[str] = []
    regressions: list[float] = []
    bottlenecks: dict[str, int] = {}
    classical_bottlenecks: dict[str, int] = {}
    architecture_bottlenecks: dict[str, int] = {}
    algorithmic_bottlenecks: dict[str, int] = {}
    head_service_tpots: set[float] = set()
    capacity_scope_failures: list[str] = []
    for group in evidence.groups:
        modes = group.by_mode()
        for mode, measurement in modes.items():
            expected = expected_packedkv_bytes(
                group.precision,
                group.topology,
                evidence.environment.alignment_bytes,
                mode,
            )
            observed = (
                measurement.storage_bytes_per_sequence_token,
                measurement.read_bytes_per_sequence_token,
                measurement.write_bytes_per_appended_token,
            )
            if observed != expected:
                byte_failures.append(
                    f"{group.precision.role}/{group.topology.role}/{mode}"
                )
            if (
                measurement.max_abs_error > policy.absolute_tolerance
                or measurement.max_rel_error > policy.relative_tolerance
            ):
                numerical_failures.append(
                    f"{group.precision.role}/{group.topology.role}/{mode}"
                )
            bottlenecks[measurement.bottleneck] = (
                bottlenecks.get(measurement.bottleneck, 0) + 1
            )
            classical_bottlenecks[
                measurement.classical_roofline_bottleneck
            ] = (
                classical_bottlenecks.get(
                    measurement.classical_roofline_bottleneck,
                    0,
                )
                + 1
            )
            architecture_bottlenecks[
                measurement.architecture_issue_bottleneck
            ] = (
                architecture_bottlenecks.get(
                    measurement.architecture_issue_bottleneck,
                    0,
                )
                + 1
            )
            algorithmic_bottlenecks[measurement.algorithmic_bottleneck] = (
                algorithmic_bottlenecks.get(
                    measurement.algorithmic_bottleneck,
                    0,
                )
                + 1
            )
            head_service_tpots.add(measurement.head_service_tpot_ms)
            expected_scope = (
                "cross_stack_measured"
                if measurement.feasible_batch
                in evidence.environment.executable_batches
                else "analytical_projection"
            )
            if measurement.capacity_throughput_scope != expected_scope:
                capacity_scope_failures.append(
                    f"{group.precision.role}/{group.topology.role}/{mode}"
                )
        if PADDED_PER_HEAD in modes and DENSE_SELECTOR in modes:
            regressions.append(
                modes[DENSE_SELECTOR].tpot_ms / modes[PADDED_PER_HEAD].tpot_ms - 1
            )

    reductions: dict[str, float] = {}
    for precision_role in PRECISION_ROLES:
        group = groups.get((precision_role, "gqa8"))
        if group is None:
            continue
        modes = group.by_mode()
        if PADDED_PER_HEAD not in modes or DENSE_SELECTOR not in modes:
            continue
        padded = modes[PADDED_PER_HEAD]
        selector = modes[DENSE_SELECTOR]
        reductions[precision_role] = min(
            padded.storage_bytes_per_sequence_token
            / selector.storage_bytes_per_sequence_token,
            padded.read_bytes_per_sequence_token
            / selector.read_bytes_per_sequence_token,
        )

    selected_capacity: float | None = None
    selected_capacity_tps: float | None = None
    selected_capacity_tps_scope: str | None = None
    capacity_group = groups.get(("selected", "gqa8"))
    if capacity_group is not None:
        modes = capacity_group.by_mode()
        if PADDED_PER_HEAD in modes and DENSE_SELECTOR in modes:
            selected_capacity = (
                modes[DENSE_SELECTOR].feasible_batch
                / modes[PADDED_PER_HEAD].feasible_batch
            )
            selected_capacity_tps = (
                modes[DENSE_SELECTOR].capacity_limited_tokens_per_s
                / modes[PADDED_PER_HEAD].capacity_limited_tokens_per_s
            )
            selected_capacity_tps_scope = (
                "cross_stack_measured"
                if (
                    modes[PADDED_PER_HEAD].capacity_throughput_scope
                    == "cross_stack_measured"
                    and modes[DENSE_SELECTOR].capacity_throughput_scope
                    == "cross_stack_measured"
                )
                else "analytical_projection"
            )

    selected_tpot_ratios: list[float] = []
    for topology_role in TOPOLOGY_ROLES:
        group = groups.get(("selected", topology_role))
        if group is None:
            continue
        modes = group.by_mode()
        if PADDED_PER_HEAD in modes and DENSE_SELECTOR in modes:
            selected_tpot_ratios.append(
                modes[DENSE_SELECTOR].tpot_ms / modes[PADDED_PER_HEAD].tpot_ms
            )
    tpot_ratio = _geometric_mean(selected_tpot_ratios)
    selected_tpot = None if tpot_ratio is None else 1 - tpot_ratio
    max_regression = max(regressions) if regressions else None

    provenance = dict(evidence.provenance)
    missing_provenance = tuple(
        key for key in REQUIRED_PROVENANCE if not provenance.get(key)
    )
    area_overhead = evidence.selector_synthesis.area_overhead
    fmax_loss = evidence.selector_synthesis.fmax_loss
    capacity_feasibility_passed = (
        evidence.environment.capacity_model_validated
        and selected_capacity is not None
        and selected_capacity >= policy.capacity_improvement
        and capacity_group is not None
        and capacity_group.by_mode()[PADDED_PER_HEAD].capacity_limiter == "hbm"
    )
    capacity_throughput_passed = (
        capacity_feasibility_passed
        and selected_capacity_tps_scope == "cross_stack_measured"
        and selected_capacity_tps is not None
        and selected_capacity_tps > 1.0
    )
    tpot_passed = (
        evidence.pipeline_validation.overlap_validated
        and selected_tpot is not None
        and selected_tpot >= policy.tpot_improvement
    )
    candidate_headline = (
        "capacity_throughput"
        if capacity_throughput_passed
        else "analytical_capacity"
        if capacity_feasibility_passed
        else "trace_validated_tpot"
        if tpot_passed
        else None
    )

    checks = (
        PackedKVGateCheck(
            "fixed_remote_head_boundary",
            len(head_service_tpots) == 1
            and evidence.environment.output_head_mode == HEAD_SERVICE_MODE
            and evidence.environment.tpot_scope == TPOT_SCOPE,
            {
                "service_mode": evidence.environment.output_head_mode,
                "scope": evidence.environment.tpot_scope,
                "fixed_batch_head_tpot_ms": sorted(head_service_tpots),
            },
            "all fixed-batch layout points use one calibrated remote BF16 head",
        ),
        PackedKVGateCheck(
            "capacity_throughput_scope",
            not capacity_scope_failures,
            {
                "executable_batches": list(
                    evidence.environment.executable_batches
                ),
                "mismatches": capacity_scope_failures,
            },
            (
                "throughput is measured only at compiler/emulator/RTL "
                "validated serving batches"
            ),
        ),
        PackedKVGateCheck(
            "complete_causal_grid",
            grid_complete,
            {
                "groups": len(groups),
                "measurements": sum(
                    len(group.measurements) for group in evidence.groups
                ),
                "missing_groups": [
                    f"{precision}/{topology}"
                    for precision, topology in sorted(expected_keys - set(groups))
                ],
            },
            "three precision roles × four topologies × four modes",
        ),
        PackedKVGateCheck(
            "fixed_precision_and_topology_definitions",
            definitions_consistent and topology_sweep_consistent,
            {
                "precision_consistent": definitions_consistent,
                "query_head_dimension_mlen_fixed": topology_sweep_consistent,
            },
            "each role is immutable and only the KV-head count changes",
        ),
        PackedKVGateCheck(
            "physical_byte_accounting",
            not byte_failures,
            {"mismatches": byte_failures},
            "element and scale planes match aligned physical bytes exactly",
        ),
        PackedKVGateCheck(
            "numerical_equivalence",
            not numerical_failures and grid_complete,
            {"mismatches": numerical_failures},
            (
                f"absolute error <= {policy.absolute_tolerance:g} and "
                f"relative error <= {policy.relative_tolerance:g}"
            ),
        ),
        PackedKVGateCheck(
            "capacity_model_validated",
            evidence.environment.capacity_model_validated,
            {
                "validated": evidence.environment.capacity_model_validated,
                "model_id": evidence.environment.capacity_model_id,
            },
            "feasible-batch evidence comes from a validated physical-capacity model",
        ),
        PackedKVGateCheck(
            "byte_reduction",
            set(reductions) == set(PRECISION_ROLES)
            and all(
                reduction >= policy.byte_reduction
                for reduction in reductions.values()
            ),
            reductions,
            (
                f"storage and read reduction >= {policy.byte_reduction:g}x "
                "for the declared GQA8 evidence topology"
            ),
        ),
        PackedKVGateCheck(
            "selector_synthesis_calibrated",
            evidence.selector_synthesis.calibrated,
            evidence.selector_synthesis.calibrated,
            "selector-off/on reports use the declared common synthesis setup",
        ),
        PackedKVGateCheck(
            "selector_area_overhead",
            area_overhead < policy.selector_area_overhead,
            area_overhead,
            f"area overhead < {policy.selector_area_overhead:.1%}",
        ),
        PackedKVGateCheck(
            "selector_fmax_loss",
            fmax_loss < policy.selector_fmax_loss,
            fmax_loss,
            f"Fmax loss < {policy.selector_fmax_loss:.1%}",
        ),
        PackedKVGateCheck(
            "no_workload_regression",
            max_regression is not None
            and max_regression <= policy.max_workload_regression,
            max_regression,
            f"selector TPOT regression <= {policy.max_workload_regression:.1%}",
        ),
        PackedKVGateCheck(
            "headline_serving_outcome",
            candidate_headline is not None,
            {
                "outcome": candidate_headline,
                "capacity_improvement": selected_capacity,
                "capacity_limited_tps_improvement": selected_capacity_tps,
                "capacity_throughput_scope": selected_capacity_tps_scope,
                "tpot_improvement": selected_tpot,
                "pipeline_overlap_validated": (
                    evidence.pipeline_validation.overlap_validated
                ),
            },
            (
                f"capacity >= {policy.capacity_improvement:g}x with higher "
                "cross-stack capacity throughput from an HBM-limited "
                "baseline, an explicitly analytical capacity-only result, "
                "or TPOT improves >= "
                f"{policy.tpot_improvement:.1%} with cross-stack overlap traces"
            ),
        ),
        PackedKVGateCheck(
            "complete_provenance",
            not missing_provenance
            and set(provenance) == set(REQUIRED_PROVENANCE),
            {
                "missing": list(missing_provenance),
                "unexpected": sorted(
                    set(provenance) - set(REQUIRED_PROVENANCE)
                ),
            },
            "all implementation, model, and synthesis inputs are content-addressed",
        ),
    )
    evidence_hash = _canonical_hash(evidence.to_dict())
    passed = all(check.passed for check in checks)
    return PackedKVPublicationReport(
        passed=passed,
        headline_outcome=candidate_headline if passed else None,
        checks=checks,
        evidence_hash=evidence_hash,
        policy=policy,
        reduction_by_precision=tuple(sorted(reductions.items())),
        selected_capacity_improvement=selected_capacity,
        selected_capacity_throughput_improvement=selected_capacity_tps,
        selected_capacity_throughput_scope=selected_capacity_tps_scope,
        selected_tpot_improvement=selected_tpot,
        maximum_regression=max_regression,
        bottleneck_counts=tuple(sorted(bottlenecks.items())),
        classical_roofline_bottleneck_counts=tuple(
            sorted(classical_bottlenecks.items())
        ),
        architecture_issue_bottleneck_counts=tuple(
            sorted(architecture_bottlenecks.items())
        ),
        algorithmic_bottleneck_counts=tuple(
            sorted(algorithmic_bottlenecks.items())
        ),
    )


def load_packedkv_evidence(path: Path | str) -> PackedKVPublicationEvidence:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    if not isinstance(value, Mapping):
        raise ValueError("PackedKV evidence must be a JSON object")
    return PackedKVPublicationEvidence.from_dict(value)


def write_packedkv_report(
    report: PackedKVPublicationReport,
    path: Path | str,
) -> None:
    """Write a report atomically so interrupted validation cannot look complete."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report.to_dict(),
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a complete PackedKV causal-ablation artifact."
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate_packedkv_publication(load_packedkv_evidence(args.evidence))
    write_packedkv_report(report, args.output)
    print(
        json.dumps(
            {
                "passed": report.passed,
                "headline_outcome": report.headline_outcome,
                "failures": report.failures,
                "report_id": report.report_id,
            },
            sort_keys=True,
        )
    )
    return 0 if report.passed else 2


__all__ = [
    "AblationEnvironment",
    "AttentionTopology",
    "DENSE_COMPILER",
    "DENSE_SELECTOR",
    "EVIDENCE_SCHEMA",
    "IDEAL_TRAFFIC",
    "PACKEDKV_MODES",
    "PADDED_PER_HEAD",
    "PackedKVAblationGroup",
    "PackedKVGateCheck",
    "PackedKVGatePolicy",
    "PackedKVModeMeasurement",
    "PackedKVPrecision",
    "PackedKVPublicationEvidence",
    "PackedKVPublicationReport",
    "PipelineValidationEvidence",
    "SelectorSynthesisEvidence",
    "evaluate_packedkv_publication",
    "expected_packedkv_bytes",
    "load_packedkv_evidence",
    "write_packedkv_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
