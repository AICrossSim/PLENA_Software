"""Production evaluator and CLI for the exact decode hardware study."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from decode_dse.legality import (
    PackedKVRuntimeTarget,
    evaluate_profile_legality,
    evaluate_stack_capability,
)
from decode_dse.hardware.admission_cost import (
    AdmissionCorrectnessStatus,
    admission_correctness_status_valid,
    load_admission_correctness_evidence,
    missing_admission_correctness_status,
)
from decode_dse.hardware.synthesis_anchor import ExactDCAnchorIndex
from decode_dse.hardware.design_space import (
    CalibratedEnergy,
    CapacityBreakdown,
    COMPILER_TRACE_EXECUTION_MODE,
    COMPILER_TRACE_TIMING_SET_SCHEMA,
    COMPILER_TRACE_TIMING_TIER,
    ExactHardwareSpace,
    ExactHardwareStudy,
    FULL_MODEL_DECODE_SCOPE,
    HardwareCandidate,
    HardwareEvaluation,
    HardwareMetrics,
    LEGACY_AGGREGATE_BANDWIDTH_MODE,
    PHYSICAL_TRAFFIC_KEYS,
    PUBLICATION_TIMING_TIERS,
    PhysicalTraffic,
    ResourceBudget,
    ResourceBudgetStatus,
    STAGE_CALIBRATED_ANALYTIC_TIMING_TIER,
    physical_cost_signature,
)
from decode_dse.hardware.lm_head_service import (
    BF16HeadServiceStatus,
    HEAD_SERVICE_MODE,
    HEAD_SERVICE_SCHEMA,
    composite_system_calibration_id,
    load_bf16_head_service_artifact,
)
from decode_dse.hardware.power_model import (
    calibrated_area_from_simulator,
    calibrated_energy_from_simulator,
    load_simulator_power_artifact,
    required_hardware_power_signatures,
)
from decode_dse.hardware.power_bridge import (
    analytic_energy_from_simulator,
    analytic_power_provenance,
    hbm_peak_bandwidth_bytes_per_s,
)
from decode_dse.hardware.workload_events import (
    EVENT_MODEL,
    DecodeEvent,
    DenseDecoderShape,
    count_decode_events,
)
from decode_dse.legality import StackValidity
from decode_dse.manifest import (
    SweepManifest,
    SweepManifestEntry,
    load_manifest,
)
from decode_dse.profiles import (
    MXINT_FORMATS,
    PROFILE_KIND_BF16_REFERENCE,
    DecodePrecisionProfile,
    format_descriptor,
)

EVALUATOR_VERSION = "plena-exact-hardware-evaluator"
TERMINAL_RESULT_SCHEMA = "decode-sweep-result"
KV_LAYOUT = "dense_selector"
TRAFFIC_UNIT = "bytes_per_generated_token"
STEP_COMPOSITION = "max_compute_memory"
PREFILL_HANDOFF_INPUT_SCHEMA = "plena-prefill-handoff-input-v1"
PREFILL_HANDOFF_ANALYSIS_SCHEMA = "plena-prefill-handoff-analysis-v1"
PREFILL_MEASUREMENT_SCOPE = "full_model_bf16_prompt_encoding_to_kv_ready"
HANDOFF_LINK_GENERATIONS = frozenset({"nvlink3", "nvlink4", "ualink", "pcie5"})
DECODE_BF16_HEAD = "decode_bf16_unmodeled"
EXTERNAL_BF16_HEAD = "external_bf16_service"
_AREA_KEYS = (
    "MATRIX_SRAM_DEPTH",
    "VECTOR_SRAM_DEPTH",
    "INT_SRAM_DEPTH",
    "FP_SRAM_DEPTH",
    "INT_DATA_WIDTH",
    "MX_SCALE_WIDTH",
    "BLOCK_DIM",
)


#: Modules whose logic can change a hardware evaluation. They are named as
#: import paths rather than file paths so a rename fails at import time instead
#: of silently hashing an absent or stale file.
_PROVENANCE_MODULES = (
    "decode_dse.legality",
    "decode_dse.manifest",
    "decode_dse.profiles",
    "decode_dse.hardware.admission_cost",
    "decode_dse.hardware.design_space",
    "decode_dse.hardware.evaluation",
    "decode_dse.hardware.lm_head_service",
    "decode_dse.hardware.power_model",
    "decode_dse.hardware.power_bridge",
    "decode_dse.hardware.synthesis_anchor",
    "decode_dse.hardware.workload_events",
)

#: Simulator sources whose logic can change a decode timing or traffic result,
#: keyed by provenance field name and given relative to the simulator root.
_SIMULATOR_SOURCES = {
    "perf_model_sha256": "analytic_models/performance/perf_model.py",
    "decode_timing_sha256": "analytic_models/performance/decode_timing.py",
    "compiler_trace_timing_sha256": (
        "analytic_models/performance/compiler_trace_timing.py"
    ),
    "decode_model_sha256": "analytic_models/performance/disagg_decode.py",
    "memory_model_sha256": "analytic_models/memory/memory_model.py",
    "llm_memory_model_sha256": "analytic_models/memory/llm_memory_model.py",
    "utilisation_model_sha256": (
        "analytic_models/utilisation/utilisation_model.py"
    ),
    "packed_kv_model_sha256": "analytic_models/disagg_serve/packed_kv.py",
    "area_bridge_sha256": "analytic_models/disagg_serve/area.py",
    "area_package_sha256": "analytic_models/area/__init__.py",
    "area_sram_model_sha256": "analytic_models/area/sram.py",
    "area_matrix_model_sha256": "analytic_models/area/matrix.py",
    "area_evidence_model_sha256": "analytic_models/area/evidence.py",
    "area_structural_coefficients_sha256": (
        "analytic_models/area/calibration/"
        "matrix_structural_coefficients.json"
    ),
    "area_sram_macro_table_sha256": (
        "analytic_models/area/calibration/asap7_sram_macro_table.csv"
    ),
    "bandwidth_model_sha256": "analytic_models/disagg_serve/memory.py",
    "hbm_technology_model_sha256": (
        "analytic_models/disagg_serve/hbm_technology.py"
    ),
    "physical_ledger_sha256": (
        "analytic_models/disagg_serve/physical_ledger.py"
    ),
    "decode_power_sha256": "analytic_models/disagg_serve/decode_power.py",
    "handoff_model_sha256": "analytic_models/disagg_serve/handoff.py",
}

#: Bandwidth calibration tables, hashed only when calibrated bandwidth is on.
_BANDWIDTH_CALIBRATION_SOURCES = (
    "analytic_models/disagg_serve/calibration_bw.csv",
    "analytic_models/disagg_serve/calibration_dma.csv",
    "analytic_models/disagg_serve/calibration_dma_requests.csv",
)


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


def _file_sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module_source_digests() -> dict[str, str]:
    """Return the SHA-256 of every module listed in `_PROVENANCE_MODULES`."""

    digests: dict[str, str] = {}
    for name in _PROVENANCE_MODULES:
        source = getattr(importlib.import_module(name), "__file__", None)
        if source is None:
            raise RuntimeError(f"{name} has no source file to hash")
        digests[name] = _file_sha256(source)
    return digests


def simulator_root() -> Path:
    """Return the PLENA_Simulator checkout the analytic models are read from."""

    default_root = Path(__file__).resolve().parents[3] / "PLENA_Simulator"
    return Path(os.environ.get("PLENA_SIMULATOR_PATH", default_root)).resolve()


def _simulator_source_digests() -> dict[str, str]:
    """Return the SHA-256 of every source listed in `_SIMULATOR_SOURCES`."""

    root = simulator_root()
    return {
        name: _file_sha256(root / relative)
        for name, relative in _SIMULATOR_SOURCES.items()
    }


def _simulator_token(token: str) -> str:
    return token if token.startswith("MXINT") else f"MXFP_{token}"


def _require_content_addressed_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or "-" not in value:
        raise ValueError(f"{name} must be a content-addressed identity")
    prefix, digest = value.rsplit("-", 1)
    if (
        not prefix
        or prefix[0] not in "abcdefghijklmnopqrstuvwxyz"
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in prefix
        )
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be a content-addressed identity")
    return value


def _exact_object(
    value: Any,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the schema")
    return value


@dataclass(frozen=True)
class HardwareWorkload:
    """One deterministic cached-decode timing and traffic workload."""

    input_seq: int
    output_seq: int
    stride: int
    runtime_hbm_reserve_bytes: int
    kv_layout: str = KV_LAYOUT

    def __post_init__(self) -> None:
        if self.input_seq <= 0 or self.output_seq <= 0 or self.stride <= 0:
            raise ValueError("decode workload lengths and stride must be positive")
        if self.runtime_hbm_reserve_bytes < 0:
            raise ValueError("runtime HBM reserve must be non-negative")
        if self.kv_layout != KV_LAYOUT:
            raise ValueError("the deployment study requires dense_selector PackedKV")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "scope": "steady_state_cached_q1",
            "query_length": 1,
            "admission_included": False,
            "input_seq": self.input_seq,
            "output_seq": self.output_seq,
            "stride": self.stride,
            "runtime_hbm_reserve_bytes": self.runtime_hbm_reserve_bytes,
            "kv_layout": self.kv_layout,
        }


@dataclass(frozen=True)
class PrefillMeasurement:
    """Evidence-bound BF16 prefill cost for one serving batch."""

    batch: int
    latency_s: float
    energy_j: float
    latency_evidence_id: str
    energy_evidence_id: str
    evidence_tier: str

    def __post_init__(self) -> None:
        if isinstance(self.batch, bool) or not isinstance(self.batch, int):
            raise TypeError("prefill batch must be an integer")
        if self.batch <= 0:
            raise ValueError("prefill batch must be positive")
        for name in ("latency_s", "energy_j"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"prefill {name} must be numeric")
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"prefill {name} must be finite and positive")
            object.__setattr__(self, name, value)
        _require_content_addressed_id(
            self.latency_evidence_id,
            "prefill latency evidence",
        )
        _require_content_addressed_id(
            self.energy_evidence_id,
            "prefill energy evidence",
        )
        if not isinstance(self.evidence_tier, str) or not self.evidence_tier:
            raise ValueError("prefill evidence tier must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "latency_s": self.latency_s,
            "energy_j": self.energy_j,
            "latency_evidence_id": self.latency_evidence_id,
            "energy_evidence_id": self.energy_evidence_id,
            "evidence_tier": self.evidence_tier,
        }


@dataclass(frozen=True)
class PrefillHandoffArtifact:
    """Explicit prefill, admission, and scheduling inputs for E1 analysis."""

    artifact_sha256: str
    model_name: str
    model_revision: str
    prompt_tokens: int
    generation_tokens: int
    prefill_precision: str
    prefill_scope: str
    measurements: tuple[PrefillMeasurement, ...]
    decode_ready_delay_s: float
    prefill_stall_power_w: float
    decode_idle_power_w: float
    direct_link_generation: str
    host_link_generation: str
    direct_link_energy_pj_per_bit: float
    host_link_energy_pj_per_bit: float
    link_evidence_id: str
    link_evidence_tier: str
    admission_bandwidth_bytes_per_s: float
    admission_quantize_energy_j_per_element: float
    admission_memory_energy_j_per_byte: float
    admission_calibrated: bool
    admission_calibration_id: str | None
    admission_evidence_tier: str
    schema_version: str = PREFILL_HANDOFF_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREFILL_HANDOFF_INPUT_SCHEMA:
            raise ValueError("unsupported prefill handoff input schema")
        if (
            not isinstance(self.artifact_sha256, str)
            or len(self.artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.artifact_sha256
            )
        ):
            raise ValueError("prefill handoff artifact SHA-256 is invalid")
        if (
            not isinstance(self.model_name, str)
            or not isinstance(self.model_revision, str)
            or not self.model_name
            or not self.model_revision
        ):
            raise ValueError("prefill handoff model identity must be explicit")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in (self.prompt_tokens, self.generation_tokens)
        ):
            raise ValueError("prefill handoff workload lengths must be positive")
        if self.prefill_precision != "BF16":
            raise ValueError("prefill handoff wire producer must use BF16")
        if self.prefill_scope != PREFILL_MEASUREMENT_SCOPE:
            raise ValueError("prefill handoff measurement scope is unsupported")
        if any(
            not isinstance(measurement, PrefillMeasurement)
            for measurement in self.measurements
        ):
            raise TypeError("prefill measurements have the wrong type")
        batches = tuple(measurement.batch for measurement in self.measurements)
        if not batches or batches != tuple(sorted(set(batches))):
            raise ValueError("prefill measurements must have unique sorted batches")
        for name in (
            "decode_ready_delay_s",
            "prefill_stall_power_w",
            "decode_idle_power_w",
            "direct_link_energy_pj_per_bit",
            "host_link_energy_pj_per_bit",
            "admission_quantize_energy_j_per_element",
            "admission_memory_energy_j_per_byte",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        raw_bandwidth = self.admission_bandwidth_bytes_per_s
        if isinstance(raw_bandwidth, bool) or not isinstance(
            raw_bandwidth,
            (int, float),
        ):
            raise TypeError("admission bandwidth must be numeric")
        bandwidth = float(raw_bandwidth)
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("admission bandwidth must be finite and positive")
        object.__setattr__(self, "admission_bandwidth_bytes_per_s", bandwidth)
        if (
            self.direct_link_generation not in HANDOFF_LINK_GENERATIONS
            or self.host_link_generation not in HANDOFF_LINK_GENERATIONS
        ):
            raise ValueError("handoff link generation is unsupported")
        _require_content_addressed_id(
            self.link_evidence_id,
            "handoff link evidence",
        )
        if (
            not isinstance(self.link_evidence_tier, str)
            or not self.link_evidence_tier
            or not isinstance(self.admission_evidence_tier, str)
            or not self.admission_evidence_tier
        ):
            raise ValueError("handoff evidence tiers must be explicit")
        if not isinstance(self.admission_calibrated, bool):
            raise TypeError("admission calibrated flag must be boolean")
        if self.admission_calibrated != bool(self.admission_calibration_id):
            raise ValueError("admission calibration identity is inconsistent")
        if self.admission_calibration_id is not None:
            _require_content_addressed_id(
                self.admission_calibration_id,
                "admission calibration",
            )
            if not self.admission_calibration_id.startswith("admission-"):
                raise ValueError("admission calibration identity has the wrong scope")

    @property
    def artifact_id(self) -> str:
        return "prefill-handoff-" + self.artifact_sha256

    @property
    def publication_rankable(self) -> bool:
        return self.admission_calibrated

    def measurement(self, batch: int) -> PrefillMeasurement:
        match = next(
            (
                measurement
                for measurement in self.measurements
                if measurement.batch == batch
            ),
            None,
        )
        if match is None:
            raise ValueError(f"prefill handoff artifact lacks batch {batch}")
        return match

    def to_status(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "workload": {
                "prompt_tokens": self.prompt_tokens,
                "generation_tokens": self.generation_tokens,
            },
            "prefill_precision": self.prefill_precision,
            "prefill_scope": self.prefill_scope,
            "measurements": [
                measurement.to_dict() for measurement in self.measurements
            ],
            "schedule": {
                "decode_ready_delay_s": self.decode_ready_delay_s,
                "prefill_stall_power_w": self.prefill_stall_power_w,
                "decode_idle_power_w": self.decode_idle_power_w,
            },
            "links": {
                "direct_generation": self.direct_link_generation,
                "host_generation": self.host_link_generation,
                "direct_energy_pj_per_bit": (
                    self.direct_link_energy_pj_per_bit
                ),
                "host_energy_pj_per_bit": self.host_link_energy_pj_per_bit,
                "evidence_id": self.link_evidence_id,
                "evidence_tier": self.link_evidence_tier,
            },
            "admission": {
                "bandwidth_bytes_per_s": (
                    self.admission_bandwidth_bytes_per_s
                ),
                "quantize_energy_j_per_element": (
                    self.admission_quantize_energy_j_per_element
                ),
                "memory_energy_j_per_byte": (
                    self.admission_memory_energy_j_per_byte
                ),
                "calibrated": self.admission_calibrated,
                "calibration_id": self.admission_calibration_id,
                "evidence_tier": self.admission_evidence_tier,
            },
            "publication_rankable": self.publication_rankable,
        }

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        model_name: str,
        model_revision: str,
        workload: HardwareWorkload,
        required_batches: Sequence[int],
    ) -> "PrefillHandoffArtifact":
        source = Path(path)
        payload_bytes = source.read_bytes()
        raw = json.loads(payload_bytes)
        root = _exact_object(
            raw,
            {
                "schema_version",
                "model",
                "workload",
                "prefill",
                "schedule",
                "links",
                "admission",
            },
            "prefill handoff artifact",
        )
        if root["schema_version"] != PREFILL_HANDOFF_INPUT_SCHEMA:
            raise ValueError("unsupported prefill handoff input schema")
        model = _exact_object(root["model"], {"name", "revision"}, "model")
        declared_workload = _exact_object(
            root["workload"],
            {"prompt_tokens", "generation_tokens"},
            "workload",
        )
        prefill = _exact_object(
            root["prefill"],
            {"precision", "scope", "measurements"},
            "prefill",
        )
        schedule = _exact_object(
            root["schedule"],
            {
                "decode_ready_delay_s",
                "prefill_stall_power_w",
                "decode_idle_power_w",
            },
            "schedule",
        )
        links = _exact_object(
            root["links"],
            {
                "direct_generation",
                "host_generation",
                "direct_energy_pj_per_bit",
                "host_energy_pj_per_bit",
                "evidence_id",
                "evidence_tier",
            },
            "links",
        )
        admission = _exact_object(
            root["admission"],
            {
                "bandwidth_bytes_per_s",
                "quantize_energy_j_per_element",
                "memory_energy_j_per_byte",
                "calibrated",
                "calibration_id",
                "evidence_tier",
            },
            "admission",
        )
        rows = prefill["measurements"]
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError("prefill measurements must be a sequence")
        measurements = []
        for row in rows:
            item = _exact_object(
                row,
                {
                    "batch",
                    "latency_s",
                    "energy_j",
                    "latency_evidence_id",
                    "energy_evidence_id",
                    "evidence_tier",
                },
                "prefill measurement",
            )
            measurements.append(
                PrefillMeasurement(
                    batch=item["batch"],
                    latency_s=item["latency_s"],
                    energy_j=item["energy_j"],
                    latency_evidence_id=item["latency_evidence_id"],
                    energy_evidence_id=item["energy_evidence_id"],
                    evidence_tier=item["evidence_tier"],
                )
            )
        artifact = cls(
            artifact_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            model_name=model["name"],
            model_revision=model["revision"],
            prompt_tokens=declared_workload["prompt_tokens"],
            generation_tokens=declared_workload["generation_tokens"],
            prefill_precision=prefill["precision"],
            prefill_scope=prefill["scope"],
            measurements=tuple(sorted(measurements, key=lambda row: row.batch)),
            decode_ready_delay_s=schedule["decode_ready_delay_s"],
            prefill_stall_power_w=schedule["prefill_stall_power_w"],
            decode_idle_power_w=schedule["decode_idle_power_w"],
            direct_link_generation=links["direct_generation"],
            host_link_generation=links["host_generation"],
            direct_link_energy_pj_per_bit=links["direct_energy_pj_per_bit"],
            host_link_energy_pj_per_bit=links["host_energy_pj_per_bit"],
            link_evidence_id=links["evidence_id"],
            link_evidence_tier=links["evidence_tier"],
            admission_bandwidth_bytes_per_s=admission["bandwidth_bytes_per_s"],
            admission_quantize_energy_j_per_element=(
                admission["quantize_energy_j_per_element"]
            ),
            admission_memory_energy_j_per_byte=(
                admission["memory_energy_j_per_byte"]
            ),
            admission_calibrated=admission["calibrated"],
            admission_calibration_id=admission["calibration_id"],
            admission_evidence_tier=admission["evidence_tier"],
        )
        if artifact.model_name != model_name or artifact.model_revision != model_revision:
            raise ValueError("prefill handoff and decode model identities differ")
        if (
            artifact.prompt_tokens != workload.input_seq
            or artifact.generation_tokens != workload.output_seq
        ):
            raise ValueError("prefill handoff and decode workloads differ")
        required = tuple(sorted(set(int(batch) for batch in required_batches)))
        available = tuple(row.batch for row in artifact.measurements)
        missing = tuple(batch for batch in required if batch not in available)
        if missing:
            raise ValueError(
                "prefill handoff artifact lacks required batches: "
                + ",".join(str(batch) for batch in missing)
            )
        return artifact


@dataclass(frozen=True)
class PrecisionRequest:
    """Exact role bindings consumed by DecodeSimulator.make_precision."""

    weight: int | str
    activation: int | str
    key: int | str
    value: int | str
    weight_family: str
    activation_family: str
    key_family: str
    value_family: str
    block_size: int


def precision_request(profile: DecodePrecisionProfile) -> PrecisionRequest:
    """Map one hardware profile without reducing formats to effective bits."""

    if profile.kind == PROFILE_KIND_BF16_REFERENCE:
        raise ValueError("the BF16 split reference is not a PLENA hardware profile")
    if profile.block_size != 8:
        raise ValueError("the PLENA decode datapath requires MX block size 8")
    required_bf16 = {"embedding", "lm_head"}
    if not required_bf16.issubset(profile.bf16_operators):
        raise ValueError("embeddings and the LM head must remain BF16")
    quantized_coverage = (
        profile.weight_operators
        + profile.activation_operators
        + profile.kv_operators
        + profile.vector_operators
    )
    if required_bf16.intersection(quantized_coverage):
        raise ValueError(
            "embedding and LM-head quantization is accuracy-only"
        )

    def role(token: str) -> tuple[int | str, str]:
        descriptor = format_descriptor(token)
        if descriptor.family == "mxint":
            if token not in MXINT_FORMATS:
                raise ValueError(f"unsupported integer format {token!r}")
            return descriptor.element_bits, "mxint"
        if descriptor.family == "mxfp":
            return token, "mxfp"
        raise ValueError(f"unsupported matrix format {token!r}")

    weight, weight_family = role(profile.weight_format)
    activation, activation_family = role(profile.activation_format)
    key, key_family = role(profile.key_format)
    value, value_family = role(profile.value_format)
    return PrecisionRequest(
        weight=weight,
        activation=activation,
        key=key,
        value=value,
        weight_family=weight_family,
        activation_family=activation_family,
        key_family=key_family,
        value_family=value_family,
        block_size=profile.block_size,
    )


def physical_traffic_from_metrics(
    metrics: Any,
    *,
    batch: int,
) -> PhysicalTraffic:
    """Convert the simulator's canonical physical ledger to per-token roles."""

    if batch <= 0:
        raise ValueError("batch must be positive")
    raw_per_token = tuple(metrics.hbm_traffic_per_generated_token)
    raw_per_step = tuple(metrics.hbm_traffic_per_batch_step)
    per_token = dict(raw_per_token)
    per_step = dict(raw_per_step)
    if (
        len(per_token) != len(raw_per_token)
        or len(per_step) != len(raw_per_step)
    ):
        raise ValueError("simulator physical traffic keys are not unique")
    if (
        set(per_token) != PHYSICAL_TRAFFIC_KEYS
        or set(per_step) != PHYSICAL_TRAFFIC_KEYS
    ):
        raise ValueError("simulator physical traffic schema mismatch")
    for name in PHYSICAL_TRAFFIC_KEYS:
        token_value = float(per_token[name])
        step_value = float(per_step[name])
        if (
            not math.isfinite(token_value)
            or not math.isfinite(step_value)
            or token_value < 0
            or step_value < 0
        ):
            raise ValueError(f"invalid physical traffic for {name}")
        expected = token_value * batch
        if abs(step_value - expected) > max(
            1e-6,
            abs(expected) * 1e-9,
        ):
            raise ValueError(f"physical traffic unit mismatch for {name}")
    traffic = PhysicalTraffic(
        weight_bytes=(
            per_token["weight_element_read_bytes"]
            + per_token["bf16_weight_read_bytes"]
        ),
        activation_bytes=(
            per_token["activation_read_bytes"]
            + per_token["activation_write_bytes"]
        ),
        kv_read_bytes=per_token["kv_element_read_bytes"],
        kv_write_bytes=per_token["kv_element_write_bytes"],
        scale_bytes=(
            per_token["weight_scale_read_bytes"]
            + per_token["kv_scale_read_bytes"]
            + per_token["kv_scale_write_bytes"]
        ),
    )
    expected_total = float(metrics.avg_hbm_bytes_per_generated_token)
    if not math.isfinite(expected_total) or expected_total < 0:
        raise ValueError("invalid per-token HBM traffic total")
    if abs(traffic.total_bytes - expected_total) > max(
        1e-6,
        abs(expected_total) * 1e-9,
    ):
        raise ValueError(
            "physical traffic differs from DecodeSimulator timing bytes"
        )
    expected_step = traffic.total_bytes * batch
    observed_step = float(metrics.avg_hbm_bytes_per_batch_step)
    if not math.isfinite(observed_step) or observed_step < 0:
        raise ValueError("invalid per-batch-step HBM traffic total")
    if abs(expected_step - observed_step) > max(
        1e-6,
        abs(expected_step) * 1e-9,
    ):
        raise ValueError("per-token and batch-step traffic differ")
    return traffic


def capacity_from_metrics(
    metrics: Any,
    *,
    batch: int,
) -> CapacityBreakdown:
    """Convert the aligned resident planes without substituting capacity."""

    if batch <= 0:
        raise ValueError("batch must be positive")
    integer_fields = (
        "weight_element_plane_bytes",
        "weight_scale_plane_bytes",
        "weight_bf16_bytes",
        "kv_element_plane_bytes",
        "kv_scale_plane_bytes",
        "runtime_hbm_reserve_bytes",
        "hbm_capacity",
        "max_batch",
        "max_resident_batch",
        "max_synchronous_batch",
        "max_runtime_batch",
        "vector_sram_capacity_bytes",
        "vector_sram_required_bytes",
        "matrix_sram_capacity_bytes",
        "matrix_sram_required_bytes",
    )
    for name in integer_fields:
        raw = getattr(metrics, name)
        value = int(raw)
        if isinstance(raw, bool) or value != raw or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if int(metrics.hbm_capacity) <= 0:
        raise ValueError("hbm_capacity must be positive")
    for name in (
        "fits_in_hbm",
        "fits_onchip_sram",
        "fits_runtime",
    ):
        if not isinstance(getattr(metrics, name), bool):
            raise TypeError(f"{name} must be a boolean")
    weight_bytes = (
        int(metrics.weight_element_plane_bytes)
        + int(metrics.weight_scale_plane_bytes)
        + int(metrics.weight_bf16_bytes)
    )
    kv_bytes = (
        int(metrics.kv_element_plane_bytes)
        + int(metrics.kv_scale_plane_bytes)
    )
    capacity = CapacityBreakdown(
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_bytes,
        runtime_bytes=int(metrics.runtime_hbm_reserve_bytes),
        available_bytes=int(metrics.hbm_capacity),
    )
    hbm_required = float(metrics.hbm_required)
    if not math.isfinite(hbm_required) or hbm_required < 0:
        raise ValueError("hbm_required must be finite and non-negative")
    if abs(hbm_required - capacity.required_bytes) > 0.5:
        raise ValueError("capacity accounting differs from DecodeSimulator")
    if bool(metrics.fits_in_hbm) != capacity.feasible:
        raise ValueError("HBM feasibility differs from resident capacity")
    fits_sram = (
        metrics.vector_sram_required_bytes
        <= metrics.vector_sram_capacity_bytes
        and metrics.matrix_sram_required_bytes
        <= metrics.matrix_sram_capacity_bytes
    )
    if bool(metrics.fits_onchip_sram) != fits_sram:
        raise ValueError("SRAM feasibility differs from physical capacity")
    expected_runtime = capacity.feasible and fits_sram
    if bool(metrics.fits_runtime) != expected_runtime:
        raise ValueError("runtime feasibility differs from physical capacity")
    if int(metrics.max_batch) != int(metrics.max_runtime_batch):
        raise ValueError("legacy and runtime batch ceilings differ")
    if int(metrics.max_runtime_batch) > int(metrics.max_resident_batch):
        raise ValueError("runtime batch ceiling exceeds resident capacity")
    if bool(metrics.fits_runtime) != (
        batch <= int(metrics.max_runtime_batch)
    ):
        raise ValueError("runtime batch ceiling differs from feasibility")
    return capacity


@dataclass(frozen=True)
class SimulatorObservation:
    """Identity-bound output from one exact simulator candidate evaluation."""

    profile_id: str
    candidate_id: str
    tpot_ms: float
    tps: float
    total_time_s: float
    analytical_area_mm2: float
    traffic: PhysicalTraffic
    capacity: CapacityBreakdown
    algorithmic_bottleneck: str
    realized_bottleneck: str
    frac_algorithmic_memory_bound: float
    frac_realized_memory_bound: float
    frac_serialization_bound: float
    generated_tokens_per_step: int
    decode_steps: int
    timing_mode: str
    timing_calibrated: bool
    timing_evidence_id: str | None
    timing_reason: str
    execution_mode: str
    compiler_trace_timing: Mapping[str, Any] | None
    kv_layout: str
    layout_id: str
    capacity_model: str
    runtime_feasible: bool
    max_batch: int
    max_resident_batch: int
    max_synchronous_batch: int
    max_runtime_batch: int
    fits_onchip_sram: bool
    vector_sram_capacity_bytes: int
    vector_sram_required_bytes: int
    matrix_sram_capacity_bytes: int
    matrix_sram_required_bytes: int
    hbm_traffic_per_batch_step: tuple[tuple[str, float], ...]
    hbm_traffic_per_generated_token: tuple[tuple[str, float], ...]
    traffic_ledger_id: str
    packedkv_selector_supported: bool
    packedkv_selector_capability_id: str
    packedkv_selector_issue_codes: tuple[str, ...]
    bandwidth_calibration_id: str | None
    total_hbm_bytes: float
    events: tuple[DecodeEvent, ...]
    output_head_location: str
    collective_time_s_per_step: float = 0.0
    collective_bytes_per_generated_token: float = 0.0
    link_generation: str = "nvlink4"
    system_area_mm2: float | None = None
    area_evidence_tier: str | None = None
    logic_area_mm2: float | None = None
    avg_ideal_compute_seconds: float | None = None
    avg_realized_compute_seconds: float | None = None
    avg_memory_seconds: float | None = None
    step_composition: str = STEP_COMPOSITION
    classical_roofline_bottleneck: str | None = None
    architecture_issue_bottleneck: str | None = None
    frac_classical_memory_bound: float | None = None
    frac_architecture_issue_memory_bound: float | None = None
    avg_peak_compute_seconds: float | None = None
    architecture_options: Mapping[str, Any] = field(default_factory=dict)
    capacity_throughput_chain: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id or not self.candidate_id:
            raise ValueError("simulator identities must be non-empty")
        for name in (
            "tpot_ms",
            "tps",
            "total_time_s",
            "analytical_area_mm2",
            "total_hbm_bytes",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("generated_tokens_per_step", "decode_steps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.algorithmic_bottleneck not in {"memory", "compute"}:
            raise ValueError("invalid algorithmic bottleneck")
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
                raise ValueError(f"invalid {name}")
        if (
            self.architecture_issue_bottleneck
            not in {None, "unavailable", self.algorithmic_bottleneck}
        ):
            raise ValueError(
                "architecture-issue and compatibility labels disagree"
            )
        if self.realized_bottleneck not in {
            "memory",
            "serialization",
            "compute",
        }:
            raise ValueError("invalid realized bottleneck")
        if not isinstance(self.timing_calibrated, bool):
            raise TypeError("timing_calibrated must be boolean")
        if self.timing_calibrated != bool(self.timing_evidence_id):
            raise ValueError("timing evidence identity is inconsistent")
        if self.execution_mode not in {
            COMPILER_TRACE_EXECUTION_MODE,
            LEGACY_AGGREGATE_BANDWIDTH_MODE,
        }:
            raise ValueError("unsupported decode execution mode")
        trace_timing = self.compiler_trace_timing
        if self.execution_mode == COMPILER_TRACE_EXECUTION_MODE:
            if not self.timing_calibrated or not isinstance(
                trace_timing,
                Mapping,
            ):
                raise ValueError(
                    "compiler mode requires calibrated request-set evidence"
                )
            trace_timing = json.loads(_canonical_bytes(dict(trace_timing)))
            if (
                trace_timing.get("schema_version")
                != COMPILER_TRACE_TIMING_SET_SCHEMA
                or trace_timing.get("execution_mode")
                != COMPILER_TRACE_EXECUTION_MODE
                or trace_timing.get("artifact_scope")
                != FULL_MODEL_DECODE_SCOPE
            ):
                raise ValueError("compiler timing provenance is inconsistent")
            expected_timing_id = (
                "compiler-trace-timing-" + _content_hash(trace_timing)
            )
            if self.timing_evidence_id != expected_timing_id:
                raise ValueError("compiler timing evidence identity differs")
            if self.bandwidth_calibration_id is not None:
                raise ValueError(
                    "aggregate-bandwidth evidence is inapplicable to compiler mode"
                )
        elif trace_timing is not None:
            raise ValueError("legacy timing cannot carry compiler-trace evidence")
        object.__setattr__(self, "compiler_trace_timing", trace_timing)
        if self.kv_layout != KV_LAYOUT or not self.layout_id:
            raise ValueError("PackedKV layout identity is invalid")
        if self.output_head_location not in {
            DECODE_BF16_HEAD,
            EXTERNAL_BF16_HEAD,
        }:
            raise ValueError("output-head location is invalid")
        for name in (
            "collective_time_s_per_step",
            "collective_bytes_per_generated_token",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not self.link_generation:
            raise ValueError("link_generation must be non-empty")
        if (self.system_area_mm2 is None) != (
            self.area_evidence_tier is None
        ):
            raise ValueError("system area and its evidence tier must be paired")
        if self.system_area_mm2 is not None:
            if (
                not math.isfinite(float(self.system_area_mm2))
                or float(self.system_area_mm2) <= 0
            ):
                raise ValueError("system_area_mm2 must be finite and positive")
            if not self.area_evidence_tier:
                raise ValueError("area_evidence_tier must be non-empty")
        if self.logic_area_mm2 is not None:
            if (
                not math.isfinite(float(self.logic_area_mm2))
                or float(self.logic_area_mm2) <= 0
            ):
                raise ValueError("logic_area_mm2 must be finite and positive")
        if self.step_composition != STEP_COMPOSITION:
            raise ValueError("decode step composition is unsupported")
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
            if int(chain.get("max_feasible_batch", -1)) != self.max_runtime_batch:
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
        has_unmodeled_head = any(
            event.signature == "UNMODELED:LM_HEAD_BF16"
            for event in self.events
        )
        if has_unmodeled_head != (
            self.output_head_location == DECODE_BF16_HEAD
        ):
            raise ValueError(
                "output-head location and event boundary disagree"
            )
        if not self.capacity_model:
            raise ValueError("capacity model identity must be non-empty")
        if not self.traffic_ledger_id:
            raise ValueError("traffic ledger identity must be non-empty")
        if not isinstance(self.runtime_feasible, bool):
            raise TypeError("runtime_feasible must be boolean")
        if not isinstance(self.fits_onchip_sram, bool):
            raise TypeError("fits_onchip_sram must be boolean")
        if not isinstance(self.packedkv_selector_supported, bool):
            raise TypeError(
                "packedkv_selector_supported must be boolean"
            )
        if not self.packedkv_selector_capability_id:
            raise ValueError(
                "selector capability identity must be non-empty"
            )
        if self.packedkv_selector_supported != (
            not self.packedkv_selector_issue_codes
        ):
            raise ValueError("selector capability evidence is inconsistent")
        canonical_issues = tuple(
            sorted(set(self.packedkv_selector_issue_codes))
        )
        if canonical_issues != self.packedkv_selector_issue_codes:
            raise ValueError(
                "selector capability issues must be unique and sorted"
            )
        if (
            self.bandwidth_calibration_id is not None
            and not self.bandwidth_calibration_id
        ):
            raise ValueError(
                "bandwidth calibration identity must be non-empty"
            )
        for name in (
            "max_batch",
            "max_resident_batch",
            "max_synchronous_batch",
            "max_runtime_batch",
            "vector_sram_capacity_bytes",
            "vector_sram_required_bytes",
            "matrix_sram_capacity_bytes",
            "matrix_sram_required_bytes",
        ):
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
        if self.max_batch != self.max_runtime_batch:
            raise ValueError("legacy and runtime batch ceilings disagree")
        if self.max_runtime_batch > self.max_resident_batch:
            raise ValueError(
                "runtime batch ceiling cannot exceed resident capacity"
            )
        if self.runtime_feasible != (
            self.generated_tokens_per_step <= self.max_runtime_batch
        ):
            raise ValueError("runtime batch ceiling disagrees with feasibility")
        physical_traffic = HardwareMetrics(
            tpot_ms=self.tpot_ms,
            tps=self.tps,
            area_mm2=self.analytical_area_mm2,
            traffic=self.traffic,
            capacity=self.capacity,
            algorithmic_bottleneck=self.algorithmic_bottleneck,
            realized_bottleneck=self.realized_bottleneck,
            frac_algorithmic_memory_bound=self.frac_algorithmic_memory_bound,
            frac_realized_memory_bound=self.frac_realized_memory_bound,
            frac_serialization_bound=self.frac_serialization_bound,
            classical_roofline_bottleneck=(
                self.classical_roofline_bottleneck
            ),
            architecture_issue_bottleneck=(
                self.architecture_issue_bottleneck
            ),
            frac_classical_memory_bound=(
                self.frac_classical_memory_bound
            ),
            frac_architecture_issue_memory_bound=(
                self.frac_architecture_issue_memory_bound
            ),
            generated_tokens_per_step=self.generated_tokens_per_step,
            capacity_model=self.capacity_model,
            runtime_feasible=self.runtime_feasible,
            max_batch=self.max_batch,
            max_resident_batch=self.max_resident_batch,
            max_synchronous_batch=self.max_synchronous_batch,
            max_runtime_batch=self.max_runtime_batch,
            fits_onchip_sram=self.fits_onchip_sram,
            vector_sram_capacity_bytes=self.vector_sram_capacity_bytes,
            vector_sram_required_bytes=self.vector_sram_required_bytes,
            matrix_sram_capacity_bytes=self.matrix_sram_capacity_bytes,
            matrix_sram_required_bytes=self.matrix_sram_required_bytes,
            hbm_traffic_per_batch_step=self.hbm_traffic_per_batch_step,
            hbm_traffic_per_generated_token=(
                self.hbm_traffic_per_generated_token
            ),
            traffic_ledger_id=self.traffic_ledger_id,
            avg_peak_compute_seconds=self.avg_peak_compute_seconds,
            avg_ideal_compute_seconds=self.avg_ideal_compute_seconds,
            avg_realized_compute_seconds=self.avg_realized_compute_seconds,
            avg_memory_seconds=self.avg_memory_seconds,
            step_composition=self.step_composition,
        )
        if not physical_traffic.traffic_evidence_complete:
            raise ValueError("physical traffic evidence is incomplete")
        expected_hbm_bytes = (
            self.traffic.total_bytes
            * self.generated_tokens_per_step
            * self.decode_steps
        )
        if abs(self.total_hbm_bytes - expected_hbm_bytes) > max(
            1e-6,
            abs(expected_hbm_bytes) * 1e-9,
        ):
            raise ValueError("workload HBM bytes do not conserve traffic")
        expected_time = self.tpot_ms / 1000.0 * self.decode_steps
        if abs(self.total_time_s - expected_time) > max(
            1e-12,
            expected_time * 1e-9,
        ):
            raise ValueError("simulator total time and TPOT are inconsistent")


@dataclass(frozen=True)
class SimulatorResourcePreflight:
    """Exact capacity and structural-area facts that precede timing pricing."""

    profile_id: str
    candidate_id: str
    analytical_area_mm2: float
    system_area_mm2: float
    capacity: CapacityBreakdown
    runtime_feasible: bool
    max_resident_batch: int
    max_runtime_batch: int
    vector_sram_capacity_bytes: int
    vector_sram_required_bytes: int
    matrix_sram_capacity_bytes: int
    matrix_sram_required_bytes: int
    link_area_mm2: float

    def __post_init__(self) -> None:
        if not self.profile_id or not self.candidate_id:
            raise ValueError("preflight identities must be non-empty")
        for name in (
            "analytical_area_mm2",
            "system_area_mm2",
            "link_area_mm2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "max_resident_batch",
            "max_runtime_batch",
            "vector_sram_capacity_bytes",
            "vector_sram_required_bytes",
            "matrix_sram_capacity_bytes",
            "matrix_sram_required_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.runtime_feasible, bool):
            raise TypeError("runtime_feasible must be boolean")


@dataclass(frozen=True)
class RankLocalAttentionGeometry:
    """Global model heads and the exact subset owned by one TP rank."""

    global_attention_heads: int
    global_kv_heads: int
    tensor_parallel_degree: int
    local_attention_heads: int
    local_kv_heads: int

    @classmethod
    def bind(
        cls,
        shape: DenseDecoderShape,
        candidate: HardwareCandidate,
    ) -> "RankLocalAttentionGeometry":
        tp = candidate.tp
        if shape.attention_heads % tp or shape.kv_heads % tp:
            raise ValueError(
                "tensor parallelism must divide attention and KV heads"
            )
        return cls(
            global_attention_heads=shape.attention_heads,
            global_kv_heads=shape.kv_heads,
            tensor_parallel_degree=tp,
            local_attention_heads=shape.attention_heads // tp,
            local_kv_heads=shape.kv_heads // tp,
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "partition_rule": "contiguous_head_ownership_by_tensor_parallel_rank",
            "global_attention_heads": self.global_attention_heads,
            "global_kv_heads": self.global_kv_heads,
            "tensor_parallel_degree": self.tensor_parallel_degree,
            "local_attention_heads": self.local_attention_heads,
            "local_kv_heads": self.local_kv_heads,
        }


def _selector_runtime_target(
    profile: DecodePrecisionProfile,
    candidate: HardwareCandidate,
    shape: DenseDecoderShape,
) -> PackedKVRuntimeTarget:
    partition = RankLocalAttentionGeometry.bind(shape, candidate)
    return PackedKVRuntimeTarget(
        mlen=candidate.mlen,
        blen=candidate.blen,
        hlen=candidate.hlen,
        batch=candidate.batch,
        kv_heads=partition.local_kv_heads,
        head_dim=shape.head_dim,
        block_size=profile.block_size,
        packed_kv=True,
        batched_attention=True,
    )


@dataclass(frozen=True)
class RefinementHardwareEntry:
    """Manifest-compatible binding for one successful refined precision."""

    ordinal: int
    source_schedule_ordinal: int
    profile: Any
    validity: StackValidity

    @property
    def profile_id(self) -> str:
        return str(self.profile.profile_id)

    @property
    def legality(self) -> Any:
        return evaluate_profile_legality(self.profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "source_schedule_ordinal": self.source_schedule_ordinal,
            "profile_id": self.profile_id,
            "profile": self.profile.to_dict(),
            "legality": self.legality.to_dict(),
            **self.validity.to_dict(),
        }


@dataclass(frozen=True)
class RefinementHardwareManifest:
    """Minimal immutable manifest adapter for exact refined-profile repricing."""

    model_name: str
    model_revision: str
    tokenizer_revision: str
    model_architecture: Mapping[str, Any]
    entries: tuple[RefinementHardwareEntry, ...]
    base_manifest_hash: str
    refinement_schedule_hash: str
    refinement_merge_content_hash: str

    @property
    def canonical_hash(self) -> str:
        return _content_hash(
            {
                "schema_version": "decode-refinement-hardware-manifest",
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "model_architecture": dict(self.model_architecture),
                "base_manifest_hash": self.base_manifest_hash,
                "refinement_schedule_hash": self.refinement_schedule_hash,
                "refinement_merge_content_hash": (
                    self.refinement_merge_content_hash
                ),
                "entries": [entry.to_dict() for entry in self.entries],
            }
        )


def _refinement_hardware_inputs(
    base_manifest: SweepManifest,
    schedule_path: str | os.PathLike[str],
    merge_receipt_path: str | os.PathLike[str],
    results_path: str | os.PathLike[str] | None,
) -> tuple[RefinementHardwareManifest, tuple[Mapping[str, Any], ...]]:
    from decode_dse.software.refinement_runner import (
        load_refinement_merged_results,
    )
    from decode_dse.software.refinement_schedule import (
        load_refinement_schedule,
    )

    schedule = load_refinement_schedule(schedule_path)
    merged = load_refinement_merged_results(
        schedule,
        merge_receipt_path,
        results_path=results_path,
    )
    selected = []
    rows = []
    for schedule_entry, row in zip(schedule.entries, merged.terminal_rows):
        legality = evaluate_profile_legality(schedule_entry.profile)
        if (
            row.get("state") != "succeeded"
            or not legality.hardware_candidate
            or any(
                getattr(schedule_entry.validity, name) is not True
                for name in (
                    "software_valid",
                    "compiler_valid",
                    "emulator_valid",
                    "rtl_valid",
                )
            )
        ):
            continue
        selected.append(
            RefinementHardwareEntry(
                ordinal=len(selected),
                source_schedule_ordinal=schedule_entry.ordinal,
                profile=schedule_entry.profile,
                validity=schedule_entry.validity,
            )
        )
        rows.append(row)
    if not selected:
        raise ValueError(
            "refined hardware repricing has no successful fully measured profiles"
        )
    adapter = RefinementHardwareManifest(
        model_name=base_manifest.model_name,
        model_revision=base_manifest.model_revision,
        tokenizer_revision=str(base_manifest.tokenizer_revision),
        model_architecture=base_manifest.model_architecture,
        entries=tuple(selected),
        base_manifest_hash=base_manifest.canonical_hash,
        refinement_schedule_hash=schedule.canonical_hash,
        refinement_merge_content_hash=str(merged.receipt["content_hash"]),
    )
    return adapter, tuple(rows)


class SimulatorBackend(Protocol):
    """Backend boundary used by dependency-light evaluator tests."""

    @property
    def provenance(self) -> Mapping[str, Any]:
        ...

    def evaluate(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        workload: HardwareWorkload,
    ) -> SimulatorObservation:
        ...

    def evaluate_handoff(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        workload: HardwareWorkload,
        artifact: PrefillHandoffArtifact,
        *,
        decode_tpot_s: float,
        decode_energy_per_token_j: float,
        decode_energy_tier: str,
        decode_timing_evidence_id: str,
        system_calibration_id: str,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class PowerOutcome:
    area_mm2: float
    energy: CalibratedEnergy
    calibration_id: str


class PowerEngine(Protocol):
    @property
    def provenance(self) -> Mapping[str, Any]:
        ...

    def evaluate(
        self,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
        observation: SimulatorObservation,
    ) -> PowerOutcome:
        ...

    def hbm_energy_per_token(
        self,
        observation: SimulatorObservation,
    ) -> tuple[float, str]:
        ...

    def anchor_prediction(
        self,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
        observation: SimulatorObservation,
    ) -> Mapping[str, Any]:
        ...


def _whole_model_energy(
    decoder: CalibratedEnergy,
    head,
    *,
    decoder_tpot_ms: float,
    batch: int,
) -> tuple[CalibratedEnergy, str]:
    """Compose two dedicated chips over the serialized decode dependency."""

    decoder_duration = decoder_tpot_ms / 1000.0 / batch
    if abs(decoder.duration_s - decoder_duration) > max(
        1e-12,
        decoder_duration * 1e-6,
    ):
        raise ValueError("decoder energy duration differs from decoder TPOT")
    whole_duration = (
        decoder_tpot_ms / 1000.0 + head.total_latency_s
    ) / batch
    decoder_leakage_power = decoder.leakage_j / decoder.duration_s
    system_id = composite_system_calibration_id(
        decoder.calibration_id,
        head.calibration_id,
        head.provenance_id,
        service_mode=head.service_mode,
    )
    combined = CalibratedEnergy(
        calibration_id=system_id,
        energy_id=system_id,
        energy_tier=decoder.energy_tier,
        compute_j=(
            decoder.compute_j
            + head.mac_dynamic_energy_j / batch
        ),
        vector_j=(
            decoder.vector_j
            + head.selection_dynamic_energy_j / batch
        ),
        sram_j=decoder.sram_j,
        hbm_j=(
            decoder.hbm_j
            + head.memory_dynamic_energy_j / batch
        ),
        leakage_j=(
            decoder_leakage_power + head.leakage_power_w
        ) * whole_duration,
        unattributed_dynamic_j=(
            decoder.unattributed_dynamic_j
            + head.fixed_dynamic_energy_j / batch
        ),
        link_j=(
            decoder.link_j
            + head.link_dynamic_energy_j / batch
        ),
        duration_s=whole_duration,
        token_latency_s=(
            decoder_tpot_ms / 1000.0 + head.total_latency_s
        ),
    )
    return combined, system_id


def _terminal_files(paths: Sequence[str | os.PathLike[str]]) -> tuple[Path, ...]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            shard_root = path / "shards" if (path / "shards").is_dir() else path
            files.extend(sorted(shard_root.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    ordered = tuple(sorted({path.resolve() for path in files}, key=str))
    if not ordered:
        raise ValueError("no numerical JSONL files were found")
    return ordered


def load_terminal_numerical_rows(
    paths: Sequence[str | os.PathLike[str]],
    manifest: SweepManifest,
    *,
    require_complete: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    """Verify checksums and return the last terminal attempt per profile."""

    entries = {entry.profile_id: entry for entry in manifest.entries}
    attempts: dict[tuple[str, int], Mapping[str, Any]] = {}
    for path in _terminal_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise TypeError(f"numerical row is not an object at {path}:{line_number}")
                row = dict(raw)
                record_hash = row.pop("record_hash", None)
                if record_hash != _content_hash(row):
                    raise ValueError(
                        f"numerical checksum mismatch at {path}:{line_number}"
                    )
                if row.get("schema_version") != TERMINAL_RESULT_SCHEMA:
                    raise ValueError(
                        f"numerical schema mismatch at {path}:{line_number}"
                    )
                if row.get("manifest_hash") != manifest.canonical_hash:
                    raise ValueError(
                        f"numerical manifest mismatch at {path}:{line_number}"
                    )
                profile_id = str(row.get("profile_id", ""))
                entry = entries.get(profile_id)
                if entry is None:
                    raise ValueError(
                        f"unknown numerical profile at {path}:{line_number}"
                    )
                if int(row.get("ordinal", -1)) != entry.ordinal:
                    raise ValueError(
                        f"numerical ordinal mismatch at {path}:{line_number}"
                    )
                if row.get("profile") != entry.profile.to_dict():
                    raise ValueError(
                        f"numerical profile mismatch at {path}:{line_number}"
                    )
                if row.get("weight_format") != entry.profile.weight_format:
                    raise ValueError(
                        f"numerical weight binding mismatch at {path}:{line_number}"
                    )
                if row.get("state") not in {"succeeded", "failed"}:
                    raise ValueError(
                        f"numerical row is not terminal at {path}:{line_number}"
                    )
                attempt = int(row.get("attempt", 0))
                if attempt <= 0:
                    raise ValueError(
                        f"numerical attempt is invalid at {path}:{line_number}"
                    )
                StackValidity.from_dict(row.get("validity"))
                runtime_seconds = row.get("runtime_seconds")
                if (
                    isinstance(runtime_seconds, bool)
                    or not isinstance(runtime_seconds, (int, float))
                    or not math.isfinite(runtime_seconds)
                    or runtime_seconds < 0
                ):
                    raise ValueError(
                        f"numerical runtime is invalid at {path}:{line_number}"
                    )
                if row["state"] == "succeeded":
                    result = row.get("result")
                    if not isinstance(result, Mapping):
                        raise ValueError(
                            f"successful numerical result is missing at "
                            f"{path}:{line_number}"
                        )
                    mean_nll = result.get("mean_nll")
                    mean_token_nll = result.get(
                        "mean_token_nll",
                        mean_nll,
                    )
                    token_count = result.get("token_count")
                    if (
                        isinstance(mean_nll, bool)
                        or not isinstance(mean_nll, (int, float))
                        or not math.isfinite(mean_nll)
                        or mean_nll < 0
                        or isinstance(mean_token_nll, bool)
                        or not isinstance(mean_token_nll, (int, float))
                        or not math.isfinite(mean_token_nll)
                        or not math.isclose(
                            mean_nll,
                            mean_token_nll,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        or isinstance(token_count, bool)
                        or not isinstance(token_count, int)
                        or token_count <= 0
                    ):
                        raise ValueError(
                            f"successful numerical metrics are invalid at "
                            f"{path}:{line_number}"
                        )
                key = profile_id, attempt
                if key in attempts:
                    raise ValueError(
                        f"duplicate numerical attempt at {path}:{line_number}"
                    )
                attempts[key] = {**row, "record_hash": record_hash}
    selected = {
        profile_id: max(
            (
                row
                for (candidate_profile, _), row in attempts.items()
                if candidate_profile == profile_id
            ),
            key=lambda row: int(row["attempt"]),
            default=None,
        )
        for profile_id in entries
    }
    if require_complete:
        missing = [
            profile_id for profile_id, row in selected.items() if row is None
        ]
        if missing:
            raise ValueError(
                f"missing {len(missing)} terminal numerical profiles"
            )
    return tuple(
        selected[entry.profile_id]
        for entry in manifest.entries
        if selected[entry.profile_id] is not None
    )

class DecodeSimulatorBackend:
    """Real adapter from canonical profiles to DecodeSimulator."""

    def __init__(
        self,
        *,
        model: str,
        model_lib: str | os.PathLike[str] | None,
        settings_toml: str | os.PathLike[str] | None,
        isa_path: str | os.PathLike[str] | None,
        timing_evidence: str | os.PathLike[str],
        calibrated_bandwidth: bool,
        execution_mode: str,
        compiler_trace_artifacts: str | os.PathLike[str] | None = None,
        request_memory_calibration: str | os.PathLike[str] | None = None,
        head_service_artifact: str | os.PathLike[str] | None = None,
        model_name: str | None = None,
        model_revision: str | None = None,
        required_batches: Sequence[int] = (),
    ) -> None:
        from decode_dse.simulator_bridge import DecodeSimulator

        _validate_execution_launch(
            execution_mode=execution_mode,
            compiler_trace_artifacts=compiler_trace_artifacts,
            publication_timing_tier=(
                COMPILER_TRACE_TIMING_TIER
                if execution_mode == COMPILER_TRACE_EXECUTION_MODE
                else STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
            ),
        )
        if (
            execution_mode == COMPILER_TRACE_EXECUTION_MODE
            and calibrated_bandwidth
        ):
            raise ValueError(
                "compiler-trace execution cannot enable aggregate-bandwidth timing"
            )
        self.sim = DecodeSimulator(
            model,
            model_lib=model_lib,
            settings_toml=settings_toml,
            isa_path=isa_path,
            timing_mode="rtl_serialized",
            timing_evidence=timing_evidence,
        )
        self.requested_execution_mode = execution_mode
        self.calibrated_bandwidth = bool(calibrated_bandwidth)
        self.compiler_trace_runtime: Any | None = None
        self.compiler_trace_artifact_path: Path | None = None
        self.request_memory_calibration_path: Path | None = None
        self.head_service_status: BF16HeadServiceStatus | None = None
        if head_service_artifact is not None:
            if not model_name or not model_revision or not required_batches:
                raise ValueError(
                    "head-service validation requires model identity and batches"
                )
            self.head_service_status = load_bf16_head_service_artifact(
                head_service_artifact,
                model_name=model_name,
                model_revision=model_revision,
                hidden_size=int(self.sim.dims["hidden"]),
                vocab_size=int(self.sim.dims["vocab"]),
                tie_embeddings=bool(
                    self.sim.dims.get("tie_embeddings", False)
                ),
                required_batches=required_batches,
            )
        self.output_head_location = (
            EXTERNAL_BF16_HEAD
            if (
                self.head_service_status is not None
                and self.head_service_status.passed
            )
            else DECODE_BF16_HEAD
        )
        if execution_mode == COMPILER_TRACE_EXECUTION_MODE:
            from compiler_trace_timing import (
                DEFAULT_REQUEST_CALIBRATION,
                create_full_model_decode_artifact_runtime,
            )

            artifact_path = Path(compiler_trace_artifacts).resolve()
            calibration_path = Path(
                request_memory_calibration or DEFAULT_REQUEST_CALIBRATION
            ).resolve()
            runtime = create_full_model_decode_artifact_runtime(
                artifact_path,
                latency_library_path=self.sim.isa_path,
                request_calibration_path=calibration_path,
            )
            self.compiler_trace_runtime = runtime
            self.compiler_trace_artifact_path = artifact_path
            self.request_memory_calibration_path = calibration_path
            self.sim.use_compiler_trace_timing(
                runtime.provider,
                runtime.binder,
            )
        elif self.calibrated_bandwidth:
            self.sim.use_calibrated_bandwidth()
        root = simulator_root()
        paths = {
            "model_json_sha256": _file_sha256(self.sim.model_json),
            "settings_sha256": _file_sha256(self.sim.settings_toml),
            "isa_sha256": _file_sha256(self.sim.isa_path),
            "timing_evidence_sha256": _file_sha256(timing_evidence),
            "simulator_bridge_sha256": _file_sha256(
                importlib.import_module(
                    "decode_dse.simulator_bridge"
                ).__file__
            ),
            "head_service_model_sha256": _file_sha256(
                importlib.import_module(
                    "decode_dse.hardware.lm_head_service"
                ).__file__
            ),
            **_simulator_source_digests(),
        }
        if self.calibrated_bandwidth:
            for relative in _BANDWIDTH_CALIBRATION_SOURCES:
                name = Path(relative).name
                paths[f"{name}_sha256"] = _file_sha256(root / relative)
        if self.compiler_trace_runtime is not None:
            paths["compiler_trace_artifacts_sha256"] = _file_sha256(
                self.compiler_trace_artifact_path
            )
            paths["request_memory_calibration_sha256"] = _file_sha256(
                self.request_memory_calibration_path
            )
        self._provenance = {
            "backend": "DecodeSimulator",
            "timing_mode": "rtl_serialized",
            "timing_evidence_tier": (
                self.sim.timing_evidence.evidence_tier
                if self.sim.timing_evidence is not None
                else None
            ),
            "calibrated_bandwidth": self.calibrated_bandwidth,
            "requested_execution_mode": self.requested_execution_mode,
            "compiler_trace_artifact_set_id": (
                self.compiler_trace_runtime.artifact_set.artifact_set_id
                if self.compiler_trace_runtime is not None
                else None
            ),
            "compiler_trace_artifact_record_count": (
                len(self.compiler_trace_runtime.artifact_set.records)
                if self.compiler_trace_runtime is not None
                else 0
            ),
            "output_head_location": self.output_head_location,
            "head_service_status": (
                self.head_service_status.to_dict()
                if self.head_service_status is not None
                else {
                    "schema_version": HEAD_SERVICE_SCHEMA,
                    "artifact_sha256": None,
                    "passed": False,
                    "failures": ["missing_head_service_artifact"],
                    "calibration_id": None,
                    "provenance_id": None,
                    "service_mode": "unmodeled",
                    "service_location": None,
                    "required_batches": [],
                }
            ),
            **paths,
        }
        if self.calibrated_bandwidth and not getattr(
            self.sim._bw_model,
            "calibration_id",
            None,
        ):
            raise ValueError("calibrated bandwidth has no artifact identity")
        self.capacity_model_id = "decode-capacity-" + _content_hash(
            {
                "version": "resident-weights-packedkv-workspace",
                "model_json_sha256": paths["model_json_sha256"],
                "settings_sha256": paths["settings_sha256"],
                "physical_ledger_sha256": paths["physical_ledger_sha256"],
                "simulator_bridge_sha256": paths[
                    "simulator_bridge_sha256"
                ],
                "kv_layout": KV_LAYOUT,
                "output_head_location": self.output_head_location,
            }
        )
        self.traffic_ledger_id = "decode-traffic-" + _content_hash(
            {
                "version": "physical-decode-step-traffic",
                "model_json_sha256": paths["model_json_sha256"],
                "settings_sha256": paths["settings_sha256"],
                "physical_ledger_sha256": paths["physical_ledger_sha256"],
                "decode_model_sha256": paths["decode_model_sha256"],
                "memory_model_sha256": paths["memory_model_sha256"],
                "llm_memory_model_sha256": paths[
                    "llm_memory_model_sha256"
                ],
                "packed_kv_model_sha256": paths[
                    "packed_kv_model_sha256"
                ],
                "simulator_bridge_sha256": paths[
                    "simulator_bridge_sha256"
                ],
                "kv_layout": KV_LAYOUT,
                "output_head_location": self.output_head_location,
            }
        )
        self._provenance["capacity_model_id"] = self.capacity_model_id
        self._provenance["traffic_ledger_id"] = self.traffic_ledger_id
        self._resource_ledger_cache: dict[tuple[Any, ...], Any] = {}
        self._resource_area_cache: dict[tuple[Any, ...], Mapping[str, Any]] = {}

    def use_compiler_trace_timing(
        self,
        provider: Any,
        request_binder: Any,
    ) -> None:
        """Enable an injected full-model compiler-trace timing runtime."""

        self.sim.use_compiler_trace_timing(provider, request_binder)

    @property
    def provenance(self) -> Mapping[str, Any]:
        execution_mode = str(self.sim.execution_mode)
        binder = self.sim.trace_request_binder
        return {
            **self._provenance,
            "execution_mode": execution_mode,
            "compiler_trace_artifact_scope": getattr(
                self.sim.trace_timing_provider,
                "artifact_scope",
                None,
            ),
            "compiler_trace_request_binder": (
                f"{type(binder).__module__}.{type(binder).__qualname__}"
                if binder is not None
                else None
            ),
            "aggregate_bandwidth_timing_used": (
                execution_mode == LEGACY_AGGREGATE_BANDWIDTH_MODE
                and self.calibrated_bandwidth
            ),
        }

    def evaluate_handoff(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        workload: HardwareWorkload,
        artifact: PrefillHandoffArtifact,
        *,
        decode_tpot_s: float,
        decode_energy_per_token_j: float,
        decode_energy_tier: str,
        decode_timing_evidence_id: str,
        system_calibration_id: str,
    ) -> Mapping[str, Any]:
        """Evaluate E1 schedules from explicit prefill and decode evidence."""

        root = simulator_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from analytic_models.disagg_serve.handoff import (
            AdmissionModel,
            evaluate_handoff_regimes,
            handoff_time,
        )

        request = precision_request(entry.profile)
        precision = self.sim.make_precision(
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
        measurement = artifact.measurement(candidate.batch)
        admission = AdmissionModel(
            bandwidth_bytes_per_s=artifact.admission_bandwidth_bytes_per_s,
            quantize_energy_j_per_element=(
                artifact.admission_quantize_energy_j_per_element
            ),
            memory_energy_j_per_byte=(
                artifact.admission_memory_energy_j_per_byte
            ),
            calibrated=artifact.admission_calibrated,
            calibration_id=artifact.admission_calibration_id,
        )
        handoff = handoff_time(
            self.sim.dims,
            precision.spec,
            workload.input_seq,
            candidate.batch,
            link_gen=artifact.direct_link_generation,
            admission=admission,
        )
        energy_tier = (
            f"prefill:{measurement.evidence_tier};"
            f"decode:{decode_energy_tier};"
            f"admission:{artifact.admission_evidence_tier};"
            f"links:{artifact.link_evidence_tier}"
        )
        regimes = evaluate_handoff_regimes(
            handoff,
            prompt_tokens=workload.input_seq,
            generation_tokens=workload.output_seq,
            precision=(
                f"W={entry.profile.weight_format};"
                f"A={entry.profile.activation_format};"
                f"K={entry.profile.key_format};"
                f"V={entry.profile.value_format}"
            ),
            prefill_latency_s=measurement.latency_s,
            decode_tpot_s=decode_tpot_s,
            decode_ready_delay_s=artifact.decode_ready_delay_s,
            prefill_energy_j=measurement.energy_j,
            decode_energy_per_token_j=decode_energy_per_token_j,
            prefill_stall_power_w=artifact.prefill_stall_power_w,
            decode_idle_power_w=artifact.decode_idle_power_w,
            direct_link_generation=artifact.direct_link_generation,
            host_link_generation=artifact.host_link_generation,
            direct_link_energy_pj_per_bit=(
                artifact.direct_link_energy_pj_per_bit
            ),
            host_link_energy_pj_per_bit=artifact.host_link_energy_pj_per_bit,
            energy_tier=energy_tier,
        )
        return {
            "schema_version": PREFILL_HANDOFF_ANALYSIS_SCHEMA,
            "scope": "request_level_prefill_decode_schedule",
            "input_artifact_id": artifact.artifact_id,
            "publication_rankable": artifact.publication_rankable,
            "unrankable_reason": (
                None
                if artifact.publication_rankable
                else "admission_cost_uncalibrated"
            ),
            "ordinary_decode_ranking_effect": "none",
            "physical_precision_signature": physical_cost_signature(
                entry.profile,
                exact_vector_format=True,
            ),
            "candidate_id": candidate.candidate_id,
            "batch": candidate.batch,
            "prefill": {
                "scope": artifact.prefill_scope,
                **measurement.to_dict(),
            },
            "decode": {
                "tpot_s": decode_tpot_s,
                "energy_per_token_j": decode_energy_per_token_j,
                "energy_tier": decode_energy_tier,
                "timing_evidence_id": decode_timing_evidence_id,
                "system_calibration_id": system_calibration_id,
            },
            "handoff": {
                "wire_bytes": handoff.wire_bytes,
                "decode_cache_bytes": handoff.decode_cache_bytes,
                "transfer_bulk_s": handoff.transfer_bulk_s,
                "transfer_streamed_s": handoff.transfer_streamed_s,
                "admission_s": handoff.admission_s,
                "admission_energy_j": handoff.admission_energy_j,
                "admission_calibrated": handoff.admission_calibrated,
                "admission_calibration_id": (
                    handoff.admission_calibration_id
                ),
                "direct_link_generation": artifact.direct_link_generation,
                "host_link_generation": artifact.host_link_generation,
                "link_evidence_id": artifact.link_evidence_id,
                "link_evidence_tier": artifact.link_evidence_tier,
            },
            "regimes": [regime.to_dict() for regime in regimes],
        }

    def resource_preflight(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        workload: HardwareWorkload,
    ) -> SimulatorResourcePreflight:
        """Price exact storage and structural area without running decode time."""

        request = precision_request(entry.profile)
        precision = self.sim.make_precision(
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
        hbm = self.sim.hbm_overrides(
            candidate.hbm_generation,
            candidate.hbm_channels,
        )
        override = {
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "VLEN": candidate.vlen,
            "HLEN": candidate.hlen,
            "TP": candidate.tp,
            "KVP": candidate.kvp,
            "LINK_PORTS": candidate.link_ports,
            "SRAM_POLICY": candidate.sram_policy,
            "LINK_GENERATION": "nvlink4",
            **hbm,
        }
        if candidate.architecture_knobs_explicit:
            override.update(
                {
                    "KV_HEAD_REUSE": candidate.kv_head_reuse,
                    "DRAIN_OVERLAPPED": candidate.drain_overlapped,
                }
            )
        hardware = self.sim.base_hw.model_copy(update=override)
        memory = self.sim.base_mem.model_copy(
            update={
                "weight_bits": precision.spec["ffn_bits"],
                "activation_bits": 16,
                "kv_cache_bits": precision.spec["kv_bits"],
                **override,
            }
        )
        hbm_per_chip = int(memory.HBM_SIZE)
        precision_id = _content_hash(precision.spec)
        ledger_key = (
            precision_id,
            candidate.mlen,
            candidate.blen,
            candidate.hlen,
            candidate.batch,
            candidate.hbm_channels,
            candidate.chip_count,
            candidate.tp,
            candidate.kvp,
            candidate.sram_policy,
        )
        ledger = self._resource_ledger_cache.get(ledger_key)
        if ledger is None:
            ledger = self.sim._dd.build_physical_decode_ledger(
                self.sim.dims,
                precision.spec,
                hardware,
                context=workload.input_seq + workload.output_seq,
                batch=candidate.batch,
                hbm_capacity_bytes=hbm_per_chip,
                runtime_hbm_reserve_bytes=workload.runtime_hbm_reserve_bytes,
                kv_layout=workload.kv_layout,
                include_lm_head=(
                    self.output_head_location == DECODE_BF16_HEAD
                ),
            )
            ledger = self.sim._dd._partition_physical_ledger(
                ledger,
                tp=int(candidate.tp),
                kvp=int(candidate.kvp),
                hbm_per_chip=hbm_per_chip,
                sram_policy=candidate.sram_policy,
                batch=candidate.batch,
            )
            if len(self._resource_ledger_cache) >= 4096:
                self._resource_ledger_cache.clear()
            self._resource_ledger_cache[ledger_key] = ledger
        capacity = CapacityBreakdown(
            weight_bytes=int(ledger.weights.resident.total_aligned),
            kv_cache_bytes=int(ledger.kv.total_bytes),
            runtime_bytes=int(ledger.runtime_hbm_reserve_bytes),
            available_bytes=int(ledger.hbm_capacity_bytes),
        )
        sram = ledger.sram
        runtime_feasible = (
            capacity.feasible
            and sram.vector_required_bytes <= sram.vector_capacity_bytes
            and sram.matrix_required_bytes <= sram.matrix_capacity_bytes
            and candidate.batch <= int(ledger.max_runtime_batch)
        )
        shape = DenseDecoderShape.from_mapping(self.sim.dims)
        attention_partition = RankLocalAttentionGeometry.bind(shape, candidate)
        if candidate.kv_head_reuse:
            reuse = self.sim._dd.kv_head_reuse_status(
                enabled=True,
                mlen=candidate.mlen,
                hlen=candidate.hlen,
                blen=candidate.blen,
                kv_heads=attention_partition.local_kv_heads,
                fp_sram_depth=int(getattr(hardware, "FP_SRAM_DEPTH", 512)),
            )
            if not bool(reuse["supported"]):
                raise ValueError(
                    "KV_HEAD_REUSE exceeds FP-SRAM/head-broadcast capacity"
                )
        self.sim._dd.set_area_model("calibrated", precision.spec)
        area_key = (
            precision_id,
            candidate.mlen,
            candidate.blen,
            candidate.hlen,
            candidate.hbm_channels,
            candidate.chip_count,
            candidate.link_ports,
            candidate.kv_head_reuse,
            candidate.drain_overlapped,
            attention_partition.local_kv_heads,
        )
        area = self._resource_area_cache.get(area_key)
        if area is None:
            area = self.sim._dd.system_area(
                hardware,
                precision.spec,
                chip_count=candidate.chip_count,
                link_ports=candidate.link_ports,
                link_generation="nvlink4",
                kv_head_reuse=candidate.kv_head_reuse,
                drain_overlapped=candidate.drain_overlapped,
                kv_heads=attention_partition.local_kv_heads,
            )
            if len(self._resource_area_cache) >= 4096:
                self._resource_area_cache.clear()
            self._resource_area_cache[area_key] = area
        chip_area = float(area["chip_area_mm2"])
        system_area = float(area["area_mm2"])
        link_area = system_area - chip_area * candidate.chip_count
        if link_area < -1e-9:
            raise ValueError("system area is smaller than its decode chips")
        return SimulatorResourcePreflight(
            profile_id=entry.profile_id,
            candidate_id=candidate.candidate_id,
            analytical_area_mm2=chip_area,
            system_area_mm2=system_area,
            capacity=capacity,
            runtime_feasible=runtime_feasible,
            max_resident_batch=int(ledger.max_resident_batch),
            max_runtime_batch=int(ledger.max_runtime_batch),
            vector_sram_capacity_bytes=int(sram.vector_capacity_bytes),
            vector_sram_required_bytes=int(sram.vector_required_bytes),
            matrix_sram_capacity_bytes=int(sram.matrix_capacity_bytes),
            matrix_sram_required_bytes=int(sram.matrix_required_bytes),
            link_area_mm2=max(0.0, link_area),
        )

    def evaluate(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        workload: HardwareWorkload,
    ) -> SimulatorObservation:
        request = precision_request(entry.profile)
        precision = self.sim.make_precision(
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
        hbm = self.sim.hbm_overrides(
            candidate.hbm_generation,
            candidate.hbm_channels,
        )
        override = {
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "VLEN": candidate.vlen,
            "HLEN": candidate.hlen,
            "TP": candidate.tp,
            "KVP": candidate.kvp,
            "LINK_PORTS": candidate.link_ports,
            "SRAM_POLICY": candidate.sram_policy,
            "LINK_GENERATION": "nvlink4",
            **hbm,
        }
        if candidate.architecture_knobs_explicit:
            override.update(
                {
                    "KV_HEAD_REUSE": candidate.kv_head_reuse,
                    "DRAIN_OVERLAPPED": candidate.drain_overlapped,
                }
            )
        self.sim._dd.set_area_model("calibrated", precision.spec)
        metrics = self.sim.evaluate(
            precision,
            batch=candidate.batch,
            input_seq=workload.input_seq,
            output_seq=workload.output_seq,
            hw_over=override,
            n_chips=candidate.chip_count,
            stride=workload.stride,
            hbm_gen=candidate.hbm_generation,
            hbm_channels=candidate.hbm_channels,
            kv_layout=workload.kv_layout,
            runtime_hbm_reserve_bytes=workload.runtime_hbm_reserve_bytes,
            output_head_location=self.output_head_location,
        )
        if metrics.n_chips != candidate.chip_count:
            raise ValueError("simulator changed the fixed chip count")
        if (
            self.calibrated_bandwidth
            and not metrics.bandwidth_calibration_id
        ):
            raise ValueError("simulator omitted the bandwidth calibration identity")
        traffic = physical_traffic_from_metrics(
            metrics,
            batch=candidate.batch,
        )
        total_hbm_bytes = (
            metrics.avg_hbm_bytes_per_batch_step
            * workload.output_seq
        )
        capacity = capacity_from_metrics(
            metrics,
            batch=candidate.batch,
        )

        shape = DenseDecoderShape.from_mapping(self.sim.dims)
        attention_partition = RankLocalAttentionGeometry.bind(shape, candidate)
        profile = entry.profile
        selector_capability = evaluate_stack_capability(
            profile,
            _selector_runtime_target(profile, candidate, shape),
        )
        selector_issue_codes = tuple(
            sorted(
                issue.code
                for issue in selector_capability.issues
                if "rtl" in issue.stages
            )
        )
        selector_capability_id = (
            "packedkv-selector-capability-"
            + _content_hash(selector_capability.to_dict())
        )
        events = count_decode_events(
            shape,
            input_seq=workload.input_seq,
            output_seq=workload.output_seq,
            batch=candidate.batch,
            mlen=candidate.mlen,
            blen=candidate.blen,
            hlen=candidate.hlen,
            linear_signature=(
                f"LINEAR:{_simulator_token(profile.weight_format)}"
                f"x{_simulator_token(profile.activation_format)}"
            ),
            qk_signature=(
                f"QK:{_simulator_token(profile.key_format)}"
                f"x{_simulator_token(profile.activation_format)}"
            ),
            pv_signature=(
                f"PV:{_simulator_token(profile.value_format)}"
                f"x{_simulator_token(profile.activation_format)}"
            ),
            vector_signature=f"VECTOR:{profile.vector_format}",
            stride=workload.stride,
            include_output_head=(
                self.output_head_location == DECODE_BF16_HEAD
            ),
        )
        architecture_issue = (
            "memory"
            if metrics.frac_architecture_issue_mem_bound >= 0.5
            else "compute"
        )
        collective = self.sim._dd.collective_cost_per_step(
            self.sim.dims,
            batch=candidate.batch,
            tp=candidate.tp,
            kvp=candidate.kvp,
            link_ports=candidate.link_ports,
            link_generation="nvlink4",
        )
        area_status = self.sim._dd.system_area(
            self.sim.base_hw.model_copy(update=override),
            precision.spec,
            chip_count=candidate.chip_count,
            link_ports=candidate.link_ports,
            link_generation="nvlink4",
            kv_head_reuse=(
                candidate.kv_head_reuse
                if candidate.architecture_knobs_explicit
                else False
            ),
            drain_overlapped=(
                candidate.drain_overlapped
                if candidate.architecture_knobs_explicit
                else False
            ),
            kv_heads=attention_partition.local_kv_heads,
        )
        chip_area_status = area_status.get("chip")
        if not isinstance(chip_area_status, Mapping):
            raise ValueError("full-chip area ledger omitted its chip breakdown")
        logic_area_mm2 = float(chip_area_status["logic_area"]) / 1e6
        fp_sram_depth = int(
            getattr(
                self.sim.base_hw.model_copy(update=override),
                "FP_SRAM_DEPTH",
                512,
            )
        )
        reuse_status = self.sim._dd.kv_head_reuse_status(
            enabled=(
                candidate.kv_head_reuse
                if candidate.architecture_knobs_explicit
                else True
            ),
            mlen=candidate.mlen,
            hlen=candidate.hlen,
            blen=candidate.blen,
            kv_heads=attention_partition.local_kv_heads,
            fp_sram_depth=fp_sram_depth,
        )
        reuse_status = {
            **reuse_status,
            "requested": (
                candidate.kv_head_reuse
                if candidate.architecture_knobs_explicit
                else None
            ),
            "legacy_implicit_default": (
                not candidate.architecture_knobs_explicit
            ),
            "legality_enforced": candidate.architecture_knobs_explicit,
        }
        drain_status = {
            "requested": (
                candidate.drain_overlapped
                if candidate.architecture_knobs_explicit
                else None
            ),
            "enabled": (
                candidate.drain_overlapped
                if candidate.architecture_knobs_explicit
                else metrics.timing_mode == "drain_overlapped"
            ),
            "timing_mode": metrics.timing_mode,
            "timing_calibrated": metrics.timing_calibrated,
            "timing_evidence_id": metrics.timing_evidence_id,
            "timing_evidence_tier": metrics.timing_evidence_tier,
            "timing_reason": metrics.timing_reason,
            "second_accumulator_bank_bytes_per_chip": (
                self.sim._dd.DRAIN_ACCUMULATOR_BYTES_PER_CHIP
                if (
                    candidate.architecture_knobs_explicit
                    and candidate.drain_overlapped
                )
                else 0
            ),
            "evidence_tier": (
                "not_applicable"
                if not (
                    candidate.drain_overlapped
                    if candidate.architecture_knobs_explicit
                    else metrics.timing_mode == "drain_overlapped"
                )
                else (
                    (
                        "compiler_trace_request_calibrated"
                        if metrics.execution_mode
                        == COMPILER_TRACE_EXECUTION_MODE
                        else (
                            "matched_analytic_emulator_timing"
                            if metrics.timing_evidence_tier == "emulator"
                            else "matched_emulator_rtl_timing"
                        )
                    )
                    if metrics.timing_calibrated
                    else "analytic_codesign_unrankable"
                )
            ),
        }
        architecture_options = {
            "schema": "plena-decode-architecture-options",
            "explicit": candidate.architecture_knobs_explicit,
            "attention_partition": attention_partition.to_dict(),
            "kv_head_reuse": reuse_status,
            "drain_overlapped": drain_status,
            "area": dict(area_status["architecture_options"]),
        }
        token_traffic = dict(metrics.hbm_traffic_per_generated_token)
        context_tokens = workload.input_seq + workload.output_seq
        kv_storage_per_sequence = (
            capacity.kv_cache_bytes / candidate.batch
        )
        capacity_throughput_chain = {
            "schema": "plena-kv-capacity-throughput-chain",
            "kv_storage_bytes_per_active_sequence": kv_storage_per_sequence,
            "kv_storage_bytes_per_sequence_context_token": (
                kv_storage_per_sequence / context_tokens
            ),
            "kv_read_bytes_per_generated_token": (
                token_traffic["kv_element_read_bytes"]
                + token_traffic["kv_scale_read_bytes"]
            ),
            "kv_write_bytes_per_generated_token": (
                token_traffic["kv_element_write_bytes"]
                + token_traffic["kv_scale_write_bytes"]
            ),
            "max_feasible_batch": metrics.max_runtime_batch,
            "evaluated_batch": candidate.batch,
            "capacity_binding": (
                candidate.batch == metrics.max_runtime_batch
            ),
            "runtime_feasible": metrics.fits_runtime,
            "evaluated_throughput_tokens_per_second": (
                metrics.tps if metrics.fits_runtime else None
            ),
            "throughput_semantics": (
                "measured_at_evaluated_batch_no_capacity_extrapolation"
            ),
            "byte_unit": "physical_bytes",
            "batch_unit": "active_sequences",
        }
        return SimulatorObservation(
            profile_id=entry.profile_id,
            candidate_id=candidate.candidate_id,
            tpot_ms=metrics.tpot * 1000.0,
            tps=metrics.tps,
            total_time_s=metrics.total_time,
            analytical_area_mm2=float(area_status["chip_area_mm2"]),
            traffic=traffic,
            capacity=capacity,
            algorithmic_bottleneck=architecture_issue,
            realized_bottleneck=metrics.bottleneck,
            frac_algorithmic_memory_bound=metrics.frac_algorithmic_mem_bound,
            frac_realized_memory_bound=metrics.frac_mem_bound,
            frac_serialization_bound=metrics.frac_serialization_bound,
            generated_tokens_per_step=candidate.batch,
            decode_steps=workload.output_seq,
            timing_mode=metrics.timing_mode,
            timing_calibrated=metrics.timing_calibrated,
            timing_evidence_id=metrics.timing_evidence_id,
            timing_reason=metrics.timing_reason,
            execution_mode=metrics.execution_mode,
            compiler_trace_timing=metrics.compiler_trace_timing,
            kv_layout=metrics.kv_layout,
            layout_id=metrics.kv_layout_id,
            capacity_model=self.capacity_model_id,
            runtime_feasible=metrics.fits_runtime,
            max_batch=metrics.max_batch,
            max_resident_batch=metrics.max_resident_batch,
            max_synchronous_batch=metrics.max_synchronous_batch,
            max_runtime_batch=metrics.max_runtime_batch,
            fits_onchip_sram=metrics.fits_onchip_sram,
            vector_sram_capacity_bytes=metrics.vector_sram_capacity_bytes,
            vector_sram_required_bytes=metrics.vector_sram_required_bytes,
            matrix_sram_capacity_bytes=metrics.matrix_sram_capacity_bytes,
            matrix_sram_required_bytes=metrics.matrix_sram_required_bytes,
            hbm_traffic_per_batch_step=(
                metrics.hbm_traffic_per_batch_step
            ),
            hbm_traffic_per_generated_token=(
                metrics.hbm_traffic_per_generated_token
            ),
            traffic_ledger_id=self.traffic_ledger_id,
            packedkv_selector_supported=not selector_issue_codes,
            packedkv_selector_capability_id=selector_capability_id,
            packedkv_selector_issue_codes=selector_issue_codes,
            bandwidth_calibration_id=metrics.bandwidth_calibration_id,
            total_hbm_bytes=total_hbm_bytes,
            events=events,
            output_head_location=metrics.output_head_location,
            collective_time_s_per_step=float(collective["time_s"]),
            collective_bytes_per_generated_token=(
                float(collective["total_bytes"]) / candidate.batch
            ),
            link_generation="nvlink4",
            system_area_mm2=float(area_status["area_mm2"]),
            area_evidence_tier=str(area_status["evidence_tier"]),
            logic_area_mm2=logic_area_mm2,
            classical_roofline_bottleneck=(
                metrics.classical_roofline_bottleneck
            ),
            architecture_issue_bottleneck=(
                metrics.architecture_issue_bottleneck
            ),
            frac_classical_memory_bound=(
                metrics.frac_classical_mem_bound
            ),
            frac_architecture_issue_memory_bound=(
                metrics.frac_architecture_issue_mem_bound
            ),
            avg_peak_compute_seconds=(
                metrics.avg_peak_compute_seconds
            ),
            avg_ideal_compute_seconds=(
                metrics.avg_ideal_compute_seconds
            ),
            avg_realized_compute_seconds=(
                metrics.avg_realized_compute_seconds
            ),
            avg_memory_seconds=metrics.avg_memory_seconds,
            step_composition=metrics.step_composition,
            architecture_options=architecture_options,
            capacity_throughput_chain=capacity_throughput_chain,
        )


class SimulatorPowerEngine:
    """Validated simulator power model with independent bridge checks."""

    def __init__(
        self,
        calibration_path: str | os.PathLike[str],
        area_config: Mapping[str, Any],
    ) -> None:
        source = Path(calibration_path).resolve()
        self.status = load_simulator_power_artifact(
            source,
            required_signatures=required_hardware_power_signatures(),
        )
        root = simulator_root()
        analytic = root / "analytic_models"
        if str(analytic) not in sys.path:
            sys.path.insert(0, str(analytic))
        from power.model import PowerCalibration

        self.calibration = PowerCalibration.load(source)
        canonical_config = {
            str(key): int(value) for key, value in area_config.items()
        }
        missing = [key for key in _AREA_KEYS if key not in canonical_config]
        if missing:
            raise ValueError(
                "area configuration is missing " + ", ".join(missing)
            )
        if any(canonical_config[key] <= 0 for key in _AREA_KEYS):
            raise ValueError("area configuration values must be positive")
        if canonical_config["BLOCK_DIM"] != 8:
            raise ValueError("area configuration must use MX block size 8")
        self.area_config = canonical_config
        calibration_status = self.status.to_dict()
        calibration_status.pop("source_path", None)
        self._provenance = {
            "engine": "PowerCalibration/estimate_power",
            "calibration_status": calibration_status,
            "area_config": dict(sorted(self.area_config.items())),
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def area_mm2(
        self,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
        events: Iterable[DecodeEvent],
    ) -> float:
        """Evaluate the calibrated chip area without timing or traffic pricing."""

        from power.model import EventCount, estimate_power

        area_config = {
            **self.area_config,
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "VLEN": candidate.vlen,
        }
        estimate = estimate_power(
            self.calibration,
            (
                EventCount(
                    event.signature,
                    event.count,
                    event.mlen,
                    event.blen,
                )
                for event in events
            ),
            elapsed_s=1.0,
            hbm_bytes=0.0,
            vector_fp=profile.vector_format,
            area_config=area_config,
            selector_enabled=True,
        )
        if not estimate.rankable:
            missing = ",".join(estimate.missing_signatures)
            raise ValueError(f"area estimate is non-rankable: {missing}")
        area = calibrated_area_from_simulator(
            self.status,
            estimate.to_dict(),
        )
        if area is None:
            raise ValueError("power bridge rejected the calibrated area")
        return area

    def evaluate(
        self,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
        observation: SimulatorObservation,
    ) -> PowerOutcome:
        from power.model import EventCount, estimate_power

        if not observation.packedkv_selector_supported:
            raise ValueError(
                "selector power cannot rank an unsupported PackedKV path"
            )
        area_config = {
            **self.area_config,
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "VLEN": candidate.vlen,
        }
        estimate = estimate_power(
            self.calibration,
            (
                EventCount(
                    event.signature,
                    event.count,
                    event.mlen,
                    event.blen,
                )
                for event in observation.events
            ),
            elapsed_s=observation.total_time_s,
            hbm_bytes=observation.total_hbm_bytes,
            vector_fp=profile.vector_format,
            area_config=area_config,
            selector_enabled=True,
        )
        estimate_dict = estimate.to_dict()
        if not estimate.rankable:
            missing = ",".join(estimate.missing_signatures)
            raise ValueError(f"power estimate is non-rankable: {missing}")
        area = calibrated_area_from_simulator(self.status, estimate_dict)
        total_energy = calibrated_energy_from_simulator(
            self.status,
            estimate_dict,
            duration_s=observation.total_time_s,
        )
        if area is None or total_energy is None:
            raise ValueError("power bridge rejected the calibrated estimate")
        generated_tokens = (
            observation.generated_tokens_per_step
            * observation.decode_steps
        )
        if generated_tokens <= 0:
            raise ValueError("power normalization token count is invalid")
        energy = CalibratedEnergy(
            calibration_id=total_energy.calibration_id,
            compute_j=total_energy.compute_j / generated_tokens,
            vector_j=total_energy.vector_j / generated_tokens,
            sram_j=total_energy.sram_j / generated_tokens,
            hbm_j=total_energy.hbm_j / generated_tokens,
            leakage_j=total_energy.leakage_j / generated_tokens,
            unattributed_dynamic_j=(
                total_energy.unattributed_dynamic_j / generated_tokens
            ),
            duration_s=total_energy.duration_s / generated_tokens,
        )
        return PowerOutcome(
            area_mm2=area,
            energy=energy,
            calibration_id=self.status.calibration_id,
        )

    def hbm_energy_per_token(
        self,
        observation: SimulatorObservation,
    ) -> tuple[float, str]:
        """Return only independently sourced HBM energy for an exact anchor."""

        if not self.status.passed:
            raise ValueError("HBM energy requires a passing power artifact")
        generated_tokens = (
            observation.generated_tokens_per_step
            * observation.decode_steps
        )
        if generated_tokens <= 0:
            raise ValueError("power normalization token count is invalid")
        coefficient = float(self.status.raw["hbm_energy_j_per_byte"])
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("HBM energy coefficient is invalid")
        return (
            observation.total_hbm_bytes
            * coefficient
            / generated_tokens,
            self.status.calibration_id,
        )

    def anchor_prediction(
        self,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
        observation: SimulatorObservation,
    ) -> Mapping[str, Any]:
        """Evaluate the model at an exact-anchor point without ranking it."""

        from power.model import EventCount, estimate_power

        area_config = {
            **self.area_config,
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "VLEN": candidate.vlen,
        }
        estimate = estimate_power(
            self.calibration,
            (
                EventCount(
                    event.signature,
                    event.count,
                    event.mlen,
                    event.blen,
                )
                for event in observation.events
            ),
            elapsed_s=observation.total_time_s,
            hbm_bytes=observation.total_hbm_bytes,
            vector_fp=profile.vector_format,
            area_config=area_config,
            selector_enabled=True,
        )
        if not estimate.calibrated or estimate.missing_signatures:
            raise ValueError(
                "exact-anchor prediction lacks calibrated event coverage"
            )
        hbm_j, hbm_calibration_id = self.hbm_energy_per_token(
            observation
        )
        return {
            "profile_id": profile.profile_id,
            "candidate_id": candidate.candidate_id,
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "selector_enabled": estimate.selector_enabled,
            "area_mm2": estimate.area_mm2,
            "dynamic_power_w": (
                estimate.dynamic_energy_j / observation.total_time_s
            ),
            "leakage_power_w": estimate.leakage_power_w,
            "hbm_j_per_token": hbm_j,
            "hbm_calibration_id": hbm_calibration_id,
            "prediction_calibration_id": self.status.calibration_id,
            "synthesis_context": dict(
                self.status.raw["synthesis_context"]
            ),
            "model_rankable_without_anchor": estimate.rankable,
            "extrapolation_reasons": list(
                estimate.extrapolation_reasons
            ),
        }


class ProductionHardwareEvaluator:
    """Fail-closed evaluator injected into ExactHardwareStudy."""

    def __init__(
        self,
        backend: SimulatorBackend,
        workload: HardwareWorkload,
        *,
        power_engine: PowerEngine | None = None,
        dc_anchor_index: ExactDCAnchorIndex | None = None,
        head_service_status: BF16HeadServiceStatus | None = None,
        admission_correctness_status: AdmissionCorrectnessStatus | None = None,
        handoff_artifact: PrefillHandoffArtifact | None = None,
        resource_budget: ResourceBudget = ResourceBudget(),
        publication_timing_tier: str = COMPILER_TRACE_TIMING_TIER,
    ) -> None:
        self.backend = backend
        self.workload = workload
        self.power_engine = power_engine
        self.dc_anchor_index = dc_anchor_index
        if publication_timing_tier not in PUBLICATION_TIMING_TIERS:
            raise ValueError(
                "publication_timing_tier must be a declared timing tier"
            )
        self.publication_timing_tier = publication_timing_tier
        if handoff_artifact is not None and not isinstance(
            handoff_artifact,
            PrefillHandoffArtifact,
        ):
            raise TypeError("handoff_artifact must be PrefillHandoffArtifact")
        if handoff_artifact is not None and not callable(
            getattr(backend, "evaluate_handoff", None)
        ):
            raise ValueError(
                "the selected backend cannot evaluate the handoff artifact"
            )
        self.handoff_artifact = handoff_artifact
        if not isinstance(resource_budget, ResourceBudget):
            raise TypeError("resource_budget must be ResourceBudget")
        self.resource_budget = resource_budget
        if dc_anchor_index is not None and power_engine is None:
            raise ValueError(
                "exact DC anchors require calibrated HBM transfer energy"
            )
        self.head_service_status = (
            head_service_status
            if head_service_status is not None
            else getattr(backend, "head_service_status", None)
        )
        self.head_service_calibration = (
            self.head_service_status.calibration
            if (
                self.head_service_status is not None
                and self.head_service_status.passed
            )
            else None
        )
        admission_status = (
            admission_correctness_status
            if admission_correctness_status is not None
            else missing_admission_correctness_status()
        )
        if (
            admission_status.passed
            and not admission_correctness_status_valid(
                admission_status.to_dict()
            )
        ):
            admission_status = missing_admission_correctness_status(
                "invalid_admission_correctness_status"
            )
        self.admission_correctness_status = admission_status
        self.admission_correctness_valid = (
            admission_correctness_status_valid(
                self.admission_correctness_status.to_dict()
            )
        )
        expected_head_location = (
            EXTERNAL_BF16_HEAD
            if self.head_service_calibration is not None
            else DECODE_BF16_HEAD
        )
        backend_location = backend.provenance.get(
            "output_head_location"
        )
        if (
            backend_location is not None
            and backend_location != expected_head_location
        ):
            raise ValueError(
                "backend and head-service artifact boundaries differ"
            )
        self.output_head_location = expected_head_location
        self.output_head_status = (
            self.head_service_status.to_dict()
            if self.head_service_status is not None
            else {
                "schema_version": HEAD_SERVICE_SCHEMA,
                "artifact_sha256": None,
                "passed": False,
                "failures": ["missing_head_service_artifact"],
                "calibration_id": None,
                "provenance_id": None,
                "service_mode": "unmodeled",
                "service_location": None,
                "required_batches": [],
            }
        )
        payload = {
            "version": EVALUATOR_VERSION,
            "event_model": EVENT_MODEL,
            "software_source_sha256": _module_source_digests(),
            "workload": workload.to_dict(),
            "backend": dict(backend.provenance),
            "power": (
                dict(power_engine.provenance)
                if power_engine is not None
                else None
            ),
            "analytic_power": analytic_power_provenance(),
            "resource_budget": self.resource_budget.to_dict(),
            "exact_dc_anchors": (
                dc_anchor_index.to_status()
                if dc_anchor_index is not None
                else None
            ),
            "traffic_unit": TRAFFIC_UNIT,
            "output_head_location": self.output_head_location,
            "head_service_status": dict(self.output_head_status),
            "admission_correctness_status": (
                self.admission_correctness_status.to_dict()
            ),
            "prefill_handoff_input": (
                handoff_artifact.to_status()
                if handoff_artifact is not None
                else None
            ),
        }
        self.provenance = payload
        self.evaluator_id = (
            f"{EVALUATOR_VERSION}:{_content_hash(payload)}"
        )

    def physical_cost_group_key(
        self,
        entry: SweepManifestEntry,
        numerical_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Declare the complete profile-side cost equivalence relation."""

        return physical_cost_signature(
            entry.profile,
            exact_vector_format=True,
        )

    def candidate_preflight(
        self,
        candidate: HardwareCandidate,
    ) -> HardwareEvaluation | None:
        """Apply precision-independent aggregate HBM resource limits."""

        simulator = getattr(self.backend, "sim", None)
        hbm_overrides = getattr(simulator, "hbm_overrides", None)
        if hbm_overrides is None:
            return None
        try:
            hbm = hbm_overrides(
                candidate.hbm_generation,
                candidate.hbm_channels,
            )
            capacity = int(hbm["HBM_SIZE"]) * candidate.chip_count
            bandwidth = (
                hbm_peak_bandwidth_bytes_per_s(
                    candidate.hbm_generation,
                    candidate.hbm_channels,
                )
                * candidate.chip_count
            )
        except Exception:
            return None
        failed = []
        if capacity > self.resource_budget.aggregate_hbm_capacity_limit_bytes:
            failed.append("hbm_capacity")
        if (
            bandwidth
            > self.resource_budget.aggregate_hbm_bandwidth_limit_bytes_per_s
        ):
            failed.append("hbm_bandwidth")
        if failed:
            return HardwareEvaluation.failed(
                "candidate_resource_budget_exceeded",
                "aggregate " + ",".join(failed) + " budget exceeded",
            )
        return None

    def preflight_group_key(
        self,
        entry: SweepManifestEntry,
        numerical_result: Mapping[str, Any],
    ) -> Mapping[str, Any] | str:
        """Keep profile-scoped exact anchors out of shared gate decisions."""

        if self.dc_anchor_index is not None:
            return f"exact-anchor-profile:{entry.profile_id}"
        return self.physical_cost_group_key(entry, numerical_result)

    def evaluation_group_key(
        self,
        entry: SweepManifestEntry,
        numerical_result: Mapping[str, Any],
    ) -> Mapping[str, Any] | str:
        """Declare when one priced result can be joined to several rows."""

        if self.dc_anchor_index is not None:
            return f"exact-anchor-profile:{entry.profile_id}"
        return self.physical_cost_group_key(entry, numerical_result)

    def _area_events(
        self,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
    ) -> tuple[DecodeEvent, ...]:
        simulator = getattr(self.backend, "sim", None)
        dimensions = getattr(simulator, "dims", None)
        if not isinstance(dimensions, Mapping):
            raise TypeError("area preflight requires simulator dimensions")
        shape = DenseDecoderShape.from_mapping(dimensions)
        return count_decode_events(
            shape,
            input_seq=self.workload.input_seq,
            output_seq=self.workload.output_seq,
            batch=candidate.batch,
            mlen=candidate.mlen,
            blen=candidate.blen,
            hlen=candidate.hlen,
            linear_signature=(
                f"LINEAR:{_simulator_token(profile.weight_format)}"
                f"x{_simulator_token(profile.activation_format)}"
            ),
            qk_signature=(
                f"QK:{_simulator_token(profile.key_format)}"
                f"x{_simulator_token(profile.activation_format)}"
            ),
            pv_signature=(
                f"PV:{_simulator_token(profile.value_format)}"
                f"x{_simulator_token(profile.activation_format)}"
            ),
            vector_signature=f"VECTOR:{profile.vector_format}",
            stride=self.workload.stride,
            include_output_head=(
                self.output_head_location == DECODE_BF16_HEAD
            ),
        )

    def preflight(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        numerical_result: Mapping[str, Any],
    ) -> HardwareEvaluation | None:
        """Reject only exact hard-constraint failures before timing pricing."""

        simulator = getattr(self.backend, "sim", None)
        dimensions = getattr(simulator, "dims", None)
        if isinstance(dimensions, Mapping):
            try:
                shape = DenseDecoderShape.from_mapping(dimensions)
                partition = RankLocalAttentionGeometry.bind(shape, candidate)
                capability = evaluate_stack_capability(
                    entry.profile,
                    _selector_runtime_target(entry.profile, candidate, shape),
                )
            except Exception:
                # A malformed proof cannot reject a point before the full
                # fail-closed evaluator has recorded the failure.
                capability = None
                partition = None
                shape = None
            if capability is not None and partition is not None and shape is not None:
                geometry_codes = {
                    issue.code
                    for issue in capability.issues
                    if "compiler" in issue.stages
                    and issue.code
                    in {
                        "packedkv_block_alignment",
                        "packedkv_hlen_tiling",
                        "packedkv_selector_stride",
                        "packedkv_row_overflow",
                        "packedkv_head_slots",
                        "packedkv_selector_encoding",
                    }
                }
                if (
                    partition.local_attention_heads
                    // partition.local_kv_heads
                    > candidate.mlen // candidate.hlen
                ):
                    geometry_codes.add("packed_attention_gqa_broadcast")
                if geometry_codes:
                    return HardwareEvaluation.failed(
                        "native_compiler_geometry_unsupported",
                        "exact rank-local geometry: "
                        + ",".join(sorted(geometry_codes)),
                    )

        resource_preflight = getattr(self.backend, "resource_preflight", None)
        if resource_preflight is None:
            return None
        try:
            status = resource_preflight(entry, candidate, self.workload)
            if not isinstance(status, SimulatorResourcePreflight):
                raise TypeError(
                    "simulator resource preflight returned an invalid status"
                )
            if status.profile_id != entry.profile_id:
                raise ValueError("preflight profile identity mismatch")
            if status.candidate_id != candidate.candidate_id:
                raise ValueError("preflight candidate identity mismatch")
        except Exception:
            # A preflight computation failure is not evidence that the point
            # is infeasible.  Preserve it for the full fail-closed evaluator.
            return None
        if not status.runtime_feasible:
            failures = []
            if not status.capacity.feasible:
                failures.append("hbm_capacity")
            if (
                status.vector_sram_required_bytes
                > status.vector_sram_capacity_bytes
            ):
                failures.append("vector_sram_capacity")
            if (
                status.matrix_sram_required_bytes
                > status.matrix_sram_capacity_bytes
            ):
                failures.append("matrix_sram_capacity")
            return HardwareEvaluation.failed(
                "runtime_capacity_exceeded",
                "physical " + ",".join(failures or ("batch_capacity",))
                + " exceeded",
            )

        per_chip_area = status.analytical_area_mm2
        exact_anchor = (
            self.dc_anchor_index.get(entry.profile_id, candidate)
            if self.dc_anchor_index is not None
            else None
        )
        if exact_anchor is not None:
            per_chip_area = float(exact_anchor.area_mm2)
        elif self.power_engine is not None:
            area_method = getattr(self.power_engine, "area_mm2", None)
            if area_method is None:
                per_chip_area = None
            else:
                try:
                    per_chip_area = float(
                        area_method(
                            entry.profile,
                            candidate,
                            self._area_events(entry.profile, candidate),
                        )
                    )
                except Exception:
                    per_chip_area = None
        aggregate_bandwidth = (
            hbm_peak_bandwidth_bytes_per_s(
                candidate.hbm_generation,
                candidate.hbm_channels,
            )
            * candidate.chip_count
        )
        failed = []
        if per_chip_area is not None:
            aggregate_area = (
                per_chip_area * candidate.chip_count
                + status.link_area_mm2
            )
            if aggregate_area > self.resource_budget.aggregate_area_limit_mm2:
                failed.append("area")
        if (
            status.capacity.available_bytes
            > self.resource_budget.aggregate_hbm_capacity_limit_bytes
        ):
            failed.append("hbm_capacity")
        if (
            aggregate_bandwidth
            > self.resource_budget.aggregate_hbm_bandwidth_limit_bytes_per_s
        ):
            failed.append("hbm_bandwidth")
        if failed:
            return HardwareEvaluation.failed(
                "resource_budget_exceeded",
                "aggregate " + ",".join(failed) + " budget exceeded",
            )
        return None

    def __call__(
        self,
        entry: SweepManifestEntry,
        candidate: HardwareCandidate,
        numerical_result: Mapping[str, Any],
    ) -> HardwareEvaluation:
        if numerical_result.get("state") != "succeeded":
            return HardwareEvaluation.failed(
                "numerical_not_succeeded",
                f"numerical state is {numerical_result.get('state')!r}",
            )
        numerical_metrics = numerical_result.get("result")
        if not isinstance(numerical_metrics, Mapping):
            return HardwareEvaluation.failed(
                "numerical_metrics_invalid",
                "successful numerical result is missing",
            )
        mean_nll = numerical_metrics.get("mean_nll")
        token_count = numerical_metrics.get("token_count")
        if (
            isinstance(mean_nll, bool)
            or not isinstance(mean_nll, (int, float))
            or not math.isfinite(mean_nll)
            or mean_nll < 0
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            return HardwareEvaluation.failed(
                "numerical_metrics_invalid",
                "successful numerical NLL or token count is invalid",
            )
        try:
            precision_request(entry.profile)
            observation = self.backend.evaluate(
                entry,
                candidate,
                self.workload,
            )
            if observation.profile_id != entry.profile_id:
                raise ValueError("simulator profile identity mismatch")
            if observation.candidate_id != candidate.candidate_id:
                raise ValueError("simulator candidate identity mismatch")
            if observation.kv_layout != self.workload.kv_layout:
                raise ValueError("simulator layout identity mismatch")
            if (
                observation.output_head_location
                != self.output_head_location
            ):
                raise ValueError("simulator output-head boundary mismatch")
            expected_tps = (
                observation.generated_tokens_per_step
                * 1000.0
                / observation.tpot_ms
            )
            if abs(observation.tps - expected_tps) > max(
                1e-9,
                expected_tps * 1e-9,
            ):
                raise ValueError("simulator TPOT and TPS are inconsistent")
        except Exception as exc:
            return HardwareEvaluation.failed(
                "simulator_evaluation_failed",
                f"{type(exc).__name__}: {exc}",
            )

        error_code: str | None = None
        error_message: str | None = None
        area = observation.analytical_area_mm2
        area_source = "analytical_uncalibrated"
        area_calibration_id: str | None = None
        energy: CalibratedEnergy | None = None
        whole_energy: CalibratedEnergy | None = None
        system_calibration_id: str | None = None
        dc_valid: bool | None = None
        dc_anchor_id: str | None = None
        dc_anchor_status: Mapping[str, Any] = {}
        head_estimate = None
        head_estimate_error: str | None = None
        if self.head_service_calibration is not None:
            try:
                head_estimate = self.head_service_calibration.estimate(
                    candidate.batch
                )
            except Exception as exc:
                head_estimate_error = f"{type(exc).__name__}: {exc}"
        resolved_timing_tier: str | None = None
        if (
            self.publication_timing_tier == COMPILER_TRACE_TIMING_TIER
            and observation.execution_mode == COMPILER_TRACE_EXECUTION_MODE
        ):
            resolved_timing_tier = COMPILER_TRACE_TIMING_TIER
        elif (
            self.publication_timing_tier
            == STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
            and observation.execution_mode == LEGACY_AGGREGATE_BANDWIDTH_MODE
            and observation.timing_calibrated
            and observation.timing_evidence_id is not None
            and observation.bandwidth_calibration_id is not None
        ):
            resolved_timing_tier = STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
        whole_rankable = (
            head_estimate is not None
            and observation.output_head_location == EXTERNAL_BF16_HEAD
            and observation.runtime_feasible
            and observation.timing_calibrated
            and resolved_timing_tier is not None
            and self.admission_correctness_valid
        )
        whole_tpot_ms = (
            observation.tpot_ms
            + head_estimate.total_latency_s * 1000.0
            if whole_rankable and head_estimate is not None
            else None
        )
        whole_tps = (
            candidate.batch * 1000.0 / whole_tpot_ms
            if whole_tpot_ms is not None
            else None
        )
        if not observation.runtime_feasible:
            error_code = "runtime_infeasible"
            error_message = "candidate exceeds physical HBM or on-chip SRAM"
        elif not observation.timing_calibrated:
            error_code = "timing_uncalibrated"
            error_message = observation.timing_reason
        elif (
            observation.execution_mode == LEGACY_AGGREGATE_BANDWIDTH_MODE
            and resolved_timing_tier is None
        ):
            error_code = "legacy_timing_sensitivity_unrankable"
            error_message = (
                "aggregate-bandwidth timing is rankable only at the "
                "explicitly requested stage_calibrated_analytic tier with "
                "calibrated bandwidth and timing evidence"
            )
        elif not observation.packedkv_selector_supported:
            error_code = "packedkv_selector_unsupported"
            error_message = (
                "PackedKV selector capability failed: "
                + ",".join(observation.packedkv_selector_issue_codes)
            )
        elif head_estimate_error is not None:
            error_code = "head_service_evaluation_failed"
            error_message = head_estimate_error
        elif head_estimate is None:
            error_code = "output_head_unmodeled"
            error_message = (
                "UNMODELED:LM_HEAD_BF16; a passing remote BF16 "
                "head-service artifact is required"
            )
        elif not self.admission_correctness_valid:
            error_code = "admission_correctness_unverified"
            error_message = ",".join(
                self.admission_correctness_status.failures
            )
        if error_code is None:
            try:
                if observation.logic_area_mm2 is None:
                    raise ValueError(
                        "simulator omitted the non-SRAM logic-area ledger"
                    )
                mac_bits = max(
                    format_descriptor(token).element_bits
                    for token in (
                        entry.profile.weight_format,
                        entry.profile.activation_format,
                        entry.profile.key_format,
                        entry.profile.value_format,
                    )
                )
                energy = analytic_energy_from_simulator(
                    candidate=candidate,
                    observation=observation,
                    mac_bits=mac_bits,
                    per_chip_logic_area_mm2=observation.logic_area_mm2,
                    collective_bytes_per_generated_token=(
                        observation.collective_bytes_per_generated_token
                    ),
                    link_generation=observation.link_generation,
                )
                area_source = "analytic_full_chip"
                if whole_rankable and head_estimate is not None:
                    whole_energy, system_calibration_id = (
                        _whole_model_energy(
                            energy,
                            head_estimate,
                            decoder_tpot_ms=observation.tpot_ms,
                            batch=candidate.batch,
                        )
                    )
            except Exception as exc:
                error_code = "analytic_power_failed"
                error_message = f"{type(exc).__name__}: {exc}"
        exact_anchor = (
            self.dc_anchor_index.get(entry.profile_id, candidate)
            if self.dc_anchor_index is not None
            else None
        )
        if error_code is None and exact_anchor is not None:
            try:
                exact_anchor.match(
                    profile=entry.profile,
                    candidate=candidate,
                    observation=observation,
                )
                if self.power_engine is None:
                    raise ValueError(
                        "exact DC anchor is missing HBM calibration"
                    )
                prediction = self.power_engine.anchor_prediction(
                    entry.profile,
                    candidate,
                    observation,
                )
                prediction_status = exact_anchor.validate_prediction(
                    prediction
                )
                duration_s = (
                    observation.tpot_ms
                    / 1000.0
                    / observation.generated_tokens_per_step
                )
                energy = exact_anchor.energy(
                    duration_s=duration_s,
                    hbm_j=float(
                        prediction_status["hbm_j_per_token"]
                    ),
                    hbm_calibration_id=str(
                        prediction_status["hbm_calibration_id"]
                    ),
                )
                area = exact_anchor.area_mm2
                area_source = "dc_calibrated"
                area_calibration_id = energy.calibration_id
                dc_anchor_id = exact_anchor.anchor_id
                dc_anchor_status = exact_anchor.to_status(
                    prediction_status
                )
                dc_valid = True
                if whole_rankable and head_estimate is not None:
                    whole_energy, system_calibration_id = (
                        _whole_model_energy(
                            energy,
                            head_estimate,
                            decoder_tpot_ms=observation.tpot_ms,
                            batch=candidate.batch,
                        )
                    )
            except Exception as exc:
                dc_valid = False
                error_code = "exact_dc_anchor_failed"
                error_message = f"{type(exc).__name__}: {exc}"
        elif error_code is None and self.power_engine is not None:
            try:
                power = self.power_engine.evaluate(
                    entry.profile,
                    candidate,
                    observation,
                )
                if power.energy.calibration_id != power.calibration_id:
                    raise ValueError("power calibration identity mismatch")
                area = power.area_mm2
                area_source = "dc_calibrated_model"
                area_calibration_id = power.calibration_id
                energy = power.energy
                if whole_rankable and head_estimate is not None:
                    whole_energy, system_calibration_id = (
                        _whole_model_energy(
                            energy,
                            head_estimate,
                            decoder_tpot_ms=observation.tpot_ms,
                            batch=candidate.batch,
                        )
                    )
                dc_valid = None
            except Exception as exc:
                dc_valid = None
                error_code = "power_calibration_failed"
                error_message = f"{type(exc).__name__}: {exc}"

        if energy is not None and energy.token_latency_s is None:
            energy = replace(
                energy,
                token_latency_s=observation.tpot_ms / 1000.0,
            )

        if observation.system_area_mm2 is None:
            raise ValueError("simulator omitted full system area")
        analytic_link_area = (
            observation.system_area_mm2
            - observation.analytical_area_mm2 * candidate.chip_count
        )
        if analytic_link_area < -1e-9:
            raise ValueError("system area is smaller than its decode chips")
        system_area = (
            area * candidate.chip_count + max(0.0, analytic_link_area)
        )
        resource_status = ResourceBudgetStatus(
            aggregate_area_mm2=system_area,
            aggregate_hbm_capacity_bytes=(
                observation.capacity.available_bytes
            ),
            aggregate_hbm_bandwidth_bytes_per_s=(
                hbm_peak_bandwidth_bytes_per_s(
                    candidate.hbm_generation,
                    candidate.hbm_channels,
                )
                * candidate.chip_count
            ),
            aggregate_multiplier_count=(
                candidate.mlen
                * candidate.blen
                * candidate.chip_count
            ),
            budget=self.resource_budget,
        )
        if error_code is None and not resource_status.feasible:
            error_code = "resource_budget_exceeded"
            failed = [
                name
                for name, passed in (
                    ("area", resource_status.area_feasible),
                    ("hbm_capacity", resource_status.hbm_capacity_feasible),
                    ("hbm_bandwidth", resource_status.hbm_bandwidth_feasible),
                )
                if not passed
            ]
            error_message = "aggregate " + ",".join(failed) + " budget exceeded"

        handoff_analysis: Mapping[str, Any] | None = None
        if self.handoff_artifact is not None:
            handoff_unavailable_reason = error_code
            if (
                handoff_unavailable_reason is None
                and (
                    not whole_rankable
                    or whole_tpot_ms is None
                    or whole_energy is None
                    or system_calibration_id is None
                    or observation.timing_evidence_id is None
                )
            ):
                handoff_unavailable_reason = (
                    "decode_whole_model_evidence_incomplete"
                )
            if handoff_unavailable_reason is None:
                try:
                    handoff_analysis = self.backend.evaluate_handoff(
                        entry,
                        candidate,
                        self.workload,
                        self.handoff_artifact,
                        decode_tpot_s=float(whole_tpot_ms) / 1000.0,
                        decode_energy_per_token_j=(
                            whole_energy.energy_per_token_j
                        ),
                        decode_energy_tier=whole_energy.energy_tier,
                        decode_timing_evidence_id=(
                            observation.timing_evidence_id
                        ),
                        system_calibration_id=system_calibration_id,
                    )
                except Exception as exc:
                    handoff_unavailable_reason = (
                        "handoff_evaluation_failed:"
                        f"{type(exc).__name__}:{exc}"
                    )
            if handoff_analysis is None:
                handoff_analysis = {
                    "schema_version": PREFILL_HANDOFF_ANALYSIS_SCHEMA,
                    "scope": "request_level_prefill_decode_schedule",
                    "input_artifact_id": self.handoff_artifact.artifact_id,
                    "publication_rankable": False,
                    "unrankable_reason": handoff_unavailable_reason,
                    "ordinary_decode_ranking_effect": "none",
                    "physical_precision_signature": physical_cost_signature(
                        entry.profile,
                        exact_vector_format=True,
                    ),
                    "candidate_id": candidate.candidate_id,
                    "batch": candidate.batch,
                    "regimes": [],
                }

        metrics = HardwareMetrics(
            tpot_ms=observation.tpot_ms,
            tps=observation.tps,
            area_mm2=area,
            traffic=observation.traffic,
            capacity=observation.capacity,
            algorithmic_bottleneck=observation.algorithmic_bottleneck,
            realized_bottleneck=observation.realized_bottleneck,
            frac_algorithmic_memory_bound=(
                observation.frac_algorithmic_memory_bound
            ),
            frac_realized_memory_bound=(
                observation.frac_realized_memory_bound
            ),
            frac_serialization_bound=observation.frac_serialization_bound,
            classical_roofline_bottleneck=(
                observation.classical_roofline_bottleneck
            ),
            architecture_issue_bottleneck=(
                observation.architecture_issue_bottleneck
            ),
            frac_classical_memory_bound=(
                observation.frac_classical_memory_bound
            ),
            frac_architecture_issue_memory_bound=(
                observation.frac_architecture_issue_memory_bound
            ),
            generated_tokens_per_step=observation.generated_tokens_per_step,
            kv_layout=observation.kv_layout,
            layout_id=observation.layout_id,
            capacity_model=observation.capacity_model,
            runtime_feasible=observation.runtime_feasible,
            max_batch=observation.max_batch,
            max_resident_batch=observation.max_resident_batch,
            max_synchronous_batch=observation.max_synchronous_batch,
            max_runtime_batch=observation.max_runtime_batch,
            fits_onchip_sram=observation.fits_onchip_sram,
            vector_sram_capacity_bytes=(
                observation.vector_sram_capacity_bytes
            ),
            vector_sram_required_bytes=(
                observation.vector_sram_required_bytes
            ),
            matrix_sram_capacity_bytes=(
                observation.matrix_sram_capacity_bytes
            ),
            matrix_sram_required_bytes=(
                observation.matrix_sram_required_bytes
            ),
            hbm_traffic_per_batch_step=(
                observation.hbm_traffic_per_batch_step
            ),
            hbm_traffic_per_generated_token=(
                observation.hbm_traffic_per_generated_token
            ),
            traffic_ledger_id=observation.traffic_ledger_id,
            area_source=area_source,
            energy=energy,
            area_calibration_id=area_calibration_id,
            dc_anchor_id=dc_anchor_id,
            dc_anchor_status=dc_anchor_status,
            timing_mode=observation.timing_mode,
            timing_calibrated=observation.timing_calibrated,
            timing_evidence_id=observation.timing_evidence_id,
            timing_reason=observation.timing_reason,
            execution_mode=observation.execution_mode,
            compiler_trace_timing=observation.compiler_trace_timing,
            bandwidth_calibration_id=(
                observation.bandwidth_calibration_id
            ),
            admission_correctness_status=(
                self.admission_correctness_status.to_dict()
            ),
            service_mode=(
                HEAD_SERVICE_MODE
                if self.head_service_calibration is not None
                else "unmodeled"
            ),
            output_head_status=self.output_head_status,
            output_head_service=head_estimate,
            whole_model_tpot_ms=whole_tpot_ms,
            whole_model_tps=whole_tps,
            whole_model_energy=whole_energy,
            system_calibration_id=system_calibration_id,
            whole_model_rankable=whole_rankable,
            publication_timing_tier=(
                resolved_timing_tier if whole_rankable else None
            ),
            avg_peak_compute_seconds=(
                observation.avg_peak_compute_seconds
            ),
            avg_ideal_compute_seconds=(
                observation.avg_ideal_compute_seconds
            ),
            avg_realized_compute_seconds=(
                observation.avg_realized_compute_seconds
            ),
            avg_memory_seconds=observation.avg_memory_seconds,
            step_composition=observation.step_composition,
            system_area_mm2=system_area,
            resource_budget=resource_status,
            architecture_options=observation.architecture_options,
            capacity_throughput_chain=(
                observation.capacity_throughput_chain
            ),
            handoff_analysis=handoff_analysis,
        )
        return HardwareEvaluation(
            metrics=metrics,
            validity=StackValidity(dc_calibrated=dc_valid),
            error_code=error_code,
            error_message=error_message,
        )


def _load_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return dict(value)


def _validate_system_boundary_config(config: Mapping[str, Any]) -> None:
    output_head = config.get("output_head_contract")
    if output_head is not None:
        expected_head = {
            "headline_location": EXTERNAL_BF16_HEAD,
            "headline_precision": "BF16",
            "local_bf16": "sensitivity_unrankable",
            "local_low_precision": "accuracy_only",
        }
        if output_head != expected_head:
            raise ValueError("output_head_contract differs from the evaluator")
    hbm = config.get("hbm_sensitivity")
    if hbm is not None:
        expected_hbm = {
            "schema_version": "decode-hbm-sensitivity",
            "source_profile_count": 4,
            "generations": [
                "HBM2",
                "HBM2E",
                "HBM3",
                "HBM3E",
                "HBM4",
            ],
            "preserve_geometry_batch_channels": True,
            "cross_generation_ranking": False,
        }
        if hbm != expected_hbm:
            raise ValueError("hbm_sensitivity differs from the evaluator")


def _hardware_space(config: Mapping[str, Any]) -> ExactHardwareSpace:
    return ExactHardwareSpace.from_study_config(config)


def _relative_perplexity_limit(config: Mapping[str, Any]) -> float:
    """Translate the configured fractional PPL tolerance into a hard limit."""

    tolerance = config.get("fp_ppl_tol")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0
    ):
        raise ValueError("fp_ppl_tol must be finite and positive")
    return 1.0 + float(tolerance)


def _code_revisions(values: Iterable[str]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for value in values:
        try:
            name, revision = value.split("=", 1)
        except ValueError as exc:
            raise ValueError("code revisions must use NAME=REVISION") from exc
        if not name or not revision or name in revisions:
            raise ValueError(f"invalid code revision {value!r}")
        revisions[name] = revision
    return revisions


def run_exact_hardware_study(
    *,
    manifest: SweepManifest,
    numerical_rows: Iterable[Mapping[str, Any]],
    space: ExactHardwareSpace,
    hidden_size: int,
    evaluator: ProductionHardwareEvaluator,
    output: str | os.PathLike[str],
    code_revisions: Mapping[str, str] | None = None,
    require_complete: bool = True,
    relative_perplexity_limit: float | None = None,
) -> Mapping[str, Any]:
    """Execute and atomically write one provenance-bound study."""

    study = ExactHardwareStudy(
        manifest=manifest,
        numerical_results=numerical_rows,
        space=space,
        hidden_size=hidden_size,
        evaluator=evaluator,
        evaluator_version=evaluator.evaluator_id,
        evaluator_provenance=evaluator.provenance,
        code_revisions=code_revisions,
        require_complete=require_complete,
        relative_perplexity_limit=relative_perplexity_limit,
    )
    artifact = study.write(output)
    return {
        "run_id": artifact.run_id,
        "path": str(artifact.path),
        "metadata_path": str(artifact.metadata_path),
        "result_count": artifact.result_count,
        "content_sha256": artifact.content_hash,
        "evaluator_id": evaluator.evaluator_id,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact decode hardware study."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--numerical-jsonl",
        action="append",
        help="terminal result JSONL or a directory containing result shards",
    )
    parser.add_argument(
        "--refinement-schedule",
        help="immutable refinement schedule for refined-profile repricing",
    )
    parser.add_argument(
        "--refinement-merge",
        help="immutable four-shard merge receipt for refined-profile repricing",
    )
    parser.add_argument(
        "--refinement-results",
        help="optional relocated merged refinement JSONL with the sealed identity",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--timing-evidence", required=True)
    parser.add_argument(
        "--execution-mode",
        choices=(
            COMPILER_TRACE_EXECUTION_MODE,
            LEGACY_AGGREGATE_BANDWIDTH_MODE,
        ),
        required=True,
        help=(
            "compiler_trace prices from full-model traces; "
            "legacy_aggregate_bandwidth prices from the stage-calibrated "
            "analytic model"
        ),
    )
    parser.add_argument(
        "--publication-timing-tier",
        choices=sorted(PUBLICATION_TIMING_TIERS),
        required=True,
        help=(
            "declared timing tier of every rankable row; must match the "
            "execution mode and is stamped on the study output"
        ),
    )
    parser.add_argument(
        "--compiler-trace-artifacts",
        help="content-addressed full-model compiler artifact set",
    )
    parser.add_argument(
        "--request-memory-calibration",
        help="structured HBM request-latency calibration CSV",
    )
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument(
        "--runtime-hbm-reserve-bytes",
        type=int,
        required=True,
    )
    parser.add_argument("--power-calibration")
    parser.add_argument("--area-config")
    parser.add_argument("--exact-dc-anchors")
    parser.add_argument("--rtl-source-tree-sha256")
    parser.add_argument("--head-service-calibration")
    parser.add_argument("--admission-receipt", required=True)
    parser.add_argument(
        "--handoff-artifact",
        help=(
            "optional evidence-bound BF16 prefill/admission artifact for "
            "prefill-to-decode schedule analysis"
        ),
    )
    parser.add_argument(
        "--local-bf16-head-sensitivity",
        action="store_true",
        help=(
            "run an explicitly unrankable local-BF16-head sensitivity "
            "instead of the calibrated remote service"
        ),
    )
    parser.add_argument("--settings-toml")
    parser.add_argument("--isa-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-revision", action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def _validate_execution_launch(
    *,
    execution_mode: str,
    compiler_trace_artifacts: str | os.PathLike[str] | None,
    publication_timing_tier: str,
) -> None:
    if execution_mode not in {
        COMPILER_TRACE_EXECUTION_MODE,
        LEGACY_AGGREGATE_BANDWIDTH_MODE,
    }:
        raise ValueError("unsupported decode execution mode")
    if publication_timing_tier not in PUBLICATION_TIMING_TIERS:
        raise ValueError("unsupported publication timing tier")
    expected_tier = (
        COMPILER_TRACE_TIMING_TIER
        if execution_mode == COMPILER_TRACE_EXECUTION_MODE
        else STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
    )
    if publication_timing_tier != expected_tier:
        raise ValueError(
            "publication timing tier differs from the execution mode: "
            f"{execution_mode} prices at {expected_tier}"
        )
    if execution_mode == COMPILER_TRACE_EXECUTION_MODE:
        if compiler_trace_artifacts is None or not os.fspath(
            compiler_trace_artifacts
        ).strip():
            raise ValueError(
                "compiler_trace requires --compiler-trace-artifacts"
            )
    elif compiler_trace_artifacts is not None:
        raise ValueError(
            "legacy timing cannot consume compiler-trace artifacts"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    refinement_mode = any(
        value is not None
        for value in (
            args.refinement_schedule,
            args.refinement_merge,
            args.refinement_results,
        )
    )
    if refinement_mode:
        if (
            args.refinement_schedule is None
            or args.refinement_merge is None
            or args.numerical_jsonl
            or args.allow_incomplete
        ):
            raise ValueError(
                "refined hardware repricing requires schedule and merge, "
                "and forbids base numerical inputs or incomplete coverage"
            )
    elif not args.numerical_jsonl:
        raise ValueError(
            "base hardware evaluation requires --numerical-jsonl"
        )
    _validate_execution_launch(
        execution_mode=args.execution_mode,
        compiler_trace_artifacts=args.compiler_trace_artifacts,
        publication_timing_tier=args.publication_timing_tier,
    )
    if (
        args.publication_timing_tier == STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
        and args.local_bf16_head_sensitivity
    ):
        raise ValueError(
            "the analytic publication tier requires the remote head service"
        )
    if (
        args.execution_mode == LEGACY_AGGREGATE_BANDWIDTH_MODE
        and args.request_memory_calibration is not None
    ):
        raise ValueError(
            "legacy timing cannot consume request-memory calibration"
        )
    if bool(args.power_calibration) != bool(args.area_config):
        raise ValueError(
            "--power-calibration and --area-config must be supplied together"
        )
    if args.exact_dc_anchors and not args.power_calibration:
        raise ValueError(
            "--exact-dc-anchors requires --power-calibration"
        )
    if bool(args.exact_dc_anchors) != bool(
        args.rtl_source_tree_sha256
    ):
        raise ValueError(
            "--exact-dc-anchors and --rtl-source-tree-sha256 "
            "must be supplied together"
        )
    if args.local_bf16_head_sensitivity:
        if args.head_service_calibration:
            raise ValueError(
                "local head sensitivity cannot use a remote head artifact"
            )
    elif not args.head_service_calibration:
        raise ValueError(
            "the headline study requires --head-service-calibration"
        )
    base_manifest = load_manifest(args.manifest)
    config = _load_json_object(args.config)
    _validate_system_boundary_config(config)
    if config.get("model_name") != base_manifest.model_name:
        raise ValueError("config and manifest model names differ")
    if config.get("model_revision") != base_manifest.model_revision:
        raise ValueError("config and manifest model revisions differ")
    if str(config.get("tokenizer_revision")) != str(
        base_manifest.tokenizer_revision
    ):
        raise ValueError("config and manifest tokenizer revisions differ")
    admission_status = load_admission_correctness_evidence(
        args.admission_receipt,
        manifest_hash=base_manifest.canonical_hash,
    )
    if not admission_status.passed:
        raise ValueError(
            "decode admission correctness evidence failed: "
            + ",".join(admission_status.failures)
        )
    if refinement_mode:
        manifest, rows = _refinement_hardware_inputs(
            base_manifest,
            args.refinement_schedule,
            args.refinement_merge,
            args.refinement_results,
        )
    else:
        manifest = base_manifest
        rows = load_terminal_numerical_rows(
            args.numerical_jsonl,
            base_manifest,
            require_complete=not args.allow_incomplete,
        )
    reference = config.get("reference_workload")
    if not isinstance(reference, Mapping):
        raise ValueError("config is missing reference_workload")
    workload = HardwareWorkload(
        input_seq=int(reference["input_seq"]),
        output_seq=int(reference["output_seq"]),
        stride=args.stride,
        runtime_hbm_reserve_bytes=args.runtime_hbm_reserve_bytes,
    )
    space = _hardware_space(config)
    handoff_artifact = (
        PrefillHandoffArtifact.load(
            args.handoff_artifact,
            model_name=manifest.model_name,
            model_revision=manifest.model_revision,
            workload=workload,
            required_batches=space.batch,
        )
        if args.handoff_artifact is not None
        else None
    )
    backend = DecodeSimulatorBackend(
        model=str(config["sim_model"]),
        model_lib=config.get("model_lib"),
        settings_toml=args.settings_toml,
        isa_path=args.isa_path,
        timing_evidence=args.timing_evidence,
        calibrated_bandwidth=(
            args.execution_mode == LEGACY_AGGREGATE_BANDWIDTH_MODE
            and config.get("bw_model") == "calibrated"
        ),
        execution_mode=args.execution_mode,
        compiler_trace_artifacts=args.compiler_trace_artifacts,
        request_memory_calibration=args.request_memory_calibration,
        head_service_artifact=args.head_service_calibration,
        model_name=manifest.model_name,
        model_revision=manifest.model_revision,
        required_batches=space.batch,
    )
    if (
        not args.local_bf16_head_sensitivity
        and backend.output_head_location != EXTERNAL_BF16_HEAD
    ):
        failures = (
            backend.head_service_status.failures
            if backend.head_service_status is not None
            else ("missing_head_service_artifact",)
        )
        raise ValueError(
            "the remote BF16 output-head service is not rankable: "
            + ",".join(failures)
        )
    if (
        args.publication_timing_tier == STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
        and not backend.calibrated_bandwidth
    ):
        raise ValueError(
            "the stage_calibrated_analytic tier requires "
            "config.bw_model == 'calibrated'"
        )
    if backend.calibrated_bandwidth:
        backend.sim.validate_calibrated_hardware_space(
            space.hbm_generation,
            space.hbm_channels,
        )
    power_engine = (
        SimulatorPowerEngine(
            args.power_calibration,
            _load_json_object(args.area_config),
        )
        if args.power_calibration
        else None
    )
    dc_anchor_index = (
        ExactDCAnchorIndex.load(
            args.exact_dc_anchors,
            model_name=manifest.model_name,
            model_revision=manifest.model_revision,
            workload=workload.to_dict(),
            rtl_source_tree_sha256=args.rtl_source_tree_sha256,
        )
        if args.exact_dc_anchors
        else None
    )
    if refinement_mode and dc_anchor_index is not None:
        refined_ids = {entry.profile_id for entry in manifest.entries}
        if not any(anchor.profile_id in refined_ids for anchor in dc_anchor_index.anchors):
            raise ValueError(
                "refined hardware repricing cannot inherit base-profile DC anchors"
            )
    evaluator = ProductionHardwareEvaluator(
        backend,
        workload,
        power_engine=power_engine,
        dc_anchor_index=dc_anchor_index,
        head_service_status=backend.head_service_status,
        admission_correctness_status=admission_status,
        handoff_artifact=handoff_artifact,
        resource_budget=space.resource_budget,
        publication_timing_tier=args.publication_timing_tier,
    )
    result = run_exact_hardware_study(
        manifest=manifest,
        numerical_rows=rows,
        space=space,
        hidden_size=int(backend.sim.dims["hidden"]),
        evaluator=evaluator,
        output=args.output,
        code_revisions=_code_revisions(args.code_revision),
        require_complete=not args.allow_incomplete,
        relative_perplexity_limit=(
            None if refinement_mode else _relative_perplexity_limit(config)
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DecodeSimulatorBackend",
    "EVALUATOR_VERSION",
    "HardwareWorkload",
    "PowerEngine",
    "PowerOutcome",
    "PREFILL_HANDOFF_ANALYSIS_SCHEMA",
    "PREFILL_HANDOFF_INPUT_SCHEMA",
    "PREFILL_MEASUREMENT_SCOPE",
    "PrefillHandoffArtifact",
    "PrefillMeasurement",
    "PrecisionRequest",
    "ProductionHardwareEvaluator",
    "RankLocalAttentionGeometry",
    "RefinementHardwareEntry",
    "RefinementHardwareManifest",
    "SimulatorBackend",
    "SimulatorObservation",
    "SimulatorPowerEngine",
    "capacity_from_metrics",
    "load_terminal_numerical_rows",
    "main",
    "physical_traffic_from_metrics",
    "precision_request",
    "run_exact_hardware_study",
]
