"""Measured BF16 GPU decode baseline with raw, provenance-bound samples."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.software.token_samples import (
    TokenSampleBundle,
    load_sample_bundle,
)
from decode_dse.software.runtime_environment import (
    RuntimeEnvironment,
    capture_runtime_environment,
)
from decode_dse.software.sweep_plan import (
    load_immutable_json,
    write_immutable_json,
)

GPU_BASELINE_CONTRACT_SCHEMA = "decode-gpu-baseline-contract"
GPU_BASELINE_RESULT_SCHEMA = "decode-gpu-baseline-result"
GPU_BASELINE_REPORT_SCHEMA = "decode-gpu-baseline-report"
GPU_BASELINE_STAGE_RECEIPT_SCHEMA = "decode-gpu-baseline-stage-receipt"
GPU_BASELINE_WORKSPACE_BINDING_SCHEMA = "decode-gpu-baseline-workspace-binding"
GPU_ENERGY_MEASUREMENT_SCHEMA = "decode-gpu-energy-measurement"
ENERGY_EVIDENCE_SCHEMA = "decode-energy-evidence"
ENERGY_COMPARISON_SCHEMA = "decode-energy-comparison"
THROUGHPUT_EVIDENCE_SCHEMA = "decode-throughput-evidence"
THROUGHPUT_COMPARISON_SCHEMA = "decode-throughput-comparison"
GPU_BASELINE_SCOPE = "bf16_aggregated_gpu_cached_q1"
GPU_BASELINE_ENERGY_SOURCE = "bf16_aggregated_gpu_cached_q1_measured_board_energy"
GPU_BASELINE_EXECUTION_MODE = "hf_uncompiled_dynamic_cache"
GPU_BASELINE_TIMING_SCOPE = (
    "cuda_event_causal_lm_forward_with_bf16_head_and_greedy_selection"
)
GPU_BASELINE_ENERGY_SCOPE = (
    "synchronized_board_energy_for_measured_decode_only_excluding_warmup"
)
NVML_TOTAL_ENERGY_METHOD = "nvml_total_energy_counter"
NVML_POWER_TRACE_METHOD = "nvml_power_trace_trapezoidal"
ENERGY_UNAVAILABLE_METHOD = "unavailable"
GPU_BASELINE_ENERGY_METER_PRIORITY = (
    NVML_TOTAL_ENERGY_METHOD,
    NVML_POWER_TRACE_METHOD,
)
NVIDIA_SMI_QUERY_FIELDS = (
    "timestamp",
    "driver_version",
    "name",
    "uuid",
    "pci.bus_id",
    "pstate",
    "memory.total",
    "memory.used",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "mig.mode.current",
)
SWEEP_PROVENANCE_SCHEMA = "decode-sweep-provenance"
MEASURED_EVIDENCE_TIER = "measured"
PEAK_ROOFLINE_EVIDENCE_TIER = "peak_roofline"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_git_commit(value: str, label: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a pinned lowercase Git commit")


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _require_utc_timestamp(value: str, label: str) -> None:
    if not value or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(
            f"{label} must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")


def _normalized_label(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "decode_dse").rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix not in {".py", ".json", ".md"}
        ):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    rank = quantile * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class GPUBaselineWorkspaceBinding:
    """Hashes that bind the measured baseline to one immutable sweep workspace."""

    manifest_hash: str
    run_plan_hash: str
    prompt_manifest_hash: str
    sweep_provenance_sha256: str
    schema_version: str = GPU_BASELINE_WORKSPACE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GPU_BASELINE_WORKSPACE_BINDING_SCHEMA:
            raise ValueError("unsupported GPU baseline workspace-binding schema")
        for label in (
            "manifest_hash",
            "run_plan_hash",
            "prompt_manifest_hash",
            "sweep_provenance_sha256",
        ):
            _require_sha256(str(getattr(self, label)), label)

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_hash": self.manifest_hash,
            "run_plan_hash": self.run_plan_hash,
            "prompt_manifest_hash": self.prompt_manifest_hash,
            "sweep_provenance_sha256": self.sweep_provenance_sha256,
        }

    @property
    def binding_hash(self) -> str:
        return _canonical_hash(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "GPUBaselineWorkspaceBinding":
        binding = cls(
            schema_version=str(value["schema_version"]),
            manifest_hash=str(value["manifest_hash"]),
            run_plan_hash=str(value["run_plan_hash"]),
            prompt_manifest_hash=str(value["prompt_manifest_hash"]),
            sweep_provenance_sha256=str(value["sweep_provenance_sha256"]),
        )
        if value.get("binding_hash") != binding.binding_hash:
            raise ValueError("GPU baseline workspace-binding hash mismatch")
        return binding


def load_gpu_baseline_workspace_binding(
    provenance_path: str | Path,
    bundle: TokenSampleBundle,
) -> GPUBaselineWorkspaceBinding:
    """Bind a baseline to the validated sweep provenance and prompt identity."""

    path = Path(provenance_path).resolve()
    provenance = load_immutable_json(path)
    if provenance.get("schema_version") != SWEEP_PROVENANCE_SCHEMA:
        raise ValueError("GPU baseline requires sweep workspace provenance")
    prompt_manifest_hash = bundle.prompt_manifest().canonical_hash
    if provenance.get("prompt_manifest_hash") != prompt_manifest_hash:
        raise ValueError("GPU baseline prompts differ from sweep provenance")
    model = provenance.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("revision") != bundle.model_revision
        or model.get("tokenizer_revision") != bundle.tokenizer_revision
    ):
        raise ValueError("GPU baseline model differs from sweep provenance")
    return GPUBaselineWorkspaceBinding(
        manifest_hash=str(provenance.get("manifest_hash", "")),
        run_plan_hash=str(provenance.get("run_plan_hash", "")),
        prompt_manifest_hash=prompt_manifest_hash,
        sweep_provenance_sha256=_sha256_file(path),
    )


@dataclass(frozen=True)
class GPUBaselinePrompt:
    """One fixed prompt available to every measured device."""

    document_id: str
    prompt_hash: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("baseline document_id must be non-empty")
        _require_sha256(self.prompt_hash, "prompt_hash")
        _require_positive_int(self.token_count, "token_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "prompt_hash": self.prompt_hash,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUBaselinePrompt":
        return cls(
            document_id=str(value["document_id"]),
            prompt_hash=str(value["prompt_hash"]),
            token_count=int(value["token_count"]),
        )


@dataclass(frozen=True)
class GPUBaselineContract:
    """Immutable workload and execution policy shared across GPU runs."""

    model_name: str
    model_revision: str
    tokenizer_revision: str
    sample_bundle_hash: str
    workspace_binding: GPUBaselineWorkspaceBinding
    prompts: tuple[GPUBaselinePrompt, ...]
    attention_implementation: str
    warmup_steps: int
    measured_steps: int
    repetitions: int
    planned_device_labels: tuple[str, ...]
    planned_batch_sizes: tuple[int, ...]
    seed: int
    source_tree_sha256: str
    energy_meter_priority: tuple[str, ...] = GPU_BASELINE_ENERGY_METER_PRIORITY
    power_trace_sample_interval_ms: int = 10
    precision: str = "BF16"
    q_len: int = 1
    first_token_owner: str = "prefill"
    prefill_logits_to_keep: int = 1
    execution_mode: str = GPU_BASELINE_EXECUTION_MODE
    timing_scope: str = GPU_BASELINE_TIMING_SCOPE
    energy_scope: str = GPU_BASELINE_ENERGY_SCOPE
    require_mig_disabled: bool = True
    scope: str = GPU_BASELINE_SCOPE
    schema_version: str = GPU_BASELINE_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GPU_BASELINE_CONTRACT_SCHEMA:
            raise ValueError("unsupported GPU baseline contract schema")
        if not self.model_name or not self.model_revision:
            raise ValueError("baseline model identity must be pinned")
        if not self.tokenizer_revision:
            raise ValueError("baseline tokenizer revision must be pinned")
        _require_git_commit(self.model_revision, "model_revision")
        _require_git_commit(self.tokenizer_revision, "tokenizer_revision")
        _require_sha256(self.sample_bundle_hash, "sample_bundle_hash")
        if not isinstance(self.workspace_binding, GPUBaselineWorkspaceBinding):
            raise TypeError("GPU baseline requires an immutable workspace binding")
        _require_sha256(self.source_tree_sha256, "source_tree_sha256")
        if self.precision != "BF16":
            raise ValueError("the measured GPU reference must use BF16")
        if self.q_len != 1 or self.first_token_owner != "prefill":
            raise ValueError("baseline must use the split cached-q1 boundary")
        if self.scope != GPU_BASELINE_SCOPE:
            raise ValueError("unsupported GPU baseline scope")
        if self.attention_implementation != "sdpa":
            raise ValueError("GPU baseline attention implementation must be sdpa")
        if self.energy_meter_priority != GPU_BASELINE_ENERGY_METER_PRIORITY:
            raise ValueError("GPU baseline energy-meter priority is not canonical")
        if (
            isinstance(self.power_trace_sample_interval_ms, bool)
            or not isinstance(self.power_trace_sample_interval_ms, int)
            or not 1 <= self.power_trace_sample_interval_ms <= 1000
        ):
            raise ValueError("power-trace sample interval must be in [1, 1000] ms")
        _require_positive_int(self.warmup_steps, "warmup_steps")
        _require_positive_int(self.measured_steps, "measured_steps")
        _require_positive_int(self.repetitions, "repetitions")
        if self.repetitions < 3:
            raise ValueError("GPU baselines require at least three repetitions")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("GPU baseline seed must be an integer")
        if self.seed < 0:
            raise ValueError("GPU baseline seed must be non-negative")
        labels = tuple(self.planned_device_labels)
        if not labels or any(not _normalized_label(label) for label in labels):
            raise ValueError("GPU baseline requires planned device labels")
        normalized_labels = tuple(_normalized_label(label) for label in labels)
        if len(normalized_labels) != len(set(normalized_labels)):
            raise ValueError("planned GPU device labels must be unique")
        batches = tuple(self.planned_batch_sizes)
        if not batches:
            raise ValueError("GPU baseline requires planned batch sizes")
        for batch_size in batches:
            _require_positive_int(batch_size, "planned batch size")
        if batches != tuple(sorted(set(batches))):
            raise ValueError(
                "planned GPU batch sizes must be unique and increasing"
            )
        if not self.prompts:
            raise ValueError("GPU baseline contract requires prompts")
        ids = tuple(prompt.document_id for prompt in self.prompts)
        if len(ids) != len(set(ids)):
            raise ValueError("GPU baseline prompt IDs must be unique")
        lengths = {prompt.token_count for prompt in self.prompts}
        if len(lengths) != 1:
            raise ValueError("GPU baseline prompts must have one context length")
        if batches[-1] > len(self.prompts):
            raise ValueError(
                "planned GPU batch size exceeds the contracted prompt set"
            )
        if self.prefill_logits_to_keep != 1:
            raise ValueError("GPU prefill must materialize only final-position logits")
        if self.execution_mode != GPU_BASELINE_EXECUTION_MODE:
            raise ValueError("unsupported GPU baseline execution mode")
        if self.timing_scope != GPU_BASELINE_TIMING_SCOPE:
            raise ValueError("unsupported GPU baseline timing scope")
        if self.energy_scope != GPU_BASELINE_ENERGY_SCOPE:
            raise ValueError("unsupported GPU baseline energy scope")
        if self.require_mig_disabled is not True:
            raise ValueError("publication GPU baselines require a full GPU")

    @property
    def context_length(self) -> int:
        return self.prompts[0].token_count

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "sample_bundle_hash": self.sample_bundle_hash,
            "workspace_binding": self.workspace_binding.to_dict(),
            "prompts": [prompt.to_dict() for prompt in self.prompts],
            "attention_implementation": self.attention_implementation,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "repetitions": self.repetitions,
            "planned_device_labels": list(self.planned_device_labels),
            "planned_batch_sizes": list(self.planned_batch_sizes),
            "seed": self.seed,
            "source_tree_sha256": self.source_tree_sha256,
            "energy_meter_priority": list(self.energy_meter_priority),
            "power_trace_sample_interval_ms": self.power_trace_sample_interval_ms,
            "precision": self.precision,
            "q_len": self.q_len,
            "first_token_owner": self.first_token_owner,
            "prefill_logits_to_keep": self.prefill_logits_to_keep,
            "execution_mode": self.execution_mode,
            "timing_scope": self.timing_scope,
            "energy_scope": self.energy_scope,
            "require_mig_disabled": self.require_mig_disabled,
            "scope": self.scope,
        }

    @property
    def contract_hash(self) -> str:
        return _canonical_hash(self._body())

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "contract_hash": self.contract_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUBaselineContract":
        contract = cls(
            schema_version=str(value["schema_version"]),
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            sample_bundle_hash=str(value["sample_bundle_hash"]),
            workspace_binding=GPUBaselineWorkspaceBinding.from_dict(
                value["workspace_binding"]
            ),
            prompts=tuple(
                GPUBaselinePrompt.from_dict(item)
                for item in value["prompts"]
            ),
            attention_implementation=str(value["attention_implementation"]),
            warmup_steps=int(value["warmup_steps"]),
            measured_steps=int(value["measured_steps"]),
            repetitions=int(value["repetitions"]),
            planned_device_labels=tuple(
                str(label) for label in value["planned_device_labels"]
            ),
            planned_batch_sizes=tuple(
                int(batch_size) for batch_size in value["planned_batch_sizes"]
            ),
            seed=int(value["seed"]),
            source_tree_sha256=str(value["source_tree_sha256"]),
            energy_meter_priority=tuple(
                str(item) for item in value["energy_meter_priority"]
            ),
            power_trace_sample_interval_ms=int(
                value["power_trace_sample_interval_ms"]
            ),
            precision=str(value["precision"]),
            q_len=int(value["q_len"]),
            first_token_owner=str(value["first_token_owner"]),
            prefill_logits_to_keep=int(value["prefill_logits_to_keep"]),
            execution_mode=str(value["execution_mode"]),
            timing_scope=str(value["timing_scope"]),
            energy_scope=str(value["energy_scope"]),
            require_mig_disabled=value["require_mig_disabled"],
            scope=str(value["scope"]),
        )
        if value.get("contract_hash") != contract.contract_hash:
            raise ValueError("GPU baseline contract hash mismatch")
        return contract


def build_gpu_baseline_contract(
    config: Mapping[str, Any],
    bundle: TokenSampleBundle,
    workspace_binding: GPUBaselineWorkspaceBinding,
    *,
    attention_implementation: str,
    warmup_steps: int,
    measured_steps: int,
    repetitions: int,
    planned_device_labels: Sequence[str],
    planned_batch_sizes: Sequence[int],
    energy_meter_priority: Sequence[str],
    power_trace_sample_interval_ms: int,
) -> GPUBaselineContract:
    """Bind the hardware validation prompts and explicit BF16 execution policy."""

    baseline_policy = config.get("gpu_baseline")
    if not isinstance(baseline_policy, Mapping):
        raise ValueError("GPU baseline config requires an explicit policy")
    expected_policy = {
        "attention_implementation": attention_implementation,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "repetitions": repetitions,
        "batch_sizes": [int(value) for value in planned_batch_sizes],
        "precision": "BF16",
        "q_len": 1,
        "first_gpu_only": True,
        "energy_meter_priority": [str(value) for value in energy_meter_priority],
        "power_trace_sample_interval_ms": power_trace_sample_interval_ms,
    }
    if dict(baseline_policy) != expected_policy:
        raise ValueError("GPU baseline launch policy differs from the config")
    if bundle.model_revision != str(config["model_revision"]):
        raise ValueError("baseline sample model revision differs from config")
    if bundle.tokenizer_revision != str(config["tokenizer_revision"]):
        raise ValueError("baseline tokenizer revision differs from config")
    if workspace_binding.prompt_manifest_hash != bundle.prompt_manifest().canonical_hash:
        raise ValueError("baseline sample bundle differs from its workspace binding")
    if len(bundle.hardware_validation) != 64:
        raise ValueError(
            "GPU baseline requires the complete 64-prompt hardware-validation set"
        )
    if any(len(sample.prompt_token_ids) != 512 for sample in bundle.hardware_validation):
        raise ValueError(
            "GPU baseline requires 512-token hardware-validation prompts"
        )
    phase = dict(config.get("phase_contract", {}))
    if (
        str(config.get("dtype")) != "bfloat16"
        or bool(config.get("trust_remote_code", False))
        or phase.get("prefill_precision") != "BF16"
        or phase.get("decode_query_length") != 1
        or phase.get("first_token_owner") != "prefill"
    ):
        raise ValueError("GPU baseline config violates the BF16 cached-q1 boundary")
    prompts = tuple(
        GPUBaselinePrompt(
            document_id=sample.document_id,
            prompt_hash=sample.prompt_hash,
            token_count=len(sample.prompt_token_ids),
        )
        for sample in bundle.hardware_validation
    )
    return GPUBaselineContract(
        model_name=str(config["model_name"]),
        model_revision=bundle.model_revision,
        tokenizer_revision=bundle.tokenizer_revision,
        sample_bundle_hash=bundle.canonical_hash,
        workspace_binding=workspace_binding,
        prompts=prompts,
        attention_implementation=attention_implementation,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        repetitions=repetitions,
        planned_device_labels=tuple(
            str(value) for value in planned_device_labels
        ),
        planned_batch_sizes=tuple(int(value) for value in planned_batch_sizes),
        energy_meter_priority=tuple(str(value) for value in energy_meter_priority),
        power_trace_sample_interval_ms=power_trace_sample_interval_ms,
        seed=int(config.get("seed", 0)),
        source_tree_sha256=_source_tree_hash(),
    )


@dataclass(frozen=True)
class GPUHardwareStateSnapshot:
    """Raw NVIDIA device state captured at one benchmark boundary."""

    phase: str
    captured_at_utc: str
    raw_query_line: str
    raw_query_sha256: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        repetition_phase = self.phase.split("_")
        valid_repetition_phase = (
            len(repetition_phase) == 3
            and repetition_phase[0] == "repetition"
            and repetition_phase[1].isdigit()
            and repetition_phase[2] in {"start", "end"}
        )
        if (
            self.phase != "run_start"
            and self.phase != "run_end"
            and not valid_repetition_phase
        ):
            raise ValueError("unsupported GPU hardware-state phase")
        _require_utc_timestamp(self.captured_at_utc, "captured_at_utc")
        expected_hash = hashlib.sha256(
            self.raw_query_line.encode("utf-8")
        ).hexdigest()
        if self.raw_query_sha256 != expected_hash:
            raise ValueError("GPU hardware-state raw-query hash mismatch")
        parsed = tuple(
            field.strip()
            for field in next(csv.reader([self.raw_query_line]))
        )
        if parsed != self.values:
            raise ValueError("GPU hardware-state values differ from raw query")
        if len(self.values) != len(NVIDIA_SMI_QUERY_FIELDS):
            raise ValueError("GPU hardware-state query has missing fields")
        if any(not value for value in self.values):
            raise ValueError("GPU hardware-state fields must be non-empty")

    @property
    def fields(self) -> Mapping[str, str]:
        return dict(zip(NVIDIA_SMI_QUERY_FIELDS, self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "captured_at_utc": self.captured_at_utc,
            "query_fields": list(NVIDIA_SMI_QUERY_FIELDS),
            "raw_query_line": self.raw_query_line,
            "raw_query_sha256": self.raw_query_sha256,
            "values": list(self.values),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "GPUHardwareStateSnapshot":
        if tuple(value["query_fields"]) != NVIDIA_SMI_QUERY_FIELDS:
            raise ValueError("GPU hardware-state query fields differ")
        return cls(
            phase=str(value["phase"]),
            captured_at_utc=str(value["captured_at_utc"]),
            raw_query_line=str(value["raw_query_line"]),
            raw_query_sha256=str(value["raw_query_sha256"]),
            values=tuple(str(item) for item in value["values"]),
        )


@dataclass(frozen=True)
class GPUPowerTraceSample:
    """One raw NVML board-power sample on the monotonic host clock."""

    timestamp_monotonic_ns: int
    power_mw: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_monotonic_ns, bool)
            or not isinstance(self.timestamp_monotonic_ns, int)
            or self.timestamp_monotonic_ns <= 0
        ):
            raise ValueError("power-trace timestamp must be a positive integer")
        if (
            isinstance(self.power_mw, bool)
            or not isinstance(self.power_mw, int)
            or self.power_mw <= 0
        ):
            raise ValueError("NVML board-power samples must be positive milliwatts")

    def to_dict(self) -> dict[str, int]:
        return {
            "timestamp_monotonic_ns": self.timestamp_monotonic_ns,
            "power_mw": self.power_mw,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUPowerTraceSample":
        return cls(
            timestamp_monotonic_ns=int(value["timestamp_monotonic_ns"]),
            power_mw=int(value["power_mw"]),
        )


def _integrate_power_trace(samples: Sequence[GPUPowerTraceSample]) -> float:
    if len(samples) < 2:
        raise ValueError("power-trace integration requires at least two samples")
    timestamps = tuple(sample.timestamp_monotonic_ns for sample in samples)
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("power-trace timestamps must increase strictly")
    energy_j = 0.0
    for left, right in zip(samples, samples[1:]):
        duration_s = (
            right.timestamp_monotonic_ns - left.timestamp_monotonic_ns
        ) / 1_000_000_000.0
        mean_power_w = (left.power_mw + right.power_mw) / 2000.0
        energy_j += mean_power_w * duration_s
    if not math.isfinite(energy_j) or energy_j <= 0:
        raise ValueError("integrated NVML board energy must be positive")
    return energy_j


@dataclass(frozen=True)
class GPUDeviceEnergyMeasurement:
    """Raw board-energy evidence for one device and measured decode region."""

    device_uuid: str
    meter_method: str
    measurement_status: str
    started_at_monotonic_ns: int
    ended_at_monotonic_ns: int
    energy_j: float | None
    counter_start_mj: int | None
    counter_end_mj: int | None
    power_trace: tuple[GPUPowerTraceSample, ...]
    requested_power_sample_interval_ms: int
    unavailable_reason: str | None
    scope: str = GPU_BASELINE_ENERGY_SCOPE
    schema_version: str = GPU_ENERGY_MEASUREMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GPU_ENERGY_MEASUREMENT_SCHEMA:
            raise ValueError("unsupported GPU energy-measurement schema")
        if not self.device_uuid or self.device_uuid.lower() == "unavailable":
            raise ValueError("GPU energy measurement requires a device UUID")
        if self.meter_method not in {
            *GPU_BASELINE_ENERGY_METER_PRIORITY,
            ENERGY_UNAVAILABLE_METHOD,
        }:
            raise ValueError("unsupported GPU board-energy meter")
        if self.measurement_status not in {"measured", "unavailable"}:
            raise ValueError("unsupported GPU energy-measurement status")
        if self.scope != GPU_BASELINE_ENERGY_SCOPE:
            raise ValueError("unsupported GPU energy-measurement scope")
        if (
            isinstance(self.started_at_monotonic_ns, bool)
            or isinstance(self.ended_at_monotonic_ns, bool)
            or not isinstance(self.started_at_monotonic_ns, int)
            or not isinstance(self.ended_at_monotonic_ns, int)
            or self.started_at_monotonic_ns <= 0
            or self.ended_at_monotonic_ns <= self.started_at_monotonic_ns
        ):
            raise ValueError("GPU energy interval must increase on the monotonic clock")
        if (
            isinstance(self.requested_power_sample_interval_ms, bool)
            or not isinstance(self.requested_power_sample_interval_ms, int)
            or not 1 <= self.requested_power_sample_interval_ms <= 1000
        ):
            raise ValueError("power sample interval must be in [1, 1000] ms")
        if self.measurement_status == "unavailable":
            if self.energy_j is not None or not self.unavailable_reason:
                raise ValueError("unavailable energy requires a reason and no joule claim")
            if (
                self.counter_start_mj is not None
                or self.counter_end_mj is not None
                or self.power_trace
            ):
                raise ValueError("unavailable energy cannot retain partial measurements")
            return
        if self.unavailable_reason is not None or self.energy_j is None:
            raise ValueError("measured energy requires joules and no unavailable reason")
        energy_j = _require_nonnegative_finite(self.energy_j, "energy_j")
        if energy_j <= 0:
            raise ValueError("measured board energy must be positive")
        if self.meter_method == NVML_TOTAL_ENERGY_METHOD:
            if (
                isinstance(self.counter_start_mj, bool)
                or isinstance(self.counter_end_mj, bool)
                or not isinstance(self.counter_start_mj, int)
                or not isinstance(self.counter_end_mj, int)
                or self.counter_start_mj < 0
                or self.counter_end_mj <= self.counter_start_mj
                or self.power_trace
            ):
                raise ValueError("invalid cumulative NVML energy-counter evidence")
            expected_j = (self.counter_end_mj - self.counter_start_mj) / 1000.0
        elif self.meter_method == NVML_POWER_TRACE_METHOD:
            if self.counter_start_mj is not None or self.counter_end_mj is not None:
                raise ValueError("NVML power traces cannot contain energy counters")
            expected_j = _integrate_power_trace(self.power_trace)
            if (
                self.power_trace[0].timestamp_monotonic_ns
                < self.started_at_monotonic_ns
                or self.power_trace[-1].timestamp_monotonic_ns
                > self.ended_at_monotonic_ns
            ):
                raise ValueError("NVML power samples fall outside the measured interval")
        else:
            raise ValueError("unavailable meter cannot claim measured energy")
        if not math.isclose(energy_j, expected_j, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("board-energy joules differ from raw NVML evidence")
        object.__setattr__(self, "energy_j", energy_j)

    @property
    def duration_ms(self) -> float:
        return (
            self.ended_at_monotonic_ns - self.started_at_monotonic_ns
        ) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device_uuid": self.device_uuid,
            "meter_method": self.meter_method,
            "measurement_status": self.measurement_status,
            "scope": self.scope,
            "started_at_monotonic_ns": self.started_at_monotonic_ns,
            "ended_at_monotonic_ns": self.ended_at_monotonic_ns,
            "duration_ms": self.duration_ms,
            "energy_j": self.energy_j,
            "counter_start_mj": self.counter_start_mj,
            "counter_end_mj": self.counter_end_mj,
            "power_trace": [sample.to_dict() for sample in self.power_trace],
            "power_trace_sample_count": len(self.power_trace),
            "requested_power_sample_interval_ms": (
                self.requested_power_sample_interval_ms
            ),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "GPUDeviceEnergyMeasurement":
        measurement = cls(
            schema_version=str(value["schema_version"]),
            device_uuid=str(value["device_uuid"]),
            meter_method=str(value["meter_method"]),
            measurement_status=str(value["measurement_status"]),
            scope=str(value["scope"]),
            started_at_monotonic_ns=int(value["started_at_monotonic_ns"]),
            ended_at_monotonic_ns=int(value["ended_at_monotonic_ns"]),
            energy_j=(
                None if value.get("energy_j") is None else float(value["energy_j"])
            ),
            counter_start_mj=(
                None
                if value.get("counter_start_mj") is None
                else int(value["counter_start_mj"])
            ),
            counter_end_mj=(
                None
                if value.get("counter_end_mj") is None
                else int(value["counter_end_mj"])
            ),
            power_trace=tuple(
                GPUPowerTraceSample.from_dict(item)
                for item in value.get("power_trace", ())
            ),
            requested_power_sample_interval_ms=int(
                value["requested_power_sample_interval_ms"]
            ),
            unavailable_reason=(
                None
                if value.get("unavailable_reason") is None
                else str(value["unavailable_reason"])
            ),
        )
        if value.get("duration_ms") != measurement.duration_ms:
            raise ValueError("GPU energy duration differs from its raw timestamps")
        if value.get("power_trace_sample_count") != len(measurement.power_trace):
            raise ValueError("GPU energy trace sample count mismatch")
        return measurement


@dataclass(frozen=True)
class GPUEnergyMeasurement:
    """Aggregate measured-region energy, retaining every device contribution."""

    repetition: int
    generated_tokens: int
    devices: tuple[GPUDeviceEnergyMeasurement, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.repetition, bool)
            or not isinstance(self.repetition, int)
            or self.repetition < 0
        ):
            raise ValueError("energy repetition index must be non-negative")
        _require_positive_int(self.generated_tokens, "generated_tokens")
        if not self.devices:
            raise ValueError("GPU energy measurement requires selected devices")
        uuids = tuple(device.device_uuid for device in self.devices)
        if len(uuids) != len(set(uuids)):
            raise ValueError("GPU energy measurement repeats a device UUID")
        if len({device.meter_method for device in self.devices}) != 1:
            raise ValueError("one energy row cannot mix NVML meter methods")
        if len({device.measurement_status for device in self.devices}) != 1:
            raise ValueError("one energy row cannot mix availability states")

    @property
    def available(self) -> bool:
        return all(
            device.measurement_status == "measured" for device in self.devices
        )

    @property
    def meter_method(self) -> str:
        return self.devices[0].meter_method

    @property
    def total_energy_j(self) -> float | None:
        if not self.available:
            return None
        return sum(float(device.energy_j) for device in self.devices)

    @property
    def energy_per_token_j(self) -> float | None:
        energy = self.total_energy_j
        return None if energy is None else energy / self.generated_tokens

    @property
    def tokens_per_joule(self) -> float | None:
        energy = self.total_energy_j
        return None if energy is None else self.generated_tokens / energy

    @property
    def unavailable_reason(self) -> str | None:
        if self.available:
            return None
        return "; ".join(
            f"{device.device_uuid}: {device.unavailable_reason}"
            for device in self.devices
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repetition": self.repetition,
            "generated_tokens": self.generated_tokens,
            "devices": [device.to_dict() for device in self.devices],
            "available": self.available,
            "meter_method": self.meter_method,
            "total_energy_j": self.total_energy_j,
            "energy_per_token_j": self.energy_per_token_j,
            "tokens_per_joule": self.tokens_per_joule,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUEnergyMeasurement":
        measurement = cls(
            repetition=int(value["repetition"]),
            generated_tokens=int(value["generated_tokens"]),
            devices=tuple(
                GPUDeviceEnergyMeasurement.from_dict(item)
                for item in value["devices"]
            ),
        )
        if measurement.to_dict() != dict(value):
            raise ValueError("aggregate GPU energy summary mismatch")
        return measurement


@dataclass(frozen=True)
class GPUBaselineRepetition:
    """Raw timing samples and execution invariants for one fresh prefill."""

    repetition: int
    document_ids: tuple[str, ...]
    prefill_ms: float
    decode_step_ms: tuple[float, ...]
    q_len_one_calls: int
    cache_growth_checks: int
    first_token_count: int
    generated_token_sha256: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    energy_measurement: GPUEnergyMeasurement

    def __post_init__(self) -> None:
        if self.repetition < 0:
            raise ValueError("repetition index must be non-negative")
        if not self.document_ids or len(self.document_ids) != len(
            set(self.document_ids)
        ):
            raise ValueError("repetition document IDs must be unique")
        _require_nonnegative_finite(self.prefill_ms, "prefill_ms")
        if not self.decode_step_ms:
            raise ValueError("repetition requires raw decode timings")
        for value in self.decode_step_ms:
            _require_nonnegative_finite(value, "decode_step_ms")
            if value == 0:
                raise ValueError("decode timing samples must be positive")
        _require_positive_int(self.q_len_one_calls, "q_len_one_calls")
        _require_positive_int(self.cache_growth_checks, "cache_growth_checks")
        if self.q_len_one_calls != self.cache_growth_checks:
            raise ValueError("every cached-q1 call requires a growth check")
        if self.first_token_count != len(self.document_ids):
            raise ValueError("prefill must produce one first token per sequence")
        _require_sha256(
            self.generated_token_sha256,
            "generated_token_sha256",
        )
        if self.peak_allocated_bytes <= 0 or self.peak_reserved_bytes <= 0:
            raise ValueError("successful runs require measured CUDA memory")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("reserved CUDA memory cannot be below allocated")
        if self.energy_measurement.repetition != self.repetition:
            raise ValueError("energy repetition index differs from timing evidence")
        if self.energy_measurement.generated_tokens != (
            len(self.document_ids) * len(self.decode_step_ms)
        ):
            raise ValueError("energy token count differs from measured decode timing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repetition": self.repetition,
            "document_ids": list(self.document_ids),
            "prefill_ms": self.prefill_ms,
            "decode_step_ms": list(self.decode_step_ms),
            "q_len_one_calls": self.q_len_one_calls,
            "cache_growth_checks": self.cache_growth_checks,
            "first_token_count": self.first_token_count,
            "generated_token_sha256": self.generated_token_sha256,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "energy_measurement": self.energy_measurement.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUBaselineRepetition":
        return cls(
            repetition=int(value["repetition"]),
            document_ids=tuple(map(str, value["document_ids"])),
            prefill_ms=float(value["prefill_ms"]),
            decode_step_ms=tuple(map(float, value["decode_step_ms"])),
            q_len_one_calls=int(value["q_len_one_calls"]),
            cache_growth_checks=int(value["cache_growth_checks"]),
            first_token_count=int(value["first_token_count"]),
            generated_token_sha256=str(value["generated_token_sha256"]),
            peak_allocated_bytes=int(value["peak_allocated_bytes"]),
            peak_reserved_bytes=int(value["peak_reserved_bytes"]),
            energy_measurement=GPUEnergyMeasurement.from_dict(
                value["energy_measurement"]
            ),
        )


@dataclass(frozen=True)
class GPUBaselineResult:
    """One terminal batch-size measurement; failures remain explicit."""

    contract_hash: str
    device_label: str
    device_name: str
    device_uuid: str
    runtime_environment: RuntimeEnvironment
    batch_size: int
    state: str
    repetitions: tuple[GPUBaselineRepetition, ...]
    hardware_state_snapshots: tuple[GPUHardwareStateSnapshot, ...]
    error_class: str | None
    error_message: str | None
    created_at_utc: str
    schema_version: str = GPU_BASELINE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != GPU_BASELINE_RESULT_SCHEMA:
            raise ValueError("unsupported GPU baseline result schema")
        _require_sha256(self.contract_hash, "contract_hash")
        if not self.device_label or not self.device_name or not self.device_uuid:
            raise ValueError("GPU baseline device identity must be complete")
        _require_positive_int(self.batch_size, "batch_size")
        if self.state not in {"succeeded", "oom", "failed"}:
            raise ValueError("unsupported GPU baseline terminal state")
        _require_utc_timestamp(self.created_at_utc, "created_at_utc")
        observed = self.runtime_environment.observation
        logical = self.runtime_environment.logical
        if str(logical.get("device_name")) != self.device_name:
            raise ValueError("result device name differs from runtime receipt")
        if str(observed.get("device_uuid")) != self.device_uuid:
            raise ValueError("result device UUID differs from runtime receipt")
        if self.device_uuid.lower() == "unavailable":
            raise ValueError("GPU baseline requires a physical device UUID")
        for snapshot in self.hardware_state_snapshots:
            fields = snapshot.fields
            if fields["uuid"] != self.device_uuid:
                raise ValueError(
                    "GPU hardware-state UUID differs from runtime receipt"
                )
        if self.state == "succeeded":
            if len(self.repetitions) < 3:
                raise ValueError("successful GPU result requires three repeats")
            if self.error_class is not None or self.error_message is not None:
                raise ValueError("successful GPU result cannot contain an error")
            indices = tuple(item.repetition for item in self.repetitions)
            if indices != tuple(range(len(self.repetitions))):
                raise ValueError("GPU repetition indices must be contiguous")
            if any(
                len(item.document_ids) != self.batch_size
                for item in self.repetitions
            ):
                raise ValueError("GPU repetition batch size mismatch")
            energy_measurements = tuple(
                item.energy_measurement for item in self.repetitions
            )
            if any(
                tuple(device.device_uuid for device in measurement.devices)
                != (self.device_uuid,)
                for measurement in energy_measurements
            ):
                raise ValueError("GPU energy evidence differs from the selected device")
            if len(
                {measurement.meter_method for measurement in energy_measurements}
            ) != 1:
                raise ValueError("one GPU result cannot mix energy-meter methods")
            lengths = {len(item.decode_step_ms) for item in self.repetitions}
            if len(lengths) != 1:
                raise ValueError("GPU repetitions have different sample counts")
            expected_phases = ["run_start"]
            for repetition in range(len(self.repetitions)):
                expected_phases.extend(
                    (
                        f"repetition_{repetition}_start",
                        f"repetition_{repetition}_end",
                    )
                )
            expected_phases.append("run_end")
            observed_phases = [
                snapshot.phase
                for snapshot in self.hardware_state_snapshots
            ]
            if observed_phases != expected_phases:
                raise ValueError(
                    "successful GPU result lacks hardware-state coverage"
                )
            if any(
                snapshot.fields["mig.mode.current"].lower() != "disabled"
                for snapshot in self.hardware_state_snapshots
            ):
                raise ValueError("GPU baseline requires MIG to be disabled")
            required_fields = (
                "driver_version",
                "pstate",
                "temperature.gpu",
                "power.draw",
                "power.limit",
                "clocks.current.graphics",
                "clocks.current.sm",
                "clocks.current.memory",
            )
            if any(
                snapshot.fields[field].lower() in {"n/a", "[n/a]"}
                for snapshot in self.hardware_state_snapshots
                for field in required_fields
            ):
                raise ValueError(
                    "successful GPU result has unavailable hardware state"
                )
            numeric_fields = (
                "memory.total",
                "memory.used",
                "temperature.gpu",
                "power.draw",
                "power.limit",
                "clocks.current.graphics",
                "clocks.current.sm",
                "clocks.current.memory",
            )
            for snapshot in self.hardware_state_snapshots:
                for field in numeric_fields:
                    _require_nonnegative_finite(
                        float(snapshot.fields[field]),
                        f"hardware state {field}",
                    )
            if len(
                {
                    snapshot.fields["driver_version"]
                    for snapshot in self.hardware_state_snapshots
                }
            ) != 1:
                raise ValueError("GPU driver changed during the baseline")
            stable_fields = (
                "driver_version",
                "name",
                "uuid",
                "pci.bus_id",
                "memory.total",
                "power.limit",
                "mig.mode.current",
            )
            for field in stable_fields:
                if len(
                    {
                        snapshot.fields[field]
                        for snapshot in self.hardware_state_snapshots
                    }
                ) != 1:
                    raise ValueError(
                        f"GPU hardware field {field} changed during the baseline"
                    )
        else:
            if self.repetitions:
                raise ValueError("failed GPU result cannot claim repetitions")
            if not self.error_class or not self.error_message:
                raise ValueError("failed GPU result requires an error receipt")

    @property
    def result_id(self) -> str:
        return f"gpu-baseline-{_canonical_hash(self._body())}"

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "device_label": self.device_label,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "runtime_environment": self.runtime_environment.to_dict(),
            "batch_size": self.batch_size,
            "state": self.state,
            "repetitions": [
                repetition.to_dict() for repetition in self.repetitions
            ],
            "hardware_state_snapshots": [
                snapshot.to_dict()
                for snapshot in self.hardware_state_snapshots
            ],
            "error_class": self.error_class,
            "error_message": self.error_message,
            "created_at_utc": self.created_at_utc,
        }

    @property
    def summary(self) -> Mapping[str, Any] | None:
        if self.state != "succeeded":
            return None
        samples = tuple(
            sample
            for repetition in self.repetitions
            for sample in repetition.decode_step_ms
        )
        mean_ms = statistics.fmean(samples)
        repetition_mean_ms = tuple(
            statistics.fmean(repetition.decode_step_ms)
            for repetition in self.repetitions
        )
        energy_measurements = tuple(
            repetition.energy_measurement for repetition in self.repetitions
        )
        energy_available = all(
            measurement.available for measurement in energy_measurements
        )
        energy_j = (
            sum(
                float(measurement.total_energy_j)
                for measurement in energy_measurements
            )
            if energy_available
            else None
        )
        generated_tokens = sum(
            measurement.generated_tokens for measurement in energy_measurements
        )
        energy_per_token_j = (
            energy_j / generated_tokens if energy_j is not None else None
        )
        tokens_per_joule = (
            generated_tokens / energy_j if energy_j is not None else None
        )
        energy_delay_product_j_s = (
            energy_per_token_j * mean_ms / 1000.0
            if energy_per_token_j is not None
            else None
        )
        hardware_fields = tuple(
            snapshot.fields for snapshot in self.hardware_state_snapshots
        )
        return {
            "raw_decode_step_sample_count": len(samples),
            "mean_batch_step_ms": mean_ms,
            "median_batch_step_ms": statistics.median(samples),
            "p95_batch_step_ms": _percentile(samples, 0.95),
            "tokens_per_second": self.batch_size * 1000.0 / mean_ms,
            "mean_batch_step_ms_by_repetition": list(
                repetition_mean_ms
            ),
            "tokens_per_second_by_repetition": [
                self.batch_size * 1000.0 / value
                for value in repetition_mean_ms
            ],
            "repeat_mean_step_cv": (
                statistics.pstdev(repetition_mean_ms)
                / statistics.fmean(repetition_mean_ms)
            ),
            "mean_prefill_ms": statistics.fmean(
                repetition.prefill_ms for repetition in self.repetitions
            ),
            "energy": {
                "available": energy_available,
                "scope": GPU_BASELINE_ENERGY_SCOPE,
                "meter_method": energy_measurements[0].meter_method,
                "device_uuids": [self.device_uuid],
                "measured_repetitions": (
                    len(energy_measurements) if energy_available else 0
                ),
                "generated_tokens": generated_tokens,
                "total_energy_j": energy_j,
                "energy_per_token_j": energy_per_token_j,
                "tokens_per_joule": tokens_per_joule,
                "energy_delay_product_j_s": energy_delay_product_j_s,
                "unavailable_reason": (
                    None
                    if energy_available
                    else "; ".join(
                        str(measurement.unavailable_reason)
                        for measurement in energy_measurements
                        if not measurement.available
                    )
                ),
            },
            "max_peak_allocated_bytes": max(
                repetition.peak_allocated_bytes
                for repetition in self.repetitions
            ),
            "max_peak_reserved_bytes": max(
                repetition.peak_reserved_bytes
                for repetition in self.repetitions
            ),
            "hardware_state": {
                "driver_version": hardware_fields[0]["driver_version"],
                "mig_mode": hardware_fields[0]["mig.mode.current"],
                "observed_pstates": sorted(
                    {fields["pstate"] for fields in hardware_fields}
                ),
                "sm_clock_mhz_min": min(
                    float(fields["clocks.current.sm"])
                    for fields in hardware_fields
                ),
                "sm_clock_mhz_max": max(
                    float(fields["clocks.current.sm"])
                    for fields in hardware_fields
                ),
                "memory_clock_mhz_min": min(
                    float(fields["clocks.current.memory"])
                    for fields in hardware_fields
                ),
                "memory_clock_mhz_max": max(
                    float(fields["clocks.current.memory"])
                    for fields in hardware_fields
                ),
                "power_draw_w_max": max(
                    float(fields["power.draw"])
                    for fields in hardware_fields
                ),
                "power_limit_w": float(
                    hardware_fields[0]["power.limit"]
                ),
                "temperature_c_max": max(
                    float(fields["temperature.gpu"])
                    for fields in hardware_fields
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "result_id": self.result_id,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUBaselineResult":
        environment_value = dict(value["runtime_environment"])
        result = cls(
            schema_version=str(value["schema_version"]),
            contract_hash=str(value["contract_hash"]),
            device_label=str(value["device_label"]),
            device_name=str(value["device_name"]),
            device_uuid=str(value["device_uuid"]),
            runtime_environment=RuntimeEnvironment.from_dict(
                environment_value
            ),
            batch_size=int(value["batch_size"]),
            state=str(value["state"]),
            repetitions=tuple(
                GPUBaselineRepetition.from_dict(item)
                for item in value["repetitions"]
            ),
            hardware_state_snapshots=tuple(
                GPUHardwareStateSnapshot.from_dict(item)
                for item in value["hardware_state_snapshots"]
            ),
            error_class=(
                None
                if value.get("error_class") is None
                else str(value["error_class"])
            ),
            error_message=(
                None
                if value.get("error_message") is None
                else str(value["error_message"])
            ),
            created_at_utc=str(value["created_at_utc"]),
        )
        if value.get("result_id") != result.result_id:
            raise ValueError("GPU baseline result identity mismatch")
        if value.get("summary") != result.summary:
            raise ValueError("GPU baseline summary mismatch")
        return result


def _cache_length(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        value = cache.get_seq_length()
        return int(value.item() if hasattr(value, "item") else value)
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "key_cache"):
        cache = tuple(zip(cache.key_cache, cache.value_cache))
    layers = tuple(cache)
    if not layers:
        raise ValueError("model returned an empty decode cache")
    return int(layers[0][0].shape[-2])


def _cache_layers(cache: Any) -> tuple[tuple[Any, Any], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache = tuple(zip(cache.key_cache, cache.value_cache))
    layers = tuple(cache)
    if not layers or any(len(layer) < 2 for layer in layers):
        raise ValueError("model returned an invalid decode cache")
    return tuple((layer[0], layer[1]) for layer in layers)


def _assert_bf16_cache(cache: Any, torch: Any) -> None:
    for key, value in _cache_layers(cache):
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise TypeError("GPU baseline cache must remain BF16")


def _timed_cuda_call(
    torch: Any,
    callable_: Any,
    *,
    device: str,
) -> tuple[Any, float]:
    with torch.cuda.device(torch.device(device)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        value = callable_()
        end.record()
        end.synchronize()
    elapsed = float(start.elapsed_time(end))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("CUDA event produced an invalid timing")
    return value, elapsed


def _select_repeat_samples(
    bundle: TokenSampleBundle,
    *,
    repetition: int,
    batch_size: int,
) -> tuple[Any, ...]:
    samples = tuple(bundle.hardware_validation)
    if batch_size > len(samples):
        raise ValueError("batch size exceeds the fixed hardware validation prompt set")
    start = (repetition * batch_size) % len(samples)
    indices = tuple((start + offset) % len(samples) for offset in range(batch_size))
    if len(set(indices)) != len(indices):
        raise AssertionError("baseline repeat selected a duplicate prompt")
    return tuple(samples[index] for index in indices)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _cuda_driver_version(torch: Any) -> str:
    candidates = (
        getattr(torch.cuda, "driver_version", None),
        getattr(torch._C, "_cuda_getDriverVersion", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = candidate() if callable(candidate) else candidate
        except (RuntimeError, TypeError):
            continue
        if value is not None:
            return str(value)
    return "unavailable"


def _capture_baseline_runtime(
    device: str,
    *,
    seed: int,
    attention_implementation: str,
    torch: Any,
) -> RuntimeEnvironment:
    base = capture_runtime_environment(device, seed=seed)
    logical = dict(base.logical)
    logical.update(
        {
            "attention_implementation": attention_implementation,
            "execution_mode": GPU_BASELINE_EXECUTION_MODE,
            "timing_scope": GPU_BASELINE_TIMING_SCOPE,
            "cuda_driver_version": _cuda_driver_version(torch),
            "accelerate": _package_version("accelerate"),
            "safetensors": _package_version("safetensors"),
            "tokenizers": _package_version("tokenizers"),
            "nvidia_ml_py": _package_version("nvidia-ml-py"),
            "flash_attn": (
                _package_version("flash_attn")
                if attention_implementation == "flash_attention_2"
                else "not_applicable"
            ),
        }
    )
    return RuntimeEnvironment(
        logical=logical,
        observation=dict(base.observation),
    )


def _capture_hardware_state(
    *,
    device_uuid: str,
    phase: str,
) -> GPUHardwareStateSnapshot:
    command = (
        "nvidia-smi",
        f"--id={device_uuid}",
        f"--query-gpu={','.join(NVIDIA_SMI_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("failed to capture NVIDIA hardware state") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "nvidia-smi query failed"
        raise RuntimeError(detail)
    lines = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    if len(lines) != 1:
        raise RuntimeError(
            "NVIDIA hardware-state query did not identify one physical GPU"
        )
    raw_line = lines[0]
    values = tuple(field.strip() for field in next(csv.reader([raw_line])))
    captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return GPUHardwareStateSnapshot(
        phase=phase,
        captured_at_utc=captured_at,
        raw_query_line=raw_line,
        raw_query_sha256=hashlib.sha256(
            raw_line.encode("utf-8")
        ).hexdigest(),
        values=values,
    )


def _meter_error_text(error: BaseException) -> str:
    detail = " ".join(str(error).split())
    return type(error).__name__ + (f": {detail}" if detail else "")


class _NVMLBoardEnergyMeter:
    """Select and operate one provenance-recorded NVML board-energy meter."""

    def __init__(
        self,
        *,
        device_uuid: str,
        method_priority: Sequence[str],
        power_trace_sample_interval_ms: int,
    ) -> None:
        if tuple(method_priority) != GPU_BASELINE_ENERGY_METER_PRIORITY:
            raise ValueError("GPU energy-meter priority differs from the contract")
        if not 1 <= power_trace_sample_interval_ms <= 1000:
            raise ValueError("power-trace sample interval must be in [1, 1000] ms")
        self.device_uuid = device_uuid
        self.power_trace_sample_interval_ms = power_trace_sample_interval_ms
        self.method = ENERGY_UNAVAILABLE_METHOD
        self.unavailable_reason = "NVML board-energy meter was not initialized"
        self._module: Any | None = None
        self._handle: Any | None = None
        self._initialized = False
        self._active = False
        self._started_at_ns: int | None = None
        self._counter_start_mj: int | None = None
        self._trace_samples: list[GPUPowerTraceSample] = []
        self._trace_error: str | None = None
        self._trace_stop: threading.Event | None = None
        self._trace_thread: threading.Thread | None = None
        self._initialize(method_priority)

    def _initialize(self, method_priority: Sequence[str]) -> None:
        failures: list[str] = []
        try:
            module = importlib.import_module("pynvml")
            module.nvmlInit()
            self._initialized = True
            self._module = module
            handle = module.nvmlDeviceGetHandleByUUID(self.device_uuid)
            observed_uuid = module.nvmlDeviceGetUUID(handle)
            if isinstance(observed_uuid, bytes):
                observed_uuid = observed_uuid.decode("utf-8")
            if str(observed_uuid).lower() != self.device_uuid.lower():
                raise RuntimeError("NVML handle resolved to a different GPU UUID")
            self._handle = handle
        except Exception as error:
            self.unavailable_reason = (
                "NVML initialization or UUID binding failed: "
                + _meter_error_text(error)
            )
            return
        for method in method_priority:
            try:
                if method == NVML_TOTAL_ENERGY_METHOD:
                    int(self._module.nvmlDeviceGetTotalEnergyConsumption(self._handle))
                elif method == NVML_POWER_TRACE_METHOD:
                    power_mw = int(self._module.nvmlDeviceGetPowerUsage(self._handle))
                    if power_mw <= 0:
                        raise RuntimeError("NVML returned non-positive board power")
                else:
                    raise ValueError(f"unsupported energy meter {method!r}")
            except Exception as error:
                failures.append(f"{method}: {_meter_error_text(error)}")
                continue
            self.method = method
            self.unavailable_reason = None
            return
        self.unavailable_reason = "no supported NVML board-energy meter; " + "; ".join(
            failures
        )

    def _read_power_sample(self) -> GPUPowerTraceSample:
        power_mw = int(self._module.nvmlDeviceGetPowerUsage(self._handle))
        return GPUPowerTraceSample(
            timestamp_monotonic_ns=time.monotonic_ns(),
            power_mw=power_mw,
        )

    def _trace_worker(self) -> None:
        if self._trace_stop is None:
            return
        interval_s = self.power_trace_sample_interval_ms / 1000.0
        while not self._trace_stop.wait(interval_s):
            try:
                self._trace_samples.append(self._read_power_sample())
            except Exception as error:
                self._trace_error = _meter_error_text(error)
                self._trace_stop.set()
                return

    def begin(self, torch: Any, *, device: str) -> None:
        if self._active:
            raise RuntimeError("GPU energy meter already has an active interval")
        torch.cuda.synchronize(torch.device(device))
        self._active = True
        self._counter_start_mj = None
        self._trace_samples = []
        self._trace_error = None
        self._trace_stop = None
        self._trace_thread = None
        if self.method == NVML_TOTAL_ENERGY_METHOD:
            try:
                self._counter_start_mj = int(
                    self._module.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                )
                self._started_at_ns = time.monotonic_ns()
            except Exception as error:
                self.method = ENERGY_UNAVAILABLE_METHOD
                self.unavailable_reason = (
                    "NVML cumulative energy counter failed at region start: "
                    + _meter_error_text(error)
                )
                self._started_at_ns = time.monotonic_ns()
        elif self.method == NVML_POWER_TRACE_METHOD:
            try:
                initial = self._read_power_sample()
                self._trace_samples.append(initial)
                self._started_at_ns = initial.timestamp_monotonic_ns
                self._trace_stop = threading.Event()
                self._trace_thread = threading.Thread(
                    target=self._trace_worker,
                    name="gpu-board-power-trace",
                    daemon=True,
                )
                self._trace_thread.start()
            except Exception as error:
                self.method = ENERGY_UNAVAILABLE_METHOD
                self.unavailable_reason = (
                    "NVML power trace failed at region start: "
                    + _meter_error_text(error)
                )
                self._trace_samples = []
                self._started_at_ns = time.monotonic_ns()
        else:
            self._started_at_ns = time.monotonic_ns()

    def _stop_trace(self) -> None:
        if self._trace_stop is not None:
            self._trace_stop.set()
        if self._trace_thread is not None:
            self._trace_thread.join(timeout=5.0)
            if self._trace_thread.is_alive():
                self._trace_error = "power-trace sampler did not terminate"
        self._trace_stop = None
        self._trace_thread = None

    def end(
        self,
        torch: Any,
        *,
        device: str,
        repetition: int,
        generated_tokens: int,
    ) -> GPUEnergyMeasurement:
        if not self._active or self._started_at_ns is None:
            raise RuntimeError("GPU energy meter has no active interval")
        torch.cuda.synchronize(torch.device(device))
        method = self.method
        started_at_ns = self._started_at_ns
        ended_at_ns = time.monotonic_ns()
        counter_end_mj: int | None = None
        energy_j: float | None = None
        unavailable_reason = self.unavailable_reason
        try:
            if method == NVML_TOTAL_ENERGY_METHOD:
                counter_end_mj = int(
                    self._module.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                )
                ended_at_ns = time.monotonic_ns()
                if self._counter_start_mj is None:
                    raise RuntimeError("cumulative energy counter lacks a start value")
                energy_j = (counter_end_mj - self._counter_start_mj) / 1000.0
                if energy_j <= 0:
                    raise RuntimeError(
                        "cumulative energy counter did not increase"
                    )
            elif method == NVML_POWER_TRACE_METHOD:
                self._stop_trace()
                final = self._read_power_sample()
                self._trace_samples.append(final)
                ended_at_ns = final.timestamp_monotonic_ns
                if self._trace_error is not None:
                    raise RuntimeError(self._trace_error)
                energy_j = _integrate_power_trace(self._trace_samples)
            else:
                unavailable_reason = unavailable_reason or (
                    "no supported NVML board-energy meter"
                )
        except Exception as error:
            unavailable_reason = (
                f"{method} failed for the measured decode region: "
                + _meter_error_text(error)
            )
            energy_j = None
            counter_end_mj = None
            self._trace_samples = []
        finally:
            self._stop_trace()
            self._active = False
            self._started_at_ns = None
        status = "measured" if energy_j is not None else "unavailable"
        device_measurement = GPUDeviceEnergyMeasurement(
            device_uuid=self.device_uuid,
            meter_method=method,
            measurement_status=status,
            started_at_monotonic_ns=started_at_ns,
            ended_at_monotonic_ns=max(ended_at_ns, started_at_ns + 1),
            energy_j=energy_j,
            counter_start_mj=(
                self._counter_start_mj if status == "measured" else None
            ),
            counter_end_mj=(counter_end_mj if status == "measured" else None),
            power_trace=(
                tuple(self._trace_samples)
                if status == "measured" and method == NVML_POWER_TRACE_METHOD
                else ()
            ),
            requested_power_sample_interval_ms=(
                self.power_trace_sample_interval_ms
            ),
            unavailable_reason=(None if status == "measured" else unavailable_reason),
        )
        self._counter_start_mj = None
        self._trace_samples = []
        return GPUEnergyMeasurement(
            repetition=repetition,
            generated_tokens=generated_tokens,
            devices=(device_measurement,),
        )

    def abort(self) -> None:
        self._stop_trace()
        self._active = False
        self._started_at_ns = None
        self._counter_start_mj = None
        self._trace_samples = []

    def close(self) -> None:
        self.abort()
        if self._initialized and self._module is not None:
            try:
                self._module.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


def _run_repetition(
    model: Any,
    torch: Any,
    samples: Sequence[Any],
    energy_meter: _NVMLBoardEnergyMeter,
    *,
    device: str,
    warmup_steps: int,
    measured_steps: int,
    repetition: int,
) -> GPUBaselineRepetition:
    batch_size = len(samples)
    prompt_length = len(samples[0].prompt_token_ids)
    total_steps = warmup_steps + measured_steps
    input_ids = torch.tensor(
        [sample.prompt_token_ids for sample in samples],
        dtype=torch.long,
        device=device,
    )
    full_attention_mask = torch.ones(
        (batch_size, prompt_length + total_steps),
        dtype=torch.long,
        device=device,
    )
    attention_mask = full_attention_mask[:, :prompt_length]
    position_ids = (
        torch.arange(prompt_length, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    cache_position = torch.arange(prompt_length, device=device)
    decode_position_ids = (
        torch.arange(
            prompt_length,
            prompt_length + total_steps,
            dtype=torch.long,
            device=device,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    decode_cache_positions = torch.arange(
        prompt_length,
        prompt_length + total_steps,
        dtype=torch.long,
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    energy_region_started = False
    energy_measurement: GPUEnergyMeasurement | None = None
    try:
        with torch.inference_mode():
            def prefill_call() -> tuple[Any, Any]:
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    logits_to_keep=1,
                )
                selected_tokens = result.logits[:, -1, :].argmax(dim=-1)
                return result, selected_tokens

            (output, next_tokens), prefill_ms = _timed_cuda_call(
                torch,
                prefill_call,
                device=device,
            )
            if output.logits.shape[:2] != (batch_size, 1):
                raise AssertionError("prefill logits have an invalid shape")
            if output.logits.dtype != torch.bfloat16:
                raise TypeError("GPU baseline output head must emit BF16 logits")
            cache = output.past_key_values
            if _cache_length(cache) != prompt_length:
                raise AssertionError("prefill cache length mismatch")
            _assert_bf16_cache(cache, torch)
            generated = [next_tokens.detach().clone()]
            del output
            timings: list[float] = []
            q_len_one_calls = 0
            growth_checks = 0
            for step in range(total_steps):
                if step == warmup_steps:
                    energy_meter.begin(torch, device=device)
                    energy_region_started = True
                previous_length = _cache_length(cache)
                if previous_length != prompt_length + step:
                    raise AssertionError("decode cache position is not sequential")
                if next_tokens.shape != (batch_size,):
                    raise AssertionError("decode tokens must have shape [batch]")
                step_attention_mask = full_attention_mask[
                    :, : previous_length + 1
                ]
                positions = decode_position_ids[:, step : step + 1]
                cache_positions = decode_cache_positions[step : step + 1]

                def decode_call() -> tuple[Any, Any]:
                    result = model(
                        input_ids=next_tokens[:, None],
                        attention_mask=step_attention_mask,
                        position_ids=positions,
                        cache_position=cache_positions,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    selected_tokens = result.logits[:, -1, :].argmax(dim=-1)
                    return result, selected_tokens

                (decoded, selected_tokens), elapsed_ms = _timed_cuda_call(
                    torch,
                    decode_call,
                    device=device,
                )
                q_len_one_calls += 1
                cache = decoded.past_key_values
                if _cache_length(cache) != previous_length + 1:
                    raise AssertionError("decode cache did not grow by one entry")
                growth_checks += 1
                if decoded.logits.shape[:2] != (batch_size, 1):
                    raise AssertionError("decode logits have an invalid shape")
                if decoded.logits.dtype != torch.bfloat16:
                    raise TypeError("GPU baseline output head must emit BF16 logits")
                next_tokens = selected_tokens
                generated.append(next_tokens.detach().clone())
                del decoded
                if step >= warmup_steps:
                    timings.append(elapsed_ms)
            energy_measurement = energy_meter.end(
                torch,
                device=device,
                repetition=repetition,
                generated_tokens=batch_size * measured_steps,
            )
            energy_region_started = False
    finally:
        if energy_region_started:
            energy_meter.abort()

    if energy_measurement is None:
        raise RuntimeError("GPU energy measurement did not reach a terminal state")

    generated_token_ids = torch.stack(generated).to("cpu").tolist()
    token_hash = _canonical_hash(
        {"generated_token_ids": generated_token_ids}
    )
    return GPUBaselineRepetition(
        repetition=repetition,
        document_ids=tuple(sample.document_id for sample in samples),
        prefill_ms=prefill_ms,
        decode_step_ms=tuple(timings),
        q_len_one_calls=q_len_one_calls,
        cache_growth_checks=growth_checks,
        first_token_count=batch_size,
        generated_token_sha256=token_hash,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        energy_measurement=energy_measurement,
    )


def run_gpu_baseline(
    config: Mapping[str, Any],
    bundle: TokenSampleBundle,
    contract: GPUBaselineContract,
    *,
    device: str,
    device_label: str,
    batch_size: int,
) -> GPUBaselineResult:
    """Measure one GPU and retain OOM separately from execution failures."""

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "GPU baseline execution requires torch and transformers"
        ) from exc

    if contract.sample_bundle_hash != bundle.canonical_hash:
        raise ValueError("GPU baseline contract differs from the sample bundle")
    if (
        contract.workspace_binding.prompt_manifest_hash
        != bundle.prompt_manifest().canonical_hash
    ):
        raise ValueError("GPU baseline workspace binding differs from the sample bundle")
    if contract.source_tree_sha256 != _source_tree_hash():
        raise ValueError("GPU baseline source tree differs from the contract")
    if (
        str(config.get("model_name")) != contract.model_name
        or str(config.get("model_revision")) != contract.model_revision
        or str(config.get("tokenizer_revision"))
        != contract.tokenizer_revision
        or int(config.get("seed", 0)) != contract.seed
        or bool(config.get("trust_remote_code", False))
    ):
        raise ValueError("GPU baseline config differs from the contract")
    baseline_policy = config.get("gpu_baseline")
    expected_baseline_policy = {
        "attention_implementation": contract.attention_implementation,
        "warmup_steps": contract.warmup_steps,
        "measured_steps": contract.measured_steps,
        "repetitions": contract.repetitions,
        "batch_sizes": list(contract.planned_batch_sizes),
        "precision": contract.precision,
        "q_len": contract.q_len,
        "first_gpu_only": True,
        "energy_meter_priority": list(contract.energy_meter_priority),
        "power_trace_sample_interval_ms": (
            contract.power_trace_sample_interval_ms
        ),
    }
    if (
        not isinstance(baseline_policy, Mapping)
        or dict(baseline_policy) != expected_baseline_policy
    ):
        raise ValueError("GPU baseline config policy differs from the contract")
    _require_positive_int(batch_size, "batch_size")
    if batch_size not in contract.planned_batch_sizes:
        raise ValueError("batch size is not in the contracted baseline plan")
    planned_labels = {
        _normalized_label(label): label
        for label in contract.planned_device_labels
    }
    normalized_requested_label = _normalized_label(device_label)
    if normalized_requested_label not in planned_labels:
        raise ValueError("device label is not in the contracted baseline plan")
    canonical_device_label = planned_labels[normalized_requested_label]
    selected = torch.device(device)
    if (
        selected.type != "cuda"
        or selected.index is None
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "GPU baseline execution requires an explicit CUDA device index"
        )
    torch.cuda.set_device(selected)
    random.seed(contract.seed)
    torch.manual_seed(contract.seed)
    torch.cuda.manual_seed_all(contract.seed)
    torch.set_float32_matmul_precision("high")
    runtime = _capture_baseline_runtime(
        device,
        seed=contract.seed,
        attention_implementation=contract.attention_implementation,
        torch=torch,
    )
    device_name = str(runtime.logical["device_name"])
    device_uuid = str(runtime.observation["device_uuid"])
    normalized_label = _normalized_label(canonical_device_label)
    normalized_name = _normalized_label(device_name)
    if normalized_label not in normalized_name:
        raise ValueError(
            "device label is not present in the measured CUDA device name"
        )
    energy_meter = _NVMLBoardEnergyMeter(
        device_uuid=device_uuid,
        method_priority=contract.energy_meter_priority,
        power_trace_sample_interval_ms=(
            contract.power_trace_sample_interval_ms
        ),
    )
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repetitions: list[GPUBaselineRepetition] = []
    hardware_state_snapshots: list[GPUHardwareStateSnapshot] = []
    model = None
    state = "succeeded"
    error_class = None
    error_message = None
    try:
        hardware_state_snapshots.append(
            _capture_hardware_state(
                device_uuid=device_uuid,
                phase="run_start",
            )
        )
        if (
            hardware_state_snapshots[-1]
            .fields["mig.mode.current"]
            .lower()
            != "disabled"
        ):
            raise RuntimeError("GPU baseline requires MIG to be disabled")
        model = AutoModelForCausalLM.from_pretrained(
            contract.model_name,
            revision=contract.model_revision,
            torch_dtype=torch.bfloat16,
            cache_dir=config.get("hf_cache_dir"),
            local_files_only=bool(config.get("local_files_only", True)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            attn_implementation=contract.attention_implementation,
            low_cpu_mem_usage=True,
        )
        model = model.to(device).eval()
        effective_attention = getattr(
            model.config,
            "_attn_implementation",
            None,
        )
        if effective_attention != contract.attention_implementation:
            raise RuntimeError(
                "loaded model did not retain the contracted attention "
                "implementation"
            )
        parameter_dtypes = {
            parameter.dtype
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
        if parameter_dtypes != {torch.bfloat16}:
            raise TypeError("GPU baseline model parameters must all be BF16")
        for repetition in range(contract.repetitions):
            hardware_state_snapshots.append(
                _capture_hardware_state(
                    device_uuid=device_uuid,
                    phase=f"repetition_{repetition}_start",
                )
            )
            samples = _select_repeat_samples(
                bundle,
                repetition=repetition,
                batch_size=batch_size,
            )
            repetitions.append(
                _run_repetition(
                    model,
                    torch,
                    samples,
                    energy_meter,
                    device=device,
                    warmup_steps=contract.warmup_steps,
                    measured_steps=contract.measured_steps,
                    repetition=repetition,
                )
            )
            hardware_state_snapshots.append(
                _capture_hardware_state(
                    device_uuid=device_uuid,
                    phase=f"repetition_{repetition}_end",
                )
            )
    except Exception as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
            "cuda out of memory" in str(error).lower()
            or "cuda error: out of memory" in str(error).lower()
        )
        state = "oom" if is_oom else "failed"
        error_class = type(error).__name__
        error_message = str(error) or repr(error)
        repetitions.clear()
    finally:
        try:
            hardware_state_snapshots.append(
                _capture_hardware_state(
                    device_uuid=device_uuid,
                    phase="run_end",
                )
            )
        except Exception as snapshot_error:
            if state == "succeeded":
                state = "failed"
                error_class = type(snapshot_error).__name__
                error_message = str(snapshot_error) or repr(snapshot_error)
                repetitions.clear()
        if model is not None:
            del model
        energy_meter.close()
        gc.collect()
        with torch.cuda.device(selected):
            torch.cuda.empty_cache()

    result_arguments = {
        "contract_hash": contract.contract_hash,
        "device_label": canonical_device_label,
        "device_name": device_name,
        "device_uuid": device_uuid,
        "runtime_environment": runtime,
        "batch_size": batch_size,
        "state": state,
        "repetitions": tuple(repetitions),
        "hardware_state_snapshots": tuple(hardware_state_snapshots),
        "error_class": error_class,
        "error_message": error_message,
        "created_at_utc": created_at,
    }
    try:
        return GPUBaselineResult(**result_arguments)
    except (TypeError, ValueError) as validation_error:
        if state != "succeeded":
            raise
        return GPUBaselineResult(
            **{
                **result_arguments,
                "state": "failed",
                "repetitions": (),
                "error_class": type(validation_error).__name__,
                "error_message": (
                    str(validation_error) or repr(validation_error)
                ),
            }
        )


def build_gpu_baseline_report(
    contract: GPUBaselineContract,
    results: Sequence[GPUBaselineResult],
) -> Mapping[str, Any]:
    """Join raw device results without extrapolating across devices or batches."""

    rows = tuple(
        sorted(
            results,
            key=lambda row: (
                row.device_label,
                row.device_uuid,
                row.batch_size,
                row.result_id,
            ),
        )
    )
    if not rows:
        raise ValueError("GPU baseline report requires results")
    if any(row.contract_hash != contract.contract_hash for row in rows):
        raise ValueError("GPU baseline results use different contracts")
    keys = tuple(
        (row.device_uuid, row.batch_size) for row in rows
    )
    if len(keys) != len(set(keys)):
        raise ValueError("GPU baseline report contains duplicate device batches")
    planned_batches = set(contract.planned_batch_sizes)
    if any(row.batch_size not in planned_batches for row in rows):
        raise ValueError("GPU baseline report contains an unplanned batch")
    planned_labels = set(contract.planned_device_labels)
    if any(row.device_label not in planned_labels for row in rows):
        raise ValueError("GPU baseline report contains an unplanned device")
    for device_label in planned_labels:
        label_uuids = {
            row.device_uuid
            for row in rows
            if row.device_label == device_label
        }
        if len(label_uuids) > 1:
            raise ValueError(
                "one planned GPU label maps to multiple physical devices"
            )
    logical_hardware_keys = {"device_name", "compute_capability"}
    software_environments = {
        _canonical_hash(
            {
                key: value
                for key, value in row.runtime_environment.logical.items()
                if key not in logical_hardware_keys
            }
        )
        for row in rows
    }
    if len(software_environments) != 1:
        raise ValueError(
            "GPU baseline results use incompatible software environments"
        )
    observed_driver_versions = {
        snapshot.fields["driver_version"]
        for row in rows
        for snapshot in row.hardware_state_snapshots
    }
    if len(observed_driver_versions) > 1:
        raise ValueError("GPU baseline results use different driver versions")
    for device_uuid in {row.device_uuid for row in rows}:
        device_rows = tuple(
            row for row in rows if row.device_uuid == device_uuid
        )
        if len({row.device_label for row in device_rows}) != 1:
            raise ValueError("one GPU UUID has conflicting device labels")
        if len({row.device_name for row in device_rows}) != 1:
            raise ValueError("one GPU UUID has conflicting device names")
        if len(
            {
                row.runtime_environment.logical_fingerprint
                for row in device_rows
            }
        ) != 1:
            raise ValueError(
                "one GPU was measured under different runtime environments"
            )
        if len(
            {
                row.runtime_environment.observation.get(
                    "total_memory_bytes"
                )
                for row in device_rows
            }
        ) != 1:
            raise ValueError("one GPU UUID has conflicting memory capacity")
    prompt_ids = tuple(prompt.document_id for prompt in contract.prompts)
    for row in rows:
        if row.state != "succeeded":
            continue
        if len(row.repetitions) != contract.repetitions:
            raise ValueError("GPU result repetition count differs from contract")
        for repetition in row.repetitions:
            if len(repetition.decode_step_ms) != contract.measured_steps:
                raise ValueError(
                    "GPU result measured-step count differs from contract"
                )
            expected_calls = contract.warmup_steps + contract.measured_steps
            if (
                repetition.q_len_one_calls != expected_calls
                or repetition.cache_growth_checks != expected_calls
            ):
                raise ValueError(
                    "GPU result cached-q1 evidence differs from contract"
                )
            start = (
                repetition.repetition * row.batch_size
            ) % len(prompt_ids)
            expected_ids = tuple(
                prompt_ids[(start + offset) % len(prompt_ids)]
                for offset in range(row.batch_size)
            )
            if repetition.document_ids != expected_ids:
                raise ValueError(
                    "GPU result prompt selection differs from contract"
                )
            if any(
                device.requested_power_sample_interval_ms
                != contract.power_trace_sample_interval_ms
                for device in repetition.energy_measurement.devices
            ):
                raise ValueError(
                    "GPU energy sample interval differs from the contract"
                )
    trajectories: dict[
        tuple[int, int, tuple[str, ...]],
        str,
    ] = {}
    for row in rows:
        if row.state != "succeeded":
            continue
        for repetition in row.repetitions:
            key = (
                row.batch_size,
                repetition.repetition,
                repetition.document_ids,
            )
            previous_hash = trajectories.setdefault(
                key,
                repetition.generated_token_sha256,
            )
            if previous_hash != repetition.generated_token_sha256:
                raise ValueError(
                    "GPU devices followed different generated-token "
                    "trajectories"
                )
    succeeded = tuple(row for row in rows if row.state == "succeeded")
    completion_by_device: dict[str, Mapping[str, Any]] = {}
    completion_by_device_label: dict[str, Mapping[str, Any]] = {}
    complete_devices: set[str] = set()
    for device_uuid in sorted({row.device_uuid for row in rows}):
        device_rows = tuple(
            row for row in rows if row.device_uuid == device_uuid
        )
        observed_batches = {row.batch_size for row in device_rows}
        missing_batches = tuple(sorted(planned_batches - observed_batches))
        failed_batches = tuple(
            sorted(
                row.batch_size
                for row in device_rows
                if row.state == "failed"
            )
        )
        complete = not missing_batches and not failed_batches
        if complete:
            complete_devices.add(device_uuid)
        completion_by_device[device_uuid] = {
            "complete": complete,
            "missing_batch_sizes": list(missing_batches),
            "failed_batch_sizes": list(failed_batches),
        }
    for device_label in contract.planned_device_labels:
        label_rows = tuple(
            row for row in rows if row.device_label == device_label
        )
        observed_batches = {row.batch_size for row in label_rows}
        missing_batches = tuple(sorted(planned_batches - observed_batches))
        failed_batches = tuple(
            sorted(
                row.batch_size
                for row in label_rows
                if row.state == "failed"
            )
        )
        uuids = sorted({row.device_uuid for row in label_rows})
        completion_by_device_label[device_label] = {
            "complete": (
                len(uuids) == 1
                and not missing_batches
                and not failed_batches
            ),
            "device_uuid": uuids[0] if len(uuids) == 1 else None,
            "missing_batch_sizes": list(missing_batches),
            "failed_batch_sizes": list(failed_batches),
        }
    best_by_device: dict[str, Mapping[str, Any]] = {}
    for device_uuid in sorted(complete_devices):
        candidates = tuple(
            row
            for row in succeeded
            if row.device_uuid == device_uuid
        )
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: float(row.summary["tokens_per_second"]),
        )
        best_by_device[device_uuid] = {
            "device_label": best.device_label,
            "device_name": best.device_name,
            "batch_size": best.batch_size,
            "result_id": best.result_id,
            "tokens_per_second": best.summary["tokens_per_second"],
            "mean_batch_step_ms": best.summary["mean_batch_step_ms"],
            "energy": best.summary["energy"],
            "selection_scope": "measured_batches_only",
        }
    energy_availability_by_device_label: dict[str, Mapping[str, Any]] = {}
    for device_label in contract.planned_device_labels:
        successful_rows = tuple(
            row
            for row in succeeded
            if row.device_label == device_label
        )
        energy_availability_by_device_label[device_label] = {
            "available_batch_sizes": [
                row.batch_size
                for row in successful_rows
                if bool(row.summary["energy"]["available"])
            ],
            "unavailable_by_batch_size": {
                str(row.batch_size): row.summary["energy"]["unavailable_reason"]
                for row in successful_rows
                if not bool(row.summary["energy"]["available"])
            },
            "meter_methods": sorted(
                {
                    str(row.summary["energy"]["meter_method"])
                    for row in successful_rows
                }
            ),
        }
    body = {
        "schema_version": GPU_BASELINE_REPORT_SCHEMA,
        "contract": contract.to_dict(),
        "results": [row.to_dict() for row in rows],
        "completion_by_device": completion_by_device,
        "completion_by_device_label": completion_by_device_label,
        "best_measured_by_device": best_by_device,
        "energy_availability_by_device_label": (
            energy_availability_by_device_label
        ),
        "measurement_scope": "single_gpu_no_extrapolation",
        "timing_scope": contract.timing_scope,
        "energy_scope": contract.energy_scope,
        "energy_meter_priority": list(contract.energy_meter_priority),
        "complete": all(
            record["complete"]
            for record in completion_by_device_label.values()
        ),
    }
    return {**body, "report_hash": _canonical_hash(body)}


@dataclass(frozen=True)
class ThroughputEvidenceRow:
    """One provenance-bound throughput observation with an explicit evidence tier."""

    system_name: str
    model_name: str
    model_revision: str
    context_length: int
    batch_size: int
    tokens_per_second: float
    evidence_tier: str
    evidence_source: str
    resource_budget_hash: str
    artifact_hash: str
    workspace_provenance_sha256: str | None
    q_len: int = 1
    schema_version: str = THROUGHPUT_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != THROUGHPUT_EVIDENCE_SCHEMA:
            raise ValueError("unsupported throughput-evidence schema")
        if not self.system_name or not self.model_name or not self.evidence_source:
            raise ValueError("throughput evidence identity must be complete")
        _require_git_commit(self.model_revision, "model_revision")
        _require_positive_int(self.context_length, "context_length")
        _require_positive_int(self.batch_size, "batch_size")
        if self.q_len != 1:
            raise ValueError("throughput evidence must use cached q_len=1 decode")
        throughput = _require_nonnegative_finite(
            self.tokens_per_second,
            "tokens_per_second",
        )
        if throughput <= 0:
            raise ValueError("tokens_per_second must be positive")
        object.__setattr__(self, "tokens_per_second", throughput)
        if self.evidence_tier not in {
            MEASURED_EVIDENCE_TIER,
            PEAK_ROOFLINE_EVIDENCE_TIER,
        }:
            raise ValueError("unsupported throughput evidence tier")
        _require_sha256(self.resource_budget_hash, "resource_budget_hash")
        _require_sha256(self.artifact_hash, "artifact_hash")
        if self.evidence_tier == MEASURED_EVIDENCE_TIER:
            if self.workspace_provenance_sha256 is None:
                raise ValueError("measured throughput requires workspace provenance")
            _require_sha256(
                self.workspace_provenance_sha256,
                "workspace_provenance_sha256",
            )
        elif self.workspace_provenance_sha256 is not None:
            _require_sha256(
                self.workspace_provenance_sha256,
                "workspace_provenance_sha256",
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "system_name": self.system_name,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "q_len": self.q_len,
            "tokens_per_second": self.tokens_per_second,
            "evidence_tier": self.evidence_tier,
            "evidence_source": self.evidence_source,
            "resource_budget_hash": self.resource_budget_hash,
            "artifact_hash": self.artifact_hash,
            "workspace_provenance_sha256": self.workspace_provenance_sha256,
        }

    @property
    def evidence_id(self) -> str:
        return f"throughput-{_canonical_hash(self._body())}"

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ThroughputEvidenceRow":
        row = cls(
            schema_version=str(value["schema_version"]),
            system_name=str(value["system_name"]),
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            context_length=int(value["context_length"]),
            batch_size=int(value["batch_size"]),
            q_len=int(value["q_len"]),
            tokens_per_second=float(value["tokens_per_second"]),
            evidence_tier=str(value["evidence_tier"]),
            evidence_source=str(value["evidence_source"]),
            resource_budget_hash=str(value["resource_budget_hash"]),
            artifact_hash=str(value["artifact_hash"]),
            workspace_provenance_sha256=(
                None
                if value.get("workspace_provenance_sha256") is None
                else str(value["workspace_provenance_sha256"])
            ),
        )
        if value.get("evidence_id") != row.evidence_id:
            raise ValueError("throughput-evidence identity mismatch")
        return row


@dataclass(frozen=True)
class EnergyEvidenceRow:
    """One measured energy-efficiency observation with raw-artifact identity."""

    system_name: str
    model_name: str
    model_revision: str
    context_length: int
    batch_size: int
    energy_per_token_j: float
    mean_decode_step_s: float
    meter_method: str
    device_ids: tuple[str, ...]
    evidence_source: str
    resource_budget_hash: str
    artifact_hash: str
    workspace_provenance_sha256: str
    q_len: int = 1
    evidence_tier: str = MEASURED_EVIDENCE_TIER
    schema_version: str = ENERGY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ENERGY_EVIDENCE_SCHEMA:
            raise ValueError("unsupported energy-evidence schema")
        if self.evidence_tier != MEASURED_EVIDENCE_TIER:
            raise ValueError("headline energy evidence must be measured")
        if (
            not self.system_name
            or not self.model_name
            or not self.meter_method
            or not self.evidence_source
        ):
            raise ValueError("energy evidence identity must be complete")
        _require_git_commit(self.model_revision, "model_revision")
        _require_positive_int(self.context_length, "context_length")
        _require_positive_int(self.batch_size, "batch_size")
        if self.q_len != 1:
            raise ValueError("energy evidence must use cached q_len=1 decode")
        energy = _require_nonnegative_finite(
            self.energy_per_token_j,
            "energy_per_token_j",
        )
        delay = _require_nonnegative_finite(
            self.mean_decode_step_s,
            "mean_decode_step_s",
        )
        if energy <= 0 or delay <= 0:
            raise ValueError("measured energy and decode delay must be positive")
        object.__setattr__(self, "energy_per_token_j", energy)
        object.__setattr__(self, "mean_decode_step_s", delay)
        if not self.device_ids or any(not value for value in self.device_ids):
            raise ValueError("energy evidence requires measurement device identities")
        if len(self.device_ids) != len(set(self.device_ids)):
            raise ValueError("energy evidence repeats a measurement device")
        for name in (
            "resource_budget_hash",
            "artifact_hash",
            "workspace_provenance_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)

    @property
    def tokens_per_joule(self) -> float:
        return 1.0 / self.energy_per_token_j

    @property
    def energy_delay_product_j_s(self) -> float:
        return self.energy_per_token_j * self.mean_decode_step_s

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "system_name": self.system_name,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "q_len": self.q_len,
            "energy_per_token_j": self.energy_per_token_j,
            "tokens_per_joule": self.tokens_per_joule,
            "mean_decode_step_s": self.mean_decode_step_s,
            "energy_delay_product_j_s": self.energy_delay_product_j_s,
            "meter_method": self.meter_method,
            "device_ids": list(self.device_ids),
            "evidence_tier": self.evidence_tier,
            "evidence_source": self.evidence_source,
            "resource_budget_hash": self.resource_budget_hash,
            "artifact_hash": self.artifact_hash,
            "workspace_provenance_sha256": self.workspace_provenance_sha256,
        }

    @property
    def evidence_id(self) -> str:
        return f"energy-{_canonical_hash(self._body())}"

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnergyEvidenceRow":
        row = cls(
            schema_version=str(value["schema_version"]),
            system_name=str(value["system_name"]),
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            context_length=int(value["context_length"]),
            batch_size=int(value["batch_size"]),
            q_len=int(value["q_len"]),
            energy_per_token_j=float(value["energy_per_token_j"]),
            mean_decode_step_s=float(value["mean_decode_step_s"]),
            meter_method=str(value["meter_method"]),
            device_ids=tuple(str(item) for item in value["device_ids"]),
            evidence_tier=str(value["evidence_tier"]),
            evidence_source=str(value["evidence_source"]),
            resource_budget_hash=str(value["resource_budget_hash"]),
            artifact_hash=str(value["artifact_hash"]),
            workspace_provenance_sha256=str(
                value["workspace_provenance_sha256"]
            ),
        )
        if value.get("tokens_per_joule") != row.tokens_per_joule:
            raise ValueError("energy evidence tokens/J mismatch")
        if (
            value.get("energy_delay_product_j_s")
            != row.energy_delay_product_j_s
        ):
            raise ValueError("energy evidence EDP mismatch")
        if value.get("evidence_id") != row.evidence_id:
            raise ValueError("energy-evidence identity mismatch")
        return row


def _resource_budget_hash(resource_budget: Mapping[str, Any]) -> str:
    required = {
        "reference_system",
        "aggregate_area_limit_mm2",
        "aggregate_hbm_capacity_limit_bytes",
        "aggregate_hbm_bandwidth_limit_bytes_per_s",
    }
    if set(resource_budget) != required:
        raise ValueError("resource budget fields differ from the comparison contract")
    if not str(resource_budget["reference_system"]):
        raise ValueError("resource budget reference system must be non-empty")
    for field in (
        "aggregate_area_limit_mm2",
        "aggregate_hbm_capacity_limit_bytes",
        "aggregate_hbm_bandwidth_limit_bytes_per_s",
    ):
        value = _require_nonnegative_finite(resource_budget[field], field)
        if value <= 0:
            raise ValueError(f"resource budget {field} must be positive")
    return _canonical_hash(dict(resource_budget))


def validate_gpu_baseline_report(
    report: Mapping[str, Any],
) -> tuple[GPUBaselineContract, Mapping[str, Any]]:
    """Rebuild a baseline report from terminal rows and require exact identity."""

    if report.get("schema_version") != GPU_BASELINE_REPORT_SCHEMA:
        raise ValueError("unsupported GPU baseline report schema")
    contract_value = report.get("contract")
    if not isinstance(contract_value, Mapping):
        raise ValueError("GPU baseline report lacks its contract")
    contract = GPUBaselineContract.from_dict(contract_value)
    results_value = report.get("results")
    if not isinstance(results_value, Sequence) or isinstance(
        results_value,
        (str, bytes),
    ):
        raise ValueError("GPU baseline report lacks result rows")
    results = tuple(
        GPUBaselineResult.from_dict(row)
        for row in results_value
        if isinstance(row, Mapping)
    )
    if len(results) != len(results_value):
        raise ValueError("GPU baseline report contains a malformed result row")
    rebuilt = build_gpu_baseline_report(contract, results)
    provided = {key: value for key, value in report.items() if key != "content_hash"}
    if provided != rebuilt:
        raise ValueError("GPU baseline report differs from its measured results")
    return contract, rebuilt


def build_gpu_baseline_stage_receipt(
    report: Mapping[str, Any],
    *,
    provenance_path: str | Path,
) -> Mapping[str, Any]:
    """Bind all terminal baseline work units to the sweep provenance receipt."""

    contract, rebuilt = validate_gpu_baseline_report(report)
    if rebuilt["complete"] is not True:
        raise ValueError("GPU baseline stage cannot seal an incomplete report")
    path = Path(provenance_path).resolve()
    provenance = load_immutable_json(path)
    binding = contract.workspace_binding
    if (
        provenance.get("schema_version") != SWEEP_PROVENANCE_SCHEMA
        or provenance.get("manifest_hash") != binding.manifest_hash
        or provenance.get("run_plan_hash") != binding.run_plan_hash
        or provenance.get("prompt_manifest_hash") != binding.prompt_manifest_hash
        or _sha256_file(path) != binding.sweep_provenance_sha256
    ):
        raise ValueError("GPU baseline report differs from sweep provenance")
    results = tuple(
        GPUBaselineResult.from_dict(value) for value in rebuilt["results"]
    )
    body = {
        "schema_version": GPU_BASELINE_STAGE_RECEIPT_SCHEMA,
        "workspace_binding": binding.to_dict(),
        "contract_hash": contract.contract_hash,
        "report_hash": rebuilt["report_hash"],
        "work_units": [
            {
                "device_label": label,
                "batch_size": batch_size,
                "first_gpu_only": True,
            }
            for label in contract.planned_device_labels
            for batch_size in contract.planned_batch_sizes
        ],
        "terminal_result_ids": [result.result_id for result in results],
        "energy_scope": contract.energy_scope,
        "energy_availability_by_device_label": rebuilt[
            "energy_availability_by_device_label"
        ],
        "complete": True,
    }
    return {**body, "receipt_hash": _canonical_hash(body)}


def validate_gpu_baseline_stage_receipt(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require a sealed stage receipt for the exact terminal baseline report."""

    contract, rebuilt = validate_gpu_baseline_report(report)
    if rebuilt["complete"] is not True:
        raise ValueError("GPU baseline stage receipt cannot cover an incomplete report")
    results = tuple(
        GPUBaselineResult.from_dict(value) for value in rebuilt["results"]
    )
    body = {
        "schema_version": GPU_BASELINE_STAGE_RECEIPT_SCHEMA,
        "workspace_binding": contract.workspace_binding.to_dict(),
        "contract_hash": contract.contract_hash,
        "report_hash": rebuilt["report_hash"],
        "work_units": [
            {
                "device_label": label,
                "batch_size": batch_size,
                "first_gpu_only": True,
            }
            for label in contract.planned_device_labels
            for batch_size in contract.planned_batch_sizes
        ],
        "terminal_result_ids": [result.result_id for result in results],
        "energy_scope": contract.energy_scope,
        "energy_availability_by_device_label": rebuilt[
            "energy_availability_by_device_label"
        ],
        "complete": True,
    }
    expected = {**body, "receipt_hash": _canonical_hash(body)}
    provided = {key: value for key, value in receipt.items() if key != "content_hash"}
    if provided != expected:
        raise ValueError("GPU baseline stage receipt differs from its report")
    return expected


def gpu_baseline_throughput_evidence(
    report: Mapping[str, Any],
    *,
    stage_receipt: Mapping[str, Any],
    device_label: str,
    resource_budget: Mapping[str, Any],
) -> ThroughputEvidenceRow:
    """Select the best measured batch for one contracted GPU device label."""

    contract, rebuilt = validate_gpu_baseline_report(report)
    validate_gpu_baseline_stage_receipt(rebuilt, stage_receipt)
    report_hash = str(rebuilt["report_hash"])
    if rebuilt["complete"] is not True:
        raise ValueError("headline GPU baseline report must be complete")
    best_by_device = rebuilt.get("best_measured_by_device")
    if not isinstance(best_by_device, Mapping):
        raise ValueError("GPU baseline report lacks best measured results")
    matches = tuple(
        row
        for row in best_by_device.values()
        if isinstance(row, Mapping) and row.get("device_label") == device_label
    )
    if len(matches) != 1:
        raise ValueError("GPU baseline device label lacks one best measured result")
    best = matches[0]
    return ThroughputEvidenceRow(
        system_name=str(best["device_name"]),
        model_name=contract.model_name,
        model_revision=contract.model_revision,
        context_length=contract.context_length,
        batch_size=int(best["batch_size"]),
        q_len=contract.q_len,
        tokens_per_second=float(best["tokens_per_second"]),
        evidence_tier=MEASURED_EVIDENCE_TIER,
        evidence_source=GPU_BASELINE_SCOPE,
        resource_budget_hash=_resource_budget_hash(resource_budget),
        artifact_hash=report_hash,
        workspace_provenance_sha256=(
            contract.workspace_binding.sweep_provenance_sha256
        ),
    )


def gpu_baseline_energy_evidence(
    report: Mapping[str, Any],
    *,
    stage_receipt: Mapping[str, Any],
    device_label: str,
    resource_budget: Mapping[str, Any],
) -> EnergyEvidenceRow:
    """Build measured GPU energy evidence or fail with the recorded reason."""

    contract, rebuilt = validate_gpu_baseline_report(report)
    validate_gpu_baseline_stage_receipt(rebuilt, stage_receipt)
    if rebuilt["complete"] is not True:
        raise ValueError("headline GPU energy report must be complete")
    best_by_device = rebuilt.get("best_measured_by_device")
    if not isinstance(best_by_device, Mapping):
        raise ValueError("GPU baseline report lacks best measured results")
    matches = tuple(
        row
        for row in best_by_device.values()
        if isinstance(row, Mapping) and row.get("device_label") == device_label
    )
    if len(matches) != 1:
        raise ValueError("GPU baseline device label lacks one measured result")
    best = matches[0]
    energy = best.get("energy")
    if not isinstance(energy, Mapping):
        raise ValueError("GPU baseline result lacks board-energy evidence")
    if energy.get("available") is not True:
        reason = str(
            energy.get("unavailable_reason")
            or "no supported measured board-energy meter"
        )
        raise ValueError(f"measured GPU energy is unavailable: {reason}")
    meter_method = str(energy.get("meter_method", ""))
    if meter_method not in GPU_BASELINE_ENERGY_METER_PRIORITY:
        raise ValueError("GPU energy evidence does not use a contracted NVML meter")
    device_ids = tuple(str(item) for item in energy.get("device_uuids", ()))
    energy_per_token = energy.get("energy_per_token_j")
    mean_step_ms = best.get("mean_batch_step_ms")
    if energy_per_token is None or mean_step_ms is None:
        raise ValueError("GPU energy evidence lacks measured joules or delay")
    return EnergyEvidenceRow(
        system_name=str(best["device_name"]),
        model_name=contract.model_name,
        model_revision=contract.model_revision,
        context_length=contract.context_length,
        batch_size=int(best["batch_size"]),
        q_len=contract.q_len,
        energy_per_token_j=float(energy_per_token),
        mean_decode_step_s=float(mean_step_ms) / 1000.0,
        meter_method=meter_method,
        device_ids=device_ids,
        evidence_source=GPU_BASELINE_ENERGY_SOURCE,
        resource_budget_hash=_resource_budget_hash(resource_budget),
        artifact_hash=str(rebuilt["report_hash"]),
        workspace_provenance_sha256=(
            contract.workspace_binding.sweep_provenance_sha256
        ),
    )


def build_headline_energy_comparison(
    *,
    plena_measurement: EnergyEvidenceRow,
    gpu_measurement: EnergyEvidenceRow,
    resource_budget: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Compare measured tokens/J and EDP without an analytic substitute."""

    budget_hash = _resource_budget_hash(resource_budget)
    rows = (plena_measurement, gpu_measurement)
    if any(row.evidence_tier != MEASURED_EVIDENCE_TIER for row in rows):
        raise ValueError("headline energy ratios require measured evidence")
    if gpu_measurement.evidence_source != GPU_BASELINE_ENERGY_SOURCE:
        raise ValueError("headline energy denominator must be measured GPU board energy")
    if plena_measurement.evidence_source == GPU_BASELINE_ENERGY_SOURCE:
        raise ValueError("headline energy numerator must be a measured PLENA result")
    for field in (
        "model_name",
        "model_revision",
        "context_length",
        "q_len",
        "resource_budget_hash",
        "workspace_provenance_sha256",
    ):
        if len({getattr(row, field) for row in rows}) != 1:
            raise ValueError(f"headline energy evidence differs on {field}")
    if plena_measurement.resource_budget_hash != budget_hash:
        raise ValueError("headline energy evidence uses a different resource budget")
    body = {
        "schema_version": ENERGY_COMPARISON_SCHEMA,
        "evidence_tier": MEASURED_EVIDENCE_TIER,
        "resource_budget": dict(resource_budget),
        "resource_budget_hash": budget_hash,
        "numerator": plena_measurement.to_dict(),
        "denominator": gpu_measurement.to_dict(),
        "tokens_per_joule_ratio": (
            plena_measurement.tokens_per_joule
            / gpu_measurement.tokens_per_joule
        ),
        "energy_per_token_improvement": (
            gpu_measurement.energy_per_token_j
            / plena_measurement.energy_per_token_j
        ),
        "energy_delay_product_improvement": (
            gpu_measurement.energy_delay_product_j_s
            / plena_measurement.energy_delay_product_j_s
        ),
        "ratio_semantics": "measured_plena_over_measured_gpu_efficiency",
        "analytic_substitution_permitted": False,
    }
    return {**body, "comparison_hash": _canonical_hash(body)}


def build_headline_throughput_comparison(
    *,
    plena_measurement: ThroughputEvidenceRow,
    gpu_measurement: ThroughputEvidenceRow,
    peak_roofline_rows: Sequence[ThroughputEvidenceRow],
    resource_budget: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build a measured-only headline ratio and a separate peak-bound table."""

    budget_hash = _resource_budget_hash(resource_budget)
    headline_rows = (plena_measurement, gpu_measurement)
    if any(row.evidence_tier != MEASURED_EVIDENCE_TIER for row in headline_rows):
        raise ValueError("headline throughput ratios require measured evidence on both sides")
    if gpu_measurement.evidence_source != GPU_BASELINE_SCOPE:
        raise ValueError("headline denominator must be the measured cached-q1 GPU baseline")
    if plena_measurement.evidence_source == GPU_BASELINE_SCOPE:
        raise ValueError("headline numerator must be a measured PLENA result")
    matched_fields = (
        "model_name",
        "model_revision",
        "context_length",
        "q_len",
        "resource_budget_hash",
        "workspace_provenance_sha256",
    )
    for field in matched_fields:
        values = {getattr(row, field) for row in headline_rows}
        if len(values) != 1:
            raise ValueError(f"headline throughput evidence differs on {field}")
    if plena_measurement.resource_budget_hash != budget_hash:
        raise ValueError("headline throughput evidence uses a different resource budget")
    peaks = tuple(peak_roofline_rows)
    if not peaks:
        raise ValueError("comparison requires a separately labelled peak-roofline table")
    for row in peaks:
        if row.evidence_tier != PEAK_ROOFLINE_EVIDENCE_TIER:
            raise ValueError("peak-roofline table contains non-peak evidence")
        if (
            row.model_name != plena_measurement.model_name
            or row.model_revision != plena_measurement.model_revision
            or row.context_length != plena_measurement.context_length
            or row.q_len != 1
        ):
            raise ValueError("peak-roofline row uses a different model workload")
    headline = {
        "evidence_tier": MEASURED_EVIDENCE_TIER,
        "numerator": plena_measurement.to_dict(),
        "denominator": gpu_measurement.to_dict(),
        "throughput_ratio": (
            plena_measurement.tokens_per_second
            / gpu_measurement.tokens_per_second
        ),
        "ratio_semantics": "measured_plena_over_measured_gpu",
    }
    body = {
        "schema_version": THROUGHPUT_COMPARISON_SCHEMA,
        "resource_budget": dict(resource_budget),
        "resource_budget_hash": budget_hash,
        "headline": headline,
        "peak_roofline_table": [row.to_dict() for row in peaks],
        "peak_roofline_ratio_permitted": False,
    }
    return {**body, "comparison_hash": _canonical_hash(body)}


def _load_config(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("GPU baseline config must contain an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--sample-bundle", required=True)
    prepare.add_argument("--provenance", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--attention-implementation",
        default="sdpa",
    )
    prepare.add_argument("--warmup-steps", type=int, default=16)
    prepare.add_argument("--measured-steps", type=int, default=128)
    prepare.add_argument("--repetitions", type=int, default=3)
    prepare.add_argument(
        "--device-labels",
        nargs="+",
        default=("A100", "H100"),
    )
    prepare.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8),
    )
    prepare.add_argument(
        "--energy-meter-priority",
        nargs="+",
        default=GPU_BASELINE_ENERGY_METER_PRIORITY,
    )
    prepare.add_argument(
        "--power-trace-sample-interval-ms",
        type=int,
        default=10,
    )

    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--sample-bundle", required=True)
    run.add_argument("--contract", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--device-label", required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument("--output", required=True)

    merge = commands.add_parser("merge")
    merge.add_argument("--contract", required=True)
    merge.add_argument("--results", nargs="+", required=True)
    merge.add_argument("--output", required=True)

    receipt = commands.add_parser("receipt")
    receipt.add_argument("--report", required=True)
    receipt.add_argument("--provenance", required=True)
    receipt.add_argument("--output", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            config = _load_config(args.config)
            bundle = load_sample_bundle(args.sample_bundle)
            workspace_binding = load_gpu_baseline_workspace_binding(
                args.provenance,
                bundle,
            )
            contract = build_gpu_baseline_contract(
                config,
                bundle,
                workspace_binding,
                attention_implementation=args.attention_implementation,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                repetitions=args.repetitions,
                planned_device_labels=args.device_labels,
                planned_batch_sizes=args.batch_sizes,
                energy_meter_priority=args.energy_meter_priority,
                power_trace_sample_interval_ms=(
                    args.power_trace_sample_interval_ms
                ),
            )
            payload = contract.to_dict()
        elif args.command == "run":
            config = _load_config(args.config)
            bundle = load_sample_bundle(args.sample_bundle)
            contract = GPUBaselineContract.from_dict(
                load_immutable_json(Path(args.contract))
            )
            destination = Path(args.output)
            if destination.exists():
                existing = GPUBaselineResult.from_dict(
                    load_immutable_json(destination)
                )
                if (
                    existing.contract_hash != contract.contract_hash
                    or existing.batch_size != args.batch_size
                    or _normalized_label(existing.device_label)
                    != _normalized_label(args.device_label)
                ):
                    raise ValueError(
                        "existing GPU baseline result belongs to another work unit"
                    )
                payload = existing.to_dict()
            else:
                payload = run_gpu_baseline(
                    config,
                    bundle,
                    contract,
                    device=args.device,
                    device_label=args.device_label,
                    batch_size=args.batch_size,
                ).to_dict()
        elif args.command == "merge":
            contract = GPUBaselineContract.from_dict(
                load_immutable_json(Path(args.contract))
            )
            results = tuple(
                GPUBaselineResult.from_dict(load_immutable_json(Path(path)))
                for path in args.results
            )
            payload = build_gpu_baseline_report(contract, results)
        else:
            payload = build_gpu_baseline_stage_receipt(
                load_immutable_json(Path(args.report)),
                provenance_path=args.provenance,
            )
        write_immutable_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENERGY_COMPARISON_SCHEMA",
    "ENERGY_EVIDENCE_SCHEMA",
    "ENERGY_UNAVAILABLE_METHOD",
    "GPU_BASELINE_CONTRACT_SCHEMA",
    "GPU_BASELINE_ENERGY_METER_PRIORITY",
    "GPU_BASELINE_ENERGY_SCOPE",
    "GPU_BASELINE_ENERGY_SOURCE",
    "GPU_BASELINE_EXECUTION_MODE",
    "GPU_BASELINE_REPORT_SCHEMA",
    "GPU_BASELINE_RESULT_SCHEMA",
    "GPU_BASELINE_SCOPE",
    "GPU_BASELINE_STAGE_RECEIPT_SCHEMA",
    "GPU_BASELINE_TIMING_SCOPE",
    "GPU_BASELINE_WORKSPACE_BINDING_SCHEMA",
    "GPU_ENERGY_MEASUREMENT_SCHEMA",
    "MEASURED_EVIDENCE_TIER",
    "NVIDIA_SMI_QUERY_FIELDS",
    "NVML_POWER_TRACE_METHOD",
    "NVML_TOTAL_ENERGY_METHOD",
    "PEAK_ROOFLINE_EVIDENCE_TIER",
    "THROUGHPUT_COMPARISON_SCHEMA",
    "THROUGHPUT_EVIDENCE_SCHEMA",
    "EnergyEvidenceRow",
    "GPUDeviceEnergyMeasurement",
    "GPUEnergyMeasurement",
    "GPUHardwareStateSnapshot",
    "GPUPowerTraceSample",
    "GPUBaselineContract",
    "GPUBaselinePrompt",
    "GPUBaselineRepetition",
    "GPUBaselineResult",
    "GPUBaselineWorkspaceBinding",
    "ThroughputEvidenceRow",
    "build_headline_energy_comparison",
    "build_headline_throughput_comparison",
    "build_gpu_baseline_contract",
    "build_gpu_baseline_report",
    "build_gpu_baseline_stage_receipt",
    "gpu_baseline_energy_evidence",
    "gpu_baseline_throughput_evidence",
    "load_gpu_baseline_workspace_binding",
    "main",
    "run_gpu_baseline",
    "validate_gpu_baseline_report",
    "validate_gpu_baseline_stage_receipt",
]
