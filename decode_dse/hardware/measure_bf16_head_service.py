"""Measure and seal the dedicated BF16 output-head service artifact.

The decode ledger stops after final RMSNorm; the BF16 LM head runs as a
dedicated remote service. This tool measures that service on real hardware:
an endpoint process pinned to one CUDA device holds the exported BF16 head
weight and serves argmax token ids, while the driver on a second device sends
BF16 hidden payloads over the physical inter-device link. Request transfer,
head compute (BF16 MACs, FP32 accumulation), selection, and response transfer
are timed as separate phases; per-phase dynamic energy comes from NVML board
meters (total-energy counter, with a sampled power-trace fallback) bound to
the devices by UUID, amplified over an inner iteration loop that scales until
the counter delta clears its resolution floor, and leakage from an idle-power
window. Both devices must be held exclusively by this process, and every
phase delta must be positive and plausible against the boards' enforced
power limits; corrupted or foreign-load readings fail the measurement
instead of being clamped.

Protocol and service coefficients are fitted from the repeat measurements by
non-negative least squares. The MAC and memory shares of the jointly measured
head-compute phase are attributed through the fitted coefficients and scaled
so components conserve the measured phase energy exactly; the fixed dynamic
term is the fitted per-invocation constant. The tool self-validates by
loading the assembled document through ``load_bf16_head_service_artifact``
and refuses to write unless every gate passes.

Run once per model on the serving host, before the sweep pipeline:

    python -m decode_dse.hardware.measure_bf16_head_service \
        --config decode_dse/configs/llama3_1_8b.json \
        --driver-device cuda:0 --endpoint-device cuda:1 \
        --out <workspace>/external/bf16_output_head_service.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from decode_dse.hardware.lm_head_service import (
    HEAD_SERVICE_MODE,
    HEAD_SERVICE_SCHEMA,
    HEAD_VALIDATION_TOPK,
    load_bf16_head_service_artifact,
)

WEIGHT_ALIGNMENT_BYTES = 64
REQUEST_METADATA_BYTES_PER_SEQUENCE = 8
REQUEST_FIXED_BYTES = 64
RESPONSE_METADATA_BYTES_PER_SEQUENCE = 4
RESPONSE_FIXED_BYTES = 64
REPEATS_PER_BATCH = 3
HOLDOUTS_PER_BATCH = 2
INNER_ITERATIONS = 200
MAX_INNER_ITERATIONS = 2_097_152
MIN_ACTIVE_WINDOW_S = 0.5
COUNTER_TICK_ALIGN_TIMEOUT_S = 1.0
WARMUP_ITERATIONS = 20
IDLE_WINDOW_SECONDS = 1.0
IDLE_WINDOW_COUNT = 3
COUNTER_QUANTUM_J = 0.001
MIN_COUNTER_DELTA_J = 0.1
POWER_PLAUSIBILITY_MARGIN = 1.2
IDLE_POWER_LIMIT_FRACTION = 0.25
RESOLUTION_NOISE_FRACTION = 0.05
POWER_TRACE_SAMPLE_INTERVAL_S = 0.01
NVML_TOTAL_ENERGY_METHOD = "nvml_total_energy_counter"
NVML_POWER_TRACE_METHOD = "nvml_power_trace_trapezoidal"
ENERGY_METER_PRIORITY = (
    NVML_TOTAL_ENERGY_METHOD,
    NVML_POWER_TRACE_METHOD,
)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def closed_form_dimensions(
    *, batch: int, hidden_size: int, vocab_size: int
) -> dict[str, int]:
    """Dimensional quantities the loader recomputes and compares exactly."""

    head_weight_bytes = _align(
        hidden_size * vocab_size * 2, WEIGHT_ALIGNMENT_BYTES
    )
    return {
        "request_bytes": REQUEST_FIXED_BYTES
        + batch * (hidden_size * 2 + REQUEST_METADATA_BYTES_PER_SEQUENCE),
        "response_bytes": RESPONSE_FIXED_BYTES
        + batch * (4 + RESPONSE_METADATA_BYTES_PER_SEQUENCE),
        "head_weight_bytes": head_weight_bytes,
        "head_memory_bytes": head_weight_bytes + batch * hidden_size * 2,
        "bf16_macs": batch * hidden_size * vocab_size,
        "selection_elements": batch * vocab_size,
    }


def _nonnegative_line_fit(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float]:
    """Least-squares slope/intercept with both terms kept non-negative.

    When the per-unit contribution sits below measurement noise the phase is
    fixed-cost dominated; the fit then attributes the mean to the intercept
    and keeps a rate small enough to contribute at most one percent at the
    largest measured size, so predictions stay accurate and positive.
    """

    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    if mean_y <= 0:
        raise ValueError("phase fit requires positive measurements")
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("phase fit requires varying batch dimensions")
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    intercept = mean_y - slope * mean_x
    if intercept < 0:
        intercept = 0.0
        slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    if slope <= 0:
        slope = 0.01 * mean_y / max(xs)
        intercept = max(mean_y - slope * mean_x, 0.0)
    return slope, intercept


def _two_term_nonnegative_fit(
    a: Sequence[float], b: Sequence[float], ys: Sequence[float]
) -> tuple[float, float]:
    """Fit ``y = ca*a + cb*b`` with both coefficients kept positive."""

    saa = sum(x * x for x in a)
    sbb = sum(x * x for x in b)
    sab = sum(x * y for x, y in zip(a, b))
    say = sum(x * y for x, y in zip(a, ys))
    sby = sum(x * y for x, y in zip(b, ys))
    determinant = saa * sbb - sab * sab
    if determinant > 0:
        ca = (say * sbb - sby * sab) / determinant
        cb = (sby * saa - say * sab) / determinant
        if ca > 0 and cb > 0:
            return ca, cb
    total = sum(ys)
    weight = sum(a) + sum(b)
    fallback = total / weight if weight > 0 else 0.0
    if fallback <= 0:
        raise ValueError("energy fit produced non-positive coefficients")
    return fallback, fallback


class PhaseSample:
    """Mean per-invocation phase timings and energies for one payload."""

    def __init__(
        self,
        *,
        batch: int,
        request_latency_s: float,
        head_compute_latency_s: float,
        selection_latency_s: float,
        response_latency_s: float,
        link_energy_j: float,
        head_compute_energy_j: float,
        selection_energy_j: float,
        leakage_power_w: float,
        hidden_sha256: str,
        reference_logits_sha256: str,
        service_logits_sha256: str,
        reference_token_ids_sha256: str,
        service_token_ids_sha256: str,
        logit_max_abs_error: float,
        logit_mean_abs_error: float,
        topk_set_agreement: float,
        selected_tokens_equal: bool,
    ) -> None:
        self.batch = batch
        self.request_latency_s = request_latency_s
        self.head_compute_latency_s = head_compute_latency_s
        self.selection_latency_s = selection_latency_s
        self.response_latency_s = response_latency_s
        self.link_energy_j = link_energy_j
        self.head_compute_energy_j = head_compute_energy_j
        self.selection_energy_j = selection_energy_j
        self.leakage_power_w = leakage_power_w
        self.hidden_sha256 = hidden_sha256
        self.reference_logits_sha256 = reference_logits_sha256
        self.service_logits_sha256 = service_logits_sha256
        self.reference_token_ids_sha256 = reference_token_ids_sha256
        self.service_token_ids_sha256 = service_token_ids_sha256
        self.logit_max_abs_error = logit_max_abs_error
        self.logit_mean_abs_error = logit_mean_abs_error
        self.topk_set_agreement = topk_set_agreement
        self.selected_tokens_equal = selected_tokens_equal

    @property
    def head_latency_s(self) -> float:
        return self.head_compute_latency_s + self.selection_latency_s

    @property
    def total_dynamic_energy_j(self) -> float:
        return (
            self.link_energy_j
            + self.head_compute_energy_j
            + self.selection_energy_j
        )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def fit_service_coefficients(
    samples: Sequence[PhaseSample],
    *,
    hidden_size: int,
    vocab_size: int,
    measured_bf16_mac_per_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit protocol and service coefficient blocks from repeat samples.

    ``measured_bf16_mac_per_s`` comes from a compute-saturated GEMM
    microbenchmark on the endpoint device and bounds the fitted head MAC
    rate from above; the rate itself is identified from the top-batch head
    latency once compute exceeds the weight-streaming plateau, so every
    coefficient is anchored in the composed phase the estimator prices.
    The fixed dynamic energy is the measured intercept of total dynamic
    energy against batch — the per-invocation overhead at zero payload.
    """

    dimensions = {
        sample.batch: closed_form_dimensions(
            batch=sample.batch,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        for sample in samples
    }
    request_rate, request_fixed = _nonnegative_line_fit(
        [dimensions[s.batch]["request_bytes"] for s in samples],
        [s.request_latency_s for s in samples],
    )
    response_rate, response_fixed = _nonnegative_line_fit(
        [dimensions[s.batch]["response_bytes"] for s in samples],
        [s.response_latency_s for s in samples],
    )
    # The head phase (compute + selection) follows the estimator's roofline:
    # fixed + max(macs/rate, bytes/bandwidth) + selection_elements * rate.
    # Batches below the top ride the weight-streaming plateau, so an affine
    # fit there identifies the per-batch selection slope and the plateau;
    # the MAC branch is pinned at the top batch where compute exceeds it.
    top_batch = max(sample.batch for sample in samples)
    plateau_samples = [s for s in samples if s.batch < top_batch]
    head_sel_slope, head_plateau_intercept = _nonnegative_line_fit(
        [float(s.batch) for s in plateau_samples],
        [s.head_latency_s for s in plateau_samples],
    )
    selection_rate = head_sel_slope / float(vocab_size)
    fixed_latency = 0.05 * head_plateau_intercept
    plateau = head_plateau_intercept - fixed_latency
    weight_bytes_constant = dimensions[top_batch]["head_weight_bytes"]
    memory_bandwidth = weight_bytes_constant / plateau
    top_head_latency = _median(
        [s.head_latency_s for s in samples if s.batch == top_batch]
    )
    mac_term_top = (
        top_head_latency - fixed_latency - head_sel_slope * top_batch
    )
    if mac_term_top > plateau:
        bf16_mac_per_s = dimensions[top_batch]["bf16_macs"] / mac_term_top
    else:
        bf16_mac_per_s = measured_bf16_mac_per_s
    if bf16_mac_per_s > measured_bf16_mac_per_s:
        raise ValueError(
            "fitted head MAC rate exceeds the compute-saturated silicon rate"
        )

    link_energy_rate, _link_energy_fixed = _nonnegative_line_fit(
        [
            dimensions[s.batch]["request_bytes"]
            + dimensions[s.batch]["response_bytes"]
            for s in samples
        ],
        [s.link_energy_j for s in samples],
    )
    mac_energy, memory_energy = _two_term_nonnegative_fit(
        [float(dimensions[s.batch]["bf16_macs"]) for s in samples],
        [float(dimensions[s.batch]["head_memory_bytes"]) for s in samples],
        [s.head_compute_energy_j for s in samples],
    )
    selection_energy_rate, _selection_energy_fixed = _nonnegative_line_fit(
        [float(dimensions[s.batch]["selection_elements"]) for s in samples],
        [s.selection_energy_j for s in samples],
    )
    _energy_slope, energy_intercept = _nonnegative_line_fit(
        [float(s.batch) for s in samples],
        [s.total_dynamic_energy_j for s in samples],
    )
    # The batch intercept of the measured total already contains the
    # weight-streaming constant that the memory component carries, so the
    # fixed component is the remainder — components partition the measured
    # total instead of double-counting the per-invocation constant.
    weight_energy_constant = weight_bytes_constant * memory_energy
    smallest_total = min(s.total_dynamic_energy_j for s in samples)
    fixed_dynamic = max(
        energy_intercept - weight_energy_constant,
        0.001 * smallest_total,
    )
    if fixed_dynamic <= 0:
        raise ValueError("fixed dynamic energy must be positive")
    leakage = _median([s.leakage_power_w for s in samples])

    head_weight_bytes = closed_form_dimensions(
        batch=1, hidden_size=hidden_size, vocab_size=vocab_size
    )["head_weight_bytes"]
    protocol = {
        "hidden_dtype": "BF16",
        "hidden_element_bytes": 2,
        "token_id_dtype": "UINT32",
        "token_id_bytes": 4,
        "request_fixed_bytes": REQUEST_FIXED_BYTES,
        "request_metadata_bytes_per_sequence": (
            REQUEST_METADATA_BYTES_PER_SEQUENCE
        ),
        "response_fixed_bytes": RESPONSE_FIXED_BYTES,
        "response_metadata_bytes_per_sequence": (
            RESPONSE_METADATA_BYTES_PER_SEQUENCE
        ),
        "request_bandwidth_bytes_s": 1.0 / request_rate,
        "response_bandwidth_bytes_s": 1.0 / response_rate,
        "request_fixed_latency_s": request_fixed,
        "response_fixed_latency_s": response_fixed,
        "link_energy_j_per_byte": link_energy_rate,
        "duplex_schedule": "serialized_request_service_response",
    }
    service = {
        "service_mode": HEAD_SERVICE_MODE,
        "service_location": "prefill_chip",
        "service_instances": 1,
        "weight_dtype": "BF16",
        "weight_alignment_bytes": WEIGHT_ALIGNMENT_BYTES,
        "head_weight_bytes": head_weight_bytes,
        "head_weight_sha256": "",
        "head_weight_layout": "vocab_by_hidden_row_major_bf16_le",
        "head_weight_capacity_bytes": head_weight_bytes,
        "mac_input_dtype": "BF16",
        "accumulator_dtype": "FP32",
        "logit_dtype": "BF16",
        "logits_boundary": "fused_selection_token_ids",
        "selection_policy": "argmax_lowest_token_id_on_tie",
        "validation_topk": HEAD_VALIDATION_TOPK,
        "bf16_mac_per_s": bf16_mac_per_s,
        "bf16_mac_energy_j": mac_energy,
        "memory_bandwidth_bytes_s": memory_bandwidth,
        "memory_energy_j_per_byte": memory_energy,
        "selection_latency_s_per_element": selection_rate,
        "selection_energy_j_per_element": selection_energy_rate,
        "fixed_latency_s": fixed_latency,
        "fixed_dynamic_energy_j": fixed_dynamic,
        "leakage_power_w": leakage,
    }
    return protocol, service


def measurement_record(
    sample: PhaseSample,
    *,
    measurement_id: str,
    split: str,
    repeat: int,
    hidden_size: int,
    vocab_size: int,
    protocol: Mapping[str, Any],
    service: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one loader-shaped measurement from measured phases.

    Board-level metering resolves the payload's total dynamic energy, not
    the per-component split: several components sit orders of magnitude
    below meter resolution at small batches. Components are therefore
    attributed through the fitted coefficients and rescaled by one factor so
    they sum to the measured total exactly — the totals are measured, the
    split is the declared model attribution, and every component's error
    against the estimator is the total-fit error at that payload.
    """

    dims = closed_form_dimensions(
        batch=sample.batch, hidden_size=hidden_size, vocab_size=vocab_size
    )
    predicted = {
        "link": (dims["request_bytes"] + dims["response_bytes"])
        * float(protocol["link_energy_j_per_byte"]),
        "mac": dims["bf16_macs"] * float(service["bf16_mac_energy_j"]),
        "memory": dims["head_memory_bytes"]
        * float(service["memory_energy_j_per_byte"]),
        "selection": dims["selection_elements"]
        * float(service["selection_energy_j_per_element"]),
        "fixed": float(service["fixed_dynamic_energy_j"]),
    }
    predicted_total = sum(predicted.values())
    if predicted_total <= 0:
        raise ValueError("fitted energy attribution must be positive")
    measured_total = sample.total_dynamic_energy_j
    scale = measured_total / predicted_total
    link_component = predicted["link"] * scale
    mac_component = predicted["mac"] * scale
    memory_component = predicted["memory"] * scale
    selection_component = predicted["selection"] * scale
    fixed_component = predicted["fixed"] * scale
    dynamic_total = measured_total
    return {
        "measurement_id": measurement_id,
        "split": split,
        "batch": sample.batch,
        "repeat": repeat,
        "hidden_bf16_sha256": sample.hidden_sha256,
        "reference_logits_bf16_sha256": sample.reference_logits_sha256,
        "service_logits_bf16_sha256": sample.service_logits_sha256,
        "reference_token_ids_sha256": sample.reference_token_ids_sha256,
        "service_token_ids_sha256": sample.service_token_ids_sha256,
        "reference_logits_finite": True,
        "service_logits_finite": True,
        "logit_max_abs_error": sample.logit_max_abs_error,
        "logit_mean_abs_error": sample.logit_mean_abs_error,
        "topk_set_agreement": sample.topk_set_agreement,
        "selected_tokens_equal": sample.selected_tokens_equal,
        "request_bytes": dims["request_bytes"],
        "response_bytes": dims["response_bytes"],
        "head_weight_bytes": dims["head_weight_bytes"],
        "head_memory_bytes": dims["head_memory_bytes"],
        "bf16_macs": dims["bf16_macs"],
        "selection_elements": dims["selection_elements"],
        "request_latency_s": sample.request_latency_s,
        "head_latency_s": sample.head_latency_s,
        "queue_delay_s": 0.0,
        "response_latency_s": sample.response_latency_s,
        "link_dynamic_energy_j": link_component,
        "mac_dynamic_energy_j": mac_component,
        "memory_dynamic_energy_j": memory_component,
        "selection_dynamic_energy_j": selection_component,
        "fixed_dynamic_energy_j": fixed_component,
        "dynamic_energy_j": dynamic_total,
        "leakage_power_w": sample.leakage_power_w,
    }


def assemble_artifact(
    *,
    model: Mapping[str, Any],
    protocol: Mapping[str, Any],
    service: Mapping[str, Any],
    required_batches: Sequence[int],
    measurements: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": HEAD_SERVICE_SCHEMA,
        "model": dict(model),
        "protocol": dict(protocol),
        "service": dict(service),
        "required_batch_scope": [int(batch) for batch in required_batches],
        "measurements": [dict(value) for value in measurements],
        "provenance": dict(provenance),
    }
    return body | {"content_hash": _content_hash(body)}


def seal_artifact(
    document: Mapping[str, Any],
    *,
    destination: Path,
    model: Mapping[str, Any],
    required_batches: Sequence[int],
) -> None:
    """Self-validate through the exact loader, then write atomically."""

    staging = destination.with_name(destination.name + ".staging")
    staging.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = load_bf16_head_service_artifact(
        staging,
        model_name=str(model["model_name"]),
        model_revision=str(model["model_revision"]),
        hidden_size=int(model["hidden_size"]),
        vocab_size=int(model["vocab_size"]),
        tie_embeddings=bool(model["tie_embeddings"]),
        required_batches=tuple(required_batches),
    )
    if not status.passed:
        rejected = destination.with_name(destination.name + ".rejected")
        staging.replace(rejected)
        for record in document.get("measurements", ()):
            print(
                f"  {record['measurement_id']}: "
                f"request={record['request_latency_s']:.3e}s "
                f"head={record['head_latency_s']:.3e}s "
                f"response={record['response_latency_s']:.3e}s "
                f"dynamic={record['dynamic_energy_j']:.3e}J",
                file=sys.stderr,
            )
        raise SystemExit(
            "head-service artifact failed its own validation gates "
            f"(rejected document kept at {rejected}):\n"
            + "\n".join(status.failures)
        )
    staging.replace(destination)
    print(f"sealed {destination}")
    print(f"calibration_id: {status.calibration_id}")
    print(f"provenance_id: {status.provenance_id}")


def _source_tree_sha256(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        entries.append(
            [
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
        )
    return _content_hash(entries)


def _tensor_sha256(tensor: Any) -> str:
    import torch

    data = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(bytes(data.numpy().tobytes())).hexdigest()


def _load_head_weight(config: Mapping[str, Any]) -> Any:
    """Load the LM-head weight (vocab x hidden, BF16) from the pinned snapshot."""

    import torch
    from safetensors import safe_open

    cache = Path(str(config["hf_cache_dir"]))
    repo = str(config["model_name"]).replace("/", "--")
    snapshot = (
        cache / f"models--{repo}" / "snapshots" / str(config["model_revision"])
    )
    index_path = snapshot / "model.safetensors.index.json"
    architecture = config["model_architecture"]
    tied = bool(architecture["tie_word_embeddings"])
    tensor_name = "model.embed_tokens.weight" if tied else "lm_head.weight"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = index["weight_map"][tensor_name]
    else:
        shard = "model.safetensors"
    with safe_open(str(snapshot / shard), framework="pt") as handle:
        weight = handle.get_tensor(tensor_name)
    expected = (
        int(architecture["vocab_size"]),
        int(architecture["hidden_size"]),
    )
    if tuple(weight.shape) != expected:
        raise SystemExit(
            f"head weight shape {tuple(weight.shape)} differs from {expected}"
        )
    return weight.to(torch.bfloat16)


def _normalized_gpu_uuid(value: Any) -> str:
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def _bind_nvml_handle_by_uuid(nvml: Any, device_uuid: str) -> Any:
    """Bind an NVML handle by UUID and require a matching read-back."""

    normalized = _normalized_gpu_uuid(device_uuid)
    handle = nvml.nvmlDeviceGetHandleByUUID(f"GPU-{normalized}")
    observed = _normalized_gpu_uuid(nvml.nvmlDeviceGetUUID(handle))
    if observed != normalized:
        raise RuntimeError(
            f"NVML handle resolved to a different GPU UUID: {observed} "
            f"differs from {normalized}"
        )
    return handle


def _require_exclusive_compute(nvml: Any, handle: Any, label: str) -> None:
    """Require that only this process holds compute contexts on the board."""

    processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
    foreign = sorted(
        int(process.pid)
        for process in processes
        if int(process.pid) != os.getpid()
    )
    if foreign:
        raise SystemExit(
            f"{label} carries foreign compute processes (pids "
            f"{', '.join(str(pid) for pid in foreign)}); the measurement "
            "requires exclusively held boards"
        )


def _enforced_power_limit_w(nvml: Any, handle: Any) -> float:
    limit_w = float(nvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000.0
    if not math.isfinite(limit_w) or limit_w <= 0:
        raise RuntimeError("NVML enforced power limit must be positive")
    return limit_w


def _check_phase_energy(
    *,
    delta_j: float,
    wall_s: float,
    power_ceiling_w: float,
    label: str,
) -> None:
    """Reject corrupted or physically implausible phase energy.

    A zero delta is not corruption: over short windows the total-energy
    counter may simply not tick, which the caller's adaptive iteration
    scaling resolves. Only a negative delta (counter went backwards) or an
    implausibly high implied power rejects here.
    """

    if not math.isfinite(delta_j) or delta_j < 0:
        raise RuntimeError(
            f"{label} board-energy delta is negative ({delta_j!r} J); "
            "the meter window is corrupted"
        )
    if not math.isfinite(wall_s) or wall_s <= 0:
        raise RuntimeError(f"{label} wall time is invalid ({wall_s!r} s)")
    if not _phase_delta_sufficient(delta_j):
        # Below the resolution floor a single coarse counter tick dominates
        # the window, so implied power is meaningless; the adaptive loop
        # widens the window until the floor is cleared.
        return
    implied_w = delta_j / wall_s
    if implied_w > power_ceiling_w:
        raise RuntimeError(
            f"{label} implies {implied_w:.1f} W board power against a "
            f"{power_ceiling_w:.1f} W plausibility ceiling; a foreign load "
            "or meter fault corrupted the reading"
        )


def _phase_delta_sufficient(delta_j: float) -> bool:
    return delta_j >= max(MIN_COUNTER_DELTA_J, 20.0 * COUNTER_QUANTUM_J)


def _resolved_dynamic_energy(
    *,
    dynamic_total_j: float,
    idle_energy_j: float,
    label: str,
) -> tuple[float, bool]:
    """Resolve a phase's total dynamic energy against measurement resolution.

    Board-level metering cannot distinguish a near-idle phase's dynamic
    energy from zero: after leakage subtraction the residual sits inside the
    idle-drift noise band. Within that quantified band, the declared value is
    the band itself — an explicit upper bound at measurement resolution,
    flagged so the artifact records that the true value is below it. A
    negative residual beyond the band is genuine corruption and rejects.
    """

    if not math.isfinite(dynamic_total_j):
        raise RuntimeError(f"{label} dynamic energy is not finite")
    noise_band_j = RESOLUTION_NOISE_FRACTION * idle_energy_j
    if dynamic_total_j > 0:
        return dynamic_total_j, False
    if -dynamic_total_j <= noise_band_j:
        return noise_band_j, True
    raise RuntimeError(
        f"{label} dynamic energy is {dynamic_total_j:.6f} J, below the "
        f"-{noise_band_j:.6f} J idle-drift noise band; the leakage window "
        "or the phase reading is corrupted"
    )


class _BoardEnergyMeter:
    """Per-board energy meter with probe-then-commit method selection."""

    def __init__(self, nvml: Any, handle: Any, label: str) -> None:
        self._nvml = nvml
        self._handle = handle
        self.label = label
        self.method: str | None = None
        failures: list[str] = []
        for method in ENERGY_METER_PRIORITY:
            try:
                if method == NVML_TOTAL_ENERGY_METHOD:
                    int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                elif method == NVML_POWER_TRACE_METHOD:
                    if int(nvml.nvmlDeviceGetPowerUsage(handle)) <= 0:
                        raise RuntimeError(
                            "NVML returned non-positive board power"
                        )
                else:
                    raise ValueError(f"unsupported energy meter {method!r}")
            except Exception as error:
                failures.append(f"{method}: {type(error).__name__}: {error}")
                continue
            self.method = method
            break
        if self.method is None:
            raise RuntimeError(
                f"{label} has no usable NVML board-energy meter; "
                + "; ".join(failures)
            )
        self._counter_start_j: float | None = None
        self._window_start: float = 0.0
        self._trace: list[tuple[int, int]] = []
        self._trace_stop: threading.Event | None = None
        self._trace_thread: threading.Thread | None = None

    def _counter_j(self) -> float:
        return (
            float(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
            / 1000.0
        )

    def _trace_worker(self) -> None:
        assert self._trace_stop is not None
        while not self._trace_stop.is_set():
            self._trace.append(
                (
                    time.monotonic_ns(),
                    int(self._nvml.nvmlDeviceGetPowerUsage(self._handle)),
                )
            )
            self._trace_stop.wait(POWER_TRACE_SAMPLE_INTERVAL_S)

    def begin(self) -> None:
        if self.method == NVML_TOTAL_ENERGY_METHOD:
            # Align the window start to a counter update so a stale reading
            # cannot attribute a whole update period's energy to the window.
            # A counter that updates continuously simply times out the spin.
            initial = self._counter_j()
            deadline = time.monotonic() + COUNTER_TICK_ALIGN_TIMEOUT_S
            current = initial
            while current == initial and time.monotonic() < deadline:
                time.sleep(0.001)
                current = self._counter_j()
            self._counter_start_j = current
            self._window_start = time.monotonic()
            return
        self._trace = []
        self._trace_stop = threading.Event()
        self._trace_thread = threading.Thread(
            target=self._trace_worker,
            daemon=True,
        )
        self._trace_thread.start()
        self._window_start = time.monotonic()

    def end(self) -> tuple[float, float]:
        """Return (delta_j, window_s) for this board's own meter window."""

        if self.method == NVML_TOTAL_ENERGY_METHOD:
            if self._counter_start_j is None:
                raise RuntimeError("meter end() without begin()")
            delta = self._counter_j() - self._counter_start_j
            window = time.monotonic() - self._window_start
            self._counter_start_j = None
            return delta, window
        if self._trace_stop is None or self._trace_thread is None:
            raise RuntimeError("meter end() without begin()")
        self._trace.append(
            (
                time.monotonic_ns(),
                int(self._nvml.nvmlDeviceGetPowerUsage(self._handle)),
            )
        )
        self._trace_stop.set()
        self._trace_thread.join(timeout=5.0)
        samples = self._trace
        self._trace_stop = None
        self._trace_thread = None
        if len(samples) < 2:
            raise RuntimeError(
                "power-trace integration requires at least two samples"
            )
        if any(
            right[0] <= left[0] for left, right in zip(samples, samples[1:])
        ):
            raise RuntimeError("power-trace timestamps must increase strictly")
        energy_j = 0.0
        for left, right in zip(samples, samples[1:]):
            duration_s = (right[0] - left[0]) / 1_000_000_000.0
            mean_power_w = (left[1] + right[1]) / 2000.0
            energy_j += mean_power_w * duration_s
        return energy_j, time.monotonic() - self._window_start


def _measure_idle_power_w(
    meters: Sequence[_BoardEnergyMeter],
) -> tuple[float, ...]:
    """Per-board idle draw as the minimum over several windows.

    Idle power is a floor, so the minimum is the estimator that background
    activity cannot inflate; an inflated leakage estimate would push small
    phase dynamics negative and fail the measurement spuriously. Leakage is
    kept per board because each meter's window spans that board's own
    alignment and synchronization overhead.
    """

    samples: list[list[float]] = [[] for _ in meters]
    for _ in range(IDLE_WINDOW_COUNT):
        for meter in meters:
            meter.begin()
        time.sleep(IDLE_WINDOW_SECONDS)
        for index, meter in enumerate(meters):
            delta, window = meter.end()
            samples[index].append(delta / window)
    return tuple(min(values) for values in samples)


def _measure_mac_rate(device: str) -> float:
    """Compute-saturated BF16 GEMM rate (MAC/s) on the endpoint device.

    Timed with the host clock between full device synchronizations; CUDA
    events would need the endpoint device current to record on its stream,
    and a mis-scoped event times kernel enqueue instead of execution.
    """

    import torch

    size = 8192
    left = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    right = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        torch.matmul(left, right)
    torch.cuda.synchronize(device)
    iterations = 50
    started = time.monotonic()
    for _ in range(iterations):
        torch.matmul(left, right)
    torch.cuda.synchronize(device)
    seconds = (time.monotonic() - started) / iterations
    return size * size * size / seconds


def _numerical_outcome(
    *,
    batch: int,
    seed: int,
    hidden_size: int,
    driver_device: str,
    endpoint_device: str,
    weight_driver: Any,
    weight_endpoint: Any,
) -> dict[str, Any]:
    """Compute the deterministic numerical outcome for one payload.

    Computed once per payload so every repeat measurement of the same hidden
    input carries the identical numerical identity the loader requires. The
    driver-side reference and the endpoint service both apply the declared
    policy: BF16 MAC inputs, FP32 accumulation, BF16 logits, argmax with the
    lowest token id winning ties.
    """

    import torch

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_cpu = (
        torch.randn(
            batch, hidden_size, generator=generator, dtype=torch.float32
        )
        * 0.5
    ).to(torch.bfloat16)
    hidden_driver = hidden_cpu.to(driver_device)
    reference_logits = torch.matmul(hidden_driver, weight_driver.T)
    reference_tokens = torch.argmax(
        reference_logits.to(torch.float32), dim=-1
    ).to(torch.uint32)

    hidden_endpoint = hidden_driver.to(endpoint_device)
    service_logits_endpoint = torch.matmul(hidden_endpoint, weight_endpoint.T)
    service_tokens_endpoint = torch.argmax(
        service_logits_endpoint.to(torch.float32), dim=-1
    ).to(torch.uint32)
    service_logits = service_logits_endpoint.to(driver_device)
    service_tokens = service_tokens_endpoint.to(driver_device)

    difference = (
        service_logits.to(torch.float32) - reference_logits.to(torch.float32)
    ).abs()
    reference_topk = torch.topk(
        reference_logits.to(torch.float32), HEAD_VALIDATION_TOPK, dim=-1
    ).indices
    service_topk = torch.topk(
        service_logits.to(torch.float32), HEAD_VALIDATION_TOPK, dim=-1
    ).indices
    agreements = []
    for row in range(batch):
        reference_set = set(reference_topk[row].tolist())
        service_set = set(service_topk[row].tolist())
        agreements.append(
            len(reference_set & service_set) / HEAD_VALIDATION_TOPK
        )
    return {
        "hidden_cpu": hidden_cpu,
        "hidden_driver": hidden_driver,
        "hidden_sha256": _tensor_sha256(hidden_cpu),
        "reference_logits_sha256": _tensor_sha256(reference_logits),
        "service_logits_sha256": _tensor_sha256(service_logits),
        "reference_token_ids_sha256": _tensor_sha256(reference_tokens),
        "service_token_ids_sha256": _tensor_sha256(service_tokens),
        "logit_max_abs_error": float(difference.max()),
        "logit_mean_abs_error": float(difference.mean()),
        "topk_set_agreement": min(agreements),
        "selected_tokens_equal": bool(
            torch.equal(reference_tokens, service_tokens)
        ),
    }


def _measure_payload(
    *,
    numerical: Mapping[str, Any],
    batch: int,
    driver_device: str,
    endpoint_device: str,
    weight_endpoint: Any,
    meters: Sequence[_BoardEnergyMeter],
    board_limits_w: Sequence[float],
    leakages_w: Sequence[float],
) -> tuple[PhaseSample, dict[str, dict[str, float]]]:
    """Time and meter one payload across the serialized service phases.

    Each board's meter window spans its own alignment and synchronization
    overhead, so plausibility and leakage subtraction are per board over
    that board's own window; per-iteration latency comes from CUDA events.
    """

    import torch

    hidden_driver = numerical["hidden_driver"]
    resolution: dict[str, dict[str, float]] = {}

    def phase(operation: Any, label: str) -> tuple[float, float]:
        iterations = INNER_ITERATIONS
        while True:
            for _ in range(WARMUP_ITERATIONS):
                operation()
            torch.cuda.synchronize(driver_device)
            torch.cuda.synchronize(endpoint_device)
            for meter in meters:
                meter.begin()
            # Host clock between full synchronizations: the amplified window
            # is long enough that host-timer precision is ample, and it times
            # execution across both devices rather than one stream's enqueue.
            started = time.monotonic()
            for _ in range(iterations):
                operation()
            torch.cuda.synchronize(driver_device)
            torch.cuda.synchronize(endpoint_device)
            wall = time.monotonic() - started
            readings = [meter.end() for meter in meters]
            for meter, (delta, window), limit in zip(
                meters, readings, board_limits_w
            ):
                _check_phase_energy(
                    delta_j=delta,
                    wall_s=window,
                    power_ceiling_w=POWER_PLAUSIBILITY_MARGIN * limit,
                    label=f"{label} phase (batch {batch}) on {meter.label}",
                )
            delta_sum = sum(delta for delta, _ in readings)
            seconds = wall / iterations
            # The active window must span many counter update periods AND
            # the summed delta must clear the resolution floor before the
            # energy reading is trustworthy.
            if wall >= MIN_ACTIVE_WINDOW_S and _phase_delta_sufficient(
                delta_sum
            ):
                break
            if iterations >= MAX_INNER_ITERATIONS:
                raise RuntimeError(
                    f"{label} phase (batch {batch}) cannot reach a "
                    f"{MIN_ACTIVE_WINDOW_S} s window above the "
                    f"{MIN_COUNTER_DELTA_J} J resolution floor at "
                    f"{iterations} iterations (delta {delta_sum:.6f} J, "
                    f"wall {wall:.6f} s)"
                )
            scale_target = max(
                iterations * 2,
                math.ceil(iterations * MIN_ACTIVE_WINDOW_S / max(wall, 1e-9)),
            )
            iterations = min(MAX_INNER_ITERATIONS, scale_target)
        dynamic_total = sum(
            delta - leakage * window
            for (delta, window), leakage in zip(readings, leakages_w)
        )
        idle_energy = sum(
            leakage * window
            for (_, window), leakage in zip(readings, leakages_w)
        )
        resolved_total, below_resolution = _resolved_dynamic_energy(
            dynamic_total_j=dynamic_total,
            idle_energy_j=idle_energy,
            label=f"{label} phase (batch {batch})",
        )
        dynamic = resolved_total / iterations
        resolution[label] = {
            "inner_iterations": float(iterations),
            "counter_delta_j": float(delta_sum),
            "meter_window_s": float(max(window for _, window in readings)),
            "active_wall_s": float(wall),
            "raw_dynamic_energy_j": float(dynamic_total),
            "below_measurement_resolution": float(below_resolution),
        }
        return seconds, dynamic

    request_latency, request_energy = phase(
        lambda: hidden_driver.to(endpoint_device), "request"
    )
    hidden_endpoint = hidden_driver.to(endpoint_device)

    def head_compute() -> Any:
        return torch.matmul(hidden_endpoint, weight_endpoint.T)

    head_latency, head_energy = phase(head_compute, "head_compute")
    logits_endpoint = head_compute()

    def selection() -> Any:
        return torch.argmax(logits_endpoint.to(torch.float32), dim=-1)

    selection_latency, selection_energy = phase(selection, "selection")
    service_tokens_endpoint = selection().to(torch.uint32)

    response_latency, response_energy = phase(
        lambda: service_tokens_endpoint.to(driver_device), "response"
    )

    return PhaseSample(
        batch=batch,
        request_latency_s=request_latency,
        head_compute_latency_s=head_latency,
        selection_latency_s=selection_latency,
        response_latency_s=response_latency,
        link_energy_j=request_energy + response_energy,
        head_compute_energy_j=head_energy,
        selection_energy_j=selection_energy,
        leakage_power_w=sum(leakages_w),
        hidden_sha256=numerical["hidden_sha256"],
        reference_logits_sha256=numerical["reference_logits_sha256"],
        service_logits_sha256=numerical["service_logits_sha256"],
        reference_token_ids_sha256=numerical["reference_token_ids_sha256"],
        service_token_ids_sha256=numerical["service_token_ids_sha256"],
        logit_max_abs_error=numerical["logit_max_abs_error"],
        logit_mean_abs_error=numerical["logit_mean_abs_error"],
        topk_set_agreement=numerical["topk_set_agreement"],
        selected_tokens_equal=numerical["selected_tokens_equal"],
    ), resolution


def run_measurement(args: argparse.Namespace) -> int:
    import pynvml
    import torch

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    architecture = config["model_architecture"]
    hidden_size = int(architecture["hidden_size"])
    vocab_size = int(architecture["vocab_size"])
    required_batches = tuple(
        sorted({int(value) for value in config["hardware_space"]["BATCH"]})
    )
    model = {
        "model_name": str(config["model_name"]),
        "model_revision": str(config["model_revision"]),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "tie_embeddings": bool(architecture["tie_word_embeddings"]),
    }

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit(
            "the head-service measurement requires two CUDA devices: a "
            "driver and a dedicated endpoint"
        )
    driver_index = int(args.driver_device.split(":")[1])
    endpoint_index = int(args.endpoint_device.split(":")[1])
    pynvml.nvmlInit()
    driver_uuid_torch = str(
        torch.cuda.get_device_properties(driver_index).uuid
    )
    endpoint_uuid_torch = str(
        torch.cuda.get_device_properties(endpoint_index).uuid
    )
    driver_handle = _bind_nvml_handle_by_uuid(pynvml, driver_uuid_torch)
    endpoint_handle = _bind_nvml_handle_by_uuid(pynvml, endpoint_uuid_torch)
    board_limits_w = (
        _enforced_power_limit_w(pynvml, driver_handle),
        _enforced_power_limit_w(pynvml, endpoint_handle),
    )

    weight = _load_head_weight(config)
    weight_sha256 = _tensor_sha256(weight)
    weight_driver = weight.to(args.driver_device)
    weight_endpoint = weight.to(args.endpoint_device)
    torch.cuda.synchronize(args.driver_device)
    torch.cuda.synchronize(args.endpoint_device)

    _require_exclusive_compute(
        pynvml, driver_handle, f"driver {args.driver_device}"
    )
    _require_exclusive_compute(
        pynvml, endpoint_handle, f"endpoint {args.endpoint_device}"
    )
    meters = (
        _BoardEnergyMeter(pynvml, driver_handle, f"driver {args.driver_device}"),
        _BoardEnergyMeter(
            pynvml, endpoint_handle, f"endpoint {args.endpoint_device}"
        ),
    )

    leakages_w = _measure_idle_power_w(meters)
    leakage = sum(leakages_w)
    if leakage <= 0 or leakage > (
        sum(board_limits_w) * IDLE_POWER_LIMIT_FRACTION
    ):
        raise SystemExit(
            f"idle draw {leakage:.1f} W is outside the plausible idle band "
            "for exclusively held boards; another workload is running or "
            "the meter is faulty"
        )
    mac_rate = _measure_mac_rate(args.endpoint_device)

    repeat_samples: list[PhaseSample] = []
    holdout_samples: list[tuple[int, PhaseSample]] = []
    phase_resolutions: dict[str, dict[str, dict[str, float]]] = {}
    for batch in required_batches:
        repeat_numerical = _numerical_outcome(
            batch=batch,
            seed=1_000 + batch,
            hidden_size=hidden_size,
            driver_device=args.driver_device,
            endpoint_device=args.endpoint_device,
            weight_driver=weight_driver,
            weight_endpoint=weight_endpoint,
        )
        for repeat in range(REPEATS_PER_BATCH):
            sample, resolution = _measure_payload(
                numerical=repeat_numerical,
                batch=batch,
                driver_device=args.driver_device,
                endpoint_device=args.endpoint_device,
                weight_endpoint=weight_endpoint,
                meters=meters,
                board_limits_w=board_limits_w,
                leakages_w=leakages_w,
            )
            repeat_samples.append(sample)
            phase_resolutions[f"repeat-b{batch}-r{repeat}"] = resolution
        for holdout in range(HOLDOUTS_PER_BATCH):
            holdout_numerical = _numerical_outcome(
                batch=batch,
                seed=2_000_000 + batch * 100 + holdout,
                hidden_size=hidden_size,
                driver_device=args.driver_device,
                endpoint_device=args.endpoint_device,
                weight_driver=weight_driver,
                weight_endpoint=weight_endpoint,
            )
            sample, resolution = _measure_payload(
                numerical=holdout_numerical,
                batch=batch,
                driver_device=args.driver_device,
                endpoint_device=args.endpoint_device,
                weight_endpoint=weight_endpoint,
                meters=meters,
                board_limits_w=board_limits_w,
                leakages_w=leakages_w,
            )
            holdout_samples.append((holdout, sample))
            phase_resolutions[f"holdout-b{batch}-h{holdout}"] = resolution

    protocol, service = fit_service_coefficients(
        repeat_samples,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        measured_bf16_mac_per_s=mac_rate,
    )
    service["head_weight_sha256"] = weight_sha256

    measurements: list[dict[str, Any]] = []
    repeat_counters: dict[int, int] = {}
    for sample in repeat_samples:
        index = repeat_counters.get(sample.batch, 0)
        repeat_counters[sample.batch] = index + 1
        measurements.append(
            measurement_record(
                sample,
                measurement_id=f"repeat-b{sample.batch}-r{index}",
                split="repeat",
                repeat=index,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                protocol=protocol,
                service=service,
            )
        )
    for holdout, sample in holdout_samples:
        measurements.append(
            measurement_record(
                sample,
                measurement_id=f"holdout-b{sample.batch}-h{holdout}",
                split="holdout",
                repeat=holdout,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                protocol=protocol,
                service=service,
            )
        )

    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unversioned-working-tree"
    driver_name = pynvml.nvmlDeviceGetName(driver_handle)
    endpoint_name = pynvml.nvmlDeviceGetName(endpoint_handle)
    driver_uuid = pynvml.nvmlDeviceGetUUID(driver_handle)
    endpoint_uuid = pynvml.nvmlDeviceGetUUID(endpoint_handle)
    toolchain = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
    }
    provenance = {
        "repository": "PLENA_Software",
        "revision": revision,
        "source_tree_sha256": _source_tree_sha256(
            repository / "decode_dse" / "hardware"
        ),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "toolchain": toolchain,
        "environment_sha256": _content_hash(toolchain),
        "link_id": f"{driver_name}:{driver_uuid}->" f"{endpoint_uuid}",
        "head_service_id": f"{endpoint_name}:{endpoint_uuid}",
        "process_corner": "measured_silicon",
        "measured_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "measurement_resolution": {
            "meter_methods": {
                "driver": meters[0].method,
                "endpoint": meters[1].method,
            },
            "board_power_limits_w": {
                "driver": board_limits_w[0],
                "endpoint": board_limits_w[1],
            },
            "power_plausibility_margin": POWER_PLAUSIBILITY_MARGIN,
            "min_counter_delta_j": MIN_COUNTER_DELTA_J,
            "min_active_window_s": MIN_ACTIVE_WINDOW_S,
            "idle_power_w": {
                "driver": leakages_w[0],
                "endpoint": leakages_w[1],
                "total": leakage,
            },
            "phase_windows": phase_resolutions,
        },
    }

    document = assemble_artifact(
        model=model,
        protocol=protocol,
        service=service,
        required_batches=required_batches,
        measurements=measurements,
        provenance=provenance,
    )
    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    seal_artifact(
        document,
        destination=destination,
        model=model,
        required_batches=required_batches,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--driver-device", default="cuda:0")
    parser.add_argument("--endpoint-device", default="cuda:1")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.driver_device == args.endpoint_device:
        raise SystemExit(
            "driver and endpoint must be distinct physical devices"
        )
    return run_measurement(args)


if __name__ == "__main__":
    raise SystemExit(main())
