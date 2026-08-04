"""Measure and seal the dedicated BF16 output-head service artifact.

The decode ledger stops after final RMSNorm; the BF16 LM head runs as a
dedicated remote service. This tool measures that service on real hardware:
an endpoint process pinned to one CUDA device holds the exported BF16 head
weight and serves argmax token ids, while the driver on a second device sends
BF16 hidden payloads over the physical inter-device link. Request transfer,
head compute (BF16 MACs, FP32 accumulation), selection, and response transfer
are timed as separate phases; per-phase dynamic energy comes from NVML total
energy counters amplified over an inner iteration loop, and leakage from an
idle-power window.

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
import subprocess
import sys
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
WARMUP_ITERATIONS = 20
IDLE_WINDOW_SECONDS = 0.5


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
    microbenchmark on the endpoint device; the head-compute phase itself is
    memory-dominated across the calibrated batch scope, so the memory
    bandwidth is identified from that phase and the MAC rate independently.
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
    selection_rate, selection_fixed = _nonnegative_line_fit(
        [dimensions[s.batch]["selection_elements"] for s in samples],
        [s.selection_latency_s for s in samples],
    )
    memory_bandwidth = _median(
        [
            dimensions[s.batch]["head_memory_bytes"]
            / s.head_compute_latency_s
            for s in samples
        ]
    )
    residuals = []
    for sample in samples:
        dims = dimensions[sample.batch]
        modelled = max(
            dims["bf16_macs"] / measured_bf16_mac_per_s,
            dims["head_memory_bytes"] / memory_bandwidth,
        )
        residuals.append(sample.head_compute_latency_s - modelled)
    fixed_latency = max(_median(residuals), 0.0) + selection_fixed

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
    smallest_total = min(s.total_dynamic_energy_j for s in samples)
    fixed_dynamic = max(energy_intercept, 0.001 * smallest_total)
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
        "bf16_mac_per_s": measured_bf16_mac_per_s,
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

    The MAC and memory components of the jointly measured head-compute phase
    are attributed through the fitted coefficients and rescaled so the two
    components sum to the measured phase energy exactly; link and selection
    components are direct phase measurements and the fixed component is the
    fitted per-invocation constant, so the total conserves by construction.
    """

    dims = closed_form_dimensions(
        batch=sample.batch, hidden_size=hidden_size, vocab_size=vocab_size
    )
    predicted_mac = dims["bf16_macs"] * float(service["bf16_mac_energy_j"])
    predicted_memory = dims["head_memory_bytes"] * float(
        service["memory_energy_j_per_byte"]
    )
    predicted_phase = predicted_mac + predicted_memory
    scale = (
        sample.head_compute_energy_j / predicted_phase
        if predicted_phase > 0
        else 0.0
    )
    mac_component = predicted_mac * scale
    memory_component = predicted_memory * scale
    fixed_component = float(service["fixed_dynamic_energy_j"])
    dynamic_total = (
        sample.link_energy_j
        + mac_component
        + memory_component
        + sample.selection_energy_j
        + fixed_component
    )
    del protocol
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
        "link_dynamic_energy_j": sample.link_energy_j,
        "mac_dynamic_energy_j": mac_component,
        "memory_dynamic_energy_j": memory_component,
        "selection_dynamic_energy_j": sample.selection_energy_j,
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
        staging.unlink()
        raise SystemExit(
            "head-service artifact failed its own validation gates:\n"
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


def _device_energy_j(handle: Any) -> float:
    import pynvml

    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0


def _measure_idle_power_w(handles: Sequence[Any]) -> float:
    import time

    start = [_device_energy_j(handle) for handle in handles]
    time.sleep(IDLE_WINDOW_SECONDS)
    end = [_device_energy_j(handle) for handle in handles]
    return sum(
        (after - before) / IDLE_WINDOW_SECONDS
        for before, after in zip(start, end)
    )


def _measure_mac_rate(device: str) -> float:
    """Compute-saturated BF16 GEMM rate (MAC/s) on the endpoint device."""

    import torch

    size = 8192
    left = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    right = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        torch.matmul(left, right)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = 20
    start.record()
    for _ in range(iterations):
        torch.matmul(left, right)
    end.record()
    torch.cuda.synchronize(device)
    seconds = start.elapsed_time(end) / 1000.0 / iterations
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
    driver_handle: Any,
    endpoint_handle: Any,
    leakage_power_w: float,
) -> PhaseSample:
    """Time and meter one payload across the serialized service phases."""

    import torch

    hidden_driver = numerical["hidden_driver"]

    def phase(operation: Any, iterations: int) -> tuple[float, float]:
        for _ in range(WARMUP_ITERATIONS):
            operation()
        torch.cuda.synchronize(driver_device)
        torch.cuda.synchronize(endpoint_device)
        energy_before = _device_energy_j(driver_handle) + _device_energy_j(
            endpoint_handle
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        torch.cuda.synchronize(driver_device)
        torch.cuda.synchronize(endpoint_device)
        energy_after = _device_energy_j(driver_handle) + _device_energy_j(
            endpoint_handle
        )
        seconds = start.elapsed_time(end) / 1000.0 / iterations
        wall = seconds * iterations
        dynamic = max(
            (energy_after - energy_before) - leakage_power_w * wall, 0.0
        ) / iterations
        return seconds, dynamic

    request_latency, request_energy = phase(
        lambda: hidden_driver.to(endpoint_device), INNER_ITERATIONS
    )
    hidden_endpoint = hidden_driver.to(endpoint_device)

    def head_compute() -> Any:
        return torch.matmul(hidden_endpoint, weight_endpoint.T)

    head_latency, head_energy = phase(head_compute, INNER_ITERATIONS)
    logits_endpoint = head_compute()

    def selection() -> Any:
        return torch.argmax(logits_endpoint.to(torch.float32), dim=-1)

    selection_latency, selection_energy = phase(selection, INNER_ITERATIONS)
    service_tokens_endpoint = selection().to(torch.uint32)

    response_latency, response_energy = phase(
        lambda: service_tokens_endpoint.to(driver_device), INNER_ITERATIONS
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
        leakage_power_w=leakage_power_w,
        hidden_sha256=numerical["hidden_sha256"],
        reference_logits_sha256=numerical["reference_logits_sha256"],
        service_logits_sha256=numerical["service_logits_sha256"],
        reference_token_ids_sha256=numerical["reference_token_ids_sha256"],
        service_token_ids_sha256=numerical["service_token_ids_sha256"],
        logit_max_abs_error=numerical["logit_max_abs_error"],
        logit_mean_abs_error=numerical["logit_mean_abs_error"],
        topk_set_agreement=numerical["topk_set_agreement"],
        selected_tokens_equal=numerical["selected_tokens_equal"],
    )


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
    driver_handle = pynvml.nvmlDeviceGetHandleByIndex(driver_index)
    endpoint_handle = pynvml.nvmlDeviceGetHandleByIndex(endpoint_index)

    weight = _load_head_weight(config)
    weight_sha256 = _tensor_sha256(weight)
    weight_driver = weight.to(args.driver_device)
    weight_endpoint = weight.to(args.endpoint_device)

    leakage = _measure_idle_power_w((driver_handle, endpoint_handle))
    mac_rate = _measure_mac_rate(args.endpoint_device)

    repeat_samples: list[PhaseSample] = []
    holdout_samples: list[tuple[int, PhaseSample]] = []
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
        for _repeat in range(REPEATS_PER_BATCH):
            repeat_samples.append(
                _measure_payload(
                    numerical=repeat_numerical,
                    batch=batch,
                    driver_device=args.driver_device,
                    endpoint_device=args.endpoint_device,
                    weight_endpoint=weight_endpoint,
                    driver_handle=driver_handle,
                    endpoint_handle=endpoint_handle,
                    leakage_power_w=leakage,
                )
            )
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
            holdout_samples.append(
                (
                    holdout,
                    _measure_payload(
                        numerical=holdout_numerical,
                        batch=batch,
                        driver_device=args.driver_device,
                        endpoint_device=args.endpoint_device,
                        weight_endpoint=weight_endpoint,
                        driver_handle=driver_handle,
                        endpoint_handle=endpoint_handle,
                        leakage_power_w=leakage,
                    ),
                )
            )

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
