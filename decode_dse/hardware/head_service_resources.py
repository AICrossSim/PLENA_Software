"""Content-addressed resource boundary for a remote BF16 output head."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from decode_dse.hardware.lm_head_service import (
    BF16HeadServiceStatus,
    HEAD_SERVICE_MODE,
    require_content_addressed_id,
)

HEAD_RESOURCE_SCHEMA = "bf16-output-head-endpoint-resources/v1"
HEAD_RESOURCE_INPUT_SCHEMA = "bf16-output-head-endpoint-resource-input/v1"
HEAD_RESOURCE_DEPLOYMENT_SCOPE = (
    "prefill_endpoint_with_bf16_head_service_fully_accounted"
)
MEASUREMENT_DRIVER_ROLE = "instrumentation_only_not_deployed"
AREA_COMPARISON_BASIS = (
    "aggregate_physical_compute_silicon_area_mm2_unscaled_excludes_hbm"
)
HBM_CAPACITY_BASIS = "installed_endpoint_capacity_bytes"
HBM_BANDWIDTH_BASIS = "vendor_peak_theoretical_bytes_per_s"
DECODER_INTERFACE_ENERGY_SCOPE = (
    "decoder_request_response_interface_only_excludes_endpoint"
)
DEPLOYMENT_LINK_TIMING_SCOPE = (
    "plena_decoder_to_prefill_endpoint_bound_interface"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_INPUT_FIELDS = {
    "schema_version",
    "deployment_scope",
    "service_instances",
    "endpoint_instances",
    "endpoint_resources_included_once",
    "endpoint_shared_with_decoder",
    "endpoint_shared_with_prefill",
    "decoder_resources_included",
    "prefill_resources_included",
    "measurement_driver_role",
    "measurement_driver_resources_included",
    "endpoint_device_name",
    "endpoint_device_uuid",
    "endpoint_aggregate_compute_silicon_area_mm2",
    "endpoint_compute_die_count",
    "endpoint_hbm_capacity_bytes",
    "endpoint_hbm_bandwidth_bytes_per_s",
    "prefill_resident_bytes",
    "head_resident_bytes",
    "endpoint_runtime_reserve_bytes",
    "decoder_interface_energy_j_per_byte",
    "decoder_interface_energy_scope",
    "deployment_request_bandwidth_bytes_s",
    "deployment_response_bandwidth_bytes_s",
    "deployment_link_peak_bandwidth_bytes_s",
    "deployment_request_fixed_latency_s",
    "deployment_response_fixed_latency_s",
    "deployment_link_timing_scope",
    "measurement_driver_timing_used",
    "area_comparison_basis",
    "hbm_capacity_basis",
    "hbm_bandwidth_basis",
    "source",
}
_SOURCE_FIELDS = {
    "publisher",
    "title",
    "revision",
    "locator",
    "retrieved_at_utc",
    "area_basis_statement",
    "deployment_link_basis_statement",
}


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


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _validate_source(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        raise ValueError("endpoint resource source fields differ")
    result = {name: str(value[name]) for name in _SOURCE_FIELDS}
    if any(not item for item in result.values()):
        raise ValueError("endpoint resource source fields must be non-empty")
    if not result["retrieved_at_utc"].endswith("Z"):
        raise ValueError("endpoint resource retrieval time must be UTC")
    return result


def validate_endpoint_resource_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the manually curated, cited physical-resource input."""

    raw = dict(value)
    if set(raw) != _INPUT_FIELDS:
        raise ValueError("endpoint resource input fields differ")
    if raw["schema_version"] != HEAD_RESOURCE_INPUT_SCHEMA:
        raise ValueError("endpoint resource input schema differs")
    exact = {
        "deployment_scope": HEAD_RESOURCE_DEPLOYMENT_SCOPE,
        "service_instances": 1,
        "endpoint_instances": 1,
        "endpoint_resources_included_once": True,
        "endpoint_shared_with_decoder": False,
        "endpoint_shared_with_prefill": True,
        "decoder_resources_included": False,
        "prefill_resources_included": True,
        "measurement_driver_role": MEASUREMENT_DRIVER_ROLE,
        "measurement_driver_resources_included": False,
        "area_comparison_basis": AREA_COMPARISON_BASIS,
        "hbm_capacity_basis": HBM_CAPACITY_BASIS,
        "hbm_bandwidth_basis": HBM_BANDWIDTH_BASIS,
        "decoder_interface_energy_scope": DECODER_INTERFACE_ENERGY_SCOPE,
        "deployment_link_timing_scope": DEPLOYMENT_LINK_TIMING_SCOPE,
        "measurement_driver_timing_used": False,
    }
    for name, expected in exact.items():
        if raw[name] != expected:
            raise ValueError(f"endpoint resource input {name} differs")
    for name in ("endpoint_device_name", "endpoint_device_uuid"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError(f"endpoint resource input {name} is empty")
    raw["endpoint_aggregate_compute_silicon_area_mm2"] = _positive_float(
        raw["endpoint_aggregate_compute_silicon_area_mm2"],
        "endpoint_aggregate_compute_silicon_area_mm2",
    )
    raw["endpoint_compute_die_count"] = _positive_int(
        raw["endpoint_compute_die_count"], "endpoint_compute_die_count"
    )
    raw["endpoint_hbm_capacity_bytes"] = _positive_int(
        raw["endpoint_hbm_capacity_bytes"], "endpoint_hbm_capacity_bytes"
    )
    raw["endpoint_hbm_bandwidth_bytes_per_s"] = _positive_float(
        raw["endpoint_hbm_bandwidth_bytes_per_s"],
        "endpoint_hbm_bandwidth_bytes_per_s",
    )
    raw["decoder_interface_energy_j_per_byte"] = _positive_float(
        raw["decoder_interface_energy_j_per_byte"],
        "decoder_interface_energy_j_per_byte",
    )
    for name in (
        "deployment_request_bandwidth_bytes_s",
        "deployment_response_bandwidth_bytes_s",
        "deployment_link_peak_bandwidth_bytes_s",
    ):
        raw[name] = _positive_float(raw[name], name)
    if raw["deployment_link_peak_bandwidth_bytes_s"] > raw[
        "endpoint_hbm_bandwidth_bytes_per_s"
    ]:
        raise ValueError("deployment link peak exceeds endpoint HBM bandwidth")
    if any(
        raw[name] > raw["deployment_link_peak_bandwidth_bytes_s"]
        for name in (
            "deployment_request_bandwidth_bytes_s",
            "deployment_response_bandwidth_bytes_s",
        )
    ):
        raise ValueError("deployment transfer rate exceeds cited link peak")
    for name in (
        "deployment_request_fixed_latency_s",
        "deployment_response_fixed_latency_s",
    ):
        raw[name] = _positive_float(raw[name], name)
    raw["prefill_resident_bytes"] = _positive_int(
        raw["prefill_resident_bytes"], "prefill_resident_bytes"
    )
    raw["head_resident_bytes"] = _positive_int(
        raw["head_resident_bytes"], "head_resident_bytes"
    )
    if (
        isinstance(raw["endpoint_runtime_reserve_bytes"], bool)
        or not isinstance(raw["endpoint_runtime_reserve_bytes"], int)
        or raw["endpoint_runtime_reserve_bytes"] < 0
    ):
        raise TypeError("endpoint_runtime_reserve_bytes must be non-negative")
    required = (
        raw["prefill_resident_bytes"]
        + raw["head_resident_bytes"]
        + raw["endpoint_runtime_reserve_bytes"]
    )
    if required > raw["endpoint_hbm_capacity_bytes"]:
        raise ValueError("prefill, head, and runtime residency exceed endpoint HBM")
    raw["source"] = _validate_source(raw["source"])
    return raw


def qwen3_moe_bf16_parameter_census(
    architecture: Mapping[str, Any],
) -> dict[str, int | str | bool]:
    """Derive the exact untied Qwen3-MoE BF16 residency from architecture."""

    if architecture.get("model_type") != "qwen3_moe":
        raise ValueError("parameter census requires qwen3_moe architecture")
    if architecture.get("tie_word_embeddings") is not False:
        raise ValueError("parameter census requires an untied output head")
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    )
    values = {name: _positive_int(architecture[name], name) for name in required}
    h = values["hidden_size"]
    layers = values["num_hidden_layers"]
    query = values["num_attention_heads"] * values["head_dim"]
    kv = values["num_key_value_heads"] * values["head_dim"]
    embedding = values["vocab_size"] * h
    lm_head = values["vocab_size"] * h
    attention = h * query + 2 * h * kv + query * h
    qk_norm = 2 * values["head_dim"]
    layer_norm = 2 * h
    router = h * values["num_experts"]
    experts = (
        values["num_experts"]
        * 3
        * h
        * values["moe_intermediate_size"]
    )
    per_layer = attention + qk_norm + layer_norm + router + experts
    final_norm = h
    total = embedding + lm_head + layers * per_layer + final_norm
    excluding_head = total - lm_head
    return {
        "schema_version": "qwen3-moe-parameter-census/v1",
        "tie_word_embeddings": False,
        "embedding_parameters": embedding,
        "lm_head_parameters": lm_head,
        "attention_parameters_per_layer": attention,
        "qk_norm_parameters_per_layer": qk_norm,
        "decoder_norm_parameters_per_layer": layer_norm,
        "router_parameters_per_layer": router,
        "expert_parameters_per_layer": experts,
        "decoder_parameters_per_layer": per_layer,
        "final_norm_parameters": final_norm,
        "total_parameters": total,
        "prefill_model_excluding_lm_head_parameters": excluding_head,
        "bf16_total_resident_bytes": total * 2,
        "prefill_model_excluding_lm_head_bf16_bytes": excluding_head * 2,
        "lm_head_bf16_bytes": lm_head * 2,
    }


@dataclass(frozen=True)
class BF16HeadEndpointResourceReceipt:
    """Verified resources of the deployed endpoint, excluding instrumentation."""

    source_path: Path
    artifact_sha256: str
    content_hash: str
    receipt_id: str
    head_service_artifact_sha256: str
    head_service_calibration_id: str
    head_service_provenance_id: str
    endpoint_device_name: str
    endpoint_device_uuid: str
    endpoint_aggregate_compute_silicon_area_mm2: float
    endpoint_compute_die_count: int
    endpoint_hbm_capacity_bytes: int
    endpoint_hbm_bandwidth_bytes_per_s: float
    prefill_resident_bytes: int
    head_resident_bytes: int
    endpoint_runtime_reserve_bytes: int
    decoder_interface_energy_j_per_byte: float
    deployment_request_bandwidth_bytes_s: float
    deployment_response_bandwidth_bytes_s: float
    deployment_link_peak_bandwidth_bytes_s: float
    deployment_request_fixed_latency_s: float
    deployment_response_fixed_latency_s: float
    input_artifact_sha256: str
    specification_artifact_sha256: str
    source: Mapping[str, str]

    def to_status(self) -> dict[str, Any]:
        return {
            "schema_version": HEAD_RESOURCE_SCHEMA,
            "artifact_sha256": self.artifact_sha256,
            "content_hash": self.content_hash,
            "receipt_id": self.receipt_id,
            "passed": True,
            "failures": [],
            "head_service_artifact_sha256": (
                self.head_service_artifact_sha256
            ),
            "head_service_calibration_id": self.head_service_calibration_id,
            "head_service_provenance_id": self.head_service_provenance_id,
            "service_mode": HEAD_SERVICE_MODE,
            "service_location": "prefill_chip",
            "deployment_scope": HEAD_RESOURCE_DEPLOYMENT_SCOPE,
            "service_instances": 1,
            "endpoint_instances": 1,
            "endpoint_resources_included_once": True,
            "endpoint_shared_with_decoder": False,
            "endpoint_shared_with_prefill": True,
            "decoder_resources_included": False,
            "prefill_resources_included": True,
            "measurement_driver_role": MEASUREMENT_DRIVER_ROLE,
            "measurement_driver_resources_included": False,
            "endpoint": {
                "device_name": self.endpoint_device_name,
                "device_uuid": self.endpoint_device_uuid,
                "aggregate_compute_silicon_area_mm2": (
                    self.endpoint_aggregate_compute_silicon_area_mm2
                ),
                "compute_die_count": self.endpoint_compute_die_count,
                "hbm_capacity_bytes": self.endpoint_hbm_capacity_bytes,
                "hbm_bandwidth_bytes_per_s": (
                    self.endpoint_hbm_bandwidth_bytes_per_s
                ),
                "prefill_resident_bytes": self.prefill_resident_bytes,
                "head_resident_bytes": self.head_resident_bytes,
                "runtime_reserve_bytes": self.endpoint_runtime_reserve_bytes,
                "resident_total_bytes": (
                    self.prefill_resident_bytes
                    + self.head_resident_bytes
                    + self.endpoint_runtime_reserve_bytes
                ),
                "area_comparison_basis": AREA_COMPARISON_BASIS,
                "hbm_capacity_basis": HBM_CAPACITY_BASIS,
                "hbm_bandwidth_basis": HBM_BANDWIDTH_BASIS,
            },
            "model_residency": {
                "precision": "BF16",
                "prefill_model_excluding_lm_head_bytes": (
                    self.prefill_resident_bytes
                ),
                "lm_head_bytes": self.head_resident_bytes,
                "untied_lm_head_counted_once": True,
            },
            "composed_link_energy": {
                "decoder_interface_energy_j_per_byte": (
                    self.decoder_interface_energy_j_per_byte
                ),
                "decoder_interface_energy_scope": (
                    DECODER_INTERFACE_ENERGY_SCOPE
                ),
                "endpoint_interface_energy_scope": (
                    "endpoint_receive_transmit_incremental_only"
                ),
                "measurement_driver_dynamic_included": False,
                "complete": True,
            },
            "deployment_link_timing": {
                "request_bandwidth_bytes_s": (
                    self.deployment_request_bandwidth_bytes_s
                ),
                "response_bandwidth_bytes_s": (
                    self.deployment_response_bandwidth_bytes_s
                ),
                "link_peak_bandwidth_bytes_s": (
                    self.deployment_link_peak_bandwidth_bytes_s
                ),
                "request_fixed_latency_s": (
                    self.deployment_request_fixed_latency_s
                ),
                "response_fixed_latency_s": (
                    self.deployment_response_fixed_latency_s
                ),
                "scope": DEPLOYMENT_LINK_TIMING_SCOPE,
                "measurement_driver_timing_used": False,
                "complete": True,
            },
            "evidence": {
                "input_artifact_sha256": self.input_artifact_sha256,
                "specification_artifact_sha256": (
                    self.specification_artifact_sha256
                ),
                "source": dict(self.source),
            },
        }


@dataclass(frozen=True)
class BF16HeadEndpointResourceStatus:
    """Fail-closed result of loading an endpoint-resource receipt."""

    source_path: Path
    artifact_sha256: str
    failures: tuple[str, ...]
    receipt: BF16HeadEndpointResourceReceipt | None

    @property
    def passed(self) -> bool:
        return not self.failures and self.receipt is not None

    def to_dict(self) -> dict[str, Any]:
        if self.receipt is not None:
            return self.receipt.to_status()
        return {
            "schema_version": HEAD_RESOURCE_SCHEMA,
            "artifact_sha256": self.artifact_sha256 or None,
            "content_hash": None,
            "receipt_id": None,
            "passed": False,
            "failures": list(self.failures),
            "head_service_artifact_sha256": None,
            "head_service_calibration_id": None,
            "head_service_provenance_id": None,
            "service_mode": "unmodeled",
            "service_location": None,
            "deployment_scope": None,
            "service_instances": 0,
            "endpoint_instances": 0,
            "endpoint_resources_included_once": False,
            "endpoint_shared_with_decoder": None,
            "endpoint_shared_with_prefill": None,
            "decoder_resources_included": None,
            "prefill_resources_included": None,
            "measurement_driver_role": None,
            "measurement_driver_resources_included": None,
            "endpoint": None,
            "model_residency": None,
            "composed_link_energy": None,
            "deployment_link_timing": None,
            "evidence": None,
        }


def assemble_endpoint_resource_receipt(
    resource_input: Mapping[str, Any],
    *,
    input_artifact_sha256: str,
    specification_artifact_sha256: str,
    head_service_status: BF16HeadServiceStatus,
    prefill_model_excluding_head_bytes: int,
) -> dict[str, Any]:
    """Bind cited endpoint resources to one exact measured head service."""

    if not head_service_status.passed or head_service_status.calibration is None:
        raise ValueError("endpoint resources require a passing head service")
    raw = validate_endpoint_resource_input(resource_input)
    expected_prefill = _positive_int(
        prefill_model_excluding_head_bytes,
        "prefill_model_excluding_head_bytes",
    )
    if raw["prefill_resident_bytes"] != expected_prefill:
        raise ValueError(
            "prefill residency differs from the config-derived BF16 model "
            "excluding its untied LM head"
        )
    for name, digest in (
        ("input_artifact_sha256", input_artifact_sha256),
        ("specification_artifact_sha256", specification_artifact_sha256),
    ):
        if not _SHA256.fullmatch(str(digest)):
            raise ValueError(f"{name} is invalid")
    calibration = head_service_status.calibration
    service_id = str(calibration.provenance["head_service_id"])
    expected_name, separator, expected_uuid = service_id.rpartition(":")
    if not separator:
        raise ValueError("head-service endpoint identity is not separable")
    if (
        raw["endpoint_device_name"] != expected_name
        or raw["endpoint_device_uuid"] != expected_uuid
    ):
        raise ValueError("resource input and measured endpoint identities differ")
    if raw["head_resident_bytes"] != int(
        calibration.service["head_weight_capacity_bytes"]
    ):
        raise ValueError("resource input and calibrated head residency differ")
    if raw["endpoint_hbm_bandwidth_bytes_per_s"] < float(
        calibration.service["memory_bandwidth_bytes_s"]
    ):
        raise ValueError("endpoint peak HBM bandwidth is below measured service rate")
    body = {
        "schema_version": HEAD_RESOURCE_SCHEMA,
        "head_service": {
            "artifact_sha256": head_service_status.artifact_sha256,
            "calibration_id": head_service_status.calibration_id,
            "provenance_id": head_service_status.provenance_id,
            "service_mode": HEAD_SERVICE_MODE,
            "service_location": "prefill_chip",
        },
        "model_residency": {
            "precision": "BF16",
            "prefill_model_excluding_lm_head_bytes": expected_prefill,
            "lm_head_bytes": raw["head_resident_bytes"],
            "untied_lm_head_counted_once": True,
        },
        "deployment": {
            name: raw[name]
            for name in (
                "deployment_scope",
                "service_instances",
                "endpoint_instances",
                "endpoint_resources_included_once",
                "endpoint_shared_with_decoder",
                "endpoint_shared_with_prefill",
                "decoder_resources_included",
                "prefill_resources_included",
                "measurement_driver_role",
                "measurement_driver_resources_included",
            )
        },
        "endpoint": {
            "device_name": raw["endpoint_device_name"],
            "device_uuid": raw["endpoint_device_uuid"],
            "aggregate_compute_silicon_area_mm2": raw[
                "endpoint_aggregate_compute_silicon_area_mm2"
            ],
            "compute_die_count": raw["endpoint_compute_die_count"],
            "hbm_capacity_bytes": raw["endpoint_hbm_capacity_bytes"],
            "hbm_bandwidth_bytes_per_s": raw[
                "endpoint_hbm_bandwidth_bytes_per_s"
            ],
            "prefill_resident_bytes": raw["prefill_resident_bytes"],
            "head_resident_bytes": raw["head_resident_bytes"],
            "runtime_reserve_bytes": raw["endpoint_runtime_reserve_bytes"],
            "area_comparison_basis": raw["area_comparison_basis"],
            "hbm_capacity_basis": raw["hbm_capacity_basis"],
            "hbm_bandwidth_basis": raw["hbm_bandwidth_basis"],
        },
        "composed_link_energy": {
            "decoder_interface_energy_j_per_byte": raw[
                "decoder_interface_energy_j_per_byte"
            ],
            "decoder_interface_energy_scope": raw[
                "decoder_interface_energy_scope"
            ],
            "endpoint_interface_energy_scope": (
                "endpoint_receive_transmit_incremental_only"
            ),
            "measurement_driver_dynamic_included": False,
            "complete": True,
        },
        "deployment_link_timing": {
            "request_bandwidth_bytes_s": raw[
                "deployment_request_bandwidth_bytes_s"
            ],
            "response_bandwidth_bytes_s": raw[
                "deployment_response_bandwidth_bytes_s"
            ],
            "link_peak_bandwidth_bytes_s": raw[
                "deployment_link_peak_bandwidth_bytes_s"
            ],
            "request_fixed_latency_s": raw[
                "deployment_request_fixed_latency_s"
            ],
            "response_fixed_latency_s": raw[
                "deployment_response_fixed_latency_s"
            ],
            "scope": raw["deployment_link_timing_scope"],
            "measurement_driver_timing_used": raw[
                "measurement_driver_timing_used"
            ],
            "complete": True,
        },
        "evidence": {
            "input_artifact_sha256": str(input_artifact_sha256),
            "specification_artifact_sha256": str(
                specification_artifact_sha256
            ),
            "source": raw["source"],
        },
    }
    return body | {"content_hash": _content_hash(body)}


def load_bf16_head_endpoint_resource_receipt(
    path: str | Path,
    *,
    head_service_status: BF16HeadServiceStatus,
    prefill_model_excluding_head_bytes: int,
) -> BF16HeadEndpointResourceStatus:
    """Load a receipt and rebind every field to the measured endpoint."""

    source_path = Path(path).resolve()
    artifact_sha256 = ""
    failures: list[str] = []
    receipt: BF16HeadEndpointResourceReceipt | None = None
    try:
        payload = source_path.read_bytes()
        artifact_sha256 = hashlib.sha256(payload).hexdigest()
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(raw, Mapping):
            raise TypeError("endpoint resource receipt root must be an object")
        if set(raw) != {
            "schema_version",
            "head_service",
            "model_residency",
            "deployment",
            "endpoint",
            "composed_link_energy",
            "deployment_link_timing",
            "evidence",
            "content_hash",
        }:
            raise ValueError("endpoint resource receipt fields differ")
        body = {name: raw[name] for name in raw if name != "content_hash"}
        content_hash = str(raw["content_hash"])
        if content_hash != _content_hash(body):
            raise ValueError("endpoint resource receipt content hash differs")
        if raw["schema_version"] != HEAD_RESOURCE_SCHEMA:
            raise ValueError("endpoint resource receipt schema differs")
        if not head_service_status.passed or head_service_status.calibration is None:
            raise ValueError("endpoint resource receipt requires head evidence")
        head = raw["head_service"]
        if not isinstance(head, Mapping) or set(head) != {
            "artifact_sha256",
            "calibration_id",
            "provenance_id",
            "service_mode",
            "service_location",
        }:
            raise ValueError("endpoint resource head binding fields differ")
        expected_head = {
            "artifact_sha256": head_service_status.artifact_sha256,
            "calibration_id": head_service_status.calibration_id,
            "provenance_id": head_service_status.provenance_id,
            "service_mode": HEAD_SERVICE_MODE,
            "service_location": "prefill_chip",
        }
        if dict(head) != expected_head:
            raise ValueError("endpoint resource and head-service identities differ")
        expected_prefill = _positive_int(
            prefill_model_excluding_head_bytes,
            "prefill_model_excluding_head_bytes",
        )
        if raw["model_residency"] != {
            "precision": "BF16",
            "prefill_model_excluding_lm_head_bytes": expected_prefill,
            "lm_head_bytes": raw["endpoint"].get("head_resident_bytes"),
            "untied_lm_head_counted_once": True,
        }:
            raise ValueError("endpoint resource model residency differs")
        deployment = raw["deployment"]
        if not isinstance(deployment, Mapping):
            raise ValueError("endpoint resource deployment must be an object")
        endpoint = raw["endpoint"]
        composed_link = raw["composed_link_energy"]
        deployment_timing = raw["deployment_link_timing"]
        evidence = raw["evidence"]
        synthetic_input = {
            "schema_version": HEAD_RESOURCE_INPUT_SCHEMA,
            **dict(deployment),
            "endpoint_device_name": endpoint.get("device_name"),
            "endpoint_device_uuid": endpoint.get("device_uuid"),
            "endpoint_aggregate_compute_silicon_area_mm2": endpoint.get(
                "aggregate_compute_silicon_area_mm2"
            ),
            "endpoint_compute_die_count": endpoint.get("compute_die_count"),
            "endpoint_hbm_capacity_bytes": endpoint.get("hbm_capacity_bytes"),
            "endpoint_hbm_bandwidth_bytes_per_s": endpoint.get(
                "hbm_bandwidth_bytes_per_s"
            ),
            "prefill_resident_bytes": endpoint.get("prefill_resident_bytes"),
            "head_resident_bytes": endpoint.get("head_resident_bytes"),
            "endpoint_runtime_reserve_bytes": endpoint.get(
                "runtime_reserve_bytes"
            ),
            "decoder_interface_energy_j_per_byte": (
                composed_link.get("decoder_interface_energy_j_per_byte")
                if isinstance(composed_link, Mapping)
                else None
            ),
            "decoder_interface_energy_scope": (
                composed_link.get("decoder_interface_energy_scope")
                if isinstance(composed_link, Mapping)
                else None
            ),
            "deployment_request_bandwidth_bytes_s": (
                deployment_timing.get("request_bandwidth_bytes_s")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "deployment_response_bandwidth_bytes_s": (
                deployment_timing.get("response_bandwidth_bytes_s")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "deployment_link_peak_bandwidth_bytes_s": (
                deployment_timing.get("link_peak_bandwidth_bytes_s")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "deployment_request_fixed_latency_s": (
                deployment_timing.get("request_fixed_latency_s")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "deployment_response_fixed_latency_s": (
                deployment_timing.get("response_fixed_latency_s")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "deployment_link_timing_scope": (
                deployment_timing.get("scope")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "measurement_driver_timing_used": (
                deployment_timing.get("measurement_driver_timing_used")
                if isinstance(deployment_timing, Mapping)
                else None
            ),
            "area_comparison_basis": endpoint.get("area_comparison_basis"),
            "hbm_capacity_basis": endpoint.get("hbm_capacity_basis"),
            "hbm_bandwidth_basis": endpoint.get("hbm_bandwidth_basis"),
            "source": evidence.get("source") if isinstance(evidence, Mapping) else None,
        }
        validated = validate_endpoint_resource_input(synthetic_input)
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "input_artifact_sha256",
            "specification_artifact_sha256",
            "source",
        }:
            raise ValueError("endpoint resource evidence fields differ")
        for name in ("input_artifact_sha256", "specification_artifact_sha256"):
            if not _SHA256.fullmatch(str(evidence[name])):
                raise ValueError(f"endpoint resource {name} is invalid")
        rebound = assemble_endpoint_resource_receipt(
            validated,
            input_artifact_sha256=str(evidence["input_artifact_sha256"]),
            specification_artifact_sha256=str(
                evidence["specification_artifact_sha256"]
            ),
            head_service_status=head_service_status,
            prefill_model_excluding_head_bytes=expected_prefill,
        )
        if rebound != dict(raw):
            raise ValueError("endpoint resource receipt is not canonical")
        receipt_id = "bf16-head-endpoint-resources-" + _content_hash(
            {
                "artifact_sha256": artifact_sha256,
                "content_hash": content_hash,
                "head_service_calibration_id": head_service_status.calibration_id,
            }
        )
        receipt = BF16HeadEndpointResourceReceipt(
            source_path=source_path,
            artifact_sha256=artifact_sha256,
            content_hash=content_hash,
            receipt_id=receipt_id,
            head_service_artifact_sha256=head_service_status.artifact_sha256,
            head_service_calibration_id=str(head_service_status.calibration_id),
            head_service_provenance_id=str(head_service_status.provenance_id),
            endpoint_device_name=validated["endpoint_device_name"],
            endpoint_device_uuid=validated["endpoint_device_uuid"],
            endpoint_aggregate_compute_silicon_area_mm2=validated[
                "endpoint_aggregate_compute_silicon_area_mm2"
            ],
            endpoint_compute_die_count=validated["endpoint_compute_die_count"],
            endpoint_hbm_capacity_bytes=validated[
                "endpoint_hbm_capacity_bytes"
            ],
            endpoint_hbm_bandwidth_bytes_per_s=validated[
                "endpoint_hbm_bandwidth_bytes_per_s"
            ],
            prefill_resident_bytes=validated["prefill_resident_bytes"],
            head_resident_bytes=validated["head_resident_bytes"],
            endpoint_runtime_reserve_bytes=validated[
                "endpoint_runtime_reserve_bytes"
            ],
            decoder_interface_energy_j_per_byte=validated[
                "decoder_interface_energy_j_per_byte"
            ],
            deployment_request_bandwidth_bytes_s=validated[
                "deployment_request_bandwidth_bytes_s"
            ],
            deployment_response_bandwidth_bytes_s=validated[
                "deployment_response_bandwidth_bytes_s"
            ],
            deployment_link_peak_bandwidth_bytes_s=validated[
                "deployment_link_peak_bandwidth_bytes_s"
            ],
            deployment_request_fixed_latency_s=validated[
                "deployment_request_fixed_latency_s"
            ],
            deployment_response_fixed_latency_s=validated[
                "deployment_response_fixed_latency_s"
            ],
            input_artifact_sha256=str(evidence["input_artifact_sha256"]),
            specification_artifact_sha256=str(
                evidence["specification_artifact_sha256"]
            ),
            source=validated["source"],
        )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    return BF16HeadEndpointResourceStatus(
        source_path=source_path,
        artifact_sha256=artifact_sha256,
        failures=tuple(failures),
        receipt=receipt,
    )


def head_endpoint_resource_status_valid(value: Mapping[str, Any]) -> bool:
    """Validate the self-contained status serialized into hardware rows."""

    try:
        if (
            value.get("schema_version") != HEAD_RESOURCE_SCHEMA
            or value.get("passed") is not True
            or value.get("failures") != []
            or not _SHA256.fullmatch(str(value.get("artifact_sha256", "")))
            or not _SHA256.fullmatch(str(value.get("content_hash", "")))
            or value.get("service_mode") != HEAD_SERVICE_MODE
            or value.get("service_location") != "prefill_chip"
            or value.get("deployment_scope") != HEAD_RESOURCE_DEPLOYMENT_SCOPE
            or value.get("service_instances") != 1
            or value.get("endpoint_instances") != 1
            or value.get("endpoint_resources_included_once") is not True
            or value.get("endpoint_shared_with_decoder") is not False
            or value.get("endpoint_shared_with_prefill") is not True
            or value.get("decoder_resources_included") is not False
            or value.get("prefill_resources_included") is not True
            or value.get("measurement_driver_role") != MEASUREMENT_DRIVER_ROLE
            or value.get("measurement_driver_resources_included") is not False
        ):
            return False
        require_content_addressed_id(
            "endpoint resource receipt",
            value.get("receipt_id"),
            prefix="bf16-head-endpoint-resources-",
        )
        require_content_addressed_id(
            "head calibration",
            value.get("head_service_calibration_id"),
            prefix="bf16-head-service-",
        )
        require_content_addressed_id(
            "head provenance",
            value.get("head_service_provenance_id"),
            prefix="bf16-head-provenance-",
        )
        if not _SHA256.fullmatch(
            str(value.get("head_service_artifact_sha256", ""))
        ):
            return False
        endpoint = value.get("endpoint")
        model_residency = value.get("model_residency")
        composed_link = value.get("composed_link_energy")
        deployment_timing = value.get("deployment_link_timing")
        evidence = value.get("evidence")
        if (
            not isinstance(endpoint, Mapping)
            or not isinstance(model_residency, Mapping)
            or not isinstance(composed_link, Mapping)
            or not isinstance(deployment_timing, Mapping)
            or not isinstance(evidence, Mapping)
        ):
            return False
        if endpoint.get("area_comparison_basis") != AREA_COMPARISON_BASIS:
            return False
        if endpoint.get("hbm_capacity_basis") != HBM_CAPACITY_BASIS:
            return False
        if endpoint.get("hbm_bandwidth_basis") != HBM_BANDWIDTH_BASIS:
            return False
        _positive_float(
            endpoint.get("aggregate_compute_silicon_area_mm2"),
            "endpoint.aggregate_compute_silicon_area_mm2",
        )
        _positive_int(
            endpoint.get("compute_die_count"), "endpoint.compute_die_count"
        )
        _positive_int(
            endpoint.get("hbm_capacity_bytes"), "endpoint.hbm_capacity_bytes"
        )
        _positive_float(
            endpoint.get("hbm_bandwidth_bytes_per_s"),
            "endpoint.hbm_bandwidth_bytes_per_s",
        )
        if (
            deployment_timing.get("scope") != DEPLOYMENT_LINK_TIMING_SCOPE
            or deployment_timing.get("measurement_driver_timing_used")
            is not False
            or deployment_timing.get("complete") is not True
        ):
            return False
        for name in (
            "request_bandwidth_bytes_s",
            "response_bandwidth_bytes_s",
            "link_peak_bandwidth_bytes_s",
        ):
            _positive_float(deployment_timing.get(name), name)
        if deployment_timing["link_peak_bandwidth_bytes_s"] > endpoint[
            "hbm_bandwidth_bytes_per_s"
        ]:
            return False
        if any(
            deployment_timing[name]
            > deployment_timing["link_peak_bandwidth_bytes_s"]
            for name in (
                "request_bandwidth_bytes_s",
                "response_bandwidth_bytes_s",
            )
        ):
            return False
        for name in ("request_fixed_latency_s", "response_fixed_latency_s"):
            _positive_float(deployment_timing.get(name), name)
        prefill = _positive_int(
            endpoint.get("prefill_resident_bytes"),
            "endpoint.prefill_resident_bytes",
        )
        head = _positive_int(
            endpoint.get("head_resident_bytes"),
            "endpoint.head_resident_bytes",
        )
        runtime = endpoint.get("runtime_reserve_bytes")
        if isinstance(runtime, bool) or not isinstance(runtime, int) or runtime < 0:
            return False
        if endpoint.get("resident_total_bytes") != prefill + head + runtime:
            return False
        if endpoint["resident_total_bytes"] > endpoint["hbm_capacity_bytes"]:
            return False
        if model_residency != {
            "precision": "BF16",
            "prefill_model_excluding_lm_head_bytes": prefill,
            "lm_head_bytes": head,
            "untied_lm_head_counted_once": True,
        }:
            return False
        if (
            composed_link.get("decoder_interface_energy_scope")
            != DECODER_INTERFACE_ENERGY_SCOPE
            or composed_link.get("endpoint_interface_energy_scope")
            != "endpoint_receive_transmit_incremental_only"
            or composed_link.get("measurement_driver_dynamic_included")
            is not False
            or composed_link.get("complete") is not True
        ):
            return False
        _positive_float(
            composed_link.get("decoder_interface_energy_j_per_byte"),
            "composed_link_energy.decoder_interface_energy_j_per_byte",
        )
        if not endpoint.get("device_name") or not endpoint.get("device_uuid"):
            return False
        for name in ("input_artifact_sha256", "specification_artifact_sha256"):
            if not _SHA256.fullmatch(str(evidence.get(name, ""))):
                return False
        _validate_source(evidence.get("source"))
        return True
    except (TypeError, ValueError):
        return False


__all__ = [
    "AREA_COMPARISON_BASIS",
    "DECODER_INTERFACE_ENERGY_SCOPE",
    "DEPLOYMENT_LINK_TIMING_SCOPE",
    "BF16HeadEndpointResourceReceipt",
    "BF16HeadEndpointResourceStatus",
    "HBM_BANDWIDTH_BASIS",
    "HBM_CAPACITY_BASIS",
    "HEAD_RESOURCE_DEPLOYMENT_SCOPE",
    "HEAD_RESOURCE_INPUT_SCHEMA",
    "HEAD_RESOURCE_SCHEMA",
    "MEASUREMENT_DRIVER_ROLE",
    "assemble_endpoint_resource_receipt",
    "head_endpoint_resource_status_valid",
    "load_bf16_head_endpoint_resource_receipt",
    "validate_endpoint_resource_input",
]
