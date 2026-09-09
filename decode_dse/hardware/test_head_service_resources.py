"""Strict endpoint-resource receipt regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from decode_dse.hardware.head_service_resources import (
    HEAD_RESOURCE_DEPLOYMENT_SCOPE,
    HEAD_RESOURCE_INPUT_SCHEMA,
    assemble_endpoint_resource_receipt,
    head_endpoint_resource_status_valid,
    load_bf16_head_endpoint_resource_receipt,
    qwen3_moe_bf16_parameter_census,
)
from decode_dse.hardware.lm_head_service import (
    load_bf16_head_service_artifact,
)
from decode_dse.hardware.test_evaluation_contracts import (
    _assembled_synthetic_head_artifact,
)


def _fixture(tmp_path):
    head_path, model = _assembled_synthetic_head_artifact(tmp_path)
    head = load_bf16_head_service_artifact(
        head_path,
        model_name=model["model_name"],
        model_revision=model["model_revision"],
        hidden_size=model["hidden_size"],
        vocab_size=model["vocab_size"],
        tie_embeddings=model["tie_embeddings"],
        required_batches=(1, 4, 8),
    )
    assert head.passed and head.calibration is not None
    head_bytes = int(head.calibration.service["head_weight_capacity_bytes"])
    resource_input = {
        "schema_version": HEAD_RESOURCE_INPUT_SCHEMA,
        "deployment_scope": HEAD_RESOURCE_DEPLOYMENT_SCOPE,
        "service_instances": 1,
        "endpoint_instances": 1,
        "endpoint_resources_included_once": True,
        "endpoint_shared_with_decoder": False,
        "endpoint_shared_with_prefill": True,
        "decoder_resources_included": False,
        "prefill_resources_included": True,
        "measurement_driver_role": "instrumentation_only_not_deployed",
        "measurement_driver_resources_included": False,
        "endpoint_device_name": "Synthetic Accelerator",
        "endpoint_device_uuid": "GPU-SYNTHETIC",
        "endpoint_aggregate_compute_silicon_area_mm2": 100.0,
        "endpoint_compute_die_count": 2,
        "endpoint_hbm_capacity_bytes": head_bytes * 4,
        "endpoint_hbm_bandwidth_bytes_per_s": (
            float(head.calibration.service["memory_bandwidth_bytes_s"]) * 2
        ),
        "prefill_resident_bytes": head_bytes,
        "head_resident_bytes": head_bytes,
        "endpoint_runtime_reserve_bytes": head_bytes,
        "decoder_interface_energy_j_per_byte": 1e-12,
        "decoder_interface_energy_scope": (
            "decoder_request_response_interface_only_excludes_endpoint"
        ),
        "deployment_request_bandwidth_bytes_s": 1e9,
        "deployment_response_bandwidth_bytes_s": 1e9,
        "deployment_link_peak_bandwidth_bytes_s": 1e9,
        "deployment_request_fixed_latency_s": 1e-6,
        "deployment_response_fixed_latency_s": 1e-6,
        "deployment_link_timing_scope": (
            "plena_decoder_to_prefill_endpoint_bound_interface"
        ),
        "measurement_driver_timing_used": False,
        "area_comparison_basis": (
            "aggregate_physical_compute_silicon_area_mm2_unscaled_excludes_hbm"
        ),
        "hbm_capacity_basis": "installed_endpoint_capacity_bytes",
        "hbm_bandwidth_basis": "vendor_peak_theoretical_bytes_per_s",
        "source": {
            "publisher": "Synthetic vendor",
            "title": "Synthetic retained specification",
            "revision": "1",
            "locator": "retained://synthetic-specification.txt",
            "retrieved_at_utc": "2026-08-20T00:00:00Z",
            "area_basis_statement": (
                "two compute dies, aggregate compute silicon, HBM excluded"
            ),
            "deployment_link_basis_statement": (
                "bound decoder-to-endpoint interface timing and energy"
            ),
        },
    }
    digest = hashlib.sha256(b"retained evidence").hexdigest()
    document = assemble_endpoint_resource_receipt(
        resource_input,
        input_artifact_sha256=digest,
        specification_artifact_sha256=digest,
        head_service_status=head,
        prefill_model_excluding_head_bytes=head_bytes,
    )
    receipt_path = tmp_path / "endpoint_resources.json"
    receipt_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    return head, receipt_path, resource_input


def test_endpoint_resource_receipt_binds_prefill_and_excludes_driver(tmp_path):
    head, receipt_path, resource_input = _fixture(tmp_path)
    status = load_bf16_head_endpoint_resource_receipt(
        receipt_path,
        head_service_status=head,
        prefill_model_excluding_head_bytes=resource_input[
            "prefill_resident_bytes"
        ],
    )
    assert status.passed
    serialized = status.to_dict()
    assert head_endpoint_resource_status_valid(serialized)
    assert serialized["prefill_resources_included"] is True
    assert serialized["measurement_driver_resources_included"] is False
    assert serialized["endpoint"]["resident_total_bytes"] <= serialized[
        "endpoint"
    ]["hbm_capacity_bytes"]


def test_endpoint_resource_receipt_rejects_content_tampering(tmp_path):
    head, receipt_path, resource_input = _fixture(tmp_path)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    value["endpoint"]["aggregate_compute_silicon_area_mm2"] += 1
    receipt_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    status = load_bf16_head_endpoint_resource_receipt(
        receipt_path,
        head_service_status=head,
        prefill_model_excluding_head_bytes=resource_input[
            "prefill_resident_bytes"
        ],
    )
    assert not status.passed
    assert "content hash differs" in status.failures[0]


def test_endpoint_resource_input_rejects_missing_prefill_residency(tmp_path):
    head, _, resource_input = _fixture(tmp_path)
    resource_input["prefill_resources_included"] = False
    digest = hashlib.sha256(b"retained evidence").hexdigest()
    try:
        assemble_endpoint_resource_receipt(
            resource_input,
            input_artifact_sha256=digest,
            specification_artifact_sha256=digest,
            head_service_status=head,
            prefill_model_excluding_head_bytes=resource_input[
                "prefill_resident_bytes"
            ],
        )
    except ValueError as exc:
        assert "prefill_resources_included" in str(exc)
    else:
        raise AssertionError("missing prefill residency was accepted")


def test_target_parameter_census_matches_the_sealed_config():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "qwen3_30b_a3b_thinking_2507.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    census = qwen3_moe_bf16_parameter_census(
        config["model_architecture"]
    )
    assert census == config["parameter_census"]
    assert census["total_parameters"] == 30_532_122_624
    assert (
        census["prefill_model_excluding_lm_head_bf16_bytes"]
        + census["lm_head_bf16_bytes"]
        == census["bf16_total_resident_bytes"]
    )
