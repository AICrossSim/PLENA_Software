"""Write a non-rankable endpoint-resource input draft with derived residency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decode_dse.hardware.head_service_resources import (
    AREA_COMPARISON_BASIS,
    DECODER_INTERFACE_ENERGY_SCOPE,
    DEPLOYMENT_LINK_TIMING_SCOPE,
    HBM_BANDWIDTH_BASIS,
    HBM_CAPACITY_BASIS,
    HEAD_RESOURCE_DEPLOYMENT_SCOPE,
    HEAD_RESOURCE_INPUT_SCHEMA,
    MEASUREMENT_DRIVER_ROLE,
    qwen3_moe_bf16_parameter_census,
)
from decode_dse.hardware.lm_head_service import (
    load_bf16_head_service_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--head-service-calibration", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    architecture = config["model_architecture"]
    census = qwen3_moe_bf16_parameter_census(architecture)
    if config.get("parameter_census") != census:
        raise SystemExit("config parameter_census differs from architecture")
    batches = tuple(sorted(set(config["hardware_space"]["BATCH"])))
    head = load_bf16_head_service_artifact(
        args.head_service_calibration,
        model_name=config["model_name"],
        model_revision=config["model_revision"],
        hidden_size=architecture["hidden_size"],
        vocab_size=architecture["vocab_size"],
        tie_embeddings=architecture["tie_word_embeddings"],
        required_batches=batches,
    )
    if not head.passed or head.calibration is None:
        raise SystemExit("a passing head-service artifact is required")
    service_id = str(head.calibration.provenance["head_service_id"])
    device_name, separator, device_uuid = service_id.rpartition(":")
    if not separator:
        raise SystemExit("head-service endpoint identity is not separable")
    draft = {
        "schema_version": HEAD_RESOURCE_INPUT_SCHEMA,
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
        "endpoint_device_name": device_name,
        "endpoint_device_uuid": device_uuid,
        "endpoint_aggregate_compute_silicon_area_mm2": None,
        "endpoint_compute_die_count": None,
        "endpoint_hbm_capacity_bytes": None,
        "endpoint_hbm_bandwidth_bytes_per_s": None,
        "prefill_resident_bytes": census[
            "prefill_model_excluding_lm_head_bf16_bytes"
        ],
        "head_resident_bytes": int(
            head.calibration.service["head_weight_capacity_bytes"]
        ),
        "endpoint_runtime_reserve_bytes": None,
        "decoder_interface_energy_j_per_byte": None,
        "decoder_interface_energy_scope": DECODER_INTERFACE_ENERGY_SCOPE,
        "deployment_request_bandwidth_bytes_s": None,
        "deployment_response_bandwidth_bytes_s": None,
        "deployment_link_peak_bandwidth_bytes_s": None,
        "deployment_request_fixed_latency_s": None,
        "deployment_response_fixed_latency_s": None,
        "deployment_link_timing_scope": DEPLOYMENT_LINK_TIMING_SCOPE,
        "measurement_driver_timing_used": False,
        "area_comparison_basis": AREA_COMPARISON_BASIS,
        "hbm_capacity_basis": HBM_CAPACITY_BASIS,
        "hbm_bandwidth_basis": HBM_BANDWIDTH_BASIS,
        "source": {
            "publisher": None,
            "title": None,
            "revision": None,
            "locator": None,
            "retrieved_at_utc": None,
            "area_basis_statement": None,
            "deployment_link_basis_statement": None,
        },
    }
    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"refusing to replace existing draft: {destination}")
    destination.write_text(
        json.dumps(draft, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote non-rankable draft {destination}")
    print("replace every null with retained, cited evidence before sealing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
