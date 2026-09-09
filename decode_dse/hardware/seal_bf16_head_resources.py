"""Seal cited endpoint resources against one measured BF16 head service.

The resource input is deliberately manual: no GPU area or peak-HBM number is
inferred from a marketing name.  The operator supplies an exact JSON input and
the cited specification file from which its values were transcribed.  Both
files are hashed into the receipt, while the measured artifact binds the
physical endpoint UUID.  The measurement driver is instrumentation and must
not appear as a deployed service endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from decode_dse.hardware.head_service_resources import (
    assemble_endpoint_resource_receipt,
    load_bf16_head_endpoint_resource_receipt,
    qwen3_moe_bf16_parameter_census,
)
from decode_dse.hardware.lm_head_service import (
    load_bf16_head_service_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = config["model_architecture"]
    derived_census = qwen3_moe_bf16_parameter_census(architecture)
    if config.get("parameter_census") != derived_census:
        raise SystemExit("config parameter_census differs from architecture")
    batches = tuple(
        sorted({int(value) for value in config["hardware_space"]["BATCH"]})
    )
    head_path = Path(args.head_service_calibration).resolve()
    head_status = load_bf16_head_service_artifact(
        head_path,
        model_name=str(config["model_name"]),
        model_revision=str(config["model_revision"]),
        hidden_size=int(architecture["hidden_size"]),
        vocab_size=int(architecture["vocab_size"]),
        tie_embeddings=bool(architecture["tie_word_embeddings"]),
        required_batches=batches,
    )
    if not head_status.passed:
        raise SystemExit(
            "head-service artifact is not valid:\n"
            + "\n".join(head_status.failures)
        )
    input_path = Path(args.resource_input).resolve()
    specification_path = Path(args.specification_evidence).resolve()
    resource_input = json.loads(input_path.read_text(encoding="utf-8"))
    document = assemble_endpoint_resource_receipt(
        resource_input,
        input_artifact_sha256=_sha256(input_path),
        specification_artifact_sha256=_sha256(specification_path),
        head_service_status=head_status,
        prefill_model_excluding_head_bytes=int(
            derived_census[
                "prefill_model_excluding_lm_head_bf16_bytes"
            ]
        ),
    )
    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".staging")
    staging.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = load_bf16_head_endpoint_resource_receipt(
        staging,
        head_service_status=head_status,
        prefill_model_excluding_head_bytes=int(
            derived_census[
                "prefill_model_excluding_lm_head_bf16_bytes"
            ]
        ),
    )
    if not status.passed:
        rejected = destination.with_name(destination.name + ".rejected")
        staging.replace(rejected)
        raise SystemExit(
            "endpoint resource receipt failed self-validation "
            f"(kept at {rejected}):\n" + "\n".join(status.failures)
        )
    staging.replace(destination)
    assert status.receipt is not None
    print(f"sealed {destination}")
    print(f"receipt_id: {status.receipt.receipt_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--head-service-calibration", required=True)
    parser.add_argument(
        "--resource-input",
        required=True,
        help="manual endpoint/deployment resource JSON",
    )
    parser.add_argument(
        "--specification-evidence",
        required=True,
        help="retained cited specification or manual used for the values",
    )
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return seal(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
