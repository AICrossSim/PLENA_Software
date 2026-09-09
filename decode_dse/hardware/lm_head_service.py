"""Validated BF16 output-head service boundary for cached decode."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from decode_dse.hardware.statistics import (
    percentile,
    spearman_rank_correlation,
)
from decode_dse.profiles import local_head_matrix_family_contract

HEAD_SERVICE_SCHEMA = "bf16-output-head-service"
HEAD_SERVICE_MODE = "remote_bf16_head_dedicated"
LOCAL_MX_HEAD_SCHEMA = "decode-local-mx-output-head/v2"
LOCAL_MX_HEAD_MODE = "decode_local_mx_head"
LOCAL_MX_HEAD_PRECISION_POLICY = "profile_w_a_bf16_logits"
LOCAL_MX_HEAD_LOGIT_DTYPE = "BF16"
LOCAL_MX_HEAD_SELECTION_POLICY = (
    "streaming_topk20_topp0.95_with_argmax_diagnostic_lowest_id_ties"
)
#: Service mode of the decode-local BF16 output head.  The head's weights and
#: its per-decode-step HBM traffic are charged to the decode chip's physical
#: ledger exactly like every other resident tensor; only the head's *compute*
#: is idealized, in the sense that no measured instruction-level timing or
#: energy signature is bound to it.  The idealization below travels with every
#: row priced at this boundary so it can never be read as a modelled head.
LOCAL_HEAD_MODE = "decode_local_bf16_head"
LOCAL_HEAD_COMPUTE_IDEALIZATION = "local_bf16_head_compute_idealized"
LOCAL_HEAD_IDEALIZATIONS: tuple[str, ...] = (
    LOCAL_HEAD_COMPUTE_IDEALIZATION,
)
# Where the BF16 output head runs.  ``DECODE_BF16_HEAD`` charges the head's
# weights and its per-decode-step traffic to the decode chip and idealizes only
# the head's compute; ``EXTERNAL_BF16_HEAD`` stops the decode ledger after the
# final RMSNorm and prices the head from a measured remote service.
DECODE_BF16_HEAD = "decode_bf16_unmodeled"
DECODE_MX_HEAD = "decode_local_mx_head"
EXTERNAL_BF16_HEAD = "external_bf16_service"
OUTPUT_HEAD_LOCATIONS = frozenset(
    {DECODE_BF16_HEAD, DECODE_MX_HEAD, EXTERNAL_BF16_HEAD}
)
#: Service mode implied by each output-head location.
OUTPUT_HEAD_SERVICE_MODES: Mapping[str, str] = {
    DECODE_BF16_HEAD: LOCAL_HEAD_MODE,
    DECODE_MX_HEAD: LOCAL_MX_HEAD_MODE,
    EXTERNAL_BF16_HEAD: HEAD_SERVICE_MODE,
}
#: Scope idealizations disclosed by each output-head location.
OUTPUT_HEAD_IDEALIZATIONS: Mapping[str, tuple[str, ...]] = {
    DECODE_BF16_HEAD: LOCAL_HEAD_IDEALIZATIONS,
    DECODE_MX_HEAD: (),
    EXTERNAL_BF16_HEAD: (),
}
HEAD_LOGIT_MAX_ABS_ERROR = 0.25
HEAD_LOGIT_MEAN_ABS_ERROR = 0.02
HEAD_TOPK_MIN_AGREEMENT = 0.90
HEAD_VALIDATION_TOPK = 10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_ADDRESSED_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[0-9a-f]{64}$"
)
_MODEL_FIELDS = {
    "model_name",
    "model_revision",
    "hidden_size",
    "vocab_size",
    "tie_embeddings",
}
_PROTOCOL_FIELDS = {
    "hidden_dtype",
    "hidden_element_bytes",
    "token_id_dtype",
    "token_id_bytes",
    "request_fixed_bytes",
    "request_metadata_bytes_per_sequence",
    "response_fixed_bytes",
    "response_metadata_bytes_per_sequence",
    "request_bandwidth_bytes_s",
    "response_bandwidth_bytes_s",
    "request_fixed_latency_s",
    "response_fixed_latency_s",
    "link_energy_j_per_byte",
    "link_dynamic_energy_scope",
    "duplex_schedule",
}
_SERVICE_FIELDS = {
    "service_mode",
    "service_location",
    "service_instances",
    "weight_dtype",
    "weight_alignment_bytes",
    "head_weight_bytes",
    "head_weight_sha256",
    "head_weight_layout",
    "head_weight_capacity_bytes",
    "mac_input_dtype",
    "accumulator_dtype",
    "logit_dtype",
    "logits_boundary",
    "selection_policy",
    "validation_topk",
    "bf16_mac_per_s",
    "bf16_mac_energy_j",
    "memory_bandwidth_bytes_s",
    "memory_energy_j_per_byte",
    "selection_latency_s_per_element",
    "selection_energy_j_per_element",
    "fixed_latency_s",
    "fixed_dynamic_energy_j",
    "leakage_power_w",
}
_MEASUREMENT_FIELDS = {
    "measurement_id",
    "split",
    "batch",
    "repeat",
    "hidden_bf16_sha256",
    "reference_logits_bf16_sha256",
    "service_logits_bf16_sha256",
    "reference_token_ids_sha256",
    "service_token_ids_sha256",
    "reference_logits_finite",
    "service_logits_finite",
    "logit_max_abs_error",
    "logit_mean_abs_error",
    "topk_set_agreement",
    "selected_tokens_equal",
    "request_bytes",
    "response_bytes",
    "head_weight_bytes",
    "head_memory_bytes",
    "bf16_macs",
    "selection_elements",
    "request_latency_s",
    "head_latency_s",
    "queue_delay_s",
    "response_latency_s",
    "link_dynamic_energy_j",
    "mac_dynamic_energy_j",
    "memory_dynamic_energy_j",
    "selection_dynamic_energy_j",
    "fixed_dynamic_energy_j",
    "dynamic_energy_j",
    "leakage_power_w",
}
_PROVENANCE_FIELDS = {
    "repository",
    "revision",
    "source_tree_sha256",
    "command",
    "toolchain",
    "environment_sha256",
    "link_id",
    "head_service_id",
    "process_corner",
    "measured_at_utc",
    "measurement_resolution",
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


def require_content_addressed_id(
    name: str,
    value: Any,
    *,
    prefix: str | None = None,
) -> str:
    """Return a named SHA-256 identity or reject an unbound label."""

    if not isinstance(value, str) or not _CONTENT_ADDRESSED_ID.fullmatch(value):
        raise ValueError(f"{name} must be a content-addressed identity")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"{name} must use the {prefix} identity family")
    return value


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def composite_system_calibration_id(
    decoder_calibration_id: str,
    head_calibration_id: str,
    head_provenance_id: str,
    *,
    service_mode: str = HEAD_SERVICE_MODE,
) -> str:
    """Bind decoder and remote-head calibrations into one system identity."""

    decoder_calibration_id = require_content_addressed_id(
        "decoder calibration",
        decoder_calibration_id,
    )
    head_calibration_id = require_content_addressed_id(
        "head calibration",
        head_calibration_id,
        prefix="bf16-head-service-",
    )
    head_provenance_id = require_content_addressed_id(
        "head provenance",
        head_provenance_id,
        prefix="bf16-head-provenance-",
    )
    if service_mode != HEAD_SERVICE_MODE:
        raise ValueError("system calibration requires the remote head service")
    return "decode-head-system-" + _content_hash(
        {
            "decoder_calibration_id": decoder_calibration_id,
            "head_calibration_id": head_calibration_id,
            "head_provenance_id": head_provenance_id,
            "service_mode": service_mode,
        }
    )


def local_head_system_calibration_id(decoder_calibration_id: str) -> str:
    """System identity for the fully priced decode-local MX output head."""

    decoder_calibration_id = require_content_addressed_id(
        "decoder calibration",
        decoder_calibration_id,
    )
    return "decode-local-head-system-" + _content_hash(
        {
            "decoder_calibration_id": decoder_calibration_id,
            "service_mode": LOCAL_MX_HEAD_MODE,
            "precision_policy": LOCAL_MX_HEAD_PRECISION_POLICY,
            "logit_dtype": LOCAL_MX_HEAD_LOGIT_DTYPE,
            "selection_policy": LOCAL_MX_HEAD_SELECTION_POLICY,
            "idealizations": [],
        }
    )


def local_head_boundary_status() -> dict[str, Any]:
    """Recorded boundary disclosure for the decode-local BF16 head.

    The fields mirror the remote-service status so a reader can compare the two
    placements field by field, but ``service_mode`` and ``idealizations`` state
    plainly that this arm rests on an unmeasured head compute cost.
    """

    return {
        "schema_version": HEAD_SERVICE_SCHEMA,
        "artifact_sha256": None,
        "passed": False,
        "failures": [LOCAL_HEAD_COMPUTE_IDEALIZATION],
        "calibration_id": None,
        "provenance_id": None,
        "service_mode": LOCAL_HEAD_MODE,
        "service_location": "decode_chip",
        "required_batches": [],
    }


def _local_mx_head_arithmetic(
    weight_format: str,
    activation_format: str,
    vector_format: str,
) -> tuple[bool, tuple[str, ...]]:
    """Return the exact profile-bound matrix chain and deployment verdict."""

    contract = local_head_matrix_family_contract(
        weight_format,
        activation_format,
        vector_format,
    )
    return (
        bool(contract["operand_family_deployment_supported"]),
        tuple(str(item) for item in contract["arithmetic_chain"]),
    )


def _local_mx_head_family_contract(
    weight_format: str,
    activation_format: str,
    vector_format: str,
) -> dict[str, Any]:
    """Return the canonical profile helper without duplicating its semantics."""

    return dict(
        local_head_matrix_family_contract(
            weight_format,
            activation_format,
            vector_format,
        )
    )


def local_mx_head_boundary_status(
    *,
    profile_id: str,
    weight_format: str,
    activation_format: str,
    vector_format: str,
    matrix_mlen: int,
) -> dict[str, Any]:
    """Return the exact, non-reusable contract for one local-head profile."""

    require_content_addressed_id("profile_id", profile_id, prefix="dqp-")
    for name, value in (
        ("weight_format", weight_format),
        ("activation_format", activation_format),
        ("vector_format", vector_format),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
    family_contract = _local_mx_head_family_contract(
        weight_format,
        activation_format,
        vector_format,
    )
    supported = bool(family_contract["operand_family_deployment_supported"])
    arithmetic_chain = tuple(family_contract["arithmetic_chain"])
    _positive_int(matrix_mlen, "matrix_mlen")
    failures = [] if supported else [
        "mixed_matrix_family_unsupported_without_trace_evidence"
    ]
    return {
        "schema_version": LOCAL_MX_HEAD_SCHEMA,
        "passed": supported,
        "failures": failures,
        "service_mode": LOCAL_MX_HEAD_MODE,
        "service_location": "decode_chip",
        "profile_id": profile_id,
        "precision_policy": LOCAL_MX_HEAD_PRECISION_POLICY,
        "weight_format": weight_format,
        "activation_format": activation_format,
        "numerical_matrix_mlen": matrix_mlen,
        "scale_format": "E8M0",
        "block_size": 8,
        "accumulator_dtype": "signed_fixed16.16",
        "operand_family_binding": family_contract["operand_family_binding"],
        "numerical_oracle_rule": family_contract["numerical_oracle_rule"],
        "partial_conversion_rule": family_contract[
            "partial_conversion_rule"
        ],
        "operand_family_deployment_supported": supported,
        "hardware_bit_parity_verified": family_contract[
            "hardware_bit_parity_verified"
        ],
        "accumulation_chain": list(arithmetic_chain),
        "matrix_numeric_format": vector_format,
        "matrix_storage_format": vector_format,
        "logit_container_format": LOCAL_MX_HEAD_LOGIT_DTYPE,
        "bf16_container_precision_recovery": False,
        "selection_policy": LOCAL_MX_HEAD_SELECTION_POLICY,
        "tp_selection_merge": (
            "topk20_gather_owner_then_u32_token_broadcast_charged"
        ),
        "serving_logits_materialization": "tile_streamed_not_full_vocab",
        "offline_nll_logits_materialization": "full_vocab_allowed",
        "weight_source": "profile_weight_format",
        "activation_source": "profile_activation_format",
        "uses_existing_matrix_units": True,
        "uses_existing_vector_units": True,
        "additional_compute_area_mm2": 0.0,
        "idealizations": [],
    }


def local_mx_head_status_valid(
    value: Mapping[str, Any],
    *,
    profile_id: str | None = None,
    weight_format: str | None = None,
    activation_format: str | None = None,
    vector_format: str | None = None,
    matrix_mlen: int | None = None,
) -> bool:
    """Return whether a serialized local-head contract is exact and complete."""

    try:
        resolved = {
            "profile_id": profile_id or str(value["profile_id"]),
            "weight_format": weight_format or str(value["weight_format"]),
            "activation_format": (
                activation_format or str(value["activation_format"])
            ),
            "vector_format": vector_format or str(
                value["matrix_numeric_format"]
            ),
            "matrix_mlen": matrix_mlen or int(
                value["numerical_matrix_mlen"]
            ),
        }
        return dict(value) == local_mx_head_boundary_status(**resolved)
    except (KeyError, TypeError, ValueError):
        return False


def local_mx_head_breakdown_valid(
    value: Mapping[str, Any],
    *,
    profile_id: str | None = None,
    weight_format: str | None = None,
    activation_format: str | None = None,
    vector_format: str | None = None,
    matrix_mlen: int | None = None,
    require_passed: bool = True,
) -> bool:
    """Validate the per-row physical cost decomposition of the local head."""

    try:
        if set(value) != {
            "schema_version",
            "passed",
            "failures",
            "operator",
            "profile_id",
            "numerical_matrix_mlen",
            "candidate_matrix_mlen",
            "numerical_matrix_mlen_exact_match",
            "batch_geometry",
            "padding_preparation",
            "precision_policy",
            "weight_format",
            "activation_format",
            "weight_element_bits",
            "weight_effective_bits",
            "activation_element_bits",
            "activation_effective_bits",
            "block_size",
            "scale_format",
            "accumulator_dtype",
            "operand_family_binding",
            "numerical_oracle_rule",
            "partial_conversion_rule",
            "operand_family_deployment_supported",
            "hardware_bit_parity_verified",
            "accumulation_chain",
            "matrix_numeric_format",
            "matrix_storage_format",
            "logit_container_format",
            "bf16_container_precision_recovery",
            "selection",
            "cycles_per_batch_step",
            "time_s_per_batch_step",
            "hbm_read_bytes_per_batch_step",
            "resident_bytes",
            "algorithmic_flops_per_batch_step",
            "flops_per_batch_step",
            "padding_flops_per_batch_step",
            "fractions",
            "topology",
            "compiler_lowering_receipt",
            "compiler_lowering_blocker",
        }:
            return False
        if (
            value["schema_version"] != "decode-local-mx-head-breakdown/v3"
            or not isinstance(value["passed"], bool)
            or not isinstance(value["failures"], list)
            or value["operator"] != "decode_lm_head"
            or not isinstance(value["profile_id"], str)
            or not _CONTENT_ADDRESSED_ID.fullmatch(value["profile_id"])
            or not value["profile_id"].startswith("dqp-")
            or value["precision_policy"] != LOCAL_MX_HEAD_PRECISION_POLICY
            or value["block_size"] != 8
            or value["scale_format"] != "E8M0"
            or value["accumulator_dtype"] != "signed_fixed16.16"
            or value["matrix_storage_format"]
            != value["matrix_numeric_format"]
            or value["logit_container_format"] != "BF16"
            or value["bf16_container_precision_recovery"] is not False
            or not isinstance(value["weight_format"], str)
            or not value["weight_format"]
            or not isinstance(value["activation_format"], str)
            or not value["activation_format"]
        ):
            return False
        expected_profile_id = profile_id or value["profile_id"]
        expected_weight = weight_format or value["weight_format"]
        expected_activation = activation_format or value["activation_format"]
        expected_vector = vector_format or value["matrix_numeric_format"]
        expected_mlen = matrix_mlen or value["numerical_matrix_mlen"]
        family_contract = _local_mx_head_family_contract(
            expected_weight,
            expected_activation,
            expected_vector,
        )
        supported = bool(
            family_contract["operand_family_deployment_supported"]
        )
        expected_chain = tuple(family_contract["arithmetic_chain"])
        if (
            value["profile_id"] != expected_profile_id
            or value["weight_format"] != expected_weight
            or value["activation_format"] != expected_activation
            or value["matrix_numeric_format"] != expected_vector
            or value["numerical_matrix_mlen"] != expected_mlen
            or value["operand_family_binding"]
            != family_contract["operand_family_binding"]
            or value["numerical_oracle_rule"]
            != family_contract["numerical_oracle_rule"]
            or value["partial_conversion_rule"]
            != family_contract["partial_conversion_rule"]
            or value["operand_family_deployment_supported"] is not supported
            or value["hardware_bit_parity_verified"]
            is not family_contract["hardware_bit_parity_verified"]
            or value["accumulation_chain"] != list(expected_chain)
        ):
            return False
        numerical_mlen = _positive_int(
            value["numerical_matrix_mlen"],
            "numerical_matrix_mlen",
        )
        candidate_mlen = _positive_int(
            value["candidate_matrix_mlen"],
            "candidate_matrix_mlen",
        )
        exact_mlen = numerical_mlen == candidate_mlen
        if value["numerical_matrix_mlen_exact_match"] is not exact_mlen:
            return False
        batch_geometry = value["batch_geometry"]
        if (
            not isinstance(batch_geometry, Mapping)
            or set(batch_geometry)
            != {"active_rows", "physical_rows", "zero_padded_rows"}
        ):
            return False
        active_rows = _positive_int(
            batch_geometry["active_rows"], "active_rows"
        )
        physical_rows = _positive_int(
            batch_geometry["physical_rows"], "physical_rows"
        )
        padded_rows = _nonnegative_int(
            batch_geometry["zero_padded_rows"], "zero_padded_rows"
        )
        if physical_rows != active_rows + padded_rows:
            return False
        padding = value["padding_preparation"]
        if (
            not isinstance(padding, Mapping)
            or set(padding)
            != {
                "schedule",
                "weight_padding",
                "activation_zero_fill_elements_per_rank",
                "activation_zero_fill_vector_events_per_rank",
                "activation_zero_fill_cycles_per_rank",
                "padded_vocab_mask",
                "slowest_serving_rank",
                "padded_vocab_mask_by_tp_rank",
                "padded_vocab_mask_elements_slowest_rank",
                "padded_vocab_mask_vector_events_slowest_rank",
                "padded_vocab_mask_cycles_slowest_rank",
                "padded_vocab_mask_elements_system",
                "padded_vocab_mask_vector_events_system",
                "padded_vocab_mask_cycles_system",
                "analytic_cycles_charged",
                "compiler_lowered",
            }
            or padding["schedule"]
            != "v_basic_full_width_chunks_before_matrix_and_selection"
            or padding["weight_padding"]
            != "offline_zero_fill_included_in_head_hbm_planes"
            or padding["padded_vocab_mask"] != "negative_infinity"
            or padding["analytic_cycles_charged"] is not True
            or padding["compiler_lowered"] is not False
        ):
            return False
        for name in (
            "activation_zero_fill_elements_per_rank",
            "activation_zero_fill_vector_events_per_rank",
            "activation_zero_fill_cycles_per_rank",
            "padded_vocab_mask_elements_slowest_rank",
            "padded_vocab_mask_vector_events_slowest_rank",
            "padded_vocab_mask_cycles_slowest_rank",
            "padded_vocab_mask_elements_system",
            "padded_vocab_mask_vector_events_system",
            "padded_vocab_mask_cycles_system",
        ):
            _nonnegative_int(padding[name], name)
        activation_padding_elements = padding[
            "activation_zero_fill_elements_per_rank"
        ]
        activation_padding_events = padding[
            "activation_zero_fill_vector_events_per_rank"
        ]
        if activation_padding_events != math.ceil(
            activation_padding_elements / candidate_mlen
        ):
            return False
        rank_masks = padding["padded_vocab_mask_by_tp_rank"]
        if not isinstance(rank_masks, list) or not rank_masks:
            return False
        normalized_rank_masks: list[dict[str, int]] = []
        for rank_mask in rank_masks:
            if not isinstance(rank_mask, Mapping) or set(rank_mask) != {
                "rank",
                "logical_vocab",
                "physical_vocab",
                "elements",
                "vector_events",
                "cycles",
            }:
                return False
            rank = _nonnegative_int(rank_mask["rank"], "rank")
            logical_vocab = _positive_int(
                rank_mask["logical_vocab"], "logical_vocab"
            )
            physical_vocab = _positive_int(
                rank_mask["physical_vocab"], "physical_vocab"
            )
            elements = _nonnegative_int(rank_mask["elements"], "elements")
            events = _nonnegative_int(
                rank_mask["vector_events"], "vector_events"
            )
            event_cycles = _nonnegative_int(rank_mask["cycles"], "cycles")
            if (
                physical_vocab < logical_vocab
                or physical_vocab
                != math.ceil(logical_vocab / candidate_mlen) * candidate_mlen
                or elements != active_rows * (physical_vocab - logical_vocab)
                or events != math.ceil(elements / candidate_mlen)
                or (events == 0) != (elements == 0)
                or (event_cycles == 0) != (events == 0)
            ):
                return False
            normalized_rank_masks.append(
                {
                    "rank": rank,
                    "logical_vocab": logical_vocab,
                    "physical_vocab": physical_vocab,
                    "elements": elements,
                    "vector_events": events,
                    "cycles": event_cycles,
                }
            )
        if (
            padding["padded_vocab_mask_elements_system"]
            < padding["padded_vocab_mask_elements_slowest_rank"]
            or padding["padded_vocab_mask_vector_events_system"]
            < padding["padded_vocab_mask_vector_events_slowest_rank"]
            or padding["padded_vocab_mask_cycles_system"]
            < padding["padded_vocab_mask_cycles_slowest_rank"]
        ):
            return False
        receipt = value["compiler_lowering_receipt"]
        receipt_blocker = value["compiler_lowering_blocker"]
        if exact_mlen and isinstance(receipt, Mapping):
            if receipt_blocker is not None:
                return False
            receipt_body = dict(receipt)
            receipt_hash = receipt_body.pop("contract_sha256", None)
            receipt_profile = receipt.get("profile")
            receipt_geometry = receipt.get("matrix_geometry")
            receipt_numeric = receipt.get("numeric_semantics")
            receipt_identity = receipt.get("numerical_identity")
            receipt_validity = receipt.get("validity")
            receipt_blockers = receipt.get("blockers")
            if (
                receipt.get("schema_version")
                != "plena-local-lm-head-lowering/v1"
                or receipt.get("operation") != "decode_lm_head"
                or receipt.get("profile_id") != value["profile_id"]
                or receipt.get("profile_sha256")
                != value["profile_id"].removeprefix("dqp-")
                or receipt_hash != _content_hash(receipt_body)
                or not isinstance(receipt_profile, Mapping)
                or receipt_profile.get("weight_format")
                != value["weight_format"]
                or receipt_profile.get("activation_format")
                != value["activation_format"]
                or receipt_profile.get("vector_format")
                != value["matrix_numeric_format"]
                or not isinstance(receipt_geometry, Mapping)
                or receipt_geometry.get("mlen") != candidate_mlen
                or not isinstance(receipt_identity, Mapping)
                or receipt_identity.get("profile_id") != value["profile_id"]
                or receipt_identity.get("mlen") != candidate_mlen
                or receipt.get("numerical_identity_sha256")
                != _content_hash(receipt_identity)
                or not isinstance(receipt_numeric, Mapping)
                or receipt_numeric.get("matrix_k_partition") != "MLEN"
                or receipt_numeric.get("partial_conversion")
                != "round_each_mlen_partial_to_profile.vector_format"
                or receipt_numeric.get("partial_rounding")
                != "round_to_nearest_even_to_profile.vector_format"
                or receipt_numeric.get("cross_instruction_accumulator")
                != "signed_fixed16_16_wraparound"
                or receipt_numeric.get("matrix_numeric_format")
                != "profile.vector_format"
                or receipt_numeric.get("matrix_storage_format")
                != "profile.vector_format"
                or receipt_numeric.get("matrix_writeout_numeric_format")
                != "profile.vector_format"
                or receipt_numeric.get("logit_container_format") != "BF16"
                or receipt_numeric.get("bf16_reference_writeout_rounding")
                != "mantissa_truncation"
                or receipt_numeric.get("no_per_stage_bf16_matrix_switch")
                is not True
                or receipt_numeric.get("precision_recovery") is not False
                or not isinstance(receipt_validity, Mapping)
                or not isinstance(
                    receipt_validity.get("publication_valid"), bool
                )
                or not isinstance(receipt_blockers, list)
                or any(
                    not isinstance(item, str) or not item
                    for item in receipt_blockers
                )
                or (
                    receipt_validity["publication_valid"] is False
                    and not receipt_blockers
                )
            ):
                return False
            receipt_model = receipt.get("model_geometry")
            if (
                not isinstance(receipt_model, Mapping)
                or receipt_model.get("active_batch") != active_rows
                or receipt_model.get("physical_batch") != physical_rows
                or receipt_model.get("zero_padded_batch_rows") != padded_rows
            ):
                return False
        elif exact_mlen:
            if (
                receipt is not None
                or receipt_blocker
                != "tensor_parallel_local_head_compiler_lowering_unavailable"
            ):
                return False
        elif (
            receipt is not None
            or receipt_blocker
            != (
                "numerical_matrix_mlen_mismatch_no_profile_bound_"
                "compiler_receipt"
            )
        ):
            return False
        failures = value["failures"]
        allowed_failures = {
            "body_weight_physical_padding_unmodelled",
            "local_head_tp_sharded_physical_shape_unmodelled",
            "compiler_trace_head_stage_breakdown_unavailable",
            "legacy_ideal_parallelism_omits_global_topk_merge",
            "mixed_matrix_family_unsupported_without_trace_evidence",
            "numerical_matrix_mlen_mismatch",
        }
        if (
            any(not isinstance(item, str) for item in failures)
            or len(failures) != len(set(failures))
            or not set(failures).issubset(allowed_failures)
            or value["passed"] is not (not failures)
            or (
                supported
                == (
                    "mixed_matrix_family_unsupported_without_trace_evidence"
                    in failures
                )
            )
            or (
                exact_mlen
                == ("numerical_matrix_mlen_mismatch" in failures)
            )
            or (require_passed and failures)
        ):
            return False
        for name in (
            "weight_element_bits",
            "activation_element_bits",
            "algorithmic_flops_per_batch_step",
            "flops_per_batch_step",
        ):
            if (
                isinstance(value[name], bool)
                or not isinstance(value[name], int)
                or value[name] <= 0
            ):
                return False
        padding_flops = _nonnegative_int(
            value["padding_flops_per_batch_step"],
            "padding_flops_per_batch_step",
        )
        if value["flops_per_batch_step"] != (
            value["algorithmic_flops_per_batch_step"] + padding_flops
        ):
            return False
        for name in ("weight_effective_bits", "activation_effective_bits"):
            if _positive_float(value[name], name) < value[
                name.replace("effective", "element")
            ]:
                return False
        selection = value["selection"]
        if (
            not isinstance(selection, Mapping)
            or set(selection)
            != {
                "serving_policy",
                "diagnostic_policy",
                "full_vocab_logits_materialized",
                "top_k",
                "top_p",
                "min_p",
                "logit_tile_bytes_per_chip",
                "state_bytes_per_chip",
                "workspace_bytes_per_chip",
                "distributed_merge",
            }
            or selection["serving_policy"]
            != "streaming_topk20_topp0.95_minp0"
            or selection["diagnostic_policy"]
            != "argmax_lowest_token_id_on_tie"
            or selection["full_vocab_logits_materialized"] is not False
            or selection["top_k"] != 20
            or selection["top_p"] != 0.95
            or selection["min_p"] != 0.0
        ):
            return False
        for name in (
            "logit_tile_bytes_per_chip",
            "state_bytes_per_chip",
            "workspace_bytes_per_chip",
        ):
            _positive_int(selection[name], name)
        if selection["workspace_bytes_per_chip"] != (
            selection["logit_tile_bytes_per_chip"]
            + selection["state_bytes_per_chip"]
        ):
            return False
        merge = selection["distributed_merge"]
        if (
            not isinstance(merge, Mapping)
            or set(merge)
            != {
                "mode",
                "candidate_pair_bytes",
                "candidate_count_per_rank_per_sequence",
                "aggregate_link_bytes_per_batch_step",
                "slowest_path_serialization_time_s",
                "charged_to_system_collective_energy",
            }
            or merge["mode"]
            != "tp_topk20_gather_owner_then_u32_token_broadcast"
            or merge["candidate_pair_bytes"] != 8
            or merge["candidate_count_per_rank_per_sequence"] != 20
            or merge["charged_to_system_collective_energy"] is not True
        ):
            return False
        link_bytes = _nonnegative_float(
            merge["aggregate_link_bytes_per_batch_step"],
            "aggregate_link_bytes_per_batch_step",
        )
        link_time = _nonnegative_float(
            merge["slowest_path_serialization_time_s"],
            "slowest_path_serialization_time_s",
        )
        cycles = value["cycles_per_batch_step"]
        if not isinstance(cycles, Mapping) or set(cycles) != {
            "matrix_slowest_rank",
            "selection_slowest_rank",
            "argmax_diagnostic_slowest_rank",
            "activation_zero_fill_per_rank",
            "padded_vocab_mask_slowest_rank",
            "padded_vocab_mask_system",
            "serving_slowest_rank",
        }:
            return False
        for name in (
            "matrix_slowest_rank",
            "selection_slowest_rank",
            "argmax_diagnostic_slowest_rank",
        ):
            _positive_float(cycles[name], name)
        for name in (
            "activation_zero_fill_per_rank",
            "padded_vocab_mask_slowest_rank",
            "padded_vocab_mask_system",
        ):
            _nonnegative_int(cycles[name], name)
        if (
            cycles["activation_zero_fill_per_rank"]
            != padding["activation_zero_fill_cycles_per_rank"]
            or cycles["padded_vocab_mask_slowest_rank"]
            != padding["padded_vocab_mask_cycles_slowest_rank"]
            or cycles["padded_vocab_mask_system"]
            != padding["padded_vocab_mask_cycles_system"]
        ):
            return False
        serving_cycles = _positive_float(
            cycles["serving_slowest_rank"],
            "serving_slowest_rank",
        )
        # A passing v3 receipt is selection-eligible, so its serving total must
        # conserve every charged slowest-rank component.  Failed legacy
        # projections may use an undisclosed compatibility divisor and remain
        # inspectable only through ``require_passed=False``; they cannot rank.
        if value["passed"] and not math.isclose(
            serving_cycles,
            float(cycles["matrix_slowest_rank"])
            + float(cycles["selection_slowest_rank"])
            + float(cycles["activation_zero_fill_per_rank"])
            + float(cycles["padded_vocab_mask_slowest_rank"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return False
        timing = value["time_s_per_batch_step"]
        if (
            not isinstance(timing, Mapping)
            or set(timing)
            != {
                "compute_slowest_rank",
                "hbm_slowest_rank",
                "local_roofline",
                "distributed_selection_collective",
                "isolated_head_with_collective",
                "scope",
            }
            or timing["scope"]
            != "local_roofline_plus_dependency_bound_tp_selection_merge"
        ):
            return False
        compute_time = _positive_float(
            timing["compute_slowest_rank"],
            "compute_slowest_rank",
        )
        hbm_time = _positive_float(
            timing["hbm_slowest_rank"],
            "hbm_slowest_rank",
        )
        roofline = _positive_float(
            timing["local_roofline"],
            "local_roofline",
        )
        collective = _nonnegative_float(
            timing["distributed_selection_collective"],
            "distributed_selection_collective",
        )
        isolated = _positive_float(
            timing["isolated_head_with_collective"],
            "isolated_head_with_collective",
        )
        if (
            not math.isclose(roofline, max(compute_time, hbm_time), rel_tol=1e-12)
            or not math.isclose(collective, link_time, rel_tol=1e-12)
            or not math.isclose(isolated, roofline + collective, rel_tol=1e-12)
            or (link_bytes == 0.0) != (collective == 0.0)
        ):
            return False
        hbm = value["hbm_read_bytes_per_batch_step"]
        if not isinstance(hbm, Mapping) or set(hbm) != {
            "aggregate_system_before_overfetch",
            "aggregate_system_after_overfetch",
            "slowest_rank_after_overfetch",
        }:
            return False
        before = _positive_float(
            hbm["aggregate_system_before_overfetch"],
            "aggregate_system_before_overfetch",
        )
        after = _positive_float(
            hbm["aggregate_system_after_overfetch"],
            "aggregate_system_after_overfetch",
        )
        rank = _positive_float(
            hbm["slowest_rank_after_overfetch"],
            "slowest_rank_after_overfetch",
        )
        if after < before or rank > after:
            return False
        resident = value["resident_bytes"]
        if not isinstance(resident, Mapping) or set(resident) != {
            "aggregate_system",
            "element_plane",
            "scale_plane",
            "bf16_plane",
        }:
            return False
        aggregate = _positive_int(
            resident["aggregate_system"],
            "aggregate_system",
        )
        element = _positive_int(resident["element_plane"], "element_plane")
        scale = _positive_int(resident["scale_plane"], "scale_plane")
        if resident["bf16_plane"] != 0 or aggregate != element + scale:
            return False
        fractions = value["fractions"]
        if not isinstance(fractions, Mapping) or set(fractions) != {
            "isolated_roofline_time_over_decoder_tpot",
            "hbm_read_over_decoder_traffic",
            "resident_over_decoder_capacity",
        }:
            return False
        for name, raw in fractions.items():
            fraction = _positive_float(raw, name)
            if fraction > 1.0:
                return False
        topology = value["topology"]
        if not isinstance(topology, Mapping) or set(topology) != {
            "tp",
            "kvp",
            "chip_count",
        }:
            return False
        tp = _positive_int(topology["tp"], "tp")
        kvp = _positive_int(topology["kvp"], "kvp")
        chips = _positive_int(topology["chip_count"], "chip_count")
        has_receipt = isinstance(receipt, Mapping)
        footprint = (
            receipt.get("hbm_footprint")
            if exact_mlen and has_receipt
            else None
        )
        receipt_selection = (
            receipt.get("selection")
            if exact_mlen and has_receipt
            else None
        )
        receipt_tp = (
            receipt_selection.get("tensor_parallel")
            if isinstance(receipt_selection, Mapping)
            else None
        )
        body_layout_missing = (
            "body_weight_physical_padding_unmodelled" in failures
        )
        head_tp_layout_missing = (
            "local_head_tp_sharded_physical_shape_unmodelled" in failures
        )
        if (
            tp * kvp != chips
            or (head_tp_layout_missing and not body_layout_missing)
            or (
                body_layout_missing
                and head_tp_layout_missing != (tp > 1)
            )
        ):
            return False
        expected_mask_ranks = 1 if body_layout_missing else tp
        mask_replicas = 1 if body_layout_missing else kvp
        if (
            len(normalized_rank_masks) != expected_mask_ranks
            or [value["rank"] for value in normalized_rank_masks]
            != list(range(expected_mask_ranks))
        ):
            return False
        slowest_serving_rank = _nonnegative_int(
            padding["slowest_serving_rank"], "slowest_serving_rank"
        )
        if slowest_serving_rank >= expected_mask_ranks:
            return False
        slowest_mask = normalized_rank_masks[slowest_serving_rank]
        if (
            padding["padded_vocab_mask_elements_slowest_rank"]
            != slowest_mask["elements"]
            or padding["padded_vocab_mask_vector_events_slowest_rank"]
            != slowest_mask["vector_events"]
            or padding["padded_vocab_mask_cycles_slowest_rank"]
            != slowest_mask["cycles"]
            or padding["padded_vocab_mask_elements_system"]
            != mask_replicas
            * sum(value["elements"] for value in normalized_rank_masks)
            or padding["padded_vocab_mask_vector_events_system"]
            != mask_replicas
            * sum(value["vector_events"] for value in normalized_rank_masks)
            or padding["padded_vocab_mask_cycles_system"]
            != mask_replicas
            * sum(value["cycles"] for value in normalized_rank_masks)
        ):
            return False
        event_costs = {
            value["cycles"] / value["vector_events"]
            for value in normalized_rank_masks
            if value["vector_events"]
        }
        activation_events = padding[
            "activation_zero_fill_vector_events_per_rank"
        ]
        activation_cycles = padding["activation_zero_fill_cycles_per_rank"]
        if activation_events:
            event_costs.add(activation_cycles / activation_events)
        elif activation_cycles:
            return False
        if len(event_costs) > 1 or any(
            not float(cost).is_integer() or cost <= 0 for cost in event_costs
        ):
            return False
        if exact_mlen:
            if tp == 1:
                if (
                    not has_receipt
                    or receipt_blocker is not None
                    or not isinstance(footprint, Mapping)
                    or footprint.get("weight_data_bytes") * kvp != element
                    or footprint.get("scale_bytes") * kvp != scale
                    or not isinstance(receipt_tp, Mapping)
                    or receipt_tp.get("ranks") != tp
                    or receipt_tp.get(
                        "required_local_candidates_per_rank_per_row"
                    )
                    != 20
                    or receipt_tp.get("selected_token_broadcast") is not True
                ):
                    return False
            elif (
                has_receipt
                or receipt_blocker
                != "tensor_parallel_local_head_compiler_lowering_unavailable"
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _relative_error(measured: float, predicted: float) -> float:
    if measured <= 0:
        raise ValueError("measured values must be positive")
    return abs(predicted - measured) / measured


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("repeat variation requires two or more values")
    mean = sum(values) / len(values)
    if mean <= 0:
        raise ValueError("repeat measurements must be positive")
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


@dataclass(frozen=True)
class HeadServiceMeasurement:
    """One dimensional and measured head-service observation."""

    measurement_id: str
    split: str
    batch: int
    repeat: int
    hidden_bf16_sha256: str
    reference_logits_bf16_sha256: str
    service_logits_bf16_sha256: str
    reference_token_ids_sha256: str
    service_token_ids_sha256: str
    reference_logits_finite: bool
    service_logits_finite: bool
    logit_max_abs_error: float
    logit_mean_abs_error: float
    topk_set_agreement: float
    selected_tokens_equal: bool
    request_bytes: int
    response_bytes: int
    head_weight_bytes: int
    head_memory_bytes: int
    bf16_macs: int
    selection_elements: int
    request_latency_s: float
    head_latency_s: float
    queue_delay_s: float
    response_latency_s: float
    link_dynamic_energy_j: float
    mac_dynamic_energy_j: float
    memory_dynamic_energy_j: float
    selection_dynamic_energy_j: float
    fixed_dynamic_energy_j: float
    dynamic_energy_j: float
    leakage_power_w: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HeadServiceMeasurement":
        if set(value) != _MEASUREMENT_FIELDS:
            raise ValueError("head-service measurement fields differ")
        measurement_id = str(value["measurement_id"])
        if not measurement_id:
            raise ValueError("measurement_id must be non-empty")
        split = str(value["split"])
        if split not in {"repeat", "holdout"}:
            raise ValueError("measurement split must be repeat or holdout")
        hashes = {}
        for name in (
            "hidden_bf16_sha256",
            "reference_logits_bf16_sha256",
            "service_logits_bf16_sha256",
            "reference_token_ids_sha256",
            "service_token_ids_sha256",
        ):
            item = str(value[name])
            if not _SHA256.fullmatch(item):
                raise ValueError(f"measurement {name} is invalid")
            hashes[name] = item
        for name in (
            "reference_logits_finite",
            "service_logits_finite",
            "selected_tokens_equal",
        ):
            if not isinstance(value[name], bool):
                raise TypeError(f"measurement {name} must be boolean")
        logit_max_abs_error = _nonnegative_float(
            value["logit_max_abs_error"],
            "measurement.logit_max_abs_error",
        )
        logit_mean_abs_error = _nonnegative_float(
            value["logit_mean_abs_error"],
            "measurement.logit_mean_abs_error",
        )
        if logit_mean_abs_error > logit_max_abs_error:
            raise ValueError("mean logit error cannot exceed maximum error")
        topk_set_agreement = _nonnegative_float(
            value["topk_set_agreement"],
            "measurement.topk_set_agreement",
        )
        if topk_set_agreement > 1:
            raise ValueError("top-k agreement cannot exceed one")
        return cls(
            measurement_id=measurement_id,
            split=split,
            batch=_positive_int(value["batch"], "measurement.batch"),
            repeat=_nonnegative_int(value["repeat"], "measurement.repeat"),
            hidden_bf16_sha256=hashes["hidden_bf16_sha256"],
            reference_logits_bf16_sha256=hashes[
                "reference_logits_bf16_sha256"
            ],
            service_logits_bf16_sha256=hashes[
                "service_logits_bf16_sha256"
            ],
            reference_token_ids_sha256=hashes[
                "reference_token_ids_sha256"
            ],
            service_token_ids_sha256=hashes[
                "service_token_ids_sha256"
            ],
            reference_logits_finite=value["reference_logits_finite"],
            service_logits_finite=value["service_logits_finite"],
            logit_max_abs_error=logit_max_abs_error,
            logit_mean_abs_error=logit_mean_abs_error,
            topk_set_agreement=topk_set_agreement,
            selected_tokens_equal=value["selected_tokens_equal"],
            request_bytes=_positive_int(
                value["request_bytes"],
                "measurement.request_bytes",
            ),
            response_bytes=_positive_int(
                value["response_bytes"],
                "measurement.response_bytes",
            ),
            head_weight_bytes=_positive_int(
                value["head_weight_bytes"],
                "measurement.head_weight_bytes",
            ),
            head_memory_bytes=_positive_int(
                value["head_memory_bytes"],
                "measurement.head_memory_bytes",
            ),
            bf16_macs=_positive_int(
                value["bf16_macs"],
                "measurement.bf16_macs",
            ),
            selection_elements=_positive_int(
                value["selection_elements"],
                "measurement.selection_elements",
            ),
            request_latency_s=_positive_float(
                value["request_latency_s"],
                "measurement.request_latency_s",
            ),
            head_latency_s=_positive_float(
                value["head_latency_s"],
                "measurement.head_latency_s",
            ),
            queue_delay_s=_nonnegative_float(
                value["queue_delay_s"],
                "measurement.queue_delay_s",
            ),
            response_latency_s=_positive_float(
                value["response_latency_s"],
                "measurement.response_latency_s",
            ),
            link_dynamic_energy_j=_positive_float(
                value["link_dynamic_energy_j"],
                "measurement.link_dynamic_energy_j",
            ),
            mac_dynamic_energy_j=_positive_float(
                value["mac_dynamic_energy_j"],
                "measurement.mac_dynamic_energy_j",
            ),
            memory_dynamic_energy_j=_positive_float(
                value["memory_dynamic_energy_j"],
                "measurement.memory_dynamic_energy_j",
            ),
            selection_dynamic_energy_j=_positive_float(
                value["selection_dynamic_energy_j"],
                "measurement.selection_dynamic_energy_j",
            ),
            fixed_dynamic_energy_j=_positive_float(
                value["fixed_dynamic_energy_j"],
                "measurement.fixed_dynamic_energy_j",
            ),
            dynamic_energy_j=_positive_float(
                value["dynamic_energy_j"],
                "measurement.dynamic_energy_j",
            ),
            leakage_power_w=_positive_float(
                value["leakage_power_w"],
                "measurement.leakage_power_w",
            ),
        )

    @property
    def total_latency_s(self) -> float:
        return (
            self.request_latency_s
            + self.head_latency_s
            + self.queue_delay_s
            + self.response_latency_s
        )

    @property
    def component_dynamic_energy_j(self) -> float:
        return (
            self.link_dynamic_energy_j
            + self.mac_dynamic_energy_j
            + self.memory_dynamic_energy_j
            + self.selection_dynamic_energy_j
            + self.fixed_dynamic_energy_j
        )


@dataclass(frozen=True)
class BF16HeadServiceEstimate:
    """Calibrated remote-head cost for one batch decode step."""

    calibration_id: str
    service_mode: str
    service_location: str
    provenance_id: str
    batch: int
    hidden_size: int
    vocab_size: int
    request_bytes: int
    response_bytes: int
    head_weight_bytes: int
    head_weight_capacity_bytes: int
    head_memory_bytes: int
    bf16_macs: int
    selection_elements: int
    request_latency_s: float
    head_latency_s: float
    queue_delay_s: float
    response_latency_s: float
    link_dynamic_energy_j: float
    mac_dynamic_energy_j: float
    memory_dynamic_energy_j: float
    selection_dynamic_energy_j: float
    fixed_dynamic_energy_j: float
    leakage_power_w: float

    def __post_init__(self) -> None:
        require_content_addressed_id(
            "calibration_id",
            self.calibration_id,
            prefix="bf16-head-service-",
        )
        require_content_addressed_id(
            "provenance_id",
            self.provenance_id,
            prefix="bf16-head-provenance-",
        )
        for name in ("service_mode", "service_location"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self,
                name,
            ):
                raise ValueError(f"{name} must be non-empty")
        if self.service_mode != HEAD_SERVICE_MODE:
            raise ValueError("head-service estimate mode is unsupported")
        if self.service_location != "prefill_chip":
            raise ValueError("head-service estimate location is unsupported")
        for name in (
            "batch",
            "hidden_size",
            "vocab_size",
            "request_bytes",
            "response_bytes",
            "head_weight_bytes",
            "head_weight_capacity_bytes",
            "head_memory_bytes",
            "bf16_macs",
            "selection_elements",
        ):
            _positive_int(getattr(self, name), name)
        if not self.head_capacity_feasible:
            raise ValueError("head-service estimate exceeds weight capacity")
        for name in (
            "request_latency_s",
            "head_latency_s",
            "response_latency_s",
            "leakage_power_w",
        ):
            _positive_float(getattr(self, name), name)
        if _nonnegative_float(
            self.queue_delay_s,
            "queue_delay_s",
        ) != 0:
            raise ValueError("dedicated head service cannot queue")
        for name in (
            "link_dynamic_energy_j",
            "mac_dynamic_energy_j",
            "memory_dynamic_energy_j",
            "selection_dynamic_energy_j",
            "fixed_dynamic_energy_j",
        ):
            _nonnegative_float(getattr(self, name), name)
        if self.dynamic_energy_j <= 0:
            raise ValueError("head-service dynamic energy must be positive")

    @property
    def total_latency_s(self) -> float:
        return (
            self.request_latency_s
            + self.head_latency_s
            + self.queue_delay_s
            + self.response_latency_s
        )

    @property
    def dynamic_energy_j(self) -> float:
        return (
            self.link_dynamic_energy_j
            + self.mac_dynamic_energy_j
            + self.memory_dynamic_energy_j
            + self.selection_dynamic_energy_j
            + self.fixed_dynamic_energy_j
        )

    @property
    def head_capacity_feasible(self) -> bool:
        return self.head_weight_bytes <= self.head_weight_capacity_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "service_mode": self.service_mode,
            "service_location": self.service_location,
            "provenance_id": self.provenance_id,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "request": {
                "unit": "bytes_per_batch_step",
                "bytes": self.request_bytes,
                "latency_s": self.request_latency_s,
            },
            "response": {
                "unit": "bytes_per_batch_step",
                "bytes": self.response_bytes,
                "latency_s": self.response_latency_s,
            },
            "head_weight": {
                "unit": "bytes",
                "required_bytes": self.head_weight_bytes,
                "capacity_bytes": self.head_weight_capacity_bytes,
                "feasible": self.head_capacity_feasible,
            },
            "head_memory_bytes_per_batch_step": self.head_memory_bytes,
            "bf16_macs_per_batch_step": self.bf16_macs,
            "selection_elements_per_batch_step": self.selection_elements,
            "head_latency_s": self.head_latency_s,
            "queue_delay_s": self.queue_delay_s,
            "total_latency_s": self.total_latency_s,
            "dynamic_energy_j_per_batch_step": {
                "link": self.link_dynamic_energy_j,
                "bf16_mac": self.mac_dynamic_energy_j,
                "head_memory": self.memory_dynamic_energy_j,
                "selection": self.selection_dynamic_energy_j,
                "fixed": self.fixed_dynamic_energy_j,
                "total": self.dynamic_energy_j,
            },
            "leakage_power_w": self.leakage_power_w,
        }


@dataclass(frozen=True)
class BF16HeadServiceCalibration:
    """Validated coefficients and identities for one dedicated head service."""

    artifact_sha256: str
    content_hash: str
    model_name: str
    model_revision: str
    hidden_size: int
    vocab_size: int
    tie_embeddings: bool
    required_batches: tuple[int, ...]
    protocol: Mapping[str, Any]
    service: Mapping[str, Any]
    provenance: Mapping[str, Any]
    validation: Mapping[str, float | int]

    @property
    def calibration_id(self) -> str:
        return "bf16-head-service-" + _content_hash(
            {
                "artifact_sha256": self.artifact_sha256,
                "content_hash": self.content_hash,
                "model_revision": self.model_revision,
                "required_batches": self.required_batches,
            }
        )

    @property
    def provenance_id(self) -> str:
        return "bf16-head-provenance-" + _content_hash(self.provenance)

    def _request_bytes(self, batch: int) -> int:
        return int(self.protocol["request_fixed_bytes"]) + batch * (
            self.hidden_size * int(self.protocol["hidden_element_bytes"])
            + int(self.protocol["request_metadata_bytes_per_sequence"])
        )

    def _response_bytes(self, batch: int) -> int:
        return int(self.protocol["response_fixed_bytes"]) + batch * (
            int(self.protocol["token_id_bytes"])
            + int(self.protocol["response_metadata_bytes_per_sequence"])
        )

    def _dimensions(self, batch: int) -> tuple[int, int, int]:
        weight = int(self.service["head_weight_bytes"])
        memory = weight + batch * self.hidden_size * 2
        macs = batch * self.hidden_size * self.vocab_size
        selection = batch * self.vocab_size
        return memory, macs, selection

    def estimate(self, batch: int) -> BF16HeadServiceEstimate:
        if batch not in self.required_batches:
            raise ValueError(
                f"batch {batch} is outside calibrated head-service scope"
            )
        request_bytes = self._request_bytes(batch)
        response_bytes = self._response_bytes(batch)
        memory_bytes, macs, selection = self._dimensions(batch)
        request_latency = (
            float(self.protocol["request_fixed_latency_s"])
            + request_bytes
            / float(self.protocol["request_bandwidth_bytes_s"])
        )
        response_latency = (
            float(self.protocol["response_fixed_latency_s"])
            + response_bytes
            / float(self.protocol["response_bandwidth_bytes_s"])
        )
        compute_latency = macs / float(self.service["bf16_mac_per_s"])
        memory_latency = (
            memory_bytes / float(self.service["memory_bandwidth_bytes_s"])
        )
        head_latency = (
            float(self.service["fixed_latency_s"])
            + max(compute_latency, memory_latency)
            + selection
            * float(self.service["selection_latency_s_per_element"])
        )
        link_energy = (
            request_bytes + response_bytes
        ) * float(self.protocol["link_energy_j_per_byte"])
        return BF16HeadServiceEstimate(
            calibration_id=self.calibration_id,
            service_mode=HEAD_SERVICE_MODE,
            service_location="prefill_chip",
            provenance_id=self.provenance_id,
            batch=batch,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            head_weight_bytes=int(self.service["head_weight_bytes"]),
            head_weight_capacity_bytes=int(
                self.service["head_weight_capacity_bytes"]
            ),
            head_memory_bytes=memory_bytes,
            bf16_macs=macs,
            selection_elements=selection,
            request_latency_s=request_latency,
            head_latency_s=head_latency,
            queue_delay_s=0.0,
            response_latency_s=response_latency,
            link_dynamic_energy_j=link_energy,
            mac_dynamic_energy_j=(
                macs * float(self.service["bf16_mac_energy_j"])
            ),
            memory_dynamic_energy_j=(
                memory_bytes
                * float(self.service["memory_energy_j_per_byte"])
            ),
            selection_dynamic_energy_j=(
                selection
                * float(self.service["selection_energy_j_per_element"])
            ),
            fixed_dynamic_energy_j=float(
                self.service["fixed_dynamic_energy_j"]
            ),
            leakage_power_w=float(self.service["leakage_power_w"]),
        )


@dataclass(frozen=True)
class BF16HeadServiceStatus:
    """Fail-closed result of validating one service artifact."""

    source_path: Path
    artifact_sha256: str
    failures: tuple[str, ...]
    calibration: BF16HeadServiceCalibration | None

    @property
    def passed(self) -> bool:
        return not self.failures and self.calibration is not None

    @property
    def calibration_id(self) -> str | None:
        return (
            self.calibration.calibration_id
            if self.calibration is not None
            else None
        )

    @property
    def provenance_id(self) -> str | None:
        return (
            self.calibration.provenance_id
            if self.calibration is not None
            else None
        )

    @property
    def service_mode(self) -> str:
        return HEAD_SERVICE_MODE if self.passed else "unmodeled"

    def to_dict(self) -> dict[str, Any]:
        calibration = self.calibration
        return {
            "schema_version": HEAD_SERVICE_SCHEMA,
            "artifact_sha256": self.artifact_sha256,
            "passed": self.passed,
            "failures": list(self.failures),
            "calibration_id": self.calibration_id,
            "provenance_id": self.provenance_id,
            "service_mode": (
                HEAD_SERVICE_MODE if self.passed else "unmodeled"
            ),
            "service_location": (
                "prefill_chip" if self.passed else None
            ),
            "required_batches": (
                list(calibration.required_batches)
                if calibration is not None
                else []
            ),
            "head_weight_sha256": (
                str(calibration.service["head_weight_sha256"])
                if calibration is not None
                else None
            ),
            "cost_scope": (
                {
                    "dynamic_energy": "endpoint_only",
                    "leakage": "endpoint_only",
                    "link_dynamic_energy": calibration.protocol[
                        "link_dynamic_energy_scope"
                    ],
                    "measurement_link_timing": (
                        "instrumentation_driver_to_endpoint_not_deployment"
                    ),
                    "measurement_driver_dynamic_included": False,
                    "measurement_driver_leakage_included": False,
                }
                if calibration is not None
                else None
            ),
            "numerical_policy": (
                {
                    "mac_input_dtype": calibration.service[
                        "mac_input_dtype"
                    ],
                    "accumulator_dtype": calibration.service[
                        "accumulator_dtype"
                    ],
                    "logit_dtype": calibration.service["logit_dtype"],
                    "selection_policy": calibration.service[
                        "selection_policy"
                    ],
                    "validation_topk": calibration.service[
                        "validation_topk"
                    ],
                    "logit_max_abs_error_limit": (
                        HEAD_LOGIT_MAX_ABS_ERROR
                    ),
                    "logit_mean_abs_error_limit": (
                        HEAD_LOGIT_MEAN_ABS_ERROR
                    ),
                    "topk_set_agreement_min": (
                        HEAD_TOPK_MIN_AGREEMENT
                    ),
                }
                if calibration is not None
                else None
            ),
            "numerical_validation": (
                dict(calibration.validation)
                if calibration is not None
                else None
            ),
        }


def head_service_status_valid(value: Mapping[str, Any]) -> bool:
    """Return whether a serialized status carries passing v2 evidence."""

    try:
        if (
            value.get("schema_version") != HEAD_SERVICE_SCHEMA
            or value.get("passed") is not True
            or value.get("failures") != []
            or value.get("service_mode") != HEAD_SERVICE_MODE
            or value.get("service_location") != "prefill_chip"
            or not _SHA256.fullmatch(str(value.get("artifact_sha256", "")))
            or not _SHA256.fullmatch(
                str(value.get("head_weight_sha256", ""))
            )
            or value.get("cost_scope")
            != {
                "dynamic_energy": "endpoint_only",
                "leakage": "endpoint_only",
                "link_dynamic_energy": (
                    "endpoint_receive_transmit_incremental_only"
                ),
                "measurement_link_timing": (
                    "instrumentation_driver_to_endpoint_not_deployment"
                ),
                "measurement_driver_dynamic_included": False,
                "measurement_driver_leakage_included": False,
            }
        ):
            return False
        require_content_addressed_id(
            "head calibration",
            value.get("calibration_id"),
            prefix="bf16-head-service-",
        )
        require_content_addressed_id(
            "head provenance",
            value.get("provenance_id"),
            prefix="bf16-head-provenance-",
        )
        batches = value.get("required_batches")
        if (
            not isinstance(batches, list)
            or len(batches) < 3
            or any(
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch <= 0
                for batch in batches
            )
            or batches != sorted(set(batches))
        ):
            return False
        policy = value.get("numerical_policy")
        expected_policy = {
            "mac_input_dtype": "BF16",
            "accumulator_dtype": "FP32",
            "logit_dtype": "BF16",
            "selection_policy": "argmax_lowest_token_id_on_tie",
            "validation_topk": HEAD_VALIDATION_TOPK,
            "logit_max_abs_error_limit": HEAD_LOGIT_MAX_ABS_ERROR,
            "logit_mean_abs_error_limit": HEAD_LOGIT_MEAN_ABS_ERROR,
            "topk_set_agreement_min": HEAD_TOPK_MIN_AGREEMENT,
        }
        if policy != expected_policy:
            return False
        validation = value.get("numerical_validation")
        if not isinstance(validation, Mapping):
            return False
        measurement_count = _positive_int(
            validation.get("measurement_count"),
            "measurement_count",
        )
        sample_count = _positive_int(
            validation.get("numerical_sample_count"),
            "numerical_sample_count",
        )
        holdout_count = _positive_int(
            validation.get("holdout_count"),
            "holdout_count",
        )
        exact_match_count = _positive_int(
            validation.get("selected_token_exact_match_count"),
            "selected_token_exact_match_count",
        )
        if (
            measurement_count != sample_count
            or exact_match_count != sample_count
            or measurement_count < 4 * len(batches)
            or holdout_count < len(batches)
        ):
            return False
        max_error = _nonnegative_float(
            validation.get("sampled_logit_max_abs_error"),
            "sampled_logit_max_abs_error",
        )
        mean_error = _nonnegative_float(
            validation.get("sampled_logit_mean_abs_error_max"),
            "sampled_logit_mean_abs_error_max",
        )
        topk_agreement = _nonnegative_float(
            validation.get("sampled_topk_set_agreement_min"),
            "sampled_topk_set_agreement_min",
        )
        if (
            max_error > HEAD_LOGIT_MAX_ABS_ERROR
            or mean_error > HEAD_LOGIT_MEAN_ABS_ERROR
            or mean_error > max_error
            or topk_agreement < HEAD_TOPK_MIN_AGREEMENT
            or topk_agreement > 1
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _parse_calibration(
    raw: Mapping[str, Any],
    *,
    artifact_sha256: str,
    model_name: str,
    model_revision: str,
    hidden_size: int,
    vocab_size: int,
    tie_embeddings: bool,
    required_batches: tuple[int, ...],
) -> BF16HeadServiceCalibration:
    body = dict(raw)
    content_hash = str(body.pop("content_hash", ""))
    if content_hash != _content_hash(body):
        raise ValueError("head-service content hash mismatch")
    if set(body) != {
        "schema_version",
        "model",
        "protocol",
        "service",
        "required_batch_scope",
        "measurements",
        "provenance",
    }:
        raise ValueError("head-service artifact fields differ")
    if body["schema_version"] != HEAD_SERVICE_SCHEMA:
        raise ValueError("unsupported head-service schema")

    model = body["model"]
    if not isinstance(model, Mapping) or set(model) != _MODEL_FIELDS:
        raise ValueError("head-service model identity fields differ")
    expected_model = {
        "model_name": model_name,
        "model_revision": model_revision,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "tie_embeddings": tie_embeddings,
    }
    if dict(model) != expected_model:
        raise ValueError("head-service model identity mismatch")

    protocol = body["protocol"]
    if not isinstance(protocol, Mapping) or set(protocol) != _PROTOCOL_FIELDS:
        raise ValueError("head-service protocol fields differ")
    if protocol["hidden_dtype"] != "BF16":
        raise ValueError("head-service hidden payload must be BF16")
    if protocol["token_id_dtype"] != "UINT32":
        raise ValueError("head-service response must use UINT32 token IDs")
    if protocol["duplex_schedule"] != "serialized_request_service_response":
        raise ValueError("head-service duplex schedule is unsupported")
    if _positive_int(
        protocol["hidden_element_bytes"],
        "hidden_element_bytes",
    ) != 2:
        raise ValueError("BF16 hidden elements must occupy two bytes")
    if _positive_int(protocol["token_id_bytes"], "token_id_bytes") != 4:
        raise ValueError("UINT32 token IDs must occupy four bytes")
    for name in (
        "request_fixed_bytes",
        "request_metadata_bytes_per_sequence",
        "response_fixed_bytes",
        "response_metadata_bytes_per_sequence",
    ):
        _nonnegative_int(protocol[name], name)
    for name in (
        "request_bandwidth_bytes_s",
        "response_bandwidth_bytes_s",
        "link_energy_j_per_byte",
    ):
        _positive_float(protocol[name], name)
    if (
        protocol["link_dynamic_energy_scope"]
        != "endpoint_receive_transmit_incremental_only"
    ):
        raise ValueError("head-service link energy must be endpoint-only")
    for name in ("request_fixed_latency_s", "response_fixed_latency_s"):
        _nonnegative_float(protocol[name], name)

    service = body["service"]
    if not isinstance(service, Mapping) or set(service) != _SERVICE_FIELDS:
        raise ValueError("head-service coefficient fields differ")
    if service["service_mode"] != HEAD_SERVICE_MODE:
        raise ValueError("head-service mode must be dedicated and queue-free")
    if service["service_location"] != "prefill_chip":
        raise ValueError("output-head service must run on the prefill chip")
    if _positive_int(service["service_instances"], "service_instances") != 1:
        raise ValueError("one dedicated service instance is required")
    if service["weight_dtype"] != "BF16":
        raise ValueError("output-head weights must be BF16")
    if not _SHA256.fullmatch(str(service["head_weight_sha256"])):
        raise ValueError("output-head weight checksum is invalid")
    if service["head_weight_layout"] != "vocab_by_hidden_row_major_bf16_le":
        raise ValueError("output-head weight layout is unsupported")
    if service["mac_input_dtype"] != "BF16":
        raise ValueError("output-head MAC inputs must be BF16")
    if service["accumulator_dtype"] != "FP32":
        raise ValueError("output-head accumulation must be FP32")
    if service["logit_dtype"] != "BF16":
        raise ValueError("output-head validation logits must be BF16")
    if service["logits_boundary"] != "fused_selection_token_ids":
        raise ValueError("output-head service must return token IDs only")
    if service["selection_policy"] != "argmax_lowest_token_id_on_tie":
        raise ValueError("output-head selection policy is unsupported")
    if (
        _positive_int(service["validation_topk"], "validation_topk")
        != HEAD_VALIDATION_TOPK
    ):
        raise ValueError("output-head validation top-k differs")
    alignment = _positive_int(
        service["weight_alignment_bytes"],
        "weight_alignment_bytes",
    )
    expected_weight = _align(hidden_size * vocab_size * 2, alignment)
    if _positive_int(
        service["head_weight_bytes"],
        "head_weight_bytes",
    ) != expected_weight:
        raise ValueError("head-service BF16 weight size is inconsistent")
    if _positive_int(
        service["head_weight_capacity_bytes"],
        "head_weight_capacity_bytes",
    ) < expected_weight:
        raise ValueError("head-service weight capacity is insufficient")
    for name in (
        "bf16_mac_per_s",
        "bf16_mac_energy_j",
        "memory_bandwidth_bytes_s",
        "memory_energy_j_per_byte",
        "selection_latency_s_per_element",
        "selection_energy_j_per_element",
        "fixed_dynamic_energy_j",
        "leakage_power_w",
    ):
        _positive_float(service[name], name)
    _nonnegative_float(service["fixed_latency_s"], "fixed_latency_s")

    batch_scope = body["required_batch_scope"]
    if (
        not isinstance(batch_scope, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in batch_scope
        )
        or tuple(batch_scope) != required_batches
    ):
        raise ValueError("head-service batch scope mismatch")
    if len(required_batches) < 3:
        raise ValueError(
            "head-service calibration requires at least three batch points"
        )

    provenance = body["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _PROVENANCE_FIELDS
    ):
        raise ValueError("head-service provenance fields differ")
    for name in (
        "repository",
        "revision",
        "link_id",
        "head_service_id",
        "process_corner",
    ):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(f"head-service provenance {name} is empty")
    measured_at = provenance["measured_at_utc"]
    if not isinstance(measured_at, str) or not measured_at.endswith("Z"):
        raise ValueError("head-service measurement time must be UTC")
    try:
        parsed_time = datetime.fromisoformat(
            measured_at[:-1] + "+00:00"
        )
    except ValueError as exc:
        raise ValueError("head-service measurement time is invalid") from exc
    if parsed_time.tzinfo != timezone.utc:
        raise ValueError("head-service measurement time must be UTC")
    for name in ("source_tree_sha256", "environment_sha256"):
        if not _SHA256.fullmatch(str(provenance[name])):
            raise ValueError(f"head-service provenance {name} is invalid")
    command = provenance["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise ValueError("head-service measurement command is invalid")
    toolchain = provenance["toolchain"]
    if (
        not isinstance(toolchain, Mapping)
        or not toolchain
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in toolchain.items()
        )
    ):
        raise ValueError("head-service toolchain provenance is invalid")
    resolution = provenance["measurement_resolution"]
    if not isinstance(resolution, Mapping):
        raise ValueError("head-service measurement resolution is invalid")
    idle_power = resolution.get("idle_power_w")
    if (
        not isinstance(idle_power, Mapping)
        or set(idle_power) != {"driver", "endpoint", "total"}
    ):
        raise ValueError("head-service idle-power provenance is incomplete")
    driver_idle = _positive_float(idle_power["driver"], "driver idle power")
    endpoint_idle = _positive_float(
        idle_power["endpoint"], "endpoint idle power"
    )
    total_idle = _positive_float(idle_power["total"], "total idle power")
    if not math.isclose(
        total_idle,
        driver_idle + endpoint_idle,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("head-service idle-power provenance does not conserve")
    service_leakage = _positive_float(
        resolution.get("service_leakage_power_w"),
        "service leakage power",
    )
    if (
        resolution.get("service_leakage_scope") != "endpoint_only"
        or resolution.get("measurement_driver_idle_role")
        != "instrumentation_only_not_deployed"
        or resolution.get("measurement_driver_leakage_included") is not False
        or not math.isclose(
            service_leakage,
            endpoint_idle,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            service_leakage,
            float(service["leakage_power_w"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("head-service leakage attribution is not endpoint-only")

    raw_measurements = body["measurements"]
    if not isinstance(raw_measurements, list):
        raise ValueError("head-service measurements must be a list")
    measurements = tuple(
        HeadServiceMeasurement.from_dict(value)
        for value in raw_measurements
    )
    measurement_ids = tuple(value.measurement_id for value in measurements)
    if len(measurement_ids) != len(set(measurement_ids)):
        raise ValueError("head-service measurement IDs repeat")

    provisional = BF16HeadServiceCalibration(
        artifact_sha256=artifact_sha256,
        content_hash=content_hash,
        model_name=model_name,
        model_revision=model_revision,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        tie_embeddings=tie_embeddings,
        required_batches=required_batches,
        protocol=dict(protocol),
        service=dict(service),
        provenance=dict(provenance),
        validation={},
    )
    repeats_by_batch: dict[int, list[HeadServiceMeasurement]] = {
        batch: [] for batch in required_batches
    }
    holdouts: list[HeadServiceMeasurement] = []
    for measurement in measurements:
        if measurement.batch not in repeats_by_batch:
            raise ValueError("measurement batch is outside required scope")
        estimate = provisional.estimate(measurement.batch)
        expected_dimensions = (
            estimate.request_bytes,
            estimate.response_bytes,
            estimate.head_weight_bytes,
            estimate.head_memory_bytes,
            estimate.bf16_macs,
            estimate.selection_elements,
        )
        measured_dimensions = (
            measurement.request_bytes,
            measurement.response_bytes,
            measurement.head_weight_bytes,
            measurement.head_memory_bytes,
            measurement.bf16_macs,
            measurement.selection_elements,
        )
        if measured_dimensions != expected_dimensions:
            raise ValueError("head-service measurement dimensions differ")
        if measurement.queue_delay_s != 0:
            raise ValueError("dedicated head service cannot report queue delay")
        if (
            not measurement.reference_logits_finite
            or not measurement.service_logits_finite
        ):
            raise ValueError("head-service logits must be finite")
        if (
            not measurement.selected_tokens_equal
            or measurement.reference_token_ids_sha256
            != measurement.service_token_ids_sha256
        ):
            raise ValueError(
                "head-service selected tokens differ from the BF16 reference"
            )
        if measurement.logit_max_abs_error > HEAD_LOGIT_MAX_ABS_ERROR:
            raise ValueError("head-service maximum logit error exceeds limit")
        if measurement.logit_mean_abs_error > HEAD_LOGIT_MEAN_ABS_ERROR:
            raise ValueError("head-service mean logit error exceeds limit")
        if measurement.topk_set_agreement < HEAD_TOPK_MIN_AGREEMENT:
            raise ValueError("head-service top-k agreement is below limit")
        if not math.isclose(
            measurement.component_dynamic_energy_j,
            measurement.dynamic_energy_j,
            rel_tol=1e-9,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "head-service dynamic energy components do not conserve"
            )
        if measurement.split == "repeat":
            repeats_by_batch[measurement.batch].append(measurement)
        else:
            holdouts.append(measurement)

    repeat_hidden_hashes: set[str] = set()
    for batch, values in repeats_by_batch.items():
        numerical_identities = {
            (
                value.hidden_bf16_sha256,
                value.reference_logits_bf16_sha256,
                value.service_logits_bf16_sha256,
                value.reference_token_ids_sha256,
                value.service_token_ids_sha256,
            )
            for value in values
        }
        if len(numerical_identities) != 1:
            raise ValueError(
                f"batch {batch} repeat measurements changed numerical input"
            )
        repeat_hidden_hashes.add(values[0].hidden_bf16_sha256)
    if len(repeat_hidden_hashes) != len(required_batches):
        raise ValueError("repeat batches must use distinct hidden payloads")
    holdout_hidden_hashes = {
        value.hidden_bf16_sha256 for value in holdouts
    }
    if len(holdout_hidden_hashes) != len(holdouts):
        raise ValueError("holdout hidden payloads must be distinct")
    if repeat_hidden_hashes & holdout_hidden_hashes:
        raise ValueError("repeat and holdout hidden payloads must be disjoint")

    repeat_latency_cv: list[float] = []
    repeat_energy_cv: list[float] = []
    repeat_leakage_cv: list[float] = []
    for batch, values in repeats_by_batch.items():
        if len(values) < 3:
            raise ValueError(
                f"batch {batch} requires at least three repeat measurements"
            )
        repeat_ids = tuple(value.repeat for value in values)
        if len(repeat_ids) != len(set(repeat_ids)):
            raise ValueError(f"batch {batch} repeats a repeat index")
        repeat_latency_cv.append(
            _coefficient_of_variation(
                [value.total_latency_s for value in values]
            )
        )
        repeat_energy_cv.append(
            _coefficient_of_variation(
                [value.dynamic_energy_j for value in values]
            )
        )
        repeat_leakage_cv.append(
            _coefficient_of_variation(
                [value.leakage_power_w for value in values]
            )
        )
    if max(repeat_latency_cv) > 0.05:
        raise ValueError("head-service repeat latency CV exceeds 5%")
    if max(repeat_energy_cv) > 0.10:
        raise ValueError("head-service repeat energy CV exceeds 10%")
    if max(repeat_leakage_cv) > 0.10:
        raise ValueError("head-service repeat leakage CV exceeds 10%")

    holdouts_by_batch = {
        batch: [value for value in holdouts if value.batch == batch]
        for batch in required_batches
    }
    if any(not values for values in holdouts_by_batch.values()):
        raise ValueError("every required batch needs a holdout measurement")
    latency_errors: list[float] = []
    dynamic_errors: list[float] = []
    leakage_errors: list[float] = []
    component_errors: list[float] = []
    dynamic_component_errors: list[float] = []
    for measurement in holdouts:
        estimate = provisional.estimate(measurement.batch)
        predicted_components = (
            estimate.request_latency_s,
            estimate.head_latency_s,
            estimate.response_latency_s,
        )
        measured_components = (
            measurement.request_latency_s,
            measurement.head_latency_s,
            measurement.response_latency_s,
        )
        component_errors.extend(
            _relative_error(measured, predicted)
            for measured, predicted in zip(
                measured_components,
                predicted_components,
            )
        )
        predicted_dynamic_components = (
            estimate.link_dynamic_energy_j,
            estimate.mac_dynamic_energy_j,
            estimate.memory_dynamic_energy_j,
            estimate.selection_dynamic_energy_j,
            estimate.fixed_dynamic_energy_j,
        )
        measured_dynamic_components = (
            measurement.link_dynamic_energy_j,
            measurement.mac_dynamic_energy_j,
            measurement.memory_dynamic_energy_j,
            measurement.selection_dynamic_energy_j,
            measurement.fixed_dynamic_energy_j,
        )
        dynamic_component_errors.extend(
            _relative_error(measured, predicted)
            for measured, predicted in zip(
                measured_dynamic_components,
                predicted_dynamic_components,
            )
        )
        latency_errors.append(
            _relative_error(
                measurement.total_latency_s,
                estimate.total_latency_s,
            )
        )
        dynamic_errors.append(
            _relative_error(
                measurement.dynamic_energy_j,
                estimate.dynamic_energy_j,
            )
        )
        leakage_errors.append(
            _relative_error(
                measurement.leakage_power_w,
                float(service["leakage_power_w"]),
            )
        )
    if max(component_errors) > 0.15:
        raise ValueError("head-service component latency error exceeds 15%")
    if max(dynamic_component_errors) > 0.25:
        raise ValueError(
            "head-service component dynamic-energy error exceeds 25%"
        )
    if percentile(latency_errors, 0.5) > 0.10 or max(latency_errors) > 0.15:
        raise ValueError("head-service latency holdout gate failed")
    if percentile(dynamic_errors, 0.5) > 0.15 or max(dynamic_errors) > 0.25:
        raise ValueError("head-service dynamic-energy holdout gate failed")
    if percentile(leakage_errors, 0.5) > 0.15 or max(leakage_errors) > 0.25:
        raise ValueError("head-service leakage holdout gate failed")
    latency_measured = [
        sum(value.total_latency_s for value in holdouts_by_batch[batch])
        / len(holdouts_by_batch[batch])
        for batch in required_batches
    ]
    latency_predicted = [
        provisional.estimate(batch).total_latency_s
        for batch in required_batches
    ]
    dynamic_measured = [
        sum(value.dynamic_energy_j for value in holdouts_by_batch[batch])
        / len(holdouts_by_batch[batch])
        for batch in required_batches
    ]
    dynamic_predicted = [
        provisional.estimate(batch).dynamic_energy_j
        for batch in required_batches
    ]
    latency_rank = spearman_rank_correlation(
        latency_measured,
        latency_predicted,
    )
    energy_rank = spearman_rank_correlation(
        dynamic_measured,
        dynamic_predicted,
    )
    if latency_rank < 0.90:
        raise ValueError("head-service latency rank gate failed")
    if energy_rank < 0.90:
        raise ValueError("head-service energy rank gate failed")

    validation = {
        "repeat_latency_cv_max": max(repeat_latency_cv),
        "repeat_dynamic_energy_cv_max": max(repeat_energy_cv),
        "repeat_leakage_cv_max": max(repeat_leakage_cv),
        "holdout_latency_median_error": percentile(latency_errors, 0.5),
        "holdout_latency_max_error": max(latency_errors),
        "holdout_dynamic_energy_median_error": percentile(
            dynamic_errors,
            0.5,
        ),
        "holdout_dynamic_energy_max_error": max(dynamic_errors),
        "holdout_dynamic_component_max_error": max(
            dynamic_component_errors
        ),
        "holdout_leakage_median_error": percentile(
            leakage_errors,
            0.5,
        ),
        "holdout_leakage_max_error": max(leakage_errors),
        "holdout_latency_rank_correlation": latency_rank,
        "holdout_dynamic_energy_rank_correlation": energy_rank,
        "sampled_logit_max_abs_error": max(
            value.logit_max_abs_error for value in measurements
        ),
        "sampled_logit_mean_abs_error_max": max(
            value.logit_mean_abs_error for value in measurements
        ),
        "sampled_topk_set_agreement_min": min(
            value.topk_set_agreement for value in measurements
        ),
        "selected_token_exact_match_count": len(measurements),
        "numerical_sample_count": len(measurements),
        "holdout_count": len(holdouts),
        "measurement_count": len(measurements),
    }
    return replace(provisional, validation=validation)


def load_bf16_head_service_artifact(
    path: str | Path,
    *,
    model_name: str,
    model_revision: str,
    hidden_size: int,
    vocab_size: int,
    tie_embeddings: bool,
    required_batches: Sequence[int],
) -> BF16HeadServiceStatus:
    """Load and validate one exact remote-head calibration artifact."""

    source = Path(path).resolve()
    payload = b""
    artifact_sha256 = ""
    failures: list[str] = []
    calibration: BF16HeadServiceCalibration | None = None
    try:
        payload = source.read_bytes()
        artifact_sha256 = hashlib.sha256(payload).hexdigest()
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(raw, Mapping):
            raise TypeError("head-service artifact root must be an object")
        batches = tuple(sorted({_positive_int(value, "batch") for value in required_batches}))
        if not batches:
            raise ValueError("required batch scope must be non-empty")
        calibration = _parse_calibration(
            raw,
            artifact_sha256=artifact_sha256,
            model_name=model_name,
            model_revision=model_revision,
            hidden_size=_positive_int(hidden_size, "hidden_size"),
            vocab_size=_positive_int(vocab_size, "vocab_size"),
            tie_embeddings=bool(tie_embeddings),
            required_batches=batches,
        )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    return BF16HeadServiceStatus(
        source_path=source,
        artifact_sha256=artifact_sha256,
        failures=tuple(failures),
        calibration=calibration,
    )


__all__ = [
    "BF16HeadServiceCalibration",
    "BF16HeadServiceEstimate",
    "BF16HeadServiceStatus",
    "HEAD_LOGIT_MAX_ABS_ERROR",
    "HEAD_LOGIT_MEAN_ABS_ERROR",
    "HEAD_SERVICE_MODE",
    "HEAD_SERVICE_SCHEMA",
    "HEAD_TOPK_MIN_AGREEMENT",
    "HEAD_VALIDATION_TOPK",
    "HeadServiceMeasurement",
    "DECODE_BF16_HEAD",
    "DECODE_MX_HEAD",
    "EXTERNAL_BF16_HEAD",
    "LOCAL_HEAD_COMPUTE_IDEALIZATION",
    "LOCAL_HEAD_IDEALIZATIONS",
    "LOCAL_HEAD_MODE",
    "LOCAL_MX_HEAD_LOGIT_DTYPE",
    "LOCAL_MX_HEAD_MODE",
    "LOCAL_MX_HEAD_PRECISION_POLICY",
    "LOCAL_MX_HEAD_SCHEMA",
    "LOCAL_MX_HEAD_SELECTION_POLICY",
    "OUTPUT_HEAD_IDEALIZATIONS",
    "OUTPUT_HEAD_LOCATIONS",
    "OUTPUT_HEAD_SERVICE_MODES",
    "composite_system_calibration_id",
    "head_service_status_valid",
    "local_head_boundary_status",
    "local_head_system_calibration_id",
    "local_mx_head_boundary_status",
    "local_mx_head_breakdown_valid",
    "local_mx_head_status_valid",
    "load_bf16_head_service_artifact",
    "require_content_addressed_id",
]
