"""Fail-closed routed-MoE power-event accounting.

The calibrated decode-power model consumes generic operation signatures, but
the routed expert schedule must first be represented without losing its
semantic operations or parallel placement.  This module builds that semantic
ledger from the simulator's exact physical body layout and ragged expert
timing.  Only operations covered by an existing generic calibration class are
translated.  In particular, the canonical BF16 router is never relabelled as
an MX8 matrix operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


MOE_POWER_EVENT_LEDGER_SCHEMA = "decode-moe-power-event-ledger/v1"
MOE_POWER_EVENT_RECEIPT_SCHEMA = "decode-moe-power-event-receipt/v1"
MOE_POWER_EVENT_INPUT_SCHEMA = "decode-moe-power-event-input-binding/v1"
MOE_POWER_EVENT_BLOCKER = "moe_power_event_receipt_missing_or_invalid"
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
TARGET_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
BF16_ROUTER_CALIBRATION_BLOCKER = (
    "bf16_router_matrix_event_calibration_unavailable"
)
ROUTE_FILTER_CALIBRATION_BLOCKER = (
    "route_filter_event_calibration_unavailable"
)
BODY_RECEIPT_BLOCKER = "exact_route_body_receipt_unavailable"
POWER_RECEIPT_BLOCKER = "moe_power_calibration_receipt_unavailable"
EXPERT_ID_TRACE_BLOCKER = "expert_id_rank_route_trace_unavailable"
ASSIGNMENT_EQUIVALENT_SEMANTICS = (
    "each_logical_routed_expert_assignment_counts_once_per_kvp_replica_"
    "and_is_partitioned_across_tp_assignment_equivalents;matrix_event_"
    "counts_independently_include_every_executed_rank_local_shard"
)
RANK_ORDER = "kvp_major_then_tp_rank"
EXPERT_TENSOR_PARALLEL = "tensor_parallel"
EXPERT_ID_PARALLEL = "expert_id_parallel"

_REQUIRED_OPERATIONS = (
    "moe_router",
    "moe_topk",
    "moe_route_filter",
    "moe_expert_gate_up",
    "moe_expert_swiglu",
    "moe_expert_down",
    "moe_combine",
    "moe_hidden_dispatch",
    "moe_expert_output_collective",
)
_GENERIC_TRANSLATABLE_OPERATIONS = frozenset(
    {
        "moe_topk",
        "moe_expert_gate_up",
        "moe_expert_swiglu",
        "moe_expert_down",
        "moe_combine",
    }
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_moe_power_event_input_binding(
    *,
    config_sha256: str,
    workload: Mapping[str, Any],
    power_calibration_sha256: str,
) -> dict[str, str]:
    """Bind a future v1 receipt to immutable evaluator inputs."""

    if not _is_sha256(config_sha256):
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(workload, Mapping):
        raise TypeError("workload must be an object")
    if not _is_sha256(power_calibration_sha256):
        raise ValueError(
            "power_calibration_sha256 must be a lowercase SHA-256 digest"
        )
    return {
        "schema_version": MOE_POWER_EVENT_INPUT_SCHEMA,
        "config_sha256": config_sha256,
        "workload_sha256": _content_hash(dict(workload)),
        "power_calibration_sha256": power_calibration_sha256,
    }


def validate_moe_power_event_input_binding(value: Mapping[str, Any]) -> bool:
    """Return whether an evaluator input binding matches the Results v1 keyset."""

    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "schema_version",
            "config_sha256",
            "workload_sha256",
            "power_calibration_sha256",
        }
        and value.get("schema_version") == MOE_POWER_EVENT_INPUT_SCHEMA
        and all(
            _is_sha256(value.get(key))
            for key in (
                "config_sha256",
                "workload_sha256",
                "power_calibration_sha256",
            )
        )
    )


def _ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("event dimensions must be non-negative")
    return (value + divisor - 1) // divisor


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _balanced_shards(value: int, ranks: int) -> tuple[int, ...]:
    _nonnegative_int(value, "assignment count")
    _positive_int(ranks, "rank count")
    quotient, remainder = divmod(value, ranks)
    return tuple(
        quotient + int(rank < remainder) for rank in range(ranks)
    )


def _expand_kvp(per_tp_rank: Sequence[int], kvp: int) -> list[int]:
    _positive_int(kvp, "KVP")
    return [int(value) for _ in range(kvp) for value in per_tp_rank]


def _event_entry(per_rank: Sequence[int]) -> dict[str, object]:
    values = [int(value) for value in per_rank]
    if any(value < 0 for value in values):
        raise ValueError("per-rank events must be non-negative")
    return {"per_rank": values, "aggregate_system": sum(values)}


def _ragged_route_ledger(
    moe_workload: Mapping[str, Any],
    *,
    batch: int,
    experts: int,
    top_k: int,
    blen: int,
) -> tuple[int, int, Mapping[str, Any]]:
    """Validate the simulator's one-layer ragged expert schedule."""

    if moe_workload.get("schema") != "plena-routed-moe-decode-workload/v1":
        raise ValueError("routed-MoE workload schema differs")
    if (
        moe_workload.get("tokens_per_step") != batch
        or moe_workload.get("num_experts") != experts
        or moe_workload.get("experts_per_token") != top_k
    ):
        raise ValueError("routed-MoE workload geometry differs")
    ledger = moe_workload.get("expert_batch_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError("routed-MoE workload omitted its ragged ledger")
    assignments = batch * top_k
    if ledger.get("route_assignments") != assignments:
        raise ValueError("ragged route assignments do not conserve batch*top-k")
    histogram = ledger.get("expert_token_count_histogram")
    if not isinstance(histogram, Mapping) or not histogram:
        raise ValueError("ragged route histogram is missing")
    observed_assignments = 0
    observed_experts = 0
    row_tiles = 0
    for token_count_raw, expert_count_raw in histogram.items():
        token_count = _positive_int(int(token_count_raw), "expert token count")
        expert_count = _positive_int(expert_count_raw, "expert count")
        observed_assignments += token_count * expert_count
        observed_experts += expert_count
        row_tiles += _ceil_div(token_count, blen) * expert_count
    if (
        observed_assignments != assignments
        or observed_experts != ledger.get("active_experts")
        or row_tiles != ledger.get("expert_row_tiles")
    ):
        raise ValueError("ragged expert ledger does not conserve routes or rows")
    return assignments, row_tiles, ledger


def _rank_local_expert_widths(
    body_layout: Mapping[str, Any],
    *,
    tp: int,
    kvp: int,
    mlen: int,
    expert_parallel_mode: str,
) -> tuple[tuple[int, ...], bool, Mapping[str, Any]]:
    if body_layout.get("schema_version") != (
        "plena-body-weight-physical-layout/v1"
    ):
        raise ValueError("body physical-layout schema differs")
    provenance = body_layout.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("body physical layout omitted provenance")
    if (
        provenance.get("analytic_layout_valid") is not True
        or provenance.get("tp") != tp
        or provenance.get("kvp") != kvp
        or provenance.get("mlen") != mlen
        or provenance.get("expert_parallel_mode") != expert_parallel_mode
        or provenance.get("padding_order")
        != "partition_rank_local_then_pad_each_matrix_to_mlen"
    ):
        raise ValueError("body physical layout differs from the event topology")
    rank_shapes = provenance.get("rank_shapes")
    expert_shapes = (
        rank_shapes.get("experts") if isinstance(rank_shapes, Mapping) else None
    )
    if not isinstance(expert_shapes, list) or len(expert_shapes) != tp:
        raise ValueError("body layout omitted TP-rank expert shapes")
    widths: list[int] = []
    for rank, shape in enumerate(expert_shapes):
        if not isinstance(shape, Mapping) or shape.get("rank") != rank:
            raise ValueError("body expert ranks are not canonical")
        gate = shape.get("gate_up")
        down = shape.get("down")
        if not isinstance(gate, Mapping) or not isinstance(down, Mapping):
            raise ValueError("body expert rank omitted matrix shapes")
        width = _positive_int(gate.get("logical_rows"), "expert width")
        if down.get("logical_columns") != width:
            raise ValueError("gate/up and down expert widths disagree")
        widths.append(width)
    return (
        tuple(widths),
        bool(provenance.get("expert_route_assignment_exact")),
        provenance,
    )


def build_moe_power_event_ledger(
    *,
    dims: Mapping[str, Any],
    profile_id: str,
    candidate_id: str,
    expert_linear_signature: str,
    vector_signature: str,
    body_layout: Mapping[str, Any],
    moe_workload: Mapping[str, Any],
    batch: int,
    decode_steps: int,
    mlen: int,
    blen: int,
    vlen: int,
    tp: int,
    kvp: int,
    expert_parallel_mode: str,
) -> dict[str, Any]:
    """Build a semantic per-rank ledger and its safe generic translations."""

    for label, value in (
        ("batch", batch),
        ("decode_steps", decode_steps),
        ("MLEN", mlen),
        ("BLEN", blen),
        ("VLEN", vlen),
        ("TP", tp),
        ("KVP", kvp),
    ):
        _positive_int(value, label)
    if mlen % blen:
        raise ValueError("MLEN must be divisible by BLEN")
    if not profile_id or not candidate_id:
        raise ValueError("power-event identities must be non-empty")
    if not expert_linear_signature.startswith("LINEAR:"):
        raise ValueError("expert signature must be a LINEAR calibration class")
    if not vector_signature.startswith("VECTOR:"):
        raise ValueError("vector signature must be a VECTOR calibration class")
    if expert_parallel_mode not in {
        EXPERT_TENSOR_PARALLEL,
        EXPERT_ID_PARALLEL,
    }:
        raise ValueError("unsupported expert parallel mode")

    hidden = _positive_int(dims.get("hidden"), "hidden size")
    inter = _positive_int(dims.get("inter"), "expert intermediate size")
    layers = _positive_int(dims.get("layers"), "layer count")
    experts = _positive_int(dims.get("num_experts"), "expert count")
    top_k = _positive_int(dims.get("experts_per_token"), "top-k")
    if experts <= 1 or top_k > experts:
        raise ValueError("routed-MoE geometry is invalid")
    if int(dims.get("router_weight_bits", 16)) != 16:
        raise ValueError("canonical routed-MoE ledger requires the BF16 router")

    assignments_per_layer_step, row_tiles, ragged = _ragged_route_ledger(
        moe_workload,
        batch=batch,
        experts=experts,
        top_k=top_k,
        blen=blen,
    )
    widths, route_assignment_exact, body_provenance = (
        _rank_local_expert_widths(
            body_layout,
            tp=tp,
            kvp=kvp,
            mlen=mlen,
            expert_parallel_mode=expert_parallel_mode,
        )
    )
    logical_assignments = (
        assignments_per_layer_step * layers * decode_steps
    )
    physical_assignment_equivalents = logical_assignments * kvp
    assignment_equivalents_by_rank: list[int] | None
    assignment_conservation_exact = (
        expert_parallel_mode == EXPERT_TENSOR_PARALLEL
        or route_assignment_exact
    )
    if expert_parallel_mode == EXPERT_TENSOR_PARALLEL:
        assignment_equivalents_by_rank = _expand_kvp(
            _balanced_shards(logical_assignments, tp),
            kvp,
        )
    else:
        # Whole-expert placement needs layer-specific rank route histograms.
        # A global active-expert count cannot safely reconstruct them.
        assignment_equivalents_by_rank = None
        assignment_conservation_exact = False

    repeated_scale = layers * decode_steps
    router_events_rank = (
        _ceil_div(batch, blen)
        * _ceil_div(hidden, mlen)
        * _ceil_div(experts, blen)
        * repeated_scale
    )
    topk_events_rank = (
        batch
        * (
            _ceil_div(experts, vlen)
            + 3 * _ceil_div(top_k, vlen)
        )
        * repeated_scale
    )
    route_filter_events_rank = logical_assignments
    combine_events_rank = (
        logical_assignments * _ceil_div(hidden, vlen) * 2
    )

    gate_up_by_tp_rank: list[int] = []
    swiglu_by_tp_rank: list[int] = []
    down_by_tp_rank: list[int] = []
    for width in widths:
        local_width = (
            width
            if expert_parallel_mode == EXPERT_TENSOR_PARALLEL
            else inter
        )
        gate_up_by_tp_rank.append(
            2
            * row_tiles
            * _ceil_div(local_width, blen)
            * _ceil_div(hidden, mlen)
            * repeated_scale
        )
        swiglu_by_tp_rank.append(
            assignments_per_layer_step
            * _ceil_div(local_width, vlen)
            * 6
            * repeated_scale
        )
        down_by_tp_rank.append(
            row_tiles
            * _ceil_div(hidden, blen)
            * _ceil_div(local_width, mlen)
            * repeated_scale
        )

    replicated_rank = [router_events_rank] * (tp * kvp)
    topk_rank = [topk_events_rank] * (tp * kvp)
    route_filter_rank = [route_filter_events_rank] * (tp * kvp)
    combine_rank = [combine_events_rank] * (tp * kvp)
    operations = {
        "moe_router": _event_entry(replicated_rank),
        "moe_topk": _event_entry(topk_rank),
        "moe_route_filter": _event_entry(route_filter_rank),
        "moe_expert_gate_up": _event_entry(
            _expand_kvp(gate_up_by_tp_rank, kvp)
        ),
        "moe_expert_swiglu": _event_entry(
            _expand_kvp(swiglu_by_tp_rank, kvp)
        ),
        "moe_expert_down": _event_entry(
            _expand_kvp(down_by_tp_rank, kvp)
        ),
        "moe_combine": _event_entry(combine_rank),
        "moe_hidden_dispatch": _event_entry([0] * (tp * kvp)),
        "moe_expert_output_collective": _event_entry(
            [layers * decode_steps if tp > 1 else 0] * (tp * kvp)
        ),
    }

    generic_groups = (
        (
            expert_linear_signature,
            ("moe_expert_gate_up", "moe_expert_down"),
        ),
        (
            vector_signature,
            (
                "moe_topk",
                "moe_expert_swiglu",
                "moe_combine",
            ),
        ),
    )
    generic_events = []
    for signature, semantic_operations in generic_groups:
        per_rank = [
            sum(
                int(operations[operation]["per_rank"][rank])
                for operation in semantic_operations
            )
            for rank in range(tp * kvp)
        ]
        generic_events.append(
            {
                "signature": signature,
                "semantic_operations": list(semantic_operations),
                **_event_entry(per_rank),
            }
        )

    blockers = [
        BF16_ROUTER_CALIBRATION_BLOCKER,
        ROUTE_FILTER_CALIBRATION_BLOCKER,
        BODY_RECEIPT_BLOCKER,
        POWER_RECEIPT_BLOCKER,
    ]
    if not assignment_conservation_exact:
        blockers.append(EXPERT_ID_TRACE_BLOCKER)
    event_counts = {
        "per_operation": operations,
        "aggregate_system_events": sum(
            int(value["aggregate_system"]) for value in operations.values()
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": MOE_POWER_EVENT_LEDGER_SCHEMA,
        "receipt_schema_target": MOE_POWER_EVENT_RECEIPT_SCHEMA,
        "receipt_emitted": False,
        "publication_valid": False,
        "selection_eligible": False,
        "power_engine_eligible": False,
        "dense_ffn_fallback_used": False,
        "profile_id": profile_id,
        "candidate_id": candidate_id,
        "model": {
            "model_type": str(dims.get("model_type", "qwen3_moe")),
            "layers": layers,
            "experts": experts,
            "experts_per_token": top_k,
            "hidden_size": hidden,
            "expert_intermediate_size": inter,
        },
        "workload": {
            "batch": batch,
            "decode_steps": decode_steps,
            "q_len": 1,
        },
        "topology": {
            "tp": tp,
            "kvp": kvp,
            "chip_count": tp * kvp,
            "expert_parallel_mode": expert_parallel_mode,
            "rank_order": RANK_ORDER,
        },
        "dispatch_policy": {
            "mapping": (
                "replicated_hidden_local_route_filter_then_output_allreduce"
            ),
            "hidden_dispatch_event_count": 0,
            "expert_output_collective": "allreduce_if_tp_gt_1",
        },
        "assignment_count_semantics": ASSIGNMENT_EQUIVALENT_SEMANTICS,
        "assignment_conservation": {
            "exact": assignment_conservation_exact,
            "batch": batch,
            "layers": layers,
            "top_k": top_k,
            "decode_steps": decode_steps,
            "expected_logical_assignments": logical_assignments,
            "logical_observed_assignments": logical_assignments,
            "physical_replication_factor": kvp,
            "expected_physical_executed_assignments": (
                physical_assignment_equivalents
            ),
            "observed_physical_executed_assignments": (
                physical_assignment_equivalents
                if assignment_equivalents_by_rank is not None
                else None
            ),
            "physical_executed_assignments_per_rank": (
                assignment_equivalents_by_rank
            ),
        },
        "event_counts": event_counts,
        "operation_semantics": {
            "event_unit": "issued_operation_events",
            "matrix_events_exclude_separately_accounted_writeout_drains": True,
            "router": "replicated_bf16_matrix_accumulate_issues",
            "topk": "replicated_topk_and_route_probability_vector_issues",
            "route_filter": "replicated_owned_id_filter_control_issues",
            "gate_up": "rank_local_ragged_matrix_accumulate_issues",
            "swiglu": "rank_local_silu_reciprocal_and_multiply_vector_issues",
            "down": "rank_local_ragged_matrix_accumulate_issues",
            "combine": "replicated_weighted_sum_vector_issues",
            "hidden_dispatch": "zero_for_replicated_hidden_mapping",
            "expert_output_collective": "one_invocation_per_layer_if_tp_gt_1",
        },
        "generic_calibration_events": generic_events,
        "untranslated_operations": {
            "moe_router": BF16_ROUTER_CALIBRATION_BLOCKER,
            "moe_route_filter": ROUTE_FILTER_CALIBRATION_BLOCKER,
            "moe_hidden_dispatch": "zero_events",
            "moe_expert_output_collective": (
                "priced_by_explicit_link_bytes_not_generic_on_chip_signature"
            ),
        },
        "provenance": {
            "body_layout_sha256": _content_hash(body_layout),
            "moe_workload_sha256": _content_hash(moe_workload),
            "ragged_ledger_sha256": _content_hash(ragged),
            "body_compiler_layout_valid": body_provenance.get(
                "compiler_layout_valid"
            ),
            "body_rtl_layout_valid": body_provenance.get("rtl_layout_valid"),
            "route_assignment_exact": assignment_conservation_exact,
            "router_weight_precision_bits": 16,
            "router_calibration_signature": None,
            "generic_translation_operations": sorted(
                _GENERIC_TRANSLATABLE_OPERATIONS
            ),
        },
        "blockers": blockers,
    }
    payload["content_hash"] = _content_hash(payload)
    if not validate_moe_power_event_ledger(payload):
        raise AssertionError("constructed MoE power-event ledger is invalid")
    return payload


def validate_moe_power_event_ledger(value: Mapping[str, Any]) -> bool:
    """Return whether a semantic ledger is intact and fail-closed."""

    try:
        if set(value) != {
            "schema_version",
            "content_hash",
            "receipt_schema_target",
            "receipt_emitted",
            "publication_valid",
            "selection_eligible",
            "power_engine_eligible",
            "dense_ffn_fallback_used",
            "profile_id",
            "candidate_id",
            "model",
            "workload",
            "topology",
            "dispatch_policy",
            "assignment_count_semantics",
            "assignment_conservation",
            "event_counts",
            "operation_semantics",
            "generic_calibration_events",
            "untranslated_operations",
            "provenance",
            "blockers",
        }:
            return False
        body = dict(value)
        observed_hash = body.pop("content_hash")
        if observed_hash != _content_hash(body):
            return False
        if (
            value["schema_version"] != MOE_POWER_EVENT_LEDGER_SCHEMA
            or value["receipt_schema_target"]
            != MOE_POWER_EVENT_RECEIPT_SCHEMA
            or value["receipt_emitted"] is not False
            or value["publication_valid"] is not False
            or value["selection_eligible"] is not False
            or value["power_engine_eligible"] is not False
            or value["dense_ffn_fallback_used"] is not False
            or value["assignment_count_semantics"]
            != ASSIGNMENT_EQUIVALENT_SEMANTICS
            or not isinstance(value["profile_id"], str)
            or not value["profile_id"]
            or not isinstance(value["candidate_id"], str)
            or not value["candidate_id"]
        ):
            return False
        topology = value["topology"]
        if not isinstance(topology, Mapping) or set(topology) != {
            "tp",
            "kvp",
            "chip_count",
            "expert_parallel_mode",
            "rank_order",
        }:
            return False
        tp = _positive_int(topology["tp"], "TP")
        kvp = _positive_int(topology["kvp"], "KVP")
        chips = _positive_int(topology["chip_count"], "chip count")
        if (
            tp * kvp != chips
            or topology["rank_order"] != RANK_ORDER
            or topology["expert_parallel_mode"]
            not in {EXPERT_TENSOR_PARALLEL, EXPERT_ID_PARALLEL}
        ):
            return False
        dispatch = value["dispatch_policy"]
        if not isinstance(dispatch, Mapping) or dict(dispatch) != {
            "mapping": (
                "replicated_hidden_local_route_filter_then_output_allreduce"
            ),
            "hidden_dispatch_event_count": 0,
            "expert_output_collective": "allreduce_if_tp_gt_1",
        }:
            return False
        events = value["event_counts"]
        if (
            not isinstance(events, Mapping)
            or set(events) != {"per_operation", "aggregate_system_events"}
            or not isinstance(events["per_operation"], Mapping)
            or set(events["per_operation"]) != set(_REQUIRED_OPERATIONS)
        ):
            return False
        aggregate = 0
        for operation, entry in events["per_operation"].items():
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"per_rank", "aggregate_system"}
                or not isinstance(entry["per_rank"], list)
                or len(entry["per_rank"]) != chips
            ):
                return False
            per_rank = [
                _nonnegative_int(item, f"{operation} per-rank events")
                for item in entry["per_rank"]
            ]
            system = _nonnegative_int(
                entry["aggregate_system"], f"{operation} system events"
            )
            if sum(per_rank) != system:
                return False
            if operation == "moe_hidden_dispatch" and system != 0:
                return False
            if operation == "moe_expert_output_collective" and (
                (tp > 1) != (system > 0)
            ):
                return False
            if operation not in {
                "moe_hidden_dispatch",
                "moe_expert_output_collective",
            } and system == 0:
                return False
            aggregate += system
        if events["aggregate_system_events"] != aggregate:
            return False
        conservation = value["assignment_conservation"]
        if not isinstance(conservation, Mapping) or set(conservation) != {
            "exact",
            "batch",
            "layers",
            "top_k",
            "decode_steps",
            "expected_logical_assignments",
            "logical_observed_assignments",
            "physical_replication_factor",
            "expected_physical_executed_assignments",
            "observed_physical_executed_assignments",
            "physical_executed_assignments_per_rank",
        }:
            return False
        expected_logical = (
            _positive_int(conservation["batch"], "batch")
            * _positive_int(conservation["layers"], "layers")
            * _positive_int(conservation["top_k"], "top-k")
            * _positive_int(conservation["decode_steps"], "decode steps")
        )
        expected_physical = expected_logical * kvp
        if (
            conservation["expected_logical_assignments"] != expected_logical
            or conservation["logical_observed_assignments"] != expected_logical
            or conservation["physical_replication_factor"] != kvp
            or conservation["expected_physical_executed_assignments"]
            != expected_physical
        ):
            return False
        exact = conservation["exact"]
        if not isinstance(exact, bool):
            return False
        per_rank_assignments = conservation[
            "physical_executed_assignments_per_rank"
        ]
        if exact:
            if (
                not isinstance(per_rank_assignments, list)
                or len(per_rank_assignments) != chips
            ):
                return False
            assignments = [
                _nonnegative_int(item, "assignment-equivalent count")
                for item in per_rank_assignments
            ]
            if (
                sum(assignments) != expected_physical
                or conservation["observed_physical_executed_assignments"]
                != expected_physical
            ):
                return False
        elif (
            per_rank_assignments is not None
            or conservation["observed_physical_executed_assignments"] is not None
            or topology["expert_parallel_mode"] != EXPERT_ID_PARALLEL
        ):
            return False
        generic_events = value["generic_calibration_events"]
        if not isinstance(generic_events, list) or not generic_events:
            return False
        translated: set[str] = set()
        signatures: set[str] = set()
        for generic in generic_events:
            if not isinstance(generic, Mapping) or set(generic) != {
                "signature",
                "semantic_operations",
                "per_rank",
                "aggregate_system",
            }:
                return False
            signature = generic["signature"]
            semantic_operations = generic["semantic_operations"]
            if (
                not isinstance(signature, str)
                or not signature.startswith(("LINEAR:", "VECTOR:"))
                or signature in signatures
                or not isinstance(semantic_operations, list)
                or not semantic_operations
                or not set(semantic_operations).issubset(
                    _GENERIC_TRANSLATABLE_OPERATIONS
                )
            ):
                return False
            signatures.add(signature)
            translated.update(semantic_operations)
            per_rank = generic["per_rank"]
            if not isinstance(per_rank, list) or len(per_rank) != chips:
                return False
            normalized = [
                _nonnegative_int(item, "generic per-rank events")
                for item in per_rank
            ]
            if (
                sum(normalized) != generic["aggregate_system"]
                or normalized
                != [
                    sum(
                        int(events["per_operation"][operation]["per_rank"][rank])
                        for operation in semantic_operations
                    )
                    for rank in range(chips)
                ]
            ):
                return False
        if translated != _GENERIC_TRANSLATABLE_OPERATIONS:
            return False
        blockers = value["blockers"]
        if (
            not isinstance(blockers, list)
            or len(blockers) != len(set(blockers))
            or not {
                BF16_ROUTER_CALIBRATION_BLOCKER,
                ROUTE_FILTER_CALIBRATION_BLOCKER,
                BODY_RECEIPT_BLOCKER,
                POWER_RECEIPT_BLOCKER,
            }.issubset(blockers)
            or (EXPERT_ID_TRACE_BLOCKER in blockers) is exact
        ):
            return False
        untranslated = value["untranslated_operations"]
        provenance = value["provenance"]
        if (
            not isinstance(untranslated, Mapping)
            or untranslated.get("moe_router")
            != BF16_ROUTER_CALIBRATION_BLOCKER
            or "moe_router" in translated
            or not isinstance(provenance, Mapping)
            or provenance.get("router_weight_precision_bits") != 16
            or provenance.get("router_calibration_signature") is not None
            or provenance.get("route_assignment_exact") is not exact
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def generic_calibration_event_counts(
    ledger: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    """Return only the generic signatures backed by existing event classes."""

    if not validate_moe_power_event_ledger(ledger):
        raise ValueError("invalid routed-MoE power-event ledger")
    return tuple(
        (str(event["signature"]), int(event["aggregate_system"]))
        for event in ledger["generic_calibration_events"]
    )


def validate_moe_power_event_receipt_for_engine(
    receipt: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    calibration_id: str,
    calibration_sha256: str,
) -> bool:
    """Validate the engine-visible portion of the Results v1 receipt.

    The enclosing hardware artifact authenticates the row record and artifact
    provenance after serialization.  This check covers every field the power
    engine can authenticate before that serialization boundary.  A canonical
    BF16-router ledger intentionally cannot become engine-eligible today.
    """

    try:
        if (
            not validate_moe_power_event_ledger(ledger)
            or not validate_moe_power_event_input_binding(input_binding)
            or not isinstance(receipt, Mapping)
            or set(receipt)
            != {
                "schema_version",
                "content_hash",
                "publication_valid",
                "profile_id",
                "candidate_id",
                "model",
                "config_sha256",
                "artifact_provenance_sha256",
                "workload",
                "topology",
                "dispatch_policy",
                "assignment_conservation",
                "event_counts",
                "body_layout_sha256",
                "route_provenance",
                "power_calibration",
                "dense_ffn_fallback_used",
                "power_engine_consumed_exact_receipt",
            }
        ):
            return False
        body = dict(receipt)
        observed_hash = body.pop("content_hash")
        if observed_hash != _content_hash(body):
            return False
        model = receipt["model"]
        topology = receipt["topology"]
        workload = receipt["workload"]
        conservation = receipt["assignment_conservation"]
        route = receipt["route_provenance"]
        calibration = receipt["power_calibration"]
        ledger_model = ledger["model"]
        ledger_topology = ledger["topology"]
        ledger_workload = ledger["workload"]
        ledger_conservation = dict(ledger["assignment_conservation"])
        ledger_conservation.pop("exact")
        if (
            receipt["schema_version"] != MOE_POWER_EVENT_RECEIPT_SCHEMA
            or receipt["publication_valid"] is not True
            or receipt["profile_id"] != ledger["profile_id"]
            or receipt["candidate_id"] != ledger["candidate_id"]
            or receipt["dense_ffn_fallback_used"] is not False
            or receipt["power_engine_consumed_exact_receipt"] is not True
            or dict(model)
            != {
                "model_name": TARGET_MODEL,
                "model_revision": TARGET_REVISION,
                "tokenizer_revision": TARGET_REVISION,
                "model_type": "qwen3_moe",
                "layers": 48,
                "experts": 128,
                "experts_per_token": 8,
            }
            or ledger_model["model_type"] != "qwen3_moe"
            or ledger_model["layers"] != 48
            or ledger_model["experts"] != 128
            or ledger_model["experts_per_token"] != 8
            or receipt["config_sha256"]
            != input_binding["config_sha256"]
            or not _is_sha256(receipt["artifact_provenance_sha256"])
            or dict(workload)
            != {
                "sha256": input_binding["workload_sha256"],
                "batch": ledger_workload["batch"],
                "decode_steps": ledger_workload["decode_steps"],
                "q_len": 1,
            }
            or dict(topology)
            != {
                "tp": ledger_topology["tp"],
                "kvp": ledger_topology["kvp"],
                "chip_count": ledger_topology["chip_count"],
                "expert_parallel_mode": ledger_topology[
                    "expert_parallel_mode"
                ],
            }
            or receipt["dispatch_policy"] != ledger["dispatch_policy"]
            or conservation != ledger_conservation
            or receipt["event_counts"] != ledger["event_counts"]
            or receipt["body_layout_sha256"]
            != ledger["provenance"]["body_layout_sha256"]
            or not isinstance(route, Mapping)
            or set(route) != {"source", "sha256", "publication_valid"}
            or not isinstance(route["source"], str)
            or not route["source"]
            or route["sha256"]
            != ledger["provenance"]["moe_workload_sha256"]
            or route["publication_valid"] is not True
            or dict(calibration)
            != {
                "calibration_id": calibration_id,
                "sha256": calibration_sha256,
            }
            or calibration_sha256
            != input_binding["power_calibration_sha256"]
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "ASSIGNMENT_EQUIVALENT_SEMANTICS",
    "BF16_ROUTER_CALIBRATION_BLOCKER",
    "BODY_RECEIPT_BLOCKER",
    "EXPERT_ID_TRACE_BLOCKER",
    "MOE_POWER_EVENT_INPUT_SCHEMA",
    "MOE_POWER_EVENT_LEDGER_SCHEMA",
    "MOE_POWER_EVENT_BLOCKER",
    "MOE_POWER_EVENT_RECEIPT_SCHEMA",
    "POWER_RECEIPT_BLOCKER",
    "ROUTE_FILTER_CALIBRATION_BLOCKER",
    "TARGET_MODEL",
    "TARGET_REVISION",
    "build_moe_power_event_input_binding",
    "build_moe_power_event_ledger",
    "generic_calibration_event_counts",
    "validate_moe_power_event_input_binding",
    "validate_moe_power_event_ledger",
    "validate_moe_power_event_receipt_for_engine",
]
