"""Content-addressed Qwen3 rank-local GQA KV-head-reuse sensitivity.

The lane derives one ``KV_HEAD_REUSE=true`` candidate from each otherwise
identical explicit false control, runs both through the complete analytic
decode loop, and records the exact physical KV traffic and all retained model
components.  Rank-local Hkv=1 is emitted only as a structural-prune receipt.
Every result is projected evidence and is ineligible for strict selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

from decode_dse.hardware.design_space import (
    KV_HEAD_REUSE_NOOP_REASON,
    HardwareCandidate,
    kv_head_reuse_candidate_status,
)
from decode_dse.simulator_bridge import DecodeSimulator


SENSITIVITY_SCHEMA = "decode-qwen3-gqa-kv-head-reuse-sensitivity/v1"
RECEIPT_SCHEMA = "decode-qwen3-gqa-kv-head-reuse-sensitivity-receipt/v1"
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
TARGET_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
TARGET_KV_HEADS = 4
TARGET_TP_DEGREES = (1, 2, 4)
MEASURED_EMULATOR_LATENCY_DELTA = {2: -0.0061, 4: -0.0116}
CLASSIFICATION = {
    "nonpublication_sensitivity": True,
    "publication_valid": False,
    "publication_rankable": False,
    "hardware_rankable": False,
    "selection_eligible": False,
    "strict_selection_eligible": False,
    "timing_selection_allowed": False,
    "compiler_valid": False,
    "emulator_valid": False,
    "rtl_valid": False,
    "power_valid": False,
    "full_model_timing_measured": False,
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


def _file_hash(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_bytes(_jsonable(value)))


def _hashed(value: Mapping[str, Any], *, field: str = "content_hash") -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = _content_hash(result)
    return result


def _validate_hashed(value: Mapping[str, Any], *, field: str = "content_hash") -> None:
    digest = value.get(field)
    _require_sha256(field, digest)
    body = dict(value)
    body.pop(field, None)
    if digest != _content_hash(body):
        raise ValueError(f"{field} does not authenticate its object")


@dataclass(frozen=True)
class KVHeadReuseWorkload:
    """Fixed serving loop shared by every false/true pair."""

    input_sequence_tokens: int
    output_sequence_tokens: int
    stride: int = 1
    runtime_hbm_reserve_bytes_per_chip: int = 536_870_912
    batch_packed_attention: bool = False
    kv_layout: str = "dense_selector"

    def __post_init__(self) -> None:
        for name in ("input_sequence_tokens", "output_sequence_tokens", "stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.stride > self.output_sequence_tokens:
            raise ValueError("stride cannot exceed output_sequence_tokens")
        if (
            isinstance(self.runtime_hbm_reserve_bytes_per_chip, bool)
            or not isinstance(self.runtime_hbm_reserve_bytes_per_chip, int)
            or self.runtime_hbm_reserve_bytes_per_chip < 0
        ):
            raise ValueError(
                "runtime_hbm_reserve_bytes_per_chip must be a non-negative integer"
            )
        if not isinstance(self.batch_packed_attention, bool):
            raise TypeError("batch_packed_attention must be boolean")
        if self.kv_layout != "dense_selector":
            raise ValueError("KV-head reuse sensitivity requires dense_selector KV")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_sequence_tokens": self.input_sequence_tokens,
            "output_sequence_tokens": self.output_sequence_tokens,
            "stride": self.stride,
            "runtime_hbm_reserve_bytes_per_chip": (
                self.runtime_hbm_reserve_bytes_per_chip
            ),
            "batch_packed_attention": self.batch_packed_attention,
            "kv_layout": self.kv_layout,
        }

    @property
    def workload_id(self) -> str:
        return "kv-reuse-workload-" + _content_hash(self.to_dict())


@dataclass(frozen=True)
class _EvaluationContext:
    simulator: DecodeSimulator
    precision: Mapping[str, Any]
    workload: KVHeadReuseWorkload


def _source_file_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _file_hash(resolved)}


def _source_bindings(
    *,
    simulator: DecodeSimulator,
    model_config_path: Path,
    hardware_config_path: Path,
    custom_isa_path: Path,
) -> dict[str, Any]:
    simulator_root = Path(simulator.model_json).resolve().parents[4]
    # A custom model path need not live below the Simulator checkout.
    try:
        from decode_dse.hardware.power_bridge import resolve_simulator_root

        simulator_root = resolve_simulator_root().root.resolve()
    except (ImportError, FileNotFoundError):
        pass
    files = {
        "sensitivity_lane": Path(__file__).resolve(),
        "software_design_space": Path(__file__).with_name("design_space.py"),
        "model_config": model_config_path,
        "hardware_config": hardware_config_path,
        "custom_isa": custom_isa_path,
        "simulator_disagg_decode": (
            simulator_root / "analytic_models" / "performance" / "disagg_decode.py"
        ),
        "simulator_perf_model": (
            simulator_root / "analytic_models" / "performance" / "perf_model.py"
        ),
        "simulator_packed_kv": (
            simulator_root / "analytic_models" / "disagg_serve" / "packed_kv.py"
        ),
        "simulator_physical_ledger": (
            simulator_root / "analytic_models" / "disagg_serve" / "physical_ledger.py"
        ),
        "simulator_body_weight_layout": (
            simulator_root
            / "analytic_models"
            / "disagg_serve"
            / "body_weight_layout.py"
        ),
        "simulator_memory_model": (
            simulator_root / "analytic_models" / "memory" / "memory_model.py"
        ),
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "sensitivity source files are missing: " + ",".join(missing)
        )
    return _hashed(
        {
            "schema": "decode-qwen3-kv-head-reuse-source-binding/v1",
            "files": {
                name: _source_file_binding(path)
                for name, path in sorted(files.items())
            },
        }
    )


def _validate_target_dims(dims: Mapping[str, Any]) -> None:
    expected = {
        "model_type": "qwen3_moe",
        "hidden": 2048,
        "heads": 32,
        "kv_heads": TARGET_KV_HEADS,
        "head_dim": 128,
        "layers": 48,
        "inter": 768,
        "vocab": 151_936,
        "num_experts": 128,
        "experts_per_token": 8,
    }
    for field, wanted in expected.items():
        if dims.get(field) != wanted:
            raise ValueError(f"target decoder dimension {field} differs")


def _validate_precision(
    precision: Mapping[str, Any], candidates: Sequence[HardwareCandidate]
) -> None:
    required = {
        "profile_id",
        "attn_elem",
        "attn_bits",
        "ffn_elem",
        "ffn_bits",
        "kv_elem",
        "kv_bits",
        "block_size",
        "m_bits",
        "density_exp",
        "head_elem",
        "head_bits",
        "head_label",
        "head_activation_bits",
        "head_activation_elem",
        "head_activation_label",
        "head_vector_format",
        "head_matrix_storage_format",
        "head_logit_container_format",
        "head_bf16_container_precision_recovery",
        "head_operand_family_supported",
        "head_operand_family_binding",
        "head_numerical_oracle_rule",
        "head_partial_conversion_rule",
        "head_hardware_bit_parity_verified",
        "head_accumulation_chain",
        "head_numerical_matrix_mlen",
    }
    missing = sorted(required - set(precision))
    if missing:
        raise ValueError(
            "local MX head v3 precision binding is incomplete: " + ",".join(missing)
        )
    if not isinstance(precision["profile_id"], str) or not precision["profile_id"]:
        raise ValueError("precision profile_id must be non-empty")
    if int(precision["block_size"]) != 8:
        raise ValueError("sensitivity requires the native MX block size")
    if (
        int(precision["head_elem"]) != int(precision["attn_elem"])
        or not math.isclose(
            float(precision["head_bits"]),
            float(precision["attn_bits"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or precision["head_matrix_storage_format"]
        != precision["head_vector_format"]
        or precision["head_logit_container_format"] != "BF16"
        or precision["head_bf16_container_precision_recovery"] is not False
    ):
        raise ValueError("precision does not satisfy the local MX head v3 boundary")
    candidate_mlen = {candidate.mlen for candidate in candidates}
    if candidate_mlen != {int(precision["head_numerical_matrix_mlen"])}:
        raise ValueError("candidate MLEN differs from local-head numerical MLEN")


def _candidate_family_key(candidate: HardwareCandidate) -> dict[str, Any]:
    value = candidate.to_dict()
    for name in ("TP", "CHIP_COUNT", "LINK_PORTS", "KV_HEAD_REUSE"):
        value.pop(name, None)
    return value


def _validate_baseline_candidates(
    candidates: Sequence[HardwareCandidate],
) -> tuple[HardwareCandidate, ...]:
    ordered = tuple(sorted(candidates, key=lambda candidate: int(candidate.tp)))
    if tuple(int(candidate.tp) for candidate in ordered) != TARGET_TP_DEGREES:
        raise ValueError("sensitivity requires exactly one TP1, TP2, and TP4 control")
    if len({candidate.candidate_id for candidate in ordered}) != len(ordered):
        raise ValueError("baseline candidate identities must be unique")
    family = _candidate_family_key(ordered[0])
    for candidate in ordered:
        if not candidate.architecture_knobs_explicit:
            raise ValueError("sensitivity controls require explicit architecture knobs")
        if candidate.kv_head_reuse:
            raise ValueError("sensitivity inputs must be KV_HEAD_REUSE=false controls")
        if candidate.chip_count != int(candidate.tp) * int(candidate.kvp):
            raise ValueError("candidate topology is inconsistent")
        if TARGET_KV_HEADS % int(candidate.tp):
            raise ValueError("TP must own complete target KV heads")
        if _candidate_family_key(candidate) != family:
            raise ValueError(
                "TP controls must preserve geometry, KVP, workload, and all other knobs"
            )
    return ordered


def _candidate_override(
    simulator: DecodeSimulator, candidate: HardwareCandidate
) -> dict[str, Any]:
    return {
        "MLEN": candidate.mlen,
        "BLEN": candidate.blen,
        "VLEN": candidate.vlen,
        "HLEN": candidate.hlen,
        "HBM_M_Prefetch_Amount": candidate.mlen,
        "TP": int(candidate.tp),
        "KVP": int(candidate.kvp),
        "LINK_PORTS": int(candidate.link_ports),
        "SRAM_POLICY": candidate.sram_policy,
        "KV_HEAD_REUSE": bool(candidate.kv_head_reuse),
        "DRAIN_OVERLAPPED": bool(candidate.drain_overlapped),
        "EXPERT_PARALLEL_MODE": str(candidate.expert_parallel_mode),
        "LINK_GENERATION": "nvlink4",
        **simulator.hbm_overrides(
            candidate.hbm_generation, candidate.hbm_channels
        ),
    }


def _full_decode_evaluator(
    context: _EvaluationContext, candidate: HardwareCandidate
) -> Mapping[str, Any]:
    simulator = context.simulator
    decode = simulator._dd
    override = _candidate_override(simulator, candidate)
    hardware = simulator.base_hw.model_copy(update=override)
    decode.set_area_model("calibrated", dict(context.precision))
    return decode.evaluate(
        simulator.model_json,
        simulator.dims,
        hardware,
        simulator.isa_path,
        simulator.base_mem,
        dict(context.precision),
        candidate.batch,
        context.workload.input_sequence_tokens,
        context.workload.output_sequence_tokens,
        hw_over=override,
        stride=context.workload.stride,
        n_chips=candidate.chip_count,
        batch_packed_attention=context.workload.batch_packed_attention,
        hbm_gen=candidate.hbm_generation,
        hbm_channels=candidate.hbm_channels,
        kv_layout=context.workload.kv_layout,
        timing_mode=simulator.timing_mode,
        runtime_hbm_reserve_bytes=(
            context.workload.runtime_hbm_reserve_bytes_per_chip
        ),
        output_head_location=decode.DECODE_MX_HEAD,
        execution_mode=decode.LEGACY_AGGREGATE_BANDWIDTH,
    )


def _stable_local_head(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_clone(value)
    result.pop("fractions", None)
    return result


def _metric_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "tpot",
        "tps",
        "total_time",
        "first_step",
        "traffic_breakdown_per_batch_step",
        "traffic_breakdown_per_generated_token",
        "architecture_options",
        "capacity_throughput_chain",
        "body_physical_layout",
        "moe_workload",
        "local_output_head",
        "power",
        "timing_mode",
        "timing_calibrated",
        "timing_reason",
        "packed_q1_timing_validated",
        "packed_q1_timing_reason",
        "execution_mode",
        "n_chips",
        "hbm_required",
        "hbm_capacity",
        "fits_in_hbm",
        "fits_runtime",
        "max_runtime_batch",
        "parallelism",
        "avg_peak_compute_seconds",
        "avg_realized_compute_seconds",
        "avg_ideal_compute_seconds",
        "avg_memory_seconds",
        "avg_collective_seconds",
        "collective_bytes_per_batch_step",
        "collective_bytes_per_generated_token",
        "collective_breakdown_per_batch_step",
        "batch_packed_attention",
        "layer_exact_moe_route_projection",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("full decode result is incomplete: " + ",".join(missing))
    traffic_step = {
        str(name): float(value)
        for name, value in raw["traffic_breakdown_per_batch_step"].items()
    }
    traffic_token = {
        str(name): float(value)
        for name, value in raw["traffic_breakdown_per_generated_token"].items()
    }
    for traffic in (traffic_step, traffic_token):
        for name in (
            "kv_element_read_bytes",
            "kv_scale_read_bytes",
            "kv_element_write_bytes",
            "kv_scale_write_bytes",
        ):
            if (
                name not in traffic
                or not math.isfinite(traffic[name])
                or traffic[name] < 0
            ):
                raise ValueError(f"full decode traffic field {name} is invalid")
    options = _json_clone(raw["architecture_options"])
    option_area = options.get("area", {})
    control_area = float(
        option_area.get("breakdown_mm2_per_chip", {}).get(
            "KVHeadReuseControl", 0.0
        )
    )
    body = _json_clone(raw["body_physical_layout"])
    moe = _json_clone(raw["moe_workload"])
    head = _json_clone(raw["local_output_head"])
    power = _json_clone(raw["power"])
    if (
        not isinstance(body, dict)
        or not isinstance(moe, dict)
        or not isinstance(head, dict)
    ):
        raise ValueError("full model component ledgers are missing")
    capacity = {
        "system_required_bytes": int(raw["hbm_required"]),
        "system_capacity_bytes": int(raw["hbm_capacity"]),
        "slowest_rank_required_bytes": int(
            body["capacity"]["slowest_rank_required_bytes"]
        ),
        "per_chip_capacity_bytes": int(body["capacity"]["per_chip_capacity_bytes"]),
        "system_feasible": bool(body["capacity"]["system_feasible"]),
        "slowest_rank_feasible": bool(body["capacity"]["slowest_rank_feasible"]),
        "overall_feasible": bool(body["capacity"]["overall_feasible"]),
        "runtime_feasible": bool(raw["fits_runtime"]),
        "max_runtime_batch": int(raw["max_runtime_batch"]),
    }
    projection = {
        "performance": {
            "tpot_s": float(raw["tpot"]),
            "tpot_ms": float(raw["tpot"]) * 1000.0,
            "tokens_per_second": float(raw["tps"]),
            "total_time_s": float(raw["total_time"]),
            "first_step_s": float(raw["first_step"]),
        },
        "kv_traffic": {
            "per_batch_step": {
                "element_read_bytes": traffic_step["kv_element_read_bytes"],
                "scale_read_bytes": traffic_step["kv_scale_read_bytes"],
                "total_read_bytes": (
                    traffic_step["kv_element_read_bytes"]
                    + traffic_step["kv_scale_read_bytes"]
                ),
                "element_write_bytes": traffic_step["kv_element_write_bytes"],
                "scale_write_bytes": traffic_step["kv_scale_write_bytes"],
                "total_write_bytes": (
                    traffic_step["kv_element_write_bytes"]
                    + traffic_step["kv_scale_write_bytes"]
                ),
            },
            "per_generated_token": {
                "element_read_bytes": traffic_token["kv_element_read_bytes"],
                "scale_read_bytes": traffic_token["kv_scale_read_bytes"],
                "total_read_bytes": (
                    traffic_token["kv_element_read_bytes"]
                    + traffic_token["kv_scale_read_bytes"]
                ),
                "element_write_bytes": traffic_token["kv_element_write_bytes"],
                "scale_write_bytes": traffic_token["kv_scale_write_bytes"],
                "total_write_bytes": (
                    traffic_token["kv_element_write_bytes"]
                    + traffic_token["kv_scale_write_bytes"]
                ),
            },
        },
        "traffic_breakdown_per_batch_step": traffic_step,
        "traffic_breakdown_per_generated_token": traffic_token,
        "capacity": capacity,
        "capacity_throughput_chain": _json_clone(
            raw["capacity_throughput_chain"]
        ),
        "architecture_options": options,
        "kv_head_reuse_control_area_mm2_per_chip": control_area,
        "retained_loop_accounting": {
            "avg_peak_compute_seconds": float(raw["avg_peak_compute_seconds"]),
            "avg_realized_compute_seconds": float(
                raw["avg_realized_compute_seconds"]
            ),
            "avg_ideal_compute_seconds": float(
                raw["avg_ideal_compute_seconds"]
            ),
            "avg_memory_seconds": float(raw["avg_memory_seconds"]),
            "avg_collective_seconds": float(raw["avg_collective_seconds"]),
            "collective_bytes_per_batch_step": float(
                raw["collective_bytes_per_batch_step"]
            ),
            "collective_bytes_per_generated_token": float(
                raw["collective_bytes_per_generated_token"]
            ),
            "collective_breakdown_per_batch_step": _json_clone(
                raw["collective_breakdown_per_batch_step"]
            ),
            "batch_packed_attention": bool(raw["batch_packed_attention"]),
            "layer_exact_moe_route_projection": _json_clone(
                raw["layer_exact_moe_route_projection"]
            ),
        },
        "evidence": {
            "execution_mode": str(raw["execution_mode"]),
            "timing_mode": str(raw["timing_mode"]),
            "timing_calibrated": bool(raw["timing_calibrated"]),
            "timing_reason": str(raw["timing_reason"]),
            "timing_evidence_id": raw.get("timing_evidence_id"),
            "bandwidth_calibration_id": raw.get("bandwidth_calibration_id"),
            "packed_q1_timing_validated": bool(raw["packed_q1_timing_validated"]),
            "packed_q1_timing_reason": str(raw["packed_q1_timing_reason"]),
            "packed_q1_timing_contract_id": raw.get(
                "packed_q1_timing_contract_id"
            ),
            "full_model_timing_measured": False,
        },
        "topology": _json_clone(raw["parallelism"]),
        "components": {
            "local_output_head": head,
            "body_physical_layout": body,
            "moe_workload": moe,
            "power": power,
        },
        "component_hashes": {
            "local_output_head": _content_hash(head),
            "local_output_head_stable_boundary": _content_hash(
                _stable_local_head(head)
            ),
            "body_physical_layout": _content_hash(body),
            "moe_workload": _content_hash(moe),
            "power": _content_hash(power),
        },
    }
    return _hashed(projection, field="metrics_content_hash")


def _simulator_reuse_status(
    context: _EvaluationContext,
    candidate: HardwareCandidate,
    *,
    enabled: bool,
) -> dict[str, Any]:
    local_heads = TARGET_KV_HEADS // int(candidate.tp)
    value = context.simulator._dd.kv_head_reuse_status(
        enabled=enabled,
        mlen=candidate.mlen,
        hlen=candidate.hlen,
        blen=candidate.blen,
        kv_heads=local_heads,
        fp_sram_depth=int(getattr(context.simulator.base_hw, "FP_SRAM_DEPTH", 512)),
    )
    return _json_clone(value)


def _software_reuse_status(
    context: _EvaluationContext,
    candidate: HardwareCandidate,
    *,
    enabled: bool,
) -> dict[str, Any]:
    return _json_clone(
        kv_head_reuse_candidate_status(
            enabled=enabled,
            mlen=candidate.mlen,
            blen=candidate.blen,
            hlen=candidate.hlen,
            local_kv_heads=TARGET_KV_HEADS // int(candidate.tp),
            fp_sram_depth=int(
                getattr(context.simulator.base_hw, "FP_SRAM_DEPTH", 512)
            ),
        )
    )


def _pair_delta(
    baseline: Mapping[str, Any], variant: Mapping[str, Any], local_heads: int
) -> dict[str, Any]:
    base_perf = baseline["performance"]
    true_perf = variant["performance"]
    base_kv = baseline["kv_traffic"]["per_generated_token"]
    true_kv = variant["kv_traffic"]["per_generated_token"]
    base_area = float(baseline["kv_head_reuse_control_area_mm2_per_chip"])
    true_area = float(variant["kv_head_reuse_control_area_mm2_per_chip"])
    base_loop = baseline["retained_loop_accounting"]
    true_loop = variant["retained_loop_accounting"]
    return {
        "tpot_s": float(true_perf["tpot_s"]) - float(base_perf["tpot_s"]),
        "tpot_fraction": (
            float(true_perf["tpot_s"]) / float(base_perf["tpot_s"]) - 1.0
        ),
        "tokens_per_second": (
            float(true_perf["tokens_per_second"])
            - float(base_perf["tokens_per_second"])
        ),
        "tokens_per_second_fraction": (
            float(true_perf["tokens_per_second"])
            / float(base_perf["tokens_per_second"])
            - 1.0
        ),
        "kv_read_bytes_per_generated_token": (
            float(true_kv["total_read_bytes"])
            - float(base_kv["total_read_bytes"])
        ),
        "kv_read_fraction": (
            float(true_kv["total_read_bytes"]) / float(base_kv["total_read_bytes"])
            - 1.0
        ),
        "exact_kv_read_reduction_factor": local_heads,
        "control_area_mm2_per_chip": true_area - base_area,
        "avg_realized_compute_seconds": (
            float(true_loop["avg_realized_compute_seconds"])
            - float(base_loop["avg_realized_compute_seconds"])
        ),
        "avg_ideal_compute_seconds": (
            float(true_loop["avg_ideal_compute_seconds"])
            - float(base_loop["avg_ideal_compute_seconds"])
        ),
        "avg_memory_seconds": (
            float(true_loop["avg_memory_seconds"])
            - float(base_loop["avg_memory_seconds"])
        ),
        "avg_collective_seconds": (
            float(true_loop["avg_collective_seconds"])
            - float(base_loop["avg_collective_seconds"])
        ),
        "system_required_hbm_bytes": (
            int(variant["capacity"]["system_required_bytes"])
            - int(baseline["capacity"]["system_required_bytes"])
        ),
        "slowest_rank_required_hbm_bytes": (
            int(variant["capacity"]["slowest_rank_required_bytes"])
            - int(baseline["capacity"]["slowest_rank_required_bytes"])
        ),
        "measured_emulator_latency_delta_fraction_reference": (
            MEASURED_EMULATOR_LATENCY_DELTA[local_heads]
        ),
        "measured_delta_applied_to_full_model_projection": False,
    }


def _pair_fields_except_reuse(
    baseline: HardwareCandidate, variant: HardwareCandidate
) -> bool:
    false_value = baseline.to_dict()
    true_value = variant.to_dict()
    false_value.pop("KV_HEAD_REUSE")
    true_value.pop("KV_HEAD_REUSE")
    return false_value == true_value


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-6)


def _validate_candidate_record(
    record: Mapping[str, Any], *, expected_reuse: bool
) -> HardwareCandidate:
    candidate = HardwareCandidate.from_dict(record["candidate"])
    if candidate.candidate_id != record.get("candidate_id"):
        raise ValueError("candidate ID does not authenticate its fields")
    if candidate.kv_head_reuse is not expected_reuse:
        raise ValueError("candidate reuse setting differs from its pair role")
    if not candidate.architecture_knobs_explicit:
        raise ValueError("sensitivity candidate knobs must remain explicit")
    return candidate


def _validate_metric_candidate_binding(
    metrics: Mapping[str, Any], candidate: HardwareCandidate
) -> None:
    topology = metrics["topology"]
    expected_topology = {
        "tp": int(candidate.tp),
        "kvp": int(candidate.kvp),
        "chip_count": candidate.chip_count,
        "link_ports": int(candidate.link_ports),
        "sram_policy": candidate.sram_policy,
        "kv_head_reuse": bool(candidate.kv_head_reuse),
    }
    for field, expected in expected_topology.items():
        if topology.get(field) != expected:
            raise ValueError(f"full-loop topology field {field} differs")
    options = metrics["architecture_options"]
    reuse = options.get("kv_head_reuse", {})
    if (
        options.get("explicit") is not True
        or options.get("expert_parallel_mode") != candidate.expert_parallel_mode
        or reuse.get("requested") is not candidate.kv_head_reuse
        or reuse.get("enabled") is not candidate.kv_head_reuse
    ):
        raise ValueError("full-loop architecture option binding differs")
    head = metrics["components"]["local_output_head"]
    if (
        head.get("schema_version") != "decode-local-mx-head-breakdown/v3"
        or head.get("operator") != "decode_lm_head"
        or head.get("topology", {}).get("tp") != int(candidate.tp)
        or head.get("topology", {}).get("kvp") != int(candidate.kvp)
        or head.get("topology", {}).get("chip_count") != candidate.chip_count
    ):
        raise ValueError("decode-local MX head binding differs")


def _validate_pair(case: Mapping[str, Any]) -> None:
    local_heads = int(case["rank_local_kv_heads"])
    baseline = case["baseline"]
    variant = case["reuse_variant"]
    if variant is None:
        raise ValueError("legal reuse case omitted its true variant")
    false_object = _validate_candidate_record(baseline, expected_reuse=False)
    true_object = _validate_candidate_record(variant, expected_reuse=True)
    if int(false_object.tp) != int(case["tp"]) or int(true_object.tp) != int(
        case["tp"]
    ):
        raise ValueError("pair candidate topology differs from its case")
    false_candidate = dict(baseline["candidate"])
    true_candidate = dict(variant["candidate"])
    if false_candidate.get("KV_HEAD_REUSE") is not False:
        raise ValueError("pair baseline did not disable KV-head reuse")
    if true_candidate.get("KV_HEAD_REUSE") is not True:
        raise ValueError("pair variant did not enable KV-head reuse")
    false_candidate.pop("KV_HEAD_REUSE")
    true_candidate.pop("KV_HEAD_REUSE")
    if false_candidate != true_candidate:
        raise ValueError("reuse pair changed fields other than KV_HEAD_REUSE")
    for metrics in (baseline["metrics"], variant["metrics"]):
        _validate_hashed(metrics, field="metrics_content_hash")
    _validate_metric_candidate_binding(baseline["metrics"], false_object)
    _validate_metric_candidate_binding(variant["metrics"], true_object)
    false_metrics = baseline["metrics"]
    true_metrics = variant["metrics"]
    for scope in ("per_batch_step", "per_generated_token"):
        false_kv = false_metrics["kv_traffic"][scope]
        true_kv = true_metrics["kv_traffic"][scope]
        for plane in ("element_read_bytes", "scale_read_bytes", "total_read_bytes"):
            if not _close(false_kv[plane], float(true_kv[plane]) * local_heads):
                raise ValueError("KV read traffic does not match the rank-local factor")
        for plane in ("element_write_bytes", "scale_write_bytes", "total_write_bytes"):
            if not _close(false_kv[plane], true_kv[plane]):
                raise ValueError("KV-head reuse changed KV storage/write traffic")
    if false_metrics["capacity"] != true_metrics["capacity"]:
        raise ValueError("KV-head reuse changed the persistent capacity ledger")
    for scope in (
        "traffic_breakdown_per_batch_step",
        "traffic_breakdown_per_generated_token",
    ):
        false_traffic = false_metrics[scope]
        true_traffic = true_metrics[scope]
        if set(false_traffic) != set(true_traffic):
            raise ValueError("reuse pair traffic schemas differ")
        for field in false_traffic:
            if field in {"kv_element_read_bytes", "kv_scale_read_bytes"}:
                continue
            if not _close(false_traffic[field], true_traffic[field]):
                raise ValueError(f"KV-head reuse changed non-read traffic {field}")
    false_loop = false_metrics["retained_loop_accounting"]
    true_loop = true_metrics["retained_loop_accounting"]
    for field in (
        "avg_peak_compute_seconds",
        "avg_collective_seconds",
        "collective_bytes_per_batch_step",
        "collective_bytes_per_generated_token",
        "collective_breakdown_per_batch_step",
        "batch_packed_attention",
        "layer_exact_moe_route_projection",
    ):
        if false_loop[field] != true_loop[field]:
            raise ValueError(f"KV-head reuse changed retained loop field {field}")
    if float(false_metrics["kv_head_reuse_control_area_mm2_per_chip"]) != 0.0:
        raise ValueError("false control unexpectedly carries reuse-control area")
    if float(true_metrics["kv_head_reuse_control_area_mm2_per_chip"]) <= 0.0:
        raise ValueError("true reuse variant omitted its control area")
    false_hashes = false_metrics["component_hashes"]
    true_hashes = true_metrics["component_hashes"]
    for field in (
        "local_output_head_stable_boundary",
        "body_physical_layout",
        "moe_workload",
    ):
        if false_hashes[field] != true_hashes[field]:
            raise ValueError(f"KV-head reuse changed retained {field}")
    expected_delta = MEASURED_EMULATOR_LATENCY_DELTA[local_heads]
    if case["measured_emulator_latency_delta_fraction"] != expected_delta:
        raise ValueError("case measured delta reference differs")
    if case["multi_chip_rank_local_packed_q1_timing_matched"] is not False:
        raise ValueError("rank-local PackedQ1 timing was promoted")
    if case.get("delta") != _pair_delta(false_metrics, true_metrics, local_heads):
        raise ValueError("reported pair delta differs from exact metrics")
    if any(
        CLASSIFICATION[field] != case["classification"][field]
        for field in CLASSIFICATION
    ):
        raise ValueError("case classification differs from sensitivity scope")


def validate_qwen3_kv_head_reuse_sensitivity(
    artifact: Mapping[str, Any], *, verify_source_files: bool = True
) -> None:
    """Fail closed on schema, hashes, target, pair parity, and evidence scope."""

    _validate_hashed(artifact)
    if artifact.get("schema") != SENSITIVITY_SCHEMA:
        raise ValueError("unsupported KV-head-reuse sensitivity schema")
    target = artifact.get("target", {})
    if (
        target.get("model") != TARGET_MODEL
        or target.get("model_revision") != TARGET_REVISION
        or int(target.get("global_kv_heads", 0)) != TARGET_KV_HEADS
        or target.get("output_head_location") != "decode_local_mx_head"
    ):
        raise ValueError("sensitivity target binding differs")
    if artifact.get("classification") != CLASSIFICATION:
        raise ValueError("sensitivity classification differs")
    _require_sha256("precision content hash", artifact["precision"]["content_hash"])
    precision_body = dict(artifact["precision"])
    precision_digest = precision_body.pop("content_hash")
    if precision_digest != _content_hash(precision_body):
        raise ValueError("precision binding is not content addressed")
    if (
        artifact["precision"]["specification"].get("profile_id")
        != artifact["precision"]["profile_id"]
    ):
        raise ValueError("precision profile identity differs")
    workload = artifact["workload"]
    if workload.get("workload_id") != "kv-reuse-workload-" + _content_hash(
        workload["settings"]
    ):
        raise ValueError("workload identity differs")
    _validate_hashed(workload)
    cases = artifact.get("cases")
    if not isinstance(cases, list) or [case.get("tp") for case in cases] != list(
        TARGET_TP_DEGREES
    ):
        raise ValueError("sensitivity cases must cover TP1, TP2, and TP4")
    for case in cases:
        _validate_hashed(case, field="case_content_hash")
        tp = int(case["tp"])
        local_heads = TARGET_KV_HEADS // tp
        if int(case["rank_local_kv_heads"]) != local_heads:
            raise ValueError("rank-local KV-head count differs")
        _validate_hashed(case["baseline"]["metrics"], field="metrics_content_hash")
        if tp in (1, 2):
            _validate_pair(case)
        else:
            baseline_candidate = _validate_candidate_record(
                case["baseline"], expected_reuse=False
            )
            if int(baseline_candidate.tp) != tp:
                raise ValueError("TP4 baseline topology differs")
            _validate_metric_candidate_binding(
                case["baseline"]["metrics"], baseline_candidate
            )
            if case["reuse_variant"] is not None:
                raise ValueError("TP4 structural no-op must not be evaluated")
            prune = case.get("structural_prune", {})
            if (
                prune.get("reason") != KV_HEAD_REUSE_NOOP_REASON
                or prune.get("structural_no_op") is not True
                or prune.get("true_candidate_constructed") is not False
                or prune.get("true_candidate_evaluated") is not False
                or float(prune.get("control_area_mm2_per_chip", -1.0)) != 0.0
            ):
                raise ValueError("TP4 structural-prune receipt differs")
        for metrics in (
            case["baseline"]["metrics"],
            *(
                (case["reuse_variant"]["metrics"],)
                if case["reuse_variant"] is not None
                else ()
            ),
        ):
            if (
                metrics["components"]["local_output_head"].get("profile_id")
                != artifact["precision"]["profile_id"]
            ):
                raise ValueError("full-loop local head profile differs")
        if int(case["baseline"]["candidate"]["BATCH"]) != int(
            workload["batch"]
        ):
            raise ValueError("case batch differs from workload binding")
    source_bindings = artifact.get("source_bindings", {})
    _validate_hashed(source_bindings)
    if source_bindings.get("schema") != (
        "decode-qwen3-kv-head-reuse-source-binding/v1"
    ):
        raise ValueError("sensitivity source-binding schema differs")
    sources = source_bindings.get("files", {})
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sensitivity source bindings are missing")
    for name, binding in sources.items():
        _require_sha256(f"source {name}", binding.get("sha256"))
        path = Path(str(binding.get("path", "")))
        if verify_source_files:
            if not path.is_file() or _file_hash(path) != binding["sha256"]:
                raise ValueError(f"source binding {name} changed")


def build_qwen3_kv_head_reuse_sensitivity(
    *,
    precision: Mapping[str, Any],
    baseline_candidates: Sequence[HardwareCandidate],
    workload: KVHeadReuseWorkload,
    model_config_path: str | os.PathLike[str],
    hardware_config_path: str | os.PathLike[str],
    custom_isa_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run the isolated TP1/2 sensitivity and authenticate the TP4 prune."""

    model_path = Path(model_config_path).resolve()
    hardware_path = Path(hardware_config_path).resolve()
    isa_path = Path(custom_isa_path).resolve()
    for name, path in (
        ("model_config_path", model_path),
        ("hardware_config_path", hardware_path),
        ("custom_isa_path", isa_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is not a file: {path}")
    candidates = _validate_baseline_candidates(baseline_candidates)
    _validate_precision(precision, candidates)
    simulator = DecodeSimulator(
        str(model_path), settings_toml=hardware_path, isa_path=isa_path
    )
    _validate_target_dims(simulator.dims)
    context = _EvaluationContext(
        simulator=simulator,
        precision=_json_clone(precision),
        workload=workload,
    )
    cases: list[dict[str, Any]] = []
    for candidate in candidates:
        tp = int(candidate.tp)
        local_heads = TARGET_KV_HEADS // tp
        baseline_status = _simulator_reuse_status(
            context, candidate, enabled=False
        )
        baseline_metrics = _metric_projection(
            _full_decode_evaluator(context, candidate)
        )
        baseline = {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
            "reuse_status": baseline_status,
            "metrics": baseline_metrics,
        }
        software_true = _software_reuse_status(context, candidate, enabled=True)
        simulator_true = _simulator_reuse_status(context, candidate, enabled=True)
        if tp == 4:
            if (
                software_true["legal"] is not False
                or software_true["structural_no_op"] is not True
                or software_true["legality_reason"] != KV_HEAD_REUSE_NOOP_REASON
                or simulator_true["supported"] is not False
                or simulator_true["legality_reason"] != KV_HEAD_REUSE_NOOP_REASON
            ):
                raise ValueError("TP4 rank-local Hkv=1 was not structurally pruned")
            case = {
                "schema": "decode-qwen3-gqa-kv-head-reuse-case/v1",
                "tp": tp,
                "kvp": int(candidate.kvp),
                "chip_count": candidate.chip_count,
                "rank_local_kv_heads": local_heads,
                "baseline": baseline,
                "reuse_variant": None,
                "structural_prune": {
                    "reason": KV_HEAD_REUSE_NOOP_REASON,
                    "structural_no_op": True,
                    "software_legality": software_true,
                    "simulator_legality": simulator_true,
                    "true_candidate_constructed": False,
                    "true_candidate_evaluated": False,
                    "control_area_mm2_per_chip": 0.0,
                    "projected_delta": "not_applicable",
                },
                "measured_emulator_latency_delta_fraction": None,
                "multi_chip_rank_local_packed_q1_timing_matched": False,
                "classification": dict(CLASSIFICATION),
            }
        else:
            if (
                local_heads not in MEASURED_EMULATOR_LATENCY_DELTA
                or software_true["legal"] is not True
                or simulator_true["supported"] is not True
                or simulator_true["measured_latency_delta_fraction"]
                != MEASURED_EMULATOR_LATENCY_DELTA[local_heads]
            ):
                raise ValueError(
                    f"TP{tp} is not a legal measured rank-local Hkv{local_heads} case"
                )
            variant_candidate = replace(candidate, kv_head_reuse=True)
            if not _pair_fields_except_reuse(candidate, variant_candidate):
                raise ValueError("derived variant changed fields other than reuse")
            variant_metrics = _metric_projection(
                _full_decode_evaluator(context, variant_candidate)
            )
            variant = {
                "candidate_id": variant_candidate.candidate_id,
                "candidate": variant_candidate.to_dict(),
                "reuse_status": simulator_true,
                "metrics": variant_metrics,
            }
            case = {
                "schema": "decode-qwen3-gqa-kv-head-reuse-case/v1",
                "tp": tp,
                "kvp": int(candidate.kvp),
                "chip_count": candidate.chip_count,
                "rank_local_kv_heads": local_heads,
                "baseline": baseline,
                "reuse_variant": variant,
                "structural_prune": None,
                "delta": _pair_delta(
                    baseline_metrics, variant_metrics, local_heads
                ),
                "measured_emulator_latency_delta_fraction": (
                    MEASURED_EMULATOR_LATENCY_DELTA[local_heads]
                ),
                "measured_delta_applied_to_full_model_projection": False,
                "multi_chip_rank_local_packed_q1_timing_matched": False,
                "evidence_scope": (
                    "exact analytic physical K/V traffic; measured emulator "
                    f"kernel delta reference at rank-local Hkv={local_heads}; "
                    "full-model TPOT/TPS and multi-chip execution remain projected"
                ),
                "classification": dict(CLASSIFICATION),
            }
        cases.append(_hashed(case, field="case_content_hash"))
    precision_binding = {
        "profile_id": str(precision["profile_id"]),
        "specification": _json_clone(precision),
    }
    precision_binding["content_hash"] = _content_hash(precision_binding)
    workload_binding = _hashed(
        {
            "workload_id": workload.workload_id,
            "settings": workload.to_dict(),
            "batch": candidates[0].batch,
        }
    )
    artifact = {
        "schema": SENSITIVITY_SCHEMA,
        "target": {
            "model": TARGET_MODEL,
            "model_revision": TARGET_REVISION,
            "tokenizer_revision": TARGET_REVISION,
            "global_kv_heads": TARGET_KV_HEADS,
            "tp_degrees": list(TARGET_TP_DEGREES),
            "output_head_location": "decode_local_mx_head",
        },
        "precision": precision_binding,
        "workload": workload_binding,
        "source_bindings": _source_bindings(
            simulator=simulator,
            model_config_path=model_path,
            hardware_config_path=hardware_path,
            custom_isa_path=isa_path,
        ),
        "protocol": {
            "derived_space": "explicit_false_control_plus_legal_true_variant",
            "opened_true_rank_local_kv_heads": [2, 4],
            "structurally_pruned_rank_local_kv_heads": [1],
            "pair_difference": ["KV_HEAD_REUSE"],
            "full_analytic_decode_loop_invoked": True,
            "measured_kernel_delta_used_as_repricing_correction": False,
            "persistent_capacity_changes_with_reuse": False,
            "tp4_control_area_charged": False,
            "moe_power_event_receipt_issued": False,
        },
        "cases": cases,
        "classification": dict(CLASSIFICATION),
        "limitations": [
            "rank_local PackedQ1 multi-chip timing has no matched compiler/RTL receipt",
            "full-model TPOT/TPS values are analytic projections",
            "emulator latency deltas are references and are not applied as corrections",
            "the lane cannot enter strict or publication selection",
        ],
    }
    artifact = _hashed(artifact)
    validate_qwen3_kv_head_reuse_sensitivity(artifact)
    return artifact


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize_qwen3_kv_head_reuse_sensitivity(
    artifact: Mapping[str, Any], output_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Write one immutable artifact directory and authenticated receipt."""

    validate_qwen3_kv_head_reuse_sensitivity(artifact)
    digest = str(artifact["content_hash"])
    destination = Path(output_dir).resolve() / ("kv-head-reuse-" + digest)
    artifact_path = destination / "sensitivity.json"
    _atomic_json(artifact_path, artifact)
    receipt = _hashed(
        {
            "schema": RECEIPT_SCHEMA,
            "artifact_relative_path": artifact_path.name,
            "artifact_sha256": _file_hash(artifact_path),
            "artifact_content_hash": digest,
            "classification": dict(CLASSIFICATION),
        },
        field="receipt_content_hash",
    )
    receipt_path = destination / "receipt.json"
    _atomic_json(receipt_path, receipt)
    return {
        "schema": RECEIPT_SCHEMA,
        "artifact_path": str(artifact_path),
        "receipt_path": str(receipt_path),
        "artifact_sha256": receipt["artifact_sha256"],
        "artifact_content_hash": digest,
        "receipt_sha256": _file_hash(receipt_path),
        "receipt_content_hash": receipt["receipt_content_hash"],
        "classification": dict(CLASSIFICATION),
    }


def load_qwen3_kv_head_reuse_sensitivity(
    path: str | os.PathLike[str], *, verify_source_files: bool = True
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sensitivity artifact root must be an object")
    validate_qwen3_kv_head_reuse_sensitivity(
        value, verify_source_files=verify_source_files
    )
    return value


__all__ = [
    "CLASSIFICATION",
    "KVHeadReuseWorkload",
    "MEASURED_EMULATOR_LATENCY_DELTA",
    "RECEIPT_SCHEMA",
    "SENSITIVITY_SCHEMA",
    "TARGET_MODEL",
    "TARGET_REVISION",
    "build_qwen3_kv_head_reuse_sensitivity",
    "load_qwen3_kv_head_reuse_sensitivity",
    "materialize_qwen3_kv_head_reuse_sensitivity",
    "validate_qwen3_kv_head_reuse_sensitivity",
]
