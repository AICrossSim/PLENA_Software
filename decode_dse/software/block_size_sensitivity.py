"""Controlled block-size sensitivity for selected decode profiles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.profiles import (
    MX_BLOCK_SIZE,
    PROFILE_KIND_QUANTIZED,
    DecodePrecisionProfile,
    format_descriptor,
)

BLOCK_SENSITIVITY_SCHEMA = "decode-block-sensitivity"
BLOCK_SENSITIVITY_REPORT_SCHEMA = "decode-block-sensitivity-report"
BLOCK_SENSITIVITY_BLOCKS = (8, 16, 32)
BLOCK_SENSITIVITY_SOURCE_COUNT = 4


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BlockSensitivityPoint:
    """One numerical block-size variant of a canonical block-8 profile."""

    source_profile: DecodePrecisionProfile
    block_size: int
    schema_version: str = BLOCK_SENSITIVITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_SENSITIVITY_SCHEMA:
            raise ValueError(
                f"unsupported block-sensitivity schema {self.schema_version!r}"
            )
        if self.source_profile.kind != PROFILE_KIND_QUANTIZED:
            raise ValueError("block sensitivity requires a quantized source profile")
        if self.source_profile.block_size != MX_BLOCK_SIZE:
            raise ValueError("the source profile must use the native block size")
        if self.block_size not in BLOCK_SENSITIVITY_BLOCKS:
            raise ValueError(
                f"block_size must be one of {BLOCK_SENSITIVITY_BLOCKS}"
            )

    @property
    def native_datapath(self) -> bool:
        return self.block_size == MX_BLOCK_SIZE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_profile_id": self.source_profile.profile_id,
            "source_profile": self.source_profile.to_dict(),
            "block_size": self.block_size,
            "native_datapath": self.native_datapath,
        }

    @property
    def point_id(self) -> str:
        return "dbs-" + _content_hash(self.to_dict())


@dataclass(frozen=True)
class BlockSensitivitySchedule:
    points: tuple[BlockSensitivityPoint, ...]

    def __post_init__(self) -> None:
        expected = BLOCK_SENSITIVITY_SOURCE_COUNT * len(
            BLOCK_SENSITIVITY_BLOCKS
        )
        if len(self.points) != expected:
            raise ValueError(f"block-sensitivity schedule requires {expected} points")
        point_ids = tuple(point.point_id for point in self.points)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("block-sensitivity schedule contains duplicate points")
        source_ids = tuple(
            dict.fromkeys(point.source_profile.profile_id for point in self.points)
        )
        if len(source_ids) != BLOCK_SENSITIVITY_SOURCE_COUNT:
            raise ValueError(
                "block-sensitivity schedule requires four source profiles"
            )
        expected_pairs = tuple(
            (source_id, block_size)
            for source_id in source_ids
            for block_size in BLOCK_SENSITIVITY_BLOCKS
        )
        observed_pairs = tuple(
            (point.source_profile.profile_id, point.block_size)
            for point in self.points
        )
        if observed_pairs != expected_pairs:
            raise ValueError("block-sensitivity points are not canonically ordered")

    @property
    def source_profile_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(point.source_profile.profile_id for point in self.points)
        )

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": BLOCK_SENSITIVITY_SCHEMA,
            "blocks": list(BLOCK_SENSITIVITY_BLOCKS),
            "native_block": MX_BLOCK_SIZE,
            "source_profile_ids": list(self.source_profile_ids),
            "point_count": len(self.points),
            "points": [
                {
                    "ordinal": ordinal,
                    "point_id": point.point_id,
                    **point.to_dict(),
                }
                for ordinal, point in enumerate(self.points)
            ],
        }
        return {**body, "schedule_hash": _content_hash(body)}

    @property
    def canonical_hash(self) -> str:
        return str(self.to_dict()["schedule_hash"])


def build_block_sensitivity_schedule(
    selected_profiles: Sequence[DecodePrecisionProfile],
) -> BlockSensitivitySchedule:
    """Create the fixed 4-by-3 post-selection numerical experiment."""

    sources = tuple(sorted(selected_profiles, key=lambda profile: profile.profile_id))
    if len(sources) != BLOCK_SENSITIVITY_SOURCE_COUNT:
        raise ValueError("exactly four selected profiles are required")
    source_ids = tuple(profile.profile_id for profile in sources)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("selected profiles must be unique")
    return BlockSensitivitySchedule(
        points=tuple(
            BlockSensitivityPoint(source_profile=profile, block_size=block_size)
            for profile in sources
            for block_size in BLOCK_SENSITIVITY_BLOCKS
        )
    )


def block_sensitivity_to_decode_quant_spec(
    point: BlockSensitivityPoint,
) -> Any:
    """Map one sensitivity point to an isolated MASE quantization build."""

    from decode_dse.software.precision_bindings import DecodeQuantSpec

    def operand(token: str) -> tuple[str, Any]:
        descriptor = format_descriptor(token)
        if descriptor.family == "mxint":
            return descriptor.family, descriptor.element_bits
        return descriptor.family, (
            descriptor.exponent_bits,
            descriptor.mantissa_bits,
        )

    profile = point.source_profile
    weight_family, weight_width = operand(profile.weight_format)
    activation_family, activation_width = operand(profile.activation_format)
    kv_family, kv_width = operand(profile.kv_format)
    return DecodeQuantSpec(
        attn_w=weight_width,
        ffn_w=weight_width,
        kv=kv_width,
        w_fmt=weight_family,
        kv_fmt=kv_family,
        weight_block=point.block_size,
        kv_block=point.block_size,
        act_w=activation_width,
        act_fmt=activation_family,
        act_block=point.block_size,
        use_gptq=False,
        use_rotation=False,
        fp_setting=profile.vector_format,
        fp_setting_attention=True,
        quant_attn_internals=True,
    )


@dataclass(frozen=True)
class BlockSensitivityObservation:
    point_id: str
    state: str
    mean_token_nll: float | None = None
    error_class: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.point_id:
            raise ValueError("point_id is required")
        if self.state not in {"succeeded", "failed", "oom"}:
            raise ValueError(f"unsupported observation state {self.state!r}")
        if self.state == "succeeded":
            if (
                self.mean_token_nll is None
                or not math.isfinite(self.mean_token_nll)
                or self.mean_token_nll < 0
            ):
                raise ValueError("successful observation requires finite NLL")
            if self.error_class is not None or self.error_message is not None:
                raise ValueError("successful observation cannot carry an error")
        else:
            if self.mean_token_nll is not None:
                raise ValueError("failed observation cannot carry NLL")
            if not self.error_class:
                raise ValueError("failed observation requires error_class")

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "state": self.state,
            "mean_token_nll": self.mean_token_nll,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


def build_block_sensitivity_report(
    schedule: BlockSensitivitySchedule,
    observations: Iterable[BlockSensitivityObservation],
) -> dict[str, Any]:
    """Join exact terminal observations without converting failures to scores."""

    indexed: dict[str, BlockSensitivityObservation] = {}
    for observation in observations:
        if observation.point_id in indexed:
            raise ValueError(
                f"duplicate block-sensitivity observation {observation.point_id}"
            )
        indexed[observation.point_id] = observation
    expected_ids = {point.point_id for point in schedule.points}
    unknown = sorted(set(indexed) - expected_ids)
    missing = sorted(expected_ids - set(indexed))
    if unknown:
        raise ValueError(f"unknown block-sensitivity points: {unknown}")

    baselines = {
        point.source_profile.profile_id: indexed.get(point.point_id)
        for point in schedule.points
        if point.native_datapath
    }
    rows: list[dict[str, Any]] = []
    comparable = not missing
    for ordinal, point in enumerate(schedule.points):
        observation = indexed.get(point.point_id)
        baseline = baselines[point.source_profile.profile_id]
        delta = None
        if (
            observation is not None
            and observation.state == "succeeded"
            and baseline is not None
            and baseline.state == "succeeded"
        ):
            delta = observation.mean_token_nll - baseline.mean_token_nll
        else:
            comparable = False
        rows.append(
            {
                "ordinal": ordinal,
                "point_id": point.point_id,
                "source_profile_id": point.source_profile.profile_id,
                "block_size": point.block_size,
                "native_datapath": point.native_datapath,
                "deployment_candidate": point.native_datapath,
                "observation": (
                    observation.to_dict() if observation is not None else None
                ),
                "nll_delta_from_block8": delta,
            }
        )

    body = {
        "schema_version": BLOCK_SENSITIVITY_REPORT_SCHEMA,
        "schedule_hash": schedule.canonical_hash,
        "complete": not missing,
        "comparable": comparable,
        "missing_point_ids": missing,
        "rows": rows,
    }
    return {**body, "report_hash": _content_hash(body)}


def write_block_sensitivity_schedule(
    path: str | Path,
    schedule: BlockSensitivitySchedule,
) -> Path:
    from decode_dse.software.sweep_plan import write_immutable_json

    return write_immutable_json(path, schedule.to_dict())


__all__ = [
    "BLOCK_SENSITIVITY_BLOCKS",
    "BLOCK_SENSITIVITY_REPORT_SCHEMA",
    "BLOCK_SENSITIVITY_SCHEMA",
    "BLOCK_SENSITIVITY_SOURCE_COUNT",
    "BlockSensitivityObservation",
    "BlockSensitivityPoint",
    "BlockSensitivitySchedule",
    "block_sensitivity_to_decode_quant_spec",
    "build_block_sensitivity_report",
    "build_block_sensitivity_schedule",
    "write_block_sensitivity_schedule",
]
