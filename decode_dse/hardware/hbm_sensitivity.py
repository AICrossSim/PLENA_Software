"""Controlled post-selection HBM technology sensitivity schedule."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from decode_dse.hardware.admission_cost import (
    admission_correctness_status_valid,
)
from decode_dse.hardware.design_space import HardwareCandidate
from decode_dse.hardware.lm_head_service import (
    HEAD_SERVICE_MODE,
    composite_system_calibration_id,
    head_service_status_valid,
    require_content_addressed_id,
)
from decode_dse.profiles import (
    MX_BLOCK_SIZE,
    PROFILE_KIND_QUANTIZED,
    DecodePrecisionProfile,
)
from decode_dse.simulator_bridge import HBMOperatingPointStatus

HBM_SENSITIVITY_SCHEMA = "decode-hbm-sensitivity"
HBM_SENSITIVITY_GENERATIONS = (
    "HBM2",
    "HBM2E",
    "HBM3",
    "HBM3E",
    "HBM4",
)
HBM_SENSITIVITY_SOURCE_COUNT = 4
HBM_SENSITIVITY_BASELINE = "HBM2"
HBM_SENSITIVITY_OUTPUT_HEAD = "external_bf16_service"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class HBMOperatingPointResolver(Protocol):
    def hbm_operating_point(
        self,
        generation: str,
        channels: int,
    ) -> HBMOperatingPointStatus:
        ...


@dataclass(frozen=True)
class HBMSensitivitySource:
    """One selected native-datapath profile and its fixed hardware point."""

    profile: DecodePrecisionProfile
    candidate: HardwareCandidate
    numerical_result_hash: str
    hardware_result_hash: str
    head_service_artifact_sha256: str
    head_service_calibration_id: str
    head_service_provenance_id: str
    system_calibration_id: str
    timing_evidence_id: str
    baseline_bandwidth_calibration_id: str
    admission_correctness_evidence_id: str
    admission_validation_hash: str

    def __post_init__(self) -> None:
        if self.profile.kind != PROFILE_KIND_QUANTIZED:
            raise ValueError("HBM sensitivity requires quantized hardware profiles")
        if self.profile.block_size != MX_BLOCK_SIZE:
            raise ValueError("HBM sensitivity requires the native block size")
        if self.candidate.hbm_generation != HBM_SENSITIVITY_BASELINE:
            raise ValueError("HBM sensitivity sources must use the HBM2 anchor")
        _require_sha256("numerical_result_hash", self.numerical_result_hash)
        _require_sha256("hardware_result_hash", self.hardware_result_hash)
        _require_sha256(
            "head_service_artifact_sha256",
            self.head_service_artifact_sha256,
        )
        require_content_addressed_id(
            "head_service_calibration_id",
            self.head_service_calibration_id,
            prefix="bf16-head-service-",
        )
        require_content_addressed_id(
            "head_service_provenance_id",
            self.head_service_provenance_id,
            prefix="bf16-head-provenance-",
        )
        require_content_addressed_id(
            "system_calibration_id",
            self.system_calibration_id,
            prefix="decode-head-system-",
        )
        require_content_addressed_id(
            "timing_evidence_id",
            self.timing_evidence_id,
            prefix="timing-",
        )
        require_content_addressed_id(
            "baseline_bandwidth_calibration_id",
            self.baseline_bandwidth_calibration_id,
            prefix="bandwidth-operating-point-",
        )
        require_content_addressed_id(
            "admission_correctness_evidence_id",
            self.admission_correctness_evidence_id,
            prefix="admission-correctness-",
        )
        _require_sha256(
            "admission_validation_hash",
            self.admission_validation_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "profile": self.profile.to_dict(),
            "candidate_id": self.candidate.candidate_id,
            "candidate": self.candidate.to_dict(),
            "numerical_result_hash": self.numerical_result_hash,
            "hardware_result_hash": self.hardware_result_hash,
            "output_head_location": HBM_SENSITIVITY_OUTPUT_HEAD,
            "head_service_artifact_sha256": (
                self.head_service_artifact_sha256
            ),
            "head_service_calibration_id": self.head_service_calibration_id,
            "head_service_provenance_id": self.head_service_provenance_id,
            "system_calibration_id": self.system_calibration_id,
            "timing_evidence_id": self.timing_evidence_id,
            "baseline_bandwidth_calibration_id": (
                self.baseline_bandwidth_calibration_id
            ),
            "admission_correctness_evidence_id": (
                self.admission_correctness_evidence_id
            ),
            "admission_validation_hash": self.admission_validation_hash,
        }

    @property
    def source_id(self) -> str:
        return "hbm-source-" + _content_hash(self.to_dict())


@dataclass(frozen=True)
class HBMSensitivityPoint:
    """One fixed-design rerun at an explicit HBM operating point."""

    source: HBMSensitivitySource
    generation: str
    operating_point: HBMOperatingPointStatus
    schema_version: str = HBM_SENSITIVITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != HBM_SENSITIVITY_SCHEMA:
            raise ValueError("unsupported HBM sensitivity schema")
        if self.generation not in HBM_SENSITIVITY_GENERATIONS:
            raise ValueError("unsupported HBM sensitivity generation")
        if self.operating_point.generation != self.generation:
            raise ValueError("HBM technology identity differs from the point")
        if (
            self.operating_point.interface_units
            != self.source.candidate.hbm_channels
        ):
            raise ValueError("HBM sensitivity changed the interface-unit count")
        if (
            self.generation == HBM_SENSITIVITY_BASELINE
            and self.operating_point.calibration_id
            != self.source.baseline_bandwidth_calibration_id
        ):
            raise ValueError(
                "HBM2 anchor differs from the selected hardware evidence"
            )

    @property
    def candidate(self) -> HardwareCandidate:
        return replace(
            self.source.candidate,
            hbm_generation=self.generation,
        )

    @property
    def executable(self) -> bool:
        return self.operating_point.reason not in {
            "calibrated_traffic_class_missing",
            "channel_count_outside_calibration",
        }

    @property
    def cost_class(self) -> str:
        if self.operating_point.rankable:
            return "calibrated_anchor"
        return "technology_peak_sensitivity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source.source_id,
            "source_profile_id": self.source.profile.profile_id,
            "source_candidate_id": self.source.candidate.candidate_id,
            "candidate_id": self.candidate.candidate_id,
            "candidate": self.candidate.to_dict(),
            "generation": self.generation,
            "operating_point": self.operating_point.to_dict(),
            "analytical_executable": self.executable,
            "emulator_calibrated": self.operating_point.rankable,
            "cost_class": self.cost_class,
            "cross_generation_rankable": False,
            "deployment_candidate": False,
        }

    @property
    def point_id(self) -> str:
        return "hbm-sensitivity-" + _content_hash(self.to_dict())


@dataclass(frozen=True)
class HBMSensitivitySchedule:
    """The exact four-profile post-selection technology experiment."""

    sources: tuple[HBMSensitivitySource, ...]
    points: tuple[HBMSensitivityPoint, ...]

    def __post_init__(self) -> None:
        if len(self.sources) != HBM_SENSITIVITY_SOURCE_COUNT:
            raise ValueError("HBM sensitivity requires four source profiles")
        source_ids = tuple(source.source_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("HBM sensitivity source identities must be unique")
        profile_ids = tuple(source.profile.profile_id for source in self.sources)
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("HBM sensitivity profiles must be unique")
        if tuple(sorted(profile_ids)) != profile_ids:
            raise ValueError("HBM sensitivity sources are not canonically ordered")
        expected = len(self.sources) * len(HBM_SENSITIVITY_GENERATIONS)
        if len(self.points) != expected:
            raise ValueError(f"HBM sensitivity requires {expected} points")
        expected_pairs = tuple(
            (source.source_id, generation)
            for source in self.sources
            for generation in HBM_SENSITIVITY_GENERATIONS
        )
        observed_pairs = tuple(
            (point.source.source_id, point.generation)
            for point in self.points
        )
        if observed_pairs != expected_pairs:
            raise ValueError("HBM sensitivity points are not canonically ordered")
        point_ids = tuple(point.point_id for point in self.points)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("HBM sensitivity point identities must be unique")
        for field in (
            "head_service_artifact_sha256",
            "head_service_calibration_id",
            "head_service_provenance_id",
            "system_calibration_id",
            "timing_evidence_id",
            "baseline_bandwidth_calibration_id",
            "admission_correctness_evidence_id",
            "admission_validation_hash",
        ):
            identities = {getattr(source, field) for source in self.sources}
            if len(identities) != 1:
                raise ValueError(
                    f"HBM sensitivity sources have inconsistent {field}"
                )

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": HBM_SENSITIVITY_SCHEMA,
            "baseline_generation": HBM_SENSITIVITY_BASELINE,
            "generations": list(HBM_SENSITIVITY_GENERATIONS),
            "source_profile_count": len(self.sources),
            "point_count": len(self.points),
            "numerical_manifest_points_added": 0,
            "varied_fields": ["HBM_GENERATION"],
            "preserved_fields": [
                "PROFILE_ID",
                "PRECISION_PROFILE",
                "MLEN",
                "BLEN",
                "VLEN",
                "HLEN",
                "BATCH",
                "HBM_CHANNELS",
                "CHIP_COUNT",
                "OUTPUT_HEAD_SERVICE",
                "TIMING_EVIDENCE",
                "ADMISSION_CORRECTNESS_EVIDENCE",
            ],
            "cross_generation_ranking": False,
            "sources": [
                {
                    "ordinal": ordinal,
                    "source_id": source.source_id,
                    **source.to_dict(),
                }
                for ordinal, source in enumerate(self.sources)
            ],
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


def build_hbm_sensitivity_schedule(
    selected: Sequence[HBMSensitivitySource],
    resolver: HBMOperatingPointResolver,
) -> HBMSensitivitySchedule:
    """Build post-selection technology points without numerical expansion."""

    sources = tuple(
        sorted(selected, key=lambda source: source.profile.profile_id)
    )
    if len(sources) != HBM_SENSITIVITY_SOURCE_COUNT:
        raise ValueError("exactly four selected hardware profiles are required")
    points = tuple(
        HBMSensitivityPoint(
            source=source,
            generation=generation,
            operating_point=resolver.hbm_operating_point(
                generation,
                source.candidate.hbm_channels,
            ),
        )
        for source in sources
        for generation in HBM_SENSITIVITY_GENERATIONS
    )
    return HBMSensitivitySchedule(sources=sources, points=points)


def hbm_sensitivity_sources_from_hardware_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[HBMSensitivitySource, ...]:
    """Extract four remote-head deployment points from verified hardware rows."""

    values = tuple(rows)
    if len(values) != HBM_SENSITIVITY_SOURCE_COUNT:
        raise ValueError("exactly four hardware result rows are required")
    sources: list[HBMSensitivitySource] = []
    for raw in values:
        row = dict(raw)
        record_hash = str(row.pop("record_hash", ""))
        _require_sha256("hardware result hash", record_hash)
        if _content_hash(row) != record_hash:
            raise ValueError("hardware result checksum mismatch")
        if row.get("deployment_valid") is not True:
            raise ValueError("HBM sensitivity requires deployment-valid sources")
        profile = DecodePrecisionProfile.from_dict(row["profile"])
        if row.get("profile_id") != profile.profile_id:
            raise ValueError("hardware row profile identity mismatch")
        hardware = row["hardware"]
        if not isinstance(hardware, Mapping):
            raise TypeError("hardware candidate must be an object")
        candidate = HardwareCandidate.from_dict(
            hardware,
            allow_legacy_single_chip=True,
        )
        candidate_ids = {candidate.candidate_id}
        if set(hardware) == set(HardwareCandidate.LEGACY_FIELDS):
            candidate_ids.add(
                "hw-" + _content_hash(candidate.to_legacy_dict())
            )
        if row.get("candidate_id") not in candidate_ids:
            raise ValueError("hardware row candidate identity mismatch")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TypeError("hardware row metrics must be an object")
        admission = metrics.get("admission_boundary")
        if (
            not isinstance(admission, Mapping)
            or not admission_correctness_status_valid(admission)
        ):
            raise ValueError(
                "HBM sensitivity requires verified decode admission"
            )
        whole_model = metrics.get("whole_model")
        output_head = metrics.get("output_head_boundary")
        if not isinstance(whole_model, Mapping) or not isinstance(
            output_head,
            Mapping,
        ):
            raise ValueError("hardware row system boundary is incomplete")
        status = output_head.get("status")
        estimate = output_head.get("estimate")
        decoder_stack = metrics.get("decoder_stack")
        decoder_energy = (
            decoder_stack.get("calibrated_energy")
            if isinstance(decoder_stack, Mapping)
            else None
        )
        whole_energy = whole_model.get("calibrated_energy")
        if (
            whole_model.get("rankable") is not True
            or output_head.get("service_mode")
            != HEAD_SERVICE_MODE
            or not isinstance(status, Mapping)
            or not isinstance(estimate, Mapping)
            or not head_service_status_valid(status)
            or status.get("service_location") != "prefill_chip"
            or estimate.get("service_mode") != HEAD_SERVICE_MODE
            or estimate.get("service_location") != "prefill_chip"
            or not isinstance(decoder_energy, Mapping)
            or not isinstance(whole_energy, Mapping)
        ):
            raise ValueError(
                "HBM sensitivity requires the calibrated remote BF16 head"
            )
        head_calibration_id = str(status.get("calibration_id", ""))
        head_provenance_id = str(status.get("provenance_id", ""))
        if (
            estimate.get("calibration_id") != head_calibration_id
            or estimate.get("provenance_id") != head_provenance_id
        ):
            raise ValueError("remote-head estimate identity mismatch")
        decoder_calibration_id = str(
            decoder_energy.get("calibration_id", "")
        )
        system_calibration_id = composite_system_calibration_id(
            decoder_calibration_id,
            head_calibration_id,
            head_provenance_id,
        )
        if (
            whole_model.get("system_calibration_id")
            != system_calibration_id
            or whole_energy.get("calibration_id")
            != system_calibration_id
        ):
            raise ValueError("whole-model calibration identity mismatch")
        if metrics.get("timing_calibrated") is not True:
            raise ValueError("HBM sensitivity requires calibrated timing")
        timing = metrics.get("timing_decomposition")
        if (
            not isinstance(timing, Mapping)
            or timing.get("composition") != "max_compute_memory"
            or timing.get("unit") != "seconds_per_batch_step"
        ):
            raise ValueError("HBM sensitivity requires timing decomposition")
        for name in (
            "ideal_compute_seconds",
            "realized_compute_seconds",
            "memory_seconds",
        ):
            value = timing.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError("HBM timing decomposition is invalid")
        if float(timing["realized_compute_seconds"]) < float(
            timing["ideal_compute_seconds"]
        ):
            raise ValueError("HBM timing decomposition is inconsistent")
        sources.append(
            HBMSensitivitySource(
                profile=profile,
                candidate=candidate,
                numerical_result_hash=str(row["numerical_result_hash"]),
                hardware_result_hash=record_hash,
                head_service_artifact_sha256=str(
                    status.get("artifact_sha256", "")
                ),
                head_service_calibration_id=head_calibration_id,
                head_service_provenance_id=head_provenance_id,
                system_calibration_id=system_calibration_id,
                timing_evidence_id=str(metrics["timing_evidence_id"]),
                baseline_bandwidth_calibration_id=str(
                    metrics["bandwidth_calibration_id"]
                ),
                admission_correctness_evidence_id=str(
                    admission["evidence_id"]
                ),
                admission_validation_hash=str(
                    admission["numerical_validation_hash"]
                ),
            )
        )
    return tuple(
        sorted(sources, key=lambda source: source.profile.profile_id)
    )


def write_hbm_sensitivity_schedule(
    path: str | Path,
    schedule: HBMSensitivitySchedule,
) -> Path:
    """Write one immutable content-addressed schedule."""

    from decode_dse.software.sweep_plan import write_immutable_json

    return write_immutable_json(path, schedule.to_dict())


__all__ = [
    "HBMOperatingPointResolver",
    "HBMSensitivityPoint",
    "HBMSensitivitySchedule",
    "HBMSensitivitySource",
    "HBM_SENSITIVITY_BASELINE",
    "HBM_SENSITIVITY_GENERATIONS",
    "HBM_SENSITIVITY_OUTPUT_HEAD",
    "HBM_SENSITIVITY_SCHEMA",
    "HBM_SENSITIVITY_SOURCE_COUNT",
    "build_hbm_sensitivity_schedule",
    "hbm_sensitivity_sources_from_hardware_rows",
    "write_hbm_sensitivity_schedule",
]
