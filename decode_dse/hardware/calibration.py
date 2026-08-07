"""Holdout validation gates for calibrated area, power, and timing models."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from decode_dse.hardware.statistics import spearman_rank_correlation

CALIBRATION_REPORT_SCHEMA = "decode-calibration-report"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CalibrationPair:
    """One positive measured/predicted holdout pair."""

    label: str
    measured: float
    predicted: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("calibration labels must be non-empty")
        if not math.isfinite(self.measured) or self.measured <= 0:
            raise ValueError("measured values must be positive and finite")
        if not math.isfinite(self.predicted) or self.predicted < 0:
            raise ValueError("predicted values must be non-negative and finite")

    @property
    def absolute_relative_error(self) -> float:
        return abs(self.predicted - self.measured) / self.measured

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "measured": self.measured,
            "predicted": self.predicted,
            "absolute_relative_error": self.absolute_relative_error,
        }


@dataclass(frozen=True)
class CalibrationThresholds:
    area_median_error: float = 0.10
    area_max_error: float = 0.15
    dynamic_power_median_error: float = 0.15
    dynamic_power_max_error: float = 0.25
    leakage_power_median_error: float = 0.15
    leakage_power_max_error: float = 0.25
    ranking_correlation: float = 0.90
    cycle_max_error: float = 0.05
    latency_mape: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.ranking_correlation > 1:
            raise ValueError("ranking_correlation cannot exceed one")

    def to_dict(self) -> dict[str, float]:
        return {
            "area_median_error": self.area_median_error,
            "area_max_error": self.area_max_error,
            "dynamic_power_median_error": self.dynamic_power_median_error,
            "dynamic_power_max_error": self.dynamic_power_max_error,
            "leakage_power_median_error": self.leakage_power_median_error,
            "leakage_power_max_error": self.leakage_power_max_error,
            "ranking_correlation": self.ranking_correlation,
            "cycle_max_error": self.cycle_max_error,
            "latency_mape": self.latency_mape,
        }


@dataclass(frozen=True)
class CalibrationGateReport:
    """Publication gate and its complete holdout diagnostics."""

    passed: bool
    coverage_complete: bool
    missing_signatures: tuple[str, ...]
    area_median_error: float | None
    area_max_error: float | None
    dynamic_power_median_error: float | None
    dynamic_power_max_error: float | None
    leakage_power_median_error: float | None
    leakage_power_max_error: float | None
    ranking_correlation: float | None
    cycle_max_error: float | None
    latency_mape: float | None
    failures: tuple[str, ...]
    thresholds: CalibrationThresholds
    observations_hash: str
    schema_version: str = CALIBRATION_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "coverage_complete": self.coverage_complete,
            "missing_signatures": list(self.missing_signatures),
            "area_median_error": self.area_median_error,
            "area_max_error": self.area_max_error,
            "dynamic_power_median_error": self.dynamic_power_median_error,
            "dynamic_power_max_error": self.dynamic_power_max_error,
            "leakage_power_median_error": self.leakage_power_median_error,
            "leakage_power_max_error": self.leakage_power_max_error,
            "ranking_correlation": self.ranking_correlation,
            "cycle_max_error": self.cycle_max_error,
            "latency_mape": self.latency_mape,
            "failures": list(self.failures),
            "thresholds": self.thresholds.to_dict(),
            "observations_hash": self.observations_hash,
        }

    @property
    def calibration_id(self) -> str:
        return f"dc-cal-{_canonical_hash(self.to_dict())}"


def _unique_pairs(
    values: Iterable[CalibrationPair],
    category: str,
) -> tuple[CalibrationPair, ...]:
    pairs = tuple(sorted(values, key=lambda pair: pair.label))
    labels = [pair.label for pair in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate {category} holdout labels")
    return pairs


def _errors(
    pairs: Sequence[CalibrationPair],
) -> tuple[float | None, float | None]:
    if not pairs:
        return None, None
    values = [pair.absolute_relative_error for pair in pairs]
    return statistics.median(values), max(values)


def validate_calibration(
    *,
    area: Iterable[CalibrationPair],
    dynamic_power: Iterable[CalibrationPair],
    leakage_power: Iterable[CalibrationPair],
    cycles: Iterable[CalibrationPair],
    latency: Iterable[CalibrationPair],
    required_signatures: Iterable[str],
    measured_signatures: Iterable[str],
    thresholds: CalibrationThresholds = CalibrationThresholds(),
) -> CalibrationGateReport:
    """Apply all holdout and signature-coverage gates without imputation."""

    area_pairs = _unique_pairs(area, "area")
    power_pairs = _unique_pairs(dynamic_power, "dynamic power")
    leakage_pairs = _unique_pairs(leakage_power, "leakage power")
    cycle_pairs = _unique_pairs(cycles, "cycle")
    latency_pairs = _unique_pairs(latency, "latency")
    required = tuple(sorted({str(value) for value in required_signatures}))
    measured = frozenset(str(value) for value in measured_signatures)
    missing = tuple(value for value in required if value not in measured)

    area_median, area_max = _errors(area_pairs)
    power_median, power_max = _errors(power_pairs)
    leakage_median, leakage_max = _errors(leakage_pairs)
    _, cycle_max = _errors(cycle_pairs)
    latency_mape, _ = _errors(latency_pairs)
    rank = (
        spearman_rank_correlation(
            [pair.measured for pair in power_pairs],
            [pair.predicted for pair in power_pairs],
        )
        if (
            len(power_pairs) >= 2
            and len({pair.measured for pair in power_pairs}) >= 2
            and len({pair.predicted for pair in power_pairs}) >= 2
        )
        else None
    )

    failures: list[str] = []
    if not required:
        failures.append("signature_requirements")
    if missing:
        failures.append("signature_coverage")
    for category, pairs in (
        ("area_holdouts", area_pairs),
        ("dynamic_power_holdouts", power_pairs),
        ("leakage_power_holdouts", leakage_pairs),
        ("cycle_holdouts", cycle_pairs),
        ("latency_holdouts", latency_pairs),
    ):
        if len(pairs) < 2:
            failures.append(category)
    if area_median is not None and area_median > thresholds.area_median_error:
        failures.append("area_median_error")
    if area_max is not None and area_max > thresholds.area_max_error:
        failures.append("area_max_error")
    if (
        power_median is not None
        and power_median > thresholds.dynamic_power_median_error
    ):
        failures.append("dynamic_power_median_error")
    if power_max is not None and power_max > thresholds.dynamic_power_max_error:
        failures.append("dynamic_power_max_error")
    if (
        leakage_median is not None
        and leakage_median > thresholds.leakage_power_median_error
    ):
        failures.append("leakage_power_median_error")
    if (
        leakage_max is not None
        and leakage_max > thresholds.leakage_power_max_error
    ):
        failures.append("leakage_power_max_error")
    if rank is None or rank < thresholds.ranking_correlation:
        failures.append("ranking_correlation")
    if cycle_max is not None and cycle_max > thresholds.cycle_max_error:
        failures.append("cycle_max_error")
    if latency_mape is not None and latency_mape > thresholds.latency_mape:
        failures.append("latency_mape")

    observation_content = {
        "area": [pair.to_dict() for pair in area_pairs],
        "dynamic_power": [pair.to_dict() for pair in power_pairs],
        "leakage_power": [pair.to_dict() for pair in leakage_pairs],
        "cycles": [pair.to_dict() for pair in cycle_pairs],
        "latency": [pair.to_dict() for pair in latency_pairs],
        "required_signatures": list(required),
        "measured_signatures": sorted(measured),
        "thresholds": thresholds.to_dict(),
    }
    return CalibrationGateReport(
        passed=not failures,
        coverage_complete=not missing,
        missing_signatures=missing,
        area_median_error=area_median,
        area_max_error=area_max,
        dynamic_power_median_error=power_median,
        dynamic_power_max_error=power_max,
        leakage_power_median_error=leakage_median,
        leakage_power_max_error=leakage_max,
        ranking_correlation=rank,
        cycle_max_error=cycle_max,
        latency_mape=latency_mape,
        failures=tuple(failures),
        thresholds=thresholds,
        observations_hash=_canonical_hash(observation_content),
    )


__all__ = [
    "CALIBRATION_REPORT_SCHEMA",
    "CalibrationGateReport",
    "CalibrationPair",
    "CalibrationThresholds",
    "validate_calibration",
]
