"""Accuracy, fidelity, uncertainty, and deployment selection gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from decode_dse.hardware.statistics import percentile, spearman_rank_correlation
from decode_dse.hardware.lm_head_service import (
    HEAD_SERVICE_MODE,
    composite_system_calibration_id,
    require_content_addressed_id,
)
from decode_dse.legality import StackValidity
from decode_dse.profiles import (
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    PROFILE_KIND_VECTOR_BF16_CONTROL,
    DecodePrecisionProfile,
    format_descriptor,
)

SELECTION_REPORT_SCHEMA = "decode-selection-report"
DEFAULT_REQUIRED_TASKS = ("gsm8k", "ifeval")
DEFAULT_RULER_LENGTHS = (4096, 8192, 16384, 32768)
BF16_REFERENCE_SCOPE = "split_execution_software_accuracy"
ENERGY_TIERS = frozenset({"analytic_anchored", "dc_calibrated"})


def _energy_tier_rank(value: str | None) -> int:
    if value == "dc_calibrated":
        return 0
    if value == "analytic_anchored":
        return 1
    return 2


class CalibrationEvidence(Protocol):
    passed: bool

    @property
    def calibration_id(self) -> str:
        ...


class TimingCalibrationEvidence(Protocol):
    passed: bool
    mode: str

    @property
    def evidence_id(self) -> str:
        ...


class HeadServiceEvidence(Protocol):
    passed: bool
    service_mode: str

    @property
    def calibration_id(self) -> str | None:
        ...

    @property
    def provenance_id(self) -> str | None:
        ...


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + ("\n" if newline else "")
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_optional(name: str, value: float | None, *, positive: bool = False) -> None:
    if value is None:
        return
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ParetoPoint:
    """One profile-level point used for validation fidelity and refinement."""

    profile: DecodePrecisionProfile
    mean_nll: float
    tpot_ms: float | None
    tps: float | None
    energy_per_token_j: float | None
    area_mm2: float | None
    candidate_id: str | None = None
    power_calibration_id: str | None = None
    cost_scope: str | None = None
    system_calibration_id: str | None = None
    head_service_calibration_id: str | None = None
    whole_model_rankable: bool = False
    energy_tier: str | None = None
    publication_timing_tier: str | None = None

    def __post_init__(self) -> None:
        _finite_optional("mean_nll", self.mean_nll)
        for name in ("tpot_ms", "tps", "energy_per_token_j", "area_mm2"):
            _finite_optional(name, getattr(self, name), positive=True)
        if self.publication_timing_tier is not None:
            from decode_dse.hardware.design_space import (
                PUBLICATION_TIMING_TIERS,
            )

            if self.publication_timing_tier not in PUBLICATION_TIMING_TIERS:
                raise ValueError("publication_timing_tier is unsupported")
        if self.whole_model_rankable and self.publication_timing_tier is None:
            raise ValueError(
                "rankable points require a publication timing tier"
            )
        has_power_cost = (
            self.energy_per_token_j is not None or self.area_mm2 is not None
        )
        if has_power_cost != bool(self.power_calibration_id):
            raise ValueError(
                "energy or area costs require one power calibration identity"
            )
        tier = self.energy_tier
        if has_power_cost and tier is None:
            tier = "dc_calibrated"
            object.__setattr__(self, "energy_tier", tier)
        if tier is not None and tier not in ENERGY_TIERS:
            raise ValueError("energy_tier is unsupported")
        if not has_power_cost and tier is not None:
            raise ValueError("energy_tier requires serving costs")
        has_system_cost = any(
            value is not None
            for value in (
                self.tpot_ms,
                self.tps,
                self.energy_per_token_j,
            )
        )
        if has_system_cost:
            if (
                not self.whole_model_rankable
                or self.cost_scope != "whole_model"
                or not self.system_calibration_id
                or not self.head_service_calibration_id
            ):
                raise ValueError(
                    "serving costs require a calibrated whole-model boundary"
                )
        elif any(
            (
                self.whole_model_rankable,
                self.cost_scope is not None,
                self.system_calibration_id is not None,
                self.head_service_calibration_id is not None,
            )
        ):
            raise ValueError(
                "whole-model identities require serving costs"
            )

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    @property
    def tokens_per_joule(self) -> float | None:
        if self.energy_per_token_j is None:
            return None
        return 1.0 / self.energy_per_token_j

    @property
    def edp_j_s(self) -> float | None:
        if self.energy_per_token_j is None or self.tpot_ms is None:
            return None
        return self.energy_per_token_j * self.tpot_ms / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile": self.profile.to_dict(),
            "mean_nll": self.mean_nll,
            "tpot_ms": self.tpot_ms,
            "tps": self.tps,
            "energy_per_token_j": self.energy_per_token_j,
            "energy_tier": self.energy_tier,
            "tokens_per_joule": self.tokens_per_joule,
            "edp_j_s": self.edp_j_s,
            "area_mm2": self.area_mm2,
            "candidate_id": self.candidate_id,
            "power_calibration_id": self.power_calibration_id,
            "cost_scope": self.cost_scope,
            "system_calibration_id": self.system_calibration_id,
            "head_service_calibration_id": (
                self.head_service_calibration_id
            ),
            "whole_model_rankable": self.whole_model_rankable,
        }


@dataclass(frozen=True)
class EpsilonPolicy:
    """Additive tolerances for the five promotion objectives."""

    mean_nll: float = 0.0
    tpot_ms: float = 0.0
    tps: float = 0.0
    energy_per_token_j: float = 0.0
    area_mm2: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} epsilon must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_nll": self.mean_nll,
            "tpot_ms": self.tpot_ms,
            "tps": self.tps,
            "energy_per_token_j": self.energy_per_token_j,
            "area_mm2": self.area_mm2,
        }


def _min_value(value: float | None) -> float:
    return float("inf") if value is None else value


def _max_value(value: float | None) -> float:
    return float("-inf") if value is None else value


def epsilon_dominates(
    left: ParetoPoint,
    right: ParetoPoint,
    epsilon: EpsilonPolicy,
) -> bool:
    """Return whether left additively epsilon-dominates right."""

    left_min = (
        left.mean_nll,
        _min_value(left.tpot_ms),
        _min_value(left.energy_per_token_j),
        _min_value(left.area_mm2),
    )
    right_min = (
        right.mean_nll,
        _min_value(right.tpot_ms),
        _min_value(right.energy_per_token_j),
        _min_value(right.area_mm2),
    )
    min_eps = (
        epsilon.mean_nll,
        epsilon.tpot_ms,
        epsilon.energy_per_token_j,
        epsilon.area_mm2,
    )
    weak_min = all(
        left_value <= right_value + tolerance
        for left_value, right_value, tolerance in zip(left_min, right_min, min_eps)
    )
    strict_min = any(
        left_value < right_value - tolerance
        for left_value, right_value, tolerance in zip(left_min, right_min, min_eps)
    )
    left_tps = _max_value(left.tps)
    right_tps = _max_value(right.tps)
    weak_tps = left_tps >= right_tps - epsilon.tps
    strict_tps = left_tps > right_tps + epsilon.tps
    return weak_min and weak_tps and (strict_min or strict_tps)


def epsilon_pareto_fronts(
    points: Iterable[ParetoPoint],
    epsilon: EpsilonPolicy = EpsilonPolicy(),
) -> tuple[tuple[ParetoPoint, ...], ...]:
    """Return deterministic candidate-level fronts without stochastic ranking."""

    remaining = sorted(tuple(points), key=_point_order)
    identities = tuple(_point_identity(point) for point in remaining)
    if len(identities) != len(set(identities)):
        raise ValueError("Pareto points contain duplicate candidate identities")
    fronts: list[tuple[ParetoPoint, ...]] = []
    while remaining:
        front = tuple(
            point
            for point in remaining
            if not any(
                _point_identity(other) != _point_identity(point)
                and epsilon_dominates(other, point, epsilon)
                for other in remaining
            )
        )
        if not front:
            raise RuntimeError("epsilon dominance did not produce a front")
        fronts.append(tuple(sorted(front, key=_point_order)))
        front_ids = {_point_identity(point) for point in front}
        remaining = [
            point for point in remaining if _point_identity(point) not in front_ids
        ]
    return tuple(fronts)


def _point_order(point: ParetoPoint) -> tuple[Any, ...]:
    return (
        point.mean_nll,
        _energy_tier_rank(point.energy_tier),
        _min_value(point.energy_per_token_j),
        _min_value(point.tpot_ms),
        -_max_value(point.tps),
        _min_value(point.area_mm2),
        point.profile_id,
        point.candidate_id or "",
    )


def _point_identity(point: ParetoPoint) -> tuple[str, str]:
    return point.profile_id, point.candidate_id or ""


def _deduplicate_profile_points(
    points: Iterable[ParetoPoint],
) -> dict[str, ParetoPoint]:
    selected: dict[str, ParetoPoint] = {}
    for point in points:
        current = selected.get(point.profile_id)
        if current is None or _point_order(point) < _point_order(current):
            selected[point.profile_id] = point
    return selected


def _hardware_frontier(points: Iterable[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Keep every exact per-profile latency-energy tradeoff, per evidence tier."""

    grouped: dict[tuple[str, str | None], list[ParetoPoint]] = {}
    for point in points:
        if (
            point.candidate_id is None
            or point.tpot_ms is None
            or point.energy_per_token_j is None
        ):
            continue
        grouped.setdefault((point.profile_id, point.energy_tier), []).append(point)
    retained = []
    for values in grouped.values():
        identities = tuple(_point_identity(point) for point in values)
        if len(identities) != len(set(identities)):
            raise ValueError("hardware points contain duplicate candidate identities")
        for point in values:
            dominated = False
            for other in values:
                if _point_identity(other) == _point_identity(point):
                    continue
                weak = (
                    float(other.tpot_ms) <= float(point.tpot_ms)
                    and float(other.energy_per_token_j)
                    <= float(point.energy_per_token_j)
                )
                strict = (
                    float(other.tpot_ms) < float(point.tpot_ms)
                    or float(other.energy_per_token_j)
                    < float(point.energy_per_token_j)
                )
                exact_tie = (
                    float(other.tpot_ms) == float(point.tpot_ms)
                    and float(other.energy_per_token_j)
                    == float(point.energy_per_token_j)
                    and _point_identity(other) < _point_identity(point)
                )
                if weak and (strict or exact_tie):
                    dominated = True
                    break
            if not dominated:
                retained.append(point)
    return tuple(sorted(retained, key=_point_order))


def _families(point: ParetoPoint) -> set[str]:
    return {
        format_descriptor(token).family
        for token in (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        )
        if token != "BF16"
    }


def _best_matching(
    points: Sequence[ParetoPoint],
    predicate,
) -> ParetoPoint | None:
    matched = [point for point in points if predicate(point)]
    return min(matched, key=_point_order) if matched else None


@dataclass(frozen=True)
class PromotionResult:
    """Promoted profiles and the control category assigned to each."""

    points: tuple[ParetoPoint, ...]
    vector_controls: tuple[ParetoPoint, ...]
    controls: tuple[tuple[str, str], ...]
    missing_controls: tuple[str, ...]
    missing_vector_controls: tuple[str, ...]
    epsilon: EpsilonPolicy
    hardware_alternatives: tuple[ParetoPoint, ...] = ()

    @property
    def controls_complete(self) -> bool:
        return not self.missing_controls and not self.missing_vector_controls

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [point.to_dict() for point in self.points],
            "vector_controls": [
                point.to_dict() for point in self.vector_controls
            ],
            "controls": dict(self.controls),
            "missing_controls": list(self.missing_controls),
            "missing_vector_controls": list(self.missing_vector_controls),
            "controls_complete": self.controls_complete,
            "epsilon": self.epsilon.to_dict(),
            "hardware_alternatives": [
                point.to_dict() for point in self.hardware_alternatives
            ],
        }


REFINEMENT_SOURCE_ROLES = (
    "uniform_mxint8",
    "uniform_mxint4",
    "mxint_kv2",
    "accuracy_constrained_deployment",
)


@dataclass(frozen=True)
class RefinementSourceSelection:
    """Four deterministic, measured sources for precision refinement."""

    promotion: PromotionResult
    source_roles: tuple[tuple[str, str], ...]
    reference_mean_nll: float
    relative_perplexity_limit: float
    deployment_accuracy_gate_passed: bool
    deployment_fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if tuple(role for role, _ in self.source_roles) != REFINEMENT_SOURCE_ROLES:
            raise ValueError("refinement source roles differ from the fixed policy")
        profile_ids = tuple(profile_id for _, profile_id in self.source_roles)
        if len(profile_ids) != 4 or len(set(profile_ids)) != 4:
            raise ValueError("refinement selection requires four distinct sources")
        if tuple(point.profile_id for point in self.promotion.points) != profile_ids:
            raise ValueError("refinement promotion order differs from its source roles")
        if not self.promotion.controls_complete:
            raise ValueError("refinement source controls are incomplete")
        if not math.isfinite(self.reference_mean_nll) or self.reference_mean_nll < 0:
            raise ValueError("reference_mean_nll must be finite and non-negative")
        if (
            not math.isfinite(self.relative_perplexity_limit)
            or self.relative_perplexity_limit <= 1.0
        ):
            raise ValueError(
                "relative_perplexity_limit must be finite and greater than one"
            )
        if self.deployment_accuracy_gate_passed:
            if self.deployment_fallback_reason is not None:
                raise ValueError("a passed accuracy gate cannot carry a fallback reason")
        elif not self.deployment_fallback_reason:
            raise ValueError("a failed accuracy gate requires a fallback reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "four_measured_decode_sources/v1",
            "source_roles": dict(self.source_roles),
            "reference_mean_nll": self.reference_mean_nll,
            "relative_perplexity_limit": self.relative_perplexity_limit,
            "maximum_mean_nll": (
                self.reference_mean_nll
                + math.log(self.relative_perplexity_limit)
            ),
            "deployment_accuracy_gate_passed": (
                self.deployment_accuracy_gate_passed
            ),
            "deployment_fallback_reason": self.deployment_fallback_reason,
            "promotion": self.promotion.to_dict(),
        }


def _uniform_mxint(point: ParetoPoint, bits: int) -> bool:
    token = f"MXINT{bits}"
    return (
        point.profile.kind == PROFILE_KIND_QUANTIZED
        and (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        )
        == (token, token, token)
    )


def _deployment_order(point: ParetoPoint) -> tuple[Any, ...]:
    return (
        _energy_tier_rank(point.energy_tier),
        _min_value(point.energy_per_token_j),
        _min_value(point.tpot_ms),
        -_max_value(point.tps),
        _min_value(point.area_mm2),
        point.mean_nll,
        point.profile_id,
        point.candidate_id or "",
    )


def select_refinement_sources(
    points: Iterable[ParetoPoint],
    *,
    reference_mean_nll: float,
    relative_perplexity_limit: float = 1.01,
    epsilon: EpsilonPolicy = EpsilonPolicy(),
) -> RefinementSourceSelection:
    """Select four distinct, hardware-measured refinement sources."""

    if not math.isfinite(reference_mean_nll) or reference_mean_nll < 0:
        raise ValueError("reference_mean_nll must be finite and non-negative")
    if (
        not math.isfinite(relative_perplexity_limit)
        or relative_perplexity_limit <= 1.0
    ):
        raise ValueError(
            "relative_perplexity_limit must be finite and greater than one"
        )
    values = tuple(points)
    identities = tuple(_point_identity(point) for point in values)
    if len(identities) != len(set(identities)):
        raise ValueError("refinement points contain duplicate candidate identities")
    quantized = tuple(
        point
        for point in values
        if point.profile.kind == PROFILE_KIND_QUANTIZED
        and point.whole_model_rankable
        and all(
            value is not None
            for value in (
                point.tpot_ms,
                point.tps,
                point.energy_per_token_j,
                point.area_mm2,
            )
        )
    )
    uniform_i8 = _best_matching(
        quantized,
        lambda point: _uniform_mxint(point, 8),
    )
    uniform_i4 = _best_matching(
        quantized,
        lambda point: _uniform_mxint(point, 4),
    )
    kv2 = _best_matching(
        quantized,
        lambda point: (
            point.profile.kv_format == "MXINT2"
            and point.profile.weight_format in {"MXINT4", "MXINT8"}
            and point.profile.activation_format in {"MXINT4", "MXINT8"}
        ),
    )
    required = {
        "uniform_mxint8": uniform_i8,
        "uniform_mxint4": uniform_i4,
        "mxint_kv2": kv2,
    }
    missing = tuple(name for name, point in required.items() if point is None)
    if missing:
        raise ValueError(f"required refinement sources are missing: {missing}")
    selected = tuple(point for point in required.values() if point is not None)
    selected_ids = {point.profile_id for point in selected}
    remaining = tuple(
        point for point in quantized if point.profile_id not in selected_ids
    )
    if not remaining:
        raise ValueError("no distinct deployment source remains after controls")
    pareto = epsilon_pareto_fronts(remaining, epsilon)[0]
    maximum_nll = reference_mean_nll + math.log(relative_perplexity_limit)
    accurate = tuple(point for point in pareto if point.mean_nll <= maximum_nll)
    if accurate:
        deployment = min(accurate, key=_deployment_order)
        gate_passed = True
        fallback_reason = None
    else:
        deployment = min(pareto, key=_point_order)
        gate_passed = False
        fallback_reason = "no_distinct_pareto_source_met_relative_perplexity_limit"
    sources = selected + (deployment,)
    roles = tuple(
        (role, point.profile_id)
        for role, point in zip(REFINEMENT_SOURCE_ROLES, sources)
    )
    vector_controls: list[ParetoPoint] = []
    missing_vector: list[str] = []
    seen_triples: set[tuple[str, str, str]] = set()
    for point in sources:
        triple = (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        )
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        control = _best_matching(
            tuple(
                candidate
                for candidate in values
                if candidate.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
            ),
            lambda candidate, expected=triple: (
                candidate.profile.weight_format,
                candidate.profile.activation_format,
                candidate.profile.kv_format,
            )
            == expected,
        )
        if control is None:
            missing_vector.append("/".join(triple))
        else:
            vector_controls.append(control)
    promotion = PromotionResult(
        points=sources,
        vector_controls=tuple(vector_controls),
        controls=roles,
        missing_controls=(),
        missing_vector_controls=tuple(missing_vector),
        epsilon=epsilon,
        hardware_alternatives=_hardware_frontier(
            point for point in quantized if point.profile_id in selected_ids | {deployment.profile_id}
        ),
    )
    if not promotion.controls_complete:
        raise ValueError(
            "vector-BF16 controls are missing for selected refinement sources: "
            f"{promotion.missing_vector_controls}"
        )
    return RefinementSourceSelection(
        promotion=promotion,
        source_roles=roles,
        reference_mean_nll=reference_mean_nll,
        relative_perplexity_limit=relative_perplexity_limit,
        deployment_accuracy_gate_passed=gate_passed,
        deployment_fallback_reason=fallback_reason,
    )


def promote_epsilon_pareto(
    points: Iterable[ParetoPoint],
    *,
    limit: int = 24,
    epsilon: EpsilonPolicy = EpsilonPolicy(),
    preserve_controls: bool = True,
) -> PromotionResult:
    """Promote epsilon fronts while retaining the required attribution points."""

    if limit <= 0:
        raise ValueError("promotion limit must be positive")
    values = tuple(points)
    identities = tuple(_point_identity(point) for point in values)
    if len(identities) != len(set(identities)):
        raise ValueError("promotion points contain duplicate candidate identities")
    quantized = tuple(
        point for point in values if point.profile.kind == PROFILE_KIND_QUANTIZED
    )
    ranked_candidates = quantized or values
    ranked = tuple(
        point
        for front in epsilon_pareto_fronts(ranked_candidates, epsilon)
        for point in front
    )
    control_points: dict[str, ParetoPoint | None] = {}
    if preserve_controls:
        control_points = {
            "uniform_i8": _best_matching(
                quantized,
                lambda point: all(
                    token == "MXINT8"
                    for token in (
                        point.profile.weight_format,
                        point.profile.activation_format,
                        point.profile.kv_format,
                    )
                ),
            ),
            "uniform_i4": _best_matching(
                quantized,
                lambda point: all(
                    token == "MXINT4"
                    for token in (
                        point.profile.weight_format,
                        point.profile.activation_format,
                        point.profile.kv_format,
                    )
                ),
            ),
            "all_mxfp": _best_matching(
                quantized,
                lambda point: _families(point) == {"mxfp"},
            ),
            "mixed_family_oracle": _best_matching(
                quantized,
                lambda point: len(_families(point)) > 1,
            ),
        }
        for bits in (2, 4, 8):
            control_points[f"kv{bits}"] = _best_matching(
                quantized,
                lambda point, expected=bits: (
                    format_descriptor(point.profile.kv_format).element_bits == expected
                ),
            )

    missing = tuple(
        name for name, point in sorted(control_points.items()) if point is None
    )
    controls = tuple(
        (name, point.profile_id)
        for name, point in sorted(control_points.items())
        if point is not None
    )
    promoted: list[ParetoPoint] = []
    promoted_ids: set[str] = set()
    for _, profile_id in controls:
        point = min(
            (point for point in values if point.profile_id == profile_id),
            key=_point_order,
        )
        if profile_id not in promoted_ids:
            promoted.append(point)
            promoted_ids.add(profile_id)
    if len(promoted) > limit:
        raise ValueError("promotion limit is smaller than the unique control set")
    for point in ranked:
        if len(promoted) >= limit:
            break
        if point.profile_id not in promoted_ids:
            promoted.append(point)
            promoted_ids.add(point.profile_id)
    vector_controls: list[ParetoPoint] = []
    missing_vector: list[str] = []
    seen_triples: set[tuple[str, str, str]] = set()
    for point in promoted:
        if point.profile.kind != PROFILE_KIND_QUANTIZED:
            continue
        triple = (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        )
        if triple in seen_triples:
            continue
        seen_triples.add(triple)
        control = _best_matching(
            tuple(
                candidate
                for candidate in values
                if candidate.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
            ),
            lambda candidate, expected=triple: (
                candidate.profile.weight_format,
                candidate.profile.activation_format,
                candidate.profile.kv_format,
            )
            == expected,
        )
        if control is None:
            missing_vector.append("/".join(triple))
        else:
            vector_controls.append(control)
    return PromotionResult(
        points=tuple(promoted),
        vector_controls=tuple(vector_controls),
        controls=controls,
        missing_controls=missing,
        missing_vector_controls=tuple(missing_vector),
        epsilon=epsilon,
        hardware_alternatives=_hardware_frontier(
            point for point in quantized if point.profile_id in promoted_ids
        ),
    )


@dataclass(frozen=True)
class FidelityReport:
    """Numerical-screen to hardware-validation ranking and recovery gates."""

    spearman: float
    top_k_recall: float
    common_profiles: int
    top_k: int
    spearman_threshold: float
    recall_threshold: float
    numerical_screen_top_ids: tuple[str, ...]
    hardware_validation_top_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.spearman >= self.spearman_threshold
            and self.top_k_recall >= self.recall_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "spearman": self.spearman,
            "top_k_recall": self.top_k_recall,
            "common_profiles": self.common_profiles,
            "top_k": self.top_k,
            "spearman_threshold": self.spearman_threshold,
            "recall_threshold": self.recall_threshold,
            "numerical_screen_top_ids": list(self.numerical_screen_top_ids),
            "hardware_validation_top_ids": list(self.hardware_validation_top_ids),
        }


def evaluate_screening_fidelity(
    numerical_screen_points: Iterable[ParetoPoint],
    hardware_validation_points: Iterable[ParetoPoint],
    *,
    top_k: int = 24,
    epsilon: EpsilonPolicy = EpsilonPolicy(),
    spearman_threshold: float = 0.90,
    recall_threshold: float = 0.90,
) -> FidelityReport:
    """Compare common-profile NLL ranks and epsilon-Pareto top-set recall."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= spearman_threshold <= 1.0:
        raise ValueError("spearman_threshold must be in [0, 1]")
    if not 0.0 <= recall_threshold <= 1.0:
        raise ValueError("recall_threshold must be in [0, 1]")
    numerical_screen_all = _deduplicate_profile_points(numerical_screen_points)
    hardware_validation_all = _deduplicate_profile_points(hardware_validation_points)
    numerical_screen_quantized = {
        profile_id: point
        for profile_id, point in numerical_screen_all.items()
        if point.profile.kind == PROFILE_KIND_QUANTIZED
    }
    hardware_validation_quantized = {
        profile_id: point
        for profile_id, point in hardware_validation_all.items()
        if point.profile.kind == PROFILE_KIND_QUANTIZED
    }
    numerical_screen = numerical_screen_quantized or numerical_screen_all
    hardware_validation = hardware_validation_quantized or hardware_validation_all
    common = tuple(sorted(set(numerical_screen) & set(hardware_validation)))
    if len(common) < 2:
        raise ValueError("fidelity requires at least two common profiles")
    correlation = spearman_rank_correlation(
        [numerical_screen[profile_id].mean_nll for profile_id in common],
        [hardware_validation[profile_id].mean_nll for profile_id in common],
    )
    numerical_screen_ranked = [
        point
        for front in epsilon_pareto_fronts(numerical_screen.values(), epsilon)
        for point in front
    ]
    hardware_validation_ranked = [
        point
        for front in epsilon_pareto_fronts(hardware_validation.values(), epsilon)
        for point in front
    ]
    numerical_screen_top = tuple(point.profile_id for point in numerical_screen_ranked[:top_k])
    hardware_validation_top = tuple(point.profile_id for point in hardware_validation_ranked[:top_k])
    denominator = len(hardware_validation_top)
    recall = len(set(numerical_screen_top) & set(hardware_validation_top)) / denominator if denominator else 0.0
    return FidelityReport(
        spearman=correlation,
        top_k_recall=recall,
        common_profiles=len(common),
        top_k=top_k,
        spearman_threshold=spearman_threshold,
        recall_threshold=recall_threshold,
        numerical_screen_top_ids=numerical_screen_top,
        hardware_validation_top_ids=hardware_validation_top,
    )


@dataclass(frozen=True)
class DocumentNLL:
    document_id: str
    nll_sum: float
    token_count: int

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if not math.isfinite(self.nll_sum) or self.nll_sum < 0:
            raise ValueError("nll_sum must be finite and non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")


@dataclass(frozen=True)
class PairedDocumentScore:
    document_id: str
    reference_score: float
    candidate_score: float

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if not all(
            math.isfinite(value)
            for value in (self.reference_score, self.candidate_score)
        ):
            raise ValueError("paired scores must be finite")


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int
    cluster_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
            "seed": self.seed,
            "cluster_count": self.cluster_count,
        }


def _validate_bootstrap(
    count: int,
    samples: int,
    confidence: float,
) -> None:
    if count < 2:
        raise ValueError("at least two document clusters are required")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")


def clustered_nll_bootstrap(
    documents: Iterable[DocumentNLL],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap micro-average NLL by resampling whole documents."""

    values = tuple(sorted(documents, key=lambda value: value.document_id))
    if len({value.document_id for value in values}) != len(values):
        raise ValueError("document IDs must be unique")
    _validate_bootstrap(len(values), samples, confidence)
    estimate = sum(value.nll_sum for value in values) / sum(
        value.token_count for value in values
    )
    generator = random.Random(seed)
    replicates: list[float] = []
    for _ in range(samples):
        selected = [values[generator.randrange(len(values))] for _ in values]
        replicates.append(
            sum(value.nll_sum for value in selected)
            / sum(value.token_count for value in selected)
        )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=estimate,
        lower=percentile(replicates, alpha),
        upper=percentile(replicates, 1.0 - alpha),
        confidence=confidence,
        samples=samples,
        seed=seed,
        cluster_count=len(values),
    )


def paired_task_bootstrap(
    documents: Iterable[PairedDocumentScore],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap paired candidate-minus-reference score differences by document."""

    values = tuple(sorted(documents, key=lambda value: value.document_id))
    if len({value.document_id for value in values}) != len(values):
        raise ValueError("document IDs must be unique")
    _validate_bootstrap(len(values), samples, confidence)
    differences = [
        value.candidate_score - value.reference_score for value in values
    ]
    estimate = sum(differences) / len(differences)
    generator = random.Random(seed)
    replicates = [
        sum(differences[generator.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=estimate,
        lower=percentile(replicates, alpha),
        upper=percentile(replicates, 1.0 - alpha),
        confidence=confidence,
        samples=samples,
        seed=seed,
        cluster_count=len(values),
    )


@dataclass(frozen=True)
class PublicationCandidate:
    """Full-evaluation row used by the final deployment gate."""

    evaluation_class: str
    profile_id: str
    candidate_id: str
    profile_kind: str
    perplexity: float
    tpot_ms: float | None
    energy_per_token_j: float | None
    validity: StackValidity
    hardware_candidate: bool = True
    power_calibration_id: str | None = None
    cost_scope: str | None = None
    system_calibration_id: str | None = None
    head_service_calibration_id: str | None = None
    whole_model_rankable: bool = False
    timing_mode: str = "rtl_serialized"
    timing_calibrated: bool = False
    timing_evidence_id: str | None = None
    task_delta_lower_ci: tuple[tuple[str, float], ...] = ()
    ruler_scores: tuple[tuple[int, float, float], ...] = ()
    energy_tier: str | None = None

    def __post_init__(self) -> None:
        if not self.evaluation_class or not self.profile_id or not self.candidate_id:
            raise ValueError("publication candidate identities must be non-empty")
        _finite_optional("perplexity", self.perplexity, positive=True)
        _finite_optional("tpot_ms", self.tpot_ms, positive=True)
        _finite_optional(
            "energy_per_token_j",
            self.energy_per_token_j,
            positive=True,
        )
        if (self.energy_per_token_j is not None) != bool(
            self.power_calibration_id
        ):
            raise ValueError(
                "energy requires one power calibration identity"
            )
        tier = self.energy_tier
        if self.energy_per_token_j is not None and tier is None:
            tier = "dc_calibrated"
            object.__setattr__(self, "energy_tier", tier)
        if tier is not None and tier not in ENERGY_TIERS:
            raise ValueError("energy_tier is unsupported")
        if self.energy_per_token_j is None and tier is not None:
            raise ValueError("energy_tier requires energy")
        has_serving_cost = (
            self.tpot_ms is not None
            or self.energy_per_token_j is not None
        )
        if has_serving_cost:
            if (
                not self.whole_model_rankable
                or self.cost_scope != "whole_model"
                or not self.system_calibration_id
                or not self.head_service_calibration_id
            ):
                raise ValueError(
                    "deployment costs require a calibrated whole-model boundary"
                )
        elif any(
            (
                self.whole_model_rankable,
                self.cost_scope is not None,
                self.system_calibration_id is not None,
                self.head_service_calibration_id is not None,
            )
        ):
            raise ValueError(
                "whole-model identities require deployment costs"
            )
        is_reference_class = self.evaluation_class == "bf16_reference"
        is_reference_kind = self.profile_kind == PROFILE_KIND_BF16_REFERENCE
        if is_reference_class != is_reference_kind:
            raise ValueError("BF16 reference class and profile kind must match")
        if is_reference_class:
            if self.hardware_candidate:
                raise ValueError("the BF16 reference is not a PLENA candidate")
            if (
                self.tpot_ms is not None
                or self.energy_per_token_j is not None
                or self.power_calibration_id is not None
                or self.cost_scope is not None
                or self.system_calibration_id is not None
                or self.head_service_calibration_id is not None
                or self.whole_model_rankable
                or self.energy_tier is not None
                or self.timing_calibrated
                or self.timing_evidence_id is not None
            ):
                raise ValueError(
                    "the BF16 accuracy reference cannot carry PLENA cost evidence"
                )
        if not self.timing_mode:
            raise ValueError("timing_mode must be non-empty")
        if self.timing_calibrated != bool(self.timing_evidence_id):
            raise ValueError(
                "timing calibration requires a timing evidence identity"
            )
        tasks = tuple(sorted((str(name), float(value)) for name, value in self.task_delta_lower_ci))
        if len({name for name, _ in tasks}) != len(tasks):
            raise ValueError("task confidence bounds must have unique names")
        if not all(math.isfinite(value) for _, value in tasks):
            raise ValueError("task confidence bounds must be finite")
        ruler = tuple(
            sorted(
                (int(length), float(candidate), float(reference))
                for length, candidate, reference in self.ruler_scores
            )
        )
        if len({length for length, _, _ in ruler}) != len(ruler):
            raise ValueError("RULER lengths must be unique")
        if any(
            length <= 0
            or not math.isfinite(candidate)
            or not math.isfinite(reference)
            or reference <= 0
            for length, candidate, reference in ruler
        ):
            raise ValueError("RULER scores and lengths must be valid")
        object.__setattr__(self, "task_delta_lower_ci", tasks)
        object.__setattr__(self, "ruler_scores", ruler)

    @property
    def fully_hardware_valid(self) -> bool:
        stack_fields = (
            "software_valid",
            "compiler_valid",
            "emulator_valid",
            "rtl_valid",
        )
        return (
            self.timing_calibrated
            and self.power_calibration_id is not None
            and self.energy_tier in ENERGY_TIERS
            and self.whole_model_rankable
            and self.cost_scope == "whole_model"
            and self.system_calibration_id is not None
            and self.head_service_calibration_id is not None
            and all(getattr(self.validity, field) is True for field in stack_fields)
            and (
                self.energy_tier != "dc_calibrated"
                or self.validity.dc_calibrated is True
            )
        )

    @property
    def tokens_per_joule(self) -> float | None:
        if self.energy_per_token_j is None:
            return None
        return 1.0 / self.energy_per_token_j

    @property
    def edp_j_s(self) -> float | None:
        if self.energy_per_token_j is None or self.tpot_ms is None:
            return None
        return self.energy_per_token_j * self.tpot_ms / 1000.0

    @property
    def reference_scope(self) -> str | None:
        if self.evaluation_class == "bf16_reference":
            return BF16_REFERENCE_SCOPE
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_class": self.evaluation_class,
            "profile_id": self.profile_id,
            "candidate_id": self.candidate_id,
            "profile_kind": self.profile_kind,
            "perplexity": self.perplexity,
            "tpot_ms": self.tpot_ms,
            "energy_per_token_j": self.energy_per_token_j,
            "energy_tier": self.energy_tier,
            "tokens_per_joule": self.tokens_per_joule,
            "edp_j_s": self.edp_j_s,
            "validity": self.validity.to_dict(),
            "hardware_candidate": self.hardware_candidate,
            "reference_scope": self.reference_scope,
            "power_calibration_id": self.power_calibration_id,
            "cost_scope": self.cost_scope,
            "system_calibration_id": self.system_calibration_id,
            "head_service_calibration_id": (
                self.head_service_calibration_id
            ),
            "whole_model_rankable": self.whole_model_rankable,
            "timing_mode": self.timing_mode,
            "timing_calibrated": self.timing_calibrated,
            "timing_evidence_id": self.timing_evidence_id,
            "task_delta_lower_ci": dict(self.task_delta_lower_ci),
            "ruler_scores": [
                {
                    "length": length,
                    "candidate_score": candidate,
                    "reference_score": reference,
                    "retention": candidate / reference,
                }
                for length, candidate, reference in self.ruler_scores
            ],
        }


@dataclass(frozen=True)
class FinalSelectionDecision:
    """Selection result with explicit global and per-candidate failures."""

    selected: PublicationCandidate | None
    calibration_id: str | None
    timing_evidence_id: str | None
    timing_mode: str | None
    head_service_calibration_id: str | None
    system_calibration_id: str | None
    global_failures: tuple[str, ...]
    candidate_failures: tuple[tuple[str, tuple[str, ...]], ...]
    relative_ppl_limit: float
    task_margin_points: float
    ruler_retention: float
    energy_tier: str | None = None
    bf16_reference_scope: str = BF16_REFERENCE_SCOPE
    schema_version: str = SELECTION_REPORT_SCHEMA

    @property
    def selected_profile_id(self) -> str | None:
        return self.selected.profile_id if self.selected else None

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "selected": self.selected.to_dict() if self.selected else None,
            "calibration_id": self.calibration_id,
            "timing_evidence_id": self.timing_evidence_id,
            "timing_mode": self.timing_mode,
            "head_service_calibration_id": (
                self.head_service_calibration_id
            ),
            "system_calibration_id": self.system_calibration_id,
            "energy_tier": self.energy_tier,
            "global_failures": list(self.global_failures),
            "candidate_failures": {
                key: list(value) for key, value in self.candidate_failures
            },
            "relative_ppl_limit": self.relative_ppl_limit,
            "task_margin_points": self.task_margin_points,
            "ruler_retention": self.ruler_retention,
            "bf16_reference_scope": self.bf16_reference_scope,
            "cost_scope": "whole_model",
            "objective": ["energy_per_token_j", "tpot_ms"],
        }
        return {**body, "decision_hash": _hash(body)}


def select_final_deployment(
    candidates: Iterable[PublicationCandidate],
    *,
    calibration: CalibrationEvidence | None,
    timing_evidence: TimingCalibrationEvidence | None,
    head_service_evidence: HeadServiceEvidence | None = None,
    relative_ppl_limit: float = 0.01,
    task_margin_points: float = 2.0,
    ruler_retention: float = 0.98,
    required_tasks: Sequence[str] = DEFAULT_REQUIRED_TASKS,
    required_ruler_lengths: Sequence[int] = DEFAULT_RULER_LENGTHS,
) -> FinalSelectionDecision:
    """Select the strongest available energy tier, then energy and TPOT."""

    if relative_ppl_limit < 0 or not math.isfinite(relative_ppl_limit):
        raise ValueError("relative_ppl_limit must be finite and non-negative")
    if task_margin_points < 0 or not math.isfinite(task_margin_points):
        raise ValueError("task_margin_points must be finite and non-negative")
    if not 0.0 < ruler_retention <= 1.0:
        raise ValueError("ruler_retention must be in (0, 1]")
    values = tuple(candidates)
    identities = [(value.profile_id, value.candidate_id) for value in values]
    if len(identities) != len(set(identities)):
        raise ValueError("publication configurations must be distinct")
    global_failures: list[str] = []
    calibration_passed = False
    calibration_id = None
    if calibration is not None:
        candidate_calibration_id = calibration.calibration_id
        try:
            calibration_id = require_content_addressed_id(
                "power calibration",
                candidate_calibration_id,
            )
        except ValueError:
            calibration_id = None
        calibration_passed = bool(calibration.passed and calibration_id)
    if timing_evidence is None:
        global_failures.append("timing_evidence_missing")
        timing_evidence_id = None
        timing_mode = None
    else:
        timing_evidence_id = timing_evidence.evidence_id
        timing_mode = timing_evidence.mode
        try:
            timing_evidence_id = require_content_addressed_id(
                "timing evidence",
                timing_evidence_id,
                prefix="timing-",
            )
        except ValueError:
            global_failures.append("timing_evidence_identity")
            timing_evidence_id = None
        if not timing_mode:
            global_failures.append("timing_evidence_mode")
        if not timing_evidence.passed:
            global_failures.append("timing_evidence_gate")
    if head_service_evidence is None:
        global_failures.append("head_service_evidence_missing")
        head_service_calibration_id = None
        head_service_provenance_id = None
    else:
        head_service_calibration_id = (
            head_service_evidence.calibration_id
        )
        try:
            head_service_calibration_id = require_content_addressed_id(
                "head-service calibration",
                head_service_calibration_id,
                prefix="bf16-head-service-",
            )
        except ValueError:
            global_failures.append("head_service_evidence_identity")
            head_service_calibration_id = None
        head_service_provenance_id = (
            head_service_evidence.provenance_id
        )
        try:
            head_service_provenance_id = require_content_addressed_id(
                "head-service provenance",
                head_service_provenance_id,
                prefix="bf16-head-provenance-",
            )
        except ValueError:
            global_failures.append("head_service_provenance_identity")
            head_service_provenance_id = None
        if head_service_evidence.service_mode != HEAD_SERVICE_MODE:
            global_failures.append("head_service_mode")
        if not head_service_evidence.passed:
            global_failures.append("head_service_evidence_gate")
    if (
        not head_service_calibration_id
        or not head_service_provenance_id
        or (
            head_service_evidence is not None
            and head_service_evidence.service_mode != HEAD_SERVICE_MODE
        )
    ):
        system_calibration_id = None
    else:
        system_calibration_id = None
    def bound_system_identity(value: PublicationCandidate) -> bool:
        if (
            not value.power_calibration_id
            or not head_service_calibration_id
            or not head_service_provenance_id
        ):
            return False
        try:
            power_id = require_content_addressed_id(
                "candidate energy",
                value.power_calibration_id,
            )
            expected = composite_system_calibration_id(
                power_id,
                head_service_calibration_id,
                head_service_provenance_id,
            )
        except ValueError:
            return False
        if value.system_calibration_id != expected:
            return False
        if value.energy_tier == "dc_calibrated":
            return calibration_passed and power_id == calibration_id
        return value.energy_tier == "analytic_anchored"

    by_class = {value.evaluation_class for value in values}
    reference_rows = [
        value for value in values if value.evaluation_class == "bf16_reference"
    ]
    if len(reference_rows) != 1:
        if not reference_rows:
            global_failures.append("missing_bf16_reference")
        else:
            global_failures.append("count_bf16_reference")
    else:
        reference = reference_rows[0]
        if (
            reference.hardware_candidate
            or reference.validity.software_valid is not True
            or reference.reference_scope != BF16_REFERENCE_SCOPE
        ):
            global_failures.append("invalid_bf16_accuracy_reference")
    for required_class in (
        "uniform_i8",
        "uniform_i4",
        "pareto_candidate",
    ):
        if required_class not in by_class:
            global_failures.append(f"missing_{required_class}")
        elif sum(
            value.evaluation_class == required_class for value in values
        ) != 1:
            global_failures.append(f"count_{required_class}")
        elif not any(
            value.evaluation_class == required_class
            and value.hardware_candidate
            and value.fully_hardware_valid
            and bound_system_identity(value)
            and value.head_service_calibration_id
            == head_service_calibration_id
            and value.timing_evidence_id == timing_evidence_id
            and value.timing_mode == timing_mode
            for value in values
        ):
            global_failures.append(f"invalid_{required_class}")
    references = [
        value
        for value in values
        if value.evaluation_class == "bf16_reference"
        and value.profile_kind == PROFILE_KIND_BF16_REFERENCE
    ]
    if len(references) != 1:
        global_failures.append("bf16_reference_count")
        reference_ppl = None
    else:
        reference_ppl = references[0].perplexity

    required_task_set = frozenset(str(value) for value in required_tasks)
    required_length_set = frozenset(int(value) for value in required_ruler_lengths)
    if not required_task_set:
        raise ValueError("at least one required task is needed")
    if not required_length_set or any(length <= 0 for length in required_length_set):
        raise ValueError("required RULER lengths must be positive")
    failures_by_candidate: list[tuple[str, tuple[str, ...]]] = []
    eligible: list[PublicationCandidate] = []
    for value in values:
        if value in references:
            continue
        failures: list[str] = []
        if not value.hardware_candidate:
            failures.append("static_hardware_legality")
        if not value.fully_hardware_valid:
            failures.append("cross_stack_validity")
        if value.energy_per_token_j is None or value.tpot_ms is None:
            failures.append("calibrated_cost_missing")
        if value.energy_tier not in ENERGY_TIERS:
            failures.append("energy_tier")
        if not bound_system_identity(value):
            if value.energy_tier == "dc_calibrated":
                if not calibration_passed:
                    failures.append("power_calibration_gate")
                elif value.power_calibration_id != calibration_id:
                    failures.append("power_calibration_identity")
            else:
                failures.append("energy_identity")
        if (
            value.head_service_calibration_id
            != head_service_calibration_id
        ):
            failures.append("head_service_calibration_identity")
        if not bound_system_identity(value):
            failures.append("system_calibration_identity")
        if (
            value.timing_evidence_id != timing_evidence_id
            or value.timing_mode != timing_mode
        ):
            failures.append("timing_evidence_identity")
        if reference_ppl is None:
            failures.append("bf16_perplexity_missing")
        elif value.perplexity / reference_ppl - 1.0 > relative_ppl_limit:
            failures.append("relative_perplexity")
        task_bounds = dict(value.task_delta_lower_ci)
        if not required_task_set.issubset(task_bounds):
            failures.append("task_confidence_missing")
        elif any(
            task_bounds[name] < -task_margin_points for name in required_task_set
        ):
            failures.append("task_confidence")
        ruler = {
            length: candidate / reference
            for length, candidate, reference in value.ruler_scores
        }
        if not required_length_set.issubset(ruler):
            failures.append("ruler_lengths_missing")
        elif any(ruler[length] < ruler_retention for length in required_length_set):
            failures.append("ruler_retention")
        key = f"{value.profile_id}/{value.candidate_id}"
        failures_by_candidate.append((key, tuple(failures)))
        if not failures:
            eligible.append(value)
    selected = None
    if not global_failures and eligible:
        selected = min(
            eligible,
            key=lambda value: (
                _energy_tier_rank(value.energy_tier),
                float(value.energy_per_token_j),
                float(value.tpot_ms),
                value.profile_id,
                value.candidate_id,
            ),
        )
    selected_calibration_id = (
        selected.power_calibration_id if selected is not None else calibration_id
    )
    selected_system_id = (
        selected.system_calibration_id if selected is not None else None
    )
    return FinalSelectionDecision(
        selected=selected,
        calibration_id=selected_calibration_id,
        timing_evidence_id=timing_evidence_id,
        timing_mode=timing_mode,
        head_service_calibration_id=head_service_calibration_id,
        system_calibration_id=selected_system_id,
        global_failures=tuple(global_failures),
        candidate_failures=tuple(failures_by_candidate),
        relative_ppl_limit=relative_ppl_limit,
        task_margin_points=task_margin_points,
        ruler_retention=ruler_retention,
        energy_tier=(selected.energy_tier if selected is not None else None),
    )


def write_selection_report(
    path: str | os.PathLike[str],
    decision: FinalSelectionDecision,
    *,
    provenance_hashes: Mapping[str, str],
) -> Path:
    """Atomically create an immutable selection report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": SELECTION_REPORT_SCHEMA,
        "provenance_hashes": dict(sorted(provenance_hashes.items())),
        "decision": decision.to_dict(),
    }
    payload = _canonical_bytes(
        {**body, "report_hash": _hash(body)},
        newline=True,
    )
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace a different selection report: {destination}"
            )
        return destination
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        os.link(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


__all__ = [
    "BF16_REFERENCE_SCOPE",
    "BootstrapInterval",
    "DEFAULT_REQUIRED_TASKS",
    "DEFAULT_RULER_LENGTHS",
    "DocumentNLL",
    "EpsilonPolicy",
    "FidelityReport",
    "FinalSelectionDecision",
    "HeadServiceEvidence",
    "PairedDocumentScore",
    "ParetoPoint",
    "PromotionResult",
    "PublicationCandidate",
    "REFINEMENT_SOURCE_ROLES",
    "RefinementSourceSelection",
    "clustered_nll_bootstrap",
    "epsilon_dominates",
    "epsilon_pareto_fronts",
    "evaluate_screening_fidelity",
    "paired_task_bootstrap",
    "promote_epsilon_pareto",
    "select_refinement_sources",
    "select_final_deployment",
    "write_selection_report",
]
