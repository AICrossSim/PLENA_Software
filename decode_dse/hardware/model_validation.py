"""Validation quality of the pricing models the design space is priced by.

Admission into the reported results rests on the *pricing model* being
validated and identified, not on every priced point having been individually
compiled and emulated (see ``evaluate_publication_admission`` in
``design_space.py`` and section 10 of ``docs/evidence_tiers.md``).  That
framing is only honest if the model's validation quality travels with the
claim, so this module reads the validation artifacts themselves and returns
their real figures.

Nothing here is hard-coded.  Every number is read from the artifact that
produced it:

``decode_timing_evidence.json``
    analytical-versus-emulator cycle agreement for the timing model, written by
    the timing-evidence stage and named on every priced row by
    ``metrics.timing_evidence_id``.
``analytic_models/disagg_serve/calibration_dma_requests.validation.json``
    the descriptor-aware holdout of the request-latency (memory bandwidth)
    calibration, including its per-plane error bands.
``analytic_models/area/calibration/matrix_structural_coefficients.json``
    the fitted structural area model, its holdout report, and - through the
    calibration CSVs - the geometry grid it was fitted on.
``analytic_models/disagg_serve/decode_power.py``
    the analytic energy tier and the provenance of each energy component.

A figure this module cannot read is reported as missing rather than defaulted:
an absent artifact must never look like a passing one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from decode_dse.hardware.power_bridge import (
    analytic_power_provenance,
    resolve_simulator_root,
)

#: Schema the timing-evidence artifact must declare.
TIMING_EVIDENCE_SCHEMA = "plena-decode-timing-evidence"

#: Schema the request-latency (bandwidth) holdout artifact must declare.
BANDWIDTH_VALIDATION_SCHEMA = "plena-request-latency-validation"

#: Simulator-relative paths of the calibration artifacts read below.
BANDWIDTH_VALIDATION_PATH = Path(
    "analytic_models/disagg_serve/calibration_dma_requests.validation.json"
)
BANDWIDTH_PROVENANCE_PATH = Path(
    "analytic_models/disagg_serve/CALIBRATION_PROVENANCE.md"
)
AREA_COEFFICIENT_PATH = Path(
    "analytic_models/area/calibration/matrix_structural_coefficients.json"
)
AREA_CALIBRATION_SOURCES = (
    Path("analytic_models/area/calibration/matrix_machine_mxint.csv"),
    Path("analytic_models/area/calibration/matrix_machine_mxfp.csv"),
    Path("analytic_models/area/calibration/full_chip_anchors.csv"),
)

#: An independent gate-level campaign that no area coefficient was fitted to.
#: It is read as evidence *about* the census, never as an input to it, so it is
#: deliberately absent from ``AREA_CALIBRATION_SOURCES`` above.
AREA_GATE_LEVEL_VALIDATION_PATH = Path(
    "analytic_models/area/calibration/matrix_gate_level_validation.json"
)
AREA_GATE_LEVEL_SCHEMA = "plena-matrix-gate-level-validation"

#: Every calibration row underneath these models is simulator-measured, not
#: measured silicon.  The label travels with the figures so a reader cannot
#: mistake a model holdout for a silicon error bar.
MODEL_ERROR_SCOPE = "simulator_calibrated_model_error_not_measured_silicon"

#: Area-domain verdicts.  The searched MLEN/BLEN grid runs well past the
#: synthesised grid, so most of the design space is priced by extrapolation and
#: must say so.
AREA_DOMAIN_INSIDE = "inside_fitted_domain"
AREA_DOMAIN_OUTSIDE = "outside_fitted_domain"
AREA_DOMAIN_UNKNOWN = "geometry_not_declared"


class ModelValidationUnavailable(RuntimeError):
    """A validation artifact could not be read, so its figures are unknown."""


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelValidationUnavailable(
            f"validation artifact {path} is missing"
        ) from error
    except json.JSONDecodeError as error:
        raise ModelValidationUnavailable(
            f"validation artifact {path} is not valid JSON"
        ) from error
    if not isinstance(document, Mapping):
        raise ModelValidationUnavailable(
            f"validation artifact {path} is not a JSON object"
        )
    return document


def _float(value: Any, *, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelValidationUnavailable(f"{path} has no numeric {name}")
    return float(value)


# --------------------------------------------------------------------------
# Timing: analytical model versus the emulator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingModelValidation:
    """Analytical-versus-emulator agreement for the timing model."""

    evidence_id: str
    evidence_tier: str
    mode: str
    step_composition: str
    analytical_mape: float
    analytical_mape_limit: float
    anchor_max_error: float
    anchor_max_error_limit: float
    anchor_count: int
    layer_anchor_count: int
    passed: bool
    execution_identity_matched: bool
    trace_identities_matched: bool
    provenance_hashes: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "decode_step_timing",
            "reference": "compiler_emitted_program_under_the_emulator",
            "evidence_id": self.evidence_id,
            "evidence_tier": self.evidence_tier,
            "mode": self.mode,
            "step_composition": self.step_composition,
            "analytical_vs_emulator_mape": self.analytical_mape,
            "analytical_vs_emulator_mape_limit": self.analytical_mape_limit,
            "anchor_max_error": self.anchor_max_error,
            "anchor_max_error_limit": self.anchor_max_error_limit,
            "anchor_count": self.anchor_count,
            "layer_anchor_count": self.layer_anchor_count,
            "passed": self.passed,
            "execution_identity_matched": self.execution_identity_matched,
            "trace_identities_matched": self.trace_identities_matched,
            "provenance_hashes": dict(self.provenance_hashes),
            "error_scope": MODEL_ERROR_SCOPE,
        }


def timing_model_validation(path: str | Path) -> TimingModelValidation:
    """Read the timing-evidence artifact the priced rows were timed against."""

    source = Path(path)
    document = _read_json(source)
    if document.get("schema") != TIMING_EVIDENCE_SCHEMA:
        raise ModelValidationUnavailable(
            f"{source} does not declare {TIMING_EVIDENCE_SCHEMA}"
        )
    evidence_id = document.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ModelValidationUnavailable(f"{source} has no evidence identity")
    anchors = document.get("anchors")
    provenance = document.get("provenance_hashes")
    return TimingModelValidation(
        evidence_id=evidence_id,
        evidence_tier=str(document.get("evidence_tier", "")),
        mode=str(document.get("mode", "")),
        step_composition=str(document.get("step_composition", "")),
        analytical_mape=_float(
            document.get("analytical_mape"),
            name="analytical_mape",
            path=source,
        ),
        analytical_mape_limit=_float(
            document.get("analytical_mape_limit"),
            name="analytical_mape_limit",
            path=source,
        ),
        anchor_max_error=_float(
            document.get("anchor_max_error"),
            name="anchor_max_error",
            path=source,
        ),
        anchor_max_error_limit=_float(
            document.get("anchor_max_error_limit"),
            name="anchor_max_error_limit",
            path=source,
        ),
        anchor_count=len(anchors) if isinstance(anchors, list) else 0,
        layer_anchor_count=int(document.get("layer_anchor_count", 0)),
        passed=document.get("passed") is True,
        execution_identity_matched=(
            document.get("execution_identity_matched") is True
        ),
        trace_identities_matched=(
            document.get("trace_identities_matched") is True
        ),
        provenance_hashes=(
            {str(key): str(value) for key, value in provenance.items()}
            if isinstance(provenance, Mapping)
            else {}
        ),
    )


# --------------------------------------------------------------------------
# Memory bandwidth: descriptor-aware holdout of the request-latency fit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BandwidthModelValidation:
    """Holdout quality of the request-latency (memory bandwidth) calibration."""

    calibration_id: str
    split_unit: str
    training_count: int
    holdout_count: int
    holdout_fraction: float
    median_absolute_error_percent: float
    p95_absolute_error_percent: float
    p99_absolute_error_percent: float
    worst_absolute_error_percent: float
    worst_group_by_p95: tuple[str, float, float, int]
    worst_group_by_median: tuple[str, float, float, int]
    group_count: int

    @staticmethod
    def _group(record: tuple[str, float, float, int]) -> dict[str, Any]:
        name, median, p95, count = record
        return {
            "group": name,
            "median_absolute_error_percent": median,
            "p95_absolute_error_percent": p95,
            "holdout_count": count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "request_latency_effective_bandwidth",
            "reference": "ramulator2_simulated_request_traces",
            "calibration_id": self.calibration_id,
            "split_unit": self.split_unit,
            "training_count": self.training_count,
            "holdout_count": self.holdout_count,
            "holdout_fraction": self.holdout_fraction,
            "holdout_median_absolute_error_percent": (
                self.median_absolute_error_percent
            ),
            "holdout_p95_absolute_error_percent": (
                self.p95_absolute_error_percent
            ),
            "holdout_p99_absolute_error_percent": (
                self.p99_absolute_error_percent
            ),
            "holdout_worst_absolute_error_percent": (
                self.worst_absolute_error_percent
            ),
            "worst_retained_group_by_p95": self._group(self.worst_group_by_p95),
            "worst_retained_group_by_median": self._group(
                self.worst_group_by_median
            ),
            "retained_group_count": self.group_count,
            "error_scope": MODEL_ERROR_SCOPE,
        }


def bandwidth_model_validation(
    simulator_root: Path | None = None,
) -> BandwidthModelValidation:
    """Read the descriptor-aware holdout of the request-latency calibration."""

    root = Path(simulator_root) if simulator_root else resolve_simulator_root().root
    source = root / BANDWIDTH_VALIDATION_PATH
    document = _read_json(source)
    if document.get("schema_version") != BANDWIDTH_VALIDATION_SCHEMA:
        raise ModelValidationUnavailable(
            f"{source} does not declare {BANDWIDTH_VALIDATION_SCHEMA}"
        )
    calibration_id = document.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ModelValidationUnavailable(f"{source} has no calibration identity")
    per_group = document.get("per_group")
    if not isinstance(per_group, Mapping) or not per_group:
        raise ModelValidationUnavailable(f"{source} reports no retained groups")
    groups: list[tuple[str, float, float, int]] = []
    for name, record in per_group.items():
        if not isinstance(record, Mapping):
            continue
        groups.append(
            (
                str(name),
                _float(
                    record.get("median_absolute_error_percent"),
                    name=f"{name} median",
                    path=source,
                ),
                _float(
                    record.get("p95_absolute_error_percent"),
                    name=f"{name} p95",
                    path=source,
                ),
                int(record.get("count", 0)),
            )
        )
    if not groups:
        raise ModelValidationUnavailable(f"{source} reports no usable groups")
    return BandwidthModelValidation(
        calibration_id=calibration_id,
        split_unit=str(document.get("split_unit", "")),
        training_count=int(document.get("training_count", 0)),
        holdout_count=int(document.get("holdout_count", 0)),
        holdout_fraction=_float(
            document.get("holdout_fraction"),
            name="holdout_fraction",
            path=source,
        ),
        median_absolute_error_percent=_float(
            document.get("median_absolute_error_percent"),
            name="median_absolute_error_percent",
            path=source,
        ),
        p95_absolute_error_percent=_float(
            document.get("p95_absolute_error_percent"),
            name="p95_absolute_error_percent",
            path=source,
        ),
        p99_absolute_error_percent=_float(
            document.get("p99_absolute_error_percent"),
            name="p99_absolute_error_percent",
            path=source,
        ),
        worst_absolute_error_percent=_float(
            document.get("worst_absolute_error_percent"),
            name="worst_absolute_error_percent",
            path=source,
        ),
        # Sorted on the error first and the group name second so the reported
        # worst group is deterministic when two planes tie exactly.
        worst_group_by_p95=max(groups, key=lambda item: (item[2], item[0])),
        worst_group_by_median=max(groups, key=lambda item: (item[1], item[0])),
        group_count=len(groups),
    )


# --------------------------------------------------------------------------
# Area: structural census holdout and the grid it was fitted on
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaModelValidation:
    """Holdout quality and fitted domain of the structural area model."""

    model_version: str
    per_family: Mapping[str, Mapping[str, Any]]
    full_chip: Mapping[str, Any]
    fitted_mlen: tuple[int, ...]
    fitted_blen: tuple[int, ...]
    #: The independent gate-level cross-validation, or a reason it is missing.
    gate_level: Mapping[str, Any] = field(default_factory=dict)

    def domain_status(
        self,
        mlen: int | None,
        blen: int | None,
    ) -> str:
        """Report whether one geometry sits inside the synthesised grid."""

        if mlen is None or blen is None:
            return AREA_DOMAIN_UNKNOWN
        inside = int(mlen) in self.fitted_mlen and int(blen) in self.fitted_blen
        return AREA_DOMAIN_INSIDE if inside else AREA_DOMAIN_OUTSIDE

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "precision_aware_structural_area_census",
            "reference": "synopsys_design_compiler_7nm_areas",
            "model_version": self.model_version,
            "per_family_holdout": {
                family: dict(report) for family, report in self.per_family.items()
            },
            "full_chip_holdout": dict(self.full_chip),
            "fitted_domain": {
                "MLEN": list(self.fitted_mlen),
                "BLEN": list(self.fitted_blen),
            },
            "domain_note": (
                "the searched MLEN/BLEN grid runs past the synthesised grid, so "
                "an outside-domain point is priced by structural extrapolation "
                "and the in-domain holdout is not an error bar on it"
            ),
            "independent_gate_level_cross_validation": dict(self.gate_level),
            "error_scope": MODEL_ERROR_SCOPE,
        }


def _fitted_area_domain(root: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read the synthesised MLEN/BLEN grid from the calibration sources."""

    mlen: set[int] = set()
    blen: set[int] = set()
    for relative in AREA_CALIBRATION_SOURCES:
        source = root / relative
        if not source.is_file():
            continue
        with source.open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                for key, sink in (("MLEN", mlen), ("BLEN", blen)):
                    value = record.get(key)
                    if value in (None, ""):
                        continue
                    try:
                        sink.add(int(value))
                    except ValueError:
                        continue
    if not mlen or not blen:
        raise ModelValidationUnavailable(
            "the area calibration sources declare no MLEN/BLEN grid"
        )
    return tuple(sorted(mlen)), tuple(sorted(blen))


def _gate_level_area_validation(root: Path) -> dict[str, Any]:
    """Summarise the independent gate-level campaign, or say why it is missing.

    Only the headline verdict is lifted, and it is lifted together with the
    campaign's scope. These are one-block figures in um^2 at MLEN 16-64 and
    25 C, while the priced geometries are full-chip mm^2 at MLEN 128-1024; a
    summary that carried the agreement without the boundary would invite
    exactly the comparison the boundary exists to prevent.
    """

    source = root / AREA_GATE_LEVEL_VALIDATION_PATH
    try:
        document = _read_json(source)
    except (ModelValidationUnavailable, FileNotFoundError, ValueError) as error:
        return {"unavailable": str(error)}
    if document.get("schema") != AREA_GATE_LEVEL_SCHEMA:
        return {
            "unavailable": f"{source} does not declare {AREA_GATE_LEVEL_SCHEMA}",
        }
    census = document.get("census_cross_validation")
    if not isinstance(census, Mapping):
        return {"unavailable": f"{source} carries no cross-validation report"}
    residual = census.get("shape_and_precision_error_after_offset_pct", {})
    return {
        "artifact": str(AREA_GATE_LEVEL_VALIDATION_PATH),
        "n_points": document.get("n_points"),
        "independent_of_the_fit": document.get("independent_of_the_fit"),
        "coefficients_changed": document.get("coefficients_changed"),
        "uniform_offset_census_over_campaign": census.get(
            "uniform_offset_census_over_campaign"
        ),
        "shape_and_precision_error_after_offset_pct": dict(residual)
        if isinstance(residual, Mapping)
        else {},
        "known_model_limit": census.get("known_model_limit"),
        "scope": dict(document.get("scope", {})),
    }


def area_model_validation(
    simulator_root: Path | None = None,
) -> AreaModelValidation:
    """Read the structural area model's holdout report and fitted grid."""

    root = Path(simulator_root) if simulator_root else resolve_simulator_root().root
    source = root / AREA_COEFFICIENT_PATH
    document = _read_json(source)
    report = document.get("report")
    if not isinstance(report, Mapping):
        raise ModelValidationUnavailable(f"{source} carries no holdout report")
    blocks = report.get("full_chip_blocks")
    full_chip = (
        blocks.get("full_chip")
        if isinstance(blocks, Mapping) and isinstance(blocks.get("full_chip"), Mapping)
        else {}
    )
    per_family = {
        str(family): dict(record)
        for family, record in report.items()
        if family != "full_chip_blocks" and isinstance(record, Mapping)
    }
    if not per_family:
        raise ModelValidationUnavailable(
            f"{source} reports no per-family holdout errors"
        )
    fitted_mlen, fitted_blen = _fitted_area_domain(root)
    return AreaModelValidation(
        model_version=str(document.get("model_version", "")),
        per_family=per_family,
        full_chip=dict(full_chip),
        fitted_mlen=fitted_mlen,
        fitted_blen=fitted_blen,
        gate_level=_gate_level_area_validation(root),
    )


# --------------------------------------------------------------------------
# Energy: tier and per-component provenance
# --------------------------------------------------------------------------


def energy_model_validation() -> dict[str, Any]:
    """Return the analytic energy tier and each component's evidence scope.

    The analytic tier is a coefficient set, not a fitted model with a holdout,
    so what travels with it is the identity of every coefficient and the
    declared scope of each one -- including the two that are explicitly *not*
    calibrated (leakage and link).  Stating that is the disclosure; there is no
    holdout figure to quote and none is invented here.

    Two coefficients now additionally carry gate-level evidence that did not
    change them: an independent envelope that brackets the compute anchor, and
    a leakage density measured at a corner and scope too narrow to adopt.  Both
    are reported as cross-checks rather than as calibration, because that is
    what they are.
    """

    provenance = analytic_power_provenance()
    leakage = provenance.get("leakage") or {}
    measured_bound = (
        leakage.get("measured_lower_bound") if isinstance(leakage, Mapping) else None
    )
    cross_checks: dict[str, Any] = {
        "compute_anchor": provenance.get("compute_energy_cross_check"),
        "leakage_measured_lower_bound": (
            measured_bound
            if isinstance(measured_bound, Mapping)
            else {
                "unavailable": (
                    "the leakage coefficient declares no measured lower bound"
                ),
            }
        ),
    }
    return {
        "model": "analytic_decode_energy",
        "energy_tier": provenance.get("energy_tier"),
        "energy_id": provenance.get("energy_id"),
        "engine": provenance.get("engine"),
        "component_provenance": {
            "sram": provenance.get("sram"),
            "leakage": provenance.get("leakage"),
            "link": provenance.get("link"),
        },
        "gate_level_cross_checks": cross_checks,
        "gate_level_cross_check_note": (
            "independent gate-level evidence on the compute and leakage "
            "coefficients; both are matrix-machine-only figures at 25 C and "
            "neither changed a coefficient, so the tier is unchanged"
        ),
        "sram_access_accounting": provenance.get("sram_access_accounting"),
        "holdout": None,
        "holdout_note": (
            "the analytic tier is an identified coefficient set, not a fitted "
            "model; leakage and link coefficients are declared, not calibrated"
        ),
        "simulator_root": provenance.get("simulator_root"),
        "simulator_root_source": provenance.get("simulator_root_source"),
    }


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def pricing_model_validation(
    *,
    timing_evidence_path: str | Path | None = None,
    mlen: int | None = None,
    blen: int | None = None,
    simulator_root: Path | None = None,
) -> dict[str, Any]:
    """Compose every pricing model's validation quality into one record.

    Each component is reported independently, and a component whose artifact
    cannot be read is recorded as ``{"unavailable": <reason>}`` rather than
    omitted, so a missing figure is visible instead of silently absent.
    """

    def attempt(loader) -> Any:
        try:
            return loader()
        except (ModelValidationUnavailable, FileNotFoundError, ValueError) as error:
            return {"unavailable": str(error)}

    area = attempt(lambda: area_model_validation(simulator_root))
    area_record: Any
    if isinstance(area, AreaModelValidation):
        area_record = area.to_dict()
        area_record["queried_geometry"] = {"MLEN": mlen, "BLEN": blen}
        area_record["domain_status"] = area.domain_status(mlen, blen)
    else:
        area_record = area

    timing: Any
    if timing_evidence_path is None:
        timing = {
            "unavailable": "no timing-evidence artifact path was supplied",
        }
    else:
        timing = attempt(lambda: timing_model_validation(timing_evidence_path))
        if isinstance(timing, TimingModelValidation):
            timing = timing.to_dict()

    bandwidth = attempt(lambda: bandwidth_model_validation(simulator_root))
    if isinstance(bandwidth, BandwidthModelValidation):
        bandwidth = bandwidth.to_dict()

    return {
        "schema_version": "plena-pricing-model-validation",
        "admission_basis": (
            "priced by a validated pricing model; individual-validation "
            "coverage is disclosed per row"
        ),
        "timing": timing,
        "bandwidth": bandwidth,
        "area": area_record,
        "energy": attempt(energy_model_validation),
    }


__all__ = [
    "AREA_DOMAIN_INSIDE",
    "AREA_DOMAIN_OUTSIDE",
    "AREA_DOMAIN_UNKNOWN",
    "AREA_GATE_LEVEL_SCHEMA",
    "AREA_GATE_LEVEL_VALIDATION_PATH",
    "AreaModelValidation",
    "BandwidthModelValidation",
    "MODEL_ERROR_SCOPE",
    "ModelValidationUnavailable",
    "TimingModelValidation",
    "area_model_validation",
    "bandwidth_model_validation",
    "energy_model_validation",
    "pricing_model_validation",
    "timing_model_validation",
]
