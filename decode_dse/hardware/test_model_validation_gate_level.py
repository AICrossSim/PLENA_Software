"""Contracts for surfacing gate-level cross-validation in the pricing record.

An independent Design Compiler campaign now bears on two pricing coefficients:
it cross-validates the structural area census, and it brackets the analytic
compute-energy anchor. Neither changed a coefficient, and the reporting has to
keep saying so. Three failure modes are guarded here:

* the evidence disappearing from the record entirely, which would lose an
  independent check on the two most load-bearing pricing models;
* the evidence appearing without its scope, which would invite comparing
  one-block micrometre-squared figures at MLEN 16-64 against full-chip
  millimetre-squared figures at MLEN 128-1024; and
* a *missing* artifact reading as a passing one, which is the failure mode the
  whole module exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from decode_dse.hardware import model_validation, power_bridge
from decode_dse.hardware.model_validation import (
    AREA_CALIBRATION_SOURCES,
    AREA_GATE_LEVEL_SCHEMA,
    AREA_GATE_LEVEL_VALIDATION_PATH,
    AreaModelValidation,
    energy_model_validation,
)


def _record(**overrides: Any) -> dict[str, Any]:
    """A minimal artifact shaped like the one the simulator writes."""

    document: dict[str, Any] = {
        "schema": AREA_GATE_LEVEL_SCHEMA,
        "n_points": 8,
        "independent_of_the_fit": True,
        "coefficients_changed": False,
        "scope": {
            "block": "matrix_machine",
            "unit": "um^2",
            "measured_mlen": [16, 32, 64],
            "not_comparable_to": (
                "full-chip mm^2 estimates at MLEN 128-1024; this campaign "
                "measures one block, in um^2, over MLEN 16-64, at 25 C"
            ),
        },
        "census_cross_validation": {
            "uniform_offset_census_over_campaign": 1.1242,
            "shape_and_precision_error_after_offset_pct": {
                "median": 0.41,
                "mean": 0.82,
                "max": 2.97,
            },
            "known_model_limit": "equal width formats are indistinguishable",
        },
    }
    document.update(overrides)
    return document


def _root(tmp_path: Path, document: Any | None) -> Path:
    if document is not None:
        target = tmp_path / AREA_GATE_LEVEL_VALIDATION_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document), encoding="utf-8")
    return tmp_path


def test_gate_level_summary_carries_its_verdict_and_its_scope(
    tmp_path: Path,
) -> None:
    summary = model_validation._gate_level_area_validation(
        _root(tmp_path, _record())
    )
    assert "unavailable" not in summary
    assert summary["n_points"] == 8
    assert summary["independent_of_the_fit"] is True
    assert summary["coefficients_changed"] is False
    assert summary["uniform_offset_census_over_campaign"] == pytest.approx(1.1242)
    assert summary["shape_and_precision_error_after_offset_pct"]["max"] == 2.97
    # The scope must travel with the agreement figure, never separately.
    assert summary["scope"]["unit"] == "um^2"
    assert "full-chip mm^2" in summary["scope"]["not_comparable_to"]
    assert "MLEN 128-1024" in summary["scope"]["not_comparable_to"]


def test_a_missing_campaign_reads_as_missing_not_as_passing(
    tmp_path: Path,
) -> None:
    summary = model_validation._gate_level_area_validation(_root(tmp_path, None))
    assert "unavailable" in summary
    assert "uniform_offset_census_over_campaign" not in summary


def test_an_undeclared_schema_is_rejected_rather_than_parsed(
    tmp_path: Path,
) -> None:
    summary = model_validation._gate_level_area_validation(
        _root(tmp_path, _record(schema="something-else"))
    )
    assert AREA_GATE_LEVEL_SCHEMA in summary["unavailable"]


def test_an_artifact_without_a_cross_validation_report_is_not_a_pass(
    tmp_path: Path,
) -> None:
    document = _record()
    del document["census_cross_validation"]
    summary = model_validation._gate_level_area_validation(
        _root(tmp_path, document)
    )
    assert "unavailable" in summary


def test_the_campaign_is_never_treated_as_a_calibration_source() -> None:
    """It validates the census; it must not become an input to the fit."""

    assert AREA_GATE_LEVEL_VALIDATION_PATH not in AREA_CALIBRATION_SOURCES


def test_the_area_record_always_declares_the_cross_validation_slot() -> None:
    validation = AreaModelValidation(
        model_version="test",
        per_family={"mxfp": {"holdout_mape_pct": 2.0}},
        full_chip={},
        fitted_mlen=(16, 32, 64),
        fitted_blen=(4, 8, 16),
    )
    record = validation.to_dict()
    assert "independent_gate_level_cross_validation" in record
    assert record["independent_gate_level_cross_validation"] == {}


def test_energy_validation_always_reports_both_gate_level_cross_checks() -> None:
    record = energy_model_validation()
    checks = record["gate_level_cross_checks"]
    assert set(checks) == {"compute_anchor", "leakage_measured_lower_bound"}
    # Whether the resolved checkout carries the records or not, each slot is a
    # mapping that either states its figures or states why it has none.
    for slot in checks.values():
        assert isinstance(slot, dict) and slot
    assert "25 C" in record["gate_level_cross_check_note"]
    assert "neither changed a coefficient" in record["gate_level_cross_check_note"]


def test_a_checkout_without_the_cross_check_says_so(monkeypatch) -> None:
    class _Bare:
        """A simulator checkout predating the campaign."""

    marker = power_bridge._optional_record(_Bare(), "COMPUTE_ENERGY_CROSS_CHECK")
    assert "COMPUTE_ENERGY_CROSS_CHECK" in marker["unavailable"]
    assert "anchor_pj_per_mac" not in marker


def test_a_declared_cross_check_is_passed_through_with_its_caveat() -> None:
    class _Declared:
        COMPUTE_ENERGY_CROSS_CHECK = {
            "anchor_pj_per_mac": 0.203,
            "coefficient_changed": False,
            "caveat": "declared-activity vectorless analysis",
        }

    record = power_bridge._optional_record(
        _Declared(),
        "COMPUTE_ENERGY_CROSS_CHECK",
    )
    assert record["anchor_pj_per_mac"] == 0.203
    assert record["coefficient_changed"] is False
    assert "declared-activity" in record["caveat"]
    # A copy, so a consumer cannot mutate the model's own declaration.
    record["anchor_pj_per_mac"] = 0.0
    assert _Declared.COMPUTE_ENERGY_CROSS_CHECK["anchor_pj_per_mac"] == 0.203
