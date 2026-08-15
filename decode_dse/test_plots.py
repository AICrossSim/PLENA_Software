from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decode_dse.plots import (
    ANALYSIS_SCHEMA,
    HardwarePoint,
    NumericalPoint,
    _capacity_table,
    _handoff_table,
    _hardware_table,
    _landscape_table,
    _load_decode_analysis,
    _load_selected_publication_rows,
    _model_validation_table,
    _multichip_table,
    _numerical_table,
    _packedkv_table,
    _stage_breakdown_table,
    _vector_table,
    _write_csv,
    plot_accuracy_by_weight,
    plot_accuracy_landscape,
    plot_completion_matrix,
    plot_decode_capacity,
    plot_handoff_regimes,
    plot_hardware_pareto,
    plot_model_validation,
    plot_multichip_scaling,
    plot_packedkv_ablation,
    plot_screening_fidelity,
    plot_selected_deployment_evidence,
    plot_stage_breakdown,
    plot_vector_sensitivity,
)
from decode_dse.hardware.packedkv_claims import PACKEDKV_MODES, PRECISION_ROLES
from decode_dse.profiles import (
    DECODE_FORMATS,
    PROFILE_KIND_BF16_REFERENCE,
    DecodePrecisionProfile,
    enumerate_decode_profiles,
)


def _assert_rendered(paths: tuple[Path, ...]) -> None:
    assert paths
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_hardware_pareto_handles_single_point_and_missing_energy_tier(
    tmp_path: Path,
) -> None:
    profile = DecodePrecisionProfile.quantized(
        "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
    )
    point = HardwarePoint(
        profile=profile,
        candidate_id="synthetic-candidate",
        delta_nll=0.01,
        relative_perplexity_percent=1.01,
        tpot_ms=1.25,
        tps=204.8,
        energy_j=0.08,
        area_mm2=420.0,
        max_runtime_batch=256,
        chip_count=4,
        tp=2,
        kvp=2,
        energy_tier=None,
        area_budget_mm2=908.6,
    )
    _assert_rendered(
        plot_hardware_pareto(
            (point,),
            model_name="Synthetic decode model",
            output_dir=tmp_path,
            formats=("svg",),
        )
    )
    table = _hardware_table((point,))
    assert table[0]["energy_tier"] == ""
    assert table[0]["tokens_per_joule"] == 12.5
    _write_csv(
        tmp_path / "hardware.csv",
        table,
        fieldnames=tuple(table[0]),
    )


def test_selected_deployment_plot_keeps_evidence_tiers_separate(
    tmp_path: Path,
) -> None:
    common = {
        "headline_ratio_permitted": False,
        "ratio_block_reason": "throughput_evidence_tiers_differ",
        "throughput_ratio": "",
        "peak_roofline_row": False,
        "peak_roofline_ratio_permitted": False,
    }
    rows = (
        {
            "system_role": "selected_plena_deployment",
            "tokens_per_second": 240.0,
            "throughput_evidence_tier": "compiler_trace_request_calibrated",
            "energy_per_token_j": 0.08,
            "energy_evidence_tier": "analytic_anchored",
            **common,
        },
        {
            "system_role": "measured_gpu_baseline",
            "tokens_per_second": 180.0,
            "throughput_evidence_tier": "measured",
            "energy_per_token_j": "",
            "energy_evidence_tier": "unavailable",
            **common,
        },
    )
    _assert_rendered(
        plot_selected_deployment_evidence(
            rows,
            model_name="Synthetic decode model",
            output_dir=tmp_path,
            formats=("svg",),
        )
    )
    assert all(row["headline_ratio_permitted"] is False for row in rows)
    assert all(row["throughput_ratio"] == "" for row in rows)


def test_selected_publication_rows_bind_every_consumed_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from decode_dse import plots
    from decode_dse.hardware.design_space import HARDWARE_STORAGE_REVISION
    from decode_dse.software import gpu_baseline
    from decode_dse.software.benchmark_runner import PublicationContract
    from decode_dse.software.sweep_plan import write_immutable_json

    model_revision = "1" * 40
    tokenizer_revision = "2" * 40
    manifest = SimpleNamespace(
        model_name="synthetic/model",
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        canonical_hash="3" * 64,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": manifest.model_name,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "hardware_space": {
                    "RESOURCE_BUDGET": {
                        "reference_system": "B200x1",
                        "aggregate_area_limit_mm2": 826.0,
                        "aggregate_hbm_capacity_limit_bytes": 192_000_000_000,
                        "aggregate_hbm_bandwidth_limit_bytes_per_s": (
                            8_000_000_000_000.0
                        ),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_report_path = write_immutable_json(
        tmp_path / "gpu_report.json",
        {"fixture": "gpu-report"},
    )
    baseline_receipt_path = write_immutable_json(
        tmp_path / "gpu_receipt.json",
        {"fixture": "gpu-receipt"},
    )
    contract_path = write_immutable_json(
        tmp_path / "publication_contract.json",
        {"fixture": "publication-contract"},
    )
    contract_hash = "4" * 64
    configuration_id = "pub-configuration"
    alternative_id = "pubhw-alternative"
    profile_id = "refined-profile"
    candidate_id = "exact-candidate"
    record_hash = "5" * 64
    selected_configuration = SimpleNamespace(
        configuration_id=configuration_id,
        role="pareto",
        profile=SimpleNamespace(profile_id=profile_id),
    )
    refined_hardware_path = tmp_path / "refined_hardware.jsonl"
    refined_hardware_path.write_text("factorized synthetic fixture\n", encoding="utf-8")
    refined_hardware_sha256 = hashlib.sha256(
        refined_hardware_path.read_bytes()
    ).hexdigest()
    alternative = SimpleNamespace(
        alternative_id=alternative_id,
        configuration_id=configuration_id,
        profile_id=profile_id,
        candidate_id=candidate_id,
        record_hash=record_hash,
        hardware_artifact_sha256=refined_hardware_sha256,
        tpot_ms=4.0,
        energy_per_token_j=0.08,
        energy_tier="analytic_anchored",
    )
    contract = SimpleNamespace(
        canonical_hash=contract_hash,
        protocol=SimpleNamespace(
            model_name=manifest.model_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        ),
        configurations=(selected_configuration,),
        hardware_alternatives=(alternative,),
    )
    monkeypatch.setattr(
        PublicationContract,
        "from_dict",
        classmethod(lambda cls, value: contract),
    )
    publication_report_path = write_immutable_json(
        tmp_path / "publication_report.json",
        {
            "schema_version": "decode-publication-report",
            "contract_hash": contract_hash,
            "selection": {
                "selected": True,
                "accuracy_configuration_ids": [configuration_id],
            },
        },
    )
    final_selection_path = write_immutable_json(
        tmp_path / "final_selection.json",
        {
            "schema_version": "decode-final-publication-selection",
            "contract_hash": contract_hash,
            "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "benchmark_report_sha256": hashlib.sha256(
                publication_report_path.read_bytes()
            ).hexdigest(),
            "accuracy_pass_configuration_ids": [configuration_id],
            "hardware_artifacts": [{"sha256": refined_hardware_sha256}],
            "selection": {
                "configuration_id": configuration_id,
                "role": "pareto",
                "alternative_id": alternative_id,
                "profile_id": profile_id,
                "candidate_id": candidate_id,
                "hardware_record_hash": record_hash,
                "hardware_artifact_sha256": refined_hardware_sha256,
            },
        },
    )
    baseline_contract = SimpleNamespace(
        model_name=manifest.model_name,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        workspace_binding=SimpleNamespace(manifest_hash=manifest.canonical_hash),
        planned_device_labels=("B200",),
    )
    rebuilt_baseline = {
        "best_measured_by_device": {
            "GPU-test": {
                "device_label": "B200",
                "mean_batch_step_ms": 5.0,
                "energy": {"available": True},
            }
        }
    }
    monkeypatch.setattr(
        gpu_baseline,
        "validate_gpu_baseline_report",
        lambda report: (baseline_contract, rebuilt_baseline),
    )
    monkeypatch.setattr(
        gpu_baseline,
        "validate_gpu_baseline_stage_receipt",
        lambda report, receipt: receipt,
    )
    monkeypatch.setattr(
        gpu_baseline,
        "gpu_baseline_throughput_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            system_name="NVIDIA B200",
            batch_size=2,
            tokens_per_second=400.0,
            evidence_tier="measured",
        ),
    )
    monkeypatch.setattr(
        gpu_baseline,
        "gpu_baseline_energy_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            energy_per_token_j=0.5,
            tokens_per_joule=2.0,
            evidence_tier="measured",
        ),
    )
    hardware_row = {
        "record_hash": record_hash,
        "deployment_valid": True,
        "profile_id": profile_id,
        "candidate_id": candidate_id,
        "hardware": {"BATCH": 2},
        "metrics": {
            "execution_mode": "compiler_trace",
            "timing_calibrated": True,
            "timing_evidence_id": "timing-" + "6" * 64,
            "whole_model": {
                "rankable": True,
                "publication_timing_tier": "compiler_trace_request_calibrated",
                "tpot_ms": 4.0,
                "tps": 500.0,
                "calibrated_energy": {"total_j": 0.08},
            },
        },
    }
    monkeypatch.setattr(
        plots,
        "load_hardware_artifact",
        lambda path: (
            {
                "storage_revision": HARDWARE_STORAGE_REVISION,
                "provenance": {
                    "model_revision": model_revision,
                    "tokenizer_revision": tokenizer_revision,
                },
            },
            (hardware_row,),
        ),
    )

    rows, receipt = _load_selected_publication_rows(
        config_path=config_path,
        manifest=manifest,
        gpu_baseline_report_path=baseline_report_path,
        gpu_baseline_receipt_path=baseline_receipt_path,
        publication_contract_path=contract_path,
        publication_report_path=publication_report_path,
        final_selection_path=final_selection_path,
        refined_hardware_artifact_path=refined_hardware_path,
    )
    assert [row["system_role"] for row in rows] == [
        "selected_plena_deployment",
        "measured_gpu_baseline",
    ]
    assert all(row["headline_ratio_permitted"] is False for row in rows)
    assert all(row["throughput_ratio"] == "" for row in rows)
    assert receipt["sources"]["final_selection_sha256"] == hashlib.sha256(
        final_selection_path.read_bytes()
    ).hexdigest()


def test_decode_analysis_figures_and_csv_tables_render(
    tmp_path: Path,
) -> None:
    validation = tuple(
        {
            "model": "Synthetic-8B",
            "component": component,
            "evidence_tier": tier,
            "relative_error_percent": error,
            "evaluation_seconds": 0.02,
        }
        for component, tier, error in (
            ("compute", "emulator_calibrated", -2.0),
            ("memory", "ramulator_calibrated", 3.0),
            ("area", "dc_synthesised", 1.0),
            ("power", "analytic_anchored", 4.0),
        )
    )
    stages = tuple(
        {
            "stage": stage,
            "variant": variant,
            "matrix_cycles": 100.0 * scale,
            "vector_cycles": 70.0 * scale,
            "scalar_cycles": 40.0 * scale,
            "control_cycles": 10.0 * scale,
        }
        for stage in ("QKV projection", "Flash attention")
        for variant, scale in (("baseline", 1.0), ("enhanced", 0.8))
    )
    capacity = tuple(
        {
            "kv_format": kv_format,
            "context_tokens": context,
            "kv_bytes_per_token": bytes_per_token,
            "feasible_batch": max(1, 131_072 // context),
            "tpot_ms": context / 1_000.0,
        }
        for kv_format, bytes_per_token in (("MXINT4", 512), ("MXINT8", 1024))
        for context in (128, 1024)
    )
    handoff = tuple(
        {
            "regime": regime,
            "prefill_s": 0.2,
            "transfer_s": 0.1,
            "admission_s": 0.02,
            "wait_s": wait,
            "host_spill_s": spill,
            "ttft_s": 0.32 + wait + spill,
            "energy_j": 1.0 + spill,
            "prefill_utilization": utilization,
            "prefill_decode_ratio": 2.0,
            "prompt_tokens": 1024,
            "generation_tokens": 128,
            "precision": "MXINT4",
        }
        for regime, wait, spill, utilization in (
            ("fully_pipelined", 0.0, 0.0, 0.98),
            ("back_pressure", 0.2, 0.0, 0.70),
            ("host_buffered", 0.0, 0.4, 0.95),
        )
    )
    multichip = tuple(
        {
            "chip_count": chips,
            "tp": tp,
            "kvp": kvp,
            "tps": 100.0 * chips**0.8,
            "energy_per_token_j": 0.1 * chips**0.2,
            "energy_tier": tier,
        }
        for chips, tp, kvp, tier in (
            (1, 1, 1, "analytic_anchored"),
            (2, 2, 1, "analytic_anchored"),
            (4, 2, 2, ""),
        )
    )
    plot_specs = (
        (plot_model_validation, validation, _model_validation_table),
        (plot_stage_breakdown, stages, _stage_breakdown_table),
        (plot_decode_capacity, capacity, _capacity_table),
        (plot_handoff_regimes, handoff, _handoff_table),
        (plot_multichip_scaling, multichip, _multichip_table),
    )
    for index, (plotter, rows, table_builder) in enumerate(plot_specs):
        _assert_rendered(
            plotter(rows, output_dir=tmp_path, formats=("svg",))
        )
        table = table_builder(rows)
        _write_csv(
            tmp_path / f"table-{index}.csv",
            table,
            fieldnames=tuple(table[0]),
        )

    artifact_path = tmp_path / "decode_analysis.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": ANALYSIS_SCHEMA,
                "model_name": "Synthetic-8B",
                "model_validation": validation,
                "stage_breakdown": stages,
                "capacity": capacity,
                "handoff": handoff,
                "multichip": multichip,
            }
        ),
        encoding="utf-8",
    )
    loaded = _load_decode_analysis(artifact_path)
    assert loaded["model_name"] == "Synthetic-8B"
    assert len(loaded["multichip"]) == 3


def test_packedkv_causal_ablation_renders_all_controls(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    groups = []
    for precision_index, role in enumerate(PRECISION_ROLES):
        measurements = {}
        for mode_index, mode in enumerate(PACKEDKV_MODES):
            traffic_scale = (1.0, 0.72, 0.55, 0.48)[mode_index]
            measurements[mode] = SimpleNamespace(
                read_bytes_per_sequence_token=(
                    4096.0 * (precision_index + 1) * traffic_scale
                ),
                feasible_batch=64 * (mode_index + 1),
                capacity_limited_tokens_per_s=100.0 * (mode_index + 1),
                tpot_ms=10.0 / (mode_index + 1),
            )
        groups.append(
            SimpleNamespace(
                key=("gqa8", role),
                precision=SimpleNamespace(
                    role=role,
                    format_id=("MXINT8" if role == "i8" else "MXINT4"),
                ),
                topology=SimpleNamespace(role="gqa8"),
                by_mode=lambda values=measurements: values,
            )
        )
    evidence = SimpleNamespace(groups=tuple(groups))
    monkeypatch.setattr(
        "decode_dse.plots.load_packedkv_evidence",
        lambda _path: evidence,
    )
    monkeypatch.setattr(
        "decode_dse.plots.evaluate_packedkv_publication",
        lambda _evidence: SimpleNamespace(passed=True),
    )

    _assert_rendered(
        plot_packedkv_ablation(
            tmp_path / "synthetic-evidence.json",
            output_dir=tmp_path,
            formats=("svg",),
        )
    )
    table = _packedkv_table(evidence)
    assert len(table) == len(PRECISION_ROLES) * len(PACKEDKV_MODES)
    assert {row["mode"] for row in table} == set(PACKEDKV_MODES)


def test_completion_matrix_renders_an_all_failed_column(tmp_path: Path) -> None:
    entries = []
    terminal_rows = []
    for weight_format in DECODE_FORMATS:
        for kv_format in DECODE_FORMATS:
            profile = DecodePrecisionProfile.quantized(
                weight_format,
                weight_format,
                kv_format,
                "FP_E3M2",
            )
            entry = SimpleNamespace(profile=profile, profile_id=profile.profile_id)
            entries.append(entry)
            terminal_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "state": "failed" if kv_format == DECODE_FORMATS[0] else "succeeded",
                    "error_class": "SyntheticFailure" if kv_format == DECODE_FORMATS[0] else None,
                }
            )
    manifest = SimpleNamespace(entries=tuple(entries))
    _assert_rendered(
        plot_completion_matrix(
            manifest,
            terminal_rows,
            model_name="Synthetic decode model",
            output_dir=tmp_path,
            formats=("svg",),
        )
    )


def test_numerical_figures_and_tables_render_full_manifest(tmp_path: Path) -> None:
    profiles = enumerate_decode_profiles()
    points = tuple(
        NumericalPoint(
            ordinal=ordinal,
            profile=profile,
            mean_nll=(
                5.0
                if profile.kind == PROFILE_KIND_BF16_REFERENCE
                else 5.01 + ordinal * 1e-5
            ),
            runtime_seconds=0.01,
        )
        for ordinal, profile in enumerate(profiles)
    )
    validation_points = tuple(
        NumericalPoint(
            ordinal=point.ordinal,
            profile=point.profile,
            mean_nll=point.mean_nll + point.ordinal * 1e-7,
            runtime_seconds=0.02,
        )
        for point in points
    )
    reference_nll = 5.0
    for plotter in (
        plot_accuracy_by_weight,
        plot_accuracy_landscape,
        plot_vector_sensitivity,
    ):
        _assert_rendered(
            plotter(
                points,
                reference_nll=reference_nll,
                model_name="Synthetic decode model",
                output_dir=tmp_path,
                formats=("svg",),
            )
        )
    fidelity_paths, fidelity = plot_screening_fidelity(
        points,
        validation_points,
        model_name="Synthetic decode model",
        output_dir=tmp_path,
        formats=("svg",),
    )
    _assert_rendered(fidelity_paths)
    assert fidelity["passed"] is True

    entries = tuple(
        SimpleNamespace(
            ordinal=point.ordinal,
            profile=point.profile,
            profile_id=point.profile_id,
        )
        for point in points
    )
    terminal_rows = tuple(
        {
            "profile_id": point.profile_id,
            "state": "succeeded",
            "attempt": 1,
            "runtime_seconds": point.runtime_seconds,
            "result": {"mean_nll": point.mean_nll},
            "record_hash": "a" * 64,
            "validity": {"passed": True},
        }
        for point in points
    )
    manifest = SimpleNamespace(entries=entries)
    tables = (
        _numerical_table(
            manifest,
            terminal_rows,
            reference_nll=reference_nll,
        ),
        _landscape_table(points, reference_nll=reference_nll),
        _vector_table(points),
    )
    for index, table in enumerate(tables):
        _write_csv(
            tmp_path / f"numerical-table-{index}.csv",
            table,
            fieldnames=tuple(table[0]),
        )


def test_hardware_pareto_draws_dual_accuracy_envelopes(tmp_path: Path) -> None:
    def _point(weight_format: str, ppl_percent: float, tpot: float, energy: float):
        return HardwarePoint(
            profile=DecodePrecisionProfile.quantized(
                weight_format, "MXINT8", "MXINT8", "FP_E5M6"
            ),
            candidate_id=f"cand-{weight_format}",
            delta_nll=ppl_percent / 100.0,
            relative_perplexity_percent=ppl_percent,
            tpot_ms=tpot,
            tps=1000.0 / tpot,
            energy_j=energy,
            area_mm2=180.0,
            max_runtime_batch=64,
            chip_count=4,
            energy_tier="analytic_anchored",
            publication_timing_tier="stage_calibrated_analytic",
        )

    points = (
        _point("MXINT8", 0.5, 2.0, 0.10),
        _point("MXINT4", 4.0, 1.0, 0.05),
    )
    rendered = plot_hardware_pareto(
        points,
        model_name="Synthetic decode model",
        output_dir=tmp_path,
        formats=("svg",),
        accuracy_budgets={
            "strict_relative_perplexity": 1.01,
            "relaxed_relative_perplexity": 1.05,
        },
    )
    assert rendered
    assert all(path.is_file() and path.stat().st_size > 0 for path in rendered)
    # A strict budget that admits nothing must still render the relaxed
    # envelope rather than raising.
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    sparse = plot_hardware_pareto(
        points,
        model_name="Synthetic decode model",
        output_dir=sparse_dir,
        formats=("svg",),
        accuracy_budgets={
            "strict_relative_perplexity": 1.001,
            "relaxed_relative_perplexity": 1.05,
        },
    )
    assert sparse
    assert all(path.is_file() and path.stat().st_size > 0 for path in sparse)


def test_energy_efficiency_figure_and_context_semantics(tmp_path: Path) -> None:
    from decode_dse.plots import plot_energy_efficiency
    from decode_dse.software.gpu_baseline import build_analytic_energy_context

    point = HardwarePoint(
        profile=DecodePrecisionProfile.quantized(
            "MXINT4", "MXINT8", "MXINT4", "FP_E5M6"
        ),
        candidate_id="cand-energy",
        delta_nll=0.01,
        relative_perplexity_percent=1.0,
        tpot_ms=1.0,
        tps=1000.0,
        energy_j=0.02,
        area_mm2=300.0,
        max_runtime_batch=64,
        chip_count=4,
        energy_tier="analytic_anchored",
        publication_timing_tier="stage_calibrated_analytic",
    )
    baseline_report = {
        "energy_scope": "synchronized_board_energy_for_measured_decode_only",
        "results": [
            {
                "batch_size": 32,
                "device_label": "b200",
                "summary": {
                    "tokens_per_second": 1631.0,
                    "energy": {
                        "available": True,
                        "energy_per_token_j": 0.5,
                    },
                },
            }
        ],
    }
    context = build_analytic_energy_context(
        plena_points=(
            {
                "profile_id": point.profile.profile_id,
                "candidate_id": point.candidate_id,
                "energy_per_token_j": point.energy_j,
                "tokens_per_second": point.tps,
                "energy_tier": point.energy_tier,
                "publication_timing_tier": point.publication_timing_tier,
            },
        ),
        baseline_report=baseline_report,
    )
    assert context["not_a_headline_claim"] is True
    assert context["numerator_tier"] == "analytic_anchored"
    assert context["denominator_tier"] == "measured"
    assert context["ratio_semantics"] == "model_estimate_over_measured_gpu"
    assert context["context_energy_ratio_gpu_over_plena"] == 25.0

    # A measured PLENA numerator must be refused: that is headline territory.
    with pytest.raises(ValueError, match="headline comparison"):
        build_analytic_energy_context(
            plena_points=(
                {
                    "energy_per_token_j": 0.02,
                    "energy_tier": "measured",
                },
            ),
            baseline_report=baseline_report,
        )

    rendered = plot_energy_efficiency(
        (point,),
        baseline_rows=tuple(context["gpu_measured"]),
        model_name="Synthetic decode model",
        output_dir=tmp_path,
        formats=("svg",),
    )
    assert rendered
    assert all(path.is_file() and path.stat().st_size > 0 for path in rendered)
    table = _hardware_table((point,))
    assert table[0]["average_system_power_w"] == pytest.approx(20.0)


def _accuracy_budget_rows(
    reference_mean_nll: float,
) -> tuple[tuple[HardwarePoint, ...], tuple[object, ...]]:
    """Return matched figure and selection rows for one synthetic study.

    Each profile appears once on each side with identical identities and
    identical accuracy, expressed the way each side stores it: the figure
    carries a percentage perplexity increase, the selection record carries
    mean NLL.
    """

    import math

    from decode_dse.hardware.selection import ParetoPoint
    from decode_dse.plots import _relative_perplexity_percent

    rows = (
        ("MXINT8", 1.002, 4.0, 0.20),
        ("E4M3", 1.008, 2.0, 0.12),
        ("MXINT4", 1.020, 1.5, 0.07),
        ("E2M1", 1.040, 1.0, 0.05),
        ("MXINT2", 1.300, 0.8, 0.03),
    )
    hardware: list[HardwarePoint] = []
    selection: list[object] = []
    for weight_format, relative_perplexity, tpot, energy in rows:
        profile = DecodePrecisionProfile.quantized(
            weight_format, "MXINT8", "MXINT8", "FP_E5M6"
        )
        delta_nll = math.log(relative_perplexity)
        candidate_id = f"cand-{weight_format}"
        hardware.append(
            HardwarePoint(
                profile=profile,
                candidate_id=candidate_id,
                delta_nll=delta_nll,
                relative_perplexity_percent=_relative_perplexity_percent(delta_nll),
                tpot_ms=tpot,
                tps=1000.0 / tpot,
                energy_j=energy,
                area_mm2=180.0,
                max_runtime_batch=64,
                chip_count=4,
                energy_tier="analytic_anchored",
                publication_timing_tier="stage_calibrated_analytic",
            )
        )
        selection.append(
            ParetoPoint(
                profile=profile,
                mean_nll=reference_mean_nll + delta_nll,
                tpot_ms=tpot,
                tps=1000.0 / tpot,
                energy_per_token_j=energy,
                area_mm2=180.0,
                candidate_id=candidate_id,
                power_calibration_id="synthetic-power",
                cost_scope="whole_model",
                system_calibration_id="synthetic-system",
                head_service_calibration_id="synthetic-head",
                whole_model_rankable=True,
                energy_tier="analytic_anchored",
                publication_timing_tier="stage_calibrated_analytic",
            )
        )
    return tuple(hardware), tuple(selection)


def test_pareto_figure_and_selection_record_share_one_accuracy_budget(
    tmp_path: Path,
) -> None:
    # The figure stores accuracy as a percentage perplexity increase and the
    # selection record stores it as mean NLL against a reference. Before these
    # were reduced to one comparison the two could admit different point sets
    # while both looking correct, so the emitted frontier and the published
    # figure could disagree silently.
    from decode_dse.hardware.selection import (
        dual_accuracy_frontiers,
        relative_perplexity_from_mean_nll,
        relative_perplexity_from_percent,
    )
    from decode_dse.plots import (
        FRONTIER_FROM_LOCAL_RECOMPUTE,
        FRONTIER_FROM_RECORD,
        hardware_accuracy_envelopes,
    )

    reference = 2.52709
    hardware, selection = _accuracy_budget_rows(reference)
    budgets = {
        "strict_relative_perplexity": 1.01,
        "relaxed_relative_perplexity": 1.05,
    }
    record = dual_accuracy_frontiers(
        selection,
        reference_mean_nll=reference,
        strict_relative_perplexity=budgets["strict_relative_perplexity"],
        relaxed_relative_perplexity=budgets["relaxed_relative_perplexity"],
    )

    # Both sides reduce their own accuracy column to the same ratio.
    for figure_row, selection_row in zip(hardware, selection):
        assert relative_perplexity_from_percent(
            figure_row.relative_perplexity_percent
        ) == pytest.approx(
            relative_perplexity_from_mean_nll(selection_row.mean_nll, reference)
        )

    local = hardware_accuracy_envelopes(hardware, accuracy_budgets=budgets)
    assert local.source == FRONTIER_FROM_LOCAL_RECOMPUTE
    assert local.note.startswith("accuracy frontier: ")
    # Budget membership is the shared decision, and it agrees exactly.
    assert len(local.strict_members) == record["budgets"]["strict"]["admitted_points"]
    assert len(local.relaxed_members) == record["budgets"]["relaxed"]["admitted_points"]
    assert {point.candidate_id for point in local.strict_members} == {
        "cand-MXINT8",
        "cand-E4M3",
    }

    joined = hardware_accuracy_envelopes(
        hardware,
        accuracy_budgets=budgets,
        frontier_record=record,
    )
    assert joined.source == FRONTIER_FROM_RECORD
    assert joined.strict_members == local.strict_members
    assert joined.relaxed_members == local.relaxed_members
    # The record-driven envelope is exactly the emitted front, which keeps
    # accuracy as a dominance objective and so may be wider than the figure's
    # own two-objective envelope. That difference is now disclosed on the
    # figure instead of being invisible.
    for name, envelope in (("strict", joined.strict), ("relaxed", joined.relaxed)):
        emitted = {
            entry["candidate_id"] for entry in record["budgets"][name]["front"]
        }
        assert {point.candidate_id for point in envelope} == emitted
    assert set(local.strict).issubset(set(joined.strict))

    with_record = tmp_path / "with-record"
    with_record.mkdir()
    rendered = plot_hardware_pareto(
        hardware,
        model_name="Synthetic decode model",
        output_dir=with_record,
        formats=("svg",),
        accuracy_budgets=budgets,
        frontier_record=record,
    )
    _assert_rendered(rendered)
    # The figure must still render with no frontier record at all.
    without_record = tmp_path / "without-record"
    without_record.mkdir()
    fallback = plot_hardware_pareto(
        hardware,
        model_name="Synthetic decode model",
        output_dir=without_record,
        formats=("svg",),
        accuracy_budgets=budgets,
    )
    _assert_rendered(fallback)


def test_pareto_figure_fails_closed_on_a_disagreeing_frontier_record(
    tmp_path: Path,
) -> None:
    from decode_dse.hardware.selection import dual_accuracy_frontiers
    from decode_dse.plots import hardware_accuracy_envelopes

    reference = 2.52709
    hardware, selection = _accuracy_budget_rows(reference)
    budgets = {
        "strict_relative_perplexity": 1.01,
        "relaxed_relative_perplexity": 1.05,
    }
    record = dual_accuracy_frontiers(
        selection,
        reference_mean_nll=reference,
        strict_relative_perplexity=budgets["strict_relative_perplexity"],
        relaxed_relative_perplexity=budgets["relaxed_relative_perplexity"],
    )

    other_budgets = dict(budgets, strict_relative_perplexity=1.02)
    with pytest.raises(ValueError, match="different budgets"):
        hardware_accuracy_envelopes(
            hardware,
            accuracy_budgets=other_budgets,
            frontier_record=record,
        )

    unknown_schema = json.loads(json.dumps(record))
    unknown_schema["schema_version"] = "decode-some-other-record"
    with pytest.raises(ValueError, match="unsupported schema"):
        hardware_accuracy_envelopes(
            hardware,
            accuracy_budgets=budgets,
            frontier_record=unknown_schema,
        )

    foreign_row = json.loads(json.dumps(record))
    foreign_row["budgets"]["strict"]["front"][0]["candidate_id"] = "cand-elsewhere"
    with pytest.raises(ValueError, match="did not load"):
        hardware_accuracy_envelopes(
            hardware,
            accuracy_budgets=budgets,
            frontier_record=foreign_row,
        )

    # A record whose front names a row the figure's budget filter rejects is
    # exactly the silent divergence this path exists to prevent.
    outside_budget = json.loads(json.dumps(record))
    outside_budget["budgets"]["strict"]["front"] = [
        entry
        for entry in outside_budget["budgets"]["relaxed"]["front"]
    ]
    with pytest.raises(ValueError, match="budget filter rejects"):
        hardware_accuracy_envelopes(
            hardware,
            accuracy_budgets=budgets,
            frontier_record=outside_budget,
        )
