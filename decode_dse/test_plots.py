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
            **common,
        },
        {
            "system_role": "measured_gpu_baseline",
            "tokens_per_second": 180.0,
            "throughput_evidence_tier": "measured",
            "energy_per_token_j": "",
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
