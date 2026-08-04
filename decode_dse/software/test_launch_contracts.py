from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from decode_dse.hardware.selection import _deduplicate_profile_points
from decode_dse.legality import (
    StackValidity,
    evaluate_profile_legality,
    load_built_stack_validity,
)
from decode_dse.manifest import (
    QuantizerProvenance,
    QuantizerSource,
    ResolvedImportOrigin,
    SweepManifest,
    SweepManifestEntry,
    build_exhaustive_manifest,
    validate_sweep_config,
)
from decode_dse.profiles import DecodePrecisionProfile, enumerate_decode_profiles
from decode_dse.plots import (
    PORTABLE_WORKSPACE_PROVENANCE,
    RESULTS_PROVENANCE_SCHEMA,
    _build_results_provenance,
    _copy_sweep_provenance,
    _load_sweep_provenance,
    _write_json,
)
from decode_dse.software.runtime_environment import (
    GPUObservation,
    LaunchEnvironmentObservation,
    LaunchPreflightError,
    MutablePathObservation,
    RuntimeEnvironment,
    _architecture_supported,
    _observe_model_assets,
    dense_decoder_parameter_count,
    estimate_artifact_footprint,
    evaluate_launch_preflight,
)
from decode_dse.software.gpu_baseline import (
    ENERGY_UNAVAILABLE_METHOD,
    GPU_BASELINE_REPORT_SCHEMA,
    GPU_BASELINE_ENERGY_SOURCE,
    GPU_BASELINE_SCOPE,
    GPU_BASELINE_TIMING_SCOPE,
    MEASURED_EVIDENCE_TIER,
    NVML_POWER_TRACE_METHOD,
    NVML_TOTAL_ENERGY_METHOD,
    PEAK_ROOFLINE_EVIDENCE_TIER,
    EnergyEvidenceRow,
    GPUDeviceEnergyMeasurement,
    GPUEnergyMeasurement,
    GPUHardwareStateSnapshot,
    GPUPowerTraceSample,
    GPUBaselineContract,
    GPUBaselinePrompt,
    GPUBaselineRepetition,
    GPUBaselineResult,
    GPUBaselineWorkspaceBinding,
    ThroughputEvidenceRow,
    build_headline_energy_comparison,
    build_gpu_baseline_report,
    build_gpu_baseline_stage_receipt,
    build_headline_throughput_comparison,
    gpu_baseline_energy_evidence,
    gpu_baseline_throughput_evidence,
    load_gpu_baseline_workspace_binding,
)
from decode_dse.software.refinement_schedule import _hardware_point
from decode_dse.software.benchmark_runner import (
    LocalDatasetSnapshot,
    PublicationBenchmark,
    PublicationConfiguration,
    PublicationContract,
    PublicationHardwareAlternative,
    PublicationProtocol,
    build_publication_benchmark_manifest,
    load_publication_benchmark_manifest,
)
from decode_dse.software.sweep_plan import (
    GPUBaselinePlan,
    PromptManifest,
    PromptRecord,
    build_quantizer_provenance,
    build_run_plan,
    load_immutable_json,
    validate_run_plan,
    write_immutable_json,
)
from decode_dse.software.sweep_runner import (
    EvaluationOutcome,
    ExhaustiveSweepRunner,
)

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY / "decode_dse/configs/qwen3_32b.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _model_config() -> dict:
    return {
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
    }


def _good_observation() -> LaunchEnvironmentObservation:
    capacity = 100_000 * (1 << 20)
    path = MutablePathObservation(
        label="workspace",
        path="/publication/workspace",
        writable=True,
        lockable=True,
        free_bytes=256 * (1 << 30),
    )
    return LaunchEnvironmentObservation(
        package_versions={
            "torch": "2.8.0",
            "transformers": "4.53.2",
            "datasets": "4.0.0",
            "numpy": "2.3.1",
            "nvidia-ml-py": "13.595.45",
        },
        cuda_devices=tuple(
            GPUObservation(
                index=index,
                name="NVIDIA B200",
                total_bytes=capacity,
                free_bytes=capacity,
                compute_capability="sm_100",
                bf16_supported=True,
            )
            for index in range(4)
        ),
        torch_arch_list=("sm_80", "sm_86", "sm_90", "sm_100"),
        host_available_bytes=256 * (1 << 30),
        model_snapshot="/data/models/pinned-qwen3-32b",
        model_config=_model_config(),
        model_weight_bytes=65_524_246_528,
        model_assets_complete=True,
        model_asset_error=None,
        dataset_assets={
            "evaluation": "/data/datasets/evaluation-revision",
            "refinement_calibration": "/data/datasets/calibration-revision",
        },
        mase_origin=str((REPOSITORY.parent / "mase/src/chop/__init__.py").resolve()),
        mutable_paths=(path,),
    )


def _small_manifest() -> SweepManifest:
    provenance = QuantizerProvenance(
        sources=(
            QuantizerSource(
                component="test",
                path="mase/src/chop/nn/quantizers/mxint/fake.py",
                sha256="1" * 64,
            ),
        ),
        resolved_imports=(
            ResolvedImportOrigin(
                module="chop.nn.quantizers.mxint.fake",
                path="mase/src/chop/nn/quantizers/mxint/fake.py",
            ),
        ),
    )
    candidates = enumerate_decode_profiles()
    weight_format = candidates[0].weight_format
    profiles = tuple(
        profile for profile in candidates if profile.weight_format == weight_format
    )[:2]
    entries = tuple(
        SweepManifestEntry(
            ordinal=ordinal,
            profile=profile,
            legality=evaluate_profile_legality(profile),
        )
        for ordinal, profile in enumerate(profiles)
    )
    return SweepManifest(
        model_name="Qwen/Qwen3-32B",
        model_revision="a" * 40,
        model_architecture=_config()["model_architecture"],
        tokenizer_revision="a" * 40,
        quantizer_provenance=provenance,
        entries=entries,
    )


def test_aggregate_launch_preflight_and_exact_memory_arithmetic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config()
    good = evaluate_launch_preflight(
        config,
        _good_observation(),
        repository_root=REPOSITORY,
        device_labels=("b200",),
    )
    assert good.passed
    numerical, hardware, refinement = good.memory_estimates
    assert numerical.parameter_count == 32_762_123_264
    assert numerical.weight_bytes == 65_524_246_528
    assert numerical.kv_cache_bytes == 2_214_592_512
    assert numerical.required_bytes == 70_000 * (1 << 20)
    assert hardware.kv_cache_bytes == 1_140_850_688
    assert hardware.required_bytes == 70_000 * (1 << 20)
    assert refinement.kv_cache_bytes == 4_294_967_296
    assert refinement.required_bytes == 90_000 * (1 << 20)

    bad_path = replace(
        _good_observation().mutable_paths[0],
        writable=False,
        lockable=False,
        free_bytes=1 << 30,
        error="permission denied",
    )
    bad = replace(
        _good_observation(),
        package_versions={
            "torch": None,
            "transformers": None,
            "datasets": None,
            "numpy": None,
            "nvidia-ml-py": None,
        },
        cuda_devices=(),
        host_available_bytes=1 << 30,
        model_assets_complete=False,
        model_asset_error="missing pinned shards",
        dataset_assets={"evaluation": None, "refinement_calibration": None},
        mase_origin="/wrong/chop/__init__.py",
        mutable_paths=(bad_path,),
    )
    report = evaluate_launch_preflight(
        config,
        bad,
        repository_root=REPOSITORY,
        device_labels=("cuda:0",),
    )
    failure_codes = {check.code for check in report.checks if not check.passed}
    assert {
        "packages",
        "model_assets",
        "dataset_evaluation",
        "dataset_refinement_calibration",
        "mase_origin",
        "cuda",
        "device_labels",
        "gpu_memory_numerical_screen",
        "gpu_memory_hardware_validation",
        "gpu_memory_refinement",
        "host_memory",
        "path_workspace",
    } <= failure_codes
    with pytest.raises(LaunchPreflightError) as failure:
        report.require_passed()
    message = str(failure.value)
    assert "no CUDA device is visible" in message
    assert "weights 62488.79 + KV 2112.00" in message
    assert "Automatic device_map remains disabled" in message

    from decode_dse.software import sweep

    monkeypatch.setattr(
        sweep,
        "run_launch_preflight",
        lambda *args, **kwargs: good,
    )
    summary = sweep.create_workspace(
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "good-plan",
        device_labels=("b200",),
        dry_run=True,
    )
    assert summary["launch_preflight"]["passed"] is True
    assert summary["quantizer_provenance_hash"]
    qwen_trace_plan = summary["compiler_trace_preflight"]
    assert qwen_trace_plan["structurally_legal_hardware_candidates"] == 1_413_216
    assert qwen_trace_plan["compiler_geometry_eligible_hardware_candidates"] == 79_488
    assert qwen_trace_plan["compiler_base_hardware_signatures"] == 16_216
    assert qwen_trace_plan["exact_batch_record_signatures"] == 912
    assert qwen_trace_plan["unique_compiler_lowering_instantiations"] == 5_472
    assert qwen_trace_plan["unique_lazy_trace_instantiations"] == 912
    assert qwen_trace_plan["raw_profile_candidate_pairs"] == 811_185_984
    assert qwen_trace_plan["raw_context_point_resolutions"] == (
        2_491_963_342_848
    )
    assert qwen_trace_plan["physical_signature_pairs"] == 476_928
    assert qwen_trace_plan["projected_context_timing_resolutions"] == 476_928
    assert qwen_trace_plan["physical_context_step_outcomes"] == 1_465_122_816
    assert qwen_trace_plan["projected_full_evaluator_calls"] == 476_928
    assert qwen_trace_plan["projected_joined_identities"] == 476_928
    assert qwen_trace_plan["projected_trace_bytes"] == 9_622_781_952
    assert qwen_trace_plan["projected_digest_updates"] == 953_856
    assert "projected_wall_clock_seconds" not in qwen_trace_plan
    assert qwen_trace_plan["compiler_trace_preflight_feasible"] is True


def test_quantizer_sources_change_manifest_and_run_plan_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config()
    first = build_quantizer_provenance(REPOSITORY, config)
    assert dict((origin.module, origin.path) for origin in first.resolved_imports)[
        "chop.nn.quantizers.mxint.fake"
    ].endswith("/chop/nn/quantizers/mxint/fake.py")

    from decode_dse.software import sweep_plan

    original_hash = sweep_plan._sha256_source

    def changed_hash(path: Path) -> str:
        if path.name == "precision_bindings.py":
            return "f" * 64
        return original_hash(path)

    monkeypatch.setattr(sweep_plan, "_sha256_source", changed_hash)
    second = build_quantizer_provenance(REPOSITORY, config)
    assert first.canonical_hash != second.canonical_hash

    manifest = build_exhaustive_manifest(
        config["model_name"],
        config["model_revision"],
        config["model_architecture"],
        first,
        config["tokenizer_revision"],
    )
    changed_manifest = replace(manifest, quantizer_provenance=second)
    assert manifest.canonical_hash != changed_manifest.canonical_hash
    run_plan = build_run_plan(manifest, device_labels=("h100",))
    changed_plan = replace(
        run_plan,
        manifest_hash=changed_manifest.canonical_hash,
        quantizer_provenance=second,
    )
    assert run_plan.canonical_hash != changed_plan.canonical_hash
    with pytest.raises(ValueError, match="different quantizer arithmetic"):
        validate_run_plan(
            replace(run_plan, quantizer_provenance=second),
            manifest,
        )

    from decode_dse.software.sweep import _provenance, _validate_provenance
    from decode_dse.software.sweep_plan import write_immutable_json

    optional_config = _config()
    optional_refinement = dict(optional_config["refinement"])
    optional_refinement.pop("calibration_data")
    optional_config["refinement"] = optional_refinement
    optional_config_path = tmp_path / "config-without-refinement-dataset.json"
    optional_config_path.write_text(
        json.dumps(optional_config),
        encoding="utf-8",
    )
    optional_provenance = _provenance(
        repository=REPOSITORY,
        config_path=optional_config_path,
        manifest=manifest,
        plan=run_plan,
        prompts=None,
        created_at_utc="2026-08-01T00:00:00Z",
    )
    assert tuple(optional_provenance["datasets"]) == ("evaluation",)

    incomplete_config = dict(optional_config)
    incomplete_refinement = dict(optional_refinement)
    incomplete_refinement["calibration_data"] = {"dataset_name": "Salesforce/wikitext"}
    incomplete_config["refinement"] = incomplete_refinement
    incomplete_config_path = tmp_path / "config-with-incomplete-dataset.json"
    incomplete_config_path.write_text(
        json.dumps(incomplete_config),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="descriptor is incomplete"):
        _provenance(
            repository=REPOSITORY,
            config_path=incomplete_config_path,
            manifest=manifest,
            plan=run_plan,
            prompts=None,
            created_at_utc="2026-08-01T00:00:00Z",
        )

    provenance_path = tmp_path / "provenance.json"
    write_immutable_json(
        provenance_path,
        _provenance(
            repository=REPOSITORY,
            config_path=CONFIG_PATH,
            manifest=manifest,
            plan=run_plan,
            prompts=None,
            created_at_utc="2026-08-01T00:00:00Z",
        ),
    )
    with pytest.raises(RuntimeError, match="quantizer sources differ"):
        _validate_provenance(
            provenance_path,
            repository=REPOSITORY,
            config_path=CONFIG_PATH,
            manifest=manifest,
            plan=run_plan,
            prompts=None,
        )

    wrong_origin = tuple(
        (
            replace(
                origin,
                path="mase/src/chop/nn/quantizers/mxint_hardware.py",
            )
            if origin.module == "chop.nn.quantizers.mxint.fake"
            else origin
        )
        for origin in first.resolved_imports
    )
    with pytest.raises(ValueError, match="mxint.fake"):
        replace(first, resolved_imports=wrong_origin)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Executor:
    def __init__(self, clock: _Clock, *, fail_first: bool = False) -> None:
        self.clock = clock
        self.fail_first = fail_first
        self.evaluations = 0

    @contextmanager
    def open_weight_bank(self, weight_format, entries):
        self.clock.advance(5.0)
        yield weight_format

    @contextmanager
    def open_kv_admission_cache(self, kv_format):
        yield kv_format

    def evaluate(self, entry, *, weight_bank, kv_admission_cache):
        del entry, weight_bank, kv_admission_cache
        self.clock.advance(2.0)
        self.evaluations += 1
        if self.fail_first and self.evaluations == 1:
            raise RuntimeError("deliberate transient failure")
        return EvaluationOutcome(metrics={"mean_token_nll": 1.0})


class _InterruptingExecutor(_Executor):
    def __init__(self, clock: _Clock, *, interrupt_on: int) -> None:
        super().__init__(clock)
        self.interrupt_on = interrupt_on

    def evaluate(self, entry, *, weight_bank, kv_admission_cache):
        if self.evaluations + 1 == self.interrupt_on:
            self.evaluations += 1
            raise KeyboardInterrupt("simulated process termination")
        return super().evaluate(
            entry,
            weight_bank=weight_bank,
            kv_admission_cache=kv_admission_cache,
        )


def test_runner_reports_first_completion_continuous_eta_failure_and_resume(
    tmp_path: Path,
) -> None:
    manifest = _small_manifest()
    clock = _Clock()
    emitted: list[str] = []
    executor = _Executor(clock, fail_first=True)
    summary = ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=tmp_path,
        executor=executor,
        max_attempts=2,
        stage="numerical-screen",
        clock=clock,
        emit_progress=emitted.append,
    ).run()
    assert summary.succeeded == 2
    assert summary.failed_terminal == 0
    assert executor.evaluations == 3
    assert any("event=retry-required" in line for line in emitted)
    assert any("event=first-completion" in line for line in emitted)
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["work_class"] == "numerical"
    assert progress["completed_profiles"] == 2
    assert progress["unique_weight_banks_opened"] == 1
    assert progress["mean_trial_seconds"] == 3.0
    assert progress["estimated_remaining_seconds"] == 0.0

    resumed_messages: list[str] = []
    resumed_executor = _Executor(clock)
    resumed = ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=tmp_path,
        executor=resumed_executor,
        max_attempts=2,
        stage="numerical-screen",
        clock=clock,
        emit_progress=resumed_messages.append,
    ).run()
    assert resumed.succeeded == 2
    assert resumed_executor.evaluations == 0
    assert any("event=resume" in line for line in resumed_messages)


def test_runner_counts_terminal_failure(tmp_path: Path) -> None:
    manifest = replace(_small_manifest(), entries=_small_manifest().entries[:1])
    clock = _Clock()
    executor = _Executor(clock, fail_first=True)
    summary = ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=tmp_path,
        executor=executor,
        max_attempts=1,
        stage="hardware-validation",
        clock=clock,
        emit_progress=lambda message: None,
    ).run()
    assert summary.failed_terminal == 1
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["work_class"] == "hardware-validation"
    assert progress["failed_terminal_profiles"] == 1
    assert progress["event"] == "first-completion"


def test_interrupted_and_boundary_resumes_preserve_the_exact_manifest(
    tmp_path: Path,
) -> None:
    manifest = _small_manifest()
    reference = tmp_path / "reference"
    ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=reference,
        executor=_Executor(_Clock()),
        stage="numerical-screen",
        emit_progress=lambda message: None,
    ).run()
    expected_manifest = (reference / "manifest.json").read_bytes()

    interrupted = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt, match="simulated process"):
        ExhaustiveSweepRunner(
            manifest=manifest,
            output_dir=interrupted,
            executor=_InterruptingExecutor(_Clock(), interrupt_on=2),
            stage="numerical-screen",
            emit_progress=lambda message: None,
        ).run()
    with (interrupted / "status.jsonl").open("ab") as handle:
        handle.write(b'{"incomplete"')
    resumed = ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=interrupted,
        executor=_Executor(_Clock()),
        stage="numerical-screen",
        emit_progress=lambda message: None,
    ).run()
    assert resumed.succeeded == len(manifest.entries)
    assert (interrupted / "manifest.json").read_bytes() == expected_manifest

    boundary = tmp_path / "boundary"
    first = ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=boundary,
        executor=_Executor(_Clock()),
        stage="numerical-screen",
        emit_progress=lambda message: None,
    ).run(limit=1)
    assert first.pending == 1
    completed = ExhaustiveSweepRunner(
        manifest=manifest,
        output_dir=boundary,
        executor=_Executor(_Clock()),
        stage="numerical-screen",
        emit_progress=lambda message: None,
    ).run()
    assert completed.succeeded == len(manifest.entries)
    assert (boundary / "manifest.json").read_bytes() == expected_manifest


def test_model_general_resource_and_dry_run_contracts(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    llama_path = REPOSITORY / "decode_dse/configs/llama3_1_8b.json"
    llama = json.loads(llama_path.read_text(encoding="utf-8"))
    qwen = _config()
    validate_sweep_config(llama)
    assert dense_decoder_parameter_count(llama["model_architecture"]) == 8_030_261_248
    assert dense_decoder_parameter_count(qwen["model_architecture"]) == 32_762_123_264
    for config in (llama, qwen):
        footprint = estimate_artifact_footprint(
            config,
            config["model_architecture"],
        )
        assert footprint.policy == "content_addressed_recompute_per_format"
        assert (
            footprint.admission_peak_persisted_bytes
            < footprint.admission_total_logical_bytes
        )
        assert footprint.required_workspace_bytes > footprint.prefill_persisted_bytes
        assert footprint.to_dict()["total_logical_workspace_bytes"] == (
            footprint.prefill_persisted_bytes
            + footprint.admission_total_logical_bytes
        )

    from decode_dse.software import sweep

    report = evaluate_launch_preflight(
        qwen,
        _good_observation(),
        repository_root=REPOSITORY,
        device_labels=("b200",),
    )
    monkeypatch.setattr(
        sweep,
        "run_launch_preflight",
        lambda *args, **kwargs: report,
    )
    stable_software_hash = sweep._software_tree_hash(REPOSITORY)
    monkeypatch.setattr(
        sweep,
        "_software_tree_hash",
        lambda repository: stable_software_hash,
    )
    first = sweep.create_workspace(
        config_path=llama_path,
        output_dir=tmp_path / "dry-run",
        device_labels=("b200",),
        dry_run=True,
    )
    second = sweep.create_workspace(
        config_path=llama_path,
        output_dir=tmp_path / "dry-run",
        device_labels=("b200",),
        dry_run=True,
    )
    assert first["manifest"] == second["manifest"]
    assert first["run_plan"] == second["run_plan"]
    assert first["provenance"] == second["provenance"]
    assert len(first["manifest"]["entries"]) == 3585
    assert first["cost_declaration"]["manifest_profiles"] == 3585
    assert first["gpu_baseline_measurements"] == 4
    assert len(first["gpu_baseline_work_units"]) == 4
    assert first["cost_declaration"]["gpu_baseline_measurements"] == 4
    assert first["cost_declaration"]["gpu_baseline_prefill_runs"] == 12
    assert first["cost_declaration"]["gpu_baseline_decode_steps"] == 1728
    assert first["run_plan"]["gpu_baseline"] == llama["gpu_baseline"]
    assert first["cost_declaration"]["total_work_units"] == (
        first["cost_declaration"]["total_profile_evaluations"] + 4
    )
    llama_trace_plan = first["compiler_trace_preflight"]
    assert llama_trace_plan["structurally_legal_hardware_candidates"] == 1_848_096
    assert llama_trace_plan["compiler_geometry_eligible_hardware_candidates"] == 272_160
    assert llama_trace_plan["compiler_base_hardware_signatures"] == 21_192
    assert llama_trace_plan["exact_batch_record_signatures"] == 3_144
    assert llama_trace_plan["unique_compiler_lowering_instantiations"] == 18_864
    assert llama_trace_plan["unique_lazy_trace_instantiations"] == 3_144
    assert llama_trace_plan["raw_profile_candidate_pairs"] == 1_060_807_104
    assert llama_trace_plan["raw_context_point_resolutions"] == (
        3_258_799_423_488
    )
    assert llama_trace_plan["physical_signature_pairs"] == 1_632_960
    assert llama_trace_plan["projected_context_timing_resolutions"] == 1_632_960
    assert llama_trace_plan["physical_context_step_outcomes"] == 5_016_453_120
    assert llama_trace_plan["projected_full_evaluator_calls"] == 1_632_960
    assert llama_trace_plan["projected_joined_identities"] == 1_632_960
    assert llama_trace_plan["projected_trace_bytes"] == 13_392_936_960
    assert llama_trace_plan["projected_digest_updates"] == 3_265_920
    assert "projected_wall_clock_seconds" not in llama_trace_plan
    assert (
        first["cost_declaration"]["projection_status"]
        == "awaiting_first_completed_profile"
    )
    assert not (tmp_path / "dry-run").exists()

    pipeline_workspace = tmp_path / "pipeline-plan"
    write_immutable_json(
        pipeline_workspace / "run_plan.json",
        first["run_plan"],
    )
    monkeypatch.setattr(
        sweep,
        "capture_runtime_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pipeline dry-run must not inspect CUDA")
        ),
    )
    assert (
        sweep.sweep_pipeline_main(
            (
                "--config",
                str(llama_path),
                "--output-dir",
                str(pipeline_workspace),
                "--device-label",
                "b200",
                "--gpus",
                "0,1",
                "--dry-run",
            )
        )
        == 0
    )
    pipeline_dry_run = json.loads(capsys.readouterr().out)
    pipeline_commands = pipeline_dry_run["commands"]
    assert pipeline_commands[0]["name"] == "compiler-trace-artifacts"
    assert pipeline_commands[0]["outputs"] == [
        str(
            pipeline_workspace
            / "external"
            / "compiler_trace_artifacts.json"
        )
    ]
    assert pipeline_dry_run["compiler_trace_preflight_feasible"] is True
    assert pipeline_dry_run["unique_compiler_family_artifacts"] == 1
    assert pipeline_dry_run["unique_lazy_trace_instantiations"] > 0
    assert pipeline_dry_run["projected_trace_generation_calls"] == (
        pipeline_dry_run["unique_lazy_trace_instantiations"]
    )
    assert pipeline_dry_run["projected_trace_bytes"] > 0
    assert [
        command["name"]
        for command in pipeline_commands
        if command["name"].startswith("gpu-baseline-batch-")
    ] == [
        "gpu-baseline-batch-1",
        "gpu-baseline-batch-2",
        "gpu-baseline-batch-4",
        "gpu-baseline-batch-8",
    ]
    baseline_contract_command = next(
        command
        for command in pipeline_commands
        if command["name"] == "gpu-baseline-contract"
    )
    attention_index = baseline_contract_command["argv"].index(
        "--attention-implementation"
    )
    assert baseline_contract_command["argv"][attention_index + 1] == "sdpa"
    meter_index = baseline_contract_command["argv"].index(
        "--energy-meter-priority"
    )
    assert baseline_contract_command["argv"][meter_index + 1 : meter_index + 3] == [
        NVML_TOTAL_ENERGY_METHOD,
        NVML_POWER_TRACE_METHOD,
    ]
    interval_index = baseline_contract_command["argv"].index(
        "--power-trace-sample-interval-ms"
    )
    assert baseline_contract_command["argv"][interval_index + 1] == "10"

    missing_baseline = dict(llama)
    missing_baseline.pop("gpu_baseline")
    with pytest.raises(ValueError, match="gpu_baseline must be an explicit object"):
        validate_sweep_config(missing_baseline)
    mistyped_baseline = json.loads(json.dumps(llama))
    mistyped_baseline["gpu_baseline"]["first_gpu_only"] = 1
    with pytest.raises(ValueError, match="gpu_baseline.first_gpu_only"):
        validate_sweep_config(mistyped_baseline)

    failed_report = replace(
        report,
        checks=tuple(
            replace(
                check,
                passed=False,
                observed=(
                    "FileNotFoundError: pinned tokenizer assets are incomplete"
                ),
            )
            if check.code == "model_assets"
            else check
            for check in report.checks
        ),
    )
    monkeypatch.setattr(
        sweep,
        "run_launch_preflight",
        lambda *args, **kwargs: failed_report,
    )
    blocked_dry_run = sweep.create_workspace(
        config_path=llama_path,
        output_dir=tmp_path / "blocked-dry-run",
        device_labels=("b200",),
        dry_run=True,
    )
    assert blocked_dry_run["launch_preflight"]["passed"] is False
    assert blocked_dry_run["compiler_trace_preflight_feasible"] is True
    assert len(blocked_dry_run["manifest"]["entries"]) == 3585
    assert blocked_dry_run["provenance"]["model"]["name"] == (
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert not (tmp_path / "blocked-dry-run").exists()
    with pytest.raises(LaunchPreflightError, match="pinned tokenizer"):
        sweep.create_workspace(
            config_path=llama_path,
            output_dir=tmp_path / "blocked-execution",
            device_labels=("b200",),
            dry_run=False,
        )
    assert not (tmp_path / "blocked-execution").exists()


def test_compiler_trace_artifact_command_builds_the_configured_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from decode_dse.software import sweep

    config = _config()
    workspace = tmp_path / "workspace"
    output = workspace / "external" / "compiler_trace_artifacts.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    points = (SimpleNamespace(name="first"), SimpleNamespace(name="second"))
    monkeypatch.setattr(sweep, "_build_manifest", lambda *args: _small_manifest())
    monkeypatch.setattr(
        sweep,
        "_compiler_trace_feasibility",
        lambda *args: {
            "compiler_trace_preflight_feasible": True,
            "compiler_trace_preflight_blockers": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "_compiler_trace_generation_points",
        lambda *args: points,
    )
    observed: dict[str, object] = {}

    def build(point_contexts, destination, *, dry_run):
        resolved = tuple(point_contexts)
        observed["points"] = tuple(point for point, _ in resolved)
        observed["contexts"] = tuple(
            (contexts.start, contexts.stop, contexts.step)
            for _, contexts in resolved
        )
        observed["destination"] = destination
        observed["dry_run"] = dry_run
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{"artifact_set_id":"fixture"}\n', encoding="utf-8")
        return {"artifact_set_id": "fixture", "native_compile_calls": 2}

    monkeypatch.setitem(
        sys.modules,
        "compiler_trace_timing",
        SimpleNamespace(build_full_model_decode_artifact_set=build),
    )
    assert (
        sweep.compiler_trace_artifacts_main(
            (
                "--config",
                str(config_path),
                "--output-dir",
                str(workspace),
                "--output",
                str(output),
            )
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert observed == {
        "points": points,
        "contexts": ((512, 3584, 1), (512, 3584, 1)),
        "destination": output,
        "dry_run": False,
    }
    assert receipt["point_count"] == 2
    assert receipt["build"]["native_compile_calls"] == 2
    assert receipt["output"] == str(output)
    assert output.is_file()


def test_measured_gpu_baseline_binding_and_strict_comparison_tiers(
    tmp_path: Path,
) -> None:
    config = _config()
    provenance_path = tmp_path / "provenance.json"
    write_immutable_json(
        provenance_path,
        {
            "schema_version": "decode-sweep-provenance",
            "manifest_hash": "1" * 64,
            "run_plan_hash": "2" * 64,
            "prompt_manifest_hash": "3" * 64,
            "model": {
                "revision": config["model_revision"],
                "tokenizer_revision": config["tokenizer_revision"],
            },
        },
    )

    class _BundleIdentity:
        model_revision = config["model_revision"]
        tokenizer_revision = config["tokenizer_revision"]

        @staticmethod
        def prompt_manifest():
            class _PromptIdentity:
                canonical_hash = "3" * 64

            return _PromptIdentity()

    binding = load_gpu_baseline_workspace_binding(
        provenance_path,
        _BundleIdentity(),
    )
    assert GPUBaselineWorkspaceBinding.from_dict(binding.to_dict()) == binding
    contract = GPUBaselineContract(
        model_name=config["model_name"],
        model_revision=config["model_revision"],
        tokenizer_revision=config["tokenizer_revision"],
        sample_bundle_hash="5" * 64,
        workspace_binding=binding,
        prompts=tuple(
            GPUBaselinePrompt(
                document_id=f"hardware-{index:03d}",
                prompt_hash=hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
                token_count=512,
            )
            for index in range(64)
        ),
        attention_implementation="sdpa",
        warmup_steps=1,
        measured_steps=2,
        repetitions=3,
        planned_device_labels=("B200",),
        planned_batch_sizes=(2,),
        seed=0,
        source_tree_sha256="6" * 64,
    )
    assert GPUBaselineContract.from_dict(contract.to_dict()) == contract

    device_uuid = "GPU-publication-test"
    runtime = RuntimeEnvironment(
        logical={"device_name": "NVIDIA B200", "stack": "test"},
        observation={"device_uuid": device_uuid, "total_memory_bytes": 192_000},
    )

    def snapshot(phase: str) -> GPUHardwareStateSnapshot:
        values = (
            "2026/08/02 00:00:00.000",
            "600.00",
            "NVIDIA B200",
            device_uuid,
            "00000000:01:00.0",
            "P0",
            "192000",
            "1000",
            "40",
            "300",
            "1000",
            "1800",
            "1800",
            "3500",
            "Disabled",
        )
        raw = ", ".join(values)
        return GPUHardwareStateSnapshot(
            phase=phase,
            captured_at_utc="2026-08-02T00:00:00Z",
            raw_query_line=raw,
            raw_query_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            values=values,
        )

    repetitions = tuple(
        GPUBaselineRepetition(
            repetition=index,
            document_ids=(
                f"hardware-{index * 2:03d}",
                f"hardware-{index * 2 + 1:03d}",
            ),
            prefill_ms=20.0,
            decode_step_ms=(10.0, 10.0),
            q_len_one_calls=3,
            cache_growth_checks=3,
            first_token_count=2,
            generated_token_sha256=hashlib.sha256(
                f"tokens-{index}".encode()
            ).hexdigest(),
            peak_allocated_bytes=1_000,
            peak_reserved_bytes=2_000,
            energy_measurement=GPUEnergyMeasurement(
                repetition=index,
                generated_tokens=4,
                devices=(
                    GPUDeviceEnergyMeasurement(
                        device_uuid=device_uuid,
                        meter_method=NVML_TOTAL_ENERGY_METHOD,
                        measurement_status="measured",
                        started_at_monotonic_ns=(
                            1_000_000_000 + index * 20_000_000
                        ),
                        ended_at_monotonic_ns=(
                            1_010_000_000 + index * 20_000_000
                        ),
                        energy_j=2.0,
                        counter_start_mj=10_000 + index * 3_000,
                        counter_end_mj=12_000 + index * 3_000,
                        power_trace=(),
                        requested_power_sample_interval_ms=10,
                        unavailable_reason=None,
                    ),
                ),
            ),
        )
        for index in range(3)
    )
    phases = ["run_start"]
    for index in range(3):
        phases.extend((f"repetition_{index}_start", f"repetition_{index}_end"))
    phases.append("run_end")
    result = GPUBaselineResult(
        contract_hash=contract.contract_hash,
        device_label="B200",
        device_name="NVIDIA B200",
        device_uuid=device_uuid,
        runtime_environment=runtime,
        batch_size=2,
        state="succeeded",
        repetitions=repetitions,
        hardware_state_snapshots=tuple(snapshot(phase) for phase in phases),
        error_class=None,
        error_message=None,
        created_at_utc="2026-08-02T00:00:00Z",
    )
    report = build_gpu_baseline_report(contract, (result,))
    receipt = build_gpu_baseline_stage_receipt(
        report,
        provenance_path=provenance_path,
    )
    assert receipt["complete"] is True
    assert receipt["work_units"] == [
        {"device_label": "B200", "batch_size": 2, "first_gpu_only": True}
    ]
    budget = {
        "reference_system": "B200x1",
        "aggregate_area_limit_mm2": 826.0,
        "aggregate_hbm_capacity_limit_bytes": 192_000_000_000,
        "aggregate_hbm_bandwidth_limit_bytes_per_s": 8_000_000_000_000.0,
    }
    denominator = gpu_baseline_throughput_evidence(
        report,
        stage_receipt=receipt,
        device_label="B200",
        resource_budget=budget,
    )
    assert denominator.evidence_tier == MEASURED_EVIDENCE_TIER
    assert denominator.evidence_source == GPU_BASELINE_SCOPE
    energy_denominator = gpu_baseline_energy_evidence(
        report,
        stage_receipt=receipt,
        device_label="B200",
        resource_budget=budget,
    )
    assert energy_denominator.evidence_source == GPU_BASELINE_ENERGY_SOURCE
    assert energy_denominator.tokens_per_joule == pytest.approx(2.0)
    assert energy_denominator.energy_delay_product_j_s == pytest.approx(0.005)
    energy_numerator = EnergyEvidenceRow(
        system_name="PLENA measured prototype",
        model_name=energy_denominator.model_name,
        model_revision=energy_denominator.model_revision,
        context_length=energy_denominator.context_length,
        batch_size=2,
        energy_per_token_j=0.25,
        mean_decode_step_s=0.005,
        meter_method="measured_dc_power_rail",
        device_ids=("plena-prototype",),
        evidence_source="plena_cached_q1_measured_board_energy",
        resource_budget_hash=energy_denominator.resource_budget_hash,
        artifact_hash="9" * 64,
        workspace_provenance_sha256=(
            energy_denominator.workspace_provenance_sha256
        ),
    )
    energy_comparison = build_headline_energy_comparison(
        plena_measurement=energy_numerator,
        gpu_measurement=energy_denominator,
        resource_budget=budget,
    )
    assert energy_comparison["tokens_per_joule_ratio"] == pytest.approx(2.0)
    assert energy_comparison["energy_delay_product_improvement"] == pytest.approx(4.0)
    assert energy_comparison["analytic_substitution_permitted"] is False
    trace_measurement = GPUDeviceEnergyMeasurement(
        device_uuid=device_uuid,
        meter_method=NVML_POWER_TRACE_METHOD,
        measurement_status="measured",
        started_at_monotonic_ns=2_000_000_000,
        ended_at_monotonic_ns=3_000_000_000,
        energy_j=100.0,
        counter_start_mj=None,
        counter_end_mj=None,
        power_trace=(
            GPUPowerTraceSample(2_000_000_000, 100_000),
            GPUPowerTraceSample(3_000_000_000, 100_000),
        ),
        requested_power_sample_interval_ms=10,
        unavailable_reason=None,
    )
    assert trace_measurement.to_dict()["power_trace_sample_count"] == 2

    unavailable_repetitions = tuple(
        replace(
            repetition,
            energy_measurement=GPUEnergyMeasurement(
                repetition=repetition.repetition,
                generated_tokens=4,
                devices=(
                    GPUDeviceEnergyMeasurement(
                        device_uuid=device_uuid,
                        meter_method=ENERGY_UNAVAILABLE_METHOD,
                        measurement_status="unavailable",
                        started_at_monotonic_ns=(
                            4_000_000_000
                            + repetition.repetition * 20_000_000
                        ),
                        ended_at_monotonic_ns=(
                            4_010_000_000
                            + repetition.repetition * 20_000_000
                        ),
                        energy_j=None,
                        counter_start_mj=None,
                        counter_end_mj=None,
                        power_trace=(),
                        requested_power_sample_interval_ms=10,
                        unavailable_reason=(
                            "NVML energy is not supported by publication GPU"
                        ),
                    ),
                ),
            ),
        )
        for repetition in repetitions
    )
    unavailable_report = build_gpu_baseline_report(
        contract,
        (replace(result, repetitions=unavailable_repetitions),),
    )
    unavailable_receipt = build_gpu_baseline_stage_receipt(
        unavailable_report,
        provenance_path=provenance_path,
    )
    unavailable_throughput = gpu_baseline_throughput_evidence(
        unavailable_report,
        stage_receipt=unavailable_receipt,
        device_label="B200",
        resource_budget=budget,
    )
    assert unavailable_throughput.tokens_per_second == denominator.tokens_per_second
    with pytest.raises(ValueError, match="not supported by publication GPU"):
        gpu_baseline_energy_evidence(
            unavailable_report,
            stage_receipt=unavailable_receipt,
            device_label="B200",
            resource_budget=budget,
        )
    numerator = ThroughputEvidenceRow(
        system_name="PLENA measured prototype",
        model_name=denominator.model_name,
        model_revision=denominator.model_revision,
        context_length=denominator.context_length,
        batch_size=2,
        tokens_per_second=100.0,
        evidence_tier=MEASURED_EVIDENCE_TIER,
        evidence_source="plena_cached_q1_measurement",
        resource_budget_hash=denominator.resource_budget_hash,
        artifact_hash="7" * 64,
        workspace_provenance_sha256=denominator.workspace_provenance_sha256,
    )
    peak = ThroughputEvidenceRow(
        system_name="A100 peak roofline",
        model_name=denominator.model_name,
        model_revision=denominator.model_revision,
        context_length=denominator.context_length,
        batch_size=2,
        tokens_per_second=10_000.0,
        evidence_tier=PEAK_ROOFLINE_EVIDENCE_TIER,
        evidence_source="published_peak_roofline",
        resource_budget_hash=denominator.resource_budget_hash,
        artifact_hash="8" * 64,
        workspace_provenance_sha256=None,
    )
    comparison = build_headline_throughput_comparison(
        plena_measurement=numerator,
        gpu_measurement=denominator,
        peak_roofline_rows=(peak,),
        resource_budget=budget,
    )
    assert comparison["headline"]["throughput_ratio"] == 0.5
    assert comparison["peak_roofline_ratio_permitted"] is False
    assert all(
        "throughput_ratio" not in row
        for row in comparison["peak_roofline_table"]
    )
    with pytest.raises(ValueError, match="measured evidence"):
        build_headline_throughput_comparison(
            plena_measurement=replace(
                numerator,
                evidence_tier=PEAK_ROOFLINE_EVIDENCE_TIER,
            ),
            gpu_measurement=denominator,
            peak_roofline_rows=(peak,),
            resource_budget=budget,
        )
    with pytest.raises(ValueError, match="workspace_provenance"):
        build_headline_throughput_comparison(
            plena_measurement=replace(
                numerator,
                workspace_provenance_sha256="9" * 64,
            ),
            gpu_measurement=denominator,
            peak_roofline_rows=(peak,),
            resource_budget=budget,
        )


def test_measured_gpu_energy_meter_counter_and_trace_fallback(monkeypatch) -> None:
    from decode_dse.software import gpu_baseline

    device_uuid = "GPU-meter-test"

    class _Torch:
        @staticmethod
        def device(value):
            return value

        class cuda:
            @staticmethod
            def synchronize(device):
                assert device == "cuda:0"

    class _FakeNVML:
        def __init__(self, *, counter_supported: bool) -> None:
            self.counter_supported = counter_supported
            self.counter_values = iter((1_000, 2_000, 5_000))
            self.shutdown = False

        @staticmethod
        def nvmlInit() -> None:
            return None

        @staticmethod
        def nvmlDeviceGetHandleByUUID(value):
            assert value == device_uuid
            return "handle"

        @staticmethod
        def nvmlDeviceGetUUID(handle):
            assert handle == "handle"
            return device_uuid

        def nvmlDeviceGetTotalEnergyConsumption(self, handle):
            assert handle == "handle"
            if not self.counter_supported:
                raise RuntimeError("energy counter is not supported")
            return next(self.counter_values)

        @staticmethod
        def nvmlDeviceGetPowerUsage(handle):
            assert handle == "handle"
            return 100_000

        def nvmlShutdown(self) -> None:
            self.shutdown = True

    counter_nvml = _FakeNVML(counter_supported=True)
    monkeypatch.setattr(
        gpu_baseline.importlib,
        "import_module",
        lambda name: counter_nvml,
    )
    counter_meter = gpu_baseline._NVMLBoardEnergyMeter(
        device_uuid=device_uuid,
        method_priority=gpu_baseline.GPU_BASELINE_ENERGY_METER_PRIORITY,
        power_trace_sample_interval_ms=10,
    )
    counter_meter.begin(_Torch, device="cuda:0")
    counter_energy = counter_meter.end(
        _Torch,
        device="cuda:0",
        repetition=0,
        generated_tokens=4,
    )
    assert counter_energy.meter_method == NVML_TOTAL_ENERGY_METHOD
    assert counter_energy.total_energy_j == pytest.approx(3.0)
    counter_meter.close()
    assert counter_nvml.shutdown is True

    trace_nvml = _FakeNVML(counter_supported=False)
    monkeypatch.setattr(
        gpu_baseline.importlib,
        "import_module",
        lambda name: trace_nvml,
    )
    trace_meter = gpu_baseline._NVMLBoardEnergyMeter(
        device_uuid=device_uuid,
        method_priority=gpu_baseline.GPU_BASELINE_ENERGY_METER_PRIORITY,
        power_trace_sample_interval_ms=10,
    )
    trace_meter.begin(_Torch, device="cuda:0")
    trace_energy = trace_meter.end(
        _Torch,
        device="cuda:0",
        repetition=0,
        generated_tokens=4,
    )
    assert trace_energy.meter_method == NVML_POWER_TRACE_METHOD
    assert trace_energy.available is True
    assert len(trace_energy.devices[0].power_trace) >= 2
    assert trace_energy.total_energy_j > 0
    trace_meter.close()
    assert trace_nvml.shutdown is True

    with pytest.raises(ValueError, match="cannot mix NVML meter methods"):
        GPUEnergyMeasurement(
            repetition=0,
            generated_tokens=4,
            devices=(
                counter_energy.devices[0],
                replace(
                    trace_energy.devices[0],
                    device_uuid="GPU-meter-test-second",
                ),
            ),
        )


def test_incomplete_tokenizer_keeps_model_architecture_evidence(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (REPOSITORY / "decode_dse/configs/llama3_1_8b.json").read_text(encoding="utf-8")
    )
    cache = tmp_path / "models"
    config["hf_cache_dir"] = str(cache)
    snapshot = (
        cache
        / "models--meta-llama--Llama-3.1-8B-Instruct"
        / "snapshots"
        / config["model_revision"]
    )
    snapshot.mkdir(parents=True)
    model_config = dict(config["model_architecture"])
    model_config.pop("head_dim")
    model_config["torch_dtype"] = "bfloat16"
    (snapshot / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    shard_name = "model-00001-of-00001.safetensors"
    (snapshot / shard_name).write_bytes(b"stub")
    expected_weight_bytes = 8_030_261_248 * 2
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": expected_weight_bytes},
                "weight_map": {"model.embed_tokens.weight": shard_name},
            }
        ),
        encoding="utf-8",
    )

    path, observed_config, weight_bytes, complete, error = _observe_model_assets(config)
    assert path == str(snapshot)
    assert weight_bytes == expected_weight_bytes
    assert not complete
    assert error == (
        "FileNotFoundError: pinned tokenizer assets are incomplete: "
        f"{snapshot / 'tokenizer.json'}, {snapshot / 'tokenizer_config.json'}"
    )
    observation = replace(
        _good_observation(),
        model_snapshot=path,
        model_config=observed_config,
        model_weight_bytes=weight_bytes,
        model_assets_complete=complete,
        model_asset_error=error,
    )
    report = evaluate_launch_preflight(
        config,
        observation,
        repository_root=REPOSITORY,
        device_labels=("b200",),
    )
    checks = {
        check.code: check
        for check in report.checks
    }
    assert not checks["model_assets"].passed
    assert checks["model_assets"].observed == error
    assert checks["model_architecture"].passed
    assert checks["model_weight_size"].passed
    assert len(report.memory_estimates) == 3
    assert max(item.required_bytes for item in report.memory_estimates) == 36_000 * (
        1 << 20
    )


def test_refinement_hardware_points_preserve_energy_identity_and_tier() -> None:
    profile = _small_manifest().entries[0].profile

    def point(tier: str, identity: str, total_j: float):
        row = {
            "candidate_id": f"candidate-{tier}",
            "deployment_valid": True,
            "packedkv_selector_valid": True,
            "error_code": None,
            "metrics": {
                "area_source": "analytic_full_chip",
                "area_mm2": 100.0,
                "timing_calibrated": True,
                "runtime_feasible": True,
                "capacity": {"feasible": True},
                "whole_model": {
                    "rankable": True,
                    "tpot_ms": 10.0,
                    "tps": 100.0,
                    "system_calibration_id": "system-calibration",
                    "calibrated_energy": {
                        "energy_tier": tier,
                        "energy_id": identity,
                        "calibration_id": "underlying-calibration",
                        "total_j": total_j,
                    },
                },
                "output_head_boundary": {
                    "estimate": {"calibration_id": "head-calibration"}
                },
            },
        }
        return _hardware_point(row, profile=profile, mean_nll=1.0)

    analytic = point("analytic_anchored", "analytic-energy", 1.0)
    calibrated = point("dc_calibrated", "dc-energy", 10.0)
    assert analytic is not None
    assert calibrated is not None
    assert analytic.power_calibration_id == "analytic-energy"
    assert analytic.energy_tier == "analytic_anchored"
    assert calibrated.power_calibration_id == "dc-energy"
    assert (
        _deduplicate_profile_points((analytic, calibrated))[profile.profile_id]
        is calibrated
    )


def test_stubbed_stage_pipeline_is_idempotent_end_to_end(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from decode_dse.software import sweep

    config = _config()
    report = evaluate_launch_preflight(
        config,
        _good_observation(),
        repository_root=REPOSITORY,
        device_labels=("b200",),
    )
    monkeypatch.setattr(
        sweep,
        "run_launch_preflight",
        lambda *args, **kwargs: report,
    )
    monkeypatch.setattr(
        sweep,
        "_require_compiler_trace_feasible",
        lambda plan: None,
    )
    prompts = PromptManifest(
        dataset_name="stubbed-publication-inputs",
        dataset_revision="1" * 40,
        numerical_screen=tuple(
            PromptRecord(
                document_id=f"numerical-{index:03d}",
                prompt_hash=hashlib.sha256(f"n{index}".encode()).hexdigest(),
            )
            for index in range(16)
        ),
        hardware_validation=tuple(
            PromptRecord(
                document_id=f"hardware-{index:03d}",
                prompt_hash=hashlib.sha256(f"h{index}".encode()).hexdigest(),
            )
            for index in range(64)
        ),
    )
    prompt_path = tmp_path / "prompts.json"
    write_immutable_json(prompt_path, prompts.to_dict())
    workspace = tmp_path / "workspace"
    sweep.create_workspace(
        config_path=CONFIG_PATH,
        output_dir=workspace,
        device_labels=("b200",),
        prompt_manifest_path=prompt_path,
        numerical_screen_workers=1,
        hardware_validation_workers=1,
    )
    manifest_bytes = (workspace / "manifest.json").read_bytes()
    recorded_software_hash = json.loads(
        (workspace / "provenance.json").read_text(encoding="utf-8")
    )["software_tree_sha256"]
    monkeypatch.setattr(
        sweep,
        "_software_tree_hash",
        lambda repository: recorded_software_hash,
    )
    plan = sweep._load_plan(workspace / "run_plan.json")
    stage_ids = {
        stage: sweep._stage_ids(plan, stage)[:2]
        for stage in (
            "preflight",
            "validation-pilot",
            "numerical-screen",
            "hardware-validation",
        )
    }
    monkeypatch.setattr(
        sweep,
        "_stage_ids",
        lambda plan, stage: stage_ids[stage],
    )
    monkeypatch.setattr(
        sweep,
        "_require_stage_completion",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sweep,
        "_load_executor_factory",
        lambda spec: (lambda context: _Executor(_Clock())),
    )

    class _PassingGates:
        def require_passed(self) -> None:
            return None

    monkeypatch.setattr(sweep, "load_preflight_evidence", lambda path: object())
    monkeypatch.setattr(
        sweep,
        "evaluate_preflight_gates",
        lambda *args, **kwargs: _PassingGates(),
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    stage_manifests: dict[str, bytes] = {}
    for stage in stage_ids:
        summary = sweep.launch_stage(
            config_path=CONFIG_PATH,
            output_dir=workspace,
            stage=stage,
            executor_factory="decode_dse.tests:stub_executor",
            evidence_path=(
                evidence
                if stage in {"numerical-screen", "hardware-validation"}
                else None
            ),
            device_label="b200",
        )
        assert summary.succeeded == 2
        stage_root = workspace / stage
        stage_manifests[stage] = (stage_root / "manifest.json").read_bytes()
        assert (stage_root / "invocation.json").is_file()
        assert len(tuple((stage_root / "completed").glob("*.json"))) == 2

    for stage in stage_ids:
        resumed = sweep.launch_stage(
            config_path=CONFIG_PATH,
            output_dir=workspace,
            stage=stage,
            executor_factory="decode_dse.tests:stub_executor",
            evidence_path=(
                evidence
                if stage in {"numerical-screen", "hardware-validation"}
                else None
            ),
            device_label="b200",
        )
        assert resumed.attempts_written == 0
        assert (workspace / stage / "manifest.json").read_bytes() == stage_manifests[
            stage
        ]
    assert (workspace / "manifest.json").read_bytes() == manifest_bytes


@pytest.fixture
def local_publication_dataset_fixture(tmp_path: Path):
    cache_root = (tmp_path / "dataset-cache").resolve()
    cache_root.mkdir()
    rows = {
        "Salesforce/wikitext": (
            {"text": "first document"},
            {"text": "second document"},
        ),
        "google/IFEval": (
            {
                "key": 1000,
                "prompt": "Write a short answer.",
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"num_words": 10}],
            },
            {
                "key": 1001,
                "prompt": "Use two bullets.",
                "instruction_id_list": ["detectable_format:number_bullet_lists"],
                "kwargs": [{"num_bullets": 2}],
            },
        ),
        "openai/gsm8k": (
            {"question": "What is 1 + 1?", "answer": "#### 2"},
            {"question": "What is 2 + 2?", "answer": "#### 4"},
        ),
    }
    revisions = {
        "Salesforce/wikitext": "1" * 40,
        "google/IFEval": "2" * 40,
        "openai/gsm8k": "3" * 40,
    }
    declarations = {
        "wikitext2": {
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "dataset_revision": revisions["Salesforce/wikitext"],
            "split": "test",
            "cache_dir": str(cache_root),
            "content_columns": ["text"],
            "id_column": None,
            "source_item_count": 2,
        },
        "ifeval": {
            "dataset_name": "google/IFEval",
            "dataset_config": "default",
            "dataset_revision": revisions["google/IFEval"],
            "split": "train",
            "cache_dir": str(cache_root),
            "content_columns": [
                "key",
                "prompt",
                "instruction_id_list",
                "kwargs",
            ],
            "id_column": "key",
            "source_item_count": 2,
        },
        "gsm8k": {
            "dataset_name": "openai/gsm8k",
            "dataset_config": "main",
            "dataset_revision": revisions["openai/gsm8k"],
            "split": "test",
            "cache_dir": str(cache_root),
            "content_columns": ["question", "answer"],
            "id_column": None,
            "source_item_count": 2,
        },
    }
    sources = {}
    for index, (dataset_name, records) in enumerate(rows.items()):
        source = cache_root / f"snapshot-{index:02d}.jsonl"
        source.write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in records
            ),
            encoding="utf-8",
        )
        sources[dataset_name] = source

    calls = []

    def load_snapshot(specification):
        dataset_name = str(specification["dataset_name"])
        calls.append(dict(specification))
        return LocalDatasetSnapshot(
            records=rows[dataset_name],
            source_files=(sources[dataset_name],),
        )

    return (
        {"publication": {"benchmark_datasets": declarations}},
        load_snapshot,
        calls,
        sources,
    )


def test_local_only_publication_benchmark_manifest_is_deterministic(
    monkeypatch,
    tmp_path: Path,
    local_publication_dataset_fixture,
) -> None:
    from decode_dse.software import benchmark_runner

    config, loader, calls, sources = local_publication_dataset_fixture
    first = build_publication_benchmark_manifest(
        config,
        load_snapshot=loader,
    )
    second = build_publication_benchmark_manifest(
        config,
        load_snapshot=loader,
    )
    assert first == second
    assert [record["name"] for record in first["benchmarks"]] == [
        "wikitext2",
        "ifeval",
        "gsm8k",
    ]
    assert all(record["full_evaluation"] for record in first["benchmarks"])
    assert all(record["dataset_config"] for record in first["benchmarks"])
    assert all(
        PublicationBenchmark.from_dict(record).source_item_count == 2
        for record in first["benchmarks"]
    )
    assert len(calls) == 6

    config_path = tmp_path / "publication-config.json"
    output_path = tmp_path / "benchmarks.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_load_local_huggingface_snapshot",
        loader,
    )
    assert benchmark_runner.build_benchmark_manifest_main(
        ("--config", str(config_path), "--output", str(output_path))
    ) == 0
    output_bytes = output_path.read_bytes()
    loaded = load_publication_benchmark_manifest(output_path)
    assert tuple(benchmark.name for benchmark in loaded) == (
        "wikitext2",
        "ifeval",
        "gsm8k",
    )
    measured = StackValidity(True, True, True, True, True)
    configurations = (
        PublicationConfiguration(
            "bf16",
            DecodePrecisionProfile.bf16_reference(),
            StackValidity(),
        ),
        PublicationConfiguration(
            "uniform_i8",
            DecodePrecisionProfile.quantized(
                "MXINT8", "MXINT8", "MXINT8", "FP_E3M2"
            ),
            measured,
        ),
        PublicationConfiguration(
            "uniform_i4",
            DecodePrecisionProfile.quantized(
                "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
            ),
            measured,
        ),
        PublicationConfiguration(
            "pareto",
            DecodePrecisionProfile.quantized(
                "MXINT8", "MXINT4", "MXINT4", "FP_E2M3"
            ),
            measured,
        ),
    )
    alternatives = tuple(
        PublicationHardwareAlternative(
            configuration_id=configuration.configuration_id,
            profile_id=configuration.profile.profile_id,
            source_profile_id=configuration.profile.profile_id,
            candidate_id=f"hardware-{configuration.role}",
            record_hash=f"{index + 1:x}" * 64,
            hardware_artifact_sha256="a" * 64,
            tpot_ms=1.0 + index,
            energy_per_token_j=0.5 + index,
            energy_tier="analytic_anchored",
        )
        for index, configuration in enumerate(configurations[1:])
    )
    contract = PublicationContract(
        configurations=configurations,
        hardware_alternatives=alternatives,
        benchmarks=loaded,
        protocol=PublicationProtocol(
            model_name="fixture/model",
            model_revision="4" * 40,
            tokenizer_revision="5" * 40,
            chat_template_sha256="6" * 64,
            thinking_mode="disabled",
            enable_thinking=False,
            greedy=True,
            temperature=0.0,
            token_budgets=(
                ("wikitext2", 128),
                ("ifeval", 128),
                ("gsm8k", 128),
                ("ruler", 128),
            ),
        ),
    )
    assert len(contract.benchmarks) == 3
    assert benchmark_runner.build_benchmark_manifest_main(
        ("--config", str(config_path), "--output", str(output_path))
    ) == 0
    assert output_path.read_bytes() == output_bytes

    sources["openai/gsm8k"].write_text(
        '{"changed":true}\n',
        encoding="utf-8",
    )
    changed = build_publication_benchmark_manifest(
        config,
        load_snapshot=loader,
    )
    assert changed["manifest_hash"] != first["manifest_hash"]


def test_publication_chat_template_is_sealed_only_from_pinned_local_inputs() -> None:
    from decode_dse.software.benchmark_runner import (
        seal_publication_chat_template,
    )

    qwen_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    qwen = seal_publication_chat_template(
        qwen_config,
        config_path=CONFIG_PATH,
    )
    assert qwen["schema_version"] == "decode-chat-template"
    assert qwen["model_revision"] == qwen_config["model_revision"]
    assert qwen["source"]["kind"] == "pinned_local_asset"
    assert hashlib.sha256(qwen["chat_template"].encode()).hexdigest() == qwen[
        "chat_template_sha256"
    ]

    llama_config = json.loads(
        (REPOSITORY / "decode_dse/configs/llama3_1_8b.json").read_text(
            encoding="utf-8"
        )
    )
    calls = []

    class LocalTokenizer:
        @staticmethod
        def get_chat_template():
            return "{{ messages | tojson }}"

    def local_loader(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return LocalTokenizer()

    llama = seal_publication_chat_template(
        llama_config,
        tokenizer_loader=local_loader,
    )
    assert llama["chat_template"] == "{{ messages | tojson }}"
    assert llama["source"]["kind"] == "pinned_local_tokenizer_execution"
    assert calls == [
        (
            llama_config["model_name"],
            {
                "revision": llama_config["tokenizer_revision"],
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]

    def unavailable(*args, **kwargs):
        raise OSError("not cached")

    with pytest.raises(RuntimeError, match="unavailable locally"):
        seal_publication_chat_template(
            llama_config,
            tokenizer_loader=unavailable,
        )


def test_refinement_launch_maps_four_logical_shards_to_two_gpu_waves() -> None:
    from decode_dse.software.refinement_runner import (
        _refinement_launch_waves,
        parse_gpu_pool,
    )

    devices = parse_gpu_pool("0,1")
    waves = _refinement_launch_waves(devices)
    assert waves == (
        ((0, "0"), (1, "1")),
        ((2, "0"), (3, "1")),
    )
    assert tuple(index for wave in waves for index, _ in wave) == tuple(range(4))
    assert all(len(wave) <= 2 for wave in waves)
    assert all(len({device for _, device in wave}) == len(wave) for wave in waves)
    with pytest.raises(ValueError, match="unique GPU identifiers"):
        parse_gpu_pool("0,0")


def test_pipeline_directory_identity_covers_the_complete_tree(
    tmp_path: Path,
) -> None:
    from decode_dse.software import sweep

    root = tmp_path / "output"
    nested = root / "nested"
    nested.mkdir(parents=True)
    first = root / "first.json"
    second = nested / "second.bin"
    first.write_text('{"value":1}\n', encoding="utf-8")
    second.write_bytes(b"payload")

    baseline = sweep._pipeline_output_identity(root)
    assert baseline == sweep._pipeline_output_identity(root)
    assert baseline["entry_count"] == 3
    assert baseline["file_count"] == 2
    assert baseline["directory_count"] == 1
    assert baseline["size_bytes"] == first.stat().st_size + second.stat().st_size

    first.write_text('{"value":2}\n', encoding="utf-8")
    assert sweep._pipeline_output_identity(root) != baseline
    first.write_text('{"value":1}\n', encoding="utf-8")
    assert sweep._pipeline_output_identity(root) == baseline

    second.unlink()
    assert sweep._pipeline_output_identity(root) != baseline
    second.write_bytes(b"payload")
    assert sweep._pipeline_output_identity(root) == baseline

    added = root / "added.txt"
    added.write_text("added\n", encoding="utf-8")
    assert sweep._pipeline_output_identity(root) != baseline
    added.unlink()
    assert sweep._pipeline_output_identity(root) == baseline

    link = root / "link"
    link.symlink_to(first)
    with pytest.raises(ValueError, match="symbolic links"):
        sweep._pipeline_output_identity(root)
    link.unlink()

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="unsupported file type"):
        sweep._pipeline_output_identity(root)


def test_publication_configuration_builder_separates_accuracy_and_hardware(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from decode_dse.hardware import design_space, evaluation
    from decode_dse.software import refinement_runner
    from decode_dse.software.benchmark_runner import (
        build_publication_configuration_manifest,
        load_publication_configuration_manifest,
    )
    from decode_dse.software.refinement_schedule import (
        DecodeRefinementProfile,
        DoomedGateDecision,
        DoomedGatePolicy,
        RefinementSchedule,
        RefinementScheduleEntry,
        write_refinement_schedule,
    )

    base_manifest = _small_manifest()
    sources = {
        "uniform_mxint8": DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT8", "MXINT8", "FP_E3M2"
        ),
        "uniform_mxint4": DecodePrecisionProfile.quantized(
            "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
        ),
        "mxint_kv2": DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT8", "MXINT2", "FP_E3M2"
        ),
        "accuracy_constrained_deployment": DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT4", "MXINT4", "FP_E2M3"
        ),
    }
    manifest_profiles = (
        DecodePrecisionProfile.bf16_reference(),
        *sources.values(),
    )
    manifest = SweepManifest(
        model_name=base_manifest.model_name,
        model_revision=base_manifest.model_revision,
        model_architecture=base_manifest.model_architecture,
        tokenizer_revision=base_manifest.tokenizer_revision,
        quantizer_provenance=base_manifest.quantizer_provenance,
        entries=tuple(
            SweepManifestEntry(
                ordinal=index,
                profile=profile,
                legality=evaluate_profile_legality(profile),
            )
            for index, profile in enumerate(manifest_profiles)
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profiles = (
        DecodeRefinementProfile(sources["uniform_mxint8"], "MXINT8", "MXINT8"),
        DecodeRefinementProfile(sources["uniform_mxint8"], "MXINT8", "MXINT4"),
        DecodeRefinementProfile(sources["uniform_mxint4"], "MXINT4", "MXINT4"),
        DecodeRefinementProfile(sources["mxint_kv2"], "MXINT2", "MXINT2"),
        DecodeRefinementProfile(
            sources["accuracy_constrained_deployment"], "MXINT4", "MXINT4"
        ),
    )
    validity = StackValidity(True, True, True, True, True)
    scheduled_gate = DoomedGateDecision("scheduled", "fixture", 2.0, 1.0)
    skipped_gate = DoomedGateDecision(
        "skipped_doomed",
        "source_accuracy_exceeds_gate",
        2.0,
        3.0,
    )
    gates = (
        scheduled_gate,
        scheduled_gate,
        scheduled_gate,
        skipped_gate,
        scheduled_gate,
    )
    schedule = RefinementSchedule(
        entries=tuple(
            RefinementScheduleEntry(index, profile, gates[index], validity)
            for index, profile in enumerate(profiles)
        ),
        source_profile_ids=tuple(
            sorted(profile.profile_id for profile in sources.values())
        ),
        reference_mean_nll=1.0,
        policy=DoomedGatePolicy(),
    )
    schedule_path = write_refinement_schedule(
        tmp_path / "refinement_schedule.json",
        schedule,
    )
    source_selection_path = write_immutable_json(
        tmp_path / "source_selection.json",
        {
            "schema_version": "decode-refinement-source-selection",
            "manifest_hash": manifest.canonical_hash,
            "schedule_hash": schedule.canonical_hash,
            "source_selection": {
                "source_roles": {
                    role: profile.profile_id for role, profile in sources.items()
                }
            },
        },
    )

    means = (1.1, 1.0, 1.2, 1.3, 1.05)
    sample_bundle_hash = "7" * 64
    result_rows = []
    for entry, mean_nll in zip(schedule.entries, means):
        artifacts = []
        if entry.gate.executable:
            artifact_path = tmp_path / f"result-{entry.ordinal}.json"
            artifact_path.write_text(
                json.dumps({"mean_token_nll": mean_nll}) + "\n",
                encoding="utf-8",
            )
            artifacts.append(
                refinement_runner._artifact_identity(str(artifact_path))
            )
        body = {
            "schema_version": refinement_runner.REFINEMENT_RESULT_SCHEMA,
            "schedule_hash": schedule.canonical_hash,
            "sample_bundle_hash": sample_bundle_hash,
            "ordinal": entry.ordinal,
            "profile_id": entry.profile_id,
            "profile": entry.profile.to_dict(),
            "gate": entry.gate.to_dict(),
            "bank_id": "fixture-bank" if entry.gate.executable else None,
            "attempt": 1 if entry.gate.executable else 0,
            "state": (
                "succeeded"
                if entry.gate.executable
                else entry.gate.execution_state
            ),
            "result": (
                {"mean_token_nll": mean_nll}
                if entry.gate.executable
                else {"gate": entry.gate.to_dict()}
            ),
            "artifacts": artifacts,
            "error_class": None,
            "error_message": None,
            "traceback": None,
            "runtime_seconds": 1.0,
            "completed_at": "2026-08-02T00:00:00Z",
        }
        result_rows.append(
            {**body, "record_hash": refinement_runner._content_hash(body)}
        )
    results_path = tmp_path / "refinement_results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in result_rows
        ),
        encoding="utf-8",
    )
    merge_path = write_immutable_json(
        tmp_path / "refinement_merge.json",
        {
            "schema_version": refinement_runner.REFINEMENT_MERGE_SCHEMA,
            "master_schedule_hash": schedule.canonical_hash,
            "sample_bundle_hash": sample_bundle_hash,
            "profile_count": len(schedule.entries),
            "profile_ids": [entry.profile_id for entry in schedule.entries],
            "terminal": [
                {
                    "profile_id": row["profile_id"],
                    "state": row["state"],
                    "attempt": row["attempt"],
                    "record_hash": row["record_hash"],
                }
                for row in result_rows
            ],
            "result_rows": len(result_rows),
            "merged_results": {
                "path": str(results_path.resolve()),
                "size_bytes": results_path.stat().st_size,
                "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
            },
        },
    )
    hardware_path = tmp_path / "refined_hardware.jsonl"
    hardware_path.write_text("factorized fixture\n", encoding="utf-8")
    chosen_profiles = (profiles[1], profiles[2], profiles[4])
    hardware_rows = tuple(
        {
            "profile_id": profile.profile_id,
            "profile": profile.to_dict(),
            "candidate_id": f"candidate-{index}",
            "record_hash": hashlib.sha256(
                f"{profile.profile_id}:candidate-{index}".encode()
            ).hexdigest(),
            "retention_labels": ["profile_frontier"],
            "deployment_valid": True,
            "validity": validity.to_dict(),
            "metrics": {
                "whole_model": {
                    "rankable": True,
                    "tpot_ms": 2.0 + index,
                    "calibrated_energy": {
                        "total_j": 0.2 + index * 0.1,
                        "energy_tier": "analytic_anchored",
                    },
                }
            },
        }
        for index, profile in enumerate(chosen_profiles)
    )
    monkeypatch.setattr(
        design_space,
        "load_hardware_artifact",
        lambda path: (
            {
                "storage_revision": "factorized-exact",
                "provenance": {
                    "model_revision": manifest.model_revision,
                    "tokenizer_revision": manifest.tokenizer_revision,
                },
            },
            hardware_rows,
        ),
    )
    built = build_publication_configuration_manifest(
        manifest_path=manifest_path,
        schedule_path=schedule_path,
        source_selection_path=source_selection_path,
        merge_receipt_path=merge_path,
        merged_results_path=results_path,
        hardware_artifacts=(hardware_path,),
    )
    assert [item["role"] for item in built["configurations"]] == [
        "bf16",
        "uniform_i8",
        "uniform_i4",
        "pareto",
    ]
    assert built["configurations"][1]["profile_id"] == profiles[1].profile_id
    assert len(built["hardware_alternatives"]) == 3
    assert built["selection_semantics"]["source_hardware_costs_inherited"] is False
    output = write_immutable_json(tmp_path / "configurations.json", built)
    loaded = load_publication_configuration_manifest(output)
    assert loaded["manifest_hash"] == built["manifest_hash"]
    refined_manifest, refined_rows = evaluation._refinement_hardware_inputs(
        manifest,
        schedule_path,
        merge_path,
        results_path,
    )
    assert len(refined_manifest.entries) == len(refined_rows) == len(profiles) - 1
    assert refined_manifest.refinement_schedule_hash == schedule.canonical_hash

    results_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        build_publication_configuration_manifest(
            manifest_path=manifest_path,
            schedule_path=schedule_path,
            source_selection_path=source_selection_path,
            merge_receipt_path=merge_path,
            merged_results_path=results_path,
            hardware_artifacts=(hardware_path,),
        )


def test_full_publication_pipeline_receipts_partition_and_resume(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    from decode_dse.software import sweep
    from decode_dse.hardware import evaluation
    from decode_dse.hardware.lm_head_service import (
        load_bf16_head_service_artifact,
    )

    config = _config()
    config["publication_pipeline"]["resources"]["publication_enabled"] = True
    config["publication_pipeline"]["artifacts"][
        "head_service_calibration"
    ] = "workspace://test-fixtures/synthetic_test_only_head.json"
    config["publication_pipeline"]["artifacts"][
        "packedkv_evidence"
    ] = "workspace://test-fixtures/synthetic_test_only_packedkv.json"
    config["publication_pipeline"]["artifacts"][
        "decode_analysis"
    ] = "workspace://test-fixtures/synthetic_test_only_decode_analysis.json"
    workspace = (tmp_path / "workspace").resolve()
    synthetic_head = workspace / "test-fixtures" / "synthetic_test_only_head.json"
    synthetic_head.parent.mkdir(parents=True)
    synthetic_head.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-test-only-head",
                "production_provenance": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "synthetic_test_only_packedkv.json",
        "synthetic_test_only_decode_analysis.json",
    ):
        (synthetic_head.parent / name).write_text(
            json.dumps(
                {
                    "schema_version": "synthetic-test-only",
                    "production_provenance": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    head_status = load_bf16_head_service_artifact(
        synthetic_head,
        model_name=str(config["model_name"]),
        model_revision=str(config["model_revision"]),
        hidden_size=int(config["model_architecture"]["hidden_size"]),
        vocab_size=int(config["model_architecture"]["vocab_size"]),
        tie_embeddings=bool(
            config["model_architecture"]["tie_word_embeddings"]
        ),
        required_batches=tuple(config["hardware_space"]["BATCH"]),
    )
    assert head_status.passed is False
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_exhaustive_manifest(
        str(config["model_name"]),
        str(config["model_revision"]),
        dict(config["model_architecture"]),
        _small_manifest().quantizer_provenance,
        str(config["tokenizer_revision"]),
    )
    plan = build_run_plan(
        manifest,
        device_labels=("b200",),
        numerical_screen_workers=2,
        hardware_validation_workers=2,
        gpu_baseline=GPUBaselinePlan.from_config(config["gpu_baseline"]),
    )
    monkeypatch.setattr(sweep, "validate_inputs", lambda **kwargs: plan)
    commands = sweep.build_pipeline(
        config=config_path.resolve(),
        output_dir=workspace,
        device_label="b200",
        gpus=("0", "1"),
        plan=plan,
        stack_validity=workspace / "stack_validity.json",
    )
    command_names = tuple(command.name for command in commands)
    assert command_names[:4] == (
        "compiler-trace-artifacts",
        "publication-evidence-gate",
        "publication-benchmark-manifest",
        "publication-chat-template",
    )
    command_by_argv = {command.argv: command for command in commands}
    calls: list[str] = []
    first_gpu_environments: list[tuple[str, str | None]] = []
    interrupted = False
    partition_manifests: dict[Path, bytes] = {}
    hardware_outputs = tuple(
        command.outputs[0]
        for command in commands
        if command.name.startswith("exact-hardware-study")
    )
    final_hardware_outputs = tuple(
        command.outputs[0]
        for command in commands
        if command.name == "refined-hardware-study"
    )
    for command in commands:
        if command.name.startswith("exact-hardware-study"):
            parsed = evaluation._parser().parse_args(command.argv[3:])
            assert parsed.execution_mode == "compiler_trace"
            assert parsed.compiler_trace_artifacts
            assert parsed.request_memory_calibration
    figure_stems = (
        "00_numerical_completion",
        "01_accuracy_by_weight",
        "02_weight_kv_accuracy_landscape",
        "03_vector_precision_sensitivity",
        "04_screening_fidelity",
        "05_hardware_pareto",
        "06_packedkv_causal_ablation",
        "07_model_validation",
        "08_decode_stage_breakdown",
        "09_decode_capacity",
        "10_handoff_regimes",
        "11_multichip_scaling",
        "12_selected_deployment",
    )
    csv_names = tuple(f"{stem}_data.csv" for stem in figure_stems)

    def create_output(path: Path, command_name: str) -> None:
        if command_name == "publication-figures":
            path.mkdir(parents=True, exist_ok=True)
            figure_names = tuple(
                f"{stem}.{format_name}"
                for stem in figure_stems
                for format_name in ("png", "pdf", "svg")
            )
            for name in (*figure_names, *csv_names):
                (path / name).write_bytes(
                    f"synthetic test output: {name}\n".encode("utf-8")
                )
            figure_command = next(
                item for item in commands if item.name == "publication-figures"
            )
            source_flags = (
                "--gpu-baseline-report",
                "--gpu-baseline-receipt",
                "--publication-contract",
                "--publication-report",
                "--final-selection",
                "--refined-hardware-artifact",
            )
            consumed = []
            for flag in source_flags:
                argument_index = figure_command.argv.index(flag)
                source_path = Path(figure_command.argv[argument_index + 1])
                consumed.append(
                    {
                        "flag": flag,
                        "path": str(source_path),
                        "sha256": hashlib.sha256(
                            source_path.read_bytes()
                        ).hexdigest(),
                    }
                )
            identity_header = ",".join(
                item["flag"].removeprefix("--").replace("-", "_")
                + "_sha256"
                for item in consumed
            )
            identity_row = ",".join(item["sha256"] for item in consumed)
            (path / "12_selected_deployment_data.csv").write_text(
                identity_header + "\n" + identity_row + "\n",
                encoding="utf-8",
            )
            (path / "figure_manifest.json").write_text(
                json.dumps(
                    {
                        "evidence_tier": "synthetic_test_only",
                        "figures": list(figure_names),
                        "data_tables": list(csv_names),
                        "consumed_publication_sources": consumed,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            if command_name == "final-publication-selection":
                payload = {
                    "selected": True,
                    "hardware_artifacts": [
                        {
                            "sha256": hashlib.sha256(item.read_bytes()).hexdigest()
                        }
                        for item in final_hardware_outputs
                    ],
                }
            else:
                payload = {"command": command_name, "path": str(path)}
            path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_bytes(
                json.dumps(
                    {"command": command_name, "partition": path.name},
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
        partition_manifests.setdefault(path, manifest_path.read_bytes())
        journal = path / "journal.jsonl"
        payload = journal.read_bytes() if journal.exists() else b""
        if payload and not payload.endswith(b"\n"):
            payload = payload[: payload.rfind(b"\n") + 1]
        if b'"boundary":"terminal"' not in payload:
            payload += b'{"boundary":"terminal"}\n'
        journal.write_bytes(payload)

    def stub_run(argv, *, cwd, env, check):
        nonlocal interrupted
        command = command_by_argv[tuple(argv)]
        calls.append(command.name)
        if command.first_gpu_only:
            first_gpu_environments.append(
                (command.name, env.get("CUDA_VISIBLE_DEVICES"))
            )
        if command.name == "numerical-screen" and not interrupted:
            interrupted = True
            for output in command.outputs:
                output.mkdir(parents=True, exist_ok=True)
                manifest_path = output / "manifest.json"
                manifest_path.write_bytes(
                    json.dumps(
                        {"command": command.name, "partition": output.name},
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                partition_manifests[output] = manifest_path.read_bytes()
                (output / "journal.jsonl").write_bytes(
                    b'{"boundary":"completed"}\n{"truncated"'
                )
            raise KeyboardInterrupt("stubbed interruption")
        for output in command.outputs:
            create_output(output, command.name)
        return None

    monkeypatch.setattr(sweep.subprocess, "run", stub_run)
    arguments = (
        "--config",
        str(config_path),
        "--output-dir",
        str(workspace),
        "--device-label",
        "b200",
        "--gpus",
        "0,1",
    )
    with pytest.raises(KeyboardInterrupt, match="stubbed interruption"):
        sweep.sweep_pipeline_main(arguments)
    contract_bytes = (workspace / "pipeline" / "contract.json").read_bytes()
    assert not any(
        name.startswith("exact-hardware-study") for name in calls
    )

    assert sweep.sweep_pipeline_main(arguments) == 0
    assert (workspace / "pipeline" / "contract.json").read_bytes() == contract_bytes
    assert set(calls) == {command.name for command in commands}
    assert len(
        tuple((workspace / "pipeline" / "completed").glob("*.json"))
    ) == len(commands)
    assert (workspace / "pipeline" / "receipt.json").is_file()
    assert (workspace / "publication" / "final_selection.json").is_file()
    assert (
        workspace / "publication" / "figures" / "figure_manifest.json"
    ).is_file()
    figure_root = workspace / "publication" / "figures"
    assert all(
        (figure_root / f"{stem}.{format_name}").is_file()
        for stem in figure_stems
        for format_name in ("png", "pdf", "svg")
    )
    assert all((figure_root / name).is_file() for name in csv_names)
    figure_manifest = json.loads(
        (figure_root / "figure_manifest.json").read_text(encoding="utf-8")
    )
    consumed_sources = figure_manifest["consumed_publication_sources"]
    assert {item["flag"] for item in consumed_sources} == {
        "--gpu-baseline-report",
        "--gpu-baseline-receipt",
        "--publication-contract",
        "--publication-report",
        "--final-selection",
        "--refined-hardware-artifact",
    }
    selected_table = (
        figure_root / "12_selected_deployment_data.csv"
    ).read_text(encoding="utf-8")
    assert all(item["sha256"] in selected_table for item in consumed_sources)
    final_selection = json.loads(
        (workspace / "publication" / "final_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        item["sha256"] for item in final_selection["hardware_artifacts"]
    ] == [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in final_hardware_outputs
    ]
    for root, expected in partition_manifests.items():
        assert (root / "manifest.json").read_bytes() == expected
        assert (root / "journal.jsonl").read_bytes().endswith(
            b'{"boundary":"terminal"}\n'
        )
    assert first_gpu_environments
    assert all(device == "0" for _, device in first_gpu_environments)

    call_count = len(calls)
    receipt_bytes = (workspace / "pipeline" / "receipt.json").read_bytes()
    assert sweep.sweep_pipeline_main(arguments) == 0
    assert len(calls) == call_count
    assert (workspace / "pipeline" / "contract.json").read_bytes() == contract_bytes
    assert (workspace / "pipeline" / "receipt.json").read_bytes() == receipt_bytes

    directory_output = next(
        output
        for command in commands
        for output in command.outputs
        if not output.suffix
    )
    tracked_file = directory_output / "manifest.json"
    tracked_bytes = tracked_file.read_bytes()
    capsys.readouterr()

    tracked_file.write_bytes(tracked_bytes + b'{"tampered":true}\n')
    assert sweep.sweep_pipeline_main(arguments) == 2
    assert "pipeline output identity changed" in capsys.readouterr().err
    tracked_file.write_bytes(tracked_bytes)
    assert sweep.sweep_pipeline_main(arguments) == 0
    capsys.readouterr()

    tracked_file.unlink()
    assert sweep.sweep_pipeline_main(arguments) == 2
    assert "pipeline output identity changed" in capsys.readouterr().err
    tracked_file.write_bytes(tracked_bytes)
    assert sweep.sweep_pipeline_main(arguments) == 0
    capsys.readouterr()

    unexpected_file = directory_output / "unexpected.json"
    unexpected_file.write_text('{"unexpected":true}\n', encoding="utf-8")
    assert sweep.sweep_pipeline_main(arguments) == 2
    assert "pipeline output identity changed" in capsys.readouterr().err
    unexpected_file.unlink()
    assert sweep.sweep_pipeline_main(arguments) == 0
    assert len(calls) == call_count
    assert (workspace / "pipeline" / "receipt.json").read_bytes() == receipt_bytes


def test_publication_evidence_gate_rejects_missing_physical_head_before_sweep(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from decode_dse.software import sweep
    from decode_dse import simulator_bridge

    config = _config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    workspace = tmp_path / "workspace"
    timing = workspace / "external" / "decode_timing_evidence.json"
    timing.parent.mkdir(parents=True)
    timing.write_text("{}\n", encoding="utf-8")

    class PassingTiming:
        mode = "rtl_serialized"
        passed = True
        evidence_id = "timing-" + "1" * 64

    class TimingLoader:
        @staticmethod
        def load(path):
            assert Path(path) == timing
            return PassingTiming()

    class SimulatorTiming:
        TimingEvidence = TimingLoader

    monkeypatch.setattr(simulator_bridge, "_disagg", lambda: SimulatorTiming())
    with pytest.raises(
        FileNotFoundError,
        match="required publication artifact head_service_calibration is missing",
    ):
        sweep.validate_publication_evidence(
            config=config_path,
            output_dir=workspace,
            report=workspace / "publication_evidence_gate.json",
        )
    assert not (workspace / "preflight").exists()


def test_final_publication_gate_joins_benchmarks_to_exact_hardware(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from decode_dse.software import benchmark_runner, sweep
    from decode_dse.hardware import design_space

    profile = DecodePrecisionProfile.quantized(
        "MXINT8", "MXINT4", "MXINT4", "FP_E2M3"
    )
    profile_id = profile.profile_id
    candidate_id = "candidate-identity"
    configuration_id = "configuration-identity"
    contract_hash = "1" * 64
    hardware_path = tmp_path / "hardware.jsonl"
    hardware_path.write_text("sealed hardware partition\n", encoding="utf-8")
    hardware_sha256 = hashlib.sha256(hardware_path.read_bytes()).hexdigest()
    alternative = benchmark_runner.PublicationHardwareAlternative(
        configuration_id=configuration_id,
        profile_id=profile_id,
        source_profile_id=profile_id,
        candidate_id=candidate_id,
        record_hash="2" * 64,
        hardware_artifact_sha256=hardware_sha256,
        tpot_ms=1.5,
        energy_per_token_j=0.25,
        energy_tier="analytic_anchored",
    )
    contract = SimpleNamespace(
        canonical_hash=contract_hash,
        configurations=(
            SimpleNamespace(
                configuration_id=configuration_id,
                role="pareto",
                profile=profile,
            ),
        ),
        hardware_alternatives=(alternative,),
    )
    monkeypatch.setattr(
        benchmark_runner.PublicationContract,
        "from_dict",
        staticmethod(lambda value: contract),
    )
    contract_path = tmp_path / "contract.json"
    report_path = tmp_path / "publication_report.json"
    output_path = tmp_path / "selection.json"
    write_immutable_json(contract_path, {"contract_hash": contract_hash})
    write_immutable_json(
        report_path,
        {
            "schema_version": "decode-publication-report",
            "contract_hash": contract_hash,
            "selection": {
                "selected": True,
                "accuracy_configuration_ids": [configuration_id],
            },
        },
    )
    monkeypatch.setattr(
        design_space,
        "load_hardware_artifact",
        lambda path: (
            {
                "storage_revision": "factorized-exact",
                "run_id": "hardware-run",
                "factor_evaluation_count": 1,
                "factor_evaluation_sha256": "3" * 64,
                "ordered_membership_map_sha256": "4" * 64,
                "expansion_contract_sha256": "5" * 64,
                "conceptual_result_count": 1,
            },
            (
                {
                    "record_hash": alternative.record_hash,
                    "candidate_id": candidate_id,
                    "profile_id": profile_id,
                    "profile": profile.to_dict(),
                    "deployment_valid": True,
                    "retention_labels": ["profile_frontier"],
                    "metrics": {
                        "whole_model": {
                            "rankable": True,
                            "tpot_ms": 1.5,
                            "calibrated_energy": {
                                "total_j": 0.25,
                                "energy_tier": "analytic_anchored",
                            },
                        }
                    },
                },
            ),
        ),
    )
    assert (
        sweep.publication_gate_main(
            (
                "--contract",
                str(contract_path),
                "--report",
                str(report_path),
                "--hardware-artifact",
                str(hardware_path),
                "--output",
                str(output_path),
            )
        )
        == 0
    )
    selection = load_immutable_json(output_path)
    assert selection["selection"]["candidate_id"] == candidate_id
    assert selection["selection"]["alternative_id"] == alternative.alternative_id
    assert selection["hardware_join"]["candidate_count"] == 1
    assert selection["hardware_artifacts"][0]["sha256"] == hardware_sha256


def test_csv_export_requires_checksum_bound_workspace_provenance(
    tmp_path: Path,
) -> None:
    manifest = _small_manifest()
    body = {
        "schema_version": "decode-sweep-provenance",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": "2" * 64,
        "quantizer_provenance_hash": (manifest.quantizer_provenance.canonical_hash),
        "model": {
            "name": manifest.model_name,
            "revision": manifest.model_revision,
            "tokenizer_revision": manifest.tokenizer_revision,
            "dtype": "bfloat16",
        },
        "datasets": {
            "evaluation": {
                "name": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1",
                "revision": "3" * 40,
                "split": "validation",
            }
        },
    }
    workspace = tmp_path / "workspace"
    export = tmp_path / "publication"

    def write_provenance(value: dict, directory: Path = workspace) -> Path:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "provenance.json"
        path.write_text(
            json.dumps(value | {"content_hash": hashlib.sha256(payload).hexdigest()}),
            encoding="utf-8",
        )
        return path

    source_provenance = write_provenance(body)
    loaded = _load_sweep_provenance(source_provenance, manifest)
    assert loaded["run_plan_hash"] == "2" * 64
    assert RESULTS_PROVENANCE_SCHEMA == "decode-sweep-results-provenance"
    portable_provenance = _copy_sweep_provenance(
        source_provenance,
        export,
        expected_content_hash=loaded["content_hash"],
    )
    assert portable_provenance.name == PORTABLE_WORKSPACE_PROVENANCE
    assert portable_provenance.read_bytes() == source_provenance.read_bytes()
    assert (
        _copy_sweep_provenance(
            source_provenance,
            export,
            expected_content_hash=loaded["content_hash"],
        )
        == portable_provenance
    )
    table = export / "00_numerical_completion_data.csv"
    table.write_bytes(b"profile_id,state\nabc,succeeded\n")
    sidecar = _build_results_provenance(
        sweep_provenance_path=portable_provenance,
        sweep_provenance=loaded,
        manifest=manifest,
        data_tables=(table,),
        created_at_utc="2026-08-01T00:00:01Z",
    )
    assert sidecar["workspace_provenance"] == {
        "path": PORTABLE_WORKSPACE_PROVENANCE,
        "content_hash": loaded["content_hash"],
    }
    assert sidecar["tables"] == [
        {
            "filename": table.name,
            "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            "size_bytes": table.stat().st_size,
        }
    ]
    sidecar_body = dict(sidecar)
    content_hash = sidecar_body.pop("content_hash")
    assert (
        content_hash
        == hashlib.sha256(
            json.dumps(
                sidecar_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    table.write_bytes(table.read_bytes() + b"def,failed\n")
    changed_sidecar = _build_results_provenance(
        sweep_provenance_path=portable_provenance,
        sweep_provenance=loaded,
        manifest=manifest,
        data_tables=(table,),
        created_at_utc="2026-08-01T00:00:01Z",
    )
    assert changed_sidecar["tables"][0]["sha256"] != sidecar["tables"][0]["sha256"]
    assert changed_sidecar["content_hash"] != sidecar["content_hash"]
    results_provenance_path = export / "sweep_results_provenance.json"
    _write_json(results_provenance_path, changed_sidecar)

    portable_bytes = portable_provenance.read_bytes()
    portable_provenance.write_bytes(b"conflicting provenance\n")
    with pytest.raises(FileExistsError, match="differs"):
        _copy_sweep_provenance(
            source_provenance,
            export,
            expected_content_hash=loaded["content_hash"],
        )
    portable_provenance.write_bytes(portable_bytes)
    workspace.rename(tmp_path / "moved-workspace")
    assert not source_provenance.exists()
    relocated = _load_sweep_provenance(portable_provenance, manifest)
    assert relocated["content_hash"] == loaded["content_hash"]
    stored_sidecar = json.loads(results_provenance_path.read_text(encoding="utf-8"))
    stored_body = dict(stored_sidecar)
    stored_hash = stored_body.pop("content_hash")
    assert (
        stored_hash
        == hashlib.sha256(
            json.dumps(
                stored_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    portable_reference = export / stored_sidecar["workspace_provenance"]["path"]
    assert portable_reference == portable_provenance
    assert (
        _load_sweep_provenance(portable_reference, manifest)["content_hash"]
        == stored_sidecar["workspace_provenance"]["content_hash"]
    )
    stored_table = stored_sidecar["tables"][0]
    assert hashlib.sha256(table.read_bytes()).hexdigest() == stored_table["sha256"]
    assert table.stat().st_size == stored_table["size_bytes"]
    relocated_sidecar = _build_results_provenance(
        sweep_provenance_path=portable_provenance,
        sweep_provenance=relocated,
        manifest=manifest,
        data_tables=(table,),
        created_at_utc="2026-08-01T00:00:01Z",
    )
    assert relocated_sidecar == changed_sidecar

    with pytest.raises(ValueError, match="does not bind"):
        _load_sweep_provenance(
            write_provenance(
                body | {"manifest_hash": "0" * 64},
                tmp_path / "wrong-workspace",
            ),
            manifest,
        )


@pytest.mark.parametrize(
    ("capability", "arch_list", "supported"),
    (
        ("sm_100", ("sm_90", "sm_100"), True),
        ("sm_100", ("sm_90", "sm_120"), False),
        ("sm_86", ("sm_80",), True),
        ("sm_80", ("sm_86",), False),
        ("sm_90", ("compute_90",), True),
        ("sm_90", (), False),
        ("", ("sm_90",), False),
    ),
)
def test_architecture_compatibility_follows_the_cuda_rule(
    capability: str,
    arch_list: tuple[str, ...],
    supported: bool,
) -> None:
    """A cubin for sm_XY also runs on sm_XZ for Z >= Y, same major version."""

    assert _architecture_supported(capability, arch_list) is supported


def _write_stack_validity_document(
    path: Path,
    *,
    run_plan_hash: str,
    manifest_hash: str,
    profile_ids: tuple[str, ...],
) -> Path:
    document = {
        "run_plan_hash": run_plan_hash,
        "manifest_hash": manifest_hash,
        "profiles": {
            profile_id: {
                "software_valid": None,
                "compiler_valid": True,
                "emulator_valid": True,
                "rtl_valid": None,
                "dc_calibrated": None,
            }
            for profile_id in profile_ids
        },
    }
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_stack_validity_loader_accepts_the_launch_call_signature(
    tmp_path: Path,
) -> None:
    """The sweep and executor call the loader with a manifest binding."""

    run_plan_hash = "a" * 64
    manifest = SimpleNamespace(canonical_hash="b" * 64)
    profile_ids = ("profile-0", "profile-1")
    path = _write_stack_validity_document(
        tmp_path / "stack_validity.json",
        run_plan_hash=run_plan_hash,
        manifest_hash=manifest.canonical_hash,
        profile_ids=profile_ids,
    )
    validity = load_built_stack_validity(
        path,
        manifest=manifest,
        scope_profile_ids=profile_ids,
        required_stages=("compiler", "emulator"),
        scope_name="hardware-validation",
        run_plan_hash=run_plan_hash,
    )
    assert set(validity) == set(profile_ids)
    for record in validity.values():
        assert record.compiler_valid is True
        assert record.emulator_valid is True


def test_stack_validity_loader_rejects_a_foreign_manifest(tmp_path: Path) -> None:
    """A manifest-hash mismatch means the measurements bind another sweep."""

    run_plan_hash = "a" * 64
    path = _write_stack_validity_document(
        tmp_path / "stack_validity.json",
        run_plan_hash=run_plan_hash,
        manifest_hash="b" * 64,
        profile_ids=("profile-0",),
    )
    with pytest.raises(ValueError, match="different sweep manifest"):
        load_built_stack_validity(
            path,
            manifest=SimpleNamespace(canonical_hash="c" * 64),
            scope_profile_ids=("profile-0",),
            required_stages=("compiler", "emulator"),
            scope_name="hardware-validation",
            run_plan_hash=run_plan_hash,
        )


def _synthetic_stage_report(stage: str, calibration_ids: list[str]) -> dict:
    from decode_dse.software.stack_validity import _canonical_content_hash

    report = {
        "schema": "plena-stack-stage-report",
        "stage": stage,
        "provenance": {
            "started_at_utc": "2026-08-03T10:00:00Z",
            "completed_at_utc": "2026-08-03T10:05:00Z",
            "command": [f"{stage}_stage"],
            "host": "test-host",
        },
        "artifacts": {"primary": "d" * 64},
        "calibration_ids": sorted(calibration_ids),
    }
    report["content_hash"] = _canonical_content_hash(report)
    return report


def _synthetic_calibration_artifact(path: Path, calibration_id: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "plena-decode-emulator-calibration",
                "calibration_id": calibration_id,
                "passed": True,
                "execution_contract": {"timing_mode": "rtl_serialized"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_stack_validity_artifact_builder_satisfies_the_launch_gate(
    tmp_path: Path,
) -> None:
    """The produced artifact passes the loader and the preflight evidence check."""

    from decode_dse.software.preflight import _stack_evidence_preparation
    from decode_dse.software.stack_validity import build_stack_validity_artifact
    from decode_dse.software.sweep_plan import GPUBaselinePlan, build_run_plan

    config = json.loads(
        (REPOSITORY / "decode_dse/configs/llama3_1_8b.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_exhaustive_manifest(
        str(config["model_name"]),
        str(config["model_revision"]),
        dict(config["model_architecture"]),
        _small_manifest().quantizer_provenance,
        str(config["tokenizer_revision"]),
    )
    plan = build_run_plan(
        manifest,
        device_labels=("b200",),
        numerical_screen_workers=2,
        hardware_validation_workers=2,
        gpu_baseline=GPUBaselinePlan.from_config(config["gpu_baseline"]),
    )
    calibration_id = "emucal-" + "e" * 64
    calibration = _synthetic_calibration_artifact(
        tmp_path / "decode_kv128.json", calibration_id
    )
    reports = {}
    for stage in ("compiler", "emulator"):
        report_path = tmp_path / f"{stage}_report.json"
        report_path.write_text(
            json.dumps(_synthetic_stage_report(stage, [calibration_id])) + "\n",
            encoding="utf-8",
        )
        reports[stage] = report_path
    destination = tmp_path / "stack_validity.json"
    document = build_stack_validity_artifact(
        manifest=manifest,
        plan=plan,
        compiler_report_path=reports["compiler"],
        emulator_report_path=reports["emulator"],
        calibration_paths=(calibration,),
        destination=destination,
    )
    assert set(document["profiles"]) == set(plan.hardware_validation_profile_ids)
    validity = load_built_stack_validity(
        destination,
        manifest=manifest,
        scope_profile_ids=plan.hardware_validation_profile_ids,
        required_stages=("compiler", "emulator"),
        scope_name="hardware-validation",
        run_plan_hash=plan.canonical_hash,
    )
    assert len(validity) == len(plan.hardware_validation_profile_ids)
    evidence = _stack_evidence_preparation(
        path=destination, manifest=manifest, plan=plan
    )
    assert evidence["compiler_seconds"] > 0
    assert evidence["emulator_seconds"] > 0
    assert evidence["stage_report_hashes"]["compiler"]
    assert evidence["stage_report_hashes"]["emulator"]


def test_stack_validity_builder_rejects_a_tampered_stage_report(
    tmp_path: Path,
) -> None:
    from decode_dse.software.stack_validity import load_stage_report

    report = _synthetic_stage_report("compiler", ["emucal-" + "e" * 64])
    report["artifacts"]["primary"] = "f" * 64
    path = tmp_path / "compiler_report.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content-hash"):
        load_stage_report(path, stage="compiler")


def test_stack_validity_builder_requires_bound_calibration_ids(
    tmp_path: Path,
) -> None:
    from decode_dse.software.stack_validity import (
        build_stack_validity_document,
    )

    calibration_id = "emucal-" + "e" * 64
    manifest = SimpleNamespace(entries=(), canonical_hash="a" * 64)
    plan = SimpleNamespace(
        hardware_validation_profile_ids=(), canonical_hash="b" * 64
    )
    with pytest.raises(ValueError, match="does not bind calibration ids"):
        build_stack_validity_document(
            manifest=manifest,
            plan=plan,
            compiler_report=_synthetic_stage_report("compiler", []),
            emulator_report=_synthetic_stage_report("emulator", [calibration_id]),
            calibration_ids=(calibration_id,),
        )


def test_stack_validity_builder_refuses_a_forbidden_capability_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from decode_dse.software import stack_validity as module

    profile = SimpleNamespace(profile_id="profile-0")
    entry = SimpleNamespace(profile_id="profile-0", profile=profile)
    manifest = SimpleNamespace(entries=(entry,), canonical_hash="a" * 64)
    plan = SimpleNamespace(
        hardware_validation_profile_ids=("profile-0",), canonical_hash="b" * 64
    )
    floor = StackValidity(compiler_valid=False)
    monkeypatch.setattr(
        module,
        "evaluate_stack_capability",
        lambda _profile: SimpleNamespace(validity_floor=floor),
    )
    calibration_id = "emucal-" + "e" * 64
    with pytest.raises(ValueError, match="structural capability forbids"):
        module.build_stack_validity_document(
            manifest=manifest,
            plan=plan,
            compiler_report=_synthetic_stage_report("compiler", [calibration_id]),
            emulator_report=_synthetic_stage_report("emulator", [calibration_id]),
            calibration_ids=(calibration_id,),
        )


def test_chat_template_sealing_pins_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sealing tool writes the asset and pins both publication fields."""

    import decode_dse.software.seal_chat_template_asset as sealer

    source_config = REPOSITORY / "decode_dse/configs/llama3_1_8b.json"
    config_path = tmp_path / "llama3_1_8b.json"
    config_path.write_text(
        source_config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template = "{{ bos_token }}{% for message in messages %}x{% endfor %}"
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    sealed = {
        "schema_version": "decode-chat-template",
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "enable_thinking": False,
        "chat_template_sha256": template_hash,
        "chat_template": template,
        "source": {"kind": "pinned_local_tokenizer_execution"},
    }
    monkeypatch.setattr(
        sealer,
        "seal_publication_chat_template",
        lambda extraction_config, config_path: sealed,
    )
    asset_path = (
        REPOSITORY
        / "decode_dse/configs/publication_chat_template_llama3_1_8b.json"
    )
    wrote_asset = not asset_path.exists()
    monkeypatch.setattr(
        "sys.argv",
        [
            "seal_chat_template_asset",
            "--config",
            str(config_path),
            "--asset",
            str(tmp_path / "asset.json"),
        ],
    )
    try:
        monkeypatch.setattr(
            sealer,
            "_repo_relative",
            lambda path, repository: "decode_dse/configs/asset.json",
        )
        assert sealer.main() == 0
    finally:
        if wrote_asset and asset_path.exists():
            asset_path.unlink()
    asset = json.loads((tmp_path / "asset.json").read_text(encoding="utf-8"))
    assert set(asset) == {
        "schema_version",
        "model_name",
        "model_revision",
        "tokenizer_revision",
        "enable_thinking",
        "chat_template_sha256",
        "chat_template",
    }
    assert asset["chat_template_sha256"] == template_hash
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert (
        updated["publication"]["chat_template_asset"]
        == "decode_dse/configs/asset.json"
    )
    assert (
        updated["publication"]["chat_template_sha256"] == template_hash
    )
    assert updated["model_architecture"] == config["model_architecture"]
