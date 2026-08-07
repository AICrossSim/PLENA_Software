"""Render checksum-verified publication figures from decode artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from decode_dse.hardware.evaluation import (  # noqa: E402
    load_terminal_numerical_rows,
)
from decode_dse.hardware.design_space import load_hardware_artifact  # noqa: E402
from decode_dse.hardware.packedkv_claims import (  # noqa: E402
    PACKEDKV_MODES,
    PRECISION_ROLES,
    evaluate_packedkv_publication,
    load_packedkv_evidence,
)
from decode_dse.manifest import load_manifest  # noqa: E402
from decode_dse.profiles import (  # noqa: E402
    DECODE_FORMATS,
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    PROFILE_KIND_VECTOR_BF16_CONTROL,
    VECTOR_FORMATS,
    DecodePrecisionProfile,
)
from decode_dse.software.sweep_plan import load_immutable_json  # noqa: E402

INK = "#1A1A1A"
MUTED = "#666666"
GRID = "#D9D9D9"
SURFACE = "#FFFFFF"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#8A8A8A"
FAMILY_COLOURS = {"mxint": BLUE, "mxfp": ORANGE, "bf16": GREY}
MODE_COLOURS = (GREY, SKY, BLUE, GREEN)
FIGURE_SCHEMA = "decode-publication-figures"
FINAL_PUBLICATION_SELECTION_SCHEMA = "decode-final-publication-selection"
RESULTS_PROVENANCE_SCHEMA = "decode-sweep-results-provenance"
ANALYSIS_SCHEMA = "decode-publication-analysis"
PORTABLE_WORKSPACE_PROVENANCE = "workspace_provenance.json"
FIGURE_DPI = 600


@dataclass(frozen=True)
class NumericalPoint:
    ordinal: int
    profile: DecodePrecisionProfile
    mean_nll: float
    runtime_seconds: float

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id


@dataclass(frozen=True)
class HardwarePoint:
    profile: DecodePrecisionProfile
    candidate_id: str
    delta_nll: float
    relative_perplexity_percent: float
    tpot_ms: float
    tps: float
    energy_j: float
    area_mm2: float
    max_runtime_batch: int
    chip_count: int = 1
    tp: int = 1
    kvp: int = 1
    energy_tier: str | None = None
    area_budget_mm2: float | None = None
    retention_labels: tuple[str, ...] = ("legacy_full_row",)
    sample_seed: str | None = None
    sample_limit: int | None = None
    scatter_population_count: int | None = None
    dominated_population_count: int | None = None

    @property
    def tokens_per_j(self) -> float:
        return 1.0 / self.energy_j

    @property
    def edp_j_s(self) -> float:
        return self.energy_j * self.tpot_ms / 1_000.0


def _load_selected_publication_rows(
    *,
    config_path: Path,
    manifest: Any,
    gpu_baseline_report_path: Path,
    gpu_baseline_receipt_path: Path,
    publication_contract_path: Path,
    publication_report_path: Path,
    final_selection_path: Path,
    refined_hardware_artifact_path: Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Validate and join the measured GPU baseline to the final exact row."""

    from decode_dse.hardware.design_space import HARDWARE_STORAGE_REVISION
    from decode_dse.software.benchmark_runner import (
        PUBLICATION_REPORT_SCHEMA,
        PublicationContract,
    )
    from decode_dse.software.gpu_baseline import (
        MEASURED_EVIDENCE_TIER,
        gpu_baseline_energy_evidence,
        gpu_baseline_throughput_evidence,
        validate_gpu_baseline_report,
        validate_gpu_baseline_stage_receipt,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("publication configuration must contain an object")
    if any(
        config.get(name) != getattr(manifest, name)
        for name in ("model_name", "model_revision", "tokenizer_revision")
    ):
        raise ValueError("publication configuration differs from the sweep manifest")
    hardware_space = config.get("hardware_space")
    resource_budget = (
        hardware_space.get("RESOURCE_BUDGET")
        if isinstance(hardware_space, Mapping)
        else None
    )
    if not isinstance(resource_budget, Mapping):
        raise ValueError("publication comparison requires a resource budget")

    baseline_report = load_immutable_json(gpu_baseline_report_path)
    baseline_receipt = load_immutable_json(gpu_baseline_receipt_path)
    baseline_contract, rebuilt_baseline = validate_gpu_baseline_report(
        baseline_report
    )
    validate_gpu_baseline_stage_receipt(rebuilt_baseline, baseline_receipt)
    if (
        baseline_contract.model_name != manifest.model_name
        or baseline_contract.model_revision != manifest.model_revision
        or baseline_contract.tokenizer_revision != manifest.tokenizer_revision
        or baseline_contract.workspace_binding.manifest_hash
        != manifest.canonical_hash
    ):
        raise ValueError("measured GPU baseline differs from the sweep manifest")
    if len(baseline_contract.planned_device_labels) != 1:
        raise ValueError("publication comparison requires one measured GPU label")
    device_label = baseline_contract.planned_device_labels[0]
    gpu_throughput = gpu_baseline_throughput_evidence(
        rebuilt_baseline,
        stage_receipt=baseline_receipt,
        device_label=device_label,
        resource_budget=resource_budget,
    )
    best_by_device = rebuilt_baseline.get("best_measured_by_device")
    best_rows = tuple(
        row
        for row in best_by_device.values()
        if isinstance(best_by_device, Mapping)
        and isinstance(row, Mapping)
        and row.get("device_label") == device_label
    ) if isinstance(best_by_device, Mapping) else ()
    if len(best_rows) != 1:
        raise ValueError("measured GPU baseline has no unique selected row")
    best_gpu = best_rows[0]
    gpu_energy_value = best_gpu.get("energy")
    if not isinstance(gpu_energy_value, Mapping):
        raise ValueError("measured GPU baseline lacks energy status")
    gpu_energy_j: float | str = ""
    gpu_tokens_per_joule: float | str = ""
    gpu_energy_tier = "unavailable"
    gpu_energy_status = str(
        gpu_energy_value.get("unavailable_reason") or "unavailable"
    )
    if gpu_energy_value.get("available") is True:
        gpu_energy = gpu_baseline_energy_evidence(
            rebuilt_baseline,
            stage_receipt=baseline_receipt,
            device_label=device_label,
            resource_budget=resource_budget,
        )
        gpu_energy_j = gpu_energy.energy_per_token_j
        gpu_tokens_per_joule = gpu_energy.tokens_per_joule
        gpu_energy_tier = gpu_energy.evidence_tier
        gpu_energy_status = "measured"

    contract_value = load_immutable_json(publication_contract_path)
    contract_value.pop("content_hash", None)
    contract = PublicationContract.from_dict(contract_value)
    if (
        contract.protocol.model_name != manifest.model_name
        or contract.protocol.model_revision != manifest.model_revision
        or contract.protocol.tokenizer_revision != manifest.tokenizer_revision
    ):
        raise ValueError("publication contract differs from the sweep manifest")
    publication_report = load_immutable_json(publication_report_path)
    if (
        publication_report.get("schema_version") != PUBLICATION_REPORT_SCHEMA
        or publication_report.get("contract_hash") != contract.canonical_hash
    ):
        raise ValueError("publication benchmark report differs from its contract")
    report_selection = publication_report.get("selection")
    passing_ids = (
        report_selection.get("accuracy_configuration_ids")
        if isinstance(report_selection, Mapping)
        else None
    )
    if (
        not isinstance(report_selection, Mapping)
        or report_selection.get("selected") is not True
        or not isinstance(passing_ids, list)
        or not passing_ids
    ):
        raise ValueError("publication accuracy gates selected no configuration")

    final_selection = load_immutable_json(final_selection_path)
    if (
        final_selection.get("schema_version")
        != FINAL_PUBLICATION_SELECTION_SCHEMA
        or final_selection.get("contract_hash") != contract.canonical_hash
        or final_selection.get("contract_sha256")
        != _sha256(publication_contract_path)
        or final_selection.get("benchmark_report_sha256")
        != _sha256(publication_report_path)
        or final_selection.get("accuracy_pass_configuration_ids") != passing_ids
    ):
        raise ValueError("final publication selection provenance differs")
    selected = final_selection.get("selection")
    if not isinstance(selected, Mapping):
        raise ValueError("final publication selection is missing")
    configuration_by_id = {
        item.configuration_id: item for item in contract.configurations
    }
    selected_configuration = configuration_by_id.get(
        str(selected.get("configuration_id", ""))
    )
    if (
        selected_configuration is None
        or selected_configuration.role == "bf16"
        or selected_configuration.configuration_id not in passing_ids
        or selected.get("role") != selected_configuration.role
        or selected.get("profile_id")
        != selected_configuration.profile.profile_id
    ):
        raise ValueError("selected deployment differs from accuracy gates")
    matching_alternatives = tuple(
        alternative
        for alternative in contract.hardware_alternatives
        if alternative.alternative_id == selected.get("alternative_id")
    )
    if len(matching_alternatives) != 1:
        raise ValueError("selected hardware alternative is absent from contract")
    alternative = matching_alternatives[0]
    if (
        alternative.configuration_id
        != selected_configuration.configuration_id
        or alternative.profile_id != selected_configuration.profile.profile_id
        or alternative.candidate_id != selected.get("candidate_id")
        or alternative.record_hash != selected.get("hardware_record_hash")
    ):
        raise ValueError("selected hardware alternative binding differs")

    refined_hardware_sha256 = _sha256(refined_hardware_artifact_path)
    declared_artifacts = final_selection.get("hardware_artifacts")
    if (
        alternative.hardware_artifact_sha256 != refined_hardware_sha256
        or selected.get("hardware_artifact_sha256") != refined_hardware_sha256
        or not isinstance(declared_artifacts, list)
        or refined_hardware_sha256
        not in {
            item.get("sha256")
            for item in declared_artifacts
            if isinstance(item, Mapping)
        }
    ):
        raise ValueError("selected refined hardware artifact identity differs")
    hardware_header, hardware_rows = load_hardware_artifact(
        refined_hardware_artifact_path
    )
    provenance = hardware_header.get("provenance")
    if (
        hardware_header.get("storage_revision") != HARDWARE_STORAGE_REVISION
        or not isinstance(provenance, Mapping)
        or provenance.get("model_revision") != manifest.model_revision
        or provenance.get("tokenizer_revision") != manifest.tokenizer_revision
    ):
        raise ValueError("refined hardware artifact provenance differs")
    matched_rows = tuple(
        row
        for row in hardware_rows
        if row.get("record_hash") == selected.get("hardware_record_hash")
    )
    if len(matched_rows) != 1:
        raise ValueError("selected refined hardware row is missing or duplicated")
    hardware_row = matched_rows[0]
    metrics = hardware_row.get("metrics")
    whole_model = metrics.get("whole_model") if isinstance(metrics, Mapping) else None
    energy = (
        whole_model.get("calibrated_energy")
        if isinstance(whole_model, Mapping)
        else None
    )
    hardware = hardware_row.get("hardware")
    timing_tier_by_mode = {
        "compiler_trace": "compiler_trace_request_calibrated",
        "legacy_aggregate_bandwidth": "stage_calibrated_analytic",
    }
    selected_timing_tier = (
        whole_model.get("publication_timing_tier")
        if isinstance(whole_model, Mapping)
        else None
    )
    if (
        hardware_row.get("deployment_valid") is not True
        or hardware_row.get("profile_id") != alternative.profile_id
        or hardware_row.get("candidate_id") != alternative.candidate_id
        or not isinstance(metrics, Mapping)
        or timing_tier_by_mode.get(str(metrics.get("execution_mode")))
        != selected_timing_tier
        or metrics.get("timing_calibrated") is not True
        or not metrics.get("timing_evidence_id")
        or not isinstance(whole_model, Mapping)
        or whole_model.get("rankable") is not True
        or not isinstance(energy, Mapping)
        or not isinstance(hardware, Mapping)
    ):
        raise ValueError("selected refined hardware evidence is not rankable")
    plena_tps = _finite(whole_model.get("tps"), "selected whole-model TPS")
    plena_tpot_ms = _finite(
        whole_model.get("tpot_ms"), "selected whole-model TPOT"
    )
    plena_energy_j = _finite(energy.get("total_j"), "selected whole-model energy")
    if (
        plena_tps <= 0
        or plena_tpot_ms <= 0
        or plena_energy_j <= 0
        or not math.isclose(
            plena_tpot_ms,
            alternative.tpot_ms,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        or not math.isclose(
            plena_energy_j,
            alternative.energy_per_token_j,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError("selected refined hardware metrics differ from contract")

    identities = {
        "gpu_baseline_report_sha256": _sha256(gpu_baseline_report_path),
        "gpu_baseline_receipt_sha256": _sha256(gpu_baseline_receipt_path),
        "publication_contract_sha256": _sha256(publication_contract_path),
        "publication_report_sha256": _sha256(publication_report_path),
        "final_selection_sha256": _sha256(final_selection_path),
        "refined_hardware_artifact_sha256": refined_hardware_sha256,
    }
    common = {
        "headline_ratio_permitted": False,
        "ratio_block_reason": "throughput_evidence_tiers_differ",
        "throughput_ratio": "",
        "peak_roofline_row": False,
        "peak_roofline_ratio_permitted": False,
        "accuracy_pass_configuration_ids": ";".join(passing_ids),
        **identities,
    }
    rows = (
        {
            "system_role": "selected_plena_deployment",
            "system_name": "PLENA decode with dedicated BF16 output head",
            "configuration_role": selected_configuration.role,
            "configuration_id": selected_configuration.configuration_id,
            "profile_id": alternative.profile_id,
            "candidate_id": alternative.candidate_id,
            "batch_size": int(hardware["BATCH"]),
            "tpot_ms": plena_tpot_ms,
            "tokens_per_second": plena_tps,
            "throughput_evidence_tier": str(selected_timing_tier),
            "energy_per_token_j": plena_energy_j,
            "tokens_per_joule": 1.0 / plena_energy_j,
            "energy_evidence_tier": alternative.energy_tier,
            "energy_status": "rankable",
            "accuracy_gate_passed": True,
            "selected_deployment": True,
            **common,
        },
        {
            "system_role": "measured_gpu_baseline",
            "system_name": gpu_throughput.system_name,
            "configuration_role": "bf16",
            "configuration_id": "",
            "profile_id": "",
            "candidate_id": "",
            "batch_size": gpu_throughput.batch_size,
            "tpot_ms": _finite(
                best_gpu.get("mean_batch_step_ms"),
                "measured GPU batch-step latency",
            ),
            "tokens_per_second": gpu_throughput.tokens_per_second,
            "throughput_evidence_tier": gpu_throughput.evidence_tier,
            "energy_per_token_j": gpu_energy_j,
            "tokens_per_joule": gpu_tokens_per_joule,
            "energy_evidence_tier": gpu_energy_tier,
            "energy_status": gpu_energy_status,
            "accuracy_gate_passed": "reference",
            "selected_deployment": False,
            **common,
        },
    )
    if gpu_throughput.evidence_tier != MEASURED_EVIDENCE_TIER:
        raise ValueError("GPU baseline throughput is not measured evidence")
    return rows, {
        "accuracy_pass_configuration_ids": list(passing_ids),
        "selected_configuration_id": selected_configuration.configuration_id,
        "selected_alternative_id": alternative.alternative_id,
        "headline_ratio_permitted": False,
        "ratio_block_reason": "throughput_evidence_tiers_differ",
        "peak_roofline_ratio_permitted": False,
        "peak_roofline_status": "not_part_of_headline_comparison",
        "sources": identities,
    }


def plot_selected_deployment_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    """Show selected and measured systems without a cross-tier ratio."""

    ordered = tuple(rows)
    if tuple(row.get("system_role") for row in ordered) != (
        "selected_plena_deployment",
        "measured_gpu_baseline",
    ):
        raise ValueError("selected-deployment comparison rows are incomplete")
    labels = ("Selected PLENA", "Measured GPU")
    colours = (BLUE, GREY)
    throughput = [
        _finite(row.get("tokens_per_second"), "comparison throughput")
        for row in ordered
    ]
    if any(value <= 0 for value in throughput):
        raise ValueError("comparison throughput must be positive")
    fig, (throughput_ax, energy_ax) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.1),
        constrained_layout=True,
    )
    x = np.arange(len(ordered))
    bars = throughput_ax.bar(x, throughput, color=colours, width=0.62)
    throughput_ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    throughput_ax.set_xticks(x, labels)
    throughput_ax.set_ylabel("Whole-model throughput (tokens/s)")
    for index, row in enumerate(ordered):
        throughput_ax.text(
            index,
            throughput[index] * 0.04,
            str(row["throughput_evidence_tier"]),
            rotation=90,
            ha="center",
            va="bottom",
            color=SURFACE,
            fontsize=7.2,
        )
    _set_title(
        throughput_ax,
        "Selected deployment and measured GPU baseline",
        f"{model_name} · evidence tiers differ, so no throughput ratio is reported",
    )

    energy_values = []
    energy_positions = []
    energy_colours = []
    for index, row in enumerate(ordered):
        value = row.get("energy_per_token_j")
        if value in (None, ""):
            continue
        numeric = _finite(value, "comparison energy")
        if numeric <= 0:
            raise ValueError("comparison energy must be positive")
        energy_positions.append(index)
        energy_values.append(numeric)
        energy_colours.append(colours[index])
    if energy_values:
        energy_bars = energy_ax.bar(
            energy_positions,
            energy_values,
            color=energy_colours,
            width=0.62,
        )
        energy_ax.bar_label(energy_bars, fmt="%.4g", padding=3, fontsize=8)
    energy_ax.set_xticks(x, labels)
    energy_ax.set_ylabel("Energy per generated token (J)")
    for index, row in enumerate(ordered):
        if row.get("energy_per_token_j") in (None, ""):
            energy_ax.text(
                index,
                0.02,
                "Unavailable",
                transform=energy_ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color=MUTED,
                fontsize=8,
            )
    _set_title(
        energy_ax,
        "Energy evidence",
        "Tiers remain explicit; unavailable GPU board energy is not substituted",
    )
    return _save(
        fig,
        stem="12_selected_deployment",
        output_dir=output_dir,
        formats=formats,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ("Arial", "Liberation Sans", "DejaVu Sans"),
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _set_title(ax: Any, title: str, subtitle: str) -> None:
    # The subtitle is offset in points, matching the units of the title pad, so
    # the gap between the two is fixed. An axes-relative offset scales with axes
    # height and lets the subtitle ride up into the title on tall panels.
    ax.set_title(title, loc="left", color=INK, pad=13)
    ax.annotate(
        subtitle,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, 2.5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=8.3,
        annotation_clip=False,
    )


def _save(
    fig: Any,
    *,
    stem: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    paths = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        options: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
        }
        if suffix == "png":
            options["dpi"] = FIGURE_DPI
        fig.savefig(path, **options)
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    """Write one deterministic figure-data table atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=tuple(fieldnames),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Write a deterministic JSON receipt atomically."""

    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _load_sweep_provenance(path: Path, manifest: Any) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"publication CSVs require the workspace provenance sidecar: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    content_hash = value.pop("content_hash", None)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if content_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError("workspace provenance content hash mismatch")
    if (
        value.get("schema_version") != "decode-sweep-provenance"
        or value.get("manifest_hash") != manifest.canonical_hash
        or value.get("quantizer_provenance_hash")
        != manifest.quantizer_provenance.canonical_hash
        or not value.get("run_plan_hash")
        or not value.get("created_at_utc")
    ):
        raise ValueError("workspace provenance does not bind this sweep manifest")
    return value | {"content_hash": content_hash}


def _copy_sweep_provenance(
    source: Path,
    output_dir: Path,
    *,
    expected_content_hash: str,
) -> Path:
    """Copy validated provenance bytes into a self-contained export."""

    payload = source.read_bytes()
    value = json.loads(payload)
    recorded_hash = value.pop("content_hash", None)
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if (
        recorded_hash != expected_content_hash
        or hashlib.sha256(canonical).hexdigest() != expected_content_hash
    ):
        raise ValueError("workspace provenance changed after validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / PORTABLE_WORKSPACE_PROVENANCE
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != payload
        ):
            raise FileExistsError(
                f"existing portable provenance differs: {destination}"
            )
        return destination

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{PORTABLE_WORKSPACE_PROVENANCE}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != payload
            ):
                raise FileExistsError(
                    f"existing portable provenance differs: {destination}"
                )
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _build_results_provenance(
    *,
    sweep_provenance_path: Path,
    sweep_provenance: Mapping[str, Any],
    manifest: Any,
    data_tables: Sequence[Path],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if sweep_provenance_path.name != PORTABLE_WORKSPACE_PROVENANCE:
        raise ValueError("results must bind the portable workspace provenance")
    portable = _load_sweep_provenance(sweep_provenance_path, manifest)
    if portable["content_hash"] != sweep_provenance["content_hash"]:
        raise ValueError("portable workspace provenance identity differs")
    if any(path.parent != sweep_provenance_path.parent for path in data_tables):
        raise ValueError("publication tables and provenance must share a directory")
    body = {
        "schema_version": RESULTS_PROVENANCE_SCHEMA,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": sweep_provenance["model"],
        "datasets": sweep_provenance["datasets"],
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": sweep_provenance["run_plan_hash"],
        "quantizer_provenance_hash": sweep_provenance[
            "quantizer_provenance_hash"
        ],
        "workspace_provenance": {
            "path": sweep_provenance_path.name,
            "content_hash": sweep_provenance["content_hash"],
        },
        "tables": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in data_tables
        ],
    }
    return body | {
        "content_hash": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    }


def _source_files(path: Path) -> tuple[Path, ...]:
    """Resolve only the immutable result shards consumed by a loader."""

    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(path)
    shard_root = path / "shards" if (path / "shards").is_dir() else path
    files = tuple(sorted(shard_root.glob("*.jsonl")))
    if not files:
        raise ValueError(f"no result shards found under {path}")
    return files


def _load_numerical_points(
    manifest_path: Path,
    result_paths: Sequence[Path],
    *,
    require_complete: bool,
    stage: str | None = None,
) -> tuple[Any, tuple[NumericalPoint, ...], tuple[Mapping[str, Any], ...]]:
    manifest = load_manifest(manifest_path)
    invocation_paths = tuple(
        invocation
        for path in result_paths
        if path.is_dir()
        for invocation in path.rglob("invocation.json")
    )
    if invocation_paths:
        if stage not in {"numerical-screen", "hardware-validation"}:
            raise ValueError("sharded numerical inputs require their stage identity")
        from decode_dse.software.refinement_schedule import (
            _load_sharded_stage_rows,
        )
        from decode_dse.software.sweep_plan import (
            SweepRunPlan,
            load_immutable_json,
        )

        plan_value = load_immutable_json(manifest_path.parent / "run_plan.json")
        plan_value.pop("content_hash", None)
        plan = SweepRunPlan.from_dict(plan_value)
        profile_ids = (
            plan.numerical_screen_profile_ids
            if stage == "numerical-screen"
            else plan.hardware_validation_profile_ids
        )
        rows, _ = _load_sharded_stage_rows(
            result_paths,
            manifest=manifest,
            plan=plan,
            stage=stage,
            profile_ids=profile_ids,
        )
    else:
        rows = load_terminal_numerical_rows(
            result_paths,
            manifest,
            require_complete=require_complete,
        )
    points = []
    for row in rows:
        if row["state"] != "succeeded":
            continue
        result = row.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("successful numerical row has no metric object")
        points.append(
            NumericalPoint(
                ordinal=int(row["ordinal"]),
                profile=DecodePrecisionProfile.from_dict(row["profile"]),
                mean_nll=_finite(result.get("mean_nll"), "mean_nll"),
                runtime_seconds=_finite(
                    row.get("runtime_seconds"),
                    "runtime_seconds",
                ),
            )
        )
    if not points:
        raise ValueError("numerical artifact contains no successful profiles")
    return manifest, tuple(points), rows


def _reference_nll(points: Sequence[NumericalPoint]) -> float:
    references = [
        point.mean_nll
        for point in points
        if point.profile.kind == PROFILE_KIND_BF16_REFERENCE
    ]
    if len(references) != 1:
        raise ValueError("figures require exactly one successful BF16 reference")
    return references[0]


def _family(format_id: str) -> str:
    if format_id.startswith("MXINT"):
        return "mxint"
    if format_id.startswith("E"):
        return "mxfp"
    return "bf16"


def _jitter(count: int, ordinal: int) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    phase = (ordinal % 97) / 97.0
    return np.linspace(-0.19, 0.19, count) + (phase - 0.5) * 0.01


def plot_completion_matrix(
    manifest: Any,
    terminal_rows: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    """Show complete, failed, and pending profiles without dropping failures."""

    row_by_id = {str(row["profile_id"]): row for row in terminal_rows}
    totals = np.zeros((len(DECODE_FORMATS), len(DECODE_FORMATS)), dtype=int)
    succeeded = np.zeros_like(totals)
    format_index = {format_id: index for index, format_id in enumerate(DECODE_FORMATS)}
    states: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    for entry in manifest.entries:
        profile = entry.profile
        row = row_by_id.get(entry.profile_id)
        state = str(row["state"]) if row is not None else "pending"
        states[state] += 1
        if state == "failed":
            failure_classes[str(row.get("error_class") or "unclassified")] += 1
        if profile.kind == PROFILE_KIND_BF16_REFERENCE:
            continue
        row_index = format_index[profile.weight_format]
        column_index = format_index[profile.kv_format]
        totals[row_index, column_index] += 1
        if state == "succeeded":
            succeeded[row_index, column_index] += 1
    if np.any(totals == 0):
        raise ValueError("completion matrix has incomplete manifest coverage")
    completion = np.divide(
        succeeded,
        totals,
        out=np.zeros_like(succeeded, dtype=float),
        where=totals > 0,
    )
    fig, (matrix_ax, status_ax) = plt.subplots(
        1,
        2,
        figsize=(9.4, 5.15),
        gridspec_kw={"width_ratios": (3.5, 1.0)},
        constrained_layout=True,
    )
    image = matrix_ax.imshow(
        completion,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        aspect="equal",
    )
    matrix_ax.set_xticks(np.arange(len(DECODE_FORMATS)))
    matrix_ax.set_xticklabels(DECODE_FORMATS, rotation=35, ha="right")
    matrix_ax.set_yticks(np.arange(len(DECODE_FORMATS)))
    matrix_ax.set_yticklabels(DECODE_FORMATS)
    matrix_ax.set_xlabel("KV-cache format")
    matrix_ax.set_ylabel("Weight format")
    matrix_ax.grid(False)
    for row_index in range(totals.shape[0]):
        for column_index in range(totals.shape[1]):
            fraction = completion[row_index, column_index]
            matrix_ax.text(
                column_index,
                row_index,
                f"{succeeded[row_index, column_index]}/"
                f"{totals[row_index, column_index]}",
                ha="center",
                va="center",
                fontsize=6.8,
                color=SURFACE if fraction >= 0.65 else INK,
            )
    colourbar = fig.colorbar(image, ax=matrix_ax, shrink=0.82, pad=0.03)
    colourbar.set_label("Successful terminal fraction")
    _set_title(
        matrix_ax,
        "Numerical sweep completion",
        f"{model_name} · cells retain all activation and vector settings",
    )

    state_order = ("succeeded", "failed", "pending")
    state_colours = (GREEN, RED, GREY)
    values = [states[state] for state in state_order]
    bars = status_ax.bar(
        np.arange(len(state_order)),
        values,
        color=state_colours,
        width=0.62,
    )
    status_ax.bar_label(bars, padding=3, fontsize=8)
    status_ax.set_xticks(np.arange(len(state_order)))
    status_ax.set_xticklabels(
        ("Succeeded", "Failed", "Pending"),
        rotation=30,
        ha="right",
    )
    status_ax.set_ylabel("Profiles")
    status_ax.set_ylim(0, max(values + [1]) * 1.15)
    failure_note = "No terminal failures"
    if failure_classes:
        failure_note = ", ".join(
            f"{name}: {count}" for name, count in failure_classes.most_common(3)
        )
    _set_title(
        status_ax,
        "Terminal states",
        failure_note,
    )
    return _save(
        fig,
        stem="00_numerical_completion",
        output_dir=output_dir,
        formats=formats,
    )


def plot_accuracy_by_weight(
    points: Sequence[NumericalPoint],
    *,
    reference_nll: float,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    groups = [
        [
            point.mean_nll - reference_nll
            for point in points
            if point.profile.weight_format == format_id
            and point.profile.kind != PROFILE_KIND_BF16_REFERENCE
        ]
        for format_id in DECODE_FORMATS
    ]
    if any(not group for group in groups):
        raise ValueError("weight-format accuracy figure has incomplete coverage")
    fig, ax = plt.subplots(figsize=(7.25, 4.15), constrained_layout=True)
    box = ax.boxplot(
        groups,
        positions=np.arange(len(groups)),
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.3},
        whiskerprops={"color": MUTED, "linewidth": 0.8},
        capprops={"color": MUTED, "linewidth": 0.8},
        boxprops={"edgecolor": INK, "linewidth": 0.7},
    )
    for patch, format_id in zip(box["boxes"], DECODE_FORMATS):
        patch.set_facecolor(FAMILY_COLOURS[_family(format_id)])
        patch.set_alpha(0.55)
    for index, (values, format_id) in enumerate(zip(groups, DECODE_FORMATS)):
        ordered = np.asarray(sorted(values), dtype=float)
        ax.scatter(
            index + _jitter(len(ordered), index),
            ordered,
            s=7,
            alpha=0.12,
            color=FAMILY_COLOURS[_family(format_id)],
            edgecolors="none",
            rasterized=True,
        )
    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set_xticks(np.arange(len(DECODE_FORMATS)))
    ax.set_xticklabels(DECODE_FORMATS, rotation=30, ha="right")
    ax.set_ylabel("Δ NLL from BF16 (nats/token, symlog)")
    ax.set_xlabel("Weight format")
    _set_title(
        ax,
        "Decode accuracy sensitivity across weight formats",
        f"{model_name} · distributions span every activation, KV, and vector setting",
    )
    return _save(
        fig,
        stem="01_accuracy_by_weight",
        output_dir=output_dir,
        formats=formats,
    )


def plot_accuracy_landscape(
    points: Sequence[NumericalPoint],
    *,
    reference_nll: float,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    matrix = np.full((len(DECODE_FORMATS), len(DECODE_FORMATS)), np.nan)
    for row, weight_format in enumerate(DECODE_FORMATS):
        for column, kv_format in enumerate(DECODE_FORMATS):
            values = [
                point.mean_nll - reference_nll
                for point in points
                if point.profile.weight_format == weight_format
                and point.profile.kv_format == kv_format
                and point.profile.kind != PROFILE_KIND_BF16_REFERENCE
            ]
            if values:
                matrix[row, column] = min(values)
    if np.isnan(matrix).any():
        raise ValueError("accuracy landscape has incomplete W/KV coverage")
    magnitude = max(float(np.max(np.abs(matrix))), 0.01)
    norm = SymLogNorm(
        linthresh=0.01,
        linscale=0.8,
        vmin=-magnitude,
        vmax=magnitude,
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    image = ax.imshow(
        matrix,
        cmap="RdYlBu_r",
        norm=norm,
        origin="upper",
        aspect="equal",
    )
    ax.set_xticks(np.arange(len(DECODE_FORMATS)))
    ax.set_xticklabels(DECODE_FORMATS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(DECODE_FORMATS)))
    ax.set_yticklabels(DECODE_FORMATS)
    ax.set_xlabel("KV-cache format")
    ax.set_ylabel("Weight format")
    ax.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            label = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                color=INK if luminance > 0.58 else SURFACE,
            )
    colourbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    colourbar.set_label("Best Δ NLL from BF16 (nats/token)")
    _set_title(
        ax,
        "Weight–KV accuracy landscape",
        f"{model_name} · best observed activation/vector setting in each cell",
    )
    return _save(
        fig,
        stem="02_weight_kv_accuracy_landscape",
        output_dir=output_dir,
        formats=formats,
    )


def plot_vector_sensitivity(
    points: Sequence[NumericalPoint],
    *,
    reference_nll: float,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    control_by_matrix_profile = {
        (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        ): point.mean_nll
        for point in points
        if point.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
    }
    groups = [
        [
            (
                point.mean_nll
                - control_by_matrix_profile[
                    (
                        point.profile.weight_format,
                        point.profile.activation_format,
                        point.profile.kv_format,
                    )
                ]
            )
            for point in points
            if point.profile.vector_format == vector_format
            and point.profile.kind != PROFILE_KIND_BF16_REFERENCE
            and (
                point.profile.weight_format,
                point.profile.activation_format,
                point.profile.kv_format,
            )
            in control_by_matrix_profile
        ]
        for vector_format in VECTOR_FORMATS
    ]
    if any(not group for group in groups):
        raise ValueError("vector-format figure has incomplete coverage")
    fig, ax = plt.subplots(figsize=(7.25, 4.15), constrained_layout=True)
    violin = ax.violinplot(
        groups,
        positions=np.arange(len(groups)),
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        points=120,
    )
    for body, vector_format in zip(violin["bodies"], VECTOR_FORMATS):
        body.set_facecolor(GREY if vector_format == "BF16" else PURPLE)
        body.set_edgecolor(INK)
        body.set_alpha(0.55)
        body.set_linewidth(0.6)
    violin["cmedians"].set_color(INK)
    violin["cmedians"].set_linewidth(1.4)
    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set_xticks(np.arange(len(VECTOR_FORMATS)))
    ax.set_xticklabels(VECTOR_FORMATS, rotation=30, ha="right")
    ax.set_ylabel("Paired Δ NLL vs vector-BF16 (nats/token, symlog)")
    ax.set_xlabel("Vector-datapath format")
    _set_title(
        ax,
        "Paired vector precision attribution",
        f"{model_name} · each point holds W, A, and KV formats fixed",
    )
    return _save(
        fig,
        stem="03_vector_precision_sensitivity",
        output_dir=output_dir,
        formats=formats,
    )


def _relative_perplexity_percent(delta_nll: float) -> float:
    if delta_nll > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.inf
    return math.expm1(delta_nll) * 100.0


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return deterministic one-based average ranks for tied values."""

    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def plot_screening_fidelity(
    screening_points: Sequence[NumericalPoint],
    validation_points: Sequence[NumericalPoint],
    *,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """Compare the two measured fidelities only on identical profile IDs."""

    screening = {
        point.profile_id: point
        for point in screening_points
        if point.profile.kind == PROFILE_KIND_QUANTIZED
    }
    validation = {
        point.profile_id: point
        for point in validation_points
        if point.profile.kind == PROFILE_KIND_QUANTIZED
    }
    common = tuple(sorted(set(screening) & set(validation)))
    if len(common) < 2:
        raise ValueError("fidelity figure requires at least two common profiles")
    screen_nll = np.asarray(
        [screening[profile_id].mean_nll for profile_id in common],
        dtype=float,
    )
    validation_nll = np.asarray(
        [validation[profile_id].mean_nll for profile_id in common],
        dtype=float,
    )
    screen_ranks = _average_ranks(screen_nll)
    validation_ranks = _average_ranks(validation_nll)
    spearman = float(np.corrcoef(screen_ranks, validation_ranks)[0, 1])
    if not math.isfinite(spearman):
        raise ValueError("fidelity rank correlation is non-finite")
    top_k = min(24, len(common))
    screen_top = {
        common[index] for index in np.argsort(screen_nll, kind="mergesort")[:top_k]
    }
    validation_top = {
        common[index] for index in np.argsort(validation_nll, kind="mergesort")[:top_k]
    }
    recall = len(screen_top & validation_top) / top_k
    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    colours = [
        FAMILY_COLOURS[_family(screening[profile_id].profile.weight_format)]
        for profile_id in common
    ]
    ax.scatter(
        screen_ranks,
        validation_ranks,
        c=colours,
        s=20,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        (1, len(common)),
        (1, len(common)),
        color=INK,
        linestyle="--",
        linewidth=0.9,
        label="Equal rank",
    )
    ax.set_xlim(1, len(common))
    ax.set_ylim(1, len(common))
    ax.set_xlabel("Numerical-screen NLL rank (lower is better)")
    ax.set_ylabel("Hardware-validation NLL rank (lower is better)")
    ax.legend(loc="best")
    gate = spearman >= 0.90 and recall >= 0.90
    _set_title(
        ax,
        "Screening fidelity on shared profiles",
        (
            f"{model_name} · Spearman ρ={spearman:.3f}; "
            f"top-{top_k} NLL recall={recall:.1%}; "
            f"gate {'passed' if gate else 'not passed'}"
        ),
    )
    paths = _save(
        fig,
        stem="04_screening_fidelity",
        output_dir=output_dir,
        formats=formats,
    )
    return paths, {
        "common_profiles": len(common),
        "spearman": spearman,
        "top_k": top_k,
        "top_k_nll_recall": recall,
        "passed": gate,
    }


def _load_hardware_points(
    path: Path,
    *,
    profile_by_id: Mapping[str, DecodePrecisionProfile],
    nll_by_id: Mapping[str, float],
    reference_nll: float,
) -> tuple[HardwarePoint, ...]:
    header, rows = load_hardware_artifact(path)
    retention = header.get("retention")
    retention = retention if isinstance(retention, Mapping) else {}
    points = []
    for row in rows:
        if row.get("deployment_valid") is not True:
            continue
        profile_id = str(row["profile_id"])
        profile = profile_by_id.get(profile_id)
        if profile is None or profile_id not in nll_by_id:
            raise ValueError("hardware artifact references an unknown profile")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("deployment-valid row has no hardware metrics")
        whole = metrics.get("whole_model")
        capacity = metrics.get("runtime_capacity_evidence")
        if not isinstance(whole, Mapping) or whole.get("rankable") is not True:
            raise ValueError("deployment-valid row has no rankable system metrics")
        energy = whole.get("calibrated_energy")
        if not isinstance(capacity, Mapping):
            raise ValueError("deployment-valid row lacks energy or capacity")
        if isinstance(energy, Mapping):
            energy_j = _finite(energy.get("total_j"), "energy.total_j")
            energy_tier_raw = energy.get("energy_tier", whole.get("energy_tier"))
        else:
            energy_j = _finite(
                whole.get("energy_per_token_j"),
                "whole_model.energy_per_token_j",
            )
            energy_tier_raw = whole.get("energy_tier")
        energy_tier = (
            str(energy_tier_raw) if energy_tier_raw not in (None, "") else None
        )
        hardware = row.get("hardware")
        if not isinstance(hardware, Mapping):
            raise ValueError("deployment-valid row has no hardware identity")
        resource_budget = metrics.get("resource_budget")
        area_budget = None
        if isinstance(resource_budget, Mapping):
            raw_budget = resource_budget.get(
                "aggregate_area_limit_mm2",
                resource_budget.get("area_budget_mm2"),
            )
            if raw_budget is not None:
                area_budget = _finite(raw_budget, "resource_budget.area_budget_mm2")
        delta_nll = nll_by_id[profile_id] - reference_nll
        points.append(
            HardwarePoint(
                profile=profile,
                candidate_id=str(row["candidate_id"]),
                delta_nll=delta_nll,
                relative_perplexity_percent=_relative_perplexity_percent(delta_nll),
                tpot_ms=_finite(whole.get("tpot_ms"), "whole_model.tpot_ms"),
                tps=_finite(whole.get("tps"), "whole_model.tps"),
                energy_j=energy_j,
                area_mm2=_finite(
                    metrics.get(
                        "system_area_mm2",
                        resource_budget.get("aggregate_area_mm2")
                        if isinstance(resource_budget, Mapping)
                        else metrics.get("area_mm2"),
                    ),
                    "area_mm2",
                ),
                max_runtime_batch=int(capacity["max_runtime_batch"]),
                chip_count=int(hardware.get("CHIP_COUNT", 1)),
                tp=int(hardware.get("TP", 1)),
                kvp=int(hardware.get("KVP", 1)),
                energy_tier=energy_tier,
                area_budget_mm2=area_budget,
                retention_labels=tuple(row.get("retention_labels", ())),
                sample_seed=(
                    str(retention["sample_seed"])
                    if retention.get("sample_seed") is not None
                    else None
                ),
                sample_limit=(
                    int(retention["sample_limit"])
                    if retention.get("sample_limit") is not None
                    else None
                ),
                scatter_population_count=(
                    int(retention["scatter_population_count"])
                    if retention.get("scatter_population_count") is not None
                    else None
                ),
                dominated_population_count=(
                    int(retention["dominated_population_count"])
                    if retention.get("dominated_population_count") is not None
                    else None
                ),
            )
        )
    if not points:
        raise ValueError(
            "hardware figure requires at least one deployment-valid energy-ranked row"
        )
    return tuple(points)


def _pareto_indices(
    points: Sequence[HardwarePoint],
    x: str,
    y: str,
) -> tuple[int, ...]:
    result = []
    for index, point in enumerate(points):
        x_value = float(getattr(point, x))
        y_value = float(getattr(point, y))
        dominated = False
        for other_index, other in enumerate(points):
            if other_index == index:
                continue
            other_x = float(getattr(other, x))
            other_y = float(getattr(other, y))
            if (
                other_x <= x_value
                and other_y <= y_value
                and (other_x < x_value or other_y < y_value)
            ):
                dominated = True
                break
        if not dominated:
            result.append(index)
    return tuple(result)


def plot_hardware_pareto(
    points: Sequence[HardwarePoint],
    *,
    model_name: str,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    finite = [
        point
        for point in points
        if all(
            math.isfinite(value) and value > 0.0
            for value in (point.tpot_ms, point.energy_j, point.area_mm2, point.tps)
        )
    ]
    if not finite:
        raise ValueError("hardware Pareto has no finite positive cost points")
    areas = np.asarray([point.area_mm2 for point in finite], dtype=float)
    if float(np.max(areas)) == float(np.min(areas)):
        marker_sizes = np.full(len(areas), 46.0)
    else:
        marker_sizes = 28.0 + 70.0 * (
            (areas - float(np.min(areas)))
            / (float(np.max(areas)) - float(np.min(areas)))
        )
    chip_counts = tuple(sorted({point.chip_count for point in finite}))
    palette = (BLUE, ORANGE, GREEN, PURPLE, SKY, RED, GREY)
    chip_colours = {
        chip_count: palette[index % len(palette)]
        for index, chip_count in enumerate(chip_counts)
    }
    family_markers = {"mxint": "o", "mxfp": "s", "bf16": "D"}
    declared = [point for point in finite if point.energy_tier]
    frontier_source = declared
    sampled_count = sum(
        bool(
            {"sampled_dominated", "sampled_unrankable"}
            & set(point.retention_labels)
        )
        for point in finite
    )

    fig, (latency_ax, area_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.35),
        constrained_layout=True,
    )
    for point, marker_size in zip(finite, marker_sizes):
        family = _family(point.profile.weight_format)
        sampled = bool(
            {"sampled_dominated", "sampled_unrankable"}
            & set(point.retention_labels)
        )
        latency_ax.scatter(
            point.tpot_ms,
            point.energy_j,
            color=chip_colours[point.chip_count],
            marker=family_markers.get(family, "X"),
            s=float(marker_size),
            alpha=(0.34 if sampled else 0.72) if point.energy_tier else 0.22,
            edgecolors=INK if point.energy_tier else RED,
            linewidths=0.45 if point.energy_tier else 1.0,
            rasterized=True,
        )
        area_ax.scatter(
            point.tps,
            point.area_mm2,
            color=chip_colours[point.chip_count],
            marker=family_markers.get(family, "X"),
            s=float(marker_size),
            alpha=(0.34 if sampled else 0.72) if point.energy_tier else 0.22,
            edgecolors=INK if point.energy_tier else RED,
            linewidths=0.45 if point.energy_tier else 1.0,
            rasterized=True,
        )

    front = []
    if frontier_source:
        front = sorted(
            (
                frontier_source[index]
                for index in _pareto_indices(
                    frontier_source,
                    "tpot_ms",
                    "energy_j",
                )
            ),
            key=lambda point: point.tpot_ms,
        )
        latency_ax.plot(
            [point.tpot_ms for point in front],
            [point.energy_j for point in front],
            color=INK,
            marker="o",
            markerfacecolor=SURFACE,
            markersize=4.0,
            linewidth=1.35,
            label="Latency–energy Pareto front",
            zorder=5,
        )

        callouts: dict[str, tuple[HardwarePoint, list[str]]] = {}
        for label, point in (
            ("Fastest", min(frontier_source, key=lambda item: item.tpot_ms)),
            ("Lowest energy", min(frontier_source, key=lambda item: item.energy_j)),
            ("Best EDP", min(frontier_source, key=lambda item: item.edp_j_s)),
        ):
            entry = callouts.setdefault(point.candidate_id, (point, []))
            entry[1].append(label)
        offsets = ((10, 12), (10, 10), (-68, 12))
        for ordinal, (_, (point, labels)) in enumerate(sorted(callouts.items())):
            latency_ax.annotate(
                " / ".join(labels),
                xy=(point.tpot_ms, point.energy_j),
                xytext=offsets[ordinal % len(offsets)],
                textcoords="offset points",
                fontsize=7.8,
                color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
                zorder=6,
            )

    latency_ax.set_xscale("log")
    latency_ax.set_yscale("log")
    latency_ax.set_xlabel("Whole-model TPOT (ms/token, log)")
    latency_ax.set_ylabel("Energy per generated token (J, log)")
    _set_title(
        latency_ax,
        "Latency–energy co-design frontier",
        (
            f"{model_name} · area sets marker size · "
            f"exact decision/frontier rows + {sampled_count} deterministic "
            "sampled dominated row(s)"
        ),
    )

    # The frontier is the lower-left envelope, so the region below it holds
    # no points; the full-space inset lives at the lower left, and the y
    # floor extends until the inset rectangle is verifiably point-free.
    _INSET_RECT = (0.06, 0.08, 0.40, 0.36)
    full_ax = latency_ax.inset_axes(_INSET_RECT)
    for point, marker_size in zip(finite, marker_sizes):
        full_ax.scatter(
            point.tpot_ms,
            point.energy_j,
            color=chip_colours[point.chip_count],
            marker=family_markers.get(_family(point.profile.weight_format), "X"),
            s=max(8.0, float(marker_size) * 0.25),
            alpha=0.55,
            edgecolors="none",
            rasterized=True,
        )
    full_ax.set_xscale("log")
    full_ax.set_yscale("log")
    full_ax.set_title(
        "Full sampled design space",
        fontsize=7.2,
        loc="left",
        pad=2,
    )
    full_ax.tick_params(labelsize=6.2, length=2)
    full_ax.grid(True, linewidth=0.35)

    if front:
        front_x = [point.tpot_ms for point in front]
        front_y = [point.energy_j for point in front]

        def padded_limits(values: Sequence[float]) -> tuple[float, float]:
            low, high = min(values), max(values)
            if low == high:
                return low / 1.8, high * 1.8
            return low / 1.25, high * 1.25

        x_low, x_high = padded_limits(front_x)
        y_low, y_high = padded_limits(front_y)
        latency_ax.set_xlim(x_low, x_high)
        inset_x0, inset_y0, inset_w, inset_h = _INSET_RECT
        for _ in range(8):
            latency_ax.set_ylim(y_low, y_high)
            to_axes = (
                latency_ax.transData + latency_ax.transAxes.inverted()
            ).transform
            occupied = any(
                inset_x0 <= x <= inset_x0 + inset_w
                and inset_y0 <= y <= inset_y0 + inset_h
                for x, y in to_axes(
                    [
                        (point.tpot_ms, point.energy_j)
                        for point in frontier_source
                    ]
                )
            )
            if not occupied:
                break
            y_low /= 1.6

    budgets = sorted(
        {point.area_budget_mm2 for point in finite if point.area_budget_mm2 is not None}
    )
    if len(budgets) > 1:
        raise ValueError("hardware rows declare inconsistent aggregate area budgets")
    if budgets:
        area_ax.axhline(
            budgets[0],
            color=RED,
            linestyle="--",
            linewidth=1.1,
        )
        area_ax.annotate(
            f"Area budget ({budgets[0]:g} mm²)",
            xy=(0.02, budgets[0]),
            xycoords=("axes fraction", "data"),
            xytext=(0, -11),
            textcoords="offset points",
            fontsize=7.8,
            color=RED,
            va="top",
        )
    area_ax.set_xscale("log")
    area_ax.set_xlabel("Whole-model throughput (tokens/s, log)")
    area_ax.set_ylabel("Aggregate silicon area (mm²)")
    _set_title(
        area_ax,
        "Resource envelope",
        "Only like-for-like aggregate silicon budgets define eligibility",
    )

    chip_handles = [
        Line2D(
            (),
            (),
            marker="o",
            linestyle="none",
            markerfacecolor=chip_colours[chip_count],
            markeredgecolor="none",
            label=f"{chip_count} chip{'s' if chip_count != 1 else ''}",
        )
        for chip_count in chip_counts
    ]
    family_handles = [
        Line2D(
            (),
            (),
            marker=marker,
            linestyle="none",
            markerfacecolor=GREY,
            markeredgecolor=INK,
            label=family.upper(),
        )
        for family, marker in family_markers.items()
        if any(_family(point.profile.weight_format) == family for point in finite)
    ]
    fig.legend(
        handles=[*chip_handles, *family_handles],
        loc="outside lower center",
        ncols=min(6, len(chip_handles) + len(family_handles)),
    )
    return _save(
        fig,
        stem="05_hardware_pareto",
        output_dir=output_dir,
        formats=formats,
    )


def plot_packedkv_ablation(
    evidence_path: Path,
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    evidence = load_packedkv_evidence(evidence_path)
    report = evaluate_packedkv_publication(evidence)
    groups = [group for group in evidence.groups if group.topology.role == "gqa8"]
    groups = sorted(
        groups,
        key=lambda group: PRECISION_ROLES.index(group.precision.role),
    )
    if tuple(group.precision.role for group in groups) != PRECISION_ROLES:
        raise ValueError("PackedKV evidence lacks the three Qwen GQA-8 controls")
    precision_order = [
        f"{group.precision.role.upper()}\n({group.precision.format_id})"
        for group in groups
    ]
    mode_order = tuple(PACKEDKV_MODES)
    x = np.arange(len(precision_order), dtype=float)
    width = 0.19
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.0, 3.9),
        constrained_layout=True,
    )
    fields = (
        ("read_bytes_per_sequence_token", "HBM read bytes/token", True),
        ("feasible_batch", "Maximum feasible batch", False),
        ("capacity_limited_tokens_per_s", "Capacity-limited tokens/s", False),
    )
    for axis, (field, label, normalize) in zip(axes, fields):
        for mode_index, mode in enumerate(mode_order):
            values = []
            for group in groups:
                measurement = group.by_mode()[mode]
                value = float(getattr(measurement, field))
                if normalize:
                    baseline = float(getattr(group.by_mode()[mode_order[0]], field))
                    value /= baseline
                values.append(value)
            axis.bar(
                x + (mode_index - 1.5) * width,
                values,
                width,
                color=MODE_COLOURS[mode_index],
                label=mode.replace("_", " ").title(),
            )
        axis.set_xticks(x)
        axis.set_xticklabels(precision_order, rotation=25, ha="right")
        axis.set_ylabel("Normalized HBM reads (baseline = 1.0)" if normalize else label)
        axis.set_xlabel("Precision role")
    _set_title(
        axes[0],
        "PackedKV physical traffic",
        "Qwen GQA-8 · fixed precision, geometry, clock, and HBM",
    )
    _set_title(
        axes[1],
        "PackedKV capacity",
        "Maximum batch under the identical physical-capacity model",
    )
    _set_title(
        axes[2],
        "PackedKV serving throughput",
        f"Gate status: {'passed' if report.passed else 'not passed'}",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(mode_order),
    )
    return _save(
        fig,
        stem="06_packedkv_causal_ablation",
        output_dir=output_dir,
        formats=formats,
    )


def _model_validation_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Aggregate validation observations without combining evidence tiers."""

    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for row in rows:
        model = str(row.get("model", "")).strip()
        component = str(row.get("component", "")).strip().lower()
        tier = str(row.get("evidence_tier", "")).strip()
        if not model or component not in {"compute", "memory", "area", "power"}:
            raise ValueError("model-validation rows require a model and known component")
        if not tier:
            raise ValueError("model-validation rows require an evidence tier")
        error = abs(_finite(row.get("relative_error_percent"), "relative error"))
        elapsed = _finite(row.get("evaluation_seconds"), "evaluation seconds")
        if elapsed < 0:
            raise ValueError("evaluation seconds must be non-negative")
        grouped.setdefault((model, component, tier), []).append((error, elapsed))
    if not grouped:
        raise ValueError("model validation requires at least one observation")
    result = []
    for (model, component, tier), observations in sorted(grouped.items()):
        errors = np.asarray([value[0] for value in observations], dtype=float)
        elapsed = np.asarray([value[1] for value in observations], dtype=float)
        result.append(
            {
                "model": model,
                "component": component,
                "evidence_tier": tier,
                "sample_count": len(observations),
                "median_absolute_error_percent": float(np.median(errors)),
                "p95_absolute_error_percent": float(np.percentile(errors, 95)),
                "average_evaluation_seconds": float(np.mean(elapsed)),
            }
        )
    return tuple(result)


def plot_model_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    summary = _model_validation_table(rows)
    labels = [f"{row['model']}\n{str(row['component']).title()}" for row in summary]
    # The evidence tier rides in the tick label so it never crosses a bar.
    tier_labels = [
        f"{label}\n({str(row['evidence_tier']).replace('_', ' ')})"
        for label, row in zip(labels, summary)
    ]
    x = np.arange(len(summary), dtype=float)
    median = [float(row["median_absolute_error_percent"]) for row in summary]
    p95 = [float(row["p95_absolute_error_percent"]) for row in summary]
    elapsed = [float(row["average_evaluation_seconds"]) for row in summary]
    width = 0.36
    fig, (error_ax, time_ax) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (2.1, 1.0)},
    )
    error_ax.bar(x - width / 2, median, width, color=BLUE, label="Median")
    error_ax.bar(x + width / 2, p95, width, color=ORANGE, label="P95")
    error_ax.set_xticks(x)
    error_ax.set_xticklabels(tier_labels, rotation=25, ha="right", fontsize=7.6)
    error_ax.set_ylabel("Absolute error against reference (%)")
    error_ax.legend(loc="upper left")
    _set_title(
        error_ax,
        "Analytic-model validation",
        "Errors remain separated by model, component, and evidence tier",
    )
    time_ax.barh(np.arange(len(summary)), elapsed, color=GREEN)
    time_ax.set_yticks(np.arange(len(summary)))
    time_ax.set_yticklabels(labels)
    time_ax.invert_yaxis()
    time_ax.set_xlabel("Average evaluation time (s)")
    _set_title(
        time_ax,
        "Evaluation cost",
        "Mean wall time per validation observation",
    )
    return _save(
        fig,
        stem="07_model_validation",
        output_dir=output_dir,
        formats=formats,
    )


def _stage_breakdown_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    components = ("matrix_cycles", "vector_cycles", "scalar_cycles", "control_cycles")
    normalized: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        stage = str(row.get("stage", "")).strip()
        variant = str(row.get("variant", "")).strip().lower()
        if not stage or variant not in {"baseline", "enhanced"}:
            raise ValueError("stage rows require a stage and baseline/enhanced variant")
        key = (stage, variant)
        if key in normalized:
            raise ValueError("stage breakdown contains duplicate stage variants")
        item: dict[str, Any] = {"stage": stage, "variant": variant}
        for component in components:
            value = _finite(row.get(component), component)
            if value < 0:
                raise ValueError("cycle counts must be non-negative")
            item[component] = value
        item["total_cycles"] = sum(float(item[name]) for name in components)
        normalized[key] = item
    # Stages render in decode pipeline order; any stage outside the canonical
    # pipeline keeps its first-seen position after the known ones.
    pipeline_order = (
        "RMSNorm",
        "QKV projection",
        "Flash attention",
        "Residual",
        "FFN",
        "LM head",
    )
    seen = tuple(dict.fromkeys(stage for stage, _ in normalized))
    stages = tuple(
        sorted(
            seen,
            key=lambda stage: (
                pipeline_order.index(stage)
                if stage in pipeline_order
                else len(pipeline_order) + seen.index(stage)
            ),
        )
    )
    if not stages:
        raise ValueError("stage breakdown requires at least one stage")
    for stage in stages:
        if (stage, "baseline") not in normalized or (stage, "enhanced") not in normalized:
            raise ValueError("every stage requires baseline and enhanced observations")
        baseline = float(normalized[(stage, "baseline")]["total_cycles"])
        if baseline <= 0:
            raise ValueError("baseline stage totals must be positive")
        for variant in ("baseline", "enhanced"):
            total = float(normalized[(stage, variant)]["total_cycles"])
            normalized[(stage, variant)]["delta_from_baseline_percent"] = (
                100.0 * (total - baseline) / baseline
            )
    return tuple(
        normalized[(stage, variant)]
        for stage in stages
        for variant in ("baseline", "enhanced")
    )


def plot_stage_breakdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    table = _stage_breakdown_table(rows)
    by_key = {(str(row["stage"]), str(row["variant"])): row for row in table}
    stages = tuple(dict.fromkeys(str(row["stage"]) for row in table))
    x = np.arange(len(stages), dtype=float)
    width = 0.36
    component_style = (
        ("matrix_cycles", "Matrix", BLUE),
        ("vector_cycles", "Vector", ORANGE),
        ("scalar_cycles", "Scalar", GREEN),
        ("control_cycles", "Control", PURPLE),
    )
    fig, ax = plt.subplots(figsize=(max(8.5, 1.15 * len(stages)), 4.5), constrained_layout=True)
    for variant_index, variant in enumerate(("baseline", "enhanced")):
        bottom = np.zeros(len(stages), dtype=float)
        offset = (-0.5 if variant == "baseline" else 0.5) * width
        for component, label, colour in component_style:
            values = np.asarray(
                [float(by_key[(stage, variant)][component]) for stage in stages],
                dtype=float,
            )
            ax.bar(
                x + offset,
                values,
                width,
                bottom=bottom,
                color=colour,
                alpha=0.48 if variant == "baseline" else 0.92,
                label=label if variant == "enhanced" else None,
            )
            bottom += values
        if variant == "enhanced":
            for index, stage in enumerate(stages):
                delta = float(by_key[(stage, variant)]["delta_from_baseline_percent"])
                ax.text(
                    x[index] + offset,
                    bottom[index],
                    f"{delta:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    color=INK,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=25, ha="right")
    ax.set_ylabel("Cycles")
    ax.set_xlabel("Decode stage (left bar baseline, right bar enhanced)")
    fig.legend(loc="outside upper right", ncols=4)
    _set_title(
        ax,
        "Decode-stage cycle composition",
        "Opaque bars are enhanced RTL; percentage labels report total-cycle change",
    )
    return _save(
        fig,
        stem="08_decode_stage_breakdown",
        output_dir=output_dir,
        formats=formats,
    )


def _capacity_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalized = []
    identities: set[tuple[str, int]] = set()
    for row in rows:
        kv_format = str(row.get("kv_format", "")).strip()
        context = int(row.get("context_tokens", 0))
        if not kv_format or context <= 0:
            raise ValueError("capacity rows require a KV format and positive context")
        identity = (kv_format, context)
        if identity in identities:
            raise ValueError("capacity rows contain duplicate format/context points")
        identities.add(identity)
        item = {
            "kv_format": kv_format,
            "context_tokens": context,
            "kv_bytes_per_token": _finite(row.get("kv_bytes_per_token"), "KV bytes/token"),
            "feasible_batch": int(row.get("feasible_batch", 0)),
            "tpot_ms": _finite(row.get("tpot_ms"), "TPOT"),
        }
        if any(float(item[name]) <= 0 for name in ("kv_bytes_per_token", "feasible_batch", "tpot_ms")):
            raise ValueError("capacity metrics must be positive")
        normalized.append(item)
    if not normalized:
        raise ValueError("capacity figure requires at least one row")
    return tuple(sorted(normalized, key=lambda item: (str(item["kv_format"]), int(item["context_tokens"]))))


def plot_decode_capacity(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    table = _capacity_table(rows)
    formats_present = tuple(sorted({str(row["kv_format"]) for row in table}))
    fig, (latency_ax, batch_ax, chain_ax) = plt.subplots(
        1,
        3,
        figsize=(12.6, 4.0),
        constrained_layout=True,
    )
    palette = (BLUE, ORANGE, GREEN, PURPLE, SKY, RED, GREY)
    for index, kv_format in enumerate(formats_present):
        subset = [row for row in table if row["kv_format"] == kv_format]
        current = palette[index % len(palette)]
        contexts = [int(row["context_tokens"]) for row in subset]
        latency_ax.plot(contexts, [float(row["tpot_ms"]) for row in subset], marker="o", color=current, label=kv_format)
        batch_ax.plot(contexts, [int(row["feasible_batch"]) for row in subset], marker="o", color=current, label=kv_format)
        chain_ax.plot(
            [float(row["kv_bytes_per_token"]) for row in subset],
            [int(row["feasible_batch"]) for row in subset],
            marker="o",
            color=current,
            label=kv_format,
        )
    for axis in (latency_ax, batch_ax):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Context length (tokens, log₂)")
    latency_ax.set_yscale("log")
    latency_ax.set_ylabel("TPOT (ms/token, log)")
    _set_title(latency_ax, "Decode capacity wall", "Latency rises as resident KV traffic grows")
    batch_ax.set_yscale("log", base=2)
    batch_ax.set_ylabel("Maximum feasible batch (log₂)")
    _set_title(batch_ax, "Capacity-enabled batch", "Physical capacity is enforced before throughput ranking")
    chain_ax.set_xscale("log", base=2)
    chain_ax.set_yscale("log", base=2)
    chain_ax.set_xlabel("KV bytes per sequence token (log₂)")
    chain_ax.set_ylabel("Maximum feasible batch (log₂)")
    _set_title(chain_ax, "PackedKV causal chain", "KV bytes/token ↓ → feasible batch ↑")
    handles, labels = latency_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=max(1, len(labels)))
    return _save(
        fig,
        stem="09_decode_capacity",
        output_dir=output_dir,
        formats=formats,
    )


def _handoff_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    segments = ("prefill_s", "transfer_s", "admission_s", "wait_s", "host_spill_s")
    normalized = []
    regimes: set[str] = set()
    for row in rows:
        regime = str(row.get("regime", "")).strip()
        if not regime or regime in regimes:
            raise ValueError("handoff rows require unique named regimes")
        regimes.add(regime)
        item: dict[str, Any] = {"regime": regime}
        for name in segments:
            value = _finite(row.get(name, 0.0), name)
            if value < 0:
                raise ValueError("handoff timeline durations must be non-negative")
            item[name] = value
        for name in ("ttft_s", "energy_j", "prefill_utilization", "prefill_decode_ratio"):
            value = _finite(row.get(name), name)
            if value < 0:
                raise ValueError("handoff metrics must be non-negative")
            item[name] = value
        item["prompt_tokens"] = int(row.get("prompt_tokens", 0))
        item["generation_tokens"] = int(row.get("generation_tokens", 0))
        item["precision"] = str(row.get("precision", ""))
        normalized.append(item)
    if not normalized:
        raise ValueError("handoff figure requires at least one row")
    return tuple(normalized)


def plot_handoff_regimes(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    table = _handoff_table(rows)
    regimes = [str(row["regime"]).replace("_", " ").title() for row in table]
    y = np.arange(len(table), dtype=float)
    fig, (timeline_ax, ttft_ax, energy_ax, balance_ax) = plt.subplots(
        1,
        4,
        figsize=(13.6, 4.1),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.9, 0.8, 0.8, 1.0)},
    )
    segment_style = (
        ("prefill_s", "Prefill", BLUE),
        ("transfer_s", "KV transfer", ORANGE),
        ("admission_s", "Admission", GREEN),
        ("wait_s", "Decode wait", RED),
        ("host_spill_s", "Host spill", PURPLE),
    )
    left = np.zeros(len(table), dtype=float)
    for name, label, colour in segment_style:
        values = np.asarray([float(row[name]) for row in table], dtype=float)
        timeline_ax.barh(y, values, left=left, color=colour, label=label)
        left += values
    timeline_ax.set_yticks(y)
    timeline_ax.set_yticklabels(regimes)
    timeline_ax.invert_yaxis()
    timeline_ax.set_xlabel("Critical-path time (s)")
    # The first (shortest) schedule leaves the upper right of the panel empty.
    timeline_ax.legend(loc="upper right", ncols=1, fontsize=7.2)
    _set_title(timeline_ax, "Prefill→decode schedules", "Timeline exposes stalls and PCIe host-spill penalties")

    # TTFT and energy carry different units, so each keeps its own panel and
    # single axis rather than sharing a dual-scale chart.
    x = np.arange(len(table), dtype=float)
    ttft_ax.bar(x, [float(row["ttft_s"]) for row in table], 0.62, color=BLUE)
    ttft_ax.set_xticks(x)
    ttft_ax.set_xticklabels(regimes, rotation=30, ha="right")
    ttft_ax.set_ylabel("TTFT (s)")
    _set_title(ttft_ax, "Request TTFT", "Critical-path time to first token")
    energy_ax.bar(
        x, [float(row["energy_j"]) for row in table], 0.62, color=ORANGE
    )
    energy_ax.set_xticks(x)
    energy_ax.set_xticklabels(regimes, rotation=30, ha="right")
    energy_ax.set_ylabel("Request energy (J)")
    _set_title(energy_ax, "Request energy", "Includes spill write and read")

    balance_ax.scatter(
        [float(row["prefill_decode_ratio"]) for row in table],
        [100.0 * float(row["prefill_utilization"]) for row in table],
        c=[
            (BLUE, ORANGE, GREEN, PURPLE, SKY, RED, GREY)[index % 7]
            for index in range(len(table))
        ],
        s=52,
    )
    label_offsets = ((6, 6), (6, -11), (-6, 6))
    label_alignment = ("left", "left", "right")
    for index, label in enumerate(regimes):
        balance_ax.annotate(
            label,
            xy=(float(table[index]["prefill_decode_ratio"]), 100.0 * float(table[index]["prefill_utilization"])),
            xytext=label_offsets[index % len(label_offsets)],
            textcoords="offset points",
            ha=label_alignment[index % len(label_alignment)],
            fontsize=7.0,
        )
    balance_ax.set_xlabel("Balanced prefill : decode chip ratio")
    balance_ax.set_ylabel("Prefill utilisation (%)")
    balance_ax.set_ylim(0, 105)
    _set_title(balance_ax, "Pipeline balance", "Ratio, utilisation, prompt, generation, and precision are recorded")
    return _save(
        fig,
        stem="10_handoff_regimes",
        output_dir=output_dir,
        formats=formats,
    )


def _multichip_table(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalized = []
    identities: set[tuple[int, int, int]] = set()
    for row in rows:
        chips = int(row.get("chip_count", 0))
        tp = int(row.get("tp", 0))
        kvp = int(row.get("kvp", 0))
        if chips <= 0 or tp <= 0 or kvp <= 0 or chips != tp * kvp:
            raise ValueError("multi-chip rows require CHIP_COUNT = TP × KVP")
        identity = (chips, tp, kvp)
        if identity in identities:
            raise ValueError("multi-chip rows contain duplicate topologies")
        identities.add(identity)
        tps = _finite(row.get("tps"), "TPS")
        energy_j = _finite(row.get("energy_per_token_j"), "energy/token")
        if tps <= 0 or energy_j <= 0:
            raise ValueError("multi-chip throughput and energy must be positive")
        tier_raw = row.get("energy_tier")
        normalized.append(
            {
                "chip_count": chips,
                "tp": tp,
                "kvp": kvp,
                "parallelism": "TP" if kvp == 1 else ("KVP" if tp == 1 else "TP+KVP"),
                "tps": tps,
                "energy_per_token_j": energy_j,
                "tokens_per_joule": 1.0 / energy_j,
                "energy_tier": str(tier_raw) if tier_raw not in (None, "") else "",
            }
        )
    if not normalized:
        raise ValueError("multi-chip scaling requires at least one row")
    return tuple(sorted(normalized, key=lambda item: (int(item["chip_count"]), int(item["tp"]), int(item["kvp"]))))


def plot_multichip_scaling(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    table = _multichip_table(rows)
    fig, (throughput_ax, efficiency_ax) = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    styles = {
        "TP": (BLUE, "o"),
        "KVP": (ORANGE, "s"),
        "TP+KVP": (GREEN, "D"),
    }
    for mode, (colour, marker) in styles.items():
        subset = [row for row in table if row["parallelism"] == mode]
        if not subset:
            continue
        x = [int(row["chip_count"]) for row in subset]
        throughput_ax.plot(x, [float(row["tps"]) for row in subset], color=colour, marker=marker, label=mode)
        efficiency_ax.plot(x, [float(row["tokens_per_joule"]) for row in subset], color=colour, marker=marker, label=mode)
        for axis, field in ((throughput_ax, "tps"), (efficiency_ax, "tokens_per_joule")):
            for row in subset:
                if not row["energy_tier"]:
                    axis.scatter(int(row["chip_count"]), float(row[field]), facecolors="none", edgecolors=RED, marker=marker, s=70, linewidths=1.1)
    for axis in (throughput_ax, efficiency_ax):
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({int(row["chip_count"]) for row in table}))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Decode chips")
    throughput_ax.set_ylabel("Throughput (tokens/s)")
    _set_title(throughput_ax, "Multi-chip throughput", "TP and KVP are explicit rather than an ideal chip-count divider")
    efficiency_ax.set_ylabel("Energy efficiency (tokens/J)")
    _set_title(efficiency_ax, "Multi-chip energy efficiency", "Tier-undeclared points are outlined in red")
    handles, labels = throughput_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=max(1, len(labels)))
    return _save(
        fig,
        stem="11_multichip_scaling",
        output_dir=output_dir,
        formats=formats,
    )


def _load_decode_analysis(path: Path) -> Mapping[str, Any]:
    """Load the compact artifact that drives decode-specific figures."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("decode analysis must be a JSON object")
    sections = {
        "model_validation",
        "stage_breakdown",
        "capacity",
        "handoff",
        "multichip",
    }
    allowed = {"schema_version", "model_name", *sections}
    if set(value) - allowed:
        raise ValueError("decode analysis contains unknown fields")
    if value.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError("decode analysis schema is unsupported")
    model_name = str(value.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("decode analysis requires a model name")
    normalized: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "model_name": model_name,
    }
    for section in sections:
        rows = value.get(section, [])
        if not isinstance(rows, list) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise TypeError(f"decode analysis section {section!r} must be an object list")
        normalized[section] = tuple(dict(row) for row in rows)
    return normalized


def _numerical_table(
    manifest: Any,
    terminal_rows: Sequence[Mapping[str, Any]],
    *,
    reference_nll: float,
) -> tuple[dict[str, Any], ...]:
    row_by_id = {str(row["profile_id"]): row for row in terminal_rows}
    result = []
    for entry in manifest.entries:
        profile = entry.profile
        row = row_by_id.get(entry.profile_id)
        metrics = row.get("result") if row is not None else None
        mean_nll = (
            _finite(metrics.get("mean_nll"), "mean_nll")
            if isinstance(metrics, Mapping) and row["state"] == "succeeded"
            else None
        )
        result.append(
            {
                "ordinal": entry.ordinal,
                "profile_id": entry.profile_id,
                "kind": profile.kind,
                "weight_format": profile.weight_format,
                "activation_format": profile.activation_format,
                "key_format": profile.key_format,
                "value_format": profile.value_format,
                "vector_format": profile.vector_format,
                "state": str(row["state"]) if row is not None else "pending",
                "attempt": int(row["attempt"]) if row is not None else "",
                "runtime_seconds": (
                    float(row["runtime_seconds"]) if row is not None else ""
                ),
                "mean_nll": mean_nll if mean_nll is not None else "",
                "delta_nll_from_bf16": (
                    mean_nll - reference_nll if mean_nll is not None else ""
                ),
                "error_class": (
                    str(row.get("error_class") or "") if row is not None else ""
                ),
                "error_message": (
                    str(row.get("error_message") or "") if row is not None else ""
                ),
                "validity_json": (
                    json.dumps(
                        row.get("validity"),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if row is not None
                    else ""
                ),
                "record_hash": (
                    str(row.get("record_hash") or "") if row is not None else ""
                ),
            }
        )
    return tuple(result)


def _landscape_table(
    points: Sequence[NumericalPoint],
    *,
    reference_nll: float,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for weight_format in DECODE_FORMATS:
        for kv_format in DECODE_FORMATS:
            candidates = [
                point
                for point in points
                if point.profile.weight_format == weight_format
                and point.profile.kv_format == kv_format
                and point.profile.kind != PROFILE_KIND_BF16_REFERENCE
            ]
            if not candidates:
                raise ValueError("landscape table has incomplete format coverage")
            best = min(candidates, key=lambda point: (point.mean_nll, point.profile_id))
            rows.append(
                {
                    "weight_format": weight_format,
                    "kv_format": kv_format,
                    "best_profile_id": best.profile_id,
                    "best_activation_format": best.profile.activation_format,
                    "best_vector_format": best.profile.vector_format,
                    "best_mean_nll": best.mean_nll,
                    "best_delta_nll_from_bf16": best.mean_nll - reference_nll,
                }
            )
    return tuple(rows)


def _vector_table(
    points: Sequence[NumericalPoint],
) -> tuple[dict[str, Any], ...]:
    controls = {
        (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        ): point
        for point in points
        if point.profile.kind == PROFILE_KIND_VECTOR_BF16_CONTROL
    }
    rows = []
    for point in points:
        if point.profile.kind == PROFILE_KIND_BF16_REFERENCE:
            continue
        key = (
            point.profile.weight_format,
            point.profile.activation_format,
            point.profile.kv_format,
        )
        control = controls.get(key)
        if control is None:
            continue
        rows.append(
            {
                "profile_id": point.profile_id,
                "control_profile_id": control.profile_id,
                "weight_format": key[0],
                "activation_format": key[1],
                "kv_format": key[2],
                "vector_format": point.profile.vector_format,
                "mean_nll": point.mean_nll,
                "control_mean_nll": control.mean_nll,
                "paired_delta_nll": point.mean_nll - control.mean_nll,
            }
        )
    return tuple(rows)


def _hardware_table(
    points: Sequence[HardwarePoint],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "profile_id": point.profile.profile_id,
            "candidate_id": point.candidate_id,
            "weight_format": point.profile.weight_format,
            "activation_format": point.profile.activation_format,
            "kv_format": point.profile.kv_format,
            "vector_format": point.profile.vector_format,
            "delta_nll": point.delta_nll,
            "relative_perplexity_increase_percent": (point.relative_perplexity_percent),
            "whole_model_tpot_ms": point.tpot_ms,
            "whole_model_tps": point.tps,
            "energy_j_per_generated_token": point.energy_j,
            "tokens_per_joule": point.tokens_per_j,
            "edp_j_s": point.edp_j_s,
            "energy_tier": point.energy_tier or "",
            "aggregate_area_mm2": point.area_mm2,
            "aggregate_area_budget_mm2": point.area_budget_mm2 or "",
            "max_runtime_batch": point.max_runtime_batch,
            "chip_count": point.chip_count,
            "tp": point.tp,
            "kvp": point.kvp,
            "retention_labels": ";".join(point.retention_labels),
            "sample_seed": point.sample_seed or "",
            "sample_limit": (
                point.sample_limit if point.sample_limit is not None else ""
            ),
            "scatter_population_count": (
                point.scatter_population_count
                if point.scatter_population_count is not None
                else ""
            ),
            "dominated_population_count": (
                point.dominated_population_count
                if point.dominated_population_count is not None
                else ""
            ),
        }
        for point in points
    )


def _packedkv_table(evidence: Any) -> tuple[dict[str, Any], ...]:
    rows = []
    for group in sorted(evidence.groups, key=lambda item: item.key):
        for mode in PACKEDKV_MODES:
            measurement = group.by_mode()[mode]
            rows.append(
                {
                    "precision_role": group.precision.role,
                    "precision_format": group.precision.format_id,
                    "topology_role": group.topology.role,
                    "mode": mode,
                    "read_bytes_per_sequence_token": (
                        measurement.read_bytes_per_sequence_token
                    ),
                    "feasible_batch": measurement.feasible_batch,
                    "capacity_limited_tokens_per_s": (
                        measurement.capacity_limited_tokens_per_s
                    ),
                    "tpot_ms": measurement.tpot_ms,
                }
            )
    return tuple(rows)


def render(args: argparse.Namespace) -> dict[str, Any]:
    _configure_style()
    manifest_path = Path(args.manifest).resolve()
    numerical_paths = tuple(Path(path).resolve() for path in args.numerical)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(args.formats)
    manifest, points, terminal_rows = _load_numerical_points(
        manifest_path,
        numerical_paths,
        require_complete=not args.allow_incomplete,
        stage="numerical-screen",
    )
    sweep_provenance_path = manifest_path.parent / "provenance.json"
    sweep_provenance = _load_sweep_provenance(
        sweep_provenance_path,
        manifest,
    )
    portable_provenance_path = _copy_sweep_provenance(
        sweep_provenance_path,
        output_dir,
        expected_content_hash=str(sweep_provenance["content_hash"]),
    )
    reference_nll = _reference_nll(points)
    outputs: list[Path] = []
    data_tables: list[Path] = []
    omitted_figures: list[dict[str, str]] = []
    outputs.extend(
        plot_completion_matrix(
            manifest,
            terminal_rows,
            model_name=manifest.model_name,
            output_dir=output_dir,
            formats=formats,
        )
    )
    numerical_table = _numerical_table(
        manifest,
        terminal_rows,
        reference_nll=reference_nll,
    )
    data_tables.append(
        _write_csv(
            output_dir / "00_numerical_completion_data.csv",
            numerical_table,
            fieldnames=tuple(numerical_table[0]),
        )
    )

    def render_numerical_figure(name: str, callback: Any) -> None:
        try:
            outputs.extend(callback())
        except ValueError as exc:
            if not args.allow_incomplete:
                raise
            omitted_figures.append({"figure": name, "reason": str(exc)})

    render_numerical_figure(
        "accuracy_by_weight",
        lambda: plot_accuracy_by_weight(
            points,
            reference_nll=reference_nll,
            model_name=manifest.model_name,
            output_dir=output_dir,
            formats=formats,
        ),
    )
    weight_rows = tuple(
        row
        for row in numerical_table
        if row["state"] == "succeeded" and row["kind"] != PROFILE_KIND_BF16_REFERENCE
    )
    if weight_rows:
        data_tables.append(
            _write_csv(
                output_dir / "01_accuracy_by_weight_data.csv",
                weight_rows,
                fieldnames=tuple(weight_rows[0]),
            )
        )
    render_numerical_figure(
        "weight_kv_accuracy_landscape",
        lambda: plot_accuracy_landscape(
            points,
            reference_nll=reference_nll,
            model_name=manifest.model_name,
            output_dir=output_dir,
            formats=formats,
        ),
    )
    try:
        landscape_rows = _landscape_table(
            points,
            reference_nll=reference_nll,
        )
    except ValueError:
        if not args.allow_incomplete:
            raise
    else:
        data_tables.append(
            _write_csv(
                output_dir / "02_weight_kv_accuracy_landscape_data.csv",
                landscape_rows,
                fieldnames=tuple(landscape_rows[0]),
            )
        )
    render_numerical_figure(
        "vector_precision_sensitivity",
        lambda: plot_vector_sensitivity(
            points,
            reference_nll=reference_nll,
            model_name=manifest.model_name,
            output_dir=output_dir,
            formats=formats,
        ),
    )
    vector_rows = _vector_table(points)
    if vector_rows:
        data_tables.append(
            _write_csv(
                output_dir / "03_vector_precision_sensitivity_data.csv",
                vector_rows,
                fieldnames=tuple(vector_rows[0]),
            )
        )

    fidelity: dict[str, Any] | None = None
    publication_selection: dict[str, Any] | None = None
    source_paths = [manifest_path, *numerical_paths]
    if args.validation_numerical:
        validation_paths = tuple(
            Path(path).resolve() for path in args.validation_numerical
        )
        (
            validation_manifest,
            validation_points,
            _,
        ) = _load_numerical_points(
            manifest_path,
            validation_paths,
            require_complete=False,
            stage="hardware-validation",
        )
        if validation_manifest.canonical_hash != manifest.canonical_hash:
            raise ValueError("validation results use a different manifest")
        fidelity_outputs, fidelity = plot_screening_fidelity(
            points,
            validation_points,
            model_name=manifest.model_name,
            output_dir=output_dir,
            formats=formats,
        )
        outputs.extend(fidelity_outputs)
        screening_by_id = {point.profile_id: point for point in points}
        validation_by_id = {point.profile_id: point for point in validation_points}
        common_ids = tuple(sorted(set(screening_by_id) & set(validation_by_id)))
        screen_ranks = _average_ranks(
            [screening_by_id[profile_id].mean_nll for profile_id in common_ids]
        )
        validation_ranks = _average_ranks(
            [validation_by_id[profile_id].mean_nll for profile_id in common_ids]
        )
        fidelity_rows = tuple(
            {
                "profile_id": profile_id,
                "weight_format": screening_by_id[profile_id].profile.weight_format,
                "activation_format": screening_by_id[
                    profile_id
                ].profile.activation_format,
                "kv_format": screening_by_id[profile_id].profile.kv_format,
                "vector_format": screening_by_id[profile_id].profile.vector_format,
                "numerical_screen_mean_nll": screening_by_id[profile_id].mean_nll,
                "hardware_validation_mean_nll": validation_by_id[profile_id].mean_nll,
                "numerical_screen_rank": float(screen_ranks[index]),
                "hardware_validation_rank": float(validation_ranks[index]),
            }
            for index, profile_id in enumerate(common_ids)
        )
        data_tables.append(
            _write_csv(
                output_dir / "04_screening_fidelity_data.csv",
                fidelity_rows,
                fieldnames=tuple(fidelity_rows[0]),
            )
        )
        source_paths.extend(validation_paths)

    profile_by_id = {point.profile_id: point.profile for point in points}
    nll_by_id = {point.profile_id: point.mean_nll for point in points}
    if args.hardware_artifact:
        raw_hardware_paths = (
            (args.hardware_artifact,)
            if isinstance(args.hardware_artifact, (str, Path))
            else tuple(args.hardware_artifact)
        )
        hardware_paths = tuple(
            Path(path).resolve() for path in raw_hardware_paths
        )
        hardware_points = tuple(
            point
            for hardware_path in hardware_paths
            for point in _load_hardware_points(
                hardware_path,
                profile_by_id=profile_by_id,
                nll_by_id=nll_by_id,
                reference_nll=reference_nll,
            )
        )
        hardware_identities = tuple(
            (point.profile_id, point.candidate_id) for point in hardware_points
        )
        if len(hardware_identities) != len(set(hardware_identities)):
            raise ValueError("hardware artifact partitions overlap")
        outputs.extend(
            plot_hardware_pareto(
                hardware_points,
                model_name=manifest.model_name,
                output_dir=output_dir,
                formats=formats,
            )
        )
        hardware_rows = _hardware_table(hardware_points)
        data_tables.append(
            _write_csv(
                output_dir / "05_hardware_pareto_data.csv",
                hardware_rows,
                fieldnames=tuple(hardware_rows[0]),
            )
        )
        for hardware_path in hardware_paths:
            source_paths.append(hardware_path)
            metadata_path = hardware_path.with_name(
                f"{hardware_path.name}.meta.json"
            )
            if metadata_path.is_file():
                source_paths.append(metadata_path)
    if args.packedkv_evidence:
        packedkv_path = Path(args.packedkv_evidence).resolve()
        packedkv_evidence = load_packedkv_evidence(packedkv_path)
        outputs.extend(
            plot_packedkv_ablation(
                packedkv_path,
                output_dir=output_dir,
                formats=formats,
            )
        )
        packedkv_rows = _packedkv_table(packedkv_evidence)
        data_tables.append(
            _write_csv(
                output_dir / "06_packedkv_causal_ablation_data.csv",
                packedkv_rows,
                fieldnames=tuple(packedkv_rows[0]),
            )
        )
        source_paths.append(packedkv_path)
    if args.decode_analysis:
        analysis_path = Path(args.decode_analysis).resolve()
        analysis = _load_decode_analysis(analysis_path)
        if analysis["model_name"] != manifest.model_name:
            raise ValueError("decode analysis uses a different model")
        analysis_specs = (
            (
                "model_validation",
                plot_model_validation,
                _model_validation_table,
                "07_model_validation_data.csv",
            ),
            (
                "stage_breakdown",
                plot_stage_breakdown,
                _stage_breakdown_table,
                "08_decode_stage_breakdown_data.csv",
            ),
            (
                "capacity",
                plot_decode_capacity,
                _capacity_table,
                "09_decode_capacity_data.csv",
            ),
            (
                "handoff",
                plot_handoff_regimes,
                _handoff_table,
                "10_handoff_regimes_data.csv",
            ),
            (
                "multichip",
                plot_multichip_scaling,
                _multichip_table,
                "11_multichip_scaling_data.csv",
            ),
        )
        for section, plotter, table_builder, filename in analysis_specs:
            rows = analysis[section]
            if not rows:
                continue
            outputs.extend(
                plotter(rows, output_dir=output_dir, formats=formats)
            )
            table = table_builder(rows)
            data_tables.append(
                _write_csv(
                    output_dir / filename,
                    table,
                    fieldnames=tuple(table[0]),
                )
            )
        source_paths.append(analysis_path)

    publication_inputs = {
        "config": getattr(args, "config", None),
        "gpu_baseline_report": getattr(args, "gpu_baseline_report", None),
        "gpu_baseline_receipt": getattr(args, "gpu_baseline_receipt", None),
        "publication_contract": getattr(args, "publication_contract", None),
        "publication_report": getattr(args, "publication_report", None),
        "final_selection": getattr(args, "final_selection", None),
        "refined_hardware_artifact": getattr(
            args,
            "refined_hardware_artifact",
            None,
        ),
    }
    supplied_publication_inputs = {
        name: value
        for name, value in publication_inputs.items()
        if value is not None
    }
    if supplied_publication_inputs and len(supplied_publication_inputs) != len(
        publication_inputs
    ):
        missing = sorted(set(publication_inputs) - set(supplied_publication_inputs))
        raise ValueError(
            "selected-deployment rendering requires all publication inputs; "
            "missing " + ", ".join(missing)
        )
    if supplied_publication_inputs:
        publication_paths = {
            name: Path(str(value)).resolve()
            for name, value in publication_inputs.items()
        }
        selected_rows, publication_selection = _load_selected_publication_rows(
            config_path=publication_paths["config"],
            manifest=manifest,
            gpu_baseline_report_path=publication_paths[
                "gpu_baseline_report"
            ],
            gpu_baseline_receipt_path=publication_paths[
                "gpu_baseline_receipt"
            ],
            publication_contract_path=publication_paths[
                "publication_contract"
            ],
            publication_report_path=publication_paths["publication_report"],
            final_selection_path=publication_paths["final_selection"],
            refined_hardware_artifact_path=publication_paths[
                "refined_hardware_artifact"
            ],
        )
        outputs.extend(
            plot_selected_deployment_evidence(
                selected_rows,
                model_name=manifest.model_name,
                output_dir=output_dir,
                formats=formats,
            )
        )
        data_tables.append(
            _write_csv(
                output_dir / "12_selected_deployment_data.csv",
                selected_rows,
                fieldnames=tuple(selected_rows[0]),
            )
        )
        source_paths.extend(publication_paths.values())
        refined_metadata = publication_paths[
            "refined_hardware_artifact"
        ].with_name(
            f"{publication_paths['refined_hardware_artifact'].name}.meta.json"
        )
        if refined_metadata.is_file():
            source_paths.append(refined_metadata)

    terminal_counts: dict[str, int] = {}
    for row in terminal_rows:
        state = str(row["state"])
        terminal_counts[state] = terminal_counts.get(state, 0) + 1
    resolved_sources = tuple(
        sorted(
            {
                source_file.resolve()
                for path in source_paths
                for source_file in _source_files(path)
            },
            key=str,
        )
    )
    generator_path = Path(__file__).resolve()
    receipt = {
        "schema_version": FIGURE_SCHEMA,
        "manifest_hash": manifest.canonical_hash,
        "model_name": manifest.model_name,
        "reference_mean_nll": reference_nll,
        "allow_incomplete": bool(args.allow_incomplete),
        "command": [sys.executable, "-m", "decode_dse.plots", *sys.argv[1:]],
        "generator": {
            "path": str(generator_path),
            "sha256": _sha256(generator_path),
        },
        "rendering": {
            "png_dpi": FIGURE_DPI,
            "formats": list(formats),
            "font_embedding": {
                "pdf_fonttype": 42,
                "svg_fonttype": "none",
            },
        },
        "selection_policy": {
            "accuracy": "successful numerical rows only; exact format IDs",
            "vector_attribution": (
                "paired to the vector-BF16 control with identical W/A/KV"
            ),
            "hardware": (
                "deployment_valid rows with rankable whole-model timing and "
                "explicit analytic-anchored or DC-calibrated energy"
            ),
            "packedkv": "checksum-valid packedkv-publication-evidence/v4",
            "decode_analysis": ANALYSIS_SCHEMA,
            "selected_deployment": (
                "post-accuracy exact refined-hardware join with a separately "
                "validated measured GPU denominator"
            ),
            "cross_tier_ratio": "forbidden",
            "peak_roofline": "separate from the headline comparison",
        },
        "terminal_counts": terminal_counts,
        "fidelity": fidelity,
        "publication_selection": publication_selection,
        "omitted_figures": omitted_figures,
        "sources": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in resolved_sources
        ],
        "figures": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "data_tables": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in data_tables
        ],
    }
    results_provenance = _build_results_provenance(
        sweep_provenance_path=portable_provenance_path,
        sweep_provenance=sweep_provenance,
        manifest=manifest,
        data_tables=data_tables,
    )
    results_provenance_path = output_dir / "sweep_results_provenance.json"
    _write_json(results_provenance_path, results_provenance)
    receipt["results_provenance"] = {
        "path": str(results_provenance_path),
        "content_hash": results_provenance["content_hash"],
        "schema_version": RESULTS_PROVENANCE_SCHEMA,
    }
    receipt_path = output_dir / "figure_manifest.json"
    _write_json(receipt_path, receipt)
    return receipt | {"receipt_path": str(receipt_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render publication figures from checksum-verified numerical and "
            "hardware artifacts."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--numerical",
        action="append",
        required=True,
        help=(
            "Numerical-screen result directory or JSONL file; repeat for "
            "multiple shards."
        ),
    )
    parser.add_argument(
        "--validation-numerical",
        action="append",
        help=(
            "Higher-fidelity result directory or JSONL file; repeat for "
            "multiple shards to render the screening-fidelity figure."
        ),
    )
    parser.add_argument(
        "--hardware-artifact",
        action="append",
        help="Checksum-verified hardware partition; repeat for sharded studies.",
    )
    parser.add_argument("--config")
    parser.add_argument("--gpu-baseline-report")
    parser.add_argument("--gpu-baseline-receipt")
    parser.add_argument("--publication-contract")
    parser.add_argument("--publication-report")
    parser.add_argument("--final-selection")
    parser.add_argument("--refined-hardware-artifact")
    parser.add_argument("--packedkv-evidence")
    parser.add_argument(
        "--decode-analysis",
        help=(
            "Compact JSON artifact for validation, stage, capacity, handoff, "
            "and multi-chip figures."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Render diagnostics from a partial numerical run.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    receipt = render(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
