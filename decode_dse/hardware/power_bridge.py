"""Bridge simulator traffic and timing into the analytic decode-energy model."""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from decode_dse.hardware.design_space import CalibratedEnergy, HardwareCandidate


def _simulator_module(name: str):
    default_root = Path(__file__).resolve().parents[3] / "PLENA_Simulator"
    root = Path(os.environ.get("PLENA_SIMULATOR_PATH", default_root)).resolve()
    source = root / "analytic_models" / "disagg_serve"
    if not source.is_dir():
        raise FileNotFoundError(f"simulator energy models are missing: {source}")
    token = str(root)
    if token not in sys.path:
        sys.path.insert(0, token)
    return importlib.import_module(f"analytic_models.disagg_serve.{name}")


def analytic_power_provenance() -> dict[str, Any]:
    """Return the coefficient identity and declared evidence scopes."""

    model = _simulator_module("decode_power")
    return {
        "engine": "decode_analytic_power_bridge",
        "energy_tier": model.ANALYTIC_ENERGY_TIER,
        "energy_id": model.analytic_energy_identity(),
        "sram": dict(model.SRAM_ENERGY_SOURCE),
        "leakage": dict(model.LEAKAGE_SOURCE),
        "link": dict(model.LINK_ENERGY_SOURCE),
        "sram_access_accounting": (
            "HBM fills plus four vector-workspace reads and one write per step"
        ),
    }


def hbm_peak_bandwidth_bytes_per_s(
    generation: str,
    interface_units: int,
) -> float:
    """Return peak bandwidth for the versioned HBM operating point."""

    if interface_units <= 0:
        raise ValueError("interface_units must be positive")
    technology = _simulator_module("hbm_technology").hbm_technology(
        generation
    )
    return (
        interface_units
        * 64.0
        * float(technology.pin_rate_gbps)
        * 1e9
        / 8.0
    )


def _traffic_rates(
    observation: Any,
) -> tuple[float, float]:
    values = dict(observation.hbm_traffic_per_generated_token)
    if not values:
        raise ValueError("analytic energy requires physical HBM traffic")
    read_per_token = sum(
        float(value)
        for name, value in values.items()
        if name.endswith("_read_bytes")
    )
    write_per_token = sum(
        float(value)
        for name, value in values.items()
        if name.endswith("_write_bytes")
    )
    if min(read_per_token, write_per_token) < 0:
        raise ValueError("physical HBM traffic must be non-negative")
    return (
        read_per_token * float(observation.tps),
        write_per_token * float(observation.tps),
    )


def analytic_energy_from_simulator(
    *,
    candidate: HardwareCandidate,
    observation: Any,
    mac_bits: int,
    per_chip_logic_area_mm2: float,
    collective_bytes_per_generated_token: float,
    link_generation: str = "nvlink4",
) -> CalibratedEnergy:
    """Return rankable analytic energy per generated token.

    HBM and link traffic are aggregate-system values.  Logic geometry, SRAM
    traffic and HBM capacity are converted to per-chip inputs before the power
    model applies ``chip_count`` exactly once.
    """

    model = _simulator_module("decode_power")
    hbm = _simulator_module("hbm_technology")
    if mac_bits <= 0:
        raise ValueError("mac_bits must be positive")
    if per_chip_logic_area_mm2 <= 0 or not math.isfinite(
        per_chip_logic_area_mm2
    ):
        raise ValueError("per-chip logic area must be finite and positive")
    if (
        collective_bytes_per_generated_token < 0
        or not math.isfinite(collective_bytes_per_generated_token)
    ):
        raise ValueError("collective bytes must be finite and non-negative")
    read_rate, write_rate = _traffic_rates(observation)
    chip_count = candidate.chip_count
    step_rate = float(observation.tps) / candidate.batch
    vector_workspace_rate = (
        float(observation.vector_sram_required_bytes) * step_rate
    )
    sram_read_rate_per_chip = (
        read_rate + 4.0 * vector_workspace_rate
    ) / chip_count
    sram_write_rate_per_chip = (
        read_rate + write_rate + vector_workspace_rate
    ) / chip_count
    active_fraction = min(
        1.0,
        float(observation.avg_realized_compute_seconds)
        / (float(observation.tpot_ms) / 1000.0),
    )
    estimate = model.decode_power(
        hbm.hbm_technology(candidate.hbm_generation),
        capacity_bytes=(
            float(observation.capacity.available_bytes) / chip_count
        ),
        read_bytes_per_second=read_rate / chip_count,
        write_bytes_per_second=write_rate / chip_count,
        multipliers=candidate.mlen * candidate.blen,
        clock_hz=1.0e9,
        mac_bits=mac_bits,
        array_active_fraction=active_fraction,
        tokens_per_second=float(observation.tps),
        chip_count=chip_count,
        sram_read_bytes_per_second=sram_read_rate_per_chip,
        sram_write_bytes_per_second=sram_write_rate_per_chip,
        logic_area_mm2=per_chip_logic_area_mm2,
        link_bytes_per_second=(
            collective_bytes_per_generated_token
            * float(observation.tps)
        ),
        link_generation=link_generation,
        token_latency_s=float(observation.tpot_ms) / 1000.0,
    )
    if estimate.tokens_per_second <= 0:
        raise ValueError("analytic energy requires positive throughput")
    per_token = 1.0 / estimate.tokens_per_second
    duration = (
        float(observation.tpot_ms)
        / 1000.0
        / int(observation.generated_tokens_per_step)
    )
    return CalibratedEnergy(
        calibration_id=str(estimate.energy_id),
        energy_id=str(estimate.energy_id),
        energy_tier=str(estimate.energy_tier),
        compute_j=estimate.compute_watts * per_token,
        vector_j=0.0,
        sram_j=estimate.sram_watts * per_token,
        hbm_j=estimate.memory_watts * per_token,
        leakage_j=estimate.leakage_watts * per_token,
        link_j=estimate.link_watts * per_token,
        duration_s=duration,
        token_latency_s=float(observation.tpot_ms) / 1000.0,
    )


__all__ = [
    "analytic_energy_from_simulator",
    "analytic_power_provenance",
    "hbm_peak_bandwidth_bytes_per_s",
]
