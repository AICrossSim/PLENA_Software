"""Optuna co-design search over the PLENA decode-chip hardware.

Optimises the analytic metric directly — maximise TPS, minimise TPOT — instead of
a memory proxy. Multi-objective NSGA-II returns the TPS/TPOT Pareto front for one
fixed precision; ``fits_in_hbm`` is a hard constraint (a chip that cannot hold
weights + KV is infeasible), and area is reported so the smallest chip on the
front is visible.

Search knobs: array geometry MLEN/BLEN/VLEN/HLEN, serving BATCH, and HBM channel
count (technology is fixed; only channels scale). Trials that break the array
geometry, the compiler constraint (``hidden % VLEN``), or the bandwidth cap are
rejected as infeasible. The chip count is pinned to one so the HBM-fit constraint
binds — with auto-chips the model would add chips (free bandwidth) and
``fits_in_hbm`` would always be true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna

from decode_dse.simulator_bridge import DecodeSimulator, Precision

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HardwareSpace:
    """Discrete search grid for the decode-chip hardware knobs.

    VLEN is not searched: the vector unit is tied to the matrix unit (VLEN=MLEN,
    the compiler constraint). MLEN starts at 16 because HLEN (the FlashAttention
    head tile) needs BLEN <= HLEN <= MLEN and the smallest useful head tile is 16,
    so smaller MLENs only waste trials. HBM technology is fixed (hbm_gen); channels
    scale bandwidth and capacity together. The channel span reaches from a small
    chip (8 ch) to A100-class aggregate bandwidth (128 ch).
    """

    MLEN: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    BLEN: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    HLEN: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    BATCH: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64, 128, 256])
    HBM_CHANNELS: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    hbm_gen: str = "HBM2"  # fixed technology (emulator-validated preset); channels scale

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "HardwareSpace":
        if not d:
            return cls()
        known = {f: d[f] for f in ("MLEN", "BLEN", "HLEN", "BATCH", "HBM_CHANNELS", "hbm_gen") if f in d}
        return cls(**known)


def _geometry_ok(mlen: int, blen: int, hlen: int, hidden: int) -> bool:
    """Legal array geometry (VLEN=MLEN >= BLEN is implicit) plus the compiler
    constraint that the hidden size tiles the vector unit exactly
    (``hidden % VLEN == 0``, VLEN=MLEN)."""
    return (mlen % blen == 0 and mlen % hlen == 0 and blen <= hlen <= mlen
            and hidden % mlen == 0)


def make_sampler(name: str, seed: int, constraints_func=None) -> optuna.samplers.BaseSampler:
    """Multi-objective sampler by name.

    ``nsga2`` (default) is the evolutionary NSGA-II sampler, robust on this mixed
    categorical space and used for all recorded runs. ``tpe`` is the
    multi-objective TPE (MOTPE), a Bayesian alternative. Both honour the same
    feasibility constraints.
    """
    if name == "tpe":
        kw: dict = {"seed": seed, "multivariate": True}
        if constraints_func is not None:
            kw["constraints_func"] = constraints_func
        return optuna.samplers.TPESampler(**kw)
    if name == "nsga2":
        kw = {"seed": seed}
        if constraints_func is not None:
            kw["constraints_func"] = constraints_func
        return optuna.samplers.NSGAIISampler(**kw)
    raise ValueError(f"unknown sampler {name!r} (choose 'nsga2' or 'tpe').")


@dataclass
class TrialRecord:
    hw: dict[str, int]
    tps: float
    tpot_ms: float
    area_mm2: float
    fits: bool
    mem_bound: bool
    feasible: bool


class CodesignSearch:
    """NSGA-II co-design for one precision on one model."""

    def __init__(
        self,
        sim: DecodeSimulator,
        precision: Precision,
        workload: dict[str, int],
        space: HardwareSpace,
        *,
        n_chips: int = 1,
        seed: int = 0,
        sampler: str = "nsga2",
    ):
        self.sim = sim
        self.prec = precision
        self.workload = workload
        self.space = space
        self.n_chips = n_chips
        self.seed = seed
        self.sampler = sampler
        self._hbm = sim.hbm_overrides(space.hbm_gen)  # base HBM (channels overridden per trial)

    # -- single-candidate evaluation ----------------------------------------
    def _hw_over(self, params: dict[str, int]) -> dict[str, int]:
        gen = self.space.hbm_gen
        hbm = self.sim.hbm_overrides(gen, channels=params["HBM_CHANNELS"])
        return {
            "MLEN": params["MLEN"], "BLEN": params["BLEN"],
            "VLEN": params["MLEN"], "HLEN": params["HLEN"],  # VLEN tied to MLEN
            "HBM_WIDTH": hbm["HBM_WIDTH"], "HBM_SIZE": hbm["HBM_SIZE"],
        }

    def evaluate(self, params: dict[str, int]) -> TrialRecord:
        over = self._hw_over(params)
        geom = _geometry_ok(params["MLEN"], params["BLEN"], params["HLEN"],
                            self.sim.dims["hidden"])
        bw_ok = params["MLEN"] <= self.sim.mlen_cap(self.prec, hw_over=over)
        if not (geom and bw_ok):
            return TrialRecord(over, 0.0, 1e9, self.sim._dd.area_mm2(  # penalised
                self.sim.base_hw.model_copy(update=over)), False, False, feasible=False)
        m = self.sim.evaluate(
            self.prec, batch=params["BATCH"],
            input_seq=self.workload["input_seq"], output_seq=self.workload["output_seq"],
            hw_over=over, n_chips=self.n_chips,
            # Keys the calibrated bandwidth lookup (when active) so a searched
            # channel count is priced at its measured sublinear scaling, not peak.
            hbm_gen=self.space.hbm_gen, hbm_channels=params["HBM_CHANNELS"],
        )
        return TrialRecord(
            {**over, "BATCH": params["BATCH"]}, m.tps, m.tpot * 1e3, m.area_mm2,
            m.fits_in_hbm, m.mem_bound, feasible=m.fits_in_hbm,
        )

    # -- Optuna study --------------------------------------------------------
    def _objective(self, trial: optuna.Trial) -> tuple[float, float, float]:
        params = {
            "MLEN": trial.suggest_categorical("MLEN", self.space.MLEN),
            "BLEN": trial.suggest_categorical("BLEN", self.space.BLEN),
            "HLEN": trial.suggest_categorical("HLEN", self.space.HLEN),
            "BATCH": trial.suggest_categorical("BATCH", self.space.BATCH),
            "HBM_CHANNELS": trial.suggest_categorical("HBM_CHANNELS", self.space.HBM_CHANNELS),
        }
        rec = self.evaluate(params)
        # Constraint (<=0 feasible): geometry/bandwidth + HBM fit folded into one.
        trial.set_user_attr("constraints", (0.0 if rec.feasible else 1.0,))
        trial.set_user_attr("fits", rec.fits)
        trial.set_user_attr("mem_bound", rec.mem_bound)
        # Three co-design objectives: throughput, latency, area.
        return rec.tps, rec.tpot_ms, rec.area_mm2

    @staticmethod
    def _constraints(trial: optuna.trial.FrozenTrial):
        return trial.user_attrs.get("constraints", (0.0,))

    def run(self, n_trials: int = 128) -> list[TrialRecord]:
        sampler = make_sampler(self.sampler, self.seed, constraints_func=self._constraints)
        study = optuna.create_study(
            # TPS up, TPOT down, area down.
            directions=["maximize", "minimize", "minimize"], sampler=sampler,
        )
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=False)
        self.study = study
        return self._front(study)

    def _front(self, study: optuna.Study) -> list[TrialRecord]:
        """Feasible Pareto-optimal trials, rebuilt as records (best TPS first)."""
        recs = []
        for t in study.best_trials:
            if t.user_attrs.get("constraints", (1.0,))[0] > 0:
                continue
            hw = {k: t.params[k] for k in ("MLEN", "BLEN", "HLEN", "BATCH")}
            hw["HBM_CHANNELS"] = t.params["HBM_CHANNELS"]
            recs.append(TrialRecord(
                hw, t.values[0], t.values[1], t.values[2],
                t.user_attrs.get("fits", True), t.user_attrs.get("mem_bound", False), True,
            ))
        return sorted(recs, key=lambda r: r.tps, reverse=True)


def search_precision(
    sim: DecodeSimulator,
    precision: Precision,
    workload: dict[str, int],
    space: HardwareSpace,
    *,
    n_trials: int = 128,
    n_chips: int = 1,
    seed: int = 0,
    sampler: str = "nsga2",
) -> list[TrialRecord]:
    """Convenience wrapper: run one co-design study and return its Pareto front."""
    return CodesignSearch(sim, precision, workload, space,
                          n_chips=n_chips, seed=seed, sampler=sampler).run(n_trials)
