#!/usr/bin/env python3
"""Online DSE for PLENA: GP + Expected Improvement on TPS.

Single-objective Bayesian optimization over the PLENA hardware design space
(MLEN/BLEN/VLEN/HLEN + memory knobs). Each candidate is evaluated in-process
via PLENA_Simulator's LLaMAModel.

Usage (from PLENA_Software repo root):
    python plena_experiments/online_dse/scripts/online_dse_gp_ei.py \\
        plena_experiments/online_dse/configs/dse_llama3_8b.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.utils.transforms import normalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False

DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_BOTORCH else None
)
DTYPE = torch.double if HAS_BOTORCH else None

# HardwareConfig fields the DSE varies. Order = feature-vector order.
KNOBS = (
    "BLEN",
    "MLEN",
    "VLEN",
    "HLEN",
    "VECTOR_SRAM_SIZE",
    "HBM_V_Prefetch_Amount",
    "HBM_M_Prefetch_Amount",
    "HBM_SIZE",
    "HBM_WIDTH",
)


# ---------------------------------------------------------------------------
# Design space enumeration
# ---------------------------------------------------------------------------
def build_candidates(
    search_grid: Dict[str, List[int]],
) -> Tuple[List[Dict[str, int]], np.ndarray]:
    """Enumerate the discrete Cartesian product, filtering invalid combos.

    HardwareConfig validators require MLEN % BLEN == 0 and VLEN >= BLEN.
    """
    missing = [k for k in KNOBS if k not in search_grid]
    if missing:
        raise SystemExit(f"`search` missing required knobs: {missing}")

    grids = [search_grid[k] for k in KNOBS]
    candidates: List[Dict[str, int]] = []
    features: List[List[float]] = []
    for combo in itertools.product(*grids):
        d = dict(zip(KNOBS, combo))
        if d["MLEN"] % d["BLEN"] != 0:
            continue
        if d["VLEN"] < d["BLEN"]:
            continue
        candidates.append(d)
        features.append([float(v) for v in combo])

    if not candidates:
        raise SystemExit(
            "No valid candidates after MLEN%BLEN / VLEN>=BLEN filtering."
        )
    return candidates, np.asarray(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Disk-backed evaluation cache
# ---------------------------------------------------------------------------
class EvalCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            print(f"  Loaded {len(self.data)} cached evaluations from {self.path}")

    @staticmethod
    def _key(cand: Dict[str, int]) -> str:
        raw = json.dumps(cand, sort_keys=True).encode()
        return hashlib.md5(raw).hexdigest()

    def get(self, cand: Dict[str, int]) -> Optional[float]:
        v = self.data.get(self._key(cand))
        if v is None or v == "__FAILED__":
            return None
        return float(v)

    def put(self, cand: Dict[str, int], tps: Optional[float]) -> None:
        self.data[self._key(cand)] = tps if tps is not None else "__FAILED__"

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1))


# ---------------------------------------------------------------------------
# PLENA simulator evaluator (in-process)
# ---------------------------------------------------------------------------
class PlenaEvaluator:
    """Wraps PLENA_Simulator's LLaMAModel for in-process candidate evaluation."""

    def __init__(self, cfg: Dict[str, Any]):
        sim_root = Path(cfg["plena_simulator_path"]).expanduser().resolve()
        perf_dir = sim_root / "analytic_models" / "performance"
        if not perf_dir.exists():
            raise SystemExit(
                f"plena_simulator_path looks wrong — missing {perf_dir}. "
                f"Set it to a local clone of github.com/AICrossSim/PLENA_Simulator."
            )
        # Script-style imports inside the simulator require this dir on sys.path.
        if str(perf_dir) not in sys.path:
            sys.path.insert(0, str(perf_dir))

        from llama_model import LLaMAModel  # noqa: E402
        from perf_model import HardwareConfig, load_hardware_config_from_toml  # noqa: E402

        self._LLaMAModel = LLaMAModel
        self._HardwareConfig = HardwareConfig

        toml_path = cfg.get("plena_settings_toml") or str(sim_root / "plena_settings.toml")
        isa_path = cfg.get("custom_isa_path") or str(perf_dir / "customISA_lib.json")
        self.model_config_path = str(Path(cfg["model_config_path"]).expanduser().resolve())
        self.custom_isa_path = isa_path
        self._base_dump = load_hardware_config_from_toml(toml_path).model_dump()
        self.batch_size = int(cfg.get("batch_size", 1))
        self.input_seq_len = int(cfg.get("input_seq_len", 2048))
        self.output_seq_len = int(cfg.get("output_seq_len", 128))

    def evaluate(self, cand: Dict[str, int]) -> Optional[float]:
        try:
            hw = self._HardwareConfig(**(self._base_dump | cand))
            model = self._LLaMAModel(
                model_config_path=self.model_config_path,
                hardware_config=hw,
                custom_isa_path=self.custom_isa_path,
                batch_size=self.batch_size,
                input_seq_len=self.input_seq_len,
                output_seq_len=self.output_seq_len,
            )
            _, tps = model.compute_performance(verbose=False)
            return float(tps)
        except Exception as e:
            print(f"    eval failed for {cand}: {type(e).__name__}: {e}")
            return None


# ---------------------------------------------------------------------------
# GP + Expected Improvement (single objective)
# ---------------------------------------------------------------------------
class OnlineBOSearch:
    """GP-based BO over a finite candidate set; argmax LogEI for selection."""

    def __init__(self, X: np.ndarray, n_init: int, seed: int):
        self._X = torch.tensor(X, dtype=DTYPE, device=DEVICE)
        self._bounds = torch.stack([self._X.min(0).values, self._X.max(0).values])
        # Guard against zero-width dims (e.g. single-valued knob).
        if (self._bounds[1] - self._bounds[0] == 0).any():
            self._bounds[1] = torch.where(
                self._bounds[1] == self._bounds[0],
                self._bounds[0] + 1.0,
                self._bounds[1],
            )
        self._X_norm = normalize(self._X, self._bounds)
        self.n_total = X.shape[0]
        self.n_init = n_init
        self.rng = np.random.RandomState(seed)
        self.observed: List[int] = []
        self.y_obs: List[float] = []
        self.best_trace: List[float] = []

    def _unobs(self) -> np.ndarray:
        seen = set(self.observed)
        return np.array([i for i in range(self.n_total) if i not in seen])

    def suggest(self) -> int:
        if len(self.y_obs) < self.n_init:
            return int(self.rng.choice(self._unobs()))
        return self._suggest_ei()

    def _suggest_ei(self) -> int:
        train_X = self._X_norm[self.observed]
        train_y = torch.tensor(self.y_obs, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
        gp = SingleTaskGP(train_X, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"    GP fit failed ({type(e).__name__}: {e}) — falling back to random.")
                return int(self.rng.choice(self._unobs()))

        best_f = float(max(self.y_obs))
        acq = LogExpectedImprovement(gp, best_f=best_f, maximize=True)
        unobs = self._unobs()
        X_cand = self._X_norm[unobs].unsqueeze(1)  # (N, q=1, D)
        with torch.no_grad():
            vals = acq(X_cand)
        return int(unobs[int(vals.argmax())])

    def observe(self, idx: int, y: float) -> None:
        self.observed.append(idx)
        self.y_obs.append(y)
        self.best_trace.append(max(self.y_obs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Online DSE for PLENA (GP + EI on TPS).")
    ap.add_argument("run_config", type=str, help="Path to DSE run config JSON.")
    ap.add_argument(
        "--cache",
        type=str,
        default="results/online_dse/cache.json",
        help="Disk path for the evaluation cache.",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/online_dse/results.json",
        help="Path to write the final result summary.",
    )
    args = ap.parse_args()

    if not HAS_BOTORCH:
        raise SystemExit("Install BO deps: `uv sync --extra dse` (botorch + gpytorch).")

    cfg = json.loads(Path(args.run_config).read_text())
    bo_cfg = cfg.get("bo", {})
    n_init = int(bo_cfg.get("n_init", 8))
    n_budget = int(bo_cfg.get("n_budget", 40))
    seed = int(bo_cfg.get("seed", 0))

    print("Building candidate list...")
    candidates, X = build_candidates(cfg["search"])
    n_budget = min(n_budget, len(candidates))
    n_init = min(n_init, n_budget)
    print(f"  {len(candidates)} valid candidates × {X.shape[1]} knobs")
    print(f"  BO: n_init={n_init}, n_budget={n_budget}, seed={seed}")

    cache = EvalCache(Path(args.cache))
    evaluator = PlenaEvaluator(cfg)
    bo = OnlineBOSearch(X, n_init=n_init, seed=seed)

    t0 = time.time()
    n_done = 0
    n_fail = 0
    while n_done < n_budget:
        if len(bo._unobs()) == 0:
            print(f"  Candidate pool exhausted after {n_done} evals.")
            break
        idx = bo.suggest()
        cand = candidates[idx]
        tps = cache.get(cand)
        if tps is None:
            tps = evaluator.evaluate(cand)
            cache.put(cand, tps)
            cache.flush()
        if tps is None:
            n_fail += 1
            bo.observe(idx, 0.0)  # treat failures as worst-possible
        else:
            bo.observe(idx, tps)
        n_done += 1
        phase = "init" if n_done <= n_init else "BO"
        if n_done % 5 == 0 or n_done == n_budget:
            print(f"  eval {n_done}/{n_budget}  best_tps={bo.best_trace[-1]:.2f}  ({phase})")

    elapsed = time.time() - t0
    best_pos = int(np.argmax(bo.y_obs)) if bo.y_obs else -1
    best_cand = candidates[bo.observed[best_pos]] if best_pos >= 0 else None
    best_tps = float(max(bo.y_obs)) if bo.y_obs else 0.0

    out = {
        "config": str(Path(args.run_config).resolve()),
        "n_evaluated": n_done,
        "n_failed": n_fail,
        "elapsed_s": round(elapsed, 2),
        "best_tps": best_tps,
        "best_candidate": best_cand,
        "best_tps_trace": bo.best_trace,
        "observed": [
            {"candidate": candidates[i], "tps": y}
            for i, y in zip(bo.observed, bo.y_obs)
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nDone in {elapsed:.1f}s, {n_fail} failures.")
    print(f"  Best TPS:       {best_tps:.2f}")
    print(f"  Best candidate: {best_cand}")
    print(f"  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
