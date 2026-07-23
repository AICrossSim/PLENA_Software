"""End-to-end decode results: accuracy x throughput/latency, with vs without
the decode optimisation.

Joins Stage A (software accuracy per precision) with Stage B (hardware
co-design) into one summary table. Three hardware/precision regimes are
compared on the same workload so the two gains stay separable:

  baseline  : FP16 decode on the shipped (un-searched) chip        — no optimisation
  quant-only: quantised decode on the shipped chip                 — precision gain
  co-design : quantised decode on the Optuna-searched chip         — precision + hardware

Fairness rules, so the speed-up measures the optimisation and not a hidden
hardware upgrade:

* every regime uses the same HBM technology (``hbm_gen``); the baseline sits at a
  fixed channel count (``baseline_hbm_channels``, default 32 HBM2 ch = 512 GB/s)
  while the co-design may search channels, and the searched count is reported;
* the shipped chip is clamped to a legal design point per precision
  (``MLEN <= HBM_WIDTH / stream_bits``, VLEN == MLEN) — the raw TOML geometry
  breaks the bandwidth bound at FP16;
* the co-designed chip is also evaluated at the baseline batch (equal-batch
  columns), so batch-scaling and design gains stay separable;
* baseline/quant rows auto-resolve the chip count (FP16 may exceed one HBM stack,
  the capacity wall) and report it; the co-design is pinned to one chip so its
  HBM-fit constraint binds.

Reports TPS and TPOT for each, plus the accuracy (PPL / GSM8K / IFEval) the
quantised precision costs, so the reader sees what the speed-up buys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decode_dse.simulator_bridge import DecodeSimulator
from decode_dse.hardware.codesign_search import HardwareSpace, search_precision
from decode_dse.results import (
    load_software_rows,
    prefer_best_method,
    select_front,
)


def _best_by_tpot_budget(front: list, tpot_budget_ms: float | None) -> Any:
    """Pick the highest-TPS feasible config within a TPOT budget (or overall)."""
    feasible = [r for r in front if r.fits]
    if tpot_budget_ms is not None:
        within = [r for r in feasible if r.tpot_ms <= tpot_budget_ms]
        if within:
            return max(within, key=lambda r: r.tps)
    return max(feasible, key=lambda r: r.tps) if feasible else None


def _codesign_over(sim: DecodeSimulator, gen: str, hw: dict) -> dict[str, int]:
    """Hardware overrides for a searched design point (VLEN tied to MLEN)."""
    hbm = sim.hbm_overrides(gen, hw["HBM_CHANNELS"])
    return {"MLEN": hw["MLEN"], "BLEN": hw["BLEN"], "VLEN": hw["MLEN"],
            "HLEN": hw["HLEN"], **hbm}


def build_table(cfg: dict, csv_path: Path, *, n_trials: int, tpot_budget_ms: float | None) -> list[dict]:
    sim = DecodeSimulator(cfg["sim_model"], model_lib=cfg.get("model_lib"))
    if cfg.get("bw_model", "calibrated") == "calibrated":
        sim.use_calibrated_bandwidth()
    workload = cfg["reference_workload"]
    base_batch = workload["batch"]
    space = HardwareSpace.from_dict(cfg.get("hardware_space"))
    n_chips = int(cfg.get("n_chips", 1))
    # Baseline HBM: the same technology the co-design searches, at a fixed channel
    # count (32 HBM2 ch = 512 GB/s, the plena_settings bandwidth).
    base_hbm = sim.hbm_overrides(space.hbm_gen, int(cfg.get("baseline_hbm_channels", 32)))

    # FP16 baseline on the shipped chip (no optimisation), clamped to a legal
    # design point; chips auto-resolved (FP16 may exceed one stack).
    fp16 = sim.precision_from_eff_bits(16.0, 16.0, 16.0)
    base_fp = sim.evaluate(fp16, batch=base_batch,
                           input_seq=workload["input_seq"], output_seq=workload["output_seq"],
                           hw_over=sim.shipped_over(fp16, base_hbm), n_chips=0)

    # One row per precision (best available method), ordered by the accuracy front.
    all_rows = list(prefer_best_method(load_software_rows(csv_path)).values())
    front_pts = select_front(all_rows, int(cfg.get("front_size", 6)))

    rows: list[dict] = []
    for p in front_pts:
        prec = sim.precision_from_row(p)
        # quant-only: same shipped chip (same HBM budget), quantised precision.
        q_base = sim.evaluate(prec, batch=base_batch,
                              input_seq=workload["input_seq"], output_seq=workload["output_seq"],
                              hw_over=sim.shipped_over(prec, base_hbm), n_chips=0)
        # co-design: search the chip for this precision (one chip; fit binds).
        pareto = search_precision(sim, prec, workload, space,
                                  n_trials=n_trials, n_chips=n_chips, seed=cfg.get("seed", 0),
                                  sampler=cfg.get("sampler", "nsga2"))
        best = _best_by_tpot_budget(pareto, tpot_budget_ms)
        # equal-batch: the co-designed chip evaluated at the baseline batch, so
        # batch-scaling and design gains stay separable. The max-TPS chip is tuned
        # for its own batch and can lose to the shipped clamp at a small batch, so
        # this column picks the searched chip best at the baseline batch. The
        # shipped clamp seeds the pick (it is a legal search point), so a precision
        # the search can't improve reports x1.0, never a spurious slowdown.
        eq = q_base if q_base.fits_in_hbm else None
        for r in pareto:
            if not r.fits:
                continue
            m = sim.evaluate(prec, batch=base_batch,
                             input_seq=workload["input_seq"], output_seq=workload["output_seq"],
                             hw_over=_codesign_over(sim, space.hbm_gen, r.hw),
                             n_chips=n_chips)
            if m.fits_in_hbm and (eq is None or m.tps > eq.tps):
                eq = m
        rows.append({
            "tag": p["tag"], "ppl_fp": p.get("prefill_ppl") or "",
            "ppl_quant": p["cont_ppl"], "gsm8k": p.get("gsm8k") or "", "ifeval": p.get("ifeval") or "",
            "base_tps": round(base_fp.tps, 1), "base_tpot_ms": round(base_fp.tpot * 1e3, 2),
            "base_chips": base_fp.n_chips,
            "quant_base_tps": round(q_base.tps, 1), "quant_base_tpot_ms": round(q_base.tpot * 1e3, 2),
            "quant_chips": q_base.n_chips,
            "codesign_tps": round(best.tps, 1) if best else None,
            "codesign_tpot_ms": round(best.tpot_ms, 2) if best else None,
            "codesign_area_mm2": round(best.area_mm2, 3) if best else None,
            "codesign_hw": best.hw if best else None,
            "eq_batch_tps": round(eq.tps, 1) if eq else None,
            "eq_batch_tpot_ms": round(eq.tpot * 1e3, 2) if eq else None,
            "tps_speedup": round(best.tps / base_fp.tps, 2) if best else None,
            "tpot_reduction": round(base_fp.tpot * 1e3 / best.tpot_ms, 2) if best else None,
            "eq_tps_speedup": round(eq.tps / base_fp.tps, 2) if eq else None,
            "eq_tpot_reduction": round(base_fp.tpot / eq.tpot, 2) if eq else None,
        })
    return rows


def _print_table(rows: list[dict]) -> None:
    """`opt` columns are the max-TPS pick at its own searched batch (full HBM use).
    That TPOT belongs to that batch, so the latency comparison is printed from the
    equal-batch columns instead."""
    hdr = (f"{'precision':<34}{'PPL(q)':>8}{'GSM8K':>7}{'IFEval':>7}"
           f"{'base TPS':>10}{'base TPOT':>11}{'chips':>6}{'opt TPS':>9}{'opt TPOT':>10}"
           f"{'area':>8}{'TPS x':>7}{'eq TPS':>8}{'eq TPOT':>9}{'eq TPS x':>9}{'eq TPOT x':>10}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<34}{_f(r['ppl_quant'],3):>8}{_f(r['gsm8k'],3):>7}{_f(r['ifeval'],3):>7}"
              f"{r['base_tps']:>10}{r['base_tpot_ms']:>11}{r['base_chips']:>6}"
              f"{_f(r['codesign_tps'],1):>9}{_f(r['codesign_tpot_ms'],2):>10}"
              f"{_f(r['codesign_area_mm2'],3):>8}{_f(r['tps_speedup'],2):>7}"
              f"{_f(r['eq_batch_tps'],1):>8}{_f(r['eq_batch_tpot_ms'],2):>9}"
              f"{_f(r['eq_tps_speedup'],2):>9}{_f(r['eq_tpot_reduction'],2):>10}")
    print("=" * len(hdr))
    print("  base = FP16 decode on the shipped chip (same HBM generation, fixed channels; chips auto to fit)")
    print("  opt  = co-designed chip at ITS searched batch (max-TPS pick; its TPOT belongs to that batch)")
    print("  eq   = the same co-designed chip at the baseline batch — the iso-workload speed-ups")


def _f(v, nd):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode end-to-end table (with vs without optimisation).")
    ap.add_argument("config", help="decode DSE config JSON")
    ap.add_argument("--software-csv", default=None)
    ap.add_argument("--n-trials", type=int, default=128)
    ap.add_argument("--tpot-budget-ms", type=float, default=None,
                    help="cap TPOT when picking the 'with-optimisation' config (default: max TPS)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(cfg.get("output_dir", "results/decode_dse")) / cfg["model_name"].split("/")[-1]
    csv_path = Path(args.software_csv) if args.software_csv else out_dir / "software_disagg_decode.csv"

    rows = build_table(cfg, csv_path, n_trials=args.n_trials, tpot_budget_ms=args.tpot_budget_ms)
    _print_table(rows)

    out = Path(args.out) if args.out else out_dir / "end_to_end.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
