"""Re-score a finished Stage-A CSV under different matrix-timing models.

The matrix-op cycle model changed after the llama8b run. The open question is
pipelined (`customISA_lib.json` "pipelined" column, an ideal fully-pipelined
array) vs serialized (the current RTL control's stall-behind-active-MCU
reading). Until the RTL microbenchmark settles it, this tool recomputes every
row's reference TPS/TPOT under both ISA libraries and reports how much the
numbers and the PPL-vs-TPS front move.

No model evaluation involved — pure analytic re-scoring, runs in seconds.

Usage:
  python -m decode_dse.software.rescore_timing \
      --config decode_dse/configs/llama3_8b.json \
      [--isa-a path/to/pipelined.json] [--isa-b path/to/serialized.json] \
      [--out results/decode_dse/<model>/rescore_timing.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from decode_dse.simulator_bridge import DecodeSimulator, resolve_model_json  # noqa: F401


def _ref_metrics(sim: DecodeSimulator, row: dict, workload: dict, base_hbm: dict):
    """Same reference-chip metrics as run_software_dse._ref_metrics, kept
    torch-free so re-scoring never loads a model."""
    act = row.get("act_bits")
    prec = sim.precision_from_eff_bits(
        float(row["attn_w_bits"]), float(row["ffn_w_bits"]), float(row["kv_bits"]),
        act_bits=float(act) if act not in (None, "") else None,
        block=int(float(row.get("block") or 32)),
    )
    over = sim.shipped_over(prec, base_hbm)
    m_tps = sim.evaluate(prec, batch=workload["batch"], input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over)
    if not m_tps.fits_in_hbm:
        return 0.0, float("inf")
    m_lat = sim.evaluate(prec, batch=1, input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over)
    return round(m_tps.tps, 2), round(m_lat.tpot * 1e3, 3)


def _pareto_tags(rows: list[dict], tps_key: str) -> set[str]:
    """Tags on the (min ppl, max tps) Pareto front."""
    front = set()
    for r in rows:
        if not r.get("cont_ppl") or not r.get(tps_key):
            continue
        dominated = any(
            float(o["cont_ppl"]) <= float(r["cont_ppl"])
            and float(o[tps_key]) >= float(r[tps_key])
            and (float(o["cont_ppl"]), float(o[tps_key]))
            != (float(r["cont_ppl"]), float(r[tps_key]))
            for o in rows
            if o.get("cont_ppl") and o.get(tps_key)
        )
        if not dominated:
            front.add(r["tag"])
    return front


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", default=None, help="Stage-A CSV (default: from config out_dir)")
    ap.add_argument("--isa-a", default=None, help="ISA lib A (default: current customISA_lib.json)")
    ap.add_argument("--isa-b", default=None, help="ISA lib B (optional second timing model)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    model = cfg["model_name"].split("/")[-1]
    out_dir = Path(cfg.get("out_dir", f"results/decode_dse/{model}"))
    csv_path = Path(args.csv) if args.csv else out_dir / "software_disagg_decode.csv"
    rows = [r for r in csv.DictReader(open(csv_path)) if not r.get("error")]

    workload = cfg.get("workload", {"batch": 4, "input_seq": 1024, "output_seq": 3072})
    gen = cfg.get("baseline_hbm_gen", "HBM2")
    channels = int(cfg.get("baseline_hbm_channels", 32))

    sims = {}
    sims["A"] = DecodeSimulator(cfg.get("sim_model", model.lower()), isa_path=args.isa_a)
    if args.isa_b:
        sims["B"] = DecodeSimulator(cfg.get("sim_model", model.lower()), isa_path=args.isa_b)

    for r in rows:
        for label, sim in sims.items():
            base_hbm = sim.hbm_overrides(gen, channels)
            tps, tpot = _ref_metrics(sim, r, workload, base_hbm)
            r[f"tps_{label}"], r[f"tpot_ms_{label}"] = tps, tpot

    # Summary: distinct TPS values (degeneracy check) + front movement.
    old_front = _pareto_tags(rows, "ref_tps")
    print(f"rows re-scored: {len(rows)}")
    for label in sims:
        vals = sorted({r[f"tps_{label}"] for r in rows})
        front = _pareto_tags(rows, f"tps_{label}")
        kept = len(front & old_front)
        print(
            f"ISA {label}: {len(vals)} distinct TPS values "
            f"(min {vals[0]}, max {vals[-1]}); Pareto front size {len(front)}, "
            f"{kept}/{len(old_front)} of the original front retained"
        )

    out = Path(args.out) if args.out else out_dir / "rescore_timing.csv"
    keep = ["tag", "cont_ppl", "attn_w_bits", "ffn_w_bits", "kv_bits", "act_bits",
            "block", "ref_tps", "ref_tpot_ms"] + [
        k for label in sims for k in (f"tps_{label}", f"tpot_ms_{label}")
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
