"""Multi-objective co-design search over the decode-chip quantization space
Sampler: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
Instead of finding one best config find a set of good compromises
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch
import transformers

from decode_dse.quant import (
    MODEL_NAME,
    MXINT_WIDTHS,
    MXFP_FORMATS,
    BLOCK_SIZES,
    GPTQ_VALIDATED_SPEC,
    build_pass_args,
    build_gptq_pass_args,
    decode_cost,
    width_label,
)
from decode_dse.decode_eval import (
    load_prefill,
    quantize_decode_model,
    continuation_ppl,
    wikitext_chunks,
)

HERE        = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
GPTQ_RECIPE = HERE / "configs" / "w4_mxint_gptq_rot.toml"

PPL_PENALTY = 1.0e6  # To avoid crashing


# -----------------------------------------------------------------------------
# Search-space sampling (RTN)
# -----------------------------------------------------------------------------
def suggest_spec(trial) -> dict:
    """Sample one RTN decode-precision point."""
    fmt = trial.suggest_categorical("fmt", ["mxint", "mxfp"])
    block = trial.suggest_categorical("block", BLOCK_SIZES)
    if fmt == "mxint":
        return {"fmt": fmt, "block": block,
                "w":  trial.suggest_categorical("w_mxint", MXINT_WIDTHS),
                "a":  trial.suggest_categorical("a_mxint", MXINT_WIDTHS),
                "kv": trial.suggest_categorical("kv_mxint", MXINT_WIDTHS)}
    keys = list(MXFP_FORMATS)
    wk = trial.suggest_categorical("w_mxfp", keys)
    ak = trial.suggest_categorical("a_mxfp", keys)
    kk = trial.suggest_categorical("kv_mxfp", keys)
    return {"fmt": fmt, "block": block,
            "w": MXFP_FORMATS[wk], "a": MXFP_FORMATS[ak], "kv": MXFP_FORMATS[kk]}


def spec_tag(spec: dict, gptq: bool = False) -> str:
    g = "(gptq)" if gptq else ""
    return (f"W:{width_label(spec['fmt'], spec['w'])}{g} "
            f"A:{width_label(spec['fmt'], spec['a'])} "
            f"KV:{width_label(spec['fmt'], spec['kv'])} b{spec['block']}")


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------
class Objective:
    """Holds the fixed prefill chip + eval chunks so trials don't reload them."""

    def __init__(self, prefill, chunks, half, device, dtype):
        self.prefill, self.chunks, self.half = prefill, chunks, half
        self.device, self.dtype = device, dtype

    def __call__(self, trial) -> tuple[float, float]:
        spec = suggest_spec(trial)
        cost = decode_cost(spec["fmt"], spec["w"], spec["a"], spec["kv"], spec["block"])
        tag = spec_tag(spec)

        trial.set_user_attr("tag", tag)
        trial.set_user_attr("fmt", spec["fmt"])
        trial.set_user_attr("block", spec["block"])
        trial.set_user_attr("gptq", False)
        trial.set_user_attr("w_label", width_label(spec["fmt"], spec["w"]))
        trial.set_user_attr("a_label", width_label(spec["fmt"], spec["a"]))
        trial.set_user_attr("kv_label", width_label(spec["fmt"], spec["kv"]))
        if spec["fmt"] == "mxint":     # raw ints, so the GPTQ refine can rebuild the spec
            trial.set_user_attr("w_raw", spec["w"])
            trial.set_user_attr("a_raw", spec["a"])
            trial.set_user_attr("kv_raw", spec["kv"])
        for k, v in cost.items():
            trial.set_user_attr(k, v)

        model = None
        try:
            pass_args = build_pass_args(spec["fmt"], spec["w"], spec["a"], spec["kv"], spec["block"])
            t0 = time.time()
            _, model = quantize_decode_model(pass_args, self.device, self.dtype, verbose=False)
            kv_cfg = {"fmt": spec["fmt"], "bits": spec["kv"]}
            ppl = continuation_ppl(self.prefill, model, self.chunks, kv_cfg, self.half)
            trial.set_user_attr("eval_seconds", round(time.time() - t0, 1))
            print(f"  [trial {trial.number:>4}] {tag:<46}  "
                  f"PPL={ppl:8.3f}  cost={cost['cost_mb_per_token']:7.1f} MB/tok", flush=True)
            return float(ppl), float(cost["cost_mb_per_token"])
        except Exception as e:  # noqa: BLE001 - keep the study alive
            print(f"  [trial {trial.number:>4}] {tag:<46}  FAILED: {type(e).__name__}: {e}", flush=True)
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            return PPL_PENALTY, float(cost["cost_mb_per_token"])
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()


def enqueue_anchors(study) -> None:
    """Seed known-good RTN configs so the front is populated from trial 0."""
    anchors = [
        {"fmt": "mxint", "block": 32, "w_mxint": 8, "a_mxint": 8, "kv_mxint": 8},  # near-lossless
        {"fmt": "mxint", "block": 32, "w_mxint": 4, "a_mxint": 8, "kv_mxint": 4},  # aggressive W4/A8/KV4
        {"fmt": "mxint", "block": 32, "w_mxint": 4, "a_mxint": 8, "kv_mxint": 8},  # W4 only
        {"fmt": "mxfp",  "block": 32, "w_mxfp": "E4M3", "a_mxfp": "E4M3", "kv_mxfp": "E4M3"},
        {"fmt": "mxfp",  "block": 32, "w_mxfp": "E2M1", "a_mxfp": "E4M3", "kv_mxfp": "E2M1"},
    ]
    for a in anchors:
        try:
            study.enqueue_trial(a, skip_if_exists=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [seed] could not enqueue {a}: {e}")


def eval_gptq_specs(prefill, chunks, half, device, dtype, recipe, specs) -> list[dict]:
    """GPTQ-calibrate each (w, a, kv, block) spec and score continuation PPL"""
    rows = []
    for (w, a, kv, block) in specs:
        cost = decode_cost("mxint", w, a, kv, block)
        tag = f"W:MXINT{w}(gptq) A:MXINT{a} KV:MXINT{kv} b{block}"
        print(f"\n  [gptq] calibrating {tag} ...", flush=True)
        model = None
        try:
            pa = build_gptq_pass_args(recipe, w, a, kv, block)
            t0 = time.time()
            _, model = quantize_decode_model(pa, device, dtype, verbose=False)
            ppl = continuation_ppl(prefill, model, chunks, {"fmt": "mxint", "bits": kv}, half)
            secs = time.time() - t0
            print(f"  [gptq] {tag:<46}  PPL={ppl:8.3f}  cost={cost['cost_mb_per_token']:7.1f} MB/tok "
                  f"({secs:.0f}s)", flush=True)
            rows.append({
                "trial": -1, "tag": tag, "fmt": "mxint", "block": block, "gptq": True,
                "W": f"MXINT{w}", "A": f"MXINT{a}", "KV": f"MXINT{kv}",
                "cont_ppl": round(float(ppl), 4),
                "cost_mb_per_token": cost["cost_mb_per_token"],
                "w_eff_bits": cost["w_eff_bits"], "a_eff_bits": cost["a_eff_bits"],
                "kv_eff_bits": cost["kv_eff_bits"],
                "weight_MB_per_token": cost["weight_MB_per_token"],
                "kv_MB_per_token": cost["kv_MB_per_token"],
                "eval_seconds": round(secs, 1), "error": "",
            })
        except Exception as e:  # noqa: BLE001
            print(f"  [gptq] {tag} FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def select_refine_specs(rows: list[dict], k: int, ppl_budget: float) -> list[tuple]:
    """Pick the K GPTQ-refine targets most likely to pay off"""
    floor = min((r["cont_ppl"] for r in rows if not r.get("error")), default=ppl_budget)
    lo, hi = 1.1 * floor, 5.0 * ppl_budget # recoverable band
    cands = []
    for r in rows:
        if r.get("gptq") or r.get("fmt") != "mxint" or r.get("error"):
            continue
        try:
            w, a, kv = (int(str(r[x]).replace("MXINT", "")) for x in ("W", "A", "KV"))
        except ValueError:
            continue
        # weights low enough for GPTQ to matter; activations stay at 8 (GPTQ does
        # not help activations, and lowering A buys no decode bandwidth anyway).
        if w <= 4 and a >= 8 and lo <= r["cont_ppl"] <= hi:
            cands.append((r["cost_mb_per_token"], (w, a, kv, int(r["block"]))))

    seen, out = set(), []
    for _, spec in sorted(cands):                   # cheapest first
        if spec != GPTQ_VALIDATED_SPEC and spec not in seen:
            seen.add(spec)
            out.append(spec)
        if len(out) >= k:
            break
    return out


def dedupe(rows: list[dict]) -> list[dict]:
    """Collapse identical (fmt, W, A, KV, block, gptq) configs to their best PPL.
    NSGA-II resamples categorical points, so the raw trials contain duplicates."""
    best = {}
    for r in rows:
        if r.get("cont_ppl") is None:
            continue
        key = (r.get("fmt"), r.get("W"), r.get("A"), r.get("KV"), r.get("block"), bool(r.get("gptq")))
        cur = best.get(key)
        if cur is None or r["cont_ppl"] < cur["cont_ppl"]:
            best[key] = r
    return list(best.values())


def load_rows_from_csv(path: str) -> list[dict]:
    """Rebuild rows from an existing trials.csv (regenerate front/plot, no re-run)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("cont_ppl", "cost_mb_per_token", "w_eff_bits", "a_eff_bits", "kv_eff_bits",
                      "weight_MB_per_token", "kv_MB_per_token", "eval_seconds"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = None
            try:
                r["block"] = int(r["block"])
            except (TypeError, ValueError):
                pass
            r["gptq"] = str(r.get("gptq")).lower() in ("true", "1")
            rows.append(r)
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated set minimizing (cont_ppl, cost_mb_per_token)."""
    valid = [r for r in rows if not r.get("error") and r["cont_ppl"] < PPL_PENALTY / 2]
    front = []
    for p in valid:
        dominated = any(
            q["cont_ppl"] <= p["cont_ppl"] and q["cost_mb_per_token"] <= p["cost_mb_per_token"]
            and (q["cont_ppl"] < p["cont_ppl"] or q["cost_mb_per_token"] < p["cost_mb_per_token"])
            for q in valid)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda r: r["cost_mb_per_token"])


def trial_rows(study) -> list[dict]:
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        ua = t.user_attrs
        rows.append({
            "trial": t.number, "tag": ua.get("tag", ""),
            "fmt": ua.get("fmt"), "block": ua.get("block"), "gptq": ua.get("gptq", False),
            "W": ua.get("w_label"), "A": ua.get("a_label"), "KV": ua.get("kv_label"),
            "cont_ppl": round(t.values[0], 4), "cost_mb_per_token": round(t.values[1], 3),
            "w_eff_bits": ua.get("w_eff_bits"), "a_eff_bits": ua.get("a_eff_bits"),
            "kv_eff_bits": ua.get("kv_eff_bits"),
            "weight_MB_per_token": ua.get("weight_MB_per_token"),
            "kv_MB_per_token": ua.get("kv_MB_per_token"),
            "eval_seconds": ua.get("eval_seconds"), "error": ua.get("error", ""),
        })
    return rows


def export_results(rows: list[dict], ppl_budget: float = 30.0) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = dedupe(rows)
    rows = sorted(rows, key=lambda r: (r["cont_ppl"], r["cost_mb_per_token"]))

    csv_path = RESULTS_DIR / "trials.csv"
    with csv_path.open("w", newline="") as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"\n[write] {csv_path}  ({len(rows)} unique configs)")

    # The recommended front keeps only usable points (PPL <= budget); the cheap-but-
    # broken low-bit corner is Pareto-optimal by domination but useless to deploy.
    front = [p for p in pareto_front(rows) if p["cont_ppl"] <= ppl_budget]
    points = []
    for p in front:
        tag = (f"{p['W']}_{p['A']}_{p['KV']}" + ("_gptq" if p.get("gptq") else "")
               ).replace("MXINT", "I").replace("MXFP_", "F")
        points.append({
            "tag": tag, "fmt": "Mx",
            "weight_bits": p.get("w_eff_bits"), "act_bits": p.get("a_eff_bits"),
            "kv_bits": p.get("kv_eff_bits"),
            "cont_ppl": p["cont_ppl"], "cost_mb_per_token": p["cost_mb_per_token"],
            "block": p.get("block"), "gptq": p.get("gptq", False), "note": p.get("tag"),
        })
    (RESULTS_DIR / "pareto.json").write_text(json.dumps(points, indent=2))
    print(f"[write] {RESULTS_DIR / 'pareto.json'}  ({len(points)} Pareto points)")

    print("\n" + "=" * 94)
    print("  PARETO FRONT (accuracy vs decode byte-traffic) — paste into analytic PRECISION_POINTS")
    print("=" * 94)
    print(f"  {'tag':<26} | {'cont.PPL':>8} | {'cost MB/tok':>11} | {'W_b':>5} {'A_b':>5} {'KV_b':>5} | note")
    print("  " + "-" * 90)
    for p in sorted(points, key=lambda q: q["cont_ppl"]):
        print(f"  {p['tag']:<26} | {p['cont_ppl']:>8.3f} | {p['cost_mb_per_token']:>11.1f} | "
              f"{p['weight_bits']:>5} {p['act_bits']:>5} {p['kv_bits']:>5} | {p['note']}")
    print("=" * 94)
    _plot(rows, front, ppl_budget)


def _plot(rows, front, ppl_budget) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed; skipping scatter")
        return
    good = [r for r in rows if not r.get("error") and r["cont_ppl"] < PPL_PENALTY / 2]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for fmt, color, lbl in (("mxint", "steelblue", "MXINT (RTN)"),
                            ("mxfp", "indianred", "MXFP (RTN)")):
        pts = [r for r in good if r["fmt"] == fmt and not r.get("gptq")]
        if pts:
            ax.scatter([r["cost_mb_per_token"] for r in pts], [r["cont_ppl"] for r in pts],
                       s=16, alpha=0.35, color=color, label=f"{lbl} ({len(pts)})")
    gptq = [r for r in good if r.get("gptq")]
    if gptq:
        ax.scatter([r["cost_mb_per_token"] for r in gptq], [r["cont_ppl"] for r in gptq],
                   s=70, marker="*", color="seagreen", edgecolor="k", zorder=5,
                   label=f"MXINT + GPTQ ({len(gptq)})")
    pf = sorted(front, key=lambda p: p["cost_mb_per_token"])
    ax.plot([p["cost_mb_per_token"] for p in pf], [p["cont_ppl"] for p in pf],
            "o-", color="darkorange", linewidth=2, markersize=5,
            label=f"usable Pareto front ({len(pf)})", zorder=6)
    for p in pf:
        ax.annotate(p["tag"], (p["cost_mb_per_token"], p["cont_ppl"]),
                    fontsize=7, alpha=0.9, rotation=18, xytext=(3, 4), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(min(r["cont_ppl"] for r in good) * 0.95, max(60.0, ppl_budget * 3.0))  # usable knee
    ax.axhline(ppl_budget, ls=":", color="gray", alpha=0.7, label=f"usable budget = PPL {ppl_budget:g}")
    ax.set_xlabel("MB / token")
    ax.set_ylabel("Continuation Perplexity (Log)")
    ax.set_title("Decode co-design search: Accuracy vs Decode cost\n"
                 "Llama-3.2-1B")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[write] {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--n-trials", type=int, default=300, help="RTN trials THIS process runs")
    ap.add_argument("--chunks", type=int, default=80, help="eval chunks for continuation PPL")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///.../study.db "
                         "(enables resume + multi-GPU fan-out). Default: in-memory.")
    ap.add_argument("--study-name", default="decode_codesign")
    ap.add_argument("--refine-gptq", type=int, default=0, metavar="K",
                    help="after the search, GPTQ-calibrate the K best low-bit frontier points "
                         "(~15-30 min each). 0 = only the cached W4/A8/KV4 anchor.")
    ap.add_argument("--export-only", action="store_true",
                    help="skip the RTN search; just re-export (and --refine-gptq) an existing study")
    ap.add_argument("--ppl-budget", type=float, default=30.0,
                    help="accuracy budget: recommended front + plot keep only configs with PPL <= this")
    ap.add_argument("--from-csv", default=None,
                    help="regenerate front/plot/json from an existing trials.csv (no model run)")
    args = ap.parse_args()

    # Fast path: rebuild the cleaned front + plot from existing results, no GPU needed.
    if args.from_csv:
        export_results(load_rows_from_csv(args.from_csv), args.ppl_budget)
        return 0

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage, load_if_exists=True,
        directions=["minimize", "minimize"], sampler=optuna.samplers.NSGAIISampler(seed=0))

    dmap = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dmap[args.dtype]
    half = args.chunk_len // 2

    print("=" * 94)
    print(f"  Decode co-design search — {MODEL_NAME}")
    print(f"  objectives: (continuation PPL, decode MB/token)   sampler: NSGA-II")
    print(f"  device={args.device}  n_trials={0 if args.export_only else args.n_trials}  "
          f"chunks={args.chunks}  refine_gptq={args.refine_gptq}  storage={args.storage or 'in-memory'}")
    print("=" * 94)
    transformers.set_seed(0)

    tok, prefill = load_prefill(args.device, dtype)
    chunks = wikitext_chunks(tok, args.device, args.chunks, args.chunk_len)

    if not args.export_only:
        enqueue_anchors(study)
        study.optimize(Objective(prefill, chunks, half, args.device, dtype),
                       n_trials=args.n_trials, gc_after_trial=True)

    # --- GPTQ post-step: cached anchor (cheap) + optional refine of the front ---
    rows = trial_rows(study)
    from quant_eval.quantize import load_quant_config
    recipe = load_quant_config(str(GPTQ_RECIPE))
    specs = [GPTQ_VALIDATED_SPEC]                             # validated, cache-hit
    if args.refine_gptq > 0:
        specs += select_refine_specs(rows, args.refine_gptq, args.ppl_budget)
    rows += eval_gptq_specs(prefill, chunks, half, args.device, dtype, recipe, specs)

    export_results(rows, args.ppl_budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
