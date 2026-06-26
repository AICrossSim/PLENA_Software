"""Decode-chip precision DSE for Qwen3-32B

Bayesian (Optuna TPE) multi-objective search over the PER-COMPONENT precision space -- attention-
weight, FFN-weight, and KV each searched independently (activations are pinned to 8-bit MX, since
they stay on-chip and cost no HBM bandwidth) -- minimising (continuation-PPL proxy, decode MB/token).

The fast PPL proxy lets the search explore hundreds of points. Then GPTQ is applied to AGGRESSIVE
low-bit configs (which RTN can't handle and never reach the front) to rescue them, with block-wise
clip + selective rotation. Finally STRICT IFEval accuracy (prompt + instruction) is measured on the
Pareto frontier + the rescued points.
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

from decode_dse_qwen import disagg_serve, quant, ifeval

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PPL_PENALTY = 1.0e6   # returned for a crashed trial so the study survives


# search space
def suggest_spec(trial) -> dict:
    """Sample one per-component precision point: attention-weight, FFN-weight, and KV searched
    independently"""
    fmt = trial.suggest_categorical("fmt", ["mxint", "mxfp"])
    block = trial.suggest_categorical("block", quant.BLOCK_SIZES)
    # NOTE: param names are format-suffixed ("attn_w_mxint" vs "attn_w_mxfp") because Optuna forbids
    # one categorical name having two value spaces (ints for MXINT, format-strings for MXFP).
    if fmt == "mxint":
        w = lambda n: trial.suggest_categorical(f"{n}_mxint", quant.MXINT_WIDTHS)
    else:
        F, keys = quant.MXFP_FORMATS, list(quant.MXFP_FORMATS)
        w = lambda n: F[trial.suggest_categorical(f"{n}_mxfp", keys)]
    return {"fmt": fmt, "block": block, "attn_w": w("attn_w"), "ffn_w": w("ffn_w"), "kv": w("kv")}


def _spec_json(spec: dict) -> dict:
    """JSON-safe spec for Optuna user_attrs (tuples -> lists)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}


def _spec_from_json(d: dict) -> dict:
    """Rebuild a spec (lists -> tuples for MXFP widths)."""
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}


# objective (PPL proxy)
class Objective:
    """Scores one sampled precision: re-pick the freest GPUs, quantise the decode chip there, score
    PPL on the cached prefill KV, free"""

    def __init__(self, ppl_caches, dims, gpus_n, dtype):
        self.ppl_caches, self.dims, self.gpus_n, self.dtype = ppl_caches, dims, gpus_n, dtype

    def __call__(self, trial):
        spec = suggest_spec(trial)
        cost = quant.decode_cost(self.dims, spec, batch=16, s_in=256, s_out=16384)
        tag = quant.spec_tag(spec)
        trial.set_user_attr("spec", _spec_json(spec))
        trial.set_user_attr("tag", tag)
        for k, v in cost.items():
            trial.set_user_attr(k, v)

        decode = None
        try:
            gpus = disagg_serve.select_gpus(self.gpus_n)             # migrate to whatever is free now
            t0 = time.time()
            decode = disagg_serve.quantize_decode_model(spec, gpus, self.dtype, verbose=False)
            ppl = disagg_serve.continuation_ppl(decode, self.ppl_caches, spec, gpus)
            trial.set_user_attr("eval_seconds", round(time.time() - t0, 1))
            print(f"  [trial {trial.number:>4}] {tag:<54} PPL={ppl:8.2f} "
                  f"cost={cost['cost_mb_per_token']:7.0f} MB/tok", flush=True)
            return float(ppl), float(cost["cost_mb_per_token"])
        except Exception as e:                                       # keep the study alive
            print(f"  [trial {trial.number:>4}] {tag:<54} FAILED: {type(e).__name__}: {e}", flush=True)
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            return PPL_PENALTY, float(cost["cost_mb_per_token"])
        finally:
            if decode is not None:
                disagg_serve.free_decode(decode)
                del decode
            gc.collect()
            torch.cuda.empty_cache()


def enqueue_anchors(study):
    """Seed known-good per-component configs so the front is populated from trial 0 (activations are
    pinned, so anchors set only the searched axes: weights + KV + block + fmt)."""
    for a in [   # keys must match the format-suffixed param names in suggest_spec
        dict(fmt="mxint", block=32, attn_w_mxint=8, ffn_w_mxint=8, kv_mxint=8),   # near-lossless
        dict(fmt="mxint", block=32, attn_w_mxint=4, ffn_w_mxint=4, kv_mxint=4),   # uniform W4
        dict(fmt="mxint", block=32, attn_w_mxint=8, ffn_w_mxint=4, kv_mxint=8),   # cheap FFN only
        dict(fmt="mxint", block=32, attn_w_mxint=4, ffn_w_mxint=8, kv_mxint=4),   # cheap attn + KV
        dict(fmt="mxfp",  block=32, attn_w_mxfp="E4M3", ffn_w_mxfp="E2M1", kv_mxfp="E4M3"),
    ]:
        try:
            study.enqueue_trial(a, skip_if_exists=True)
        except Exception as e:
            print(f"  [seed] could not enqueue {a}: {e}")


# frontier selection
def trial_rows(study) -> list[dict]:
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        ua = t.user_attrs
        rows.append({"trial": t.number, "tag": ua.get("tag", ""), "spec": ua.get("spec"),
                     "gptq": False, "cont_ppl": round(t.values[0], 3),
                     "cost_mb_per_token": round(t.values[1], 2),
                     "attn_w_bits": ua.get("attn_w_bits"), "ffn_w_bits": ua.get("ffn_w_bits"),
                     "kv_bits": ua.get("kv_bits"), "weight_MB_per_token": ua.get("weight_MB_per_token"),
                     "kv_MB_per_token": ua.get("kv_MB_per_token"), "error": ua.get("error", "")})
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated set minimising (cont_ppl, cost). Drops crashed/penalty points."""
    valid = [r for r in rows if not r.get("error") and r["cont_ppl"] < PPL_PENALTY / 2]
    front = [p for p in valid if not any(
        q["cont_ppl"] <= p["cont_ppl"] and q["cost_mb_per_token"] <= p["cost_mb_per_token"]
        and (q["cont_ppl"] < p["cont_ppl"] or q["cost_mb_per_token"] < p["cost_mb_per_token"])
        for q in valid)]
    return sorted(front, key=lambda r: r["cost_mb_per_token"])


def aggressive_gptq_specs(k: int, block: int, rotation: bool) -> list[dict]:
    """The AGGRESSIVE low-bit MXINT configs GPTQ should try to RESCUE"""
    grid = [{"fmt": "mxint", "block": block, "attn_w": w, "ffn_w": w, "kv": kv,
             "gptq": True, "clip_search_y": True, "rotation": rotation}
            for w in (3, 4) for kv in (2, 3, 4)]
    grid.sort(key=lambda s: (s["ffn_w"], s["kv"]))   # ~cost order: lower weight + KV = cheaper
    return grid[:k]


def gptq_refine(specs, ppl_caches, dims, gpus_n, dtype, recipe):
    """GPTQ-calibrate each aggressive spec and score PPL on the cached prefill KV. Re-picks the freest
    GPUs per point; a failed point is isolated, so the search keeps its RTN frontier regardless."""
    rows = []
    for spec in specs:
        tag = quant.spec_tag(spec)
        print(f"\n  [gptq] calibrating {tag} ...", flush=True)
        decode = None
        try:
            gpus = disagg_serve.select_gpus(gpus_n)
            t0 = time.time()
            decode = disagg_serve.quantize_decode_model(spec, gpus, dtype, recipe=recipe, verbose=False)
            ppl = disagg_serve.continuation_ppl(decode, ppl_caches, spec, gpus)
            cost = quant.decode_cost(dims, spec, batch=16, s_in=256, s_out=16384)
            print(f"  [gptq] {tag:<54} PPL={ppl:8.2f} ({time.time()-t0:.0f}s)", flush=True)
            rows.append({"trial": -1, "tag": tag, "spec": _spec_json(spec), "gptq": True,
                         "cont_ppl": round(float(ppl), 3), "cost_mb_per_token": cost["cost_mb_per_token"],
                         "attn_w_bits": cost["attn_w_bits"], "ffn_w_bits": cost["ffn_w_bits"],
                         "kv_bits": cost["kv_bits"], "weight_MB_per_token": cost["weight_MB_per_token"],
                         "kv_MB_per_token": cost["kv_MB_per_token"], "error": ""})
        except Exception as e:
            print(f"  [gptq] {tag} FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            if decode is not None:
                disagg_serve.free_decode(decode); del decode
            gc.collect(); torch.cuda.empty_cache()
    return rows


# IFEval on the frontier
def ifeval_frontier(front_specs, tok, prompt_caches, dims, gpus_n, dtype, recipe, budget):
    """Score the frontier (+ an unquantised gold) with strict IFEval in thinking mode, on the cached
    prompts. Returns per-config accuracy rows and per-prompt records (for the length plot)."""
    sampling = dict(disagg_serve.THINKING_SAMPLING)
    rows, records = [], {}

    def run(label, decode, spec, gpus):
        agg, recs = ifeval.evaluate(decode, tok, prompt_caches, spec, gpus,
                                    max_new_tokens=budget, sampling=sampling)
        cost = quant.decode_cost(dims, spec, batch=16, s_in=256, s_out=16384)
        print(f"  [ifeval] {label:<54} prompt={agg['strict_prompt_acc']:.3f} "
              f"inst={agg['strict_inst_acc']:.3f}", flush=True)
        rows.append({"label": label, "tag": quant.spec_tag(spec), "gptq": bool(spec.get("gptq")),
                     "cost_mb_per_token": cost["cost_mb_per_token"], **agg})
        records[label] = recs

    # unquantised gold reference (bf16 decode chip)
    gpus = disagg_serve.select_gpus(gpus_n)
    gold = disagg_serve.load_model(dtype, gpus, attn_implementation="sdpa")
    gold_spec = {"fmt": "bf16", "block": 32, "attn_w": 16, "ffn_w": 16, "kv": "bf16"}
    try:
        run("GOLD (unquantised bf16)", gold, gold_spec, gpus)
    finally:
        disagg_serve.free_decode(gold); del gold; gc.collect(); torch.cuda.empty_cache()

    for spec in front_specs:
        decode = None
        try:
            gpus = disagg_serve.select_gpus(gpus_n)
            decode = disagg_serve.quantize_decode_model(spec, gpus, dtype, recipe=recipe, verbose=False)
            run(quant.spec_tag(spec), decode, spec, gpus)
        except Exception as e:
            print(f"  [ifeval] {quant.spec_tag(spec)} FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            if decode is not None:
                disagg_serve.free_decode(decode); del decode
            gc.collect(); torch.cuda.empty_cache()
    return rows, records


def _write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r if k != "spec"})
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)
    print(f"[write] {path}")


def plot_pareto(rows, front, ppl_budget):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    good = [r for r in rows if not r.get("error") and r["cont_ppl"] < PPL_PENALTY / 2]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for g, color, lbl in ((False, "steelblue", "RTN"), (True, "seagreen", "GPTQ")):
        pts = [r for r in good if bool(r.get("gptq")) == g]
        ax.scatter([r["cost_mb_per_token"] for r in pts], [r["cont_ppl"] for r in pts],
                   s=70 if g else 16, marker="*" if g else "o", alpha=0.85 if g else 0.35,
                   color=color, edgecolor="k" if g else "none", linewidth=0.4, zorder=5 if g else 1,
                   label=f"{lbl} ({len(pts)})")
    if front:
        ax.plot([p["cost_mb_per_token"] for p in front], [p["cont_ppl"] for p in front],
                "o-", color="darkorange", lw=2, ms=6, zorder=6, label="Pareto front")
    ax.axhline(ppl_budget, ls="--", color="gray", alpha=0.6, label=f"PPL budget = {ppl_budget:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Decode memory traffic (MB / token)")
    ax.set_ylabel("Continuation perplexity (log)")
    ax.set_title("Qwen3-32B decode chip: Accuracy vs Memory bandwidth")
    ax.grid(True, alpha=0.25, which="both"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "pareto.png", dpi=150); plt.close(fig)
    print(f"[write] {RESULTS / 'pareto.png'}")


def plot_accuracy_vs_length(records):
    """Strict instruction accuracy vs binned output length: gold reference vs best-quantised."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gold = next((k for k in records if k.startswith("GOLD")), None)
    quantised = [k for k in records if k != gold]
    if not gold or not quantised:
        return
    # best-quantised = highest overall strict instruction accuracy
    best = max(quantised, key=lambda k: ifeval.aggregate(records[k])["strict_inst_acc"])
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for label, style, color in ((gold, "o--", "gray"), (best, "o-", "darkorange")):
        bins = ifeval.length_bins(records[label])
        if bins:
            ax.plot([b["mid"] for b in bins], [b["acc"] for b in bins], style, color=color, lw=2,
                    ms=7, label=("gold (unquantised)" if label == gold else f"best quantised\n{best}"))
    ax.set_xscale("log")
    ax.set_xlabel("Output length (Binned)")
    ax.set_ylabel("Strict instruction accuracy")
    ax.set_title("Accuracy vs Output Length: Gold vs Best-Quantised (IFEval)")
    ax.grid(True, alpha=0.25, which="both"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "accuracy_vs_length.png", dpi=150); plt.close(fig)
    print(f"[write] {RESULTS / 'accuracy_vs_length.png'}")


def export(rows, ifeval_rows, records, ppl_budget):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["cont_ppl"], r["cost_mb_per_token"]))
    _write_csv(RESULTS / "trials.csv", rows)
    front = [p for p in pareto_front(rows) if p["cont_ppl"] <= ppl_budget]
    (RESULTS / "pareto.json").write_text(json.dumps(
        [{"tag": p["tag"], "cont_ppl": p["cont_ppl"], "cost_mb_per_token": p["cost_mb_per_token"],
          "gptq": p["gptq"], "spec": p["spec"]} for p in front], indent=2))
    print(f"[write] {RESULTS / 'pareto.json'}  ({len(front)} points)")
    if ifeval_rows:
        _write_csv(RESULTS / "frontier_ifeval.csv", ifeval_rows)
    plot_pareto(rows, front, ppl_budget)
    if records:
        plot_accuracy_vs_length(records)
    return front


def _checkpoint_csv(study, trial):
    """Dump trials.csv every 10 trials so a long unattended run always has fresh partial results."""
    if trial.number % 10 == 0:
        RESULTS.mkdir(parents=True, exist_ok=True)
        _write_csv(RESULTS / "trials.csv",
                   sorted(trial_rows(study), key=lambda r: (r["cont_ppl"], r["cost_mb_per_token"])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--n-trials", type=int, default=250, help="TOTAL TPE trials to reach (resume-aware)")
    ap.add_argument("--decode-gpus", type=int, default=2, help="GPUs per chip (prefill is transient, then freed)")
    ap.add_argument("--chunks", type=int, default=48, help="WikiText chunks for continuation PPL")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--storage", default=None, help="optuna URL (sqlite:///...) for resume; default in-memory")
    ap.add_argument("--study-name", default="decode_percomponent")
    ap.add_argument("--refine-gptq", type=int, default=3,
                    help="GPTQ-rescue K aggressive low-bit MXINT configs (W3/4 x KV2/3/4), cheapest first")
    ap.add_argument("--gptq-rotation", type=int, default=1,
                    help="1=add selective rotation (rotation_search) to the GPTQ stack; 0=GPTQ+clip only")
    ap.add_argument("--ifeval-subset", type=int, default=64, help="IFEval prompts for the frontier eval (0=skip)")
    ap.add_argument("--ifeval-topk", type=int, default=4, help="evaluate the K best-PPL frontier pts on IFEval")
    ap.add_argument("--ifeval-budget", type=int, default=32768, help="thinking budget for the frontier eval")
    ap.add_argument("--ppl-budget", type=float, default=30.0, help="recommended-front PPL ceiling")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import optuna, transformers
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    transformers.set_seed(args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    dims = disagg_serve.load_dims()
    half = args.chunk_len // 2
    gpus0 = disagg_serve.select_gpus(args.decode_gpus)

    print("=" * 96)
    print(f"  Qwen3-32B decode per-component precision DSE   sampler: TPE (Bayesian, multi-objective)")
    print(f"  objectives: (continuation PPL, MB/token)   GPUs/chip: {args.decode_gpus} (re-picked per trial)")
    print(f"  n_trials(target)={args.n_trials}  gptq_rescue={args.refine_gptq}"
          f"(rotation={'on' if args.gptq_rotation else 'off'})  ifeval_subset={args.ifeval_subset}")
    print("=" * 96)

    tok = disagg_serve.load_tokenizer()
    print(f"loading bf16 prefill on {gpus0} (transient: builds the KV caches, then freed) ...", flush=True)
    prefill = disagg_serve.load_prefill(gpus0, dtype)
    chunks = disagg_serve.wikitext_chunks(tok, disagg_serve.first_device(gpus0), args.chunks, args.chunk_len)
    print(f"  prefilling {len(chunks)} PPL chunks + {args.ifeval_subset} IFEval prompts ...", flush=True)
    ppl_caches = disagg_serve.precompute_ppl_caches(prefill, chunks, half, gpus0)
    prompt_caches = (disagg_serve.precompute_prompt_caches(
        prefill, tok, ifeval.load_ifeval(args.ifeval_subset), gpus0, enable_thinking=True)
        if args.ifeval_subset > 0 else [])
    disagg_serve.free_decode(prefill); del prefill
    gc.collect(); torch.cuda.empty_cache()
    print("  prefill freed -> search runs on the decode chip only.", flush=True)

    study = optuna.create_study(study_name=args.study_name, storage=args.storage, load_if_exists=True,
                                directions=["minimize", "minimize"],
                                sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True))
    enqueue_anchors(study)
    done = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.n_trials - done)
    print(f"  study: {done} trials already done; running {remaining} more toward target {args.n_trials}.", flush=True)
    if remaining:
        study.optimize(Objective(ppl_caches, dims, args.decode_gpus, dtype),
                       n_trials=remaining, gc_after_trial=True, callbacks=[_checkpoint_csv])

    rows = trial_rows(study)
    from quant_eval.quantize import load_quant_config
    recipe = load_quant_config(str(quant.GPTQ_RECIPE))
    if args.refine_gptq > 0:
        # GPTQ targets AGGRESSIVE off-front configs to rescue (not the cheapest front points).
        specs = aggressive_gptq_specs(args.refine_gptq, block=32, rotation=bool(args.gptq_rotation))
        rows += gptq_refine(specs, ppl_caches, dims, args.decode_gpus, dtype, recipe)
    front = pareto_front(rows)

    ifeval_rows, records = [], {}
    if prompt_caches:
        # confirm the K best-PPL frontier points in thinking mode (the deployed metric)
        chosen = sorted([f for f in front if f["cont_ppl"] <= args.ppl_budget],
                        key=lambda r: r["cont_ppl"])[:args.ifeval_topk]
        front_specs = [_spec_from_json(p["spec"]) | ({"gptq": True, "clip_search_y": True} if p["gptq"] else {})
                       for p in chosen]
        ifeval_rows, records = ifeval_frontier(front_specs, tok, prompt_caches, dims, args.decode_gpus,
                                               dtype, recipe, args.ifeval_budget)

    export(rows, ifeval_rows, records, args.ppl_budget)
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
