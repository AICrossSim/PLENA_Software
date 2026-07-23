"""Publication figures for the decode-chip DSE.

Every figure in the decode study, ordered as the story reads:

  00  task length distributions   — why these workloads stress decode
  01  PPL vs reference-chip TPS    — headline accuracy/throughput front
  02  PPL vs reference-chip TPOT   — accuracy/latency view
  03  task accuracy vs TPS         — calibrated front only
  04  co-design TPS/TPOT frontier  — what the Stage B search buys
  05  right-sizing (TPS vs area)   — smallest chip at peak throughput
  06  batch sweep                  — throughput/latency vs serving batch
  07  HBM channel sweep            — bandwidth + capacity scaling
  08  roofline                     — memory- vs compute-bound at the design point
  09  precision heatmap            — W x KV accuracy map (log colours)
  10  with/without optimisation    — the end-to-end table as a figure
  11  per-task serving batch       — ideal decode batch per workload
  12  FP_SETTING ablation          — vector-unit width vs PPL
  13  KV hand-off                  — quantise-on-write shrinks prefill->decode TTFT
  14  K/V residency                — measured HBM-read cut vs context
  15  serialization gap            — RTL matmul cost vs the pipelined ideal
  16  disaggregated serving Pareto — throughput-per-chip vs per-user rate

Core set: 01/02 (accuracy fronts), 08 (roofline), 09 (precision map), 10
(end-to-end), and the decode-specific 13/14/15. The rest are supporting
diagnostics; the final listing groups the two.

Figures are saved as PNGs under ``<output_dir>/<model>/plots/``. Run after
Stage A (01/02/09/12 need ``software_disagg_decode.csv``) and Stage B (10 needs
``end_to_end.json``); 14 needs ``kv_residency_measured.csv``. A figure whose
input is missing is skipped, so partial pipelines still render the rest.

    python -m decode_dse.plots decode_dse/configs/llama3_8b.json

Style: a colourblind-safe palette (one hue per role), recessive grid and axes,
direct labels on the points that matter, no dual axes. Diverged precisions (PPL
above ``DIVERGED_PPL``) are counted in an annotation, not plotted.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from decode_dse.results import (  # noqa: E402
    accuracy_throughput_front,
    load_software_rows,
    prefer_best_method,
    select_front,
)
from decode_dse.simulator_bridge import DecodeSimulator  # noqa: E402
from decode_dse.hardware.codesign_search import HardwareSpace, search_precision  # noqa: E402

# --- colourblind-safe palette (light surface) -----
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MEM, COMPUTE = "#eb6834", "#2a78d6"          # bottleneck: memory=orange, compute=blue
BASE_C, QUANT_C, OPT_C = "#898781", "#1baf7a", "#2a78d6"  # baseline / quant-only / co-design
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

DIVERGED_PPL = 100.0   # above this a precision is unusable; annotate, don't plot


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5, "axes.titlesize": 12.5, "axes.labelsize": 11,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.labelcolor": INK2, "text.color": INK, "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
    })


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, color=INK, fontweight="bold", pad=16 if subtitle else 8, loc="left")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")


def _fmt_w(v: str) -> str:
    """'i4' -> '4', 'E2M1' stays (MXINT widths shown as plain bits)."""
    return v[1:] if v.startswith("i") and v[1:].isdigit() else v


def _short(tag: str) -> str:
    """Compact precision label from a run tag, e.g. 'W4/KV8/A4' or 'W4/KVE2M1/A4'."""
    parts = dict(p.split("-", 1) for p in tag.split("__")[1:] if "-" in p)
    return (f"W{_fmt_w(parts.get('aw', '?'))}"
            f"/KV{_fmt_w(parts.get('kv', '?'))}"
            f"/A{_fmt_w(parts.get('a', '?'))}")


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _annotate_no_clip(ax, x, y, text, dy=6):
    """A direct label that flips left near the right edge so it never clips."""
    x0, x1 = ax.get_xlim()
    if x > x0 + 0.72 * (x1 - x0):
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(-6, dy),
                    fontsize=8, color=INK2, ha="right")
    else:
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, dy),
                    fontsize=8, color=INK2, ha="left")


# ---------------------------------------------------------------------------
# 00. Task length distributions
# ---------------------------------------------------------------------------
def plot_length_distributions(lengths: dict, plot_dir: Path, meta: str) -> None:
    """ISL / OSL / ratio histograms per task (justifies the decode workload).

    Lengths are profiled under eval semantics (few-shot prompts, the task's stop
    strings and generation cap; see profile_lengths.py). p50 sizes the typical
    request, mean drives throughput, p95 sizes the KV capacity a batch slot must
    reserve — all three are marked. If a tail of generations still hits the cap,
    the OSL panel says so rather than let the capped bar look like a real mode.
    """
    for task, d in lengths.items():
        isl, osl = np.array(d["isl"], float), np.array(d["osl"], float)
        if len(isl) == 0:
            continue
        gen_cap = int(d.get("cap", 512))
        shots = d.get("num_fewshot")
        ratio = osl / np.maximum(isl, 1)
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
        panels = (
            (axes[0], isl, "ISL — prompt tokens (prefill chip)", CAT[0]),
            (axes[1], osl, "OSL — generated tokens (decode chip)", CAT[1]),
            (axes[2], ratio, "OSL / ISL — decode dominance", CAT[4]),
        )
        for ax, data, name, c in panels:
            ax.hist(data, bins=24, color=c, alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
            p50, mean, p95 = float(np.median(data)), float(np.mean(data)), float(np.percentile(data, 95))
            ax.axvline(p50, color=INK, lw=1.6, ls="--", label=f"p50 = {p50:.0f}")
            ax.axvline(mean, color=INK2, lw=1.4, label=f"mean = {mean:.0f}")
            ax.axvline(p95, color=MUTED, lw=1.2, ls=":", label=f"p95 = {p95:.0f}")
            ax.set_xlabel(name)
            ax.set_ylabel("requests")
            ax.legend(loc="upper right", fontsize=8.5)
        capped = float(d.get("capped_frac", np.mean(osl >= gen_cap)))
        if capped > 0.05:
            # Narrow 3-line block so it never collides with the upper-right legend.
            axes[1].text(0.02, 0.98, f"{capped:.0%} hit the {gen_cap}-token\n"
                         "eval cap (true OSL longer;\np50 is a lower bound)",
                         transform=axes[1].transAxes, va="top", fontsize=8.5, color="#a33b3b")
        shot_txt = f"  ·  {shots}-shot prompts (eval semantics)" if shots is not None else ""
        fig.suptitle(f"{task.upper()} request lengths  ·  {meta.split('  ·')[0]}{shot_txt}",
                     color=INK, fontweight="bold", fontsize=13, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plot_dir / f"00_lengths_{task}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 01/02. Accuracy vs reference-chip throughput / latency
# ---------------------------------------------------------------------------
def plot_accuracy_frontier(rows: list[dict], out: Path, meta: str, *, xkey: str, xlabel: str,
                           title: str, front_size: int) -> None:
    """PPL against a fixed-reference-chip metric. The metric is discrete (the MLEN
    cap moves in powers of two), so configs stack into columns; the line traces the
    Pareto front and only front points are labelled. Diverged precisions go in a
    corner note instead of stretching the log axis."""
    usable = [r for r in rows if r.get(xkey) not in (None, float("inf"))
              and r["cont_ppl"] is not None]
    pts = [r for r in usable if r["cont_ppl"] <= DIVERGED_PPL]
    n_div = len(usable) - len(pts)
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = np.array([r[xkey] for r in pts])
    ys = np.array([r["cont_ppl"] for r in pts])
    ax.scatter(xs, ys, s=34, color=CAT[0], alpha=0.35, edgecolor=SURFACE, linewidth=0.5,
               zorder=3, label=f"precision config (n={len(pts)})")

    lower_better_x = xkey.endswith("tpot_ms")
    if lower_better_x:
        # For latency the front is min-PPL at each achievable TPOT.
        inv = [dict(r, _neg=-r[xkey]) for r in pts]
        front = accuracy_throughput_front(inv, tps_key="_neg")
    else:
        front = accuracy_throughput_front(pts, tps_key=xkey)
    fx = np.array([r[xkey] for r in front])
    fy = np.array([r["cont_ppl"] for r in front])
    order = np.argsort(fx)
    ax.step(fx[order], fy[order], where="post" if lower_better_x else "pre",
            color=CAT[0], lw=2, zorder=4, label="Pareto front")
    ax.scatter(fx, fy, s=76, color=CAT[0], edgecolor=SURFACE, linewidth=1.0, zorder=5)

    ax.set_yscale("log")
    ax.margins(x=0.16, y=0.22)
    # Stagger labels up/down along the front so close neighbours don't overlap.
    for i, r in enumerate(sorted(front[:front_size], key=lambda r: r[xkey])):
        _annotate_no_clip(ax, r[xkey], r["cont_ppl"], _short(r["tag"]),
                          dy=10 if i % 2 == 0 else -16)
    if n_div:
        ax.text(0.5, 0.02, f"{n_div} diverged configs (PPL > {DIVERGED_PPL:.0f}) not shown",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("WikiText-2 perplexity  (log, lower is better)")
    _title(ax, title, meta)
    ax.legend(loc="upper center")
    _save(fig, out)


def plot_task_vs_throughput(rows: list[dict], out: Path, meta: str) -> None:
    """Downstream task accuracy vs reference-chip throughput (calibrated front only)."""
    tasks = [("gsm8k", "GSM8K", CAT[1]), ("ifeval", "IFEval", CAT[4])]
    have = [(k, lbl, c) for k, lbl, c in tasks if any(r.get(k) is not None for r in rows)]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for k, lbl, c in have:
        pts = sorted((r for r in rows if r.get(k) is not None and r.get("ref_tps")),
                     key=lambda r: r["ref_tps"])
        if not pts:
            continue
        ax.plot([r["ref_tps"] for r in pts], [r[k] * 100 for r in pts], "-o", color=c, ms=7,
                lw=1.8, label=lbl, markeredgecolor=SURFACE, markeredgewidth=0.7)
        for r in pts:
            _annotate_no_clip(ax, r["ref_tps"], r[k] * 100, _short(r["tag"]), dy=-14)
    ax.set_xlabel("Decode throughput on the reference chip  (tokens/s, higher is better)")
    ax.set_ylabel("Task accuracy  (%, higher is better)")
    _title(ax, "Downstream accuracy vs decode throughput",
           f"{meta}  ·  calibrated front (GPTQ+Erry, selective rotation, task-aligned)")
    ax.legend(loc="lower left")
    _save(fig, out)


# ---------------------------------------------------------------------------
# 04-08. Hardware co-design views (Stage B searches, analytic — CPU-cheap)
# ---------------------------------------------------------------------------
def plot_throughput_latency(sim, front, workload, space, out: Path, meta: str,
                            sampler: str = "nsga2") -> None:
    """Per-precision co-design frontier: the non-dominated points where more
    latency buys more throughput, up to saturation."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    drew = False
    for i, p in enumerate(front[:4]):
        prec = sim.precision_from_row(p)
        recs = search_precision(sim, prec, workload, space, n_trials=160, sampler=sampler)
        fr, best = [], -1.0
        for r in sorted(recs, key=lambda r: (r.tpot_ms, -r.tps)):
            if r.tps > best + 1e-9:
                fr.append(r); best = r.tps
        if fr:
            drew = True
            ax.plot([r.tpot_ms for r in fr], [r.tps for r in fr], "-o", color=CAT[i % len(CAT)],
                    ms=5, lw=1.7, label=_short(p["tag"]), markeredgecolor=SURFACE, markeredgewidth=0.5)
    if not drew:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Latency  TPOT (ms/token, log, lower is better)")
    ax.set_ylabel("Throughput  (tokens/s, higher is better)")
    _title(ax, "Co-designed throughput vs latency per precision",
           f"{meta}  ·  each curve = one precision, points = searched chips")
    ax.legend(loc="upper left", title="W / KV / A")
    _save(fig, out)


def plot_disagg_pareto(sim, front, workload, space, out: Path, meta: str,
                       sampler: str = "nsga2") -> None:
    """Decode-chip serving Pareto, the industry-standard view: throughput-per-chip
    vs per-user rate.

      x = tok/s/user  = 1 / TPOT   (single-stream rate; decode dominates it)
      y = tok/s/GPU   = TPS        (per-chip throughput; n_chips pinned to 1)

    Since TPS = batch / TPOT, batch is the knob: sweeping it traces each chip's
    serving curve (more batch -> more throughput per chip, worse per-user rate). We
    sweep the top decode-tuned chips against the aggregated reference design
    (MLEN=2048, BLEN=32, 4/4/4); curves up-and-right of it are the disaggregation
    win. The coupling cost is the KV hand-off in fig 13."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    def _sweep(prec, hw, label, color, ls, lw):
        over = {"MLEN": hw["MLEN"], "BLEN": hw["BLEN"], "HLEN": hw.get("HLEN", 128),
                "VLEN": hw["MLEN"], **sim.hbm_overrides(space.hbm_gen, hw.get("HBM_CHANNELS", 32))}
        pts = []
        for b in space.BATCH:
            m = sim.evaluate(prec, batch=b, input_seq=workload["input_seq"],
                             output_seq=workload["output_seq"], hw_over=over, n_chips=1)
            if m.fits_in_hbm and m.tps > 0 and m.tpot > 0:
                pts.append((1.0 / m.tpot, m.tps))     # tok/s/user, tok/s/GPU
        if len(pts) >= 2:
            pts.sort()
            ax.plot([x for x, _ in pts], [y for _, y in pts], ls, color=color, lw=lw,
                    marker="o", ms=4, markeredgecolor=SURFACE, markeredgewidth=0.4, label=label)
            return True
        return False

    # Holding area fixed at the 2048x32 array isolates the precision co-design:
    # fixed W/A/KV=4/4/4 vs decode-tuned. Decode is capacity-bound, so aggressive
    # KV frees HBM for a larger batch and more throughput at the same silicon.
    aggregated_hw = {"MLEN": 2048, "BLEN": 32, "HLEN": 128}
    drew = False
    try:
        drew |= _sweep(sim.precision_from_eff_bits(4.0, 4.0, 4.0), aggregated_hw,
                       "aggregated-PLENA  (2048×32, W/A/KV 4/4/4)", INK, "--", 2.2)
    except Exception:
        pass
    # Decode-tuned: same weights/activations (W/A 4) but aggressive KV (KV3).
    # Decode is capacity-bound, so the freed HBM buys a larger batch at equal silicon.
    drew |= _sweep(sim.precision_from_eff_bits(4.0, 4.0, 3.0), aggregated_hw,
                   "decode-tuned  (2048×32, W/A 4, KV 3)", CAT[2], "-", 2.4)
    if not drew:
        plt.close(fig); return

    ax.set_xlabel("Per-user rate  tok/s/user  (1 / TPOT · higher is better →)")
    ax.set_ylabel("Throughput per chip  tok/s/GPU  (higher is better ↑)")
    _title(ax, "Decode-chip serving Pareto (disaggregated)",
           f"{meta}  ·  batch sweep at equal silicon (2048×32)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.text(0.02, 0.02,
            "Rate match: decode tok/s must equal the prefill chip's output rate;\n"
            "the KV hand-off (fig 13) is the disaggregation coupling cost.",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=MUTED, lw=0.6, alpha=0.9))
    _save(fig, out)


def plot_right_sizing(sim, prec_row, workload, space, out: Path, meta: str,
                      sampler: str = "nsga2") -> None:
    """Throughput vs matrix-array area; the knee is the smallest chip at peak TPS."""
    prec = sim.precision_from_row(prec_row)
    recs = search_precision(sim, prec, workload, space, n_trials=200, sampler=sampler)
    if not recs:
        return
    area_tps: dict[float, float] = {}
    for r in recs:
        area_tps[r.area_mm2] = max(area_tps.get(r.area_mm2, 0.0), r.tps)
    xs, ys = zip(*sorted(area_tps.items()))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(xs, ys, "-o", color=CAT[1], ms=6, lw=1.8, markeredgecolor=SURFACE, markeredgewidth=0.6)
    peak = max(ys)
    knee = min(a for a, t in area_tps.items() if t >= 0.99 * peak)
    ax.scatter([knee], [area_tps[knee]], s=170, marker="*", color=CAT[5], zorder=6,
               edgecolor=SURFACE, linewidth=0.8, label=f"right-size: {knee:.3f} mm² at ≥99% peak")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix-array area  (mm², log₂, smaller is cheaper)")
    ax.set_ylabel("Peak throughput  (tokens/s, higher is better)")
    _title(ax, "Hardware right-sizing", f"{meta}  ·  precision {_short(prec_row['tag'])}")
    ax.legend(loc="lower right")
    _save(fig, out)


def plot_batch_sweep(sim, prec_row, hw, workload, space, out: Path, meta: str) -> None:
    """Batch drives throughput up (saturating) and latency up — two panels, one axis each."""
    prec = sim.precision_from_row(prec_row)
    over = {"MLEN": hw["MLEN"], "BLEN": hw["BLEN"], "HLEN": hw["HLEN"], "VLEN": hw["MLEN"],
            **sim.hbm_overrides(space.hbm_gen, hw["HBM_CHANNELS"])}
    batches, tps, tpot, fits = [], [], [], []
    for b in space.BATCH:
        m = sim.evaluate(prec, batch=b, input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over, n_chips=1)
        batches.append(b); tps.append(m.tps); tpot.append(m.tpot * 1e3); fits.append(m.fits_in_hbm)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, y, lbl, c in ((axL, tps, "Throughput (tokens/s, higher is better)", CAT[0]),
                          (axR, tpot, "Latency TPOT (ms/token, lower is better)", CAT[7])):
        ax.plot(batches, y, "-o", color=c, ms=6, lw=1.8, markeredgecolor=SURFACE, markeredgewidth=0.6)
        for b, f in zip(batches, fits):
            if not f:
                ax.axvline(b, ls=":", color=BASE_C, lw=1.2)
                ax.text(b, ax.get_ylim()[1], " exceeds HBM", color=MUTED, fontsize=8, va="top")
                break
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size (log₂)")
        ax.set_ylabel(lbl)
    _title(axL, "Throughput vs batch", meta)
    _title(axR, "Latency vs batch", f"precision {_short(prec_row['tag'])}, co-designed chip")
    _save(fig, out)


def plot_hbm_channels(sim, prec_row, hw, workload, space, out: Path, meta: str) -> None:
    """Throughput vs HBM channel count (bandwidth + capacity scale together).
    One chip is pinned so the capacity wall is visible as missing points."""
    prec = sim.precision_from_row(prec_row)
    chans, tps = [], []
    for ch in space.HBM_CHANNELS:
        hbm = sim.hbm_overrides(space.hbm_gen, ch)
        over = {"MLEN": hw["MLEN"], "BLEN": hw["BLEN"], "HLEN": hw["HLEN"], "VLEN": hw["MLEN"], **hbm}
        m = sim.evaluate(prec, batch=hw["BATCH"], input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over, n_chips=1)
        chans.append(ch); tps.append(m.tps if m.fits_in_hbm else np.nan)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(chans, tps, "-o", color=CAT[4], ms=6, lw=1.8, markeredgecolor=SURFACE, markeredgewidth=0.6)
    if any(np.isnan(t) for t in tps):
        first_bad = chans[int(np.argmax(np.isnan(tps)))]
        ax.text(first_bad, np.nanmin(tps), " model+KV do not fit", color=MUTED, fontsize=8.5)
    g = sim._dd.HBM_GENS[space.hbm_gen]
    ax.set_xlabel(f"HBM channels  ({space.hbm_gen}, "
                  f"{g['ch_bits'] * g['gbps'] / 8:.0f} GB/s + {g['ch_gb']:.0f} GB each)")
    ax.set_ylabel("Throughput  (tokens/s, higher is better)")
    _title(ax, "Throughput vs HBM channels", f"{meta}  ·  precision {_short(prec_row['tag'])}")
    _save(fig, out)


def plot_roofline(sim, prec_row, hw, workload, space, out: Path, meta: str) -> None:
    """Roofline for the co-designed chip at one precision: HBM slant, compute
    ceiling, the workload's arithmetic intensity, and the achieved point."""
    dd = sim._dd
    prec = sim.precision_from_row(prec_row)
    over = {"MLEN": hw["MLEN"], "BLEN": hw["BLEN"], "HLEN": hw["HLEN"], "VLEN": hw["MLEN"],
            **sim.hbm_overrides(space.hbm_gen, hw["HBM_CHANNELS"])}
    hw_cfg = sim.base_hw.model_copy(update={**over, "HBM_M_Prefetch_Amount": hw["MLEN"]})
    peak_bw = dd.peak_hbm_bw_bytes(hw_cfg)
    peak_compute = 2 * hw_cfg.MLEN * hw_cfg.BLEN * dd.FREQ_HZ * dd.compute_density(prec.spec) / 1e12
    m = sim.evaluate(prec, batch=hw["BATCH"], input_seq=workload["input_seq"],
                     output_seq=workload["output_seq"], hw_over=over, n_chips=1)
    avg_kv = workload["input_seq"] + workload["output_seq"] // 2
    flops = dd.decode_step_flops(sim.dims, avg_kv, hw["BATCH"])
    ai = flops / m.avg_bytes_per_token
    achieved = flops / m.tpot / 1e12
    ridge = peak_compute / (peak_bw / 1e12)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = np.logspace(-1, np.log10(max(ridge, ai) * 8), 200)
    ax.plot(xs, np.minimum(xs * peak_bw / 1e12, peak_compute), color=INK2, lw=2, zorder=2,
            label="roofline (HBM slant + compute ceiling)")
    ax.axvline(ai, ls="--", color=CAT[5], lw=1.6, zorder=3,
               label=f"decode intensity ({ai:.0f} FLOP/byte)")
    ax.axvline(ridge, ls=":", color=MUTED, lw=1.2)
    ax.text(ridge, peak_compute, " ridge", color=MUTED, fontsize=8, va="bottom", ha="left")
    bound = "memory-bound" if m.mem_bound else "compute-bound"
    ax.scatter([ai], [achieved], s=130, color=MEM if m.mem_bound else COMPUTE, zorder=5,
               edgecolor=SURFACE, linewidth=1.0, label=f"achieved: {achieved:.1f} TFLOP/s ({bound})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity  (FLOP/byte)")
    ax.set_ylabel("Performance  (TFLOP/s)")
    _title(ax, "Decode-step roofline", f"{meta}  ·  precision {_short(prec_row['tag'])}")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 09. Precision heatmap
# ---------------------------------------------------------------------------
def plot_precision_heatmap(rows: list[dict], out: Path, meta: str) -> None:
    """Weight-bits x KV-bits grid coloured by the best perplexity over the other
    knobs (format / activation / block). Log colours, since quantisation failures
    span orders of magnitude; diverged cells are flagged rather than left to
    flatten the scale."""
    ws = sorted({round(r["attn_bits"]) for r in rows})
    kvs = sorted({round(r["kv_bits"]) for r in rows})
    if len(ws) < 2 or len(kvs) < 2:
        return
    grid = np.full((len(ws), len(kvs)), np.nan)
    for r in rows:
        i, j = ws.index(round(r["attn_bits"])), kvs.index(round(r["kv_bits"]))
        v = r["cont_ppl"]
        if np.isnan(grid[i, j]) or v < grid[i, j]:
            grid[i, j] = v

    shown = np.where(grid <= DIVERGED_PPL, grid, np.nan)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad("#f1f0ec")
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    vmin, vmax = np.nanmin(shown), np.nanmax(shown)
    im = ax.imshow(shown, cmap=cmap, aspect="auto", origin="lower",
                   norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01)))
    ax.set_xticks(range(len(kvs))); ax.set_xticklabels([f"{k}b" for k in kvs])
    ax.set_yticks(range(len(ws))); ax.set_yticklabels([f"{w}b" for w in ws])
    ax.set_xlabel("KV-cache precision (effective bits)")
    ax.set_ylabel("Weight precision (effective bits)")
    med = np.nanmedian(shown)
    for i in range(len(ws)):
        for j in range(len(kvs)):
            if np.isnan(grid[i, j]):
                ax.text(j, i, "·", ha="center", va="center", color=MUTED, fontsize=12)
            elif grid[i, j] > DIVERGED_PPL:
                ax.text(j, i, "diverged", ha="center", va="center", fontsize=8, color="#a33b3b")
            else:
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color=INK if grid[i, j] < med else "#ffffff")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("WikiText-2 perplexity (log scale, lower is better)", color=INK2)
    ax.grid(False)
    _title(ax, "Mixed-precision accuracy map",
           f"{meta}  ·  best over formats / activation / block per cell")
    _save(fig, out)


# ---------------------------------------------------------------------------
# 10. With / without optimisation — the end-to-end table as a figure
# ---------------------------------------------------------------------------
def plot_with_without(e2e_rows: list[dict], out: Path, meta: str) -> None:
    """Grouped bars from ``end_to_end.json``: FP16 baseline vs quantised-on-the-
    same-chip vs co-designed, all at the same batch (the equal-batch columns), so
    precision and hardware gains stay separable. Chip counts annotate the bars
    whose design needed more than one HBM stack."""
    rows = [r for r in e2e_rows if r.get("eq_batch_tps")]
    if not rows:
        return
    rows = rows[:4]
    labels = [_short(r["tag"]) for r in rows]
    x = np.arange(len(rows))
    w = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    series = [
        ("FP16 baseline (shipped chip)", BASE_C, [r["base_tps"] for r in rows],
         [r["base_tpot_ms"] for r in rows], [r.get("base_chips", 1) for r in rows]),
        ("quantised (same chip)", QUANT_C, [r["quant_base_tps"] for r in rows],
         [r["quant_base_tpot_ms"] for r in rows], [r.get("quant_chips", 1) for r in rows]),
        ("quantised + co-design", OPT_C, [r["eq_batch_tps"] for r in rows],
         [r["eq_batch_tpot_ms"] for r in rows], [1] * len(rows)),
    ]
    for s, (lbl, c, tps, tpot, chips) in enumerate(series):
        xoff = x + (s - 1) * (w + 0.02)
        ax1.bar(xoff, tps, w, color=c, zorder=3, label=lbl)
        ax2.bar(xoff, tpot, w, color=c, zorder=3, label=lbl)
        for xi, v, n in zip(xoff, tps, chips):
            ax1.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 2),
                         ha="center", va="bottom", fontsize=8, color=INK2)
            # Chip count (when >1) sits above the value in a smaller, muted line so
            # it reads clearly without overrunning the neighbouring bar.
            if n and n > 1:
                ax1.annotate(f"{n} chips", (xi, v), textcoords="offset points",
                             xytext=(0, 12), ha="center", va="bottom", fontsize=7,
                             color=MUTED)
        for xi, v in zip(xoff, tpot):
            ax2.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    speed = [r["eq_batch_tps"] / r["base_tps"] for r in rows]
    gain = (f"end-to-end speed-up ×{min(speed):.1f}" if min(speed) == max(speed)
            else f"end-to-end speed-up ×{min(speed):.1f}–×{max(speed):.1f}")
    for ax, ylabel, title, sub in (
        (ax1, "Throughput (tokens/s, higher is better)", "Decode throughput", f"{meta}  ·  {gain}"),
        (ax2, "TPOT (ms/token, lower is better)", "Decode latency",
         "same batch for all three bars (iso-workload)"),
    ):
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
        ax.set_ylabel(ylabel)
        ax.margins(y=0.18)   # headroom so bar-top labels clear the legend
        _title(ax, title, sub)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncols=3,
               bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Per-task serving batch (uses profiled p50 lengths)
# ---------------------------------------------------------------------------
def plot_task_batch(sim, prec_row, hw, lengths: dict, space, out: Path, meta: str,
                    tpot_budget_ms: float = 50.0) -> None:
    """Ideal decode batch per task: sweep the serving batch at each task's profiled
    p50 lengths on the co-designed chip. The pick is the smallest batch within 5% of
    peak TPS that fits in HBM; larger batches only add latency. A reference line
    shows the interactive TPOT budget."""
    prec = sim.precision_from_row(prec_row)
    over = {"MLEN": hw["MLEN"], "BLEN": hw["BLEN"], "HLEN": hw["HLEN"], "VLEN": hw["MLEN"],
            **sim.hbm_overrides(space.hbm_gen, hw["HBM_CHANNELS"])}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    drew = False
    for i, (task, d) in enumerate(sorted(lengths.items())):
        if not d.get("isl"):
            continue
        isl = int(np.median(d["isl"])); osl = max(int(np.median(d["osl"])), 16)
        bs, tps, tpot = [], [], []
        for b in space.BATCH:
            m = sim.evaluate(prec, batch=b, input_seq=isl, output_seq=osl,
                             hw_over=over, n_chips=1)
            if not m.fits_in_hbm:
                break
            bs.append(b); tps.append(m.tps); tpot.append(m.tpot * 1e3)
        if not bs:
            continue
        drew = True
        c = CAT[i % len(CAT)]
        peak = max(tps)
        pick = next(b for b, t in zip(bs, tps) if t >= 0.95 * peak)
        lbl = f"{task} (ISL {isl}, OSL {osl})"
        axL.plot(bs, tps, "-o", color=c, ms=5.5, lw=1.7, label=lbl,
                 markeredgecolor=SURFACE, markeredgewidth=0.5)
        axR.plot(bs, tpot, "-o", color=c, ms=5.5, lw=1.7, label=lbl,
                 markeredgecolor=SURFACE, markeredgewidth=0.5)
        j = bs.index(pick)
        axL.scatter([pick], [tps[j]], s=150, marker="*", color=c, zorder=6,
                    edgecolor=SURFACE, linewidth=0.8)
        axL.annotate(f"batch {pick}", (pick, tps[j]), textcoords="offset points",
                     xytext=(6, -12), fontsize=8.5, color=INK2)
    if not drew:
        plt.close(fig)
        return
    axR.axhline(tpot_budget_ms, ls="--", color=MUTED, lw=1.2)
    axR.text(space.BATCH[0], tpot_budget_ms, f" {tpot_budget_ms:.0f} ms interactive budget",
             color=MUTED, fontsize=8.5, va="bottom")
    axL.legend(loc="lower right", fontsize=8.5)
    axR.legend(loc="upper left", fontsize=8.5)
    for ax, ylab in ((axL, "Throughput (tokens/s, higher is better)"),
                     (axR, "TPOT (ms/token, lower is better)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Serving batch (log₂)")
        ax.set_ylabel(ylab)
    _title(axL, "Ideal decode batch per task",
           f"{meta}  ·  ★ = smallest batch within 5% of peak TPS that fits HBM")
    _title(axR, "Latency cost of batching", f"precision {_short(prec_row['tag'])}, co-designed chip")
    _save(fig, out)


# ---------------------------------------------------------------------------
# 12. FP_SETTING (vector-unit width) ablation from the stage-2 sweep JSONs
# ---------------------------------------------------------------------------
def plot_fp_setting(trial_dir: Path, out: Path, meta: str, chosen: dict[str, str]) -> None:
    """PPL vs vector-unit minifloat width from the front sweep: how narrow the
    SiLU/RMSNorm/softmax/rope unit can go before accuracy suffers. One curve per
    calibrated front precision; the chosen (cheapest-within-tolerance) point is
    starred."""
    groups: dict[str, list[tuple[int, str, float]]] = {}
    for f in sorted(trial_dir.glob("*__ppl.json")):
        row = json.loads(f.read_text())
        if row.get("error") or row.get("cont_ppl") in ("", None):
            continue
        tag = row.get("tag", f.stem)
        base = tag.split("_fp")[0]
        if "_fp" in tag:
            e, m = tag.split("_fp")[1].split("-")
            bits, lbl = 1 + int(e) + int(m), f"E{e}M{m}"
        else:
            bits, lbl = 16, "bf16"
        groups.setdefault(base, []).append((bits, lbl, float(row["cont_ppl"])))
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for i, (base, pts) in enumerate(sorted(groups.items())):
        pts.sort()
        c = CAT[i % len(CAT)]
        ax.plot([p[0] for p in pts], [p[2] for p in pts], "-o", color=c, ms=6, lw=1.7,
                label=_short(base), markeredgecolor=SURFACE, markeredgewidth=0.6)
        pick = chosen.get(base)
        for bits, lbl, ppl in pts:
            if pick is not None and lbl == pick:
                ax.scatter([bits], [ppl], s=170, marker="*", color=c, zorder=6,
                           edgecolor=SURFACE, linewidth=0.8)
    ax.set_xlabel("Vector-unit width  (1 + E + M bits; bf16 = 16)")
    ax.set_ylabel("WikiText-2 perplexity")
    _title(ax, "FP_SETTING: how narrow can the vector unit go?",
           f"{meta}  ·  ★ = chosen (cheapest within PPL tolerance)")
    ax.legend(loc="upper right", title="W / KV / A", fontsize=8.5)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 13. Prefill -> decode KV hand-off (the disaggregation contribution)
# ---------------------------------------------------------------------------
def plot_kv_handoff(sim, prec_row, workload, out: Path, meta: str) -> None:
    """Time to ship the prompt KV cache from the prefill chip to the decode chip
    vs the decode KV precision. The transfer is quantise-on-write, so lower-bit KV
    shrinks both the wire bytes and the added time-to-first-token — interconnect
    time is a third thing KV quantisation buys, on top of HBM bandwidth and
    capacity. Two link tiers and the streamed vs bulk overlap cases bound the cost."""
    import importlib
    import sys as _sys
    _sys.path.insert(0, str(_sim_root_from(sim)))
    handoff = importlib.import_module("analytic_models.disagg_serve.handoff")

    kv_bits = [16, 8, 6, 4, 3, 2]
    isl, batch = workload["input_seq"], workload["batch"]
    links = [("nvlink4", "NVLink4 (450 GB/s)", CAT[0]),
             ("pcie5", "PCIe 5.0 x16 (64 GB/s)", CAT[7])]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # Left: wire bytes vs KV precision (precision-only, link-independent).
    gb = [handoff.kv_wire_bytes(sim.dims, {"kv_bits": float(b)}, isl, batch) / 1e9 for b in kv_bits]
    axL.plot(kv_bits, gb, "-o", color=CAT[1], ms=7, lw=2, markeredgecolor=SURFACE, markeredgewidth=0.7)
    for b, g in zip(kv_bits, gb):
        axL.annotate(f"{g:.2f} GB", (b, g), textcoords="offset points", xytext=(0, 7),
                     fontsize=8, color=INK2, ha="center")
    axL.set_xlabel("Decode KV-cache precision  (effective bits)")
    axL.set_ylabel("KV cache shipped  (GB)")
    axL.invert_xaxis()
    _title(axL, "KV hand-off volume", f"{meta}  ·  prompt {isl} tok × batch {batch}, quantise-on-write")

    # Right: added TTFT vs KV precision, per link, streamed and bulk.
    for link, lbl, c in links:
        bulk = [handoff.handoff_time(sim.dims, {"kv_bits": float(b)}, isl, batch, link).bulk_s * 1e3
                for b in kv_bits]
        strm = [handoff.handoff_time(sim.dims, {"kv_bits": float(b)}, isl, batch, link).streamed_s * 1e3
                for b in kv_bits]
        axR.plot(kv_bits, bulk, "-o", color=c, ms=6, lw=1.9, label=f"{lbl} · bulk",
                 markeredgecolor=SURFACE, markeredgewidth=0.6)
        axR.plot(kv_bits, strm, "--s", color=c, ms=5, lw=1.5, alpha=0.8, label=f"{lbl} · streamed",
                 markeredgecolor=SURFACE, markeredgewidth=0.6)
    axR.set_xlabel("Decode KV-cache precision  (effective bits)")
    axR.set_ylabel("Added decode-side TTFT  (ms)")
    axR.set_yscale("log")
    axR.invert_xaxis()
    axR.legend(loc="upper left", fontsize=8)
    _title(axR, "Hand-off cost to time-to-first-token",
           "bulk = whole cache after prefill · streamed = layer-wise")
    _save(fig, out)


# ---------------------------------------------------------------------------
# 14. K/V SRAM residency — measured HBM-read reduction (compiler optimisation)
# ---------------------------------------------------------------------------
def plot_kv_residency(csv_path: Path, out: Path, meta: str) -> None:
    """Decode-step HBM reads with vs without the K/V-residency compiler change,
    measured on the emulator across context length. Giving K and V their own
    Matrix-SRAM tiles and fetching each once per KV group (instead of re-loading V
    per Q-head) removes redundant reads; the saving grows with context because KV
    traffic dominates. Outputs are bit-exact, so this is a pure traffic win."""
    if not csv_path.exists():
        return
    import csv as _csv
    with_r, without_r = {}, {}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            kv = int(row["kv_size"]); b = float(row["hbm_read_bytes"])
            (with_r if row["variant"] == "with_residency" else without_r)[kv] = b
    kvs = sorted(set(with_r) & set(without_r))
    if len(kvs) < 2:
        return
    wo = np.array([without_r[k] / 1e3 for k in kvs])
    wi = np.array([with_r[k] / 1e3 for k in kvs])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axL.plot(kvs, wo, "-o", color=BASE_C, ms=7, lw=2, label="baseline (V re-loaded per head)",
             markeredgecolor=SURFACE, markeredgewidth=0.7)
    axL.plot(kvs, wi, "-o", color=OPT_C, ms=7, lw=2, label="K/V resident (fetch once per group)",
             markeredgecolor=SURFACE, markeredgewidth=0.7)
    axL.fill_between(kvs, wi, wo, color=OPT_C, alpha=0.10)
    axL.set_xlabel("KV-cache length  (tokens)")
    axL.set_ylabel("Decode-step HBM reads  (KB)")
    axL.legend(loc="upper left", fontsize=9)
    _title(axL, "K/V residency cuts decode HBM reads", f"{meta}  ·  measured on the transactional emulator")

    red = (1 - wi / wo) * 100
    axR.plot(kvs, red, "-o", color=CAT[3], ms=7, lw=2, markeredgecolor=SURFACE, markeredgewidth=0.7)
    for k, r in zip(kvs, red):
        axR.annotate(f"−{r:.0f}%", (k, r), textcoords="offset points", xytext=(0, 7),
                     fontsize=8.5, color=INK2, ha="center")
    axR.set_xlabel("KV-cache length  (tokens)")
    axR.set_ylabel("HBM-read reduction  (%)")
    axR.set_ylim(0, max(red) * 1.25)
    _title(axR, "Saving grows with context", "KV traffic dominates decode as the cache grows")
    _save(fig, out)


# ---------------------------------------------------------------------------
# 15. Matrix-array serialization gap (the RTL timing finding)
# ---------------------------------------------------------------------------
def plot_serialization_gap(hw, out: Path, meta: str) -> None:
    """Per-M_MM cost and array utilisation vs BLEN. The RTL serialises matmuls
    (drain-then-start): microbenchmarks measured 23 cycles at BLEN=4 and 35 at
    BLEN=8, giving cost = 3·BLEN + 11, versus ~BLEN for a fully-pipelined array.
    Utilisation = BLEN / cost falls to ~30% at BLEN=32, making the pipelined array
    the largest decode-compute lever, and the reason the chip can look
    compute-bound at large batch."""
    blens = np.array([2, 4, 8, 16, 32, 64])
    serial = 3 * blens + 11
    pipelined = blens.astype(float)          # ideal fully-pipelined steady state
    measured = {4: 23, 8: 35}                # RTL microbenchmark points

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axL.plot(blens, serial, "-o", color=MEM, ms=6, lw=2, label="serialized (current RTL)",
             markeredgecolor=SURFACE, markeredgewidth=0.6)
    axL.plot(blens, pipelined, "--", color=COMPUTE, lw=2, label="fully-pipelined (ideal)")
    axL.scatter(list(measured), list(measured.values()), s=150, marker="*", color=INK, zorder=6,
                edgecolor=SURFACE, linewidth=0.8, label="RTL microbenchmark")
    axL.set_xscale("log", base=2); axL.set_yscale("log", base=2)
    axL.set_xlabel("BLEN  (systolic block size, log₂)")
    axL.set_ylabel("Cycles per M_MM  (log₂)")
    axL.legend(loc="upper left", fontsize=8.5)
    _title(axL, "Matrix-array serialization", f"{meta}  ·  cost = 3·BLEN + 11 (measured)")

    util = blens / serial * 100
    axR.plot(blens, util, "-o", color=MEM, ms=6, lw=2, markeredgecolor=SURFACE, markeredgewidth=0.6)
    axR.axhline(100, ls="--", color=COMPUTE, lw=1.6)
    axR.text(blens[0], 100, " pipelined ceiling", color=COMPUTE, fontsize=8.5, va="bottom")
    baseline_blen = int(hw["BLEN"]) if hw and hw.get("BLEN") in blens else 32
    j = int(np.where(blens == baseline_blen)[0][0])
    axR.scatter([baseline_blen], [util[j]], s=170, marker="*", color=INK, zorder=6,
                edgecolor=SURFACE, linewidth=0.8)
    axR.annotate(f"BLEN {baseline_blen}: {util[j]:.0f}% of peak", (baseline_blen, util[j]),
                 textcoords="offset points", xytext=(8, 4), fontsize=8.5, color=INK2)
    axR.set_xscale("log", base=2)
    axR.set_xlabel("BLEN  (log₂)")
    axR.set_ylabel("Matrix-array utilisation  (%)")
    axR.set_ylim(0, 108)
    _title(axR, "Utilisation lost to serialization",
           "realising the pipelined array is the top decode-compute lever")
    _save(fig, out)


def _sim_root_from(sim) -> Path:
    """PLENA_Simulator root, from the simulator's model-json path."""
    return Path(sim.model_json).resolve().parents[3]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Render all decode-DSE figures from a config + CSV.")
    ap.add_argument("config")
    ap.add_argument("--software-csv", default=None)
    ap.add_argument("--n-trials", type=int, default=160)
    args = ap.parse_args()

    set_style()
    cfg = json.loads(Path(args.config).read_text())
    model = cfg["model_name"].split("/")[-1]
    out_dir = Path(cfg.get("output_dir", "results/decode_dse")) / model
    csv_path = Path(args.software_csv) if args.software_csv else out_dir / "software_disagg_decode.csv"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    wl = cfg["reference_workload"]
    meta = f"{model}  ·  batch {wl['batch']}, context {wl['input_seq']}→+{wl['output_seq']} tok"

    def fig(name, fn, *a, **kw):
        """Render one figure; a failure skips it (partial pipelines still plot)."""
        try:
            fn(*a, **kw)
        except Exception:
            print(f"[plots] {name} failed:")
            traceback.print_exc()

    lengths = {}
    lengths_path = out_dir / "task_lengths.json"
    if lengths_path.exists():
        lengths = json.loads(lengths_path.read_text())
        fig("00_lengths", plot_length_distributions, lengths, plot_dir, meta)

    all_rows = load_software_rows(csv_path)
    rows = list(prefer_best_method(all_rows).values())
    if not rows:
        raise SystemExit(f"no usable rows in {csv_path} — run Stage A first.")
    front = select_front(rows, int(cfg.get("front_size", 6)))

    sim = DecodeSimulator(cfg["sim_model"], model_lib=cfg.get("model_lib"))
    space = HardwareSpace.from_dict(cfg.get("hardware_space"))
    sampler = cfg.get("sampler", "nsga2")
    n_chips = int(cfg.get("n_chips", 1))

    # Accuracy views (CSV only). The scatter uses every recorded config so the
    # density is honest; fronts and labels use the best method per precision.
    fig("01_accuracy_vs_throughput", plot_accuracy_frontier, all_rows,
        plot_dir / "01_accuracy_vs_throughput.png", meta,
        xkey="ref_tps", xlabel="Decode throughput on the reference chip  (tokens/s, higher is better)",
        title="Accuracy vs decode throughput", front_size=int(cfg.get("front_size", 6)))
    fig("02_accuracy_vs_latency", plot_accuracy_frontier, all_rows,
        plot_dir / "02_accuracy_vs_latency.png", meta,
        xkey="ref_tpot_ms", xlabel="Single-request decode latency  TPOT@batch1 (ms/token, lower is better)",
        title="Accuracy vs decode latency", front_size=int(cfg.get("front_size", 6)))
    fig("03_task_vs_throughput", plot_task_vs_throughput, rows,
        plot_dir / "03_task_vs_throughput.png", meta)
    fig("09_precision_heatmap", plot_precision_heatmap, rows,
        plot_dir / "09_precision_heatmap.png", meta)

    # Hardware co-design views (Stage B analytic searches — CPU, ~seconds).
    fig("04_throughput_latency", plot_throughput_latency, sim, front, wl, space,
        plot_dir / "04_throughput_latency.png", meta, sampler)
    fig("16_disagg_pareto", plot_disagg_pareto, sim, front, wl, space,
        plot_dir / "16_disagg_pareto.png", meta, sampler)
    top = front[len(front) // 2] if front else rows[0]   # representative mid-front precision
    recs = search_precision(sim, sim.precision_from_row(top),
                            wl, space, n_trials=args.n_trials, n_chips=n_chips, sampler=sampler)
    best_hw = max(recs, key=lambda r: r.tps).hw if recs else None
    fig("05_right_sizing", plot_right_sizing, sim, top, wl, space,
        plot_dir / "05_right_sizing.png", meta, sampler)
    if best_hw:
        fig("06_batch_sweep", plot_batch_sweep, sim, top, best_hw, wl, space,
            plot_dir / "06_batch_sweep.png", meta)
        fig("07_hbm_channels", plot_hbm_channels, sim, top, best_hw, wl, space,
            plot_dir / "07_hbm_channels.png", meta)
        fig("08_roofline", plot_roofline, sim, top, best_hw, wl, space,
            plot_dir / "08_roofline.png", meta)
        if lengths:
            fig("11_task_batch", plot_task_batch, sim, top, best_hw, lengths, space,
                plot_dir / "11_task_batch.png", meta)

    # End-to-end with/without (Stage B table as a figure).
    e2e_path = out_dir / "end_to_end.json"
    if e2e_path.exists():
        fig("10_with_without", plot_with_without, json.loads(e2e_path.read_text()),
            plot_dir / "10_with_without.png", meta)

    # Decode-specific contributions: the disaggregation hand-off, the measured
    # compiler optimisation, and the RTL timing finding.
    fig("13_kv_handoff", plot_kv_handoff, sim, top, wl,
        plot_dir / "13_kv_handoff.png", meta)
    fig("14_kv_residency", plot_kv_residency, out_dir / "kv_residency_measured.csv",
        plot_dir / "14_kv_residency.png", meta)
    fig("15_serialization_gap", plot_serialization_gap, best_hw,
        plot_dir / "15_serialization_gap.png", meta)

    # FP_SETTING ablation from the stage-2 sweep trial files.
    chosen = {r["tag"].split("_fp")[0]: (r.get("fp_setting") or "bf16")
              for r in rows if r.get("use_rotation") or r.get("use_gptq")}
    fig("12_fp_setting", plot_fp_setting, out_dir / "trials",
        plot_dir / "12_fp_setting.png", meta, chosen)

    # Split the headline figures from the supporting diagnostics so the listing
    # makes the selection obvious.
    CORE = {"01_accuracy_vs_throughput", "02_accuracy_vs_latency", "08_roofline",
            "09_precision_heatmap", "10_with_without", "13_kv_handoff",
            "14_kv_residency", "15_serialization_gap"}
    print(f"[plots] wrote figures to {plot_dir}\n  -- core --")
    for p in sorted(plot_dir.glob("*.png")):
        if p.stem in CORE or p.name.startswith("00_"):
            print(f"    {p.name}")
    print("  -- supporting diagnostics --")
    for p in sorted(plot_dir.glob("*.png")):
        if p.stem not in CORE and not p.name.startswith("00_"):
            print(f"    {p.name}")


if __name__ == "__main__":
    main()
