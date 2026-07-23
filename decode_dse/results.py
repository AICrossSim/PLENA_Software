"""Read Stage A results and form the accuracy-vs-throughput front.

The link between software (accuracy) and hardware is the analytic model's
end-to-end throughput/latency on a fixed reference chip (the shipped design,
clamped legal per precision, at the baseline HBM channel count) — not a byte
proxy, and not the co-design ceiling. Searching channels+batch per precision
lets the hardware buy back the precision difference, collapsing the throughput
axis to one value per stream-width class and starving the search of signal.

Fixed-chip TPS is discrete (the MLEN bandwidth cap doubles only when the widest
operand crosses a power of two), so the strict Pareto front holds one point per
TPS level. ``select_front`` fills the requested size with the next-best-PPL
points per level — the ablation rows the end-to-end table wants anyway.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_software_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse ``software_disagg_decode.csv`` into typed rows (error rows dropped)."""
    path = Path(csv_path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for r in csv.DictReader(path.open()):
        if r.get("error") or r.get("cont_ppl") in ("", None):
            continue
        try:
            row = {
                "tag": r.get("tag", ""),
                "cont_ppl": float(r["cont_ppl"]),
                "prefill_ppl": _maybe_float(r.get("prefill_ppl")),
                "gsm8k": _maybe_float(r.get("gsm8k")),
                "ifeval": _maybe_float(r.get("ifeval")),
                "attn_bits": float(r["attn_w_bits"]),
                "ffn_bits": float(r["ffn_w_bits"]),
                "kv_bits": float(r["kv_bits"]),
                "act_bits": _maybe_float(r.get("act_bits")),
                "block": int(float(r.get("block") or 32)),
                "fp_setting": r.get("fp_setting") or None,
                "use_gptq": str(r.get("use_gptq", "")).lower() in {"true", "1"},
                "use_rotation": str(r.get("use_rotation", "")).lower() in {"true", "1"},
                "ref_tps": _maybe_float(r.get("ref_tps")),
                "ref_tpot_ms": _maybe_float(r.get("ref_tpot_ms")),
            }
        except (KeyError, ValueError):
            continue
        rows.append(row)
    return rows


def _maybe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def accuracy_throughput_front(
    rows: list[dict[str, Any]],
    *,
    tps_key: str = "ref_tps",
    ppl_key: str = "cont_ppl",
) -> list[dict[str, Any]]:
    """Non-dominated points in (throughput up, perplexity down).

    Keeps a point if no other has both higher throughput and lower-or-equal
    perplexity. Falls back to pure-accuracy ordering when throughput isn't
    populated yet (before the reference-chip bridge has run).
    """
    scored = [r for r in rows if r.get(tps_key) is not None]
    if not scored:
        # No throughput yet: rank by accuracy so the caller still gets a list.
        return sorted(rows, key=lambda r: r[ppl_key])

    front: list[dict[str, Any]] = []
    # Sort by throughput descending, keep each point that lowers perplexity.
    best_ppl = float("inf")
    for r in sorted(scored, key=lambda r: (-r[tps_key], r[ppl_key])):
        if r[ppl_key] < best_ppl - 1e-9:
            front.append(r)
            best_ppl = r[ppl_key]
    # Ordered high-throughput/high-ppl -> low-throughput/low-ppl.
    return front


def select_front(
    rows: list[dict[str, Any]],
    size: int,
    *,
    tps_key: str = "ref_tps",
    ppl_key: str = "cont_ppl",
    max_ppl: float = 100.0,
) -> list[dict[str, Any]]:
    """Strict Pareto front, filled to ``size`` with per-TPS-level runner-ups.

    Fixed-chip TPS takes one value per stream-width class, so the strict front
    holds ~one point per level. The fill walks TPS levels round-robin (highest
    first), adding the next-best-PPL point at each level, giving a same-throughput
    accuracy ablation instead of a single row per level. Diverged points
    (ppl > max_ppl, e.g. RTN 2-bit weights) never enter the front.
    """
    usable = [r for r in rows if r.get(tps_key) is not None and r[ppl_key] <= max_ppl]
    front = accuracy_throughput_front(usable, tps_key=tps_key, ppl_key=ppl_key)[:size]
    if len(front) >= size:
        return front

    chosen = {id(r) for r in front}
    by_level: dict[float, list[dict]] = {}
    for r in usable:
        if id(r) not in chosen:
            by_level.setdefault(round(r[tps_key], 2), []).append(r)
    for lvl in by_level:
        by_level[lvl].sort(key=lambda r: r[ppl_key])
    levels = sorted(by_level, reverse=True)

    out = list(front)
    while len(out) < size and any(by_level[lvl] for lvl in levels):
        for lvl in levels:
            if by_level[lvl] and len(out) < size:
                out.append(by_level[lvl].pop(0))
    return out


def prefer_best_method(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Index rows by (attn, ffn, kv) effective bits, keeping the most accurate
    method available per precision (rotation > GPTQ > RTN)."""
    def rank(r: dict) -> int:
        return 2 if r["use_rotation"] else 1 if r["use_gptq"] else 0

    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (round(r["attn_bits"], 2), round(r["ffn_bits"], 2), round(r["kv_bits"], 2))
        if key not in by_key or rank(r) > rank(by_key[key]):
            by_key[key] = r
    return by_key
