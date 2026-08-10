"""Decompose the decode throughput gap before trusting any framing of it.

For a set of representative hardware points this script prices the decode
loop on the stage-calibrated analytic model and reports where each point's
step time actually goes: the realized memory-bound fraction, the classical
peak-roofline view, the ideal-issue architecture view, and the matrix-issue
serialization fraction, alongside bytes per generated token and the
capacity-limited batch. When a measured GPU baseline report is supplied,
each point is placed next to the measured tokens-per-second rows as
context — the record carries not_a_headline_claim, and no ratio against a
peak-roofline number is ever formed.

The launch gate reads the emitted JSON: if the memory-bound fraction turns
out to be a minor term at the candidate points, the bandwidth-first framing
of the study needs revisiting before the sweep is sealed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

GAP_DECOMPOSITION_SCHEMA = "decode-gap-decomposition/v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _point_report(sim: Any, precision: Any, *, label: str, spec: Mapping[str, Any],
                  workload: Mapping[str, Any], stride: int) -> dict[str, Any]:
    hw_over = {
        "MLEN": int(spec["mlen"]),
        "VLEN": int(spec["mlen"]),
        "BLEN": int(spec["blen"]),
        "HLEN": int(spec["hlen"]),
        **sim.hbm_overrides(str(spec["hbm_gen"]), int(spec["hbm_channels"])),
    }
    metrics = sim.evaluate(
        precision,
        batch=int(spec["batch"]),
        input_seq=int(workload["input_seq"]),
        output_seq=int(workload["output_seq"]),
        hw_over=hw_over,
        n_chips=int(spec["chip_count"]),
        stride=stride,
        hbm_gen=str(spec["hbm_gen"]),
        hbm_channels=int(spec["hbm_channels"]),
    )
    values = {
        "label": label,
        "point": dict(spec),
        "tokens_per_second": metrics.tps,
        "tpot_ms": metrics.tpot * 1e3,
        "bottleneck": metrics.bottleneck,
        "classical_roofline_bottleneck": metrics.classical_roofline_bottleneck,
        "architecture_issue_bottleneck": metrics.architecture_issue_bottleneck,
        "frac_mem_bound": metrics.frac_mem_bound,
        "frac_classical_mem_bound": metrics.frac_classical_mem_bound,
        "frac_architecture_issue_mem_bound": (
            metrics.frac_architecture_issue_mem_bound
        ),
        "frac_serialization_bound": metrics.frac_serialization_bound,
        "avg_hbm_bytes_per_generated_token": (
            metrics.avg_hbm_bytes_per_generated_token
        ),
        "max_runtime_batch": metrics.max_runtime_batch,
        "fits_runtime": metrics.fits_runtime,
        "area_mm2": metrics.area_mm2,
        "n_chips": metrics.n_chips,
    }
    for name, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label}: {name} is not finite")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-evidence", required=True)
    parser.add_argument("--gpu-baseline-report")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args(argv)

    config = _load_json(args.config)
    from decode_dse.simulator_bridge import DecodeSimulator

    workload = config["reference_workload"]
    space = config["hardware_space"]
    reference = config["reference_chip"]
    generation = str(space["HBM_GENERATION"])
    sim = DecodeSimulator(
        str(config["sim_model"]),
        timing_evidence=args.timing_evidence,
    )
    sim.use_calibrated_bandwidth()
    sim._dd.set_area_model("proxy")

    def _precision(bits: int) -> Any:
        return sim.make_precision(
            attn_w=bits, ffn_w=bits, key=bits, value=bits,
            w_fmt="mxint", key_fmt="mxint", value_fmt="mxint",
            act_w=min(8, bits * 2), act_fmt="mxint", block=8,
        )

    channels = tuple(int(value) for value in space["HBM_CHANNELS"])
    chip_counts = tuple(int(value) for value in space["CHIP_COUNT"])
    batches = tuple(int(value) for value in space["BATCH"])
    points = {
        "reference_chip_narrowest": {
            "mlen": reference["MLEN"], "blen": reference["BLEN"],
            "hlen": reference["HLEN"], "hbm_gen": generation,
            "hbm_channels": min(channels), "chip_count": min(chip_counts),
            "batch": int(workload["batch"]),
        },
        "reference_chip_widest": {
            "mlen": reference["MLEN"], "blen": reference["BLEN"],
            "hlen": reference["HLEN"], "hbm_gen": generation,
            "hbm_channels": max(channels), "chip_count": max(chip_counts),
            "batch": max(batches),
        },
    }
    rows = []
    for bits, tag in ((8, "w8"), (4, "w4")):
        precision = _precision(bits)
        for label, spec in points.items():
            rows.append(
                _point_report(
                    sim,
                    precision,
                    label=f"{label}_{tag}",
                    spec=spec,
                    workload=workload,
                    stride=max(1, int(args.stride)),
                )
            )

    context = None
    if args.gpu_baseline_report:
        report = _load_json(args.gpu_baseline_report)
        measured = [
            {
                "batch_size": int(result["batch_size"]),
                "device_label": str(result.get("device_label", "")),
                "tokens_per_second": float(result["summary"]["tokens_per_second"]),
                "evidence_tier": "measured",
            }
            for result in report.get("results", ())
        ]
        best = max(measured, key=lambda row: row["tokens_per_second"])
        context = {
            "gpu_measured": measured,
            "gpu_best": best,
            "ratio_semantics": "model_estimate_over_measured_gpu",
            "not_a_headline_claim": True,
            "per_point_context_ratio": {
                row["label"]: row["tokens_per_second"] / best["tokens_per_second"]
                for row in rows
            },
        }

    memory_fracs = [row["frac_mem_bound"] for row in rows]
    payload = {
        "schema_version": GAP_DECOMPOSITION_SCHEMA,
        "model_name": config["model_name"],
        "workload": dict(workload),
        "stride": max(1, int(args.stride)),
        "timing_note": (
            "analytic stage-calibrated pricing; area uses the labelled proxy "
            "for speed and is not a ranking number"
        ),
        "parallelism_note": (
            "points are priced without a tensor-parallel compute split, so "
            "multi-chip compute ceilings here are pessimistic; the study "
            "prices TP/KVP per candidate"
        ),
        "points": rows,
        "gpu_context": context,
        "bandwidth_first_framing_supported": max(memory_fracs) >= 0.5,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "bandwidth_first_framing_supported": payload[
            "bandwidth_first_framing_supported"
        ],
        "max_frac_mem_bound": max(memory_fracs),
        "min_frac_mem_bound": min(memory_fracs),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
