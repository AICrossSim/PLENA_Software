#!/usr/bin/env python
"""Collect true BFCL multiple prompt/output token lengths from an FP model.

The normal BFCL generate path records the local OpenAI-compatible server usage
fields as ``input_token_count`` and ``output_token_count`` in the result JSONL.
This helper keeps the FP model resident for the full BFCL multiple pass, then
summarizes those fields into CSV/JSONL/summary artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = (
    "/data/models/pgf23_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/"
    "snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
)
DEFAULT_ALIAS = "Qwen/Qwen3-235B-A22B-Instruct-2507-FC"
DEFAULT_GPUS = "2,3,4"
DEFAULT_BFCL_ENV_BIN = ".conda/envs/plena-bfcl/bin"
DEFAULT_BFCL_PROJECT_ROOT = ".bfcl"

LENGTH_FIELDS = ["id", "input_tokens", "output_tokens", "total_tokens", "osl_isl_ratio", "latency"]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        yield payload
        return

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record in {path} line {lineno}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object in {path} line {lineno}, got {type(item).__name__}.")
        yield item


def _find_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob("**/BFCL_v*_multiple_result.json"))
    if not files:
        files = sorted(path.glob("**/*multiple*result*.json"))
    if not files:
        raise FileNotFoundError(f"No BFCL multiple result JSON files found under {path}")
    return files


def load_length_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_file in _find_result_files(path):
        for record in _iter_json_records(result_file):
            input_tokens = record.get("input_token_count")
            output_tokens = record.get("output_token_count")
            if input_tokens is None or output_tokens is None:
                usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
            if input_tokens is None or output_tokens is None:
                raise ValueError(
                    f"Missing input/output token counts for record {record.get('id')!r} in {result_file}"
                )

            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
            rows.append(
                {
                    "id": str(record.get("id", "")),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "osl_isl_ratio": (output_tokens / input_tokens) if input_tokens else "",
                    "latency": float(record["latency"]) if record.get("latency") is not None else "",
                }
            )
    rows.sort(key=lambda row: _natural_id_key(row["id"]))
    return rows


def _natural_id_key(value: str) -> tuple[str, int | str]:
    prefix, sep, suffix = value.rpartition("_")
    if sep and suffix.isdigit():
        return (prefix, int(suffix))
    return (value, value)


def _nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": _nearest_rank(values, 90),
        "p95": _nearest_rank(values, 95),
        "p99": _nearest_rank(values, 99),
        "max": max(values),
    }


def build_summary(rows: list[dict[str, Any]], *, max_new_tokens: int | None) -> dict[str, Any]:
    latencies = [float(row["latency"]) for row in rows if row["latency"] != ""]
    output_tokens = [int(row["output_tokens"]) for row in rows]
    summary = {
        "count": len(rows),
        "isl": _metric_summary([int(row["input_tokens"]) for row in rows]),
        "osl": _metric_summary(output_tokens),
        "total_tokens": _metric_summary([int(row["total_tokens"]) for row in rows]),
        "osl_isl_ratio": _metric_summary([float(row["osl_isl_ratio"]) for row in rows if row["osl_isl_ratio"] != ""]),
        "latency": _metric_summary(latencies),
        "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens_count": (
            sum(1 for value in output_tokens if value >= max_new_tokens)
            if max_new_tokens is not None
            else None
        ),
    }
    return summary


def write_outputs(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_new_tokens: int | None,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "lengths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LENGTH_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = out_dir / "lengths.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(rows, max_new_tokens=max_new_tokens)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    config_path = out_dir / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    _write_length_figure(rows, out_dir)

    return summary


def _write_length_figure(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    metrics = [
        ("ISL", [float(row["input_tokens"]) for row in rows], "Prompt tokens"),
        ("OSL", [float(row["output_tokens"]) for row in rows], "Output tokens"),
        ("OSL / ISL", [float(row["osl_isl_ratio"]) for row in rows if row["osl_isl_ratio"] != ""], "Ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), constrained_layout=True)
    for ax, (title, values, ylabel) in zip(axes, metrics, strict=True):
        bins = "auto"
        if title == "OSL / ISL":
            # Keep a long-thinking outlier from flattening the visible mass.
            p99 = _nearest_rank(values, 99)
            plot_values = [min(value, p99) for value in values] if p99 is not None else values
            xlabel = f"{ylabel} (clipped at p99={p99:.2f})" if p99 is not None else ylabel
        else:
            plot_values = values
            xlabel = ylabel
        ax.hist(plot_values, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.7)
        mean = statistics.fmean(values)
        p50 = statistics.median(values)
        ax.axvline(mean, color="#E15759", linewidth=1.2, label=f"mean={mean:.1f}")
        ax.axvline(p50, color="#59A14F", linewidth=1.2, linestyle="--", label=f"p50={p50:.1f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.legend(fontsize=7, frameon=False)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle("BFCL Multiple Length Distributions", fontsize=12)
    fig.savefig(figure_dir / "bfcl_multiple_isl_osl_ratio_bars.pdf")
    fig.savefig(figure_dir / "bfcl_multiple_isl_osl_ratio_bars.png", dpi=220)
    fig.savefig(figure_dir / "bfcl_multiple_isl_osl_ratio_hist.pdf")
    fig.savefig(figure_dir / "bfcl_multiple_isl_osl_ratio_hist.png", dpi=220)
    plt.close(fig)


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    project_root = _resolve_repo_path(args.bfcl_project_root)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    bfcl_bin = _resolve_repo_path(args.bfcl_env_bin)
    os.environ["PATH"] = f"{bfcl_bin}:{os.environ.get('PATH', '')}"


def run_bfcl_generate(args: argparse.Namespace, run_dir: Path) -> Path:
    _configure_environment(args)

    # Import after CUDA_VISIBLE_DEVICES is set so torch sees the intended GPU set.
    from quant_eval.cli.eval_phase_bfcl import main as eval_phase_bfcl_main

    bfcl_log_dir = run_dir / "bfcl_generate"
    result = eval_phase_bfcl_main(
        model_name=args.model_name,
        device_id=args.device_id,
        dtype=args.dtype,
        quant_config="none",
        model_parallel=True,
        bfcl_test_categories=["multiple"],
        bfcl_model_alias=args.bfcl_model_alias,
        bfcl_adapter=args.bfcl_adapter,
        model_family=args.model_family,
        bfcl_num_threads=args.num_threads,
        server_host=args.server_host,
        server_port=args.server_port,
        bfcl_tool_mode="return",
        bfcl_max_new_tokens=args.max_new_tokens,
        bfcl_generate_mode=args.bfcl_generate_mode,
        bfcl_batch_size=args.batch_size,
        bfcl_batch_length_bucket=args.batch_length_bucket,
        limit=args.limit,
        log_dir=str(bfcl_log_dir),
        run_evaluate=args.run_evaluate,
    )
    result_dir_value = result.get("bfcl_result_dir") or result.get("bfcl_result_path")
    if result_dir_value:
        result_dir = Path(result_dir_value)
    else:
        result_candidates = sorted(bfcl_log_dir.glob("**/bfcl_results"))
        if not result_candidates:
            raise KeyError("eval_phase_bfcl result did not include bfcl_result_dir and no bfcl_results dir was found")
        result_dir = result_candidates[-1]
    if not result_dir.exists():
        raise FileNotFoundError(f"BFCL result dir was not created: {result_dir}")
    return result_dir


def run_bfcl_evaluate(args: argparse.Namespace, result_path: Path, out_dir: Path) -> dict[str, Any]:
    _configure_environment(args)
    score_dir = out_dir / "bfcl_scores"
    cmd = [
        "bfcl",
        "evaluate",
        "--model", args.bfcl_model_alias,
        "--test-category", "multiple",
        "--result-dir", str(result_path),
        "--score-dir", str(score_dir),
    ]
    if args.limit is not None:
        cmd.append("--partial-eval")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    (out_dir / "bfcl_evaluate_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "bfcl_evaluate_stderr.log").write_text(proc.stderr, encoding="utf-8")
    score_jsons = sorted(score_dir.glob("**/BFCL_v*_multiple_score.json"))
    overall_csvs = sorted(score_dir.glob("**/data_overall.csv"))
    return {
        "returncode": proc.returncode,
        "score_dir": str(score_dir),
        "score_files": [str(path) for path in score_jsons],
        "overall_csv": str(overall_csvs[-1]) if overall_csvs else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="Physical GPUs to expose, e.g. 2,3,4.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--bfcl-model-alias", default=DEFAULT_ALIAS)
    parser.add_argument("--bfcl-adapter", default="qwen3_fc")
    parser.add_argument("--model-family", default="qwen3_moe")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-id", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--bfcl-generate-mode", choices=["cli", "batched"], default="batched")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length-bucket", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-evaluate", action="store_true")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parse-only", default=None, help="Parse an existing BFCL result file or directory.")
    parser.add_argument("--bfcl-env-bin", default=DEFAULT_BFCL_ENV_BIN)
    parser.add_argument("--bfcl-project-root", default=DEFAULT_BFCL_PROJECT_ROOT)
    parser.add_argument("--online", dest="offline", action="store_false", help="Do not set HF offline environment flags.")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(offline=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "prefill_DSE" / "runs" / f"bfcl_multiple_lengths_{_timestamp()}"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    run_config = {
        "model_name": args.model_name,
        "bfcl_model_alias": args.bfcl_model_alias,
        "bfcl_adapter": args.bfcl_adapter,
        "model_family": args.model_family,
        "category": "multiple",
        "gpus": args.gpus,
        "model_parallel": True,
        "quant_config": "none",
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "num_threads": args.num_threads,
        "bfcl_generate_mode": args.bfcl_generate_mode,
        "batch_size": args.batch_size,
        "batch_length_bucket": args.batch_length_bucket,
        "run_evaluate": args.run_evaluate,
        "server": f"http://{args.server_host}:{args.server_port}",
        "parse_only": args.parse_only,
        "offline": args.offline,
    }

    command = " ".join([sys.executable, *sys.argv])
    if args.dry_run:
        print(f"Output dir          : {out_dir}")
        print(f"CUDA_VISIBLE_DEVICES: {args.gpus}")
        print(f"Model               : {args.model_name}")
        print(f"BFCL alias          : {args.bfcl_model_alias}")
        print(f"Max new tokens      : {args.max_new_tokens}")
        print(f"Limit               : {args.limit if args.limit is not None else 'full multiple'}")
        print(f"Generate mode       : {args.bfcl_generate_mode}")
        print(f"Batch size          : {args.batch_size}")
        print(f"Run evaluate        : {args.run_evaluate}")
        print(f"Parse only          : {args.parse_only or 'no'}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    if args.parse_only:
        result_path = _resolve_repo_path(args.parse_only)
    else:
        result_path = run_bfcl_generate(args, out_dir)
    run_config["bfcl_result_path"] = str(result_path)

    rows = load_length_rows(result_path)
    if args.parse_only and args.run_evaluate:
        run_config["bfcl_evaluate"] = run_bfcl_evaluate(args, result_path, out_dir)
    summary = write_outputs(rows, out_dir, max_new_tokens=args.max_new_tokens, run_config=run_config)

    print(f"Rows        : {len(rows)}")
    print(f"Output dir  : {out_dir}")
    print(f"lengths.csv : {out_dir / 'lengths.csv'}")
    print(f"summary.json: {out_dir / 'summary.json'}")
    print(f"figure      : {out_dir / 'figures' / 'bfcl_multiple_isl_osl_ratio_bars.pdf'}")
    print(
        "ISL mean/max: "
        f"{summary['isl']['mean']:.2f}/{summary['isl']['max']} | "
        "OSL mean/max: "
        f"{summary['osl']['mean']:.2f}/{summary['osl']['max']}"
    )
    print(
        "OSL/ISL mean/p50/max: "
        f"{summary['osl_isl_ratio']['mean']:.4f}/"
        f"{summary['osl_isl_ratio']['p50']:.4f}/"
        f"{summary['osl_isl_ratio']['max']:.4f}"
    )
    print(f"Hit max_new_tokens: {summary['hit_max_new_tokens_count']}")


if __name__ == "__main__":
    main()
