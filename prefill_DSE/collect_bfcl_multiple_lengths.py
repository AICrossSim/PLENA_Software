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

LENGTH_FIELDS = ["id", "input_tokens", "output_tokens", "total_tokens", "latency"]


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

    return summary


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
        gpu_memory_reserve_mb=0,
        gpu_memory_reserve_disable=True,
        limit=args.limit,
        log_dir=str(bfcl_log_dir),
        run_evaluate=False,
    )
    result_dir = Path(result["bfcl_result_dir"])
    if not result_dir.exists():
        raise FileNotFoundError(f"BFCL result dir was not created: {result_dir}")
    return result_dir


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
    summary = write_outputs(rows, out_dir, max_new_tokens=args.max_new_tokens, run_config=run_config)

    print(f"Rows        : {len(rows)}")
    print(f"Output dir  : {out_dir}")
    print(f"lengths.csv : {out_dir / 'lengths.csv'}")
    print(f"summary.json: {out_dir / 'summary.json'}")
    print(
        "ISL mean/max: "
        f"{summary['isl']['mean']:.2f}/{summary['isl']['max']} | "
        "OSL mean/max: "
        f"{summary['osl']['mean']:.2f}/{summary['osl']['max']}"
    )
    print(f"Hit max_new_tokens: {summary['hit_max_new_tokens_count']}")


if __name__ == "__main__":
    main()
