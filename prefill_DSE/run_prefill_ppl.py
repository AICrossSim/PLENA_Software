#!/usr/bin/env python3
"""Batch WikiText2 PPL evaluator for prefill-only DSE.

This runner mirrors the BFCL DSE grid but evaluates teacher-forcing PPL.  It
uses subprocess isolation per trial and defaults to GPTQ cache load-only so PPL
runs cannot mutate or regenerate the BFCL GPTQ weight cache unless explicitly
requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "search_space.yaml"
RESULT_FIELDS = [
    "trial_id",
    "act",
    "kv",
    "fp_setting",
    "status",
    "ppl",
    "nll",
    "num_tokens",
    "nsamples",
    "runtime_sec",
    "gpu",
    "returncode",
    "gptq_cache_mode",
    "gptq_cache_key",
    "gptq_cache_hit",
    "gptq_cache_path",
    "log_dir",
    "error_message",
    "proxy_cost",
    "act_bits_proxy",
    "kv_bits_proxy",
    "fp_bits_proxy",
]
JOIN_FIELDS = [
    "trial_id",
    "act",
    "kv",
    "fp_setting",
    "bfcl_accuracy",
    "bfcl_correct",
    "bfcl_total",
    "bfcl_format_error_count",
    "ppl",
    "nll",
    "num_tokens",
    "ppl_nsamples",
    "ppl_status",
]


@dataclass(frozen=True)
class Trial:
    trial_id: str
    act: str
    kv: str
    fp_setting: str
    index: int


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"Expected list in search space, got {type(value).__name__}")


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _safe_token(text: str) -> str:
    return text.replace("/", "_").replace("=", "-").replace(",", "_")


def _make_trial_id(act: str, kv: str, fp: str) -> str:
    stem = f"act-{_safe_token(act)}__kv-{_safe_token(kv)}__fp-{_safe_token(fp)}"
    return f"{stem}__{_short_hash(stem)}"


def build_trials(cfg: dict[str, Any]) -> list[Trial]:
    space = cfg.get("search_space", {})
    acts = _as_list(space.get("ACT", []))
    kvs = _as_list(space.get("KV", []))
    fps = _as_list(space.get("FP_SETTING", []))
    trials: list[Trial] = []
    idx = 0
    for act in acts:
        for kv in kvs:
            for fp in fps:
                trials.append(Trial(_make_trial_id(act, kv, fp), act, kv, fp, idx))
                idx += 1
    return trials


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _mx_bits(token: str) -> int:
    if token.startswith("MXINT_"):
        return int(token.rsplit("_", 1)[1])
    if token.startswith("MXFP_E"):
        body = token.removeprefix("MXFP_E")
        exp_s, mant_s = body.split("M", 1)
        return 1 + int(exp_s) + int(mant_s)
    return 0


def _fp_bits(token: str) -> int:
    if token.startswith("FP_E"):
        body = token.removeprefix("FP_E")
        exp_s, mant_s = body.split("M", 1)
        return 1 + int(exp_s) + int(mant_s)
    return 0


def _proxy_cost(act: str, kv: str, fp: str) -> tuple[int, int, int, int]:
    act_b = _mx_bits(act)
    kv_b = _mx_bits(kv)
    fp_b = _fp_bits(fp)
    return act_b + kv_b + fp_b, act_b, kv_b, fp_b


def _parse_nullable_int(value: str | int | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if str(value).lower() in {"none", "null", "full", ""}:
        return None
    return int(value)


def _read_status(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _upsert_csv(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    clean_row = {k: row.get(k, "") for k in RESULT_FIELDS}
    trial_id = str(clean_row.get("trial_id", ""))
    if not trial_id:
        raise ValueError("Cannot write PPL result row without trial_id")

    with lock:
        rows: list[dict[str, Any]] = []
        replaced = False
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for existing in csv.DictReader(f):
                    if existing.get("trial_id") == trial_id:
                        if not replaced:
                            rows.append(clean_row)
                            replaced = True
                    else:
                        rows.append({k: existing.get(k, "") for k in RESULT_FIELDS})
        if not replaced:
            rows.append(clean_row)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)


def _write_design_space(path: Path, trials: list[Trial]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "index",
                "act",
                "kv",
                "fp_setting",
                "proxy_cost",
                "act_bits_proxy",
                "kv_bits_proxy",
                "fp_bits_proxy",
            ],
        )
        writer.writeheader()
        for t in trials:
            pc, ab, kb, fb = _proxy_cost(t.act, t.kv, t.fp_setting)
            writer.writerow({
                "trial_id": t.trial_id,
                "index": t.index,
                "act": t.act,
                "kv": t.kv,
                "fp_setting": t.fp_setting,
                "proxy_cost": pc,
                "act_bits_proxy": ab,
                "kv_bits_proxy": kb,
                "fp_bits_proxy": fb,
            })


def _find_eval_results_json(trial_dir: Path) -> Path | None:
    candidates = sorted(trial_dir.glob("run-*/results.json"))
    return candidates[-1] if candidates else None


def _extract_ppl_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cache = data.get("gptq_cache", {})
    if not isinstance(cache, dict):
        cache = {}
    return {
        "ppl": data.get("ppl", ""),
        "nll": data.get("nll", ""),
        "num_tokens": data.get("num_tokens", ""),
        "nsamples": data.get("nsamples", ""),
        "gptq_cache_mode": cache.get("mode", ""),
        "gptq_cache_key": cache.get("key", ""),
        "gptq_cache_hit": cache.get("hit", ""),
        "gptq_cache_path": cache.get("path", ""),
    }


def _base_command(cfg: dict[str, Any], trial: Trial, trial_dir: Path, args: argparse.Namespace) -> list[str]:
    model = cfg["model"]
    gptq = cfg.get("gptq", {})
    runtime = cfg.get("runtime", {})
    ppl = cfg.get("ppl", {})

    max_samples = _parse_nullable_int(args.ppl_max_samples, ppl.get("max_samples", 64))
    seqlen = int(args.ppl_seqlen or ppl.get("seqlen", 1024))
    gptq_max_layers = _parse_nullable_int(args.gptq_max_layers, gptq.get("max_layers"))
    gptq_cache_mode = args.gptq_cache_mode or "require"
    reserve_mb = args.gpu_memory_reserve_mb
    if reserve_mb is None:
        reserve_mb = runtime.get("gpu_memory_reserve_mb", 0)
    reserve_wait_sec = args.gpu_memory_reserve_wait_sec
    if reserve_wait_sec is None:
        reserve_wait_sec = runtime.get("gpu_memory_reserve_wait_sec", 600)
    reserve_poll_sec = args.gpu_memory_reserve_poll_sec
    if reserve_poll_sec is None:
        reserve_poll_sec = runtime.get("gpu_memory_reserve_poll_sec", 5.0)
    reserve_chunk_mb = args.gpu_memory_reserve_chunk_mb
    if reserve_chunk_mb is None:
        reserve_chunk_mb = runtime.get("gpu_memory_reserve_chunk_mb", 512)
    reserve_disable = args.gpu_memory_reserve_disable or bool(runtime.get("gpu_memory_reserve_disable", False))

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "quant_eval.cli.eval_phase_ppl",
        "--model_name", str(model["model_name"]),
        "--dtype", str(model.get("dtype", "bfloat16")),
        "--quant_config", str(model.get("quant_config", "quant_eval/configs/llama_mxint4.toml")),
        "--model_family", str(model.get("model_family", "llama")),
        "--device_id", "cuda:0",
        "--dataset", str(args.dataset or ppl.get("dataset", "wikitext")),
        "--subset", str(args.subset if args.subset is not None else ppl.get("subset", "wikitext-2-raw-v1")),
        "--split", str(args.split or ppl.get("split", "test")),
        "--seqlen", str(seqlen),
        "--max_samples", "null" if max_samples is None else str(max_samples),
        "--gptq_dataset", str(gptq["dataset"]),
        "--gptq_nsamples", str(gptq.get("nsamples", 32)),
        "--gptq_seqlen", str(gptq.get("seqlen", 1024)),
        "--gptq_format", str(gptq.get("format", "mxint")),
        "--gptq_weight_width", str(gptq.get("weight_width", 8)),
        "--gptq_weight_block_size", str(gptq.get("weight_block_size", 32)),
        "--gptq_cali_batch_size", str(gptq.get("cali_batch_size", 1)),
        "--gptq_cache_mode", str(gptq_cache_mode),
        "--decode_weight_mode", str(runtime.get("decode_weight_mode", "fp")),
        "--gpu_memory_reserve_mb", str(reserve_mb or 0),
        "--gpu_memory_reserve_wait_sec", str(reserve_wait_sec),
        "--gpu_memory_reserve_poll_sec", str(reserve_poll_sec),
        "--gpu_memory_reserve_chunk_mb", str(reserve_chunk_mb),
        "--act_element_width_prefill", trial.act,
        "--kv_element_width_prefill", trial.kv,
        "--fp_setting_prefill", trial.fp_setting,
        "--log_dir", str(trial_dir),
    ]
    if reserve_disable:
        cmd += ["--gpu_memory_reserve_disable", "true"]
    if gptq.get("cache_dir"):
        cmd += ["--gptq_cache_dir", str(gptq.get("cache_dir"))]
    if gptq_max_layers is not None:
        cmd += ["--gptq_max_layers", str(gptq_max_layers)]
    return [x for x in cmd if x != ""]


def _run_trial(cfg: dict[str, Any], trial: Trial, trial_dir: Path, gpu: str, args: argparse.Namespace) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_command(cfg, trial, trial_dir, args)
    timeout = int(args.trial_timeout_sec or cfg.get("runtime", {}).get("trial_timeout_sec", 7200))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MPLCONFIGDIR"] = str(trial_dir / "matplotlib")

    _write_json(trial_dir / "trial_config.json", {"trial": trial.__dict__, "gpu": gpu, "timeout_sec": timeout})
    (trial_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    start = time.time()
    status = "failed"
    error = ""
    returncode = None
    try:
        with (trial_dir / "stdout.log").open("w", encoding="utf-8") as out, (trial_dir / "stderr.log").open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=err, timeout=timeout)
        returncode = proc.returncode
        status = "success" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            error = f"returncode={proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        error = f"timeout after {timeout}s"
    except Exception as exc:
        status = "failed"
        error = repr(exc)

    runtime_sec = time.time() - start
    metrics = _extract_ppl_metrics(_find_eval_results_json(trial_dir))
    pc, ab, kb, fb = _proxy_cost(trial.act, trial.kv, trial.fp_setting)
    row = {
        "trial_id": trial.trial_id,
        "act": trial.act,
        "kv": trial.kv,
        "fp_setting": trial.fp_setting,
        "status": status,
        "ppl": metrics.get("ppl", ""),
        "nll": metrics.get("nll", ""),
        "num_tokens": metrics.get("num_tokens", ""),
        "nsamples": metrics.get("nsamples", ""),
        "runtime_sec": f"{runtime_sec:.2f}",
        "gpu": gpu,
        "returncode": returncode if returncode is not None else "",
        "gptq_cache_mode": metrics.get("gptq_cache_mode", args.gptq_cache_mode),
        "gptq_cache_key": metrics.get("gptq_cache_key", ""),
        "gptq_cache_hit": metrics.get("gptq_cache_hit", ""),
        "gptq_cache_path": metrics.get("gptq_cache_path", ""),
        "log_dir": str(trial_dir),
        "error_message": error,
        "proxy_cost": pc,
        "act_bits_proxy": ab,
        "kv_bits_proxy": kb,
        "fp_bits_proxy": fb,
    }
    _write_json(trial_dir / "status.json", row)
    return row


def _should_run(trial_dir: Path, args: argparse.Namespace) -> bool:
    if args.rerun_success:
        return True
    status = _read_status(trial_dir)
    if status is None:
        return True
    if status.get("status") == "success":
        return False
    if args.retry_failed:
        return True
    return False


def _worker(worker_idx: int, gpu: str, cfg: dict[str, Any], q: queue.Queue, args: argparse.Namespace, run_dir: Path, results_csv: Path, csv_lock: threading.Lock, pbar: tqdm) -> None:
    while True:
        try:
            trial = q.get_nowait()
        except queue.Empty:
            return
        try:
            trial_dir = run_dir / "ppl_trials" / trial.trial_id
            row = _run_trial(cfg, trial, trial_dir, gpu, args)
            _upsert_csv(results_csv, row, csv_lock)
        finally:
            q.task_done()
            pbar.update(1)


def _read_csv_by_trial(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row.get("trial_id", ""): row for row in csv.DictReader(f) if row.get("trial_id")}


def _write_joined(bfcl_csv: Path, ppl_csv: Path, out_csv: Path) -> None:
    bfcl = _read_csv_by_trial(bfcl_csv)
    ppl = _read_csv_by_trial(ppl_csv)
    trial_ids = list(bfcl.keys()) if bfcl else list(ppl.keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOIN_FIELDS)
        writer.writeheader()
        for trial_id in trial_ids:
            b = bfcl.get(trial_id, {})
            p = ppl.get(trial_id, {})
            writer.writerow({
                "trial_id": trial_id,
                "act": p.get("act") or b.get("act", ""),
                "kv": p.get("kv") or b.get("kv", ""),
                "fp_setting": p.get("fp_setting") or b.get("fp_setting", ""),
                "bfcl_accuracy": b.get("accuracy", ""),
                "bfcl_correct": b.get("correct", ""),
                "bfcl_total": b.get("total", ""),
                "bfcl_format_error_count": b.get("format_error_count", ""),
                "ppl": p.get("ppl", ""),
                "nll": p.get("nll", ""),
                "num_tokens": p.get("num_tokens", ""),
                "ppl_nsamples": p.get("nsamples", ""),
                "ppl_status": p.get("status", ""),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiText2 PPL batch evaluator for prefill DSE")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--only-trials", default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerun-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trial-timeout-sec", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--ppl-seqlen", type=int, default=None)
    parser.add_argument("--ppl-max-samples", default="64", help="Number of WikiText2 blocks; use 'none' for full split")
    parser.add_argument("--gptq-max-layers", default=None)
    parser.add_argument("--gptq-cache-mode", default="require", choices=["off", "auto", "refresh", "require"])
    parser.add_argument("--gpu-memory-reserve-mb", type=int, default=None)
    parser.add_argument("--gpu-memory-reserve-wait-sec", type=int, default=None)
    parser.add_argument("--gpu-memory-reserve-poll-sec", type=float, default=None)
    parser.add_argument("--gpu-memory-reserve-chunk-mb", type=int, default=None)
    parser.add_argument("--gpu-memory-reserve-disable", action="store_true")
    parser.add_argument("--join-bfcl-results", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _load_yaml(cfg_path)
    all_trials = build_trials(cfg)

    only = set(_split_csv(args.only_trials) or [])
    trials = [t for t in all_trials if not only or t.trial_id in only]
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    run_name = args.run_name or f"prefill_ppl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = REPO_ROOT / "prefill_DSE" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_design_space(run_dir / "design_space.csv", all_trials)
    _write_json(run_dir / "run_config.json", {"config_path": str(cfg_path), "args": vars(args), "n_design_points": len(all_trials)})

    pending = [t for t in trials if _should_run(run_dir / "ppl_trials" / t.trial_id, args)]
    print(f"Design points total: {len(all_trials)}")
    print(f"Selected trials     : {len(trials)}")
    print(f"Pending trials      : {len(pending)}")
    print(f"Run dir             : {run_dir}")

    if args.dry_run:
        for t in pending[:10]:
            cmd = _base_command(cfg, t, run_dir / "ppl_trials" / t.trial_id, args)
            print(f"[{t.trial_id}] {' '.join(cmd)}")
        if args.join_bfcl_results:
            print(f"Join target: {run_dir / 'joined_bfcl_ppl.csv'}")
        return

    q: queue.Queue[Trial] = queue.Queue()
    for t in pending:
        q.put(t)

    gpus = _split_csv(args.gpus) or ["0"]
    csv_lock = threading.Lock()
    results_csv = run_dir / "results_ppl.csv"
    with tqdm(total=len(pending), desc="Prefill PPL", unit="trial") as pbar:
        threads = []
        for idx, gpu in enumerate(gpus):
            th = threading.Thread(
                target=_worker,
                args=(idx, gpu, cfg, q, args, run_dir, results_csv, csv_lock, pbar),
                daemon=True,
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

    print(f"PPL results CSV: {results_csv}")
    if args.join_bfcl_results:
        bfcl_csv = Path(args.join_bfcl_results)
        if not bfcl_csv.is_absolute():
            bfcl_csv = REPO_ROOT / bfcl_csv
        joined = run_dir / "joined_bfcl_ppl.csv"
        _write_joined(bfcl_csv, results_csv, joined)
        print(f"Joined CSV: {joined}")


if __name__ == "__main__":
    main()
