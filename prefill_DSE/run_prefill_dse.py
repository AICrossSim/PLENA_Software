#!/usr/bin/env python3
"""Batch evaluator for prefill-only BFCL DSE.

The default YAML is a full run: full GPTQ and full BFCL multiple. Use CLI
overrides such as --limit 2 and --gptq-max-layers 1 for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import socket
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
    "accuracy",
    "correct",
    "total",
    "format_error_count",
    "raw_output_count",
    "runtime_sec",
    "gpu",
    "server_port",
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


def _parse_gptq_max_layers(value: str | None, default: Any) -> int | None:
    if value is None:
        return default
    if value.lower() in {"none", "null", "full", ""}:
        return None
    return int(value)


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _mx_bits(token: str) -> int:
    if token.startswith("MXINT_"):
        return int(token.rsplit("_", 1)[1])
    if token.startswith("MXFP_E"):
        # E4M3 means 1 sign + 4 exponent + 3 mantissa bits.
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


def _read_status(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _upsert_csv(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    """Write one trial result, replacing an existing row for the same trial.

    Retry runs should update the canonical result for a design point instead of
    appending a second row after the old failed entry.  Keep first-seen order so
    the CSV remains aligned with the deterministic design-space order, but make
    the row content reflect the latest completed attempt.
    """
    clean_row = {k: row.get(k, "") for k in RESULT_FIELDS}
    trial_id = str(clean_row.get("trial_id", ""))
    if not trial_id:
        raise ValueError("Cannot write DSE result row without trial_id")

    with lock:
        rows: list[dict[str, Any]] = []
        replaced = False
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for existing in reader:
                    if existing.get("trial_id") == trial_id:
                        if not replaced:
                            rows.append(clean_row)
                            replaced = True
                        # Drop duplicate stale rows for the same trial.
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
        writer = csv.DictWriter(f, fieldnames=["trial_id", "index", "act", "kv", "fp_setting", "proxy_cost", "act_bits_proxy", "kv_bits_proxy", "fp_bits_proxy"])
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


def _find_score_dir(trial_dir: Path) -> Path | None:
    candidates = sorted(trial_dir.glob("**/bfcl_scores"))
    return candidates[-1] if candidates else None


def _find_results_dir(trial_dir: Path) -> Path | None:
    candidates = sorted(trial_dir.glob("**/bfcl_results"))
    return candidates[-1] if candidates else None


def _find_eval_results_json(trial_dir: Path) -> Path | None:
    candidates = sorted(trial_dir.glob("run-*/results.json"))
    return candidates[-1] if candidates else None


def _extract_results_json_metrics(path: Path | None) -> tuple[Any, Any, Any, int, int]:
    if path is None or not path.exists():
        return "", "", "", 0, 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", "", "", 0, 0
    per_category = data.get("per_category", {})
    accuracy = correct = total = ""
    raw_count = 0
    format_errors = 0
    for entries in per_category.values():
        if isinstance(entries, dict):
            if "accuracy" in entries and accuracy == "":
                accuracy = entries.get("accuracy", "")
                correct = entries.get("correct_count", "")
                total = entries.get("total_count", "")
            continue
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            if "accuracy" in item and accuracy == "":
                accuracy = item.get("accuracy", "")
                correct = item.get("correct_count", "")
                total = item.get("total_count", "")
                continue
            if "model_result_raw" in item:
                raw_count += 1
            error_type = str(item.get("error_type", "")).lower()
            if "decoder" in error_type or "format" in error_type:
                format_errors += 1
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        total_int = None
    if total_int is not None and raw_count < total_int:
        raw_count = total_int
    return accuracy, correct, total, raw_count, format_errors



def _extract_gptq_cache_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cache = data.get("gptq_cache", {})
    return cache if isinstance(cache, dict) else {}


def _extract_accuracy(score_dir: Path | None) -> tuple[Any, Any, Any]:
    if score_dir is None:
        return "", "", ""
    overall = score_dir / "data_overall.csv"
    if not overall.exists():
        return "", "", ""
    try:
        rows = list(csv.DictReader(overall.open("r", encoding="utf-8")))
    except Exception:
        return "", "", ""
    if not rows:
        return "", "", ""
    row = rows[0]
    accuracy = row.get("accuracy") or row.get("Accuracy") or row.get("overall_accuracy") or ""
    correct = row.get("correct") or row.get("Correct") or ""
    total = row.get("total") or row.get("Total") or ""
    return accuracy, correct, total


def _count_raw_outputs(result_dir: Path | None) -> tuple[int, int]:
    if result_dir is None:
        return 0, 0
    raw_count = 0
    format_errors = 0
    for path in result_dir.rglob("*.json*"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            if not line.strip():
                continue
            raw_count += 1
            lower = line.lower()
            if "decoder_failed" in lower or "ast_decoder" in lower or "format" in lower and "error" in lower:
                format_errors += 1
    return raw_count, format_errors


def _cleanup_trial_tmp(trial_dir: Path, compact: bool) -> None:
    patterns = ["tmp", "*.pt.tmp", "*gptq_cache*", "*checkpoint*", "*.safetensors.tmp"]
    for pattern in patterns:
        for p in trial_dir.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    if compact:
        for name in ("bfcl_results", "bfcl_scores"):
            for p in trial_dir.glob(f"**/{name}"):
                shutil.rmtree(p, ignore_errors=True)


def _base_command(cfg: dict[str, Any], trial: Trial, trial_dir: Path, port: int, args: argparse.Namespace) -> list[str]:
    model = cfg["model"]
    bfcl = cfg.get("bfcl", {})
    gptq = cfg.get("gptq", {})
    runtime = cfg.get("runtime", {})

    limit = args.limit if args.limit is not None else bfcl.get("limit")
    gptq_max_layers = _parse_gptq_max_layers(args.gptq_max_layers, gptq.get("max_layers"))

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "quant_eval.cli.eval_phase_bfcl",
        "--model_name", str(model["model_name"]),
        "--bfcl_model_alias", str(model.get("bfcl_model_alias", "")),
        "--bfcl_adapter", str(model.get("bfcl_adapter", "auto")),
        "--model_family", str(model.get("model_family", "llama")),
        "--dtype", str(model.get("dtype", "bfloat16")),
        "--quant_config", str(model.get("quant_config", "quant_eval/configs/llama_mxint4.toml")),
        "--device_id", "cuda:0",
        "--bfcl_test_categories", str(bfcl.get("test_categories", "multiple")),
        "--bfcl_tool_mode", str(bfcl.get("tool_mode", "return")),
        "--bfcl_num_threads", str(bfcl.get("num_threads", 1)),
        "--bfcl_max_new_tokens", str(bfcl.get("max_new_tokens", 256)),
        "--server_port", str(port),
        "--gptq_dataset", str(gptq["dataset"]),
        "--gptq_nsamples", str(gptq.get("nsamples", 32)),
        "--gptq_seqlen", str(gptq.get("seqlen", 1024)),
        "--gptq_format", str(gptq.get("format", "mxint")),
        "--gptq_weight_width", str(gptq.get("weight_width", 8)),
        "--gptq_weight_block_size", str(gptq.get("weight_block_size", 32)),
        "--gptq_cali_batch_size", str(gptq.get("cali_batch_size", 1)),
        "--decode_weight_mode", str(runtime.get("decode_weight_mode", "fp")),
        "--gpu_memory_reserve_mb", str(runtime.get("gpu_memory_reserve_mb", 20000)),
        "--gpu_memory_reserve_wait_sec", str(runtime.get("gpu_memory_reserve_wait_sec", 600)),
        "--gpu_memory_reserve_poll_sec", str(runtime.get("gpu_memory_reserve_poll_sec", 5.0)),
        "--gpu_memory_reserve_chunk_mb", str(runtime.get("gpu_memory_reserve_chunk_mb", 512)),
        "--act_element_width_prefill", trial.act,
        "--kv_element_width_prefill", trial.kv,
        "--fp_setting_prefill", trial.fp_setting,
        "--log_dir", str(trial_dir),
    ]
    if gptq.get("cache_dir"):
        cmd += ["--gptq_cache_dir", str(gptq.get("cache_dir"))]
    if gptq.get("cache_mode"):
        cmd += ["--gptq_cache_mode", str(gptq.get("cache_mode"))]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if gptq_max_layers is not None:
        cmd += ["--gptq_max_layers", str(gptq_max_layers)]
    return [x for x in cmd if x != ""]


def _run_trial(cfg: dict[str, Any], trial: Trial, trial_dir: Path, gpu: str, port: int, args: argparse.Namespace) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = _base_command(cfg, trial, trial_dir, port, args)
    gptq = cfg.get("gptq", {})
    runtime = cfg.get("runtime", {})
    timeout = int(args.trial_timeout_sec or runtime.get("trial_timeout_sec", 7200))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MPLCONFIGDIR"] = str(trial_dir / "matplotlib")
    env["BFCL_PROJECT_ROOT"] = str((REPO_ROOT / runtime.get("bfcl_project_root", ".bfcl")).resolve())
    bfcl_bin = runtime.get("bfcl_env_bin")
    if bfcl_bin:
        env["PATH"] = f"{bfcl_bin}:{env.get('PATH', '')}"

    _write_json(trial_dir / "trial_config.json", {"trial": trial.__dict__, "gpu": gpu, "port": port, "timeout_sec": timeout})
    (trial_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    start = time.time()
    status = "failed"
    error = ""
    returncode = None
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
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
    score_dir = _find_score_dir(trial_dir)
    result_dir = _find_results_dir(trial_dir)
    eval_results_json = _find_eval_results_json(trial_dir)
    gptq_cache_meta = _extract_gptq_cache_metadata(eval_results_json)
    accuracy, correct, total, raw_count, format_errors = _extract_results_json_metrics(eval_results_json)
    if accuracy == "":
        accuracy, correct, total = _extract_accuracy(score_dir)
    if raw_count == 0:
        raw_count, format_errors = _count_raw_outputs(result_dir)
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        total_int = None
    if total_int is not None and raw_count < total_int:
        raw_count = total_int
    pc, ab, kb, fb = _proxy_cost(trial.act, trial.kv, trial.fp_setting)
    row = {
        "trial_id": trial.trial_id,
        "act": trial.act,
        "kv": trial.kv,
        "fp_setting": trial.fp_setting,
        "status": status,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "format_error_count": format_errors,
        "raw_output_count": raw_count,
        "runtime_sec": f"{runtime_sec:.2f}",
        "gpu": gpu,
        "server_port": port,
        "returncode": returncode if returncode is not None else "",
        "gptq_cache_mode": gptq_cache_meta.get("mode", gptq.get("cache_mode", "")),
        "gptq_cache_key": gptq_cache_meta.get("key", ""),
        "gptq_cache_hit": gptq_cache_meta.get("hit", ""),
        "gptq_cache_path": gptq_cache_meta.get("path", ""),
        "log_dir": str(trial_dir),
        "error_message": error,
        "proxy_cost": pc,
        "act_bits_proxy": ab,
        "kv_bits_proxy": kb,
        "fp_bits_proxy": fb,
    }
    _write_json(trial_dir / "status.json", row)
    if (trial_dir / "results.json").exists():
        pass
    _cleanup_trial_tmp(trial_dir, compact=args.compact_artifacts)
    return row


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _claim_port(host: str, preferred: int, active_ports: set[int], lock: threading.Lock, span: int = 1000) -> int:
    for port in range(preferred, preferred + span):
        with lock:
            if port in active_ports:
                continue
            if _port_is_available(host, port):
                active_ports.add(port)
                return port
    raise RuntimeError(f"No free port found in [{preferred}, {preferred + span}).")


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


def _worker(worker_idx: int, gpu: str, cfg: dict[str, Any], q: queue.Queue, args: argparse.Namespace, run_dir: Path, results_csv: Path, csv_lock: threading.Lock, port_lock: threading.Lock, active_ports: set[int], pbar: tqdm) -> None:
    preferred_port = args.base_port + worker_idx
    server_host = str(cfg.get("runtime", {}).get("server_host", "127.0.0.1"))
    while True:
        try:
            trial = q.get_nowait()
        except queue.Empty:
            return
        trial_dir = run_dir / "trials" / trial.trial_id
        port = _claim_port(server_host, preferred_port, active_ports, port_lock)
        try:
            row = _run_trial(cfg, trial, trial_dir, gpu, port, args)
            _upsert_csv(results_csv, row, csv_lock)
        finally:
            with port_lock:
                active_ports.discard(port)
            q.task_done()
            pbar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill-only BFCL DSE batch evaluator")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gptq-max-layers", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--only-trials", default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerun-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--trial-timeout-sec", type=int, default=None)
    parser.add_argument("--compact-artifacts", action="store_true")
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

    run_name = args.run_name or f"prefill_dse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = REPO_ROOT / "prefill_DSE" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_design_space(run_dir / "design_space.csv", all_trials)
    _write_json(run_dir / "run_config.json", {"config_path": str(cfg_path), "args": vars(args), "n_design_points": len(all_trials)})

    pending = [t for t in trials if _should_run(run_dir / "trials" / t.trial_id, args)]
    print(f"Design points total: {len(all_trials)}")
    print(f"Selected trials     : {len(trials)}")
    print(f"Pending trials      : {len(pending)}")
    print(f"Run dir             : {run_dir}")

    if args.dry_run:
        preview = pending[:10]
        for t in preview:
            cmd = _base_command(cfg, t, run_dir / "trials" / t.trial_id, args.base_port, args)
            print(f"[{t.trial_id}] {' '.join(cmd)}")
        return

    q: queue.Queue[Trial] = queue.Queue()
    for t in pending:
        q.put(t)

    gpus = _split_csv(args.gpus) or ["0"]
    csv_lock = threading.Lock()
    port_lock = threading.Lock()
    active_ports: set[int] = set()
    results_csv = run_dir / "results.csv"
    with tqdm(total=len(pending), desc="Prefill DSE", unit="trial") as pbar:
        threads = []
        for idx, gpu in enumerate(gpus):
            th = threading.Thread(
                target=_worker,
                args=(idx, gpu, cfg, q, args, run_dir, results_csv, csv_lock, port_lock, active_ports, pbar),
                daemon=True,
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

    print(f"Results CSV: {results_csv}")


if __name__ == "__main__":
    main()
