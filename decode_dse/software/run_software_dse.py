"""Search the mixed-precision decode space (weights, KV, activations, and the
FP_SETTING vector-unit precision) for the accuracy/throughput front.

Weight format is searchable alongside width: ``search.weight_w`` accepts MXINT
widths and MXFP tokens. Weight rotation is left to the selective search, which
only rotates the activation/KV matmuls that actually benefit.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from decode_dse.simulator_bridge import DecodeSimulator
from decode_dse.software.decode_quant import DecodeQuantSpec, gptq_cache_key, parse_prec_token
from decode_dse.software.build_task_calib import calib_path
from decode_dse.results import select_front
from decode_dse.hardware.codesign_search import make_sampler

CSV_FIELDS = [
    "tag", "cont_ppl", "prefill_ppl", "gsm8k", "ifeval",
    "attn_w_bits", "ffn_w_bits", "kv_bits", "act_bits", "block", "fp_setting",
    "ref_tps", "ref_tpot_ms", "ref_fits", "w_fmt", "kv_fmt", "act_fmt",
    "use_gptq", "use_rotation", "calib", "runtime_sec", "error",
]


def _load_cfg(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def _grid(search: dict[str, Any]) -> list[dict[str, Any]]:
    """MXINT and MXFP are searched on weights, KV and activations
    alike. ``mixed_weight`` (default false) shares one weight token across
    attention and FFN."""
    w_toks = search["weight_w"]
    kv_toks, act_toks = search["kv"], search.get("act_w", [8])
    blocks = _as_list(search.get("block", 32))
    weight_pairs = (
        [(a, f) for a in w_toks for f in w_toks] if search.get("mixed_weight", False)
        else [(w, w) for w in w_toks]
    )
    points = []
    for (awt, fwt), kvt, actt, blk in itertools.product(weight_pairs, kv_toks, act_toks, blocks):
        w_fmt, aw = parse_prec_token(awt)
        fw_fmt, fw = parse_prec_token(fwt)
        if fw_fmt != w_fmt:
            continue  # attention and FFN share the weight datapath format
        kv_fmt, kv_w = parse_prec_token(kvt)
        act_fmt, act_w = parse_prec_token(actt)
        points.append({"w_fmt": w_fmt, "attn_w": aw, "ffn_w": fw, "kv": kv_w, "kv_fmt": kv_fmt,
                       "act_w": act_w, "act_fmt": act_fmt, "block": blk})
    return points


def _precision_choices(search: dict) -> dict[str, list[str]]:
    return {
        "weight_w": [str(w) for w in search["weight_w"]],          # MXINT + MXFP tokens
        "kv": [str(t) for t in search["kv"]],                      # MXINT + MXFP tokens
        "act_w": [str(t) for t in search.get("act_w", ["MXINT8"])],
        "block": [str(b) for b in _as_list(search.get("block", 32))],
    }


def _pt_from_params(params: dict) -> dict:
    w_fmt, w = parse_prec_token(params["weight_w"])
    kv_fmt, kv_w = parse_prec_token(params["kv"])
    act_fmt, act_w = parse_prec_token(params["act_w"])
    return {"w_fmt": w_fmt, "attn_w": w, "ffn_w": w, "kv": kv_w, "kv_fmt": kv_fmt,
            "act_w": act_w, "act_fmt": act_fmt, "block": int(params["block"])}


def _front_fp_list(search: dict) -> list[Any]:
    """FP_SETTING values refined on the front (accuracy-only)"""
    return [tuple(v) if v else None for v in search.get("front_fp_setting", [None])]


def _spec(cfg: dict, pt: dict, *, method: str = "rtn", clip: bool = False,
          fp_setting: Any = None, full_decode: bool = False) -> DecodeQuantSpec:
    """Build a spec. ``method`` in {rtn, gptq, rotation}; ``clip`` enables Erry
    clipping. ``full_decode`` turns on the decode fidelity used on the front —
    qk/av matmul quantisation and softmax/rope FP_SETTING (both eager)"""
    s = cfg["search"]
    blk = pt["block"]
    return DecodeQuantSpec(
        attn_w=pt["attn_w"], ffn_w=pt["ffn_w"], kv=pt["kv"],
        w_fmt=pt.get("w_fmt", "mxint"), kv_fmt=pt.get("kv_fmt", "mxint"),
        weight_block=blk, kv_block=blk,
        act_w=pt.get("act_w"), act_fmt=pt.get("act_fmt", "mxint"), act_block=blk,
        use_gptq=(method == "gptq"), use_rotation=(method == "rotation"),
        fp_setting=fp_setting,
        fp_setting_attention=full_decode and bool(s.get("fp_setting_attention", True)),
        quant_attn_internals=full_decode and bool(s.get("quant_attn_matmuls", True)),
    )


def _cli_tok(fmt: str, width: Any) -> str:
    """Render a (format, width) as a CLI precision token the worker re-parses."""
    if width is None:
        return "16"
    if fmt == "mxint":
        return str(int(width))
    e, m = width
    return f"E{e}M{m}"


def _ref_metrics(sim: DecodeSimulator, row: dict, workload: dict,
                 base_hbm: dict, n_chips: int = 1) -> tuple[float, float, bool]:
    """One chip is pinned so the HBM capacity wall binds. TPS is evaluated at
    the capacity-limited batch: start from the workload's target batch and
    halve until weights+KV fit, so a lower-precision KV earns throughput by
    holding MORE streams (the capacity wall is part of the objective, not just
    a pass/fail gate). A precision that cannot fit even one stream scores
    zero. TPOT stays a single-stream latency."""
    act = row.get("act_bits")
    prec = sim.precision_from_eff_bits(
        float(row["attn_w_bits"]), float(row["ffn_w_bits"]), float(row["kv_bits"]),
        act_bits=float(act) if act not in (None, "") else None,
        block=int(float(row.get("block") or 32)),
    )
    over = sim.shipped_over(prec, base_hbm)
    gen = workload.get("hbm_gen", "HBM2")
    ch = int(workload.get("hbm_channels", 32))
    target = int(workload.get("batch", 64))
    # Probe once for the largest batch that fits, then evaluate at
    # min(target, capacity). Lower-bit KV frees capacity, so it earns TPS by
    # serving more streams even when a compute ceiling flattens per-batch TPS.
    probe = sim.evaluate(prec, batch=1, input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over,
                         n_chips=n_chips, hbm_gen=gen, hbm_channels=ch)
    if not probe.fits_in_hbm or probe.max_batch < 1:
        return 0.0, float("inf"), False
    batch = max(1, min(target, probe.max_batch))
    m_tps = sim.evaluate(prec, batch=batch, input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over,
                         n_chips=n_chips, hbm_gen=gen, hbm_channels=ch)
    if not m_tps.fits_in_hbm:
        return 0.0, float("inf"), False
    m_lat = sim.evaluate(prec, batch=1, input_seq=workload["input_seq"],
                         output_seq=workload["output_seq"], hw_over=over,
                         n_chips=n_chips, hbm_gen=gen, hbm_channels=ch)
    return round(m_tps.tps, 2), round(m_lat.tpot * 1e3, 3), True


def _build_task_calib(cfg: dict, task: str) -> str:
    """Ensure a task-aligned calibration file exists; return its ``file:`` spec."""
    g = cfg.get("gptq", {})
    n, s = int(g.get("nsamples", 64)), int(g.get("seqlen", 1024))
    out = calib_path(cfg["model_name"], task, n, s)
    if not out.exists():
        args = [
            sys.executable, "-m", "decode_dse.software.build_task_calib",
            "--model_name", cfg["model_name"], "--task", task,
            "--nsamples", str(n), "--seqlen", str(s),
            "--device", cfg.get("device", "cuda:0"),
        ]
        if cfg.get("local_files_only", True):
            args += ["--local_files_only", "true"]
        if cfg.get("trust_remote_code"):
            args += ["--trust_remote_code", "true"]
        if cfg.get("hf_token"):
            args += ["--hf_token", cfg["hf_token"]]
        subprocess.run(args, check=False)
    return f"file:{out}"


def _gpu_free_map() -> dict[str, int]:
    """Free MiB per physical GPU index ({} when nvidia-smi is unavailable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
        free = {}
        for line in out.strip().splitlines():
            idx, mb = line.split(",")
            free[idx.strip()] = int(mb)
        return free
    except Exception:
        return {}


def _pick_gpu(min_free_mb: int, poll_sec: int = 120, max_wait_sec: int = 21600) -> str | None:
    """Physical index of the freest GPU, waiting until one has ``min_free_mb``"""
    waited = 0
    while True:
        free = _gpu_free_map()
        if not free:
            return None
        gpu = max(free, key=lambda g: free[g])
        if free[gpu] >= min_free_mb or waited >= max_wait_sec:
            return gpu
        print(f"  [gpu-wait] freest GPU {gpu} has {free[gpu]} MiB < {min_free_mb}; "
              f"waiting {poll_sec}s ({waited + poll_sec}/{max_wait_sec}s)...", flush=True)
        time.sleep(poll_sec)
        waited += poll_sec


_RETRYABLE = ("OutOfMemoryError", "no output (worker died)")


def _list_free_gpus(min_free_mb: int, want: int) -> list[str]:
    """Up to `want` physical GPU indices with >= min_free_mb free, freest first.

    Used to spread the Stage-2 front points across only the GPUs that are
    actually free — this is a shared cluster, so we never assume all of them.
    If none currently clear the bar (fully contended cluster) we still return the
    freest one: the per-trial retry re-picks via ``_pick_gpu``, which blocks until
    capacity appears, so the run self-schedules instead of crashing."""
    free = _gpu_free_map()
    if not free:
        return []
    ok = sorted((g for g in free if free[g] >= min_free_mb), key=lambda g: -free[g])
    if not ok:
        ok = [max(free, key=lambda g: free[g])]
    return ok[:max(1, want)]


def _run_trial(cfg: dict, spec: DecodeQuantSpec, *, calib_dataset: str, tasks: str,
               eval_ppl: bool, clip: bool, gptq_ckpt: str | None, out_json: Path,
               attempts: int = 3, fp_only: bool = False, pin_gpu: str | None = None) -> dict[str, Any]:
    """Invoke the eval worker as a subprocess and return its JSON row.

    ``pin_gpu`` pins the worker to one physical GPU (parallel Stage-2 points);
    when None the worker picks the freest GPU itself (sequential Stage-1)."""
    g = cfg.get("gptq", {})
    args = [
        sys.executable, "-m", "decode_dse.software.eval_decode",
        "--model_name", cfg["model_name"], "--device", cfg.get("device", "cuda:0"),
        "--dtype", cfg.get("dtype", "bfloat16"),
        "--attn_w", _cli_tok(spec.w_fmt, spec.attn_w),
        "--ffn_w", _cli_tok(spec.w_fmt, spec.ffn_w),
        "--kv", _cli_tok(spec.kv_fmt, spec.kv),
        "--w_fmt", spec.w_fmt, "--kv_fmt", spec.kv_fmt,
        "--weight_block", str(spec.weight_block), "--kv_block", str(spec.kv_block),
        "--act_w", _cli_tok(spec.act_fmt, spec.act_w), "--act_fmt", spec.act_fmt,
        "--act_block", str(spec.act_block),
        "--use_gptq", "true" if spec.use_gptq else "false",
        "--use_rotation", "true" if spec.use_rotation else "false",
        "--clip_search_y", "true" if clip else "false",
        "--calib_dataset", calib_dataset,
        "--eval_ppl", "true" if eval_ppl else "false",
        "--eval_ppl_nsamples", str(cfg.get("eval_ppl_nsamples", 40)),
        "--eval_ppl_seqlen", str(cfg.get("eval_ppl_seqlen", 2048)),
        "--tasks", tasks, "--task_batch_size", str(cfg.get("task_batch_size", 8)),
        "--gptq_nsamples", str(g.get("nsamples", 128)),
        "--gptq_seqlen", str(g.get("seqlen", 2048)),
        "--gptq_cali_batch_size", str(g.get("cali_batch_size", 8)),
        "--out", str(out_json),
    ]
    if spec.fp_setting:
        args += ["--fp_setting", f"{spec.fp_setting[0]},{spec.fp_setting[1]}"]
        args += ["--fp_setting_attention", "true" if spec.fp_setting_attention else "false"]
    if spec.quant_attn_internals:
        args += ["--quant_attn_internals", "true"]
    if cfg.get("task_limit"):
        args += ["--task_limit", str(cfg["task_limit"])]
    if cfg.get("local_files_only", True):
        args += ["--local_files_only", "true"]
    if cfg.get("trust_remote_code"):
        args += ["--trust_remote_code", "true"]
    if cfg.get("hf_token"):
        args += ["--hf_token", cfg["hf_token"]]
    if g.get("max_layers"):
        args += ["--gptq_max_layers", str(g["max_layers"])]
    if spec.gptq_weights and gptq_ckpt:
        args += ["--gptq_checkpoint_dir", gptq_ckpt]
    if fp_only:
        args += ["--fp_only", "true"]

    heavy = (spec.gptq_weights or bool(tasks)) and not fp_only
    min_free = (int(cfg.get("gpu_min_free_mb_calibrated", 36000)) if heavy
                else int(cfg.get("gpu_min_free_mb", 21000)))
    label = "fp_reference" if fp_only else spec.tag

    row: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        env = None
        if str(cfg.get("device", "cuda:0")).startswith("cuda"):
            # First attempt uses the pinned GPU; a retry (usually OOM from a
            # co-tenant on this shared cluster) falls back to the freest GPU now.
            gpu = pin_gpu if (pin_gpu is not None and attempt == 1) else _pick_gpu(min_free)
            if gpu is not None:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        subprocess.run(args, check=False, env=env)
        row = (json.loads(out_json.read_text()) if out_json.exists()
               else {"tag": label, "cont_ppl": "", "error": "no output (worker died)"})
        err = row.get("error") or ""
        if not any(r in err for r in _RETRYABLE) or attempt == attempts:
            return row
        print(f"  [retry {attempt}/{attempts - 1}] {label}: {err.splitlines()[0][:100]}",
              flush=True)
        out_json.unlink(missing_ok=True)
    return row


def _fp_reference(cfg: dict, trial_dir: Path) -> dict[str, Any]:
    """The per-model FP perplexity reference (cached; computed once).

    Forced-prefill scoring bypasses every decode quantiser, so ``prefill_ppl``
    is identical across trials; computing it here once lets the trial workers
    skip it AND drop their FP weight bank, halving their VRAM"""
    oj = trial_dir / "fp_reference.json"
    row = _cached(oj)
    if row is None:
        row = _run_trial(cfg, DecodeQuantSpec(), calib_dataset="wikitext2", tasks="",
                         eval_ppl=True, clip=False, gptq_ckpt=None, out_json=oj,
                         fp_only=True)
    return row


def _cached(out_json: Path) -> dict[str, Any] | None:
    if not out_json.exists():
        return None
    row = json.loads(out_json.read_text())
    return None if row.get("error") else row


def _ckpt_dir(cfg: dict, root: Path, spec: DecodeQuantSpec, calib_dataset: str, clip: bool) -> str:
    """Cache dir for one calibrated bank (keyed by weight width x block x calib x clip)."""
    key = gptq_cache_key(cfg["model_name"], spec, {**cfg.get("gptq", {}),
                                                   "dataset": calib_dataset, "clip_search_y": clip})
    return str(root / key)


def _clear_banks(ckpt: str | None) -> None:
    """Delete the ~16 GB GPTQ weight banks in a checkpoint dir once no further trial needs them"""
    if not ckpt:
        return
    d = Path(ckpt)
    if d.exists():
        for pat in ("*.safetensors", "*.safetensors.tmp", "*.pt.tmp"):
            for f in d.glob(pat):
                f.unlink(missing_ok=True)


def _seed_rotation_cache(src_ckpt: str | None, dst_ckpt: str | None) -> None:
    """Reuse the wikitext2 rotation decisions for a task-aligned bank"""
    if not src_ckpt or not dst_ckpt:
        return
    src = Path(src_ckpt) / "rotation_decisions.json"
    dst = Path(dst_ckpt) / "rotation_decisions.json"
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _fp_bits(fp: Any) -> int:
    """Total vector-unit width of an FP_SETTING (bf16 when None): 1+E+M bits."""
    return 16 if fp is None else 1 + fp[0] + fp[1]


def _pick_fp(sweep: list[tuple[Any, dict]], tol: float) -> tuple[Any, dict] | None:
    """Cheapest FP_SETTING within ``tol`` relative PPL of the best sweep point.

    Minimum PPL would always pick the bf16 vector unit; the sweep exists to
    find how NARROW the vector unit can go without hurting accuracy."""
    ok = [(fp, r) for fp, r in sweep if not r.get("error") and r.get("cont_ppl") not in ("", None)]
    if not ok:
        return None
    best_ppl = min(float(r["cont_ppl"]) for _, r in ok)
    within = [(fp, r) for fp, r in ok if float(r["cont_ppl"]) <= best_ppl * (1.0 + tol)]
    return min(within, key=lambda x: (_fp_bits(x[0]), float(x[1]["cont_ppl"])))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(_fp_str(r))


def _fp_str(row: dict) -> dict:
    """Render an [E,M] fp_setting as a compact string for the CSV."""
    r = dict(row)
    fp = r.get("fp_setting")
    if isinstance(fp, (list, tuple)):
        r["fp_setting"] = f"E{fp[0]}M{fp[1]}"
    return r


def _is_doomed(ppl: Any, fp_ref: Any, cfg: dict) -> bool:
    """A front point is doomed if it is unrecoverable at the best-case nonlinear
    precision (bf16 FP setting): GPTQ recovers weight error and rotation recovers
    KV/act error, but neither rescues a point already catastrophic there — so the
    expensive FP sweep + task generation would only confirm garbage (e.g. the
    22 h wasted on a 2-bit-KV point that scored ppl 235). An empty/errored ppl is
    NOT doomed — that is a crashed eval (e.g. OOM), handled separately as a
    retryable error so a transient failure is never mistaken for a bad precision."""
    if ppl in ("", None):
        return False
    ppl = float(ppl)
    ratio = float(cfg.get("doomed_ppl_ratio", 3.0))
    absmax = float(cfg.get("doomed_ppl_abs", 100.0))
    gate = max(absmax, ratio * float(fp_ref)) if fp_ref and float(fp_ref) > 0 else absmax
    return ppl > gate


def _process_front_point(cfg, idx, n_front, fr, pt, *, front_fps, tasks, fp_tol, method,
                         cache_root, trial_dir, fp_ppl, fp_ref_ppl, pin_gpu) -> dict | None:
    """One Stage-2 front point on a pinned GPU: FP_SETTING PPL sweep -> pick the
    cheapest within tol -> per-task eval. Gated so a doomed point skips the sweep
    and tasks. Uses per-point (tag-suffixed) bank dirs so parallel points never
    race on a shared GPTQ bank; clears its own banks when done. Returns the merged
    row (or None if skipped); the caller adds reference-chip metrics under a lock."""
    if pt is None or pt["attn_w"] != pt["ffn_w"]:
        if pt:
            print(f"  skip calibration for mixed-weight {fr['tag']} (RTN row kept)", flush=True)
        return None
    t0 = time.time()
    tag = fr["tag"]
    wiki_spec = _spec(cfg, pt, method=method, clip=True, full_decode=True)
    wiki_ckpt = f"{_ckpt_dir(cfg, cache_root, wiki_spec, 'wikitext2', True)}__{tag}"
    own_banks = {wiki_ckpt}

    sweep: list[tuple[Any, dict]] = []
    for j, fp in enumerate(front_fps):
        spec = _spec(cfg, pt, method=method, clip=True, fp_setting=fp, full_decode=True)
        oj = trial_dir / f"{spec.tag}__ppl.json"
        r = _cached(oj) or _run_trial(cfg, spec, calib_dataset="wikitext2", tasks="",
                                      eval_ppl=True, clip=True, gptq_ckpt=wiki_ckpt,
                                      out_json=oj, pin_gpu=pin_gpu)
        r["fp_setting"] = fp
        sweep.append((fp, r))
        # First (reference / bf16-nonlinear) setting. A crashed eval (OOM/error)
        # keeps its `error` field, so it is dropped downstream and re-tried on a
        # rerun rather than mistaken for a bad precision.
        if j == 0 and (r.get("error") or r.get("cont_ppl") in ("", None)):
            _clear_banks(wiki_ckpt)
            print(f"  [{idx+1}/{n_front}] {tag}: ERROR at ref FP "
                  f"({str(r.get('error'))[:70]}) — not doomed; retries on rerun "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)
            return dict(r, fp_setting=fp)
        if j == 0 and _is_doomed(r.get("cont_ppl"), fp_ref_ppl, cfg):
            _clear_banks(wiki_ckpt)
            merged = dict(r)
            merged.update(fp_setting=fp, skipped="doomed_ppl")
            if fp_ppl and merged.get("prefill_ppl") in ("", None):
                merged["prefill_ppl"] = fp_ppl
            print(f"  [{idx+1}/{n_front}] {tag}: DOOMED ppl={r.get('cont_ppl')} at ref FP "
                  f"(FP ref={fp_ref_ppl}) — skipped sweep+tasks ({(time.time()-t0)/60:.0f} min)",
                  flush=True)
            return merged

    picked = _pick_fp(sweep, fp_tol)
    if picked is None:
        _clear_banks(wiki_ckpt)
        print(f"  [{idx+1}/{n_front}] {tag}: every FP_SETTING failed — skipping", flush=True)
        return None
    best_fp, merged = picked
    merged = dict(merged)
    spec = _spec(cfg, pt, method=method, clip=True, fp_setting=best_fp, full_decode=True)

    for task in tasks:
        calib = _build_task_calib(cfg, task)
        ckpt_t = f"{_ckpt_dir(cfg, cache_root, spec, calib, True)}__{tag}"
        _seed_rotation_cache(wiki_ckpt, ckpt_t)
        out_t = trial_dir / f"{spec.tag}__{task}.json"
        rt = _cached(out_t) or _run_trial(cfg, spec, calib_dataset=calib, tasks=task,
                                          eval_ppl=False, clip=True, gptq_ckpt=ckpt_t,
                                          out_json=out_t, pin_gpu=pin_gpu)
        own_banks.add(ckpt_t)
        if rt.get(task) not in ("", None):
            merged[task] = rt[task]
    merged["fp_setting"] = best_fp
    if fp_ppl and merged.get("prefill_ppl") in ("", None):
        merged["prefill_ppl"] = fp_ppl
    for ck in own_banks:
        _clear_banks(ck)
    print(f"  [{idx+1}/{n_front}] {merged.get('tag')}: ppl={merged.get('cont_ppl')} fp={best_fp} "
          + " ".join(f"{t}={merged.get(t)}" for t in tasks)
          + f"  ({(time.time()-t0)/60:.0f} min)", flush=True)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode-accuracy DSE")
    ap.add_argument("config", help="path to a decode DSE config JSON")
    ap.add_argument("--out", default=None, help="CSV path (default: <output_dir>/software_disagg_decode.csv)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    out_dir = Path(cfg.get("output_dir", "results/decode_dse")) / cfg["model_name"].split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) if args.out else out_dir / "software_disagg_decode.csv"
    trial_dir = out_dir / "trials"
    trial_dir.mkdir(exist_ok=True)
    cache_root = Path(cfg.get("scratch_dir", "/tmp/decode_dse")) / cfg["model_name"].split("/")[-1] / "gptq_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    sim = DecodeSimulator(cfg["sim_model"], model_lib=cfg.get("model_lib"))
    # Memory-bound reference operating point: a large array evaluated at
    # capacity-limited batch, with memory priced at the emulator-calibrated
    # effective bandwidth. The old MLEN*BLEN=4096 default saturated compute at
    # any batch, collapsing TPS to a couple of discrete values.
    if cfg.get("bw_model", "calibrated") == "calibrated":
        sim.use_calibrated_bandwidth()
    ref_chip = cfg.get("reference_chip", {"MLEN": 2048, "BLEN": 32, "VLEN": 2048, "HLEN": 128})
    workload = cfg["reference_workload"]
    base_hbm = sim.hbm_overrides(cfg.get("hardware_space", {}).get("hbm_gen", "HBM2"),
                                 int(cfg.get("baseline_hbm_channels", 32)))
    base_hbm = {**base_hbm, **ref_chip}
    seed = int(cfg.get("seed", 0))
    n_chips = int(cfg.get("n_chips", 1))  # pinned so the HBM-capacity wall binds
    sampler = cfg.get("sampler", "nsga2")  # 'nsga2' (evolutionary) or 'tpe' (Bayesian MOTPE)
    method = "rotation" if cfg.get("use_rotation") else "gptq"
    tasks = cfg.get("tasks", ["gsm8k", "ifeval"])
    fp_tol = float(cfg.get("fp_ppl_tol", 0.01))

    # Optuna precision search (NSGA-II) over the mixed-precision space
    # (weight/KV/act/block, MXINT+MXFP on every axis). Objectives: minimise PPL,
    # maximise reference-chip TPS, minimise reference-chip latency. RTN + SDPA
    # here; qk/av/softmax don't change throughput and wait for the front.
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # FP prefill reference: one worker run, shared by every trial row 
    fp_ref = _fp_reference(cfg, trial_dir)
    fp_ppl = fp_ref.get("prefill_ppl") if not fp_ref.get("error") else ""
    print(f"[stage 1] FP prefill reference: ppl={fp_ppl or 'unavailable'}", flush=True)

    rows: list[dict] = []
    tag_to_pt: dict[str, dict] = {}
    tag_to_row: dict[str, dict] = {}

    def _record(pt: dict) -> dict:
        """Evaluate one precision point (cached by tag); append + checkpoint.
        Error rows are cached too — resampling a failed point must return its
        error row, not crash the study (StopIteration aborts study.optimize)."""
        spec = _spec(cfg, pt)  # full_decode=False -> SDPA, weight+KV+act only
        if spec.tag in tag_to_row:
            return tag_to_row[spec.tag]
        tag_to_pt[spec.tag] = pt
        oj = trial_dir / f"{spec.tag}.json"
        legacy = trial_dir / f"rtn__{spec.tag}.json"   # pre-fix cache name (double prefix)
        row = _cached(oj) or _cached(legacy)
        if row is None:
            row = _run_trial(cfg, spec, calib_dataset="wikitext2", tasks="", eval_ppl=True,
                             clip=False, gptq_ckpt=None, out_json=oj)
        row["tag"] = row.get("tag") or spec.tag
        row.setdefault("fp_setting", None)
        if not row.get("error") and row.get("cont_ppl") not in ("", None):
            if fp_ppl and row.get("prefill_ppl") in ("", None):
                row["prefill_ppl"] = fp_ppl
            row["ref_tps"], row["ref_tpot_ms"], row["ref_fits"] = _ref_metrics(
                sim, row, workload, base_hbm, n_chips)
        tag_to_row[spec.tag] = row
        rows.append(row)
        _write_csv(csv_path, rows)
        return row

    choices = _precision_choices(cfg["search"])
    budget = int(cfg.get("search_budget", 60))
    print(f"[stage 1] precision search: {budget} trials over "
          f"{len(_grid(cfg['search']))} configs "
          f"(min PPL / max ref TPS / min ref TPOT@b1)", flush=True)

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in choices.items()}
        r = _record(_pt_from_params(params))
        if r.get("error") or r.get("cont_ppl") in ("", None):
            return 1e6, 0.0, 1e9
        print(f"  [{len(rows)}] {r.get('tag')}: ppl={r.get('cont_ppl')} "
              f"TPS={r.get('ref_tps')} TPOT@b1={r.get('ref_tpot_ms')}ms", flush=True)
        return float(r["cont_ppl"]), float(r.get("ref_tps") or 0.0), float(r.get("ref_tpot_ms") or 1e9)

    study = optuna.create_study(
        directions=["minimize", "maximize", "minimize"],  # PPL, TPS, TPOT@b1
        sampler=make_sampler(sampler, seed),
    )
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    # Full-decode fidelity on the accuracy/throughput front ------
    # Eager: qk/av quantised + softmax/rope FP_SETTING (swept), weights = GPTQ +
    # Erry clip, activations/KV = selective rotation, task-aligned calib.
    front = select_front([r for r in rows if r.get("ref_tps") is not None],
                         int(cfg.get("front_size", 4)))
    front_fps = _front_fp_list(cfg["search"])
    print(f"[stage 2] {method} + Erry-clip on {len(front)} front points; "
          f"FP_SETTING sweep={front_fps} (cheapest within {fp_tol:.1%} PPL); "
          f"tasks={tasks} (task-aligned calib, eager)", flush=True)

    # Stage 2 runs the front points in parallel across the free GPUs (freest
    # first). Each point is pinned to one GPU and builds its own tag-suffixed
    # banks, so parallel points never race on a shared bank; a doomed precision
    # skips the sweep + tasks. Task calibrations are built once up front.
    import queue as _queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    front_pts = [(fr, tag_to_pt.get(fr["tag"])) for fr in front]
    for task in tasks:
        _build_task_calib(cfg, task)
    fp_ref_ppl = fp_ref.get("cont_ppl") or fp_ppl
    on_cuda = str(cfg.get("device", "cuda:0")).startswith("cuda")
    max_par = int(cfg.get("max_parallel_points", 4))
    gpus = _list_free_gpus(int(cfg.get("gpu_min_free_mb_calibrated", 36000)), max_par) if on_cuda else [None]
    gate = max(float(cfg.get("doomed_ppl_abs", 100.0)), 3.0 * (float(fp_ref_ppl) if fp_ref_ppl else 0.0))
    print(f"[stage 2] {len(front)} front points across {len(gpus)} free GPU(s) {gpus} "
          f"(parallel; doomed-gate at ppl>{gate:.0f})", flush=True)

    lock = threading.Lock()
    gpu_q: "_queue.Queue" = _queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    def _worker(item):
        idx, (fr, pt) = item
        g = gpu_q.get()
        try:
            merged = _process_front_point(cfg, idx, len(front), fr, pt, front_fps=front_fps,
                                          tasks=tasks, fp_tol=fp_tol, method=method,
                                          cache_root=cache_root, trial_dir=trial_dir,
                                          fp_ppl=fp_ppl, fp_ref_ppl=fp_ref_ppl, pin_gpu=g)
        finally:
            gpu_q.put(g)
        if merged is not None:
            with lock:  # DecodeSimulator + CSV are shared state: serialize
                if not merged.get("error") and merged.get("cont_ppl") not in ("", None):
                    merged["ref_tps"], merged["ref_tpot_ms"], merged["ref_fits"] = _ref_metrics(
                        sim, merged, workload, base_hbm, n_chips)
                rows.append(merged)
                _write_csv(csv_path, rows)
        return merged

    with ThreadPoolExecutor(max_workers=max(1, len(gpus))) as ex:
        list(ex.map(_worker, list(enumerate(front_pts))))

    print(f"\n[done] wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
