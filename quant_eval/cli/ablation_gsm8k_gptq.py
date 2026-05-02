"""
GSM8K-aligned GPTQ ablation using TokenCollector.

Pipeline (per-run, executed once per (width, condition) combo):
    1. (one-shot) Collect GSM8K calibration tokens via ``TokenCollector``:
       a fresh model is loaded, GSM8K train rows are formatted as
       ``Question: ... Answer: ...`` and forwarded; the hook saves
       ``calib/<model>_gsm8k_<...>.pt`` in ``get_loaders`` format.
    2. For each (width, condition) combo: load a fresh model, build
       ``pass_args`` with optional ``gptq`` block (pointing at the
       calibration file or ``"wikitext2"``), run
       ``quantize_module_transform_pass`` to do GPTQ + MXInt module
       replacement, then run ``evaluate_with_lm_eval`` on GSM8K.
    3. Dump per-combo timing / metrics / module-class histogram to JSON.

Conditions (cartesian-producted with widths):
    * ``no_gptq``           — only MXInt module replacement (baseline)
    * ``gptq_wikitext2``    — GPTQ on wikitext2 + MXInt replacement
    * ``gptq_gsm8k``        — GPTQ on GSM8K-aligned tokens + MXInt

Run::

    python -m quant_eval.cli.ablation_gsm8k_gptq \\
        --model_name Qwen/Qwen3-0.6B \\
        --widths "[4,8]" \\
        --conditions "[no_gptq,gptq_gsm8k]" \\
        --limit 20
"""

from __future__ import annotations

import gc
import json
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Union

import torch
import transformers

from quant_eval.utils import get_logger, set_logging_verbosity, setup_model
from quant_eval.eval.lm_eval import evaluate_with_lm_eval
from chop.passes.module.transforms.gptq import TokenCollector

logger = get_logger(__name__)
set_logging_verbosity("info")


LINEAR_REGEXES = (
    r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
    r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj",
)


# ---------------------------------------------------------------------------
# Stage 1 — Collect GSM8K-aligned calibration tokens with TokenCollector.
# ---------------------------------------------------------------------------

def collect_gsm8k_calibration(
    model_name: str,
    dtype: torch.dtype,
    device: str,
    save_path: Path,
    nsamples: int,
    seqlen: int,
    overwrite: bool,
    max_prompts: int = 512,
) -> None:
    """Forward GSM8K-formatted prompts through a fresh model so TokenCollector
    captures the token stream and saves a get_loaders-compatible file."""
    if save_path.exists() and not overwrite:
        logger.info("calibration file %s exists; reuse (overwrite=False).", save_path)
        return
    if save_path.exists():
        save_path.unlink()

    import datasets

    tokenizer, model = setup_model(
        model_name=model_name,
        model_parallel=False,
        dtype=dtype,
        device=device,
        attn_implementation="eager",
    )
    model.eval().to(device)

    collector = TokenCollector(
        model=model,
        target_nsamples=nsamples,
        seqlen=seqlen,
        save_path=save_path,
        overwrite=False,
        min_prefill_tokens=8,
    ).attach()

    ds = datasets.load_dataset("gsm8k", "main", split="train").shuffle(seed=0)

    used = 0
    for row in ds:
        if collector.complete or used >= max_prompts:
            break
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seqlen,
        )
        ids = enc.input_ids.to(device)
        if ids.shape[-1] < 8:
            continue
        with torch.no_grad():
            try:
                model(ids)
            except Exception as e:
                logger.warning("forward failed on prompt %d: %s", used, e)
                break
        used += 1

    if not collector.complete:
        collector.finalize()

    logger.info(
        "calibration: forwarded=%d prompts, saved=%s, complete=%s",
        used, save_path, collector.complete,
    )

    del model, tokenizer, collector
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2 — Build pass_args + run one (width, condition) combo.
# ---------------------------------------------------------------------------

def build_pass_args(
    width: int,
    block_size: int,
    gptq_config: dict | None,
) -> dict:
    proj_cfg = {
        "name": "mxint",
        "weight_block_size": block_size,
        "weight_width": width,
        "data_in_block_size": block_size,
        "data_in_width": width,
    }
    pass_args: dict = {"by": "regex_name"}
    for rgx in LINEAR_REGEXES:
        pass_args[rgx] = {"config": dict(proj_cfg)}
    if gptq_config is not None:
        pass_args["gptq"] = gptq_config
    return pass_args


def _flatten_metrics(results: dict) -> dict:
    flat = {}
    for task, mvals in results.get("results", {}).items():
        for k, v in mvals.items():
            if isinstance(v, (int, float)):
                flat[f"{task}/{k}"] = float(v)
    return flat


def run_one(
    model_name: str,
    dtype: torch.dtype,
    device: str,
    pass_args: dict,
    tasks: Union[str, list[str]],
    seqlen: int,
    batch_size: Union[int, str],
    limit,
) -> dict:
    from chop.passes.module.transforms import quantize_module_transform_pass

    tokenizer, model = setup_model(
        model_name=model_name,
        model_parallel=False,
        dtype=dtype,
        device=device,
        attn_implementation="eager",
    )
    model.eval()

    t0 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    t_quant = time.time() - t0

    cls_count = dict(Counter(
        type(m).__name__
        for _, m in model.named_modules()
        if "MX" in type(m).__name__
    ))

    model.to(device)

    t0 = time.time()
    results = evaluate_with_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        max_length=seqlen,
        batch_size=batch_size,
        log_samples=False,
        limit=limit,
    )
    t_eval = time.time() - t0

    metrics = _flatten_metrics(results)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "t_quant_s": t_quant,
        "t_eval_s": t_eval,
        "module_classes": cls_count,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------

def _gptq_config_for(
    cond: str,
    *,
    model_name: str,
    device: str,
    width: int,
    block_size: int,
    calib_path: Path,
    calib_nsamples: int,
    calib_seqlen: int,
    cali_batch_size: int,
) -> dict | None:
    if cond == "no_gptq":
        return None
    base = {
        "model_name": model_name,
        "device": device,
        "nsamples": calib_nsamples,
        "seqlen": calib_seqlen,
        "format": "mxint",
        "weight_config": {
            "weight_block_size": block_size,
            "weight_width": width,
        },
        "quantile_search": True,
        "clip_search_y": False,
        "cali_batch_size": cali_batch_size,
        "checkpoint_dir": None,
    }
    if cond == "gptq_gsm8k":
        base["dataset"] = f"file:{calib_path}"
    elif cond == "gptq_wikitext2":
        base["dataset"] = "wikitext2"
    else:
        raise ValueError(f"unknown condition: {cond}")
    return base


def main(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    widths: list[int] = [4],
    block_size: int = 32,
    seqlen: int = 2048,
    batch_size: Union[int, str] = 4,
    limit: Union[int, float, None] = 20,
    conditions: list[str] = ["no_gptq", "gptq_gsm8k"],
    calib_nsamples: int = 32,
    calib_seqlen: int = 1024,
    cali_batch_size: int = 32,
    calib_overwrite: bool = False,
    calib_dir: str = "calib",
    log_dir: str = "logs/ablation_gsm8k_gptq",
    max_calib_prompts: int = 512,
):
    """GSM8K-aligned GPTQ ablation. See module docstring."""
    transformers.set_seed(0)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype_t = dtype_map[dtype]

    log_p = Path(log_dir)
    log_p.mkdir(parents=True, exist_ok=True)
    calib_p = Path(calib_dir)
    calib_p.mkdir(parents=True, exist_ok=True)
    calib_path = calib_p / (
        f"{model_name.replace('/', '_')}_gsm8k_n{calib_nsamples}_s{calib_seqlen}.pt"
    )

    # Stage 1 — collect once if any combo needs GSM8K calibration.
    need_gsm8k = any(c == "gptq_gsm8k" for c in conditions)
    if need_gsm8k:
        logger.info("==> Stage 1: collecting GSM8K calibration tokens")
        collect_gsm8k_calibration(
            model_name=model_name,
            dtype=dtype_t,
            device=device,
            save_path=calib_path,
            nsamples=calib_nsamples,
            seqlen=calib_seqlen,
            overwrite=calib_overwrite,
            max_prompts=max_calib_prompts,
        )

    combos = list(product(widths, conditions))
    rows: list[dict] = []

    print("=" * 72)
    print(f"GSM8K GPTQ ablation: {len(combos)} combos on {model_name}")
    print(f"  widths     = {widths}")
    print(f"  conditions = {conditions}")
    print(f"  block_size = {block_size}")
    print(f"  calib_path = {calib_path} (n={calib_nsamples}, seqlen={calib_seqlen})")
    print(f"  eval limit = {limit}")
    print("=" * 72)

    t_all = time.time()
    for i, (w, cond) in enumerate(combos, 1):
        tag = f"w{w}_{cond}"
        print(f"\n[{i}/{len(combos)}] >>> {tag}")

        gptq_config = _gptq_config_for(
            cond,
            model_name=model_name, device=device, width=w,
            block_size=block_size, calib_path=calib_path,
            calib_nsamples=calib_nsamples, calib_seqlen=calib_seqlen,
            cali_batch_size=cali_batch_size,
        )
        pass_args = build_pass_args(w, block_size, gptq_config)

        try:
            out = run_one(
                model_name=model_name, dtype=dtype_t, device=device,
                pass_args=pass_args, tasks="gsm8k",
                seqlen=seqlen, batch_size=batch_size, limit=limit,
            )
            out["error"] = None
        except Exception as e:
            logger.exception("combo %s failed", tag)
            out = {
                "t_quant_s": 0.0, "t_eval_s": 0.0,
                "module_classes": {}, "metrics": {},
                "error": f"{type(e).__name__}: {e}",
            }

        out.update({
            "tag": tag,
            "config": {
                "width": w, "block_size": block_size, "condition": cond,
                "pass_args": pass_args,
            },
        })
        rows.append(out)

        if out["error"]:
            print(f"  ERROR: {out['error']}")
        else:
            print(
                f"  t_quant={out['t_quant_s']:.1f}s "
                f"t_eval={out['t_eval_s']:.1f}s "
                f"metrics={out['metrics']}"
            )
            print(f"  classes={out['module_classes']}")

    t_total = time.time() - t_all

    out_path = log_p / f"ablation_gsm8k_gptq_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model_name": model_name, "device": device, "dtype": dtype,
            "widths": widths, "block_size": block_size,
            "conditions": conditions,
            "calib_path": str(calib_path),
            "calib_nsamples": calib_nsamples,
            "calib_seqlen": calib_seqlen,
            "cali_batch_size": cali_batch_size,
            "seqlen": seqlen, "batch_size": batch_size, "limit": limit,
            "total_runtime_s": t_total, "rows": rows,
        }, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"Total runtime: {t_total:.1f}s")
    print(f"Saved        : {out_path}")
    print("=" * 72)

    metric_keys = sorted({k for r in rows for k in r["metrics"].keys()})
    header = f"{'tag':<28} {'t_quant':>8} {'t_eval':>8}"
    for mk in metric_keys:
        header += f" {mk:>22}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['tag']:<28} {r['t_quant_s']:>8.1f} {r['t_eval_s']:>8.1f}"
        for mk in metric_keys:
            v = r['metrics'].get(mk)
            line += f" {v:>22.4f}" if v is not None else f" {'-':>22}"
        if r["error"]:
            line += f"   ERR: {r['error']}"
        print(line)

    return rows


if __name__ == "__main__":
    from jsonargparse import CLI

    start = time.time()
    CLI(main)
    print(f"\n[INFO] Wall time: {time.time() - start:.2f} s")
