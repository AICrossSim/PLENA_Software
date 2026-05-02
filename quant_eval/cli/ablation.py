"""
Ablation across three axes for MXInt quantization on a HF causal LM:

    * Activation/weight precision  (e.g. 4, 8, 16 bits)
    * Online Hadamard rotation     (on / off)
    * Attention-module quantization (on / off, i.e. whether we quantize
      Q@K^T, attn@V and the KV cache; the q/k/v/o_proj linears are
      always quantized so the "off" case still differs from FP)

For every combination the script:
    1. Builds the quantization ``pass_args`` dict in memory (no TOML
       round-trip), so the resolved config can be dumped back out.
    2. Loads a fresh model copy, runs ``quantize_module_transform_pass``,
       times the operation, then runs ``evaluate_with_lm_eval`` and times
       that as well.
    3. Records the ``pass_args`` dict, the post-quant module-class
       histogram (so you can verify ``RotateMXIntLinear`` etc. landed),
       and the per-task metrics from lm-eval.
    4. Writes one JSON file with all rows + prints a summary table.

Run:
    python -m quant_eval.cli.ablation \\
        --model_name Qwen/Qwen3-0.6B \\
        --device_id cuda:0 \\
        --tasks wikitext \\
        --limit 0.05
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

from quant_eval.utils import (
    get_logger,
    set_logging_verbosity,
    setup_model,
)
from quant_eval.eval.lm_eval import evaluate_with_lm_eval

logger = get_logger(__name__)
set_logging_verbosity("info")




# ---------------------------------------------------------------------------
# Build a regex_name pass_args dict equivalent to the qwen3_mxint*.toml files,
# parameterized by (width, block_size, rotate, quant_attn).
# ---------------------------------------------------------------------------

LINEAR_REGEXES = (
    r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
    r"model\.layers\.\d+\.mlp\.(gate|up|down)_proj",
)
ATTN_REGEX = r"model\.layers\.\d+\.self_attn$"


def build_quant_config(
    width: int,
    block_size: int,
    rotate: bool,
    quant_attn: bool,
) -> dict:
    name = "mxint_rotate" if rotate else "mxint"

    proj_cfg = {
        "name": name,
        "weight_block_size": block_size,
        "weight_width": width,
        "data_in_block_size": block_size,
        "data_in_width": width,
    }

    pass_args: dict = {"by": "regex_name"}
    for rgx in LINEAR_REGEXES:
        pass_args[rgx] = {"config": dict(proj_cfg)}

    if quant_attn:
        pass_args[ATTN_REGEX] = {
            "config": {
                "name": name,
                "qk_matmul": {
                    "data_in_block_size": block_size,
                    "data_in_width": width,
                },
                "av_matmul": {
                    "data_in_block_size": block_size,
                    "data_in_width": width,
                },
                "kv_cache": {
                    "data_in_block_size": block_size,
                    "data_in_width": width,
                },
                "softmax": {"bypass": True},
                "rope": {"bypass": True},
            }
        }

    return pass_args


# ---------------------------------------------------------------------------
# Run a single (width, rotate, quant_attn) combo end-to-end and collect
# timing + metrics + observed module classes.
# ---------------------------------------------------------------------------

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
        model_name,
        model_parallel=False,
        dtype=dtype,
        device=device,
        attn_implementation="eager",  # required by all MXInt/Rotate attns
    )
    model.eval()

    t0 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    t_quant = time.time() - t0

    cls_count = dict(
        Counter(
            type(m).__name__
            for _, m in model.named_modules()
            if "MX" in type(m).__name__
        )
    )

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

    # Free GPU memory before the next combo.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "t_quant_s": t_quant,
        "t_eval_s": t_eval,
        "module_classes": cls_count,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Main loop — sweep the cartesian product, print a summary, write JSON.
# ---------------------------------------------------------------------------

def main(
    model_name: str = "Qwen/Qwen3-0.6B",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    tasks: Union[str, list[str]] = "wikitext",
    seqlen: int = 2048,
    batch_size: Union[int, str] = 4,
    limit: Union[int, float, None] = 0.05,
    widths: list[int] = [4, 8, 16],
    block_size: int = 32,
    rotate_options: list[bool] = [False, True],
    quant_attn_options: list[bool] = [False, True],
    log_dir: str = "logs/ablation",
):
    """
    Sweep precision × hadamard × quant_attn for an MXInt-quantized HF LM.

    Args:
        model_name: HuggingFace model id.
        device_id:  CUDA device.
        dtype:      Model dtype (bfloat16 recommended).
        tasks:      lm-eval task(s).
        seqlen:     Context window for lm-eval.
        batch_size: Eval batch size; pass "auto" for auto-batching.
        limit:      Cap samples (int absolute, float in (0,1) fraction,
                    None = full eval).
        widths:     List of bit-widths to sweep (applied to both weight
                    and activation widths).
        block_size: MX block size (32 matches all qwen3_mxint*.toml configs).
        rotate_options:     Booleans to sweep for online Hadamard rotation.
        quant_attn_options: Booleans to sweep for attention-module
                            quantization (qk@av matmul + KV cache).
        log_dir:    Output directory; one ``ablation_<timestamp>.json``
                    is written here.
    """
    transformers.set_seed(0)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    log_dir_p = Path(log_dir)
    log_dir_p.mkdir(parents=True, exist_ok=True)

    combos = list(product(widths, rotate_options, quant_attn_options))
    rows = []

    print("=" * 72)
    print(f"Ablation: {len(combos)} combos on {model_name} ({device_id}, {dtype})")
    print(f"  widths             = {widths}")
    print(f"  rotate_options     = {rotate_options}")
    print(f"  quant_attn_options = {quant_attn_options}")
    print(f"  tasks              = {tasks}")
    print(f"  limit              = {limit}")
    print("=" * 72)

    t_all = time.time()
    for i, (w, rotate, quant_attn) in enumerate(combos, 1):
        tag = f"w{w}_rot{int(rotate)}_attn{int(quant_attn)}"
        print(f"\n[{i}/{len(combos)}] >>> {tag}")

        pass_args = build_quant_config(w, block_size, rotate, quant_attn)
        try:
            out = run_one(
                model_name=model_name,
                dtype=torch_dtype,
                device=device_id,
                pass_args=pass_args,
                tasks=tasks,
                seqlen=seqlen,
                batch_size=batch_size,
                limit=limit,
            )
            out["error"] = None
        except Exception as e:
            logger.exception("Combo %s failed", tag)
            out = {
                "t_quant_s": 0.0,
                "t_eval_s": 0.0,
                "module_classes": {},
                "metrics": {},
                "error": f"{type(e).__name__}: {e}",
            }

        out.update(
            {
                "tag": tag,
                "config": {
                    "width": w,
                    "block_size": block_size,
                    "rotate": rotate,
                    "quant_attn": quant_attn,
                    "pass_args": pass_args,
                },
            }
        )
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

    # -----------------------------------------------------------------------
    # Persist + summary table
    # -----------------------------------------------------------------------
    out_path = log_dir_p / f"ablation_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "device_id": device_id,
                "dtype": dtype,
                "tasks": tasks,
                "seqlen": seqlen,
                "batch_size": batch_size,
                "limit": limit,
                "block_size": block_size,
                "total_runtime_s": t_total,
                "rows": rows,
            },
            f,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 72)
    print(f"Total runtime: {t_total:.1f}s")
    print(f"Saved        : {out_path}")
    print("=" * 72)

    # Compact summary
    metric_keys = sorted({k for r in rows for k in r["metrics"].keys()})
    header = f"{'tag':<28} {'t_quant':>8} {'t_eval':>8}"
    for mk in metric_keys:
        header += f" {mk:>22}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['tag']:<28} {r['t_quant_s']:>8.1f} {r['t_eval_s']:>8.1f}"
        )
        for mk in metric_keys:
            v = r["metrics"].get(mk)
            line += f" {v:>22.4f}" if v is not None else f" {'-':>22}"
        if r["error"]:
            line += f"   ERR: {r['error']}"
        print(line)

    return rows


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    print(f"\n[INFO] Wall time: {time.time() - start_time:.2f} s")
