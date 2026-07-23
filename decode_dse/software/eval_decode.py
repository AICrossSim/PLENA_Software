"""Per-trial decode-accuracy worker.

Loads a model, installs decode-only phase-split quantisation (prefill unquantised,
decode MXINT/MXFP), and scores the decode chip:

* Continuation perplexity (fast; drives the precision front) — cache-free
  forwards forced through the decode numerics (a conservative proxy);
* GSM8K and IFEval (generative; run only on the front) — the prompt pass runs
  prefill-FP with KV quantise-on-write handoff, generated steps run quantised.

Example:
    python -m decode_dse.software.eval_decode \
        --model_name meta-llama/Llama-3.1-8B-Instruct --device cuda:0 \
        --attn_w 4 --ffn_w 4 --kv 4 --act_w 8 --use_gptq true \
        --gptq_checkpoint_dir results/gptq_cache/<key> \
        --tasks gsm8k,ifeval --out results/trial.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from decode_dse.software.decode_quant import (
    DecodeQuantSpec,
    build_decode_pass_args,
    eff_bits,
)


def _make_phase_decode_hflm(model, tokenizer, batch_size: int):
    """Wrap the quantised model in an lm-eval HFLM that scores the DECODE chip.

    Generation (`_model_generate`) runs with NO phase override: the decoder-layer
    pre-hooks infer the phase from cache semantics, so the prompt pass runs as
    prefill and every generated-token step runs as decode (quantised)
    """
    from lm_eval.models.huggingface import HFLM
    from chop.nn.quantized.modules.phase_context import force_runtime_phase
    from chop.passes.module.transforms.quantize.quantize import (
        install_phase_context_pre_hooks,
    )

    class _PhaseDecodeHFLM(HFLM):
        def __init__(self):
            super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
            install_phase_context_pre_hooks(self._model)

        def _model_generate(self, *args, **kwargs):
            # No override: hooks give prefill-FP (+ KV handoff) then quantised decode.
            return super()._model_generate(*args, **kwargs)

        def _model_call(self, *args, **kwargs):
            with force_runtime_phase("decode"):
                return super()._model_call(*args, **kwargs)

    return _PhaseDecodeHFLM()


def _run_tasks(model, tokenizer, tasks: list[str], limit: int | None, batch_size: int) -> dict[str, float]:
    """Score generative tasks (GSM8K / IFEval) through the decode chip."""
    from lm_eval import simple_evaluate

    lm = _make_phase_decode_hflm(model, tokenizer, batch_size)
    results = simple_evaluate(model=lm, tasks=tasks, limit=limit, bootstrap_iters=0)
    scores: dict[str, float] = {}
    for task, metrics in results["results"].items():
        # Pick the primary accuracy metric each task reports.
        for key in (
            "exact_match,strict-match", "exact_match,flexible-extract",
            "prompt_level_strict_acc,none", "inst_level_strict_acc,none",
            "acc,none", "exact_match", "acc",
        ):
            if key in metrics:
                scores[task] = float(metrics[key])
                break
    return scores


def build_spec(cfg: argparse.Namespace) -> DecodeQuantSpec:
    """The precision spec for this worker invocation (also used to tag error rows)."""
    return DecodeQuantSpec(
        attn_w=cfg.attn_w, ffn_w=cfg.ffn_w, kv=cfg.kv,
        w_fmt=cfg.w_fmt, kv_fmt=cfg.kv_fmt,
        weight_block=cfg.weight_block, kv_block=cfg.kv_block,
        act_w=cfg.act_w, act_block=cfg.act_block,
        act_fmt=cfg.act_fmt,
        use_gptq=cfg.use_gptq, use_rotation=cfg.use_rotation,
        fp_setting=tuple(cfg.fp_setting) if cfg.fp_setting else None,
        fp_setting_attention=cfg.fp_setting_attention,
        quant_attn_internals=cfg.quant_attn_internals,
    )


def evaluate_fp_reference(cfg: argparse.Namespace) -> dict[str, Any]:
    """Perplexity of the unquantised model — the FP reference every trial shares.

    Forced-prefill scoring of a quantised model bypasses all decode numerics, so
    ``prefill_ppl`` is a per-model constant; computing it once here (instead of
    inside every trial) lets trial workers drop their FP weight bank entirely.
    """
    from transformers import AutoModelForCausalLM
    from chop.passes.module.transforms.gptq.data_utils import get_loaders
    from chop.passes.module.transforms.quantize.rotation_search import (
        _compute_calibration_perplexity,
    )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    load_kwargs = {"local_files_only": cfg.local_files_only, "trust_remote_code": cfg.trust_remote_code}
    if cfg.hf_token:
        load_kwargs["token"] = cfg.hf_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, **load_kwargs)
    model = model.to(cfg.device).eval()
    loader = get_loaders(
        "wikitext2", nsamples=cfg.eval_ppl_nsamples, seed=0,
        seqlen=cfg.eval_ppl_seqlen, model=cfg.model_name, hf_token=cfg.hf_token,
    )
    ppl = round(_compute_calibration_perplexity(
        model, loader, cfg.device, label="fp_reference", score_phase="prefill"), 4)
    return {
        "tag": "fp_reference", "cont_ppl": ppl, "prefill_ppl": ppl,
        "attn_w_bits": 16.0, "ffn_w_bits": 16.0, "kv_bits": 16.0, "act_bits": 16.0,
        "runtime_sec": round(time.time() - t0, 1), "error": "",
    }


def evaluate_decode_point(cfg: argparse.Namespace) -> dict[str, Any]:
    """Quantise for one decode precision and return an accuracy result row."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chop.passes.module.transforms.gptq.data_utils import get_loaders
    from chop.passes.module.transforms.quantize.quantize import (
        install_phase_context_pre_hooks,
        quantize_module_transform_pass,
    )
    from chop.passes.module.transforms.quantize.rotation_search import (
        _compute_calibration_perplexity,
    )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    spec = build_spec(cfg)

    row: dict[str, Any] = {
        "tag": spec.tag,
        "attn_w_bits": round(eff_bits(spec.w_fmt, spec.attn_w, spec.weight_block), 4),
        "ffn_w_bits": round(eff_bits(spec.w_fmt, spec.ffn_w, spec.weight_block), 4),
        "kv_bits": round(eff_bits(spec.kv_fmt, spec.kv, spec.kv_block), 4),
        "act_bits": round(eff_bits(spec.act_fmt, spec.act_w, spec.act_block), 4) if spec.act_w is not None else 16.0,
        "w_fmt": spec.w_fmt, "kv_fmt": spec.kv_fmt, "act_fmt": spec.act_fmt,
        "block": spec.weight_block,
        "use_gptq": spec.use_gptq, "use_rotation": spec.use_rotation,
        "cont_ppl": "", "prefill_ppl": "", "gsm8k": "", "ifeval": "",
        "runtime_sec": "", "error": "",
    }

    t0 = time.time()
    load_kwargs = {"local_files_only": cfg.local_files_only, "trust_remote_code": cfg.trust_remote_code}
    if cfg.hf_token:
        load_kwargs["token"] = cfg.hf_token
    # Softmax/rope FP_SETTING and qk/av quantisation live inside eager attention.
    if spec.needs_eager:
        load_kwargs["attn_implementation"] = "eager"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, **load_kwargs)
    model = model.eval()

    # GPTQ/rotation calibrate with on-device forwards, so the model goes on the
    # GPU for the pass. RTN needs no forwards: the model stays on CPU while the
    # bank build streams weight chunks through the GPU (MASE_PHASE_BANK_DEVICE),
    # so the GPU never holds FP weights + bank + temporaries at once.
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        os.environ.setdefault("MASE_PHASE_BANK_DEVICE", cfg.device)
    if spec.gptq_weights:
        model = model.to(cfg.device)

    # Task-aligned calibration: the calib set (wikitext2 for PPL, a task-token
    # file for a task) + Erry clip drive the decode weight bank.
    gptq_cfg: dict[str, Any] = {
        "dataset": cfg.calib_dataset, "nsamples": cfg.gptq_nsamples,
        "seqlen": cfg.gptq_seqlen, "cali_batch_size": cfg.gptq_cali_batch_size,
        "clip_search_y": cfg.clip_search_y, "max_layers": cfg.gptq_max_layers,
        "hf_token": cfg.hf_token, "checkpoint_dir": cfg.gptq_checkpoint_dir,
    }
    pass_args = build_decode_pass_args(
        cfg.model_name, cfg.device, spec, gptq_cfg if spec.gptq_weights else None
    )

    model, _ = quantize_module_transform_pass(model, pass_args)

    # Decode-PPL-only trials never score the FP prefill path, so the FP weight
    # bank is dead weight: fold the decode bank into the parameters and drop it.
    # This halves resident VRAM (~2x model -> ~1x), the difference between fitting
    # and OOMing on a shared 48 GB GPU.
    tasks = [t for t in (cfg.tasks or "").split(",") if t.strip()]
    if not tasks and not cfg.eval_prefill_ppl:
        for m in model.modules():
            if hasattr(m, "collapse_to_decode_bank"):
                m.collapse_to_decode_bank()

    model = model.to(cfg.device).eval()
    install_phase_context_pre_hooks(model)
    row["calib"] = cfg.calib_dataset

    # Perplexity: forced-decode (and optionally the forced-prefill FP reference).
    if cfg.eval_ppl:
        loader = get_loaders(
            "wikitext2", nsamples=cfg.eval_ppl_nsamples, seed=0,
            seqlen=cfg.eval_ppl_seqlen, model=cfg.model_name, hf_token=cfg.hf_token,
        )
        if cfg.eval_prefill_ppl:
            row["prefill_ppl"] = round(_compute_calibration_perplexity(
                model, loader, cfg.device, label="prefill_fp", score_phase="prefill"), 4)
        row["cont_ppl"] = round(_compute_calibration_perplexity(
            model, loader, cfg.device, label="decode_quant", score_phase="decode"), 4)

    if tasks:
        scores = _run_tasks(model, tokenizer, tasks, cfg.task_limit, cfg.task_batch_size)
        for t, v in scores.items():
            row[t] = round(v, 4)

    row["runtime_sec"] = round(time.time() - t0, 1)
    return row


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode-accuracy worker for one precision point")
    p.add_argument("--model_name", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--local_files_only", type=_boolarg, default=True)
    p.add_argument("--trust_remote_code", type=_boolarg, default=False)
    p.add_argument("--hf_token", default=None)

    # Precision knobs
    p.add_argument("--attn_w", type=_wtype, default=4)
    p.add_argument("--ffn_w", type=_wtype, default=4)
    p.add_argument("--kv", type=_wtype, default=4)
    p.add_argument("--w_fmt", default="mxint", choices=("mxint", "mxfp"))
    p.add_argument("--kv_fmt", default="mxint", choices=("mxint", "mxfp"))
    p.add_argument("--weight_block", type=int, default=32)
    p.add_argument("--kv_block", type=int, default=32)
    p.add_argument("--act_w", type=_wtype, default=8)
    p.add_argument("--act_fmt", default="mxint", choices=("mxint", "mxfp"))
    p.add_argument("--act_block", type=int, default=32)
    p.add_argument("--use_gptq", type=_boolarg, default=False)
    p.add_argument("--use_rotation", type=_boolarg, default=False,
                   help="selective phase-aware rotation (decode-only); runs GPTQ internally")
    p.add_argument("--fp_setting", type=_pair, default=None,
                   help="FP_SETTING vector-unit minifloat 'E,M' (SiLU/RMSNorm, +softmax/rope below)")
    p.add_argument("--fp_setting_attention", type=_boolarg, default=False,
                   help="extend FP_SETTING to softmax/rope (needs eager attention)")
    p.add_argument("--quant_attn_internals", type=_boolarg, default=False)

    # Calibration (task-aligned) + GPTQ/Erry-clip
    p.add_argument("--calib_dataset", default="wikitext2",
                   help="GPTQ/rotation calibration set: 'wikitext2' or 'file:calib/<...>.pt' (task-aligned)")
    p.add_argument("--clip_search_y", type=_boolarg, default=False,
                   help="Erry (output-norm) block-wise clipping in GPTQ")
    p.add_argument("--gptq_nsamples", type=int, default=128)
    p.add_argument("--gptq_seqlen", type=int, default=2048)
    p.add_argument("--gptq_cali_batch_size", type=int, default=32)
    p.add_argument("--gptq_max_layers", type=int, default=None)
    p.add_argument("--gptq_checkpoint_dir", default=None)

    # Perplexity + tasks
    p.add_argument("--fp_only", type=_boolarg, default=False,
                   help="score the unquantised model once (shared FP prefill reference) and exit")
    p.add_argument("--eval_ppl", type=_boolarg, default=True, help="compute decode PPL")
    p.add_argument("--eval_prefill_ppl", type=_boolarg, default=False,
                   help="also score forced-prefill PPL in this trial (needs the FP bank kept "
                        "resident; normally computed once via --fp_only instead)")
    p.add_argument("--eval_ppl_nsamples", type=int, default=40)
    p.add_argument("--eval_ppl_seqlen", type=int, default=2048)
    p.add_argument("--tasks", default="", help="comma list, e.g. gsm8k,ifeval (empty = ppl only)")
    p.add_argument("--task_limit", type=int, default=None)
    p.add_argument("--task_batch_size", type=int, default=8)

    p.add_argument("--out", required=True, help="path to write the JSON result row")
    return p.parse_args()


def _boolarg(x: str | bool) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _wtype(x: str):
    """A width token: int for MXINT, 'E*M*' string for MXFP"""
    return int(x) if str(x).isdigit() else x


def _pair(x: str | None):
    if not x:
        return None
    a, b = str(x).split(",")
    return (int(a), int(b))


def main() -> None:
    cfg = _parse_args()
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        tag = "fp_reference" if cfg.fp_only else build_spec(cfg).tag
    except Exception:
        tag = ""
    try:
        row = evaluate_fp_reference(cfg) if cfg.fp_only else evaluate_decode_point(cfg)
    except Exception as e:  # a crashed trial writes an error row (tagged, so the
        # orchestrator can cache/dedupe it instead of losing its identity)
        row = {"tag": tag, "cont_ppl": "", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
    out.write_text(json.dumps(row, indent=2))
    print(f"[eval_decode] wrote {out}  (cont_ppl={row.get('cont_ppl')}, error={row.get('error') or 'none'})")


if __name__ == "__main__":
    main()
