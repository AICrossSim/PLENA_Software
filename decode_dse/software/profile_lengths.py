"""Profile task input/output token lengths (ISL / OSL).

Measures, per task, the prompt length (ISL) the prefill chip processes and the
generated length (OSL) the decode chip produces, under the same semantics the
task eval uses: lm-eval's few-shot prompt construction, the task's own stop
strings, and its own generation cap.

* ISL: ``task.fewshot_context`` at the task's default ``num_fewshot`` (GSM8K is
  5-shot; a zero-shot prompt would understate ISL).
* OSL: generation ends at EOS or the task's first stop string ("until"), capped
  at the task's ``max_gen_toks``.

    python -m decode_dse.software.profile_lengths \
        --model_name meta-llama/Llama-3.1-8B-Instruct --tasks gsm8k,ifeval --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# lm-eval's HFLM default generation cap, used when a task sets no max_gen_toks.
LM_EVAL_DEFAULT_MAX_GEN_TOKS = 256


def _task_setup(task_name: str, limit: int | None, max_new_override: int | None):
    """The exact (prompt, generation-kwargs) pairs lm-eval's evaluator sends.

    Built via ``task.build_all_requests()`` — the SAME request-construction
    path ``simple_evaluate`` uses (few-shot sampler, de-limiters, per-request
    resolved generation kwargs)"""
    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict([task_name], TaskManager())[task_name]
    task.build_all_requests(limit=limit, rank=0, world_size=1)
    if not task.instances:
        raise RuntimeError(f"{task_name}: build_all_requests produced no instances.")

    prompts, gen_settings = [], set()
    for inst in task.instances:
        if inst.request_type != "generate_until":
            raise ValueError(
                f"{task_name} is {inst.request_type!r}, not generative — "
                "length profiling only applies to generate_until tasks.")
        ctx, gen_kwargs = inst.args
        prompts.append(ctx)
        gen_settings.add((tuple(s for s in (gen_kwargs.get("until") or []) if s),
                          gen_kwargs.get("max_gen_toks")))

    # One profile describes one workload: a task whose requests disagree on
    # stop strings or caps would need per-instance handling, not a silent
    # last-one-wins — fail loudly so the variation is looked at.
    if len(gen_settings) != 1:
        raise RuntimeError(
            f"{task_name}: instances disagree on generation settings ({gen_settings}); "
            "profile them per group instead of assuming one workload.")
    (until_t, max_gen), = gen_settings
    until = list(until_t)
    max_new = int(max_new_override or max_gen or LM_EVAL_DEFAULT_MAX_GEN_TOKS)
    num_fewshot = int(task.config.num_fewshot or 0)
    return prompts, num_fewshot, until, max_new


def _osl_tokens(tok, row: torch.Tensor, until: list[str], max_new: int) -> tuple[int, bool]:
    """Generated length for one row: EOS position, else stop-string cut, else
    the full generation (capped). Returns (osl, hit_cap)."""
    eos = (row == tok.eos_token_id).nonzero()
    n = int(eos[0].item() + 1) if len(eos) else int((row != tok.pad_token_id).sum().item())
    text = tok.decode(row[:n], skip_special_tokens=True)
    cuts = [text.find(s) for s in until if s in text]
    if cuts:
        # The serving stack stops generating at the stop string; count the
        # tokens up to it
        n = max(1, len(tok(text[:min(cuts)], add_special_tokens=False).input_ids))
        return n, False
    return n, (len(eos) == 0 and n >= max_new)


@torch.no_grad()
def profile(cfg: argparse.Namespace) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load = {"local_files_only": cfg.local_files_only, "trust_remote_code": cfg.trust_remote_code}
    if cfg.hf_token:
        load["token"] = cfg.hf_token
    tok = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left", **load)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, **load)
    model = model.to(cfg.device).eval()

    result: dict[str, dict] = {}
    for task in [t.strip() for t in cfg.tasks.split(",") if t.strip()]:
        prompts, num_fewshot, until, max_new = _task_setup(task, cfg.limit, cfg.max_new_tokens)
        print(f"  [{task}] {num_fewshot}-shot, until={until or 'EOS only'}, cap={max_new}")
        isl, osl, capped = [], [], 0
        for i in range(0, len(prompts), cfg.batch_size):
            batch = prompts[i:i + cfg.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg.max_input).to(cfg.device)
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j in range(len(batch)):
                isl.append(int(enc["attention_mask"][j].sum().item()))
                n, hit = _osl_tokens(tok, new[j], until, max_new)
                osl.append(n)
                capped += hit
            print(f"  [{task}] {min(i + cfg.batch_size, len(prompts))}/{len(prompts)}")
        result[task] = {"isl": isl, "osl": osl, "num_fewshot": num_fewshot,
                        "until": until, "cap": max_new,
                        "capped_frac": round(capped / max(len(osl), 1), 4)}
        import numpy as np
        print(f"  [{task}] ISL p50={np.median(isl):.0f} mean={np.mean(isl):.0f} | "
              f"OSL p50={np.median(osl):.0f} mean={np.mean(osl):.0f} p95={np.percentile(osl, 95):.0f} | "
              f"capped {capped}/{len(osl)}")

    out = Path(cfg.output_dir) / cfg.model_name.split("/")[-1] / "task_lengths.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[profile_lengths] wrote {out}")
    return out


def _boolarg(x):
    return x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    p = argparse.ArgumentParser(description="Profile task ISL/OSL token lengths in FP (eval semantics).")
    p.add_argument("--model_name", required=True)
    p.add_argument("--tasks", default="gsm8k,ifeval")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="override the generation cap (default: the task's max_gen_toks, else lm-eval's 256)")
    p.add_argument("--max_input", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/decode_dse")
    p.add_argument("--local_files_only", type=_boolarg, default=True)
    p.add_argument("--trust_remote_code", type=_boolarg, default=False)
    p.add_argument("--hf_token", default=None)
    profile(p.parse_args())


if __name__ == "__main__":
    main()
