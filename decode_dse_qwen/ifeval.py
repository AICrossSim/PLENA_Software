"""IFEval metric for the Qwen3-32B decode chip.

IFEval measures *verifiable* instruction following: each prompt carries
programmatic checks (respond in JSON, >=3 bullets, no commas, ...). We generate a response on the
disaggregated decode chip, strip the thinking, and score it with lm-eval-harness's official IFEval
scorer with the following metrics:

    strict prompt accuracy       -- fraction of prompts where ALL instructions are satisfied
    strict instruction accuracy  -- fraction of individual instructions satisfied

Each evaluated prompt also records its output length and per-instruction correctness, which feeds
the "accuracy vs output length" plot (does quantisation hurt more on long, decode-heavy generations?).
"""

from __future__ import annotations

from pathlib import Path

import torch

from decode_dse_qwen import disagg_serve

DATASET, SPLIT = "google/IFEval", "train"   # IFEval ships its 541-prompt eval set as 'train'


def load_ifeval(n: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=SPLIT)
    docs = [{"key": d["key"], "prompt": d["prompt"],
             "instruction_id_list": d["instruction_id_list"], "kwargs": d["kwargs"]} for d in ds]
    return docs if n is None else docs[:n]


def _score_strict(doc: dict, answer: str):
    """Official strict check -> (all_satisfied: bool, per_instruction: list[bool])."""
    from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict
    inp = InputExample(key=doc["key"], instruction_id_list=doc["instruction_id_list"],
                       prompt=doc["prompt"], kwargs=doc["kwargs"])
    out = test_instruction_following_strict(inp, answer)
    return bool(out.follow_all_instructions), [bool(x) for x in out.follow_instruction_list]


@torch.no_grad()
def evaluate(decode, tok, prompt_caches, spec, gpus, *, max_new_tokens, sampling, seed=0):
    """Generate + strict-score every prompt on the decode chip, using PRECOMPUTED prefill caches (no
    prefill model resident). Returns aggregate metrics and per-prompt records (n_generated + per-
    instruction correctness, for the length plot)"""
    records = []
    for i, c in enumerate(prompt_caches):
        answer, n_gen = disagg_serve.disagg_generate(
            decode, tok, c, spec, gpus, max_new_tokens=max_new_tokens, sampling=sampling, seed=seed + i)
        prompt_ok, inst_ok = _score_strict(c["doc"], answer)
        records.append({"key": c["doc"]["key"], "n_generated": n_gen,
                        "prompt_strict": prompt_ok, "inst_strict": inst_ok})
    return aggregate(records), records


def aggregate(records: list[dict]) -> dict:
    """Collapse per-prompt records into the two strict IFEval numbers."""
    if not records:
        return {"strict_prompt_acc": 0.0, "strict_inst_acc": 0.0, "n_prompts": 0}
    insts = [x for r in records for x in r["inst_strict"]]
    return {
        "strict_prompt_acc": sum(r["prompt_strict"] for r in records) / len(records),
        "strict_inst_acc": (sum(insts) / len(insts)) if insts else 0.0,
        "n_prompts": len(records),
    }


def length_bins(records: list[dict], edges=(0, 256, 1024, 4096, 16384, 1 << 30)) -> list[dict]:
    """Strict instruction accuracy binned by output length -> the (x=length, y=accuracy) plot data."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        insts = [x for r in records if lo <= r["n_generated"] < hi for x in r["inst_strict"]]
        if insts:
            out.append({"lo": lo, "hi": hi, "mid": (lo + min(hi, 2 * lo + 1)) / 2,
                        "acc": sum(insts) / len(insts), "n_inst": len(insts),
                        "n_prompts": sum(1 for r in records if lo <= r["n_generated"] < hi)})
    return out


def build_calib_file(tok, out_path: str, *, n_prompts: int, seqlen: int = 2048,
                     enable_thinking: bool = True, hold_out_from: int = 0) -> str:
    """Write held-out IFEval prompts as a chop GPTQ `file:` calibration loader (same-task calibration).
    Prompts come from the front of IFEval; the search evaluates on the disjoint tail."""
    texts = [tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=enable_thinking)
             for d in load_ifeval()[hold_out_from:]]
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids.flatten()
    n = min(n_prompts, ids.numel() // seqlen)
    if n == 0:
        raise RuntimeError(f"IFEval calib: only {ids.numel()} tokens, need >= {seqlen}")
    loader = []
    for i in range(n):
        inp = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).clone()
        tar = inp.clone(); tar[:, :-1] = -100
        loader.append((inp, tar))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"loader": loader, "seqlen": seqlen, "source": "ifeval"}, out_path)
    return f"file:{out_path}"
