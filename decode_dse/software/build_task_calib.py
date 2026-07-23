"""Build task-aligned calibration token files.

GPTQ/rotation on the decode weight bank should calibrate on the task's own
tokens, not generic wikitext2. This util runs the task's formatted prompts
through the model with a ``TokenCollector`` attached and saves the captured
stream to ``file:calib/<model>_<task>_nN_sS.pt``, which the GPTQ/rotation
``dataset`` field then points at.

Example:
    python -m decode_dse.software.build_task_calib \
        --model_name meta-llama/Llama-3.1-8B-Instruct --task gsm8k \
        --nsamples 64 --seqlen 1024 --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def calib_path(model_name: str, task: str, nsamples: int, seqlen: int,
               calib_dir: str = "calib") -> Path:
    """Canonical calibration-file path for a (model, task) pair."""
    slug = model_name.replace("/", "_")
    return Path(calib_dir) / f"{slug}_{task}_n{nsamples}_s{seqlen}.pt"


def _task_prompt_text(task_name: str, limit: int | None) -> str:
    """Concatenate a task's formatted prompts (input + target) into one stream.

    Uses lm-eval's own prompt construction so the calibration distribution
    matches what the task eval will actually feed the model.
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict([task_name], TaskManager())[task_name]
    if task.has_test_docs():
        docs = task.test_docs()
    elif task.has_validation_docs():
        docs = task.validation_docs()
    else:
        docs = task.training_docs()

    parts: list[str] = []
    for i, doc in enumerate(docs):
        if limit is not None and i >= limit:
            break
        text = task.doc_to_text(doc)
        target = task.doc_to_target(doc)
        target = target if isinstance(target, str) else str(target)
        parts.append(f"{text}{target}")
    if not parts:
        raise RuntimeError(f"no documents found for task {task_name!r}.")
    return "\n\n".join(parts)


@torch.no_grad()
def build_calib(cfg: argparse.Namespace) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chop.passes.module.transforms.gptq import CollectorFull, TokenCollector

    out = calib_path(cfg.model_name, cfg.task, cfg.nsamples, cfg.seqlen, cfg.calib_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not cfg.overwrite:
        print(f"[build_task_calib] {out} exists (use --overwrite to rebuild)")
        return out

    load_kwargs = {"local_files_only": cfg.local_files_only, "trust_remote_code": cfg.trust_remote_code}
    if cfg.hf_token:
        load_kwargs["token"] = cfg.hf_token
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, **load_kwargs
    ).to(cfg.device).eval()

    text = _task_prompt_text(cfg.task, cfg.doc_limit)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_chunks = ids.shape[0] // cfg.seqlen
    if n_chunks == 0:
        raise RuntimeError(
            f"task {cfg.task} produced only {ids.shape[0]} tokens (< seqlen {cfg.seqlen})."
        )

    collector = TokenCollector(
        model, target_nsamples=cfg.nsamples, seqlen=cfg.seqlen,
        save_path=str(out), overwrite=cfg.overwrite, raise_on_full=True,
    ).attach()

    try:
        for c in range(n_chunks):
            chunk = ids[c * cfg.seqlen:(c + 1) * cfg.seqlen].unsqueeze(0).to(cfg.device)
            model(input_ids=chunk)
    except CollectorFull:
        pass  # buffer full and saved — the whole point
    finally:
        collector.finalize()  # flush a partial buffer if the stream ran short

    print(f"[build_task_calib] wrote {out} ({collector.total_tokens} tokens, task={cfg.task})")
    return out


def _boolarg(x: str | bool) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    p = argparse.ArgumentParser(description="Build a task-aligned GPTQ/rotation calibration file.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--task", required=True, help="lm-eval task name, e.g. gsm8k or ifeval")
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--calib_dir", default="calib")
    p.add_argument("--doc_limit", type=int, default=512, help="cap task docs scanned (None-like 0 = all)")
    p.add_argument("--overwrite", type=_boolarg, default=False)
    p.add_argument("--local_files_only", type=_boolarg, default=True)
    p.add_argument("--trust_remote_code", type=_boolarg, default=False)
    p.add_argument("--hf_token", default=None)
    cfg = p.parse_args()
    if cfg.doc_limit == 0:
        cfg.doc_limit = None
    build_calib(cfg)


if __name__ == "__main__":
    main()
