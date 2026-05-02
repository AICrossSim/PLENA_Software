"""Collect task-aligned calibration tokens via TokenCollector.

Generic helper: load any model, attach ``TokenCollector``, forward prompts
from a configurable source (HF dataset / JSONL / plain text), and save a
``get_loaders``-compatible token loader to disk.

Prompt sources (--prompt_source):
    hf:<path>[:<config>][:<split>][:<col1,col2,...>]
        Load a HuggingFace dataset. Columns are concatenated with newline
        per row; if no columns are given, the row's first text-like field
        is used.
        Examples:
            hf:gsm8k:main:train:question,answer
            hf:openai/openai_humaneval::test:prompt
            hf:wikitext:wikitext-2-raw-v1:train:text

    jsonl:<path>[:<field>]
        One JSON object per line. ``field`` defaults to ``"text"`` and may
        be a comma list to concat multiple fields.
        Example:
            jsonl:bfcl/queries.jsonl:question

    txt:<path>
        Plain text, one prompt per line.

    lm_eval:<task>[:<limit>]
        Drive forwards via lm-eval-harness on the given task. The hook
        captures whatever lm-eval feeds into ``model.forward`` (so the
        calibration tokens are exactly what eval will see — few-shot
        context, chat template, etc.). ``limit`` defaults to 20; once the
        TokenCollector cap is hit the hook silently detaches and the
        remaining lm-eval forwards are wasted but harmless.
        Example:
            lm_eval:gsm8k:20

A ``--prompt_template`` (Python str.format) is applied per row; when
omitted, columns are joined with newlines.

Usage:
    python -m quant_eval.eval.collect_calib \\
        --model_name Qwen/Qwen3-0.6B \\
        --device cuda:0 \\
        --prompt_source "hf:gsm8k:main:train:question,answer" \\
        --prompt_template "Question: {question}\\nAnswer: {answer}" \\
        --save_path calib/qwen3_gsm8k.pt \\
        --nsamples 32 --seqlen 1024 --max_prompts 256
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterator

import torch

from quant_eval.utils import get_logger, set_logging_verbosity, setup_model
from chop.passes.module.transforms.gptq import CollectorFull, TokenCollector

logger = get_logger(__name__)
set_logging_verbosity("info")


# ---------------------------------------------------------------------------
# Prompt source parsing.
# ---------------------------------------------------------------------------

def _iter_hf(spec: str, template: str | None) -> Iterator[str]:
    """spec body: '<path>[:<config>][:<split>][:<col1,col2,...>]'"""
    import datasets

    parts = spec.split(":")
    path = parts[0]
    cfg = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    cols = parts[3].split(",") if len(parts) > 3 and parts[3] else None

    ds = datasets.load_dataset(path, cfg, split=split).shuffle(seed=0)
    if cols is None:
        # Auto-pick first string column.
        feature_cols = [
            n for n, t in ds.features.items()
            if getattr(t, "dtype", None) == "string"
        ]
        if not feature_cols:
            raise ValueError(f"no string columns in {path}; pass cols explicitly.")
        cols = feature_cols[:1]
        logger.info("[hf] auto-selected columns: %s", cols)

    for row in ds:
        if template:
            try:
                yield template.format(**row)
            except KeyError as e:
                raise KeyError(
                    f"template references missing column {e}; row keys = {list(row.keys())}"
                ) from None
        else:
            yield "\n".join(str(row[c]) for c in cols)


def _iter_jsonl(spec: str, template: str | None) -> Iterator[str]:
    """spec body: '<path>[:<field1,field2,...>]'"""
    parts = spec.split(":", 1)
    path = parts[0]
    fields = parts[1].split(",") if len(parts) > 1 and parts[1] else ["text"]
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if template:
                yield template.format(**row)
            else:
                yield "\n".join(str(row[c]) for c in fields)


def _iter_txt(path: str, template: str | None) -> Iterator[str]:
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield template.format(text=line) if template else line


def iter_prompts(prompt_source: str, template: str | None) -> Iterator[str]:
    if prompt_source.startswith("hf:"):
        yield from _iter_hf(prompt_source[3:], template)
    elif prompt_source.startswith("jsonl:"):
        yield from _iter_jsonl(prompt_source[6:], template)
    elif prompt_source.startswith("txt:"):
        yield from _iter_txt(prompt_source[4:], template)
    else:
        raise ValueError(
            f"unknown prompt_source prefix: {prompt_source!r}. "
            "Use one of: hf:..., jsonl:..., txt:..."
        )


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def main(
    model_name: str = "Qwen/Qwen3-0.6B",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    prompt_source: str = "hf:gsm8k:main:train:question,answer",
    prompt_template: str | None = "Question: {question}\nAnswer: {answer}",
    save_path: str = "calib/calib.pt",
    nsamples: int = 32,
    seqlen: int = 1024,
    max_prompts: int = 256,
    overwrite: bool = False,
    min_prefill_tokens: int = 8,
):
    """Forward task-aligned prompts through a fresh model with TokenCollector
    attached and save a calibration token loader to disk.

    Args:
        model_name:        HF model id (used both for tokenizer and forward).
        device:            CUDA device.
        dtype:             Model dtype (float16/bfloat16/float32).
        prompt_source:     ``hf:...`` / ``jsonl:...`` / ``txt:...`` (see module
                           docstring for grammar).
        prompt_template:   Optional Python ``str.format`` template; if None,
                           columns/fields are newline-joined.
        save_path:         Output path (will be created/overwritten by hook).
        nsamples:          Number of seqlen-sized chunks to save.
        seqlen:            Tokens per chunk.
        max_prompts:       Hard cap on prompts forwarded (safety net).
        overwrite:         Replace existing save_path; otherwise reuse.
        min_prefill_tokens: Skip prompts shorter than this (decode-step filter).
    """
    save_p = Path(save_path)
    if save_p.exists() and not overwrite:
        logger.info("calibration file %s exists; reuse (overwrite=False).", save_p)
        return
    if save_p.exists():
        save_p.unlink()
    save_p.parent.mkdir(parents=True, exist_ok=True)

    dtype_t = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}[dtype]

    tokenizer, model = setup_model(
        model_name=model_name, model_parallel=False, dtype=dtype_t,
        device=device, attn_implementation="eager",
    )
    model.eval().to(device)

    raise_on_full = prompt_source.startswith("lm_eval:")
    collector = TokenCollector(
        model=model, target_nsamples=nsamples, seqlen=seqlen,
        save_path=save_p, overwrite=False,
        min_prefill_tokens=min_prefill_tokens,
        raise_on_full=raise_on_full,
    ).attach()

    if prompt_source.startswith("lm_eval:"):
        # Drive forwards with a real lm-eval pass; TokenCollector captures the
        # exact prefill ids that lm-eval feeds into the model. The hook raises
        # CollectorFull once the buffer is full so we abort the eval here
        # rather than wasting compute on forwards we won't use.
        from quant_eval.eval.lm_eval import evaluate_with_lm_eval

        parts = prompt_source[len("lm_eval:"):].split(":")
        task = parts[0]
        eval_limit = int(parts[1]) if len(parts) > 1 and parts[1] else 20
        logger.info(
            "[lm_eval driver] task=%s limit=%d (will abort once buffer is full)",
            task, eval_limit,
        )
        try:
            evaluate_with_lm_eval(
                model=model, tokenizer=tokenizer, tasks=task,
                max_length=seqlen, batch_size=4, log_samples=False,
                limit=eval_limit,
            )
        except CollectorFull as e:
            logger.info("[lm_eval driver] aborted as planned: %s", e)
        used = -1  # not meaningful here; the lm-eval pass handled iteration
    else:
        # Manual prompt iteration: tokenize each row and forward it ourselves.
        used = 0
        for text in iter_prompts(prompt_source, prompt_template):
            if collector.complete or used >= max_prompts:
                break
            ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=seqlen,
            ).input_ids.to(device)
            if ids.shape[-1] < min_prefill_tokens:
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
        "calibration: source=%s forwarded=%s saved=%s complete=%s",
        prompt_source, used, save_p, collector.complete,
    )

    del model, tokenizer, collector
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
