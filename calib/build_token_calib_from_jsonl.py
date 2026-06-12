#!/usr/bin/env python3
"""Build a GPTQ token-loader calibration file from prompt-only JSONL.

This lightweight path is useful for very large checkpoints where loading the
full model just to collect input ids is unnecessary.  The output mirrors the
TokenCollector checkpoint structure consumed by MASE GPTQ:

    {
        "loader": [(input_ids, target_ids), ...],
        "seqlen": int,
        "target_nsamples": int,
        "collected_samples": int,
        "format_version": 1,
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def _iter_prompts(path: Path, field: str):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get(field)
            if prompt:
                yield str(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GPTQ calibration token chunks from JSONL prompts")
    parser.add_argument("--model-name", required=True, help="Tokenizer model id or local path")
    parser.add_argument("--jsonl", required=True, help="Prompt-only JSONL path")
    parser.add_argument("--field", default="prompt")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--max-prompts", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    save_path = Path(args.save_path).expanduser().resolve()
    if save_path.exists() and not args.overwrite:
        print(f"reuse existing calibration file: {save_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    buffer: list[int] = []
    used = 0
    for prompt in _iter_prompts(Path(args.jsonl).expanduser().resolve(), args.field):
        if used >= args.max_prompts or len(chunks) >= args.nsamples:
            break
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if not ids:
            continue
        buffer.extend(ids)
        used += 1
        while len(buffer) >= args.seqlen and len(chunks) < args.nsamples:
            sample = torch.tensor(buffer[: args.seqlen], dtype=torch.long).unsqueeze(0)
            chunks.append((sample, sample.clone()))
            buffer = buffer[args.seqlen :]

    if len(chunks) < args.nsamples and buffer:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        padded = buffer[: args.seqlen] + [eos] * max(0, args.seqlen - len(buffer))
        sample = torch.tensor(padded[: args.seqlen], dtype=torch.long).unsqueeze(0)
        chunks.append((sample, sample.clone()))

    if not chunks:
        raise RuntimeError(f"No calibration samples produced from {args.jsonl}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "loader": chunks[: args.nsamples],
            "seqlen": args.seqlen,
            "target_nsamples": args.nsamples,
            "collected_samples": min(len(chunks), args.nsamples),
            "format_version": 1,
        },
        save_path,
    )
    print(f"source: {args.jsonl}")
    print(f"tokenizer: {args.model_name}")
    print(f"output: {save_path}")
    print(f"samples: {min(len(chunks), args.nsamples)}/{args.nsamples}")
    print(f"prompts_used: {used}")


if __name__ == "__main__":
    main()
