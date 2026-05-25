#!/usr/bin/env python3
"""Build BFCL official calibration prompts for Llama 3.1 (prompt-only output).

This script is for GPTQ calibration data preparation only.
It outputs one JSON object per line with a single field:

    {"prompt": "..."}

The prompt is rendered from official BFCL entries using Llama-3.1
`apply_chat_template(messages, tools=..., add_generation_prompt=True)` so it
matches what the model actually consumes during inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_LLAMA31_SNAPSHOT = (
    "/home/yh3525/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)


def _detect_bfcl_data_path(cli_value: str | None) -> Path:
    if cli_value:
        p = Path(cli_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--bfcl-data-path does not exist: {p}")
        return p

    try:
        from bfcl_eval.constants.eval_config import PROMPT_PATH  # type: ignore

        p = (Path(PROMPT_PATH) / "BFCL_v4_multiple.json").resolve()
        if p.exists():
            return p
    except Exception:
        pass

    fallback = Path(
        "/home/yh3525/FYP/Coprocessor_for_Llama/acc_simulator/third_party/"
        "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/"
        "BFCL_v4_multiple.json"
    ).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Unable to locate BFCL_v4_multiple.json. Provide --bfcl-data-path explicitly."
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No entries loaded from {path}")
    return rows


def _import_local_bfcl_utils() -> Any:
    local_repo_root = Path(
        "/home/yh3525/FYP/Coprocessor_for_Llama/acc_simulator/third_party/"
        "gorilla/berkeley-function-call-leaderboard"
    )
    if not local_repo_root.exists():
        raise FileNotFoundError(f"Local gorilla repo not found: {local_repo_root}")

    sys.path.insert(0, str(local_repo_root))
    import bfcl_eval.utils as bfcl_utils  # type: ignore

    return bfcl_utils


def _apply_official_preprocessing(rows: list[dict[str, Any]], bfcl_utils: Any) -> list[dict[str, Any]]:
    rows = deepcopy(rows)
    rows = bfcl_utils.process_agentic_test_case(rows)
    rows = bfcl_utils.populate_test_cases_with_predefined_functions(rows)
    rows = bfcl_utils.add_language_specific_hint_to_function_doc(rows)
    return rows


def _extract_first_turn_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    q = row.get("question", [])
    if q and isinstance(q, list) and isinstance(q[0], list):
        first_turn = q[0]
    elif isinstance(q, list):
        first_turn = q
    else:
        first_turn = []

    return [
        {
            "role": str(msg.get("role", "user")),
            "content": str(msg.get("content", "")),
        }
        for msg in first_turn
    ]


def _normalize_schema_types(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_schema_types(x) for x in node]
    if not isinstance(node, dict):
        return node

    out = deepcopy(node)
    t = out.get("type")
    mapping = {
        "float": "number",
        "double": "number",
        "int": "integer",
        "integer": "integer",
        "bool": "boolean",
        "boolean": "boolean",
        "str": "string",
        "string": "string",
        "dict": "object",
        "map": "object",
        "list": "array",
        "array": "array",
        "any": "string",
        "ArrayList": "array",
        "Array": "array",
    }
    if isinstance(t, str):
        out["type"] = mapping.get(t, t)

    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {
            k: _normalize_schema_types(v) for k, v in out["properties"].items()
        }
    if "items" in out:
        out["items"] = _normalize_schema_types(out["items"])

    return out


def _to_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for func in deepcopy(functions):
        name = str(func.get("name", ""))
        func["name"] = re.sub(r"\.", "_", name)

        params = func.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        params = _normalize_schema_types(params)
        params["type"] = "object"
        func["parameters"] = params

        tools.append({"type": "function", "function": func})
    return tools


def _render_prompt(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template returned non-string output")
    return rendered


def build_prompt_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        messages = _extract_first_turn_messages(row)
        functions = row.get("function", [])
        if not isinstance(functions, list):
            raise TypeError("function field is not a list")

        tools = _to_tools(functions)
        prompt = _render_prompt(tokenizer, messages=messages, tools=tools)
        out.append({"prompt": prompt})
    return out


def _write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BFCL official wrapped prompts for Llama 3.1 GPTQ calibration"
    )
    parser.add_argument(
        "--bfcl-data-path",
        type=str,
        default=None,
        help="Optional explicit path to BFCL_v4_multiple.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_LLAMA31_SNAPSHOT,
        help="Llama-3.1 snapshot path for chat template rendering",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="calib/bfcl_multiple_prompts_official_wrapped_llama31.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    src_path = _detect_bfcl_data_path(args.bfcl_data_path)
    model_path = Path(args.model_path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"--model-path does not exist: {model_path}")

    bfcl_utils = _import_local_bfcl_utils()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    raw_rows = _load_jsonl(src_path)
    processed_rows = _apply_official_preprocessing(raw_rows, bfcl_utils=bfcl_utils)
    prompt_rows = build_prompt_rows(processed_rows, tokenizer=tokenizer)
    _write_jsonl(prompt_rows, out_path)

    print(f"source: {src_path}")
    print(f"model_path: {model_path}")
    print(f"output: {out_path}")
    print(f"rows: {len(prompt_rows)}")


if __name__ == "__main__":
    main()
