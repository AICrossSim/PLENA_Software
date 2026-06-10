#!/usr/bin/env python3
"""Build BFCL official calibration prompts for Qwen3 FC.

This script is for GPTQ calibration data preparation only.  It renders BFCL
multiple prompts with the same structure as BFCL's official QwenFCHandler and
writes prompt-only JSONL rows:

    {"prompt": "..."}

No id/source metadata is written because GPTQ only consumes model input text.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_BFCL_REPO = Path(
    "/home/yh3525/FYP/Coprocessor_for_Llama/acc_simulator/third_party/"
    "gorilla/berkeley-function-call-leaderboard"
)
DEFAULT_BFCL_DATA = DEFAULT_BFCL_REPO / "bfcl_eval/data/BFCL_v4_multiple.json"
DEFAULT_OUT = "calib/bfcl_multiple_prompts_official_wrapped_qwen3.jsonl"


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

    if DEFAULT_BFCL_DATA.exists():
        return DEFAULT_BFCL_DATA.resolve()
    raise FileNotFoundError("Unable to locate BFCL_v4_multiple.json; pass --bfcl-data-path.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No BFCL rows loaded from {path}")
    return rows


def _import_local_bfcl_utils() -> Any:
    try:
        import bfcl_eval.utils as bfcl_utils  # type: ignore

        return bfcl_utils
    except Exception:
        pass

    if not DEFAULT_BFCL_REPO.exists():
        raise FileNotFoundError(f"Local BFCL repo not found: {DEFAULT_BFCL_REPO}")
    sys.path.insert(0, str(DEFAULT_BFCL_REPO))
    import bfcl_eval.utils as bfcl_utils  # type: ignore

    return bfcl_utils


def _apply_official_preprocessing(rows: list[dict[str, Any]], bfcl_utils: Any) -> list[dict[str, Any]]:
    rows = deepcopy(rows)
    rows = bfcl_utils.process_agentic_test_case(rows)
    rows = bfcl_utils.populate_test_cases_with_predefined_functions(rows)
    rows = bfcl_utils.add_language_specific_hint_to_function_doc(rows)
    return rows


def _extract_first_turn_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    q = row.get("question", [])
    if q and isinstance(q, list) and isinstance(q[0], list):
        first_turn = q[0]
    elif isinstance(q, list):
        first_turn = q
    else:
        first_turn = []
    messages: list[dict[str, str]] = []
    for msg in first_turn:
        if not isinstance(msg, dict):
            continue
        messages.append({
            "role": str(msg.get("role", "user")),
            "content": str(msg.get("content", "")),
        })
    if not messages:
        raise ValueError(f"BFCL row {row.get('id', '<unknown>')} has no messages")
    return messages


def render_qwen3_fc_prompt(messages: list[dict[str, str]], functions: list[dict[str, Any]]) -> str:
    """Render the official BFCL QwenFCHandler prompt.

    This mirrors bfcl_eval/model_handler/local_inference/qwen_fc.py without
    importing bfcl_eval at runtime.  The model is asked to return JSON objects
    under <tool_call> tags with an "arguments" field.
    """
    formatted = ""

    if functions:
        formatted += "<|im_start|>system\n"
        if messages and messages[0].get("role") == "system":
            formatted += messages[0].get("content", "") + "\n\n"
        formatted += (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>"
        )
        for tool in functions:
            formatted += "\n" + json.dumps(tool, ensure_ascii=False)
        formatted += (
            "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    elif messages and messages[0].get("role") == "system":
        formatted += f"<|im_start|>system\n{messages[0].get('content', '')}<|im_end|>\n"

    last_query_index = len(messages) - 1
    for offset, message in enumerate(reversed(messages)):
        idx = len(messages) - 1 - offset
        content = message.get("content", "")
        if (
            message.get("role") == "user"
            and isinstance(content, str)
            and not (content.startswith("<tool_response>") and content.endswith("</tool_response>"))
        ):
            last_query_index = idx
            break

    for idx, message in enumerate(messages):
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "user" or (role == "system" and idx != 0):
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif role == "assistant":
            reasoning_content = ""
            if "</think>" in content:
                parts = content.split("</think>")
                reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = parts[-1].lstrip("\n")
            if idx > last_query_index:
                if idx == len(messages) - 1 or reasoning_content:
                    formatted += (
                        f"<|im_start|>{role}\n<think>\n"
                        + reasoning_content.strip("\n")
                        + "\n</think>\n\n"
                        + content.lstrip("\n")
                    )
                else:
                    formatted += f"<|im_start|>{role}\n{content}"
            else:
                formatted += f"<|im_start|>{role}\n{content}"
            formatted += "<|im_end|>\n"
        elif role == "tool":
            prev_role = messages[idx - 1].get("role") if idx > 0 else None
            next_role = messages[idx + 1].get("role") if idx < len(messages) - 1 else None
            if idx == 0 or prev_role != "tool":
                formatted += "<|im_start|>user"
            formatted += f"\n<tool_response>\n{content}\n</tool_response>"
            if idx == len(messages) - 1 or next_role != "tool":
                formatted += "<|im_end|>\n"

    formatted += "<|im_start|>assistant\n"
    return formatted


def build_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        functions = row.get("function", [])
        if not isinstance(functions, list):
            raise TypeError(f"BFCL row {row.get('id', '<unknown>')} function field is not a list")
        prompt = render_qwen3_fc_prompt(_extract_first_turn_messages(row), functions)
        out.append({"prompt": prompt})
    return out


def _write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official BFCL multiple prompts for Qwen3 FC GPTQ calibration")
    parser.add_argument("--bfcl-data-path", type=str, default=None)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--skip-official-preprocessing", action="store_true")
    args = parser.parse_args()

    src_path = _detect_bfcl_data_path(args.bfcl_data_path)
    rows = _load_jsonl(src_path)
    if not args.skip_official_preprocessing:
        rows = _apply_official_preprocessing(rows, _import_local_bfcl_utils())
    prompt_rows = build_prompt_rows(rows)
    out_path = Path(args.out).expanduser().resolve()
    _write_jsonl(prompt_rows, out_path)

    print(f"source: {src_path}")
    print(f"output: {out_path}")
    print(f"rows: {len(prompt_rows)}")


if __name__ == "__main__":
    main()
