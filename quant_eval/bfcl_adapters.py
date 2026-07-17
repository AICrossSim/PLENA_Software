"""Model-specific BFCL response adapters.

The local OpenAI-compatible server has to return whatever shape the selected
BFCL model handler expects.  Keep model-specific transport cleanup here instead
of sharing Llama-only heuristics across Qwen and future model families.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

BFCL_ADAPTER_NAMES = ("auto", "qwen3_fc", "llama31_fc_legacy", "raw")
MAX_LITERAL_PARSE_CHARS = 65536


def _loads_json_or_literal(payload: str) -> Any:
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, RecursionError, MemoryError):
        if len(payload) > MAX_LITERAL_PARSE_CHARS:
            raise ValueError("Tool payload is too large for literal parsing")
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError) as exc:
            raise ValueError("Unable to parse tool payload") from exc


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _arguments_from_payload(parsed: dict[str, Any]) -> Any:
    if "arguments" in parsed:
        args = parsed["arguments"]
    elif "parameters" in parsed:
        args = parsed["parameters"]
    else:
        args = {k: v for k, v in parsed.items() if k != "name"}
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, RecursionError):
            return args
    return args


def _append_openai_tool_call(tool_calls: list[dict[str, Any]], parsed: Any) -> None:
    if not isinstance(parsed, dict):
        return
    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        return
    args = _arguments_from_payload(parsed)
    tool_calls.append({
        "id": f"call_{len(tool_calls)}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    })


def _parse_payload_sequence(payload: str) -> list[dict[str, Any]]:
    payload = (payload or "").strip()
    if not payload:
        return []
    if payload[0] not in "[{":
        return []

    tool_calls: list[dict[str, Any]] = []
    try:
        parts = [p.strip() for p in payload.split(";") if p.strip()]
        if len(parts) > 1:
            for part in parts:
                _append_openai_tool_call(tool_calls, _loads_json_or_literal(part))
            return tool_calls

        parsed = _loads_json_or_literal(payload)
        if isinstance(parsed, list):
            for item in parsed:
                _append_openai_tool_call(tool_calls, item)
        else:
            _append_openai_tool_call(tool_calls, parsed)
    except (json.JSONDecodeError, SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
        return []
    return tool_calls


def _iter_balanced_payloads(text: str):
    if not text:
        return
    open_to_close = {"{": "}", "[": "]"}
    closers = set(open_to_close.values())
    n = len(text)
    i = 0
    while i < n:
        if text[i] not in open_to_close:
            i += 1
            continue
        start = i
        stack = [open_to_close[text[i]]]
        quote = None
        escaped = False
        i += 1
        while i < n and stack:
            ch = text[i]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            else:
                if ch in ("'", '"'):
                    quote = ch
                elif ch in open_to_close:
                    stack.append(open_to_close[ch])
                elif ch in closers:
                    if ch != stack[-1]:
                        break
                    stack.pop()
            i += 1
        if not stack:
            yield text[start:i]
        else:
            i = start + 1


def _strip_markdown_json_fence(text: str) -> str:
    value = (text or "").strip().strip("`").strip()
    if value.lower().startswith("json\n"):
        value = value[5:].strip()
    return value


def _strip_qwen_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def _parse_tool_call_tags(text: str) -> tuple[list[dict[str, Any]], str]:
    # Official QwenFCHandler uses newlines around the JSON payload.  Accept a
    # relaxed form too so we can canonicalize generated variants back to the
    # official shape before BFCL sees them.
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text or "", flags=re.DOTALL)
    tool_calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            _append_openai_tool_call(tool_calls, _loads_json_or_literal(match.strip()))
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
            continue
    leftover = re.sub(pattern, "", text or "", flags=re.DOTALL).strip()
    return tool_calls, leftover


def _parse_fenced_json(text: str) -> tuple[list[dict[str, Any]], str]:
    pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    matches = re.findall(pattern, text or "", flags=re.DOTALL)
    tool_calls: list[dict[str, Any]] = []
    for match in matches:
        tool_calls.extend(_parse_payload_sequence(match.strip()))
    leftover = re.sub(pattern, "", text or "", flags=re.DOTALL).strip()
    return tool_calls, leftover


@dataclass(frozen=True)
class BFCLAdapter:
    name: str

    def parse_tool_calls(self, text: str) -> tuple[list[dict[str, Any]] | None, str]:
        return None, text

    def normalize_return_text(self, text: str) -> str:
        return text

    def return_text_from_tool_calls(self, tool_calls: list[dict[str, Any]], raw_text: str | None = None) -> str:
        return raw_text or self.normalize_return_text("")


class RawBFCLAdapter(BFCLAdapter):
    def __init__(self) -> None:
        super().__init__("raw")


class Qwen3FCAdapter(BFCLAdapter):
    """Adapter for BFCL's official QwenFCHandler protocol."""

    def __init__(self) -> None:
        super().__init__("qwen3_fc")

    def parse_tool_calls(self, text: str) -> tuple[list[dict[str, Any]] | None, str]:
        cleaned = _strip_qwen_think(text or "")
        for parser in (_parse_tool_call_tags, _parse_fenced_json):
            calls, leftover = parser(cleaned)
            if calls:
                return calls, leftover

        payload = _strip_markdown_json_fence(cleaned)
        calls = _parse_payload_sequence(payload)
        if calls:
            return calls, ""

        # If the model says "Here is the JSON:" before a valid payload, strip
        # only the transport prose.  Do not repair function names or argument
        # values.
        embedded: list[dict[str, Any]] = []
        for candidate in _iter_balanced_payloads(payload):
            embedded.extend(_parse_payload_sequence(candidate))
        if embedded:
            return embedded, ""
        return None, text

    def normalize_return_text(self, text: str) -> str:
        cleaned = _strip_qwen_think(text or "").strip()
        calls, _ = _parse_tool_call_tags(cleaned)
        if calls:
            return self.return_text_from_tool_calls(calls, raw_text=cleaned)
        return cleaned

    def return_text_from_tool_calls(self, tool_calls: list[dict[str, Any]], raw_text: str | None = None) -> str:
        payloads: list[str] = []
        for tc in tool_calls or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, RecursionError):
                    # Preserve malformed argument strings as model output. BFCL
                    # should count that as a format/semantic failure.
                    pass
            payloads.append(
                "<tool_call>\n"
                + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                + "\n</tool_call>"
            )
        return "\n".join(payloads) if payloads else (raw_text or "")


class Llama31FCLegacyAdapter(BFCLAdapter):
    """Legacy adapter for the earlier Llama 3.1 BFCL experiments."""

    def __init__(self) -> None:
        super().__init__("llama31_fc_legacy")

    def normalize_return_text(self, text: str) -> str:
        if not text:
            return text
        normalized = text.strip()
        header = "<|start_header_id|>assistant<|end_header_id|>"
        if header in normalized:
            normalized = normalized.split(header, 1)[1].strip()
        if "<|python_tag|>" in normalized:
            normalized = normalized.split("<|python_tag|>", 1)[1].strip()
        for stop in ("<|eom_id|>", "<|eot_id|>", "<|end_of_text|>"):
            if stop in normalized:
                normalized = normalized.split(stop, 1)[0].strip()
        return _strip_markdown_json_fence(normalized) or text

    def parse_tool_calls(self, text: str) -> tuple[list[dict[str, Any]] | None, str]:
        cleaned = self.normalize_return_text(text)
        for parser in (_parse_tool_call_tags, _parse_fenced_json):
            calls, leftover = parser(cleaned)
            if calls:
                return calls, leftover
        calls = _parse_payload_sequence(cleaned)
        if calls:
            return calls, ""
        embedded: list[dict[str, Any]] = []
        for candidate in _iter_balanced_payloads(cleaned):
            embedded.extend(_parse_payload_sequence(candidate))
        if embedded:
            return embedded, ""
        return None, text

    def return_text_from_tool_calls(self, tool_calls: list[dict[str, Any]], raw_text: str | None = None) -> str:
        payloads: list[dict[str, Any]] = []
        for tc in tool_calls or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, RecursionError):
                    pass
            payloads.append({"name": name, "parameters": args})
        if not payloads:
            return raw_text or ""
        if len(payloads) == 1:
            # BFCL's Llama 3.1 local handler historically eval()s this path.
            return repr(payloads[0])
        return ";".join(_json_dumps_compact(payload) for payload in payloads)


def resolve_bfcl_adapter(adapter: str, model_name: str | None = None, model_alias: str | None = None) -> BFCLAdapter:
    value = (adapter or "auto").strip().lower()
    if value not in BFCL_ADAPTER_NAMES:
        raise ValueError(f"Unsupported bfcl_adapter={adapter!r}; expected one of {BFCL_ADAPTER_NAMES}.")
    if value == "auto":
        joined = " ".join(x for x in (model_name, model_alias) if x).lower()
        if "qwen" in joined and "qwen3" in joined:
            value = "qwen3_fc"
        elif "llama" in joined:
            value = "llama31_fc_legacy"
        else:
            value = "raw"
    if value == "qwen3_fc":
        return Qwen3FCAdapter()
    if value == "llama31_fc_legacy":
        return Llama31FCLegacyAdapter()
    return RawBFCLAdapter()
