"""
BFCL web-search evaluation with phase- and layer-type-dependent MX quantization.

Serves an MX-quantized model through a lightweight OpenAI-compatible HTTP
server (backed by HuggingFace ``generate``), then drives the standard BFCL
CLI against it. Activation precision is set independently for each
(phase, layer_type) pair via the prefill/decode × attn/FFN flags.

BFCL is a two-step flow:

1. ``bfcl generate`` calls the local server to produce model responses.
2. ``bfcl evaluate`` scores those responses (no model needed).

This script orchestrates both steps automatically and exposes the local
server on ``server_host:server_port``.

Requires the ``bfcl`` extra:

    uv sync --extra bfcl

Example:

    python -m quant_eval.cli.eval_phase_bfcl \\
        --model_name Qwen/Qwen2.5-1.5B \\
        --quant_config quant_eval/configs/llama_mxint4.toml \\
        --prefill_attn_width 4 --prefill_ffn_width 4 \\
        --decode_attn_width  8 --decode_ffn_width  8 \\
        --bfcl_test_categories web_search_base \\
        --limit 50
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Sequence, Union

import torch
import transformers

from quant_eval.utils import (
    get_logger,
    set_logging_verbosity,
    setup_model,
    move_to_gpu,
    create_experiment_log_dir,
    save_args,
    save_results,
)
from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.quantize import load_quant_config

from fastapi import FastAPI, Request

import httpx
from bs4 import BeautifulSoup
import markdownify, random


logger = get_logger(__name__)
set_logging_verbosity("debug")

# ── Default BFCL V4 web-search categories ─────────────────────────────────────
BFCL_WEB_SEARCH_CATEGORIES = ("web_search_base", "web_search_no_snippet")
BFCL_TOOL_MODES = ("auto", "return", "execute")

# ── OpenAI-compatible server defaults ─────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8915

def _execute_tool(tool_call: dict) -> str:
    """Execute duckduckgo_search or fetch_url_content and return result as string."""
    import random
    import httpx
    from bs4 import BeautifulSoup

    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to parse tool arguments"})

    print(f"DEBUG: Executing tool '{name}' with arguments: {args}")

    # ── duckduckgo_search ──────────────────────────────────────────
    if name in ("duckduckgo_search", "search_engine_query"):
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                print(f"DEBUG: Performing DuckDuckGo search with keywords='{args.get('keywords', '')}' and region='{args.get('region', 'wt-wt')}'")
                results = list(ddgs.text(
                    args.get("keywords", args.get("query", "")),  # positional, not keyword
                    region=args.get("region", "wt-wt"),
                    max_results=args.get("max_results", 10),
                ))
                print(f"DEBUG: DuckDuckGo search returned {len(results)} results")
                print(f"DEBUG: Sample result: {results[0] if results else 'No results'}")
            # Normalise keys to what BFCL expects: title, href, body
            results = [{"title": r["title"], "url": r["href"]} for r in results]
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": f"duckduckgo_search failed: {e}"})

    # ── fetch_url_content ──────────────────────────────────────────
    elif name == "fetch_url_content":
        url  = args.get("url", "")
        mode = args.get("mode", "raw")

        # Simulate probabilistic failures (matches BFCL benchmark behaviour)
        error_templates = [
            f"503 Server Error: Service Unavailable for url: {url}",
            f"429 Client Error: Too Many Requests for url: {url}",
            f"403 Client Error: Forbidden for url: {url}",
            f"HTTPSConnectionPool(host='{url}', port=443): Max retries exceeded",
            f"HTTPSConnectionPool(host='{url}', port=443): Read timed out. (read timeout=5)",
        ]
        if random.random() < 0.1:
            return json.dumps({"error": random.choice(error_templates)})

        try:
            resp = httpx.get(url, timeout=10, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            html = resp.text

            if mode == "raw":
                return html[:20000]  # cap to avoid huge prompts

            elif mode == "markdown":
                try:
                    import markdownify
                    return markdownify.markdownify(html)[:20000]
                except ImportError:
                    # fallback to truncate if markdownify not installed
                    mode = "truncate"

            if mode == "truncate":
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "head"]):
                    tag.decompose()
                text = " ".join(soup.get_text().split())
                return text[:8000]

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Unknown tool ───────────────────────────────────────────────
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})



# ══════════════════════════════════════════════════════════════════════════════
#  Minimal OpenAI-compatible chat-completion server
# ══════════════════════════════════════════════════════════════════════════════

def _build_server_app(model, tokenizer, device: str, tool_mode: str = "execute"):
    """
    Return a FastAPI application that exposes POST /v1/chat/completions.

    The quantized model (with PhaseLayerAutoSwitch already enabled) is called
    directly — no additional process boundary.  Tool/function calls are passed
    through transparently so that BFCL can exercise them.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError(
            "fastapi and uvicorn are required to serve the model locally.\n"
            "Install them with:  pip install fastapi uvicorn"
        ) from exc

    if tool_mode not in ("return", "execute"):
        raise ValueError(f"tool_mode must be 'return' or 'execute', got {tool_mode!r}.")

    app = FastAPI(title="quant-model-server")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        print("🚀 REQUEST HIT")
        body = await request.json()
        messages     = list(body.get("messages", []))  # make a mutable copy
        tools        = body.get("tools", None)
        temperature  = body.get("temperature", 0.0)
        max_new_toks = body.get("max_tokens", 1024)

        MAX_TURNS = 15  # prevent infinite agentic loops

        for turn in range(MAX_TURNS):
            print(f"🔄 Agentic turn {turn + 1}/{MAX_TURNS}, messages so far: {len(messages)}")

            # ── Build prompt ───────────────────────────────────────────
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt_ids = tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(device)
                except Exception:
                    prompt_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(device)
            else:
                text = "\n".join(
                    f"{m.get('role','user').upper()}: {m.get('content','')}"
                    for m in messages
                ) + "\nASSISTANT:"
                prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            # ── Inference ──────────────────────────────────────────────
            with torch.no_grad():
                attention_mask = torch.ones_like(prompt_ids)
                output_ids = model.generate(
                    prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_toks,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            print(f"DEBUG: output_ids shape: {output_ids.shape}")
            generated_ids = output_ids[0][prompt_ids.shape[-1]:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            print(f"DEBUG: raw_text: {raw_text[:200]}")

            logger.debug("Prompt tokens: %s", raw_text[:200])
            # ── Parse tool calls ───────────────────────────────────────
            tool_calls, content = _parse_tool_calls(raw_text)
            content = content or ""

            # ── No tool calls → final answer, return immediately ───────
            if not tool_calls:
                print(f"✅ Final answer reached at turn {turn + 1}")
                message = {"role": "assistant", "content": content}
                response = {
                    "id":      f"chatcmpl-{int(time.time()*1000)}",
                    "object":  "chat.completion",
                    "created": int(time.time()),
                    "model":   tokenizer.name_or_path,
                    "choices": [{
                        "index":         0,
                        "message":       message,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens":     int(prompt_ids.shape[-1]),
                        "completion_tokens": int(generated_ids.shape[-1]),
                        "total_tokens":      int(prompt_ids.shape[-1] + generated_ids.shape[-1]),
                    },
                }
                return JSONResponse(content=response)

            # BFCL non-live categories (for example `multiple`) expect the
            # model's function call itself as the answer. Executing arbitrary
            # benchmark functions here would append synthetic tool results and
            # force a second generation, changing the response that BFCL scores.
            if tool_mode == "return":
                print(f"↩️ Returning tool calls at turn {turn + 1}: {[tc['function']['name'] for tc in tool_calls]}")
                message = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                response = {
                    "id":      f"chatcmpl-{int(time.time()*1000)}",
                    "object":  "chat.completion",
                    "created": int(time.time()),
                    "model":   tokenizer.name_or_path,
                    "choices": [{
                        "index":         0,
                        "message":       message,
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {
                        "prompt_tokens":     int(prompt_ids.shape[-1]),
                        "completion_tokens": int(generated_ids.shape[-1]),
                        "total_tokens":      int(prompt_ids.shape[-1] + generated_ids.shape[-1]),
                    },
                }
                return JSONResponse(content=response)

            # ── Tool calls found → execute them and loop back ──────────
            print(f"🔧 Tool calls at turn {turn + 1}: {[tc['function']['name'] for tc in tool_calls]}")

            # Append the assistant's tool-call message to history
            messages.append({
                "role":       "assistant",
                "content":    content,
                "tool_calls": tool_calls,
            })

            # Execute each tool and append results
            for tc in tool_calls:
                tool_name   = tc["function"]["name"]
                logger.info("Executing tool '%s' with arguments: %s", tool_name, tc["function"]["arguments"])
                tool_result = _execute_tool(tc)
                print(f"🔍 Tool '{tool_name}' result preview: {str(tool_result)[:200]}")
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "name":         tool_name,
                    "content":      tool_result,
                })

        # ── Exceeded MAX_TURNS → return whatever we have ───────────────
        print(f"⚠️ MAX_TURNS ({MAX_TURNS}) exceeded, returning last content")
        message = {"role": "assistant", "content": content}
        response = {
            "id":      f"chatcmpl-{int(time.time()*1000)}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   tokenizer.name_or_path,
            "choices": [{
                "index":         0,
                "message":       message,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     int(prompt_ids.shape[-1]),
                "completion_tokens": int(generated_ids.shape[-1]),
                "total_tokens":      int(prompt_ids.shape[-1] + generated_ids.shape[-1]),
            },
        }
        return JSONResponse(content=response)

    @app.post("/v1/completions")
    async def completions(request: Request):
        print("🚀 /v1/completions REQUEST HIT")
        body = await request.json()

        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        tools        = body.get("tools", None)
        temperature  = body.get("temperature", 0.0)
        max_new_toks = body.get("max_tokens", 1024)

        MAX_TURNS = 15  # prevent infinite agentic loops

        total_prompt_tokens     = 0
        total_completion_tokens = 0

        for turn in range(MAX_TURNS):
            print(f"🔄 Agentic turn {turn + 1}/{MAX_TURNS}")

            # Tokenize the current prompt string directly (no chat template).
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                attention_mask = torch.ones_like(prompt_ids)
                output_ids = model.generate(
                    prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_toks,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][prompt_ids.shape[-1]:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            print(raw_text)

            total_prompt_tokens     += int(prompt_ids.shape[-1])
            total_completion_tokens += int(generated_ids.shape[-1])

            # ── Parse tool calls ───────────────────────────────────────
            tool_calls, content = _parse_tool_calls(raw_text)
            content = content or ""
            print(tool_calls)

            # ── No tool calls → final answer, return immediately ───────
            if not tool_calls:
                print(f"✅ Final answer reached at turn {turn + 1}")
                response_text = raw_text
                if tool_mode == "return":
                    response_text = _normalize_bfcl_return_text(raw_text)
                response = {
                    "id":      f"cmpl-{int(time.time()*1000)}",
                    "object":  "text_completion",
                    "created": int(time.time()),
                    "model":   tokenizer.name_or_path,
                    "choices": [
                        {
                            "index":         0,
                            "text":          response_text,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens":     total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens":      total_prompt_tokens + total_completion_tokens,
                    },
                }
                return JSONResponse(content=response)

            # BFCL non-live categories should receive the model's function
            # call response. Llama 3.1 FC emits special tokens around JSON
            # (<|python_tag|>...<|eom_id|>); strip those before BFCL evaluates.
            if tool_mode == "return":
                print(f"↩️ Returning completion tool calls at turn {turn + 1}: {[tc['function']['name'] for tc in tool_calls]}")
                response_text = _normalize_bfcl_return_text(raw_text)
                response = {
                    "id":      f"cmpl-{int(time.time()*1000)}",
                    "object":  "text_completion",
                    "created": int(time.time()),
                    "model":   tokenizer.name_or_path,
                    "choices": [
                        {
                            "index":         0,
                            "text":          response_text,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens":     total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens":      total_prompt_tokens + total_completion_tokens,
                    },
                }
                return JSONResponse(content=response)

            # ── Tool calls found → execute them and append to prompt ───
            # Append the model's output (with tool calls) to the running prompt.
            prompt = prompt + raw_text

            # Execute each tool and append results in a structured block.
            for tc in tool_calls:
                tool_name   = tc["function"]["name"]
                logger.info("Executing tool '%s' with arguments: %s", tool_name, tc["function"]["arguments"])
                tool_result = _execute_tool(tc)
                print(f"🔍 Tool '{tool_name}' result preview: {str(tool_result)[:200]}")
                prompt += (
                    f"\n<tool_response>\n"
                    f"{json.dumps({'tool_call_id': tc['id'], 'name': tool_name, 'content': tool_result})}\n"
                    f"</tool_response>\n"
                )

        # ── Exceeded MAX_TURNS → return whatever we have ───────────────
        print(f"⚠️ MAX_TURNS ({MAX_TURNS}) exceeded, returning last output")
        response = {
            "id":      f"cmpl-{int(time.time()*1000)}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   tokenizer.name_or_path,
            "choices": [
                {
                    "index":         0,
                    "text":          raw_text,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens":     total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens":      total_prompt_tokens + total_completion_tokens,
            },
        }
        return JSONResponse(content=response)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse(content={
            "object": "list",
            "data": [{"id": tokenizer.name_or_path, "object": "model"}],
        })

    return app



def _normalize_bfcl_return_text(text: str) -> str:
    """Normalize local Llama FC text before BFCL AST decoding.

    BFCL's Llama 3.1 handler expects the generated payload itself, e.g.
    ``{"name": "fn", "parameters": {...}}`` or a semicolon-separated list of
    those JSON objects. The HF Llama template wraps that payload with
    ``<|python_tag|>`` and generation often includes ``<|eom_id|>``. Returning
    those tokens verbatim makes BFCL's AST decoder fail before it can score the
    actual call. This helper only removes transport/template wrappers; it does
    not repair wrong function names or wrong argument values.
    """
    if not text:
        return text

    normalized = text.strip()

    # If the model generated an assistant header, score only the assistant body.
    header = "<|start_header_id|>assistant<|end_header_id|>"
    if header in normalized:
        normalized = normalized.split(header, 1)[1].strip()

    if "<|python_tag|>" in normalized:
        normalized = normalized.split("<|python_tag|>", 1)[1].strip()

    # Stop at Llama special tokens that are not part of the callable payload.
    for stop in ("<|eom_id|>", "<|eot_id|>", "<|end_of_text|>"):
        if stop in normalized:
            normalized = normalized.split(stop, 1)[0].strip()

    # Remove common markdown wrappers without changing the model payload.
    normalized = normalized.strip().strip("`").strip()
    if normalized.lower().startswith("json\n"):
        normalized = normalized[5:].strip()

    if normalized != text.strip():
        print(f"DEBUG: Normalized BFCL return text: {normalized[:300]}")
    return normalized or text


def _loads_tool_payload(payload: str):
    """Load JSON or Python-literal tool payloads emitted by Llama."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        import ast
        return ast.literal_eval(payload)


def _coerce_tool_arguments(parsed: dict) -> dict:
    """Return OpenAI-style tool arguments from common FC JSON shapes."""
    if not isinstance(parsed, dict):
        return {}
    if "arguments" in parsed:
        args = parsed["arguments"]
    elif "parameters" in parsed:
        args = parsed["parameters"]
    else:
        args = {k: v for k, v in parsed.items() if k != "name"}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return args if isinstance(args, dict) else {"value": args}


def _append_tool_call(tool_calls: list, parsed: dict) -> None:
    """Append one parsed call in OpenAI tool_calls shape if it has a name."""
    if not isinstance(parsed, dict):
        return
    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        return
    tool_calls.append({
        "id":       f"call_{len(tool_calls)}",
        "type":     "function",
        "function": {
            "name":      name,
            "arguments": json.dumps(_coerce_tool_arguments(parsed)),
        },
    })


def _parse_tool_calls(text: str) -> tuple[list | None, str]:
    """
    Attempt to extract OpenAI-format tool_calls from model output.

    Handles two common patterns emitted by instruction-tuned models:
      1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
      2. ```json\\n{"name": "...", "arguments": {...}}\\n```

    Returns (tool_calls_list, leftover_text).
    If no tool calls are found, returns (None, original_text).
    """
    import re

    patterns = [
        r"<tool_call>(.*?)</tool_call>",
        r"```(?:json)?\s*(\{.*?\})\s*```",
    ]

    for pat in patterns:
        matches = re.findall(pat, text, re.DOTALL)
        if matches:
            print(f"DEBUG: Found {len(matches)} tool call(s) with pattern: {pat}")
            tool_calls = []
            for m in matches:
                try:
                    parsed = _loads_tool_payload(m.strip())
                    _append_tool_call(tool_calls, parsed)
                    print(f"DEBUG: Parsed tool call {len(tool_calls) - 1}: {tool_calls[-1]}")
                except (json.JSONDecodeError, SyntaxError, ValueError, IndexError):
                    print(f"WARNING: Failed to parse tool call JSON: {m[:200]}")
                    continue
            if tool_calls:
                leftover = re.sub(pat, "", text, flags=re.DOTALL).strip()
                return tool_calls, leftover

    # Llama 3.1 function-calling output uses <|python_tag|> followed by one
    # JSON object, a list of JSON objects, or semicolon-separated JSON objects.
    llama_payload = _normalize_bfcl_return_text(text)
    if llama_payload != text.strip():
        tool_calls = []
        payloads = [p.strip() for p in llama_payload.split(";") if p.strip()]
        try:
            if len(payloads) > 1:
                for payload in payloads:
                    _append_tool_call(tool_calls, _loads_tool_payload(payload))
            else:
                parsed = _loads_tool_payload(llama_payload)
                if isinstance(parsed, list):
                    for item in parsed:
                        _append_tool_call(tool_calls, item)
                else:
                    _append_tool_call(tool_calls, parsed)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            print(f"WARNING: Failed to parse Llama tool-call JSON: {llama_payload[:200]}")
        if tool_calls:
            return tool_calls, ""

    return None, text


def _start_server(app, host: str, port: int) -> threading.Thread:
    """Launch uvicorn in a daemon thread; block until the server is ready."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to serve the model.\n"
            "Install it with:  pip install uvicorn"
        ) from exc

    config = uvicorn.Config(app, host=host, port=port, log_level="debug")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to accept connections (up to 60 s).
    import httpx
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            httpx.get(f"http://{host}:{port}/v1/models", timeout=1)
            logger.info("Server ready at http://%s:%d", host, port)
            return thread
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(f"Server at http://{host}:{port} did not start in time.")


# ══════════════════════════════════════════════════════════════════════════════
#  BFCL CLI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bfcl_model_name(model_name: str, model_alias: str | None = None) -> str:
    """
    Map a HuggingFace model ID to the BFCL result-file name.

    BFCL replaces '/' with '__' internally; we follow the same convention and
    append '-FC' to signal native function-calling support.
    """
    if model_alias:
        return model_alias

    normalized = model_name.replace("/", "__")
    if normalized.endswith("-FC"):
        return normalized
    return normalized + "-FC"


def _normalize_bfcl_categories(
    categories: Union[list[str], str, None],
) -> list[str]:
    """Normalize category input to a non-empty list of strings."""
    if categories is None:
        return list(BFCL_WEB_SEARCH_CATEGORIES)

    if isinstance(categories, str):
        raw = categories.strip()
        if not raw:
            raise ValueError("bfcl_test_categories cannot be empty.")

        # Accept shell-friendly JSON lists and comma-separated strings.
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON for bfcl_test_categories: {raw}"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("bfcl_test_categories JSON must be a list.")
            categories = parsed
        else:
            categories = [c.strip() for c in raw.split(",") if c.strip()]

    if not isinstance(categories, list):
        raise ValueError(
            "bfcl_test_categories must be a list, JSON list string, or comma-separated string."
        )

    normalized = []
    for cat in categories:
        if not isinstance(cat, str):
            raise ValueError("bfcl_test_categories entries must be strings.")
        cat = cat.strip()
        if cat:
            normalized.append(cat)

    if not normalized:
        raise ValueError("bfcl_test_categories resolved to an empty list.")
    return normalized


def _resolve_bfcl_tool_mode(categories: Sequence[str], mode: str) -> str:
    """Resolve auto tool behavior for BFCL live vs non-live categories."""
    mode = mode.lower()
    if mode not in BFCL_TOOL_MODES:
        raise ValueError(
            f"bfcl_tool_mode must be one of {BFCL_TOOL_MODES}, got {mode!r}."
        )
    if mode != "auto":
        return mode
    # Only BFCL web-search categories need live tool execution. Standard
    # function-calling categories such as `multiple` are scored on the function
    # call returned by the model, so the local server must not execute tools.
    return "execute" if all(cat.startswith("web_search") for cat in categories) else "return"


def _normalize_gptq_dataset(dataset: str) -> str:
    """Accept either GPTQ loader specs or plain local file paths."""
    dataset = dataset.strip()
    if not dataset:
        raise ValueError("gptq_dataset cannot be empty.")
    if dataset.startswith(("file:", "hf:", "jsonl:", "txt:", "lm_eval:")):
        return dataset
    return "file:" + dataset


def _inject_gptq_config(
    pass_args: dict,
    *,
    model_name: str,
    device_id: str,
    dataset: str | None,
    nsamples: int,
    seqlen: int,
    fmt: str,
    weight_width: int,
    weight_block_size: int,
    cali_batch_size: int,
    max_layers: int | None,
) -> dict | None:
    """Merge CLI GPTQ options into pass_args; CLI values take precedence."""
    if dataset is None:
        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id
            return pass_args["gptq"]
        return None

    if nsamples <= 0:
        raise ValueError("gptq_nsamples must be positive.")
    if seqlen <= 0:
        raise ValueError("gptq_seqlen must be positive.")
    if cali_batch_size <= 0:
        raise ValueError("gptq_cali_batch_size must be positive.")

    gptq_cfg = dict(pass_args.get("gptq", {}))
    gptq_cfg.update({
        "model_name": model_name,
        "device": device_id,
        "dataset": _normalize_gptq_dataset(dataset),
        "nsamples": nsamples,
        "seqlen": seqlen,
        "format": fmt,
        "weight_config": {
            **dict(gptq_cfg.get("weight_config", {})),
            "weight_width": weight_width,
            "weight_block_size": weight_block_size,
        },
        "cali_batch_size": cali_batch_size,
    })
    if max_layers is not None:
        if max_layers <= 0:
            raise ValueError("gptq_max_layers must be positive when set.")
        gptq_cfg["max_layers"] = max_layers
    else:
        gptq_cfg.pop("max_layers", None)

    pass_args["gptq"] = gptq_cfg
    return gptq_cfg



def _run_bfcl_generate(
    model_name: str,
    test_categories: Sequence[str],
    host: str,
    port: int,
    result_dir: Path,
    num_threads: int,
    limit: int | None,
    model_alias: str | None = None,
) -> int:
    """Call ``bfcl generate`` against the local server; return the exit code."""
    bfcl_name = _bfcl_model_name(model_name, model_alias=model_alias)
    cmd = [
        "bfcl", "generate",
        "--model",         bfcl_name,
        "--test-category", *test_categories,
        "--skip-server-setup",
        "--result-dir",    str(result_dir),
        "--num-threads",   str(num_threads),
    ]
    env = os.environ.copy()
    env["LOCAL_SERVER_ENDPOINT"] = host
    env["LOCAL_SERVER_PORT"]     = str(port)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when set.")
        # The PyPI BFCL CLI does not expose a --limit flag. Use its official
        # run-id mechanism instead: write the first N deterministic test IDs
        # to BFCL_PROJECT_ROOT/test_case_ids_to_generate.json and pass
        # --run-ids. This keeps the quick-smoke path out of bfcl-eval source.
        project_root = Path(env.get("BFCL_PROJECT_ROOT", result_dir.parent / "bfcl_project"))
        env["BFCL_PROJECT_ROOT"] = str(project_root)
        project_root.mkdir(parents=True, exist_ok=True)
        ids_path = project_root / "test_case_ids_to_generate.json"
        ids = {cat: [f"{cat}_{idx}" for idx in range(limit)] for cat in test_categories}
        ids_path.write_text(json.dumps(ids, indent=2) + "\n")
        cmd.append("--run-ids")

    print(cmd)
    print(host)
    print(port)

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _load_bfcl_score_file(path: Path) -> object:
    """Load BFCL score output across package versions.

    Older/local BFCL builds emit a single JSON object in ``*_score.json``.
    The PyPI BFCL CLI can emit newline-delimited JSON records in the same
    ``.json`` file. Treat both as valid because this parser is only used for
    reporting; BFCL itself has already finished evaluation by this point.
    """
    text = path.read_text().strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def _run_bfcl_evaluate(
    model_name: str,
    test_categories: list[str],
    result_dir: Path,
    score_dir: Path,
    model_alias: str | None = None,
    partial_eval: bool = False,
) -> tuple[int, dict]:
    """Call ``bfcl evaluate``; return (exit_code, parsed_scores)."""
    bfcl_name = _bfcl_model_name(model_name, model_alias=model_alias)
    cmd = [
        "bfcl", "evaluate",
        "--model",         bfcl_name,
        "--test-category", *test_categories,
        "--result-dir",    str(result_dir),
        "--score-dir",     str(score_dir),
    ]

    if partial_eval:
        cmd.append("--partial-eval")

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # ── Collect per-category JSON scores ──────────────────────────────────
    scores: dict = {}
    per_cat: dict = {}
    for cat in test_categories:
        candidates = sorted(score_dir.rglob(f"BFCL_*_{cat}_score.json"))
        for json_path in candidates:
            if json_path.exists():
                per_cat[cat] = _load_bfcl_score_file(json_path)
                break

    if per_cat:
        scores["per_category"] = per_cat

    # ── Pull summary row from data_overall.csv if present ─────────────────
    csv_path = score_dir / "data_overall.csv"
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("Model") == bfcl_name:
                    scores.update({k: v for k, v in row.items() if k != "Model"})
                    break

    if proc.stdout:
        logger.info("bfcl evaluate stdout:\n%s", proc.stdout)
    if proc.stderr:
        logger.warning("bfcl evaluate stderr:\n%s", proc.stderr)

    return proc.returncode, scores


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main(
    model_name:  str = "Qwen/Qwen3-8B-FC",
    device_id:   str = "cuda:0",
    dtype:       str = "bfloat16",
    quant_config: str = "quant_eval/configs/llama_mxint4.toml",
    model_parallel: bool = False,
    # ── BFCL settings ──────────────────────────────────────────────────────
    bfcl_test_categories: Union[list[str], str, None] = None,
    bfcl_model_alias:     str | None = None,
    bfcl_num_threads:     int   = 1,
    server_host:          str   = DEFAULT_HOST,
    server_port:          int   = DEFAULT_PORT,
    bfcl_tool_mode:       str   = "auto",
    # ── GPTQ calibration/weight quantization ───────────────────────────────
    gptq_dataset:          str | None = None,
    gptq_nsamples:         int = 32,
    gptq_seqlen:           int = 1024,
    gptq_format:           str = "mxint",
    gptq_weight_width:     int = 8,
    gptq_weight_block_size:int = 32,
    gptq_cali_batch_size:  int = 1,
    gptq_max_layers:       int | None = None,
    # ── Activation precision: prefill ──────────────────────────────────────
    prefill_attn_width:      int = 4,
    prefill_ffn_width:       int = 4,
    prefill_attn_block_size: int = 32,
    prefill_ffn_block_size:  int = 32,
    # ── Activation precision: decode ───────────────────────────────────────
    decode_attn_width:       int = 8,
    decode_ffn_width:        int = 8,
    decode_attn_block_size:  int = 32,
    decode_ffn_block_size:   int = 32,
    decode_weight_mode:      str = "quantized",
    # ── Optional keyword overrides ─────────────────────────────────────────
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords:  Union[list[str], None] = None,
    limit: Union[int, None] = None,
    log_dir: Union[str, None] = None,
):
    """
    Run BFCL web-search evaluation with phase- and layer-type-dependent
    activation precision.

    Spawns a local OpenAI-compatible HTTP server backed by HF ``generate``
    so that the unmodified ``bfcl generate`` CLI can drive inference, then
    runs ``bfcl evaluate`` to score responses.

    Args:
        model_name: HuggingFace model ID. For function-calling, must be an
            instruction-tuned model with a function-call template
            (e.g. ``Qwen/Qwen3-8B-FC``).
        device_id: CUDA device string.
        dtype: Model dtype — ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        quant_config: Path to a TOML quantization recipe.
        model_parallel: Distribute across GPUs with ``device_map="auto"``.
        bfcl_test_categories: BFCL category names to evaluate (e.g.
            ``["web_search_base", "web_search_no_snippet"]``). ``None`` uses
            the default web-search category set. Also accepts JSON-list string
            or comma-separated categories.
        bfcl_model_alias: BFCL result-file model alias. If set, this exact
            value is used for ``bfcl generate/evaluate --model``.
        bfcl_num_threads: Parallel inference threads for ``bfcl generate``.
        server_host: Host for the local OpenAI-compatible server.
        server_port: Port for the local OpenAI-compatible server.
        bfcl_tool_mode: ``"auto"`` executes tools only for web-search categories;
            ``"return"`` returns model tool calls directly for BFCL function-call
            categories such as ``multiple``; ``"execute"`` preserves agentic
            web-search behavior.
        gptq_dataset: Optional GPTQ calibration dataset. Plain paths are treated
            as ``file:<path>`` for the GPTQ loader. When set, CLI GPTQ values
            override any ``[gptq]`` block in ``quant_config``.
        gptq_nsamples: Number of GPTQ calibration samples.
        gptq_seqlen: GPTQ calibration sequence length.
        gptq_format: GPTQ target format, for example ``"mxint"``.
        gptq_weight_width: GPTQ quantized weight width.
        gptq_weight_block_size: GPTQ quantized weight block size.
        gptq_cali_batch_size: GPTQ calibration batch size.
        gptq_max_layers: Optional layer cap for quick smoke tests.
        prefill_attn_width: Activation bit-width for attention during prefill.
        prefill_ffn_width: Activation bit-width for FFN during prefill.
        prefill_attn_block_size: MX block size for attention during prefill.
        prefill_ffn_block_size: MX block size for FFN during prefill.
        decode_attn_width: Activation bit-width for attention during decode.
        decode_ffn_width: Activation bit-width for FFN during decode.
        decode_attn_block_size: MX block size for attention during decode.
        decode_ffn_block_size: MX block size for FFN during decode.
        decode_weight_mode: ``"quantized"`` keeps current decode behavior;
            ``"fp"`` loads original checkpoint weights for decode Linear
            layers and bypasses Linear activation quantization.
        attn_keywords: Module-name substrings that identify attention blocks.
            ``None`` uses the built-in defaults.
        ffn_keywords: Module-name substrings that identify FFN blocks. ``None``
            uses the built-in defaults.
        limit: Cap the number of samples per category. ``None`` = full dataset.
        log_dir: Directory for ``args.json`` and ``results.json``.

    Returns:
        BFCL evaluation summary — per-category scores plus aggregate metrics,
        with ``phase_layer_configs`` recording the resolved (phase, layer)
        precision table.
    """
    bfcl_test_categories = _normalize_bfcl_categories(bfcl_test_categories)
    bfcl_tool_mode = _resolve_bfcl_tool_mode(bfcl_test_categories, bfcl_tool_mode)
    decode_weight_mode = decode_weight_mode.lower()
    if decode_weight_mode not in ("quantized", "fp"):
        raise ValueError(
            "decode_weight_mode must be 'quantized' or 'fp', "
            f"got {decode_weight_mode!r}."
        )

    # ------------------------------------------------------------------
    # Build the nested phase × layer config
    # ------------------------------------------------------------------
    decode_weight_policy = {}
    if decode_weight_mode == "fp":
        decode_weight_policy = {"weight_mode": "fp", "bypass": True}

    phase_configs = {
        "prefill": {
            "attn": {
                "data_in_width":      prefill_attn_width,
                "data_in_block_size": prefill_attn_block_size,
            },
            "ffn": {
                "data_in_width":      prefill_ffn_width,
                "data_in_block_size": prefill_ffn_block_size,
            },
        },
        "decode": {
            "attn": {
                "data_in_width":      decode_attn_width,
                "data_in_block_size": decode_attn_block_size,
                **decode_weight_policy,
            },
            "ffn": {
                "data_in_width":      decode_ffn_width,
                "data_in_block_size": decode_ffn_block_size,
                **decode_weight_policy,
            },
        },
    }

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    _pa = f"MXInt{prefill_attn_width}(bs={prefill_attn_block_size})"
    _pf = f"MXInt{prefill_ffn_width}(bs={prefill_ffn_block_size})"
    _da = f"MXInt{decode_attn_width}(bs={decode_attn_block_size}, W={decode_weight_mode})"
    _df = f"MXInt{decode_ffn_width}(bs={decode_ffn_block_size}, W={decode_weight_mode})"

    print("=" * 64)
    print("BFCL Web Search — Phase × Layer-Type Disaggregated Quantization")
    print("=" * 64)
    print(f"  Model      : {model_name}")
    if bfcl_model_alias:
        print(f"  BFCL Alias : {bfcl_model_alias}")
    print(f"  Categories : {bfcl_test_categories}")
    print(f"  Tool mode  : {bfcl_tool_mode}")
    print(f"  Weights    : {quant_config}")
    print(f"  Decode W   : {decode_weight_mode}")
    if gptq_dataset:
        print(f"  GPTQ       : dataset={gptq_dataset}, nsamples={gptq_nsamples}, seqlen={gptq_seqlen}, max_layers={gptq_max_layers}")
    print(f"  Server     : http://{server_host}:{server_port}")
    print()
    print(f"  {'':10s}  {'attn':>24s}  {'ffn':>24s}")
    print(f"  {'prefill':10s}  {_pa:>24s}  {_pf:>24s}")
    print(f"  {'decode':10s}  {_da:>24s}  {_df:>24s}")
    print("=" * 64)
    logger.info("Model Parallel", model_parallel)

    # ------------------------------------------------------------------
    # Resolve output directories (persistent if log_dir given)
    # ------------------------------------------------------------------
    _tmpdir_ctx = tempfile.TemporaryDirectory()
    _tmpdir     = Path(_tmpdir_ctx.name)

    result_dir = _tmpdir / "bfcl_results"
    score_dir  = _tmpdir / "bfcl_scores"
    result_dir.mkdir(parents=True)
    score_dir.mkdir(parents=True)

    if log_dir:
        log_dir    = create_experiment_log_dir(log_dir)
        result_dir = log_dir / "bfcl_results"
        score_dir  = log_dir / "bfcl_scores"
        result_dir.mkdir(parents=True)
        score_dir.mkdir(parents=True)
        save_args(log_dir, locals().copy())
        import shutil
        shutil.copy(quant_config, log_dir / "quant_config.toml")

    transformers.set_seed(0)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    tokenizer, model = setup_model(
        model_name,
        model_parallel,
        dtype=torch_dtype,
        device=device_id if not model_parallel else None,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Weight quantization
    # ------------------------------------------------------------------
    from chop.passes.module.transforms import quantize_module_transform_pass

    pass_args = load_quant_config(quant_config)
    resolved_gptq_config = _inject_gptq_config(
        pass_args,
        model_name=model_name,
        device_id=device_id,
        dataset=gptq_dataset,
        nsamples=gptq_nsamples,
        seqlen=gptq_seqlen,
        fmt=gptq_format,
        weight_width=gptq_weight_width,
        weight_block_size=gptq_weight_block_size,
        cali_batch_size=gptq_cali_batch_size,
        max_layers=gptq_max_layers,
    )
    if resolved_gptq_config:
        logger.info(
            "GPTQ enabled: dataset=%s nsamples=%s seqlen=%s format=%s max_layers=%s",
            resolved_gptq_config.get("dataset"),
            resolved_gptq_config.get("nsamples"),
            resolved_gptq_config.get("seqlen"),
            resolved_gptq_config.get("format"),
            resolved_gptq_config.get("max_layers"),
        )

    n_linear = sum(
        1 for _, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    )
    logger.info("Quantizing %d linear layers...", n_linear)
    t0 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    logger.info("Quantization complete in %.1fs", time.time() - t0)

    if model_parallel:
        model = move_to_gpu(model, model_parallel)
    else:
        model.to(device_id)

    # ------------------------------------------------------------------
    # Enable disaggregated quantization hook
    # ------------------------------------------------------------------
    switch_kwargs = {}
    if attn_keywords:
        switch_kwargs["attn_keywords"] = tuple(attn_keywords)
    if ffn_keywords:
        switch_kwargs["ffn_keywords"] = tuple(ffn_keywords)
    if decode_weight_mode == "fp":
        switch_kwargs["model_name"] = model_name

    switch = PhaseLayerAutoSwitch(model, phase_configs, **switch_kwargs)
    switch.enable()
    logger.info("\n%s", switch.summary())

    # ------------------------------------------------------------------
    # Start the OpenAI-compatible server (hook fires on every request)
    # ------------------------------------------------------------------
    device_str = device_id if not model_parallel else "cuda"
    app = _build_server_app(model, tokenizer, device_str, tool_mode=bfcl_tool_mode)
    _start_server(app, server_host, server_port)


    # ------------------------------------------------------------------
    # Step 1: bfcl generate  (calls the local server)
    # ------------------------------------------------------------------
    print("\n[1/2] Generating BFCL responses via local server...")
    gen_rc = _run_bfcl_generate(
        model_name      = model_name,
        test_categories = bfcl_test_categories,
        host            = server_host,
        port            = server_port,
        result_dir      = result_dir,
        num_threads     = bfcl_num_threads,
        limit           = limit,
        model_alias     = bfcl_model_alias,
    )
    if gen_rc != 0:
        logger.error("bfcl generate exited with code %d", gen_rc)

#     # ------------------------------------------------------------------
#     # Step 2: bfcl evaluate  (pure scoring, no model needed)
#     # ------------------------------------------------------------------
    print("[2/2] Evaluating BFCL responses...")
    eval_rc, scores = _run_bfcl_evaluate(
        model_name      = model_name,
        test_categories = bfcl_test_categories,
        result_dir      = result_dir,
        score_dir       = score_dir,
        model_alias     = bfcl_model_alias,
        partial_eval    = limit is not None,
    )

    switch.disable()

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Results:")
    print("=" * 64)
    print(f"\n  {'':10s}  {'attn':>24s}  {'ffn':>24s}")
    print(f"  {'prefill':10s}  {_pa:>24s}  {_pf:>24s}")
    print(f"  {'decode':10s}  {_da:>24s}  {_df:>24s}")
    print()

    per_cat = scores.pop("per_category", {})
    for cat, cat_scores in per_cat.items():
        print(f"  {cat}:")
        if isinstance(cat_scores, dict):
            for metric, value in cat_scores.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
        else:
            print(f"    {cat_scores}")

    if scores:
        print("\n  Overall (from data_overall.csv):")
        for k, v in scores.items():
            print(f"    {k}: {v}")

    # Restore per_category before saving.
    scores["per_category"] = per_cat
    scores["phase_layer_configs"] = phase_configs
    scores["bfcl_categories"] = bfcl_test_categories
    scores["bfcl_tool_mode"] = bfcl_tool_mode
    scores["decode_weight_mode"] = decode_weight_mode
    if resolved_gptq_config:
        scores["gptq"] = {
            "dataset": resolved_gptq_config.get("dataset"),
            "nsamples": resolved_gptq_config.get("nsamples"),
            "seqlen": resolved_gptq_config.get("seqlen"),
            "format": resolved_gptq_config.get("format"),
            "weight_config": resolved_gptq_config.get("weight_config"),
            "cali_batch_size": resolved_gptq_config.get("cali_batch_size"),
            "max_layers": resolved_gptq_config.get("max_layers"),
        }

    if log_dir:
        save_results(log_dir, scores)

    _tmpdir_ctx.cleanup()
    return scores


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
