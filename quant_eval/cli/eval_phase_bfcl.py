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

import copy
import gc
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import torch
import transformers
from torch import Tensor

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
from quant_eval.precision import apply_dse_quant_config, parse_fp_setting, parse_mx_precision, mx_data_config, fp_data_config
from quant_eval.eval.unified_mx import (
    ATTENTION_MEMORY_MODES,
    AttentionQueryChunkController,
    apply_qwen3_gptq_cache_unified_wrappers,
    apply_unified_mx_wrappers,
    get_attention_memory_stats,
)
from quant_eval.bfcl_adapters import BFCL_ADAPTER_NAMES, resolve_bfcl_adapter

from fastapi import FastAPI, Request

import httpx
from bs4 import BeautifulSoup
import markdownify


logger = get_logger(__name__)
set_logging_verbosity("debug")

# ── Default BFCL V4 web-search categories ─────────────────────────────────────
BFCL_WEB_SEARCH_CATEGORIES = ("web_search_base", "web_search_no_snippet")
BFCL_TOOL_MODES = ("auto", "return", "execute")
BFCL_GENERATE_MODES = ("cli", "batched")
BFCL_WEB_SEARCH_BACKENDS = (
    "bfcl_web_search_api_serpapi_duckduckgo",
    "bfcl_harness_ddgs_duckduckgo",
)
BFCL_WEB_SEARCH_CACHE_MODES = ("off", "record", "replay", "auto")
ATTN_IMPLEMENTATIONS = ("auto", "sdpa", "eager")

# ── OpenAI-compatible server defaults ─────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8915

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_attention_backend(
    requested: str | None,
    *,
    qwen_model_family: bool,
    quant_config_is_none: bool,
    codesign_tokens_enabled: bool,
) -> str:
    """Choose an HF attention backend that matches PLENA runtime wrappers."""
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        # Qwen3 unified attention wrappers use an eager-style attention core
        # when QK/softmax/AV are quantized. Loading Qwen3 with SDPA and then
        # switching into that eager-style core can produce backend-mismatch
        # failures such as repeated chat-template tokens. Keep FP baselines on
        # SDPA by default, but use eager for quantized Qwen3/Qwen3-MoE paths.
        if qwen_model_family and not quant_config_is_none and codesign_tokens_enabled:
            return "eager"
        return "sdpa"
    if value not in {"sdpa", "eager"}:
        raise ValueError(f"attn_implementation must be one of {ATTN_IMPLEMENTATIONS}, got {requested!r}.")
    return value


def _execute_tool(tool_call: dict) -> str:
    """Legacy in-server web tools used only by explicit ``tool_mode=execute``."""
    import httpx
    from bs4 import BeautifulSoup

    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to parse tool arguments"})

    logger.debug("Executing tool %s with arguments: %s", name, args)

    # ── duckduckgo_search ──────────────────────────────────────────
    if name in ("duckduckgo_search", "search_engine_query"):
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                logger.debug(
                    "Performing DuckDuckGo search with keywords=%r and region=%r",
                    args.get("keywords", ""),
                    args.get("region", "wt-wt"),
                )
                results = list(ddgs.text(
                    args.get("keywords", args.get("query", "")),  # positional, not keyword
                    region=args.get("region", "wt-wt"),
                    max_results=args.get("max_results", 10),
                ))
                logger.debug("DuckDuckGo search returned %d results", len(results))
            # Match BFCL's WebSearchAPI schema. In particular, base-category
            # scoring expects snippets (``body``); dropping them silently
            # changes the task into the harder no-snippet variant.
            results = [
                {
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", ""),
                }
                for result in results
            ]
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": f"duckduckgo_search failed: {e}"})

    # ── fetch_url_content ──────────────────────────────────────────
    elif name == "fetch_url_content":
        url  = args.get("url", "")
        mode = args.get("mode", "raw")

        try:
            resp = httpx.get(url, timeout=10, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            html = resp.text

            if mode == "raw":
                return json.dumps({"content": html[:20000]})

            elif mode == "markdown":
                try:
                    import markdownify
                    return json.dumps(
                        {"content": markdownify.markdownify(html)[:20000]}
                    )
                except ImportError:
                    # fallback to truncate if markdownify not installed
                    mode = "truncate"

            if mode == "truncate":
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "head"]):
                    tag.decompose()
                text = " ".join(soup.get_text().split())
                return json.dumps({"content": text[:8000]})

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Unknown tool ───────────────────────────────────────────────
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})



# ══════════════════════════════════════════════════════════════════════════════
#  Minimal OpenAI-compatible chat-completion server
# ══════════════════════════════════════════════════════════════════════════════

def _is_cuda_oom(error: Exception) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _generate_with_oom_retry(
    model,
    *,
    oom_retries: int = 2,
    request_label: str = "generation",
    **generate_kwargs,
):
    """Retry a batch-1 server generation after releasing allocator cache."""
    for attempt in range(int(oom_retries) + 1):
        try:
            return model.generate(**generate_kwargs)
        except Exception as error:
            if not _is_cuda_oom(error) or attempt >= int(oom_retries):
                raise
            logger.warning(
                "CUDA OOM during %s (attempt %d/%d); clearing transient "
                "allocations and retrying the same request.",
                request_label,
                attempt + 1,
                int(oom_retries) + 1,
            )
            gc.collect()
            if torch.cuda.is_available():
                for device_index in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()


def _build_server_app(
    model,
    tokenizer,
    device: str,
    tool_mode: str = "execute",
    max_new_tokens: int | None = 2048,
    bfcl_adapter=None,
):
    """
    Return a FastAPI application that exposes POST /v1/chat/completions.

    The quantized model (with PhaseLayerAutoSwitch already enabled) is called
    directly — no additional process boundary.  Tool/function calls are passed
    through transparently so that BFCL can exercise them.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError(
            "fastapi and uvicorn are required to serve the model locally.\n"
            "Install them with:  pip install fastapi uvicorn"
        ) from exc

    if tool_mode not in ("return", "execute"):
        raise ValueError(f"tool_mode must be 'return' or 'execute', got {tool_mode!r}.")
    if bfcl_adapter is None:
        bfcl_adapter = resolve_bfcl_adapter("raw", getattr(tokenizer, "name_or_path", None), None)
    if max_new_tokens is not None and int(max_new_tokens) <= 0:
        raise ValueError(f"max_new_tokens must be positive or None, got {max_new_tokens!r}.")

    def _input_device() -> torch.device:
        try:
            return model.get_input_embeddings().weight.device
        except Exception:
            return torch.device(device)

    def _cap_max_new_tokens(requested: int | None) -> int:
        requested_toks = 1024 if requested is None else int(requested)
        if max_new_tokens is None:
            return requested_toks
        return min(requested_toks, int(max_new_tokens))

    model_context_length = getattr(
        getattr(model, "config", None), "max_position_embeddings", None
    )
    if not isinstance(model_context_length, int) or model_context_length <= 0:
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        model_context_length = (
            int(tokenizer_limit)
            if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000_000
            else None
        )

    app = FastAPI(title="quant-model-server")
    def _generate_serialized(**generate_kwargs):
        with torch.inference_mode():
            if tool_mode == "execute" and torch.cuda.is_available():
                # Agentic turns grow their eager-attention matrices over time.
                # Release inactive blocks between turns so the next large
                # FP32 softmax can obtain a contiguous allocation on tight
                # model-parallel deployments.
                for device_index in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
            return _generate_with_oom_retry(model, **generate_kwargs)

    def _prepare_prompt_ids(builder: Callable[[], Tensor]) -> Tensor:
        prompt_ids = builder()
        if not isinstance(prompt_ids, Tensor) or prompt_ids.ndim != 2:
            raise TypeError("BFCL tokenizer must return a rank-2 token tensor.")
        return prompt_ids.to(device=_input_device(), dtype=torch.long)

    def _generate_turn(
        *,
        prompt_ids: Tensor,
        max_new_toks: int,
        temperature: float,
        request_label: str,
    ) -> Tensor:
        prompt_length = int(prompt_ids.shape[-1])
        if model_context_length is not None:
            available_generation_tokens = (
                int(model_context_length) - prompt_length - 2
            )
            if available_generation_tokens <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompt length {prompt_length} exceeds the model "
                        f"context window {model_context_length}."
                    ),
                )
            max_new_toks = min(max_new_toks, available_generation_tokens)
        attention_mask = torch.ones_like(prompt_ids)
        generate_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_toks,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "pad_token_id": tokenizer.eos_token_id,
            "request_label": request_label,
        }
        return _generate_serialized(**generate_kwargs)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        logger.debug("Chat-completions request received")
        body = await request.json()
        messages     = list(body.get("messages", []))  # make a mutable copy
        tools        = body.get("tools", None)
        temperature  = body.get("temperature", 0.0)
        max_new_toks = _cap_max_new_tokens(body.get("max_tokens", 1024))
        MAX_TURNS = 15  # prevent infinite agentic loops
        last_prompt_tokens = 0
        last_completion_tokens = 0

        for turn in range(MAX_TURNS):
            logger.debug(
                "Chat-completions agentic turn %d/%d with %d messages",
                turn + 1,
                MAX_TURNS,
                len(messages),
            )

            # ── Build prompt ───────────────────────────────────────────
            def build_chat_prompt_ids() -> Tensor:
                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        return tokenizer.apply_chat_template(
                            messages,
                            tools=tools,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                    except Exception:
                        return tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                text = "\n".join(
                    f"{m.get('role','user').upper()}: {m.get('content','')}"
                    for m in messages
                ) + "\nASSISTANT:"
                return tokenizer(text, return_tensors="pt").input_ids

            prompt_ids = _prepare_prompt_ids(build_chat_prompt_ids)

            # ── Inference ──────────────────────────────────────────────
            output_ids = _generate_turn(
                prompt_ids=prompt_ids,
                max_new_toks=max_new_toks,
                temperature=temperature,
                request_label=f"chat completion turn {turn + 1}",
            )

            logger.debug("Generated output shape: %s", tuple(output_ids.shape))
            generated_ids = output_ids[0][prompt_ids.shape[-1]:]
            last_prompt_tokens = int(prompt_ids.shape[-1])
            last_completion_tokens = int(generated_ids.shape[-1])
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            logger.debug("Generated text preview: %s", raw_text[:200])

            logger.debug("Prompt tokens: %s", raw_text[:200])
            # ── Parse tool calls ───────────────────────────────────────
            tool_calls, content = bfcl_adapter.parse_tool_calls(raw_text)
            content = content or ""

            # ── No tool calls → final answer, return immediately ───────
            if not tool_calls:
                logger.debug("Final answer reached at turn %d", turn + 1)
                message = {
                    "role": "assistant",
                    "content": bfcl_adapter.normalize_return_text(content or raw_text),
                }
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
                logger.debug(
                    "Returning tool calls at turn %d: %s",
                    turn + 1,
                    [tc["function"]["name"] for tc in tool_calls],
                )
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
            logger.debug(
                "Executing tool calls at turn %d: %s",
                turn + 1,
                [tc["function"]["name"] for tc in tool_calls],
            )

            # Append the assistant's tool-call message to history
            messages.append({
                "role":       "assistant",
                "content":    content,
                "tool_calls": tool_calls,
            })

            # Do not retain one request's CUDA tensors while it waits for a
            # web tool and another concurrent request enters generation.
            del output_ids, generated_ids, prompt_ids

            # Execute each tool and append results
            for tc in tool_calls:
                tool_name   = tc["function"]["name"]
                logger.info("Executing tool '%s' with arguments: %s", tool_name, tc["function"]["arguments"])
                tool_result = _execute_tool(tc)
                logger.debug(
                    "Tool %s result preview: %s", tool_name, str(tool_result)[:200]
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "name":         tool_name,
                    "content":      tool_result,
                })

        # ── Exceeded MAX_TURNS → return whatever we have ───────────────
        logger.warning("MAX_TURNS (%d) exceeded; returning last content", MAX_TURNS)
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
                "prompt_tokens":     last_prompt_tokens,
                "completion_tokens": last_completion_tokens,
                "total_tokens":      last_prompt_tokens + last_completion_tokens,
            },
        }
        return JSONResponse(content=response)

    @app.post("/v1/completions")
    async def completions(request: Request):
        logger.debug("Text-completions request received")
        body = await request.json()

        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        tools        = body.get("tools", None)
        temperature  = body.get("temperature", 0.0)
        max_new_toks = _cap_max_new_tokens(body.get("max_tokens", 1024))
        MAX_TURNS = 15  # prevent infinite agentic loops

        total_prompt_tokens     = 0
        total_completion_tokens = 0

        for turn in range(MAX_TURNS):
            logger.debug("Text-completions agentic turn %d/%d", turn + 1, MAX_TURNS)

            # Tokenize the current prompt string directly (no chat template).
            # BFCL's QwenFCHandler has already rendered all Qwen control tokens
            # into the prompt string. Do not let the tokenizer prepend another
            # model-level special token.
            prompt_ids = _prepare_prompt_ids(
                lambda: tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids
            )

            output_ids = _generate_turn(
                prompt_ids=prompt_ids,
                max_new_toks=max_new_toks,
                temperature=temperature,
                request_label=f"text completion turn {turn + 1}",
            )

            generated_ids = output_ids[0][prompt_ids.shape[-1]:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            logger.debug("Generated text preview: %s", raw_text[:500])

            total_prompt_tokens     += int(prompt_ids.shape[-1])
            total_completion_tokens += int(generated_ids.shape[-1])

            # Official BFCL handlers own parsing. Returning the model text
            # before any relaxed local parsing ensures malformed JSON, missing
            # tags, and reasoning blocks are scored exactly as generated.
            if tool_mode == "return":
                response_text = bfcl_adapter.completion_response_text(raw_text)
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

            # ── Parse tool calls for the legacy in-server agent loop ───
            tool_calls, content = bfcl_adapter.parse_tool_calls(raw_text)
            content = content or ""
            logger.debug("Parsed tool calls: %s", tool_calls)

            # ── No tool calls → final answer, return immediately ───────
            if not tool_calls:
                logger.debug("Final answer reached at turn %d", turn + 1)
                response_text = bfcl_adapter.normalize_return_text(raw_text)
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

            # The text is now materialized on CPU; release CUDA references
            # before this request waits on tools and yields to another request.
            del output_ids, generated_ids, prompt_ids

            # Execute each tool and append results in a structured block.
            for tc in tool_calls:
                tool_name   = tc["function"]["name"]
                logger.info("Executing tool '%s' with arguments: %s", tool_name, tc["function"]["arguments"])
                tool_result = _execute_tool(tc)
                logger.debug(
                    "Tool %s result preview: %s", tool_name, str(tool_result)[:200]
                )
                prompt += (
                    f"\n<tool_response>\n"
                    f"{json.dumps({'tool_call_id': tc['id'], 'name': tool_name, 'content': tool_result})}\n"
                    f"</tool_response>\n"
                )

        # ── Exceeded MAX_TURNS → return whatever we have ───────────────
        logger.warning("MAX_TURNS (%d) exceeded; returning last output", MAX_TURNS)
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
        logger.debug("Normalized BFCL return text: %s", normalized[:300])
    return normalized or text



def _bfcl_return_text_from_tool_calls(tool_calls: list) -> str:
    """Serialize parsed tool calls into BFCL AST-decoder input.

    BFCL non-live categories score the callable payload, not the assistant's
    surrounding prose. This intentionally preserves the model's function names
    and argument values exactly as parsed; it only removes transport wrappers
    such as markdown fences, <tool_call> tags, and Llama special tokens.
    """
    payloads = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue

        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                # Preserve malformed/non-dict argument payloads rather than
                # guessing a semantic repair. BFCL should score that failure.
                pass
        if not isinstance(args, dict):
            args = {"value": args}

        payloads.append({"name": name, "parameters": args})

    if len(payloads) == 1:
        # BFCL's Llama 3.1 handler decodes a single call with eval(), not
        # json.loads(), so Python literals are required for booleans/None.
        return repr(payloads[0])

    # For multiple calls the same handler switches to the semicolon path and
    # json.loads() each segment, so keep standard JSON there.
    return ";".join(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        for payload in payloads
    )


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



def _parse_bfcl_payload_tool_calls(payload: str) -> list:
    """Parse BFCL-style callable payload text into OpenAI tool_calls shape.

    Supported payloads are a single JSON/Python dict, a list of dicts, or a
    semicolon-separated sequence of dict payloads. This parser is deliberately
    syntax-only: it does not coerce argument values to match BFCL schemas.
    """
    payload = (payload or "").strip()
    if not payload:
        return []

    if payload[0] not in "[{":
        return []

    tool_calls = []
    payloads = [p.strip() for p in payload.split(";") if p.strip()]
    try:
        if len(payloads) > 1:
            for item in payloads:
                _append_tool_call(tool_calls, _loads_tool_payload(item))
        else:
            parsed = _loads_tool_payload(payload)
            if isinstance(parsed, list):
                for item in parsed:
                    _append_tool_call(tool_calls, item)
            else:
                _append_tool_call(tool_calls, parsed)
    except (json.JSONDecodeError, SyntaxError, ValueError):
        logger.warning("Failed to parse BFCL tool-call payload: %s", payload[:200])
        return []
    return tool_calls


def _iter_balanced_payloads(text: str):
    """Yield balanced dict/list substrings embedded in model prose.

    Some Llama outputs are transport-wrapped as plain prose followed by an
    inline callable JSON object, without markdown fences. BFCL should score
    the callable payload, not the surrounding prose. This scanner only finds
    syntactically balanced ``{...}`` or ``[...]`` spans while respecting quoted
    strings; it does not modify the payload contents.
    """
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


def _parse_embedded_bfcl_payload_tool_calls(text: str) -> list:
    """Parse callable payloads embedded in prose without semantic repair."""
    tool_calls = []
    for candidate in _iter_balanced_payloads(text):
        candidate_calls = _parse_bfcl_payload_tool_calls(candidate)
        if candidate_calls:
            tool_calls.extend(candidate_calls)
    return tool_calls


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
            logger.debug("Found %d tool calls with pattern %s", len(matches), pat)
            tool_calls = []
            for m in matches:
                try:
                    parsed = _loads_tool_payload(m.strip())
                    _append_tool_call(tool_calls, parsed)
                    logger.debug(
                        "Parsed tool call %d: %s",
                        len(tool_calls) - 1,
                        tool_calls[-1],
                    )
                except (json.JSONDecodeError, SyntaxError, ValueError, IndexError):
                    logger.warning("Failed to parse tool-call JSON: %s", m[:200])
                    continue
            if tool_calls:
                leftover = re.sub(pat, "", text, flags=re.DOTALL).strip()
                return tool_calls, leftover

    # Llama 3.1 function-calling output uses <|python_tag|> followed by one
    # JSON object, a list of JSON objects, or semicolon-separated JSON objects.
    # Also try raw JSON payloads without transport tokens so completion-mode
    # output is normalized through the same BFCL serialization path.
    llama_payload = _normalize_bfcl_return_text(text)
    tool_calls = _parse_bfcl_payload_tool_calls(llama_payload)
    if tool_calls:
        return tool_calls, ""

    # Finally, handle prose + inline callable payloads, e.g.
    # "Here is the JSON:\n\n{\"name\": ..., \"parameters\": ...}".
    # This remains syntax-only and does not turn arbitrary code/prose into
    # function calls.
    tool_calls = _parse_embedded_bfcl_payload_tool_calls(llama_payload)
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
    # Keep an explicit lifecycle handle so repeated persistent trials can stop
    # their embedded server deterministically.
    app.state.uvicorn_server = server
    app.state.uvicorn_thread = thread
    thread.start()

    # Wait for *this* uvicorn instance to start. A stale process can already
    # be bound to the requested port; merely getting an HTTP response would
    # then route BFCL to the wrong model server and corrupt scores.
    import httpx
    deadline = time.time() + 60
    last_error = None
    while time.time() < deadline:
        if not thread.is_alive():
            raise RuntimeError(
                f"Server thread exited before binding http://{host}:{port}. "
                "The port is likely already in use."
            )
        if getattr(server, "started", False):
            try:
                httpx.get(f"http://{host}:{port}/v1/models", timeout=1)
                logger.info("Server ready at http://%s:%d", host, port)
                return thread
            except Exception as exc:
                last_error = exc
        time.sleep(0.5)

    raise RuntimeError(
        f"Server at http://{host}:{port} did not start in time."
        + (f" Last error: {last_error}" if last_error else "")
    )


def _stop_server(app, timeout: float = 60.0) -> None:
    """Stop the embedded server deterministically."""
    if app is None:
        return

    server = getattr(app.state, "uvicorn_server", None)
    thread = getattr(app.state, "uvicorn_thread", None)
    if server is not None:
        server.should_exit = True
    if thread is not None and thread.is_alive():
        thread.join(timeout=max(0.0, float(timeout)))
    if thread is not None and thread.is_alive():
        logger.warning(
            "BFCL server did not stop within %.1fs; forcing uvicorn exit.",
            float(timeout),
        )
        if server is not None:
            server.force_exit = True
        thread.join(timeout=5.0)

    app.state.uvicorn_server = None
    app.state.uvicorn_thread = None


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
    """Resolve server tool behavior for BFCL generation.

    BFCL owns the multi-step executable-tool loop, including WebSearchAPI
    state, snippets, and inference logs. The local model endpoint must return
    each tool call immediately. ``execute`` remains an explicit legacy mode
    for experiments that intentionally use PLENA's in-server tool runner.
    """
    mode = mode.lower()
    if mode not in BFCL_TOOL_MODES:
        raise ValueError(
            f"bfcl_tool_mode must be one of {BFCL_TOOL_MODES}, got {mode!r}."
        )
    if mode != "auto":
        return mode
    return "return"


def _normalize_gptq_dataset(dataset: str) -> str:
    """Accept either GPTQ loader specs or plain local file paths."""
    dataset = dataset.strip()
    if not dataset:
        raise ValueError("gptq_dataset cannot be empty.")
    if dataset.startswith(("file:", "hf:", "jsonl:", "txt:", "lm_eval:")):
        return dataset
    return "file:" + dataset


def _normalize_device_id(device_id: str) -> str:
    """Normalize CLI shorthand like ``0`` to a torch device string."""
    device_id = str(device_id).strip()
    if device_id.isdigit():
        return f"cuda:{device_id}"
    return device_id


def _mark_gptq_projection_configs(pass_args: dict) -> int:
    """Tell module replacement not to PTQ weights already handled by GPTQ.

    Chop's GPTQ pass writes optimized weights back into the original nn.Linear
    modules before regex replacement.  LinearMXInt/LinearMXFP will quantize
    weights again during replacement unless their config has ``gptq=True``.
    Mark every weight-quantized Linear config so GPTQ remains the only weight
    quantizer; activation quantization still runs normally in forward().
    """
    marked = 0
    for key, entry in pass_args.items():
        if key in {"by", "gptq", "token_collector", "rotation_search"}:
            continue
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config")
        if not isinstance(cfg, dict):
            cfg = entry
        is_mx_linear_config = cfg.get("name") in {"mxint", "mxfp"}
        has_weight_quant = any(
            k in cfg
            for k in (
                "weight_width",
                "weight_exponent_width",
                "weight_frac_width",
                "weight_block_size",
            )
        )
        if is_mx_linear_config and has_weight_quant:
            cfg["gptq"] = True
            marked += 1
    return marked


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
    weight_exponent_width: int | None,
    weight_frac_width: int | None,
    weight_block_size: int,
    cali_batch_size: int,
    max_layers: int | None,
    device_map_aware: bool = False,
    cpu_staged: bool = False,
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
    weight_config = {
        **dict(gptq_cfg.get("weight_config", {})),
        "weight_width": weight_width,
        "weight_block_size": weight_block_size,
    }
    if weight_exponent_width is not None:
        weight_config["weight_exponent_width"] = int(weight_exponent_width)
    if weight_frac_width is not None:
        weight_config["weight_frac_width"] = int(weight_frac_width)

    gptq_cfg.update({
        "model_name": model_name,
        "device": device_id,
        "dataset": _normalize_gptq_dataset(dataset),
        "nsamples": nsamples,
        "seqlen": seqlen,
        "format": fmt,
        "weight_config": weight_config,
        "cali_batch_size": cali_batch_size,
        "device_map_aware": bool(device_map_aware),
        "cpu_staged": bool(cpu_staged),
    })
    if max_layers is not None:
        if max_layers <= 0:
            raise ValueError("gptq_max_layers must be positive when set.")
        gptq_cfg["max_layers"] = max_layers
    else:
        gptq_cfg.pop("max_layers", None)

    pass_args["gptq"] = gptq_cfg
    return gptq_cfg



_GPTQ_CACHE_MODES = {"off", "auto", "refresh", "require", "memory"}
_GPTQ_CACHE_MODEL_NAME = "quantized_model"


def _stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _gptq_cache_fingerprint(gptq_config: dict, total_layers: int) -> dict:
    """Return only fields that can affect GPTQ-produced weights."""
    max_layers = gptq_config.get("max_layers", None)
    expected_count = total_layers if max_layers is None else min(int(max_layers), total_layers)
    return {
        "model_name": gptq_config.get("model_name"),
        "dataset": gptq_config.get("dataset"),
        "nsamples": gptq_config.get("nsamples"),
        "seqlen": gptq_config.get("seqlen"),
        "format": gptq_config.get("format"),
        "weight_config": gptq_config.get("weight_config", {}),
        "quantile_search": gptq_config.get("quantile_search", True),
        "clip_search_y": gptq_config.get("clip_search_y", False),
        "cali_batch_size": gptq_config.get("cali_batch_size", 32),
        "max_layers": max_layers,
        "expected_layers": list(range(expected_count)),
        "total_layers": total_layers,
    }


class _GptqWeightCache:
    """Load-only GPTQ cache for DSE sweeps that do not vary weight config.

    Chop's built-in GPTQ checkpoint_dir is a resume mechanism. This wrapper adds
    stricter semantics: a complete matching cache is loaded and GPTQ is skipped;
    otherwise one process creates a fresh cache under a fingerprint-specific lock.
    """

    def __init__(self, *, cache_dir: str | Path, mode: str, gptq_config: dict, total_layers: int):
        mode = str(mode or "off").lower()
        if mode not in _GPTQ_CACHE_MODES:
            raise ValueError(f"gptq_cache_mode must be one of {_GPTQ_CACHE_MODES}, got {mode!r}.")
        self.mode = mode
        self.root = Path(cache_dir).expanduser().resolve()
        self.fingerprint = _gptq_cache_fingerprint(gptq_config, total_layers)
        self.key = _stable_json_hash(self.fingerprint)
        self.cache_path = self.root / self.key
        self.lock_path = self.root / f"{self.key}.lock"
        self.expected_layers = list(self.fingerprint["expected_layers"])
        self.hit = False
        self.loaded_layers = 0
        self.partial_layers = 0
        self.resuming = False
        self.model_path_alias_hit = False
        self._lock_acquired = False
        self._wait_sec = 7200
        self._poll_sec = 5.0

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "key": self.key,
            "hit": self.hit,
            "path": str(self.cache_path),
            "loaded_layers": self.loaded_layers,
            "partial_layers": self.partial_layers,
            "resuming": self.resuming,
            "model_path_alias_hit": self.model_path_alias_hit,
            "expected_layers": self.expected_layers,
        }

    def _metadata_path(self) -> Path:
        return self.cache_path / "metadata.json"

    def _layer_path(self, layer_idx: int) -> Path:
        return self.cache_path / f"{_GPTQ_CACHE_MODEL_NAME}_layer_{layer_idx}.safetensors"

    def _existing_layers(self) -> list[int]:
        if not self.cache_path.exists():
            return []
        existing = []
        for layer_idx in self.expected_layers:
            if self._layer_path(layer_idx).exists():
                existing.append(layer_idx)
        return existing

    def _is_complete(self) -> bool:
        meta_path = self._metadata_path()
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not meta.get("complete"):
            return False
        if meta.get("cache_key") != self.key:
            return False
        if meta.get("fingerprint") != self.fingerprint:
            return False
        if meta.get("expected_layers") != self.expected_layers:
            return False
        return all(self._layer_path(i).exists() for i in self.expected_layers)

    @staticmethod
    def _snapshot_identity(model_name: object) -> str | None:
        value = str(model_name or "").rstrip("/")
        marker = "/snapshots/"
        if marker in value:
            revision = value.split(marker, 1)[1].split("/", 1)[0]
            return f"snapshot:{revision}" if revision else None
        return None

    @staticmethod
    def _dataset_identity(dataset: object) -> str:
        """Fingerprint local calibration contents independently of path."""
        value = str(dataset or "")
        if not value.startswith("file:"):
            return f"spec:{value}"
        raw_path = Path(value[len("file:"):]).expanduser()
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates = [Path.cwd() / raw_path, Path(__file__).resolve().parents[2] / raw_path]
        path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return f"missing-file:{raw_path}"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"file-sha256:{path.stat().st_size}:{digest.hexdigest()}"

    def _find_model_path_alias_cache(self) -> Path | None:
        """Find a complete cache whose only fingerprint change is local path."""
        requested_identity = self._snapshot_identity(self.fingerprint.get("model_name"))
        if requested_identity is None or not self.root.is_dir():
            return None
        requested = dict(self.fingerprint)
        requested.pop("model_name", None)
        requested_dataset = self._dataset_identity(requested.pop("dataset", None))
        for metadata_path in sorted(self.root.glob("*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidate_fingerprint = metadata.get("fingerprint")
            if not metadata.get("complete") or not isinstance(candidate_fingerprint, dict):
                continue
            if self._snapshot_identity(candidate_fingerprint.get("model_name")) != requested_identity:
                continue
            candidate = dict(candidate_fingerprint)
            candidate.pop("model_name", None)
            candidate_dataset = self._dataset_identity(candidate.pop("dataset", None))
            if candidate_dataset != requested_dataset:
                continue
            if candidate != requested:
                continue
            candidate_path = metadata_path.parent
            if all(
                (candidate_path / f"{_GPTQ_CACHE_MODEL_NAME}_layer_{i}.safetensors").is_file()
                for i in self.expected_layers
            ):
                return candidate_path
        return None

    def _acquire_lock(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self._wait_sec
        while True:
            try:
                os.mkdir(self.lock_path)
                self._lock_acquired = True
                owner = {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                (self.lock_path / "owner.json").write_text(
                    json.dumps(owner, indent=2, sort_keys=True), encoding="utf-8"
                )
                return
            except FileExistsError:
                owner_path = self.lock_path / "owner.json"
                try:
                    owner = json.loads(owner_path.read_text(encoding="utf-8"))
                except Exception:
                    owner = {}
                owner_pid = owner.get("pid")
                if owner.get("hostname") == socket.gethostname() and isinstance(owner_pid, int):
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        logger.warning(
                            "Removing stale GPTQ cache lock owned by dead pid %s: %s",
                            owner_pid,
                            self.lock_path,
                        )
                        shutil.rmtree(self.lock_path, ignore_errors=True)
                        continue
                    except PermissionError:
                        pass
                logger.info("Waiting for GPTQ cache lock: key=%s", self.key)
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for GPTQ cache lock {self.lock_path}")
                time.sleep(self._poll_sec)

    def release(self) -> None:
        if self._lock_acquired:
            shutil.rmtree(self.lock_path, ignore_errors=True)
            self._lock_acquired = False

    def load(self, model) -> int:
        from safetensors.torch import load_file

        decoder = getattr(model, "model", None)
        layers = getattr(decoder, "layers", None)
        direct_layer_load = layers is not None and all(
            0 <= layer_idx < len(layers) for layer_idx in self.expected_layers
        )
        if direct_layer_load:
            logger.info(
                "Loading GPTQ cache directly into %d decoder layers with one-layer I/O prefetch",
                len(self.expected_layers),
            )
        else:
            logger.warning(
                "Model does not expose model.layers for direct GPTQ cache loading; "
                "falling back to whole-model state_dict loading."
            )

        started = time.perf_counter()
        total = len(self.expected_layers)

        # Keep only one layer prefetched: layer checkpoints are large for MoE
        # models, so an unbounded parallel map would create a severe RAM spike.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gptq-cache") as executor:
            next_future = None
            for position, layer_idx in enumerate(self.expected_layers, start=1):
                layer_started = time.perf_counter()
                if next_future is None:
                    layer_state = load_file(str(self._layer_path(layer_idx)))
                else:
                    layer_state = next_future.result()

                if position < total:
                    next_layer_idx = self.expected_layers[position]
                    next_future = executor.submit(
                        load_file, str(self._layer_path(next_layer_idx))
                    )
                else:
                    next_future = None

                if direct_layer_load:
                    incompatible = layers[layer_idx].load_state_dict(layer_state, strict=False)
                    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
                    if unexpected:
                        raise RuntimeError(
                            f"GPTQ cache layer {layer_idx} has unexpected keys: "
                            f"{unexpected[:8]}"
                        )
                else:
                    model_state = {
                        f"model.layers.{layer_idx}.{name}": value
                        for name, value in layer_state.items()
                    }
                    incompatible = model.load_state_dict(model_state, strict=False)
                    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
                    if unexpected:
                        raise RuntimeError(
                            f"GPTQ cache layer {layer_idx} has unexpected keys: "
                            f"{unexpected[:8]}"
                        )

                del layer_state
                if position == 1 or position == total or position % 5 == 0:
                    logger.info(
                        "Loaded GPTQ cache layer %d/%d (decoder layer %d) in %.1fs",
                        position,
                        total,
                        layer_idx,
                        time.perf_counter() - layer_started,
                    )
        self.hit = True
        self.loaded_layers = len(self.expected_layers)
        logger.info(
            "GPTQ cache hit: loaded %d cached layers in %.1fs, skipping GPTQ (key=%s)",
            self.loaded_layers,
            time.perf_counter() - started,
            self.key,
        )
        return self.loaded_layers

    def _accept_complete_cache(self, model, *, load_on_hit: bool) -> bool:
        if load_on_hit:
            self.load(model)
        else:
            self.hit = True
            self.loaded_layers = 0
            logger.info(
                "GPTQ cache-only hit: verified %d cached layers without loading weights (key=%s)",
                len(self.expected_layers),
                self.key,
            )
        return True

    def prepare(self, model, *, load_on_hit: bool = True) -> bool:
        if self.mode == "off":
            return False
        if self.mode != "refresh" and self._is_complete():
            return self._accept_complete_cache(model, load_on_hit=load_on_hit)
        if self.mode != "refresh":
            alias_path = self._find_model_path_alias_cache()
            if alias_path is not None:
                logger.info(
                    "GPTQ cache model-path alias hit: requested=%s cached=%s path=%s",
                    self.fingerprint.get("model_name"),
                    json.loads((alias_path / "metadata.json").read_text(encoding="utf-8"))
                    .get("fingerprint", {})
                    .get("model_name"),
                    alias_path,
                )
                self.cache_path = alias_path
                self.model_path_alias_hit = True
                return self._accept_complete_cache(model, load_on_hit=load_on_hit)

        self._acquire_lock()
        try:
            if self.mode != "refresh" and self._is_complete():
                self._accept_complete_cache(model, load_on_hit=load_on_hit)
                self.release()
                return True
            if self.mode == "require":
                raise FileNotFoundError(
                    f"GPTQ cache miss for key={self.key} at {self.cache_path}; "
                    "run with gptq_cache_mode=auto or refresh to populate it."
                )
            if self.mode == "refresh":
                logger.info("GPTQ cache refresh requested: key=%s", self.key)
                shutil.rmtree(self.cache_path, ignore_errors=True)
                self.cache_path.mkdir(parents=True, exist_ok=True)
            else:
                existing_layers = self._existing_layers()
                self.partial_layers = len(existing_layers)
                self.cache_path.mkdir(parents=True, exist_ok=True)
                if existing_layers:
                    self.resuming = True
                    logger.info(
                        "GPTQ partial cache found: key=%s layers=%s; "
                        "resuming via checkpoint_dir=%s",
                        self.key,
                        existing_layers,
                        self.cache_path,
                    )
                else:
                    logger.info("GPTQ cache miss: key=%s, running GPTQ once", self.key)
            return False
        except Exception:
            self.release()
            raise

    def finalize(self) -> None:
        missing = [i for i in self.expected_layers if not self._layer_path(i).exists()]
        if missing:
            raise RuntimeError(f"GPTQ cache incomplete for key={self.key}; missing layers={missing}")
        meta = {
            "complete": True,
            "cache_key": self.key,
            "fingerprint": self.fingerprint,
            "expected_layers": self.expected_layers,
            "created_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_name": _GPTQ_CACHE_MODEL_NAME,
        }
        self._metadata_path().write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        self.hit = False
        self.loaded_layers = 0
        logger.info(
            "GPTQ cache populated: key=%s layers=%d path=%s",
            self.key,
            len(self.expected_layers),
            self.cache_path,
        )


def _configure_bfcl_web_backend_env(
    env: dict[str, str],
    *,
    search_backend: str,
    cache_mode: str,
    cache_dir: str | Path | None,
    stats_path: str | Path | None,
    max_search_concurrency: int = 1,
) -> None:
    """Configure the standalone BFCL process without patching site-packages."""
    backend = str(search_backend or BFCL_WEB_SEARCH_BACKENDS[0]).strip().lower()
    if backend not in BFCL_WEB_SEARCH_BACKENDS:
        raise ValueError(
            f"Unknown BFCL-Web search backend {search_backend!r}; expected one "
            f"of {BFCL_WEB_SEARCH_BACKENDS}."
        )
    if backend != "bfcl_harness_ddgs_duckduckgo":
        return

    selected_cache_mode = str(cache_mode or "off").strip().lower()
    if selected_cache_mode not in BFCL_WEB_SEARCH_CACHE_MODES:
        raise ValueError(
            "BFCL-Web cache mode must be one of "
            f"{BFCL_WEB_SEARCH_CACHE_MODES}, got {cache_mode!r}."
        )
    if selected_cache_mode != "off" and cache_dir is None:
        raise ValueError(
            "A BFCL-Web cache directory is required for DDGS cache mode "
            f"{selected_cache_mode!r}."
        )

    overlay_dir = _REPO_ROOT / "quant_eval" / "bfcl_runtime"
    python_path_entries = [str(overlay_dir), str(_REPO_ROOT)]
    python_path_entries.extend(
        entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry
    )
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_path_entries))
    env["PLENA_BFCL_WEB_SEARCH_BACKEND"] = backend
    env["PLENA_BFCL_WEB_CACHE_MODE"] = selected_cache_mode
    if int(max_search_concurrency) <= 0:
        raise ValueError("BFCL-Web search concurrency must be positive.")
    env["PLENA_BFCL_WEB_DDGS_MAX_CONCURRENCY"] = str(
        int(max_search_concurrency)
    )
    if cache_dir is not None:
        resolved_cache_dir = Path(cache_dir).expanduser().resolve()
        if selected_cache_mode in {"record", "auto"}:
            resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        env["PLENA_BFCL_WEB_CACHE_DIR"] = str(resolved_cache_dir)
    if stats_path is not None:
        resolved_stats_path = Path(stats_path).expanduser().resolve()
        resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_stats_path.unlink(missing_ok=True)
        env["PLENA_BFCL_WEB_STATS_PATH"] = str(resolved_stats_path)


def _run_bfcl_generate(
    model_name: str,
    test_categories: Sequence[str],
    host: str,
    port: int,
    result_dir: Path,
    num_threads: int,
    limit: int | None,
    model_alias: str | None = None,
    temperature: float | None = None,
    allow_overwrite: bool = False,
    web_search_backend: str = "bfcl_web_search_api_serpapi_duckduckgo",
    web_search_cache_mode: str = "off",
    web_search_cache_dir: str | Path | None = None,
    web_search_stats_path: str | Path | None = None,
    web_search_max_concurrency: int = 1,
) -> int:
    """Call ``bfcl generate`` against the local server; return the exit code."""
    bfcl_name = _bfcl_model_name(model_name, model_alias=model_alias)
    result_dir = result_dir.resolve()
    cmd = [
        "bfcl", "generate",
        "--model",         bfcl_name,
        "--test-category", *test_categories,
        "--skip-server-setup",
        "--result-dir",    str(result_dir),
        "--num-threads",   str(num_threads),
    ]
    local_model_path = Path(model_name)
    if local_model_path.is_dir():
        cmd.extend(["--local-model-path", str(local_model_path.resolve())])
    if temperature is not None:
        cmd.extend(["--temperature", str(float(temperature))])
    if allow_overwrite:
        cmd.append("--allow-overwrite")
    env = os.environ.copy()
    env["LOCAL_SERVER_ENDPOINT"] = host
    env["LOCAL_SERVER_PORT"]     = str(port)
    if local_model_path.is_dir():
        env.setdefault("REMOTE_OPENAI_TOKENIZER_PATH", str(local_model_path.resolve()))
    if any(category in BFCL_WEB_SEARCH_CATEGORIES for category in test_categories):
        if (
            web_search_backend == "bfcl_harness_ddgs_duckduckgo"
            and web_search_stats_path is None
        ):
            web_search_stats_path = result_dir / "bfcl_web_backend_stats.json"
        _configure_bfcl_web_backend_env(
            env,
            search_backend=web_search_backend,
            cache_mode=web_search_cache_mode,
            cache_dir=web_search_cache_dir,
            stats_path=web_search_stats_path,
            max_search_concurrency=web_search_max_concurrency,
        )

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
        _ensure_bfcl_eval_importable()
        try:
            from bfcl_eval.utils import load_dataset_entry  # type: ignore

            ids = {
                cat: [str(row["id"]) for row in load_dataset_entry(cat)[:limit]]
                for cat in test_categories
            }
        except Exception as exc:
            raise RuntimeError(
                "Unable to resolve official BFCL IDs for a limited run."
            ) from exc
        if any(len(values) != limit for values in ids.values()):
            raise ValueError(
                f"Requested limit={limit}, but selected BFCL IDs were {ids}."
            )
        ids_path.write_text(json.dumps(ids, indent=2) + "\n")
        cmd.append("--run-ids")

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _bfcl_site_packages_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_bin = os.environ.get("PATH", "").split(os.pathsep)
    for entry in env_bin:
        p = Path(entry)
        if p.name == "bin":
            candidates.extend((p.parent / "lib").glob("python*/site-packages"))
    candidates.extend((_REPO_ROOT / ".conda" / "envs" / "plena-bfcl" / "lib").glob("python*/site-packages"))
    return [p for p in candidates if p.exists()]


def _ensure_bfcl_eval_importable() -> None:
    try:
        import bfcl_eval  # noqa: F401
        return
    except Exception:
        pass
    for site_packages in _bfcl_site_packages_candidates():
        value = str(site_packages)
        if value not in sys.path:
            sys.path.insert(0, value)
        try:
            import bfcl_eval  # noqa: F401
            return
        except Exception:
            continue


def _load_bfcl_multiple_entries(limit: int | None) -> list[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when set.")
    _ensure_bfcl_eval_importable()
    try:
        from bfcl_eval.utils import load_dataset_entry  # type: ignore

        rows = load_dataset_entry("multiple")
    except Exception as exc:
        data_path = os.environ.get("BFCL_MULTIPLE_DATA_PATH")
        candidates = [Path(data_path)] if data_path else []
        for site_packages in _bfcl_site_packages_candidates():
            candidates.append(site_packages / "bfcl_eval" / "data" / "BFCL_v4_multiple.json")
        candidates.append(_REPO_ROOT / ".conda" / "envs" / "plena-bfcl" / "lib" / "python3.11" / "site-packages" / "bfcl_eval" / "data" / "BFCL_v4_multiple.json")
        for path in candidates:
            if path and path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                break
        else:
            raise FileNotFoundError("Unable to locate BFCL_v4_multiple.json for batched generation.") from exc
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("No BFCL multiple rows selected for batched generation.")
    return rows


def _extract_first_turn_messages(row: dict) -> list[dict[str, str]]:
    question = row.get("question", [])
    if question and isinstance(question, list) and isinstance(question[0], list):
        first_turn = question[0]
    elif isinstance(question, list):
        first_turn = question
    else:
        first_turn = []
    messages: list[dict[str, str]] = []
    for message in first_turn:
        if isinstance(message, dict):
            messages.append({
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            })
    if not messages:
        raise ValueError(f"BFCL row {row.get('id', '<unknown>')} has no first-turn messages.")
    return messages


def _render_qwen3_bfcl_prompt(row: dict, *, disable_thinking: bool = False) -> str:
    from calib.build_bfcl_official_wrapped_qwen3 import render_qwen3_fc_prompt

    functions = row.get("function", [])
    if not isinstance(functions, list):
        raise TypeError(f"BFCL row {row.get('id', '<unknown>')} function field is not a list.")
    prompt = render_qwen3_fc_prompt(_extract_first_turn_messages(row), functions)
    if disable_thinking:
        # Mirrors Qwen3's tokenizer chat_template when
        # apply_chat_template(..., enable_thinking=False) is used.
        assistant_prefix = "<|im_start|>assistant\n"
        if not prompt.endswith(assistant_prefix):
            raise ValueError("Qwen3 BFCL prompt did not end with the expected assistant prefix.")
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def _batch_input_device(model, fallback_device: str) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device(fallback_device)


def _trim_generated_padding(tokens: torch.Tensor, pad_token_id: int | None, eos_token_id: int | None) -> torch.Tensor:
    if tokens.numel() == 0 or pad_token_id is None or pad_token_id == eos_token_id:
        return tokens
    end = tokens.numel()
    while end > 0 and int(tokens[end - 1]) == int(pad_token_id):
        end -= 1
    return tokens[:end]


def _write_bfcl_multiple_results(result_dir: Path, bfcl_name: str, records: list[dict]) -> Path:
    out_dir = result_dir / bfcl_name.replace("/", "_") / "non_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "BFCL_v4_multiple_result.json"
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def _run_batched_bfcl_multiple_generate(
    *,
    model,
    tokenizer,
    device: str,
    result_dir: Path,
    model_name: str,
    model_alias: str | None,
    bfcl_adapter,
    max_new_tokens: int | None,
    limit: int | None,
    batch_size: int,
    batch_length_bucket: bool,
    disable_thinking: bool,
) -> int:
    if batch_size <= 0:
        raise ValueError("bfcl_batch_size must be positive.")

    rows = _load_bfcl_multiple_entries(limit)
    prompts = [_render_qwen3_bfcl_prompt(row, disable_thinking=disable_thinking) for row in rows]
    prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
    indices = list(range(len(rows)))
    if batch_length_bucket:
        indices.sort(key=lambda idx: prompt_lengths[idx])

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    old_pad_token = tokenizer.pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    input_device = _batch_input_device(model, device)
    max_new_toks = 1024 if max_new_tokens is None else int(max_new_tokens)
    results_by_index: dict[int, dict] = {}

    try:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_prompts = [prompts[idx] for idx in batch_indices]
            encoded = input_ids = attention_mask = output_ids = generated = None
            try:
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = encoded.input_ids.to(input_device)
                attention_mask = encoded.attention_mask.to(input_device)
                t0 = time.time()
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_toks,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                batch_latency = time.time() - t0
                generated = output_ids[:, input_ids.shape[-1]:]
                for row_pos, idx in enumerate(batch_indices):
                    gen_ids = _trim_generated_padding(
                        generated[row_pos].detach().cpu(),
                        tokenizer.pad_token_id,
                        tokenizer.eos_token_id,
                    )
                    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
                    # As with the OpenAI Completions endpoint, retain the
                    # model's exact callable syntax for BFCL's official parser.
                    # The adapter may only remove transport-level stop tokens.
                    result_text = bfcl_adapter.completion_response_text(raw_text)
                    prompt_tokens = int(attention_mask[row_pos].sum().item())
                    completion_tokens = int(gen_ids.numel())
                    results_by_index[idx] = {
                        "id": rows[idx]["id"],
                        "result": result_text,
                        "input_token_count": prompt_tokens,
                        "output_token_count": completion_tokens,
                        "latency": batch_latency,
                    }
                logger.info(
                    "Batched BFCL multiple generated %d/%d samples (batch_size=%d, latency=%.2fs)",
                    min(start + len(batch_indices), len(indices)),
                    len(indices),
                    len(batch_indices),
                    batch_latency,
                )
                partial_records = [results_by_index[idx] for idx in range(len(rows)) if idx in results_by_index]
                _write_bfcl_multiple_results(
                    result_dir,
                    _bfcl_model_name(model_name, model_alias=model_alias),
                    partial_records,
                )
            finally:
                del generated, output_ids, attention_mask, input_ids, encoded
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.pad_token = old_pad_token

    records = [results_by_index[idx] for idx in range(len(rows))]
    out_path = _write_bfcl_multiple_results(
        result_dir,
        _bfcl_model_name(model_name, model_alias=model_alias),
        records,
    )
    logger.info("Wrote batched BFCL multiple results: %s", out_path)
    return 0


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
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        records = []
        skipped = 0
        for line in lines:
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, RecursionError) as exc:
                skipped += 1
                logger.warning("Skipping unparsable BFCL score record in %s: %s", path, exc)
        if skipped and records and isinstance(records[0], dict):
            records[0]["_parse_warning"] = f"skipped {skipped} unparsable score record(s)"
        return records
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        records = []
        skipped = 0
        for line in lines:
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, RecursionError) as exc:
                skipped += 1
                logger.warning("Skipping unparsable BFCL score record in %s: %s", path, exc)
        if skipped and records and isinstance(records[0], dict):
            records[0]["_parse_warning"] = f"skipped {skipped} unparsable score record(s)"
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
    result_dir = result_dir.resolve()
    score_dir = score_dir.resolve()
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

    returncode = proc.returncode
    if (
        partial_eval
        and returncode != 0
        and per_cat
        and "StatisticsError" in proc.stderr
        and "stdev requires at least two data points" in proc.stderr
    ):
        logger.warning(
            "bfcl evaluate produced per-category scores but failed while "
            "building the leaderboard latency stdev for a one-sample partial eval; "
            "treating the score as usable."
        )
        scores["partial_eval_warning"] = "bfcl_latency_stdev_single_sample"
        returncode = 0

    return returncode, scores


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _task_success(task_dir: Path, protocol: dict, expected_count: int) -> dict | None:
    success_path = task_dir / "task_success.json"
    if not success_path.is_file():
        return None
    try:
        payload = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("protocol_digest") == _json_digest(protocol)
        and payload.get("sample_count") == expected_count
        and payload.get("status") == "complete"
    ):
        return payload
    return None


def _first_nested_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_nested_string(item)
            if found is not None:
                return found
    return None


def _gsm8k_metrics(
    results: dict,
    *,
    tokenizer=None,
    max_new_tokens: int | None = None,
    response_scope: str = "decoded_response",
) -> dict:
    task_metrics = results.get("results", {}).get("gsm8k", {})
    samples = results.get("samples", {}).get("gsm8k", [])

    def metric(name: str) -> float | None:
        for key, value in task_metrics.items():
            if (key == name or key.split(",", 1)[0] == name) and isinstance(value, (int, float)):
                return float(value)
        return None

    metrics = {
        # lm-eval duplicates logged sample records for each output filter
        # (strict-match and flexible-extract). ``sample_len`` is the number of
        # evaluated documents and is therefore the completion invariant.
        "sample_count": int(task_metrics.get("sample_len", len(samples))),
        "strict_match": metric("exact_match,strict-match"),
        "flexible_extract": metric("exact_match,flexible-extract"),
    }
    if tokenizer is None:
        return metrics

    # Logged samples are duplicated once per output filter. Deduplicate by
    # document id before reporting response-length diagnostics.
    responses_by_doc: dict[object, str] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        doc_id = sample.get("doc_id", len(responses_by_doc))
        response = _first_nested_string(sample.get("resps"))
        if response is not None:
            responses_by_doc.setdefault(doc_id, response)
    output_lengths = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in responses_by_doc.values()
    ]
    cap_hits = (
        sum(length >= int(max_new_tokens) for length in output_lengths)
        if max_new_tokens is not None
        else 0
    )
    sorted_lengths = sorted(output_lengths)
    metrics.update(
        {
            "decoded_response_count": len(output_lengths),
            "decoded_output_tokens_mean": (
                sum(output_lengths) / len(output_lengths) if output_lengths else None
            ),
            "decoded_output_tokens_p50": (
                sorted_lengths[(len(sorted_lengths) - 1) // 2]
                if sorted_lengths
                else None
            ),
            "decoded_output_tokens_max": max(output_lengths, default=None),
            "decoded_token_cap_hits": cap_hits,
            "decoded_token_cap_hit_rate": (
                cap_hits / len(output_lengths) if output_lengths else None
            ),
            "decoded_response_scope": response_scope,
        }
    )
    return metrics


def _validate_bfcl_web_environment(workload: dict) -> None:
    """Fail before generation when a BFCL-harness web run is misconfigured."""
    if workload.get("tool_execution_owner") != "bfcl_harness":
        return
    if str(workload.get("generate_mode", "cli")) != "cli":
        raise ValueError("BFCL-Web harness execution requires generate_mode='cli'.")
    if str(workload.get("tool_mode", "return")) != "return":
        raise ValueError(
            "BFCL-Web harness execution requires tool_mode='return' so the BFCL "
            "harness owns tool execution."
        )
    search_backend = str(workload.get("search_backend", "")).strip().lower()
    if search_backend not in BFCL_WEB_SEARCH_BACKENDS:
        raise ValueError(
            "BFCL-Web search_backend must be one of "
            f"{BFCL_WEB_SEARCH_BACKENDS}, got {search_backend!r}."
        )
    if int(workload.get("max_turns", 20)) != 20:
        raise ValueError("BFCL-Web 2026.3.23 uses max_turns=20.")
    max_new_tokens = workload.get("max_new_tokens")
    if max_new_tokens is not None and int(max_new_tokens) < 4096:
        raise ValueError(
            "Official QwenFCHandler requests up to 4096 tokens per step; "
            "max_new_tokens must be at least 4096."
        )
    num_threads = int(workload.get("num_threads", 1))
    if search_backend == "bfcl_web_search_api_serpapi_duckduckgo":
        if bool(workload.get("require_serpapi_api_key", True)) and not os.getenv(
            "SERPAPI_API_KEY"
        ):
            raise RuntimeError(
                "Official BFCL-Web requires SERPAPI_API_KEY for BFCL's "
                "WebSearchAPI. Select search_backend="
                "'bfcl_harness_ddgs_duckduckgo' for the free controlled "
                "DuckDuckGo transport."
            )
        return

    if num_threads != 1:
        raise ValueError(
            "Direct-DDGS BFCL-Web uses the correctness-first serial protocol; "
            "set num_threads=1."
        )
    if int(workload.get("search_max_concurrency", 1)) <= 0:
        raise ValueError("BFCL-Web search_max_concurrency must be positive.")
    cache_mode = str(workload.get("search_cache_mode", "off")).strip().lower()
    if cache_mode not in BFCL_WEB_SEARCH_CACHE_MODES:
        raise ValueError(
            "BFCL-Web search_cache_mode must be one of "
            f"{BFCL_WEB_SEARCH_CACHE_MODES}, got {cache_mode!r}."
        )
    if cache_mode != "off" and not workload.get("search_cache_dir"):
        raise ValueError(
            "Direct DDGS BFCL-Web runs require search_cache_dir unless "
            "search_cache_mode='off'."
        )


def _validate_official_bfcl_web_environment(workload: dict) -> None:
    """Backward-compatible alias for existing callers and tests."""
    _validate_bfcl_web_environment(workload)


def _release_persistent_task_cuda_memory(label: str) -> None:
    """Release transient allocations while keeping model weights resident."""
    gc.collect()
    if not torch.cuda.is_available():
        return

    device_stats = []
    for device_index in range(torch.cuda.device_count()):
        allocated_before = torch.cuda.memory_allocated(device_index)
        reserved_before = torch.cuda.memory_reserved(device_index)
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved(device_index)
        device_stats.append(
            "cuda:%d allocated=%.2fGiB reserved=%.2f->%.2fGiB"
            % (
                device_index,
                allocated_before / (1024**3),
                reserved_before / (1024**3),
                reserved_after / (1024**3),
            )
        )
    logger.info("Persistent task CUDA cleanup (%s): %s", label, "; ".join(device_stats))


def _bfcl_category_metrics(scores: dict, category: str) -> dict:
    entry = scores.get("per_category", {}).get(category, {})
    if isinstance(entry, list):
        summary = entry[0] if entry and isinstance(entry[0], dict) else {}
    elif isinstance(entry, dict):
        summary = entry
    else:
        summary = {}
    return {
        "accuracy": summary.get("accuracy"),
        "correct": int(summary.get("correct_count", 0)),
        "total": int(summary.get("total_count", 0)),
    }


def _bfcl_raw_stats(result_dir: Path, category: str, max_new_tokens: int) -> dict:
    candidates = sorted(result_dir.rglob(f"BFCL_v4_{category}_result.json"))
    if not candidates:
        return {
            "sample_count": 0,
            "token_cap_hits": 0,
            "tool_error_records": 0,
            "inference_error_records": 0,
            "result_path": None,
        }
    path = candidates[0]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def flattened_numbers(value: object) -> list[int]:
        if isinstance(value, bool):
            return []
        if isinstance(value, (int, float)):
            return [int(value)]
        if isinstance(value, list):
            result: list[int] = []
            for item in value:
                result.extend(flattened_numbers(item))
            return result
        if isinstance(value, dict):
            result = []
            for item in value.values():
                result.extend(flattened_numbers(item))
            return result
        return []

    cap_hits = 0
    error_records = 0
    inference_error_records = 0
    for record in records:
        output_counts = flattened_numbers(record.get("output_token_count"))
        cap_hits += int(any(count >= max_new_tokens for count in output_counts))
        serialized = json.dumps(record, ensure_ascii=False).lower()
        error_records += int('"error"' in serialized or "tool_error" in serialized)
        result_text = str(record.get("result", "")).lower()
        inference_error_records += int(
            "error during inference" in result_text
            or "internal server error" in result_text
        )
    return {
        "sample_count": len(records),
        "token_cap_hits": cap_hits,
        "tool_error_records": error_records,
        "inference_error_records": inference_error_records,
        "result_path": str(path),
    }


def _run_persistent_generation_suite(
    *,
    model,
    tokenizer,
    workloads: list[dict],
    switch: PhaseLayerAutoSwitch | None,
    model_name: str,
    model_alias: str | None,
    server_host: str,
    server_port: int,
    gptq_cache_info: dict,
    decode_weight_mode: str,
    decode_weight_residency: str,
    attention_backend: str,
) -> dict:
    """Run report tasks while retaining one prepared model for the profile."""
    suite_results = {
        "model_name": model_name,
        "profile_id": workloads[0].get("profile_id") if workloads else None,
        "attention_backend": attention_backend,
        "decode_weight_mode": decode_weight_mode,
        "decode_weight_residency": decode_weight_residency,
        "gptq_cache": gptq_cache_info,
        "tasks": [],
    }
    for workload in workloads:
        task = str(workload["task"])
        requested_memory_mode = str(
            workload.get("attention_memory_mode", "full")
        ).strip().lower()
        controller = getattr(
            model, "_plena_attention_query_chunk_controller", None
        )
        if isinstance(controller, AttentionQueryChunkController):
            controller.set_mode(requested_memory_mode)
        elif requested_memory_mode != "full":
            raise RuntimeError(
                f"Persistent task {task!r} requested attention memory mode "
                f"{requested_memory_mode!r}, but the model has no unified "
                "Qwen attention chunk controller."
            )
        task_dir = Path(str(workload["output_dir"])).resolve()
        task_dir.mkdir(parents=True, exist_ok=True)
        expected_count = int(workload["expected_count"])
        protocol = {
            key: value
            for key, value in workload.items()
            if key not in {"output_dir"}
        }
        prior = _task_success(task_dir, protocol, expected_count)
        if prior is not None:
            logger.info("Skipping completed persistent task %s: %s", task, task_dir)
            suite_results["tasks"].append(prior)
            continue

        _release_persistent_task_cuda_memory(f"before {task}")
        if switch is not None:
            switch._on_phase_transition("prefill", None)
            switch._phase[0] = "prefill"

        started = time.time()
        logger.info("Starting persistent evaluation task %s (expected=%d)", task, expected_count)
        if task == "humaneval":
            from quant_eval.eval.evalplus import (
                evaluate_with_evalplus,
                summarize_evalplus_results,
            )

            raw_results = evaluate_with_evalplus(
                model=model,
                tokenizer=tokenizer,
                dataset="humaneval",
                batch_size=int(workload.get("batch_size", 1)),
                greedy=True,
                n_samples=1,
                max_new_tokens=int(workload["max_new_tokens"]),
                output_dir=str(task_dir / "evalplus"),
                parallel=int(workload.get("scoring_workers", 16)),
                version=str(workload.get("version", "default")),
                limit=workload.get("limit"),
                batch_length_bucket=bool(
                    workload.get("batch_length_bucket", True)
                ),
                adaptive_batch=bool(workload.get("adaptive_batch", True)),
            )
            metrics = summarize_evalplus_results(raw_results)
            sample_count = int(raw_results.get("generated_count", metrics["total"]))
            result = {
                "task": task,
                "protocol": protocol,
                "metrics": metrics,
                "sample_count": sample_count,
                "partial": bool(raw_results.get("partial", False)),
                "samples_path": raw_results.get("samples_path"),
                "batch_stats": raw_results.get("batch_stats"),
                "elapsed_sec": time.time() - started,
            }
            _write_json_atomic(task_dir / "results.json", result)
        elif task == "gsm8k":
            from quant_eval.eval.lm_eval import evaluate_with_lm_eval

            requested_batch_size = max(1, int(workload.get("batch_size", 1)))
            current_batch_size = requested_batch_size
            attempted_batch_sizes = []
            adaptive_batch = bool(workload.get("adaptive_batch", True))
            while True:
                attempted_batch_sizes.append(current_batch_size)
                oom_message = None
                try:
                    raw_results = evaluate_with_lm_eval(
                        model=model,
                        tokenizer=tokenizer,
                        tasks="gsm8k",
                        max_length=int(workload.get("max_length", 32768)),
                        batch_size=current_batch_size,
                        log_samples=True,
                        limit=workload.get("limit"),
                        num_fewshot=int(workload.get("num_fewshot", 5)),
                        apply_chat_template=bool(
                            workload.get("apply_chat_template", False)
                        ),
                        fewshot_as_multiturn=bool(
                            workload.get("fewshot_as_multiturn", False)
                        ),
                        enable_thinking=workload.get("enable_thinking"),
                        think_end_token=workload.get("think_end_token"),
                        chat_template_args=workload.get("chat_template_args"),
                        gen_kwargs=dict(
                            workload.get("gen_kwargs")
                            or {
                                "max_gen_toks": int(workload["max_new_tokens"]),
                                "do_sample": False,
                            }
                        ),
                        random_seed=int(workload.get("random_seed", 0)),
                        numpy_random_seed=int(
                            workload.get("numpy_random_seed", 1234)
                        ),
                        torch_random_seed=int(
                            workload.get("torch_random_seed", 1234)
                        ),
                        fewshot_random_seed=int(
                            workload.get("fewshot_random_seed", 1234)
                        ),
                    )
                except Exception as error:
                    if (
                        not adaptive_batch
                        or current_batch_size <= 1
                        or not _is_cuda_oom(error)
                    ):
                        raise
                    oom_message = str(error).splitlines()[0]

                if oom_message is None:
                    break
                next_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "GSM8K CUDA OOM at batch_size=%d; retrying at batch_size=%d (%s)",
                    current_batch_size,
                    next_batch_size,
                    oom_message,
                )
                current_batch_size = next_batch_size
                _release_persistent_task_cuda_memory("GSM8K OOM retry")

            metrics = _gsm8k_metrics(
                raw_results,
                tokenizer=tokenizer,
                max_new_tokens=int(workload["max_new_tokens"]),
                response_scope=(
                    "post_thinking_strip"
                    if workload.get("think_end_token") is not None
                    else "full_decoded_response"
                ),
            )
            sample_count = int(metrics["sample_count"])
            result = {
                "task": task,
                "protocol": protocol,
                "metrics": metrics,
                "sample_count": sample_count,
                "batch_stats": {
                    "requested_batch_size": requested_batch_size,
                    "effective_batch_size": current_batch_size,
                    "attempted_batch_sizes": attempted_batch_sizes,
                    "adaptive_batch": adaptive_batch,
                },
                "elapsed_sec": time.time() - started,
            }
            save_results(task_dir, raw_results)
            _write_json_atomic(task_dir / "summary.json", result)
        elif task == "bfcl_web":
            _validate_official_bfcl_web_environment(workload)
            category = str(workload.get("category", "web_search_base"))
            result_dir = task_dir / "bfcl_results"
            score_dir = task_dir / "bfcl_scores"
            result_dir.mkdir(parents=True, exist_ok=True)
            score_dir.mkdir(parents=True, exist_ok=True)
            num_threads = max(1, int(workload.get("num_threads", 1)))
            backend_stats_path = workload.get(
                "search_stats_path", task_dir / "backend_stats.json"
            )
            gen_rc = _run_bfcl_generate(
                model_name=model_name,
                test_categories=[category],
                host=server_host,
                port=server_port,
                result_dir=result_dir,
                num_threads=num_threads,
                limit=workload.get("limit"),
                model_alias=model_alias,
                temperature=float(workload.get("temperature", 0.001)),
                allow_overwrite=True,
                web_search_backend=str(
                    workload.get(
                        "search_backend",
                        "bfcl_web_search_api_serpapi_duckduckgo",
                    )
                ),
                web_search_cache_mode=str(
                    workload.get("search_cache_mode", "off")
                ),
                web_search_cache_dir=workload.get("search_cache_dir"),
                web_search_stats_path=backend_stats_path,
                web_search_max_concurrency=int(
                    workload.get("search_max_concurrency", 1)
                ),
            )
            eval_rc, raw_scores = _run_bfcl_evaluate(
                model_name=model_name,
                test_categories=[category],
                result_dir=result_dir,
                score_dir=score_dir,
                model_alias=model_alias,
                partial_eval=workload.get("limit") is not None,
            )
            raw_stats = _bfcl_raw_stats(
                result_dir, category, int(workload["max_new_tokens"])
            )
            selected_search_backend = str(
                workload.get(
                    "search_backend",
                    "bfcl_web_search_api_serpapi_duckduckgo",
                )
            )
            web_backend_stats = {}
            backend_stats_file = Path(str(backend_stats_path))
            if (
                selected_search_backend == "bfcl_harness_ddgs_duckduckgo"
                and backend_stats_file.is_file()
            ):
                web_backend_stats = json.loads(
                    backend_stats_file.read_text(encoding="utf-8")
                )
            elif selected_search_backend == "bfcl_harness_ddgs_duckduckgo":
                web_backend_stats = {
                    "backend": selected_search_backend,
                    "errors": 1,
                    "status": "missing_backend_stats",
                }
            metrics = _bfcl_category_metrics(raw_scores, category)
            sample_count = int(raw_stats["sample_count"])
            result = {
                "task": task,
                "protocol": protocol,
                "metrics": metrics,
                "raw_stats": raw_stats,
                "sample_count": sample_count,
                "generate_returncode": gen_rc,
                "evaluate_returncode": eval_rc,
                "scores": raw_scores,
                "search_backend": selected_search_backend,
                "web_search_backend": web_backend_stats,
                "elapsed_sec": time.time() - started,
            }
            _write_json_atomic(task_dir / "results.json", result)
            if (
                gen_rc != 0
                or eval_rc != 0
                or int(raw_stats.get("inference_error_records", 0)) > 0
                or int(web_backend_stats.get("errors", 0)) > 0
            ):
                raise RuntimeError(
                    "BFCL Web failed: "
                    f"generate rc={gen_rc}, evaluate rc={eval_rc}, "
                    "inference errors="
                    f"{raw_stats.get('inference_error_records', 0)}, "
                    "web backend errors="
                    f"{web_backend_stats.get('errors', 0)}."
                )
        else:
            raise ValueError(f"Unsupported persistent workload: {task!r}.")

        if sample_count != expected_count:
            raise RuntimeError(
                f"Incomplete {task} result: {sample_count}/{expected_count} samples in {task_dir}."
            )
        result["attention_memory"] = get_attention_memory_stats(model)
        success = {
            **result,
            "status": "complete",
            "protocol_digest": _json_digest(protocol),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json_atomic(task_dir / "task_success.json", success)
        suite_results["tasks"].append(success)
        _release_persistent_task_cuda_memory(f"after {task}")

    suite_results["complete"] = len(suite_results["tasks"]) == len(workloads)
    suite_results["attention_memory"] = get_attention_memory_stats(model)
    return suite_results


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main(
    model_name:  str = "Qwen/Qwen3-8B",
    device_id:   str = "cuda:0",
    dtype:       str = "bfloat16",
    quant_config: str = "quant_eval/configs/qwen3_mxint16.toml",
    model_parallel: bool = False,
    attn_implementation: str = "auto",
    attention_memory_mode: str = "full",
    attention_query_chunk_size: int = 512,
    attention_query_chunk_min_size: int = 128,
    attention_chunk_oom_fallback: bool = True,
    # ── BFCL settings ──────────────────────────────────────────────────────
    bfcl_test_categories: Union[list[str], str, None] = None,
    bfcl_model_alias:     str | None = "Qwen/Qwen3-8B-FC",
    bfcl_adapter:        str = "auto",
    model_family:        str = "qwen3",
    bfcl_num_threads:     int   = 1,
    server_host:          str   = DEFAULT_HOST,
    server_port:          int   = DEFAULT_PORT,
    bfcl_tool_mode:       str   = "auto",
    bfcl_max_new_tokens:  int | None = 2048,
    bfcl_generate_mode:   str = "cli",
    bfcl_batch_size:      int = 16,
    bfcl_batch_length_bucket: bool = True,
    bfcl_disable_thinking: bool = False,
    bfcl_web_search_backend: str = "bfcl_web_search_api_serpapi_duckduckgo",
    bfcl_web_search_cache_mode: str = "off",
    bfcl_web_search_cache_dir: str | None = None,
    bfcl_web_search_stats_path: str | None = None,
    bfcl_web_search_max_concurrency: int = 1,
    # ── GPTQ calibration/weight quantization ───────────────────────────────
    gptq_dataset:          str | None = None,
    gptq_nsamples:         int = 32,
    gptq_seqlen:           int = 1024,
    gptq_format:           str = "mxint",
    gptq_weight_width:     int = 8,
    gptq_weight_exponent_width: int | None = None,
    gptq_weight_frac_width: int | None = None,
    gptq_weight_block_size:int = 32,
    gptq_cali_batch_size:  int = 1,
    gptq_max_layers:       int | None = None,
    gptq_device_map_aware: bool = False,
    gptq_cache_dir:        str | None = None,
    gptq_cache_mode:       str = "off",
    # ── Legacy width controls; overridden by precision tokens below ───────
    prefill_attn_width:      int = 4,
    prefill_ffn_width:       int = 4,
    prefill_attn_block_size: int = 32,
    prefill_ffn_block_size:  int = 32,
    decode_attn_width:       int = 8,
    decode_ffn_width:        int = 8,
    decode_attn_block_size:  int = 32,
    decode_ffn_block_size:   int = 32,
    # ── Codesign-style precision controls ──────────────────────────────────
    act_element_width_prefill: str | None = None,
    act_element_width_decode:  str | None = None,
    kv_element_width_prefill:  str | None = None,
    kv_element_width_decode:   str | None = None,
    fp_setting_prefill:        str | None = None,
    fp_setting_decode:         str | None = None,
    dse_mx_block_size:         int = 16,
    dse_weight_precision:      str | None = None,
    dse_weight_block_size:     int | None = None,
    decode_weight_mode:        str = "quantized",
    decode_weight_residency:   str = "disk_reload",
    runtime_weight_only:       bool = False,
    runtime_bypass_components: str | None = None,
    # ── Optional keyword overrides ─────────────────────────────────────────
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords:  Union[list[str], None] = None,
    limit: Union[int, None] = None,
    log_dir: Union[str, None] = None,
    run_evaluate: bool = True,
    persistent_trials: list[dict] | None = None,
    persistent_progress_path: str | None = None,
    persistent_workloads: list[dict] | None = None,
    persistent_suite_output_path: str | None = None,
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
        quant_config: Path to a TOML quantization recipe. Use ``"none"``
            for a true FP baseline that skips quantization and phase switching.
        model_parallel: Distribute across GPUs with ``device_map="auto"``.
        attn_implementation: HF attention backend used when loading the model.
            ``"auto"`` keeps FP baselines on SDPA but uses eager for quantized
            Qwen3/Qwen3-MoE paths where runtime wrappers insert quantization
            inside the attention core.
        attention_memory_mode: ``"full"`` preserves the existing eager
            attention allocation. ``"query_chunked"`` computes complete-key
            attention for bounded query slices to avoid materializing full
            ``Q x K`` score/probability tensors.
        attention_query_chunk_size: Initial query chunk size for chunked Qwen
            attention.
        attention_query_chunk_min_size: Lowest chunk size used by local CUDA
            OOM fallback.
        attention_chunk_oom_fallback: Halve the query chunk after a CUDA OOM,
            down to ``attention_query_chunk_min_size``.
        bfcl_test_categories: BFCL category names to evaluate (e.g.
            ``["web_search_base", "web_search_no_snippet"]``). ``None`` uses
            the default web-search category set. Also accepts JSON-list string
            or comma-separated categories.
        bfcl_model_alias: BFCL result-file model alias. If set, this exact
            value is used for ``bfcl generate/evaluate --model``.
        bfcl_adapter: Response adapter for BFCL handler protocol. ``auto`` maps
            Qwen3 aliases to official Qwen FC ``<tool_call>`` output and Llama
            aliases to the legacy Llama 3.1 payload normalizer.
        model_family: Quantized model family for DSE config generation. Supported
            values are ``qwen3`` and ``llama``.
        bfcl_num_threads: Parallel inference threads for ``bfcl generate``.
        bfcl_generate_mode: ``"cli"`` uses the official BFCL generator;
            ``"batched"`` directly writes BFCL multiple results from batched
            in-process HF generation.
        bfcl_disable_thinking: In batched Qwen3 generation, prefill an empty
            thinking block after the assistant prefix, matching the tokenizer
            template's ``enable_thinking=False`` behavior.
        bfcl_web_search_backend: Tool transport used by the BFCL harness.
            The default is BFCL's SerpAPI-backed DuckDuckGo implementation;
            ``bfcl_harness_ddgs_duckduckgo`` uses direct DDGS without an API
            key and is reported as a controlled, non-leaderboard protocol.
        bfcl_web_search_cache_mode: Direct-DDGS cache policy: off, record,
            replay, or auto (replay hits and record misses).
        bfcl_web_search_cache_dir: Shared response-cache directory for direct
            DDGS search and URL-fetch tool calls.
        bfcl_web_search_stats_path: Optional per-run JSON file receiving DDGS
            live-call, cache-hit, cache-miss, and error counters.
        bfcl_web_search_max_concurrency: Maximum number of simultaneous live
            direct-DDGS search requests.
        server_host: Host for the local OpenAI-compatible server.
        server_port: Port for the local OpenAI-compatible server.
        bfcl_tool_mode: ``"auto"`` and ``"return"`` return model tool calls to
            the BFCL harness, which owns official multi-turn tool execution.
            ``"execute"`` preserves the legacy PLENA in-server tool loop and is
            not directly comparable to official BFCL-Web results.
        bfcl_max_new_tokens: Local cap applied to BFCL-requested ``max_tokens``
            before calling ``model.generate``. ``None`` disables the cap.
        gptq_dataset: Optional GPTQ calibration dataset. Plain paths are treated
            as ``file:<path>`` for the GPTQ loader. When set, CLI GPTQ values
            override any ``[gptq]`` block in ``quant_config``.
        gptq_nsamples: Number of GPTQ calibration samples.
        gptq_seqlen: GPTQ calibration sequence length.
        gptq_format: GPTQ target format, for example ``"mxint"``.
        gptq_weight_width: GPTQ quantized weight width.
        gptq_weight_exponent_width/gptq_weight_frac_width: GPTQ MXFP element
            exponent/fraction widths when ``gptq_format="mxfp"``.
        gptq_weight_block_size: GPTQ quantized weight block size.
        gptq_cali_batch_size: GPTQ calibration batch size.
        gptq_max_layers: Optional layer cap for quick smoke tests.
        gptq_cache_dir: Optional directory for load-only GPTQ weight cache.
        gptq_cache_mode: One of off/auto/refresh/require.
        prefill_attn_width/prefill_ffn_width/decode_attn_width/decode_ffn_width:
            Legacy MXInt width controls. These are ignored for any phase where
            codesign-style precision tokens are provided.
        act_element_width_prefill/act_element_width_decode: Codesign ACT precision
            tokens, e.g. ``MXINT_4`` or ``MXFP_E4M3``.
        kv_element_width_prefill/kv_element_width_decode: Codesign KV precision
            tokens.
        fp_setting_prefill/fp_setting_decode: Codesign nonlinear minifloat
            tokens for RoPE/softmax/SiLU/RMSNorm, e.g. ``FP_E3M2``.
        dse_mx_block_size: MX block size used by codesign-style precision tokens.
        decode_weight_mode: ``"quantized"`` keeps current decode behavior;
            ``"fp"`` loads original checkpoint weights for decode Linear
            layers and bypasses Linear activation quantization.
        decode_weight_residency: ``"disk_reload"`` keeps the legacy FP decode
            weight reload path. ``"host_cached"`` retains safetensors mmaps and
            uses Linux page cache plus per-device parallel copies without a
            second GPU-resident weight set. ``"gpu_dual"`` keeps original FP
            weights in GPU-resident wrapper buffers and phase-switches by flag.
        runtime_weight_only: Debug/sanity mode for quantized runs. If true,
            keep GPTQ/fake-quantized weights but bypass runtime activation,
            KV, attention nonlinear, MLP nonlinear, and RMSNorm quantization.
        runtime_bypass_components: Optional comma-separated component list for
            debugging runtime quantization. Valid entries: attn, ffn, mlp,
            rms_norm, all. This bypasses the selected runtime components in
            both prefill and decode while keeping weight quantization intact.
        attn_keywords: Module-name substrings that identify attention blocks.
            ``None`` uses the built-in defaults.
        ffn_keywords: Module-name substrings that identify FFN blocks. ``None``
            uses the built-in defaults.
        limit: Cap the number of samples per category. ``None`` = full dataset.
        log_dir: Directory for ``args.json`` and ``results.json``.
        run_evaluate: Run ``bfcl evaluate`` after generation. Set to ``False``
            when only raw generation artifacts, for example token length
            statistics, are needed.
        persistent_workloads: Optional generation-suite task specifications.
            The prepared model remains resident while HumanEval, GSM8K and
            BFCL-Web execute sequentially.

    Returns:
        BFCL evaluation summary — per-category scores plus aggregate metrics,
        with ``phase_layer_configs`` recording the resolved (phase, layer)
        precision table.
    """
    bfcl_test_categories = _normalize_bfcl_categories(bfcl_test_categories)
    bfcl_tool_mode = _resolve_bfcl_tool_mode(bfcl_test_categories, bfcl_tool_mode)
    bfcl_generate_mode = str(bfcl_generate_mode or "cli").lower()
    if bfcl_generate_mode not in BFCL_GENERATE_MODES:
        raise ValueError(f"bfcl_generate_mode must be one of {BFCL_GENERATE_MODES}, got {bfcl_generate_mode!r}.")
    if bfcl_generate_mode == "batched":
        if bfcl_test_categories != ["multiple"] or bfcl_tool_mode != "return":
            raise ValueError(
                "bfcl_generate_mode='batched' currently supports only "
                "bfcl_test_categories='multiple' with bfcl_tool_mode='return'. "
                "Use bfcl_generate_mode='cli' for other BFCL categories."
            )
        if int(bfcl_batch_size) <= 0:
            raise ValueError("bfcl_batch_size must be positive.")
    if (
        bfcl_tool_mode == "return"
        and any(category in BFCL_WEB_SEARCH_CATEGORIES for category in bfcl_test_categories)
        and (
            persistent_workloads is None
            or any(
                str(workload.get("task")) == "bfcl_web"
                for workload in persistent_workloads
            )
        )
    ):
        persistent_web_workload = next(
            (
                workload
                for workload in (persistent_workloads or [])
                if str(workload.get("task")) == "bfcl_web"
            ),
            None,
        )
        _validate_bfcl_web_environment(
            persistent_web_workload
            or {
                "tool_execution_owner": "bfcl_harness",
                "generate_mode": bfcl_generate_mode,
                "tool_mode": bfcl_tool_mode,
                "search_backend": bfcl_web_search_backend,
                "search_cache_mode": bfcl_web_search_cache_mode,
                "search_cache_dir": bfcl_web_search_cache_dir,
                "search_max_concurrency": bfcl_web_search_max_concurrency,
                "num_threads": bfcl_num_threads,
                "max_turns": 20,
                "max_new_tokens": bfcl_max_new_tokens,
                "require_serpapi_api_key": (
                    bfcl_web_search_backend
                    == "bfcl_web_search_api_serpapi_duckduckgo"
                ),
            }
        )
    bfcl_adapter_obj = resolve_bfcl_adapter(bfcl_adapter, model_name=model_name, model_alias=bfcl_model_alias)
    resolved_bfcl_adapter = bfcl_adapter_obj.name
    model_family = model_family.lower()
    qwen_model_family = model_family in {"qwen3", "qwen3_moe"}
    decode_weight_mode = decode_weight_mode.lower()
    if decode_weight_mode not in ("quantized", "fp"):
        raise ValueError(
            "decode_weight_mode must be 'quantized' or 'fp', "
            f"got {decode_weight_mode!r}."
        )
    decode_weight_residency = str(decode_weight_residency or "disk_reload").lower()
    if decode_weight_residency not in {"disk_reload", "host_cached", "gpu_dual"}:
        raise ValueError(
            "decode_weight_residency must be 'disk_reload', 'host_cached', "
            "or 'gpu_dual', "
            f"got {decode_weight_residency!r}."
        )
    if (
        decode_weight_residency in {"host_cached", "gpu_dual"}
        and decode_weight_mode != "fp"
    ):
        raise ValueError(
            f"decode_weight_residency={decode_weight_residency!r} requires "
            "decode_weight_mode='fp'."
        )
    device_id = _normalize_device_id(device_id)
    quant_config_is_none = quant_config is None or str(quant_config).strip().lower() in {"", "none", "fp", "false"}
    if quant_config_is_none:
        quant_config = "none"
        quant_related = {
            "gptq_dataset": gptq_dataset,
            "act_element_width_prefill": act_element_width_prefill,
            "act_element_width_decode": act_element_width_decode,
            "kv_element_width_prefill": kv_element_width_prefill,
            "kv_element_width_decode": kv_element_width_decode,
            "fp_setting_prefill": fp_setting_prefill,
            "fp_setting_decode": fp_setting_decode,
            "dse_weight_precision": dse_weight_precision,
        }
        active_quant_related = [name for name, value in quant_related.items() if value is not None]
        if active_quant_related:
            raise ValueError(
                "--quant_config none requests a true FP baseline, but quantization "
                f"options were also provided: {active_quant_related}."
            )
        if decode_weight_mode != "quantized":
            raise ValueError(
                "--decode_weight_mode is only meaningful for quantized runs; "
                "use the default with --quant_config none."
            )
        if decode_weight_residency != "disk_reload":
            raise ValueError("--decode_weight_residency is only meaningful for quantized runs.")

    attention_memory_mode = str(attention_memory_mode or "full").strip().lower()
    if attention_memory_mode not in ATTENTION_MEMORY_MODES:
        raise ValueError(
            f"attention_memory_mode must be one of {ATTENTION_MEMORY_MODES}, "
            f"got {attention_memory_mode!r}."
        )
    if int(attention_query_chunk_size) <= 0 or int(attention_query_chunk_min_size) <= 0:
        raise ValueError("Attention query chunk sizes must be positive integers.")
    if int(attention_query_chunk_min_size) > int(attention_query_chunk_size):
        raise ValueError(
            "attention_query_chunk_min_size cannot exceed "
            "attention_query_chunk_size."
        )
    if attention_memory_mode == "query_chunked" and (
        not qwen_model_family or quant_config_is_none
    ):
        raise ValueError(
            "attention_memory_mode='query_chunked' currently requires a "
            "quantized qwen3 or qwen3_moe unified-wrapper run. FP baselines "
            "should use native SDPA."
        )
    # ------------------------------------------------------------------
    # Build the nested phase × layer/nonlinear config
    # ------------------------------------------------------------------
    decode_weight_policy = {}
    if decode_weight_mode == "fp":
        decode_weight_policy = {"weight_mode": "fp", "bypass": True}

    def _legacy_mx_config(width: int, block_size: int) -> dict:
        return {"data_in_width": width, "data_in_block_size": block_size}

    def _resolve_phase_precision(phase: str) -> dict:
        if phase == "prefill":
            act = act_element_width_prefill
            kv = kv_element_width_prefill
            fp = fp_setting_prefill
            legacy_attn = _legacy_mx_config(prefill_attn_width, prefill_attn_block_size)
            legacy_ffn = _legacy_mx_config(prefill_ffn_width, prefill_ffn_block_size)
        else:
            act = act_element_width_decode
            kv = kv_element_width_decode
            fp = fp_setting_decode
            legacy_attn = _legacy_mx_config(decode_attn_width, decode_attn_block_size)
            legacy_ffn = _legacy_mx_config(decode_ffn_width, decode_ffn_block_size)

        provided = {"ACT_ELEMENT_WIDTH": act, "KV_ELEMENT_WIDTH": kv, "FP_SETTING": fp}
        if any(v is not None for v in provided.values()) and not all(v is not None for v in provided.values()):
            missing = [name for name, value in provided.items() if value is None]
            raise ValueError(
                f"{phase} DSE precision requires ACT_ELEMENT_WIDTH, "
                f"KV_ELEMENT_WIDTH, and FP_SETTING together; missing {missing}."
            )

        if act is None and kv is None and fp is None:
            if qwen_model_family and not quant_config_is_none:
                # Qwen-first default: use the same full nonlinear precision
                # semantics as explicit DSE points, but choose a conservative
                # high-precision tuple.  This keeps RoPE/softmax, MLP SiLU, and
                # RMSNorm input/weight under FP_SETTING instead of silently
                # bypassing them in the default Qwen path.
                act, kv, fp = "MXINT_8", "MXINT_8", "FP_E8M5"
            else:
                attn_cfg = dict(legacy_attn)
                return {
                    "attn": attn_cfg,
                    "ffn": dict(legacy_ffn),
                    "mlp": {},
                    "rms_norm": {},
                    "display": f"MXInt{legacy_attn['data_in_width']}(bs={legacy_attn['data_in_block_size']})",
                    "ffn_display": f"MXInt{legacy_ffn['data_in_width']}(bs={legacy_ffn['data_in_block_size']})",
                    "metadata": {
                        "ACT_ELEMENT_WIDTH": f"MXINT_{legacy_attn['data_in_width']}",
                        "KV_ELEMENT_WIDTH": f"MXINT_{legacy_attn['data_in_width']}",
                        "FP_SETTING": None,
                    },
                }

        act_spec = parse_mx_precision(act or "MXINT_4")
        kv_spec = parse_mx_precision(kv or act_spec.canonical)
        fp_spec = parse_fp_setting(fp or "FP_E3M2")
        act_cfg = mx_data_config(act_spec, dse_mx_block_size)
        kv_cfg = mx_data_config(kv_spec, dse_mx_block_size)
        fp_cfg = fp_data_config(fp_spec)
        attn_cfg = {**act_cfg, "kv_cache": dict(kv_cfg), "softmax": dict(fp_cfg), "rope": dict(fp_cfg)}
        ffn_cfg = dict(act_cfg)
        mlp_cfg = dict(fp_cfg)
        rms_cfg = {
            **fp_cfg,
            "weight_exponent_width": fp_spec.exp,
            "weight_frac_width": fp_spec.frac,
            "weight_is_finite": True,
            "weight_round_mode": "rn",
        }
        return {
            "attn": attn_cfg,
            "ffn": ffn_cfg,
            "mlp": mlp_cfg,
            "rms_norm": rms_cfg,
            "display": f"{act_spec.canonical}/KV={kv_spec.canonical}/NL={fp_spec.canonical}(B{dse_mx_block_size})",
            "ffn_display": f"{act_spec.canonical}/NL={fp_spec.canonical}(B{dse_mx_block_size})",
            "metadata": {
                "ACT_ELEMENT_WIDTH": act_spec.canonical,
                "KV_ELEMENT_WIDTH": kv_spec.canonical,
                "FP_SETTING": fp_spec.canonical,
            },
        }

    _prefill_precision = _resolve_phase_precision("prefill")
    _decode_precision = _resolve_phase_precision("decode")

    def _resolve_dse_weight_precision() -> tuple[str, int | None]:
        # ACT/KV/FP_SETTING DSE should not silently lower projection weight
        # precision. If GPTQ is enabled, keep the module-replacement weight
        # config aligned with the GPTQ CLI weight format so non-GPTQ layers in
        # max_layers smoke runs are not accidentally PTQ'd to INT4.
        if dse_weight_precision is not None:
            return dse_weight_precision, dse_weight_block_size
        if gptq_dataset is not None:
            if gptq_format.lower() != "mxint":
                raise ValueError(
                    "dse_weight_precision must be set explicitly when "
                    f"gptq_format={gptq_format!r}; only mxint can be inferred "
                    "from gptq_weight_width."
                )
            return f"MXINT_{gptq_weight_width}", (
                dse_weight_block_size if dse_weight_block_size is not None else gptq_weight_block_size
            )
        return "MXINT_8", dse_weight_block_size

    _resolved_dse_weight_precision, _resolved_dse_weight_block_size = _resolve_dse_weight_precision()

    decode_nonlinear_policy = {"bypass": True} if decode_weight_mode == "fp" else {}
    phase_configs = {
        "prefill": {
            "attn": _prefill_precision["attn"],
            "ffn": _prefill_precision["ffn"],
            "mlp": _prefill_precision["mlp"],
            "rms_norm": _prefill_precision["rms_norm"],
        },
        "decode": {
            "attn": {**_decode_precision["attn"], **decode_weight_policy},
            "ffn": {**_decode_precision["ffn"], **decode_weight_policy},
            "mlp": {**_decode_precision["mlp"], **decode_nonlinear_policy},
            "rms_norm": {**_decode_precision["rms_norm"], **decode_nonlinear_policy},
        },
    }
    bypass_components = {
        item.strip().lower()
        for item in str(runtime_bypass_components or "").split(",")
        if item.strip()
    }
    if "all" in bypass_components:
        bypass_components.update({"attn", "ffn", "mlp", "rms_norm"})
        bypass_components.discard("all")
    valid_bypass_components = {
        "attn",
        "attn_linear",
        "attn_core",
        "qk",
        "av",
        "kv_cache",
        "softmax",
        "rope",
        "ffn",
        "mlp",
        "rms_norm",
    }
    invalid_bypass_components = bypass_components - valid_bypass_components
    if invalid_bypass_components:
        raise ValueError(
            f"Unsupported runtime_bypass_components={sorted(invalid_bypass_components)}; "
            f"expected entries from {sorted(valid_bypass_components)} or 'all'."
        )
    if runtime_weight_only:
        bypass_components.update(valid_bypass_components)

    def _apply_runtime_bypass(phase: dict) -> dict:
        if not bypass_components:
            return phase
        if quant_config_is_none:
            raise ValueError("runtime bypass controls are only meaningful for quantized runs.")

        if "attn" in bypass_components:
            phase["attn"] = {
                **phase.get("attn", {}),
                "bypass": True,
                "kv_cache": {"bypass": True},
                "softmax": {"bypass": True},
                "rope": {"bypass": True},
            }
        else:
            attn_phase = {**phase.get("attn", {})}
            if "attn_linear" in bypass_components:
                attn_phase["linear_bypass"] = True
            if "attn_core" in bypass_components or "qk" in bypass_components:
                attn_phase["qk_matmul"] = {"bypass": True}
            if "attn_core" in bypass_components or "av" in bypass_components:
                attn_phase["av_matmul"] = {"bypass": True}
            if "attn_core" in bypass_components or "kv_cache" in bypass_components:
                attn_phase["kv_cache"] = {"bypass": True}
            if "attn_core" in bypass_components or "softmax" in bypass_components:
                attn_phase["softmax"] = {"bypass": True}
            if "attn_core" in bypass_components or "rope" in bypass_components:
                attn_phase["rope"] = {"bypass": True}
            phase["attn"] = attn_phase
        if "ffn" in bypass_components:
            phase["ffn"] = {**phase.get("ffn", {}), "bypass": True}
        if "mlp" in bypass_components:
            phase["mlp"] = {"bypass": True}
        if "rms_norm" in bypass_components:
            phase["rms_norm"] = {"bypass": True}
        return phase

    _apply_runtime_bypass(phase_configs["prefill"])
    _apply_runtime_bypass(phase_configs["decode"])

    precision_metadata = {
        "prefill": _prefill_precision["metadata"],
        "decode": _decode_precision["metadata"],
        "dse_mx_block_size": dse_mx_block_size,
        "dse_weight_precision": _resolved_dse_weight_precision,
        "dse_weight_block_size": _resolved_dse_weight_block_size,
        "runtime_weight_only": bool(runtime_weight_only),
        "runtime_bypass_components": sorted(bypass_components),
    }

    def _prefill_phase_from_tokens(act: str, kv: str, fp: str) -> tuple[dict, dict]:
        act_spec = parse_mx_precision(act)
        kv_spec = parse_mx_precision(kv)
        fp_spec = parse_fp_setting(fp)
        act_cfg = mx_data_config(act_spec, dse_mx_block_size)
        kv_cfg = mx_data_config(kv_spec, dse_mx_block_size)
        fp_cfg = fp_data_config(fp_spec)
        rms_cfg = {
            **fp_cfg,
            "weight_exponent_width": fp_spec.exp,
            "weight_frac_width": fp_spec.frac,
            "weight_is_finite": True,
            "weight_round_mode": "rn",
        }
        phase = {
            "attn": {**act_cfg, "kv_cache": dict(kv_cfg), "softmax": dict(fp_cfg), "rope": dict(fp_cfg)},
            "ffn": dict(act_cfg),
            "mlp": dict(fp_cfg),
            "rms_norm": rms_cfg,
        }
        return (
            _apply_runtime_bypass(phase),
            {
                "ACT_ELEMENT_WIDTH": act_spec.canonical,
                "KV_ELEMENT_WIDTH": kv_spec.canonical,
                "FP_SETTING": fp_spec.canonical,
            },
        )

    def _decode_phase_from_tokens(act: str, kv: str, fp: str) -> tuple[dict, dict]:
        phase, meta = _prefill_phase_from_tokens(act, kv, fp)
        phase["attn"] = {**phase["attn"], **decode_weight_policy}
        phase["ffn"] = {**phase["ffn"], **decode_weight_policy}
        phase["mlp"] = {**phase["mlp"], **decode_nonlinear_policy}
        phase["rms_norm"] = {**phase["rms_norm"], **decode_nonlinear_policy}
        return phase, meta

    _qwen3_default_precision_enabled = qwen_model_family and not quant_config_is_none
    _codesign_tokens_enabled = _qwen3_default_precision_enabled or any(v is not None for v in (
        act_element_width_prefill, act_element_width_decode,
        kv_element_width_prefill, kv_element_width_decode,
        fp_setting_prefill, fp_setting_decode,
    ))
    if _codesign_tokens_enabled:
        # Parsing above is the validation. Mixed ACT/KV and prefill/decode MX
        # families are supported by quant_eval's unified MX wrappers, which are
        # installed immediately after the Chop quantization pass.
        parse_mx_precision(precision_metadata["prefill"]["ACT_ELEMENT_WIDTH"])
        parse_mx_precision(precision_metadata["prefill"]["KV_ELEMENT_WIDTH"])
        parse_mx_precision(precision_metadata["decode"]["ACT_ELEMENT_WIDTH"])
        parse_mx_precision(precision_metadata["decode"]["KV_ELEMENT_WIDTH"])

    requested_attn_implementation = str(attn_implementation or "auto").strip().lower()
    resolved_attn_implementation = resolve_attention_backend(
        requested_attn_implementation,
        qwen_model_family=qwen_model_family,
        quant_config_is_none=quant_config_is_none,
        codesign_tokens_enabled=_codesign_tokens_enabled,
    )
    if (
        requested_attn_implementation == "sdpa"
        and qwen_model_family
        and not quant_config_is_none
        and _codesign_tokens_enabled
    ):
        logger.warning(
            "Qwen3/Qwen3-MoE quantized attention was explicitly initialized with SDPA. "
            "PLENA unified attention-core quantization is eager-style; use "
            "attn_implementation='auto' or 'eager' unless you are intentionally "
            "reproducing the SDPA/eager mismatch failure mode."
        )

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    _pa = _prefill_precision["display"]
    _pf = _prefill_precision["ffn_display"]
    if quant_config_is_none:
        _pa = _pf = _da = _df = f"native {dtype}"
    elif decode_weight_mode == "fp":
        _da = "Linear FP/bypass, attn FP/bypass, old KV no-requant"
        _df = "Linear FP/bypass"
    else:
        _da = f"{_decode_precision['display']}, W=quantized"
        _df = f"{_decode_precision['ffn_display']}, W=quantized"

    print("=" * 64)
    print("BFCL Web Search — Phase × Layer-Type Disaggregated Quantization")
    print("=" * 64)
    print(f"  Model      : {model_name}")
    if bfcl_model_alias:
        print(f"  BFCL Alias : {bfcl_model_alias}")
    print(f"  Categories : {bfcl_test_categories}")
    print(f"  Tool mode  : {bfcl_tool_mode}")
    print(f"  Adapter    : {resolved_bfcl_adapter} (requested={bfcl_adapter})")
    print(f"  Family     : {model_family}")
    print(f"  Attention  : {resolved_attn_implementation} (requested={requested_attn_implementation})")
    print(
        f"  Attn memory: {attention_memory_mode}"
        + (
            f" (chunk={int(attention_query_chunk_size)}, "
            f"min={int(attention_query_chunk_min_size)}, "
            f"oom_fallback={bool(attention_chunk_oom_fallback)})"
            if attention_memory_mode == "query_chunked"
            else ""
        )
    )
    print(f"  Max new tok: {bfcl_max_new_tokens if bfcl_max_new_tokens is not None else 'uncapped'}")
    print(f"  Generate   : {bfcl_generate_mode}" + (f" (batch={bfcl_batch_size})" if bfcl_generate_mode == "batched" else ""))
    print("  HTTP mode  : serial")
    if bfcl_generate_mode == "batched":
        print(f"  Thinking   : {'disabled' if bfcl_disable_thinking else 'enabled'}")
    print(f"  Weights    : {'FP baseline (no quantization)' if quant_config_is_none else quant_config}")
    print(f"  Decode W   : {'n/a (native weights)' if quant_config_is_none else decode_weight_mode}")
    print(f"  Weight res : {'n/a' if quant_config_is_none else decode_weight_residency}")
    if bypass_components:
        print(f"  Runtime    : bypass {','.join(sorted(bypass_components))}")
    if gptq_dataset:
        print(f"  GPTQ       : dataset={gptq_dataset}, nsamples={gptq_nsamples}, seqlen={gptq_seqlen}, max_layers={gptq_max_layers}")
        print(f"  GPTQ cache : mode={gptq_cache_mode}, dir={gptq_cache_dir or 'none'}")
    print(f"  Server     : http://{server_host}:{server_port}")
    print()
    print(f"  {'':10s}  {'attn':>24s}  {'ffn':>24s}")
    print(f"  {'prefill':10s}  {_pa:>24s}  {_pf:>24s}")
    print(f"  {'decode':10s}  {_da:>24s}  {_df:>24s}")
    print("=" * 64)
    logger.info("Model Parallel: %s", model_parallel)

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
        if not quant_config_is_none:
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
        attn_implementation=resolved_attn_implementation,
        layer_balanced_device_map=bool(
            model_parallel
            and qwen_model_family
            and (
                decode_weight_residency in {"host_cached", "gpu_dual"}
                or gptq_device_map_aware
            )
        ),
    )
    model.eval()
    app = None

    try:
        # ------------------------------------------------------------------
        # Weight quantization
        # ------------------------------------------------------------------
        resolved_gptq_config = None
        gptq_cache_info = {"mode": str(gptq_cache_mode or "off").lower(), "hit": False}
        if quant_config_is_none:
            logger.info("True FP baseline requested: skipping quantize_module_transform_pass and phase switching.")
            if model_parallel:
                model = move_to_gpu(model, model_parallel)
            else:
                model.to(device_id)
            switch = None
        else:
            from chop.passes.module.transforms import quantize_module_transform_pass

            pass_args = load_quant_config(quant_config)
            if _codesign_tokens_enabled:
                apply_dse_quant_config(
                    pass_args,
                    act_precision=precision_metadata["prefill"]["ACT_ELEMENT_WIDTH"],
                    kv_precision=precision_metadata["prefill"]["KV_ELEMENT_WIDTH"],
                    fp_setting=precision_metadata["prefill"]["FP_SETTING"],
                    mx_block_size=dse_mx_block_size,
                    weight_precision=_resolved_dse_weight_precision,
                    weight_block_size=_resolved_dse_weight_block_size,
                    model_family=model_family,
                )
            resolved_gptq_config = _inject_gptq_config(
                pass_args,
                model_name=model_name,
                device_id=device_id,
                dataset=gptq_dataset,
                nsamples=gptq_nsamples,
                seqlen=gptq_seqlen,
                fmt=gptq_format,
                weight_width=gptq_weight_width,
                weight_exponent_width=gptq_weight_exponent_width,
                weight_frac_width=gptq_weight_frac_width,
                weight_block_size=gptq_weight_block_size,
                cali_batch_size=gptq_cali_batch_size,
                max_layers=gptq_max_layers,
                device_map_aware=bool(gptq_device_map_aware),
            )
            gptq_cache = None
            gptq_cache_info = {"mode": str(gptq_cache_mode or "off").lower(), "hit": False}
            if resolved_gptq_config:
                marked_gptq_configs = _mark_gptq_projection_configs(pass_args)
                logger.info(
                    "GPTQ enabled: dataset=%s nsamples=%s seqlen=%s format=%s max_layers=%s marked_weight_configs=%s",
                    resolved_gptq_config.get("dataset"),
                    resolved_gptq_config.get("nsamples"),
                    resolved_gptq_config.get("seqlen"),
                    resolved_gptq_config.get("format"),
                    resolved_gptq_config.get("max_layers"),
                    marked_gptq_configs,
                )
                normalized_cache_mode = str(gptq_cache_mode or "off").lower()
                if normalized_cache_mode == "memory":
                    gptq_cache_info = {
                        "mode": "memory",
                        "hit": False,
                        "path": "",
                        "loaded_layers": 0,
                        "partial_layers": 0,
                        "resuming": False,
                    }
                elif normalized_cache_mode != "off":
                    if not gptq_cache_dir:
                        raise ValueError("gptq_cache_dir must be set when gptq_cache_mode is not 'off'.")
                    gptq_cache = _GptqWeightCache(
                        cache_dir=gptq_cache_dir,
                        mode=normalized_cache_mode,
                        gptq_config=resolved_gptq_config,
                        total_layers=len(model.model.layers),
                    )
                    cache_hit = gptq_cache.prepare(model)
                    gptq_cache_info = gptq_cache.summary()
                    if cache_hit:
                        pass_args.pop("gptq", None)
                    else:
                        resolved_gptq_config["checkpoint_dir"] = str(gptq_cache.cache_path)

            n_linear = sum(
                1 for _, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)
            )
            logger.info("Quantizing %d linear layers...", n_linear)
            t0 = time.time()
            try:
                qwen_cache_fast_path = bool(
                    qwen_model_family
                    and gptq_cache is not None
                    and gptq_cache.hit
                    and (_codesign_tokens_enabled or qwen_model_family)
                )
                if qwen_cache_fast_path:
                    logger.info("GPTQ cache hit Qwen fast path: skipping Chop transform pass.")
                else:
                    t_chop = time.time()
                    model, _ = quantize_module_transform_pass(model, pass_args)
                    logger.info("Chop quantize_module_transform_pass complete in %.1fs", time.time() - t_chop)
                    if gptq_cache is not None and not gptq_cache.hit:
                        gptq_cache.finalize()
                        gptq_cache_info = gptq_cache.summary()
                        # The cache lock only protects GPTQ cache population.
                        # Runtime wrapper installation is trial-local and can run
                        # concurrently on other workers once metadata is complete.
                        gptq_cache.release()
                if _codesign_tokens_enabled or qwen_model_family:
                    _qwen3_moe_experts_config = None
                    if model_family == "qwen3_moe" and phase_configs["prefill"]["mlp"]:
                        _qwen3_moe_experts_config = {
                            "act": phase_configs["prefill"]["ffn"],
                            "nonlinear": phase_configs["prefill"]["mlp"],
                        }
                    t_unified = time.time()
                    wrapper_kwargs = {
                        "qwen3_attention_config": phase_configs["prefill"]["attn"] if model_family == "qwen3" else None,
                        "qwen3_mlp_config": phase_configs["prefill"]["mlp"] if model_family == "qwen3" and phase_configs["prefill"]["mlp"] else None,
                        "qwen3_rms_norm_config": phase_configs["prefill"]["rms_norm"] if model_family == "qwen3" and phase_configs["prefill"]["rms_norm"] else None,
                        "qwen3_moe_attention_config": phase_configs["prefill"]["attn"] if model_family == "qwen3_moe" else None,
                        "qwen3_moe_experts_config": _qwen3_moe_experts_config,
                        "qwen3_moe_rms_norm_config": phase_configs["prefill"]["rms_norm"] if model_family == "qwen3_moe" and phase_configs["prefill"]["rms_norm"] else None,
                        "attention_memory_mode": attention_memory_mode,
                        "attention_query_chunk_size": int(attention_query_chunk_size),
                        "attention_query_chunk_min_size": int(attention_query_chunk_min_size),
                        "attention_chunk_oom_fallback": bool(attention_chunk_oom_fallback),
                    }
                    if qwen_cache_fast_path:
                        unified_counts = apply_qwen3_gptq_cache_unified_wrappers(
                            model,
                            attn_linear_config=phase_configs["prefill"]["attn"],
                            ffn_linear_config=phase_configs["prefill"]["ffn"],
                            **wrapper_kwargs,
                        )
                    else:
                        unified_counts = apply_unified_mx_wrappers(model, **wrapper_kwargs)
                    logger.info(
                        "Installed unified MX wrappers: %d Linear, %d direct GPTQ Linear, %d attention (llama=%d, qwen3=%d, qwen3_moe=%d), qwen3_mlp=%d, qwen3_rms_norm=%d, qwen3_moe_experts=%d, qwen3_moe_rms_norm=%d",
                        unified_counts.get("linear", 0),
                        unified_counts.get("direct_gptq_linear", 0),
                        unified_counts.get("attention", 0),
                        unified_counts.get("llama_attention", 0),
                        unified_counts.get("qwen3_attention", 0),
                        unified_counts.get("qwen3_moe_attention", 0),
                        unified_counts.get("qwen3_mlp", 0),
                        unified_counts.get("qwen3_rms_norm", 0),
                        unified_counts.get("qwen3_moe_experts", 0),
                        unified_counts.get("qwen3_moe_rms_norm", 0),
                    )
                    logger.info("Unified wrapper install complete in %.1fs", time.time() - t_unified)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                logger.info("Quantization complete in %.1fs", time.time() - t0)
            finally:
                if gptq_cache is not None:
                    gptq_cache.release()

            t_move = time.time()
            if model_parallel:
                model = move_to_gpu(model, model_parallel)
            else:
                model.to(device_id)
            logger.info("Model device placement complete in %.1fs", time.time() - t_move)

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
                switch_kwargs["weight_residency"] = decode_weight_residency
                if decode_weight_residency in {"disk_reload", "host_cached"}:
                    quant_checkpoint_dir = gptq_cache_info.get("path")
                    if quant_checkpoint_dir:
                        switch_kwargs["quant_checkpoint_dir"] = quant_checkpoint_dir

            t_switch = time.time()
            switch = PhaseLayerAutoSwitch(model, phase_configs, **switch_kwargs)
            switch.enable()
            logger.info("Phase switch setup complete in %.1fs", time.time() - t_switch)
            logger.info("\n%s", switch.summary())

        # ------------------------------------------------------------------
        # Start the OpenAI-compatible server (hook fires on every request)
        # ------------------------------------------------------------------
        device_str = device_id if not model_parallel else "cuda"
        if bfcl_generate_mode == "cli":
            app = _build_server_app(
                model,
                tokenizer,
                device_str,
                tool_mode=bfcl_tool_mode,
                max_new_tokens=bfcl_max_new_tokens,
                bfcl_adapter=bfcl_adapter_obj,
            )
            _start_server(app, server_host, server_port)
        else:
            logger.info("Skipping local OpenAI server; using in-process batched BFCL generation.")

        if persistent_workloads:
            suite_results = _run_persistent_generation_suite(
                model=model,
                tokenizer=tokenizer,
                workloads=persistent_workloads,
                switch=switch,
                model_name=model_name,
                model_alias=bfcl_model_alias,
                server_host=server_host,
                server_port=server_port,
                gptq_cache_info=gptq_cache_info,
                decode_weight_mode=decode_weight_mode,
                decode_weight_residency=decode_weight_residency,
                attention_backend=resolved_attn_implementation,
            )
            if persistent_suite_output_path:
                _write_json_atomic(Path(persistent_suite_output_path), suite_results)
            if switch is not None:
                switch.disable()
            return suite_results

        if persistent_trials:
            persistent_results = {
                "model_family": model_family,
                "decode_weight_mode": decode_weight_mode,
                "decode_weight_residency": decode_weight_residency,
                "gptq_cache": gptq_cache_info,
                "attention_memory": get_attention_memory_stats(model),
                "trials": [],
            }
            for trial_spec in persistent_trials:
                trial_log_dir = Path(str(trial_spec["log_dir"]))
                trial_log_dir.mkdir(parents=True, exist_ok=True)
                trial_result_dir = trial_log_dir / "bfcl_results"
                trial_score_dir = trial_log_dir / "bfcl_scores"
                trial_result_dir.mkdir(parents=True, exist_ok=True)
                trial_score_dir.mkdir(parents=True, exist_ok=True)

                if switch is not None:
                    prefill_phase, prefill_meta = _prefill_phase_from_tokens(
                        str(trial_spec["act"]),
                        str(trial_spec["kv"]),
                        str(trial_spec["fp_setting"]),
                    )
                    decode_phase, decode_meta = _decode_phase_from_tokens(
                        str(trial_spec.get("decode_act", trial_spec["act"])),
                        str(trial_spec.get("decode_kv", trial_spec["kv"])),
                        str(trial_spec.get("decode_fp_setting", trial_spec["fp_setting"])),
                    )
                    switch.phase_configs["prefill"] = prefill_phase
                    switch.phase_configs["decode"] = decode_phase
                    precision_metadata["prefill"] = prefill_meta
                    precision_metadata["decode"] = decode_meta
                    switch._on_phase_transition("prefill", None)
                    switch._phase[0] = "prefill"

                if bfcl_generate_mode == "batched":
                    gen_rc = _run_batched_bfcl_multiple_generate(
                        model=model,
                        tokenizer=tokenizer,
                        device=device_str,
                        result_dir=trial_result_dir,
                        model_name=model_name,
                        model_alias=bfcl_model_alias,
                        bfcl_adapter=bfcl_adapter_obj,
                        max_new_tokens=bfcl_max_new_tokens,
                        limit=limit,
                        batch_size=int(bfcl_batch_size),
                        batch_length_bucket=bool(bfcl_batch_length_bucket),
                        disable_thinking=bool(bfcl_disable_thinking),
                    )
                else:
                    gen_rc = _run_bfcl_generate(
                        model_name=model_name,
                        test_categories=bfcl_test_categories,
                        host=server_host,
                        port=server_port,
                        result_dir=trial_result_dir,
                        num_threads=bfcl_num_threads,
                        limit=limit,
                        model_alias=bfcl_model_alias,
                        web_search_backend=bfcl_web_search_backend,
                        web_search_cache_mode=bfcl_web_search_cache_mode,
                        web_search_cache_dir=bfcl_web_search_cache_dir,
                        web_search_stats_path=bfcl_web_search_stats_path,
                        web_search_max_concurrency=bfcl_web_search_max_concurrency,
                    )
                if run_evaluate:
                    eval_rc, scores = _run_bfcl_evaluate(
                        model_name=model_name,
                        test_categories=bfcl_test_categories,
                        result_dir=trial_result_dir,
                        score_dir=trial_score_dir,
                        model_alias=bfcl_model_alias,
                        partial_eval=limit is not None,
                    )
                else:
                    eval_rc, scores = 0, {}

                scores.update({
                    "bfcl_generate_returncode": gen_rc,
                    "bfcl_evaluate_returncode": eval_rc,
                    "bfcl_result_dir": str(trial_result_dir),
                    "bfcl_score_dir": str(trial_score_dir),
                    "phase_layer_configs": phase_configs,
                    "precision_metadata": dict(precision_metadata),
                    "bfcl_categories": bfcl_test_categories,
                    "bfcl_tool_mode": bfcl_tool_mode,
                    "bfcl_adapter": resolved_bfcl_adapter,
                    "bfcl_adapter_requested": bfcl_adapter,
                    "bfcl_generate_mode": bfcl_generate_mode,
                    "bfcl_batch_size": int(bfcl_batch_size) if bfcl_generate_mode == "batched" else None,
                    "bfcl_batch_length_bucket": bool(bfcl_batch_length_bucket) if bfcl_generate_mode == "batched" else None,
                    "bfcl_disable_thinking": bool(bfcl_disable_thinking) if bfcl_generate_mode == "batched" else None,
                    "model_family": model_family,
                    "attn_implementation_requested": requested_attn_implementation,
                    "attn_implementation": resolved_attn_implementation,
                    "bfcl_max_new_tokens": bfcl_max_new_tokens,
                    "decode_weight_mode": decode_weight_mode,
                    "decode_weight_residency": decode_weight_residency,
                    "gptq_cache": gptq_cache_info,
                    "attention_memory": get_attention_memory_stats(model),
                    "persistent_trial": {
                        "trial_id": trial_spec.get("trial_id", ""),
                        "act": trial_spec.get("act", ""),
                        "kv": trial_spec.get("kv", ""),
                        "fp_setting": trial_spec.get("fp_setting", ""),
                    },
                })
                save_results(trial_log_dir, scores)
                persistent_results["trials"].append(scores)
                if persistent_progress_path:
                    progress_path = Path(persistent_progress_path)
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    with progress_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "trial_id": trial_spec.get("trial_id", ""),
                            "status": "done",
                            "time": time.time(),
                        }, sort_keys=True) + "\n")
                        f.flush()

            if switch is not None:
                switch.disable()
            persistent_results["attention_memory"] = get_attention_memory_stats(model)
            return persistent_results

        # ------------------------------------------------------------------
        # Step 1: bfcl generate  (calls the local server)
        # ------------------------------------------------------------------
        print("\n[1/2] Generating BFCL responses...")
        if bfcl_generate_mode == "batched":
            gen_rc = _run_batched_bfcl_multiple_generate(
                model=model,
                tokenizer=tokenizer,
                device=device_str,
                result_dir=result_dir,
                model_name=model_name,
                model_alias=bfcl_model_alias,
                bfcl_adapter=bfcl_adapter_obj,
                max_new_tokens=bfcl_max_new_tokens,
                limit=limit,
                batch_size=int(bfcl_batch_size),
                batch_length_bucket=bool(bfcl_batch_length_bucket),
                disable_thinking=bool(bfcl_disable_thinking),
            )
        else:
            gen_rc = _run_bfcl_generate(
                model_name      = model_name,
                test_categories = bfcl_test_categories,
                host            = server_host,
                port            = server_port,
                result_dir      = result_dir,
                num_threads     = bfcl_num_threads,
                limit           = limit,
                model_alias     = bfcl_model_alias,
                web_search_backend = bfcl_web_search_backend,
                web_search_cache_mode = bfcl_web_search_cache_mode,
                web_search_cache_dir = bfcl_web_search_cache_dir,
                web_search_stats_path = bfcl_web_search_stats_path,
                web_search_max_concurrency = bfcl_web_search_max_concurrency,
            )
        if gen_rc != 0:
            logger.error("bfcl generate exited with code %d", gen_rc)

        if not run_evaluate:
            if switch is not None:
                switch.disable()
            scores = {
                "bfcl_generate_returncode": gen_rc,
                "bfcl_result_dir": str(result_dir),
                "bfcl_score_dir": str(score_dir),
                "phase_layer_configs": phase_configs,
                "bfcl_categories": bfcl_test_categories,
                "bfcl_tool_mode": bfcl_tool_mode,
                "bfcl_adapter": resolved_bfcl_adapter,
                "bfcl_adapter_requested": bfcl_adapter,
                "bfcl_generate_mode": bfcl_generate_mode,
                "bfcl_batch_size": int(bfcl_batch_size) if bfcl_generate_mode == "batched" else None,
                "bfcl_batch_length_bucket": bool(bfcl_batch_length_bucket) if bfcl_generate_mode == "batched" else None,
                "bfcl_disable_thinking": bool(bfcl_disable_thinking) if bfcl_generate_mode == "batched" else None,
                "model_family": model_family,
                "attn_implementation_requested": requested_attn_implementation,
                "attn_implementation": resolved_attn_implementation,
                "bfcl_max_new_tokens": bfcl_max_new_tokens,
                "decode_weight_mode": decode_weight_mode,
                "decode_weight_residency": decode_weight_residency,
                "precision_metadata": precision_metadata,
                "attention_memory": get_attention_memory_stats(model),
            }
            if resolved_gptq_config:
                scores["gptq"] = {
                    "dataset": resolved_gptq_config.get("dataset"),
                    "nsamples": resolved_gptq_config.get("nsamples"),
                    "seqlen": resolved_gptq_config.get("seqlen"),
                    "format": resolved_gptq_config.get("format"),
                    "weight_config": resolved_gptq_config.get("weight_config"),
                    "cali_batch_size": resolved_gptq_config.get("cali_batch_size"),
                    "max_layers": resolved_gptq_config.get("max_layers"),
                    "device_map_aware": resolved_gptq_config.get("device_map_aware", False),
                }
                scores["gptq_cache"] = gptq_cache_info
            if log_dir:
                save_results(log_dir, scores)
            return scores

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

        if switch is not None:
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
        scores["bfcl_adapter"] = resolved_bfcl_adapter
        scores["bfcl_adapter_requested"] = bfcl_adapter
        scores["bfcl_generate_mode"] = bfcl_generate_mode
        scores["bfcl_batch_size"] = int(bfcl_batch_size) if bfcl_generate_mode == "batched" else None
        scores["bfcl_batch_length_bucket"] = bool(bfcl_batch_length_bucket) if bfcl_generate_mode == "batched" else None
        scores["bfcl_disable_thinking"] = bool(bfcl_disable_thinking) if bfcl_generate_mode == "batched" else None
        scores["model_family"] = model_family
        scores["attn_implementation_requested"] = requested_attn_implementation
        scores["attn_implementation"] = resolved_attn_implementation
        scores["bfcl_max_new_tokens"] = bfcl_max_new_tokens
        scores["decode_weight_mode"] = decode_weight_mode
        scores["decode_weight_residency"] = decode_weight_residency
        scores["precision_metadata"] = precision_metadata
        scores["attention_memory"] = get_attention_memory_stats(model)
        if resolved_gptq_config:
            scores["gptq"] = {
                "dataset": resolved_gptq_config.get("dataset"),
                "nsamples": resolved_gptq_config.get("nsamples"),
                "seqlen": resolved_gptq_config.get("seqlen"),
                "format": resolved_gptq_config.get("format"),
                "weight_config": resolved_gptq_config.get("weight_config"),
                "cali_batch_size": resolved_gptq_config.get("cali_batch_size"),
                "max_layers": resolved_gptq_config.get("max_layers"),
                "device_map_aware": resolved_gptq_config.get("device_map_aware", False),
            }
            scores["gptq_cache"] = gptq_cache_info

        if log_dir:
            save_results(log_dir, scores)

        return scores
    finally:
        _stop_server(app)
        _tmpdir_ctx.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
