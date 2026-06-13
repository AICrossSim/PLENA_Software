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

import atexit
import gc
import hashlib
import json
import os
import shutil
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
from quant_eval.precision import apply_dse_quant_config, parse_fp_setting, parse_mx_precision, mx_data_config, fp_data_config
from quant_eval.eval.unified_mx import apply_unified_mx_wrappers
from quant_eval.bfcl_adapters import BFCL_ADAPTER_NAMES, resolve_bfcl_adapter

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

_CUDA_RESERVE_MB = 1024 * 1024


class _FixedCudaMemoryReserve:
    """Hold a fixed CUDA allocation during low-memory phases of eval.

    This guard makes GPTQ's low-memory phase keep a fixed amount of GPU memory
    occupied, then releases it immediately before BFCL generation.
    """

    def __init__(
        self,
        *,
        device: str,
        reserve_mb: int,
        wait_sec: int,
        poll_sec: float,
        chunk_mb: int,
        enabled: bool,
        release_label: str = "before BFCL generate",
    ):
        self.device = torch.device(device)
        self.reserve_mb = int(reserve_mb or 0)
        self.wait_sec = int(wait_sec)
        self.poll_sec = float(poll_sec)
        self.chunk_mb = int(chunk_mb)
        self.enabled = bool(enabled and self.reserve_mb > 0)
        self.release_label = release_label
        self._buffers: list[torch.Tensor] = []
        self._reserved_mb = 0
        self._total_mb = 0
        self._free_before_mb = 0

        if self.enabled:
            if self.device.type != "cuda":
                raise ValueError(
                    "GPU memory reservation requires a CUDA device; "
                    f"got {device!r}. Disable it with --gpu_memory_reserve_disable true."
                )
            if self.chunk_mb <= 0:
                raise ValueError("gpu_memory_reserve_chunk_mb must be > 0.")
            if self.wait_sec < 0:
                raise ValueError("gpu_memory_reserve_wait_sec must be >= 0.")
            if self.poll_sec <= 0:
                raise ValueError("gpu_memory_reserve_poll_sec must be > 0.")

    def _device_index(self) -> int:
        if self.device.index is None:
            return torch.cuda.current_device()
        return self.device.index

    def _memory_info_mb(self) -> tuple[int, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(self._device_index())
        return free_bytes // _CUDA_RESERVE_MB, total_bytes // _CUDA_RESERVE_MB

    def acquire(self) -> None:
        if not self.enabled:
            logger.info("GPU reserve disabled.")
            return
        if not torch.cuda.is_available():
            raise RuntimeError("GPU reserve requested but torch.cuda is not available.")

        torch.cuda.set_device(self._device_index())
        start = time.monotonic()
        last_error = "unknown allocation failure"
        while True:
            free_mb, total_mb = self._memory_info_mb()
            self._free_before_mb = int(free_mb)
            self._total_mb = int(total_mb)
            if free_mb >= self.reserve_mb:
                try:
                    remaining_mb = self.reserve_mb
                    while remaining_mb > 0:
                        alloc_mb = min(self.chunk_mb, remaining_mb)
                        self._buffers.append(
                            torch.empty(
                                alloc_mb * _CUDA_RESERVE_MB,
                                dtype=torch.uint8,
                                device=self.device,
                            )
                        )
                        self._reserved_mb += alloc_mb
                        remaining_mb -= alloc_mb
                    logger.info(
                        "GPU reserve acquired: total=%dMB free_before=%dMB reserved=%dMB",
                        self._total_mb,
                        self._free_before_mb,
                        self._reserved_mb,
                    )
                    print(
                        "GPU reserve acquired: "
                        f"total={self._total_mb}MB "
                        f"free_before={self._free_before_mb}MB "
                        f"reserved={self._reserved_mb}MB"
                    )
                    return
                except RuntimeError as exc:
                    last_error = str(exc)
                    self.release(log=False)
                    torch.cuda.empty_cache()
            else:
                last_error = (
                    f"free={free_mb}MB is below requested reserve={self.reserve_mb}MB "
                    f"on total={total_mb}MB GPU"
                )

            if time.monotonic() - start >= self.wait_sec:
                raise RuntimeError(
                    "Timed out acquiring GPU reserve after "
                    f"{self.wait_sec}s: {last_error}. Lower --gpu_memory_reserve_mb, "
                    "choose a freer GPU, or disable reservation."
                )
            logger.info(
                "Waiting for GPU reserve: reserve=%dMB free=%dMB total=%dMB retry_in=%.1fs",
                self.reserve_mb,
                free_mb,
                total_mb,
                self.poll_sec,
            )
            time.sleep(self.poll_sec)

    def release(self, *, log: bool = True) -> None:
        if not self._buffers and self._reserved_mb == 0:
            return
        released_mb = self._reserved_mb
        free_before_mb, total_mb = (0, 0)
        free_after_mb = 0

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self._device_index())
            torch.cuda.synchronize(self._device_index())
            free_before_mb, total_mb = self._memory_info_mb()

        self._buffers.clear()
        self._reserved_mb = 0
        gc.collect()

        if torch.cuda.is_available() and self.device.type == "cuda":
            # Make reserve release visible to subsequent BFCL generate requests
            # before the server starts handling decode allocations. Without the
            # synchronizes, PyTorch/CUDA allocator bookkeeping can transiently
            # overlap the reservation with the first generation memory peak.
            torch.cuda.synchronize(self._device_index())
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self._device_index())
            free_after_mb, total_mb = self._memory_info_mb()

        if log:
            logger.info(
                "GPU reserve released %s: released=%dMB free_before=%dMB free_after=%dMB total=%dMB",
                self.release_label,
                released_mb,
                free_before_mb,
                free_after_mb,
                total_mb,
            )
            print(
                f"GPU reserve released {self.release_label}: "
                f"released={released_mb}MB "
                f"free_before={free_before_mb}MB "
                f"free_after={free_after_mb}MB "
                f"total={total_mb}MB"
            )

    def summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "reserve_mb": self.reserve_mb,
            "reserved_mb": self._reserved_mb,
            "total_mb": self._total_mb,
            "free_before_mb": self._free_before_mb,
            "chunk_mb": self.chunk_mb,
            "wait_sec": self.wait_sec,
            "poll_sec": self.poll_sec,
        }


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

def _build_server_app(model, tokenizer, device: str, tool_mode: str = "execute", max_new_tokens: int | None = 2048, bfcl_adapter=None):
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

    app = FastAPI(title="quant-model-server")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        print("🚀 REQUEST HIT")
        body = await request.json()
        messages     = list(body.get("messages", []))  # make a mutable copy
        tools        = body.get("tools", None)
        temperature  = body.get("temperature", 0.0)
        max_new_toks = _cap_max_new_tokens(body.get("max_tokens", 1024))

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
                    ).to(_input_device())
                except Exception:
                    prompt_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(_input_device())
            else:
                text = "\n".join(
                    f"{m.get('role','user').upper()}: {m.get('content','')}"
                    for m in messages
                ) + "\nASSISTANT:"
                prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(_input_device())

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
            tool_calls, content = bfcl_adapter.parse_tool_calls(raw_text)
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
        max_new_toks = _cap_max_new_tokens(body.get("max_tokens", 1024))

        MAX_TURNS = 15  # prevent infinite agentic loops

        total_prompt_tokens     = 0
        total_completion_tokens = 0

        for turn in range(MAX_TURNS):
            print(f"🔄 Agentic turn {turn + 1}/{MAX_TURNS}")

            # Tokenize the current prompt string directly (no chat template).
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(_input_device())

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
            tool_calls, content = bfcl_adapter.parse_tool_calls(raw_text)
            content = content or ""
            print(tool_calls)

            # ── No tool calls → final answer, return immediately ───────
            if not tool_calls:
                print(f"✅ Final answer reached at turn {turn + 1}")
                response_text = raw_text
                if tool_mode == "return":
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

            # BFCL non-live categories should receive the model's function
            # call response. Llama 3.1 FC emits special tokens around JSON
            # (<|python_tag|>...<|eom_id|>); strip those before BFCL evaluates.
            if tool_mode == "return":
                print(f"↩️ Returning completion tool calls at turn {turn + 1}: {[tc['function']['name'] for tc in tool_calls]}")
                response_text = bfcl_adapter.return_text_from_tool_calls(tool_calls, raw_text=raw_text)
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
        print(f"WARNING: Failed to parse BFCL tool-call payload: {payload[:200]}")
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
    weight_block_size: int,
    cali_batch_size: int,
    max_layers: int | None,
    device_map_aware: bool = False,
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
        "device_map_aware": bool(device_map_aware),
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

    def _acquire_lock(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self._wait_sec
        while True:
            try:
                os.mkdir(self.lock_path)
                self._lock_acquired = True
                return
            except FileExistsError:
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

        for layer_idx in self.expected_layers:
            layer_state = load_file(str(self._layer_path(layer_idx)))
            model_state = {
                f"model.layers.{layer_idx}.{name}": value
                for name, value in layer_state.items()
            }
            model.load_state_dict(model_state, strict=False)
        self.hit = True
        self.loaded_layers = len(self.expected_layers)
        logger.info(
            "GPTQ cache hit: loaded %d cached layers, skipping GPTQ (key=%s)",
            self.loaded_layers,
            self.key,
        )
        return self.loaded_layers

    def prepare(self, model) -> bool:
        if self.mode == "off":
            return False
        if self.mode != "refresh" and self._is_complete():
            self.load(model)
            return True

        self._acquire_lock()
        try:
            if self.mode != "refresh" and self._is_complete():
                self.load(model)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main(
    model_name:  str = "Qwen/Qwen3-8B",
    device_id:   str = "cuda:0",
    dtype:       str = "bfloat16",
    quant_config: str = "quant_eval/configs/qwen3_mxint16.toml",
    model_parallel: bool = False,
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
    # ── GPU reservation guard ───────────────────────────────────────────────
    gpu_memory_reserve_mb: int = 0,
    gpu_memory_reserve_wait_sec: int = 600,
    gpu_memory_reserve_poll_sec: float = 5.0,
    gpu_memory_reserve_chunk_mb: int = 512,
    gpu_memory_reserve_disable: bool = False,
    # ── GPTQ calibration/weight quantization ───────────────────────────────
    gptq_dataset:          str | None = None,
    gptq_nsamples:         int = 32,
    gptq_seqlen:           int = 1024,
    gptq_format:           str = "mxint",
    gptq_weight_width:     int = 8,
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
    # ── Optional keyword overrides ─────────────────────────────────────────
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords:  Union[list[str], None] = None,
    limit: Union[int, None] = None,
    log_dir: Union[str, None] = None,
    run_evaluate: bool = True,
    persistent_trials: list[dict] | None = None,
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
        server_host: Host for the local OpenAI-compatible server.
        server_port: Port for the local OpenAI-compatible server.
        bfcl_tool_mode: ``"auto"`` executes tools only for web-search categories;
            ``"return"`` returns model tool calls directly for BFCL function-call
            categories such as ``multiple``; ``"execute"`` preserves agentic
            web-search behavior.
        bfcl_max_new_tokens: Local cap applied to BFCL-requested ``max_tokens``
            before calling ``model.generate``. ``None`` disables the cap.
        gpu_memory_reserve_mb: Fixed CUDA memory reservation held during GPTQ and
            quantization, then released before BFCL generation. ``0`` disables it.
        gpu_memory_reserve_wait_sec: Seconds to wait for enough free memory.
        gpu_memory_reserve_poll_sec: Poll interval while waiting for memory.
        gpu_memory_reserve_chunk_mb: Chunk size for the reservation tensors.
        gpu_memory_reserve_disable: Disable the reservation guard entirely.
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
            weight reload path. ``"gpu_dual"`` keeps original FP Linear weights
            in GPU-resident wrapper buffers and phase-switches by flag.
        attn_keywords: Module-name substrings that identify attention blocks.
            ``None`` uses the built-in defaults.
        ffn_keywords: Module-name substrings that identify FFN blocks. ``None``
            uses the built-in defaults.
        limit: Cap the number of samples per category. ``None`` = full dataset.
        log_dir: Directory for ``args.json`` and ``results.json``.
        run_evaluate: Run ``bfcl evaluate`` after generation. Set to ``False``
            when only raw generation artifacts, for example token length
            statistics, are needed.

    Returns:
        BFCL evaluation summary — per-category scores plus aggregate metrics,
        with ``phase_layer_configs`` recording the resolved (phase, layer)
        precision table.
    """
    bfcl_test_categories = _normalize_bfcl_categories(bfcl_test_categories)
    bfcl_tool_mode = _resolve_bfcl_tool_mode(bfcl_test_categories, bfcl_tool_mode)
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
    if decode_weight_residency not in {"disk_reload", "gpu_dual"}:
        raise ValueError(
            "decode_weight_residency must be 'disk_reload' or 'gpu_dual', "
            f"got {decode_weight_residency!r}."
        )
    if decode_weight_residency == "gpu_dual" and decode_weight_mode != "fp":
        raise ValueError("decode_weight_residency='gpu_dual' requires decode_weight_mode='fp'.")
    if decode_weight_residency == "gpu_dual" and not model_parallel:
        raise ValueError("decode_weight_residency='gpu_dual' is only supported with model_parallel=true.")
    device_id = _normalize_device_id(device_id)
    gpu_memory_reserve_enabled = (
        not gpu_memory_reserve_disable
        and gpu_memory_reserve_mb is not None
        and int(gpu_memory_reserve_mb) > 0
    )
    if gpu_memory_reserve_enabled and model_parallel:
        raise ValueError(
            "GPU memory reservation currently supports only single-GPU eval; "
            "disable it with --gpu_memory_reserve_disable true for model_parallel runs."
        )
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
    precision_metadata = {
        "prefill": _prefill_precision["metadata"],
        "decode": _decode_precision["metadata"],
        "dse_mx_block_size": dse_mx_block_size,
        "dse_weight_precision": _resolved_dse_weight_precision,
        "dse_weight_block_size": _resolved_dse_weight_block_size,
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
        return (
            {
                "attn": {**act_cfg, "kv_cache": dict(kv_cfg), "softmax": dict(fp_cfg), "rope": dict(fp_cfg)},
                "ffn": dict(act_cfg),
                "mlp": dict(fp_cfg),
                "rms_norm": rms_cfg,
            },
            {
                "ACT_ELEMENT_WIDTH": act_spec.canonical,
                "KV_ELEMENT_WIDTH": kv_spec.canonical,
                "FP_SETTING": fp_spec.canonical,
            },
        )

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

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    _pa = _prefill_precision["display"]
    _pf = _prefill_precision["ffn_display"]
    if decode_weight_mode == "fp":
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
    print(f"  Max new tok: {bfcl_max_new_tokens if bfcl_max_new_tokens is not None else 'uncapped'}")
    print(f"  Weights    : {'FP baseline (no quantization)' if quant_config_is_none else quant_config}")
    print(f"  Decode W   : {decode_weight_mode}")
    print(f"  Weight res : {decode_weight_residency}")
    if gptq_dataset:
        print(f"  GPTQ       : dataset={gptq_dataset}, nsamples={gptq_nsamples}, seqlen={gptq_seqlen}, max_layers={gptq_max_layers}")
        print(f"  GPTQ cache : mode={gptq_cache_mode}, dir={gptq_cache_dir or 'none'}")
    print(f"  Server     : http://{server_host}:{server_port}")
    if gpu_memory_reserve_enabled:
        print(
            "  GPU reserve: "
            f"reserve={int(gpu_memory_reserve_mb)}MB, "
            f"wait={gpu_memory_reserve_wait_sec}s, "
            f"chunk={gpu_memory_reserve_chunk_mb}MB"
        )
    else:
        print("  GPU reserve: disabled")
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
    )
    model.eval()

    gpu_memory_reserve = _FixedCudaMemoryReserve(
        device=device_id,
        reserve_mb=int(gpu_memory_reserve_mb or 0),
        wait_sec=gpu_memory_reserve_wait_sec,
        poll_sec=gpu_memory_reserve_poll_sec,
        chunk_mb=gpu_memory_reserve_chunk_mb,
        enabled=gpu_memory_reserve_enabled,
    )
    gpu_memory_reserve.acquire()
    atexit.register(gpu_memory_reserve.release)

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
                model, _ = quantize_module_transform_pass(model, pass_args)
                if gptq_cache is not None and not gptq_cache.hit:
                    gptq_cache.finalize()
                    gptq_cache_info = gptq_cache.summary()
                    # The cache lock only protects GPTQ cache population.
                    # Runtime wrapper installation is trial-local and can run
                    # concurrently on other workers once metadata is complete.
                    gptq_cache.release()
                if _codesign_tokens_enabled or qwen_model_family:
                    _qwen3_moe_experts_config = None
                    if model_family == "qwen3_moe" and _prefill_precision["mlp"]:
                        _qwen3_moe_experts_config = {
                            **_prefill_precision["ffn"],
                            **_prefill_precision["mlp"],
                        }
                    unified_counts = apply_unified_mx_wrappers(
                        model,
                        qwen3_attention_config=_prefill_precision["attn"] if model_family == "qwen3" else None,
                        qwen3_mlp_config=_prefill_precision["mlp"] if model_family == "qwen3" and _prefill_precision["mlp"] else None,
                        qwen3_rms_norm_config=_prefill_precision["rms_norm"] if model_family == "qwen3" and _prefill_precision["rms_norm"] else None,
                        qwen3_moe_attention_config=_prefill_precision["attn"] if model_family == "qwen3_moe" else None,
                        qwen3_moe_experts_config=_qwen3_moe_experts_config,
                        qwen3_moe_rms_norm_config=_prefill_precision["rms_norm"] if model_family == "qwen3_moe" and _prefill_precision["rms_norm"] else None,
                    )
                    logger.info(
                        "Installed unified MX wrappers: %d Linear, %d attention (llama=%d, qwen3=%d, qwen3_moe=%d), qwen3_mlp=%d, qwen3_rms_norm=%d, qwen3_moe_experts=%d, qwen3_moe_rms_norm=%d",
                        unified_counts.get("linear", 0),
                        unified_counts.get("attention", 0),
                        unified_counts.get("llama_attention", 0),
                        unified_counts.get("qwen3_attention", 0),
                        unified_counts.get("qwen3_moe_attention", 0),
                        unified_counts.get("qwen3_mlp", 0),
                        unified_counts.get("qwen3_rms_norm", 0),
                        unified_counts.get("qwen3_moe_experts", 0),
                        unified_counts.get("qwen3_moe_rms_norm", 0),
                    )
                logger.info("Quantization complete in %.1fs", time.time() - t0)
            finally:
                if gptq_cache is not None:
                    gptq_cache.release()

            # Release the guard before any final device move or phase-switch
            # setup. Those steps can allocate temporary tensors (for example
            # during module.to()), so holding the fixed reserve here can leave
            # too little headroom and OOM before BFCL generation even starts.
            gpu_memory_reserve.release()

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
                switch_kwargs["weight_residency"] = decode_weight_residency

            switch = PhaseLayerAutoSwitch(model, phase_configs, **switch_kwargs)
            switch.enable()
            logger.info("\n%s", switch.summary())

        # ------------------------------------------------------------------
        # Start the OpenAI-compatible server (hook fires on every request)
        # ------------------------------------------------------------------
        device_str = device_id if not model_parallel else "cuda"
        app = _build_server_app(
            model,
            tokenizer,
            device_str,
            tool_mode=bfcl_tool_mode,
            max_new_tokens=bfcl_max_new_tokens,
            bfcl_adapter=bfcl_adapter_obj,
        )
        _start_server(app, server_host, server_port)

        if persistent_trials:
            persistent_results = {
                "model_family": model_family,
                "decode_weight_mode": decode_weight_mode,
                "decode_weight_residency": decode_weight_residency,
                "gptq_cache": gptq_cache_info,
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
                    switch.phase_configs["prefill"] = prefill_phase
                    precision_metadata["prefill"] = prefill_meta
                    switch._on_phase_transition("prefill", None)
                    switch._phase[0] = "prefill"

                gen_rc = _run_bfcl_generate(
                    model_name=model_name,
                    test_categories=bfcl_test_categories,
                    host=server_host,
                    port=server_port,
                    result_dir=trial_result_dir,
                    num_threads=bfcl_num_threads,
                    limit=limit,
                    model_alias=bfcl_model_alias,
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
                    "model_family": model_family,
                    "bfcl_max_new_tokens": bfcl_max_new_tokens,
                    "decode_weight_mode": decode_weight_mode,
                    "decode_weight_residency": decode_weight_residency,
                    "gpu_memory_reserve": gpu_memory_reserve.summary(),
                    "gptq_cache": gptq_cache_info,
                    "persistent_trial": {
                        "trial_id": trial_spec.get("trial_id", ""),
                        "act": trial_spec.get("act", ""),
                        "kv": trial_spec.get("kv", ""),
                        "fp_setting": trial_spec.get("fp_setting", ""),
                    },
                })
                save_results(trial_log_dir, scores)
                persistent_results["trials"].append(scores)

            if switch is not None:
                switch.disable()
            return persistent_results

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
                "model_family": model_family,
                "bfcl_max_new_tokens": bfcl_max_new_tokens,
                "decode_weight_mode": decode_weight_mode,
                "decode_weight_residency": decode_weight_residency,
                "precision_metadata": precision_metadata,
                "gpu_memory_reserve": gpu_memory_reserve.summary(),
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
            _tmpdir_ctx.cleanup()
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
        scores["model_family"] = model_family
        scores["bfcl_max_new_tokens"] = bfcl_max_new_tokens
        scores["decode_weight_mode"] = decode_weight_mode
        scores["decode_weight_residency"] = decode_weight_residency
        scores["precision_metadata"] = precision_metadata
        scores["gpu_memory_reserve"] = gpu_memory_reserve.summary()
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

        _tmpdir_ctx.cleanup()
        return scores
    finally:
        gpu_memory_reserve.release(log=False)


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
