"""
Plain OpenAI-compatible server for MX-quantized HuggingFace models.

Loads a HF model, applies weight + activation quantization from a single TOML
config, and serves it on an OpenAI-compatible HTTP endpoint. Whatever
precision the TOML bakes in is what the model runs with for the lifetime of
the server -- there is no prefill/decode phase switching.

For phase-disaggregated activation precision, use ``quant_hf_serve.py``.

Endpoints:
    POST /v1/chat/completions   (passthrough: one generate() call, raw text out)
    POST /v1/completions        (passthrough: raw text completion)
    GET  /v1/models

Tool-call parsing and execution are the client's responsibility.

Prerequisites:
    uv pip install fastapi uvicorn

Usage:
    python -m quant_eval.cli.plain_quant_serve --help

Example:
    python -m quant_eval.cli.plain_quant_serve \\
        --model_name Qwen/Qwen3-8B-FC \\
        --quant_config quant_eval/configs/qwen3_mxint8.toml \\
        --host 127.0.0.1 --port 8915
"""

import time
from typing import Union

import torch
import transformers

from quant_eval.utils import get_logger, set_logging_verbosity
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8915


# ══════════════════════════════════════════════════════════════════════════════
#  Minimal OpenAI-compatible chat-completion server
# ══════════════════════════════════════════════════════════════════════════════

def build_server_app(model, tokenizer, device):
    """
    Return a FastAPI application that exposes POST /v1/chat/completions,
    POST /v1/completions, and GET /v1/models.

    Generation calls ``model.generate()`` directly (single-GPU only when
    ``device`` is a concrete device id; for HF auto device-map dispatch, pass
    ``device="cuda"``).
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="plain-quant-serve")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        print(f">>> {request.method} {request.url.path}")
        try:
            response = await call_next(request)
        except Exception as e:
            print(f"!!! Exception: {e}")
            raise
        print(f"<<< {request.method} {request.url.path} -> {response.status_code}")
        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Passthrough mode: one generate() call, return the raw text as
        assistant content. Tool-call parsing and tool execution are the
        client's responsibility."""
        print("REQUEST HIT /v1/chat/completions")
        body = await request.json()
        messages     = list(body.get("messages", []))
        tools        = body.get("tools", None)
        temperature  = body.get("temperature", 0.0)
        max_new_toks = body.get("max_tokens", 1024)

        # ── Build prompt ───────────────────────────────────────────────
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except Exception:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            if isinstance(encoded, torch.Tensor):
                prompt_ids = encoded.to(device)
                attention_mask = torch.ones_like(prompt_ids)
            else:
                prompt_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get(
                    "attention_mask", torch.ones_like(prompt_ids)
                ).to(device)
        else:
            text = "\n".join(
                f"{m.get('role','user').upper()}: {m.get('content','')}"
                for m in messages
            ) + "\nASSISTANT:"
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            prompt_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

        # ── Inference (single call, no loop) ───────────────────────────
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_toks,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_ids.shape[-1]:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # ── Return raw text as assistant content ──────────────────────
        message = {"role": "assistant", "content": raw_text}
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
        print("REQUEST HIT /v1/completions")
        body = await request.json()

        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        temperature  = body.get("temperature", 0.0)
        max_new_toks = body.get("max_tokens", 1024)

        # Don't auto-add BOS — BFCL's _format_prompt already includes
        # <|begin_of_text|> in the prompt string, so add_special_tokens=True
        # would produce a double BOS token that breaks Llama 3.x reasoning.
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_toks,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_ids.shape[-1]:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        response = {
            "id":      f"cmpl-{int(time.time()*1000)}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   tokenizer.name_or_path,
            "choices": [{
                "index":         0,
                "text":          raw_text,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     int(prompt_ids.shape[-1]),
                "completion_tokens": int(generated_ids.shape[-1]),
                "total_tokens":      int(prompt_ids.shape[-1] + generated_ids.shape[-1]),
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


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main(
    model_name:          str  = "Qwen/Qwen3-8B-FC",
    device_id:           str  = "cuda:0",
    dtype:               str  = "bfloat16",
    quant_config:        Union[str, None] = None,
    model_parallel:      bool = False,
    host:                str  = DEFAULT_HOST,
    port:                int  = DEFAULT_PORT,
    attn_implementation: str  = "eager",
):
    """
    Load a (possibly quantized) HF model and serve it via an
    OpenAI-compatible HTTP server.

    The TOML at ``quant_config`` is the single source of truth for both
    weight and activation precision; whatever it specifies is what the
    server runs with throughout. Pass ``--quant_config`` unset to serve the
    fp model unmodified.

    The server blocks in the foreground until killed (Ctrl-C).
    """
    import uvicorn

    print("=" * 64)
    print("Quantized HF Model Server (OpenAI-compatible, phase-agnostic)")
    print("=" * 64)
    print(f"  Model      : {model_name}")
    print(f"  Quant cfg  : {quant_config or 'none (fp)'}")
    print(f"  Attn impl  : {attn_implementation}")
    print(f"  Server     : http://{host}:{port}")
    print("=" * 64)

    transformers.set_seed(0)

    # ── Model setup ────────────────────────────────────────────────────────
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load straight onto the target device(s) before quantizing. With
    # model_parallel=True we let HF pick a layout via its native device_map
    # ("auto" respects _no_split_modules), which avoids the 212 GB CPU load
    # for MoE models and the balanced-override layer-split bug.
    #
    # ``attn_implementation`` defaults to "eager" because that's the only
    # backend on which the MX attention wrappers' qk_matmul / av_matmul /
    # softmax stages run. The wrappers' updated dispatch (Qwen3AttentionMXFP /
    # MXInt: see qwen3/attention.py:127) detects the all-bypass case and
    # routes to HF's configured backend instead, so sdpa / flash_attention_2
    # / flash_attention_3 are safe iff the TOML bypasses qk + av + softmax.
    # KV-cache MX runs before attention dispatch and is unaffected by this
    # choice. Pure FP16 (no quant_config) can use any backend.
    if model_parallel:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        print(f"Device map: {model.hf_device_map}")
        server_device = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).to(device_id)
        server_device = device_id
    model.eval()

    # ── Optional TokenCollector (calibration-mode TOMLs) ──────────────────
    # ``token_collector`` is a side-effect pass that just attaches a hook;
    # if it's the only thing in the TOML this run is "calibration mode" and
    # we'll skip module quantization entirely. With ``raise_on_full = true``
    # in the TOML, the hook raises CollectorFull once the buffer is full,
    # which propagates out of the next forward and tears the server down --
    # that's the calibrate-done signal. Drive forwards by pointing a client
    # (e.g. bfcl generate) at the local server.
    pass_args = load_quant_config(quant_config) if quant_config else None

    if pass_args and "token_collector" in pass_args:
        from chop.passes.module.transforms import attach_token_collector_pass

        tc_cfg = pass_args.pop("token_collector")
        logger.info("Attaching TokenCollector: %s", tc_cfg)
        model, _collector_info = attach_token_collector_pass(model, tc_cfg)

    # ── Quantization (skipped in pure calibration mode) ───────────────────
    has_quant = pass_args is not None and (
        "gptq" in pass_args
        or any(k != "by" for k in pass_args.keys())
    )
    if has_quant:
        from chop.passes.module.transforms import quantize_module_transform_pass

        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id
        if "rotation_search" in pass_args:
            pass_args["rotation_search"]["device"] = device_id
            pass_args["rotation_search"].setdefault("model_name", model_name)

        n_linear = sum(
            1 for _, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
        )
        logger.info("Quantizing %d linear layers...", n_linear)
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

        # Print the post-quant module map (name → class).
        quant_modules = [
            (name, type(m).__name__)
            for name, m in model.named_modules()
            if "MX" in type(m).__name__
        ]
        logger.info(
            "Post-quant module map (%d quantized modules):\n%s",
            len(quant_modules),
            "\n".join(f"  {n} -> {c}" for n, c in quant_modules),
        )

    # ── Build FastAPI app and run uvicorn in the main thread ──────────────
    app = build_server_app(model, tokenizer, server_device)

    print(f"\nServing on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
