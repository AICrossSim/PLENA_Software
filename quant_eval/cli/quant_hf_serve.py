"""
Standalone OpenAI-compatible server for phase- and layer-type-dependent
MX-quantized HuggingFace models.

Loads a HF model, applies weight quantization (via chop), enables
PhaseLayerAutoSwitch for disaggregated activation precision, and serves
the result on an OpenAI-compatible HTTP endpoint.

Endpoints:
    POST /v1/chat/completions   (passthrough: one generate() call, raw text out)
    POST /v1/completions        (passthrough: raw text completion)
    GET  /v1/models
    GET  /v1/config             (read current phase x layer-type config)
    PUT  /v1/config             (update phase x layer-type config at runtime)

Tool-call parsing and execution are the client's responsibility.

Prerequisites:
    uv pip install fastapi uvicorn

Usage:
    python -m quant_eval.cli.quant_hf_serve --help

Example:
    python -m quant_eval.cli.quant_hf_serve \\
        --model_name Qwen/Qwen3-8B-FC \\
        --quant_config quant_eval/configs/llama_mxint4.toml \\
        --prefill_attn_width 4 --prefill_ffn_width 4 \\
        --decode_attn_width  8 --decode_ffn_width  8 \\
        --host 127.0.0.1 --port 8915
"""

import time
from typing import Union

import torch
import transformers

from quant_eval.utils import get_logger, set_logging_verbosity
from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8915


# ══════════════════════════════════════════════════════════════════════════════
#  Minimal OpenAI-compatible chat-completion server
# ══════════════════════════════════════════════════════════════════════════════

def build_server_app(model, tokenizer, device, switch=None):
    """
    Return a FastAPI application that exposes POST /v1/chat/completions,
    POST /v1/completions, GET /v1/models, and GET/PUT /v1/config.

    Generation calls ``model.generate()`` directly (single-GPU only).

    If ``switch`` (a PhaseLayerAutoSwitch instance) is provided, the
    /v1/config endpoints allow reading and updating phase configs at runtime.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="quant-hf-server")

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

    # ── Runtime config endpoints ───────────────────────────────────────
    @app.get("/v1/config")
    async def get_config():
        """Return the current phase × layer-type activation config."""
        if switch is None:
            return JSONResponse(
                status_code=404,
                content={"error": "No PhaseLayerAutoSwitch configured"},
            )
        return JSONResponse(content=switch.phase_configs)

    @app.put("/v1/config")
    async def put_config(request: Request):
        """
        Update phase configs at runtime.

        Accepts full or partial updates.  Example body:
            {
                "prefill": {"attn": {"data_in_width": 8, "data_in_block_size": 32}},
                "decode":  {"ffn":  {"data_in_width": 4, "data_in_block_size": 32}}
            }

        Omitted keys are left unchanged.
        """
        if switch is None:
            return JSONResponse(
                status_code=404,
                content={"error": "No PhaseLayerAutoSwitch configured"},
            )
        body = await request.json()
        # Filter to only the recognised shape, then merge into phase_configs
        for phase in ("prefill", "decode"):
            if phase in body:
                for layer_type in ("attn", "ffn"):
                    if layer_type in body[phase]:
                        switch.phase_configs.setdefault(phase, {})[layer_type] = body[phase][layer_type]

        logger.info("Config updated: %s", switch.phase_configs)
        return JSONResponse(content=switch.phase_configs)

    return app


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ══════════════════════════════════════════════════════════════════════════════

def main(
    model_name:      str  = "Qwen/Qwen3-8B-FC",
    device_id:       str  = "cuda:0",
    dtype:           str  = "bfloat16",
    quant_config:    Union[str, None] = None,
    model_parallel:  bool = False,
    host:            str  = DEFAULT_HOST,
    port:            int  = DEFAULT_PORT,
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
    # ── Optional keyword overrides ─────────────────────────────────────────
    attn_keywords: Union[list[str], None] = None,
    ffn_keywords:  Union[list[str], None] = None,
):
    """
    Load a quantized HF model with phase-layer disaggregated activation
    precision and serve it via an OpenAI-compatible HTTP server.

    The server blocks in the foreground until killed (Ctrl-C).
    """
    import uvicorn

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
            },
            "ffn": {
                "data_in_width":      decode_ffn_width,
                "data_in_block_size": decode_ffn_block_size,
            },
        },
    }

    # ── Print header ───────────────────────────────────────────────────────
    _pa = f"MXInt{prefill_attn_width}(bs={prefill_attn_block_size})"
    _pf = f"MXInt{prefill_ffn_width}(bs={prefill_ffn_block_size})"
    _da = f"MXInt{decode_attn_width}(bs={decode_attn_block_size})"
    _df = f"MXInt{decode_ffn_width}(bs={decode_ffn_block_size})"

    print("=" * 64)
    print("Quantized HF Model Server (OpenAI-compatible)")
    print("=" * 64)
    print(f"  Model      : {model_name}")
    print(f"  Weights    : {quant_config or 'none (fp)'}")
    print(f"  Server     : http://{host}:{port}")
    if quant_config:
        print()
        print(f"  {'':10s}  {'attn':>24s}  {'ffn':>24s}")
        print(f"  {'prefill':10s}  {_pa:>24s}  {_pf:>24s}")
        print(f"  {'decode':10s}  {_da:>24s}  {_df:>24s}")
    print("=" * 64)

    transformers.set_seed(0)

    # ── Model setup ────────────────────────────────────────────────────────
    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    switch = None

    if quant_config:
        # Quant path without TP (pipeline parallel or single GPU).
        # Use HF's native device_map="auto" which respects _no_split_modules,
        # then quantize in place. This avoids the 212 GB CPU load for MoE
        # models and the balanced-override layer-split bug.
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model_parallel:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Device map: {model.hf_device_map}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device_id)
        model.eval()

        from chop.passes.module.transforms import quantize_module_transform_pass

        pass_args = load_quant_config(quant_config)
        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id

        n_linear = sum(
            1 for _, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
        )
        logger.info("Quantizing %d linear layers...", n_linear)
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

        # ── Print the post-quant module map (name → class) ─────────────
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

        # ── Enable disaggregated quantization hook ─────────────────────
        switch_kwargs = {}
        if attn_keywords:
            switch_kwargs["attn_keywords"] = tuple(attn_keywords)
        if ffn_keywords:
            switch_kwargs["ffn_keywords"] = tuple(ffn_keywords)

        switch = PhaseLayerAutoSwitch(
            model, phase_configs, model_name=model_name, **switch_kwargs,
        )
        switch.enable()
        logger.info("\n%s", switch.summary())

        server_device = device_id if not model_parallel else "cuda"

    else:
        # No quant, no TP: pipeline parallel (via accelerate) or single GPU
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model_parallel:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Device map: {model.hf_device_map}")
            server_device = "cuda"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device_id)
            server_device = device_id
        model.eval()

    # ── Build FastAPI app and run uvicorn in the main thread ──────────────
    app = build_server_app(model, tokenizer, server_device, switch=switch)

    print(f"\nServing on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
