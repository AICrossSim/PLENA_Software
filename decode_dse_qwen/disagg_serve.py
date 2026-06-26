"""The disaggregated decode engine: load the chips, quantise the decode chip to a per-component
precision, and run it (generation + a fast PPL proxy).

The bf16 PREFILL chip is run ONCE to cache every input's KV (`prefill_cache`), then freed; the
low-precision DECODE chip reuses those caches (`build_decode_cache`)

Sections: [PLACEMENT] -> [LOAD] -> [QUANTISE] -> [KV CACHE] -> [GENERATE] -> [PPL PROXY].
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from decode_dse_qwen import quant

MODEL_ID = "Qwen/Qwen3-32B"
HEADROOM_GIB = 10  # free GiB kept per GPU beyond the model shard (absorbs the quantisation spike)

# Qwen3 official sampling (https://huggingface.co/Qwen/Qwen3-32B); NEVER greedy in thinking mode.
THINKING_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
NONTHINKING_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)
THINK_CLOSE = "</think>"


def load_dims(model_id: str = MODEL_ID) -> dict:
    """Architecture sizes from the HF config (no weights downloaded) -- feeds the cost model."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    heads = cfg.num_attention_heads
    return {
        "hidden": cfg.hidden_size, "heads": heads,
        "kv_heads": getattr(cfg, "num_key_value_heads", heads),
        "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // heads,
        "layers": cfg.num_hidden_layers, "inter": cfg.intermediate_size, "vocab": cfg.vocab_size,
    }


def _n_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _free_mib() -> list[int]:
    return [int(torch.cuda.mem_get_info(i)[0] / 1024**2) for i in range(_n_gpus())]


def select_gpus(n: int = 2) -> list[int]:
    if _n_gpus() < n:
        raise RuntimeError(f"need >={n} GPUs for Qwen3-32B; found {_n_gpus()}")
    return sorted(sorted(range(_n_gpus()), key=lambda i: _free_mib()[i], reverse=True)[:n])


def first_device(gpus: list[int]) -> torch.device:
    return torch.device(f"cuda:{gpus[0]}")


def _max_memory(gpus: list[int]) -> dict:
    """Per-GPU cap for `device_map="balanced"`: (free - HEADROOM) on the chosen cards, 0 on the rest.
    Kept generous on purpose -- too tight and accelerate dumps the model on the meta device."""
    free = _free_mib()
    mm = {i: "0GiB" for i in range(_n_gpus())}
    for i in gpus:
        mm[i] = f"{max(free[i] - HEADROOM_GIB * 1024, 1024)}MiB"
    return mm


def load_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_model(dtype, gpus: list[int], attn_implementation: str, model_id: str = MODEL_ID):
    """Load Qwen3-32B. `gpus=[]` -> CPU-resident (the GPTQ master); otherwise balanced-sharded across
    `gpus`. eager attention is required for the chip whose attention gets quantised."""
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=dtype, attn_implementation=attn_implementation, trust_remote_code=True,
              low_cpu_mem_usage=True)
    if gpus:
        kw["device_map"] = "balanced"
        kw["max_memory"] = _max_memory(gpus)
    return AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


def load_prefill(gpus: list[int], dtype):
    """The unquantised bf16 prefill chip (SDPA: fast, and KV-equivalent to eager)."""
    return load_model(dtype, gpus, attn_implementation="sdpa")


def install_attention_quant(model, spec: dict) -> int:
    """Swap each layer's attention to chop's quantised Qwen3 attention IN PLACE (`__class__` swap keeps
    the already-quantised projections, accelerate hooks, and devices). Install it by hand because
    chop's auto-dispatch can't reach these classes. For now only the qk/av matmuls stay high-precision FP (bypassed)."""
    from chop.nn.quantized.modules.qwen3.attention import (
        Qwen3AttentionMXInt, Qwen3AttentionMXFP, Qwen3AttentionMXIntRotate)
    rotate = bool(spec.get("rotation"))
    cls = Qwen3AttentionMXIntRotate if rotate else \
        {"mxint": Qwen3AttentionMXInt, "mxfp": Qwen3AttentionMXFP}[spec["fmt"]]
    qcfg = quant.attn_qconfig(spec)
    for layer in model.model.layers:
        a = layer.self_attn
        a.config._attn_implementation = "eager"   # the quantised matmul forward asserts this
        a.__class__ = cls
        for stage in ("qk", "av", "rope", "softmax", "kv_cache"):
            key = {"qk": "qk_matmul", "av": "av_matmul"}.get(stage, stage)
            setattr(a, f"{stage}_config", qcfg[key])
            setattr(a, f"{stage}_bypass", qcfg[key].get("bypass", False))
        if rotate:
            for s in ("qk", "av", "kv_cache"):
                setattr(a, f"{s}_use_rotate", qcfg[{"qk": "qk_matmul", "av": "av_matmul"}.get(s, s)].get("rotate", True))
    return len(model.model.layers)


def _quantise_linears(model, pass_args: dict) -> None:
    """Swap the matching linears to chop's quantised modules, one at a time.

    `from_linear` shares the original weight Parameter and quantises it in place; the default path
    allocates a fresh weight and DOUBLES the 32B model (-> OOM). Going name-by-name (not chop's
    whole-model pass, which snapshots every module first) keeps the peak at model + one layer.
    """
    from chop.passes.module.module_modify_helper import get_module_by_name, replace_by_name
    from chop.passes.module.state_dict_map import match_a_pattern
    from chop.nn.quantized.modules import quantized_module_map
    patterns = [k for k in pass_args if k not in ("by", "default")]
    for name in [n for n, _ in model.named_modules() if match_a_pattern(n, patterns)]:
        module = get_module_by_name(model, name)
        cfg = pass_args[match_a_pattern(name, patterns)]["config"]
        new = quantized_module_map[f"linear_{cfg['name']}"].from_linear(module, cfg)
        replace_by_name(model, name, new)
        del module


def quantize_decode_model(spec: dict, gpus: list[int], dtype, recipe: dict | None = None,
                          verbose: bool = True):
    """Build the decode chip for one precision `spec`, sharded across `gpus`.

    RTN: load on `gpus`, quantise the linears in place, wrap the attention.
    GPTQ: load on CPU (chop streams one layer at a time to a GPU to calibrate), then dispatch to `gpus`.
    """
    from chop.passes.module.transforms import quantize_module_transform_pass
    t0 = time.time()
    is_gptq = bool(spec.get("gptq"))

    if is_gptq:
        pass_args = quant.build_gptq_pass_args(recipe, spec)
        dev = str(first_device(gpus))
        for k in ("gptq", "rotation_search"):
            if k in pass_args:
                pass_args[k]["device"] = dev
        model = load_model(dtype, gpus=[], attn_implementation="eager")          # CPU master
        model, _ = quantize_module_transform_pass(model, pass_args)
        with torch.no_grad():
            install_attention_quant(model, spec)
        model = _dispatch(model, gpus, dtype)
    else:
        model = load_model(dtype, gpus, attn_implementation="eager")
        with torch.no_grad():
            _quantise_linears(model, quant.build_pass_args(spec))
            install_attention_quant(model, spec)
    if verbose:
        print(f"  quantised decode chip ({'GPTQ' if is_gptq else 'RTN'}) in {time.time()-t0:.0f}s",
              flush=True)
    return model


def _dispatch(model, gpus: list[int], dtype):
    """Shard a CPU-resident (already-quantised) model across the decode GPUs for inference."""
    from accelerate import dispatch_model, infer_auto_device_map
    dm = infer_auto_device_map(model, max_memory=_max_memory(gpus),
                               no_split_module_classes=["Qwen3DecoderLayer"], dtype=dtype)
    return dispatch_model(model, device_map=dm)


def free_decode(model) -> None:
    """Release a decode chip's GPU shards between precisions. `del`+`gc` alone won't free them
    (accelerate hooks and a lingering generation frame pin the refcount); removing the hooks and
    clearing the module tree drops the ~33 GB tensors regardless. Caller should `del` its handle after."""
    import gc
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass
    for m in model.modules():
        m._parameters.clear()
        m._buffers.clear()
    model._modules.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _kv_quantizer(fmt: str, bits, block: int):
    """Returns f(tensor) that fake-quantises along head_dim to the given MX KV format."""
    if fmt == "mxfp":
        from chop.nn.quantizers.mxfp import mxfp_quantizer
        e, f = bits
        return lambda t: mxfp_quantizer(t, block_size=block, element_exp_bits=e,
                                        element_frac_bits=f, block_dim=-1)
    from chop.nn.quantizers.mxint import mxint_quantizer
    return lambda t: mxint_quantizer(t, block_size=block, element_bits=bits, block_dim=-1)


@torch.no_grad()
def prefill_cache(prefill, input_ids, gpus, want_logits: bool = False) -> dict:
    """Run the bf16 prefill chip ONCE for `input_ids`; return its reusable KV (per-layer CPU tensors)"""
    from transformers import DynamicCache
    out = prefill(input_ids.to(first_device(gpus)), use_cache=True, past_key_values=DynamicCache())
    kv = [(l.keys.detach().to("cpu"), l.values.detach().to("cpu"))
          for l in out.past_key_values.layers if getattr(l, "is_initialized", False)]
    logits = out.logits[:, -1, :].detach().to("cpu") if want_logits else None
    return {"kv": kv, "logits": logits}


def build_decode_cache(kv, spec: dict, decode):
    """Fresh DynamicCache: the cached prompt KV requantised to the decode KV format, on the decode
    model's per-layer devices"""
    from transformers import DynamicCache
    cache = DynamicCache()
    # any non-MX format (the bf16 gold) means identity -- keep the prefill KV as-is, no requant.
    q = None if spec["fmt"] not in ("mxint", "mxfp") else _kv_quantizer(spec["fmt"], spec["kv"], spec["block"])
    dec_layers = decode.model.layers
    for i, (k, v) in enumerate(kv):
        dev = next(dec_layers[i].parameters()).device
        k, v = k.to(dev), v.to(dev)
        if q is not None:
            k, v = q(k), q(v)
        cache.update(k, v, i)
    return cache


def build_prompt_ids(tok, user_text: str, enable_thinking: bool):
    """Render one user turn through Qwen3's chat template (with the thinking switch)."""
    text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)
    return tok(text, return_tensors="pt").input_ids


def split_thinking(text: str) -> str:
    """Return the answer Qwen3 emits after </think> (IFEval scores the answer, not the reasoning)."""
    return text.partition(THINK_CLOSE)[2].strip() if THINK_CLOSE in text else text.strip()


def _sample_token(logits, sampling, ref_ids):
    """Sample one token with Qwen3's warpers (temperature/top-k/top-p/min-p), in HF generate()'s order
    -- so the prefill-emitted first token is drawn exactly as the decode chip draws later ones."""
    from transformers import (LogitsProcessorList, TemperatureLogitsWarper,
                              TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper)
    w = LogitsProcessorList()
    t = sampling.get("temperature", 1.0)
    if t and t != 1.0:
        w.append(TemperatureLogitsWarper(float(t)))
    if sampling.get("top_k"):
        w.append(TopKLogitsWarper(int(sampling["top_k"])))
    if sampling.get("top_p", 1.0) < 1.0:
        w.append(TopPLogitsWarper(float(sampling["top_p"])))
    if sampling.get("min_p", 0.0) > 0:
        w.append(MinPLogitsWarper(float(sampling["min_p"])))
    probs = torch.softmax(w(ref_ids, logits), dim=-1)
    return torch.multinomial(probs, num_samples=1)            # [1, 1]


@torch.no_grad()
def disagg_generate(decode, tok, cached: dict, spec: dict, gpus, *, max_new_tokens, sampling, seed=None):
    """One disaggregated generation from a precomputed prefill cache -> (answer, n_generated_tokens).

    `cached` = {doc, ids, kv, logits}. The prefill chip already built the
    prompt KV and emitted token 1 (sampled from `cached["logits"]`); the decode chip generates the rest.
    So the prompt KV stays prefill precision and every output token's KV is decode precision.
    """
    if seed is not None:
        torch.manual_seed(seed)
    dec_dev = first_device(gpus)
    cache = build_decode_cache(cached["kv"], spec, decode)
    ids = cached["ids"].to(dec_dev)
    first = _sample_token(cached["logits"].to(dec_dev), sampling, ids).to(dec_dev)

    start = torch.cat([ids, first], dim=1)                      # decode forwards `first` (pos L) onward
    out = decode.generate(input_ids=start, attention_mask=torch.ones_like(start),
                          past_key_values=cache, use_cache=True,
                          max_new_tokens=max(max_new_tokens - 1, 1),
                          pad_token_id=tok.eos_token_id, return_dict_in_generate=True, **sampling)
    gen = out.sequences[0, ids.shape[1]:]                       # = [first, decode tokens ...]
    return split_thinking(tok.decode(gen, skip_special_tokens=True)), int(gen.shape[0])


def wikitext_chunks(tok, device, n_chunks: int, chunk_len: int):
    """Fixed-length WikiText-2 token chunks for the continuation-PPL proxy."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids.to(device)
    n = min(n_chunks, ids.shape[1] // chunk_len)
    return [ids[:, i * chunk_len:(i + 1) * chunk_len] for i in range(n)]


@torch.no_grad()
def precompute_ppl_caches(prefill, chunks, half, gpus) -> list[dict]:
    """Prefill each chunk's first half ONCE -> reusable bf16 KV + the second half to score later."""
    return [{"kv": prefill_cache(prefill, ch[:, :half], gpus)["kv"], "cont": ch[:, half:].cpu()}
            for ch in chunks]


@torch.no_grad()
def precompute_prompt_caches(prefill, tok, docs, gpus, enable_thinking: bool) -> list[dict]:
    """Prefill each IFEval prompt ONCE -> reusable bf16 KV + first-token logits + prompt ids + the doc
    (kept for strict scoring later)."""
    out = []
    for d in docs:
        ids = build_prompt_ids(tok, d["prompt"], enable_thinking)
        c = prefill_cache(prefill, ids, gpus, want_logits=True)
        out.append({"doc": d, "ids": ids.cpu(), "kv": c["kv"], "logits": c["logits"]})
    return out


@torch.no_grad()
def continuation_ppl(decode, ppl_caches, spec, gpus):
    """Teacher-forced perplexity on the cached prefill KV -- a fast proxy for RANKING precisions, not
    the deployed IFEval metric. Prefill the first half, score the second half on the decode chip."""
    dec_dev = first_device(gpus)
    nll = tok_n = 0.0
    for c in ppl_caches:
        cache = build_decode_cache(c["kv"], spec, decode)
        cont = c["cont"].to(dec_dev)
        # cont follows the cached first half, so its positions start at the cache length (pass them
        # explicitly so RoPE is correct).
        pos = torch.arange(cache.get_seq_length(), cache.get_seq_length() + cont.shape[1], device=dec_dev)
        logits = decode(cont, past_key_values=cache, use_cache=True, cache_position=pos).logits
        nll += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               cont[:, 1:].reshape(-1).to(logits.device), reduction="sum").item()
        tok_n += cont[:, 1:].numel()
    return math.exp(nll / tok_n)
