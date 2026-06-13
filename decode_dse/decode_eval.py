"""Two-chip disaggregated decode

Disaggregated serving splits inference across two physical chips:

  * PREFILL chip -- FP16. Reads the prompt and builds the KV cache.
  * DECODE chip  -- lower precision. Receives the prefill KV cache - requantized
    to its own KV precision at the hand-off and generates every subsequent token.

Score decode with continuation perplexity
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from decode_dse.quant import MODEL_NAME


def load_prefill(device, dtype):
    """The FP16 prefill chip."""
    from quant_eval.utils import setup_model
    tok, model = setup_model(MODEL_NAME, model_parallel=False, dtype=dtype,
                             device=device, attn_implementation="sdpa")
    model.eval()
    model.to(device)   # setup_model does NOT move the model; without this it runs on CPU
    return tok, model


def quantize_decode_model(pass_args, device, dtype, label="decode", verbose=True):
    """Load a fresh FP16 model and apply a MASE pass-args dict -> the decode chip."""
    from quant_eval.utils import setup_model
    from chop.passes.module.transforms import quantize_module_transform_pass
    tok, model = setup_model(MODEL_NAME, model_parallel=False, dtype=dtype,
                             device=device, attn_implementation="eager")
    model.eval()
    model.to(device)
    for k in ("gptq", "rotation_search"):
        if k in pass_args:
            pass_args[k]["device"] = device
    t0 = time.time()
    model, _ = quantize_module_transform_pass(model, pass_args)
    model.to(device)
    if verbose:
        print(f"  decode chip quantized ({label}) in {time.time()-t0:.1f}s")
    return tok, model


def requant_kv(cache, cfg, block=32):
    """KV hand-off: requantize the FP16 prefill KV cache to the decode chip's KV
    format. Once stored at low precision this is lossy and unrecoverable.

    cfg = {"fmt": "mxint"|"mxfp", "bits": int | (exp, frac)}.
    """
    fmt = cfg["fmt"]
    if fmt == "fp16" or cache is None:
        return cache
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if not isinstance(kc, list):
        return cache
    if fmt == "mxfp":
        from chop.nn.quantizers.mxfp import mxfp_quantizer
        e, f = cfg["bits"]
        q = lambda t: mxfp_quantizer(t, block_size=block, element_exp_bits=e, element_frac_bits=f, block_dim=-1)
    else:
        from chop.nn.quantizers.mxint import mxint_quantizer
        b = cfg["bits"]
        if b >= 16:
            return cache
        q = lambda t: mxint_quantizer(t, block_size=block, element_bits=b, block_dim=-1)
    for i in range(len(kc)):
        if kc[i] is not None and kc[i].numel():
            kc[i] = q(kc[i])
        if vc[i] is not None and vc[i].numel():
            vc[i] = q(vc[i])
    return cache


def wikitext_chunks(tok, device, n_chunks, chunk_len):
    """Tokenize wikitext-2 test into n_chunks fixed-length chunks."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    n = min(n_chunks, ids.shape[1] // chunk_len)
    return [ids[:, i * chunk_len:(i + 1) * chunk_len] for i in range(n)]


@torch.no_grad()
def continuation_ppl(prefill, decode, chunks, cfg, half):
    """Prefill first half on the FP16 chip, hand KV over, score second half on decode chip."""
    from transformers import DynamicCache
    total_nll, total_tok = 0.0, 0
    for chunk in chunks:
        pre, cont = chunk[:, :half], chunk[:, half:]
        cache = prefill(pre, use_cache=True, past_key_values=DynamicCache()).past_key_values
        cache = requant_kv(cache, cfg)
        logits = decode(cont, past_key_values=cache, use_cache=False).logits
        # logits[t] predicts cont[t+1]; score cont[1:] (drop the boundary token; relative metric)
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                              cont[:, 1:].reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += cont[:, 1:].numel()
    return math.exp(total_nll / total_tok)
