#!/usr/bin/env python
"""Profile Qwen3 unified attention-core perturbations on one BFCL prompt.

This is a debugging aid, not part of the DSE runner.  It loads one FP Qwen3
model, installs only the Qwen3 attention unified wrapper, and compares logits
between the all-bypass HF-equivalent path and selected attention-core quant
paths.  It is meant to answer whether qk/av/softmax quantization moves logits
enough to explain BFCL format collapse.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_eval.cli.eval_phase_bfcl import _load_bfcl_multiple_entries, _render_qwen3_bfcl_prompt
from quant_eval.eval.unified_mx import apply_unified_mx_wrappers
from quant_eval.precision import fp_data_config, mx_data_config, parse_fp_setting, parse_mx_precision


DEFAULT_MODEL = (
    "/data/models/pgf23_cache/hub/models--Qwen--Qwen3-32B/"
    "snapshots/9216db5781bf21249d130ec9da846c4624c16137"
)


@dataclass(frozen=True)
class Mode:
    name: str
    qk: bool = False
    av: bool = False
    softmax: bool = False


MODES = (
    Mode("qk", qk=True),
    Mode("av", av=True),
    Mode("softmax", softmax=True),
    Mode("qk_av_softmax", qk=True, av=True, softmax=True),
)
MODE_BY_NAME = {mode.name: mode for mode in (Mode("all_bypass"), *MODES)}


def _resolve_model_path(model_name: str) -> str:
    path = Path(model_name)
    if path.exists():
        return str(path)
    fallback = Path("/data/models/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137")
    if model_name == DEFAULT_MODEL and fallback.exists():
        return str(fallback)
    return model_name


def _set_attention_mode(model: torch.nn.Module, mode: Mode, act_cfg: dict, fp_cfg: dict) -> int:
    count = 0
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("qk_config", "av_config", "softmax_config")):
            continue
        module.qk_config.clear()
        module.qk_config.update(act_cfg if mode.qk else {"bypass": True})
        module.av_config.clear()
        module.av_config.update(act_cfg if mode.av else {"bypass": True})
        module.softmax_config.clear()
        module.softmax_config.update(fp_cfg if mode.softmax else {"bypass": True})
        if hasattr(module, "rope_config"):
            module.rope_config.clear()
            module.rope_config.update({"bypass": True})
        if hasattr(module, "kv_cache_config"):
            module.kv_cache_config.clear()
            module.kv_cache_config.update({"bypass": True})
        module.qk_bypass = not mode.qk
        module.av_bypass = not mode.av
        module.softmax_bypass = not mode.softmax
        if hasattr(module, "rope_bypass"):
            module.rope_bypass = True
        if hasattr(module, "kv_cache_bypass"):
            module.kv_cache_bypass = True
        count += 1
    return count


def _last_token_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref_last = reference[:, -1, :].float()
    cand_last = candidate[:, -1, :].float()
    ref_logp = F.log_softmax(ref_last, dim=-1)
    cand_logp = F.log_softmax(cand_last, dim=-1)
    ref_p = ref_logp.exp()
    return {
        "full_rel_l2": float((candidate.float() - reference.float()).norm() / reference.float().norm()),
        "last_rel_l2": float((cand_last - ref_last).norm() / ref_last.norm()),
        "last_max_abs": float((cand_last - ref_last).abs().max()),
        "last_kl_ref_to_candidate": float(F.kl_div(cand_logp, ref_p, reduction="batchmean", log_target=False)),
        "last_top1_changed": float(cand_last.argmax(dim=-1).ne(ref_last.argmax(dim=-1)).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--bfcl-index", type=int, default=0)
    parser.add_argument("--act", default="MXINT_8")
    parser.add_argument("--fp", default="FP_E8M5")
    parser.add_argument("--mx-block-size", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--generate-tokens", type=int, default=0)
    parser.add_argument(
        "--modes",
        default="qk,av,softmax,qk_av_softmax",
        help="Comma-separated modes for metrics/generation; generation may also include all_bypass.",
    )
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model_name = _resolve_model_path(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    model.to(args.gpu)
    model.config._attn_implementation = "eager"

    rows = _load_bfcl_multiple_entries(limit=args.bfcl_index + 1)
    row = rows[args.bfcl_index]
    prompt = _render_qwen3_bfcl_prompt(row)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if encoded.input_ids.shape[-1] > args.max_prompt_tokens:
        encoded = {k: v[:, -args.max_prompt_tokens :] for k, v in encoded.items()}
    encoded = {k: v.to(args.gpu) for k, v in encoded.items()}

    act_cfg = mx_data_config(parse_mx_precision(args.act), args.mx_block_size)
    fp_cfg = fp_data_config(parse_fp_setting(args.fp))
    counts = apply_unified_mx_wrappers(
        model,
        qwen3_attention_config={
            **act_cfg,
            "qk_matmul": {"bypass": True},
            "av_matmul": {"bypass": True},
            "softmax": {"bypass": True},
            "rope": {"bypass": True},
            "kv_cache": {"bypass": True},
        },
    )
    print(f"model={model_name}")
    print(f"bfcl_id={row.get('id')} prompt_tokens={encoded['input_ids'].shape[-1]}")
    print(f"wrapper_counts={counts}")

    with torch.inference_mode():
        _set_attention_mode(model, Mode("all_bypass"), act_cfg, fp_cfg)
        reference = model(**encoded).logits.detach()
        ref_token = int(reference[:, -1, :].argmax(dim=-1)[0])
        print(f"baseline_top1={ref_token} {tokenizer.decode([ref_token])!r}")

        selected_names = [name.strip() for name in args.modes.split(",") if name.strip()]
        invalid = [name for name in selected_names if name not in MODE_BY_NAME]
        if invalid:
            raise ValueError(f"Unknown modes {invalid}; expected one of {sorted(MODE_BY_NAME)}")
        selected_modes = [MODE_BY_NAME[name] for name in selected_names if name != "all_bypass"]

        for mode in selected_modes:
            _set_attention_mode(model, mode, act_cfg, fp_cfg)
            candidate = model(**encoded).logits.detach()
            metrics = _last_token_metrics(reference, candidate)
            cand_token = int(candidate[:, -1, :].argmax(dim=-1)[0])
            print(
                f"{mode.name}: "
                + " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
                + f" top1={cand_token} {tokenizer.decode([cand_token])!r}"
            )

        if args.generate_tokens > 0:
            generation_modes = [MODE_BY_NAME[name] for name in selected_names]
            if not generation_modes:
                generation_modes = [Mode("all_bypass")]
            for mode in generation_modes:
                _set_attention_mode(model, mode, act_cfg, fp_cfg)
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.generate_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated = output_ids[:, encoded["input_ids"].shape[-1] :]
                text = tokenizer.decode(generated[0], skip_special_tokens=False)
                compact = text.replace("\n", "\\n")
                hit_limit = generated.shape[-1] >= args.generate_tokens
                has_tool_call = "<tool_call>" in text and "</tool_call>" in text
                print(
                    f"generate[{mode.name}] tokens={generated.shape[-1]} "
                    f"hit_limit={hit_limit} has_tool_call={has_tool_call} "
                    f"head={compact[:350]!r} tail={compact[-350:]!r}"
                )


if __name__ == "__main__":
    main()
