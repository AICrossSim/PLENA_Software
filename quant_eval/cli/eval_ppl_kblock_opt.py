import time

import torch
import transformers

from quant_eval.experimental.kblock_mixed_linear_opt import replace_linears_with_kblock_mixed
from quant_eval.utils import setup_model
from quant_eval.eval import evaluate_perplexity


def _build_config(mode, element_bits, element_exp_bits, element_frac_bits, block_size):
    if mode in ("bypass", "fp"):
        return {"mode": "bypass"}
    if mode in ("mxint", "int"):
        return {"mode": "mxint", "block_size": block_size,
                "element_bits": element_bits, "element_frac_bits": element_frac_bits}
    if mode == "mxfp":
        return {"mode": "mxfp", "block_size": block_size,
                "element_exp_bits": element_exp_bits, "element_frac_bits": element_frac_bits}
    raise ValueError(f"Unknown mode: {mode!r}")


def main(
    model_name: str = "unsloth/Llama-3.2-1B",
    dataset: str = "wikitext",
    device_id: str = "cuda:0",
    dtype: str = "float16",
    seqlen: int = 2048,
    num_eval_batches: int | None = None,

    kblock_mixed: bool = False,
    k_tile: int = 128,
    low_percent: int = 50,
    placement: str = "uniform",
    physical_pattern_len: int = 16,

    high_mode: str = "bypass",
    high_element_bits: int = 8,
    high_element_exp_bits: int = 4,
    high_element_frac_bits: int = 4,

    low_mode: str = "mxint",
    low_element_bits: int = 8,
    low_element_exp_bits: int = 4,
    low_element_frac_bits: int = 4,

    block_size: int = 32,

    acc_fp_exp_bits: int = 8,
    acc_fp_mant_bits: int = 23,
    out_fp_exp_bits: int = 8,
    out_fp_mant_bits: int = 23,
):
    transformers.set_seed(0)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer, model = setup_model(
        model_name, model_parallel=False,
        dtype=dtype_map.get(dtype, torch.float16),
        device=device_id, attn_implementation="sdpa",
    )
    model.eval()

    if kblock_mixed:
        high_config = _build_config(high_mode, high_element_bits, high_element_exp_bits,
                                    high_element_frac_bits, block_size)
        low_config  = _build_config(low_mode,  low_element_bits,  low_element_exp_bits,
                                    low_element_frac_bits,  block_size)
        replaced = replace_linears_with_kblock_mixed(
            model,
            k_tile=k_tile,
            low_percent=low_percent,
            placement=placement,
            physical_pattern_len=physical_pattern_len,
            high_config=high_config,
            low_config=low_config,
            acc_fp_exp_bits=acc_fp_exp_bits,
            acc_fp_mant_bits=acc_fp_mant_bits,
            out_fp_exp_bits=out_fp_exp_bits,
            out_fp_mant_bits=out_fp_mant_bits,
        )
        print(f"replaced {len(replaced)} modules | high={high_config} | low={low_config} | "
              f"k_tile={k_tile} low_percent={low_percent}% placement={placement}")

    model.to(device_id)

    return evaluate_perplexity(
        model=model, tokenizer=tokenizer,
        dataset_name=dataset, max_length=seqlen,
        verbose=True, num_eval_batches=num_eval_batches,
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    t0 = time.time()
    CLI(main)
    print(f"total: {time.time() - t0:.1f}s")
