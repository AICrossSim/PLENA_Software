"""
Memory footprint / peak-usage profiler for Fast-dLLM v2.

Loads a model **once** (optionally quantized) and sweeps a grid of block sizes
and batch sizes, running a single generation batch per cell and recording:

  * ``model_footprint_mb`` — CUDA memory allocated by the resident model weights
    (measured once after load; identical across cells for a given quant config).
  * ``peak_memory_mb``     — peak CUDA memory allocated while generating one
    batch (weights + activations + KV cache), isolated per cell.

Each cell is emitted on stdout as a single parseable line, e.g.::

    MEM_RESULT|bd=16|bs=4|cache=false|footprint_mb=14820.50|peak_mb=15960.20|peak_reserved_mb=16210.00

``benchmark_memory.sh`` consumes these lines and merges them into
``all_results.jsonl`` by ``run_id``.

Example::

    python -m quant_eval.cli.profile_memory \\
        --model_name Efficient-Large-Model/Fast_dLLM_v2_7B \\
        --bd_sizes '[8,16,32,64]' --batch_sizes '[1,4,8,16,32]'
"""

from typing import List, Union
import time

import torch
import transformers
from datasets import load_dataset

from quant_eval.utils import get_logger, set_logging_verbosity, setup_model
from quant_eval.eval.dllm_v2.dllm_generation import (
    setup_dllm_generation,
    FAST_DLLM_MASK_ID,
)
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("info")

MB = 1024.0 * 1024.0


def _build_batch(tokenizer, questions, mask_id, device):
    """Tokenize gsm8k-style prompts and right-pad with the mask id, mirroring
    the eval harness's ``generate_until`` batching."""
    encoded = []
    for q in questions:
        text = ("Question: " + q + "\nAnswer:").replace(
            "Answer:",
            "Please reason step by step, and put your final answer within \\boxed{}.",
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded.append(tokenizer([prompt], return_tensors="pt").to(device)["input_ids"])

    max_len = max(e.shape[1] for e in encoded)
    seq_len = [e.shape[1] for e in encoded]
    padded = [
        torch.cat(
            [
                e,
                torch.full(
                    (1, max_len - e.shape[1]), mask_id, dtype=torch.long, device=device
                ),
            ],
            dim=1,
        )
        for e in encoded
    ]
    batch_ids = torch.cat(padded, dim=0)
    return batch_ids, seq_len


def main(
    model_name: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
    quant_config: Union[str, None] = None,
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    bd_sizes: List[int] = [8, 16, 32, 64],
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    small_block_size: int = 8,
    max_new_tokens: int = 512,
    threshold: float = 1.0,
    temperature: float = 0.0,
    block_caches: List[bool] = [False, True],
    mask_id: int = FAST_DLLM_MASK_ID,
):
    """
    Profile model footprint and peak generation memory across a bd_size ×
    batch_size grid. The model is loaded once; each grid cell runs a single
    generation batch with peak-memory stats reset beforehand so cells don't
    contaminate each other.
    """
    print("=" * 60)
    print("Fast-dLLM Memory Profiler")
    print(f"Model: {model_name}")
    print(f"Quant: {quant_config or 'None (baseline)'}")
    print(f"bd_sizes={bd_sizes}  batch_sizes={batch_sizes}")
    print("=" * 60)

    transformers.set_seed(0)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    device = torch.device(device_id)
    # Initialize the CUDA context before touching memory stats — the memory
    # APIs raise "Invalid device argument" if called before any CUDA init.
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer, model = setup_model(
        model_name, False, dtype=torch_dtype, device=device_id
    )
    model.eval()

    if quant_config is not None:
        from chop.passes.module.transforms import quantize_module_transform_pass

        pass_args = load_quant_config(quant_config)
        if "gptq" in pass_args:
            pass_args["gptq"]["device"] = device_id
        t0 = time.time()
        model, _ = quantize_module_transform_pass(model, pass_args)
        logger.info("Quantization complete in %.1fs", time.time() - t0)

    model.to(device_id)
    setup_dllm_generation(model)

    # Resident model footprint: allocated memory once weights are settled.
    torch.cuda.synchronize(device)
    footprint_mb = torch.cuda.memory_allocated(device) / MB
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / MB
    buffer_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / MB
    print(
        f"Model footprint: {footprint_mb:.2f} MB allocated "
        f"(params {param_mb:.2f} MB + buffers {buffer_mb:.2f} MB)"
    )

    # A pool of real gsm8k prompts to draw batches from (largest bs needed).
    ds = load_dataset("openai/gsm8k", "main", split="test")
    pool = [ds[i]["question"] for i in range(max(batch_sizes))]

    for bd in bd_sizes:
        for bs in batch_sizes:
            for bc in block_caches:
                batch_ids, seq_len = _build_batch(tokenizer, pool[:bs], mask_id, device)
                min_len = min(seq_len)

                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

                try:
                    with torch.no_grad():
                        model.mdm_sample(
                            batch_ids,
                            tokenizer=tokenizer,
                            block_size=bd,
                            small_block_size=small_block_size,
                            max_new_tokens=max_new_tokens,
                            mask_id=mask_id,
                            min_len=int(min_len),
                            seq_len=torch.tensor(seq_len, device=device),
                            use_block_cache=bc,
                            threshold=threshold,
                            temperature=temperature,
                        )
                    torch.cuda.synchronize(device)
                    peak_mb = torch.cuda.max_memory_allocated(device) / MB
                    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / MB
                    print(
                        f"MEM_RESULT|bd={bd}|bs={bs}|cache={str(bc).lower()}"
                        f"|footprint_mb={footprint_mb:.2f}"
                        f"|peak_mb={peak_mb:.2f}|peak_reserved_mb={peak_reserved_mb:.2f}"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"MEM_OOM|bd={bd}|bs={bs}|cache={str(bc).lower()} — skipped (out of memory)")

    print("\n[INFO] Memory profiling complete.")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)