"""
Run the calibration-aware per-matmul rotation search.

Loads a base TOML quantization config (with all online Hadamard rotations
OFF), runs ``rotation_search_transform_pass`` from MASE, and writes a JSON
summary with the per-matmul-type perplexities, the greedy winners, and the
final-combined perplexity.

Example:
    python -m quant_eval.cli.search_rotation \\
        --base_config plena_experiments/table9/configs/gsm8k/04_w4_act4_kv4_gptq.toml \\
        --calib_data file:calib/Qwen_Qwen3-8B_gsm8k_n64_s1024.pt \\
        --calib_nsamples 32 \\
        --calib_seqlen 1024 \\
        --output_json /data/models/cx922/plena_camera/rotation_search_qwen3_8b.json
"""

import time
from typing import Union

import torch
import transformers

from quant_eval.utils import (
    get_logger,
    set_logging_verbosity,
    setup_model,
    create_experiment_log_dir,
    save_args,
    save_results,
)
from quant_eval.quantize import load_quant_config

logger = get_logger(__name__)
set_logging_verbosity("debug")


def main(
    model_name: str = "Qwen/Qwen3-8B",
    base_config: str = "plena_experiments/table9/configs/gsm8k/05_w4_act4_kv4_gptq_erryclip.toml",
    calib_data: str = "file:calib/Qwen_Qwen3-8B_gsm8k_n64_s1024.pt",
    device_id: str = "cuda:0",
    dtype: str = "bfloat16",
    calib_nsamples: int = 32,
    calib_seqlen: int = 1024,
    matmul_types: Union[str, None] = None,
    output_json: Union[str, None] = None,
    improvement_eps: float = 0.0,
    log_dir: Union[str, None] = None,
):
    """Search per-matmul-type online Hadamard rotation by calibration ppl.

    Args:
        model_name:        HuggingFace model ID (must match base_config).
        base_config:       TOML quantize config; should set name = "mxint"
                           everywhere (no rotation). The search adds rotation
                           per matmul type and measures ppl.
        calib_data:        Path/spec for calibration loader, e.g.
                           ``"file:calib/Qwen_Qwen3-8B_gsm8k_n64_s1024.pt"``
                           or ``"wikitext2"``.
        device_id:         CUDA device for forward passes.
        dtype:             Model dtype.
        calib_nsamples:    Number of calibration samples to score ppl on.
        calib_seqlen:      Sequence length for the calib loader.
        matmul_types:      Optional comma-separated subset, e.g.
                           ``"q_proj,o_proj,qk_matmul"``. Default: all 9.
        output_json:       Where to write the JSON results summary.
        improvement_eps:   Only flag a matmul type as a winner if it beats
                           baseline ppl by more than this margin.
        log_dir:           Directory for experiment artefacts.
    """
    print("=" * 64)
    print("Calibration-aware per-matmul rotation search")
    print("=" * 64)
    print(f"  Model       : {model_name}")
    print(f"  Base config : {base_config}")
    print(f"  Calib data  : {calib_data}")
    print(f"  Calib n/seq : {calib_nsamples} x {calib_seqlen}")
    print(f"  Output JSON : {output_json}")
    print("=" * 64)

    if log_dir:
        log_dir = create_experiment_log_dir(log_dir)
        save_args(log_dir, locals().copy())
        import shutil
        shutil.copy(base_config, log_dir / "base_config.toml")

    transformers.set_seed(0)

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Quantized attention modules (MXInt + *Rotate variants) replace the
    # eager forward and assert _attn_implementation == "eager".
    tokenizer, model = setup_model(
        model_name,
        model_parallel=False,
        dtype=torch_dtype,
        device=device_id,
        attn_implementation="eager",
    )
    model.eval()

    # Load the calibration loader the same way GPTQ does — same data
    # everyone in this project already feeds into quantization.
    from chop.passes.module.transforms.gptq.data_utils import get_loaders
    calib_loader = get_loaders(
        calib_data,
        nsamples=calib_nsamples,
        seed=0,
        seqlen=calib_seqlen,
        model=model_name,
    )
    logger.info("Loaded %d calibration batches.", len(calib_loader))

    base_pass_args = load_quant_config(base_config)
    if "gptq" in base_pass_args:
        # Plumb the device through so GPTQ runs on the same GPU.
        base_pass_args["gptq"]["device"] = device_id

    selected_types = None
    if matmul_types:
        selected_types = [t.strip() for t in matmul_types.split(",") if t.strip()]

    from chop.passes.module.transforms import rotation_search_transform_pass
    from chop.passes.module.transforms.quantize import ALL_MATMUL_TYPES

    search_args = {
        "base_quantize_args": base_pass_args,
        "calib_loader": calib_loader,
        "device": device_id,
        "matmul_types": selected_types or ALL_MATMUL_TYPES,
        "output_json": output_json,
        "improvement_eps": improvement_eps,
    }

    t0 = time.time()
    model, results = rotation_search_transform_pass(model, search_args)
    logger.info("Rotation search complete in %.1fs", time.time() - t0)

    print("\n" + "=" * 64)
    print("Rotation search results:")
    print("=" * 64)
    print(f"  baseline_ppl : {results['baseline_ppl']:.4f}")
    print(f"  final_ppl    : {results['final_ppl']:.4f}  "
          f"(Δ={results['baseline_ppl']-results['final_ppl']:+.4f} from baseline)")
    print(f"  winners      : {results['winners']}  (in commit order)")
    print(f"  total trials : {results.get('n_trials', 'n/a')}")

    print("\n  Round history:")
    for r in results.get("rounds", []):
        sel = r["selected"]
        if sel is None:
            print(
                f"   round {r['round']}: stopped at ppl={r['current_ppl_after']:.4f}"
            )
            continue
        delta = r["current_ppl_before"] - r["current_ppl_after"]
        print(
            f"   round {r['round']}: +{sel:<11s}  "
            f"ppl {r['current_ppl_before']:.4f} -> {r['current_ppl_after']:.4f}  "
            f"Δ={delta:+.4f}"
        )

    if log_dir:
        save_results(log_dir, results)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    start_time = time.time()
    CLI(main)
    total_time = time.time() - start_time
    print(f"\n[INFO] Total workload time: {total_time:.2f} seconds")
