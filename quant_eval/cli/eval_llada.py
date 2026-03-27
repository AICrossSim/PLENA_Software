"""
LLaDA evaluation with Fast-dLLM v1 KV cache and optional MASE quantization.

Wraps lm-eval's cli_evaluate with the "llada_dist" model registered in
quant_eval.eval.eval_llada. Quantization is passed via model_args.

Usage:
    # Baseline (prefix cache):
    python -m quant_eval.cli.eval_llada --tasks gsm8k --num_fewshot 0 \
        --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True,show_speed=True

    # With MXINT4 KV cache quantization:
    python -m quant_eval.cli.eval_llada --tasks gsm8k --num_fewshot 0 \
        --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True,quant_config='configs/kv_only_mxint4.toml'
"""

# Import to trigger @register_model("llada_dist")
import quant_eval.eval.llada.eval_llada  # noqa: F401

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
