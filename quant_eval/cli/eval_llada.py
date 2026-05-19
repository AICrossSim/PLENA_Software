"""
LLaDA (diffusion-style language model) evaluation with optional MX quantization.

Wraps lm-eval-harness's CLI. Use ``--model llada_dist`` and pass model and
quantization options through lm-eval's ``--model_args`` flag.

Example — baseline (prefix cache):

    python -m quant_eval.cli.eval_llada \\
        --tasks gsm8k --num_fewshot 0 \\
        --model llada_dist \\
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True

Example — with MXINT4 KV-cache quantization:

    python -m quant_eval.cli.eval_llada \\
        --tasks gsm8k --num_fewshot 0 \\
        --model llada_dist \\
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True,quant_config='quant_eval/configs/llama_mxint4.toml'
"""

# Import to trigger @register_model("llada_dist")
import quant_eval.eval.llada.eval_llada  # noqa: F401

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
