"""
Fast-dLLM v2 evaluation via the legacy ``llada_dist`` lm-eval alias.

Use ``--model llada_dist`` and pass Fast-dLLM v2 options through lm-eval's
``--model_args`` flag.

Example:

    python -m quant_eval.cli.eval_llada \\
        --tasks gsm8k --num_fewshot 0 \\
        --model llada_dist \\
        --model_args model_path='Efficient-Large-Model/Fast_dLLM_v2_7B',gen_length=256,steps=256,block_length=32,use_cache=True
"""

# Import to trigger @register_model("llada_dist")
import quant_eval.eval.llada.eval_llada  # noqa: F401

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
