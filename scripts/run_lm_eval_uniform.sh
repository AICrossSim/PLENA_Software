#!/bin/bash
# lm-eval with uniform quantization (no phase switching)
# Works for any lm-eval task (generation or log-likelihood).

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
DEVICE="${DEVICE:-cuda:0}"
QUANT_CONFIG="${QUANT_CONFIG:-quant_eval/configs/llama_mxint4.toml}"
TASKS="${TASKS:-wikitext}"

python -m quant_eval.cli.eval_lm \
    --model_name "$MODEL_NAME" \
    --quant_config "$QUANT_CONFIG" \
    --device_id "$DEVICE" \
    --dtype bfloat16 \
    --tasks "$TASKS" \
    --seqlen 2048 \
    --log_dir logs
