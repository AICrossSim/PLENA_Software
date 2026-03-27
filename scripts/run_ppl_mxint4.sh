#!/bin/bash
# Perplexity evaluation — uniform MXInt4 quantization
# Pure prefill, all layers at MXInt4.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
DEVICE="${DEVICE:-cuda:0}"
QUANT_CONFIG="${QUANT_CONFIG:-quant_eval/configs/llama_mxint4.toml}"

python -m quant_eval.cli.eval_ppl \
    --model_name "$MODEL_NAME" \
    --quant_config "$QUANT_CONFIG" \
    --device_id "$DEVICE" \
    --dtype bfloat16 \
    --seqlen 2048 \
    --log_dir logs
