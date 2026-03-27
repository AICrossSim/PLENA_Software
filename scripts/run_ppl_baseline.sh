#!/bin/bash
# Perplexity evaluation — unquantized baseline
# Pure prefill, no quantization applied.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
DEVICE="${DEVICE:-cuda:0}"

python -m quant_eval.cli.eval_ppl \
    --model_name "$MODEL_NAME" \
    --device_id "$DEVICE" \
    --dtype bfloat16 \
    --seqlen 2048 \
    --log_dir logs
