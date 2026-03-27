#!/bin/bash
# Perplexity evaluation — mixed precision (MXInt8 attention, MXInt4 MLP)
# Pure prefill. Attention at higher precision to preserve KV cache quality.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
DEVICE="${DEVICE:-cuda:0}"
QUANT_CONFIG="${QUANT_CONFIG:-quant_eval/configs/qwen_mxint8_attn_mxint4_mlp.toml}"

python -m quant_eval.cli.eval_ppl \
    --model_name "$MODEL_NAME" \
    --quant_config "$QUANT_CONFIG" \
    --device_id "$DEVICE" \
    --dtype bfloat16 \
    --seqlen 2048 \
    --log_dir logs
