#!/bin/bash
# lm-eval with phase-dependent quantization (prefill vs decode)
#
# PhaseAutoSwitch detects the prefill->decode boundary automatically:
#   - seq_len > 1  -> prefill activation config
#   - seq_len == 1 -> decode activation config
#
# Use generation tasks (gsm8k, humaneval) to exercise both phases.
# Log-likelihood tasks (wikitext, mmlu) are pure prefill.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
DEVICE="${DEVICE:-cuda:0}"
QUANT_CONFIG="${QUANT_CONFIG:-quant_eval/configs/llama_mxint4.toml}"
TASKS="${TASKS:-gsm8k}"
PREFILL_WIDTH="${PREFILL_WIDTH:-4}"
DECODE_WIDTH="${DECODE_WIDTH:-8}"

python -m quant_eval.cli.eval_phase_lm \
    --model_name "$MODEL_NAME" \
    --quant_config "$QUANT_CONFIG" \
    --device_id "$DEVICE" \
    --dtype bfloat16 \
    --tasks "$TASKS" \
    --prefill_data_in_width "$PREFILL_WIDTH" \
    --decode_data_in_width "$DECODE_WIDTH" \
    --seqlen 2048 \
    --log_dir logs
