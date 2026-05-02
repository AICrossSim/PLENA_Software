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
PREFILL_WIDTH="${PREFILL_WIDTH:-16}"
DECODE_WIDTH="${DECODE_WIDTH:-16}"

for PREFILL_ATTN in  16; do
  for PREFILL_FFN in  16; do
    for DECODE_ATTN in   16; do
        for DECODE_FFN in  16; do

        LOG_FILE="logs/run_prefill${PREFILL_ATTN}_${PREFILL_FFN}_decode${DECODE_ATTN}_${DECODE_FFN}.log"

        echo "Running PREFILL_ATTN=$PREFILL_ATTN DECODE_ATTN=$DECODE_ATTN PREFILL_FFN=$PREFILL_FFN DECODE_FFN=$DECODE_FFN"

        python -m quant_eval.cli.eval_phase_lm \
            --model_name "$MODEL_NAME" \
            --quant_config "$QUANT_CONFIG" \
            --device_id "$DEVICE" \
            --dtype bfloat16 \
            --tasks "$TASKS" \
            --prefill_attn_width "$PREFILL_ATTN" \
            --decode_attn_width "$DECODE_ATTN" \
            --prefill_ffn_width "$PREFILL_FFN" \
            --decode_ffn_width "$DECODE_FFN" \
            --seqlen 2048 \
            --log_dir logs \
            --limit 0.05 \
            2>&1 | tee "$LOG_FILE"

        done
    done
  done
done

