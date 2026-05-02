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

MODEL_NAME="${MODEL_NAME:-qwen/Qwen3-32B}"
DEVICE="${DEVICE:-cuda:0}"
QUANT_CONFIG="${QUANT_CONFIG:-quant_eval/configs/qwen_mxint16.toml}"
PREFILL_WIDTH="${PREFILL_WIDTH:-16}"
DECODE_WIDTH="${DECODE_WIDTH:-16}"
TEST_CATEGORY="${TEST_CATEGORY:-web_search_base}"

BASE_PORT=8903
RUN_ID=0

for PREFILL_ATTN in 4 16; do
  for PREFILL_FFN in 4 16; do
    for DECODE_ATTN in  4 16; do
        for DECODE_FFN in 4  16; do

        PORT=$((BASE_PORT + RUN_ID))
        LOG_FILE="logs/bfcl/run_prefill${PREFILL_ATTN}_${PREFILL_FFN}_decode${DECODE_ATTN}_${DECODE_FFN}.log"

        echo "Running PREFILL_ATTN=$PREFILL_ATTN DECODE_ATTN=$DECODE_ATTN PREFILL_FFN=$PREFILL_FFN DECODE_FFN=$DECODE_FFN"

        python -m quant_eval.cli.eval_phase_bfcl \
            --model_name      "$MODEL_NAME"    \
            --quant_config    "$QUANT_CONFIG"  \
            --device_id       "$DEVICE"        \
            --dtype           bfloat16         \
            --bfcl_test_categories   "[\"$TEST_CATEGORY\"]" \
            --prefill_attn_width "$PREFILL_ATTN" \
            --prefill_ffn_width  "$PREFILL_FFN"  \
            --decode_attn_width  "$DECODE_ATTN"  \
            --decode_ffn_width   "$DECODE_FFN"   \
            --log_dir         logs             \
            --limit 15 \
            --server_port     $PORT \
            2>&1 | tee "$LOG_FILE"

        RUN_ID=$((RUN_ID + 1))

        done
    done
  done
done

