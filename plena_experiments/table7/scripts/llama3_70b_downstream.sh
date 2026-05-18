#!/usr/bin/env bash
# Table 7 — LLaMA-3-70B downstream task evaluation (lm-eval-harness)
# Reuses the GPTQ checkpoint + cached rotation_decisions produced by Table 5.
#
# Usage: CUDA_VISIBLE_DEVICES=N bash plena_experiments/table7/scripts/llama3_70b_downstream.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="meta-llama/Meta-Llama-3-70B"
QUANT_CONFIG="plena_experiments/table7/configs/llama3_70b.toml"
TASKS="${TASKS:-piqa,winogrande,hellaswag,arc_easy,arc_challenge,lambada_openai}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
SEQLEN="${SEQLEN:-2048}"
LOG_DIR_NAME="${LOG_DIR_NAME:-table7_llama3_70b_downstream}"

LOG_DIR="quant_eval/logs/${LOG_DIR_NAME}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOGFILE="${LOG_DIR}/run-${DEVICE##*:}-${TS}.log"


echo "[table7] launching: $LOG_DIR_NAME  (device=$DEVICE, dtype=$DTYPE, batch=$BATCH_SIZE)"
echo "[table7] log:       $LOGFILE"

.venv/bin/python -m quant_eval.cli.eval_lm \
    --model_name "$MODEL" \
    --tasks "$TASKS" \
    --device_id "$DEVICE" \
    --dtype "$DTYPE" \
    --seqlen "$SEQLEN" \
    --batch_size "$BATCH_SIZE" \
    --quant_config "$QUANT_CONFIG" \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOGFILE"
