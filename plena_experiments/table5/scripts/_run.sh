#!/usr/bin/env bash
# Sourced by per-config scripts in plena_experiments/tableN/scripts/.
#
# Required vars (set by caller before sourcing):
#   MODEL          HF model id   (e.g. meta-llama/Llama-2-7b-hf)
#   LOG_DIR_NAME   subdir under quant_eval/logs/   
# Optional vars:
#   QUANT_CONFIG   path to TOML  (omit for unquantized baseline)
#   DEVICE         cuda device   (default cuda:0)
#   DTYPE          float16 | bfloat16 | float32   (default float16)
#   SEQLEN         (default 2048)
#   DATASET        (default wikitext)

set -euo pipefail

: "${MODEL:?MODEL must be set}"
: "${LOG_DIR_NAME:?LOG_DIR_NAME must be set}"
: "${REPO_ROOT:?REPO_ROOT must be set}"

DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
SEQLEN="${SEQLEN:-2048}"
DATASET="${DATASET:-wikitext}"

cd "$REPO_ROOT"

LOG_DIR="quant_eval/logs/${LOG_DIR_NAME}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOGFILE="${LOG_DIR}/run-${DEVICE##*:}-${TS}.log"

cmd=(.venv/bin/python -m quant_eval.cli.eval_ppl
     --model_name "$MODEL"
     --dataset "$DATASET"
     --device_id "$DEVICE"
     --dtype "$DTYPE"
     --seqlen "$SEQLEN"
     --log_dir "$LOG_DIR")

if [[ -n "${QUANT_CONFIG:-}" ]]; then
    cmd+=(--quant_config "$QUANT_CONFIG")
fi

echo "[table5] launching: $LOG_DIR_NAME  (device=$DEVICE, dtype=$DTYPE)"
echo "[table5] log:       $LOGFILE"
echo "[table5] cmd:       ${cmd[*]}"
echo

"${cmd[@]}" 2>&1 | tee "$LOGFILE"
