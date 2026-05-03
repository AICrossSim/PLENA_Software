#!/bin/bash
# Run a sweep of ablation TOMLs through eval_evalplus.py on humaneval/mbpp.
#
# CONFIGS is REQUIRED — pass a CSV of TOML paths. No glob, no basenames.
#
# Examples (run from repo root):
#   CONFIGS=plena_experiments/table10/configs/humaneval/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
#       bash plena_experiments/table10/scripts/run_ablation_humaneval.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-32B}
DEVICE=${DEVICE:-cuda:0}
DATASET=${DATASET:-humaneval}            # humaneval or mbpp (evalplus eval target)
GREEDY=${GREEDY:-true}
N_SAMPLES=${N_SAMPLES:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
LOG_DIR=${LOG_DIR:-logs/ablation_${DATASET}_$(date +%Y%m%d_%H%M%S)}

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

if [[ -z "${CONFIGS:-}" ]]; then
    echo "ERROR: CONFIGS env var is required (CSV of TOML paths)." >&2
    echo "  Example: CONFIGS=plena_experiments/table10/configs/humaneval/06_w4_act4_kv4_gptq_erryclip_selrot.toml \\" >&2
    echo "             bash $0" >&2
    exit 1
fi

IFS=',' read -r -a selected_tomls <<< "$CONFIGS"
for toml in "${selected_tomls[@]}"; do
    if [[ ! -f "$toml" ]]; then
        echo "ERROR: $toml does not exist" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"

echo "Sweep:"
for t in "${selected_tomls[@]}"; do echo "  - $t"; done

for toml in "${selected_tomls[@]}"; do
    tag=$(basename "$toml" .toml)
    echo ">>> $tag"
    $PY -m quant_eval.cli.eval_evalplus \
        --model_name           "$MODEL" \
        --device_id            "$DEVICE" \
        --quant_config         "$toml" \
        --dataset              "$DATASET" \
        --batch_size           $BATCH_SIZE \
        --n_samples            $N_SAMPLES \
        --max_new_tokens       $MAX_NEW_TOKENS \
        --greedy               "$GREEDY" \
        --evalplus_output_dir  "$LOG_DIR/$tag/evalplus_out" \
        --log_dir              "$LOG_DIR/$tag" \
        2>&1 | tee "$LOG_DIR/$tag.log" || echo "FAIL: $tag"
done

echo "Run dir: $LOG_DIR"
