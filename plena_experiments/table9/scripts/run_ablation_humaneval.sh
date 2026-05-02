#!/bin/bash
# Run a sweep of ablation TOMLs through eval_evalplus.py on humaneval/mbpp.
#
# CONFIGS is REQUIRED — pass a CSV of TOML paths. No glob, no basenames.
#
# Stage 1 (auto): if any config uses `dataset = "file:..."` and the file
# isn't on disk, run $CALIB_CONFIG via eval_evalplus to fill it. The hook
# captures input_ids during real evalplus forwards on $DATASET; same
# framework as Stage 2, so calibration is task-aligned by construction.
# Stage 2: run each config via eval_evalplus.
#
# Examples:
#   CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/04_w4_act4_kv4_gptq.toml \
#       bash quant_eval/scripts/run_ablation_humaneval.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-8B}
DEVICE=${DEVICE:-cuda:0}
DATASET=${DATASET:-humaneval}            # humaneval or mbpp (evalplus eval target)
GREEDY=${GREEDY:-true}
N_SAMPLES=${N_SAMPLES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
CALIB_CONFIG=${CALIB_CONFIG:-quant_eval/configs/calibrate/qwen3_8b_humaneval.toml}
LOG_DIR=${LOG_DIR:-logs/ablation_humaneval_$(date +%Y%m%d_%H%M%S)}

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

if [[ -z "${CONFIGS:-}" ]]; then
    echo "ERROR: CONFIGS env var is required (CSV of TOML paths)." >&2
    echo "  Example: CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/04_w4_act4_kv4_gptq.toml \\" >&2
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

# Stage 1 — collect calibration if any selected TOML references "file:..." that isn't on disk.
calib_paths=$(
    grep -h '^dataset *= *"file:' "${selected_tomls[@]}" 2>/dev/null \
        | sed 's/.*"file:\([^"]*\)".*/\1/' \
        | sort -u
)
need_calib=0
for calib in $calib_paths; do
    [[ -f "$calib" ]] || need_calib=1
done

if [[ "$need_calib" == "1" ]]; then
    echo ">>> collecting calibration via $CALIB_CONFIG on $DATASET (evalplus driver)"
    $PY -m quant_eval.cli.eval_evalplus \
        --model_name           "$MODEL" \
        --device_id            "$DEVICE" \
        --quant_config         "$CALIB_CONFIG" \
        --dataset              "$DATASET" \
        --batch_size           $BATCH_SIZE \
        --n_samples            $N_SAMPLES \
        --max_new_tokens       $MAX_NEW_TOKENS \
        --greedy               "$GREEDY" \
        --evalplus_output_dir  "$LOG_DIR/_calibrate/evalplus_out" \
        --log_dir              "$LOG_DIR/_calibrate" \
        2>&1 | tee "$LOG_DIR/calib.log"
fi

# Stage 2 — sweep selected TOMLs sequentially through evalplus.
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
