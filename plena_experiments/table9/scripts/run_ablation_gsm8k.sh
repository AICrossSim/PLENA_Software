#!/bin/bash
# Run a sweep of ablation TOMLs through eval_lm.py on gsm8k.
#
# CONFIGS is REQUIRED — pass a CSV of TOML paths. No glob, no basenames:
# you give the script the exact files to run.
#
# Stage 1 (auto): if any config uses `dataset = "file:..."` and the file
# isn't on disk, run $CALIB_CONFIG first to collect task-aligned tokens.
# Stage 2: run each config in order.
#
# Examples:
#   CONFIGS=quant_eval/configs/ablation/gsm8k_plena-qwen3-ablation/04_w4_act4_kv4_gptq.toml \
#       bash quant_eval/scripts/run_ablation_gsm8k.sh
#
#   CONFIGS=path/a.toml,path/b.toml bash quant_eval/scripts/run_ablation_gsm8k.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-8B}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-gsm8k}
LIMIT=${LIMIT:-1319}
SEQLEN=${SEQLEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-32}
CALIB_CONFIG=${CALIB_CONFIG:-quant_eval/configs/calibrate/qwen3_8b_gsm8k.toml}
CALIB_LIMIT=${CALIB_LIMIT:-200}
LOG_DIR=${LOG_DIR:-logs/ablation_$(date +%Y%m%d_%H%M%S)}

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

if [[ -z "${CONFIGS:-}" ]]; then
    echo "ERROR: CONFIGS env var is required (CSV of TOML paths)." >&2
    echo "  Example: CONFIGS=quant_eval/configs/ablation/gsm8k_plena-qwen3-ablation/04_w4_act4_kv4_gptq.toml \\" >&2
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
    echo ">>> collecting calibration via $CALIB_CONFIG (hook aborts when full)"
    $PY -m quant_eval.cli.eval_lm \
        --model_name   "$MODEL" \
        --device_id    "$DEVICE" \
        --tasks        "$TASKS" \
        --quant_config "$CALIB_CONFIG" \
        --limit        $CALIB_LIMIT \
        --log_dir      "$LOG_DIR/_calibrate" \
        2>&1 | tee "$LOG_DIR/calib.log"
fi

# Stage 2 — sweep selected TOMLs sequentially.
echo "Sweep:"
for t in "${selected_tomls[@]}"; do echo "  - $t"; done

for toml in "${selected_tomls[@]}"; do
    tag=$(basename "$toml" .toml)
    echo ">>> $tag"
    $PY -m quant_eval.cli.eval_lm \
        --model_name   "$MODEL" \
        --tasks        "$TASKS" \
        --device_id    "$DEVICE" \
        --quant_config "$toml" \
        --seqlen       $SEQLEN \
        --batch_size   $BATCH_SIZE \
        --limit        $LIMIT \
        --log_dir      "$LOG_DIR/$tag" \
        2>&1 | tee "$LOG_DIR/$tag.log" || echo "FAIL: $tag"
done

echo "Run dir: $LOG_DIR"
