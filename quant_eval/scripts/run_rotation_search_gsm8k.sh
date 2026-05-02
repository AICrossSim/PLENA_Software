#!/bin/bash
# Run the calibration-aware per-matmul rotation search using the row-05
# (gptq + erryclip) ablation config as the non-rotated baseline.
#
# Stage 1 (auto): if the gsm8k calib token file is missing, collect it via
# the calibrate TOML — same dance as run_ablation_gsm8k.sh.
# Stage 2: invoke quant_eval.cli.search_rotation with 05 as the base config.
#         The search runs GPTQ (resumes from checkpoint_dir if populated),
#         then sweeps each of the 9 matmul types with calib-set perplexity.
#
# Usage:
#   bash quant_eval/scripts/run_rotation_search_gsm8k.sh
#   CUDA_VISIBLE_DEVICES=0 bash quant_eval/scripts/run_rotation_search_gsm8k.sh
#
# Override anything via env vars:
#   MODEL=Qwen/Qwen3-0.6B BASE_CONFIG=path/to/other.toml \
#     bash quant_eval/scripts/run_rotation_search_gsm8k.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-8B}
DEVICE=${DEVICE:-cuda:0}
BASE_CONFIG=${BASE_CONFIG:-quant_eval/configs/ablation/gsm8k_plena-qwen3-ablation/05_w4_act4_kv4_gptq_erryclip.toml}
CALIB_DATA=${CALIB_DATA:-file:calib/Qwen_Qwen3-8B_gsm8k_n64_s1024.pt}
CALIB_NSAMPLES=${CALIB_NSAMPLES:-32}
CALIB_SEQLEN=${CALIB_SEQLEN:-1024}
IMPROVEMENT_EPS=${IMPROVEMENT_EPS:-0.0}
MATMUL_TYPES=${MATMUL_TYPES:-}            # empty = all 9 (10 with kv_cache)
CALIB_CONFIG=${CALIB_CONFIG:-quant_eval/configs/calibrate/qwen3_8b_gsm8k.toml}
CALIB_LIMIT=${CALIB_LIMIT:-200}
LOG_DIR=${LOG_DIR:-logs/rotation_search_$(date +%Y%m%d_%H%M%S)}

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: BASE_CONFIG=$BASE_CONFIG does not exist" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Model       : $MODEL"
echo "Base config : $BASE_CONFIG"
echo "Calib data  : $CALIB_DATA"
echo "Calib n/seq : $CALIB_NSAMPLES x $CALIB_SEQLEN"
echo "Log dir     : $LOG_DIR"

# Stage 1 — collect calibration if the file isn't on disk.
calib_path="${CALIB_DATA#file:}"
if [[ "$CALIB_DATA" == file:* && ! -f "$calib_path" ]]; then
    echo ">>> calib file $calib_path missing — collecting via $CALIB_CONFIG"
    $PY -m quant_eval.cli.eval_lm \
        --model_name   "$MODEL" \
        --device_id    "$DEVICE" \
        --tasks        gsm8k \
        --quant_config "$CALIB_CONFIG" \
        --limit        $CALIB_LIMIT \
        --log_dir      "$LOG_DIR/_calibrate" \
        2>&1 | tee "$LOG_DIR/calib.log"
fi

OUTPUT_JSON="$LOG_DIR/rotation_search_results.json"

# Stage 2 — run the rotation search.
extra_args=()
if [[ -n "$MATMUL_TYPES" ]]; then
    extra_args+=(--matmul_types "$MATMUL_TYPES")
fi

echo ">>> rotation search (greedy forward selection)"
$PY -m quant_eval.cli.search_rotation \
    --model_name        "$MODEL" \
    --base_config       "$BASE_CONFIG" \
    --calib_data        "$CALIB_DATA" \
    --device_id         "$DEVICE" \
    --calib_nsamples    $CALIB_NSAMPLES \
    --calib_seqlen      $CALIB_SEQLEN \
    --improvement_eps   $IMPROVEMENT_EPS \
    --output_json       "$OUTPUT_JSON" \
    --log_dir           "$LOG_DIR" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_DIR/search.log"

echo
echo "Done. Results JSON: $OUTPUT_JSON"
echo "Log dir          : $LOG_DIR"
