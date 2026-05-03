#!/bin/bash
# Run a sweep of ablation TOMLs through eval_lm.py on gsm8k.
#
# CONFIGS is REQUIRED — pass a CSV of TOML paths. No glob, no basenames:
# you give the script the exact files to run.
#
# Examples (run from repo root):
#   CONFIGS=plena_experiments/table10/configs/gsm8k/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
#       bash plena_experiments/table10/scripts/run_ablation_gsm8k.sh
#
#   CONFIGS=path/a.toml,path/b.toml bash plena_experiments/table10/scripts/run_ablation_gsm8k.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-32B}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-gsm8k}
LIMIT=${LIMIT:-1319}
SEQLEN=${SEQLEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_FEWSHOT=${NUM_FEWSHOT:-4}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
FEWSHOT_AS_MULTITURN=${FEWSHOT_AS_MULTITURN:-true}
GEN_KWARGS=${GEN_KWARGS:-max_gen_toks=2048}
LOG_DIR=${LOG_DIR:-logs/ablation_${TASKS}_$(date +%Y%m%d_%H%M%S)}

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

if [[ -z "${CONFIGS:-}" ]]; then
    echo "ERROR: CONFIGS env var is required (CSV of TOML paths)." >&2
    echo "  Example: CONFIGS=plena_experiments/table10/configs/gsm8k/06_w4_act4_kv4_gptq_erryclip_selrot.toml \\" >&2
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
    $PY -m quant_eval.cli.eval_lm \
        --model_name           "$MODEL" \
        --tasks                "$TASKS" \
        --device_id            "$DEVICE" \
        --quant_config         "$toml" \
        --seqlen               $SEQLEN \
        --batch_size           $BATCH_SIZE \
        --limit                $LIMIT \
        --log_dir              "$LOG_DIR/$tag" \
        --num_fewshot          $NUM_FEWSHOT \
        --apply_chat_template  $APPLY_CHAT_TEMPLATE \
        --fewshot_as_multiturn $FEWSHOT_AS_MULTITURN \
        --gen_kwargs           "$GEN_KWARGS" \
        2>&1 | tee "$LOG_DIR/$tag.log" || echo "FAIL: $tag"
done

echo "Run dir: $LOG_DIR"
