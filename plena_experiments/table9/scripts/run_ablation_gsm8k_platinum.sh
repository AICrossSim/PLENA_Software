#!/bin/bash
# Run a sweep of ablation TOMLs through eval_lm.py on gsm8k_platinum.
#
# Plena-aligned variant: 04/05 use wikitext2 calibration (nsamples=128,
# seqlen=2048) instead of task-aligned tokens, so results are directly
# comparable to Plena's xwkv_gptq_*_int4 baselines.
#
# CONFIGS is REQUIRED — pass a CSV of TOML paths.
#
# Pre-flight: each selected TOML's `file:...` calib refs must already be
# on disk, and any TOML with [rotation_search] (row 06) must point at an
# existing `gptq.checkpoint_dir` (row 05). See plena_experiments/table9/README.md.
#
# Examples (run from repo root):
#   CONFIGS=plena_experiments/table9/configs/gsm8k_platinum/04_w4_act4_kv4_gptq.toml \
#       bash plena_experiments/table9/scripts/run_ablation_gsm8k_platinum.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-8B}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-gsm8k_platinum}
LIMIT=${LIMIT:-1209}
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
    echo "  Example: CONFIGS=plena_experiments/table9/configs/gsm8k_platinum/04_w4_act4_kv4_gptq.toml \\" >&2
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

# Pre-flight: required calib files / row-05 checkpoints must exist.
for toml in "${selected_tomls[@]}"; do
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        if [[ ! -f "$f" ]]; then
            echo "ERROR: $toml needs calib file $f, but it's missing." >&2
            echo "       Run the calibrate Step 0 from plena_experiments/table9/README.md first." >&2
            exit 1
        fi
    done < <(grep -hE '^[[:space:]]*(dataset|calib_data)[[:space:]]*=[[:space:]]*"file:' "$toml" \
              | sed 's/.*"file:\([^"]*\)".*/\1/' | sort -u)

    if grep -qE '^\[rotation_search\]' "$toml"; then
        ckpt=$(grep -E '^[[:space:]]*checkpoint_dir[[:space:]]*=' "$toml" \
               | head -1 | sed 's/.*"\([^"]*\)".*/\1/')
        if [[ -n "$ckpt" && ! -d "$ckpt" ]]; then
            echo "ERROR: $toml has [rotation_search] but checkpoint_dir $ckpt is missing." >&2
            echo "       Run row 05 (gptq+erryclip) first." >&2
            exit 1
        fi
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
