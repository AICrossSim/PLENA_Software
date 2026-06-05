#!/bin/bash
# Benchmarking script for Fast-dLLM models using the PLENA eval_dllm harness.
# Streams output to the terminal; no per-run log files are retained.
# Only the tabulated results (all_results.jsonl) persist.

export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:/home/gm1425/Fast-dLLM/PLENA_Software/.uv_cache/git-v0/checkouts/c53f323a805b777e/cb963b8/src/

LOG_DIR="logs/benchmarks"
mkdir -p $LOG_DIR

MODELS=(
    "Efficient-Large-Model/Fast_dLLM_v2_7B"
)

BLOCK_LENGTHS=(8 16 32)
BATCH_SIZES=(16 32)

CONFIGS=(
    "plena_experiments/table9/configs/gsm8k/01_fp16.toml"
    "plena_experiments/table9/configs/gsm8k/02_w4_rtn.toml"
    "plena_experiments/table9/configs/gsm8k/03_w4_act4_kv4_rtn.toml"
    "plena_experiments/table9/configs/gsm8k/04_w4_act4_kv4_gptq.toml"
    "plena_experiments/table9/configs/gsm8k/05_w4_act4_kv4_gptq_erryclip.toml"
    "plena_experiments/table9/configs/gsm8k/06_w4_act4_kv4_gptq_erryclip_selrot.toml"
)

RESULTS_FILE="$LOG_DIR/all_results.jsonl"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "[]" > "$RESULTS_FILE"
fi

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

for MODEL in "${MODELS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        for BL in "${BLOCK_LENGTHS[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
                MODEL_NAME=$(basename "$MODEL")
                CONFIG_NAME=$(basename "$CONFIG" .toml)
                RUN_ID="${MODEL_NAME}_${CONFIG_NAME}_BL${BL}_BS${BS}"

                if grep -q "\"run_id\": \"$RUN_ID\"" "$RESULTS_FILE"; then
                    echo "Skipping $RUN_ID (already exists)..."
                    continue
                fi

                echo "Benchmarking: $MODEL_NAME config=$CONFIG_NAME BL=$BL BS=$BS..."

                # Build args for PLENA eval_dllm jsonargparse CLI.
                # No --log_dir: skip persistent per-run args.json/results.json.
                ARGS=(
                    --model_name "$MODEL"
                    --tasks gsm8k
                    --device_id cuda:0
                    --batch_size "$BS"
                    --bd_size "$BL"
                    --small_block_size 8
                    --max_new_tokens 512
                    --threshold 1.0
                    --temperature 0.0
                    --num_fewshot 0
                    --show_speed true
                )

                # Baseline fp16 runs: no quant_config; quantized runs: pass config
                if [[ "$CONFIG_NAME" != *fp16* ]]; then
                    ARGS+=(--quant_config "$CONFIG")
                fi

                # Stream to terminal while capturing to a temp file for metric
                # extraction; the temp file is deleted once results are tabulated.
                TMP_LOG=$(mktemp)
                CUDA_VISIBLE_DEVICES=3 $PY -m quant_eval.cli.eval_dllm "${ARGS[@]}" \
                    2>&1 | tee "$TMP_LOG"

                # gsm8k reports two exact_match scores: "strict-match" (needs a
                # literal "#### <n>", which our \boxed{} prompt never produces ->
                # always ~0) and "flexible-extract" (last number in the output ->
                # the real score). Prefer flexible-extract; fall back to the last
                # exact_match line for tasks (e.g. minerva_math) with no filters.
                ACCURACY=$(grep -oP 'exact_match,flexible-extract: \K[0-9.]+' "$TMP_LOG" | head -n 1)
                [ -z "$ACCURACY" ] && ACCURACY=$(grep -oP 'exact_match[^,]*?: \K[0-9.]+' "$TMP_LOG" | tail -n 1)
                TOKENS_PER_SEC=$(grep -oP 'Tokens/s: \K[0-9.]+' "$TMP_LOG" | tail -n 1)
                TTFT_MS=$(grep -oP 'TTFT: \K[0-9.]+' "$TMP_LOG" | tail -n 1)
                TOTAL_TIME=$(grep -oP 'Total workload time: \K[0-9.]+' "$TMP_LOG" | tail -n 1)

                rm -f "$TMP_LOG"

                # Treat a run as failed (crash/OOM) if no speed line was emitted.
                # Don't record failed runs, so the skip-check retries them later.
                if [ -z "$TOKENS_PER_SEC" ]; then
                    echo "FAILED $RUN_ID — no result (crash/incomplete); will retry on rerun"
                    continue
                fi

                [ -z "$ACCURACY" ] && ACCURACY=0
                [ -z "$TTFT_MS" ] && TTFT_MS=0
                [ -z "$TOTAL_TIME" ] && TOTAL_TIME=0

                echo "{\"run_id\": \"$RUN_ID\", \"model\": \"$MODEL_NAME\", \"config\": \"$CONFIG_NAME\", \"block_length\": $BL, \"batch_size\": $BS, \"accuracy\": $ACCURACY, \"tokens_per_sec\": $TOKENS_PER_SEC, \"ttft_ms\": $TTFT_MS, \"total_time\": $TOTAL_TIME}" >> "$RESULTS_FILE"

                echo "Done $RUN_ID — Acc: $ACCURACY  Tps: $TOKENS_PER_SEC  TTFT: ${TTFT_MS}ms"
            done
        done
    done
done
