#!/bin/bash
# Memory profiling for Fast-dLLM models using the PLENA profile_memory CLI.
#
# For each quant config the model is loaded ONCE, then a single generation
# batch is run for every (block_length, batch_size) cell to record:
#   * model_footprint_mb — resident model weights (CUDA allocated after load)
#   * peak_memory_mb     — peak CUDA allocation while generating one batch
#
# Results are merged into the SAME all_results.jsonl produced by
# benchmark_all.sh, matched by run_id (latency/accuracy fields are preserved;
# the memory fields are added). Cells with no existing run_id are appended.

export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:/home/gm1425/Fast-dLLM/PLENA_Software/.uv_cache/git-v0/checkouts/c53f323a805b777e/cb963b8/src/

LOG_DIR="logs/benchmarks"
mkdir -p $LOG_DIR

MODELS=(
    "Efficient-Large-Model/Fast_dLLM_v2_7B"
)

BLOCK_LENGTHS=(8 16 32 64)
BATCH_SIZES=(1 4 8 16 32)

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

# JSON list literals for the profiler CLI (jsonargparse).
BL_JSON=$(printf '%s,' "${BLOCK_LENGTHS[@]}"); BL_JSON="[${BL_JSON%,}]"
BS_JSON=$(printf '%s,' "${BATCH_SIZES[@]}");   BS_JSON="[${BS_JSON%,}]"

for MODEL in "${MODELS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        MODEL_NAME=$(basename "$MODEL")
        CONFIG_NAME=$(basename "$CONFIG" .toml)

        echo "Profiling memory: $MODEL_NAME config=$CONFIG_NAME (single batch per cell)..."

        ARGS=(
            --model_name "$MODEL"
            --device_id cuda:0
            --bd_sizes "$BL_JSON"
            --batch_sizes "$BS_JSON"
            --small_block_size 8
            --max_new_tokens 512
            --threshold 1.0
            --temperature 0.0
        )
        if [[ "$CONFIG_NAME" != *fp16* ]]; then
            ARGS+=(--quant_config "$CONFIG")
        fi

        TMP_LOG=$(mktemp)
        CUDA_VISIBLE_DEVICES=3 $PY -m quant_eval.cli.profile_memory "${ARGS[@]}" \
            2>&1 | tee "$TMP_LOG"

        if ! grep -q "MEM_RESULT" "$TMP_LOG"; then
            echo "FAILED $MODEL_NAME/$CONFIG_NAME — no MEM_RESULT lines (crash/incomplete)"
            rm -f "$TMP_LOG"
            continue
        fi

        # Merge each MEM_RESULT cell into all_results.jsonl by run_id.
        "$PY" - "$RESULTS_FILE" "$TMP_LOG" "$MODEL_NAME" "$CONFIG_NAME" <<'PYEOF'
import json, re, sys

results_path, log_path, model_name, config_name = sys.argv[1:5]

# Parse memory cells from the profiler log.
cells = {}  # run_id -> (footprint_mb, peak_mb, peak_reserved_mb, bl, bs)
pat = re.compile(
    r"MEM_RESULT\|bd=(\d+)\|bs=(\d+)\|cache=(true|false)\|footprint_mb=([\d.]+)"
    r"\|peak_mb=([\d.]+)\|peak_reserved_mb=([\d.]+)"
)
with open(log_path) as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        bl, bs, cache = int(m.group(1)), int(m.group(2)), m.group(3)
        # run_id must match benchmark_all.sh exactly: ..._BL{bl}_BS{bs}_CACHE{cache}
        run_id = f"{model_name}_{config_name}_BL{bl}_BS{bs}_CACHE{cache}"
        cells[run_id] = {
            "block_length": bl,
            "batch_size": bs,
            "use_block_cache": cache == "true",
            "model_footprint_mb": round(float(m.group(4)), 2),
            "peak_memory_mb": round(float(m.group(5)), 2),
            "peak_reserved_mb": round(float(m.group(6)), 2),
        }

# Read existing lines, updating matching records in place.
out_lines = []
seen = set()
with open(results_path) as f:
    for raw in f:
        s = raw.rstrip("\n")
        if not s.strip():
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            out_lines.append(s)
            continue
        if isinstance(obj, dict) and obj.get("run_id") in cells:
            rid = obj["run_id"]
            obj.update(
                model_footprint_mb=cells[rid]["model_footprint_mb"],
                peak_memory_mb=cells[rid]["peak_memory_mb"],
                peak_reserved_mb=cells[rid]["peak_reserved_mb"],
            )
            seen.add(rid)
            out_lines.append(json.dumps(obj))
        else:
            out_lines.append(s)

# Append cells that had no existing latency/accuracy record.
for rid, c in cells.items():
    if rid in seen:
        continue
    out_lines.append(json.dumps({
        "run_id": rid,
        "model": model_name,
        "config": config_name,
        "block_length": c["block_length"],
        "batch_size": c["batch_size"],
        "use_block_cache": c["use_block_cache"],
        "model_footprint_mb": c["model_footprint_mb"],
        "peak_memory_mb": c["peak_memory_mb"],
        "peak_reserved_mb": c["peak_reserved_mb"],
    }))

with open(results_path, "w") as f:
    f.write("\n".join(out_lines) + "\n")

print(f"Merged {len(cells)} memory cells "
      f"({len(seen)} updated, {len(cells) - len(seen)} appended) for {config_name}.")
PYEOF

        rm -f "$TMP_LOG"
        echo "Done memory profiling $MODEL_NAME/$CONFIG_NAME."
    done
done
