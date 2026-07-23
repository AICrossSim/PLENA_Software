#!/usr/bin/env bash
# Decode-chip DSE pipeline — one config, end to end.
#
#   nohup bash decode_dse/run_all.sh decode_dse/configs/llama3_8b.json \
#       > logs/llama8b_$(date +%Y%m%d).log 2>&1 &
#
#   PREFLIGHT_ONLY=1 bash decode_dse/run_all.sh <config>   # health-check, CPU, seconds
#
# Resumable: trials / calibration / GPTQ banks / rotation decisions all cache,
# so re-running the same command skips finished work. Caches recorded under
# older eval semantics are purged automatically (see EVAL_SEMANTICS).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-decode_dse/configs/llama3_8b.json}"
PY="${PY:-.venv/bin/python}"
export PLENA_SIMULATOR_PATH="${PLENA_SIMULATOR_PATH:-/home/sr1325/PLENA_Simulator}"
# Offline HF: gated Llama/Qwen repos 401 online even when cached. Llama lives in
# the default hub; Qwen3-32B is complete in /data/models (set HF_HUB_CACHE for it).
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf_hub/khl22/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-/tmp}"                          # ~16 GB GPTQ banks off the home quota
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1                                # live nohup logs

stage() { echo "==== [$(date '+%F %T')] $* ===="; }

# --- [0/6] preflight: fail fast, CPU-only, before the GPU wait ---------------
stage "[0/6] preflight — $CONFIG"
[ -x "$PY" ] || { echo "ERROR: $PY not found — run 'uv sync --extra dse' first." >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: config $CONFIG not found." >&2; exit 1; }

MODEL=$("$PY" -c "import json;print(json.load(open('$CONFIG'))['model_name'])")
TASKS=$("$PY" -c "import json;print(','.join(json.load(open('$CONFIG')).get('tasks',['gsm8k','ifeval'])))")
OUT_DIR="results/decode_dse/${MODEL##*/}"
IFS=',' read -ra TS <<< "$TASKS"

CONFIG="$CONFIG" MODEL="$MODEL" "$PY" - <<'PYEOF'
import json, os, sys
cfg = json.load(open(os.environ["CONFIG"]))
model, trc = os.environ["MODEL"], bool(cfg.get("trust_remote_code"))

from decode_dse.simulator_bridge import DecodeSimulator            # analytic model reachable
sim = DecodeSimulator(cfg["sim_model"], model_lib=cfg.get("model_lib"))
print(f"  simulator OK: {cfg['sim_model']} (hidden={sim.dims['hidden']}, layers={sim.dims['layers']})")

from chop.passes.module.transforms.quantize.quantize import install_phase_context_pre_hooks  # noqa: F401
print("  mase (chop) OK: phase-split quantize importable")

from transformers import AutoConfig, AutoTokenizer                 # offline hub complete?
try:
    AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=trc)
    AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=trc)
except Exception as e:
    hub = os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")
    sys.exit(f"ERROR: {model} not fully cached in {hub} ({type(e).__name__}).\n"
             f"  Fetch once with a token: HF_HUB_OFFLINE=0 HF_HUB_CACHE={hub} .venv/bin/python -c "
             f"\"from huggingface_hub import snapshot_download; snapshot_download('{model}')\"")
print(f"  offline hub cache OK: {model} (config + tokenizer)")

import lm_eval  # noqa: F401
print(f"  lm-eval OK (tasks: {cfg.get('tasks', ['gsm8k', 'ifeval'])})")
PYEOF

# --- eval-semantics versioning: purge caches recorded under old semantics ----
#   v2 (2026-07-09): task evals generate under true disagg (prefill FP).
#   v3 (2026-07-10): mixed-format activations (a-E* on MXINT-weight linears)
#       were silently unquantised before the mase key-dispatch fix.
EVAL_SEMANTICS="v3-mixedfmt-act"
SEM_FILE="$OUT_DIR/.eval_semantics"
if [ "$(cat "$SEM_FILE" 2>/dev/null || true)" != "$EVAL_SEMANTICS" ]; then
    if [ -d "$OUT_DIR/trials" ]; then
        for t in "${TS[@]}"; do
            find "$OUT_DIR/trials" -name "*__${t}.json" -print -delete 2>/dev/null \
                | sed 's/^/  purged stale task cache: /' || true
        done
        find "$OUT_DIR/trials" -name "*__a-E*.json" -print -delete 2>/dev/null \
            | sed 's/^/  purged stale mixed-format-act cache: /' || true
    fi
    mkdir -p "$OUT_DIR" && echo "$EVAL_SEMANTICS" > "$SEM_FILE"
    echo "  eval-semantics marker set: $EVAL_SEMANTICS"
fi

[ "${PREFLIGHT_ONLY:-0}" = "1" ] && { stage "preflight PASSED — safe to launch overnight"; exit 0; }

# --- GPU wait: shared A6000s — grab the first GPU with headroom --------------
# Override MIN_FREE_MB / POLL_SEC, or set CUDA_VISIBLE_DEVICES to pin and skip.
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
POLL_SEC="${POLL_SEC:-120}"
WAIT_TIMEOUT_MIN="${WAIT_TIMEOUT_MIN:-0}"                # 0 = wait forever
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[$(date '+%F %T')] waiting for a GPU with >= ${MIN_FREE_MB} MiB free..."
    waited=0
    while :; do
        read -r gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -n | tail -1 | tr -d ' ' | tr ',' ' ')
        if [ "${free:-0}" -ge "$MIN_FREE_MB" ]; then
            export CUDA_VISIBLE_DEVICES="$gpu"
            echo "[$(date '+%F %T')] selected GPU $gpu (${free} MiB free)"
            break
        fi
        if [ "$WAIT_TIMEOUT_MIN" -gt 0 ] && [ "$waited" -ge $((WAIT_TIMEOUT_MIN * 60)) ]; then
            echo "ERROR: no GPU reached ${MIN_FREE_MB} MiB in ${WAIT_TIMEOUT_MIN} min." >&2; exit 1
        fi
        echo "[$(date '+%F %T')] freest GPU $gpu has ${free} MiB; waiting ${POLL_SEC}s..."
        sleep "$POLL_SEC"; waited=$((waited + POLL_SEC))
    done
fi

stage "decode DSE — $MODEL — $CONFIG"

stage "[1/6] prefetching datasets (online, best-effort)"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TASKS="$TASKS" "$PY" - <<'PYEOF' || echo "(prefetch skipped — offline)"
import os, datasets
from lm_eval.tasks import TaskManager, get_task_dict
try:
    datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
except Exception as e:
    print("  prefetch wikitext2 failed:", type(e).__name__)
tm = TaskManager()
for t in filter(None, os.environ["TASKS"].split(",")):
    try:
        get_task_dict([t], tm); print(f"  {t}: dataset ready")
    except Exception as e:
        print(f"  {t}: prefetch failed ({type(e).__name__}) — will rely on cache")
PYEOF

stage "[2/6] profiling task lengths (ISL/OSL — inputs to fig 00 + fig 11)"
if [ -f "$OUT_DIR/task_lengths.json" ]; then
    echo "(already profiled — skipping)"
else
    "$PY" -m decode_dse.software.profile_lengths --model_name "$MODEL" --tasks "$TASKS" --limit 200 \
        || echo "(length profiling skipped — non-fatal)"
fi

stage "[3/6] building task-aligned calibration"
for t in "${TS[@]}"; do
    "$PY" -m decode_dse.software.build_task_calib --model_name "$MODEL" --task "$t" \
        || echo "WARNING: calib build failed for '$t' — its task-aligned eval will be skipped."
done

stage "[4/6] Stage A: precision search + calibrated front (the long part)"
# Relaunch on abnormal exit (e.g. a co-tenant OOM kill of the orchestrator):
# trials, GPTQ banks, rotation decisions and per-task JSONs all cache, so a
# relaunch fast-forwards to where the previous attempt died.
STAGE_A_ATTEMPTS="${STAGE_A_ATTEMPTS:-3}"
for attempt in $(seq 1 "$STAGE_A_ATTEMPTS"); do
  if "$PY" -m decode_dse.software.run_software_dse "$CONFIG"; then
    break
  fi
  echo "[run_all] Stage A attempt ${attempt}/${STAGE_A_ATTEMPTS} exited abnormally; relaunching (caches resume)"
  [ "$attempt" -eq "$STAGE_A_ATTEMPTS" ] && { echo "[run_all] Stage A failed after ${STAGE_A_ATTEMPTS} attempts"; exit 1; }
  sleep 60
done

stage "[5/6] Stage B: hardware co-design + end-to-end table (CPU, analytic)"
"$PY" -m decode_dse.hardware.end_to_end "$CONFIG"

stage "[6/6] rendering figures"
"$PY" -m decode_dse.plots "$CONFIG"

stage "DONE — results under $OUT_DIR/"
