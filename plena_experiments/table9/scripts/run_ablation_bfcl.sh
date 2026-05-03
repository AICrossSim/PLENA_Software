#!/bin/bash
# Start plain_quant_serve for a TOML from the table9 sweep so bfcl can be
# pointed at it. Single TOML, foreground; you drive bfcl yourself.
#
# Example:
#   CONFIG=plena_experiments/table9/configs/gsm8k/04_w4_act4_kv4_gptq.toml \
#     bash plena_experiments/table9/scripts/run_ablation_bfcl.sh

set -eu

MODEL=${MODEL:-Qwen/Qwen3-32B}
DEVICE=${DEVICE:-cuda:0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8915}
ATTN=${ATTN:-sdpa}    # safe on Qwen wrappers because qk/av/softmax are bypassed in
                      # the table9/10 gemm7 TOMLs; KV-cache MX still applies.

DEFAULT_PY=$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python)
PY=${PY:-$DEFAULT_PY}

if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG env var is required (path to a TOML)." >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: $CONFIG does not exist" >&2
    exit 1
fi

exec "$PY" -m quant_eval.cli.plain_quant_serve \
    --model_name          "$MODEL" \
    --device_id           "$DEVICE" \
    --quant_config        "$CONFIG" \
    --host                "$HOST" \
    --port                "$PORT" \
    --attn_implementation "$ATTN"
