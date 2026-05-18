#!/usr/bin/env bash
# Table 5 — Llama-2-7B  W/A/KV = 4/4/4 MXInt
# GPTQ + activation-aware y-clip + selective Hadamard rotation search.
# Reuses the w4a16kv16 GPTQ checkpoint (same weight recipe); rotation cache
# redirected to llama2_7b_mxint_w4a4kv4/rotation_decisions.json.
# Usage: DEVICE=cuda:N bash plena_experiments/table5/scripts/mxint/w4a4kv4/llama2_7b.sh
MODEL="meta-llama/Llama-2-7b-hf"
QUANT_CONFIG="plena_experiments/table5/configs/mxint/w4a4kv4/llama2_7b.toml"
LOG_DIR_NAME="table5_llama2_7b_mxint_w4a4kv4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$REPO_ROOT/plena_experiments/table5/scripts/_run.sh"
