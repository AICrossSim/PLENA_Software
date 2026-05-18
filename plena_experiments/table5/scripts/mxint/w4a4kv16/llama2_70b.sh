#!/usr/bin/env bash
# Table 5 — Llama-2-70B  W/A/KV = 4/4/16 MXInt
# GPTQ + activation-aware y-clip + selective Hadamard rotation search.
# Usage: DEVICE=cuda:N bash plena_experiments/table5/scripts/mxint/w4a4kv16/llama2_70b.sh
MODEL="meta-llama/Llama-2-70b-hf"
QUANT_CONFIG="plena_experiments/table5/configs/mxint/w4a4kv16/llama2_70b.toml"
LOG_DIR_NAME="table5_llama2_70b_mxint_w4a4kv16"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$REPO_ROOT/plena_experiments/table5/scripts/_run.sh"
