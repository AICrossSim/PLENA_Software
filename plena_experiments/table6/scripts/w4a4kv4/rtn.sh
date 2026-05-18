#!/usr/bin/env bash
# Table 6 — Llama-3-8B  MXINT Full System — RTN (no GPTQ, no rotation)
# Usage: DEVICE=cuda:N bash plena_experiments/table6/scripts/w4a4kv4/rtn.sh
MODEL="meta-llama/Meta-Llama-3-8B"
QUANT_CONFIG="plena_experiments/table6/configs/w4a4kv4/rtn.toml"
LOG_DIR_NAME="table6_w4a4kv4_rtn"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO_ROOT/plena_experiments/table6/scripts/_run.sh"
