#!/usr/bin/env bash

# Qwen3-32B Decode Precision DSE Pipeline
# Pipeline: Optuna (TPE) -> Pareto Front -> GPTQ Rescue -> IFEval Verification
# 
# Run detached & monitor:
#   nohup bash decode_dse_qwen/run_all.sh > decode_dse_qwen/results/run.out 2>&1 &

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# HF caches on /data (home has a 100 GB quota a 65 GB model blows past); allocator + sampling hygiene.
export HF_HUB_CACHE=/data/models
export HF_DATASETS_CACHE=/data/models/datasets_cache
export HF_HOME=/data/models/.hf_home
export NLTK_DATA=/data/models/.nltk_data
export MPLCONFIGDIR=/data/models/.mplconfig
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$NLTK_DATA" "$MPLCONFIGDIR" decode_dse_qwen/results

.venv/bin/python -m decode_dse_qwen.search \
  --n-trials 300 \
  --decode-gpus 2 \
  --chunks 48 --chunk-len 512 \
  --refine-gptq 5 \
  --ifeval-subset 64 --ifeval-topk 4 --ifeval-budget 32768 \
  --ppl-budget 30 \
  --storage "sqlite:///$(pwd)/decode_dse_qwen/results/study.db"

echo "[run_all] DONE -> decode_dse_qwen/results/"
echo "  trials.csv  pareto.json  frontier_ifeval.csv  pareto.png  accuracy_vs_length.png"
