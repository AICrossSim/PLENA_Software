#!/bin/bash
# Cheat-sheet for the humaneval ablation sweep.
#
# Step 0 (executable): collect humaneval calibration tokens once.
#   bash quant_eval/scripts/launch_humaneval_ablation.sh
#
# Step 1–4 (manual): copy each block below into its own terminal so the
# four ablation TOMLs run in parallel on different GPUs.

set -eu

CALIB_FILE=calib/Qwen_Qwen3-8B_humaneval_n32_s512.pt

# --------------------------------------------------------------------------
# Step 0 — Calibration (sequential, one GPU). Skips if file already exists.
# --------------------------------------------------------------------------

python3 -m quant_eval.cli.eval_evalplus \
    --model_name           Qwen/Qwen3-8B \
    --device_id            cuda:0 \
    --quant_config         quant_eval/configs/calibrate/qwen3_8b_humaneval.toml \
    --dataset              humaneval \
    --batch_size           1 \
    --n_samples            1 \
    --max_new_tokens       4096 \
    --greedy               true \
    --evalplus_output_dir  logs/humaneval_calibrate/evalplus_out \
    --log_dir              logs/humaneval_calibrate

# --------------------------------------------------------------------------
# Step 1–4 — Parallel ablation runs. Run each in its own terminal.
# --------------------------------------------------------------------------

cat <<'EOF'

============================================================
Calibration ready. Launch these 4 in separate terminals:
============================================================

# 02 W4 RTN (weight-only)
CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/02_w4_rtn.toml \
  CUDA_VISIBLE_DEVICES=0 bash quant_eval/scripts/run_ablation_humaneval.sh

# 03 W4 + ACT4 + KV4 RTN
CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/03_w4_act4_kv4_rtn.toml \
  CUDA_VISIBLE_DEVICES=1 bash quant_eval/scripts/run_ablation_humaneval.sh

# 04 + GPTQ
CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/04_w4_act4_kv4_gptq.toml \
  CUDA_VISIBLE_DEVICES=1 bash quant_eval/scripts/run_ablation_humaneval.sh

# 05 + GPTQ + Erry Clip
CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/05_w4_act4_kv4_gptq_erryclip.toml \
  CUDA_VISIBLE_DEVICES=3 bash quant_eval/scripts/run_ablation_humaneval.sh

# 06 + auto selective rotation search (greedy forward, humaneval-calib ppl).
# REUSES row-05's GPTQ checkpoint_dir (no re-quantize). The first run
# does the rotation sweep (~5-10 min ppl forwards) and writes
# rotation_decisions.json next to the GPTQ safetensors. Re-runs hit the
# cache and skip the search entirely. MUST run AFTER row 05 has finished
# so the GPTQ checkpoint exists.
CONFIGS=quant_eval/configs/ablation/humaneval_plena-qwen3-ablation/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=3 bash quant_eval/scripts/run_ablation_humaneval.sh

============================================================
EOF
