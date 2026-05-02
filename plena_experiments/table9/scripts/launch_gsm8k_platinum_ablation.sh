#!/bin/bash
# Cheat-sheet for the gsm8k_platinum ablation sweep (Plena-aligned).
#
# No calibration step needed — 04/05 use wikitext2 calibration (loaded
# from HF on demand), unlike the gsm8k variant which uses file: tokens.
#
# Usage:
#   bash quant_eval/scripts/launch_gsm8k_platinum_ablation.sh
# prints the 4 parallel commands; copy each into its own terminal.

cat <<'EOF'

============================================================
gsm8k_platinum ablation (Plena-aligned, wikitext2 calib).
Launch these 4 in separate terminals:
============================================================

# 02 W4 RTN (weight-only)
CONFIGS=quant_eval/configs/ablation/gsm8k_platinum_plena-qwen3-ablation/02_w4_rtn.toml \
  CUDA_VISIBLE_DEVICES=0 bash quant_eval/scripts/run_ablation_gsm8k_platinum.sh

# 03 W4 + ACT4 + KV4 RTN
CONFIGS=quant_eval/configs/ablation/gsm8k_platinum_plena-qwen3-ablation/03_w4_act4_kv4_rtn.toml \
  CUDA_VISIBLE_DEVICES=1 bash quant_eval/scripts/run_ablation_gsm8k_platinum.sh

# 04 + GPTQ
CONFIGS=quant_eval/configs/ablation/gsm8k_platinum_plena-qwen3-ablation/04_w4_act4_kv4_gptq.toml \
  CUDA_VISIBLE_DEVICES=2 bash quant_eval/scripts/run_ablation_gsm8k_platinum.sh

# 05 + GPTQ + Erry Clip
CONFIGS=quant_eval/configs/ablation/gsm8k_platinum_plena-qwen3-ablation/05_w4_act4_kv4_gptq_erryclip.toml \
  CUDA_VISIBLE_DEVICES=3 bash quant_eval/scripts/run_ablation_gsm8k_platinum.sh

# 06 + auto selective rotation search (greedy forward, gsm8k-calib ppl).
# REUSES row-05's GPTQ checkpoint_dir (no re-quantize). The first run
# does the rotation sweep (~10 min ppl forwards) and writes
# rotation_decisions.json next to the GPTQ safetensors. Re-runs hit the
# cache and skip the search entirely. MUST run AFTER row 05 has finished
# so the GPTQ checkpoint exists.
CONFIGS=quant_eval/configs/ablation/gsm8k_platinum_plena-qwen3-ablation/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=3 bash quant_eval/scripts/run_ablation_gsm8k_platinum.sh

============================================================
EOF
