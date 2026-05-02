# Table 9 — Qwen3-8B Quantization Ablations

Three sweeps, each with 6 rows (01 FP16 → 06 +selrot). Run from repo root.

Within each sweep:

- **Row 01** (FP16) — fast, run anywhere first.
- **Rows 02–05** — can run in parallel on separate GPUs.
- **Row 06** — must run **after** row 05 (reuses 05's GPTQ checkpoint).
- Calibration (if any) — run **once first**; rows that share it would otherwise race.

---

## 1. gsm8k (task-aligned calib)

Rows 04/05/06 share `calib/Qwen_Qwen3-8B_gsm8k_n64_s1024.pt`. Pre-warm before parallel launch.

### Step 0 — calibrate (one GPU, skip if file exists)

```bash
python -m quant_eval.cli.eval_lm \
    --model_name   Qwen/Qwen3-8B \
    --device_id    cuda:2 \
    --tasks        gsm8k \
    --quant_config plena_experiments/table9/configs/calibrate/qwen3_8b_gsm8k.toml \
    --limit        200 \
    --log_dir      logs/gsm8k_calibrate
```

### Step 1 — row 01 (FP16 baseline)

```bash
CONFIGS=plena_experiments/table9/configs/gsm8k/01_fp16.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh
```

### Step 2 — rows 02–05 (4 separate terminals, parallel)

```bash
CONFIGS=plena_experiments/table9/configs/gsm8k/02_w4_rtn.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh

CONFIGS=plena_experiments/table9/configs/gsm8k/03_w4_act4_kv4_rtn.toml \
  CUDA_VISIBLE_DEVICES=1 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh

CONFIGS=plena_experiments/table9/configs/gsm8k/04_w4_act4_kv4_gptq.toml \
  CUDA_VISIBLE_DEVICES=2 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh

CONFIGS=plena_experiments/table9/configs/gsm8k/05_w4_act4_kv4_gptq_erryclip.toml \
  CUDA_VISIBLE_DEVICES=3 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh
```

### Step 3 — row 06 (after row 05 finishes)

```bash
CONFIGS=plena_experiments/table9/configs/gsm8k/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=3 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh
```

---

## 2. humaneval (task-aligned calib via evalplus)

Rows 04/05/06 share `calib/Qwen_Qwen3-8B_humaneval_n32_s512.pt`. Pre-warm before parallel launch.

### Step 0 — calibrate (one GPU, skip if file exists)

```bash
python -m quant_eval.cli.eval_evalplus \
    --model_name           Qwen/Qwen3-8B \
    --device_id            cuda:0 \
    --quant_config         plena_experiments/table9/configs/calibrate/qwen3_8b_humaneval.toml \
    --dataset              humaneval \
    --batch_size           8 \
    --n_samples            1 \
    --max_new_tokens       4096 \
    --greedy               true \
    --evalplus_output_dir  logs/humaneval_calibrate/evalplus_out \
    --log_dir              logs/humaneval_calibrate
```

### Step 1 — row 01 (FP16 baseline)

```bash
CONFIGS=plena_experiments/table9/configs/humaneval/01_fp16.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table9/scripts/run_ablation_humaneval.sh
```

### Step 2 — rows 02–05 (4 separate terminals, parallel)

```bash
CONFIGS=plena_experiments/table9/configs/humaneval/02_w4_rtn.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table9/scripts/run_ablation_humaneval.sh

CONFIGS=plena_experiments/table9/configs/humaneval/03_w4_act4_kv4_rtn.toml \
  CUDA_VISIBLE_DEVICES=1 bash plena_experiments/table9/scripts/run_ablation_humaneval.sh

CONFIGS=plena_experiments/table9/configs/humaneval/04_w4_act4_kv4_gptq.toml \
  CUDA_VISIBLE_DEVICES=2 bash plena_experiments/table9/scripts/run_ablation_humaneval.sh

CONFIGS=plena_experiments/table9/configs/humaneval/05_w4_act4_kv4_gptq_erryclip.toml \
  CUDA_VISIBLE_DEVICES=3 bash plena_experiments/table9/scripts/run_ablation_humaneval.sh
```

### Step 3 — row 06 (after row 05 finishes)

```bash
CONFIGS=plena_experiments/table9/configs/humaneval/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=3 bash plena_experiments/table9/scripts/run_ablation_humaneval.sh
```
