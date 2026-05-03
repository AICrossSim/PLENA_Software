# Table 9 — Qwen3 Quantization Ablations

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
  CUDA_VISIBLE_DEVICES=2 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh

CONFIGS=plena_experiments/table9/configs/gsm8k/04_w4_act4_kv4_gptq.toml \
  CUDA_VISIBLE_DEVICES=7 bash plena_experiments/table9/scripts/run_ablation_gsm8k.sh

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

---

## 3. bfcl (Qwen3-32B, web-search categories)

Different model size — Qwen3-32B. Rows 04/05/06 share
`calib/Qwen_Qwen3-32B_bfcl_n32_s2048.pt`. Pre-warm before parallel launch.

Unlike sweeps 1/2, this one doesn't bundle the eval into one Python process:
the script just spins up `plain_quant_serve` (an OpenAI-compatible HTTP
server) for one TOML at a time. You drive `bfcl generate` / `bfcl evaluate`
from a separate terminal against the served port.

### Step 0 — calibrate (manual; skip if file exists)

`plain_quant_serve` does not attach a TokenCollector — produce the calib
file out-of-band, e.g. via `quant_eval.eval.collect_calib` with a JSONL of
BFCL prompts. Target output: `calib/Qwen_Qwen3-32B_bfcl_n32_s2048.pt`
(see `configs/calibrate/qwen3_32b_bfcl.toml` for the expected
`target_nsamples` / `seqlen`).

### Step 1 — row 01 (FP16 baseline)

```bash
CONFIG=plena_experiments/table9/configs/bfcl/01_fp16.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table9/scripts/run_ablation_bfcl.sh
```

In another terminal, point bfcl at the local server:

```bash
LOCAL_SERVER_ENDPOINT=127.0.0.1 LOCAL_SERVER_PORT=8915 \
  bfcl generate --model Qwen/Qwen3-32B-FC \
    --test-category web_search_base --skip-server-setup \
    --result-dir logs/bfcl/01_fp16/results
bfcl evaluate --model Qwen/Qwen3-32B-FC \
    --test-category web_search_base web_search_no_snippet \
    --result-dir logs/bfcl/01_fp16/results \
    --score-dir  logs/bfcl/01_fp16/scores
```

### Step 2 — rows 02–05 (4 separate terminals, parallel)

Pick a different `PORT` per parallel server when sharing a host so the bfcl
clients can target each independently.

```bash
CONFIG=plena_experiments/table9/configs/bfcl/02_w4_rtn.toml \
  CUDA_VISIBLE_DEVICES=0 PORT=8915 bash plena_experiments/table9/scripts/run_ablation_bfcl.sh

CONFIG=plena_experiments/table9/configs/bfcl/03_w4_act4_kv4_rtn.toml \
  CUDA_VISIBLE_DEVICES=1 PORT=8916 bash plena_experiments/table9/scripts/run_ablation_bfcl.sh

CONFIG=plena_experiments/table9/configs/bfcl/04_w4_act4_kv4_gptq.toml \
  CUDA_VISIBLE_DEVICES=2 PORT=8917 bash plena_experiments/table9/scripts/run_ablation_bfcl.sh

CONFIG=plena_experiments/table9/configs/bfcl/05_w4_act4_kv4_gptq_erryclip.toml \
  CUDA_VISIBLE_DEVICES=7 PORT=8905 bash plena_experiments/table9/scripts/run_ablation_bfcl.sh
```

### Step 3 — row 06 (after row 05 finishes)

```bash
CONFIG=plena_experiments/table9/configs/bfcl/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=3 bash plena_experiments/table9/scripts/run_ablation_bfcl.sh
```
