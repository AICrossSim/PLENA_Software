# Table 10 — Qwen3-32B Quantization Ablations

Two sweeps (gsm8k + humaneval), each with 2 rows: **01** (FP16 baseline) and
**06** (W4/Act4/KV4 GPTQ + erryclip + selective rotation). Run from repo root.

Row 06 is end-to-end self-contained: its `[gptq]` + `[rotation_search]`
blocks make the chop quantize pass run GPTQ first, then rotation search on
top. First run is slow (GPTQ + ppl evals); subsequent runs hit the GPTQ
checkpoint and the cached `rotation_decisions.json` and finish in seconds.

Note: Q@K^T and attn@V are bypassed (`qk_matmul.bypass = true`,
`av_matmul.bypass = true`) and excluded from the rotation search — only
the 7 GEMMs (q/k/v/o + gate/up/down) plus the KV cache are quantized,
hence the `gemm7` parent dir on every `checkpoint_dir`.

---

## 1. gsm8k (task-aligned calib)

Row 06 uses `calib/Qwen_Qwen3-32B_gsm8k_n64_s1024.pt` for both GPTQ and
the rotation-search calib loader.

### Step 0 — calibrate (one GPU, skip if file exists)

```bash
python -m quant_eval.cli.eval_lm \
    --model_name   Qwen/Qwen3-32B \
    --device_id    cuda:2 \
    --tasks        gsm8k \
    --quant_config plena_experiments/table10/configs/calibrate/qwen3_32b_gsm8k.toml \
    --limit        200 \
    --log_dir      logs/gsm8k_calibrate_32b
```

### Step 1 — row 01 (FP16 baseline)

```bash
CONFIGS=plena_experiments/table10/configs/gsm8k/01_fp16.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table10/scripts/run_ablation_gsm8k.sh
```

### Step 2 — row 06 (GPTQ + erryclip + selrot, end-to-end)

```bash
CONFIGS=plena_experiments/table10/configs/gsm8k/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=0 bash plena_experiments/table10/scripts/run_ablation_gsm8k.sh
```

---

## 2. humaneval (task-aligned calib via evalplus)

Row 06 uses `calib/Qwen_Qwen3-32B_humaneval_n32_s512.pt` for both GPTQ and
the rotation-search calib loader.

### Step 0 — calibrate (one GPU, skip if file exists)

```bash
python -m quant_eval.cli.eval_evalplus \
    --model_name           Qwen/Qwen3-32B \
    --device_id            cuda:0 \
    --quant_config         plena_experiments/table10/configs/calibrate/qwen3_32b_humaneval.toml \
    --dataset              humaneval \
    --batch_size           8 \
    --n_samples            1 \
    --max_new_tokens       4096 \
    --greedy               true \
    --evalplus_output_dir  logs/humaneval_calibrate_32b/evalplus_out \
    --log_dir              logs/humaneval_calibrate_32b
```

### Step 1 — row 01 (FP16 baseline)

```bash
CONFIGS=plena_experiments/table10/configs/humaneval/01_fp16.toml \
  CUDA_VISIBLE_DEVICES=5 bash plena_experiments/table10/scripts/run_ablation_humaneval.sh
```

### Step 2 — row 06 (GPTQ + erryclip + selrot, end-to-end)

```bash
CONFIGS=plena_experiments/table10/configs/humaneval/06_w4_act4_kv4_gptq_erryclip_selrot.toml \
  CUDA_VISIBLE_DEVICES=5 bash plena_experiments/table10/scripts/run_ablation_humaneval.sh
```
