# Decode-Chip DSE — Disaggregated Serving on PLENA

Design-space exploration for the **decode** chip of a disaggregated PLENA
deployment. The prefill chip is a separate FP16 device; here only the decode
chip is quantised (MXINT/MXFP weights + KV, low-precision activation compute)
and its hardware is co-designed.

The search optimises the **actual analytic hardware metric** — end-to-end TPS
and TPOT on the PLENA decode-chip analytic model — rather than a memory proxy
(no `MB/token`). The software search scores every precision on a **fixed
reference chip** (the shipped design clamped legal for that precision, at the
baseline HBM channel count): throughput at the reference batch and
single-request TPOT at batch 1. It deliberately does NOT use the per-precision
co-design *ceiling* as the search signal — searching channels + batch per
precision lets the hardware buy back the precision difference, which collapses
the throughput axis to one value per stream-width class and starves the search
(observed in the 2026-07-09 llama8b run). The full co-design runs once per
front point in Stage B. Precision enters the model through HBM bytes/token
(the N-bit stream) and the MLEN bandwidth cap; compute is modelled
precision-neutral by default (the memory-bound assumption — see *Two-level
M/N* below); accuracy enters through the software eval.

## The disaggregated pipeline being modelled

1. The prompt runs on the **prefill chip in FP** (unquantised compute). It
   produces the first token; TTFT is the prefill chip's metric, not ours.
2. Its KV cache is **quantised once, at write time, into the decode chip's KV
   format** (`kv_cache_handoff="decode_format"` — quantise-on-write handoff).
3. Every subsequent token runs on the **decode chip**: quantised weights and KV
   streamed from HBM, activations computed at low precision inside the array
   and stored bf16 in on-chip SRAM (never in HBM).
4. Stage A searches the decode-side mixed precision (Optuna, three objectives:
   PPL ↓, reference-chip TPS ↑, reference-chip TPOT@batch1 ↓); Stage B
   co-designs the hardware per front precision on the analytic model.

The task evaluation reproduces exactly this: generation runs with **no phase
override** — the MASE decoder-layer hooks infer prefill vs decode from cache
semantics, so the prompt pass is FP with quantised KV writes and each generated
step is quantised. Only cache-free PPL scoring forces the decode phase (there
is no decode step for the hooks to see; it is the conservative proxy that sends
every scored token through the decode numerics, with `prefill_ppl` logged as
the FP reference).

## Two stages

```
decode_dse/
├── simulator_bridge.py         # the ONLY sys.path import of the analytic model
├── software/                   # Stage A — decode-quant accuracy
│   ├── decode_quant.py         #   prefill-FP / decode-quant pass_args + GPTQ cache key
│   ├── eval_decode.py          #   one precision -> PPL + GSM8K + IFEval
│   └── run_software_dse.py     #   Optuna precision search -> calibrated front -> CSV
├── hardware/                   # Stage B — hardware co-design
│   ├── codesign_search.py      #   Optuna (NSGA-II / MOTPE): max TPS, min TPOT, min area
│   └── end_to_end.py           #   with/without-optimisation table (accuracy x speed)
└── configs/                    # llama3_2_1b.json, llama3_8b.json, qwen3_32b.json
```

The stages meet at `software_disagg_decode.csv`: Stage B reads the accuracy
front and co-designs hardware per precision.

## Install

```bash
uv sync --extra dse                 # optuna + the analytic model's runtime deps
uv pip install -e /home/sr1325/mase --no-deps   # your phase-split MASE branch (editable)
export PLENA_SIMULATOR_PATH=/home/sr1325/PLENA_Simulator
```

The analytic model is imported in-process (never pip-installed) through
`simulator_bridge.py`; set `PLENA_SIMULATOR_PATH` if the checkout is elsewhere.

## Run

```bash
# Stage A — software accuracy (GPU). Writes results/decode_dse/<model>/software_disagg_decode.csv
python -m decode_dse.software.run_software_dse decode_dse/configs/llama3_8b.json

# Stage B — hardware co-design (CPU, analytic). Writes codesign_pareto.json
python -m decode_dse.hardware.codesign_search decode_dse/configs/llama3_8b.json --n-trials 256

# End-to-end table (with vs without optimisation)
python -m decode_dse.hardware.end_to_end decode_dse/configs/llama3_8b.json

# Publication-quality figures (writes results/decode_dse/<model>/plots/*.png)
python -m decode_dse.plots decode_dse/configs/llama3_8b.json
```

Or the whole pipeline: `bash decode_dse/run_all.sh decode_dse/configs/<model>.json`.

## Figures (`decode_dse/plots.py`)

One module renders every figure, in a colourblind-safe, print-ready style:

| File | Shows |
| --- | --- |
| `00_lengths_<task>` | ISL/OSL request-length distributions (p50/mean/p95, cap warnings) |
| `01_accuracy_vs_throughput` | PPL vs reference-chip throughput — the headline Pareto |
| `02_accuracy_vs_latency` | PPL vs reference-chip TPOT@batch1 |
| `03_task_vs_throughput` | GSM8K / IFEval accuracy vs throughput (calibrated front) |
| `04_throughput_latency` | per-precision co-design throughput–latency frontier |
| `05_right_sizing` | throughput vs matrix-array area (the knee) |
| `06_batch_sweep` | throughput & latency vs batch (two panels; the HBM-capacity wall) |
| `07_hbm_channels` | throughput vs HBM channel count |
| `08_roofline` | decode-step roofline — memory- vs compute-bound |
| `09_precision_heatmap` | weight × KV accuracy map (log colours; diverged cells flagged) |
| `10_with_without` | baseline vs quant-only vs co-design, iso-batch (from `end_to_end.json`) |
| `11_task_batch` | ideal serving batch per task at its profiled p50 lengths |
| `12_fp_setting` | vector-unit width ablation; the chosen FP_SETTING starred |

Smoke-test a single precision without the sweep:

```bash
python -m decode_dse.software.eval_decode \
    --model_name meta-llama/Llama-3.1-8B-Instruct --device cuda:0 \
    --attn_w 4 --ffn_w 4 --kv 4 --act_w 8 --use_gptq true \
    --tasks gsm8k --task_limit 20 --out /tmp/trial.json
```

## Design decisions

- **Prefill FP, decode quantised.** Every module carries only a `decode`
  bucket; the MASE config normaliser expands that to `prefill={"bypass": True}`,
  `decode_policy="quantized"`. KV handoff defaults to `decode_format` (prefill's
  KV writes are quantised once, at write time, into the decode chip's format).
- **Task evals run the real disagg semantics** (hooks infer the phase); only
  cache-free PPL forces `force_runtime_phase("decode")`.
  `install_phase_context_pre_hooks(model)` is re-run after any model rebuild
  (e.g. the lm-eval wrapper) so phase inference never silently stays inert.
- **Mixed-precision search.** The grid sweeps weights × KV × activation ×
  FP_SETTING (the vector-unit minifloat). Activations never touch HBM (stored
  bf16 on-chip) but their compute width is a MAC operand — see *Two-level M/N*.
  FP_SETTING covers SiLU/RMSNorm (SDPA-safe) and, with `fp_setting_attention`,
  softmax/rope (eager).
- **Two-level M/N precision.** N = the HBM stream widths (attnW/ffnW/KV): they
  set memory time, footprint, and the `MLEN·max(N) ≤ HBM_WIDTH` cap. M = the
  MAC operand width = max(attnW, ffnW, KV, act) — the attention GEMMs multiply
  KV by activations just as the FFN multiplies weights by activations. By
  default compute time is **precision-neutral** (`density_exp = 0`): the
  sanctioned memory-bound assumption ("assume you upcast back to the original
  compute — it doesn't even matter; the gain is effective bandwidth"). Setting
  `--density-exp 2.0` on the analytic model enables the iso-area density layer
  ((4/M)² more MACs at fixed silicon) — usable once a small-scale Synopsys-DC
  sweep calibrates the exponent by extrapolation.
- **Weight FORMAT is searched, not assumed.** ``search.weight_w`` mixes MXINT
  widths and MXFP tokens (``"E2M1"``). The weights-stay-MXINT rule was derived
  on the shared prefill+decode chip; the dedicated decode chip re-derives it —
  at equal element width the format does not change the HBM stream, so the axis
  is free on the hardware side.
- **RTN first, calibrate the front with the PLENA recipe.** RTN explores the
  grid cheaply; the accuracy/throughput front is re-run with the ablation's
  winning recipe — weights = GPTQ + **Erry** (output-norm) clipping,
  activations/KV = **selective rotation** (a greedy per-matmul search that only
  rotates layers that help — rotation on weights hurts microscaling, so the
  search leaves them alone). One cached bank per (weight width × block × calib)
  serves every KV/activation/FP_SETTING variant and every front point that
  shares it; task-aligned banks reuse the wikitext2 rotation decisions, so the
  hours-long greedy search runs once per weight config. FP_SETTING on the front
  is chosen as the cheapest vector-unit width within ``fp_ppl_tol`` (default 1%)
  of the bf16 unit's PPL — min-PPL would always pick bf16.
- **Overnight robustness.** OOM'd or killed worker trials retry (up to 3
  attempts) after waiting for VRAM headroom — the A6000s are shared and a
  co-tenant peak must not poison the front. Errored trial JSONs are re-run on
  resume; finished ones are reused.
- **Task-aligned calibration.** Perplexity calibrates on wikitext2; each
  downstream task calibrates on its own tokens (captured by a `TokenCollector`
  during a task pass, `decode_dse.software.build_task_calib`), matching the
  per-task ablations.
- **Sampler.** `"sampler": "nsga2"` (default) is the evolutionary NSGA-II
  multi-objective sampler; `"tpe"` selects Optuna's multi-objective TPE, a
  Bayesian alternative. Both honour the same feasibility constraints; all
  recorded runs used NSGA-II.
- **HBM is fixed technology.** A generation is chosen (`hbm_gen`); only the
  channel count is searched — bandwidth and capacity scale together. The
  generation data mirrors the transactional emulator's Ramulator Rust code
  (`transactional_emulator/lib/ramulator/src/raw.rs`): HBM = 128-bit channels,
  HBM2/HBM3 = 64-bit, burst 2; HBM2 is the only emulator-validated preset
  (2 Gbps, 8 ch/stack → 16 GB/s + 1 GB per channel). HBM2e is deliberately
  absent — the Rust code has no HBM2e preset to validate against.
- **The HBM-capacity constraint binds.** The co-design pins `n_chips = 1`, so a
  chip whose weights + KV exceed its searched channels' capacity is infeasible
  (auto-chips would silently add tensor-parallel chips and scale bandwidth for
  free). Baseline rows in the end-to-end table auto-resolve chips instead — an
  FP16 model that needs several stacks *is* the capacity wall — and report the
  count.
- **Compiler constraints are enforced in the search**: `MLEN % BLEN = 0`,
  `BLEN ≤ HLEN ≤ MLEN`, `MLEN % HLEN = 0`, **`VLEN = MLEN`** (matched
  matrix/vector tiles) and **`hidden % VLEN = 0`** (the embedding template's
  assert). The same ties are applied in the simulator's own `--search` /
  `--codesign` modes and `plena_settings.toml`.

## The with/without table (`hardware/end_to_end.py`)

Three regimes on the same workload, so the two gains are separable —
**baseline** (FP16 decode, shipped chip), **quant-only** (quantised decode,
shipped chip), **co-design** (quantised decode, searched chip). Fairness rules:

- every regime uses the same HBM generation; the baseline sits at a fixed
  channel count (`baseline_hbm_channels: 32` HBM2 = 512 GB/s) and the searched
  channel count is reported, so bought bandwidth is visible, never hidden;
- the shipped chip is clamped to a **legal** design point per precision
  (bandwidth cap, VLEN = MLEN) — the raw TOML geometry is infeasible at FP16;
- co-designed chips are additionally evaluated **at the baseline batch**
  (equal-batch columns), separating batch scaling from design gains.

## Deferred / open

- **Density-exponent calibration**: `--density-exp 2.0` turns on the iso-area
  M-bit compute-density layer; the exponent needs a small-scale Synopsys-DC
  sweep (extrapolated — DC results are predictable) before the deeper claim is
  publishable. All shipped results use the precision-neutral default.
- **Qwen3-32B single-GPU eval**: GPTQ calibration is layer-by-layer (fits any
  GPU), but the PPL/task forward needs the whole bf16 model resident (~64 GB) —
  run Stage A for 32B on an 80 GB device, or add an accelerate
  `dispatch_model` step after quantisation for multi-GPU eval.
- **Prefill/decode KV-format agreement**: this study quantises FP prefill KV
  directly into the decode format (single quantisation). If the prefill chip's
  own search picks a different KV format, the real handoff would double-quantise
  — agree one KV format across the two chips when the projects are joined.
