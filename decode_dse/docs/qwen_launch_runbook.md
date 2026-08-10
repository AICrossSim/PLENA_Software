# Qwen3-32B decode study — launch runbook

This runbook takes the prepared `sr1325-decode_dse-qwen` /
`sr1325-dev-qwen` worktrees to a sealed Qwen3-32B launch without touching
the live Llama workspace. Nothing here writes into
`/data`- or `/home`-level Llama paths; the Qwen study uses a fresh output
directory and its own worktree code.

## Environment

```bash
SOFTWARE=/home/sr1325/PLENA_Software-qwen
SIMULATOR=/home/sr1325/PLENA_Simulator-qwen
VENV=/home/sr1325/PLENA_Software/.venv/bin/python
export PLENA_SIMULATOR_PATH="$SIMULATOR"
export PYTHONPATH="$SOFTWARE:$SIMULATOR:$SIMULATOR/compiler"
cd "$SOFTWARE"
```

## Gate 0 — contract tests (CPU, safe any time)

```bash
$VENV -m pytest decode_dse/ -q
```

All suites must be green, including the declared-subspace, dual-frontier,
energy-context, and HBM-PHY area tests.

## Gate 1 — gap decomposition (CPU, safe any time)

```bash
$VENV -m decode_dse.scripts.gap_decomposition \
  --config decode_dse/configs/qwen3_32b.json \
  --output /home/sr1325/plena/qwen3_32b_prep/gap_decomposition.json \
  --timing-evidence /home/sr1325/plena/llama3_1_8b/external/decode_timing_evidence.json \
  --gpu-baseline-report /home/sr1325/plena/llama3_1_8b/gpu_baseline/report.json \
  --stride 16
```

Review the emitted flag before sealing anything: if
`bandwidth_first_framing_supported` is false at the candidate points, the
bandwidth-first framing of the study must be revisited. On the current
space the narrowest point is fully memory-bound at 8-bit but 83 percent
serialization-bound at 4-bit — bandwidth is a necessary lever, not a
sufficient explanation, and figures should say so.

## Gate 2 — plan projection (needs a quiet GPU window)

Wait for the Llama pipeline to release its GPUs, then:

```bash
decode_dse/scripts/launch_pipeline.sh stage plan \
  --config decode_dse/configs/qwen3_32b.json \
  --output-dir /home/sr1325/plena/qwen3_32b \
  --device-label b200
```

Read `max_projected_hours` from the emitted `run_plan.json`. The launch
gate is a projection of at most ~30 hours against the 36-hour ceiling
(the Llama run's re-seal overhead is the buffer). If over budget, the
levers in order are: `publication_pipeline.resources.stride` to 2, then
`eval_ppl_nsamples` down, then `BLEN` drop 4. Never touch the declared
precision space to save time — its exclusions are accuracy-evidence-based
only.

## Gate 3 — study pricing smoke (CPU, safe any time)

Price ~100 factor blocks against the dry-run manifest with
`--parallel-workers 8`, then extrapolate: 333,504 physical signature pairs
total at `study_parallel_workers: 48`. Confirm one emitted row carries
`metrics.whole_model.calibrated_energy.{total_j,energy_tier,energy_id}`
with `energy_tier: analytic_anchored`, an `area_mm2` that includes the
`HBMPhys` block, and the declared timing tier.

## Launch

```bash
decode_dse/scripts/launch_pipeline.sh pipeline \
  --config decode_dse/configs/qwen3_32b.json \
  --output-dir /home/sr1325/plena/qwen3_32b \
  --device-label b200 --gpus <free-gpu-list>
```

`publication_enabled` stays false for this cycle: the pipeline stops after
repricing and figures. The figure stage always receives the measured GPU
baseline, so `energy_context.json`, the `06_energy_efficiency` figure, and
the dual-accuracy envelopes render without the benchmark stages.

## After the calibration re-sweep lands

The receipted DMA request sweep with the 16-channel plane regenerates
`analytic_models/disagg_serve/calibration_dma_requests.{csv,receipt.json,validation.json}`
in the simulator worktree. When it lands after the study has launched, the
receipts machinery supports a re-price pass in a fresh workspace with the
upgraded calibration identity; published numbers should use the receipted
calibration either way. The refreshed 32-channel matrix-prefetch tail
statistics are reported, not hidden.

The aggregate effective-bandwidth tables still carry the
`aggregate_csv_without_raw_run_receipts` grade; rerunning their harness
with per-point receipts (same capture list as the request sweep) closes
the audit's remaining provenance gap.

## After the run

- `decode_dse/scripts/area_crosscheck.py` on the frontier candidates, with
  `--reference-root /home/sr1325/PLENA_Simulator-area-ref`; quote the
  spread as the area disclosure band (±8.9 percent at the reference
  geometries).
- Re-price the Llama study in a fresh workspace with this config lineage:
  the measured GPU baseline and the numerical screen are reusable; only
  the analytic pricing, selection, and figures repeat.
