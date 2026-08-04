# PackedKV decode-chip study

This package evaluates a dedicated PLENA decode chip for disaggregated
Qwen3-32B and Llama-3.1-8B serving. It keeps numerical accuracy, compiler
support, emulator support, RTL validation, timing calibration, and power
calibration as separate claims. The two pinned model configurations use the
same precision search and publication contracts; Llama-3.1-8B is the first
execution target, followed by Qwen3-32B.

The organising principle is that **no result inherits evidence from another**.
A precision profile that is accurate says nothing about whether the compiler can
emit it; a compiler that emits it says nothing about whether the RTL executes
it; and an emulator cycle count is not an RTL cycle count. Every number carries
the tier it was measured at, and a comparison between two tiers is refused
rather than silently made.

## Contents

- [Execution contract](#execution-contract) — how prefill, admission and decode divide
- [Canonical search](#canonical-search) — the 3,585-profile grid and what admits to hardware pricing
- [Workflow](#workflow) — the four protocols, launch checks, and module map
- [Validity scope](#validity-scope) — what evidence does and does not carry over
- [Publication outcomes](#publication-outcomes) — how results may be framed
- [Pipeline stages](#pipeline-stages) — what `sweep pipeline` runs, in order

To run the study on an execution host, follow `docs/server_setup.md`; this
document describes the design, not the bring-up.

## Execution contract

- Prefill runs in BF16 on a separate chip and produces the first token.
- The BF16 prompt cache crosses the chip boundary unchanged.
- Decode admission quantizes K/V once into the selected PackedKV format.
- New `q_len=1` K/V entries are quantized directly into that format.
- Admission time and energy belong to TTFT; cached decode contributes TPOT.
- Embeddings remain BF16. The headline path executes the BF16 LM head as a
  calibrated reserved endpoint on the same prefill chip after each decode-chip
  final RMSNorm; no third device is introduced.

The evaluator uses an immutable BF16 prefill artifact and a content-addressed
admission catalog. One KV format is deterministically recomputed and verified
at a time, then released before the next format, so the approximately 149 GiB
Qwen logical admission write volume is not retained simultaneously. Screening
and refinement report `post_handoff_greedy_conditioned_nll`: the
prefill-greedy handoff token is unscored, and later dataset tokens use cached
`q_len=1` teacher forcing with exact cache positions and one-entry growth. It
is not standard WikiText-2 perplexity. Cache-free perplexity is not part of
this path.

## Canonical search

The main block size is 8, the native RTL MX block. W, A, and shared K/V each
use:

```text
MXINT2 MXINT4 MXINT8 E1M2 E2M1 E3M4 E4M3 E5M2
```

Every W/A/KV point is evaluated with:

```text
FP_E3M2 FP_E2M3 FP_E6M5 FP_E5M6 FP_E4M7 FP_E8M5
```

The deterministic manifest contains:

- 3,072 quantized profiles;
- 512 vector-BF16 controls; and
- one split-execution BF16 software reference.

The 3,585 IDs are enumerated exactly. Optuna is not used for this numerical
grid. Blocks 16 and 32 are reserved for a selected-profile numerical
sensitivity study and are not deployment candidates for the native block-8
datapath. Split K/V precision is confined to promoted-profile refinement.

Accuracy is evaluated once for every precision profile and model, while the
hardware search evaluates the complete precision-by-hardware cross-product.
Final selection is joint under the accuracy, area, and HBM constraints; there
is no staged partial-objective decomposition of the canonical search.

Static datapath legality admits 574 rows to hardware pricing: 492 quantized
profiles and 82 vector-BF16 controls. The complete structural spaces contain
1,413,216 Qwen and 1,848,096 Llama candidates, giving 811,185,984 and
1,060,807,104 raw profile-candidate pairs. Exact compiler geometry admits
79,488 Qwen and 272,160 Llama candidates. The factorized artifact stores
476,928 and 1,632,960 physical-cost evaluation rows, respectively, plus the
ordered memberships required to reconstruct every eligible conceptual join;
it never materializes the raw cross-product. The scheduler compresses this
work only by exact physical-cost equivalence, prices each equivalence class
once, and joins the result back to every accuracy row. Its
provenance records the raw, hard-accuracy-gated, hard-resource-gated,
simulator-priced, and joined counts. It performs no performance, bandwidth
demand, memory-bound, or objective-based pruning.

## Workflow

Four protocols govern the study, each held separately from the code:

- the **exhaustive sweep** protocol defines immutable inputs, the measured pilot,
  state-isolation checks, runtime rebinding, homogeneous-device sharding, and
  launch gates;
- the **stack validation** protocol defines compiler, emulator, RTL, and DC
  evidence;
- the **precision refinement** protocol defines GPTQ/Erry, selective rotation,
  split K/V variants, and doomed-profile handling; and
- the **final benchmarks** protocol defines the four-configuration WikiText-2,
  IFEval, and GSM8K study, with the complete 4K/8K/16K/32K RULER set enabled
  only when long-context budget is available.

The full sweep must not start until its measured preflight report passes.
Structural capability alone cannot become a deployment claim.

Both model configurations require explicit single-device placement, BF16,
`sm_100`, minimum package versions, and declarative resource requirements.
Planning includes the full immutable manifest, run plan, provenance, logical
write volume, concurrency-bounded peak footprint, device requirement, and the
42-hour ceiling. Preflight reports only the machine on which it actually runs;
it does not substitute or project a remote host. It also does not invent a
wall-clock projection before the first completed measured profile; the runner
records that first duration and updates the ETA continuously.

The deterministic, write-free launch checks are:

```bash
python -m decode_dse.software.sweep stage plan \
  --config decode_dse/configs/llama3_1_8b.json \
  --output-dir /data/plena/llama3_1_8b \
  --device-label b200 --dry-run

python -m decode_dse.software.sweep stage plan \
  --config decode_dse/configs/qwen3_32b.json \
  --output-dir /data/plena/qwen3_32b \
  --device-label b200 --dry-run
```

Remove `--dry-run` only on a host that passes every reported package, model,
tokenizer, simulator, GPU, BF16, host-memory, and workspace-capacity gate.
Persisted (non-dry-run) planning additionally requires `--prompt-manifest`,
produced by `sweep inputs samples`. The complete execution-host bring-up —
repository layout, model/dataset staging, evidence-artifact production
(`decode_timing_evidence.json`, the stack-validity stage reports, and the
measured BF16 head service), and the full per-model launch order — is
documented in `decode_dse/docs/server_setup.md`.

The hardware evaluator optionally accepts `--handoff-artifact` for the E1
prefill-to-decode schedule analysis. That JSON must bind the pinned model and
reference workload; provide BF16 prefill latency and energy evidence for every
searched batch; and state admission, readiness, idle/stall power, direct-link,
and host-link inputs explicitly. A requested missing or mismatched artifact
fails before the study starts. Omitting it leaves steady-state decode ranking
unchanged; supplying it adds fully-pipelined, back-pressure, and host-buffered
TTFT, utilization, energy, and balanced prefill:decode chip-ratio results to
each hardware row without changing decode feasibility or Pareto ranking.

The principal modules are:

```text
profiles.py                          canonical precision profiles and the grid
legality.py                          profile legality and PackedKV capability
manifest.py                          exact manifest and restart journal
simulator_bridge.py                  analytic decode performance model
plots.py                             publication figures

software/precision_bindings.py       MASE phase-split quantization bindings
software/decode_evaluator.py         cached one-token decode evaluator
software/cached_decode.py            teacher-forced cached decode primitives
software/token_samples.py            sweep and refinement token sample bundles
software/cache_artifacts.py          BF16 prefill and admitted-cache artifacts
software/sweep_plan.py               run plan, manifests, shared executor context
software/sweep_runner.py             restartable per-profile sweep execution
software/sweep.py                    inputs / stage / shards / pipeline commands
software/preflight.py                check / evidence commands
software/refinement_schedule.py      refinement schedule and shard plans
software/refinement_runner.py        prepare / run / launch / merge commands
software/refinement_evaluator.py     sealed GPTQ, clipping, and rotation banks
software/benchmark_runner.py         manifest / configuration / benchmark commands
software/benchmark_evaluator.py      benchmark suites over split-cached decode
software/gpu_baseline.py             measured BF16 GPU cached one-token baseline
software/block_size_sensitivity.py   block 16/32 sensitivity for selected points
software/runtime_environment.py      deterministic runtime policy and identity

hardware/design_space.py             exact enumeration and factorized artifacts
hardware/evaluation.py               simulator and power evaluation
hardware/power_model.py              calibrated area and energy ingestion
hardware/selection.py                Pareto promotion and final selection
hardware/packedkv_claims.py          capacity-first PackedKV claim gate
hardware/synthesis_anchor.py         selected-candidate DC/SAIF evidence
hardware/admission_cost.py           decode-cache admission cost and evidence
hardware/workload_events.py          decode event counts for dense decoders
hardware/lm_head_service.py          BF16 output-head service boundary
hardware/calibration.py              holdout gates for area, power, timing
hardware/hbm_sensitivity.py          post-selection HBM technology schedule
hardware/statistics.py               deterministic statistical helpers
```

## Validity scope

Software accuracy is portable across hardware candidates. Successful compiler,
emulator, and RTL evidence applies only to the exact measured geometry and
batch. Fitted power is rankable only inside its measured interpolation
envelope. Deployment requires a candidate-exact full-chip DC/SAIF anchor whose
activity is bound to the serving batch. Off-target success becomes unknown
until matching evidence exists; observed failures remain false.

The PackedKV compiler emits a record for the exact synchronous equal-length
batch requested by the candidate. Compiler, emulator, RTL, and timing evidence
never carries over to a different batch, even when its geometry otherwise
matches.

## Publication outcomes

PackedKV is framed capacity first:

1. physical KV bytes decrease;
2. the feasible serving batch increases;
3. capacity-limited throughput increases.

A TPOT claim is optional and requires matching compiler, emulator, RTL, and
timing-overlap evidence. Reports keep the ideal-pipeline algorithmic limit
separate from the realized memory, serialization, or compute limit; neither is
rewritten to force a memory-bound result.

Power and timing are rankable only after their holdout gates pass. BF16 is the
software accuracy reference; it is not represented as a PLENA BF16 matrix-chip
realization.

The local BF16 LM head is an explicitly unrankable hardware sensitivity. A
locally quantized LM head is accuracy-only and cannot supply hardware costs or
enter deployment selection.

HBM technology does not multiply the 3,585-point numerical manifest. Four
selected native-datapath hardware profiles receive a separate 20-point
HBM2/HBM2E/HBM3/HBM3E/HBM4 sensitivity schedule with geometry, batch,
interface-unit count, and chip count fixed. HBM4 is a conservative 11 Gb/s
lower-bound representation of Micron's stated greater-than-11 Gb/s 36 GB
device and is technology-peak sensitivity only. Cross-generation ranking is
disabled until each generation, pin rate, and interface-unit count has matching
measured bandwidth calibration.

## Pipeline stages

`sweep pipeline` is the publication launch path for each model study, run after
that model's plan and preflight evidence pass. It executes in this order:

1. **Measured GPU baseline.** The immutable run plan lists every
   cached-`q_len=1` BF16 baseline batch; the pipeline measures those batches on
   its first visible GPU before any sweep work.
2. **Hardware-validation shards**, evaluated with compiler-trace timing.
3. **Joint source selection** over the verified partitions.
4. **Refinement**, following the declared protocol.
5. **Repricing** of the selected refined profiles.
6. **Publication figures**, rendered only once the above have completed.

### Restartability

Every command carries an immutable identity and a completion receipt. An
interrupted command is repeated; completed commands and complete directory trees
are checksum-verified and skipped. The contract, terminal rows, report and stage
receipt are all bound to the same sweep provenance.

### The final deployment table

It verifies and records the measured GPU report and stage receipt, the benchmark
contract and report, the exact refined hardware artifact, and the post-accuracy
selection. Evidence tiers appear side by side, and:

- the throughput ratio stays **empty** until a same-tier measured PLENA result
  exists;
- headline ratios accept measured evidence on both sides only;
- A100 and H100 peak-roofline values remain a separate labelled table and never
  acquire a ratio.

### Measured baseline energy

Each successful baseline row also attempts board energy, measured over only the
synchronized post-warmup decode region. The NVML cumulative-energy counter is
preferred; a timestamped NVML board-power trace with trapezoidal integration is
the fallback. The report retains the raw counters or trace samples, the physical
GPU UUIDs, the sampling interval, tokens per joule, and EDP.

If neither meter is supported, throughput remains valid and energy is reported
as explicitly unavailable. No analytic value may stand in for it in a measured
energy comparison.

### The `external/` evidence boundary

`publication_pipeline` declares every post-accuracy input, output and resource
choice. Three inputs live under the workspace `external/` boundary:

1. **Passing decode timing evidence.** Currently produced at the emulator tier
   (mode `emulator_serialized`), because RTL does not execute at the anchor
   geometry; the artifact carries `evidence_tier: emulator` and every consumer
   surfaces that tier. An RTL-tier artifact satisfies the same gate.
2. **The full-model independent-request compiler trace set.**
3. **The BF16 output-head service calibration.**

The third cannot be produced by timing `lm_head` on the execution GPU. The
headline boundary places a dedicated BF16 endpoint on the prefill chip, so the
artifact requires measurements taken at that physical endpoint: repeated and
holdout logits, remote-link request and response timing, component dynamic
energy, and leakage. The first pipeline command validates it against the pinned
model and every searched batch, and fails before any GPU sweep work if it is
missing or incomplete. No component energy or link measurement is ever inferred
from GPU timing.

### Refinement and benchmarks

Refinement is partitioned into exactly four logical source shards from one
immutable master schedule, run in two waves across the two-GPU execution pool.
The merge accepts only complete, checksum-verified terminal coverage, and every
accuracy-selected refinement profile is repriced against the exact hardware
space before publication configurations are sealed.

Before any GPU stage, the enabled publication path enumerates complete
WikiText-2, IFEval and GSM8K splits from their declared dataset ID,
configuration, immutable revision, split, expected row count and local cache
root. Datasets are read offline in two independent ways — Hugging Face offline
controls and direct prepared-Arrow reads — and the Hub-aware loader is never
called, so a missing snapshot fails rather than downloading or silently
switching revisions. Ordered item content and every local source file are
checksum-sealed into the benchmark manifest.

RULER is either omitted entirely or supplied as the complete 4K/8K/16K/32K set;
a partial set is not accepted. Contract construction, benchmark execution, the
post-accuracy exact hardware join, and the fail-closed final selection receipt
all remain inside the same restartable pipeline.
