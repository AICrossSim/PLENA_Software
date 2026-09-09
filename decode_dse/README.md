# PackedKV decode-chip study

This package evaluates a dedicated PLENA decode chip for disaggregated
Qwen3-32B and Llama-3.1-8B serving. It keeps numerical accuracy, compiler
support, emulator support, RTL validation, timing calibration, and power
calibration as separate claims. The two pinned model configurations share the
same publication contracts; each declares its own precision search space
(Llama-3.1-8B enumerates the canonical space, Qwen3-32B a declared subspace
with disclosed exclusions). Llama-3.1-8B is the first execution target,
followed by Qwen3-32B. Every physical number comes from the sibling
`PLENA_Simulator` checkout's analytic models — resolved via
`PLENA_SIMULATOR_PATH` or the adjacent `../PLENA_Simulator` directory — so
that checkout is a hard dependency of every pricing stage.

The organising principle is that **no result inherits evidence from another**.
A precision profile that is accurate says nothing about whether the compiler can
emit it; a compiler that emits it says nothing about whether the RTL executes
it; and an emulator cycle count is not an RTL cycle count. Every number carries
the tier it was measured at, and a comparison between two tiers is refused
rather than silently made.

## Contents

- [Execution contract](#execution-contract) — how prefill, admission and decode divide
- [Declared search space](#declared-search-space) — the per-study precision grid and what admits to hardware pricing
- [Declared hardware space](#declared-hardware-space) — the structural grid, the knobs held closed, and the runtime reserve
- [Workflow](#workflow) — the four protocols, launch checks, and module map
- [Validity scope](#validity-scope) — what evidence does and does not carry over
- [Publication outcomes](#publication-outcomes) — how results may be framed
- [Pipeline stages](#pipeline-stages) — what `sweep pipeline` runs, in order
- [Operator notes](#operator-notes) — the simulator dependency, admission persistence, and re-sealing

To run the study on an execution host, follow `docs/server_setup.md`; this
document describes the design, not the bring-up. `docs/evidence_tiers.md` is the
single reference for every tier string this package emits.

## Execution contract

- Prefill runs in BF16 on a separate chip and produces the first token.
- The BF16 prompt cache crosses the chip boundary unchanged.
- Decode admission quantizes K/V once into the selected PackedKV format.
- New `q_len=1` K/V entries are quantized directly into that format.
- Admission time and energy belong to TTFT; cached decode contributes TPOT.
- Embeddings remain BF16. Cached `q_len=1` decode uses an untied, decode-local
  MX LM head whose W/A formats follow the exact profile; its matrix numeric and
  storage format is the profile vector format, followed by a BF16 logit
  container with no precision recovery.
- The serving path tile-streams logits through running top-k20/top-p0.95 state
  (plus a deterministic lowest-token-id argmax diagnostic), rather than
  reserving batch-by-vocabulary SRAM.
- A decode-local BF16 head remains available as an analytic sensitivity. Its
  BF16 weights and projection traffic are charged to decode HBM, but its
  compute has no measured instruction-level timing or energy signature. Every
  such row carries `local_bf16_head_compute_idealized` and is forced to
  `whole_model.rankable=false`.

The evaluator uses an immutable BF16 prefill artifact and a content-addressed
admission catalog under the `content_addressed_recompute_per_format` persistence
policy. One KV format is deterministically recomputed and verified at a time,
then released before the next, so the large logical admission write volume the
run plan projects is never retained simultaneously. Under this policy the
persisted total is exactly zero and the logical rebuild cost is reported
separately through the resource projection; the run plan carries the projected
figure for the model actually being run. See
[Admission persistence policies](#admission-persistence-policies). Screening
and refinement report `post_handoff_greedy_conditioned_nll`: the
prefill-greedy handoff token is unscored, and later dataset tokens use cached
`q_len=1` teacher forcing with exact cache positions and one-entry growth. It
is not standard WikiText-2 perplexity. Cache-free perplexity is not part of
this path.

## Declared search space

The main block size is 8, the native RTL MX block. The canonical format
tuple for W, A, and shared K/V is:

```text
MXINT2 MXINT4 MXINT8 E1M2 E2M1 E3M4 E4M3 E5M2
```

Every W/A/KV point is evaluated with:

```text
FP_E3M2 FP_E2M3 FP_E6M5 FP_E5M6 FP_E4M7 FP_E8M5
```

Each study declares its precision space in `search`: every axis must be a
canonical-order subsequence of the canonical tuple, every excluded format
needs a disclosed rationale under `search.declared_exclusions`, and the
sealed expected counts must equal the declared space's cross product. Within
the declared space the enumeration is exhaustive and unpruned. The Llama
study declares the full canonical space (3,072 quantized profiles, 512
vector-BF16 controls, one split-execution BF16 reference — 3,585 IDs). The
Qwen study excludes MXINT2 and E2M1 from the weight and activation axes on
measured prior evidence from the Llama screen (each exceeded the relaxed
1.05x accuracy budget with margin) and enumerates 1,728 quantized profiles,
288 controls, and the reference — 2,017 IDs.

Optuna is not used for this numerical grid. Blocks 16 and 32 are reserved
for a selected-profile numerical sensitivity study and are not deployment
candidates for the native block-8 datapath. Split K/V precision is confined
to promoted-profile refinement.

Accuracy is evaluated once for every precision profile and model, while the
hardware search evaluates the complete precision-by-hardware cross-product.
Final selection is joint under the accuracy, area, and HBM constraints; there
is no staged partial-objective decomposition of the canonical search.

Static datapath legality admits the hardware-pricing rows per declared space:
574 of the 3,585 Llama IDs, 336 of the 2,017 Qwen IDs. Every one of those rows
prices, including the ones whose RTL selector path is unimplemented — see
[RTL capability is recorded, never
blocking](#rtl-capability-is-recorded-never-blocking). Every priced row is
admitted on the validated pricing model and discloses whether it was
additionally compiled and emulated at its own geometry — see [Admission is
model validation, disclosed per
row](#admission-is-model-validation-disclosed-per-row).

The factorized artifact stores the
physical-cost evaluation rows plus the ordered memberships required to
reconstruct every eligible conceptual join; it never materializes the raw
cross-product. The scheduler compresses this work only by exact
physical-cost equivalence, prices each equivalence class once, and joins the
result back to every accuracy row. Its provenance records the raw,
hard-resource-gated, simulator-priced, and joined counts. Within the
declared space it performs no performance, bandwidth demand, memory-bound,
or objective-based pruning; accuracy is priced for every profile and stamped
as an `accuracy_within_limit` label, never used as a filter.

## Declared hardware space

`hardware_space` in each study config declares the structural grid;
`ExactHardwareSpace.iter_candidates` enumerates it exhaustively in canonical
lexical order, dropping only geometrically impossible points (MLEN divisible by
BLEN and HLEN, `BLEN <= HLEN <= MLEN`, hidden size divisible by MLEN and TP,
`TP * KVP == CHIP_COUNT`, enough link ports for the declared parallelism).

The Qwen grid is additionally narrowed to compiler-legal geometry — head_dim
fixes HLEN at 128, and the grouped-query head-broadcast bound requires MLEN of
at least 1,024 with TP and KVP at most 8 — so its geometry census records zero
compiler rejections. Compiler-legality pruning removes impossible programs,
never unpromising ones.

### Knobs held closed for want of timing evidence

`KV_HEAD_REUSE` and `DRAIN_OVERLAPPED` are declared as `[false]` for both
models. This is an **evidence-availability restriction, not objective pruning**,
and it is disclosed as such:

- the drain-overlapped timing mode has no anchor in the timing evidence; and
- the packed-`q_len=1` timing contracts have not been generated.

A candidate using either knob could therefore only ever be recorded as
`timing_uncalibrated` — enumerated and priced structurally, but not rankable —
so the grid does not spend enumeration on it. Opening both knobs takes the Llama structural
census from 251,328 to 616,032 candidates and the Qwen census from 22,320 to
55,584. Nothing about those candidates has been shown to be worse; they have
only not been shown at all. Generating the missing timing contracts is the
whole of what would be required to reopen them.

Note that the knobs are not free even structurally: when enabled,
`analytic_models/performance/disagg_decode.py::system_area` charges a KV
head-reuse control block and a drain-overlap accumulator bank, and both carry
the `declared_structural_estimate` tier.

### HBM channels and the chip-side PHY

Channels are searchable at 8, 16, and 32 interface units at HBM2 — the channel
planes with measured aggregate calibration. The Qwen study declares all three;
the Llama study declares 8. `validate_calibrated_hardware_space` fails closed on
any generation/channel pair without a rankable measured calibration, so an
uncalibrated headline operating point cannot be enumerated by accident.

Attached bandwidth is not free. Each interface unit is charged
0.6875 mm² of chip-side HBM PHY, I/O and beachfront silicon on every chip
(`PLENA_Simulator/analytic_models/area/hbm_phy.py`, threaded through
`estimate_system_area` → `disagg_serve/area.py` →
`performance/disagg_decode.py::system_area`, surfacing as the `HBMPhys`
breakdown entry). That block's evidence tier is `declared_structural_estimate`:
its die-edge occupancy is published, its beachfront depth is declared. So
channel count trades against silicon area in the Pareto ranking, at a disclosed
tier weaker than the calibrated compute-and-SRAM census it is added to.

### Runtime HBM reserve

`publication_pipeline.resources.runtime_hbm_reserve_bytes` is an allowance for
working memory outside the weight, KV and activation ledger. Both configs
declare 512 MiB.

Three properties are worth stating because they are easy to assume otherwise:

- **The reserve is per chip on every topology path.** The explicit `TP × KVP`
  ledger scales it by chip count internally, and the legacy aggregate ledger
  multiplies it by chip count so the two topologies agree about the feasibility
  of the identical physical system.
- **It is clamped to a structural fraction of per-chip capacity.**
  `_effective_runtime_hbm_reserve_bytes` takes the minimum of the configured
  value and per-chip HBM capacity divided by
  `RUNTIME_HBM_RESERVE_CAPACITY_DIVISOR` (8). A reserve chosen against a large
  memory can therefore never exceed — let alone dominate — a smaller one, and a
  reserve larger than the chip cannot render every geometry infeasible
  regardless of precision, sharding or chip count.
- **Preflight and pricing use the identical clamped value**, and that value is
  part of the resource-ledger cache key. The cheap resource preflight admits a
  candidate if and only if the full simulator evaluation finds it
  capacity-feasible; the two cannot disagree, and a reserve change cannot be
  served from a stale ledger.

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
run plan's `max_projected_hours` ceiling (168 h unless the plan narrows it),
against which the launch gate checks the projection. Any tighter operational
budget belongs in the model's runbook, not here.
Preflight reports only the machine on which it actually runs;
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
optional measured BF16 head-service sensitivity), and the full per-model
launch order — is
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
hardware/packedkv_claims.py          legacy remote-BF16 PackedKV ablation gate
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

### RTL capability is recorded, never blocking

`legality.py` partitions the five capability stages into two disjoint sets:

```text
PRICING_BLOCKING_STAGES  = ("software", "compiler", "emulator")
PRICING_RECORDED_STAGES  = ("rtl", "dc")
```

The split follows from what the timing tiers actually claim.
`TIMING_TIER_REQUIRED_VALIDITY` admits both declared publication tiers on
compiler and emulator validity alone, so a structural limitation in one of those
stages — or in the software implementation feeding them — means nothing
downstream could be evaluated at all, and the point cannot be priced. A
limitation confined to the RTL implementation or the DC flow is orthogonal to
what those tiers claim.

`CrossStackCapability` therefore exposes `blocking_issues`, `recorded_issues`
(an exhaustive partition of `issues`), and `prices_at_publication_tier`, and
`hardware/evaluation.py` carries all three onto the row:
`packedkv_selector_blocking_issue_codes` decides whether a price is withheld,
`packedkv_selector_recorded_issue_codes` is disclosed alongside it, and
`packedkv_selector_issue_codes` remains the complete record. A recorded issue
still forces `rtl_valid=False` through `CrossStackCapability.validity_floor`; it
simply does not delete the row.

The PackedKV limitations that land in the recorded set today are:

| Code | What is unimplemented |
| --- | --- |
| `rtl_batched_mxfp_unsupported` | the packed batched-attention selector exists only on the MXINT matrix path |
| `rtl_mxint_scale_segment_mismatch` | the MXINT MCU requires BLEN to divide the MX block size |
| `rtl_mxint_activation_requant_unvalidated` | the RTL FP-to-MXINT activation path is validated for MXINT4 and MXINT8 only |

**This is the single largest change to what the study reports.** Under the
previous all-or-nothing rule, any RTL limitation withheld the analytic price, so
of the 574 hardware-legal Llama profiles only 84 priced; the remaining 490 (448
MXFP profiles and 42 MXINT profiles with an unvalidated activation format at the
default target) were absent from the design space rather than disclosed within
it. All 574 now price. The Qwen space moves the same way, 84 of 336 to 336 of
336.

Stated plainly: **every published MXFP number rests on compiler and emulator
evidence with an explicitly unimplemented RTL selector path.** It is an analytic
price at a tier that never claimed RTL execution, carrying the reason the RTL
selector cannot execute it. It is not an RTL result, and `rtl_valid` is false on
every one of those rows.

### Admission is model validation, disclosed per row

A row is admitted into the reported results when it was **priced by a
validated, identified pricing model** — calibrated timing bound to a
timing-evidence identity at a declared timing tier, calibrated memory timing
bound to a bandwidth calibration identity, an energy tier plus the identity of
the coefficient set behind it, an identified area model, a priced output-head
boundary, a layout and composed system identity, and demonstrated capacity,
runtime and resource-budget feasibility — with a succeeded numerical run, measured
software-stage validity, and no structural capability limitation on a blocking
stage. `evaluate_publication_admission` in `hardware/design_space.py` is that
test. This is the usual architecture-DSE contract: validate the model against a
reference implementation, then explore the space with the validated model and
report the model's error.

It deliberately does **not** claim that each individual design point was
separately compiled and emulated. A successful compiler or emulator observation
is scoped by `scope_stack_validity` to the exact geometry it was measured at,
so requiring it per point would confine the reported study to the one geometry
the hardware-validation stage ran at. `legality.py` therefore splits the
blocking stages again:

```text
MODEL_REQUIRED_VALIDITY_STAGES = ("software",)             # required, not scoped
INDIVIDUAL_VALIDATION_STAGES   = ("compiler", "emulator")  # scoped, disclosed
```

For the scoped pair the three `StackValidity` states stay distinct evidence: a
measured `False` is evidence about the point and **excludes** it, a `True`
marks the row individually validated, and a `None` means "not measured here" —
admitted, and said so. Nothing is silently promoted.

Every admitted row therefore carries `individually_validated`, the per-stage
coverage, its own runtime target and — when individually validated — the
evidence target the measurement was taken at, alongside the identities of the
pricing models its numbers rest on. Those fields appear on the admission
verdict, on promotion and selection records, in `hardware_points.csv`, in the
figure receipt (`pricing_model_validation`, with the admitted and
individually-validated point counts), in the final selection record and on
each HBM-sensitivity source.

The strict view remains selectable from the same predicate, so both framings
can be reported side by side:
`evaluate_publication_admission(row, require_individual_validation=True)`,
`select_admitted_rows(rows, require_individual_validation=True)`, and
`selection.individually_validated_points` /
`individually_validated_candidates`. `all_stages_valid` remains the strict,
fail-closed record that every stage including RTL and DC passed.

On the completed Llama study this is the difference between reporting the
validated model's design space and reporting one point of it: of 16,384 priced
rows across 574 profiles, **135 rows across 100 profiles are individually
validated**, all at the single validation geometry (MLEN 1024 / BLEN 8 /
HLEN 128 / batch 1 / 8 KV heads, best 135 tok/s). The other 16,249 rows are
priced by the same validated models, are marked `individually_validated:
false`, and reach 2,226 tok/s whole-model. Widening the individual coverage
means additional hardware-validation points, never a relaxed scoping predicate.

The validation quality the models rest on is read from the calibration
artifacts by `hardware/model_validation.py` and travels with the claim: the
timing model's analytical-versus-emulator MAPE against its limit, the
bandwidth calibration's descriptor-aware holdout median / P95 / P99 and its
worst retained group, the area census's per-family and full-chip holdout errors
together with the fitted MLEN/BLEN grid and an inside/outside verdict, and the
analytic energy tier with each component's evidence scope. All of these are
simulator-calibrated model errors, not measured-silicon errors, and the record
says so on every entry.

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

The canonical comparison scope is the steady-state decode subsystem under the
frozen A100x4 equal-resource envelope. `decode_local_mx_head` is included in
that subsystem: MLEN-padded weight and scale planes, per-step HBM reads,
matrix/vector/selection cycles, bounded SRAM state, TP candidate exchange,
area, power, and energy all travel with each row. Its per-row breakdown is
profile- and numerical-MLEN-bound. MLEN 2048/4096 rows retain analytic costs but
cannot reuse MLEN-1024 accuracy evidence.

Strict admission remains fail-closed while two physical-shape gaps are open:
body attention/expert matrices need shard-aware padding, and TP>1 heads need a
per-rank vocab/hidden layout receipt. Such rows serialize
`body_weight_physical_padding_unmodelled` and, for TP>1,
`local_head_tp_sharded_physical_shape_unmodelled`; they remain useful projected
sensitivity rows but cannot set
`whole_model.strict_system_resource_boundary_valid=true`.

The measured external BF16 prefill/head endpoint is preserved as an optional
whole-service sensitivity. Its two receipts still bind endpoint timing,
energy, residency, silicon area, HBM and deployed-interface scope without
counting the instrumentation driver. Their absence never blocks the local
decode study.

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
2. **Compiler-trace artifacts and the publication evidence gate**, binding the
   external evidence inputs before any gated stage may run.
3. **Preflight**, including the fidelity, correctness, and evidence-building
   stages that gate the sharded sweeps.
4. **Numerical screen shards** — the exhaustive accuracy pass over every
   declared profile; the BF16 reference row for the accuracy budget lives
   here.
5. **Hardware-validation shards**, evaluated with compiler-trace timing.
6. **Exact hardware study partitions** — analytic pricing of every
   profile-by-candidate signature pair, parallelised across
   `study_parallel_workers` spawn workers via deterministic factor blocks
   whose output is byte-identical to the serial order.
7. **Joint source selection** over the verified partitions; when the config
   declares `accuracy_budgets`, the promotion record also carries the strict
   and relaxed accuracy frontiers. Pass that record to the figure stage with
   `--refinement-promotion` so `05_hardware_pareto` draws the emitted
   frontiers instead of recomputing its own.
8. **Refinement**, following the declared protocol.
9. **Repricing** of the selected refined profiles.
10. **Publication benchmarks and final selection**, only when
    `publication_enabled` is true; the pipeline otherwise stops after
    repricing.
11. **Publication figures**, rendered only once the above have completed. The
    figure stage always receives the measured GPU baseline, so the analytic
    energy context and the dual-accuracy envelopes render even when the
    publication benchmark stages are disabled.

### Execution modes and timing tiers

Hardware pricing runs in one of two execution modes, and the declared
publication timing tier is bound one-to-one to the mode:

- `legacy_aggregate_bandwidth` prices decode steps from the stage-calibrated
  analytic model — `max(compute, memory)` per sampled step, with the memory
  stage priced on the emulator-measured, DMA-size-aware bandwidth curves.
  Its rows carry the `stage_calibrated_analytic` tier.
- `compiler_trace` prices from emitted full-model compiler traces with a
  latency library and carries the `compiler_trace_request_calibrated` tier.

Both tiers require compiler and emulator validity; RTL and DC validity are
recorded and disclosed on every row but never required.

### HBM operating points

The calibrated bandwidth model is measured at the emulator's 2 Gb/s pin rate.
For HBM2 that *is* the production pin rate, so HBM2 at 8, 16, and 32 interface
units are the only rankable headline operating points, and
`validate_calibrated_hardware_space` fails closed on anything else. The
calibration rows labelled HBM3 were measured at the same 2 Gb/s emulator rate
rather than a production HBM3 rate, and are not used for headline pricing.
Faster HBM generations appear only through the separate `hbm_sensitivity`
disclosure, which never ranks across generations.

Those three channel planes rest on a receipted Ramulator2 calibration: 11,520
isolated observations over four channel planes (8, 16, 32 and 128), with one
immutable receipt binding every process invocation, its op-statistics, the
emulator and compiler state, and the toolchain. On a descriptor-aware holdout
the model's absolute latency error is 6.27% median and 23.22% at P95.

The tail is not uniform across the planes, and it must travel with the results.
**The hardest retained group is HBM2 matrix prefetch at 32 channels, at 47.43%
P95 (16 channels: 20.42%; 8 channels: 9.78%), and 32 channels is also the
weakest plane for vector store (34.00% P95) and vector prefetch (31.26% P95).**
That band must be quoted wherever 32-channel results are reported. These are
simulator-calibrated model errors, not measured-silicon errors. See
`PLENA_Simulator/analytic_models/disagg_serve/CALIBRATION_PROVENANCE.md`.

### Energy tiers

Every rankable study row prices energy per generated token through
`hardware/power_bridge.py` onto the simulator's analytic decode-power model
(`analytic_models/disagg_serve/decode_power.py`) and carries the
`analytic_anchored` tier. That total is the sum of five separately reported
terms, and their evidence is deliberately mixed rather than uniform:

| Term | Evidence |
| --- | --- |
| HBM read/write per-bit energy | published experimental data, per generation |
| HBM background power | midpoint of a reported range, labelled as such |
| SRAM read energy | median of the vendored ASAP7 macro library internal-power extraction — 0.0479 pJ/bit over 36 macros at TT / 0.7 V / 25 C, replacing a scaled textbook estimate that sat at the optimistic edge of the same extracted range |
| Compute | 0.203 pJ/MAC at 4-bit operands, literature-anchored to a reported whole-chip figure |
| Leakage | 0.05 W/mm², **declared** — no complete-chip leakage report exists |

The coefficient set is hashed into an `energy_id` stamped on every priced row,
so rows priced under different coefficients are never silently mixed.

Supplying `artifacts.power_calibration` and `artifacts.area_config` together
switches pricing to the DC-calibrated event-power engine
(`analytic_models/power`, `dc_calibrated` tier), which fails closed outside its
measured interpolation domain. That is a different engine, not a different
setting: no code changes, and neither engine's numbers are adjusted to resemble
the other's. The figure stage writes `energy_context.json`, which
places the best analytic PLENA point next to the measured GPU board energy
with both tiers, an explicit `model_estimate_over_measured_gpu` ratio
semantics, and `not_a_headline_claim`; measured-versus-measured headline
energy ratios remain the exclusive province of the final deployment table.

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
- the measured baseline device is whatever the run plan declares (a single
  B200 for these studies); A100 and H100 appear only as peak-roofline values
  in a separate labelled table and never acquire a ratio.

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
choice. Two inputs are required under the workspace `external/` boundary and
two more are an optional paired sensitivity:

1. **Passing decode timing evidence.** Currently produced at the emulator tier
   (mode `emulator_serialized`), because RTL does not execute at the anchor
   geometry; the artifact carries `evidence_tier: emulator` and every consumer
   surfaces that tier. An RTL-tier artifact satisfies the same gate.
2. **The full-model independent-request compiler trace set.**
3. **Optional: the BF16 output-head service calibration.**
4. **Optional: the BF16 output-head endpoint-resource receipt**, which accounts for the
   prefill/head endpoint and binds deployment link timing/energy without the
   instrumentation driver.

The third cannot be produced by timing `lm_head` on the execution GPU. It
describes a dedicated BF16 service on the prefill endpoint, so the artifact
requires repeated and holdout logits, endpoint head/selection timing,
endpoint-only component dynamic energy, and endpoint leakage. The B200 driver
remains raw instrumentation evidence; its dynamic energy and request/response
timing cannot stand in for PLENA. The fourth artifact supplies cited/bound
deployed-interface timing and decoder-interface energy, plus the complete
physical resource boundary. If either optional file is supplied, both must be
present and pipeline preflight validates the pair against the pinned model and
every searched batch. Neither is required for the decode-local headline.

The Qwen3-30B-A3B target sets
`output_head_contract.headline_location=decode_local_mx_head`. The
evaluator takes the boundary from `--output-head-location`, defaulting to the
contract the study config declares. External evidence, when present, is
recorded beside the local row as a nonblocking comparison and never supplies
its system identity.

### Refinement and benchmarks

Refinement is partitioned into exactly four logical source shards from one
immutable master schedule, scheduled across the execution pool the run plan
declares.
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

## Operator notes

### Resolving the simulator checkout

`decode_dse` obtains every physical number — area, bandwidth, latency, energy —
from a `PLENA_Simulator` checkout. Resolution has one owner,
`hardware/power_bridge.py::resolve_simulator_root`; the simulator bridge, the
source-digest provenance and the analytic power bridge all call it, so they
cannot end up reading different trees.

`PLENA_SIMULATOR_PATH` wins whenever it is set. An empty value is a
declaration error, and a value that does not contain
`analytic_models/disagg_serve` fails immediately rather than importing a
partial tree. Only a completely undeclared environment falls back to the
*sibling directory literally named* `PLENA_Simulator` — which, from a worktree
checked out under any other name, is a different checkout than the one under
test. That fallback no longer happens silently: every evaluation records
`simulator_root` and `simulator_root_source`
(`environment` or `repository_sibling_default`) in backend provenance and in
`analytic_power_provenance()`, so a study priced against an unintended tree is
visible in its own artifacts. Export the variable for every command;
`scripts/launch_pipeline.sh` does this and should be the only entry point.

### Admission persistence policies

`hardware/admission_cost.py` accepts exactly two persistence policies, and each
pins its own persisted-byte expectation. The policy is one concept that the
on-disk artifacts spell three ways — `persistence_policy` at the top level of
the preparation receipt and its index, `persistence_contract` inside the
persisted-contract resource projection, and `policy` inside the recomputable
resource projection — so it is read through the single accessor
`admission_persistence_policy`, which accepts all three key names, rejects a
document that declares them inconsistently, and fails closed on any policy
string outside the two below. The artifact format is unchanged.

| `persistence_contract` | Persisted bytes | Meaning |
| --- | --- | --- |
| `packed_planes_plus_bf16_numerical_view` | must be `> 0` | packed element and scale planes plus the BF16 numerical view are kept on disk |
| `content_addressed_recompute_per_format` | must be exactly `0` | nothing survives preparation; each format is deterministically rebuilt from its source when required |

Any other policy string fails closed — it is unknown evidence, not a default.
Under the recompute policy the receipt still accounts for the work: the summed
per-record size is reported as the *logical* rebuild cost through
`resource_projection`, and the projected cold-artifact total must be positive,
so a degenerate receipt that projects nothing to rebuild is rejected. Both the
receipt and the index must agree on the policy and on both totals.

The recompute policy is what makes the execution contract above possible: one
KV format is rebuilt, verified and released before the next, so a large logical
admission write volume is never retained simultaneously. Reading only the
persisted contract — as the correctness check previously did — rejected every
receipt prepared under the recompute policy even when the receipt itself
validated.

### The source-tree hash and re-sealing

`software/gpu_baseline.py::_source_tree_hash` digests every `.py`, `.json` **and
`.md`** file under `decode_dse/`, and that digest is bound into a sealed
workspace's provenance. Editing documentation therefore invalidates a sealed
workspace exactly as editing code does. This is intentional — the sealed
identity covers everything that describes the study — but it means
documentation changes are never free on a live run, and must not be made
against a tree a run is executing from.

Re-sealing after any change under `decode_dse/`:

1. **Set aside** — never delete — the stale derived immutables:
   `provenance.json`, `runtime_environment.json`, `admission_preparation.json`,
   `publication_evidence_gate.json`, `preflight_evidence.json`,
   `preflight_gate_report.json`, `pipeline/contract.json`, `pipeline/completed/`,
   `gpu_baseline/`, `artifacts/prefill_bf16`, and the per-stage
   `invocation.json` files. Moving them aside keeps the superseded evidence
   auditable; deleting them destroys the record of what the workspace used to
   claim.
2. Re-capture prefill (`inputs prefill`).
3. Re-run `stage plan`.
4. Re-run `inputs admission`.
5. Resume the pipeline.

**Measured artifacts are not regenerated by this procedure.** The numerical
screen, the hardware-validation results and the external evidence artifacts
under `external/` are measurements, not derived immutables; re-sealing rebinds
them, it does not re-measure them. Re-measuring would discard evidence, not
refresh it.
