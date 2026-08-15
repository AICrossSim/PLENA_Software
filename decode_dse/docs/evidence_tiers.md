# Evidence tiers

Every number this study reports carries a label naming the evidence behind it.
This page is the single reference for those labels: what each one means, what it
does **not** claim, and where the string is defined. It is a reference, not a
rationale — the design argument is in `decode_dse/README.md`.

The organising rule is that no result inherits evidence from another. A
comparison between two tiers is refused rather than silently made, and a mixed
estimate takes the weakest tier of its parts rather than the strongest.

---

## 1. Cross-stack capability

Five implementation stages are tracked independently for every profile
(`legality.py: CAPABILITY_STAGES`):

```text
software   compiler   emulator   rtl   dc
```

`StackValidity` records `True` / `False` / `None` per stage, where `None` means
*not measured*, not *assumed passing*. Structural capability is applied only as
a **false-only floor**: a limitation can force a stage false, but it can never
mark a stage valid.

The stages are partitioned by what they are allowed to do to a price:

| Set | Stages | Effect of a limitation |
| --- | --- | --- |
| `PRICING_BLOCKING_STAGES` | `software`, `compiler`, `emulator` | withholds the analytic price; the row is recorded with error code `packedkv_selector_unsupported` |
| `PRICING_RECORDED_STAGES` | `rtl`, `dc` | disclosed on the row, forces that stage false, **does not** withhold the price |

The split is not a policy preference; it follows from
`TIMING_TIER_REQUIRED_VALIDITY`, which admits both publication timing tiers on
compiler and emulator validity alone. A row priced with a recorded RTL
limitation is an analytic price at a tier that never claimed RTL execution, and
its `rtl_valid` is false.

`CrossStackCapability` exposes `blocking_issues`, `recorded_issues` (an
exhaustive partition of `issues`), and `prices_at_publication_tier`. On the
evaluated row these appear as `packedkv_selector_blocking_issue_codes`,
`packedkv_selector_recorded_issue_codes`, and the complete
`packedkv_selector_issue_codes`.

**Scoping.** Successful compiler, emulator, RTL and timing evidence applies only
to the exact measured PackedKV geometry *and batch*; `scope_stack_validity`
demotes a `True` to `None` on any other target. DC evidence scopes on geometry
excluding batch. Observed failures stay false everywhere.

**Consequence for coverage.** The joiner credits validity per `profile_id`, then
scopes it against a *single* `evidence_target` (`design_space.py`,
`self.capability_target`). A hardware-validation stage run at one geometry
therefore leaves compiler and emulator validity unmeasured (`None`) on every
priced row away from that geometry, however wide the hardware search is.

That is a coverage limit on *individual* validation, and it is reported as one
(§10): admission rests on the pricing model being validated, and
each row states whether it was additionally compiled and emulated. Widening the
individual coverage means additional hardware-validation points (and an
`evidence_target` that can hold a set of measured geometries rather than one),
never a relaxed scoping predicate, which would extend measured validity to
programs that were never compiled or emulated.

---

## 2. Timing tiers

Defined in `hardware/design_space.py`.

| Tier | Execution mode | What it prices from | Required measured validity |
| --- | --- | --- | --- |
| `stage_calibrated_analytic` | `legacy_aggregate_bandwidth` | `max(compute, memory)` per sampled step, memory priced on emulator-measured DMA-size-aware bandwidth curves | `compiler_valid`, `emulator_valid` |
| `compiler_trace_request_calibrated` | `compiler_trace` | emitted full-model compiler traces against a latency library | `compiler_valid`, `emulator_valid` |

Both are members of `PUBLICATION_TIMING_TIERS`; the execution mode and the tier
are bound one-to-one. Neither claims RTL-simulated or DC-calibrated timing.

**Not a tier — an outcome.** A row that cannot be timed at either tier is
recorded, not discarded, with error code `timing_uncalibrated` and the reason
attached. Such rows are excluded from ranking. Related recorded outcomes:
`runtime_infeasible` (exceeds physical HBM or on-chip SRAM),
`legacy_timing_sensitivity_unrankable`, and - at the external output-head
boundary only - `output_head_unmodeled`.

The external timing-evidence artifact is currently produced at the **emulator**
tier (mode `emulator_serialized`), because RTL does not execute at the anchor
geometry. It self-labels `evidence_tier: emulator` and every consumer surfaces
that label. An RTL-tier artifact satisfies the same gate without a code change.

---

## 3. Energy tiers

| Tier | Engine | Notes |
| --- | --- | --- |
| `analytic_anchored` | `hardware/power_bridge.py` → `analytic_models/disagg_serve/decode_power.py` | what every rankable study row carries today |
| `dc_calibrated` | `hardware/power_model.py` → `analytic_models/power` | reached only when `artifacts.power_calibration` **and** `artifacts.area_config` are both supplied; fails closed outside its measured interpolation domain |

These are two different engines sharing one vocabulary, not two settings of one
engine. Neither engine's numbers are adjusted to resemble the other's.

`analytic_anchored` totals are the sum of five separately reported terms whose
evidence is deliberately mixed:

| Term | Evidence |
| --- | --- |
| HBM read/write per-bit energy | published experimental data, per generation |
| HBM background power | midpoint of a reported range, labelled as such |
| SRAM read energy | median of the vendored ASAP7 macro library internal-power extraction (0.0479 pJ/bit, 36 macros, TT / 0.7 V / 25 C) — macro-library internal power, **not** PLENA netlist power |
| Compute | 0.203 pJ/MAC at 4-bit operands, literature-anchored; independently bracketed by a gate-level envelope (below) |
| Leakage | 0.05 W/mm², **declared**; no complete-chip leakage report exists. Bounded below by a measured 25 °C figure (below) |

The coefficient set is hashed into an `energy_id` carried on every priced row,
so rows priced under different coefficients cannot be mixed unnoticed.

#### Gate-level cross-checks that did not change a coefficient

A Design Compiler campaign on the MatrixMachine (8 timing-closed points, ASAP7
RVT_TT at `PVT_0P7V_25C`, MLEN 16–64, MXFP) bears on two of the coefficients
above. Both outcomes are surfaced by `model_validation.energy_model_validation()`
under `gate_level_cross_checks`, and both are **corroboration, not calibration**
— the tier stays `analytic_anchored`.

- **Compute.** The campaign priced two mapped netlists over a declared toggle
  envelope of 0.05–0.50, giving 0.113–1.126 pJ/MAC. The 0.203 pJ/MAC anchor sits
  inside it at an implied toggle of 0.0797 (32×4) and 0.0835 (16×4) — consistent
  across geometries. The toggle rate is *assumed and propagated by the synthesis
  tool*, not measured from decode switching, so this brackets the anchor rather
  than replacing it.
- **Leakage.** Measured density is 9.21e-04 W/mm², roughly **54× below** the
  declared 0.05 W/mm². It is **not adopted**: 25 °C is the coldest library
  corner and no hot corner was synthesised, so the derating to an 85–125 °C
  operating junction is unmeasured; and the measured design is 98.3% dense
  compute array, while the coefficient is charged against whole-chip non-memory
  logic. The declared value is therefore kept as the conservative default and
  the measurement is recorded as a scoped lower bound. At a representative
  decode point the leakage term is 0.20% of total power at 0.05 W/mm² and 0.004%
  at the measured density, so the choice moves tokens/joule by about 0.2%.

**Scope guard.** These are one-block figures, in µm², at MLEN 16–64, at 25 °C.
Every coefficient in the table above is charged against a full chip in mm² at
MLEN 128–1024. The record carries that boundary in its `scope` field; nothing
here licenses comparing the two.

### Energy comparison semantics

`energy_context.json` (written by the figure stage) places the best analytic
PLENA point beside the measured GPU board energy. It records
`numerator_tier`, `denominator_tier`,
`ratio_semantics: model_estimate_over_measured_gpu`, and
`not_a_headline_claim: true`. It refuses a measured PLENA numerator — those
belong in the headline comparison.

Headline energy ratios accept **measured evidence on both sides only** and
remain exclusive to the final deployment table.

---

## 4. Area tiers

The study's own `area_source` label, from `hardware/evaluation.py`:

| `area_source` | Meaning |
| --- | --- |
| `analytical_uncalibrated` | the fallback multiplier proxy; no full-chip decomposition |
| `analytic_full_chip` | the simulator's precision-aware structural census |
| `dc_calibrated` | a candidate-exact full-chip DC/SAIF anchor; the only path that also sets `dc_calibrated=True` on the row's `StackValidity` |
| `dc_calibrated_model` | the DC-fitted event-power model's area, inside its calibration domain; leaves DC validity unmeasured (`None`) |

`--power-calibration` and `--area-config` must be supplied together, and
`--exact-dc-anchors` additionally requires `--power-calibration` and
`--rtl-source-tree-sha256`. Supplying half a pair raises rather than silently
falling back.

Underneath, the simulator's area package labels each *block*
(`analytic_models/area/evidence.py`):

| Tier | Meaning |
| --- | --- |
| `dc_synthesized_aggregate_fit` | fitted to aggregate 7 nm DC results, queried **inside** the block's calibration domain |
| `dc_synthesized_aggregate_structural_extrapolation` | same fit, queried outside that domain — the flag is `structural_extrapolation: true` |
| `published_sram_macro_geometry` | ASAP7 macro tiling; `foundry_compiler_result: false` |
| `published_density_node_scaled` | C2C link PHY from a published density, node-scaled |
| `declared_structural_estimate` | chip-side HBM PHY, and any opt-in RTL structure no retained DC point contains |
| `mixed_analytic_evidence` | more than one tier present with no dominant one |

`weakest_tier` collapses a mixed estimate downward, so a single extrapolated or
declared block degrades the whole chip's tier.

Two facts should travel with any area number from this study:

- the searched geometries (MLEN ≥ 1024) are **outside** the calibrated grid
  (which stops at 64), so headline chip areas carry the extrapolation tier, and
  the package's in-domain holdout errors are not an error bar on them; and
- charging HBM PHY requires passing `hbm_interface_units_per_chip` explicitly —
  it defaults to zero, and a zero charge leaves no evidence record behind.

The area calibration's own provenance grade is
`aggregate_area_tables_without_raw_dc_reports`, and its audit lists "raw DC
report provenance is complete" among its **unsupported** claims.

### Independent gate-level cross-validation of the census

A separate Design Compiler campaign — 8 timing-closed MatrixMachine points,
ASAP7 RVT_TT at `PVT_0P7V_25C`, MLEN 16–64, BLEN 4–8, MXFP — was never used to
fit any coefficient, so it is a true holdout for the MXFP census. The summary is
surfaced on the area entry of the pricing-model validation record as
`independent_gate_level_cross_validation`.

The census over-predicts these points by a **uniform 1.1242×**; removing that
single offset leaves **0.41% median and 2.97% worst** error across every measured
shape and precision. The offset is a level difference between two synthesis
campaigns, not a shape or precision error — the shipped calibration CSV is itself
1.113× the new campaign at the six geometries they share — and the census already
carries a uniform level constant that absorbs it. **No coefficient was refitted.**
One genuine limit is recorded rather than hidden: the census features depend on
total operand width only, so equal-width MXFP formats (E1M2, E2M1) are predicted
identically while measuring 2.66% apart.

**Scope guard.** One block, in µm², at MLEN 16–64, at 25 °C — against full-chip
mm² at MLEN 128–1024 for every priced row. This campaign is not an error bar on
the study's chip areas, and the record says so in its `scope` field.

---

## 5. Memory-bandwidth calibration

`ramulator2_simulated` on every calibration row: simulator-measured, not
measured silicon.

| Dataset | Grade |
| --- | --- |
| `calibration_dma_requests.csv` (11,520 rows, 4 channel planes) | `ramulator2_structured_csv_with_process_receipts` — every process recorded and replayable |
| `calibration_bw.csv`, `calibration_dma.csv` (aggregate) | `aggregate_csv_without_raw_run_receipts` — valid analysis, incomplete receipt |

The combined grade is limited by its weakest input and therefore stays
incomplete.

Descriptor-aware holdout on the structured dataset: **6.27% median, 23.22%
P95**. The tail is plane-dependent and must be quoted with any 32-channel
result: HBM2 matrix prefetch at 32 channels is **47.43% P95** (16 channels
20.42%, 8 channels 9.78%), and 32 channels is also the weakest plane for vector
store (34.00%) and vector prefetch (31.26%).

Only HBM2 at 8, 16 and 32 interface units is a rankable headline operating
point. Every calibration row was measured at a 2 Gb/s pin rate, which is
production rate for HBM2 and is not for HBM3; the HBM3-labelled rows exist for
emulator-consistency checks. Faster generations reach reports only through the
labelled `hbm_sensitivity` path, which never ranks across generations.

---

## 6. Accuracy labels

Accuracy is a **disclosed budget, never a filter**. Every profile is priced and
stamped with an `accuracy_within_limit` label; nothing is removed from the
design space for being inaccurate.

Where the study config declares `accuracy_budgets`, two consumers read it, and
they share one definition of "inside a budget":

- **Joint selection.** `hardware/selection.py::dual_accuracy_frontiers` reports
  three budgets — `strict`, `relaxed` and `unconstrained` — each with its
  relative-perplexity limit, the derived `mean_nll_limit`
  (`reference_mean_nll + log(limit)`), the admitted-point count, and the first
  epsilon-Pareto front among the admitted points. An empty frontier is
  reported, never raised. The result is stored under `dual_accuracy_frontiers`
  in the refinement source-selection record.
- **The hardware Pareto figure** (`plots.py`, `05_hardware_pareto`) reads
  `accuracy_budgets` from the study config and, when the source-selection
  record is passed with `--refinement-promotion`, draws the strict and relaxed
  envelopes *from the emitted frontiers themselves*. Without that record it
  recomputes the envelopes locally and says so in the figure subtitle
  (`accuracy frontier: locally recomputed (no frontier record)` versus
  `accuracy frontier: emitted dual_accuracy_frontiers record`).

Budget membership is never duplicated. Both sides reduce their own accuracy
column — the record's `mean_nll` against the reference, the figure's
`relative_perplexity_percent` — to a relative-perplexity ratio and call
`selection.within_accuracy_budget`, so they cannot admit different point sets.
When a record is supplied, every front row it names must resolve to a plotted
row that the shared filter also admits, or the figure fails closed rather than
drawing an envelope the selection record does not support.

The two envelopes still answer different questions and need not be the same
size: the selection front keeps mean NLL as a dominance objective across five
objectives, while the figure's own fallback envelope is the two-objective
latency–energy lower-left hull. Which one produced the drawn curve is stated on
the figure.

The strict envelope reproduces the deployment gate; the relaxed envelope shows
what lower-precision weight formats buy at a labelled accuracy cost. Within
either budget, mean NLL remains a dominance objective — the budget partitions,
it does not discard.

**Both studies declare the same `accuracy_budgets`** (strict 1.01, relaxed
1.05), so both promotion records carry populated dual accuracy frontiers and
both Pareto figures render budget envelopes.

The accuracy metric is `post_handoff_greedy_conditioned_nll`, relative to the
BF16 reference row measured in the same screen. It is **not** standard
WikiText-2 perplexity: the prefill-greedy handoff token is unscored and later
tokens use cached `q_len=1` teacher forcing with exact cache positions.

---

## 7. Measured evidence

`measured` (`MEASURED_EVIDENCE_TIER`) is reserved for values read from
instruments.

- The GPU baseline is measured on whatever device the run plan declares — a
  single B200 for these studies (`gpu_baseline.first_gpu_only` is true). A100
  and H100 appear only as peak-roofline values in a separate labelled table and
  never acquire a ratio.
- Board energy is measured over only the synchronised post-warmup decode
  region, preferring the NVML cumulative-energy counter and falling back to a
  timestamped NVML power trace with trapezoidal integration. If neither meter is
  supported, throughput stays valid and energy is reported as explicitly
  unavailable; **no analytic value may stand in for it.**
- The throughput ratio in the final deployment table stays empty until a
  same-tier measured PLENA result exists.

---

## 8. Holdout gates

`hardware/calibration.py::CalibrationThresholds` sets what a calibration must
achieve before its outputs are rankable at all:

| Quantity | Median | Max |
| --- | --- | --- |
| Area | 0.10 | 0.15 |
| Dynamic power | 0.15 | 0.25 |
| Leakage power | 0.15 | 0.25 |
| Cycles | — | 0.05 |

Latency error must be at most 0.10 — note that despite the field name
`latency_mape` this is the **median** absolute relative error, not the mean.
Ranking correlation must be at least 0.90; it is Spearman over the
dynamic-power pairs only, and it is the one gate that fails closed when it
cannot be computed at all. Each category needs at least two holdout pairs.
Comparisons are strict, so a value exactly at a threshold passes.

Power and timing are rankable only once their gates pass.

Separately, the DC-fitted event-power model declares its own interpolation
domain (`power_model.EXPECTED_POWER_INTERPOLATION_DOMAIN`: MLEN 16–64, BLEN
4–16). The searched MLEN grid runs to 4096, so **most of the design space is
outside that domain by construction**; `calibrated_energy_from_simulator` and
`calibrated_area_from_simulator` refuse to extrapolate rather than returning an
optimistic number.

---

## 9. Output-head boundary

Where the BF16 LM head runs is a scope decision with its own evidence, declared
by `output_head_contract.headline_location` in the study config and resolved by
`--output-head-location`.

| Location | What the decode chip pays | Head cost evidence | Disclosure |
| --- | --- | --- | --- |
| `decode_bf16_unmodeled` (**headline**) | BF16 head weights in HBM capacity, head traffic every decode step | compute idealized - no measured instruction-level timing or energy signature | `local_bf16_head_compute_idealized` on every row |
| `external_bf16_service` (comparison) | nothing past the final RMSNorm | measured reserved endpoint: repeated and holdout logits, link timing, component dynamic energy, leakage | none claimed |

Both are rankable, and both carry their location, service mode and scope
idealizations in `metrics.output_head_boundary`. The idealization list is
*derived* from the location rather than supplied, so a locally-headed row
cannot be written without its disclosure, and the measured arm cannot borrow
one. The composed system identity differs by construction
(`decode-local-head-system-…` versus `decode-head-system-…`), so the two
placements can never be confused for one another after the fact.

The external endpoint is reserved for the whole decode step, so its idle draw
is charged across that step and dominates the whole-model energy ledger. The
local boundary provisions no second device and therefore charges no external
idle power - see `_whole_model_energy` in `hardware/evaluation.py`.

---

## 10. Result admission

### What an admitted row claims

A row is admitted into the reported results when it was **priced by a
validated, identified pricing model**, and when nothing measured about the
point itself contradicts that price. Concretely, `evaluate_publication_admission`
(`hardware/design_space.py`) requires all of:

| Requirement | Field on the row |
| --- | --- |
| the numerical run for the profile succeeded | `numerical_summary.state` |
| static hardware legality | `legality.hardware_candidate` |
| measured software-stage validity | `validity.software_valid` |
| no measured compiler or emulator **failure** | `validity.compiler_valid`, `validity.emulator_valid` are not `False` |
| no structural capability limitation on a blocking stage | `capability.issues`, `capability.stage_support` |
| calibrated timing, bound to a timing-evidence identity, at a declared tier | `metrics.timing_calibrated`, `metrics.timing_evidence_id`, `whole_model.publication_timing_tier` |
| calibrated memory timing, bound to a bandwidth calibration identity | `metrics.memory_timing_calibrated`, `metrics.bandwidth_calibration_id` |
| an energy tier plus the identity of the coefficient set or fit behind it | `calibrated_energy.energy_tier`, `energy_id` |
| an identified area model (plus its calibration identity when DC calibrated) | `metrics.area_source`, `metrics.area_calibration_id` |
| a priced output-head boundary, a layout identity, a composed system identity | `metrics.output_head_boundary`, `metrics.layout_id`, `whole_model.system_calibration_id` |
| capacity, runtime and resource-budget feasibility | `metrics.capacity.feasible`, `metrics.runtime_feasible`, `metrics.resource_budget.feasible` |
| the PackedKV selector verdict, scoped to the blocking stages | `packedkv_selector_evidence` |

This is the standard architecture-DSE contract: validate the model against a
reference implementation, then explore the design space with the validated
model and report the model's error. Section 11 states where those error
figures live and how they reach the records.

### What it does not claim

Admission does **not** claim that this individual design point was separately
compiled and emulated. A successful compiler or emulator observation is scoped
by `scope_stack_validity` to the exact geometry it was measured at, so
requiring it per point would confine the reported study to the single geometry
the hardware-validation stage happened to run at. That coverage is
therefore **reported, not required**.

Nor does admission claim RTL-simulated or DC-calibrated evidence, which
neither publication timing tier ever claimed. RTL and DC validity remain
recorded and disclosed through `recorded_validity` and
`rests_on_unimplemented_rtl_path`.

### The three validity states are kept distinct

`StackValidity` records `True` / `False` / `None`, and the three are different
evidence. `legality.py` splits the blocking stages by how far a *successful*
measurement carries:

| Set | Stages | Effect |
| --- | --- | --- |
| `MODEL_REQUIRED_VALIDITY_STAGES` | `software` | not geometry scoped; measured `True` is **required** |
| `INDIVIDUAL_VALIDATION_STAGES` | `compiler`, `emulator` | geometry scoped; `True` is disclosed coverage, `None` is admitted and disclosed, `False` **excludes** |

A measured `False` is evidence about the point itself and always excludes it.
A `None` means "not measured here", never "measured and passed": nothing is
silently promoted.

### Per-row coverage disclosure

Every admitted row carries an explicit coverage record
(`IndividualValidationCoverage`), emitted on the admission verdict and carried
into the downstream records:

- `individually_validated` - `true` only when both compiler and emulator were
  measured valid at this row's own geometry;
- `individually_validated_stages`, `unmeasured_stages`, `failed_stages`;
- `runtime_target` - the row's own PackedKV geometry;
- `evidence_target` - the geometry the individual evidence was measured at,
  which is the row's own target when `individually_validated` is true, and
  `null` otherwise. Because a `True` survives scoping only on an exact
  geometry match, these two cannot disagree.

The identities of the pricing models themselves travel alongside as
`pricing_model` (timing evidence, bandwidth calibration, energy tier and id,
area source and calibration id, layout, system and head-service identities).

Where the disclosure appears:

| Record | Fields |
| --- | --- |
| `PublicationAdmission.to_dict()` | the full verdict, `admission_basis`, coverage, `pricing_model` |
| `ParetoPoint.to_dict()` (promotion records) | `individually_validated`, `individual_validation_coverage`, `pricing_model` |
| `PublicationCandidate.to_dict()` (selection records) | `individually_validated`, `individual_validation_stages`, `admission_basis`, `all_stages_valid` |
| `hardware_points.csv` (figure data table) | `individually_validated`, `individually_validated_stages`, `timing_evidence_id`, `bandwidth_calibration_id`, `area_source` |
| `figure_manifest.json` | `selection_policy.hardware`, `pricing_model_validation` (§11) with admitted and individually-validated point counts |
| `decode-final-publication-selection` | `selection.admission`, `hardware_join.individually_validated_candidate_count` |
| HBM sensitivity sources | `individually_validated` per source |

### The strict view stays available

The individually-validated subset is selectable from the same predicate, so
both framings can be reported side by side without a second policy:

- `evaluate_publication_admission(row, require_individual_validation=True)`;
- `design_space.select_admitted_rows(rows, require_individual_validation=True)`;
- `selection.individually_validated_points`, `individually_validated_candidates`.

A row refused on a *pricing-model* input reports that reason even in the strict
view, so missing coverage never masks a missing model input.

`PublicationCandidate.priced_evidence_complete` remains the candidate-level
eligibility test and the meaning of the `cross_stack_validity` failure code:
it fires when a required stage is missing or failed, or when a pricing
identity is absent. `all_stages_valid` remains the strict, fail-closed record
that every stage - blocking, RTL and DC - was measured valid, and it now
subsumes individual validation.

One exception is deliberate. A candidate that *claims* the `dc_calibrated`
energy tier must carry measured DC validity, because there the tier is the
claim. Absent DC evidence the candidate is still eligible at
`analytic_anchored`.

### Coverage on the completed llama3.1-8B study

Of the 16,384 priced rows (574 profiles), the hardware-validation stage ran at
one geometry only - MLEN 1024 / BLEN 8 / HLEN 128 / batch 1 / 8 KV heads - so
**135 rows across 100 profiles are individually validated** and every one of
them sits at that geometry. The remaining 16,249 rows are priced by the same
validated models and are marked `individually_validated: false`. Widening the
individual coverage means additional hardware-validation points, never a
relaxed scoping predicate.

---

## 11. Pricing-model validation quality

`hardware/model_validation.py` reads the validation artifacts themselves - no
figure below is hard-coded - and composes them into a
`plena-pricing-model-validation` record carried in the figure receipt. A
component whose artifact cannot be read is recorded as `unavailable` with the
reason rather than omitted.

| Model | Reference | Artifact | Holdout / agreement |
| --- | --- | --- | --- |
| decode-step timing | compiler-emitted program under the emulator | `decode_timing_evidence.json` | analytical-vs-emulator MAPE, limit, worst-anchor error, evidence tier and mode |
| request-latency (effective bandwidth) | ramulator2-simulated request traces | `calibration_dma_requests.validation.json` | descriptor-aware holdout median / P95 / P99, worst retained group by P95 and by median |
| structural area census | Synopsys DC 7 nm areas | `matrix_structural_coefficients.json` + the calibration CSVs | per-family and full-chip holdout errors, and the fitted MLEN/BLEN grid with an inside/outside verdict |
| structural area census, independent check | a later gate-level campaign no coefficient was fitted to | `matrix_gate_level_validation.json` | uniform census offset, post-offset shape/precision error, the declared model limit, and the campaign's scope |
| analytic decode energy | identified coefficient set | `decode_power.py` | no holdout exists; the record states so and names each component's evidence scope, including the two declared-not-calibrated coefficients, plus the two gate-level cross-checks that did not change a coefficient |

Every one of these is a **simulator-calibrated model error, not a
measured-silicon error**; the record carries that scope string on each entry.
The area entry additionally carries `domain_status`, because the searched
MLEN/BLEN grid runs past the synthesised grid and an outside-domain point is
priced by structural extrapolation - the in-domain holdout is not an error bar
on it (§4).

---

## Where the strings live

| Family | Source |
| --- | --- |
| capability stages, blocking/recorded split | `decode_dse/legality.py` |
| individual vs model-required stage split, `ADMISSION_BASIS` | `decode_dse/legality.py` |
| pricing-model validation figures | `decode_dse/hardware/model_validation.py` |
| timing tiers, required validity | `decode_dse/hardware/design_space.py` |
| energy tiers | `PLENA_Simulator/analytic_models/disagg_serve/decode_power.py` |
| `area_source` | `decode_dse/hardware/evaluation.py` |
| block area tiers | `PLENA_Simulator/analytic_models/area/evidence.py` |
| calibration grades | `PLENA_Simulator/analytic_models/disagg_serve/calibration_provenance.py` |
| holdout thresholds | `decode_dse/hardware/calibration.py` |
| measured tier, energy-context semantics | `decode_dse/software/gpu_baseline.py` |
| output-head locations, modes, idealizations | `decode_dse/hardware/lm_head_service.py` |
| admission verdict and coverage records | `decode_dse/hardware/design_space.py` |
| candidate-level eligibility test | `decode_dse/hardware/selection.py` |
