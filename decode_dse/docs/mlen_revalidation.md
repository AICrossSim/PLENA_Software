# MLEN-bound numerical profiles

`DecodePrecisionProfile.matrix_mlen` is an accuracy parameter. The MASE
matrix oracle splits each reduction into contiguous MLEN chunks, performs an
FP32 reduction for each chunk, rounds the partial to `profile.vector_format`,
crosses signed fixed16.16 accumulation, and truncates final writeout. Changing
MLEN therefore creates a new profile ID and a new local-head contract.

`decode_dse.software.mlen_revalidation` provides a separate numerical-only
lane for projected hardware candidates whose MLEN differs from the screened
profile:

- variants are created only with `dataclasses.replace(source,
  matrix_mlen=...)`;
- W, A, symmetric K/V, vector precision, block size, method, and operator
  coverage must remain byte-identical;
- the serialized top-level `numerical_oracle` is checked against the profile
  property and content-addressed in the plan;
- the exact MLEN1024 source, candidate MLEN variant, and BF16 reference are
  each measured on the same 32-document validation and 128-document
  refinement splits;
- one in-memory RTN weight bank is shared by all MLEN values of a weight
  format, and runtime binding must report zero weight requantizations;
- whole weight banks are assigned deterministically to at most four GPU
  shards;
- failures and CUDA OOMs are immutable terminal rows.

The lane never claims hardware bit parity. Its profile, rows, completion, and
conditional selector input all record `hardware_bit_parity_verified=false`,
and the corrected input still requires an exact-profile analytic hardware
reprice. It corrects numerical profile identity only: it does not validate
TP-local physical padding or body scheduling, and it does not make a projected
hardware row rankable.

The module CLI exposes `worker`, `launch`, and `finalize`. The Results-side
adapter builds the source and execution plan after verifying the projected
campaign and joint receipts.
