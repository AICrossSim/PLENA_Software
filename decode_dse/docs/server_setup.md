# Execution-host bring-up

## Repository layout

The four repositories are cloned side by side; the sweep plan hashes
`PLENA_RTL/PLENA_Tools/plena_quant` as a quantizer source root and fails
closed without it, so `PLENA_RTL` must be present even though the sweep is
software-only.

```
<root>/PLENA_Software    branch sr1325-decode_dse
<root>/PLENA_Simulator   branch sr1325-dev   (compiler submodule initialized)
<root>/PLENA_RTL         branch sr1325-dev   (PLENA_Compiler submodule initialized)
<root>/mase              branch sr1325/decode-phase-quant
```

Byte-identity invariants must hold before any run:
`PLENA_RTL/PLENA_Compiler == PLENA_Simulator/compiler` and
`PLENA_RTL/PLENA_Tools == PLENA_Simulator/PLENA_Tools`
(`diff -qr … -x __pycache__ -x '*.pyc' -x .git -x .pytest_cache` is empty).

## Environment

All pipeline invocations go through `decode_dse/scripts/launch_pipeline.sh`,
which exports `PLENA_SIMULATOR_PATH` and prepends
`PLENA_Simulator/compiler` and `PLENA_Simulator/PLENA_Tools` to `PYTHONPATH`.
These exports are required: the compiler frontend imports `asm_templates` as a
top-level package, which the in-process `sys.path` injection does not cover.
Do not rely on ambient shell state.

Models and datasets are staged under the paths pinned in
`decode_dse/configs/<model>.json` (`cache_dir`, `model_revision`, dataset
revisions). The Llama tokenizer/model requires an authenticated Hugging Face
fetch at the pinned revision.

## Evidence artifacts

`workspace://external/` must contain, before the pipeline's evidence gate:

| file | origin |
|---|---|
| `decode_timing_evidence.json` | copied from `PLENA_Simulator/analytic_models/performance/evidence/`; self-labelled emulator tier |
| `compiler_trace_artifacts.json` | written by the pipeline's first command (`sweep compiler-trace-artifacts`) |
| `bf16_output_head_service.json` | measured on this host by `decode_dse/hardware/measure_bf16_head_service.py`, one run per model, on two idle exclusively-held GPUs |

The stack-validity stage reports and calibration artifacts ship inside the
repositories and the workspace; they are content-hash bound and must not be
regenerated unless their bound sources changed.

## Per-model launch order

```
scripts/launch_pipeline.sh inputs samples   --config <cfg> --output-dir <ws> ...
scripts/launch_pipeline.sh stage plan       --config <cfg> --output-dir <ws> --dry-run
scripts/launch_pipeline.sh stage plan       --config <cfg> --output-dir <ws> --prompt-manifest <ws>/prompt_manifest.json
scripts/launch_pipeline.sh inputs prefill   --config <cfg> --output-dir <ws> ...
scripts/launch_pipeline.sh inputs admission --config <cfg> --output-dir <ws> ...
# stage the three external artifacts (table above)
scripts/launch_pipeline.sh pipeline --config <cfg> --output-dir <ws> --device-label <label> --gpus 0,1 --dry-run
scripts/launch_pipeline.sh pipeline --config <cfg> --output-dir <ws> --device-label <label> --gpus 0,1
```

The sealed plan fixes two workers, so exactly two GPU ids are passed
regardless of the number of physical boards. Verify with NVML that both
chosen boards are idle and hold no foreign compute processes before the
head-service measurement and before launch.
