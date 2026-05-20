# PLENA Software

Quantization and evaluation toolkit for MX-quantized LLMs.

## Documentation

Full documentation is hosted at **<https://aicrosssim.github.io/PLENA_Software/>**, including a getting-started guide, CLI reference for evaluation commands, and the quantization TOML config reference.

# Setup

## Installation

```bash
uv venv                          # Python >= 3.11.9
source .venv/bin/activate
uv sync                          # core deps (mase, fast-hadamard-transform, lm-eval, ...)
uv sync --all-extras             # add docs / evalplus / serve / bfcl as needed
```

Per-table configs and run scripts for the paper live under `plena_experiments/`.

## Key dependencies

- **mase** (`mase[mx-ptq]`) — quantization framework. Provides `quantize_module_transform_pass`, GPTQ, rotation search. Pinned via `[tool.uv.sources]` to the `releases/plena-experiments` branch of `DeepWok/mase`.
- **fast-hadamard-transform** — Hadamard kernels used by rotation search. Pulled transitively as a git dep; built with `no-build-isolation`.

