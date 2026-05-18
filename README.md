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

## Development (optional)

To hack on `mase` against a local clone instead of the pinned git revision:

```bash
git clone -b releases/plena-experiments git@github.com:DeepWok/mase.git ../mase
uv pip install -e ../mase --no-deps
```
