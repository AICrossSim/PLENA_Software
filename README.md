# PLENA Software

Quantization and evaluation toolkit for MX-quantized LLMs, and the design-space
exploration that sizes PLENA hardware for them.

The repository holds three bodies of work:

| Area | Location | What it does |
| --- | --- | --- |
| Quantization and evaluation | `plena_experiments/` | MX post-training quantization, GPTQ, rotation search, and the per-table experiment configs |
| Online hardware search | `plena_experiments/online_dse/` | Bayesian optimization over the nine PLENA hardware knobs for a given workload |
| Decode-chip study | `decode_dse/` | Exhaustive precision-by-hardware exploration for a dedicated disaggregated decode chip |

## Documentation

Full documentation is hosted at **<https://aicrosssim.github.io/PLENA_Software/>**, including a getting-started guide, CLI reference for evaluation commands, and the quantization TOML config reference.

## Setup

### Installation

```bash
uv venv                          # Python >= 3.11.9
source .venv/bin/activate
uv sync                          # core deps (mase, fast-hadamard-transform, lm-eval, ...)
uv sync --all-extras             # add docs / evalplus / serve / bfcl / dse as needed
```

Per-table configs and run scripts live under `plena_experiments/`.

### Key dependencies

- **mase** (`mase[mx-ptq]`) — quantization framework. Provides `quantize_module_transform_pass`, GPTQ, rotation search. Pinned via `[tool.uv.sources]` to the `releases/plena-experiments` branch of `DeepWok/mase`.
- **fast-hadamard-transform** — Hadamard kernels used by rotation search. Pulled transitively as a git dep; built with `no-build-isolation`.

## Online DSE (hardware design-space exploration)

GP + Expected Improvement Bayesian optimization over the 9 PLENA `HardwareConfig`
knobs (`BLEN/MLEN/VLEN/HLEN`, vector SRAM, HBM size/width/prefetch). Maximizes
TPS for a given LLM workload by evaluating candidates in-process via
PLENA_Simulator's `LLaMAModel`.

```bash
uv sync --extra dse                                   # adds botorch + gpytorch
git clone https://github.com/AICrossSim/PLENA_Simulator.git ../PLENA_Simulator

python plena_experiments/online_dse/scripts/online_dse_gp_ei.py \
    plena_experiments/online_dse/configs/dse_llama3_8b.json
```

Outputs land in `results/online_dse/{cache.json, results.json}` (override with
`--cache PATH` / `-o PATH`). See [plena_experiments/online_dse/README.md](plena_experiments/online_dse/README.md)
for the full config schema, output format, and known limitations.

## Decode-chip design-space exploration

`decode_dse/` evaluates a dedicated PLENA decode chip for disaggregated serving
of Qwen3-32B and Llama-3.1-8B. Unlike the online search above, it enumerates its
grid exhaustively — 3,585 precision profiles crossed with the legal hardware
space — and keeps numerical accuracy, compiler support, emulator support, RTL
validation, timing calibration and power calibration as separate claims, each
with its own evidence tier.

It requires three sibling checkouts (`PLENA_Simulator`, `PLENA_RTL`, `mase`) and
a host with two BF16-capable GPUs. Start with
[decode_dse/README.md](decode_dse/README.md) for the study design, and
`decode_dse/docs/server_setup.md` for execution-host bring-up and the launch
order.

