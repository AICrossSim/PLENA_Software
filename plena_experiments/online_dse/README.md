# Online DSE for PLENA — GP + Expected Improvement

Bayesian-optimization-based design-space exploration for the PLENA accelerator.
Single objective: **maximize TPS** for a given LLM workload, by varying
hardware knobs (`MLEN/BLEN/VLEN/HLEN`, vector SRAM, HBM size/width, prefetch
amounts). Each candidate is evaluated **in-process** via PLENA_Simulator's
`LLaMAModel`.

## Layout

```
plena_experiments/online_dse/
├── configs/dse_llama3_8b.json   # search grid + paths
├── scripts/online_dse_gp_ei.py  # GP + LogEI loop
└── README.md
```

## Install

```bash
uv sync --extra dse   # adds botorch + gpytorch
```

PLENA_Simulator is **not** a pip dep right now (upstream has broken git
submodules). Clone it alongside this repo:

```bash
git clone https://github.com/AICrossSim/PLENA_Simulator.git ../PLENA_Simulator
```

The default config points at `../PLENA_Simulator`. Edit
`plena_simulator_path` in the config if you cloned it elsewhere.

## Run

From the repo root:

```bash
python plena_experiments/online_dse/scripts/online_dse_gp_ei.py \
    plena_experiments/online_dse/configs/dse_llama3_8b.json
```

Optional flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cache PATH` | `results/online_dse/cache.json` | Disk cache of candidate→TPS evaluations (survives across runs). |
| `-o PATH`, `--output PATH` | `results/online_dse/results.json` | Final result summary. |

## Config schema (`configs/*.json`)

| Field | Type | Notes |
| --- | --- | --- |
| `plena_simulator_path` | str | Local clone of PLENA_Simulator. |
| `model_config_path` | str | HuggingFace-style model config JSON. |
| `plena_settings_toml` | str | Base hardware config (latency tables come from here). Optional; defaults to `<plena_simulator_path>/plena_settings.toml`. |
| `custom_isa_path` | str | `customISA_lib.json`. Optional; defaults to the bundled copy. |
| `batch_size` / `input_seq_len` / `output_seq_len` | int | Workload params passed to `LLaMAModel`. |
| `search.<KNOB>` | list[int] | Discrete grid per hardware knob (all 9 listed in [`KNOBS`](scripts/online_dse_gp_ei.py)). |
| `bo.n_init` | int | Random initialization budget before GP takes over. |
| `bo.n_budget` | int | Total evaluation budget. |
| `bo.seed` | int | RNG seed for both the random init and BO fallbacks. |

Invalid combos (`MLEN % BLEN != 0` or `VLEN < BLEN`) are filtered before
enumeration — those come from `HardwareConfig`'s validators in
PLENA_Simulator.

## Output schema (`results/online_dse/results.json`)

```jsonc
{
  "config": "...absolute path to the input config...",
  "n_evaluated": 40,
  "n_failed":    0,
  "elapsed_s":   123.45,
  "best_tps":    231.7,
  "best_candidate": { "BLEN": 128, "MLEN": 2048, ... },
  "best_tps_trace": [ ... per-step running best TPS ... ],
  "observed": [ { "candidate": {...}, "tps": 145.2 }, ... ]
}
```

## Notes / known limitations

- **Single objective.** Power and cost were objectives in an earlier
  (private) prototype; PLENA_Simulator does not expose a power model
  yet, so EHVI degenerated to LogEI on TPS alone. Multi-objective
  support can be reintroduced once a power/area model lands.
- **In-process eval.** No subprocess, no temp JSONs. Each candidate
  reuses the base `HardwareConfig` (latency fields, etc.) and only
  overrides the listed knobs.
- **sys.path injection.** `LLaMAModel` and `perf_model` use script-style
  imports (`from perf_model import ...`), so the evaluator prepends
  `<plena_simulator_path>/analytic_models/performance/` to `sys.path`.
  Will go away if upstream switches to relative imports.
