# Get started

Five minutes from clone to your first quantized perplexity number.

## 1. Install

```bash
uv venv                    # Python ≥ 3.11.9
source .venv/bin/activate
uv sync                    # core deps — mase, fast-hadamard-transform, lm-eval, ...
```

The lockfile pins `mase` to the `releases/plena-experiments` branch and pulls
`fast-hadamard-transform` transitively. See the repo `README.md` for optional
extras (`docs`, `evalplus`, `serve`, `bfcl`).

## 2. Run your first evaluation

Quantize a Llama-3.2-1B decoder to MXFP4 and measure WikiText perplexity. No
calibration, no rotation — just the simplest possible end-to-end check that
PLENA works on your machine.

```bash
cat > /tmp/quickstart.toml <<'TOML'
by = "regex_name"

["model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj"]
name = "mxfp"
weight_block_size = 32
weight_exponent_width = 2
weight_frac_width = 1
data_in_block_size = 32
data_in_exponent_width = 2
data_in_frac_width = 1

["model\\.layers\\.\\d+\\.mlp\\.(gate|up|down)_proj"]
name = "mxfp"
weight_block_size = 32
weight_exponent_width = 2
weight_frac_width = 1
data_in_block_size = 32
data_in_exponent_width = 2
data_in_frac_width = 1
TOML

python -m quant_eval.cli.eval_ppl \
    --model_name unsloth/Llama-3.2-1B \
    --quant_config /tmp/quickstart.toml \
    --device_id cuda:0
```

Runs in about a minute on a single GPU. The output ends with `ppl: …` — that
number is your model's WikiText perplexity under MXFP4 weight + activation
quantization.

## 3. Next steps

- **[Quantization configs](reference/toml-reference.md)** — every field you
  can set in a TOML config: linear quantization, composite attention,
  `[gptq]`, `[rotation_search]`.
- **[Evaluation commands](reference/cli.md)** — every CLI module with all
  its flags.

## Paper reproductions

The `plena_experiments/` directory in the repo contains config + script
bundles that reproduce each headline result table:

- `plena_experiments/table5/` — main quantization sweep (Llama-2/3 × 3 bit
  configs).
- `plena_experiments/table6/` — component-level ablations.
- `plena_experiments/table7/` — downstream task accuracy.

Each subdirectory has runnable shell scripts that drive the CLIs above.
