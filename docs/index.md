# PLENA Software — Quantization Evaluation Toolkit

**quant_eval** is a config-driven toolkit for quantizing and evaluating large language models using [MASE](https://github.com/DeepWok/mase) MX-format quantization.

## What it does

1. **Loads** a HuggingFace causal LM (e.g. Llama 3).
2. **Quantizes** it via MASE's `quantize_module_transform_pass`, configured entirely through TOML files.
3. **Evaluates** the quantized model with perplexity or the EleutherAI lm-eval harness.
4. **Logs** arguments, configs, and results to timestamped run directories.

## Supported quantization formats

| Format | Description |
|--------|-------------|
| **MXINT** | Microscaling integer (e.g. MXINT4 with block size 16 or 32) |
| **MXFP** | Microscaling floating point (e.g. E2M1 with block size 16) |
| **Minifloat** | Custom exponent/fraction widths (e.g. E3M4) |

## Quantization strategies

| Strategy | Config examples | Description |
|----------|----------------|-------------|
| Linear-only | `linear_mxint.toml`, `linear_mxfp.toml` | Replace only `nn.Linear` layers |
| Composite | `composite_mxint.toml`, `composite_mxfp.toml` | Replace attention, MLP, layernorm, and embedding modules |
| Full | `full_mxint.toml` | Composite + linear replacement in a single pass |
| GPTQ | `linear_mxint_gptq.toml` | Hessian-based weight calibration before MX conversion |
| QuaRot | `linear_mxint_rotate.toml` | Hadamard rotation before quantization |

## Quick start

```bash
# Baseline (no quantization)
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --tasks wikitext

# Quantized with MXINT4
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --quant_config configs/linear_mxint.toml \
    --tasks wikitext
```

See [Getting Started](getting-started.md) for full installation and usage instructions.
