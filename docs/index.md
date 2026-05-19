# PLENA — Quantization Evaluation Toolkit

PLENA evaluates MX-quantized large language models against a unified
benchmark surface. One TOML config defines the quantization recipe;
swap the evaluator to score the *same* model on perplexity, lm-eval
harness tasks, code generation, agentic benchmarks, or
diffusion-style decoding.

## Where to start

- **[Get started](getting-started.md)** — install and run your first
  quantized perplexity number in five minutes.
- **[Quantization configs](reference/toml-reference.md)** — every
  field you can set in a TOML quantization recipe.
- **[Evaluation commands](reference/cli.md)** — auto-generated
  reference for every CLI module, arg by arg.

## Paper-table reproductions

Reproducible recipes for the headline results live under
`plena_experiments/` in the repo:

- **table5** — main quantization sweep across Llama-2/3 sizes and three
  bit configurations.
- **table6** — ablations isolating each technique's contribution.
- **table7** — downstream task accuracy using the best Table 5
  recipes.

Each subdirectory contains the configs and run scripts for that table.

## Installation

See `README.md` in the repo root for `uv` setup steps and optional
dependency groups (docs, evalplus, serve, bfcl).
