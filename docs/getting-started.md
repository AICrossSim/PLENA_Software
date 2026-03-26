# Getting Started

## Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended)
- [MASE](https://github.com/DeepWok/mase) installed separately

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd PLENA_Software

# Install quant_eval and its dependencies
pip install -e .

# Install MASE (required for quantization)
pip install -e /path/to/mase
```

!!! note
    MASE is not on PyPI — it must be installed from source. See the [MASE repository](https://github.com/DeepWok/mase) for instructions.

## Dependencies

Installed automatically via `pip install -e .`:

| Package | Purpose |
|---------|---------|
| `torch` | Model inference and quantization |
| `transformers` | Model and tokenizer loading |
| `lm-eval` | EleutherAI evaluation harness |
| `datasets` | HuggingFace dataset loading |
| `accelerate` | Multi-GPU dispatch |
| `jsonargparse` | CLI argument parsing from TOML/YAML/CLI |
| `colorlog` | Coloured logging output |
| `safetensors` | Checkpoint serialization |

## Running an evaluation

### Baseline (no quantization)

```bash
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --tasks wikitext
```

### Quantized model

```bash
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --quant_config configs/linear_mxint.toml \
    --tasks wikitext
```

### Using lm-eval harness benchmarks

```bash
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --quant_config configs/linear_mxint.toml \
    --enable_eval_harness \
    --tasks arc_easy hellaswag
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `meta-llama/Meta-Llama-3-8B` | HuggingFace model ID |
| `--tasks` | `wikitext` | Evaluation task(s) — dataset name for PPL, or lm-eval task names |
| `--quant_config` | `None` | Path to TOML quantization config. Omit for unquantized baseline |
| `--device_id` | `cuda:0` | CUDA device |
| `--model_parallel` | `False` | Distribute model across all GPUs |
| `--enable_eval_harness` | `False` | Use lm-eval harness instead of perplexity |
| `--seqlen` | `2048` | Context window length |
| `--log_dir` | `None` | Directory name for saving run logs (relative to `quant_eval/`) |

## Logging

When `--log_dir` is provided, the tool creates a timestamped subdirectory under `quant_eval/<log_dir>/` containing:

- `args.json` — all CLI arguments for reproducibility
- `quant_config.toml` — copy of the quantization config used
- `results.json` — evaluation results

A `latest` symlink always points to the most recent run.
