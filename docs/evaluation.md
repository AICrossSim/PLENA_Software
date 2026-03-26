# Evaluation

quant_eval provides two evaluation methods, selectable via the `--enable_eval_harness` flag.

## Perplexity (default)

Computes sliding-window perplexity on a HuggingFace dataset.

```bash
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --quant_config configs/linear_mxint.toml \
    --tasks wikitext
```

**How it works:**

1. Loads the test split and tokenizes the full text.
2. Splits into non-overlapping chunks of `--seqlen` tokens.
3. Computes cross-entropy loss per chunk under `torch.no_grad()`.
4. Reports `exp(mean_loss)` as the perplexity.

**Options:**

- `--tasks` sets the dataset name (default: `wikitext`, which auto-selects `wikitext-2-raw-v1`).
- `--seqlen` controls the chunk size (default: 2048).

## LM Eval Harness

Uses [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) for standard benchmarks.

```bash
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-8B \
    --quant_config configs/linear_mxint.toml \
    --enable_eval_harness \
    --tasks arc_easy hellaswag winogrande
```

**How it works:**

1. Wraps the HuggingFace model in an `HFLM` adapter.
2. Calls `simple_evaluate()` with the specified tasks.
3. Prints a formatted results table and returns the full results dict.

**Options:**

- `--tasks` accepts one or more lm-eval task names.
- Batch size is automatically determined (`"auto"`).

## Multi-GPU evaluation

For models that don't fit on a single GPU:

```bash
python -m quant_eval.cli.eval \
    --model_name meta-llama/Meta-Llama-3-70B \
    --quant_config configs/linear_mxint.toml \
    --model_parallel \
    --tasks wikitext
```

This uses `accelerate`'s `dispatch_model` with a balanced device map across all available GPUs.

## Logging results

Add `--log_dir` to save run artifacts:

```bash
python -m quant_eval.cli.eval \
    --quant_config configs/linear_mxint.toml \
    --tasks wikitext \
    --log_dir logs/mxint4_experiment
```

This creates a timestamped directory:

```
quant_eval/logs/mxint4_experiment/
├── run-20260223-114305/
│   ├── args.json            # All CLI arguments
│   ├── quant_config.toml    # Copy of the config used
│   └── results.json         # Evaluation output
└── latest -> run-20260223-114305/
```
