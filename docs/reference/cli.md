# Evaluation commands

Every PLENA evaluation is a Python module under `quant_eval.cli`. They share a
common pattern: pass `--model_name`, a TOML via `--quant_config`, and any
eval-specific flags. The reference below is auto-generated from each module's
docstrings, so it stays in sync with the code.

## Perplexity — `eval_ppl`

::: quant_eval.cli.eval_ppl
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## lm-eval-harness — `eval_lm`

::: quant_eval.cli.eval_lm
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## Code generation — `eval_evalplus`

::: quant_eval.cli.eval_evalplus
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## Phase-dependent precision — `eval_phase_lm`

::: quant_eval.cli.eval_phase_lm
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## BFCL with phase-dependent precision — `eval_phase_bfcl`

::: quant_eval.cli.eval_phase_bfcl
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## Diffusion LLMs — `eval_dllm`

::: quant_eval.cli.eval_dllm
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## LLaDA diffusion — `eval_llada`

::: quant_eval.cli.eval_llada
    options:
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## Agentic — `eval_osworld`

::: quant_eval.cli.eval_osworld
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3

## Rotation search — `search_rotation`

::: quant_eval.cli.search_rotation
    options:
      members: [main]
      show_root_heading: false
      show_root_full_path: false
      heading_level: 3
