"""Qwen3-32B disaggregated-decode precision DSE on PLENA.

A low-precision decode chip continues generation from an unquantised (bf16) prefill chip's
handed-off KV cache. Bayesian (Optuna TPE) search over per-component precision (attention / FFN /
KV independent), ranked by a fast continuation-PPL proxy and confirmed with strict IFEval accuracy
(thinking mode) on the Pareto frontier.

  quant.py  -- precision language (search space, MASE pass-args, decode-cost proxy)
  disagg_serve.py -- disaggregated engine (load, quantise, KV hand-off, generate, PPL proxy)
  ifeval.py -- the metric (strict prompt/instruction accuracy + length records)
  search.py -- running end to end (TPE search, GPTQ refine, IFEval frontier, plots)
"""
