# Quantization configs

Every PLENA evaluation reads its quantization recipe from a TOML file passed
via `--quant_config`. This page documents every field you can set.

## Top-level fields

```toml
by = "regex_name"   # Required: matching strategy ("regex_name", "name", or "type")
```

## Module selector sections

Each TOML section key is a module selector (regex pattern when `by = "regex_name"`). The section body contains quantization parameters.

### Linear layer parameters

Used when replacing `nn.Linear` with `LinearMXInt` or `LinearMXFP`:

#### MXINT format

```toml
["model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj"]
name = "mxint"
weight_block_size = 16       # Block size for weight quantization
weight_width = 4             # Bit width for weights
data_in_block_size = 16      # Block size for input activations
data_in_width = 4            # Bit width for input activations
bias_block_size = 16         # Block size for bias (optional)
bias_width = 4               # Bit width for bias (optional)
```

#### MXFP format

```toml
["model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj"]
name = "mxfp"
weight_block_size = 16
weight_exponent_width = 2    # Exponent bits for weights
weight_frac_width = 1        # Fraction bits for weights
data_in_block_size = 16
data_in_exponent_width = 2
data_in_frac_width = 1
bias_block_size = 16
bias_exponent_width = 2
bias_frac_width = 1
```

### Composite module parameters

Used when replacing entire attention/MLP/norm/embedding modules.

#### Attention (MXINT)

```toml
["model\\.layers\\.\\d+\\.self_attn"]
name = "mxint"

    # QK matmul quantization
    ["model\\.layers\\.\\d+\\.self_attn".qk_matmul]
    data_in_block_size = 16
    data_in_width = 4

    # AV matmul quantization
    ["model\\.layers\\.\\d+\\.self_attn".av_matmul]
    data_in_block_size = 16
    data_in_width = 4

    # RoPE (minifloat)
    ["model\\.layers\\.\\d+\\.self_attn".rope]
    data_in_exponent_width = 3
    data_in_frac_width = 4

    # Softmax (minifloat)
    ["model\\.layers\\.\\d+\\.self_attn".softmax]
    data_in_exponent_width = 3
    data_in_frac_width = 4

    # KV cache
    ["model\\.layers\\.\\d+\\.self_attn".kv_cache]
    data_in_block_size = 16
    data_in_width = 4
```

#### RMSNorm (minifloat)

```toml
["model\\.layers\\.\\d+\\.(input_layernorm|post_attention_layernorm)"]
name = "minifloat"
weight_exponent_width = 3
weight_frac_width = 4
data_in_exponent_width = 3
data_in_frac_width = 4
```

#### Embedding (MXINT)

```toml
["model\\.embed_tokens"]
name = "mxint"
weight_block_size = 16
weight_width = 4
```

## `[gptq]` block

Optional. Runs Hessian-based GPTQ weight calibration as a pre-pass, then the module-replacement pass picks up the calibrated weights. Checkpoints are saved per layer to `checkpoint_dir` — subsequent runs **auto-resume** from the highest existing layer.

```toml
[gptq]
model_name      = "meta-llama/Meta-Llama-3-8B"   # required: HF model id
format          = "mxint"                        # required: "mxint" | "mxfp"
device          = "cuda:0"                       # default "cuda:0"
dataset         = "wikitext2"                    # default "wikitext2" ("wikitext2"|"c4"|"ptb")
nsamples        = 128                            # default 128 — calibration samples
seqlen          = 2048                           # default 2048 — calibration seq length
cali_batch_size = 32                             # default 32 — minibatch within calibration
quantile_search = true                           # default true — per-block quantile clipping
clip_search_y   = false                          # default false — activation-aware y-norm clip
checkpoint_dir  = "checkpoints/.../mxint4_gptq"  # optional — auto-resume location
# max_layers    = 16                             # optional — quantize only first N layers (debug)

[gptq.weight_config]
weight_block_size = 16                       # required — block size for GPTQ weights
weight_width      = 4                        # required — bit width for GPTQ weights
```

When `[gptq]` is present, add `gptq = true` to each module selector to consume the calibrated weights instead of re-quantizing with RTN:

```toml
["model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj"]
name              = "mxint"
weight_block_size = 16
weight_width      = 4
gptq              = true    # use GPTQ-calibrated weights from [gptq]
```

## `[rotation_search]` block

Optional. Performs a greedy forward search over per-matmul online Hadamard rotations on top of an already-quantized network. Each round tries adding every remaining matmul type to the current rotation set, commits the one that drops perplexity the most, and repeats until no candidate yields more than `improvement_eps`. Winning decisions are cached as JSON so subsequent runs **skip the search** entirely and just re-apply the saved winners (mirrors GPTQ's auto-resume).

When this block is present, it **routes the whole quantize step through itself** — it internally calls GPTQ, baseline module replacement, then the rotation search.

```toml
[rotation_search]
calib_nsamples  = 128                            # default 32 — calibration samples for ppl
calib_seqlen    = 2048                           # default 1024 — calibration seq length
improvement_eps = 0.0                            # default 0.0 — minimum Δppl to commit a rotation
matmul_types    = [                              # optional — restrict search to these matmul types
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "qk_matmul", "av_matmul", "kv_cache",
]
cache_path      = "checkpoints/.../rotation_decisions.json"  # default: <gptq.checkpoint_dir>/rotation_decisions.json
# cache_winners = true                           # default true — set false to force a fresh search
# device        = "cuda:0"                       # default "cuda:0"
# model_name    = "..."                          # default: inherits from [gptq].model_name
# calib_data    = "..."                          # default: inherits from [gptq].dataset
```

