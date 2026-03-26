# TOML Config Reference

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

## GPTQ section

Optional. Enables Hessian-based weight calibration before MX conversion.

```toml
[gptq]
model_name = "meta-llama/Meta-Llama-3-8B"  # Model for calibration
device = "cuda:0"                           # Device for GPTQ pass
dataset = "wikitext2"                       # Calibration dataset
nsamples = 128                              # Number of calibration samples
seqlen = 2048                               # Sequence length for calibration
format = "mxint"                            # Target MX format
quantile_search = true                      # Enable quantile-based clipping (optional)
clip_search_y = true                        # Enable Y-axis clip search (optional)
checkpoint_dir = "checkpoints/mxint4_gptq"  # Save/load GPTQ checkpoints
# max_layers = 16                           # Limit GPTQ to first N layers (optional)

    [gptq.weight_config]
    weight_block_size = 16                  # Block size for GPTQ weight quantization
    weight_width = 4                        # Bit width for GPTQ weights
```

When using GPTQ, add `gptq = true` to each module selector to use calibrated weights instead of re-quantizing:

```toml
["model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj"]
name = "mxint"
weight_block_size = 16
weight_width = 4
gptq = true    # Use GPTQ-calibrated weights
```

## Rotation section

Optional. Applies Hadamard rotation (QuaRot) before quantization.

```toml
[rotation]
online_rotate = true

["model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj"]
name = "mxint"
weight_block_size = 16
weight_width = 4
data_in_block_size = 16
data_in_width = 4
online_rotate = true    # Enable per-layer online rotation
```
