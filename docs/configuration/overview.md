# Configuration Overview

All quantization is configured through TOML files in the `configs/` directory. Each file defines:

1. A **matching strategy** (`by`) for selecting which modules to quantize.
2. **Module selectors** — regex patterns that match layer names in the model.
3. **Quantization parameters** for each matched module group.
4. Optionally, a **GPTQ** or **rotation** section for advanced calibration.

## Config directory

```
configs/
├── linear_mxint.toml              # MXINT4 on linear layers only
├── linear_mxfp.toml               # MXFP E2M1 on linear layers only
├── linear_mxint_gptq.toml         # MXINT4 + GPTQ calibration
├── linear_mxint_gptq_w_search.toml # MXINT4 + GPTQ + quantile search
├── linear_mxint_rotate.toml       # MXINT4 + QuaRot Hadamard rotation
├── composite_mxint.toml           # MXINT4 on composite modules (attn, MLP, norm, embed)
├── composite_mxfp.toml            # MXFP on composite modules
├── full_mxint.toml                # Composite + linear in one pass
└── mxint4_gptq_b32_w_search.toml  # MXINT4 GPTQ with block size 32
```

## Matching strategy

The `by` field at the top of every config selects how module selectors are interpreted:

| Value | Description |
|-------|-------------|
| `"regex_name"` | Match module names using Python regex (most common) |
| `"name"` | Match module names exactly |
| `"type"` | Match by module class name |

## Config categories

### Linear-only configs

Replace `nn.Linear` layers (attention projections and MLP projections) with quantized variants. The rest of the model (attention logic, layernorm, embedding) runs in FP16.

**When to use:** Quick experiments to measure the impact of weight/activation quantization on linear layers alone.

### Composite configs

Replace entire composite modules — attention blocks, MLP blocks, layernorm, and embeddings — with MX-quantized variants. This quantizes internal operations like matmuls, softmax, RoPE, and KV-cache.

**When to use:** Full-model quantization studies where non-linear operations also need reduced precision.

### Full configs

Combine composite and linear replacement in a single pass. Parents (composite modules) are replaced first, then children (linear layers) are replaced within them.

**When to use:** Maximum quantization coverage in one step.

### GPTQ configs

Add Hessian-based weight calibration before MX format conversion. GPTQ optimizes weights using calibration data, then the quantized weights are loaded into MX linear layers.

**When to use:** When weight-only quantization accuracy matters and you have calibration data available.

### Rotation configs

Apply Hadamard rotation (QuaRot) before quantization to reduce outlier sensitivity.

**When to use:** When activation outliers degrade quantization quality.

See [TOML Reference](toml-reference.md) for the full parameter specification.
