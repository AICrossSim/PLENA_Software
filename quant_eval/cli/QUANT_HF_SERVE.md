# quant-hf-serve

Standalone OpenAI-compatible server for HuggingFace models, with optional
MX weight quantization and phase x layer-type disaggregated activation precision.

## Quick start

### Serve without quantization (plain fp16/bf16)

```bash
python -m quant_eval.cli.quant_hf_serve \
    --model_name zai-org/GLM-4.6 \
    --model_parallel true \
    --host 127.0.0.1 --port 8915
```

### Serve with MX quantization + phase-layer disaggregation

```bash
python -m quant_eval.cli.quant_hf_serve \
    --model_name Qwen/Qwen3-8B \
    --quant_config quant_eval/configs/llama_mxint4.toml \
    --prefill_attn_width 4 --prefill_ffn_width 4 \
    --decode_attn_width  8 --decode_ffn_width  8 \
    --host 127.0.0.1 --port 8915
```

### Multi-GPU

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m quant_eval.cli.quant_hf_serve \
    --model_name zai-org/GLM-4.6 \
    --model_parallel true \
    --host 127.0.0.1 --port 8915
```

## Endpoints

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| POST   | `/v1/chat/completions`  | Chat completion (OpenAI-compatible)  |
| POST   | `/v1/completions`       | Text completion (OpenAI-compatible)  |
| GET    | `/v1/models`            | List served model                    |
| GET    | `/v1/config`            | Read current phase-layer config      |
| PUT    | `/v1/config`            | Update phase-layer config at runtime |

## Runtime config changes

When serving with `--quant_config`, you can change the activation precision
for any (phase, layer-type) combination **without restarting the server**.
Changes take effect on the next inference request.

### Read current config

```bash
curl -s http://127.0.0.1:8915/v1/config | python -m json.tool
```

Example response:

```json
{
    "prefill": {
        "attn": {"data_in_width": 4, "data_in_block_size": 32},
        "ffn":  {"data_in_width": 4, "data_in_block_size": 32}
    },
    "decode": {
        "attn": {"data_in_width": 8, "data_in_block_size": 32},
        "ffn":  {"data_in_width": 8, "data_in_block_size": 32}
    }
}
```

### Update a single setting

Omitted keys are left unchanged. For example, to change only decode FFN to 4-bit:

```bash
curl -s -X PUT http://127.0.0.1:8915/v1/config \
    -H "Content-Type: application/json" \
    -d '{"decode": {"ffn": {"data_in_width": 4, "data_in_block_size": 32}}}'
```

### Update multiple settings at once

```bash
curl -s -X PUT http://127.0.0.1:8915/v1/config \
    -H "Content-Type: application/json" \
    -d '{
        "prefill": {
            "attn": {"data_in_width": 8, "data_in_block_size": 32},
            "ffn":  {"data_in_width": 4, "data_in_block_size": 32}
        },
        "decode": {
            "attn": {"data_in_width": 8, "data_in_block_size": 32},
            "ffn":  {"data_in_width": 6, "data_in_block_size": 32}
        }
    }'
```

### Sweep script example

Run a grid search over activation widths while the server stays up (no reload):

```bash
for pa in 4 8; do
  for pf in 4 8; do
    for da in 4 6 8; do
      for df in 4 6 8; do
        curl -s -X PUT http://127.0.0.1:8915/v1/config \
          -H "Content-Type: application/json" \
          -d "{
            \"prefill\":{\"attn\":{\"data_in_width\":$pa,\"data_in_block_size\":32},\"ffn\":{\"data_in_width\":$pf,\"data_in_block_size\":32}},
            \"decode\":{\"attn\":{\"data_in_width\":$da,\"data_in_block_size\":32},\"ffn\":{\"data_in_width\":$df,\"data_in_block_size\":32}}
          }"

        bfcl generate --model Qwen/Qwen3-8B-FC \
          --test-category web_search_base \
          --skip-server-setup \
          --result-dir "results/pa${pa}_pf${pf}_da${da}_df${df}"
      done
    done
  done
done
```

## How phase detection works

The server uses PyTorch forward pre-hooks to automatically detect prefill vs decode:

- **Prefill** (`seq_len > 1`): processing the full prompt in one pass
- **Decode** (`seq_len == 1`): autoregressive token-by-token generation

On every forward pass, hooks read the current phase and patch `module.config["data_in_width"]`
on each MX linear layer in-place before it runs. Weight quantization is unchanged --
only **activation** precision is swapped.

The `/v1/config` PUT endpoint updates the same `phase_configs` dict that the hooks
reference, so changes are picked up immediately on the next request.

## Prerequisites

```bash
uv pip install fastapi uvicorn httpx beautifulsoup4 markdownify ddgs
```

## CLI reference

```
python -m quant_eval.cli.quant_hf_serve --help
```
