# Local runtime environment for this migrated checkout.
# Usage:
#   source env.local.sh

export PLENA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VIRTUAL_ENV="$PLENA_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PLENA_ROOT/.conda/envs/plena-bfcl/bin:$PATH"

export BFCL_PROJECT_ROOT="$PLENA_ROOT/.bfcl"
export MPLCONFIGDIR="$PLENA_ROOT/.cache/matplotlib"

mkdir -p "$BFCL_PROJECT_ROOT" "$MPLCONFIGDIR" "$PLENA_ROOT/prefill_DSE/gptq_cache_qwen3"
