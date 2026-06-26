# Runtime environment for the Qwen3-32B decode DSE.
#   source decode_dse_qwen/env.sh
#
# The home directory has a 100 GB quota that a 65 GB model blows past, so models, datasets, and
# caches all live on the shared /data volume. Source this before any download or run.

export PLENA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Hugging Face caches on /data (NOT home -- quota).
export HF_HUB_CACHE="/data/models"
export HF_DATASETS_CACHE="/data/models/datasets_cache"
export HF_HOME="/data/models/.hf_home"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

# Matplotlib + nltk caches off home as well.
export MPLCONFIGDIR="/data/models/.mplconfig"
export NLTK_DATA="/data/models/.nltk_data"
mkdir -p "$MPLCONFIGDIR" "$NLTK_DATA"

# Shared GPUs: reduce allocator fragmentation so the quantization spike has room.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export PATH="$PLENA_ROOT/.venv/bin:$PATH"
echo "[env] HF_HUB_CACHE=$HF_HUB_CACHE  GPUs=$(nvidia-smi -L 2>/dev/null | wc -l)  venv=$PLENA_ROOT/.venv"
