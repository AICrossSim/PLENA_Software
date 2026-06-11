import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Literal

import torch
from torch import nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
)
from accelerate import dispatch_model, infer_auto_device_map
from colorlog import ColoredFormatter

# ---------------------------------------------------------------------------
# Global Compatibility Patches for Latest Transformers (>=4.43+)
# ---------------------------------------------------------------------------

# Patch 1: RoPE 'default' type compatibility
# Remote code for LLaDA and Fast-dLLM v2 expects rope_type='default'.
# Modern transformers sometimes removes 'default' from ROPE_INIT_FUNCTIONS.
# We inject it back, mapping it to 'linear' (which is standard RoPE).
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        # Find a suitable fallback function ('linear' is standard RoPE)
        fallback_key = next(
            (k for k in ROPE_INIT_FUNCTIONS if k in ["linear", "dynamic"]),
            next(iter(ROPE_INIT_FUNCTIONS), None),
        )
        if fallback_key:
            ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS[fallback_key]
except Exception:
    pass  # Ignore if transformers version doesn't use ROPE_INIT_FUNCTIONS

# Patch 2: Tied weights keys compatibility
# Newer transformers expects these attributes during meta-device loading,
# but older remote code doesn't define them.
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}
if not hasattr(PreTrainedModel, "_tied_weights_keys"):
    PreTrainedModel._tied_weights_keys = []
if not hasattr(PreTrainedModel, "tied_weights_keys"):
    PreTrainedModel.tied_weights_keys = []

# Patch 3: KV Cache subscriptability and mutability for older remote code
# Older remote code expects past_key_values to be a list of tuples: [(k0, v0), (k1, v1), ...]
# Modern transformers uses a DynamicCache object with either .key_cache/.value_cache lists
# or a .layers list of CacheLayer objects. We patch it to support list-like operations.
try:
    from transformers.cache_utils import DynamicCache

    def _dynamic_cache_getitem(self, idx):
        # For transformers 4.43 - 4.46
        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            return (self.key_cache[idx], self.value_cache[idx])
        # For transformers 4.47+
        elif hasattr(self, "layers"):
            layer = self.layers[idx]
            return (getattr(layer, "keys", None), getattr(layer, "values", None))
        raise AttributeError(
            f"Cannot find key/value cache in {self.__class__.__name__}"
        )

    def _dynamic_cache_len(self):
        if hasattr(self, "key_cache"):
            return len(self.key_cache)
        elif hasattr(self, "layers"):
            return len(self.layers)
        return 0

    def _dynamic_cache_setitem(self, idx, value):
        key_states, value_states = value
        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            if idx == len(self.key_cache):
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[idx] = key_states
                self.value_cache[idx] = value_states
        elif hasattr(self, "layers"):
            try:
                from transformers.cache_utils import DynamicLayer
            except ImportError:
                DynamicLayer = type("DynamicLayer", (), {})
            while len(self.layers) <= idx:
                self.layers.append(DynamicLayer())
            self.layers[idx].keys = key_states
            self.layers[idx].values = value_states
            self.layers[idx].is_initialized = True

    def _dynamic_cache_append(self, value):
        key_states, value_states = value
        if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif hasattr(self, "layers"):
            try:
                from transformers.cache_utils import DynamicLayer
            except ImportError:
                DynamicLayer = type("DynamicLayer", (), {})
            layer = DynamicLayer()
            layer.keys = key_states
            layer.values = value_states
            layer.is_initialized = True
            self.layers.append(layer)

    # Only patch if it hasn't been patched already
    original_getitem = getattr(DynamicCache, "__getitem__", None)
    if (
        original_getitem is None
        or getattr(original_getitem, "__name__", "") != "_dynamic_cache_getitem"
    ):
        DynamicCache.__getitem__ = _dynamic_cache_getitem
        DynamicCache.__len__ = _dynamic_cache_len
        DynamicCache.__setitem__ = _dynamic_cache_setitem
        DynamicCache.append = _dynamic_cache_append
except ImportError:
    pass
# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    style="%",
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

root_logger = logging.getLogger("quant_eval")
root_logger.addHandler(handler)
root_logger.propagate = False


def set_logging_verbosity(level: str = "info"):
    level = level.lower()
    if level == "debug":
        root_logger.setLevel(logging.DEBUG)
    elif level == "info":
        root_logger.setLevel(logging.INFO)
    elif level == "warning":
        root_logger.setLevel(logging.WARNING)
    elif level == "error":
        root_logger.setLevel(logging.ERROR)
    elif level == "critical":
        root_logger.setLevel(logging.CRITICAL)
    else:
        raise ValueError(
            f"Unknown logging level: {level}, should be one of: debug, info, warning, error, critical"
        )

    root_logger.info(f"Set logging level to {level}")


def get_logger(name: str):
    return root_logger.getChild(name)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def create_device_map(
    model: nn.Module,
    device_map: dict[str, int] | Literal["auto", "auto-balanced"],
) -> dict[str, int]:
    if device_map == "auto":
        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules
        )
    elif device_map == "auto-balanced":
        max_memory = {
            i: torch.cuda.mem_get_info(i)[0] // 4
            for i in range(torch.cuda.device_count())
        }
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory,
        )
        n_devices = torch.cuda.device_count()
        n_decoder_layers = model.config.num_hidden_layers
        n_layers_per_device = n_decoder_layers // n_devices
        balanced_device_map = {}
        current_device = 0
        current_decoder_idx = 0

        for layer_name in device_map:
            if ".layers." in layer_name:
                if (current_decoder_idx + 1) % n_layers_per_device == 0:
                    current_device += 1
                current_decoder_idx += 1
            balanced_device_map[layer_name] = min(current_device, n_devices - 1)
        device_map = balanced_device_map
    else:
        assert isinstance(device_map, dict)
    return device_map


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _ensure_rope_scaling(config, logger=None):
    """Normalize rope_scaling/rope_type for compatibility with modern transformers."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        rope_scaling = {}

    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")

    # Force to 'default' which we have safely patched into ROPE_INIT_FUNCTIONS above
    if rope_type in (None, "default", "original"):
        rope_scaling["rope_type"] = "default"
        rope_scaling["type"] = "default"
        rope_scaling.setdefault("factor", 1.0)

    config.rope_scaling = rope_scaling

    # Also patch direct rope_type attribute if the config exposes it
    if hasattr(config, "rope_type") and config.rope_type in (
        None,
        "default",
        "original",
    ):
        config.rope_type = "default"

    if logger is not None:
        logger.debug(f"Config rope_scaling after guard: {config.rope_scaling}")
        if hasattr(config, "rope_type"):
            logger.debug(f"Config rope_type after guard: {config.rope_type}")


def _fix_rotary_buffers(model, logger=None):
    """Recompute non-persistent RoPE ``inv_freq`` buffers after loading.

    ``inv_freq`` is registered with ``persistent=False``, so it is not stored
    in the checkpoint and is instead computed in ``__init__``. Under modern
    transformers' meta-device loading, ``__init__`` runs on the meta device and
    the buffer is materialized as uninitialized memory (garbage), producing
    NaN cos/sin and NaN logits. We detect a corrupt buffer and recompute it
    from the module's own ``rope_init_fn`` on the real device.
    """
    for name, module in model.named_modules():
        init_fn = getattr(module, "rope_init_fn", None)
        inv_freq = getattr(module, "inv_freq", None)
        if init_fn is None or inv_freq is None:
            continue
        # ``inv_freq`` is non-persistent and computed in ``__init__``; under
        # meta-device loading it is materialized as uninitialized memory, which
        # may surface as NaN/inf, huge values, OR near-zero values (the latter
        # silently slips past a "too large" heuristic). Recomputing from the
        # module's own ``rope_init_fn`` is cheap and authoritative, so we always
        # recompute rather than trying to detect every flavor of garbage.
        if inv_freq.device.type == "meta":
            device = next(
                (p.device for p in model.parameters() if p.device.type != "meta"),
                torch.device("cpu"),
            )
        else:
            device = inv_freq.device
        new_inv_freq, attn_scaling = init_fn(module.config, device=device)
        new_inv_freq = new_inv_freq.to(device=device, dtype=inv_freq.dtype)
        module.inv_freq = new_inv_freq
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = new_inv_freq
        if attn_scaling is not None:
            module.attention_scaling = attn_scaling
        if logger is not None:
            logger.info(
                f"Recomputed RoPE inv_freq for '{name}' from rope_init_fn "
                f"(guards against meta-device init garbage)"
            )


def setup_model(model_name, model_parallel, dtype, device, attn_implementation="sdpa"):
    logger = get_logger("setup")
    logger.info(
        f"Setting up model {model_name} with dtype {dtype}, device {device}, "
        f"attn_implementation={attn_implementation}"
    )

    # Load config first so we can patch RoPE before remote code initializes
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _ensure_rope_scaling(config, logger)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer setup complete")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    _fix_rotary_buffers(model, logger)
    logger.info("Model setup complete")
    return tokenizer, model


def move_to_gpu(model, model_parallel=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return model
    if model_parallel:
        device_map = create_device_map(model, "auto-balanced")
        print(f"Device map: {device_map}")
        model = dispatch_model(model, device_map=device_map)
    else:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Logging / experiment tracking
# ---------------------------------------------------------------------------


def print_all_layers(model: nn.Module):
    print("=== Model Layers and Devices ===")
    for name, layer in model.named_modules():
        try:
            device = next(layer.parameters()).device
        except StopIteration:
            device = "No parameters"
        print(f"{name}: {type(layer).__name__} | device: {device}")
    print("====================")


def create_experiment_log_dir(base_dir: str = "logs") -> Path:
    log_root = Path(base_dir)
    timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d-%H%M%S")
    log_dir = log_root / f"run-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_link = log_root / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(log_dir, target_is_directory=True)

    return log_dir


def _make_serializable(obj):
    if isinstance(obj, (Path, torch.dtype)):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)


def save_args(log_dir: Path, args: dict):
    with open(log_dir / "args.json", "w") as f:
        json.dump(_make_serializable(args), f, indent=2)


def save_results(log_dir: Path, results: dict):
    with open(log_dir / "results.json", "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
