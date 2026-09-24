import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model, infer_auto_device_map
from colorlog import ColoredFormatter

# ---------------------------------------------------------------------------
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


def create_layer_balanced_device_map(config, n_devices: int) -> dict[str, int]:
    """Place contiguous decoder-layer ranges evenly across visible GPUs.

    Accelerate's automatic map is optimized for fitting one model copy.  A
    gpu-dual run later allocates an FP backup beside most decoder weights, and
    device-map-aware GPTQ needs comparable working-memory headroom on every
    device.  Keep one fewer decoder layer on the edge devices when possible
    because they also own the embedding and output modules.
    """
    if n_devices <= 0:
        raise ValueError("Layer-balanced model loading requires at least one visible CUDA device.")

    n_layers = int(getattr(config, "num_hidden_layers", 0))
    if n_layers <= 0:
        raise ValueError("Model config does not define a positive num_hidden_layers value.")

    base, remainder = divmod(n_layers, n_devices)
    layer_counts = [base] * n_devices
    # Prefer interior devices for remainder layers.  The first and last GPUs
    # additionally hold embeddings/rotary state and norm/lm_head respectively.
    remainder_order = list(range(1, max(1, n_devices - 1)))
    if n_devices > 1:
        remainder_order.extend((0, n_devices - 1))
    else:
        remainder_order = [0]
    for device in remainder_order[:remainder]:
        layer_counts[device] += 1

    device_map: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
    }
    layer_idx = 0
    for device, count in enumerate(layer_counts):
        for _ in range(count):
            device_map[f"model.layers.{layer_idx}"] = device
            layer_idx += 1
    device_map["model.norm"] = n_devices - 1
    device_map["lm_head"] = n_devices - 1
    return device_map


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def setup_model(
    model_name,
    model_parallel,
    dtype,
    device,
    attn_implementation="sdpa",
    layer_balanced_device_map: bool = False,
    cpu_staged: bool = False,
):
    logger = get_logger("setup")
    logger.info(
        f"Setting up model {model_name} with dtype {dtype}, device {device}, "
        f"attn_implementation={attn_implementation}"
    )

    if cpu_staged and model_parallel:
        raise ValueError("cpu_staged model loading is incompatible with model_parallel.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer setup complete")

    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }
    if cpu_staged:
        # Meta initialization plus shard-at-a-time loading avoids a second
        # full CPU state_dict while keeping the final model free of Accelerate
        # dispatch hooks. MASE will stage one decoder layer on the GPU itself.
        load_kwargs["low_cpu_mem_usage"] = True
        logger.info("Using low-memory CPU-staged model loading")
    elif model_parallel:
        device_map = "auto"
        if layer_balanced_device_map:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            device_map = create_layer_balanced_device_map(config, torch.cuda.device_count())
            layer_counts = [0] * torch.cuda.device_count()
            for name, mapped_device in device_map.items():
                if name.startswith("model.layers."):
                    layer_counts[mapped_device] += 1
            logger.info("Using layer-balanced device map: layers_per_device=%s", layer_counts)
        load_kwargs.update({
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        })
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if cpu_staged:
        non_cpu = [
            f"{name}={parameter.device}"
            for name, parameter in model.named_parameters()
            if parameter.device.type != "cpu"
        ]
        if non_cpu:
            raise RuntimeError(
                "CPU-staged model load produced non-CPU parameters: "
                + ", ".join(non_cpu[:8])
            )
    if model_parallel and hasattr(model, "hf_device_map"):
        print(f"Device map: {model.hf_device_map}")
    logger.info("Model setup complete")
    return tokenizer, model


def move_to_gpu(model, model_parallel=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return model
    if model_parallel:
        if hasattr(model, "hf_device_map"):
            return model
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
    # Relative paths are interpreted from CWD (matching shell convention),
    # so callers and surrounding bash tee logs land in the same tree.
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
