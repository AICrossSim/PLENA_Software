import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    match level:
        case "debug":
            root_logger.setLevel(logging.DEBUG)
        case "info":
            root_logger.setLevel(logging.INFO)
        case "warning":
            root_logger.setLevel(logging.WARNING)
        case "error":
            root_logger.setLevel(logging.ERROR)
        case "critical":
            root_logger.setLevel(logging.CRITICAL)
        case _:
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

def setup_model(model_name, model_parallel, dtype, device):
    logger = get_logger("setup")
    logger.info(f"Setting up model {model_name} with dtype {dtype} and device {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer setup complete")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True
    )
    logger.info("Model setup complete")
    return tokenizer, model


def move_to_gpu(model, model_parallel=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return model
    if model_parallel:
        device_map = create_device_map(model, "auto-balanced")
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
    root_dir = Path(__file__).resolve().parent
    log_root = root_dir / base_dir
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
