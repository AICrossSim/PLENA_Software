import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model

from quant_eval.utils import create_device_map, get_logger, set_logging_verbosity

logger = get_logger(__name__)


def create_experiment_log_dir(base_dir: str = "logs") -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    log_root = root_dir / base_dir
    timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d-%H%M%S")
    log_dir = log_root / f"run-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_link = log_root / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(log_dir, target_is_directory=True)

    return log_dir


def save_args(log_dir: Path, args: dict):
    def make_serializable(obj):
        if isinstance(obj, (Path, torch.dtype)):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return {k: make_serializable(v) for k, v in vars(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

    serializable_args = make_serializable(args)
    with open(log_dir / "args.json", "w") as f:
        json.dump(serializable_args, f, indent=2)


def save_results(log_dir: Path, results: dict):
    def make_serializable(obj):
        if isinstance(obj, (Path,)):
            return str(obj)
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

    results = make_serializable(results)

    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


def print_all_layers(model: nn.Module):
    print("=== Model Layers and Devices ===")
    for name, layer in model.named_modules():
        try:
            device = next(layer.parameters()).device
        except StopIteration:
            device = "No parameters"
        print(f"{name}: {type(layer).__name__} | device: {device}")
    print("====================")


def setup_model(model_name, model_parallel, dtype, device):
    logger.info(f"Setting up model {model_name} with dtype {dtype} and device {device}")
    if "meta" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer setup complete")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True
    )
    logger.info(f"Model setup complete")
    # Load on CPU first; caller handles device placement
    return tokenizer, model


def move_to_gpu(model, model_parallel=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return model
    if model_parallel:
        device_map = create_device_map(model, "auto-balanced")
        model = dispatch_model(model, device_map=device_map)
        logger.debug(f"device_map: {device_map}")
    else:
        model = model.to(device)
    return model
