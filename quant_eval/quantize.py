"""
Load MASE quantization config from a single TOML file.

TOML structure (regex_name example):

    by = "regex_name"

    ["model\\.layers\\.\\d+\\.self_attn"]
    name = "mxint"

    ["model\\.layers\\.\\d+\\.self_attn".qk_matmul]
    data_in_block_size = 16
    data_in_width = 4
    ...
"""

import tomllib
from copy import deepcopy
from pathlib import Path


def load_quant_config(path: str | Path) -> dict:
    """Load a TOML quantization config and return pass_args for MASE.

    The TOML top-level key ``by`` selects the matching strategy
    ("type", "name", or "regex_name").  Every other top-level key is a
    module selector whose value is wrapped in ``{"config": ...}`` so it
    can be passed directly to ``quantize_module_transform_pass``.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    by = raw.pop("by", "regex_name")
    pass_args = {"by": by}

    # Top-level utility blocks pass through verbatim (NOT wrapped in
    # {"config": ...}); regex/name/type selectors do get wrapped below.
    for util_key in ("gptq", "token_collector", "rotation_search"):
        block = raw.pop(util_key, None)
        if block is not None:
            pass_args[util_key] = deepcopy(block)

    for key, value in raw.items():
        pass_args[key] = {"config": deepcopy(value)}

    return pass_args