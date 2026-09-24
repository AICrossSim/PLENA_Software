"""
Phase-dependent quantization support.

Provides automatic prefill/decode detection for disaggregated inference
where each phase uses different MX activation precision. The weight
precision is set once at load time; only activation (data_in) precision
is swapped dynamically.

Two hooks are provided:

    PhaseAutoSwitch       — original, phase-only (prefill vs decode).
                            Single hook on the top-level model.
                            All MX layers share one config per phase.

    PhaseLayerAutoSwitch  — extends the above with per-layer-type granularity.
                            Separate configs for attention vs FFN layers,
                            independently per phase (4 configs total).
                            Hooks are registered on each named submodule so
                            the layer type is known at dispatch time.

Config schema
─────────────
PhaseAutoSwitch (unchanged):
    {
        "prefill": {"data_in_width": 4,  "data_in_block_size": 32},
        "decode":  {"data_in_width": 8,  "data_in_block_size": 32},
    }

PhaseLayerAutoSwitch (new):
    {
        "prefill": {
            "attn": {"data_in_width": 4,  "data_in_block_size": 32},
            "ffn":  {"data_in_width": 4,  "data_in_block_size": 32},
        },
        "decode": {
            "attn": {"data_in_width": 8,  "data_in_block_size": 32},
            "ffn":  {"data_in_width": 6,  "data_in_block_size": 32},
        },
    }

Any (phase, layer_type) pair that is absent from the config is left at
whatever value was set at model-load time — no silent reset to defaults.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from functools import partial
import logging
import os
from pathlib import Path
import time

import torch
from torch import nn

from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_mx_layers(model: nn.Module):
    """Return (name, module) for every Linear MX wrapper in the model."""
    from chop.nn.quantized.modules.linear import LinearMXInt, LinearMXFP
    from quant_eval.eval.unified_mx import LinearMXUnified

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (LinearMXInt, LinearMXFP, LinearMXUnified)):
            layers.append((name, module))
    return layers


# ---------------------------------------------------------------------------
# Quantized attention wrapper support
# ---------------------------------------------------------------------------
#
# Quantized eager-attention wrappers (LlamaAttentionMXInt, Qwen3AttentionMXInt,
# Glm4MoeAttentionMXInt, ...) store their MX quant configs as separate dict
# attributes (qk_config, av_config, kv_cache_config, softmax_config,
# rope_config), not on a single `.config` like LinearMXInt does. These dicts
# are read dynamically inside the forward's call to mxint_quantizer, so
# mutating them at runtime propagates immediately.

def _find_quant_attention_wrappers(model: nn.Module):
    """Return (name, module) for every quantized eager-attention wrapper.

    Detected by duck-typing: any module that has both ``qk_config`` and
    ``av_config`` attributes (covers LlamaAttentionMXInt, Qwen3AttentionMXInt,
    Qwen3MoeAttentionMXInt, Glm4MoeAttentionMXInt, GptOssAttentionMXInt, etc.).
    """
    wrappers = []
    for name, module in model.named_modules():
        if hasattr(module, "qk_config") and hasattr(module, "av_config"):
            wrappers.append((name, module))
    return wrappers


_ATTN_CONFIG_ATTRS = (
    "qk_config",
    "av_config",
    "kv_cache_config",
    "softmax_config",
    "rope_config",
)
_ATTN_BYPASS_ATTRS = (
    "qk_bypass",
    "av_bypass",
    "kv_cache_bypass",
    "softmax_bypass",
    "rope_bypass",
)

_VALID_WEIGHT_MODES = {"quantized", "fp"}
_VALID_WEIGHT_RESIDENCIES = {"disk_reload", "host_cached", "gpu_dual"}


def _set_attention_bypass_attrs(wrapper: nn.Module, bypass: bool) -> None:
    """Set all attention wrapper bypass attrs to the same value."""
    for attr_name in _ATTN_BYPASS_ATTRS:
        if hasattr(wrapper, attr_name):
            setattr(wrapper, attr_name, bool(bypass))


def _sync_attention_bypass_attrs(wrapper: nn.Module) -> None:
    """Synchronise each bypass attr from its own config dict.

    This preserves legacy configs where only softmax/rope are bypassed, while
    still allowing a top-level phase ``bypass=True`` to force all attention
    internals down the FP path.
    """
    config_attr_by_bypass = {
        "qk_bypass": "qk_config",
        "av_bypass": "av_config",
        "kv_cache_bypass": "kv_cache_config",
        "softmax_bypass": "softmax_config",
        "rope_bypass": "rope_config",
    }
    for bypass_attr, config_attr in config_attr_by_bypass.items():
        if not hasattr(wrapper, bypass_attr):
            continue
        cfg = getattr(wrapper, config_attr, None)
        setattr(wrapper, bypass_attr, bool(cfg.get("bypass", False)) if isinstance(cfg, dict) else False)


def _normalize_weight_mode(overrides: dict | None) -> str:
    """Return the requested per-phase weight mode. Defaults to quantized."""
    if not overrides:
        return "quantized"
    mode = overrides.get("weight_mode", "quantized")
    if mode not in _VALID_WEIGHT_MODES:
        raise ValueError(
            f"Unsupported weight_mode={mode!r}; expected one of "
            f"{sorted(_VALID_WEIGHT_MODES)}."
        )
    return mode


def _apply_module_config(module: nn.Module, overrides: dict) -> None:
    """Write runtime config keys to one MX layer.

    ``weight_mode`` is switch policy, not a LinearMXInt/LinearMXFP config key.
    ``bypass`` is both a config value and a module attribute read directly by
    LinearMXInt.forward(), so keep them in sync.
    """
    desired_bypass = bool(overrides.get("linear_bypass", overrides.get("bypass", False)))
    module.config["bypass"] = desired_bypass
    if hasattr(module, "bypass"):
        module.bypass = desired_bypass

    if "data_in_width" in overrides:
        module.config.pop("data_in_exponent_width", None)
        module.config.pop("data_in_frac_width", None)
    if "data_in_exponent_width" in overrides or "data_in_frac_width" in overrides:
        module.config.pop("data_in_width", None)

    for key, value in overrides.items():
        if key in ("weight_mode", "linear_bypass", "kv_cache", "softmax", "rope", "qk_matmul", "av_matmul"):
            continue
        if key == "bypass":
            continue
        module.config[key] = value


def _apply_config(mx_layers: list, overrides: dict) -> None:
    """Write ``overrides`` into each MX layer's runtime config."""
    for _, module in mx_layers:
        _apply_module_config(module, overrides)


def set_phase(model: nn.Module, phase_configs: dict, phase: str) -> None:
    """
    Imperatively set all MX layers to the config for ``phase``.

    Compatible with PhaseAutoSwitch-style flat configs only.

    Args:
        model:         The quantized model.
        phase_configs: {"prefill": {...}, "decode": {...}}
        phase:         "prefill" or "decode"
    """
    mx_layers = _find_mx_layers(model)
    overrides = phase_configs.get(phase, {})
    _apply_config(mx_layers, overrides)




# ---------------------------------------------------------------------------
# Quantized nonlinear wrapper support
# ---------------------------------------------------------------------------

def _find_quant_mlp_wrappers(model: nn.Module):
    """Return Llama/Qwen MLP wrappers with minifloat SiLU q_config."""
    wrappers = []
    for name, module in model.named_modules():
        if hasattr(module, "q_config") and module.__class__.__name__.endswith((
            "MLPMXInt",
            "MLPMXFP",
            "MLPMXUnified",
            "ExpertsMXUnified",
        )):
            wrappers.append((name, module))
    return wrappers


def _find_minifloat_rmsnorm_wrappers(model: nn.Module):
    """Return Llama/Qwen RMSNorm minifloat wrappers."""
    wrappers = []
    for name, module in model.named_modules():
        if hasattr(module, "q_config") and module.__class__.__name__.endswith(("RMSNormMinifloat", "RMSNormMinifloatUnified")):
            wrappers.append((name, module))
    return wrappers


def _rebuild_rmsnorm_minifloat_quantizers(module: nn.Module) -> None:
    """Recreate RMSNorm minifloat quantizer closures after q_config changes.

    LlamaRMSNormMinifloat builds x_quantizer/w_quantizer in __init__, so phase
    switching cannot just mutate q_config.  This mirrors that init logic.
    """
    cfg = getattr(module, "q_config", {}) or {}
    module.bypass = bool(cfg.get("bypass", False))
    module.weight_bypass = bool(cfg.get("weight_bypass", False))
    module.data_in_bypass = bool(cfg.get("data_in_bypass", False))

    if not module.bypass and not module.weight_bypass:
        module.w_quantizer = partial(
            minifloat_quantizer_sim,
            minifloat_meta=MinifloatMeta(
                exp_bits=cfg["weight_exponent_width"],
                frac_bits=cfg["weight_frac_width"],
                is_finite=cfg.get("weight_is_finite", True),
                round_mode=cfg.get("weight_round_mode", "rn"),
            ),
        )
    else:
        module.w_quantizer = None

    if not module.bypass and not module.data_in_bypass:
        module.x_quantizer = partial(
            minifloat_quantizer_sim,
            minifloat_meta=MinifloatMeta(
                exp_bits=cfg["data_in_exponent_width"],
                frac_bits=cfg["data_in_frac_width"],
                is_finite=cfg.get("data_in_is_finite", True),
                round_mode=cfg.get("data_in_round_mode", "rn"),
            ),
        )
    else:
        module.x_quantizer = None


def _apply_nonlinear_config(module: nn.Module, overrides: dict) -> None:
    if not overrides:
        return
    if not hasattr(module, "q_config"):
        return
    module.q_config.pop("bypass", None)
    module.q_config.update({k: v for k, v in overrides.items() if k != "weight_mode"})
    if hasattr(module, "bypass"):
        module.bypass = bool(module.q_config.get("bypass", False))
    if module.__class__.__name__.endswith(("RMSNormMinifloat", "RMSNormMinifloatUnified")):
        _rebuild_rmsnorm_minifloat_quantizers(module)


def _apply_expert_activation_config(module: nn.Module, overrides: dict) -> None:
    """Update routed expert ACT precision without touching FP_SETTING."""
    config = getattr(module, "act_config", None)
    if not isinstance(config, dict) or not overrides:
        return
    config.pop("bypass", None)
    config.update({k: v for k, v in overrides.items() if k != "weight_mode"})

# ---------------------------------------------------------------------------
# Layer-type classification helpers
# ---------------------------------------------------------------------------

# Substrings matched (case-insensitive) against the full dotted module name.
# Covers Llama, Qwen, Mistral, Falcon, Phi, GPT-NeoX naming conventions.
_DEFAULT_ATTN_KEYWORDS: tuple[str, ...] = (
    "attn", "attention", "self_attn", "cross_attn",
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv",
)
_DEFAULT_FFN_KEYWORDS: tuple[str, ...] = (
    "mlp", "ffn", "feed_forward",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2", "intermediate", "output.dense",
)


def _classify_module(
    name: str,
    attn_keywords: tuple[str, ...],
    ffn_keywords: tuple[str, ...],
) -> str | None:
    """Return 'attn', 'ffn', or None (unclassified)."""
    lname = name.lower()
    if any(k in lname for k in attn_keywords):
        return "attn"
    if any(k in lname for k in ffn_keywords):
        return "ffn"
    return None


# ---------------------------------------------------------------------------
# Original hook — preserved exactly
# ---------------------------------------------------------------------------

class PhaseAutoSwitch:
    """
    Automatic phase-dependent quantization hook.

    Registers a forward pre-hook on the model that detects prefill vs decode
    by checking input sequence length, and swaps MX layer configs accordingly.

    Prefill (seq_len > 1): prompt processing, batched forward pass.
    Decode  (seq_len == 1): autoregressive generation, token-by-token.

    This makes phase-dependent quantization transparent to any evaluation
    framework (lm-eval, generation, etc.). For pure-prefill tasks (PPL,
    log-likelihood), the hook stays in prefill mode throughout — use
    eval_ppl.py directly for those.

    Usage:
        switch = PhaseAutoSwitch(model, phase_configs)
        switch.enable()
        model(long_input)   # seq_len > 1 -> prefill config
        model(single_token, past_key_values=kv) # seq_len == 1 -> decode config
        switch.disable()  # restore original configs
    """

    def __init__(self, model: nn.Module, phase_configs: dict, threshold: int = 1):
        """
        Args:
            model:         Quantized model with LinearMXInt layers.
            phase_configs: {"prefill": {"data_in_width": 4}, "decode": {"data_in_width": 8}}
            threshold:     Sequence length threshold. seq_len > threshold -> prefill, else decode.
        """
        self.model = model
        self.phase_configs = phase_configs
        self.threshold = threshold
        self.mx_layers = _find_mx_layers(model)
        self._hook_handle = None
        self._original_configs = {}
        self.current_phase = None

        # Save original configs
        for name, module in self.mx_layers:
            self._original_configs[name] = dict(module.config)

    def _hook_fn(self, module, args, kwargs):
        """Forward pre-hook that detects phase from input shape."""
        input_ids = None
        if args:
            input_ids = args[0]
        elif "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        elif "inputs_embeds" in kwargs:
            input_ids = kwargs["inputs_embeds"]

        if input_ids is None:
            return

        seq_len = input_ids.shape[1] if input_ids.dim() >= 2 else 1
        phase = "prefill" if seq_len > self.threshold else "decode"

        if phase != self.current_phase:
            self.current_phase = phase
            overrides = self.phase_configs.get(phase, {})
            for _, mx_module in self.mx_layers:
                for key, value in overrides.items():
                    if key in mx_module.config:
                        mx_module.config[key] = value

    def enable(self):
        """Register the auto-switch hook."""
        self._hook_handle = self.model.register_forward_pre_hook(
            self._hook_fn, with_kwargs=True
        )
        set_phase(self.model, self.phase_configs, "prefill")
        self.current_phase = "prefill"
        return self

    def disable(self):
        """Remove the hook and restore original configs."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        for name, module in self.mx_layers:
            original = self._original_configs.get(name, {})
            for key, value in original.items():
                module.config[key] = value
        self.current_phase = None

    def __enter__(self):
        return self.enable()

    def __exit__(self, *args):
        self.disable()


# ---------------------------------------------------------------------------
# New hook — phase × layer-type disaggregated quantization
# ---------------------------------------------------------------------------

class PhaseLayerAutoSwitch:
    """
    Disaggregated quantization hook with independent configs for every
    (phase, layer_type) pair:

        prefill × attn   prefill × ffn
        decode  × attn   decode  × ffn

    Architecture
    ────────────
    A top-level hook detects phase changes from the input sequence length.
    Per-submodule hooks then apply the active attention/FFN configuration to
    the MX layers owned by that submodule.

    Layer-type classification
    ─────────────────────────
    Every named module is classified once at ``__init__`` by matching its
    full dotted name against ``attn_keywords`` / ``ffn_keywords``.
    Modules that match neither (embeddings, layer norms, etc.) get no hook
    and are never touched.

    Config schema
    ─────────────
    {
        "prefill": {
            "attn": {"data_in_width": 4,  "data_in_block_size": 32},
            "ffn":  {"data_in_width": 4,  "data_in_block_size": 32},
        },
        "decode": {
            "attn": {
                "data_in_width": 8,
                "data_in_block_size": 32,
                "weight_mode": "fp",  # optional: "quantized" (default) or "fp"
                "bypass": True,       # required for true FP Linear decode
            },
            "ffn":  {"data_in_width": 6,  "data_in_block_size": 32},
        },
    }

    Any missing (phase, layer_type) pair is silently skipped — the layer
    keeps whatever config was set at quantization time.

    Usage:
        phase_configs = {
            "prefill": {
                "attn": {"data_in_width": 4,  "data_in_block_size": 32},
                "ffn":  {"data_in_width": 4,  "data_in_block_size": 32},
            },
            "decode": {
                "attn": {"data_in_width": 8,  "data_in_block_size": 32},
                "ffn":  {"data_in_width": 6,  "data_in_block_size": 32},
            },
        }

        switch = PhaseLayerAutoSwitch(model, phase_configs)
        switch.enable()
        # ... run lm-eval, generation, etc. ...
        switch.disable()

    Context-manager form:
        with PhaseLayerAutoSwitch(model, phase_configs):
            results = evaluate_with_lm_eval(...)
    """

    def __init__(
        self,
        model: nn.Module,
        phase_configs: dict[str, dict[str, dict]],
        threshold: int = 1,
        attn_keywords: tuple[str, ...] = _DEFAULT_ATTN_KEYWORDS,
        ffn_keywords:  tuple[str, ...] = _DEFAULT_FFN_KEYWORDS,
        model_name: str | None = None,
        weight_residency: str = "disk_reload",
        quant_checkpoint_dir: str | None = None,
    ):
        """
        Args:
            model:         Quantized model with LinearMXInt / LinearMXFP layers.
            phase_configs: Nested config dict — see class docstring.
            threshold:     seq_len threshold: > threshold → prefill, else decode.
            attn_keywords: Name substrings that identify attention modules.
            ffn_keywords:  Name substrings that identify FFN modules.
            model_name:    HF model ID or local path. Required for the disk-backed
                           weight re-quant path on phase transition. If omitted,
                           weight width stays fixed at whatever the load-time
                           quant pass produced; activation / attention /
                           KV-cache config mutation still works.
            weight_residency: ``disk_reload`` keeps one active GPU weight set;
                           when a GPTQ layer cache is available it automatically
                           uses the retained mmap/page-cache I/O fast path,
                           otherwise files are reopened for each switch.
                           ``host_cached`` explicitly requires that fast path;
                           ``gpu_dual`` retains both weight sets on GPU.
            quant_checkpoint_dir: Per-layer GPTQ cache used to restore prefill
                           weights after FP decode. Required by ``host_cached``.
        """
        self.model = model
        self.phase_configs = phase_configs
        self.threshold = threshold
        self.attn_keywords = attn_keywords
        self.ffn_keywords  = ffn_keywords
        self.model_name = model_name
        self.weight_residency = str(weight_residency or "disk_reload").lower()
        self.quant_checkpoint_dir = (
            Path(quant_checkpoint_dir).expanduser().resolve()
            if quant_checkpoint_dir
            else None
        )
        if self.weight_residency not in _VALID_WEIGHT_RESIDENCIES:
            raise ValueError(
                f"Unsupported weight_residency={weight_residency!r}; expected one of "
                f"{sorted(_VALID_WEIGHT_RESIDENCIES)}."
            )
        self._host_handle_stack = ExitStack()
        self._host_safetensor_handles: dict[str, object] = {}
        self._fp_weight_requested = self._phase_configs_request_fp_weight()
        if self._fp_weight_requested and model_name is None:
            raise ValueError(
                "PhaseLayerAutoSwitch requires model_name when any phase "
                "config sets weight_mode='fp'."
            )
        if (
            self.weight_residency in {"host_cached", "gpu_dual"}
            and not self._fp_weight_requested
        ):
            raise ValueError(
                f"weight_residency={self.weight_residency!r} requires at least "
                "one phase with weight_mode='fp'."
            )

        # Shared mutable cell updated by the top-level phase-detection hook and
        # read by each per-submodule hook. A one-element list lets closures
        # mutate it without ``nonlocal``.
        self._phase: list[str] = ["prefill"]

        self._hook_handles: list = []

        # Save original MX configs for clean restore on disable().
        self._all_mx_layers = _find_mx_layers(model)
        self._original_configs: dict[str, dict] = {
            name: dict(module.config)
            for name, module in self._all_mx_layers
        }

        # For prefill-only mode, decode may temporarily replace quantized/GPTQ
        # weights with original FP checkpoint weights. Cache the quantized
        # tensors on CPU so a later decode->prefill transition can restore the
        # exact runtime state without re-running GPTQ.
        self._quant_weight_cache: dict[int, dict[str, object]] = {}
        self._fp_weight_active: set[int] = set()
        self._fp_expert_active: set[int] = set()

        # id(mx_module) -> full dotted name, for safetensors tensor-key lookup
        # during the disk-backed weight re-quant path.
        self._mx_name_by_id: dict[int, str] = {
            id(module): name for name, module in self._all_mx_layers
        }

        # Collect quantized attention/nonlinear wrappers by duck-typing.
        # These are mutated on phase transition alongside the LinearMX layers.
        self.attn_wrappers = _find_quant_attention_wrappers(model)
        self.mlp_wrappers = _find_quant_mlp_wrappers(model)
        self.rmsnorm_wrappers = _find_minifloat_rmsnorm_wrappers(model)
        self.expert_weight_wrappers = [
            (name, module)
            for name, module in self.mlp_wrappers
            if all(
                hasattr(module, attr)
                for attr in (
                    "gate_up_proj",
                    "down_proj",
                    "allocate_fp_weight_backup",
                    "copy_fp_expert_tensor",
                    "set_use_fp_weight",
                )
            )
        ]
        self._fp_expert_weight_requested = bool(self.expert_weight_wrappers) and any(
            _normalize_weight_mode((by_layer or {}).get("ffn") or {}) == "fp"
            for by_layer in self.phase_configs.values()
            if isinstance(by_layer, dict)
        )
        if (
            self._fp_expert_weight_requested
            and self.weight_residency in {"disk_reload", "host_cached"}
            and self.quant_checkpoint_dir is None
        ):
            raise ValueError(
                "Qwen3-MoE FP decode with disk/host-backed residency requires "
                "quant_checkpoint_dir so packed GPTQ expert tensors can be restored."
            )
        if self.quant_checkpoint_dir is not None and not self.quant_checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"quant_checkpoint_dir does not exist: {self.quant_checkpoint_dir}"
            )
        if self.weight_residency == "host_cached" and self.quant_checkpoint_dir is None:
            raise ValueError(
                "weight_residency='host_cached' requires quant_checkpoint_dir; "
                "otherwise restoring prefill weights would require a full anonymous CPU copy."
            )
        # A complete GPTQ layer cache lets disk_reload use the same efficient
        # file-backed implementation as host_cached without changing its GPU
        # memory contract: only the currently active weight set remains on GPU.
        # Evaluators pass this path only after cache preparation succeeds.
        self._retained_file_cache_enabled = bool(
            self._fp_weight_requested
            and self.weight_residency in {"disk_reload", "host_cached"}
            and self.quant_checkpoint_dir is not None
        )
        self._original_attn_configs: dict[str, dict[str, object]] = {}
        for name, wrapper in self.attn_wrappers:
            snap: dict[str, object] = {"bypass_attrs": {}}
            for attr_name in _ATTN_CONFIG_ATTRS:
                target = getattr(wrapper, attr_name, None)
                if isinstance(target, dict):
                    snap[attr_name] = dict(target)
            for attr_name in _ATTN_BYPASS_ATTRS:
                if hasattr(wrapper, attr_name):
                    snap["bypass_attrs"][attr_name] = bool(getattr(wrapper, attr_name))
            self._original_attn_configs[name] = snap

        self._original_mlp_configs: dict[str, dict] = {
            name: dict(getattr(module, "q_config", {}) or {})
            for name, module in self.mlp_wrappers
        }
        self._original_mlp_act_configs: dict[str, dict] = {
            name: dict(getattr(module, "act_config", {}) or {})
            for name, module in self.mlp_wrappers
            if hasattr(module, "act_config")
        }
        self._original_rmsnorm_configs: dict[str, dict] = {
            name: dict(getattr(module, "q_config", {}) or {})
            for name, module in self.rmsnorm_wrappers
        }

        # Pre-classify every module and collect its owned MX layers.
        # "Owned" = the MX layers that are direct or nested children of
        # that module but whose name starts with this module's name prefix.
        self._submodule_info: dict[int, dict] = {}  # id(module) -> info dict
        self._build_submodule_index()

        # Resolve tensor-name → safetensors shard file path, for Stage 3
        # weight re-quant. Only built when model_name is provided.
        self._shard_map: dict[str, str] | None = None
        if model_name is not None:
            self._shard_map = self._build_shard_map(model_name)
        if self._fp_weight_requested and self._shard_map is None:
            raise ValueError(
                f"Could not locate HF safetensors for model_name={model_name!r}; "
                "weight_mode='fp' cannot load original decode weights."
            )
        if self._retained_file_cache_enabled:
            if self.weight_residency == "disk_reload":
                logger.info(
                    "Auto-enabling retained mmap/page-cache fast path for "
                    "disk_reload because GPTQ cache is available: %s",
                    self.quant_checkpoint_dir,
                )
            self._prepare_host_cached_sources()
        elif self._fp_weight_requested and self.weight_residency == "gpu_dual":
            self._materialize_gpu_dual_fp_weights()

    def _phase_requests_fp_weight(self, phase: str) -> bool:
        """Return whether a specific phase requests original FP weights."""
        by_layer = self.phase_configs.get(phase, {})
        if not isinstance(by_layer, dict):
            return False
        return any(
            _normalize_weight_mode(overrides or {}) == "fp"
            for overrides in by_layer.values()
        )

    def _phase_configs_request_fp_weight(self) -> bool:
        """Validate all weight modes and report whether any phase requests FP."""
        request_fp = False
        for phase, by_layer in self.phase_configs.items():
            if not isinstance(by_layer, dict):
                continue
            for layer_type, overrides in by_layer.items():
                mode = _normalize_weight_mode(overrides or {})
                if mode == "fp":
                    request_fp = True
        return request_fp

    def _build_shard_map(self, model_name: str) -> dict[str, str] | None:
        """Locate the HF safetensors file(s) for ``model_name`` in the local
        cache and return a dict mapping tensor name → absolute shard path.

        Handles both sharded checkpoints (via ``model.safetensors.index.json``)
        and single-file checkpoints. Returns ``None`` and logs a warning if
        neither variant is locatable."""
        import json
        import logging
        import os

        try:
            from transformers.utils.hub import cached_file
        except ImportError:
            logging.getLogger(__name__).warning(
                "transformers.utils.hub not available; disk weight re-quant disabled."
            )
            return None

        # ── Try sharded first (most modern HF models) ────────────────
        try:
            index_path = cached_file(model_name, "model.safetensors.index.json")
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            index_dir = os.path.dirname(index_path)
            return {
                name: os.path.join(index_dir, shard)
                for name, shard in weight_map.items()
            }
        except (OSError, KeyError):
            pass

        # ── Fall back to single-file ──────────────────────────────────
        try:
            single = cached_file(model_name, "model.safetensors")
            from safetensors import safe_open
            with safe_open(single, framework="pt") as f:
                return {name: single for name in f.keys()}
        except OSError:
            logging.getLogger(__name__).warning(
                "Could not locate safetensors for %s in HF cache; "
                "disk weight re-quant path disabled.", model_name,
            )
            return None

    def _open_safetensor(self, path: str | Path):
        """Open a safetensors source, optionally retaining its mmap."""
        from safetensors import safe_open

        resolved = str(Path(path).expanduser().resolve())
        if not self._retained_file_cache_enabled:
            return safe_open(resolved, framework="pt")
        handle = self._host_safetensor_handles.get(resolved)
        if handle is None:
            handle = self._host_handle_stack.enter_context(
                safe_open(resolved, framework="pt")
            )
            self._host_safetensor_handles[resolved] = handle
        return nullcontext(handle)

    @staticmethod
    def _advise_page_cache(path: Path) -> None:
        """Ask Linux to retain/read the file through its reclaimable page cache."""
        advice = getattr(os, "POSIX_FADV_WILLNEED", None)
        if advice is None or not hasattr(os, "posix_fadvise"):
            return
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, advice)
        except OSError as exc:
            logger.debug("Page-cache advice failed for %s: %s", path, exc)
        finally:
            if fd is not None:
                os.close(fd)

    def _prepare_host_cached_sources(self) -> None:
        """Retain checkpoint mmaps and issue non-blocking Linux cache hints.

        This deliberately does not clone tensors into anonymous CPU memory.
        The mapped pages remain reclaimable by Linux under memory pressure.
        """
        started = time.perf_counter()
        paths = {
            Path(path).expanduser().resolve()
            for path in (self._shard_map or {}).values()
        }
        if self.quant_checkpoint_dir is not None:
            paths.update(
                path.resolve()
                for path in self.quant_checkpoint_dir.glob(
                    "quantized_model_layer_*.safetensors"
                )
            )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"host_cached source does not exist: {missing[0]}"
            )

        apparent_bytes = 0
        try:
            for path in sorted(paths):
                apparent_bytes += path.stat().st_size
                self._advise_page_cache(path)
                with self._open_safetensor(path):
                    pass
        except Exception:
            self._close_host_cached_sources()
            raise
        logger.info(
            "Prepared retained mmap/page-cache fast path for %d safetensors files "
            "(%.1f GiB apparent) in %.1fs; requested residency=%s",
            len(paths),
            apparent_bytes / (1024 ** 3),
            time.perf_counter() - started,
            self.weight_residency,
        )

    @staticmethod
    def _copy_job_bytes(job: tuple[torch.Tensor, torch.Tensor]) -> int:
        target, _ = job
        return target.numel() * target.element_size()

    def _copy_host_cached_jobs(
        self,
        jobs: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        label: str,
    ) -> None:
        """Copy mmap-backed tensors concurrently, with one stream per device.

        Jobs targeting the same GPU remain ordered. Different GPUs copy in
        parallel so a sharded model can use all available host-to-device links.
        """
        if not jobs:
            return
        by_device: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for target, source in jobs:
            by_device.setdefault(str(target.device), []).append((target, source))

        def _copy_device_group(
            group: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> None:
            device = group[0][0].device
            if device.type == "cuda":
                torch.cuda.set_device(device)
            with torch.no_grad():
                for target, source in group:
                    target.copy_(source, non_blocking=False)

        started = time.perf_counter()
        if len(by_device) == 1:
            _copy_device_group(next(iter(by_device.values())))
        else:
            with ThreadPoolExecutor(
                max_workers=len(by_device),
                thread_name_prefix="plena-host-copy",
            ) as executor:
                futures = [
                    executor.submit(_copy_device_group, group)
                    for group in by_device.values()
                ]
                for future in futures:
                    future.result()
        total_bytes = sum(self._copy_job_bytes(job) for job in jobs)
        logger.info(
            "retained file cache %s copied %.1f GiB across %d device(s) in %.1fs",
            label,
            total_bytes / (1024 ** 3),
            len(by_device),
            time.perf_counter() - started,
        )

    def _close_host_cached_sources(self) -> None:
        if not self._host_safetensor_handles:
            return
        count = len(self._host_safetensor_handles)
        self._host_safetensor_handles.clear()
        self._host_handle_stack.close()
        self._host_handle_stack = ExitStack()
        logger.info("Closed %d retained safetensors mappings", count)

    # ------------------------------------------------------------------
    # Index construction (called once at __init__)
    # ------------------------------------------------------------------

    def _build_submodule_index(self) -> None:
        """
        For each classified submodule, record:
          - layer_type: 'attn' or 'ffn'
          - owned_mx:   list of MX layer modules whose names are prefixed
                        by this submodule's name.

        We walk named_modules() once and bucket MX layers by their
        classified parent.  If an MX layer's name contains multiple
        classified prefixes (unusual but possible in custom architectures),
        it is assigned to the most specific (longest) matching parent.
        """
        # Build a sorted list of (name, layer_type) for all classified modules.
        classified: list[tuple[str, str]] = []
        for name, module in self.model.named_modules():
            layer_type = _classify_module(name, self.attn_keywords, self.ffn_keywords)
            if layer_type is not None:
                classified.append((name, layer_type))
                # Register in index; owned_mx filled below.
                self._submodule_info[id(module)] = {
                    "name":       name,
                    "layer_type": layer_type,
                    "owned_mx":   [],
                    "module":     module,
                }

        if not classified:
            return

        # Sort by name length descending so the most-specific parent wins.
        classified.sort(key=lambda t: len(t[0]), reverse=True)
        # Assign each MX layer to its most-specific classified parent.
        for mx_name, mx_module in self._all_mx_layers:
            for parent_name, layer_type in classified:
                # An MX layer belongs to a parent if its name starts with
                # the parent's name followed by '.' (or equals it exactly).
                if mx_name == parent_name or mx_name.startswith(parent_name + "."):
                    # Find the module object for parent_name.
                    for mod_id, info in self._submodule_info.items():
                        if info["name"] == parent_name:
                            info["owned_mx"].append(mx_module)
                            break
                    break  # most-specific parent found; stop searching

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_phase_detection_hook(self):
        """Top-level hook: updates self._phase from input seq_len and, on a
        real phase transition, fires ``_on_phase_transition`` which handles
        attention sub-config mutation, disk-backed weight re-quant, and
        in-place KV-cache re-quant."""
        phase_cell = self._phase

        def hook(module, args, kwargs):
            input_ids = None
            if args:
                input_ids = args[0]
            elif "input_ids" in kwargs:
                input_ids = kwargs["input_ids"]
            elif "inputs_embeds" in kwargs:
                input_ids = kwargs["inputs_embeds"]

            if input_ids is None:
                return

            seq_len = input_ids.shape[1] if input_ids.dim() >= 2 else 1
            new_phase = "prefill" if seq_len > self.threshold else "decode"

            if new_phase != phase_cell[0]:
                phase_cell[0] = new_phase
                past_kv = kwargs.get("past_key_values")
                self._on_phase_transition(new_phase, past_kv)

        return hook

    def _make_submodule_hook(self, layer_type: str, owned_mx: list):
        """Apply the active phase/layer-type config to owned MX layers."""
        phase_cell = self._phase
        phase_configs = self.phase_configs

        def hook(module, args, kwargs):
            phase = phase_cell[0]
            overrides = phase_configs.get(phase, {}).get(layer_type)
            if overrides is None:
                return
            for mx_module in owned_mx:
                _apply_module_config(mx_module, overrides)

        return hook

    # ------------------------------------------------------------------
    # Phase-transition handlers
    # ------------------------------------------------------------------

    def _on_phase_transition(self, new_phase: str, past_key_values=None) -> None:
        """Central handler called exactly once per real phase change.

        Stage 1: mutate attention/nonlinear wrapper sub-configs.
        Stage 2: prime explicit Linear weight precision for quantized phases.
        Stage 3: select or load weights according to weight_mode.
        Stage 4: in-place re-quant of existing K/V cache entries.

        Prefill-only Level 1 note: ``weight_mode='fp'`` means FP Linear
        weights plus Linear activation bypass during decode. If the decode
        attention config also sets ``bypass=True``, QK/AV/softmax/rope/new-KV
        attention paths bypass MX quantization and Stage 4 skips old-KV
        requant. This still does not reconstruct FP K/V values already lost
        during a quantized prefill.
        """
        phase_overrides = self.phase_configs.get(new_phase, {})

        # ── Stage 1: mutate attention/nonlinear wrapper configs ─────
        # Use (phase, "attn") as the single source of truth for QK, AV,
        # KV-cache, softmax, and rope sub-configs. Width/block size only
        # matter for MX paths; bypass is mirrored to wrapper attributes because
        # the attention forward reads qk_bypass/av_bypass/... directly.
        attn_overrides = phase_overrides.get("attn") or {}
        if attn_overrides:
            bypass = attn_overrides.get("bypass")
            for _, wrapper in self.attn_wrappers:
                for attr_name in _ATTN_CONFIG_ATTRS:
                    target = getattr(wrapper, attr_name, None)
                    if not isinstance(target, dict):
                        continue
                    if attr_name in ("qk_config", "av_config"):
                        sub_key = "qk_matmul" if attr_name == "qk_config" else "av_matmul"
                        sub_cfg = attn_overrides.get(sub_key)
                        target.clear()
                        for key, value in attn_overrides.items():
                            if key in (
                                "kv_cache",
                                "softmax",
                                "rope",
                                "qk_matmul",
                                "av_matmul",
                                "weight_mode",
                                "linear_bypass",
                            ):
                                continue
                            target[key] = value
                        if isinstance(sub_cfg, dict):
                            target.update(sub_cfg)
                    elif attr_name == "kv_cache_config":
                        sub_cfg = attn_overrides.get("kv_cache", {})
                        if sub_cfg:
                            target.clear()
                            target.update(sub_cfg)
                    elif attr_name == "softmax_config":
                        sub_cfg = attn_overrides.get("softmax", {})
                        if sub_cfg:
                            target.clear()
                            target.update(sub_cfg)
                    elif attr_name == "rope_config":
                        sub_cfg = attn_overrides.get("rope", {})
                        if sub_cfg:
                            target.clear()
                            target.update(sub_cfg)
                    if bypass is not None:
                        target["bypass"] = bool(bypass)
                if bypass is not None:
                    _set_attention_bypass_attrs(wrapper, bool(bypass))
                else:
                    _sync_attention_bypass_attrs(wrapper)

        ffn_overrides = phase_overrides.get("ffn") or {}
        if ffn_overrides:
            for _, wrapper in self.mlp_wrappers:
                _apply_expert_activation_config(wrapper, ffn_overrides)

        mlp_overrides = phase_overrides.get("mlp") or {}
        if mlp_overrides:
            for _, wrapper in self.mlp_wrappers:
                _apply_nonlinear_config(wrapper, mlp_overrides)

        rms_overrides = phase_overrides.get("rms_norm") or {}
        if rms_overrides:
            for _, wrapper in self.rmsnorm_wrappers:
                _apply_nonlinear_config(wrapper, rms_overrides)

        # ── Stage 2: prime explicit quantized weight config ─────────
        weight_keys = {
            "weight_width",
            "weight_exponent_width",
            "weight_frac_width",
            "weight_block_size",
        }
        for info in self._submodule_info.values():
            overrides = phase_overrides.get(info["layer_type"]) or {}
            if _normalize_weight_mode(overrides) == "fp":
                continue
            explicit = {key: overrides[key] for key in weight_keys if key in overrides}
            if not explicit:
                continue
            for mx_module in info["owned_mx"]:
                mx_module.config.update(explicit)

        # ── Stage 3: weight residency switch / reload ────────────────
        if self.weight_residency == "gpu_dual":
            self._select_gpu_dual_weight_phase(new_phase)
        elif self._shard_map is not None or self._fp_weight_active:
            self._reload_weights_for_phase(new_phase)

        # ── Stage 4: in-place re-quant of existing KV cache ───────────
        if past_key_values is not None:
            self._requant_kv_cache(past_key_values, new_phase)

    def _cache_quantized_weight(self, mx) -> None:
        """Save the current quantized/GPTQ Linear state before FP decode."""
        mx_id = id(mx)
        if mx_id in self._quant_weight_cache:
            return
        self._quant_weight_cache[mx_id] = {
            "weight": mx.weight.detach().to("cpu").clone(),
            "bias": (
                mx.bias.detach().to("cpu").clone()
                if getattr(mx, "bias", None) is not None
                else None
            ),
            "bypass_attr": bool(getattr(mx, "bypass", False)),
            "config_bypass": mx.config.get("bypass", None),
        }

    def _restore_quantized_weight(self, mx) -> None:
        """Restore the cached quantized/GPTQ Linear state after FP decode."""
        mx_id = id(mx)
        cached = self._quant_weight_cache.get(mx_id)
        if cached is None:
            return
        with torch.no_grad():
            mx.weight.data.copy_(
                cached["weight"].to(mx.weight.device, dtype=mx.weight.dtype)
            )
            cached_bias = cached.get("bias")
            if getattr(mx, "bias", None) is not None and cached_bias is not None:
                mx.bias.data.copy_(
                    cached_bias.to(mx.bias.device, dtype=mx.bias.dtype)
                )
        config_bypass = cached.get("config_bypass")
        if config_bypass is None:
            mx.config.pop("bypass", None)
        else:
            mx.config["bypass"] = config_bypass
        if hasattr(mx, "bypass"):
            mx.bypass = bool(cached.get("bypass_attr", False))
        self._fp_weight_active.discard(mx_id)

    @staticmethod
    def _decoder_layer_tensor_name(module_name: str) -> tuple[int, str]:
        """Map ``model.layers.N.foo`` to its per-layer cache key."""
        prefix = "model.layers."
        if not module_name.startswith(prefix):
            raise ValueError(
                f"GPTQ layer cache restoration requires a decoder-layer module, got {module_name!r}."
            )
        layer_text, separator, local_name = module_name[len(prefix):].partition(".")
        if not separator or not layer_text.isdigit() or not local_name:
            raise ValueError(f"Malformed decoder-layer module name: {module_name!r}.")
        return int(layer_text), local_name

    def _quant_layer_path(self, layer_idx: int) -> Path:
        if self.quant_checkpoint_dir is None:
            raise RuntimeError("No quant_checkpoint_dir is configured.")
        path = self.quant_checkpoint_dir / f"quantized_model_layer_{layer_idx}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing GPTQ layer checkpoint for disk reload: {path}"
            )
        return path

    def _restore_linears_from_quant_checkpoint(
        self,
        targets: list[tuple[str, object, dict]],
    ) -> None:
        """Restore fake-quantized Linear tensors without a full CPU backup."""
        by_layer: dict[int, list[tuple[str, object, dict]]] = {}
        for module_name, mx, overrides in targets:
            layer_idx, local_name = self._decoder_layer_tensor_name(module_name)
            by_layer.setdefault(layer_idx, []).append(
                (f"{local_name}.weight", mx, overrides)
            )

        host_jobs: list[tuple[torch.Tensor, torch.Tensor]] = []
        restored: list[tuple[object, dict]] = []
        for layer_idx, items in sorted(by_layer.items()):
            with self._open_safetensor(self._quant_layer_path(layer_idx)) as handle:
                keys = set(handle.keys())
                for tensor_name, mx, overrides in items:
                    if tensor_name not in keys:
                        raise KeyError(
                            f"GPTQ layer {layer_idx} is missing {tensor_name!r}."
                        )
                    quant_cpu = handle.get_tensor(tensor_name)
                    if self._retained_file_cache_enabled:
                        host_jobs.append((mx.weight.data, quant_cpu))
                        restored.append((mx, overrides))
                    else:
                        with torch.no_grad():
                            mx.weight.data.copy_(
                                quant_cpu.to(mx.weight.device, dtype=mx.weight.dtype)
                            )
                        _apply_module_config(mx, overrides)
                        self._fp_weight_active.discard(id(mx))
        if host_jobs:
            self._copy_host_cached_jobs(host_jobs, label="GPTQ Linear restore")
            for mx, overrides in restored:
                _apply_module_config(mx, overrides)
                self._fp_weight_active.discard(id(mx))

    @staticmethod
    def _primary_expert_target(
        expert_module,
        expert_idx: int,
        projection: str,
    ) -> torch.Tensor:
        split = int(expert_module.intermediate_dim)
        if projection == "gate":
            return expert_module.gate_up_proj[expert_idx, :split]
        if projection == "up":
            return expert_module.gate_up_proj[expert_idx, split:]
        if projection == "down":
            return expert_module.down_proj[expert_idx]
        raise ValueError(f"Unsupported expert projection: {projection!r}.")

    @classmethod
    def _copy_primary_expert_tensor(
        cls,
        expert_module,
        expert_idx: int,
        projection: str,
        tensor: torch.Tensor,
    ) -> None:
        target = cls._primary_expert_target(
            expert_module, expert_idx, projection
        )
        with torch.no_grad():
            target.copy_(tensor.to(target.device, dtype=target.dtype))

    def _restore_experts_from_quant_checkpoint(self) -> None:
        host_jobs: list[tuple[torch.Tensor, torch.Tensor]] = []
        restored_modules: list[object] = []
        for expert_name, expert_module in self.expert_weight_wrappers:
            if id(expert_module) not in self._fp_expert_active:
                continue
            layer_idx, local_name = self._decoder_layer_tensor_name(expert_name)
            gate_up_name = f"{local_name}.gate_up_proj"
            down_name = f"{local_name}.down_proj"
            with self._open_safetensor(self._quant_layer_path(layer_idx)) as handle:
                keys = set(handle.keys())
                missing = [name for name in (gate_up_name, down_name) if name not in keys]
                if missing:
                    raise KeyError(
                        f"GPTQ layer {layer_idx} is missing packed expert tensors {missing}."
                    )
                gate_up = handle.get_tensor(gate_up_name)
                down = handle.get_tensor(down_name)
                if self._retained_file_cache_enabled:
                    host_jobs.extend(
                        (
                            (expert_module.gate_up_proj.data, gate_up),
                            (expert_module.down_proj.data, down),
                        )
                    )
                    restored_modules.append(expert_module)
                else:
                    with torch.no_grad():
                        expert_module.gate_up_proj.data.copy_(
                            gate_up.to(
                                expert_module.gate_up_proj.device,
                                dtype=expert_module.gate_up_proj.dtype,
                            )
                        )
                        expert_module.down_proj.data.copy_(
                            down.to(
                                expert_module.down_proj.device,
                                dtype=expert_module.down_proj.dtype,
                            )
                        )
                    expert_module.set_use_fp_weight(False)
                    self._fp_expert_active.discard(id(expert_module))
        if host_jobs:
            self._copy_host_cached_jobs(host_jobs, label="GPTQ expert restore")
            for expert_module in restored_modules:
                expert_module.set_use_fp_weight(False)
                self._fp_expert_active.discard(id(expert_module))

    def _load_fp_experts_from_model_checkpoint(self) -> None:
        by_shard: dict[str, list[tuple[str, object, int, str]]] = {}
        for expert_name, expert_module in self.expert_weight_wrappers:
            if id(expert_module) in self._fp_expert_active:
                continue
            for expert_idx in range(int(expert_module.num_experts)):
                for projection in ("gate", "up", "down"):
                    tensor_name = f"{expert_name}.{expert_idx}.{projection}_proj.weight"
                    shard_path = None if self._shard_map is None else self._shard_map.get(tensor_name)
                    if shard_path is None:
                        raise ValueError(
                            f"Missing checkpoint tensor {tensor_name!r}; cannot enter FP decode."
                        )
                    by_shard.setdefault(shard_path, []).append(
                        (tensor_name, expert_module, expert_idx, projection)
                    )

        host_jobs: list[tuple[torch.Tensor, torch.Tensor]] = []
        loaded_modules: set[object] = set()
        for shard_path, items in by_shard.items():
            with self._open_safetensor(shard_path) as handle:
                keys = set(handle.keys())
                for tensor_name, expert_module, expert_idx, projection in items:
                    if tensor_name not in keys:
                        raise KeyError(
                            f"Checkpoint shard {shard_path} is missing {tensor_name!r}."
                        )
                    source = handle.get_tensor(tensor_name)
                    if self._retained_file_cache_enabled:
                        host_jobs.append(
                            (
                                self._primary_expert_target(
                                    expert_module, expert_idx, projection
                                ),
                                source,
                            )
                        )
                        loaded_modules.add(expert_module)
                    else:
                        self._copy_primary_expert_tensor(
                            expert_module,
                            expert_idx,
                            projection,
                            source,
                        )
                        self._fp_expert_active.add(id(expert_module))
        if host_jobs:
            self._copy_host_cached_jobs(host_jobs, label="FP expert load")
            self._fp_expert_active.update(id(module) for module in loaded_modules)

    def _reload_expert_weights_for_phase(self, phase: str) -> None:
        if not self._fp_expert_weight_requested:
            return
        phase_overrides = self.phase_configs.get(phase, {})
        use_fp = _normalize_weight_mode(phase_overrides.get("ffn") or {}) == "fp"
        if use_fp:
            self._load_fp_experts_from_model_checkpoint()
        elif self._fp_expert_active:
            self._restore_experts_from_quant_checkpoint()

    def _reload_weights_for_phase(self, phase: str) -> None:
        """Load or restore Linear weights for the requested phase.

        ``weight_mode='quantized'`` preserves the existing disk-backed
        re-quant path: read FP checkpoint weights and call ``load_state_dict``
        so LinearMXInt/LinearMXFP re-quantizes using the primed config.

        ``weight_mode='fp'`` is prefill-only mode: cache the current
        quantized/GPTQ weight, copy the original checkpoint FP tensor directly
        into the Linear module, and set ``bypass=True``. This intentionally
        avoids ``load_state_dict`` because that would either re-quantize the
        weight or skip GPTQ modules without guaranteeing activation bypass.
        """
        reload_started = time.perf_counter()
        phase_overrides = self.phase_configs.get(phase, {})

        # Group target weight tensors by shard path for I/O locality.
        by_shard: dict[str, list[tuple[str, object, str]]] = {}
        quant_checkpoint_targets: list[tuple[str, object, dict]] = []
        for info in self._submodule_info.values():
            lt_overrides = phase_overrides.get(info["layer_type"]) or {}
            mode = _normalize_weight_mode(lt_overrides)
            for mx in info["owned_mx"]:
                if mode == "quantized" and id(mx) in self._fp_weight_active:
                    mx_name = self._mx_name_by_id.get(id(mx))
                    if self.quant_checkpoint_dir is not None and mx_name is not None:
                        quant_checkpoint_targets.append((mx_name, mx, lt_overrides))
                    else:
                        self._restore_quantized_weight(mx)
                    continue

                needs_weight = mode == "fp" or any(
                    k in lt_overrides
                    for k in ("weight_width", "weight_exponent_width", "weight_frac_width")
                )
                if not needs_weight:
                    continue

                mx_name = self._mx_name_by_id.get(id(mx))
                if mx_name is None:
                    continue
                tensor_name = f"{mx_name}.weight"
                shard_path = None if self._shard_map is None else self._shard_map.get(tensor_name)
                if shard_path is None:
                    if mode == "fp":
                        raise ValueError(
                            f"Missing checkpoint tensor {tensor_name!r}; "
                            "cannot enter weight_mode='fp'."
                        )
                    continue
                by_shard.setdefault(shard_path, []).append((tensor_name, mx, mode))

        host_jobs: list[tuple[torch.Tensor, torch.Tensor]] = []
        host_fp_modules: list[object] = []
        for shard_path, items in by_shard.items():
            with self._open_safetensor(shard_path) as f:
                shard_keys = set(f.keys())
                for tensor_name, mx, mode in items:
                    fp_cpu = f.get_tensor(tensor_name)
                    if mode == "fp":
                        if self.quant_checkpoint_dir is None:
                            self._cache_quantized_weight(mx)
                        bias_name = tensor_name[:-len(".weight")] + ".bias"
                        bias_cpu = None
                        if getattr(mx, "bias", None) is not None:
                            if bias_name in shard_keys:
                                bias_cpu = f.get_tensor(bias_name)
                            elif self._shard_map is not None:
                                bias_shard = self._shard_map.get(bias_name)
                                if bias_shard is not None:
                                    with self._open_safetensor(bias_shard) as bf:
                                        bias_cpu = bf.get_tensor(bias_name)
                        if self._retained_file_cache_enabled:
                            host_jobs.append((mx.weight.data, fp_cpu))
                            if bias_cpu is not None:
                                host_jobs.append((mx.bias.data, bias_cpu))
                            host_fp_modules.append(mx)
                        else:
                            with torch.no_grad():
                                mx.weight.data.copy_(
                                    fp_cpu.to(mx.weight.device, dtype=mx.weight.dtype)
                                )
                                if bias_cpu is not None:
                                    mx.bias.data.copy_(
                                        bias_cpu.to(mx.bias.device, dtype=mx.bias.dtype)
                                    )
                            mx.config["bypass"] = True
                            if hasattr(mx, "bypass"):
                                mx.bypass = True
                            self._fp_weight_active.add(id(mx))
                    else:
                        local_sd = {
                            "weight": fp_cpu.to(
                                mx.weight.device, dtype=mx.weight.dtype,
                            ),
                        }
                        # LinearMXInt/LinearMXFP.load_state_dict re-runs the
                        # module's own weight quantizer using Stage 2's config.
                        mx.load_state_dict(local_sd, strict=False)

        if host_jobs:
            self._copy_host_cached_jobs(host_jobs, label="FP Linear load")
            for mx in host_fp_modules:
                mx.config["bypass"] = True
                if hasattr(mx, "bypass"):
                    mx.bypass = True
                self._fp_weight_active.add(id(mx))
        if quant_checkpoint_targets:
            self._restore_linears_from_quant_checkpoint(quant_checkpoint_targets)
        self._reload_expert_weights_for_phase(phase)
        logger.info(
            "Phase %s weight reload via %s completed in %.1fs",
            phase,
            self.weight_residency,
            time.perf_counter() - reload_started,
        )

    def _gpu_dual_target_modules(self) -> dict[int, tuple[str, object]]:
        """Return MX modules that need an original FP backup in gpu_dual mode."""
        targets: dict[int, tuple[str, object]] = {}
        for info in self._submodule_info.values():
            for by_layer in self.phase_configs.values():
                lt_overrides = by_layer.get(info["layer_type"]) or {}
                if _normalize_weight_mode(lt_overrides) != "fp":
                    continue
                for mx in info["owned_mx"]:
                    mx_name = self._mx_name_by_id.get(id(mx))
                    if mx_name is not None:
                        targets[id(mx)] = (mx_name, mx)
        return targets

    def _materialize_gpu_dual_fp_weights(self) -> None:
        """Load original checkpoint weights once into GPU-resident buffers."""
        from safetensors import safe_open
        from quant_eval.eval.unified_mx import LinearMXUnified

        targets = self._gpu_dual_target_modules()
        unsupported = [
            mx_name
            for mx_name, mx in targets.values()
            if not isinstance(mx, LinearMXUnified)
        ]
        if unsupported:
            preview = ", ".join(unsupported[:5])
            suffix = "..." if len(unsupported) > 5 else ""
            raise ValueError(
                "weight_residency='gpu_dual' currently supports only LinearMXUnified "
                "modules. Unsupported modules: "
                f"{preview}{suffix}. Use weight_residency='disk_reload' for this config."
            )

        by_shard: dict[str, list[tuple[str, LinearMXUnified]]] = {}
        for mx_name, mx in targets.values():
            tensor_name = f"{mx_name}.weight"
            shard_path = None if self._shard_map is None else self._shard_map.get(tensor_name)
            if shard_path is None:
                raise ValueError(
                    f"Missing checkpoint tensor {tensor_name!r}; cannot materialize gpu_dual FP backup."
                )
            by_shard.setdefault(shard_path, []).append((tensor_name, mx))

        for shard_path, items in by_shard.items():
            with safe_open(shard_path, framework="pt") as f:
                shard_keys = set(f.keys())
                for tensor_name, mx in items:
                    fp_cpu = f.get_tensor(tensor_name)
                    bias_name = tensor_name[:-len(".weight")] + ".bias"
                    bias_cpu = f.get_tensor(bias_name) if bias_name in shard_keys else None
                    mx.set_fp_weight_backup(fp_cpu, bias_cpu)

        if not self._fp_expert_weight_requested:
            logger.info(
                "Materialized gpu_dual FP backups for %d Linear weights; no packed experts requested",
                len(targets),
            )
            return

        expert_by_shard: dict[str, list[tuple[str, object, int, str]]] = {}
        expected_tensors = 0
        for expert_name, expert_module in self.expert_weight_wrappers:
            for expert_idx in range(int(expert_module.num_experts)):
                for projection in ("gate", "up", "down"):
                    tensor_name = (
                        f"{expert_name}.{expert_idx}.{projection}_proj.weight"
                    )
                    shard_path = None if self._shard_map is None else self._shard_map.get(tensor_name)
                    if shard_path is None:
                        raise ValueError(
                            f"Missing checkpoint tensor {tensor_name!r}; cannot materialize "
                            "gpu_dual FP expert backup."
                        )
                    expert_by_shard.setdefault(shard_path, []).append(
                        (tensor_name, expert_module, expert_idx, projection)
                    )
                    expected_tensors += 1

        # Validate the complete source map before committing hundreds of GiB
        # of GPU memory to packed FP expert buffers.
        for _, expert_module in self.expert_weight_wrappers:
            expert_module.allocate_fp_weight_backup()

        loaded_tensors = 0
        for shard_position, (shard_path, items) in enumerate(expert_by_shard.items(), start=1):
            with safe_open(shard_path, framework="pt") as f:
                shard_keys = set(f.keys())
                for tensor_name, expert_module, expert_idx, projection in items:
                    if tensor_name not in shard_keys:
                        raise ValueError(
                            f"Checkpoint index maps {tensor_name!r} to {shard_path}, "
                            "but the tensor is absent from that shard."
                        )
                    fp_cpu = f.get_tensor(tensor_name)
                    expert_module.copy_fp_expert_tensor(
                        expert_idx,
                        projection,
                        fp_cpu,
                    )
                    del fp_cpu
                    loaded_tensors += 1
            if shard_position == 1 or shard_position == len(expert_by_shard) or shard_position % 10 == 0:
                logger.info(
                    "Materialized gpu_dual FP experts from shard %d/%d (%d/%d tensors)",
                    shard_position,
                    len(expert_by_shard),
                    loaded_tensors,
                    expected_tensors,
                )

        if loaded_tensors != expected_tensors:
            raise RuntimeError(
                f"Incomplete FP expert backup: loaded {loaded_tensors}/{expected_tensors} tensors."
            )
        for _, expert_module in self.expert_weight_wrappers:
            expert_module.mark_fp_weight_ready()
        logger.info(
            "Materialized gpu_dual FP backups for %d Linear weights and %d packed expert modules",
            len(targets),
            len(self.expert_weight_wrappers),
        )

    def _select_gpu_dual_weight_phase(self, phase: str) -> None:
        """Select quantized or FP resident weights for this phase."""
        phase_overrides = self.phase_configs.get(phase, {})
        for info in self._submodule_info.values():
            lt_overrides = phase_overrides.get(info["layer_type"]) or {}
            use_fp = _normalize_weight_mode(lt_overrides) == "fp"
            for mx in info["owned_mx"]:
                if hasattr(mx, "set_use_fp_weight"):
                    mx.set_use_fp_weight(use_fp)
        use_fp_experts = _normalize_weight_mode(phase_overrides.get("ffn") or {}) == "fp"
        for _, expert_module in self.expert_weight_wrappers:
            expert_module.set_use_fp_weight(use_fp_experts)

    def _requant_kv_cache(self, past_key_values, new_phase: str) -> None:
        """In-place fake-quant of existing K/V entries in ``past_key_values``
        at the new phase's attn width/block_size.

        Uses the same ``mxint_quantizer`` and ``block_dim=-1`` convention as
        ``kv_cache_mxint`` in the forward path
        (``mase/src/chop/nn/quantized/functional/kvcache.py``), so re-quanted
        entries are indistinguishable from entries that would have been
        produced under the new phase to begin with.

        Important: this is independent from Linear ``weight_mode='fp'``. FP
        decode weights do not make existing KV cache entries FP. If prefill
        already produced quantized KV, this transition can only requant that
        quantized cache; it cannot reconstruct the original unquantized K/V
        tensors. With Level 1 decode attention bypass, this function returns
        early instead: old quantized prefill KV remains as-is, and subsequent
        decode tokens use the wrapper's ``kv_cache_bypass=True`` path."""
        attn_overrides = self.phase_configs.get(new_phase, {}).get("attn") or {}
        if bool(attn_overrides.get("bypass", False)):
            # Level 1 prefill-only mode: do not re-quant existing KV. This is
            # not a recovery to FP; already-quantized prefill KV remains lossy.
            return

        from quant_eval.eval.unified_mx import quantize_mx

        # Duck-type HF DynamicCache / StaticCache: both expose key_cache and
        # value_cache as list[Tensor], one entry per layer.
        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if not isinstance(key_cache, list) or not isinstance(value_cache, list):
            return

        kv_cfg = attn_overrides.get("kv_cache") or attn_overrides
        bs = kv_cfg.get("data_in_block_size")
        mxint_width = kv_cfg.get("data_in_width")
        mxfp_exp = kv_cfg.get("data_in_exponent_width")
        mxfp_frac = kv_cfg.get("data_in_frac_width")
        if bs is None:
            return
        if mxint_width is None and (mxfp_exp is None or mxfp_frac is None):
            return

        n = min(len(key_cache), len(value_cache))
        for layer_idx in range(n):
            k = key_cache[layer_idx]
            v = value_cache[layer_idx]
            if k is None or v is None:
                continue
            if not hasattr(k, "numel") or k.numel() == 0:
                continue
            if not hasattr(v, "numel") or v.numel() == 0:
                continue
            # K/V shapes: [B, num_kv_heads, seq_len, head_dim]
            # block_dim=-1 matches kv_cache_mxint (head_dim blocking).
            key_cache[layer_idx] = quantize_mx(k, kv_cfg, block_dim=-1, prefix="data_in")
            value_cache[layer_idx] = quantize_mx(v, kv_cfg, block_dim=-1, prefix="data_in")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_phase(
        self,
        phase: str,
        phase_config: dict[str, dict] | None = None,
        past_key_values=None,
    ) -> None:
        """Apply a phase and optionally replace its runtime config."""
        if phase_config is not None:
            self.phase_configs[phase] = phase_config
        elif phase not in self.phase_configs:
            raise KeyError(f"Unknown phase {phase!r}.")
        self._on_phase_transition(phase, past_key_values)
        self._phase[0] = phase

    def enable(self) -> "PhaseLayerAutoSwitch":
        """Register all hooks and initialise layers to the prefill config."""
        if self._hook_handles:
            raise RuntimeError(
                "PhaseLayerAutoSwitch already enabled — call disable() first."
            )

        # Top-level phase-detection hook (lightweight — no config writes).
        h = self.model.register_forward_pre_hook(
            self._make_phase_detection_hook(), with_kwargs=True
        )
        self._hook_handles.append(h)

        # Per-submodule hooks preserve the established runtime behavior.
        for info in self._submodule_info.values():
            if not info["owned_mx"]:
                continue
            h = info["module"].register_forward_pre_hook(
                self._make_submodule_hook(info["layer_type"], info["owned_mx"]),
                with_kwargs=True,
            )
            self._hook_handles.append(h)

        # Initialise Linear and nonlinear wrappers to prefill.
        self._phase[0] = "prefill"
        for info in self._submodule_info.values():
            overrides = self.phase_configs.get("prefill", {}).get(info["layer_type"])
            if overrides:
                _apply_config([(None, module) for module in info["owned_mx"]], overrides)
        self._on_phase_transition("prefill", None)

        return self

    def disable(self) -> None:
        """Remove all hooks and restore original MX configs.

        Restores:
          - LinearMXInt.config on every Linear MX layer
          - qk/av/kv-cache/softmax/rope config and bypass attributes on every
            quantized attention wrapper

        Restores any active FP decode weights back to the cached
        quantized/GPTQ tensors. Does NOT touch past_key_values (caller owns it).
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        if (
            self.weight_residency in {"disk_reload", "host_cached"}
            and (self._fp_weight_active or self._fp_expert_active)
        ):
            self._reload_weights_for_phase("prefill")

        for _, module in self._all_mx_layers:
            if id(module) in self._fp_weight_active:
                self._restore_quantized_weight(module)
            if hasattr(module, "set_use_fp_weight"):
                module.set_use_fp_weight(False)
        for _, expert_module in self.expert_weight_wrappers:
            expert_module.set_use_fp_weight(False)

        for name, module in self._all_mx_layers:
            original = self._original_configs.get(name, {})
            if "bypass" not in original:
                module.config.pop("bypass", None)
            for key, value in original.items():
                module.config[key] = value
            if hasattr(module, "bypass"):
                module.bypass = bool(module.config.get("bypass", False))

        for name, wrapper in self.attn_wrappers:
            snap = self._original_attn_configs.get(name, {})
            bypass_attrs = snap.get("bypass_attrs", {})
            for attr_name in _ATTN_CONFIG_ATTRS:
                original_sub = snap.get(attr_name)
                target = getattr(wrapper, attr_name, None)
                if not isinstance(target, dict) or not isinstance(original_sub, dict):
                    continue
                target.clear()
                target.update(original_sub)
            if isinstance(bypass_attrs, dict):
                for attr_name, original_value in bypass_attrs.items():
                    if hasattr(wrapper, attr_name):
                        setattr(wrapper, attr_name, bool(original_value))

        for name, wrapper in self.mlp_wrappers:
            wrapper.q_config.clear()
            wrapper.q_config.update(self._original_mlp_configs.get(name, {}))
            if hasattr(wrapper, "act_config"):
                wrapper.act_config.clear()
                wrapper.act_config.update(self._original_mlp_act_configs.get(name, {}))
            if hasattr(wrapper, "bypass"):
                wrapper.bypass = bool(wrapper.q_config.get("bypass", False))

        for name, wrapper in self.rmsnorm_wrappers:
            wrapper.q_config.clear()
            wrapper.q_config.update(self._original_rmsnorm_configs.get(name, {}))
            _rebuild_rmsnorm_minifloat_quantizers(wrapper)

        self._phase[0] = "prefill"
        self._close_host_cached_sources()

    def summary(self) -> str:
        """Human-readable table of the active phase config mapping."""
        def _fmt_mx(cfg: dict) -> str:
            if not cfg:
                return "(unchanged)"
            if cfg.get("bypass", False):
                return f"bypass  weight={_normalize_weight_mode(cfg)}"
            if cfg.get("canonical"):
                base = f"{cfg['canonical']}(B{cfg.get('data_in_block_size', '?')})"
            elif "data_in_width" in cfg:
                base = f"MXINT_{cfg['data_in_width']}(B{cfg.get('data_in_block_size', '?')})"
            elif "data_in_exponent_width" in cfg:
                base = (
                    f"MXFP_E{cfg['data_in_exponent_width']}M{cfg['data_in_frac_width']}"
                    f"(B{cfg.get('data_in_block_size', '?')})"
                )
            else:
                base = str(cfg)
            kv_cfg = cfg.get("kv_cache")
            if isinstance(kv_cfg, dict) and kv_cfg:
                kv_name = kv_cfg.get("canonical")
                if kv_name is None and "data_in_width" in kv_cfg:
                    kv_name = f"MXINT_{kv_cfg['data_in_width']}"
                elif kv_name is None and "data_in_exponent_width" in kv_cfg:
                    kv_name = f"MXFP_E{kv_cfg['data_in_exponent_width']}M{kv_cfg['data_in_frac_width']}"
                if kv_name:
                    base = f"ACT={base}, KV={kv_name}(B{kv_cfg.get('data_in_block_size', '?')})"
            subparts = []
            for label, key in (
                ("qk", "qk_matmul"),
                ("av", "av_matmul"),
                ("softmax", "softmax"),
                ("rope", "rope"),
            ):
                sub_cfg = cfg.get(key)
                if isinstance(sub_cfg, dict) and sub_cfg.get("bypass", False):
                    subparts.append(f"{label}=bypass")
            if isinstance(kv_cfg, dict) and kv_cfg.get("bypass", False):
                subparts.append("kv=bypass")
            if cfg.get("linear_bypass", False):
                subparts.append("linear=bypass")
            if subparts:
                base = f"{base}; " + ", ".join(subparts)
            return f"{base}  weight={_normalize_weight_mode(cfg)}"

        def _fmt_fp(cfg: dict) -> str:
            if not cfg:
                return "(unchanged)"
            if cfg.get("bypass", False):
                return "bypass"
            if "data_in_exponent_width" in cfg:
                return f"FP_E{cfg['data_in_exponent_width']}M{cfg['data_in_frac_width']}"
            return str(cfg)

        lines = ["PhaseLayerAutoSwitch config:"]
        lines.append(f"  {'phase':10s}  {'component':8s}  config")
        lines.append("  " + "-" * 58)
        for phase in ("prefill", "decode"):
            by_component = self.phase_configs.get(phase, {})
            for component in ("attn", "ffn", "mlp", "rms_norm"):
                cfg = by_component.get(component)
                if component in ("attn", "ffn"):
                    text = _fmt_mx(cfg or {})
                else:
                    text = _fmt_fp(cfg or {})
                lines.append(f"  {phase:10s}  {component:8s}  {text}")

        n_attn = sum(
            1 for info in self._submodule_info.values()
            if info["layer_type"] == "attn" and info["owned_mx"]
        )
        n_ffn = sum(
            1 for info in self._submodule_info.values()
            if info["layer_type"] == "ffn"  and info["owned_mx"]
        )
        lines.append(
            f"\n  Hooked submodules: {n_attn} attn, {n_ffn} ffn; "
            f"nonlinear wrappers: {len(self.mlp_wrappers)} mlp, {len(self.rmsnorm_wrappers)} rms_norm; "
            f"packed expert weight wrappers: {len(self.expert_weight_wrappers)}"
        )
        effective_io = (
            "retained_mmap_page_cache"
            if self._retained_file_cache_enabled
            else self.weight_residency
        )
        lines.append(
            f"  Weight residency: requested={self.weight_residency}, "
            f"effective_io={effective_io}"
        )
        return "\n".join(lines)

    def __enter__(self):
        return self.enable()

    def __exit__(self, *args):
        self.disable()
