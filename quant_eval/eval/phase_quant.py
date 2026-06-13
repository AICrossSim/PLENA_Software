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

import torch
from torch import nn
from functools import partial

from chop.nn.quantizers._minifloat_mx import MinifloatMeta, minifloat_quantizer_sim


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
_VALID_WEIGHT_RESIDENCIES = {"disk_reload", "gpu_dual"}


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
    desired_bypass = bool(overrides.get("bypass", False))
    module.config["bypass"] = desired_bypass
    if hasattr(module, "bypass"):
        module.bypass = desired_bypass

    if "data_in_width" in overrides:
        module.config.pop("data_in_exponent_width", None)
        module.config.pop("data_in_frac_width", None)
    if "data_in_exponent_width" in overrides or "data_in_frac_width" in overrides:
        module.config.pop("data_in_width", None)

    for key, value in overrides.items():
        if key in ("weight_mode", "kv_cache", "softmax", "rope"):
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
    Unlike PhaseAutoSwitch (one hook on the top-level model that sets all
    MX layers at once), this class registers a hook on each attention and
    FFN submodule individually.  Each hook:

      1. Reads the current phase from a shared ``_phase`` cell that is
         updated by a single lightweight top-level hook (same seq_len
         detection logic as the original).
      2. Looks up the (phase, layer_type) config and patches only the MX
         layers that are *direct children* of that submodule.

    This avoids the N-submodule × M-MX-layer cross-product walk on every
    forward pass — each hook only touches the MX layers it owns.

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
        """
        self.model = model
        self.phase_configs = phase_configs
        self.threshold = threshold
        self.attn_keywords = attn_keywords
        self.ffn_keywords  = ffn_keywords
        self.model_name = model_name
        self.weight_residency = str(weight_residency or "disk_reload").lower()
        if self.weight_residency not in _VALID_WEIGHT_RESIDENCIES:
            raise ValueError(
                f"Unsupported weight_residency={weight_residency!r}; expected one of "
                f"{sorted(_VALID_WEIGHT_RESIDENCIES)}."
            )
        self._fp_weight_requested = self._phase_configs_request_fp_weight()
        if self._fp_weight_requested and model_name is None:
            raise ValueError(
                "PhaseLayerAutoSwitch requires model_name when any phase "
                "config sets weight_mode='fp'."
            )
        if self.weight_residency == "gpu_dual" and not self._fp_weight_requested:
            raise ValueError("weight_residency='gpu_dual' requires at least one phase with weight_mode='fp'.")

        # Shared mutable cell — updated by the top-level phase-detection hook,
        # read by every per-submodule hook.  A one-element list is used so
        # closures can mutate it without 'nonlocal'.
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
        if self._fp_weight_requested and self.weight_residency == "gpu_dual":
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
        classified_names = [n for n, _ in classified]

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
        """
        Per-submodule hook: applies the (current_phase, layer_type) config
        to the owned MX layers of this submodule.
        """
        phase_cell    = self._phase
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

        Stage 1: mutate attention wrapper qk / av / kv_cache sub-configs.
        Stage 2: prime LinearMXInt ``weight_width`` in config for quantized
                 phases (FP weight phases skip this because no re-quant runs).
        Stage 3: load/restore LinearMXInt weights according to weight_mode.
        Stage 4: in-place re-quant of existing K/V cache entries.

        Prefill-only Level 1 note: ``weight_mode='fp'`` means FP Linear
        weights plus Linear activation bypass during decode. If the decode
        attention config also sets ``bypass=True``, QK/AV/softmax/rope/new-KV
        attention paths bypass MX quantization and Stage 4 skips old-KV
        requant. This still does not reconstruct FP K/V values already lost
        during a quantized prefill.
        """
        phase_overrides = self.phase_configs.get(new_phase, {})

        # ── Stage 1: mutate attention wrapper sub-configs ────────────
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
                        target.clear()
                        for key, value in attn_overrides.items():
                            if key in ("kv_cache", "softmax", "rope", "weight_mode"):
                                continue
                            target[key] = value
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

        mlp_overrides = phase_overrides.get("mlp") or {}
        if mlp_overrides:
            for _, wrapper in self.mlp_wrappers:
                _apply_nonlinear_config(wrapper, mlp_overrides)

        rms_overrides = phase_overrides.get("rms_norm") or {}
        if rms_overrides:
            for _, wrapper in self.rmsnorm_wrappers:
                _apply_nonlinear_config(wrapper, rms_overrides)

        # ── Stage 2: prime LinearMXInt weight_width in config ─────────
        for info in self._submodule_info.values():
            lt_overrides = phase_overrides.get(info["layer_type"]) or {}
            if _normalize_weight_mode(lt_overrides) == "fp":
                continue
            weight_keys = {
                "weight_width", "weight_exponent_width", "weight_frac_width", "weight_block_size"
            }
            explicit_weight_overrides = {k: v for k, v in lt_overrides.items() if k in weight_keys}
            if not explicit_weight_overrides:
                continue
            for mx in info["owned_mx"]:
                mx.config.update(explicit_weight_overrides)

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
        from safetensors import safe_open

        phase_overrides = self.phase_configs.get(phase, {})

        # Group target weight tensors by shard path for I/O locality.
        by_shard: dict[str, list[tuple[str, object, str]]] = {}
        for info in self._submodule_info.values():
            lt_overrides = phase_overrides.get(info["layer_type"]) or {}
            mode = _normalize_weight_mode(lt_overrides)
            needs_weight = mode == "fp" or any(
                k in lt_overrides
                for k in ("weight_width", "weight_exponent_width", "weight_frac_width")
            )
            if not needs_weight:
                continue
            for mx in info["owned_mx"]:
                if mode == "quantized" and id(mx) in self._fp_weight_active:
                    self._restore_quantized_weight(mx)
                    # Exact GPTQ/quantized state is back; do not fall through
                    # to disk reload, which would create a fresh quantization
                    # and lose the cached GPTQ tensor.
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

        if not by_shard:
            return

        for shard_path, items in by_shard.items():
            with safe_open(shard_path, framework="pt") as f:
                shard_keys = set(f.keys())
                for tensor_name, mx, mode in items:
                    fp_cpu = f.get_tensor(tensor_name)
                    if mode == "fp":
                        self._cache_quantized_weight(mx)
                        with torch.no_grad():
                            mx.weight.data.copy_(
                                fp_cpu.to(mx.weight.device, dtype=mx.weight.dtype)
                            )
                            bias_name = tensor_name[:-len(".weight")] + ".bias"
                            if getattr(mx, "bias", None) is not None:
                                bias_cpu = None
                                if bias_name in shard_keys:
                                    bias_cpu = f.get_tensor(bias_name)
                                elif self._shard_map is not None:
                                    bias_shard = self._shard_map.get(bias_name)
                                    if bias_shard is not None:
                                        with safe_open(bias_shard, framework="pt") as bf:
                                            bias_cpu = bf.get_tensor(bias_name)
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

    def _select_gpu_dual_weight_phase(self, phase: str) -> None:
        """Select quantized or FP resident weights for this phase."""
        phase_overrides = self.phase_configs.get(phase, {})
        for info in self._submodule_info.values():
            lt_overrides = phase_overrides.get(info["layer_type"]) or {}
            use_fp = _normalize_weight_mode(lt_overrides) == "fp"
            for mx in info["owned_mx"]:
                if hasattr(mx, "set_use_fp_weight"):
                    mx.set_use_fp_weight(use_fp)

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

        # Per-submodule hooks.
        for info in self._submodule_info.values():
            if not info["owned_mx"]:
                continue  # classified module but no MX children — skip
            h = info["module"].register_forward_pre_hook(
                self._make_submodule_hook(info["layer_type"], info["owned_mx"]),
                with_kwargs=True,
            )
            self._hook_handles.append(h)

        # Initialise everything to prefill config.
        self._phase[0] = "prefill"
        for info in self._submodule_info.values():
            overrides = self.phase_configs.get("prefill", {}).get(info["layer_type"])
            if overrides:
                _apply_config([(None, m) for m in info["owned_mx"]], overrides)
        # Also initialise attention, MLP, and RMSNorm wrapper configs.  The
        # per-submodule hooks above only touch Linear MX children during
        # forward; nonlinear wrapper state must be set once up front.
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

        for _, module in self._all_mx_layers:
            if id(module) in self._fp_weight_active:
                self._restore_quantized_weight(module)
            if hasattr(module, "set_use_fp_weight"):
                module.set_use_fp_weight(False)

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
            if hasattr(wrapper, "bypass"):
                wrapper.bypass = bool(wrapper.q_config.get("bypass", False))

        for name, wrapper in self.rmsnorm_wrappers:
            wrapper.q_config.clear()
            wrapper.q_config.update(self._original_rmsnorm_configs.get(name, {}))
            _rebuild_rmsnorm_minifloat_quantizers(wrapper)

        self._phase[0] = "prefill"

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
            f"nonlinear wrappers: {len(self.mlp_wrappers)} mlp, {len(self.rmsnorm_wrappers)} rms_norm"
        )
        return "\n".join(lines)

    def __enter__(self):
        return self.enable()

    def __exit__(self, *args):
        self.disable()
