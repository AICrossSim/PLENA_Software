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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_mx_layers(model: nn.Module):
    """Return (name, module) for every LinearMXInt / LinearMXFP in the model."""
    from chop.nn.quantized.modules.linear import LinearMXInt, LinearMXFP

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (LinearMXInt, LinearMXFP)):
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


def _set_attention_bypass_attrs(wrapper: nn.Module, bypass: bool) -> None:
    """Synchronise attention wrapper bypass attrs read by forward()."""
    for attr_name in _ATTN_BYPASS_ATTRS:
        if hasattr(wrapper, attr_name):
            setattr(wrapper, attr_name, bool(bypass))


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
    for key, value in overrides.items():
        if key == "weight_mode":
            continue
        if key == "bypass":
            module.config["bypass"] = bool(value)
            if hasattr(module, "bypass"):
                module.bypass = bool(value)
            continue
        if key in module.config:
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
        self._fp_weight_requested = self._phase_configs_request_fp_weight()
        if self._fp_weight_requested and model_name is None:
            raise ValueError(
                "PhaseLayerAutoSwitch requires model_name when any phase "
                "config sets weight_mode='fp'."
            )

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

        # Collect quantized attention wrappers (Glm4MoeAttentionMXInt,
        # Qwen3AttentionMXInt, ...) by duck-typing. These are mutated on
        # phase transition alongside the LinearMXInt layers.
        self.attn_wrappers = _find_quant_attention_wrappers(model)
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
            w  = attn_overrides.get("data_in_width")
            bs = attn_overrides.get("data_in_block_size")
            bypass = attn_overrides.get("bypass")
            for _, wrapper in self.attn_wrappers:
                for attr_name in _ATTN_CONFIG_ATTRS:
                    target = getattr(wrapper, attr_name, None)
                    if not isinstance(target, dict):
                        continue
                    if w is not None and attr_name in (
                        "qk_config",
                        "av_config",
                        "kv_cache_config",
                    ):
                        target["data_in_width"] = w
                    if bs is not None and attr_name in (
                        "qk_config",
                        "av_config",
                        "kv_cache_config",
                    ):
                        target["data_in_block_size"] = bs
                    if bypass is not None:
                        target["bypass"] = bool(bypass)
                if bypass is not None:
                    _set_attention_bypass_attrs(wrapper, bool(bypass))

        # ── Stage 2: prime LinearMXInt weight_width in config ─────────
        for info in self._submodule_info.values():
            lt_overrides = phase_overrides.get(info["layer_type"]) or {}
            if _normalize_weight_mode(lt_overrides) == "fp":
                continue
            w  = lt_overrides.get("data_in_width")
            bs = lt_overrides.get("data_in_block_size")
            if w is None:
                continue
            for mx in info["owned_mx"]:
                mx.config["weight_width"] = w
                if bs is not None:
                    mx.config["weight_block_size"] = bs

        # ── Stage 3: disk-backed weight reload / restore ──────────────
        if self._shard_map is not None or self._fp_weight_active:
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
            needs_weight = mode == "fp" or "data_in_width" in lt_overrides
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

        from chop.nn.quantizers import mxint_quantizer

        # Duck-type HF DynamicCache / StaticCache: both expose key_cache and
        # value_cache as list[Tensor], one entry per layer.
        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if not isinstance(key_cache, list) or not isinstance(value_cache, list):
            return

        w  = attn_overrides.get("data_in_width")
        bs = attn_overrides.get("data_in_block_size")
        if w is None or bs is None:
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
            key_cache[layer_idx] = mxint_quantizer(
                k, block_size=bs, element_bits=w, block_dim=-1,
            )
            value_cache[layer_idx] = mxint_quantizer(
                v, block_size=bs, element_bits=w, block_dim=-1,
            )

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
        if self._phase_requests_fp_weight("prefill"):
            self._reload_weights_for_phase("prefill")

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

        self._phase[0] = "prefill"

    def summary(self) -> str:
        """Human-readable table of the active (phase, layer_type) → config mapping."""
        lines = ["PhaseLayerAutoSwitch config:"]
        lines.append(f"  {'phase':10s}  {'layer':6s}  config")
        lines.append("  " + "-" * 44)
        for phase in ("prefill", "decode"):
            for layer_type in ("attn", "ffn"):
                cfg = self.phase_configs.get(phase, {}).get(layer_type)
                if cfg is None:
                    lines.append(f"  {phase:10s}  {layer_type:6s}  (unchanged)")
                else:
                    width = cfg.get("data_in_width", "?")
                    bsz   = cfg.get("data_in_block_size", "?")
                    mode  = _normalize_weight_mode(cfg)
                    bypass = bool(cfg.get("bypass", False))
                    bypass_note = "  bypass" if bypass else ""
                    lines.append(
                        f"  {phase:10s}  {layer_type:6s}  "
                        f"MXInt{width}  block_size={bsz}  weight={mode}{bypass_note}"
                    )
        n_attn = sum(
            1 for info in self._submodule_info.values()
            if info["layer_type"] == "attn" and info["owned_mx"]
        )
        n_ffn = sum(
            1 for info in self._submodule_info.values()
            if info["layer_type"] == "ffn"  and info["owned_mx"]
        )
        lines.append(f"\n  Hooked submodules: {n_attn} attn, {n_ffn} ffn")
        return "\n".join(lines)

    def __enter__(self):
        return self.enable()

    def __exit__(self, *args):
        self.disable()