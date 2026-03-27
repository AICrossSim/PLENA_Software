"""
Phase-dependent quantization support.

Provides automatic prefill/decode detection for disaggregated inference
where each phase uses different MX activation precision. The weight
precision is set once at load time; only activation (data_in) precision
is swapped dynamically.

Used by eval_phase_lm.py with the lm-eval harness — PhaseAutoSwitch
detects generation tasks' prefill->decode boundary automatically via
input sequence length.
"""

from torch import nn


def _find_mx_layers(model: nn.Module):
    """Find all LinearMXInt and LinearMXFP layers in the model."""
    from chop.nn.quantized.modules.linear import LinearMXInt, LinearMXFP

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (LinearMXInt, LinearMXFP)):
            layers.append((name, module))
    return layers


def set_phase(model: nn.Module, phase_configs: dict, phase: str):
    """
    Set activation quantization config to the specified phase.

    Args:
        model: The quantized model.
        phase_configs: Dict with "prefill" and "decode" keys.
        phase: "prefill" or "decode".
    """
    mx_layers = _find_mx_layers(model)
    overrides = phase_configs.get(phase, {})

    for _, module in mx_layers:
        for key, value in overrides.items():
            if key in module.config:
                module.config[key] = value


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
            model: Quantized model with LinearMXInt layers.
            phase_configs: {"prefill": {"data_in_width": 4}, "decode": {"data_in_width": 8}}
            threshold: Sequence length threshold. seq_len > threshold -> prefill, else decode.
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
