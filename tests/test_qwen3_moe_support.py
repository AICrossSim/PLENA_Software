import torch
from torch import nn
from safetensors.torch import save_file

from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.eval.unified_mx import LinearMXUnified, Qwen3MoeExpertsMXUnified, apply_unified_mx_wrappers
from quant_eval.precision import apply_dse_quant_config, parse_mx_precision


def _tiny_qwen3_moe() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=3,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        torch_dtype="float32",
    )
    return Qwen3MoeForCausalLM(config).eval()


def _act_cfg() -> dict:
    return {
        "data_in_family": "mxint",
        "data_in_width": 8,
        "data_in_block_size": 16,
    }


def _fp_cfg() -> dict:
    return {
        "data_in_exponent_width": 8,
        "data_in_frac_width": 5,
        "data_in_is_finite": True,
        "data_in_round_mode": "rn",
        "weight_exponent_width": 8,
        "weight_frac_width": 5,
        "weight_is_finite": True,
        "weight_round_mode": "rn",
    }


def test_qwen3_moe_unified_wrapper_counts_are_nonzero():
    model = _tiny_qwen3_moe()
    counts = apply_unified_mx_wrappers(
        model,
        qwen3_moe_attention_config={**_act_cfg(), "kv_cache": _act_cfg(), "softmax": _fp_cfg(), "rope": _fp_cfg()},
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg()},
        qwen3_moe_rms_norm_config=_fp_cfg(),
    )

    assert counts["qwen3_moe_attention"] == 2
    assert counts["qwen3_moe_experts"] == 2
    assert counts["qwen3_moe_rms_norm"] > 0
    assert counts["qwen3_attention"] == 0

    router_names = [name for name, _ in model.named_modules() if name.endswith(".mlp.gate")]
    assert router_names
    assert all("MX" not in module.__class__.__name__ for name, module in model.named_modules() if name in router_names)


def test_qwen3_moe_dse_config_does_not_select_router_or_sparse_experts():
    pass_args = {}
    apply_dse_quant_config(
        pass_args,
        act_precision="MXINT_8",
        kv_precision="MXINT_8",
        fp_setting="FP_E8M5",
        model_family="qwen3_moe",
    )

    joined = "\n".join(pass_args)
    assert "self_attn" in joined
    assert "mlp\\.(gate|up|down)_proj" in joined
    assert "mlp\\.gate" not in joined
    assert "mlp\\.experts" not in joined


def test_mxint16_seed_config_is_supported_for_high_precision_smoke():
    spec = parse_mx_precision("MXINT_16")
    assert spec.canonical == "MXINT_16"

    pass_args = {}
    metadata = apply_dse_quant_config(
        pass_args,
        act_precision="MXINT_8",
        kv_precision="MXINT_8",
        fp_setting="FP_E8M5",
        weight_precision="MXINT_16",
        model_family="qwen3_moe",
    )

    assert metadata["WEIGHT_PRECISION"] == "MXINT_16"
    assert pass_args[r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"]["config"]["weight_width"] == 16


def test_qwen3_moe_expert_wrapper_follows_decode_bypass_phase():
    model = _tiny_qwen3_moe()
    apply_unified_mx_wrappers(
        model,
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg()},
    )
    experts = [m for m in model.modules() if isinstance(m, Qwen3MoeExpertsMXUnified)]
    assert experts
    assert all(not expert.bypass for expert in experts)

    switch = PhaseLayerAutoSwitch(
        model,
        {
            "prefill": {"mlp": {**_act_cfg(), **_fp_cfg()}},
            "decode": {"mlp": {"bypass": True}},
        },
    ).enable()
    try:
        switch._on_phase_transition("decode", None)
        assert all(expert.bypass for expert in experts)
        switch._on_phase_transition("prefill", None)
        assert all(not expert.bypass for expert in experts)
    finally:
        switch.disable()


def test_linear_mx_unified_can_switch_to_gpu_resident_fp_weight():
    layer = LinearMXUnified(2, 1, bias=False, config={**_act_cfg(), "bypass": True})
    with torch.no_grad():
        layer.weight.fill_(1.0)
    layer.set_fp_weight_backup(torch.full_like(layer.weight, 3.0))

    x = torch.tensor([[2.0, 4.0]])
    assert torch.equal(layer(x), torch.tensor([[6.0]]))
    layer.set_use_fp_weight(True)
    assert torch.equal(layer(x), torch.tensor([[18.0]]))
    layer.set_use_fp_weight(False)
    assert torch.equal(layer(x), torch.tensor([[6.0]]))


def test_phase_switch_gpu_dual_uses_fp_backup_without_disk_reload(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.proj = LinearMXUnified(2, 1, bias=False, config={**_act_cfg(), "bypass": True})

    model = Tiny()
    with torch.no_grad():
        model.mlp.proj.weight.fill_(1.0)

    save_file(
        {"mlp.proj.weight": torch.full_like(model.mlp.proj.weight, 5.0)},
        str(tmp_path / "model.safetensors"),
    )

    switch = PhaseLayerAutoSwitch(
        model,
        {
            "prefill": {"ffn": {"bypass": True}},
            "decode": {"ffn": {"weight_mode": "fp", "bypass": True}},
        },
        model_name=str(tmp_path),
        weight_residency="gpu_dual",
    ).enable()
    try:
        assert model.mlp.proj.fp_weight is not None
        assert not model.mlp.proj.use_fp_weight
        switch._on_phase_transition("decode", None)
        assert model.mlp.proj.use_fp_weight
        out = model.mlp.proj(torch.tensor([[1.0, 1.0]]))
        assert torch.equal(out, torch.tensor([[10.0]]))
        switch._on_phase_transition("prefill", None)
        assert not model.mlp.proj.use_fp_weight
    finally:
        switch.disable()
