import torch

from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.eval.unified_mx import Qwen3MoeExpertsMXUnified, apply_unified_mx_wrappers
from quant_eval.precision import apply_dse_quant_config


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

