import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from quant_eval.eval.phase_quant import PhaseLayerAutoSwitch
from quant_eval.eval.unified_mx import (
    LinearMXUnified,
    Qwen3MoeExpertsMXUnified,
    apply_qwen3_gptq_cache_unified_wrappers,
    apply_unified_mx_wrappers,
)
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


def _mxfp_act_cfg() -> dict:
    return {
        "data_in_family": "mxfp",
        "data_in_exponent_width": 5,
        "data_in_frac_width": 2,
        "data_in_block_size": 8,
    }


def test_qwen3_moe_unified_wrapper_counts_are_nonzero():
    model = _tiny_qwen3_moe()
    original_experts = [m for m in model.modules() if isinstance(m, Qwen3MoeExperts)]
    original_gate_up_ids = [id(m.gate_up_proj) for m in original_experts]
    original_down_ids = [id(m.down_proj) for m in original_experts]

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
    wrapped_experts = [m for m in model.modules() if isinstance(m, Qwen3MoeExpertsMXUnified)]
    assert [id(m.gate_up_proj) for m in wrapped_experts] == original_gate_up_ids
    assert [id(m.down_proj) for m in wrapped_experts] == original_down_ids

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


def test_qwen3_moe_expert_keeps_mxfp_act_separate_from_fp_setting():
    model = _tiny_qwen3_moe()
    apply_unified_mx_wrappers(
        model,
        qwen3_moe_experts_config={
            "act": _mxfp_act_cfg(),
            "nonlinear": _fp_cfg(),
        },
    )
    expert = next(m for m in model.modules() if isinstance(m, Qwen3MoeExpertsMXUnified))

    assert expert.act_config["data_in_exponent_width"] == 5
    assert expert.act_config["data_in_frac_width"] == 2
    assert expert.nonlinear_config["data_in_exponent_width"] == 8
    assert expert.nonlinear_config["data_in_frac_width"] == 5

    hidden = torch.randn(3, expert.hidden_dim)
    top_k_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    top_k_weights = torch.ones(3, 1)
    output = expert(hidden, top_k_index, top_k_weights)
    assert output.shape == hidden.shape


def test_qwen3_moe_expert_wrapper_switches_packed_fp_weights():
    model = _tiny_qwen3_moe()
    apply_unified_mx_wrappers(
        model,
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg(), "bypass": True},
    )
    expert = next(m for m in model.modules() if isinstance(m, Qwen3MoeExpertsMXUnified))
    with torch.no_grad():
        expert.gate_up_proj.zero_()
        expert.down_proj.zero_()
    expert.allocate_fp_weight_backup()
    for expert_idx in range(expert.num_experts):
        expert.copy_fp_expert_weight(
            expert_idx,
            gate=torch.ones(expert.intermediate_dim, expert.hidden_dim),
            up=torch.ones(expert.intermediate_dim, expert.hidden_dim),
            down=torch.ones(expert.hidden_dim, expert.intermediate_dim),
        )
    expert.mark_fp_weight_ready()

    hidden = torch.ones(1, expert.hidden_dim)
    top_k_index = torch.zeros(1, 1, dtype=torch.long)
    top_k_weights = torch.ones(1, 1)
    quant_output = expert(hidden, top_k_index, top_k_weights)
    expert.set_use_fp_weight(True)
    fp_output = expert(hidden, top_k_index, top_k_weights)

    assert torch.count_nonzero(quant_output) == 0
    assert torch.all(fp_output > 0)


def test_qwen3_moe_expert_bypass_matches_hf_forward():
    torch.manual_seed(7)
    model = _tiny_qwen3_moe()
    original = model.model.layers[0].mlp.experts
    hidden = torch.randn(5, original.hidden_dim)
    top_k_index = torch.tensor([[0], [1], [2], [0], [1]], dtype=torch.long)
    top_k_weights = torch.rand(5, 1)
    with torch.no_grad():
        expected = original(hidden, top_k_index, top_k_weights)

    apply_unified_mx_wrappers(
        model,
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg(), "bypass": True},
    )
    wrapped = model.model.layers[0].mlp.experts
    with torch.no_grad():
        actual = wrapped(hidden, top_k_index, top_k_weights)

    torch.testing.assert_close(actual, expected)


def test_phase_switch_gpu_dual_materializes_packed_moe_experts(tmp_path):
    model = _tiny_qwen3_moe()
    apply_unified_mx_wrappers(
        model,
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg()},
    )
    checkpoint = {}
    expected = {}
    for layer_idx, layer in enumerate(model.model.layers):
        expert = layer.mlp.experts
        for expert_idx in range(expert.num_experts):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            values = {
                "gate": float(10 + layer_idx + expert_idx),
                "up": float(20 + layer_idx + expert_idx),
                "down": float(30 + layer_idx + expert_idx),
            }
            checkpoint[f"{base}.gate_proj.weight"] = torch.full(
                (expert.intermediate_dim, expert.hidden_dim), values["gate"]
            )
            checkpoint[f"{base}.up_proj.weight"] = torch.full(
                (expert.intermediate_dim, expert.hidden_dim), values["up"]
            )
            checkpoint[f"{base}.down_proj.weight"] = torch.full(
                (expert.hidden_dim, expert.intermediate_dim), values["down"]
            )
            expected[(layer_idx, expert_idx)] = values
    save_file(checkpoint, str(tmp_path / "model.safetensors"))

    switch = PhaseLayerAutoSwitch(
        model,
        {
            "prefill": {"ffn": {}, "mlp": {**_act_cfg(), **_fp_cfg()}},
            "decode": {
                "ffn": {"weight_mode": "fp", "bypass": True},
                "mlp": {"bypass": True},
            },
        },
        model_name=str(tmp_path),
        weight_residency="gpu_dual",
    ).enable()
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            expert = layer.mlp.experts
            assert expert._fp_weight_ready
            assert not expert.use_fp_weight
            for expert_idx in range(expert.num_experts):
                values = expected[(layer_idx, expert_idx)]
                split = expert.intermediate_dim
                assert torch.all(expert.fp_gate_up_proj[expert_idx, :split] == values["gate"])
                assert torch.all(expert.fp_gate_up_proj[expert_idx, split:] == values["up"])
                assert torch.all(expert.fp_down_proj[expert_idx] == values["down"])

        switch._on_phase_transition("decode", None)
        assert all(layer.mlp.experts.use_fp_weight for layer in model.model.layers)
        switch._on_phase_transition("prefill", None)
        assert all(not layer.mlp.experts.use_fp_weight for layer in model.model.layers)
    finally:
        switch.disable()


@pytest.mark.parametrize("weight_residency", ["disk_reload", "host_cached"])
def test_phase_switch_file_backed_restores_packed_moe_experts_from_gptq_cache(
    tmp_path, weight_residency
):
    model = _tiny_qwen3_moe()
    apply_unified_mx_wrappers(
        model,
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg()},
    )
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    original_checkpoint = {}
    quant_expected = {}
    for layer_idx, layer in enumerate(model.model.layers):
        expert = layer.mlp.experts
        quant_expected[layer_idx] = (
            expert.gate_up_proj.detach().clone(),
            expert.down_proj.detach().clone(),
        )
        save_file(
            {
                "mlp.experts.gate_up_proj": quant_expected[layer_idx][0],
                "mlp.experts.down_proj": quant_expected[layer_idx][1],
            },
            str(quant_dir / f"quantized_model_layer_{layer_idx}.safetensors"),
        )
        for expert_idx in range(expert.num_experts):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            original_checkpoint[f"{base}.gate_proj.weight"] = torch.full(
                (expert.intermediate_dim, expert.hidden_dim), 10.0 + layer_idx
            )
            original_checkpoint[f"{base}.up_proj.weight"] = torch.full(
                (expert.intermediate_dim, expert.hidden_dim), 20.0 + layer_idx
            )
            original_checkpoint[f"{base}.down_proj.weight"] = torch.full(
                (expert.hidden_dim, expert.intermediate_dim), 30.0 + layer_idx
            )
    save_file(original_checkpoint, str(tmp_path / "model.safetensors"))

    switch = PhaseLayerAutoSwitch(
        model,
        {
            "prefill": {"ffn": {}},
            "decode": {"ffn": {"weight_mode": "fp", "bypass": True}},
        },
        model_name=str(tmp_path),
        weight_residency=weight_residency,
        quant_checkpoint_dir=str(quant_dir),
    ).enable()
    try:
        switch._on_phase_transition("decode", None)
        assert not switch._quant_weight_cache
        for layer_idx, layer in enumerate(model.model.layers):
            expert = layer.mlp.experts
            split = expert.intermediate_dim
            assert torch.all(expert.gate_up_proj[:, :split] == 10.0 + layer_idx)
            assert torch.all(expert.gate_up_proj[:, split:] == 20.0 + layer_idx)
            assert torch.all(expert.down_proj == 30.0 + layer_idx)

        switch._on_phase_transition("prefill", None)
        for layer_idx, layer in enumerate(model.model.layers):
            expert = layer.mlp.experts
            torch.testing.assert_close(expert.gate_up_proj, quant_expected[layer_idx][0])
            torch.testing.assert_close(expert.down_proj, quant_expected[layer_idx][1])
    finally:
        switch.disable()
    assert not switch._host_safetensor_handles


def test_phase_switch_disk_reload_restores_dense_gptq_weight_without_cpu_copy(tmp_path):
    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.proj = LinearMXUnified(
                2, 1, bias=False, config={**_act_cfg(), "bypass": True}
            )

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyLayer()])

    model = Tiny()
    with torch.no_grad():
        model.model.layers[0].mlp.proj.weight.fill_(1.0)
    save_file(
        {"model.layers.0.mlp.proj.weight": torch.full((1, 2), 5.0)},
        str(tmp_path / "model.safetensors"),
    )
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    save_file(
        {"mlp.proj.weight": torch.ones(1, 2)},
        str(quant_dir / "quantized_model_layer_0.safetensors"),
    )

    switch = PhaseLayerAutoSwitch(
        model,
        {
            "prefill": {"ffn": {"bypass": True}},
            "decode": {"ffn": {"weight_mode": "fp", "bypass": True}},
        },
        model_name=str(tmp_path),
        weight_residency="disk_reload",
        quant_checkpoint_dir=str(quant_dir),
    ).enable()
    try:
        assert switch._retained_file_cache_enabled
        assert len(switch._host_safetensor_handles) == 2
        assert (
            "requested=disk_reload, effective_io=retained_mmap_page_cache"
            in switch.summary()
        )
        switch._on_phase_transition("decode", None)
        assert torch.equal(model.model.layers[0].mlp.proj.weight, torch.full((1, 2), 5.0))
        assert not switch._quant_weight_cache
        switch._on_phase_transition("prefill", None)
        assert torch.equal(model.model.layers[0].mlp.proj.weight, torch.ones(1, 2))
    finally:
        switch.disable()


def test_phase_switch_host_cached_reuses_mmaps_and_restores_dense_gptq(tmp_path):
    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.proj = LinearMXUnified(
                2, 1, bias=False, config={**_act_cfg(), "bypass": True}
            )

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyLayer()])

    model = Tiny()
    with torch.no_grad():
        model.model.layers[0].mlp.proj.weight.fill_(1.0)
    save_file(
        {"model.layers.0.mlp.proj.weight": torch.full((1, 2), 5.0)},
        str(tmp_path / "model.safetensors"),
    )
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    save_file(
        {"mlp.proj.weight": torch.ones(1, 2)},
        str(quant_dir / "quantized_model_layer_0.safetensors"),
    )

    switch = PhaseLayerAutoSwitch(
        model,
        {
            "prefill": {"ffn": {"bypass": True}},
            "decode": {"ffn": {"weight_mode": "fp", "bypass": True}},
        },
        model_name=str(tmp_path),
        weight_residency="host_cached",
        quant_checkpoint_dir=str(quant_dir),
    ).enable()
    try:
        assert len(switch._host_safetensor_handles) == 2
        assert model.model.layers[0].mlp.proj.fp_weight is None
        retained_ids = {
            path: id(handle)
            for path, handle in switch._host_safetensor_handles.items()
        }
        switch._on_phase_transition("decode", None)
        assert torch.equal(
            model.model.layers[0].mlp.proj.weight, torch.full((1, 2), 5.0)
        )
        assert not switch._quant_weight_cache
        switch._on_phase_transition("prefill", None)
        assert torch.equal(model.model.layers[0].mlp.proj.weight, torch.ones(1, 2))
        assert retained_ids == {
            path: id(handle)
            for path, handle in switch._host_safetensor_handles.items()
        }
    finally:
        switch.disable()
    assert not switch._host_safetensor_handles


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


def test_linear_mx_unified_from_existing_reuses_parameters():
    source = nn.Linear(2, 3, bias=True)
    wrapped = LinearMXUnified.from_existing(source, {**_act_cfg(), "bypass": True})

    assert wrapped.weight is source.weight
    assert wrapped.bias is source.bias


def test_qwen3_gptq_cache_fast_path_wraps_only_projection_linears():
    model = _tiny_qwen3_moe()
    q_proj = model.model.layers[0].self_attn.q_proj
    router = model.model.layers[0].mlp.gate
    lm_head = model.lm_head
    q_proj_weight = q_proj.weight
    router_weight = router.weight
    lm_head_weight = lm_head.weight

    counts = apply_qwen3_gptq_cache_unified_wrappers(
        model,
        attn_linear_config=_act_cfg(),
        ffn_linear_config=_act_cfg(),
        qwen3_moe_attention_config={**_act_cfg(), "kv_cache": _act_cfg(), "softmax": _fp_cfg(), "rope": _fp_cfg()},
        qwen3_moe_experts_config={**_act_cfg(), **_fp_cfg()},
        qwen3_moe_rms_norm_config=_fp_cfg(),
    )

    wrapped_q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(wrapped_q_proj, LinearMXUnified)
    assert wrapped_q_proj.weight is q_proj_weight
    assert wrapped_q_proj.gptq
    assert counts["direct_gptq_linear"] > 0

    assert model.model.layers[0].mlp.gate is router
    assert model.model.layers[0].mlp.gate.weight is router_weight
    assert not isinstance(model.model.layers[0].mlp.gate, LinearMXUnified)
    assert model.lm_head is lm_head
    assert model.lm_head.weight is lm_head_weight
    assert not isinstance(model.lm_head, LinearMXUnified)


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
