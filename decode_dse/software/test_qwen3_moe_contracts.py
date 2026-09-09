"""Qwen3-MoE architecture, quantization, and thinking protocol contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from decode_dse.legality import StackValidity
from decode_dse.profiles import (
    LEGACY_PROFILE_SCHEMA,
    PROFILE_SCHEMA,
    DecodePrecisionProfile,
    enumerate_decode_profiles,
)
from decode_dse.software.benchmark_runner import (
    SAMPLED_TASK_METRIC_ID,
    PublicationConfiguration,
    PublicationItemMetric,
    PublicationProtocol,
    PublicationSplitEvidence,
)
from decode_dse.software.precision_bindings import (
    DecodeQuantSpec,
    build_decode_pass_args,
    decode_binding_expectations,
)
from decode_dse.software.publication_launch import (
    build_publication_execution_config,
)
from decode_dse.software.runtime_environment import decoder_parameter_count
from decode_dse.software.runtime_environment import _observe_qwen3_moe_runtime_abi
from decode_dse.software.sweep import (
    _compiler_model_blockers,
    _publication_pipeline_config,
)


_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "qwen3_30b_a3b_thinking_2507.json"
)
_CHAT_TEMPLATE_SHA256 = (
    "aebb52d7099cbaba93c574834d9a85ac971bac93c9bdabea4532122288f07590"
)


def _architecture() -> dict:
    return json.loads(_CONFIG.read_text(encoding="utf-8"))["model_architecture"]


def _token_budgets(task_budget: int = 32768) -> tuple[tuple[str, int], ...]:
    return (
        ("wikitext2", 32768),
        ("ifeval", task_budget),
        ("gsm8k", task_budget),
        ("ruler", 8192),
    )


def test_target_config_is_full_exhaustive_thinking_grid():
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    search = config["search"]

    assert config["model_name"] == "Qwen/Qwen3-30B-A3B-Thinking-2507"
    assert config["publication"]["thinking_mode"] == "required"
    assert config["publication"]["enable_thinking"] is True
    assert config["publication"]["greedy"] is False
    assert config["publication"]["temperature"] == 0.6
    assert config["publication"]["top_p"] == 0.95
    assert config["publication"]["top_k"] == 20
    assert config["publication"]["repetitions"] == 3
    assert config["publication"]["token_budgets"]["gsm8k"] == 32768
    assert config["publication"]["token_budgets"]["ruler"] == 8192
    assert config["runtime_requirements"]["minimum_package_versions"][
        "transformers"
    ] == "5.5.0"
    assert config["runtime_requirements"]["exact_package_versions"] == {
        "transformers": "5.5.0"
    }
    assert config["model_architecture"]["norm_topk_prob"] is True
    assert "shared_expert_intermediate_size" not in config["model_architecture"]
    assert "num_shared_experts" not in config["model_architecture"]
    assert len(search["weight_w"]) == len(search["act_w"]) == len(search["kv"]) == 8
    assert "declared_exclusions" not in search
    assert search["expected_quantized_profiles"] == 8 * 8 * 8 * 6
    assert search["expected_vector_bf16_controls"] == 8 * 8 * 8
    assert search["expected_total_profiles"] == 3585
    assert config["hardware_space"]["TP"] == [1, 2, 4, 8]
    assert config["hardware_space"]["KVP"] == [1, 2, 4]
    assert config["hardware_space"]["EXPERT_PARALLEL_MODE"] == [
        "tensor_parallel",
        "expert_id_parallel",
    ]
    assert config["hardware_space"]["ALLOW_RANK_LOCAL_MLEN_PADDING"] is True
    campaign = config["serving_queue_campaign"]
    assert campaign["schema_version"] == "decode-serving-queue-campaign/v1"
    assert campaign["publication_contract_sealed"] is True
    assert [
        (item["prompt_tokens"], item["generation_tokens"])
        for item in campaign["workload_buckets"]
    ] == [
        (512, 512),
        (4096, 4096),
        (4096, 16384),
        (16384, 32768),
        (32768, 32768),
        (131072, 8192),
    ]
    assert config["use_rotation"] is False
    assert config["rotation_refinement_blocker"][
        "selector_may_emit_rotation"
    ] is False
    assert config["refinement"]["require_symmetric_kv"] is True
    numerical_only = config["numerical_only_execution"]
    assert numerical_only["schema_version"] == "decode-numerical-only-contract/v1"
    assert numerical_only["required_shards"] == 4
    assert numerical_only["allowed_stages"] == ["preflight", "numerical-screen"]
    assert numerical_only["strict_pipeline_remains_fail_closed"] is True
    assert numerical_only["publication_rankable"] is False
    assert numerical_only["hardware_rankable"] is False
    assert numerical_only["selection_eligible"] is False
    repricing = config["router_trace_repricing"]
    assert repricing["batch_source"] == "hardware_space.BATCH"
    assert repricing["supported_override_fields"] == [
        "moe_unique_experts_per_step",
        "moe_routing_imbalance_factor",
    ]
    assert repricing["resident_expert_storage"] == "all_128_experts"
    assert repricing["execution_scope"] == (
        "legacy_aggregate_analytic_route_sensitivity_only"
    )
    assert repricing["publication_rankable"] is False
    assert repricing["hardware_rankable"] is False
    assert repricing["selection_eligible"] is False
    profiles = enumerate_decode_profiles()
    assert len(profiles) == 3585
    assert len({profile.profile_id for profile in profiles}) == 3585
    assert {profile.schema_version for profile in profiles} == {PROFILE_SCHEMA}


def test_target_chat_template_digest_and_reasoning_render_paths_are_pinned():
    from transformers.utils.chat_template_utils import _compile_jinja_template

    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    publication = config["publication"]
    asset_path = _CONFIG.parents[2] / publication["chat_template_asset"]
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    template = asset["chat_template"]

    assert publication["chat_template_sha256"] == _CHAT_TEMPLATE_SHA256
    assert asset["chat_template_sha256"] == _CHAT_TEMPLATE_SHA256
    assert hashlib.sha256(template.encode("utf-8")).hexdigest() == (
        _CHAT_TEMPLATE_SHA256
    )

    renderer = _compile_jinja_template(template)
    for reasoning_field in ("reasoning", "reasoning_content"):
        rendered = renderer.render(
            messages=[
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "answer",
                    reasoning_field: "reasoning trace",
                },
            ],
            tools=None,
            add_generation_prompt=False,
        )
        assert "<think>\nreasoning trace\n</think>\n\nanswer" in rendered

    prompt = renderer.render(
        messages=[{"role": "user", "content": "question"}],
        tools=None,
        add_generation_prompt=True,
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_target_parameter_count_and_binding_counts_use_fused_experts():
    architecture = _architecture()

    assert decoder_parameter_count(architecture) == 30_532_122_624
    detected_experts, compiler_blockers = _compiler_model_blockers(architecture)
    assert detected_experts == 128
    assert compiler_blockers == [
        "mixture_of_experts_trace_evidence_not_bound"
    ]
    expectations = decode_binding_expectations(architecture)
    assert expectations.dense_layers == 0
    assert expectations.moe_layers == 48
    assert expectations.sealed_weight_modules == 5 * 48 + 1
    assert expectations.binding_targets == 12 * 48 + 2

    dense = dict(architecture)
    dense.update(model_type="qwen3", num_experts=1)
    dense_expectations = decode_binding_expectations(dense)
    assert dense_expectations.sealed_weight_modules == 7 * 48 + 1
    assert dense_expectations.binding_targets == 14 * 48 + 2


def test_decode_selector_includes_fused_experts_and_local_head_but_excludes_router():
    args = build_decode_pass_args(
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "cuda:0",
        DecodeQuantSpec(),
    )
    ffn_pattern = next(
        pattern for pattern in args if "gate_proj|up_proj|down_proj|experts" in pattern
    )

    assert re.fullmatch(ffn_pattern, "model.layers.0.mlp.experts")
    assert re.fullmatch(ffn_pattern, "model.layers.0.mlp.gate") is None
    head_pattern = next(pattern for pattern in args if pattern == "lm_head$")
    head = args[head_pattern]["config"]["decode"]
    assert re.fullmatch(head_pattern, "lm_head")
    assert head["weight_width"] == 4
    assert head["data_in_width"] == 8
    assert head["output_format"] == "BF16"
    assert head["local_head_contract"] == DecodeQuantSpec().local_head_contract
    assert "gptq" not in head


def test_profile_serializes_local_head_w_a_and_bf16_selection_boundary():
    from decode_dse.software.sweep_plan import profile_to_decode_quant_spec

    profile = DecodePrecisionProfile.quantized(
        "MXINT4", "MXINT8", "MXINT8", "FP_E3M2"
    )
    contract = profile.local_head_contract

    assert "decode_lm_head" in profile.weight_operators
    assert "decode_lm_head" in profile.activation_operators
    assert profile.bf16_operators == ("embedding",)
    assert contract["weight_format"] == "MXINT4"
    assert contract["activation_format"] == "MXINT8"
    assert contract["accumulator_rule"] == (
        "plena_fixed16_16_accumulate_truncate"
    )
    assert contract["output_rule"] == (
        "family_specific_per_mlen_to_profile_vector_then_fixed16_16_wrap_then_profile_vector_truncate_bf16_store"
    )
    assert contract["arithmetic_chain"] == [
        "block8_mxint_quantized_operands_materialized_in_fp32",
        "fp32_matmul_reduction_per_mlen_partition",
        "each_mlen_reduction_rne_to_FP_E3M2_storage_fp",
        "signed_fixed16_16_cross_instruction_accumulate_wrap",
        "truncate_to_FP_E3M2_matrix_writeout",
        "store_already_rounded_values_in_bf16_logit_container",
    ]
    assert contract["matrix_output_format"] == "FP_E3M2"
    assert contract["matrix_mlen"] == profile.matrix_mlen == 1024
    assert contract["logit_container_format"] == "BF16"
    assert contract["bf16_container_precision_recovery"] is False
    assert contract["greedy_selection_rule"] == (
        "argmax_lowest_token_id_on_tie"
    )
    assert contract["serving_selection"]["full_batch_vocab_vsram_required"] is False
    assert contract["offline_evaluation"]["materialization"] == "full_bf16_logits"
    quant_spec = profile_to_decode_quant_spec(profile)
    assert quant_spec.matrix_mlen == profile.matrix_mlen
    assert quant_spec.local_head_contract == contract
    assert DecodePrecisionProfile.from_dict(profile.to_dict()) == profile

    reference = DecodePrecisionProfile.bf16_reference()
    assert reference.bf16_operators == ("embedding", "lm_head")
    assert "lm_head" not in reference.weight_operators
    assert reference.local_head_contract["phase_ownership"][
        "decode_head_policy"
    ] == "bf16_reference"


def test_attention_only_rotation_keeps_local_head_rtn_unrotated():
    rotated = DecodeQuantSpec(use_rotation=True)

    assert rotated.gptq_weights is True
    assert rotated.local_head_contract["weight_method"] == "rtn"
    assert rotated.local_head_contract["weight_preconditioning"] == "none"
    head = build_decode_pass_args(
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "cuda:0",
        rotated,
    )["lm_head$"]["config"]["decode"]
    assert head["local_head_contract"] == rotated.local_head_contract
    assert "gptq" not in head


def test_local_head_mxfp_and_mixed_family_contracts_are_explicit():
    mxfp = DecodePrecisionProfile.quantized(
        "E2M1", "E4M3", "E3M4", "FP_E5M6"
    ).local_head_contract
    assert mxfp["operand_family_binding"] == (
        "product_cast_to_m_fp_then_fixed16_16_bank"
    )
    assert mxfp["partial_conversion_rule"] == (
        "quantized_operands_fp32_matmul_per_mlen_rne_vector_then_fixed16_16_bank"
    )
    assert mxfp["numerical_oracle_rule"] == mxfp["partial_conversion_rule"]
    assert mxfp["hardware_bit_parity_verified"] is False
    assert mxfp["partial_conversion_format"] == "FP_E5M6"
    assert mxfp["operand_family_deployment_supported"] is True
    assert mxfp["arithmetic_chain"] == [
        "block8_mxfp_quantized_operands_materialized_in_fp32",
        "fp32_matmul_reduction_per_mlen_partition",
        "each_mlen_reduction_rne_to_FP_E5M6_storage_fp",
        "signed_fixed16_16_cross_instruction_accumulate_wrap",
        "truncate_to_FP_E5M6_matrix_writeout",
        "store_already_rounded_values_in_bf16_logit_container",
    ]

    mixed = DecodePrecisionProfile.quantized(
        "MXINT8", "E4M3", "MXINT4", "FP_E3M2"
    ).local_head_contract
    assert mixed["operand_family_binding"] == (
        "deployment_unsupported_without_trace_evidence"
    )
    assert mixed["operand_family_deployment_supported"] is False


def test_matrix_mlen_is_part_of_v2_numerical_identity():
    from decode_dse.software.sweep_plan import profile_to_decode_quant_spec

    base = DecodePrecisionProfile.quantized(
        "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
    )
    wider = DecodePrecisionProfile.quantized(
        "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
    )
    wider = dataclasses.replace(wider, matrix_mlen=2048)

    assert base.profile_id != wider.profile_id
    assert wider.to_dict()["matrix_mlen"] == 2048
    assert wider.to_dict()["numerical_oracle"]["matrix_mlen"] == 2048
    assert wider.numerical_oracle_contract["hardware_bit_parity_verified"] is False
    assert wider.local_head_contract["matrix_mlen"] == 2048
    assert profile_to_decode_quant_spec(wider).matrix_mlen == 2048
    assert DecodePrecisionProfile.from_dict(wider.to_dict()) == wider


def test_v2_profile_rejects_a_tampered_numerical_oracle_contract():
    profile = DecodePrecisionProfile.quantized(
        "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
    )
    serialized = profile.to_dict()
    serialized["numerical_oracle"]["hardware_bit_parity_verified"] = True

    with pytest.raises(ValueError, match="numerical-oracle"):
        DecodePrecisionProfile.from_dict(serialized)


def test_legacy_llama_bf16_head_profile_round_trips_with_stable_identity():
    legacy = DecodePrecisionProfile(
        kind="quantized",
        weight_format="MXINT8",
        activation_format="MXINT8",
        key_format="MXINT4",
        value_format="MXINT4",
        vector_format="FP_E6M5",
        weight_operators=("attention_linear", "ffn_linear"),
        activation_operators=(
            "attention_linear",
            "ffn_linear",
            "qk_matmul",
            "pv_matmul",
        ),
        bf16_operators=("embedding", "lm_head"),
        schema_version=LEGACY_PROFILE_SCHEMA,
    )
    serialized = legacy.to_dict()

    assert "local_head" not in serialized
    assert legacy.profile_id == (
        "dqp-1ca8a18d60263c134464a2a280dc14527daf75053b1c831ed9ac160300c3d3e3"
    )
    restored = DecodePrecisionProfile.from_dict(serialized)
    assert restored.to_dict() == serialized
    assert restored.profile_id == legacy.profile_id
    assert restored.local_head_contract["weight_format"] == "BF16"


def test_real_decode_builder_transforms_tiny_fused_qwen3_moe():
    import torch
    from transformers.models.qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )
    from chop.nn.quantized.modules.qwen3_moe import (
        Qwen3MoeDecoderLayerMinifloat,
        Qwen3MoeExpertsMXInt,
        Qwen3MoeSparseMoeBlockMinifloat,
        Qwen3MoeTopKRouterBF16,
    )
    from chop.nn.quantized.modules.linear import LinearMXInt
    from chop.nn.quantized.modules.phase_context import force_runtime_phase
    from chop.passes.module.transforms.quantize.quantize import (
        quantize_module_transform_pass,
    )

    abi = _observe_qwen3_moe_runtime_abi()
    assert abi["passed"] is True
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16).eval()
    args = build_decode_pass_args(
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "cpu",
        DecodeQuantSpec(matrix_mlen=8),
    )
    model, _ = quantize_module_transform_pass(model, args)

    layer = model.model.layers[0]
    assert isinstance(layer, Qwen3MoeDecoderLayerMinifloat)
    assert isinstance(layer.mlp, Qwen3MoeSparseMoeBlockMinifloat)
    assert isinstance(layer.mlp.gate, Qwen3MoeTopKRouterBF16)
    assert layer.mlp.gate.weight.dtype == torch.bfloat16
    assert isinstance(layer.mlp.experts, Qwen3MoeExpertsMXInt)
    assert isinstance(model.lm_head, LinearMXInt)
    assert model.lm_head.decode_config["output_format"] == "BF16"
    assert model.lm_head.decode_config["local_head_contract"] == (
        DecodeQuantSpec(matrix_mlen=8).local_head_contract
    )
    assert (
        model.model.embed_tokens.weight.data_ptr()
        != model.lm_head.weight.data_ptr()
    )
    assert layer._mase_phase_hook_installed is True
    with torch.no_grad():
        output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    assert output.shape == (1, 3, 64)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    with force_runtime_phase("decode"), torch.no_grad():
        tied_logits = model.lm_head(torch.zeros(1, 1, 16, dtype=torch.bfloat16))
    assert tied_logits.dtype == torch.bfloat16
    assert int(tied_logits.argmax(dim=-1).item()) == 0


def test_simulator_bridge_preserves_moe_workload_provenance():
    from decode_dse.simulator_bridge import DecodeSimulator

    simulator = DecodeSimulator("qwen3-30b-a3b-thinking-2507")
    precision = simulator.make_precision(
        attn_w=4,
        ffn_w=4,
        kv=4,
        act_w=4,
    )
    metrics = simulator.evaluate(
        precision,
        batch=2,
        input_seq=16,
        output_seq=2,
        hw_over=simulator.shipped_over(precision),
        n_chips=1,
        stride=1,
        hbm_gen="HBM2",
        hbm_channels=8,
    )

    assert metrics.moe_workload is not None
    assert metrics.moe_workload["physical_route_assignments_per_step"] == 16
    assert metrics.moe_workload["provenance"]["publication_rankable"] is False


def test_thinking_only_protocol_rejects_disable_and_short_task_budgets():
    shared = dict(
        model_name="Qwen/Qwen3-30B-A3B-Thinking-2507",
        model_revision="1" * 40,
        tokenizer_revision="1" * 40,
        chat_template_sha256="2" * 64,
        greedy=False,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        paired_seeds=(20260725, 20260726, 20260727),
        repetitions=3,
    )
    protocol = PublicationProtocol(
        **shared,
        thinking_mode="required",
        enable_thinking=True,
        token_budgets=_token_budgets(),
    )
    assert protocol.enable_thinking is True

    with pytest.raises(ValueError, match="thinking-only"):
        PublicationProtocol(
            **shared,
            thinking_mode="disabled",
            enable_thinking=False,
            token_budgets=_token_budgets(),
        )
    with pytest.raises(ValueError, match="below their minimums"):
        PublicationProtocol(
            **shared,
            thinking_mode="required",
            enable_thinking=True,
            token_budgets=_token_budgets(2048),
        )


def test_dynamic_deduplicated_candidate_role_and_termination_evidence():
    configuration = PublicationConfiguration(
        role="pareto_07_best-power_gptq",
        profile=DecodePrecisionProfile.quantized(
            "MXINT8", "MXINT4", "MXINT4", "FP_E3M2"
        ),
        validity=StackValidity(True, True, True, True, True),
    )
    assert configuration.role == "pareto_07_best-power_gptq"

    metric = PublicationItemMetric(
        item_id="gsm8k-0",
        score=100.0,
        generation_terminations=(
            (20260725, 1024, 900, "eos"),
            (20260726, 1100, 950, "eos"),
            (20260727, 980, 850, "stop"),
        ),
    )
    assert PublicationItemMetric.from_dict(metric.to_dict()) == metric
    evidence = PublicationSplitEvidence(
        mode="seeded_sampled_cached_generation",
        metric_id=SAMPLED_TASK_METRIC_ID,
        handoff_token_source="prefill_sampled",
        prefill_precision="BF16",
        transferred_kv_precision="BF16",
        first_token_owner="prefill",
        decode_q_lengths=(1,),
        exact_cache_positions=True,
        exact_one_entry_growth=True,
        independent_caches=True,
        admission_count_per_prompt=1,
        cache_free_calls=0,
        full_item_coverage=True,
    )
    assert PublicationSplitEvidence.from_dict(evidence.to_dict()) == evidence


def test_publication_stays_disabled_until_all_late_bindings_exist():
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    assert config["publication_pipeline"]["resources"]["publication_enabled"] is False
    with pytest.raises(ValueError, match="incomplete"):
        build_publication_execution_config(config, {})

    bound = build_publication_execution_config(
        config,
        {
            "input_manifest": "/study/publication/inputs.json",
            "prefill_artifact_root": "/study/publication/prefill",
            "driver_output_root": "/study/publication/driver-output",
            "driver": {"command": ["/study/bin/driver", "{request}", "{result}"]},
            "decode_banks": {"bf16": {"path": "/study/banks/bf16"}},
        },
    )
    assert bound["publication_pipeline"]["resources"]["publication_enabled"] is True
    assert "publication_launch_blocker" not in bound
    _publication_pipeline_config(bound)


def test_exploratory_pipeline_keeps_repricing_but_omits_publication_selection(
    tmp_path,
):
    from decode_dse.software.sweep import _build_manifest, build_pipeline
    from decode_dse.software.sweep_plan import GPUBaselinePlan, build_run_plan

    repository = Path(__file__).resolve().parents[2]
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    manifest = _build_manifest(config, repository)
    plan = build_run_plan(
        manifest,
        device_labels=("b200",),
        gpu_baseline=GPUBaselinePlan.from_config(config["gpu_baseline"]),
    )
    commands = build_pipeline(
        config=_CONFIG,
        output_dir=tmp_path / "workspace",
        device_label="b200",
        gpus=("0", "1", "2", "3"),
        plan=plan,
    )
    names = tuple(command.name for command in commands)

    assert "refined-hardware-study" in names
    assert "publication-configurations" not in names
    assert "publication-benchmarks" not in names
