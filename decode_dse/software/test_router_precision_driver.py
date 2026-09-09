from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.precision_bindings import build_decode_pass_args
from decode_dse.software.router_precision_ablation import (
    DRIVER_BATCH_SCHEMA,
    HELDOUT_SOURCE_SCHEMA,
    MODEL_REVISION,
    _heldout_source_records,
    _variant,
    path_identity,
)
from decode_dse.software.router_precision_driver import (
    HeldoutTokens,
    OfflineSnapshot,
    PREFILL_CACHE_SCHEMA,
    PREFILL_RECORD_SCHEMA,
    _CONTINUATION_DECODE_CONTRACT,
    _load_batch_manifest,
    _nonrouter_parameter_identity,
    _render_token_records,
    _resolve_offline_snapshot,
    _seal_offline_snapshot,
    _validate_prefill_index,
    measure_variant,
)
from decode_dse.software.sweep_plan import (
    profile_to_decode_quant_spec,
    write_immutable_json,
)


def _tiny_quantized_model():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeForCausalLM,
    )
    from chop.passes.module.transforms.quantize.quantize import (
        install_phase_context_pre_hooks,
        quantize_module_transform_pass,
    )

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        tie_word_embeddings=False,
    )
    torch.manual_seed(17)
    model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model.set_attn_implementation("eager")
    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        prefill = model.model(input_ids=prompt_ids, use_cache=True)
    record = HeldoutTokens(
        prompt_id="tiny-heldout",
        prompt_ids=(1, 2, 3),
        target_ids=(4, 5),
        first_token_id=4,
        cache_tensors={
            "layer_00_key": prefill.past_key_values.layers[0]
            .keys.detach()
            .cpu()
            .contiguous(),
            "layer_00_value": prefill.past_key_values.layers[0]
            .values.detach()
            .cpu()
            .contiguous(),
        },
    )
    profile = DecodePrecisionProfile.quantized(
        "MXINT8", "MXINT8", "MXINT8", "FP_E5M6"
    )
    pass_args = build_decode_pass_args(
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "cpu",
        profile_to_decode_quant_spec(profile),
    )
    pass_args["collapse_decode_banks"] = True
    model, _ = quantize_module_transform_pass(model, pass_args)
    install_phase_context_pre_hooks(model)
    return model, record


def test_tiny_qwen_moe_executes_paired_fixed_input_router_measurement():
    model, record = _tiny_quantized_model()
    body_before = _nonrouter_parameter_identity(model)
    head_before = model.lm_head.weight.data_ptr()
    agreement, end_to_end = measure_variant(
        model,
        (record,),
        job={"variant": _variant("MXINT8", "MXINT8", matrix_mlen=8)},
        heldout={"content_hash": "a" * 64},
        device="cpu",
        expected_layers=1,
        expected_experts=4,
        expected_top_k=2,
    )
    assert agreement["aggregate"]["layer_token_observations"] == 1
    assert agreement["aggregate"]["mean_topk_overlap"] <= 2.0
    assert len(agreement["layer_rows"]) == 1
    assert end_to_end["bf16_router_baseline"]["token_count"] == 1
    assert end_to_end["mx_router_candidate"]["token_count"] == 1
    assert len(agreement["shadow_router_input_hash"]) == 64
    token_stream_hash = end_to_end["teacher_forced_token_stream_hash"]
    assert end_to_end["bf16_router_baseline"][
        "teacher_forced_token_stream_hash"
    ] == token_stream_hash
    assert end_to_end["mx_router_candidate"][
        "teacher_forced_token_stream_hash"
    ] == token_stream_hash
    for field in (
        "first_decode_input_stream_hash",
        "scored_decode_suffix_stream_hash",
    ):
        assert len(end_to_end[field]) == 64
        assert end_to_end["bf16_router_baseline"][field] == end_to_end[field]
        assert end_to_end["mx_router_candidate"][field] == end_to_end[field]
    assert end_to_end["continuation_decode_contract"] == (
        _CONTINUATION_DECODE_CONTRACT
    )
    assert end_to_end["task_effects"]["status"] == "unsupported"
    assert _nonrouter_parameter_identity(model) == body_before
    assert model.lm_head.weight.data_ptr() == head_before


def test_tiny_paired_measurement_is_bitwise_restart_deterministic():
    first_model, first_record = _tiny_quantized_model()
    second_model, second_record = _tiny_quantized_model()
    kwargs = {
        "job": {"variant": _variant("E4M3", "E4M3", matrix_mlen=8)},
        "heldout": {"content_hash": "b" * 64},
        "device": "cpu",
        "expected_layers": 1,
        "expected_experts": 4,
        "expected_top_k": 2,
    }
    assert measure_variant(first_model, (first_record,), **kwargs) == measure_variant(
        second_model, (second_record,), **kwargs
    )


def test_batch_manifest_is_source_sealed_and_detects_request_tampering(tmp_path):
    request_path = tmp_path / "request.json"
    request = {
        "plan_hash": "a" * 64,
        "job": {"job_id": "rpa-test", "body_profile_id": "body-test"},
    }
    write_immutable_json(request_path, request)
    output = tmp_path / "result.json"
    driver = Path(__file__).with_name("router_precision_driver.py")
    batch_body = {
        "schema_version": DRIVER_BATCH_SCHEMA,
        "plan_hash": "a" * 64,
        "shard_index": 0,
        "driver": path_identity(driver),
        "jobs": [
            {
                "job_id": "rpa-test",
                "body_profile_id": "body-test",
                "request": path_identity(request_path),
                "output": str(output.resolve()),
            }
        ],
    }
    batch_path = tmp_path / "batch.json"
    write_immutable_json(batch_path, batch_body)
    assert _load_batch_manifest(batch_path) == ((request_path.resolve(), output.resolve()),)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="changed|checksum"):
        _load_batch_manifest(batch_path)


def test_heldout_source_schema_rejects_unsealed_calibration_overlap(tmp_path):
    source = {
        "schema_version": HELDOUT_SOURCE_SCHEMA,
        "records": [
            {
                "prompt_id": "p0",
                "messages": [{"role": "user", "content": "x"}],
                "continuation": "y",
                "used_for_router_calibration": True,
            }
        ],
    }
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        _heldout_source_records(path)


def test_render_token_records_rejects_manifest_source_count_mismatch(tmp_path):
    source = {
        "schema_version": HELDOUT_SOURCE_SCHEMA,
        "records": [
            {
                "prompt_id": f"p{index}",
                "messages": [{"role": "user", "content": "x"}],
                "continuation": "y",
                "used_for_router_calibration": False,
            }
            for index in range(2)
        ],
    }
    source_path = tmp_path / "heldout.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    heldout = {
        "dataset_source": path_identity(source_path),
        "records": [{"prompt_id": "p0"}],
        "decode_target_tokens_per_record": 1,
    }
    with pytest.raises(ValueError, match="record coverage differs"):
        _render_token_records(object(), heldout)


def test_prefill_cache_reuse_revalidates_exact_fresh_tokenization(tmp_path):
    cache_key = "a" * 64
    root = tmp_path / cache_key
    tensor_path = root / "records" / "p0.safetensors"
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"sealed-cache-tensors")
    metadata_path = root / "records" / "p0.json"
    metadata = {
        "schema_version": PREFILL_RECORD_SCHEMA,
        "cache_key": cache_key,
        "prompt_id": "p0",
        "prompt_ids": [1, 2, 3],
        "target_ids": [4, 5],
        "first_token_id": 4,
        "first_token_source": "sealed_continuation_token_0",
        "scored_suffix_start_index": 1,
        "prompt_tokens": 3,
        "layers": 48,
        "cache_tensors": path_identity(tensor_path),
    }
    write_immutable_json(metadata_path, metadata)
    index_path = root / "index.json"
    index_body = {
        "schema_version": PREFILL_CACHE_SCHEMA,
        "cache_key": cache_key,
        "model_name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "model_revision": MODEL_REVISION,
        "heldout_manifest_hash": "b" * 64,
        "prefill_dtype": "BF16",
        "serving_first_token_owner": "prefill",
        "evaluation_first_decode_input_source": (
            "teacher_forced_continuation_token_0"
        ),
        "evaluation_input_generated_by_bf16_model": False,
        "prefill_lm_head_executed": False,
        "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
        "record_count": 1,
        "records": [path_identity(metadata_path)],
    }
    write_immutable_json(index_path, index_body)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    heldout = {
        "content_hash": "b" * 64,
        "decode_target_tokens_per_record": 2,
        "records": [{"prompt_id": "p0"}],
    }
    exact = ({"prompt_id": "p0", "prompt_ids": (1, 2, 3), "target_ids": (4, 5)},)
    _validate_prefill_index(
        index, heldout=heldout, cache_key=cache_key, token_records=exact
    )
    with pytest.raises(ValueError, match="metadata differs"):
        _validate_prefill_index(
            index,
            heldout=heldout,
            cache_key=cache_key,
            token_records=(
                {"prompt_id": "p0", "prompt_ids": (1, 2, 3), "target_ids": (4, 6)},
            ),
        )
    with pytest.raises(ValueError, match="metadata differs"):
        _validate_prefill_index(
            index,
            heldout=heldout,
            cache_key=cache_key,
            token_records=(
                {"prompt_id": "p0", "prompt_ids": (1, 2, 9), "target_ids": (4, 5)},
            ),
        )


def test_offline_snapshot_rehash_detects_same_stat_shard_tamper(tmp_path):
    snapshot_root = tmp_path / "snapshots" / MODEL_REVISION
    snapshot_root.mkdir(parents=True)
    model_config = snapshot_root / "config.json"
    tokenizer_json = snapshot_root / "tokenizer.json"
    tokenizer_config = snapshot_root / "tokenizer_config.json"
    weight_index = snapshot_root / "model.safetensors.index.json"
    shard = snapshot_root / "model-00001-of-00001.safetensors"
    for path, payload in (
        (model_config, b"{}"),
        (tokenizer_json, b"{}"),
        (tokenizer_config, b"{}"),
        (weight_index, b'{"weight_map":{"x":"model-00001-of-00001.safetensors"}}'),
        (shard, b"sealed-shard"),
    ):
        path.write_bytes(payload)
    snapshot = OfflineSnapshot(
        cache_root=tmp_path,
        snapshot_root=snapshot_root,
        model_config=path_identity(model_config),
        tokenizer_json=path_identity(tokenizer_json),
        tokenizer_config=path_identity(tokenizer_config),
        weight_index=path_identity(weight_index),
        shard_count=1,
        weight_bytes=shard.stat().st_size,
        shard_paths=(shard,),
    )
    sealed = _seal_offline_snapshot(snapshot, tmp_path / "driver-cache")
    seal = json.loads(Path(sealed.content_seal["path"]).read_text(encoding="utf-8"))
    assert seal["tokenizer_json"] == snapshot.tokenizer_json
    before = shard.stat()
    shard.write_bytes(b"tamper-shard")
    os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert shard.stat().st_size == before.st_size
    assert shard.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="changed after content sealing"):
        _seal_offline_snapshot(snapshot, tmp_path / "driver-cache")


def test_offline_snapshot_resolves_mandatory_pinned_tokenizer_json(
    tmp_path, monkeypatch
):
    from transformers import utils as transformers_utils

    snapshot_root = tmp_path / "snapshots" / MODEL_REVISION
    snapshot_root.mkdir(parents=True)
    filenames = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    paths = {name: snapshot_root / name for name in filenames}
    shard = snapshot_root / "model-00001-of-00001.safetensors"
    paths["config.json"].write_text("{}", encoding="utf-8")
    paths["tokenizer.json"].write_text("{}", encoding="utf-8")
    paths["tokenizer_config.json"].write_text("{}", encoding="utf-8")
    paths["model.safetensors.index.json"].write_text(
        json.dumps({"weight_map": {"x": shard.name}}), encoding="utf-8"
    )
    shard.write_bytes(b"weights")

    def cached_file(model_name, filename, **kwargs):
        assert kwargs["revision"] == MODEL_REVISION
        assert kwargs["local_files_only"] is True
        path = paths[filename]
        return str(path) if path.is_file() else None

    monkeypatch.setattr(transformers_utils, "cached_file", cached_file)
    snapshot = _resolve_offline_snapshot(tmp_path)
    assert snapshot.tokenizer_json == path_identity(paths["tokenizer.json"])

    outside = tmp_path / "tokenizer.json"
    outside.write_text("{}", encoding="utf-8")
    paths["tokenizer.json"] = outside
    with pytest.raises(ValueError, match="pinned commit"):
        _resolve_offline_snapshot(tmp_path)

    paths["tokenizer.json"] = snapshot_root / "missing-tokenizer.json"
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        _resolve_offline_snapshot(tmp_path)
