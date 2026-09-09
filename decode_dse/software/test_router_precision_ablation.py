from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from decode_dse.legality import StackValidity
from decode_dse.manifest import (
    QuantizerProvenance,
    QuantizerSource,
    ResolvedImportOrigin,
    build_exhaustive_manifest,
    write_manifest,
)
from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.router_precision_ablation import (
    AGREEMENT_SCHEMA,
    ANCESTRY_SCHEMA,
    DRIVER_RECEIPT_SCHEMA,
    HELDOUT_SCHEMA,
    HELDOUT_SOURCE_SCHEMA,
    MODEL_NAME,
    MODEL_REVISION,
    PREFILL_CACHE_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    RouterAgreementAccumulator,
    _BF16_ROUTER_CONTRACT,
    _CONTINUATION_DECODE_CONTRACT,
    _classification,
    build_plan,
    build_completion,
    build_router_variant_pass_args,
    canonical_hash,
    derive_bf16_router_binding,
    materialize_ancestry_from_current_rows,
    materialize_heldout_manifest,
    path_identity,
    run_shard,
    validate_ancestry,
    validate_result,
)
from decode_dse.software.sweep_plan import write_immutable_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "decode_dse/configs/qwen3_30b_a3b_thinking_2507.json"
def _artifacts(tmp_path: Path, *, profiles: int = 1):
    dataset = tmp_path / "heldout.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": HELDOUT_SOURCE_SCHEMA,
                "records": [
                    {
                        "prompt_id": "heldout-0001",
                        "messages": [
                            {"role": "user", "content": "held out"}
                        ],
                        "continuation": "answer tokens",
                        "used_for_router_calibration": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    heldout_path = tmp_path / "heldout-manifest.json"
    heldout = materialize_heldout_manifest(
        config_path=CONFIG,
        source_path=dataset,
        decode_target_tokens=2,
        output_path=heldout_path,
    )

    body_records = []
    derivation = None
    formats = (
        ("MXINT4", "MXINT8", "MXINT4", "FP_E3M2"),
        ("E4M3", "E5M2", "E4M3", "FP_E5M6"),
    )
    for ordinal in range(profiles):
        profile = DecodePrecisionProfile.quantized(*formats[ordinal])
        current_derivation = derive_bf16_router_binding(profile)
        if derivation is None:
            derivation = current_derivation
        assert derivation == current_derivation
        bank = tmp_path / f"bank-{ordinal}.bin"
        bank.write_bytes(f"bank-{ordinal}".encode())
        bank_identity = path_identity(bank)
        quantizer_hash = f"{ordinal + 2:x}" * 64
        quantizer_hash = quantizer_hash[:64]
        journal_body = {
            "schema_version": "decode-sweep-result",
            "profile_id": profile.profile_id,
            "state": "succeeded",
        }
        journal_hash = canonical_hash(journal_body)
        journal = tmp_path / f"source-journal-{ordinal}.jsonl"
        journal.write_text(
            json.dumps(journal_body | {"record_hash": journal_hash}) + "\n",
            encoding="utf-8",
        )
        journal_identity = path_identity(journal)
        source_row = {
            "profile_id": profile.profile_id,
            "status": "success",
            "baseline_router": dict(_BF16_ROUTER_CONTRACT),
            "baseline_router_derivation_hash": canonical_hash(derivation),
            "quantizer_provenance_hash": quantizer_hash,
            "body_bank_binding_hash": bank_identity["sha256"],
            "source_journal": journal_identity,
            "source_journal_record_hash": journal_hash,
        }
        source_path = tmp_path / f"source-row-{ordinal}.json"
        write_immutable_json(source_path, source_row)
        body_records.append(
            {
                "body_profile_id": profile.profile_id,
                "body_profile": profile.to_dict(),
                "baseline_router": dict(_BF16_ROUTER_CONTRACT),
                "body_bank": bank_identity,
                "source_numerical_result": path_identity(source_path),
                "source_numerical_journal": journal_identity,
                "source_journal_record_hash": journal_hash,
                "quantizer_provenance_hash": quantizer_hash,
                "source_result_row_hash": canonical_hash(source_row),
                "source_result_status": "success",
            }
        )
    ancestry = {
        "schema_version": ANCESTRY_SCHEMA,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "canonical_profile_count": 3585,
        "baseline_router": dict(_BF16_ROUTER_CONTRACT),
        "bf16_router_source": dict(derivation["mase_router_source"]),
        "router_contract_derivation": derivation,
        "body_baselines": body_records,
    }
    ancestry_path = tmp_path / "ancestry.json"
    write_immutable_json(ancestry_path, ancestry)
    return heldout_path, ancestry_path


def test_plan_is_deterministic_isolated_and_costs_router_roles_separately(tmp_path):
    heldout, ancestry = _artifacts(tmp_path, profiles=2)
    first = build_plan(
        config_path=CONFIG,
        heldout_path=heldout,
        ancestry_path=ancestry,
    )
    second = build_plan(
        config_path=CONFIG,
        heldout_path=heldout,
        ancestry_path=ancestry,
    )
    assert first == second
    assert first["schema_version"] == PLAN_SCHEMA
    assert first["canonical_body_profile_count"] == 3585
    assert first["canonical_body_profile_census_modified"] is False
    assert [row["weight_format"] for row in first["variants"]] == [
        "MXINT8",
        "E4M3",
        "E5M2",
    ]
    assert len(first["jobs"]) == 6
    assert {row["shard_index"] for row in first["jobs"]} == {0, 1}
    assert all(
        len({row["shard_index"] for row in first["jobs"] if row["body_profile_id"] == profile_id}) == 1
        for profile_id in {row["body_profile_id"] for row in first["jobs"]}
    )
    assert all(row["paired_baseline_router"] == _BF16_ROUTER_CONTRACT for row in first["jobs"])
    receipt = first["jobs"][0]["cost_receipt"]
    assert receipt["body_precision_unchanged"]["key_format"] == receipt[
        "body_precision_unchanged"
    ]["value_format"]
    assert receipt["offline_weight_conversion"]["conversion_events"] > 0
    assert receipt["runtime_activation_conversion_per_token"]["conversion_events"] > 0
    assert receipt["storage"]["weight_total_bytes"] > 0
    assert receipt["traffic_per_token"]["hbm"]["router_input_activation_bytes"] == 0
    assert (
        receipt["traffic_per_token"]["on_chip_vector_sram_and_conversion"][
            "bf16_logits_bytes"
        ]
        > 0
    )
    assert receipt["compute_per_token"]["logical_macs"] > 0
    assert (
        "bf16_router_has_no_calibrated_dc_event_signature_and_may_not_be_proxied_as_mx"
        in receipt["blockers"]
    )
    assert receipt["classification"]["selection_eligible"] is False
    pass_args = build_router_variant_pass_args(first["variants"][0])
    assert set(pass_args) == {"by", r"model\.layers\.\d+\.mlp\.gate$"}
    gate_config = pass_args[r"model\.layers\.\d+\.mlp\.gate$"]["config"]
    assert gate_config["decode"]["router_weight_format"] == "MXINT8"
    assert gate_config["decode"]["router_activation_format"] == "MXINT8"
    assert "lm_head" not in json.dumps(pass_args)
    assert "kv_cache" not in json.dumps(pass_args)


def test_ancestry_rejects_source_row_that_is_not_the_bound_bf16_control(tmp_path):
    _, ancestry_path = _artifacts(tmp_path)
    ancestry = json.loads(ancestry_path.read_text(encoding="utf-8"))
    ancestry.pop("content_hash")
    record = ancestry["body_baselines"][0]
    record["source_result_row_hash"] = "f" * 64
    with pytest.raises(ValueError, match="row hash differs"):
        validate_ancestry(ancestry)


def test_current_numerical_row_materializes_implicit_bf16_router_ancestry(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    provenance = QuantizerProvenance(
        sources=(
            QuantizerSource(
                component="test",
                path="mase/src/chop/nn/quantizers/mxint/fake.py",
                sha256="a" * 64,
            ),
        ),
        resolved_imports=(
            ResolvedImportOrigin(
                module="chop.nn.quantizers.mxint.fake",
                path="mase/src/chop/nn/quantizers/mxint/fake.py",
            ),
        ),
    )
    manifest = build_exhaustive_manifest(
        MODEL_NAME,
        MODEL_REVISION,
        config["model_architecture"],
        provenance,
        tokenizer_revision=MODEL_REVISION,
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    entry = next(item for item in manifest.entries if item.profile.kind == "quantized")
    row_body = {
        "schema_version": "decode-sweep-result",
        "manifest_hash": manifest.canonical_hash,
        "ordinal": entry.ordinal,
        "profile_id": entry.profile_id,
        "profile": entry.profile.to_dict(),
        "weight_format": entry.profile.weight_format,
        "attempt": 1,
        "state": "succeeded",
        "validity": StackValidity(software_valid=True).to_dict(),
        "result": {
            "mean_token_nll": 2.0,
            "weight_bank": {
                "identity_fingerprint": "b" * 64,
                "structure_fingerprint": "c" * 64,
            },
            "runtime_environment": {"mase_tree_sha256": "d" * 64},
        },
        "artifacts": [],
        "error_class": None,
        "error_message": None,
        "traceback": None,
        "runtime_seconds": 1.0,
        "completed_at": "2026-08-20T00:00:00Z",
    }
    record_hash = canonical_hash(row_body)
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "MXINT2.jsonl").write_text(
        json.dumps(row_body | {"record_hash": record_hash}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "ancestry-derived.json"
    ancestry = materialize_ancestry_from_current_rows(
        manifest_path=manifest_path,
        results_root=results_root,
        record_hashes=(record_hash,),
        output_path=output,
    )
    assert ancestry["router_contract_derivation"][
        "direct_gate_selector_present"
    ] is False
    record = ancestry["body_baselines"][0]
    assert record["body_bank"]["materialization"] == (
        "rebuild_from_sealed_manifest_profile/v1"
    )
    source = json.loads(
        Path(record["source_numerical_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert source["baseline_router"] == _BF16_ROUTER_CONTRACT
    assert source["source_journal_record_hash"] == record_hash


def _probabilities():
    baseline = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float32
    )
    candidate = torch.tensor(
        [[0.39, 0.31, 0.2, 0.1], [0.1, 0.2, 0.31, 0.39]], dtype=torch.float32
    )
    base_idx = torch.tensor([[0, 1], [3, 2]])
    candidate_idx = torch.tensor([[1, 0], [3, 2]])
    return baseline, base_idx, candidate, candidate_idx


def test_router_agreement_accumulator_is_deterministic_and_layer_complete():
    kwargs = {
        "heldout_manifest_hash": "e" * 64,
        "shadow_router_input_hash": "f" * 64,
        "layers": 2,
        "experts": 4,
        "top_k": 2,
    }
    first = RouterAgreementAccumulator(**kwargs)
    second = RouterAgreementAccumulator(**kwargs)
    values = _probabilities()
    for layer in range(2):
        first.update(layer, *values)
        second.update(layer, *values)
    one = first.finalize()
    two = second.finalize()
    assert one == two
    assert one["schema_version"] == AGREEMENT_SCHEMA
    assert one["aggregate"]["topk_set_agreement"] == 1.0
    assert one["aggregate"]["topk_order_agreement"] == 0.5
    assert one["aggregate"]["mean_topk_overlap"] == 2.0
    incomplete = RouterAgreementAccumulator(**kwargs)
    incomplete.update(0, *values)
    with pytest.raises(ValueError, match="missing layers"):
        incomplete.finalize()


def _target_agreement(heldout_manifest_hash: str):
    accumulator = RouterAgreementAccumulator(
        heldout_manifest_hash=heldout_manifest_hash,
        shadow_router_input_hash="b" * 64,
    )
    probability = torch.full((1, 128), 1.0 / 128, dtype=torch.float32)
    index = torch.arange(8).reshape(1, 8)
    for layer in range(48):
        accumulator.update(layer, probability, index, probability, index)
    return accumulator.finalize()


def test_result_validation_recomputes_paired_nll_delta(tmp_path):
    heldout, ancestry = _artifacts(tmp_path)
    plan = build_plan(config_path=CONFIG, heldout_path=heldout, ancestry_path=ancestry)
    job = plan["jobs"][0]
    model_config = tmp_path / "model-config.json"
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_config = tmp_path / "tokenizer-config.json"
    snapshot_seal = tmp_path / "snapshot-seal.json"
    driver = tmp_path / "driver.py"
    for path, payload in (
        (model_config, "{}"),
        (tokenizer_json, "{}"),
        (tokenizer_config, "{}"),
        (snapshot_seal, "{}"),
        (driver, "pass\n"),
    ):
        path.write_text(payload, encoding="utf-8")
    heldout_value = json.loads(heldout.read_text(encoding="utf-8"))
    prefill_root = tmp_path / "prefill-cache"
    prefill_index = prefill_root / "index.json"
    prefill_body = {
        "schema_version": PREFILL_CACHE_SCHEMA,
        "cache_key": "f" * 64,
        "heldout_manifest_hash": plan["bindings"]["heldout_manifest_hash"],
        "serving_first_token_owner": "prefill",
        "evaluation_first_decode_input_source": (
            "teacher_forced_continuation_token_0"
        ),
        "evaluation_input_generated_by_bf16_model": False,
        "prefill_lm_head_executed": False,
        "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
    }
    write_immutable_json(prefill_index, prefill_body)
    prefill_index_value = json.loads(prefill_index.read_text(encoding="utf-8"))
    prefill_binding = {
        "cache_key": prefill_index_value["cache_key"],
        "index_content_hash": prefill_index_value["content_hash"],
        "index": path_identity(prefill_index),
        "artifact_tree": path_identity(prefill_root),
    }
    receipt_body = {
        "schema_version": DRIVER_RECEIPT_SCHEMA,
        "offline_local_files_only": True,
        "transformers_version": "5.5.0",
        "fused_model_class": "Qwen3MoeForCausalLM",
        "model_revision": MODEL_REVISION,
        "model_snapshot_revision_verified": True,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_snapshot_revision_verified": True,
        "model_config": path_identity(model_config),
        "tokenizer_json": path_identity(tokenizer_json),
        "tokenizer_config": path_identity(tokenizer_config),
        "model_snapshot_content_seal": path_identity(snapshot_seal),
        "chat_template": heldout_value["chat_template"],
        "driver_source": path_identity(driver),
        "heldout_manifest_hash": plan["bindings"]["heldout_manifest_hash"],
        "body_profile_id": job["body_profile_id"],
        "body_bank_ancestry_verified": True,
        "body_nonrouter_parameters_unchanged": True,
        "body_weight_bank_structure_verified": True,
        "router_target_pattern": "model.layers.<index>.mlp.gate",
        "router_target_count": 48,
        "only_router_modules_replaced": True,
        "router_variant_id": job["variant"]["variant_id"],
        "decode_query_length": 1,
        "key_format": job["body_profile"]["key_format"],
        "value_format": job["body_profile"]["value_format"],
        "local_decode_head_unchanged": True,
        "prefill_owner": "external_bf16_source_model",
        "serving_first_token_owner": "prefill",
        "evaluation_first_decode_input_source": (
            "teacher_forced_continuation_token_0"
        ),
        "evaluation_input_generated_by_bf16_model": False,
        "prefill_lm_head_executed": False,
        "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
        "first_decode_input_stream_hash": "d" * 64,
        "scored_decode_suffix_stream_hash": "e" * 64,
        "teacher_forced_token_stream_hash": "c" * 64,
        "prefill_cache_binding": prefill_binding,
        "heldout_counts": {
            "records": 1,
            "continuation_tokens_per_arm": 2,
            "prefill_owned_input_tokens_per_arm": 1,
            "scored_tokens_per_arm": 1,
        },
    }
    receipt = receipt_body | {"content_hash": canonical_hash(receipt_body)}
    result = {
        "schema_version": RESULT_SCHEMA,
        "plan_hash": plan["content_hash"],
        "request_hash": "a" * 64,
        "job_id": job["job_id"],
        "body_profile_id": job["body_profile_id"],
        "variant_id": job["variant"]["variant_id"],
        "status": "success",
        "measurements": {
            "execution_receipt": receipt,
            "router_agreement": _target_agreement(
                plan["bindings"]["heldout_manifest_hash"]
            ),
            "end_to_end": {
                "paired_bf16_reexecuted": True,
                "baseline_router": dict(_BF16_ROUTER_CONTRACT),
                "heldout_manifest_hash": plan["bindings"][
                    "heldout_manifest_hash"
                ],
                "teacher_forced_token_stream_hash": "c" * 64,
                "first_decode_input_stream_hash": "d" * 64,
                "scored_decode_suffix_stream_hash": "e" * 64,
                "continuation_decode_contract": dict(
                    _CONTINUATION_DECODE_CONTRACT
                ),
                "bf16_router_baseline": {
                    "mean_token_nll": 2.0,
                    "token_count": 1,
                    "teacher_forced_token_stream_hash": "c" * 64,
                    "first_decode_input_stream_hash": "d" * 64,
                    "scored_decode_suffix_stream_hash": "e" * 64,
                },
                "mx_router_candidate": {
                    "mean_token_nll": 2.125,
                    "token_count": 1,
                    "teacher_forced_token_stream_hash": "c" * 64,
                    "first_decode_input_stream_hash": "d" * 64,
                    "scored_decode_suffix_stream_hash": "e" * 64,
                },
                "mean_token_nll_delta": 0.125,
                "task_effects": {
                    "status": "unsupported",
                    "reason": "task driver not installed",
                },
            },
        },
        "failure": None,
        "classification": _classification(),
    }
    validate_result(plan, result)
    sealed_prefill_index = prefill_index.read_bytes()
    prefill_index.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_result(plan, result)
    prefill_index.write_bytes(sealed_prefill_index)
    tokenizer_json.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        validate_result(plan, result)
    tokenizer_json.write_text("{}", encoding="utf-8")
    result["measurements"]["end_to_end"]["mx_router_candidate"][
        "teacher_forced_token_stream_hash"
    ] = "d" * 64
    with pytest.raises(ValueError, match="teacher-forced token stream"):
        validate_result(plan, result)
    result["measurements"]["end_to_end"]["mx_router_candidate"][
        "teacher_forced_token_stream_hash"
    ] = "c" * 64
    result["measurements"]["end_to_end"]["mean_token_nll_delta"] = 0.12
    with pytest.raises(ValueError, match="NLL delta differs"):
        validate_result(plan, result)


def test_runner_retains_oom_as_immutable_terminal_row(tmp_path):
    heldout, ancestry = _artifacts(tmp_path)
    plan = build_plan(config_path=CONFIG, heldout_path=heldout, ancestry_path=ancestry)
    plan_path = tmp_path / "plan.json"
    write_immutable_json(plan_path, plan)
    driver = tmp_path / "oom-driver"
    driver.write_text("#!/bin/sh\necho 'CUDA out of memory' >&2\nexit 137\n", encoding="utf-8")
    driver.chmod(0o755)
    output = tmp_path / "run"
    counts = run_shard(
        plan_path=plan_path,
        shard_index=0,
        gpu="0",
        driver_path=driver,
        output_root=output,
        timeout_seconds=5,
    )
    assert counts == {"success": 0, "failed": 0, "oom": 3}
    result_paths = tuple(sorted((output / "results").glob("*.json")))
    before = tuple(path.read_bytes() for path in result_paths)
    repeat = run_shard(
        plan_path=plan_path,
        shard_index=0,
        gpu="0",
        driver_path=driver,
        output_root=output,
        timeout_seconds=5,
    )
    assert repeat == counts
    assert tuple(path.read_bytes() for path in result_paths) == before
    completion = build_completion(plan_path, output)
    assert completion["terminal_job_count"] == 3
    assert completion["status_counts"] == counts
    assert all(row["driver"] == path_identity(driver) for row in completion["rows"])
    assert all(row["prefill_cache_binding"] is None for row in completion["rows"])


def test_batched_runner_retains_each_missing_job_after_driver_oom(tmp_path):
    heldout, ancestry = _artifacts(tmp_path)
    plan = build_plan(config_path=CONFIG, heldout_path=heldout, ancestry_path=ancestry)
    plan_path = tmp_path / "plan.json"
    write_immutable_json(plan_path, plan)
    driver = tmp_path / "batch-oom-driver"
    driver.write_text(
        "#!/bin/sh\necho 'CUDA out of memory in body group' >&2\nexit 137\n",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    output = tmp_path / "batch-run"
    counts = run_shard(
        plan_path=plan_path,
        shard_index=0,
        gpu="0",
        driver_path=driver,
        output_root=output,
        timeout_seconds=5,
        batch_driver=True,
    )
    assert counts == {"success": 0, "failed": 0, "oom": 3}
    assert (output / "requests/shard-0.batch.json").is_file()
    assert (output / "logs/shard-0.batch.stderr").is_file()
    assert len(tuple((output / "results").glob("*.json"))) == 3
