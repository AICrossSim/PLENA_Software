"""Focused fail-closed tests for the matched-geometry producer bridge."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import ClassVar
import unittest
from unittest.mock import patch

from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.mlen_revalidation import (
    _validated_oracle_hash,
    finalize,
)
from decode_dse.software.shared_geometry_comparison import (
    ORCHESTRATION_SCHEMA,
    _core,
    _content_hash,
    _campaign_candidate,
    _selector_script,
    _validate_model_architecture,
    build_evaluator_replay_invocation,
    load_producer_receipt_strict,
    resolve_mlen_numerical_evidence,
    validate_evaluator_replay_invocation,
)
from decode_dse.software.sweep_plan import (
    PromptManifest,
    PromptRecord,
    load_immutable_json,
    write_immutable_json,
)
from decode_dse.software.token_samples import (
    build_refinement_bundle_from_token_stream,
    save_refinement_sample_bundle,
)
from decode_dse.software.test_mlen_revalidation import (
    _install_plan,
    _install_rows,
)


REVISION = "3ca25493489e939d65b4161677cc24154138d127"


def _model() -> dict:
    architecture = {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "use_qk_norm": True,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
    }
    return {
        "name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "model_architecture": architecture,
        "architecture_sha256": _content_hash(architecture),
    }


def _contract(model_name: str) -> dict:
    body = {
        "schema_version": ORCHESTRATION_SCHEMA,
        "selection_role": "balanced_source",
        "shared_geometry": {"MLEN": 2048, "BLEN": 32, "VLEN": 2048},
        "specialized_derivation_rule": (
            "preserve_all_axes_except_blen_set_blen_to_65536_div_mlen"
        ),
        "metric_whitelist": [
            "decode_stage_tpot_ms",
            "one_time_handoff_ms",
            "handoff_amortized_decode_side_service_ms",
            "relative_perplexity_vs_bf16",
        ],
        "forbidden_metrics": ["goodput"],
        "claim_scope": "controlled_decode_stage_geometry_ablation",
        "handoff": {
            "mode": "config_bound_analytic",
            "analytic_contract": {
                "schema_version": "plena-config-bound-analytic-handoff/v1",
                "link_generation": "nvlink4",
                "admission_bandwidth_policy": (
                    "matched_candidate_aggregate_hbm_roofline"
                ),
                "admission_evidence_id": "a" * 64,
                "admission_evidence_tier": "optimistic_hbm_roofline",
                "decode_ready_wait_ms": 0,
            },
        },
        "numerical": {
            "source_mlen": 1024,
            "shared_mlen": 2048,
            "suite": "refinement",
            "same_plan_bf16_required": True,
        },
        "extensible_models": [model_name],
    }
    return {**body, "content_hash": _content_hash(body)}


def _write_mapped_hardware_source(
    root: Path,
    plan_body: dict,
    profiles: dict[str, DecodePrecisionProfile],
) -> None:
    mapping = plan_body["candidate_mapping"][0]
    source = profiles[mapping["source_profile_id"]]
    provenance = {"evaluator_version": "strict-mlen-test/v1"}
    run_id = "hwdse-" + _content_hash(provenance)
    header = {
        "record_type": "study",
        "run_id": run_id,
        "provenance": provenance,
        "expected_result_count": 1,
    }
    source_hardware = dict(mapping["hardware"])
    source_hardware.update({"MLEN": 1024, "BLEN": 64, "VLEN": 1024})
    row_body = {
        "run_id": run_id,
        "profile_id": source.profile_id,
        "candidate_id": "hw-" + _content_hash(source_hardware),
        "profile": source.to_dict(),
        "hardware": source_hardware,
    }
    record_hash = _content_hash(row_body)
    artifact = root / "mapped_source.jsonl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(header, sort_keys=True, separators=(",", ":"))
        + "\n"
        + json.dumps(
            {"record_type": "result", **row_body, "record_hash": record_hash},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact.with_name(f"{artifact.name}.meta.json").write_text(
        json.dumps(
            {
                "content_sha256": artifact_sha,
                "run_id": run_id,
                "provenance_hash": _content_hash(provenance),
                "data_file": artifact.name,
                "result_count": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    mapping["source_record_hash"] = record_hash
    mapping["source_artifact_path"] = str(artifact)
    mapping["source_artifact_sha256"] = artifact_sha
    mapping["hardware_identity_hash"] = _content_hash(
        {
            "source_profile_id": source.profile_id,
            "candidate_id": mapping["candidate_id"],
            "source_record_hash": record_hash,
            "source_artifact_sha256": artifact_sha,
            "hardware": mapping["hardware"],
        }
    )


def _strict_plan_body(
    original: dict,
    *,
    root: Path,
    profiles: dict[str, DecodePrecisionProfile],
) -> dict:
    body = deepcopy(original)
    body.pop("content_hash", None)
    numerical_workspace = root / "numerical_workspace"
    refinement_path = root / "refinement_samples.json"
    prompts = PromptManifest(
        dataset_name="strict-test",
        dataset_revision="strict-test-v1",
        numerical_screen=tuple(
            PromptRecord(
                f"numerical-screen-{index:04d}",
                hashlib.sha256(f"screen-{index}".encode()).hexdigest(),
            )
            for index in range(16)
        ),
        hardware_validation=tuple(
            PromptRecord(
                f"hardware-validation-{index:04d}",
                hashlib.sha256(f"validation-{index}".encode()).hexdigest(),
            )
            for index in range(32)
        ),
    )
    write_immutable_json(
        numerical_workspace / "prompt_manifest.json", prompts.to_dict()
    )
    refinement = build_refinement_bundle_from_token_stream(
        range(641 * 128),
        model_revision=REVISION,
        tokenizer_revision=REVISION,
        dataset_name="strict-refinement-test",
        dataset_revision="strict-refinement-test-v1",
    )
    save_refinement_sample_bundle(refinement, refinement_path)
    body["paths"] = {
        "output_root": str(root),
        "numerical_workspace": str(numerical_workspace),
        "refinement_sample_bundle": str(refinement_path),
    }
    body["bindings"] = {
        **body["bindings"],
        "refinement_sample_bundle_hash": refinement.canonical_hash,
        "refinement_sample_file_sha256": hashlib.sha256(
            refinement_path.read_bytes()
        ).hexdigest(),
        "prompt_manifest_hash": prompts.canonical_hash,
    }
    body["sample_suites"] = {
        "validation": {
            "prompt_set": "hardware_validation",
            "document_count": 32,
            "decode_steps": 16,
            "q_len": 1,
        },
        "refinement": {
            "sample_bundle_hash": refinement.canonical_hash,
            "document_count": 128,
            "decode_steps": 128,
            "q_len": 1,
        },
    }
    body["classification"] = {
        **body.get("classification", {}),
        "measured_numerical": True,
        "publication_rankable": False,
        "publication_selection_eligible": False,
        "hardware_bit_parity_verified": False,
    }
    mapping = body["candidate_mapping"][0]
    mapping["hardware"] = {
        "MLEN": 2048,
        "BLEN": 32,
        "VLEN": 2048,
        "HLEN": 128,
        "BATCH": 8,
        "HBM_CHANNELS": 32,
        "HBM_GENERATION": "HBM2",
        "CHIP_COUNT": 8,
        "TP": 8,
        "KVP": 1,
        "LINK_PORTS": 1,
        "SRAM_POLICY": "streaming",
        "KV_HEAD_REUSE": False,
        "DRAIN_OVERLAPPED": False,
        "EXPERT_PARALLEL_MODE": "tensor_parallel",
    }
    mapping["candidate_id"] = "hw-" + _content_hash(mapping["hardware"])
    mapping["candidate_matrix_mlen"] = 2048
    _write_mapped_hardware_source(root, body, profiles)
    return body


def _runtime_binding(
    profile: DecodePrecisionProfile,
    *,
    identity: str,
    structure: str,
) -> dict:
    quantized = profile.kind != "bf16_reference"
    events = 1 if quantized else 0
    return {
        "performed": quantized,
        "seconds": 0.001,
        "target_count": 16 if quantized else 0,
        "used_cached_targets": True,
        "weight_requantizations": 0,
        "sealed_weight_modules": 16 if quantized else 0,
        "weight_quantization_events_before": events,
        "weight_quantization_events_after": events,
        "weight_identity_before": identity,
        "weight_identity_after": identity,
        "weight_structure_fingerprint": structure,
    }


def _metric_ledger(
    profile: DecodePrecisionProfile,
    *,
    suite: str,
    nominal_mean: float,
    sample_bundle_hash: str,
    identity: str,
    structure: str,
) -> dict:
    document_count, steps = (32, 16) if suite == "validation" else (128, 128)
    documents = []
    document_prefix = (
        "hardware-validation" if suite == "validation" else "refinement-heldout-window"
    )
    for index in range(document_count):
        nll_sum = nominal_mean * steps
        document = {
            "document_id": f"{document_prefix}-{index:04d}",
            "nll_sum": nll_sum,
            "token_count": steps,
            "mean_token_nll": nll_sum / steps,
            "initial_cache_length": 512,
            "final_cache_length": 512 + steps,
        }
        if suite == "validation":
            document.update(
                {"source_cluster_id": f"cluster-{index:04d}", "first_token_id": 1}
            )
        documents.append(document)
    aggregate = math.fsum(value["nll_sum"] for value in documents)
    tokens = document_count * steps
    layers = 48
    expected_tensors = layers * 2 if suite == "refinement" else 0
    expected_quantized = (
        0
        if profile.kind == "bf16_reference"
        else expected_tensors
    )
    result = {
        "nll_sum": aggregate,
        "token_count": tokens,
        "mean_token_nll": aggregate / tokens,
        "runtime_rebinding": _runtime_binding(
            profile, identity=identity, structure=structure
        ),
        "native_append_validation": {
            "calls": 1 if suite == "refinement" else 0,
            "tensor_checks": expected_tensors,
            "quantized_tensor_checks": expected_quantized,
            "expected_tensor_checks": expected_tensors,
            "expected_quantized_tensor_checks": expected_quantized,
            "q_len": 1,
        },
        "documents": documents,
    }
    if suite == "validation":
        result.update(
            {
                "mean_nll": aggregate / tokens,
                "weight_bank": {
                    "weight_format": profile.weight_format,
                    "weight_method": profile.method,
                    "build_seconds": 1.0,
                    "parameter_count": 10,
                    "identity_fingerprint": identity,
                    "structure_fingerprint": structure,
                },
                "runtime_environment": {
                    "logical_fingerprint": "7" * 64,
                    "mase_tree_sha256": "8" * 64,
                },
                "sample_contract": {
                    "name": "mlen-geometry-validation",
                    "prompt_set": "hardware_validation",
                    "prompt_count": document_count,
                    "prefill_tokens": 512,
                    "decode_steps": steps,
                    "q_len": 1,
                    "teacher_forced_cached": True,
                    "compiler_required": False,
                    "emulator_required": False,
                },
            }
        )
        result["native_append_validation"].update(
            {
                "expected_calls": 0,
                "mode": "preflight_gated",
                "deep_oracle_enabled": False,
            }
        )
    else:
        result.update(
            {
                "sample_bundle_hash": sample_bundle_hash,
                "document_count": document_count,
                "decode_steps": steps,
                "q_len": 1,
            }
        )
    return result


def _upgrade_rows_to_strict_ledgers(
    root: Path,
    profiles: dict[str, DecodePrecisionProfile],
) -> None:
    plan = load_immutable_json(root / "mlen_revalidation_plan.json")
    sample_bundle_hash = plan["bindings"]["refinement_sample_bundle_hash"]
    rows = root / "shards" / "part-0000-of-0001" / "rows"
    for profile in profiles.values():
        path = rows / f"{profile.profile_id}.json"
        value = load_immutable_json(path)
        value.pop("content_hash")
        nominal_validation = float(value["validation"]["mean_token_nll"])
        nominal_refinement = float(value["refinement"]["mean_token_nll"])
        identity = hashlib.sha256(
            f"identity:{profile.weight_format}".encode()
        ).hexdigest()
        structure = hashlib.sha256(
            f"structure:{profile.weight_format}".encode()
        ).hexdigest()
        bank_profile_ids = sorted(
            item.profile_id
            for item in profiles.values()
            if item.weight_format == profile.weight_format
        )
        value.update(
            {
                "validation": _metric_ledger(
                    profile,
                    suite="validation",
                    nominal_mean=nominal_validation,
                    sample_bundle_hash=sample_bundle_hash,
                    identity=identity,
                    structure=structure,
                ),
                "refinement": _metric_ledger(
                    profile,
                    suite="refinement",
                    nominal_mean=nominal_refinement,
                    sample_bundle_hash=sample_bundle_hash,
                    identity=identity,
                    structure=structure,
                ),
                "weight_bank": {
                    "weight_format": profile.weight_format,
                    "weight_method": profile.method,
                    "build_seconds": 1.0,
                    "identity_fingerprint": identity,
                    "structure_fingerprint": structure,
                    "parameter_count": 10,
                    "profile_ids": bank_profile_ids,
                    "same_in_memory_bank_across_mlen_values": True,
                },
                "mixed_weight_activation_abi": None,
                "error": None,
                "runtime_seconds": 2.0,
                "classification": {
                    "measured_numerical": True,
                    "publication_rankable": False,
                    "selection_eligible": False,
                    "compiler_valid": False,
                    "emulator_valid": False,
                    "rtl_valid": False,
                    "hardware_bit_parity_verified": False,
                },
            }
        )
        path.unlink()
        write_immutable_json(path, value)


class _FakeEvaluator:
    class _Backend:
        class _Simulator:
            dims: ClassVar[dict] = {
                "model_type": "qwen3_moe",
                "hidden": 2048,
                "dense_inter": 6144,
                "inter": 768,
                "layers": 48,
                "heads": 32,
                "kv_heads": 4,
                "head_dim": 128,
                "vocab": 151936,
                "tie_embeddings": False,
                "qk_norm": True,
                "num_experts": 128,
                "experts_per_token": 8,
                "norm_topk_prob": True,
            }

        sim = _Simulator()

    backend = _Backend()


class SharedGeometryProducerTests(unittest.TestCase):
    def test_nearby_foreign_selector_cannot_self_authenticate_a_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign_root = Path(directory) / "foreign"
            foreign = campaign_root / "scripts" / "select_campaign.py"
            foreign.parent.mkdir(parents=True)
            foreign.write_text("raise SystemExit(0)\n", encoding="utf-8")
            campaign = campaign_root / "campaign.json"
            campaign.write_text("{}\n", encoding="utf-8")
            resolved = _selector_script(campaign)
            self.assertNotEqual(resolved, foreign.resolve())
            self.assertEqual(
                resolved,
                Path(__file__).resolve().parents[3]
                / "PLENA_Qwen30B_Moe_Results"
                / "scripts"
                / "select_campaign.py",
            )

    def test_cached_core_from_another_simulator_root_is_rejected(self) -> None:
        _core()  # populate the process module cache from the configured root
        with tempfile.TemporaryDirectory() as directory:
            fake = type("Resolved", (), {"root": Path(directory)})()
            with patch(
                "decode_dse.software.shared_geometry_comparison."
                "resolve_simulator_root",
                return_value=fake,
            ):
                with self.assertRaisesRegex(RuntimeError, "different Simulator root"):
                    _core()

    def test_exact_mlen_resolver_uses_one_plan_and_bf16_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "seed"
            _, profiles = _install_plan(seed)
            original = load_immutable_json(seed / "mlen_revalidation_plan.json")
            root = Path(directory) / "comparison"
            original = _strict_plan_body(
                original, root=root, profiles=profiles
            )
            plan_path = root / "mlen_revalidation_plan.json"
            write_immutable_json(plan_path, original)
            _install_rows(root, plan_path, profiles)
            _upgrade_rows_to_strict_ledgers(root, profiles)
            finalize(plan_path=plan_path)
            source = next(
                profile
                for profile in profiles.values()
                if profile.kind != "bf16_reference" and profile.matrix_mlen == 1024
            )
            evidence = resolve_mlen_numerical_evidence(
                root / "mlen_revalidation_completion.json",
                selected_source_profile_id=source.profile_id,
                selected_hardware_mlen=1024,
                model=_model(),
                suite="refinement",
            )
            self.assertEqual(evidence.specialized_entry.profile.matrix_mlen, 1024)
            self.assertEqual(evidence.shared_entry.profile.matrix_mlen, 2048)
            self.assertEqual(
                evidence.specialized_receipt["sample_set_sha256"],
                evidence.shared_receipt["sample_set_sha256"],
            )
            self.assertEqual(
                evidence.bf16_oracle["scored_tokens"],
                evidence.specialized_receipt["scored_tokens"],
            )
            self.assertEqual(
                evidence.receipt["evidence_class"],
                "source_hash_bound_measured_numerical_nonadversarial",
            )
            self.assertFalse(
                evidence.receipt["independent_numerical_execution_replayed"]
            )
            self.assertFalse(
                evidence.receipt["adversarial_tamper_resistance_claimed"]
            )

    def test_mislabelled_quantized_row_cannot_be_the_bf16_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "seed"
            _, profiles = _install_plan(seed)
            original = load_immutable_json(seed / "mlen_revalidation_plan.json")
            fake = DecodePrecisionProfile.quantized(
                "MXINT8", "MXINT8", "MXINT8", "FP_E8M5"
            )
            bf16_id = next(
                profile_id
                for profile_id, profile in profiles.items()
                if profile.kind == "bf16_reference"
            )
            profiles.pop(bf16_id)
            profiles[fake.profile_id] = fake
            for planned in original["evaluation_profiles"]:
                if planned["role"] == "same_split_bf16_reference":
                    planned["profile_id"] = fake.profile_id
                    planned["profile"] = fake.to_dict()
                    planned["numerical_oracle_sha256"] = (
                        _validated_oracle_hash(fake)
                    )
            partition = original["sharding"]["partitions"][0]
            partition["profile_ids"] = sorted(profiles)
            partition["weight_formats"] = sorted(
                set(partition["weight_formats"]) | {fake.weight_format}
            )
            root = Path(directory) / "forged"
            original = _strict_plan_body(
                original, root=root, profiles=profiles
            )
            plan_path = root / "mlen_revalidation_plan.json"
            write_immutable_json(plan_path, original)
            _install_rows(root, plan_path, profiles)
            _upgrade_rows_to_strict_ledgers(root, profiles)
            fake_row_path = (
                root
                / "shards"
                / "part-0000-of-0001"
                / "rows"
                / f"{fake.profile_id}.json"
            )
            fake_row = load_immutable_json(fake_row_path)
            fake_row.pop("content_hash")
            fake_row["role"] = "same_split_bf16_reference"
            fake_row_path.unlink()
            write_immutable_json(fake_row_path, fake_row)
            finalize(plan_path=plan_path)
            source = next(
                profile
                for profile in profiles.values()
                if profile.weight_format == "MXINT4" and profile.matrix_mlen == 1024
            )
            with self.assertRaisesRegex(ValueError, "backend BF16 reference"):
                resolve_mlen_numerical_evidence(
                    root / "mlen_revalidation_completion.json",
                    selected_source_profile_id=source.profile_id,
                    selected_hardware_mlen=1024,
                    model=_model(),
                    suite="refinement",
                )

    def test_self_checksummed_terminal_headline_nll_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "seed"
            _, profiles = _install_plan(seed)
            original = load_immutable_json(seed / "mlen_revalidation_plan.json")
            root = Path(directory) / "tampered"
            original = _strict_plan_body(
                original, root=root, profiles=profiles
            )
            plan_path = root / "mlen_revalidation_plan.json"
            write_immutable_json(plan_path, original)
            _install_rows(root, plan_path, profiles)
            _upgrade_rows_to_strict_ledgers(root, profiles)
            source = next(
                profile
                for profile in profiles.values()
                if profile.kind != "bf16_reference" and profile.matrix_mlen == 1024
            )
            row_path = (
                root
                / "shards"
                / "part-0000-of-0001"
                / "rows"
                / f"{source.profile_id}.json"
            )
            row = load_immutable_json(row_path)
            row.pop("content_hash")
            # Re-seal the forged source row and let the normal producer
            # finalize it, proving that checksums/completion arithmetic alone
            # do not authorize a copied headline metric.
            row["refinement"]["mean_token_nll"] -= 0.5
            row_path.unlink()
            write_immutable_json(row_path, row)
            finalize(plan_path=plan_path)
            with self.assertRaisesRegex(ValueError, "mean token NLL differs"):
                resolve_mlen_numerical_evidence(
                    root / "mlen_revalidation_completion.json",
                    selected_source_profile_id=source.profile_id,
                    selected_hardware_mlen=1024,
                    model=_model(),
                    suite="refinement",
                )

    def test_terminal_population_order_and_cache_origin_tamper_are_rejected(self) -> None:
        mutations = {
            "order": "ordered document population differs",
            "cache": "document/token/cache ledger differs",
            "runtime": "different logical/software runtimes",
        }
        for mutation, message in mutations.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                seed = Path(directory) / "seed"
                _, profiles = _install_plan(seed)
                original = load_immutable_json(
                    seed / "mlen_revalidation_plan.json"
                )
                root = Path(directory) / mutation
                original = _strict_plan_body(
                    original, root=root, profiles=profiles
                )
                plan_path = root / "mlen_revalidation_plan.json"
                write_immutable_json(plan_path, original)
                _install_rows(root, plan_path, profiles)
                _upgrade_rows_to_strict_ledgers(root, profiles)
                source = next(
                    profile
                    for profile in profiles.values()
                    if profile.kind != "bf16_reference"
                    and profile.matrix_mlen == 1024
                )
                row_path = (
                    root
                    / "shards"
                    / "part-0000-of-0001"
                    / "rows"
                    / f"{source.profile_id}.json"
                )
                row = load_immutable_json(row_path)
                row.pop("content_hash")
                documents = row["refinement"]["documents"]
                if mutation == "order":
                    documents[0], documents[1] = documents[1], documents[0]
                elif mutation == "cache":
                    documents[0]["initial_cache_length"] += 1
                    documents[0]["final_cache_length"] += 1
                else:
                    row["validation"]["runtime_environment"][
                        "logical_fingerprint"
                    ] = "9" * 64
                row_path.unlink()
                write_immutable_json(row_path, row)
                finalize(plan_path=plan_path)
                with self.assertRaisesRegex(ValueError, message):
                    resolve_mlen_numerical_evidence(
                        root / "mlen_revalidation_completion.json",
                        selected_source_profile_id=source.profile_id,
                        selected_hardware_mlen=1024,
                        model=_model(),
                        suite="refinement",
                    )

    def test_consistent_source_rewrite_is_never_called_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "seed"
            _, profiles = _install_plan(seed)
            original = load_immutable_json(seed / "mlen_revalidation_plan.json")
            root = Path(directory) / "consistent-rewrite"
            original = _strict_plan_body(
                original, root=root, profiles=profiles
            )
            plan_path = root / "mlen_revalidation_plan.json"
            write_immutable_json(plan_path, original)
            _install_rows(root, plan_path, profiles)
            _upgrade_rows_to_strict_ledgers(root, profiles)
            source = next(
                profile
                for profile in profiles.values()
                if profile.kind != "bf16_reference" and profile.matrix_mlen == 1024
            )
            shared = next(
                profile
                for profile in profiles.values()
                if profile.kind != "bf16_reference" and profile.matrix_mlen == 2048
            )
            row_path = (
                root
                / "shards"
                / "part-0000-of-0001"
                / "rows"
                / f"{shared.profile_id}.json"
            )
            row = load_immutable_json(row_path)
            row.pop("content_hash")
            metrics = row["refinement"]
            for document in metrics["documents"]:
                document["nll_sum"] += 16.0
                document["mean_token_nll"] = (
                    document["nll_sum"] / document["token_count"]
                )
            metrics["nll_sum"] = math.fsum(
                document["nll_sum"] for document in metrics["documents"]
            )
            metrics["mean_token_nll"] = metrics["nll_sum"] / metrics["token_count"]
            row_path.unlink()
            write_immutable_json(row_path, row)
            finalize(plan_path=plan_path)
            evidence = resolve_mlen_numerical_evidence(
                root / "mlen_revalidation_completion.json",
                selected_source_profile_id=source.profile_id,
                selected_hardware_mlen=1024,
                model=_model(),
                suite="refinement",
            )
            self.assertFalse(
                evidence.receipt["independent_numerical_execution_replayed"]
            )
            self.assertFalse(
                evidence.receipt["adversarial_tamper_resistance_claimed"]
            )
            self.assertEqual(
                evidence.receipt["evidence_class"],
                "source_hash_bound_measured_numerical_nonadversarial",
            )

    def test_balanced_source_cannot_be_relabelled(self) -> None:
        def candidate(profile: str, hardware: str) -> dict:
            return {
                "profile_id": profile,
                "candidate_id": hardware,
                "record_hash": "1" * 64,
                "profile": {"kind": "quantized"},
                "hardware": {"MLEN": 1024},
                "accuracy": {"relative_perplexity": 1.0},
            }

        first = candidate("p1", "h1")
        second = candidate("p2", "h2")
        campaign = {
            "selection": {
                "evidence_mode": "exploratory",
                "projected_joint_source_bound": True,
                "whole_study_publication_candidate": False,
                "calibrated_variants": {"emitted_methods": []},
            },
            "top_co_design_rows": [
                {**first, "rank": 1, "balanced_score": 0.1, "roles": []},
                {**second, "rank": 2, "balanced_score": 0.2, "roles": []},
            ],
            "metric_extrema": {"overall": first},
            "geometry_revalidation": {"corrected_numerical_input": None},
            "balanced_source": first,
        }
        self.assertEqual(
            _campaign_candidate(campaign, "balanced_source"), first
        )
        # The global extremum may legitimately be omitted from the displayed
        # per-profile top rows; it remains the deterministic balanced source.
        campaign["top_co_design_rows"] = [
            {**second, "rank": 1, "balanced_score": 0.2, "roles": []}
        ]
        self.assertEqual(_campaign_candidate(campaign, "balanced_source"), first)
        campaign["balanced_source"] = second
        with self.assertRaisesRegex(ValueError, "metric_extrema.overall"):
            _campaign_candidate(campaign, "balanced_source")
        with self.assertRaisesRegex(ValueError, "only balanced_source"):
            _campaign_candidate(campaign, "top_rank:1")
        campaign["balanced_source"] = first
        campaign["geometry_revalidation"]["corrected_numerical_input"] = {
            "path": "/post-hoc/corrected.json"
        }
        with self.assertRaisesRegex(ValueError, "frozen before"):
            _campaign_candidate(campaign, "balanced_source")

    def test_launch_invocation_is_content_addressed_and_tamper_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            manifest = root / "manifest.json"
            orchestration = root / "orchestration.json"
            config.write_text("{}", encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            orchestration.write_text(
                json.dumps(_contract(_model()["name"])), encoding="utf-8"
            )
            invocation = build_evaluator_replay_invocation(
                [
                    "--manifest",
                    str(manifest),
                    "--config",
                    str(config),
                    "--output",
                    str(root / "must-not-be-used.jsonl"),
                ],
                orchestration_contract_path=orchestration,
            )
            self.assertIn(
                "__plena_matched_geometry_replay_no_write__",
                invocation["argv"],
            )
            self.assertEqual(validate_evaluator_replay_invocation(invocation), invocation)
            manifest.write_text('{"tampered":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input bytes"):
                validate_evaluator_replay_invocation(invocation)
            with self.assertRaisesRegex(ValueError, "profile filters"):
                build_evaluator_replay_invocation(
                    ["--profile-id", "chosen", "--output", "ignored"],
                    orchestration_contract_path=orchestration,
                )

    def test_architecture_binding_covers_body_and_local_head(self) -> None:
        model = _model()
        _validate_model_architecture(_FakeEvaluator(), model)
        for field in ("hidden_size", "vocab_size", "num_experts"):
            changed = deepcopy(model)
            changed["model_architecture"][field] += 1
            changed["architecture_sha256"] = _content_hash(
                changed["model_architecture"]
            )
            with self.assertRaisesRegex(ValueError, "architecture differ"):
                _validate_model_architecture(_FakeEvaluator(), changed)

    def test_strict_loader_rejects_forged_tpot_and_nll(self) -> None:
        genuine = {
            "source_receipt": {
                "campaign_path": "/campaign.json",
                "hardware_artifact_paths": ["/hardware.jsonl"],
            },
            "evaluator_replay": {
                "argv": ["--output", "ignored"],
                "orchestration_contract_path": "/contract.json",
            },
            "numerical_evidence_receipt": {
                "completion_path": "/completion.json",
                "suite": "refinement",
            },
            "replay": {"strict_evaluator_replay_on_load": True},
            "comparison": {
                "result": {"latency_metrics": {"decode_specialized_tpot_ms": 8.0}},
                "input": {
                    "arms": {
                        "decode_specialized": {
                            "numerical_receipt": {"candidate_mean_token_nll": 2.1}
                        }
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "producer.json"
            path.write_text("{}", encoding="utf-8")
            for mutation in ("tpot", "nll"):
                forged = deepcopy(genuine)
                if mutation == "tpot":
                    forged["comparison"]["result"]["latency_metrics"][
                        "decode_specialized_tpot_ms"
                    ] = 1.0
                else:
                    forged["comparison"]["input"]["arms"][
                        "decode_specialized"
                    ]["numerical_receipt"]["candidate_mean_token_nll"] = 1.0
                with patch(
                    "decode_dse.software.shared_geometry_comparison."
                    "_load_producer_receipt_structural",
                    return_value=forged,
                ), patch(
                    "decode_dse.software.shared_geometry_comparison."
                    "_verify_producer_source_files"
                ), patch(
                    "decode_dse.software.shared_geometry_comparison."
                    "materialize_from_campaign",
                    return_value=genuine,
                ):
                    with self.assertRaisesRegex(ValueError, "strict evaluator replay"):
                        load_producer_receipt_strict(path)


if __name__ == "__main__":
    unittest.main()
