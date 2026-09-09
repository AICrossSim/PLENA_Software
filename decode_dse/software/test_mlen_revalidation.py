"""CPU-only regressions for fail-closed MLEN numerical revalidation."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.mlen_revalidation import (
    COMPLETION_SCHEMA,
    PLAN_SCHEMA,
    ROW_SCHEMA,
    SELECTOR_INPUT_SCHEMA,
    _assign_weight_banks,
    _content_hash,
    _matrix_geometry_receipt,
    _validated_oracle_hash,
    audit_mixed_weight_activation_abi,
    derive_mlen_variant,
    finalize,
)
from decode_dse.software.sweep_plan import load_immutable_json, write_immutable_json


def _quantized(*, mlen: int = 1024, weight: str = "MXINT4") -> DecodePrecisionProfile:
    source = DecodePrecisionProfile.quantized(
        weight,
        "MXINT4",
        "MXINT4",
        "FP_E8M5",
    )
    return source if mlen == 1024 else derive_mlen_variant(source, mlen)


class ProfileDerivationTests(unittest.TestCase):
    def test_variant_changes_only_mlen_derived_identity_and_head(self) -> None:
        source = _quantized()
        mlen2048 = derive_mlen_variant(source, 2048)
        mlen4096 = derive_mlen_variant(source, 4096)
        self.assertNotEqual(source.profile_id, mlen2048.profile_id)
        self.assertNotEqual(mlen2048.profile_id, mlen4096.profile_id)
        for variant, expected in ((mlen2048, 2048), (mlen4096, 4096)):
            self.assertEqual(variant.matrix_mlen, expected)
            self.assertEqual(variant.local_head_contract["matrix_mlen"], expected)
            self.assertEqual(
                variant.to_dict()["numerical_oracle"],
                variant.numerical_oracle_contract,
            )
            self.assertEqual(
                variant.numerical_oracle_contract["matrix_mlen"], expected
            )
            self.assertFalse(
                variant.numerical_oracle_contract[
                    "hardware_bit_parity_verified"
                ]
            )
            self.assertEqual(variant.weight_format, source.weight_format)
            self.assertEqual(variant.activation_format, source.activation_format)
            self.assertEqual(variant.kv_format, source.kv_format)
            self.assertEqual(variant.vector_format, source.vector_format)
            self.assertEqual(variant.method, "rtn")

    def test_2048_and_4096_cannot_be_numerically_deduplicated(self) -> None:
        receipt = _matrix_geometry_receipt()
        self.assertFalse(receipt["mlen2048_mlen4096_equivalent"])
        self.assertFalse(receipt["deduplication_permitted"])
        self.assertEqual(
            receipt["counterexamples"]["attention_output_projection"],
            {
                "reduction_k": 4096,
                "partials_at_mlen2048": 2,
                "partials_at_mlen4096": 1,
            },
        )

    def test_whole_weight_banks_are_never_split(self) -> None:
        profiles = [
            {"profile": _quantized(mlen=mlen, weight=weight).to_dict()}
            for weight in ("MXINT2", "MXINT4", "MXINT8")
            for mlen in (1024, 2048, 4096)
        ]
        first = _assign_weight_banks(profiles, 2)
        second = _assign_weight_banks(profiles, 2)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 2)
        locations = {}
        for shard in first:
            for weight in shard["weight_formats"]:
                self.assertNotIn(weight, locations)
                locations[weight] = shard["shard_index"]
        self.assertEqual(set(locations), {"MXINT2", "MXINT4", "MXINT8"})

    def test_missing_mase_abi_is_an_explicit_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = audit_mixed_weight_activation_abi(Path(directory))
        self.assertFalse(receipt["numerical_execution_supported"])
        self.assertFalse(receipt["hardware_deployment_supported"])
        self.assertIn("error_class", receipt)
        self.assertEqual(len(receipt["receipt_hash"]), 64)


def _install_plan(root: Path) -> tuple[Path, dict[str, DecodePrecisionProfile]]:
    source = _quantized()
    variant = derive_mlen_variant(source, 2048)
    bf16 = DecodePrecisionProfile.bf16_reference()
    profiles = {item.profile_id: item for item in (source, variant, bf16)}
    hardware = {"MLEN": 2048, "BLEN": 8}
    hardware_identity_hash = _content_hash(
        {
            "source_profile_id": source.profile_id,
            "candidate_id": "candidate-2048",
            "source_record_hash": "2" * 64,
            "source_artifact_sha256": "3" * 64,
            "hardware": hardware,
        }
    )
    plan_body = {
        "schema_version": PLAN_SCHEMA,
        "target": {
            "model_name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "model_revision": "3ca25493489e939d65b4161677cc24154138d127",
            "tokenizer_revision": "3ca25493489e939d65b4161677cc24154138d127",
        },
        "paths": {"output_root": str(root)},
        "bindings": {"source_spec_hash": "1" * 64},
        "candidate_mapping": [
            {
                "source_profile_id": source.profile_id,
                "revalidated_profile_id": variant.profile_id,
                "candidate_id": "candidate-2048",
                "candidate_matrix_mlen": 2048,
                "source_record_hash": "2" * 64,
                "source_artifact_path": "/sealed/projected.jsonl",
                "source_artifact_sha256": "3" * 64,
                "hardware_identity_hash": hardware_identity_hash,
                "hardware": hardware,
                "roles": ["best_throughput", "top_01"],
                "selection_metrics": {
                    "tps": 10.0,
                    "tpot_ms": 100.0,
                    "system_area_mm2": 20.0,
                },
            }
        ],
        "evaluation_profiles": [
            {
                "profile_id": source.profile_id,
                "profile": source.to_dict(),
                "numerical_oracle_sha256": _validated_oracle_hash(source),
                "role": "same_format_mlen1024_control",
            },
            {
                "profile_id": variant.profile_id,
                "profile": variant.to_dict(),
                "numerical_oracle_sha256": _validated_oracle_hash(variant),
                "role": "candidate_mlen_variant",
            },
            {
                "profile_id": bf16.profile_id,
                "profile": bf16.to_dict(),
                "numerical_oracle_sha256": _validated_oracle_hash(bf16),
                "role": "same_split_bf16_reference",
            },
        ],
        "sharding": {
            "shard_count": 1,
            "partitions": [
                {
                    "shard_index": 0,
                    "weight_formats": ["BF16", "MXINT4"],
                    "profile_ids": sorted(profiles),
                }
            ],
        },
        "classification": {
            "publication_rankable": False,
            "selection_eligible": False,
        },
    }
    plan_path = root / "mlen_revalidation_plan.json"
    write_immutable_json(plan_path, plan_body)
    return plan_path, profiles


def _install_rows(
    root: Path,
    plan_path: Path,
    profiles: dict[str, DecodePrecisionProfile],
    *,
    failed_profile: str | None = None,
) -> None:
    plan = load_immutable_json(plan_path)
    shard = root / "shards" / "part-0000-of-0001"
    write_immutable_json(
        shard / "invocation.json",
        {
            "schema_version": "decode-mlen-revalidation-invocation/v1",
            "plan_hash": plan["content_hash"],
        },
    )
    nll = {
        1024: (2.10, 2.20),
        2048: (2.11, 2.23),
        None: (2.00, 2.05),
    }
    for profile in profiles.values():
        is_failed = profile.profile_id == failed_profile
        values = nll[
            None if profile.kind == "bf16_reference" else profile.matrix_mlen
        ]
        body = {
            "schema_version": ROW_SCHEMA,
            "plan_hash": plan["content_hash"],
            "profile_id": profile.profile_id,
            "profile": profile.to_dict(),
            "role": (
                "same_split_bf16_reference"
                if profile.kind == "bf16_reference"
                else "same_format_mlen1024_control"
                if profile.matrix_mlen == 1024
                else "candidate_mlen_variant"
            ),
            "shard_index": 0,
            "state": "failed" if is_failed else "succeeded",
            "validation": (
                None
                if is_failed
                else {"mean_token_nll": values[0], "token_count": 512}
            ),
            "refinement": (
                None
                if is_failed
                else {"mean_token_nll": values[1], "token_count": 16384}
            ),
            "error": (
                {
                    "error_class": "OutOfMemoryError",
                    "error_message": "CUDA out of memory",
                    "oom": True,
                }
                if is_failed
                else None
            ),
            "classification": {
                "hardware_bit_parity_verified": False,
                "publication_rankable": False,
            },
        }
        write_immutable_json(shard / "rows" / f"{profile.profile_id}.json", body)
    write_immutable_json(
        shard / "summary.json",
        {
            "schema_version": "decode-mlen-revalidation-shard-summary/v1",
            "plan_hash": plan["content_hash"],
            "complete": True,
        },
    )


class FinalizationTests(unittest.TestCase):
    def test_failure_is_retained_and_selector_input_is_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, profiles = _install_plan(root)
            variant = next(
                profile
                for profile in profiles.values()
                if profile.kind != "bf16_reference" and profile.matrix_mlen == 2048
            )
            _install_rows(root, plan_path, profiles, failed_profile=variant.profile_id)
            completion = finalize(plan_path=plan_path)
            self.assertEqual(completion["schema_version"], COMPLETION_SCHEMA)
            self.assertTrue(completion["complete"])
            self.assertFalse(completion["successful"])
            self.assertTrue(completion["failed_rows_retained"])
            self.assertFalse(completion["selector_input_emitted"])
            self.assertTrue(completion["failed_rows"][0]["error"]["oom"])
            self.assertFalse((root / "corrected_projected_selector_input.json").exists())

    def test_success_reports_geometry_delta_and_matched_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, profiles = _install_plan(root)
            _install_rows(root, plan_path, profiles)
            completion = finalize(plan_path=plan_path)
            self.assertTrue(completion["successful"])
            self.assertTrue(completion["selector_input_emitted"])
            selector = load_immutable_json(
                root / "corrected_projected_selector_input.json"
            )
            self.assertEqual(selector["schema_version"], SELECTOR_INPUT_SCHEMA)
            self.assertTrue(
                selector["all_promoted_candidates_have_matching_numerical_mlen"]
            )
            self.assertFalse(selector["nll_values_reused_or_copied"])
            self.assertFalse(selector["selection_eligible_before_exact_hardware_reprice"])
            row = selector["rows"][0]
            self.assertEqual(row["profile"]["matrix_mlen"], 2048)
            self.assertEqual(row["hardware"]["MLEN"], 2048)
            self.assertAlmostEqual(
                row["suites"]["validation"]["delta_nll_due_to_mlen_geometry"],
                0.01,
            )
            self.assertAlmostEqual(
                row["suites"]["refinement"][
                    "relative_perplexity_vs_same_split_bf16"
                ],
                math.exp(2.23 - 2.05),
            )


if __name__ == "__main__":
    unittest.main()
