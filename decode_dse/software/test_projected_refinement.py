"""Focused CPU tests for the isolated projected-refinement runner."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import unittest

from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.refinement_runner import (
    PROJECTED_REFINEMENT_MERGE_SCHEMA,
    RefinementBankHandle,
    RefinementDocumentMetric,
    RefinementEvaluation,
    RefinementExecutionEvidence,
    RefinementRunner,
    build_refinement_bank_specs,
    load_projected_refinement_merged_results,
    merge_projected_refinement_shards,
    refinement_worker_command,
)
from decode_dse.software.refinement_schedule import (
    ProjectedRefinementShardPlan,
    RefinementAccuracyEvidence,
    build_projected_refinement_shard_plans,
    build_refinement_schedule,
    validate_projected_refinement_shard_plan,
)
from decode_dse.software.runtime_environment import RuntimeEnvironment
from decode_dse.software.token_samples import (
    build_refinement_bundle_from_token_stream,
)


def _schedule():
    source = DecodePrecisionProfile.quantized("MXINT4", "MXINT8", "MXINT4", "FP_E5M6")
    return build_refinement_schedule(
        (source,),
        {
            source.profile_id: RefinementAccuracyEvidence.succeeded(
                source.profile_id, 2.0
            )
        },
        reference_mean_nll=2.0,
        require_symmetric_kv=True,
    )


class ProjectedShardContractTests(unittest.TestCase):
    def test_single_symmetric_profile_has_one_active_and_three_empty_shards(self):
        schedule = _schedule()
        plans = build_projected_refinement_shard_plans(schedule)
        self.assertEqual(len(plans), 4)
        self.assertEqual([len(plan.profile_ids) for plan in plans], [1, 0, 0, 0])
        self.assertEqual(
            tuple(profile_id for plan in plans for profile_id in plan.profile_ids),
            tuple(entry.profile_id for entry in schedule.entries),
        )

    def test_partition_tamper_fails_closed(self):
        schedule = _schedule()
        plan = build_projected_refinement_shard_plans(schedule)[1]
        tampered = ProjectedRefinementShardPlan(
            master_schedule_hash=plan.master_schedule_hash,
            shard_index=plan.shard_index,
            shard_count=plan.shard_count,
            source_profile_id=plan.source_profile_id,
            profile_ids=(schedule.entries[0].profile_id,),
        )
        with self.assertRaisesRegex(ValueError, "deterministic partition"):
            validate_projected_refinement_shard_plan(schedule, tampered)

    def test_worker_command_uses_distinct_projected_flag(self):
        path = Path("/tmp/placeholder")
        common = dict(
            config=path,
            schedule=path,
            shard_plan=path,
            sample_bundle=path,
            prefill_root=path,
            admission_root=path,
            calibration=path,
            calibration_receipt=path,
            checkpoint_root=path,
            output_dir=path,
            work_root=path,
            device_label="B200",
            decode_microbatch_size=8,
            bootstrap_replicates=100,
        )
        strict = refinement_worker_command(**common)
        projected = refinement_worker_command(**common, projected=True)
        self.assertIn("--shard-plan", strict)
        self.assertNotIn("--projected-shard-plan", strict)
        self.assertIn("--projected-shard-plan", projected)
        self.assertNotIn("--shard-plan", projected)


class _Executor:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.runtime = RuntimeEnvironment(
            logical={"test_runtime": "projected-refinement/v1"},
            observation={"device": "synthetic"},
        )

    def runtime_environment(self):
        return self.runtime.to_dict()

    @contextmanager
    def open_weight_bank(self, bank, entries):
        del entries
        yield RefinementBankHandle(
            bank_id=bank.bank_id,
            checkpoint_tree_sha256="a" * 64,
            weight_identity_before="synthetic-weight-identity",
        )

    @contextmanager
    def open_split_kv_admission_cache(self, key_format, value_format, samples):
        del key_format, value_format, samples
        yield object()

    def evaluate(self, entry, *, samples, weight_bank, kv_admission_cache):
        del entry, kv_admission_cache
        documents = tuple(
            RefinementDocumentMetric(
                document_id=sample.document_id,
                source_cluster_id=sample.source_cluster_id,
                nll_sum=128.0,
                token_count=128,
                initial_cache_length=512,
                final_cache_length=640,
            )
            for sample in samples.samples
        )
        return RefinementEvaluation(
            documents=documents,
            evidence=RefinementExecutionEvidence(
                prefill_precision="BF16",
                prefill_kv_precision="BF16",
                first_token_owner="prefill",
                q_len_values=(1,),
                exact_cache_positions=True,
                independent_batch_caches=True,
                admission_count_per_prompt=1,
                direct_native_kv_append=True,
                runtime_rebinding=True,
                weight_requantizations=0,
                weight_identity_before=weight_bank.weight_identity_before,
                weight_identity_after=weight_bank.weight_identity_before,
                checkpoint_tree_sha256=weight_bank.checkpoint_tree_sha256,
            ),
            artifacts=(str(self.artifact),),
        )


class ProjectedMergeTests(unittest.TestCase):
    def test_four_shard_merge_retains_exact_terminal_coverage(self):
        schedule = _schedule()
        token_count = 128 * (512 + 1 + 128)
        samples = build_refinement_bundle_from_token_stream(
            (index % 32000 for index in range(token_count)),
            model_revision="revision",
            tokenizer_revision="revision",
            dataset_name="synthetic",
            dataset_revision="revision",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.pt"
            calibration.write_bytes(b"sealed-calibration")
            artifact = root / "measurement.json"
            artifact.write_text("{}\n", encoding="utf-8")
            calibration_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()
            plans = build_projected_refinement_shard_plans(schedule)
            shard_roots = []
            for plan in plans:
                output = root / "results" / "shards" / f"shard-{plan.shard_index:02d}"
                banks = build_refinement_bank_specs(
                    schedule,
                    model_name="target",
                    model_revision="revision",
                    calibration_dataset="synthetic",
                    calibration_revision="revision",
                    calibration_bundle_hash=calibration_hash,
                    calibration_path=calibration,
                    checkpoint_root=root
                    / "checkpoints"
                    / f"shard-{plan.shard_index:02d}",
                    profile_ids=plan.profile_ids,
                )
                summary = RefinementRunner(
                    schedule=schedule,
                    samples=samples,
                    banks=banks,
                    executor=_Executor(artifact),
                    output_dir=output,
                    bootstrap_replicates=100,
                    projected_shard_plan=plan,
                ).run()
                self.assertEqual(summary.pending, 0)
                shard_roots.append(output)
            summary = merge_projected_refinement_shards(
                schedule=schedule,
                samples=samples,
                shard_roots=shard_roots,
                output_dir=root / "results" / "merged",
            )
            loaded = load_projected_refinement_merged_results(
                schedule, summary.receipt_path
            )
            self.assertEqual(
                loaded.receipt["schema_version"],
                PROJECTED_REFINEMENT_MERGE_SCHEMA,
            )
            self.assertEqual(len(loaded.terminal_rows), 1)
            self.assertEqual(loaded.terminal_rows[0]["state"], "succeeded")
            with self.assertRaisesRegex(ValueError, "unsupported refinement merge"):
                from decode_dse.software.refinement_runner import (
                    load_refinement_merged_results,
                )

                load_refinement_merged_results(schedule, summary.receipt_path)


if __name__ == "__main__":
    unittest.main()
