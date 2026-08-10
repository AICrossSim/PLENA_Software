"""Contracts for deriving refinement validity from measured stack validity."""

from __future__ import annotations

import pytest

from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.refinement_schedule import (
    DoomedGateDecision,
    DoomedGatePolicy,
    RefinementSchedule,
    RefinementScheduleEntry,
    derive_refinement_validity,
    iter_split_kv_variants,
    _path_hash,
)


def _schedule_over(profiles) -> RefinementSchedule:
    entries = tuple(
        RefinementScheduleEntry(
            ordinal=index,
            profile=profile,
            gate=DoomedGateDecision(
                execution_state="scheduled",
                reason="synthetic",
                threshold_mean_nll=None,
                observed_mean_nll=None,
            ),
        )
        for index, profile in enumerate(profiles)
    )
    return RefinementSchedule(
        entries=entries,
        source_profile_ids=tuple(
            sorted({entry.profile.source_profile.profile_id for entry in entries})
        ),
        reference_mean_nll=1.0,
        policy=DoomedGatePolicy(),
    )


def _stack_validity_document(source_id: str) -> dict:
    return {
        "content_hash": "c" * 64,
        "profiles": {
            source_id: {
                "software_valid": None,
                "compiler_valid": True,
                "emulator_valid": True,
                "rtl_valid": False,
                "dc_calibrated": None,
            }
        },
    }


def test_equal_kv_variant_inherits_measured_validity_split_kv_stays_unmeasured():
    source = DecodePrecisionProfile.quantized(
        "MXINT8", "MXINT8", "MXINT4", "FP_E3M2"
    )
    variants = tuple(iter_split_kv_variants(source))
    schedule = _schedule_over(variants)
    document = _stack_validity_document(source.profile_id)
    manifest = derive_refinement_validity(schedule, document)
    assert manifest.source_schedule_hash == schedule.canonical_hash
    by_id = {record.profile_id: record for record in manifest.records}
    for entry in schedule.entries:
        record = by_id[entry.profile_id]
        equal_kv = (
            entry.profile.key_format
            == entry.profile.value_format
            == source.kv_format
        )
        if equal_kv:
            # Measured layers inherit, including a measured False, with the
            # stack-validity content hash as evidence for every measured field.
            assert record.validity.compiler_valid is True
            assert record.validity.emulator_valid is True
            assert record.validity.rtl_valid is False
            assert record.validity.software_valid is None
            evidence = dict(record.evidence)
            assert evidence["compiler_valid"] == "c" * 64
            assert evidence["rtl_valid"] == "c" * 64
            assert evidence["software_valid"] is None
        else:
            assert all(
                getattr(record.validity, name) is None
                for name in (
                    "software_valid",
                    "compiler_valid",
                    "emulator_valid",
                    "rtl_valid",
                    "dc_calibrated",
                )
            )
            assert all(value is None for _, value in record.evidence)
    assert any(
        entry.profile.key_format != entry.profile.value_format
        for entry in schedule.entries
    )


def test_uncovered_refinement_source_fails_closed():
    source = DecodePrecisionProfile.quantized(
        "MXINT8", "MXINT8", "MXINT4", "FP_E3M2"
    )
    schedule = _schedule_over(tuple(iter_split_kv_variants(source)))
    document = _stack_validity_document("dqp-" + "0" * 64)
    with pytest.raises(ValueError, match="does not cover refinement source"):
        derive_refinement_validity(schedule, document)


def test_path_hash_is_deterministic_for_directories_and_files(tmp_path):
    partition = tmp_path / "part-0000-of-0002"
    (partition / "shards").mkdir(parents=True)
    (partition / "invocation.json").write_text("{}", encoding="utf-8")
    (partition / "shards" / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    first = _path_hash(partition)
    assert first == _path_hash(partition)
    plain = tmp_path / "study.jsonl"
    plain.write_text("{}\n", encoding="utf-8")
    assert _path_hash(plain) != first
    (partition / "shards" / "rows.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    assert _path_hash(partition) != first
    with pytest.raises(ValueError, match="does not exist"):
        _path_hash(tmp_path / "absent")
