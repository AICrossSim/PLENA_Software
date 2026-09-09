"""Fail-closed numerical revalidation for profile/hardware MLEN mismatches.

The exhaustive screen is evaluated with one matrix instruction partition.
Hardware candidates using another ``MLEN`` therefore need a new numerical
profile: MLEN changes every per-partial rounding boundary and consequently is
part of the profile identity.  This module builds those profiles from the
exact screened source with :func:`dataclasses.replace`, executes the screened
MLEN control, candidate MLEN variants, and BF16 control on the same validation
and refinement splits, and emits no selector input until coverage is complete.

This is deliberately a numerical derivative lane.  It never grants compiler,
emulator, RTL, timing, power, area, or publication validity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence

from decode_dse.legality import StackValidity, evaluate_profile_legality
from decode_dse.manifest import (
    SweepManifest,
    SweepManifestEntry,
    load_manifest,
)
from decode_dse.profiles import (
    FOUNDATION_MATRIX_MLEN,
    PROFILE_KIND_BF16_REFERENCE,
    PROFILE_KIND_QUANTIZED,
    PROFILE_SCHEMA,
    DecodePrecisionProfile,
    format_descriptor,
)
from decode_dse.software.cached_decode import (
    ContinuationExample,
    TorchHFCachedDecodeBackend,
    evaluate_teacher_forced_cached_batched,
)
from decode_dse.software.decode_evaluator import (
    AdmissionCacheHandle,
    DecodeEvaluator,
    _DecodeCacheLRU,
    _document_token,
)
from decode_dse.software.refinement_evaluator import RefinementEvaluator
from decode_dse.software.sweep import _sweep_launcher_load_config
from decode_dse.software.sweep_plan import (
    HARDWARE_VALIDATION_SAMPLE_CONTRACT,
    ExecutorContext,
    PromptManifest,
    StageSampleContract,
    SweepRunPlan,
    load_immutable_json,
    write_immutable_json,
)
from decode_dse.software.token_samples import load_refinement_sample_bundle


PLAN_SCHEMA = "decode-mlen-revalidation-plan/v1"
INVOCATION_SCHEMA = "decode-mlen-revalidation-invocation/v1"
ROW_SCHEMA = "decode-mlen-revalidation-row/v1"
COMPLETION_SCHEMA = "decode-mlen-revalidation-completion/v1"
SELECTOR_INPUT_SCHEMA = "decode-mlen-corrected-projected-selector-input/v1"
MIXED_ABI_SCHEMA = "decode-mixed-weight-activation-mase-abi/v1"

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
SOURCE_MLEN = FOUNDATION_MATRIX_MLEN
CANDIDATE_MLENS = (2048, 4096)
MAX_SHARDS = 4
REFINEMENT_DOCUMENTS = 128
REFINEMENT_DECODE_STEPS = 128


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    token = str(value)
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _profile_axes(profile: DecodePrecisionProfile) -> dict[str, Any]:
    return {
        "kind": profile.kind,
        "weight_format": profile.weight_format,
        "activation_format": profile.activation_format,
        "key_format": profile.key_format,
        "value_format": profile.value_format,
        "vector_format": profile.vector_format,
        "block_size": profile.block_size,
        "scale_format": profile.scale_format,
        "scale_bits": profile.scale_bits,
        "method": profile.method,
        "weight_operators": list(profile.weight_operators),
        "activation_operators": list(profile.activation_operators),
        "kv_operators": list(profile.kv_operators),
        "vector_operators": list(profile.vector_operators),
        "bf16_operators": list(profile.bf16_operators),
    }


def _validated_oracle_hash(profile: DecodePrecisionProfile) -> str:
    """Validate the serialized oracle exactly and return its canonical hash."""

    serialized = profile.to_dict().get("numerical_oracle")
    oracle = profile.numerical_oracle_contract
    if serialized != oracle:
        raise ValueError("serialized numerical_oracle differs from the profile property")
    expected_mlen = (
        None if profile.kind == PROFILE_KIND_BF16_REFERENCE else profile.matrix_mlen
    )
    if (
        not isinstance(oracle, Mapping)
        or oracle.get("schema_version") != "plena-mase-matrix-oracle/v1"
        or oracle.get("matrix_mlen") != expected_mlen
        or oracle.get("hardware_bit_parity_verified") is not False
    ):
        raise ValueError("profile numerical_oracle boundary is invalid")
    return _content_hash(oracle)


def _matrix_geometry_receipt() -> dict[str, Any]:
    """Record why MLEN=2048 and 4096 are not numerically equivalent."""

    reductions = {
        "attention_qkv_projection": 2048,
        "attention_output_projection": 32 * 128,
        "routed_gate_up_projection": 2048,
        "routed_down_projection": 768,
        "decode_lm_head": 2048,
        "qk_matmul": 128,
    }
    differing = {
        name: {
            "reduction_k": width,
            "partials_at_mlen2048": math.ceil(width / 2048),
            "partials_at_mlen4096": math.ceil(width / 4096),
        }
        for name, width in reductions.items()
        if math.ceil(width / 2048) != math.ceil(width / 4096)
    }
    if differing != {
        "attention_output_projection": {
            "reduction_k": 4096,
            "partials_at_mlen2048": 2,
            "partials_at_mlen4096": 1,
        }
    }:
        raise AssertionError("target reduction-dimension proof changed")
    return {
        "schema_version": "decode-mlen-equivalence-audit/v1",
        "target_reduction_dimensions": reductions,
        "mlen2048_mlen4096_equivalent": False,
        "counterexamples": differing,
        "deduplication_permitted": False,
        "reason": (
            "attention o_proj reduces 32 query heads times head_dim 128; "
            "MLEN2048 rounds two partials and MLEN4096 rounds one"
        ),
    }


def derive_mlen_variant(
    source: DecodePrecisionProfile,
    matrix_mlen: int,
) -> DecodePrecisionProfile:
    """Derive one exact v2 variant and prove MLEN is the only changed axis."""

    if source.schema_version != PROFILE_SCHEMA:
        raise ValueError("MLEN revalidation requires a v2 source profile")
    if source.kind != PROFILE_KIND_QUANTIZED or source.method != "rtn":
        raise ValueError("MLEN revalidation is restricted to quantized RTN profiles")
    if source.matrix_mlen != SOURCE_MLEN:
        raise ValueError(f"source profile must have matrix_mlen={SOURCE_MLEN}")
    if matrix_mlen not in CANDIDATE_MLENS:
        raise ValueError(f"candidate matrix_mlen must be one of {CANDIDATE_MLENS}")
    if source.key_format != source.value_format:
        raise ValueError("MLEN revalidation requires symmetric K/V")
    variant = replace(source, matrix_mlen=matrix_mlen)
    if _profile_axes(variant) != _profile_axes(source):
        raise AssertionError("dataclasses.replace changed a non-MLEN profile axis")
    if variant.profile_id == source.profile_id:
        raise AssertionError("MLEN change did not change profile identity")
    oracle = variant.numerical_oracle_contract
    if (
        oracle.get("schema_version") != "plena-mase-matrix-oracle/v1"
        or oracle.get("matrix_mlen") != matrix_mlen
        or oracle.get("implementation")
        != "chop.nn.quantized.functional.matrix.plena_matrix_product"
        or oracle.get("operand_materialization_dtype") != "FP32"
        or oracle.get("partition_reduction_dtype") != "FP32"
        or oracle.get("partial_rounding")
        != "round_to_nearest_even_to_profile.vector_format"
        or oracle.get("hardware_bit_parity_verified") is not False
    ):
        raise AssertionError("derived profile numerical oracle is not exact-MLEN bound")
    _validated_oracle_hash(variant)
    local = variant.local_head_contract
    if (
        local.get("matrix_mlen") != matrix_mlen
        or local.get("weight_format") != variant.weight_format
        or local.get("activation_format") != variant.activation_format
        or local.get("matrix_storage_format") != variant.vector_format
    ):
        raise AssertionError("derived local-head contract is not MLEN/profile bound")
    return variant


def _mixed_weight_activation(profile: DecodePrecisionProfile) -> bool:
    if profile.kind == PROFILE_KIND_BF16_REFERENCE:
        return False
    return (
        format_descriptor(profile.weight_format).family
        != format_descriptor(profile.activation_format).family
    )


def _source_hashes(mase_src: Path) -> dict[str, str]:
    relatives = (
        "chop/nn/quantized/modules/linear.py",
        "chop/nn/quantized/modules/qwen3_moe/experts.py",
        "chop/nn/quantized/functional/matrix.py",
    )
    result = {}
    for relative in relatives:
        path = mase_src / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _file_hash(path)
    return result


def audit_mixed_weight_activation_abi(mase_src: str | Path) -> dict[str, Any]:
    """Execute both mixed W/A directions through the current MASE ABI.

    The hardware contract still labels mixed-family deployment unsupported.
    This probe answers the narrower question needed by the numerical lane:
    whether MASE really quantizes the activation with the other family rather
    than silently applying the weight-family activation quantizer.
    """

    root = Path(mase_src).resolve()
    receipt: dict[str, Any] = {
        "schema_version": MIXED_ABI_SCHEMA,
        "mase_src": str(root),
        "source_sha256": {},
        "probes": [],
        "numerical_execution_supported": False,
        "hardware_deployment_supported": False,
        "publication_validity_granted": False,
    }
    try:
        receipt["source_sha256"] = _source_hashes(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import torch
        from chop.nn.quantized.modules.linear import LinearMXFP, LinearMXInt
        from chop.nn.quantized.modules.phase_context import force_runtime_phase
        from chop.nn.quantized.modules.qwen3_moe.experts import (
            _PhaseAwareQwen3MoeExperts,
        )
        from decode_dse.software.precision_bindings import (
            _LM_HEAD_RE,
            DecodeQuantSpec,
            build_decode_pass_args,
        )

        directions = (
            ("mxint", 4, "mxfp", (2, 1), LinearMXInt),
            ("mxfp", (2, 1), "mxint", 4, LinearMXFP),
        )
        for weight_family, weight_width, act_family, act_width, cls in directions:
            spec = DecodeQuantSpec(
                attn_w=weight_width,
                ffn_w=weight_width,
                kv=weight_width,
                w_fmt=weight_family,
                kv_fmt=weight_family,
                act_w=act_width,
                act_fmt=act_family,
                fp_setting="FP_E8M5",
                matrix_mlen=8,
            )
            raw = build_decode_pass_args("abi-probe", "cpu", spec)[_LM_HEAD_RE][
                "config"
            ]
            config = dict(raw)
            name = config.pop("name")
            if name != weight_family:
                raise AssertionError("linear class route is not weight-family bound")
            layer = cls.from_linear(
                torch.nn.Linear(8, 8, bias=False, dtype=torch.float32),
                config,
            )
            value = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32).reshape(1, 8)
            with force_runtime_phase("decode"):
                output = layer(value)
            activation_config = layer.decode_config
            expected_key = (
                "data_in_width"
                if act_family == "mxint"
                else "data_in_exponent_width"
            )
            forbidden_key = (
                "data_in_exponent_width"
                if act_family == "mxint"
                else "data_in_width"
            )
            if (
                output.shape != (1, 8)
                or not torch.isfinite(output).all().item()
                or activation_config.get(expected_key) is None
                or activation_config.get(forbidden_key) is not None
            ):
                raise RuntimeError("mixed linear activation route did not execute exactly")
            # The fused expert container owns a separate activation hook.  Run
            # the hook itself in both directions; construction of a 128-expert
            # target just to test dispatch would allocate the full expert bank.
            expert = object.__new__(_PhaseAwareQwen3MoeExperts)
            expert_output = _PhaseAwareQwen3MoeExperts._quantize_activation(
                expert,
                value,
                activation_config,
            )
            if expert_output.shape != value.shape or not torch.isfinite(
                expert_output
            ).all().item():
                raise RuntimeError("mixed fused-expert activation route failed")
            receipt["probes"].append(
                {
                    "weight_family": weight_family,
                    "activation_family": act_family,
                    "linear_class": cls.__name__,
                    "linear_forward_executed": True,
                    "fused_expert_activation_executed": True,
                    "activation_config_key": expected_key,
                    "matrix_family_contract": config.get(
                        "operand_family_binding"
                    ),
                }
            )
        receipt["numerical_execution_supported"] = True
    except Exception as error:  # retained as an explicit fail-closed receipt
        receipt["error_class"] = type(error).__name__
        receipt["error_message"] = str(error)
    receipt["receipt_hash"] = _content_hash(receipt)
    return receipt


def _validate_source_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "source_profile_id",
        "candidate_id",
        "source_record_hash",
        "source_artifact_sha256",
        "source_artifact_path",
        "profile",
        "hardware",
        "roles",
        "selection_metrics",
    }
    if set(value) != required:
        raise ValueError("source candidate fields differ from the lane schema")
    profile = DecodePrecisionProfile.from_dict(value["profile"])
    if profile.profile_id != value["source_profile_id"]:
        raise ValueError("source profile identity differs from its canonical hash")
    if profile.matrix_mlen != SOURCE_MLEN:
        raise ValueError("source candidate is not from the screened MLEN")
    hardware = value["hardware"]
    if not isinstance(hardware, Mapping):
        raise TypeError("source candidate hardware must be an object")
    candidate_mlen = hardware.get("MLEN")
    if candidate_mlen not in CANDIDATE_MLENS:
        raise ValueError("source candidate hardware MLEN does not require revalidation")
    roles = value["roles"]
    if (
        not isinstance(roles, list)
        or not roles
        or roles != sorted(set(str(role) for role in roles))
    ):
        raise ValueError("source candidate roles must be a canonical nonempty list")
    source_record_hash = _sha256(value["source_record_hash"], "source record hash")
    artifact_hash = _sha256(
        value["source_artifact_sha256"], "source artifact SHA-256"
    )
    candidate_id = str(value["candidate_id"])
    if not candidate_id:
        raise ValueError("candidate_id must be nonempty")
    hardware_identity = {
        "source_profile_id": profile.profile_id,
        "candidate_id": candidate_id,
        "source_record_hash": source_record_hash,
        "source_artifact_sha256": artifact_hash,
        "hardware": dict(hardware),
    }
    return {
        **dict(value),
        "profile": profile.to_dict(),
        "hardware": dict(hardware),
        "roles": list(roles),
        "hardware_identity_hash": _content_hash(hardware_identity),
    }


def _assign_weight_banks(
    profile_records: Sequence[Mapping[str, Any]],
    max_shards: int,
) -> list[dict[str, Any]]:
    if isinstance(max_shards, bool) or not 1 <= int(max_shards) <= MAX_SHARDS:
        raise ValueError(f"max_shards must be in [1, {MAX_SHARDS}]")
    groups: dict[str, list[str]] = {}
    for record in profile_records:
        profile = DecodePrecisionProfile.from_dict(record["profile"])
        groups.setdefault(profile.weight_format, []).append(profile.profile_id)
    shard_count = min(int(max_shards), len(groups))
    if shard_count <= 0:
        raise ValueError("revalidation plan has no weight banks")
    shards = [
        {"shard_index": index, "weight_formats": [], "profile_ids": []}
        for index in range(shard_count)
    ]
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for weight_format, profile_ids in ordered:
        target = min(
            shards,
            key=lambda shard: (
                len(shard["profile_ids"]),
                len(shard["weight_formats"]),
                shard["shard_index"],
            ),
        )
        target["weight_formats"].append(weight_format)
        target["profile_ids"].extend(sorted(profile_ids))
    for shard in shards:
        shard["weight_formats"] = sorted(shard["weight_formats"])
        shard["profile_ids"] = sorted(shard["profile_ids"])
        shard["profile_count"] = len(shard["profile_ids"])
    return shards


def build_plan(
    *,
    config_path: str | Path,
    numerical_workspace: str | Path,
    refinement_workspace: str | Path,
    output_root: str | Path,
    source_spec: Mapping[str, Any],
    max_shards: int = MAX_SHARDS,
) -> dict[str, Any]:
    """Build and immutably install a geometry-revalidation plan."""

    config_path = Path(config_path).resolve()
    numerical_workspace = Path(numerical_workspace).resolve()
    refinement_workspace = Path(refinement_workspace).resolve()
    output_root = Path(output_root).resolve()
    config = _sweep_launcher_load_config(config_path)
    if (
        config.get("model_name") != MODEL_NAME
        or config.get("model_revision") != MODEL_REVISION
        or config.get("tokenizer_revision") != MODEL_REVISION
    ):
        raise ValueError("MLEN revalidation is sealed to the exact Qwen3-MoE target")
    master = load_manifest(numerical_workspace / "manifest.json")
    run_plan_value = load_immutable_json(numerical_workspace / "run_plan.json")
    run_plan = SweepRunPlan.from_dict(run_plan_value)
    prompts_value = load_immutable_json(numerical_workspace / "prompt_manifest.json")
    prompts = PromptManifest.from_dict(prompts_value)
    if (
        master.model_name != MODEL_NAME
        or master.model_revision != MODEL_REVISION
        or run_plan.manifest_hash != master.canonical_hash
    ):
        raise ValueError("numerical workspace identities are inconsistent")

    refinement = config.get("refinement")
    if not isinstance(refinement, Mapping):
        raise ValueError("config.refinement is required")

    def refinement_path(field: str) -> Path:
        token = str(refinement[field])
        if not token.startswith("workspace://"):
            raise ValueError(f"refinement.{field} must be workspace-bound")
        relative = Path(token.removeprefix("workspace://"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"refinement.{field} escapes its workspace")
        return (refinement_workspace / relative).resolve()

    refinement_samples_path = refinement_path("sample_bundle")
    refinement_prefill_root = refinement_path("prefill_artifact_root")
    refinement_admission_root = refinement_path("admission_artifact_root")
    refinement_samples = load_refinement_sample_bundle(refinement_samples_path)
    if (
        len(refinement_samples.samples) != REFINEMENT_DOCUMENTS
        or any(
            len(sample.decode_target_ids) < REFINEMENT_DECODE_STEPS
            for sample in refinement_samples.samples
        )
    ):
        raise ValueError("refinement split must contain 128 documents and decode steps")
    refinement_prefill_index = load_immutable_json(
        refinement_prefill_root / "index.json"
    )
    if refinement_prefill_index.get("sample_bundle_hash") != refinement_samples.canonical_hash:
        raise ValueError("refinement prefill index uses another sample split")

    if source_spec.get("schema_version") != "decode-mlen-revalidation-source/v1":
        raise ValueError("unsupported Results source-spec schema")
    target = source_spec.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("model_name") != MODEL_NAME
        or target.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("source spec targets another model")
    source_bindings = source_spec.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise ValueError("source spec lacks verified input bindings")
    for key, value in source_bindings.items():
        if key.endswith("sha256") or key.endswith("hash"):
            _sha256(value, f"source binding {key}")
    raw_candidates = source_spec.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("source spec has no promoted candidate")
    candidates = [_validate_source_candidate(value) for value in raw_candidates]
    candidate_keys = [
        (str(value["source_profile_id"]), str(value["candidate_id"]))
        for value in candidates
    ]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("source spec repeats a hardware candidate identity")

    master_by_id = {entry.profile_id: entry.profile for entry in master.entries}
    source_profiles: dict[str, DecodePrecisionProfile] = {}
    variants: dict[str, DecodePrecisionProfile] = {}
    candidate_mapping = []
    for candidate in candidates:
        source = DecodePrecisionProfile.from_dict(candidate["profile"])
        installed = master_by_id.get(source.profile_id)
        if installed is None or installed != source:
            raise ValueError("promoted source profile is not exact in the sealed manifest")
        source_profiles[source.profile_id] = source
        matrix_mlen = int(candidate["hardware"]["MLEN"])
        variant = derive_mlen_variant(source, matrix_mlen)
        variants[variant.profile_id] = variant
        candidate_mapping.append(
            {
                "source_profile_id": source.profile_id,
                "revalidated_profile_id": variant.profile_id,
                "candidate_id": candidate["candidate_id"],
                "candidate_matrix_mlen": matrix_mlen,
                "source_record_hash": candidate["source_record_hash"],
                "source_artifact_path": candidate["source_artifact_path"],
                "source_artifact_sha256": candidate["source_artifact_sha256"],
                "hardware_identity_hash": candidate["hardware_identity_hash"],
                "hardware": candidate["hardware"],
                "roles": candidate["roles"],
                "selection_metrics": candidate["selection_metrics"],
            }
        )

    bf16_entries = [
        entry
        for entry in master.entries
        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
    ]
    if len(bf16_entries) != 1:
        raise ValueError("sealed manifest must contain exactly one BF16 reference")
    bf16 = bf16_entries[0].profile
    evaluation_profiles = []
    for profile in sorted(source_profiles.values(), key=lambda item: item.profile_id):
        oracle = profile.numerical_oracle_contract
        if (
            oracle.get("matrix_mlen") != SOURCE_MLEN
            or oracle.get("hardware_bit_parity_verified") is not False
        ):
            raise ValueError("source profile numerical oracle is not the sealed MLEN control")
        evaluation_profiles.append(
            {
                "profile_id": profile.profile_id,
                "profile": profile.to_dict(),
                "numerical_oracle_sha256": _validated_oracle_hash(profile),
                "role": "same_format_mlen1024_control",
                "source_profile_id": profile.profile_id,
            }
        )
    for profile in sorted(variants.values(), key=lambda item: item.profile_id):
        sources = sorted(
            {
                item["source_profile_id"]
                for item in candidate_mapping
                if item["revalidated_profile_id"] == profile.profile_id
            }
        )
        evaluation_profiles.append(
            {
                "profile_id": profile.profile_id,
                "profile": profile.to_dict(),
                "numerical_oracle_sha256": _validated_oracle_hash(profile),
                "role": "candidate_mlen_variant",
                "source_profile_ids": sources,
            }
        )
    evaluation_profiles.append(
        {
            "profile_id": bf16.profile_id,
            "profile": bf16.to_dict(),
            "numerical_oracle_sha256": _validated_oracle_hash(bf16),
            "role": "same_split_bf16_reference",
            "source_profile_id": None,
        }
    )
    evaluation_profiles.sort(key=lambda value: str(value["profile_id"]))
    if len({value["profile_id"] for value in evaluation_profiles}) != len(
        evaluation_profiles
    ):
        raise AssertionError("evaluation profile identities are not unique")
    partitions = _assign_weight_banks(evaluation_profiles, max_shards)
    mixed_profiles = sorted(
        profile.profile_id
        for profile in (*source_profiles.values(), *variants.values())
        if _mixed_weight_activation(profile)
    )
    body = {
        "schema_version": PLAN_SCHEMA,
        "target": {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
        },
        "paths": {
            "config": str(config_path),
            "numerical_workspace": str(numerical_workspace),
            "refinement_workspace": str(refinement_workspace),
            "refinement_sample_bundle": str(refinement_samples_path),
            "refinement_prefill_root": str(refinement_prefill_root),
            "refinement_admission_root": str(refinement_admission_root),
            "output_root": str(output_root),
        },
        "bindings": {
            "config_sha256": _file_hash(config_path),
            "master_manifest_hash": master.canonical_hash,
            "run_plan_hash": run_plan.canonical_hash,
            "prompt_manifest_hash": prompts.canonical_hash,
            "refinement_sample_bundle_hash": refinement_samples.canonical_hash,
            "refinement_sample_file_sha256": _file_hash(refinement_samples_path),
            "refinement_prefill_index_hash": refinement_prefill_index[
                "content_hash"
            ],
            "source_spec_hash": _content_hash(source_spec),
            **dict(source_bindings),
        },
        "sample_suites": {
            "validation": {
                "prompt_set": HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_set,
                "document_count": HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count,
                "decode_steps": HARDWARE_VALIDATION_SAMPLE_CONTRACT.decode_steps,
                "q_len": 1,
            },
            "refinement": {
                "sample_bundle_hash": refinement_samples.canonical_hash,
                "document_count": REFINEMENT_DOCUMENTS,
                "decode_steps": REFINEMENT_DECODE_STEPS,
                "q_len": 1,
            },
        },
        "matrix_geometry": _matrix_geometry_receipt(),
        "candidate_mapping": sorted(
            candidate_mapping,
            key=lambda value: (
                value["source_profile_id"],
                value["candidate_id"],
            ),
        ),
        "evaluation_profiles": evaluation_profiles,
        "mixed_weight_activation": {
            "profile_ids_requiring_runtime_abi_probe": mixed_profiles,
            "failure_policy": "terminal_explicit_failure_no_silent_fallback",
            "hardware_deployment_supported": False,
        },
        "sharding": {
            "algorithm": "whole_rtn_weight_bank_lpt/v1",
            "max_shards": int(max_shards),
            "shard_count": len(partitions),
            "partitions": partitions,
            "same_weight_bank_shared_across_mlen_controls_and_variants": True,
        },
        "classification": {
            "evidence_class": "measured_mlen_numerical_revalidation",
            "measured_numerical": True,
            "publication_rankable": False,
            "publication_selection_eligible": False,
            "strict_pipeline_valid": False,
            "hardware_rankable": False,
            "compiler_valid": False,
            "emulator_valid": False,
            "rtl_valid": False,
            "hardware_bit_parity_verified": False,
            "failed_rows_retained": True,
        },
        "completion_policy": {
            "all_evaluation_profiles_terminal": True,
            "all_controls_and_variants_must_succeed_for_selector_input": True,
            "failed_or_oom_rows_are_never_dropped": True,
            "no_nll_reuse_or_copy": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "mlen_revalidation_plan.json"
    write_immutable_json(plan_path, body)
    return load_immutable_json(plan_path)


def _profile_manifest(
    master: SweepManifest,
    profiles: Sequence[DecodePrecisionProfile],
) -> SweepManifest:
    entries = tuple(
        SweepManifestEntry(
            ordinal=index,
            profile=profile,
            legality=evaluate_profile_legality(profile),
            validity=StackValidity(),
        )
        for index, profile in enumerate(profiles)
    )
    return SweepManifest(
        model_name=master.model_name,
        model_revision=master.model_revision,
        model_architecture=master.model_architecture,
        tokenizer_revision=master.tokenizer_revision,
        quantizer_provenance=master.quantizer_provenance,
        entries=entries,
    )


def _load_plan(path: str | Path) -> dict[str, Any]:
    value = load_immutable_json(Path(path).resolve())
    if value.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported MLEN revalidation plan")
    if value.get("target", {}).get("model_revision") != MODEL_REVISION:
        raise ValueError("MLEN revalidation plan targets another revision")
    profiles = _plan_profiles(value)
    mappings = value.get("candidate_mapping")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("MLEN revalidation plan has no hardware mapping")
    identities = set()
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            raise TypeError("MLEN candidate mapping must be an object")
        source = profiles.get(str(mapping.get("source_profile_id")))
        variant = profiles.get(str(mapping.get("revalidated_profile_id")))
        if source is None or variant is None:
            raise ValueError("MLEN candidate mapping names an unknown profile")
        matrix_mlen = int(mapping.get("candidate_matrix_mlen", 0))
        if (
            derive_mlen_variant(source, matrix_mlen) != variant
            or mapping.get("hardware", {}).get("MLEN") != matrix_mlen
        ):
            raise ValueError("MLEN candidate mapping is not an exact derived variant")
        identity_body = {
            "source_profile_id": source.profile_id,
            "candidate_id": str(mapping.get("candidate_id")),
            "source_record_hash": _sha256(
                mapping.get("source_record_hash"), "mapped source record hash"
            ),
            "source_artifact_sha256": _sha256(
                mapping.get("source_artifact_sha256"),
                "mapped source artifact SHA-256",
            ),
            "hardware": dict(mapping.get("hardware", {})),
        }
        identity_hash = _content_hash(identity_body)
        if mapping.get("hardware_identity_hash") != identity_hash:
            raise ValueError("MLEN candidate hardware identity hash differs")
        key = source.profile_id, str(mapping.get("candidate_id"))
        if key in identities:
            raise ValueError("MLEN candidate mapping repeats a hardware identity")
        identities.add(key)
    return value


def _plan_profiles(plan: Mapping[str, Any]) -> dict[str, DecodePrecisionProfile]:
    result = {}
    for record in plan["evaluation_profiles"]:
        profile = DecodePrecisionProfile.from_dict(record["profile"])
        if record.get("profile_id") != profile.profile_id or profile.profile_id in result:
            raise ValueError("plan evaluation-profile identity is invalid")
        if record.get("numerical_oracle_sha256") != _validated_oracle_hash(profile):
            raise ValueError("plan numerical-oracle identity differs from its profile")
        result[profile.profile_id] = profile
    return result


def _validation_context(
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    master: SweepManifest,
    run_plan: SweepRunPlan,
    prompts: PromptManifest,
    stage_manifest: SweepManifest,
    shard_index: int,
    shard_count: int,
    device_label: str,
    output_dir: Path,
) -> ExecutorContext:
    contract = StageSampleContract(
        name="mlen-geometry-validation",
        prompt_set=HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_set,
        prompt_count=HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count,
        prefill_tokens=HARDWARE_VALIDATION_SAMPLE_CONTRACT.prefill_tokens,
        decode_steps=HARDWARE_VALIDATION_SAMPLE_CONTRACT.decode_steps,
        q_len=1,
        teacher_forced_cached=True,
        compiler_required=False,
        emulator_required=False,
    )
    return ExecutorContext(
        stage="mlen-geometry-validation",
        workspace_root=Path(plan["paths"]["numerical_workspace"]),
        output_dir=output_dir,
        config=config,
        master_manifest=master,
        stage_manifest=stage_manifest,
        run_plan=run_plan,
        prompts=prompts,
        sample_contract=contract,
        shard_index=shard_index,
        shard_count=shard_count,
        device_label=device_label,
    )


@contextmanager
def _refinement_admission(
    evaluator: RefinementEvaluator,
    profile: DecodePrecisionProfile,
) -> Iterator[AdmissionCacheHandle]:
    if profile.kind != PROFILE_KIND_BF16_REFERENCE:
        with evaluator.open_split_kv_admission_cache(
            profile.key_format,
            profile.value_format,
            evaluator.bundle,
        ) as value:
            yield value[0]
        return

    engine = evaluator.engine
    runtime_parent = engine.admission_root / ".runtime"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(tempfile.mkdtemp(prefix="bf16-", dir=runtime_parent))
    provenance = engine._admission_provenance("BF16", "BF16")
    try:
        paths = {
            sample.document_id: engine._load_or_create_admitted(
                sample,
                "BF16",
                "BF16",
                provenance,
                artifact_root=runtime_root,
            )[0]
            for sample in evaluator.bundle.samples
        }
        yield AdmissionCacheHandle(kv_format="BF16", paths=paths)
    finally:
        engine.cache_lru.clear()
        shutil.rmtree(runtime_root, ignore_errors=True)


def _evaluate_refinement(
    evaluator: RefinementEvaluator,
    entry: SweepManifestEntry,
    weight_bank: Any,
    admission: AdmissionCacheHandle,
) -> dict[str, Any]:
    import torch
    from chop.nn.quantized.modules.phase_context import force_runtime_phase
    from decode_dse.software.cache_artifacts import (
        load_prefill_artifact,
    )

    engine = evaluator.engine
    binding = engine._bind_profile(weight_bank, entry.profile)
    if binding.weight_requantizations != 0:
        raise RuntimeError("MLEN runtime rebind requantized the sealed RTN bank")
    engine._native_append_validation_calls = 0
    engine._native_append_tensor_checks = 0
    engine._native_append_quantized_tensor_checks = 0
    engine._native_append_validation_seconds = 0.0
    backend = TorchHFCachedDecodeBackend(
        device=weight_bank.device,
        append_validator=(
            lambda cache, start, end, artifact: engine._validate_native_append(
                cache, start, end, artifact
            )
            if engine._native_append_validation_calls == 0
            else None
        ),
        native_append_format=True,
    )
    cuda_device = weight_bank.device if weight_bank.device.type == "cuda" else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    documents = []
    with force_runtime_phase("decode"):
        for offset in range(0, len(evaluator.bundle.samples), evaluator.decode_microbatch_size):
            batch = evaluator.bundle.samples[
                offset : offset + evaluator.decode_microbatch_size
            ]
            examples = []
            for sample in batch:
                prefill = load_prefill_artifact(
                    evaluator.prefill_root / _document_token(sample.document_id)
                )
                admitted = engine.cache_lru.get(admission.paths[sample.document_id])
                examples.append(
                    ContinuationExample(
                        document_id=sample.document_id,
                        prefill=prefill,
                        decode_cache=admitted,
                        continuation_ids=(
                            prefill.first_token.token_ids[0],
                            *sample.decode_target_ids[:REFINEMENT_DECODE_STEPS],
                        ),
                    )
                )
            documents.extend(
                evaluate_teacher_forced_cached_batched(
                    weight_bank.model,
                    examples,
                    backend,
                )
            )
    expected_tensors = int(engine.model_architecture["num_hidden_layers"]) * 2
    expected_quantized = (
        0 if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE else expected_tensors
    )
    if (
        engine._native_append_validation_calls != 1
        or engine._native_append_tensor_checks != expected_tensors
        or engine._native_append_quantized_tensor_checks != expected_quantized
    ):
        raise AssertionError("refinement native-append oracle coverage differs")
    nll_sum = sum(document.nll_sum for document in documents)
    token_count = sum(document.token_count for document in documents)
    expected_tokens = REFINEMENT_DOCUMENTS * REFINEMENT_DECODE_STEPS
    if token_count != expected_tokens or len(documents) != REFINEMENT_DOCUMENTS:
        raise AssertionError("refinement sample coverage differs")
    mean_nll = nll_sum / token_count
    peak = None
    if cuda_device is not None:
        peak = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
        }
    return {
        "sample_bundle_hash": evaluator.bundle.canonical_hash,
        "document_count": len(documents),
        "decode_steps": REFINEMENT_DECODE_STEPS,
        "q_len": 1,
        "nll_sum": nll_sum,
        "token_count": token_count,
        "mean_token_nll": mean_nll,
        "runtime_rebinding": binding.to_dict(),
        "native_append_validation": {
            "calls": engine._native_append_validation_calls,
            "tensor_checks": engine._native_append_tensor_checks,
            "quantized_tensor_checks": engine._native_append_quantized_tensor_checks,
            "expected_tensor_checks": expected_tensors,
            "expected_quantized_tensor_checks": expected_quantized,
            "q_len": 1,
        },
        "gpu_memory": peak,
        "documents": [
            {
                "document_id": document.document_id,
                "nll_sum": document.nll_sum,
                "token_count": document.token_count,
                "mean_token_nll": document.mean_nll,
                "initial_cache_length": document.initial_cache_length,
                "final_cache_length": document.final_cache_length,
            }
            for document in documents
        ],
    }


def _failed_row(
    *,
    plan: Mapping[str, Any],
    entry: SweepManifestEntry,
    role: str,
    shard_index: int,
    error: BaseException,
    runtime_seconds: float,
    validation: Mapping[str, Any] | None = None,
    mixed_abi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    is_oom = type(error).__name__ in {"OutOfMemoryError", "MemoryError"} or "out of memory" in str(
        error
    ).lower()
    return {
        "schema_version": ROW_SCHEMA,
        "plan_hash": plan["content_hash"],
        "profile_id": entry.profile_id,
        "profile": entry.profile.to_dict(),
        "role": role,
        "shard_index": shard_index,
        "state": "failed",
        "validation": dict(validation) if validation is not None else None,
        "refinement": None,
        "weight_bank": None,
        "mixed_weight_activation_abi": (
            dict(mixed_abi) if mixed_abi is not None else None
        ),
        "error": {
            "error_class": type(error).__name__,
            "error_message": str(error),
            "oom": is_oom,
            "traceback": traceback.format_exc(),
        },
        "runtime_seconds": float(runtime_seconds),
        "classification": {
            "measured_numerical": validation is not None,
            "publication_rankable": False,
            "selection_eligible": False,
            "compiler_valid": False,
            "emulator_valid": False,
            "rtl_valid": False,
            "hardware_bit_parity_verified": False,
        },
    }


def run_worker(
    *,
    plan_path: str | Path,
    shard_index: int,
    shard_count: int,
    device_label: str,
) -> dict[str, Any]:
    """Run or resume one deterministic whole-weight-bank shard."""

    plan = _load_plan(plan_path)
    sharding = plan["sharding"]
    if shard_count != int(sharding["shard_count"]):
        raise ValueError("worker shard count differs from the sealed plan")
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid MLEN worker shard index")
    partition = sharding["partitions"][shard_index]
    if int(partition["shard_index"]) != shard_index:
        raise ValueError("MLEN partition order changed")
    paths = plan["paths"]
    config_path = Path(paths["config"])
    config = _sweep_launcher_load_config(config_path)
    if _file_hash(config_path) != plan["bindings"]["config_sha256"]:
        raise ValueError("execution config changed after MLEN planning")
    numerical_workspace = Path(paths["numerical_workspace"])
    master = load_manifest(numerical_workspace / "manifest.json")
    run_plan = SweepRunPlan.from_dict(
        load_immutable_json(numerical_workspace / "run_plan.json")
    )
    prompts = PromptManifest.from_dict(
        load_immutable_json(numerical_workspace / "prompt_manifest.json")
    )
    if (
        master.canonical_hash != plan["bindings"]["master_manifest_hash"]
        or run_plan.canonical_hash != plan["bindings"]["run_plan_hash"]
        or prompts.canonical_hash != plan["bindings"]["prompt_manifest_hash"]
    ):
        raise ValueError("sealed numerical workspace changed after MLEN planning")
    profiles = _plan_profiles(plan)
    shard_profiles = [profiles[value] for value in partition["profile_ids"]]
    stage_manifest = _profile_manifest(master, shard_profiles)
    output_dir = (
        Path(paths["output_root"])
        / "shards"
        / f"part-{shard_index:04d}-of-{shard_count:04d}"
    )
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    invocation = {
        "schema_version": INVOCATION_SCHEMA,
        "plan_hash": plan["content_hash"],
        "stage_manifest_hash": stage_manifest.canonical_hash,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device_label": device_label,
        "weight_formats": partition["weight_formats"],
        "profile_ids": partition["profile_ids"],
        "failed_rows_retained": True,
        "publication_rankable": False,
    }
    write_immutable_json(output_dir / "invocation.json", invocation)
    context = _validation_context(
        plan=plan,
        config=config,
        master=master,
        run_plan=run_plan,
        prompts=prompts,
        stage_manifest=stage_manifest,
        shard_index=shard_index,
        shard_count=shard_count,
        device_label=device_label,
        output_dir=output_dir,
    )
    validation_evaluator = DecodeEvaluator(context)
    refinement_evaluator = RefinementEvaluator(
        config=config,
        sample_bundle_path=paths["refinement_sample_bundle"],
        prefill_root=paths["refinement_prefill_root"],
        admission_root=paths["refinement_admission_root"],
        workspace_root=paths["refinement_workspace"],
        device_label=device_label,
        decode_microbatch_size=int(
            config.get("executor", {})
            .get("decode_microbatch_size", {})
            .get("hardware_validation", 8)
        ),
        max_cpu_cache_gib=float(config.get("refinement", {}).get("max_cpu_cache_gib", 24)),
    )
    mase_src = validation_evaluator.mase_source_root
    mixed_abi = audit_mixed_weight_activation_abi(mase_src)
    write_immutable_json(output_dir / "mixed_weight_activation_abi.json", mixed_abi)
    role_by_id = {
        str(value["profile_id"]): str(value["role"])
        for value in plan["evaluation_profiles"]
    }
    entry_by_id = {entry.profile_id: entry for entry in stage_manifest.entries}
    terminal = 0
    succeeded = 0
    failed = 0
    for weight_format in partition["weight_formats"]:
        entries = tuple(
            entry
            for entry in stage_manifest.entries
            if entry.profile.weight_format == weight_format
            and not (rows_dir / f"{entry.profile_id}.json").is_file()
        )
        if not entries:
            continue
        group_started = time.perf_counter()
        try:
            bank_context = validation_evaluator.open_weight_bank(weight_format, entries)
            with bank_context as weight_bank:
                bank_receipt = {
                    "weight_format": weight_format,
                    "weight_method": weight_bank.weight_method,
                    "build_seconds": weight_bank.build_seconds,
                    "identity_fingerprint": weight_bank.identity_guard.fingerprint,
                    "structure_fingerprint": weight_bank.identity_guard.structure_fingerprint,
                    "parameter_count": len(weight_bank.identity_guard.parameters),
                    "profile_ids": [entry.profile_id for entry in entries],
                    "same_in_memory_bank_across_mlen_values": True,
                }
                for entry in entries:
                    destination = rows_dir / f"{entry.profile_id}.json"
                    started = time.perf_counter()
                    validation_result = None
                    try:
                        if _mixed_weight_activation(entry.profile) and not mixed_abi.get(
                            "numerical_execution_supported"
                        ):
                            raise RuntimeError(
                                "mixed_weight_activation_mase_abi_unsupported"
                            )
                        with validation_evaluator.open_kv_admission_cache(
                            entry.profile.kv_format
                        ) as validation_admission:
                            validation_outcome = validation_evaluator.evaluate(
                                entry,
                                weight_bank=weight_bank,
                                kv_admission_cache=validation_admission,
                            )
                        validation_result = dict(validation_outcome.metrics)
                        if (
                            validation_result.get("runtime_rebinding", {}).get(
                                "weight_requantizations"
                            )
                            != 0
                        ):
                            raise RuntimeError("validation rebind requantized the RTN bank")
                        with _refinement_admission(
                            refinement_evaluator, entry.profile
                        ) as refinement_admission:
                            refinement_result = _evaluate_refinement(
                                refinement_evaluator,
                                entry,
                                weight_bank,
                                refinement_admission,
                            )
                        row = {
                            "schema_version": ROW_SCHEMA,
                            "plan_hash": plan["content_hash"],
                            "profile_id": entry.profile_id,
                            "profile": entry.profile.to_dict(),
                            "role": role_by_id[entry.profile_id],
                            "shard_index": shard_index,
                            "state": "succeeded",
                            "validation": validation_result,
                            "refinement": refinement_result,
                            "weight_bank": bank_receipt,
                            "mixed_weight_activation_abi": (
                                mixed_abi if _mixed_weight_activation(entry.profile) else None
                            ),
                            "error": None,
                            "runtime_seconds": time.perf_counter() - started,
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
                    except Exception as error:
                        row = _failed_row(
                            plan=plan,
                            entry=entry,
                            role=role_by_id[entry.profile_id],
                            shard_index=shard_index,
                            error=error,
                            runtime_seconds=time.perf_counter() - started,
                            validation=validation_result,
                            mixed_abi=(
                                mixed_abi if _mixed_weight_activation(entry.profile) else None
                            ),
                        )
                        gc.collect()
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                    write_immutable_json(destination, row)
        except Exception as bank_error:
            for entry in entries:
                destination = rows_dir / f"{entry.profile_id}.json"
                if destination.exists():
                    continue
                write_immutable_json(
                    destination,
                    _failed_row(
                        plan=plan,
                        entry=entry,
                        role=role_by_id[entry.profile_id],
                        shard_index=shard_index,
                        error=bank_error,
                        runtime_seconds=time.perf_counter() - group_started,
                        mixed_abi=(
                            mixed_abi if _mixed_weight_activation(entry.profile) else None
                        ),
                    ),
                )
    for profile_id in partition["profile_ids"]:
        row = load_immutable_json(rows_dir / f"{profile_id}.json")
        terminal += 1
        if row.get("state") == "succeeded":
            succeeded += 1
        else:
            failed += 1
    summary = {
        "schema_version": "decode-mlen-revalidation-shard-summary/v1",
        "plan_hash": plan["content_hash"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "terminal_profiles": terminal,
        "succeeded_profiles": succeeded,
        "failed_profiles": failed,
        "complete": terminal == len(partition["profile_ids"]),
    }
    write_immutable_json(output_dir / "summary.json", summary)
    return load_immutable_json(output_dir / "summary.json")


def _metric(row: Mapping[str, Any], suite: str) -> float:
    value = row.get(suite)
    if not isinstance(value, Mapping):
        raise ValueError(f"successful row lacks {suite} metrics")
    metric = value.get("mean_token_nll")
    if metric is None:
        metric = value.get("mean_nll")
    result = float(metric)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{suite} mean token NLL is invalid")
    return result


def finalize(*, plan_path: str | Path) -> dict[str, Any]:
    """Seal terminal coverage and conditionally emit corrected selector input."""

    plan_path = Path(plan_path).resolve()
    plan = _load_plan(plan_path)
    root = Path(plan["paths"]["output_root"])
    rows: dict[str, dict[str, Any]] = {}
    failures = []
    row_receipts = []
    for partition in plan["sharding"]["partitions"]:
        shard_index = int(partition["shard_index"])
        shard_root = (
            root
            / "shards"
            / f"part-{shard_index:04d}-of-{plan['sharding']['shard_count']:04d}"
        )
        invocation = load_immutable_json(shard_root / "invocation.json")
        summary = load_immutable_json(shard_root / "summary.json")
        if (
            invocation.get("plan_hash") != plan["content_hash"]
            or summary.get("plan_hash") != plan["content_hash"]
            or summary.get("complete") is not True
        ):
            raise ValueError("MLEN shard receipt is incomplete or belongs to another plan")
        for profile_id in partition["profile_ids"]:
            path = shard_root / "rows" / f"{profile_id}.json"
            row = load_immutable_json(path)
            if (
                row.get("schema_version") != ROW_SCHEMA
                or row.get("plan_hash") != plan["content_hash"]
                or row.get("profile_id") != profile_id
                or profile_id in rows
            ):
                raise ValueError("MLEN terminal row identity is invalid")
            row_profile = DecodePrecisionProfile.from_dict(row["profile"])
            planned = next(
                value
                for value in plan["evaluation_profiles"]
                if value["profile_id"] == profile_id
            )
            if (
                row_profile.profile_id != profile_id
                or row["profile"] != planned["profile"]
                or planned["numerical_oracle_sha256"]
                != _validated_oracle_hash(row_profile)
                or row.get("classification", {}).get(
                    "hardware_bit_parity_verified"
                )
                is not False
            ):
                raise ValueError("MLEN terminal row numerical oracle differs")
            rows[profile_id] = row
            row_receipts.append(
                {
                    "profile_id": profile_id,
                    "path": str(path),
                    "content_hash": row["content_hash"],
                    "state": row["state"],
                }
            )
            if row["state"] != "succeeded":
                failures.append(
                    {
                        "profile_id": profile_id,
                        "role": row.get("role"),
                        "error": row.get("error"),
                    }
                )
    expected = {value["profile_id"] for value in plan["evaluation_profiles"]}
    if set(rows) != expected:
        raise ValueError("MLEN terminal coverage is not exact")
    all_succeeded = not failures
    selector_path = root / "corrected_projected_selector_input.json"
    selector_hash = None
    if all_succeeded:
        bf16_ids = [
            value["profile_id"]
            for value in plan["evaluation_profiles"]
            if value["role"] == "same_split_bf16_reference"
        ]
        if len(bf16_ids) != 1:
            raise AssertionError("MLEN plan BF16 control coverage changed")
        bf16_row = rows[bf16_ids[0]]
        reference = {
            suite: {
                "profile_id": bf16_ids[0],
                "mean_token_nll": _metric(bf16_row, suite),
                "row_hash": bf16_row["content_hash"],
            }
            for suite in ("validation", "refinement")
        }
        corrected = []
        for mapping in plan["candidate_mapping"]:
            source = rows[mapping["source_profile_id"]]
            variant = rows[mapping["revalidated_profile_id"]]
            profile = DecodePrecisionProfile.from_dict(variant["profile"])
            if (
                profile.matrix_mlen != mapping["candidate_matrix_mlen"]
                or mapping["hardware"].get("MLEN") != profile.matrix_mlen
            ):
                raise ValueError("successful MLEN row does not match its hardware candidate")
            suites = {}
            for suite in ("validation", "refinement"):
                candidate_nll = _metric(variant, suite)
                source_nll = _metric(source, suite)
                reference_nll = reference[suite]["mean_token_nll"]
                suites[suite] = {
                    "mean_token_nll": candidate_nll,
                    "same_format_mlen1024_mean_token_nll": source_nll,
                    "bf16_mean_token_nll": reference_nll,
                    "delta_nll_due_to_mlen_geometry": candidate_nll - source_nll,
                    "relative_perplexity_vs_same_split_bf16": math.exp(
                        candidate_nll - reference_nll
                    ),
                    "candidate_row_hash": variant["content_hash"],
                    "source_control_row_hash": source["content_hash"],
                    "bf16_control_row_hash": reference[suite]["row_hash"],
                }
            corrected.append(
                {
                    **dict(mapping),
                    "profile_id": profile.profile_id,
                    "profile": profile.to_dict(),
                    "matrix_partition_matched": True,
                    "suites": suites,
                    "publication_rankable": False,
                    "selection_eligible": False,
                    "hardware_bit_parity_verified": False,
                    "hardware_reprice_required": True,
                    "hardware_reprice_reason": (
                        "source projected metrics were computed with another "
                        "profile identity; rerun the analytic evaluator with "
                        "this exact MLEN-bound profile before selection"
                    ),
                }
            )
        selector_body = {
            "schema_version": SELECTOR_INPUT_SCHEMA,
            "plan_hash": plan["content_hash"],
            "target": dict(plan["target"]),
            "source_bindings": dict(plan["bindings"]),
            "same_split_bf16_reference": reference,
            "rows": corrected,
            "complete": True,
            "all_promoted_candidates_have_matching_numerical_mlen": True,
            "nll_values_reused_or_copied": False,
            "publication_rankable": False,
            "hardware_bit_parity_verified": False,
            "selection_eligible_before_exact_hardware_reprice": False,
        }
        write_immutable_json(selector_path, selector_body)
        selector_hash = load_immutable_json(selector_path)["content_hash"]
    elif selector_path.exists():
        raise RuntimeError(
            "failed MLEN coverage exists beside a selector input; use a fresh output root"
        )
    completion_body = {
        "schema_version": COMPLETION_SCHEMA,
        "plan_path": str(plan_path),
        "plan_hash": plan["content_hash"],
        "target": dict(plan["target"]),
        "terminal_profile_count": len(rows),
        "succeeded_profile_count": len(rows) - len(failures),
        "failed_profile_count": len(failures),
        "failed_rows": failures,
        "row_receipts": sorted(row_receipts, key=lambda value: value["profile_id"]),
        "complete": True,
        "successful": all_succeeded,
        "failed_rows_retained": True,
        "selector_input_path": str(selector_path) if all_succeeded else None,
        "selector_input_hash": selector_hash,
        "selector_input_emitted": all_succeeded,
        "publication_rankable": False,
        "hardware_bit_parity_verified": False,
        "classification": dict(plan["classification"]),
    }
    completion_path = root / "mlen_revalidation_completion.json"
    write_immutable_json(completion_path, completion_body)
    return load_immutable_json(completion_path)


def launch_shards(
    *,
    plan_path: str | Path,
    gpus: Sequence[str],
    python: str | Path = sys.executable,
    device_label: str = "B200",
) -> tuple[dict[str, Any], ...]:
    plan_path = Path(plan_path).resolve()
    plan = _load_plan(plan_path)
    shard_count = int(plan["sharding"]["shard_count"])
    devices = tuple(str(value) for value in gpus)
    if len(devices) < shard_count or len(set(devices[:shard_count])) != shard_count:
        raise ValueError("MLEN launch needs one distinct GPU per sealed shard")
    executable = Path(python).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    def invoke(index: int) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = devices[index]
        software_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            software_root
            if not existing_pythonpath
            else f"{software_root}{os.pathsep}{existing_pythonpath}"
        )
        command = (
            str(executable),
            "-m",
            "decode_dse.software.mlen_revalidation",
            "worker",
            "--plan",
            str(plan_path),
            "--shard-index",
            str(index),
            "--shard-count",
            str(shard_count),
            "--device-label",
            device_label,
        )
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"MLEN shard {index} exited {completed.returncode}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=shard_count) as executor:
        futures = {executor.submit(invoke, index): index for index in range(shard_count)}
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(f"shard {futures[future]}: {error}")
    if failures:
        raise RuntimeError("; ".join(sorted(failures)))
    return tuple(
        load_immutable_json(
            Path(plan["paths"]["output_root"])
            / "shards"
            / f"part-{index:04d}-of-{shard_count:04d}"
            / "summary.json"
        )
        for index in range(shard_count)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    worker.add_argument("--device-label", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--python", type=Path, default=Path(sys.executable))
    launch.add_argument("--device-label", default="B200")
    finish = commands.add_parser("finalize")
    finish.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        result = run_worker(
            plan_path=args.plan,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device_label=args.device_label,
        )
    elif args.command == "launch":
        result = launch_shards(
            plan_path=args.plan,
            gpus=tuple(value.strip() for value in args.gpus.split(",") if value.strip()),
            python=args.python,
            device_label=args.device_label,
        )
    else:
        result = finalize(plan_path=args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_MLENS",
    "COMPLETION_SCHEMA",
    "MIXED_ABI_SCHEMA",
    "PLAN_SCHEMA",
    "ROW_SCHEMA",
    "SELECTOR_INPUT_SCHEMA",
    "audit_mixed_weight_activation_abi",
    "build_plan",
    "derive_mlen_variant",
    "finalize",
    "launch_shards",
    "run_worker",
]
