"""Fail-closed Qwen3-MoE router-precision ablation campaign.

This lane derives paired router variants from already materialized body banks.
It neither extends the canonical 3,585-profile census nor feeds the hardware
selector.  Every variant invocation must replay an exact BF16-router control
on the same held-out inputs before end-to-end numerical deltas are accepted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

from decode_dse.profiles import (
    DecodePrecisionProfile,
    declared_search_space,
    enumerate_decode_profiles,
    format_descriptor,
)
from decode_dse.manifest import load_manifest
from decode_dse.software.sweep_plan import (
    load_immutable_json,
    profile_to_decode_quant_spec,
    write_immutable_json,
)


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
EXPECTED_PROFILE_COUNT = 3585
EXPECTED_LAYERS = 48
EXPECTED_HIDDEN = 2048
EXPECTED_EXPERTS = 128
EXPECTED_TOP_K = 8
EXPECTED_SHARDS = 4

HELDOUT_SCHEMA = "decode-router-precision-heldout/v3"
HELDOUT_SOURCE_SCHEMA = "decode-router-precision-heldout-source/v1"
ANCESTRY_SCHEMA = "decode-router-bf16-ancestry/v1"
PLAN_SCHEMA = "decode-router-precision-ablation-plan/v1"
REQUEST_SCHEMA = "decode-router-precision-driver-request/v1"
RESULT_SCHEMA = "decode-router-precision-result/v1"
AGREEMENT_SCHEMA = "decode-router-precision-agreement/v2"
COST_SCHEMA = "decode-router-precision-prospective-cost/v1"
COMPLETION_SCHEMA = "decode-router-precision-completion/v2"
DRIVER_BATCH_SCHEMA = "decode-router-precision-driver-batch/v1"
DRIVER_RECEIPT_SCHEMA = "decode-router-precision-driver-receipt/v3"
PREFILL_CACHE_SCHEMA = "decode-router-bf16-prefill-cache/v2"
PREFILL_RECORD_SCHEMA = "decode-router-bf16-prefill-record/v2"
DERIVATION_SCHEMA = "decode-router-bf16-binding-derivation/v1"
SOURCE_ROW_SCHEMA = "decode-router-bf16-derived-source-row/v1"

ROUTER_FORMATS = ("MXINT8", "E4M3", "E5M2")
TERMINAL_STATUSES = ("success", "failed", "oom")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CONTINUATION_DECODE_CONTRACT = {
    "continuation_includes_prefill_owned_first_token": True,
    "serving_first_token_owner": "prefill",
    "evaluation_decode_input_source": "teacher_forced_continuation_token_0",
    "evaluation_input_generated_by_bf16_model": False,
    "prefill_owned_token_index": 0,
    "decode_input_token_index": 0,
    "scored_suffix_start_index": 1,
    "minimum_continuation_tokens": 2,
}

_TARGET_ARCHITECTURE = {
    "model_type": "qwen3_moe",
    "hidden_size": EXPECTED_HIDDEN,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_hidden_layers": EXPECTED_LAYERS,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "use_qk_norm": True,
    "num_experts": EXPECTED_EXPERTS,
    "num_experts_per_tok": EXPECTED_TOP_K,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}

_BF16_ROUTER_CONTRACT = {
    "implementation": "Qwen3MoeTopKRouterBF16",
    "weight_format": "BF16",
    "activation_format": "BF16",
    "logits_format": "BF16",
    "probability_dtype": "FP32",
    "selection": "torch.topk_sorted_exact",
    "top_k": EXPECTED_TOP_K,
    "norm_topk_prob": True,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise ValueError(f"artifact tree is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        name = item.relative_to(path).as_posix().encode("utf-8")
        payload = item.read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "sha256": sha256_file(resolved),
        }
    if resolved.is_dir():
        return {
            "path": str(resolved),
            "kind": "directory",
            "sha256": sha256_tree(resolved),
        }
    raise FileNotFoundError(f"bound artifact does not exist: {resolved}")


def _verify_identity(value: Mapping[str, Any], *, label: str) -> Path:
    if set(value) != {"path", "kind", "sha256"}:
        raise ValueError(f"{label} identity fields are incomplete")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    observed = path_identity(path)
    if observed != dict(value):
        raise ValueError(f"{label} artifact is missing or changed")
    return path


def _load_target_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("study config root must be an object")
    if config.get("model_name") != MODEL_NAME:
        raise ValueError("router lane is sealed to the Qwen3-30B-A3B target")
    for key in ("model_revision", "tokenizer_revision"):
        if config.get(key) != MODEL_REVISION:
            raise ValueError(f"{key} differs from the sealed target revision")
    architecture = config.get("model_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("model_architecture is required")
    for key, expected in _TARGET_ARCHITECTURE.items():
        if architecture.get(key) != expected:
            raise ValueError(f"target architecture mismatch for {key}")
    profiles = enumerate_decode_profiles(declared_search_space(config["search"]))
    if len(profiles) != EXPECTED_PROFILE_COUNT:
        raise ValueError("canonical body profile census must remain exactly 3,585")
    return config, path_identity(path)


def _publication_template_binding(
    config: Mapping[str, Any], config_path: Path
) -> tuple[dict[str, Any], str]:
    publication = config.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("config.publication is required")
    expected_hash = str(publication.get("chat_template_sha256", ""))
    if not _SHA256.fullmatch(expected_hash):
        raise ValueError("publication chat-template SHA-256 is missing")
    configured = Path(str(publication.get("chat_template_asset", "")))
    if not configured.is_absolute():
        configured = config_path.resolve().parents[2] / configured
    asset = json.loads(configured.read_text(encoding="utf-8"))
    if not isinstance(asset, Mapping):
        raise TypeError("publication chat-template asset must be an object")
    template = str(asset.get("chat_template", ""))
    observed_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if (
        not template
        or observed_hash != expected_hash
        or asset.get("chat_template_sha256") != expected_hash
    ):
        raise ValueError("publication chat-template asset differs from config")
    return path_identity(configured), template


def _heldout_source_records(source_path: Path) -> tuple[dict[str, Any], ...]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        not isinstance(source, Mapping)
        or source.get("schema_version") != HELDOUT_SOURCE_SCHEMA
    ):
        raise ValueError("unsupported held-out router source schema")
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("held-out router source contains no records")
    validated = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("held-out router source record must be an object")
        prompt_id = str(record.get("prompt_id", ""))
        messages = record.get("messages")
        continuation = record.get("continuation")
        if not prompt_id or prompt_id in seen:
            raise ValueError("held-out router prompt IDs are missing or duplicated")
        if not isinstance(messages, list) or not messages:
            raise ValueError("held-out router record requires non-empty messages")
        for message in messages:
            if (
                not isinstance(message, Mapping)
                or set(message) != {"role", "content"}
                or str(message.get("role", "")) not in {
                    "system",
                    "user",
                    "assistant",
                }
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("held-out message has an unsupported chat shape")
        if not isinstance(continuation, str) or not continuation:
            raise ValueError("held-out router continuation must be non-empty text")
        if record.get("used_for_router_calibration") is not False:
            raise ValueError("held-out records may not be used for calibration")
        seen.add(prompt_id)
        validated.append(
            {
                "record_index": index,
                "prompt_id": prompt_id,
                "messages": [dict(message) for message in messages],
                "continuation": continuation,
                "used_for_router_calibration": False,
            }
        )
    return tuple(validated)


def materialize_heldout_manifest(
    *,
    config_path: Path,
    source_path: Path,
    decode_target_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    """Seal chat records and their continuations for the executable driver."""

    if (
        isinstance(decode_target_tokens, bool)
        or not isinstance(decode_target_tokens, int)
        or decode_target_tokens < 2
    ):
        raise ValueError("decode_target_tokens must include an input token and a scored suffix")
    config, _ = _load_target_config(config_path.resolve())
    template_identity, _ = _publication_template_binding(
        config, config_path.resolve()
    )
    records = _heldout_source_records(source_path.resolve())
    identities = [
        {
            "record_index": record["record_index"],
            "prompt_id": record["prompt_id"],
            "prompt_sha256": hashlib.sha256(
                _canonical_bytes(record["messages"])
            ).hexdigest(),
            "continuation_sha256": hashlib.sha256(
                record["continuation"].encode("utf-8")
            ).hexdigest(),
            "used_for_router_calibration": False,
        }
        for record in records
    ]
    body = {
        "schema_version": HELDOUT_SCHEMA,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "split_role": "heldout_router_precision_evaluation",
        "calibration_overlap_allowed": False,
        "thinking_mode": True,
        "chat_template": template_identity,
        "chat_template_sha256": config["publication"][
            "chat_template_sha256"
        ],
        "chat_template_args": {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        },
        "decode_query_length": 1,
        "decode_target_tokens_per_record": decode_target_tokens,
        "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
        "dataset_source": path_identity(source_path.resolve()),
        "records": identities,
        "record_identity_hash": canonical_hash(identities),
    }
    write_immutable_json(output_path.resolve(), body)
    sealed = load_immutable_json(output_path.resolve())
    validate_heldout_manifest(sealed)
    return sealed


def validate_heldout_manifest(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != HELDOUT_SCHEMA
        or value.get("model_name") != MODEL_NAME
        or value.get("model_revision") != MODEL_REVISION
        or value.get("tokenizer_revision") != MODEL_REVISION
        or value.get("split_role") != "heldout_router_precision_evaluation"
        or value.get("calibration_overlap_allowed") is not False
        or value.get("thinking_mode") is not True
        or value.get("decode_query_length") != 1
        or value.get("continuation_decode_contract")
        != _CONTINUATION_DECODE_CONTRACT
    ):
        raise ValueError("held-out router manifest has the wrong target or split")
    target_tokens = value.get("decode_target_tokens_per_record")
    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, int)
        or target_tokens < 2
    ):
        raise ValueError("held-out router target-token count is invalid")
    template = value.get("chat_template")
    if not isinstance(template, Mapping):
        raise ValueError("held-out chat-template identity is required")
    template_path = _verify_identity(template, label="held-out chat template")
    template_asset = json.loads(template_path.read_text(encoding="utf-8"))
    template_hash = str(value.get("chat_template_sha256", ""))
    if (
        not isinstance(template_asset, Mapping)
        or template_asset.get("chat_template_sha256") != template_hash
        or hashlib.sha256(
            str(template_asset.get("chat_template", "")).encode("utf-8")
        ).hexdigest()
        != template_hash
        or value.get("chat_template_args")
        != {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
    ):
        raise ValueError("held-out chat-template binding differs")
    source = value.get("dataset_source")
    if not isinstance(source, Mapping):
        raise ValueError("held-out dataset source identity is required")
    source_path = _verify_identity(source, label="held-out dataset source")
    source_records = _heldout_source_records(source_path)
    records = value.get("records")
    if (
        not isinstance(records, list)
        or not records
        or len(records) != len(source_records)
    ):
        raise ValueError("held-out router manifest contains no records")
    seen: set[str] = set()
    identities = []
    for index, (record, source_record) in enumerate(zip(records, source_records)):
        if not isinstance(record, Mapping):
            raise TypeError("held-out record must be an object")
        prompt_id = str(record.get("prompt_id", ""))
        prompt_sha = str(record.get("prompt_sha256", ""))
        continuation_sha = str(record.get("continuation_sha256", ""))
        if (
            set(record)
            != {
                "record_index",
                "prompt_id",
                "prompt_sha256",
                "continuation_sha256",
                "used_for_router_calibration",
            }
            or record.get("record_index") != index
            or source_record["record_index"] != index
            or prompt_id != source_record["prompt_id"]
            or not prompt_id
            or prompt_id in seen
            or not _SHA256.fullmatch(prompt_sha)
            or not _SHA256.fullmatch(continuation_sha)
            or prompt_sha
            != hashlib.sha256(
                _canonical_bytes(source_record["messages"])
            ).hexdigest()
            or continuation_sha
            != hashlib.sha256(
                source_record["continuation"].encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError("held-out prompt identities are missing or duplicated")
        if record.get("used_for_router_calibration") is not False:
            raise ValueError("held-out prompts may not be used for router calibration")
        seen.add(prompt_id)
        identities.append(dict(record))
    if value.get("record_identity_hash") != canonical_hash(identities):
        raise ValueError("held-out record identity hash differs")


def _assert_profile_uses_bf16_router(profile: DecodePrecisionProfile) -> None:
    """Re-evaluate the current pass selectors instead of trusting row metadata."""

    from decode_dse.software import precision_bindings
    from decode_dse.software.precision_bindings import build_decode_pass_args

    spec = profile_to_decode_quant_spec(profile)
    if spec is None:
        raise ValueError("router MX ablation requires a quantized body profile")
    pass_args = build_decode_pass_args(MODEL_NAME, "cuda:0", spec)
    gate_name = "model.layers.0.mlp.gate"
    direct_gate = [
        pattern
        for pattern in pass_args
        if isinstance(pattern, str)
        and pattern != "by"
        and re.fullmatch(pattern, gate_name)
    ]
    mlp_matches = [
        pass_args[pattern]
        for pattern in pass_args
        if isinstance(pattern, str)
        and pattern != "by"
        and re.fullmatch(pattern, "model.layers.0.mlp")
    ]
    if direct_gate or len(mlp_matches) != 1:
        raise ValueError("current precision binding does not isolate the router")
    config = mlp_matches[0].get("config")
    if not isinstance(config, Mapping) or config.get("name") != "minifloat":
        raise ValueError("current sparse-block binding does not select minifloat")
    # Keep this import live: module registry drift is a different failure from
    # selector drift and must block ancestry derivation.
    from chop.nn.quantized.modules import quantized_module_map
    from chop.nn.quantized.modules.qwen3_moe import (
        Qwen3MoeSparseMoeBlockMinifloat,
        Qwen3MoeTopKRouterBF16,
    )

    if (
        quantized_module_map.get("qwen3_moe_sparse_block_minifloat")
        is not Qwen3MoeSparseMoeBlockMinifloat
    ):
        raise ValueError("MASE sparse-block registry no longer selects the expected class")
    # The class constructor installs Qwen3MoeTopKRouterBF16; bind both source
    # files below, with the direct class identity retained for introspection.
    if Qwen3MoeTopKRouterBF16.__name__ != "Qwen3MoeTopKRouterBF16":
        raise ValueError("MASE BF16 router implementation identity differs")
    from transformers.models.qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    probe_config = Qwen3MoeConfig(
        vocab_size=32,
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
    )
    replacement = Qwen3MoeSparseMoeBlockMinifloat.from_self(
        Qwen3MoeSparseMoeBlock(probe_config),
        q_config={"decode": {"format": "BF16"}},
    )
    if not isinstance(replacement.gate, Qwen3MoeTopKRouterBF16):
        raise ValueError("MASE sparse-block replacement no longer installs a BF16 router")
    if not Path(inspect.getsourcefile(precision_bindings) or "").is_file():
        raise ValueError("precision-binding source cannot be resolved")


def derive_bf16_router_binding(
    profile: DecodePrecisionProfile,
) -> dict[str, Any]:
    """Derive the implicit BF16 router contract from current executable code."""

    _assert_profile_uses_bf16_router(profile)
    from decode_dse.software import precision_bindings
    import chop.nn.quantized.modules as mase_registry
    from chop.nn.quantized.modules.qwen3_moe import Qwen3MoeTopKRouterBF16

    precision_source = Path(inspect.getsourcefile(precision_bindings) or "")
    router_source = Path(inspect.getsourcefile(Qwen3MoeTopKRouterBF16) or "")
    registry_source = Path(inspect.getsourcefile(mase_registry) or "")
    if not all(path.is_file() for path in (precision_source, router_source, registry_source)):
        raise ValueError("BF16 router derivation sources cannot be resolved")
    return {
        "schema_version": DERIVATION_SCHEMA,
        "mode": "verified_current_precision_binding",
        "sparse_block_selector": "model.layers.<index>.mlp -> minifloat",
        "sparse_block_replacement": "Qwen3MoeSparseMoeBlockMinifloat",
        "router_replacement": "Qwen3MoeTopKRouterBF16",
        "direct_gate_selector_present": False,
        "precision_bindings_source": path_identity(precision_source),
        "mase_router_source": path_identity(router_source),
        "mase_module_registry_source": path_identity(registry_source),
    }


def router_variant_source_bindings() -> dict[str, Any]:
    """Bind every source file on the router-variant numerical path."""

    from chop.nn.quantized.functional.matrix import plena_matrix_product
    from chop.nn.quantized.modules.qwen3_moe import Qwen3MoeTopKRouterMX
    from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer

    objects = {
        "mase_router_mx_source": Qwen3MoeTopKRouterMX,
        "mase_matrix_oracle_source": plena_matrix_product,
        "mase_mxfp_quantizer_source": mxfp_quantizer,
        "mase_mxint_quantizer_source": mxint_quantizer,
    }
    bindings = {}
    for name, value in objects.items():
        path = Path(inspect.getsourcefile(value) or "")
        if not path.is_file():
            raise ValueError(f"router variant source cannot be resolved: {name}")
        bindings[name] = path_identity(path)
    return bindings


def _load_bound_journal_row(path: Path, record_hash: str) -> dict[str, Any]:
    if not path.is_file() or path.suffix != ".jsonl":
        raise ValueError("source numerical journal must be one JSONL file")
    matches = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"journal row is not an object: {path}:{line_number}")
            if row.get("record_hash") != record_hash:
                continue
            body = dict(row)
            body.pop("record_hash", None)
            if canonical_hash(body) != record_hash:
                raise ValueError(f"journal row hash differs: {path}:{line_number}")
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("source journal does not contain exactly one bound row")
    return matches[0]


def validate_ancestry(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if (
        value.get("schema_version") != ANCESTRY_SCHEMA
        or value.get("model_name") != MODEL_NAME
        or value.get("model_revision") != MODEL_REVISION
        or value.get("canonical_profile_count") != EXPECTED_PROFILE_COUNT
        or value.get("baseline_router") != _BF16_ROUTER_CONTRACT
    ):
        raise ValueError("BF16-router ancestry differs from the sealed contract")
    router_source = value.get("bf16_router_source")
    if not isinstance(router_source, Mapping):
        raise ValueError("BF16 router source identity is required")
    _verify_identity(router_source, label="BF16 router source")
    derivation = value.get("router_contract_derivation")
    if (
        not isinstance(derivation, Mapping)
        or derivation.get("schema_version") != DERIVATION_SCHEMA
        or derivation.get("mode") != "verified_current_precision_binding"
        or derivation.get("sparse_block_selector")
        != "model.layers.<index>.mlp -> minifloat"
        or derivation.get("sparse_block_replacement")
        != "Qwen3MoeSparseMoeBlockMinifloat"
        or derivation.get("router_replacement") != "Qwen3MoeTopKRouterBF16"
        or derivation.get("direct_gate_selector_present") is not False
    ):
        raise ValueError("BF16 router ancestry lacks a verified binding derivation")
    for field, label in (
        ("precision_bindings_source", "precision-binding source"),
        ("mase_router_source", "MASE router source"),
        ("mase_module_registry_source", "MASE module registry source"),
    ):
        identity = derivation.get(field)
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} identity is required")
        _verify_identity(identity, label=label)
    if derivation["mase_router_source"] != router_source:
        raise ValueError("BF16 router source and derivation source differ")
    records = value.get("body_baselines")
    if not isinstance(records, list) or not records:
        raise ValueError("BF16-router ancestry contains no body baselines")
    validated = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("body baseline must be an object")
        profile_value = record.get("body_profile")
        if not isinstance(profile_value, Mapping):
            raise ValueError("body baseline lacks its canonical precision profile")
        profile = DecodePrecisionProfile.from_dict(profile_value)
        if derive_bf16_router_binding(profile) != dict(derivation):
            raise ValueError("BF16 router derivation differs from current executable code")
        profile_id = str(record.get("body_profile_id", ""))
        if profile.profile_id != profile_id or profile_id in seen:
            raise ValueError("body baseline profile identity differs or is duplicated")
        if profile.key_format != profile.value_format:
            raise ValueError("router ablation may not relax identical K/V precision")
        if record.get("baseline_router") != _BF16_ROUTER_CONTRACT:
            raise ValueError("body baseline does not have the exact BF16 router")
        body_bank = record.get("body_bank")
        if not isinstance(body_bank, Mapping):
            raise ValueError("body bank binding is required")
        if body_bank.get("materialization") == "rebuild_from_sealed_manifest_profile/v1":
            required_bank = {
                "materialization",
                "manifest_hash",
                "quantizer_provenance_hash",
                "weight_format",
                "source_weight_bank_identity_fingerprint",
                "source_weight_bank_structure_fingerprint",
                "source_mase_tree_sha256",
            }
            if set(body_bank) != required_bank:
                raise ValueError("rebuildable body-bank contract fields differ")
            for field in (
                "manifest_hash",
                "quantizer_provenance_hash",
                "source_weight_bank_identity_fingerprint",
                "source_weight_bank_structure_fingerprint",
                "source_mase_tree_sha256",
            ):
                if not _SHA256.fullmatch(str(body_bank.get(field, ""))):
                    raise ValueError(f"rebuildable body bank has no valid {field}")
            if (
                body_bank.get("weight_format") != profile.weight_format
                or body_bank.get("quantizer_provenance_hash")
                != record.get("quantizer_provenance_hash")
            ):
                raise ValueError("rebuildable body bank differs from its profile")
            body_bank_binding_hash = canonical_hash(body_bank)
        else:
            _verify_identity(body_bank, label="body bank")
            body_bank_binding_hash = str(body_bank["sha256"])

        bound_paths: dict[str, Path] = {}
        for field, label in (("source_numerical_result", "source numerical result"),):
            identity = record.get(field)
            if not isinstance(identity, Mapping):
                raise ValueError(f"{label} identity is required")
            bound_paths[field] = _verify_identity(identity, label=label)
        for field in (
            "quantizer_provenance_hash",
            "source_result_row_hash",
        ):
            if not _SHA256.fullmatch(str(record.get(field, ""))):
                raise ValueError(f"body baseline has no valid {field}")
        source_journal = record.get("source_numerical_journal")
        if not isinstance(source_journal, Mapping):
            raise ValueError("source numerical journal identity is required")
        source_journal_path = _verify_identity(
            source_journal, label="source numerical journal"
        )
        if not _SHA256.fullmatch(str(record.get("source_journal_record_hash", ""))):
            raise ValueError("source numerical journal record hash is required")
        journal_row = _load_bound_journal_row(
            source_journal_path, str(record["source_journal_record_hash"])
        )
        if (
            journal_row.get("profile_id") != profile_id
            or journal_row.get("state") != "succeeded"
        ):
            raise ValueError("source journal row is not this successful profile")
        if record.get("source_result_status") != "success":
            raise ValueError("body baseline must descend from a successful numerical row")
        source_path = bound_paths["source_numerical_result"]
        if not source_path.is_file():
            raise ValueError("source numerical result must be one standalone JSON row")
        source_row = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source_row, dict):
            raise ValueError("source numerical result row must be an object")
        source_body = dict(source_row)
        embedded_hash = source_body.pop("content_hash", None)
        source_row_hash = canonical_hash(source_body)
        if embedded_hash is not None and embedded_hash != source_row_hash:
            raise ValueError("source numerical result content hash differs")
        if source_row_hash != record["source_result_row_hash"]:
            raise ValueError("source numerical result row hash differs")
        if (
            source_row.get("profile_id") != profile_id
            or source_row.get("status") != "success"
            or source_row.get("baseline_router") != _BF16_ROUTER_CONTRACT
            or source_row.get("quantizer_provenance_hash")
            != record["quantizer_provenance_hash"]
            or source_row.get("body_bank_binding_hash")
            != body_bank_binding_hash
            or source_row.get("baseline_router_derivation_hash")
            != canonical_hash(derivation)
            or source_row.get("source_journal") != source_journal
            or source_row.get("source_journal_record_hash")
            != record["source_journal_record_hash"]
        ):
            raise ValueError("source numerical result does not bind this exact body/BF16 ancestry")
        seen.add(profile_id)
        validated.append(dict(record))
    return tuple(validated)


def _selected_sweep_rows(
    results_root: Path,
    record_hashes: Sequence[str],
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    requested = tuple(str(value) for value in record_hashes)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not _SHA256.fullmatch(value) for value in requested)
    ):
        raise ValueError("selected result hashes must be unique SHA-256 digests")
    wanted = set(requested)
    found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in sorted(results_root.resolve().rglob("*.jsonl")):
        journal_identity = path_identity(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"result row is not an object: {path}:{line_number}")
                record_hash = str(row.get("record_hash", ""))
                if record_hash not in wanted:
                    continue
                body = dict(row)
                body.pop("record_hash", None)
                if canonical_hash(body) != record_hash:
                    raise ValueError(f"result row hash differs: {path}:{line_number}")
                if record_hash in found:
                    raise ValueError("selected result hash occurs in multiple journals")
                found[record_hash] = (row, journal_identity)
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f"selected numerical rows are missing: {missing}")
    return tuple(found[value] for value in requested)


def materialize_ancestry_from_current_rows(
    *,
    manifest_path: Path,
    results_root: Path,
    record_hashes: Sequence[str],
    output_path: Path,
) -> dict[str, Any]:
    """Derive explicit BF16 ancestry from current rows that omit router fields."""

    manifest = load_manifest(manifest_path.resolve())
    if (
        manifest.model_name != MODEL_NAME
        or manifest.model_revision != MODEL_REVISION
        or manifest.tokenizer_revision != MODEL_REVISION
        or manifest.counts.get("total") != EXPECTED_PROFILE_COUNT
        or dict(manifest.model_architecture) != _TARGET_ARCHITECTURE
    ):
        raise ValueError("source manifest differs from the sealed target census")
    entries = {entry.profile_id: entry for entry in manifest.entries}
    selected = _selected_sweep_rows(results_root, record_hashes)
    records = []
    derivation: dict[str, Any] | None = None
    receipt_root = output_path.resolve().parent / "router_ancestry_rows"
    for row, journal_identity in selected:
        profile_id = str(row.get("profile_id", ""))
        entry = entries.get(profile_id)
        if (
            entry is None
            or row.get("schema_version") != "decode-sweep-result"
            or row.get("manifest_hash") != manifest.canonical_hash
            or row.get("profile") != entry.profile.to_dict()
            or row.get("state") != "succeeded"
            or isinstance(row.get("attempt"), bool)
            or not isinstance(row.get("attempt"), int)
            or int(row["attempt"]) <= 0
            or not isinstance(row.get("result"), Mapping)
        ):
            raise ValueError("selected row is not a successful row from this manifest")
        current_derivation = derive_bf16_router_binding(entry.profile)
        if derivation is None:
            derivation = current_derivation
        elif derivation != current_derivation:
            raise ValueError("selected profiles derive different BF16 router bindings")
        metrics = row["result"]
        weight_bank = metrics.get("weight_bank")
        runtime = metrics.get("runtime_environment")
        if not isinstance(weight_bank, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError("selected row lacks weight-bank/runtime identity evidence")
        bank_contract = {
            "materialization": "rebuild_from_sealed_manifest_profile/v1",
            "manifest_hash": manifest.canonical_hash,
            "quantizer_provenance_hash": manifest.quantizer_provenance.canonical_hash,
            "weight_format": entry.profile.weight_format,
            "source_weight_bank_identity_fingerprint": str(
                weight_bank.get("identity_fingerprint", "")
            ),
            "source_weight_bank_structure_fingerprint": str(
                weight_bank.get("structure_fingerprint", "")
            ),
            "source_mase_tree_sha256": str(runtime.get("mase_tree_sha256", "")),
        }
        for field in (
            "source_weight_bank_identity_fingerprint",
            "source_weight_bank_structure_fingerprint",
            "source_mase_tree_sha256",
        ):
            if not _SHA256.fullmatch(bank_contract[field]):
                raise ValueError(f"selected row lacks a valid {field}")
        source_body = {
            "schema_version": SOURCE_ROW_SCHEMA,
            "profile_id": profile_id,
            "profile": entry.profile.to_dict(),
            "status": "success",
            "baseline_router": dict(_BF16_ROUTER_CONTRACT),
            "baseline_router_derivation_hash": canonical_hash(current_derivation),
            "quantizer_provenance_hash": manifest.quantizer_provenance.canonical_hash,
            "body_bank_binding_hash": canonical_hash(bank_contract),
            "source_journal": journal_identity,
            "source_journal_record_hash": row["record_hash"],
            "source_attempt": int(row["attempt"]),
        }
        source_path = receipt_root / f"{profile_id}-{row['record_hash'][:12]}.json"
        write_immutable_json(source_path, source_body)
        records.append(
            {
                "body_profile_id": profile_id,
                "body_profile": entry.profile.to_dict(),
                "baseline_router": dict(_BF16_ROUTER_CONTRACT),
                "body_bank": bank_contract,
                "source_numerical_result": path_identity(source_path),
                "source_numerical_journal": journal_identity,
                "source_journal_record_hash": row["record_hash"],
                "quantizer_provenance_hash": (
                    manifest.quantizer_provenance.canonical_hash
                ),
                "source_result_row_hash": canonical_hash(source_body),
                "source_result_status": "success",
            }
        )
    assert derivation is not None
    body = {
        "schema_version": ANCESTRY_SCHEMA,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "canonical_profile_count": EXPECTED_PROFILE_COUNT,
        "source_manifest": path_identity(manifest_path.resolve()),
        "source_manifest_hash": manifest.canonical_hash,
        "baseline_router": dict(_BF16_ROUTER_CONTRACT),
        "bf16_router_source": dict(derivation["mase_router_source"]),
        "router_contract_derivation": derivation,
        "body_baselines": sorted(records, key=lambda item: item["body_profile_id"]),
    }
    write_immutable_json(output_path.resolve(), body)
    ancestry = load_immutable_json(output_path.resolve())
    validate_ancestry(ancestry)
    return ancestry


def _variant(
    weight_format: str,
    activation_format: str,
    *,
    matrix_mlen: int,
) -> dict[str, Any]:
    for token in (weight_format, activation_format):
        if token not in ROUTER_FORMATS:
            raise ValueError(f"unsupported router ablation format {token!r}")
    if (
        isinstance(matrix_mlen, bool)
        or not isinstance(matrix_mlen, int)
        or matrix_mlen <= 0
        or matrix_mlen % 8
    ):
        raise ValueError("router matrix_mlen must be a positive multiple of 8")
    family_w = format_descriptor(weight_format).family
    family_a = format_descriptor(activation_format).family
    body = {
        "weight_format": weight_format,
        "activation_format": activation_format,
        "block_size": 8,
        "scale_format": "E8M0",
        "scale_bits": 8,
        "matrix_mlen": matrix_mlen,
        "matrix_arithmetic_chain": [
            "per_mlen_fp32_matmul",
            "bf16_partial_rounding",
            "truncate_partial_to_signed_fixed16_16",
            "signed_fixed16_16_wrap_across_partitions",
            "final_bf16_writeout",
        ],
        "router_logits_container": "BF16",
        "softmax_dtype": "FP32",
        "top_k": EXPECTED_TOP_K,
        "topk_selection": "torch.topk_sorted_exact",
        "topk_renormalization_dtype": "FP32",
        "operand_family_binding": (
            family_w if family_w == family_a else f"mixed:{family_w}x{family_a}"
        ),
        "mixed_family_hardware_supported": family_w == family_a,
    }
    return body | {"variant_id": f"router-{canonical_hash(body)[:20]}"}


def build_router_variant_pass_args(
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a gate-only MASE pass without rebinding body, K/V, or LM head."""

    from chop.nn.quantized.modules.qwen3_moe import router_phase_config

    sealed = _variant(
        str(variant.get("weight_format", "")),
        str(variant.get("activation_format", "")),
        matrix_mlen=int(variant.get("matrix_mlen", 0)),
    )
    if dict(variant) != sealed:
        raise ValueError("router variant differs from its sealed identity")
    return {
        "by": "regex_name",
        r"model\.layers\.\d+\.mlp\.gate$": {
            "config": {
                "name": "mx",
                **router_phase_config(
                    sealed["weight_format"],
                    sealed["activation_format"],
                    matrix_mlen=sealed["matrix_mlen"],
                ),
            }
        },
    }


def _classification() -> dict[str, bool]:
    return {
        "publication_rankable": False,
        "hardware_rankable": False,
        "selection_eligible": False,
        "may_claim_measured_router_agreement_after_success": True,
        "may_claim_measured_nll_delta_after_success": True,
        "may_claim_measured_task_delta_after_success": True,
        "may_claim_latency": False,
        "may_claim_throughput": False,
        "may_claim_power": False,
        "may_claim_energy": False,
        "may_claim_area": False,
    }


def build_prospective_cost_receipt(
    *,
    body_profile: DecodePrecisionProfile,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Count router resources without converting counts into HW claims."""

    weight = format_descriptor(str(variant["weight_format"]))
    activation = format_descriptor(str(variant["activation_format"]))
    mlen = int(variant["matrix_mlen"])
    padded_hidden = math.ceil(EXPECTED_HIDDEN / mlen) * mlen
    weight_elements = EXPECTED_LAYERS * EXPECTED_EXPERTS * padded_hidden
    weight_blocks = weight_elements // 8
    weight_data_bits = weight_elements * weight.element_bits
    weight_scale_bits = weight_blocks * 8
    activation_elements = EXPECTED_LAYERS * padded_hidden
    activation_blocks = activation_elements // 8
    activation_data_bits = activation_elements * activation.element_bits
    activation_scale_bits = activation_blocks * 8
    logical_macs = EXPECTED_LAYERS * EXPECTED_EXPERTS * EXPECTED_HIDDEN
    issued_mac_slots = EXPECTED_LAYERS * EXPECTED_EXPERTS * padded_hidden
    output_elements = EXPECTED_LAYERS * EXPECTED_EXPERTS
    blockers = [
        "bf16_router_has_no_calibrated_dc_event_signature_and_may_not_be_proxied_as_mx",
        "mx_router_compiler_lowering_receipt_missing",
        "mx_router_emulator_or_rtl_trace_missing",
        "router_specific_calibrated_power_signature_missing",
        "exact_blen_padding_and_cycle_schedule_missing",
        "system_replication_and_placement_multiplicity_unbound",
    ]
    if body_profile.matrix_mlen != mlen:
        blockers.append("router_mlen_differs_from_source_body_numerical_mlen")
    receipt = {
        "schema_version": COST_SCHEMA,
        "scope": "prospective_router_only_per_decode_token_per_model_instance",
        "body_precision_unchanged": {
            "profile_id": body_profile.profile_id,
            "weight_format": body_profile.weight_format,
            "activation_format": body_profile.activation_format,
            "key_format": body_profile.key_format,
            "value_format": body_profile.value_format,
            "vector_format": body_profile.vector_format,
            "local_head_contract": body_profile.local_head_contract,
        },
        "router_precision": dict(variant),
        "geometry": {
            "layers": EXPECTED_LAYERS,
            "hidden_size": EXPECTED_HIDDEN,
            "padded_hidden_size": padded_hidden,
            "experts": EXPECTED_EXPERTS,
            "top_k": EXPECTED_TOP_K,
            "matrix_mlen": mlen,
            "source_body_numerical_mlen": body_profile.matrix_mlen,
            "router_body_mlen_aligned": body_profile.matrix_mlen == mlen,
        },
        "offline_weight_conversion": {
            "logical_elements": EXPECTED_LAYERS
            * EXPECTED_EXPERTS
            * EXPECTED_HIDDEN,
            "padded_elements": weight_elements,
            "conversion_events": weight_elements,
            "method": "deterministic_rtn_block8",
        },
        "runtime_activation_conversion_per_token": {
            "logical_elements": EXPECTED_LAYERS * EXPECTED_HIDDEN,
            "padded_elements": activation_elements,
            "conversion_events": activation_elements,
        },
        "storage": {
            "weight_data_bits": weight_data_bits,
            "weight_scale_bits": weight_scale_bits,
            "weight_total_bytes": math.ceil(
                (weight_data_bits + weight_scale_bits) / 8
            ),
            "weight_block_count": weight_blocks,
        },
        "traffic_per_token": {
            "hbm": {
                "router_input_activation_bytes": 0,
                "router_logits_bytes": 0,
                "weight_read_upper_bound_bytes": math.ceil(
                    (weight_data_bits + weight_scale_bits) / 8
                ),
                "weight_cache_reuse_credit": 0,
                "boundary": (
                    "weights_only_conservative_streaming_bound; router input "
                    "and logits are producer-consumer on-chip values"
                ),
            },
            "on_chip_vector_sram_and_conversion": {
                "activation_data_bits": activation_data_bits,
                "activation_scale_bits": activation_scale_bits,
                "activation_total_bytes": math.ceil(
                    (activation_data_bits + activation_scale_bits) / 8
                ),
                "bf16_logits_writeout_elements": output_elements,
                "bf16_logits_bytes": output_elements * 2,
            },
        },
        "compute_per_token": {
            "logical_macs": logical_macs,
            "issued_mac_slots_before_blen_padding": issued_mac_slots,
            "mlen_partitions_per_layer": math.ceil(EXPECTED_HIDDEN / mlen),
            "fp32_softmax_rows": EXPECTED_LAYERS,
            "fp32_softmax_elements": output_elements,
            "exact_topk_rows": EXPECTED_LAYERS,
            "bf16_output_cast_elements": output_elements,
        },
        "deployment_multiplicity": "unbound_until_exact_mapping_receipt",
        "classification": _classification(),
        "blockers": blockers,
        "claim_boundary": (
            "Counts and storage/traffic bounds are prospective geometry only; "
            "they are not latency, throughput, power, energy, or area evidence."
        ),
    }
    return receipt | {"content_hash": canonical_hash(receipt)}


def build_plan(
    *,
    config_path: Path,
    heldout_path: Path,
    ancestry_path: Path,
    matrix_mlens: Sequence[int] = (1024,),
    include_cross_mxfp: bool = False,
) -> dict[str, Any]:
    config, config_identity = _load_target_config(config_path.resolve())
    heldout = load_immutable_json(heldout_path.resolve())
    validate_heldout_manifest(heldout)
    ancestry = load_immutable_json(ancestry_path.resolve())
    baseline_records = validate_ancestry(ancestry)

    pairs = [(token, token) for token in ROUTER_FORMATS]
    if include_cross_mxfp:
        pairs.extend((left, right) for left in ("E4M3", "E5M2") for right in ("E4M3", "E5M2") if left != right)
    variants = tuple(
        _variant(weight, activation, matrix_mlen=mlen)
        for mlen in matrix_mlens
        for weight, activation in pairs
    )
    if len({item["variant_id"] for item in variants}) != len(variants):
        raise ValueError("router variant IDs are not unique")

    jobs = []
    for baseline in sorted(baseline_records, key=lambda row: row["body_profile_id"]):
        profile = DecodePrecisionProfile.from_dict(baseline["body_profile"])
        body_bank = dict(baseline["body_bank"])
        body_bank_binding_hash = (
            str(body_bank["sha256"])
            if set(body_bank) == {"path", "kind", "sha256"}
            else canonical_hash(body_bank)
        )
        for variant in variants:
            job_body = {
                "body_profile_id": profile.profile_id,
                "body_profile": profile.to_dict(),
                "body_bank": body_bank,
                "body_bank_binding_hash": body_bank_binding_hash,
                "quantizer_provenance_hash": baseline[
                    "quantizer_provenance_hash"
                ],
                "source_numerical_result": dict(
                    baseline["source_numerical_result"]
                ),
                "source_numerical_journal": dict(
                    baseline["source_numerical_journal"]
                ),
                "source_journal_record_hash": baseline[
                    "source_journal_record_hash"
                ],
                "source_result_row_hash": baseline["source_result_row_hash"],
                "paired_baseline_router": dict(_BF16_ROUTER_CONTRACT),
                "variant": dict(variant),
                "heldout_manifest_hash": heldout["content_hash"],
                "measurement_contract": {
                    "replay_input": "same_bf16_router_inputs_per_layer_and_token",
                    "replay_split": "heldout_only",
                    "per_layer_topk_set_agreement": True,
                    "per_layer_topk_order_agreement": True,
                    "full_router_probability_mae_rmse_linf": True,
                    "paired_end_to_end_nll": True,
                    "task_effects": "measure_when_driver_supports_task_protocol_else_explicit_unsupported",
                    "baseline_reexecuted_in_same_invocation": True,
                    "failed_and_oom_are_terminal_rows": True,
                },
                "cost_receipt": build_prospective_cost_receipt(
                    body_profile=profile,
                    variant=variant,
                ),
            }
            job_id = f"rpa-{canonical_hash(job_body)[:24]}"
            jobs.append(job_body | {"job_id": job_id})
    jobs.sort(key=lambda row: row["job_id"])
    profile_shards = {
        profile_id: index % EXPECTED_SHARDS
        for index, profile_id in enumerate(
            sorted({str(row["body_profile_id"]) for row in jobs})
        )
    }
    jobs = [
        row | {"shard_index": profile_shards[str(row["body_profile_id"])]}
        for row in jobs
    ]
    body = {
        "schema_version": PLAN_SCHEMA,
        "target": {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "architecture": dict(_TARGET_ARCHITECTURE),
        },
        "bindings": {
            "config": config_identity,
            "config_semantic_hash": canonical_hash(config),
            "heldout_manifest": path_identity(heldout_path.resolve()),
            "heldout_manifest_hash": heldout["content_hash"],
            "bf16_ancestry": path_identity(ancestry_path.resolve()),
            "bf16_ancestry_hash": ancestry["content_hash"],
            "router_variant_sources": router_variant_source_bindings(),
        },
        "canonical_body_profile_count": EXPECTED_PROFILE_COUNT,
        "canonical_body_profile_census_modified": False,
        "execution": {
            "required_shards": EXPECTED_SHARDS,
            "partition": "body_profile_affine_round_robin/v1",
            "restart": "immutable_requests_append_only_terminal_results",
            "paired_bf16_baseline_per_job": True,
        },
        "variants": list(variants),
        "jobs": jobs,
        "classification": _classification(),
        "admission": {
            "numerical_measurements_required": True,
            "compiler_router_receipt_required_for_publication": True,
            "hardware_router_receipt_required_for_publication": True,
            "calibrated_router_power_receipt_required_for_power_claim": True,
            "current_state": "ablation_only_unrankable",
        },
        "claim_boundary": (
            "Successful rows may report paired held-out router agreement and "
            "numerical deltas only. They never enter the canonical sweep or selection."
        ),
    }
    return body | {"content_hash": canonical_hash(body)}


def validate_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("canonical_body_profile_count") != EXPECTED_PROFILE_COUNT
        or plan.get("canonical_body_profile_census_modified") is not False
        or plan.get("classification") != _classification()
    ):
        raise ValueError("router ablation plan boundary differs")
    target = plan.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("model_name") != MODEL_NAME
        or target.get("model_revision") != MODEL_REVISION
        or target.get("tokenizer_revision") != MODEL_REVISION
        or target.get("architecture") != _TARGET_ARCHITECTURE
    ):
        raise ValueError("router ablation plan target differs")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("router ablation plan bindings are missing")
    for field in ("config", "heldout_manifest", "bf16_ancestry"):
        identity = bindings.get(field)
        if not isinstance(identity, Mapping):
            raise ValueError(f"router plan lacks {field} identity")
        _verify_identity(identity, label=f"router plan {field}")
    variants = plan.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("router ablation plan contains no variants")
    matrix_mlens = tuple(dict.fromkeys(int(row["matrix_mlen"]) for row in variants))
    per_mlen = len(variants) // len(matrix_mlens)
    if per_mlen not in (3, 5) or per_mlen * len(matrix_mlens) != len(variants):
        raise ValueError("router ablation plan variant coverage differs")
    expected = build_plan(
        config_path=Path(str(bindings["config"]["path"])),
        heldout_path=Path(str(bindings["heldout_manifest"]["path"])),
        ancestry_path=Path(str(bindings["bf16_ancestry"]["path"])),
        matrix_mlens=matrix_mlens,
        include_cross_mxfp=per_mlen == 5,
    )
    if dict(plan) != expected:
        raise ValueError("router ablation plan does not reproduce from its ancestry")


@dataclass
class _LayerAccumulator:
    tokens: int = 0
    order_matches: int = 0
    set_matches: int = 0
    topk_overlap_sum: int = 0
    probability_entries: int = 0
    probability_abs_sum: float = 0.0
    probability_sq_sum: float = 0.0
    probability_linf: float = 0.0


class RouterAgreementAccumulator:
    """Deterministically aggregate paired, fixed-input router comparisons."""

    def __init__(
        self,
        *,
        heldout_manifest_hash: str,
        shadow_router_input_hash: str,
        layers: int = EXPECTED_LAYERS,
        experts: int = EXPECTED_EXPERTS,
        top_k: int = EXPECTED_TOP_K,
    ):
        self.layers = int(layers)
        self.experts = int(experts)
        self.top_k = int(top_k)
        self.heldout_manifest_hash = str(heldout_manifest_hash)
        self.shadow_router_input_hash = str(shadow_router_input_hash)
        if min(self.layers, self.experts, self.top_k) <= 0 or self.top_k > self.experts:
            raise ValueError("invalid router agreement geometry")
        if not _SHA256.fullmatch(self.heldout_manifest_hash) or not _SHA256.fullmatch(
            self.shadow_router_input_hash
        ):
            raise ValueError("router agreement requires immutable held-out/input hashes")
        self._layers: dict[int, _LayerAccumulator] = defaultdict(_LayerAccumulator)

    def bind_shadow_router_input_hash(self, shadow_router_input_hash: str) -> None:
        """Bind the completed streaming hash before finalization."""

        if not _SHA256.fullmatch(str(shadow_router_input_hash)):
            raise ValueError("shadow router-input hash must be SHA-256")
        if self.shadow_router_input_hash != "0" * 64:
            raise RuntimeError("shadow router-input hash is already bound")
        self.shadow_router_input_hash = str(shadow_router_input_hash)

    def update(
        self,
        layer_index: int,
        baseline_probabilities: torch.Tensor,
        baseline_indices: torch.Tensor,
        candidate_probabilities: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> None:
        if not 0 <= int(layer_index) < self.layers:
            raise ValueError("router layer index is outside the target")
        if baseline_probabilities.shape != candidate_probabilities.shape:
            raise ValueError("paired router probability shapes differ")
        if baseline_indices.shape != candidate_indices.shape:
            raise ValueError("paired router index shapes differ")
        if baseline_probabilities.ndim != 2 or baseline_probabilities.shape[1] != self.experts:
            raise ValueError("router probabilities have the wrong expert dimension")
        if baseline_indices.ndim != 2 or baseline_indices.shape[1] != self.top_k:
            raise ValueError("router indices have the wrong top-k dimension")
        if baseline_indices.shape[0] != baseline_probabilities.shape[0]:
            raise ValueError("paired router token counts differ")
        for tensor in (baseline_probabilities, candidate_probabilities):
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("router probabilities must be finite FP32")
            if not torch.allclose(
                tensor.sum(dim=-1),
                torch.ones(tensor.shape[0], device=tensor.device),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("router probabilities do not sum to one")
        for tensor in (baseline_indices, candidate_indices):
            if tensor.min() < 0 or tensor.max() >= self.experts:
                raise ValueError("router indices are out of range")
            if (torch.sort(tensor, dim=-1).values[:, 1:] == torch.sort(tensor, dim=-1).values[:, :-1]).any():
                raise ValueError("router top-k contains a duplicate expert")

        tokens = int(baseline_probabilities.shape[0])
        order_equal = (baseline_indices == candidate_indices).all(dim=-1)
        baseline_sorted = torch.sort(baseline_indices, dim=-1).values
        candidate_sorted = torch.sort(candidate_indices, dim=-1).values
        set_equal = (baseline_sorted == candidate_sorted).all(dim=-1)
        overlap = (
            baseline_indices.unsqueeze(-1) == candidate_indices.unsqueeze(-2)
        ).any(dim=-1).sum(dim=-1)
        difference = (
            baseline_probabilities.to(torch.float64)
            - candidate_probabilities.to(torch.float64)
        ).abs()
        acc = self._layers[int(layer_index)]
        acc.tokens += tokens
        acc.order_matches += int(order_equal.sum().item())
        acc.set_matches += int(set_equal.sum().item())
        acc.topk_overlap_sum += int(overlap.sum().item())
        acc.probability_entries += int(difference.numel())
        acc.probability_abs_sum += float(difference.sum().item())
        acc.probability_sq_sum += float(difference.square().sum().item())
        acc.probability_linf = max(acc.probability_linf, float(difference.max().item()))

    def finalize(self) -> dict[str, Any]:
        missing = sorted(set(range(self.layers)).difference(self._layers))
        if missing:
            raise ValueError(f"router agreement is missing layers: {missing}")
        rows = []
        totals = _LayerAccumulator()
        for layer_index in range(self.layers):
            value = self._layers[layer_index]
            if value.tokens <= 0:
                raise ValueError("router agreement layer has no held-out tokens")
            row = {
                "layer_index": layer_index,
                "tokens": value.tokens,
                "topk_order_matches": value.order_matches,
                "topk_order_agreement": value.order_matches / value.tokens,
                "topk_set_matches": value.set_matches,
                "topk_set_agreement": value.set_matches / value.tokens,
                "mean_topk_overlap": value.topk_overlap_sum / value.tokens,
                "probability_mae": value.probability_abs_sum / value.probability_entries,
                "probability_rmse": math.sqrt(value.probability_sq_sum / value.probability_entries),
                "probability_linf": value.probability_linf,
            }
            rows.append(row)
            for field in (
                "tokens", "order_matches", "set_matches", "topk_overlap_sum",
                "probability_entries",
            ):
                setattr(totals, field, getattr(totals, field) + getattr(value, field))
            totals.probability_abs_sum += value.probability_abs_sum
            totals.probability_sq_sum += value.probability_sq_sum
            totals.probability_linf = max(totals.probability_linf, value.probability_linf)
        body = {
            "schema_version": AGREEMENT_SCHEMA,
            "input_pairing": "same_bf16_router_inputs_per_layer_and_token",
            "split_role": "heldout_router_precision_evaluation",
            "heldout_manifest_hash": self.heldout_manifest_hash,
            "shadow_router_input_hash": self.shadow_router_input_hash,
            "layers": self.layers,
            "experts": self.experts,
            "top_k": self.top_k,
            "layer_rows": rows,
            "aggregate": {
                "layer_token_observations": totals.tokens,
                "topk_order_agreement": totals.order_matches / totals.tokens,
                "topk_set_agreement": totals.set_matches / totals.tokens,
                "mean_topk_overlap": totals.topk_overlap_sum / totals.tokens,
                "probability_mae": totals.probability_abs_sum / totals.probability_entries,
                "probability_rmse": math.sqrt(totals.probability_sq_sum / totals.probability_entries),
                "probability_linf": totals.probability_linf,
            },
        }
        return body | {"content_hash": canonical_hash(body)}


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_agreement(
    value: Mapping[str, Any],
    *,
    heldout_manifest_hash: str,
) -> None:
    if (
        value.get("schema_version") != AGREEMENT_SCHEMA
        or value.get("input_pairing")
        != "same_bf16_router_inputs_per_layer_and_token"
        or value.get("split_role") != "heldout_router_precision_evaluation"
        or value.get("heldout_manifest_hash") != heldout_manifest_hash
        or not _SHA256.fullmatch(
            str(value.get("shadow_router_input_hash", ""))
        )
        or "paired_input_hash" in value
        or value.get("layers") != EXPECTED_LAYERS
        or value.get("experts") != EXPECTED_EXPERTS
        or value.get("top_k") != EXPECTED_TOP_K
    ):
        raise ValueError("router agreement contract or geometry differs")
    body = {key: item for key, item in value.items() if key != "content_hash"}
    if value.get("content_hash") != canonical_hash(body):
        raise ValueError("router agreement content hash differs")
    rows = value.get("layer_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_LAYERS:
        raise ValueError("router agreement lacks 48 per-layer rows")
    totals = {
        "tokens": 0,
        "order": 0,
        "set": 0,
        "overlap": 0.0,
        "abs": 0.0,
        "square": 0.0,
        "entries": 0,
        "linf": 0.0,
    }
    for expected_layer, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("layer_index") != expected_layer:
            raise ValueError("router agreement layer order differs")
        tokens = row.get("tokens")
        order_matches = row.get("topk_order_matches")
        set_matches = row.get("topk_set_matches")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or isinstance(order_matches, bool)
            or not isinstance(order_matches, int)
            or isinstance(set_matches, bool)
            or not isinstance(set_matches, int)
            or not 0 <= order_matches <= set_matches <= tokens
        ):
            raise ValueError("router agreement match counts are invalid")
        order_rate = _finite_number(
            row.get("topk_order_agreement"), label="top-k order agreement"
        )
        set_rate = _finite_number(
            row.get("topk_set_agreement"), label="top-k set agreement"
        )
        overlap = _finite_number(
            row.get("mean_topk_overlap"), label="top-k overlap"
        )
        mae = _finite_number(row.get("probability_mae"), label="probability MAE")
        rmse = _finite_number(
            row.get("probability_rmse"), label="probability RMSE"
        )
        linf = _finite_number(
            row.get("probability_linf"), label="probability Linf"
        )
        if (
            not math.isclose(order_rate, order_matches / tokens, abs_tol=1e-12)
            or not math.isclose(set_rate, set_matches / tokens, abs_tol=1e-12)
            or not 0.0 <= overlap <= EXPECTED_TOP_K
            or not 0.0 <= mae <= 1.0
            or not 0.0 <= rmse <= 1.0
            or not 0.0 <= linf <= 1.0
        ):
            raise ValueError("router agreement derived values are invalid")
        entries = tokens * EXPECTED_EXPERTS
        totals["tokens"] += tokens
        totals["order"] += order_matches
        totals["set"] += set_matches
        totals["overlap"] += overlap * tokens
        totals["abs"] += mae * entries
        totals["square"] += rmse * rmse * entries
        totals["entries"] += entries
        totals["linf"] = max(totals["linf"], linf)
    aggregate = value.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("router agreement aggregate is missing")
    expected = {
        "layer_token_observations": totals["tokens"],
        "topk_order_agreement": totals["order"] / totals["tokens"],
        "topk_set_agreement": totals["set"] / totals["tokens"],
        "mean_topk_overlap": totals["overlap"] / totals["tokens"],
        "probability_mae": totals["abs"] / totals["entries"],
        "probability_rmse": math.sqrt(totals["square"] / totals["entries"]),
        "probability_linf": totals["linf"],
    }
    if set(aggregate) != set(expected):
        raise ValueError("router agreement aggregate fields differ")
    for key, required in expected.items():
        observed = aggregate.get(key)
        if isinstance(required, int):
            if observed != required:
                raise ValueError(f"router agreement aggregate {key} differs")
        elif not math.isclose(
            _finite_number(observed, label=f"aggregate {key}"),
            required,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"router agreement aggregate {key} differs")


def _validate_driver_receipt(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    heldout_manifest_hash: str,
) -> None:
    body = {key: item for key, item in value.items() if key != "content_hash"}
    if (
        value.get("schema_version") != DRIVER_RECEIPT_SCHEMA
        or value.get("content_hash") != canonical_hash(body)
        or value.get("offline_local_files_only") is not True
        or value.get("transformers_version") != "5.5.0"
        or value.get("fused_model_class") != "Qwen3MoeForCausalLM"
        or value.get("model_revision") != MODEL_REVISION
        or value.get("model_snapshot_revision_verified") is not True
        or value.get("tokenizer_revision") != MODEL_REVISION
        or value.get("tokenizer_snapshot_revision_verified") is not True
        or value.get("heldout_manifest_hash") != heldout_manifest_hash
        or value.get("body_profile_id") != job["body_profile_id"]
        or value.get("body_bank_ancestry_verified") is not True
        or value.get("body_nonrouter_parameters_unchanged") is not True
        or value.get("body_weight_bank_structure_verified") is not True
        or value.get("router_target_pattern")
        != "model.layers.<index>.mlp.gate"
        or value.get("router_target_count") != EXPECTED_LAYERS
        or value.get("only_router_modules_replaced") is not True
        or value.get("router_variant_id") != job["variant"]["variant_id"]
        or value.get("decode_query_length") != 1
        or value.get("key_format") != job["body_profile"]["key_format"]
        or value.get("value_format") != job["body_profile"]["value_format"]
        or value.get("key_format") != value.get("value_format")
        or value.get("local_decode_head_unchanged") is not True
        or value.get("prefill_owner") != "external_bf16_source_model"
        or value.get("serving_first_token_owner") != "prefill"
        or value.get("evaluation_first_decode_input_source")
        != "teacher_forced_continuation_token_0"
        or value.get("evaluation_input_generated_by_bf16_model") is not False
        or value.get("prefill_lm_head_executed") is not False
        or value.get("continuation_decode_contract")
        != _CONTINUATION_DECODE_CONTRACT
        or not _SHA256.fullmatch(
            str(value.get("first_decode_input_stream_hash", ""))
        )
        or not _SHA256.fullmatch(
            str(value.get("scored_decode_suffix_stream_hash", ""))
        )
        or not _SHA256.fullmatch(
            str(value.get("teacher_forced_token_stream_hash", ""))
        )
    ):
        raise ValueError("router driver execution receipt differs")
    for field in (
        "model_config",
        "tokenizer_json",
        "tokenizer_config",
        "chat_template",
        "driver_source",
        "model_snapshot_content_seal",
    ):
        identity = value.get(field)
        if not isinstance(identity, Mapping):
            raise ValueError(f"router driver receipt lacks {field}")
        _verify_identity(identity, label=f"router driver {field}")
    prefill = value.get("prefill_cache_binding")
    if not isinstance(prefill, Mapping):
        raise ValueError("router driver receipt lacks its BF16 prefill cache binding")
    _validate_prefill_cache_receipt_binding(
        prefill, heldout_manifest_hash=heldout_manifest_hash
    )
    counts = value.get("heldout_counts")
    if (
        not isinstance(counts, Mapping)
        or isinstance(counts.get("records"), bool)
        or not isinstance(counts.get("records"), int)
        or counts["records"] <= 0
        or isinstance(counts.get("scored_tokens_per_arm"), bool)
        or not isinstance(counts.get("scored_tokens_per_arm"), int)
        or counts["scored_tokens_per_arm"] <= 0
        or counts.get("prefill_owned_input_tokens_per_arm") != counts["records"]
        or isinstance(counts.get("continuation_tokens_per_arm"), bool)
        or not isinstance(counts.get("continuation_tokens_per_arm"), int)
        or counts["continuation_tokens_per_arm"]
        != counts["scored_tokens_per_arm"]
        + counts["prefill_owned_input_tokens_per_arm"]
    ):
        raise ValueError("router driver held-out counts are invalid")


def _validate_prefill_cache_receipt_binding(
    value: Mapping[str, Any], *, heldout_manifest_hash: str
) -> None:
    if set(value) != {
        "cache_key",
        "index_content_hash",
        "index",
        "artifact_tree",
    }:
        raise ValueError("router prefill cache binding fields differ")
    index_identity = value.get("index")
    tree_identity = value.get("artifact_tree")
    if not isinstance(index_identity, Mapping) or not isinstance(
        tree_identity, Mapping
    ):
        raise ValueError("router prefill cache binding lacks identities")
    index_path = _verify_identity(index_identity, label="router prefill cache index")
    tree_path = _verify_identity(tree_identity, label="router prefill cache tree")
    if index_path.parent != tree_path:
        raise ValueError("router prefill cache index is outside its sealed tree")
    index = load_immutable_json(index_path)
    if (
        index.get("schema_version") != PREFILL_CACHE_SCHEMA
        or index.get("cache_key") != value.get("cache_key")
        or index.get("content_hash") != value.get("index_content_hash")
        or index.get("heldout_manifest_hash") != heldout_manifest_hash
        or index.get("serving_first_token_owner") != "prefill"
        or index.get("evaluation_first_decode_input_source")
        != "teacher_forced_continuation_token_0"
        or index.get("evaluation_input_generated_by_bf16_model") is not False
        or index.get("prefill_lm_head_executed") is not False
        or index.get("continuation_decode_contract")
        != _CONTINUATION_DECODE_CONTRACT
    ):
        raise ValueError("router prefill cache binding differs from its index")


def validate_result(plan: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unsupported router ablation result schema")
    jobs = {row["job_id"]: row for row in plan["jobs"]}
    job_id = str(result.get("job_id", ""))
    if job_id not in jobs:
        raise ValueError("router result job is absent from the plan")
    job = jobs[job_id]
    if (
        result.get("plan_hash") != plan["content_hash"]
        or result.get("request_hash") is None
        or not _SHA256.fullmatch(str(result.get("request_hash")))
        or result.get("body_profile_id") != job["body_profile_id"]
        or result.get("variant_id") != job["variant"]["variant_id"]
        or result.get("classification") != _classification()
    ):
        raise ValueError("router result bindings differ from its plan job")
    status = result.get("status")
    if status not in TERMINAL_STATUSES:
        raise ValueError("router result is not terminal")
    if status != "success":
        failure = result.get("failure")
        if not isinstance(failure, Mapping) or not str(failure.get("message", "")):
            raise ValueError("terminal router failure lacks diagnostics")
        if result.get("measurements") not in (None, {}):
            raise ValueError("failed router result may not carry partial measurements")
        return
    measurements = result.get("measurements")
    if not isinstance(measurements, Mapping):
        raise ValueError("successful router result lacks measurements")
    receipt = measurements.get("execution_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("successful router result lacks an execution receipt")
    _validate_driver_receipt(
        receipt,
        job=job,
        heldout_manifest_hash=plan["bindings"]["heldout_manifest_hash"],
    )
    agreement = measurements.get("router_agreement")
    if not isinstance(agreement, Mapping):
        raise ValueError("successful router result lacks agreement evidence")
    _validate_agreement(
        agreement,
        heldout_manifest_hash=plan["bindings"]["heldout_manifest_hash"],
    )
    end_to_end = measurements.get("end_to_end")
    if (
        not isinstance(end_to_end, Mapping)
        or end_to_end.get("paired_bf16_reexecuted") is not True
        or end_to_end.get("baseline_router") != _BF16_ROUTER_CONTRACT
        or end_to_end.get("heldout_manifest_hash")
        != plan["bindings"]["heldout_manifest_hash"]
        or end_to_end.get("continuation_decode_contract")
        != _CONTINUATION_DECODE_CONTRACT
        or not _SHA256.fullmatch(
            str(end_to_end.get("teacher_forced_token_stream_hash", ""))
        )
        or not _SHA256.fullmatch(
            str(end_to_end.get("first_decode_input_stream_hash", ""))
        )
        or not _SHA256.fullmatch(
            str(end_to_end.get("scored_decode_suffix_stream_hash", ""))
        )
        or "paired_input_hash" in end_to_end
    ):
        raise ValueError("end-to-end result lacks a reexecuted paired BF16 control")
    baseline = end_to_end.get("bf16_router_baseline")
    candidate = end_to_end.get("mx_router_candidate")
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("end-to-end result lacks its paired arms")
    baseline_nll = _finite_number(baseline.get("mean_token_nll"), label="baseline NLL")
    candidate_nll = _finite_number(candidate.get("mean_token_nll"), label="candidate NLL")
    baseline_tokens = baseline.get("token_count")
    candidate_tokens = candidate.get("token_count")
    token_stream_hash = end_to_end["teacher_forced_token_stream_hash"]
    first_input_hash = end_to_end["first_decode_input_stream_hash"]
    scored_suffix_hash = end_to_end["scored_decode_suffix_stream_hash"]
    if (
        isinstance(baseline_tokens, bool)
        or not isinstance(baseline_tokens, int)
        or baseline_tokens <= 0
        or candidate_tokens != baseline_tokens
        or baseline_tokens
        != receipt["heldout_counts"]["scored_tokens_per_arm"]
        or baseline.get("teacher_forced_token_stream_hash") != token_stream_hash
        or candidate.get("teacher_forced_token_stream_hash") != token_stream_hash
        or baseline.get("first_decode_input_stream_hash") != first_input_hash
        or candidate.get("first_decode_input_stream_hash") != first_input_hash
        or baseline.get("scored_decode_suffix_stream_hash")
        != scored_suffix_hash
        or candidate.get("scored_decode_suffix_stream_hash")
        != scored_suffix_hash
        or receipt.get("teacher_forced_token_stream_hash") != token_stream_hash
        or receipt.get("first_decode_input_stream_hash") != first_input_hash
        or receipt.get("scored_decode_suffix_stream_hash")
        != scored_suffix_hash
    ):
        raise ValueError(
            "paired NLL arms must cover the same positive teacher-forced token stream"
        )
    observed_delta = _finite_number(end_to_end.get("mean_token_nll_delta"), label="NLL delta")
    if not math.isclose(observed_delta, candidate_nll - baseline_nll, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reported NLL delta differs from paired measurements")
    tasks = end_to_end.get("task_effects")
    if not isinstance(tasks, Mapping) or tasks.get("status") not in ("measured", "unsupported"):
        raise ValueError("task effects must be measured or explicitly unsupported")
    if tasks.get("status") == "unsupported" and not str(tasks.get("reason", "")):
        raise ValueError("unsupported task effects require a reason")
    if tasks.get("status") == "measured":
        if not _SHA256.fullmatch(str(tasks.get("protocol_hash", ""))):
            raise ValueError("measured task effects lack an immutable protocol hash")
        task_rows = tasks.get("tasks")
        if not isinstance(task_rows, list) or not task_rows:
            raise ValueError("measured task effects contain no task rows")
        seen = set()
        for task in task_rows:
            task_id = str(task.get("task_id", "")) if isinstance(task, Mapping) else ""
            if task_id not in {"ifeval", "gsm8k"} or task_id in seen:
                raise ValueError("measured task effects contain an unsupported task")
            baseline_score = _finite_number(
                task.get("bf16_router_score"), label=f"{task_id} BF16 score"
            )
            candidate_score = _finite_number(
                task.get("mx_router_score"), label=f"{task_id} MX score"
            )
            delta = _finite_number(task.get("delta"), label=f"{task_id} delta")
            if not math.isclose(
                delta, candidate_score - baseline_score, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("measured task delta differs from its paired scores")
            seen.add(task_id)


def build_request(
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    driver_identity: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": REQUEST_SCHEMA,
        "plan_hash": plan["content_hash"],
        "job": dict(job),
        "target": dict(plan["target"]),
        "bindings": dict(plan["bindings"]),
        "driver": dict(driver_identity),
        "required_output_schema": RESULT_SCHEMA,
        "classification": _classification(),
    }
    return body | {"content_hash": canonical_hash(body)}


def _terminal_failure(
    *,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    request_hash: str,
    status: str,
    message: str,
    returncode: int | None,
    stdout_sha256: str,
    stderr_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": RESULT_SCHEMA,
        "plan_hash": plan["content_hash"],
        "request_hash": request_hash,
        "job_id": job["job_id"],
        "body_profile_id": job["body_profile_id"],
        "variant_id": job["variant"]["variant_id"],
        "status": status,
        "measurements": None,
        "failure": {
            "message": message,
            "returncode": returncode,
            "stdout_sha256": stdout_sha256,
            "stderr_sha256": stderr_sha256,
        },
        "classification": _classification(),
    }
    return body | {"content_hash": canonical_hash(body)}


def _driver_command(driver_path: Path, *arguments: str) -> list[str]:
    resolved = driver_path.resolve()
    prefix = [sys.executable, str(resolved)] if resolved.suffix == ".py" else [str(resolved)]
    return prefix + list(arguments)


def _validate_request_bound_result(
    request: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    if result.get("request_hash") != request.get("content_hash"):
        raise ValueError("driver result does not bind its exact request")
    if result.get("status") == "success":
        receipt = result.get("measurements", {}).get("execution_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("driver_source") != request.get(
            "driver"
        ):
            raise ValueError("driver result execution receipt binds another driver")


def _run_batched_shard(
    *,
    plan: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    shard_index: int,
    gpu: str,
    driver_path: Path,
    driver_identity: Mapping[str, Any],
    output_root: Path,
    timeout_seconds: int,
    counts: dict[str, int],
) -> dict[str, int]:
    pending: list[tuple[Mapping[str, Any], dict[str, Any], Path, Path]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        result_path = output_root / "results" / f"{job_id}.json"
        if result_path.exists():
            request_path = output_root / "requests" / f"{job_id}.json"
            request = load_immutable_json(request_path)
            if request != build_request(
                plan, job, driver_identity=driver_identity
            ):
                raise ValueError("existing router request differs from this driver")
            result = load_immutable_json(result_path)
            validate_result(plan, result)
            _validate_request_bound_result(request, result)
            counts[result["status"]] += 1
            continue
        request = build_request(plan, job, driver_identity=driver_identity)
        request_path = output_root / "requests" / f"{job_id}.json"
        write_immutable_json(request_path, request)
        temporary_result = output_root / "driver-results" / f"{job_id}.json"
        temporary_result.parent.mkdir(parents=True, exist_ok=True)
        pending.append((job, request, request_path, temporary_result))
    if not pending:
        return counts

    batch_body = {
        "schema_version": DRIVER_BATCH_SCHEMA,
        "plan_hash": plan["content_hash"],
        "shard_index": shard_index,
        "driver": dict(driver_identity),
        "jobs": [
            {
                "job_id": job["job_id"],
                "body_profile_id": job["body_profile_id"],
                "request": path_identity(request_path),
                "output": str(temporary_result.resolve()),
            }
            for job, _, request_path, temporary_result in pending
        ],
    }
    batch = batch_body | {"content_hash": canonical_hash(batch_body)}
    batch_path = output_root / "requests" / f"shard-{shard_index}.batch.json"
    write_immutable_json(batch_path, batch)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"shard-{shard_index}.batch.stdout"
    stderr_path = log_root / f"shard-{shard_index}.batch.stderr"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PLENA_ROUTER_ABLATION_SHARD"] = str(shard_index)
    environment.setdefault(
        "PLENA_ROUTER_DRIVER_CACHE",
        str((output_root / "driver-cache").resolve()),
    )
    returncode: int | None = None
    message = "batched router driver failed"
    try:
        completed = subprocess.run(
            _driver_command(
                driver_path,
                "--batch-manifest",
                str(batch_path.resolve()),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        message = f"batched router driver timed out after {timeout_seconds} seconds"
    except OSError as exc:
        stdout = b""
        stderr = str(exc).encode("utf-8", errors="replace")
        message = f"batched router driver could not start: {type(exc).__name__}: {exc}"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    stdout_hash = sha256_file(stdout_path)
    stderr_hash = sha256_file(stderr_path)
    lowered = (stderr + stdout).decode("utf-8", errors="replace").casefold()
    missing_status = (
        "oom"
        if returncode in (134, 137) or "out of memory" in lowered
        else "failed"
    )
    for job, request, _, temporary_result in pending:
        result_path = output_root / "results" / f"{job['job_id']}.json"
        if temporary_result.is_file():
            try:
                result = load_immutable_json(temporary_result)
                validate_result(plan, result)
                _validate_request_bound_result(request, result)
            except Exception as exc:
                result = _terminal_failure(
                    plan=plan,
                    job=job,
                    request_hash=request["content_hash"],
                    status="failed",
                    message=f"invalid batched driver result: {exc}",
                    returncode=returncode,
                    stdout_sha256=stdout_hash,
                    stderr_sha256=stderr_hash,
                )
        else:
            result = _terminal_failure(
                plan=plan,
                job=job,
                request_hash=request["content_hash"],
                status=missing_status,
                message=message,
                returncode=returncode,
                stdout_sha256=stdout_hash,
                stderr_sha256=stderr_hash,
            )
        write_immutable_json(result_path, result)
        counts[result["status"]] += 1
    return counts


def run_shard(
    *,
    plan_path: Path,
    shard_index: int,
    gpu: str,
    driver_path: Path,
    output_root: Path,
    timeout_seconds: int,
    batch_driver: bool = False,
) -> dict[str, int]:
    plan = load_immutable_json(plan_path.resolve())
    validate_plan(plan)
    if not 0 <= shard_index < EXPECTED_SHARDS:
        raise ValueError("router shard index must be in [0, 3]")
    driver = path_identity(driver_path.resolve())
    if driver["kind"] != "file":
        raise ValueError("router ablation driver must be a file")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    counts = {status: 0 for status in TERMINAL_STATUSES}
    jobs = [row for row in plan["jobs"] if row["shard_index"] == shard_index]
    if batch_driver:
        return _run_batched_shard(
            plan=plan,
            jobs=jobs,
            shard_index=shard_index,
            gpu=gpu,
            driver_path=driver_path,
            driver_identity=driver,
            output_root=root,
            timeout_seconds=timeout_seconds,
            counts=counts,
        )
    for job in jobs:
        job_id = job["job_id"]
        result_path = root / "results" / f"{job_id}.json"
        if result_path.exists():
            request_path = root / "requests" / f"{job_id}.json"
            request = load_immutable_json(request_path)
            if request != build_request(plan, job, driver_identity=driver):
                raise ValueError("existing router request differs from this driver")
            result = load_immutable_json(result_path)
            validate_result(plan, result)
            _validate_request_bound_result(request, result)
            counts[result["status"]] += 1
            continue
        request = build_request(plan, job, driver_identity=driver)
        request_path = root / "requests" / f"{job_id}.json"
        write_immutable_json(request_path, request)
        stdout_path = root / "logs" / f"{job_id}.stdout"
        stderr_path = root / "logs" / f"{job_id}.stderr"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_result = root / "driver-results" / f"{job_id}.json"
        temporary_result.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PLENA_ROUTER_ABLATION_SHARD"] = str(shard_index)
        returncode: int | None = None
        message = "external driver failed"
        try:
            completed = subprocess.run(
                _driver_command(
                    driver_path,
                    "--request",
                    str(request_path),
                    "--output",
                    str(temporary_result),
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            message = f"external driver timed out after {timeout_seconds} seconds"
        except OSError as exc:
            stdout = b""
            stderr = str(exc).encode("utf-8", errors="replace")
            message = f"external driver could not start: {type(exc).__name__}: {exc}"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        stdout_hash = sha256_file(stdout_path)
        stderr_hash = sha256_file(stderr_path)
        if returncode == 0 and temporary_result.is_file():
            try:
                result = load_immutable_json(temporary_result)
                validate_result(plan, result)
                _validate_request_bound_result(request, result)
            except Exception as exc:
                result = _terminal_failure(
                    plan=plan, job=job, request_hash=request["content_hash"],
                    status="failed", message=f"invalid driver result: {exc}",
                    returncode=returncode, stdout_sha256=stdout_hash,
                    stderr_sha256=stderr_hash,
                )
        else:
            lowered = (stderr + stdout).decode("utf-8", errors="replace").casefold()
            status = "oom" if returncode in (134, 137) or "out of memory" in lowered else "failed"
            result = _terminal_failure(
                plan=plan, job=job, request_hash=request["content_hash"], status=status,
                message=message, returncode=returncode, stdout_sha256=stdout_hash,
                stderr_sha256=stderr_hash,
            )
        write_immutable_json(result_path, result)
        counts[result["status"]] += 1
    return counts


def build_completion(plan_path: Path, results_root: Path) -> dict[str, Any]:
    plan = load_immutable_json(plan_path.resolve())
    validate_plan(plan)
    statuses = {status: 0 for status in TERMINAL_STATUSES}
    rows = []
    for job in plan["jobs"]:
        path = results_root.resolve() / "results" / f"{job['job_id']}.json"
        request_path = (
            results_root.resolve() / "requests" / f"{job['job_id']}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"router ablation result is missing: {path}")
        if not request_path.is_file():
            raise FileNotFoundError(
                f"router ablation request is missing: {request_path}"
            )
        request = load_immutable_json(request_path)
        if (
            request.get("schema_version") != REQUEST_SCHEMA
            or request.get("plan_hash") != plan["content_hash"]
            or request.get("job") != job
        ):
            raise ValueError("router completion request differs from its plan")
        result = load_immutable_json(path)
        validate_result(plan, result)
        _validate_request_bound_result(request, result)
        statuses[result["status"]] += 1
        prefill_cache_binding = None
        if result["status"] == "success":
            prefill_cache_binding = dict(
                result["measurements"]["execution_receipt"][
                    "prefill_cache_binding"
                ]
            )
        rows.append({
            "job_id": job["job_id"],
            "status": result["status"],
            "request": path_identity(request_path),
            "request_content_hash": request["content_hash"],
            "driver": dict(request["driver"]),
            "result": path_identity(path),
            "result_content_hash": result["content_hash"],
            "prefill_cache_binding": prefill_cache_binding,
        })
    body = {
        "schema_version": COMPLETION_SCHEMA,
        "plan_hash": plan["content_hash"],
        "terminal_job_count": len(rows),
        "expected_job_count": len(plan["jobs"]),
        "status_counts": statuses,
        "rows": rows,
        "classification": _classification(),
        "selection_exported": False,
        "publication_exported": False,
    }
    return body | {"content_hash": canonical_hash(body)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    heldout_parser = subparsers.add_parser("materialize-heldout")
    heldout_parser.add_argument("--config", type=Path, required=True)
    heldout_parser.add_argument("--source", type=Path, required=True)
    heldout_parser.add_argument("--decode-target-tokens", type=int, required=True)
    heldout_parser.add_argument("--output", type=Path, required=True)
    ancestry_parser = subparsers.add_parser("materialize-ancestry")
    ancestry_parser.add_argument("--manifest", type=Path, required=True)
    ancestry_parser.add_argument("--results-root", type=Path, required=True)
    ancestry_parser.add_argument(
        "--record-hash", action="append", required=True
    )
    ancestry_parser.add_argument("--output", type=Path, required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--heldout", type=Path, required=True)
    plan_parser.add_argument("--ancestry", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--matrix-mlen", type=int, action="append", default=[])
    plan_parser.add_argument("--include-cross-mxfp", action="store_true")
    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--shard", type=int, required=True)
    run_parser.add_argument("--gpu", required=True)
    run_parser.add_argument("--driver", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=21600)
    run_parser.add_argument("--batch-driver", action="store_true")
    finish_parser = subparsers.add_parser("complete")
    finish_parser.add_argument("--plan", type=Path, required=True)
    finish_parser.add_argument("--results-root", type=Path, required=True)
    finish_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "materialize-heldout":
        materialize_heldout_manifest(
            config_path=args.config,
            source_path=args.source,
            decode_target_tokens=args.decode_target_tokens,
            output_path=args.output,
        )
        return 0
    if args.command == "materialize-ancestry":
        materialize_ancestry_from_current_rows(
            manifest_path=args.manifest,
            results_root=args.results_root,
            record_hashes=tuple(args.record_hash),
            output_path=args.output,
        )
        return 0
    if args.command == "plan":
        plan = build_plan(
            config_path=args.config,
            heldout_path=args.heldout,
            ancestry_path=args.ancestry,
            matrix_mlens=tuple(args.matrix_mlen or (1024,)),
            include_cross_mxfp=args.include_cross_mxfp,
        )
        write_immutable_json(args.output, plan)
        return 0
    if args.command == "run-shard":
        run_shard(
            plan_path=args.plan, shard_index=args.shard, gpu=args.gpu,
            driver_path=args.driver, output_root=args.output_root,
            timeout_seconds=args.timeout_seconds, batch_driver=args.batch_driver,
        )
        return 0
    completion = build_completion(args.plan, args.results_root)
    write_immutable_json(args.output, completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
