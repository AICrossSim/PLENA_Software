#!/usr/bin/env python3
"""Offline executable for paired Qwen3-MoE decode-router measurements."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence


if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from decode_dse.profiles import DecodePrecisionProfile
from decode_dse.software.router_precision_ablation import (
    AGREEMENT_SCHEMA,
    DRIVER_BATCH_SCHEMA,
    DRIVER_RECEIPT_SCHEMA,
    EXPECTED_EXPERTS,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    EXPECTED_TOP_K,
    HELDOUT_SCHEMA,
    MODEL_NAME,
    MODEL_REVISION,
    PREFILL_CACHE_SCHEMA,
    PREFILL_RECORD_SCHEMA,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    RouterAgreementAccumulator,
    _BF16_ROUTER_CONTRACT,
    _CONTINUATION_DECODE_CONTRACT,
    _TARGET_ARCHITECTURE,
    _classification,
    _heldout_source_records,
    _verify_identity,
    build_prospective_cost_receipt,
    build_router_variant_pass_args,
    canonical_hash,
    path_identity,
    sha256_file,
    validate_ancestry,
    validate_heldout_manifest,
)
from decode_dse.software.sweep_plan import (
    _mase_tree_hash,
    load_immutable_json,
    profile_to_decode_quant_spec,
    write_immutable_json,
)


SNAPSHOT_SCHEMA = "decode-router-offline-model-snapshot/v2"
SHADOW_ROUTER_INPUT_HASH_DOMAIN = b"plena-router-shadow-bf16-input-stream/v1\0"
TEACHER_FORCED_TOKEN_HASH_DOMAIN = b"plena-router-teacher-forced-token-stream/v1\0"
FIRST_DECODE_INPUT_HASH_DOMAIN = b"plena-router-first-decode-input-stream/v1\0"
SCORED_DECODE_SUFFIX_HASH_DOMAIN = b"plena-router-scored-decode-suffix/v1\0"


@dataclass(frozen=True)
class OfflineSnapshot:
    cache_root: Path
    snapshot_root: Path
    model_config: Mapping[str, Any]
    tokenizer_json: Mapping[str, Any]
    tokenizer_config: Mapping[str, Any]
    weight_index: Mapping[str, Any]
    shard_count: int
    weight_bytes: int
    shard_paths: tuple[Path, ...]
    content_seal: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HeldoutTokens:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    first_token_id: int
    cache_tensors: Mapping[str, Any]


def _software_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configure_mase(config: Mapping[str, Any]) -> Path:
    executor = config.get("executor")
    if not isinstance(executor, Mapping):
        raise ValueError("config.executor is required")
    root = Path(str(executor.get("mase_src", "")))
    if not root.is_absolute():
        root = (_software_root() / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"configured MASE source is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_config(request: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    binding = request.get("bindings", {}).get("config")
    if not isinstance(binding, Mapping):
        raise ValueError("router request lacks its config identity")
    path = _verify_identity(binding, label="router driver config")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("router driver config must be an object")
    if canonical_hash(config) != request["bindings"].get("config_semantic_hash"):
        raise ValueError("router driver config semantic hash differs")
    return config, path


def validate_request(
    request: Mapping[str, Any], *, driver_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    body = {key: item for key, item in request.items() if key != "content_hash"}
    if (
        request.get("schema_version") != REQUEST_SCHEMA
        or request.get("content_hash") != canonical_hash(body)
        or request.get("required_output_schema") != RESULT_SCHEMA
        or request.get("classification") != _classification()
    ):
        raise ValueError("router driver request contract differs")
    target = request.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("model_name") != MODEL_NAME
        or target.get("model_revision") != MODEL_REVISION
        or target.get("tokenizer_revision") != MODEL_REVISION
        or target.get("architecture") != _TARGET_ARCHITECTURE
    ):
        raise ValueError("router driver target differs")
    expected_driver = path_identity(driver_path.resolve())
    if request.get("driver") != expected_driver:
        raise ValueError("router request is bound to a different driver source")
    config, _ = _load_config(request)
    _configure_mase(config)
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("router request bindings are missing")
    heldout_path = _verify_identity(
        bindings.get("heldout_manifest", {}), label="router held-out manifest"
    )
    heldout = load_immutable_json(heldout_path)
    validate_heldout_manifest(heldout)
    if heldout.get("content_hash") != bindings.get("heldout_manifest_hash"):
        raise ValueError("router held-out manifest hash differs")
    ancestry_path = _verify_identity(
        bindings.get("bf16_ancestry", {}), label="router BF16 ancestry"
    )
    ancestry = load_immutable_json(ancestry_path)
    ancestry_records = validate_ancestry(ancestry)
    if ancestry.get("content_hash") != bindings.get("bf16_ancestry_hash"):
        raise ValueError("router BF16 ancestry hash differs")
    source_manifest_identity = ancestry.get("source_manifest")
    if not isinstance(source_manifest_identity, Mapping):
        raise ValueError("executable router ancestry lacks its source manifest")
    source_manifest_path = _verify_identity(
        source_manifest_identity, label="router source sweep manifest"
    )
    from decode_dse.manifest import load_manifest

    source_manifest = load_manifest(source_manifest_path)
    if (
        source_manifest.canonical_hash != ancestry.get("source_manifest_hash")
        or source_manifest.model_name != MODEL_NAME
        or source_manifest.model_revision != MODEL_REVISION
        or source_manifest.tokenizer_revision != MODEL_REVISION
        or source_manifest.counts.get("total") != 3585
        or dict(source_manifest.model_architecture) != _TARGET_ARCHITECTURE
    ):
        raise ValueError("router source sweep manifest differs from the target")
    sources = bindings.get("router_variant_sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("router request lacks variant source bindings")
    for name, identity in sources.items():
        if not isinstance(identity, Mapping):
            raise TypeError(f"router source identity is invalid: {name}")
        _verify_identity(identity, label=f"router source {name}")

    job = request.get("job")
    if not isinstance(job, Mapping):
        raise ValueError("router request lacks its job")
    profile_value = job.get("body_profile")
    if not isinstance(profile_value, Mapping):
        raise ValueError("router request lacks its body profile")
    profile = DecodePrecisionProfile.from_dict(profile_value)
    if (
        profile.profile_id != job.get("body_profile_id")
        or profile.kind != "quantized"
        or profile.method != "rtn"
        or profile.key_format != profile.value_format
    ):
        raise ValueError("router driver requires an RTN quantized body with K=V")
    baseline = next(
        (
            record
            for record in ancestry_records
            if record["body_profile_id"] == profile.profile_id
        ),
        None,
    )
    if baseline is None:
        raise ValueError("router job body is absent from BF16 ancestry")
    manifest_entries = {
        entry.profile_id: entry for entry in source_manifest.entries
    }
    if (
        profile.profile_id not in manifest_entries
        or manifest_entries[profile.profile_id].profile.to_dict()
        != profile.to_dict()
        or source_manifest.quantizer_provenance.canonical_hash
        != baseline.get("quantizer_provenance_hash")
    ):
        raise ValueError("router body profile or quantizer provenance differs")
    for field in (
        "body_profile",
        "body_bank",
        "quantizer_provenance_hash",
        "source_numerical_result",
        "source_numerical_journal",
        "source_journal_record_hash",
        "source_result_row_hash",
    ):
        if job.get(field) != baseline.get(field):
            raise ValueError(f"router job differs from ancestry field {field}")
    variant = job.get("variant")
    if not isinstance(variant, Mapping):
        raise ValueError("router request lacks its router variant")
    build_router_variant_pass_args(variant)
    if job.get("cost_receipt") != build_prospective_cost_receipt(
        body_profile=profile, variant=variant
    ):
        raise ValueError("router request prospective cost receipt differs")
    bank_binding_hash = (
        str(job["body_bank"]["sha256"])
        if set(job["body_bank"]) == {"path", "kind", "sha256"}
        else canonical_hash(job["body_bank"])
    )
    measurement_contract = {
        "replay_input": "same_bf16_router_inputs_per_layer_and_token",
        "replay_split": "heldout_only",
        "per_layer_topk_set_agreement": True,
        "per_layer_topk_order_agreement": True,
        "full_router_probability_mae_rmse_linf": True,
        "paired_end_to_end_nll": True,
        "task_effects": "measure_when_driver_supports_task_protocol_else_explicit_unsupported",
        "baseline_reexecuted_in_same_invocation": True,
        "failed_and_oom_are_terminal_rows": True,
    }
    job_identity_body = {
        key: value
        for key, value in job.items()
        if key not in {"job_id", "shard_index"}
    }
    if (
        job.get("body_bank_binding_hash") != bank_binding_hash
        or job.get("measurement_contract") != measurement_contract
        or job.get("job_id")
        != f"rpa-{canonical_hash(job_identity_body)[:24]}"
        or isinstance(job.get("shard_index"), bool)
        or not isinstance(job.get("shard_index"), int)
        or not 0 <= job["shard_index"] < 4
    ):
        raise ValueError("router request job identity or measurement contract differs")
    if (
        job.get("paired_baseline_router") != _BF16_ROUTER_CONTRACT
        or job.get("heldout_manifest_hash") != heldout["content_hash"]
    ):
        raise ValueError("router job pairing contract differs")
    return dict(job), config, heldout


def _resolve_offline_snapshot(cache_root: Path) -> OfflineSnapshot:
    from transformers.utils import cached_file

    cache_root = cache_root.resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"offline model cache does not exist: {cache_root}")
    kwargs = {
        "revision": MODEL_REVISION,
        "cache_dir": str(cache_root),
        "local_files_only": True,
    }
    resolved: dict[str, Path] = {}
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        value = cached_file(MODEL_NAME, filename, **kwargs)
        if value is None:
            raise FileNotFoundError(f"offline snapshot lacks {filename}")
        path = Path(value).resolve()
        if MODEL_REVISION not in path.parts:
            raise ValueError("offline snapshot path is not the pinned commit")
        resolved[filename] = path
    roots = {path.parent for path in resolved.values()}
    if len(roots) != 1:
        raise ValueError("offline model metadata resolves to different snapshots")
    snapshot_root = roots.pop()
    index = json.loads(
        resolved["model.safetensors.index.json"].read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("offline model weight index is invalid")
    shards = tuple(sorted({str(name) for name in weight_map.values()}))
    shard_paths = tuple(snapshot_root / name for name in shards)
    if any(not path.is_file() for path in shard_paths):
        raise FileNotFoundError("offline model snapshot has missing weight shards")
    return OfflineSnapshot(
        cache_root=cache_root,
        snapshot_root=snapshot_root,
        model_config=path_identity(resolved["config.json"]),
        tokenizer_json=path_identity(resolved["tokenizer.json"]),
        tokenizer_config=path_identity(resolved["tokenizer_config.json"]),
        weight_index=path_identity(resolved["model.safetensors.index.json"]),
        shard_count=len(shard_paths),
        weight_bytes=sum(path.stat().st_size for path in shard_paths),
        shard_paths=shard_paths,
    )


def _seal_offline_snapshot(
    snapshot: OfflineSnapshot, cache_root: Path
) -> OfflineSnapshot:
    seal_key = canonical_hash(
        {
            "schema_version": SNAPSHOT_SCHEMA,
            "model_revision": MODEL_REVISION,
            "snapshot_root": str(snapshot.snapshot_root),
            "model_config_sha256": snapshot.model_config["sha256"],
            "tokenizer_json_sha256": snapshot.tokenizer_json["sha256"],
            "tokenizer_config_sha256": snapshot.tokenizer_config["sha256"],
            "weight_index_sha256": snapshot.weight_index["sha256"],
            "shards": [path.name for path in snapshot.shard_paths],
        }
    )
    root = cache_root.resolve() / "model-snapshot-seals"
    seal_path = root / f"{seal_key}.json"
    with _exclusive_lock(root / f".{seal_key}.lock"):
        observed_rows = []
        for path in snapshot.shard_paths:
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise ValueError("offline model shard changed while content hashing")
            observed_rows.append(
                {
                    "name": path.name,
                    "size_bytes": after.st_size,
                    "mtime_ns": after.st_mtime_ns,
                    "sha256": digest,
                }
            )
        if sum(row["size_bytes"] for row in observed_rows) != snapshot.weight_bytes:
            raise ValueError("offline model shard bytes changed before content sealing")
        if not seal_path.is_file():
            body = {
                "schema_version": SNAPSHOT_SCHEMA,
                "seal_key": seal_key,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "snapshot_root": str(snapshot.snapshot_root),
                "model_config": dict(snapshot.model_config),
                "tokenizer_json": dict(snapshot.tokenizer_json),
                "tokenizer_config": dict(snapshot.tokenizer_config),
                "weight_index": dict(snapshot.weight_index),
                "weight_shard_count": snapshot.shard_count,
                "weight_shard_bytes": snapshot.weight_bytes,
                "weight_shards": observed_rows,
            }
            write_immutable_json(seal_path, body)
        seal = load_immutable_json(seal_path)
        if (
            seal.get("schema_version") != SNAPSHOT_SCHEMA
            or seal.get("seal_key") != seal_key
            or seal.get("model_name") != MODEL_NAME
            or seal.get("model_revision") != MODEL_REVISION
            or seal.get("snapshot_root") != str(snapshot.snapshot_root)
            or seal.get("model_config") != snapshot.model_config
            or seal.get("tokenizer_json") != snapshot.tokenizer_json
            or seal.get("tokenizer_config") != snapshot.tokenizer_config
            or seal.get("weight_index") != snapshot.weight_index
            or seal.get("weight_shard_count") != snapshot.shard_count
            or seal.get("weight_shard_bytes") != snapshot.weight_bytes
        ):
            raise ValueError("offline model snapshot content seal differs")
        rows = seal.get("weight_shards")
        if not isinstance(rows, list) or len(rows) != len(snapshot.shard_paths):
            raise ValueError("offline model snapshot shard seal coverage differs")
        for sealed, observed in zip(rows, observed_rows):
            if not isinstance(sealed, Mapping) or dict(sealed) != observed:
                raise ValueError("offline model shard changed after content sealing")
    return OfflineSnapshot(
        cache_root=snapshot.cache_root,
        snapshot_root=snapshot.snapshot_root,
        model_config=snapshot.model_config,
        tokenizer_json=snapshot.tokenizer_json,
        tokenizer_config=snapshot.tokenizer_config,
        weight_index=snapshot.weight_index,
        shard_count=snapshot.shard_count,
        weight_bytes=snapshot.weight_bytes,
        shard_paths=snapshot.shard_paths,
        content_seal=path_identity(seal_path),
    )


def _load_tokenizer(snapshot: OfflineSnapshot, heldout: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=str(snapshot.cache_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    template_path = _verify_identity(
        heldout["chat_template"], label="router driver chat template"
    )
    asset = json.loads(template_path.read_text(encoding="utf-8"))
    expected = str(asset["chat_template"])
    if (
        str(getattr(tokenizer, "chat_template", "") or "") != expected
        or hashlib.sha256(expected.encode("utf-8")).hexdigest()
        != heldout["chat_template_sha256"]
    ):
        raise ValueError("offline tokenizer chat template differs from the sealed asset")
    return tokenizer


def _load_base_model(snapshot: OfflineSnapshot, device: str) -> Any:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from chop.nn.quantized.modules.qwen3_moe.compat import (
        require_qwen3_moe_fused_abi,
    )

    if transformers.__version__ != "5.5.0":
        raise RuntimeError("router driver requires transformers==5.5.0")
    require_qwen3_moe_fused_abi()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=str(snapshot.cache_root),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    _validate_fused_model(model, _TARGET_ARCHITECTURE)
    return model.to(device).eval()


def _validate_fused_model(model: Any, architecture: Mapping[str, Any]) -> None:
    import torch

    if type(model).__name__ != "Qwen3MoeForCausalLM":
        raise TypeError("offline weights did not load Qwen3MoeForCausalLM")
    config = model.config
    for field, expected in architecture.items():
        if getattr(config, field, None) != expected:
            raise ValueError(f"loaded fused model differs at config.{field}")
    layers = tuple(model.model.layers)
    if len(layers) != int(architecture["num_hidden_layers"]):
        raise ValueError("loaded fused model layer count differs")
    hidden = int(architecture["hidden_size"])
    experts = int(architecture["num_experts"])
    moe_hidden = int(architecture["moe_intermediate_size"])
    for index, layer in enumerate(layers):
        fused = layer.mlp.experts
        if (
            tuple(fused.gate_up_proj.shape) != (experts, 2 * moe_hidden, hidden)
            or tuple(fused.down_proj.shape) != (experts, hidden, moe_hidden)
            or tuple(layer.mlp.gate.weight.shape) != (experts, hidden)
        ):
            raise ValueError(f"loaded fused expert ABI differs at layer {index}")
    if (
        tuple(model.lm_head.weight.shape)
        != (int(architecture["vocab_size"]), hidden)
        or model.lm_head.weight.data_ptr()
        == model.model.embed_tokens.weight.data_ptr()
        or str(model.config._attn_implementation) != "eager"
        or any(parameter.dtype != torch.bfloat16 for parameter in model.parameters())
    ):
        raise ValueError("loaded model head/storage/attention ABI differs")


def _render_token_records(
    tokenizer: Any,
    heldout: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    source_path = _verify_identity(
        heldout["dataset_source"], label="router held-out dataset source"
    )
    source = _heldout_source_records(source_path)
    sealed_records = heldout.get("records")
    if not isinstance(sealed_records, list) or len(sealed_records) != len(source):
        raise ValueError("held-out source and manifest record coverage differs")
    target_count = int(heldout["decode_target_tokens_per_record"])
    rows = []
    for sealed, record in zip(sealed_records, source):
        if sealed["prompt_id"] != record["prompt_id"]:
            raise ValueError("held-out source order differs from its manifest")
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = tuple(
            int(token)
            for token in tokenizer(
                rendered, add_special_tokens=False
            )["input_ids"]
        )
        target_ids = tuple(
            int(token)
            for token in tokenizer(
                record["continuation"], add_special_tokens=False
            )["input_ids"]
        )
        if target_count < 2:
            raise ValueError(
                "held-out continuation must include the prefill-owned input token "
                "and at least one scored decode token"
            )
        if not prompt_ids or len(target_ids) < target_count:
            raise ValueError("held-out record lacks the sealed token coverage")
        rows.append(
            {
                "prompt_id": record["prompt_id"],
                "prompt_ids": prompt_ids,
                "target_ids": target_ids[:target_count],
            }
        )
    return tuple(rows)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_immutable_safetensors(path: Path, tensors: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        save_file(dict(tensors), temporary)
        if path.exists():
            if sha256_file(Path(temporary)) != sha256_file(path):
                raise FileExistsError(f"immutable tensor artifact differs: {path}")
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if sha256_file(Path(temporary)) != sha256_file(path):
                    raise FileExistsError(
                        f"immutable tensor artifact differs: {path}"
                    )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cache_layers(cache: Any) -> tuple[tuple[Any, Any], ...]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("Transformers 5.5 cache does not expose fused layers")
    values = []
    for layer in layers:
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if key is None or value is None:
            raise ValueError("BF16 prefill cache contains an empty layer")
        values.append((key, value))
    return tuple(values)


def _build_prefill_cache(
    *,
    model: Any,
    token_records: Sequence[Mapping[str, Any]],
    heldout: Mapping[str, Any],
    snapshot: OfflineSnapshot,
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    cache_key = canonical_hash(
        {
            "schema_version": PREFILL_CACHE_SCHEMA,
            "model_revision": MODEL_REVISION,
            "heldout_manifest_hash": heldout["content_hash"],
            "model_config_sha256": snapshot.model_config["sha256"],
            "tokenizer_json_sha256": snapshot.tokenizer_json["sha256"],
            "tokenizer_config_sha256": snapshot.tokenizer_config["sha256"],
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "prefill_dtype": "BF16",
            "continuation_decode_contract": _CONTINUATION_DECODE_CONTRACT,
        }
    )
    root = cache_root.resolve() / cache_key
    index_path = root / "index.json"
    with _exclusive_lock(cache_root.resolve() / f".{cache_key}.lock"):
        if index_path.is_file():
            index = load_immutable_json(index_path)
            _validate_prefill_index(
                index,
                heldout=heldout,
                cache_key=cache_key,
                token_records=token_records,
            )
            return index, _prefill_cache_binding(root, index_path, index)
        records = []
        with torch.inference_mode():
            for record in token_records:
                prompt_ids = tuple(record["prompt_ids"])
                input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
                )
                output = model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    return_dict=True,
                )
                target_ids = tuple(int(token) for token in record["target_ids"])
                if len(target_ids) < 2:
                    raise ValueError("BF16 prefill record lacks a scored decode suffix")
                first_token = target_ids[0]
                layers = _cache_layers(output.past_key_values)
                if len(layers) != EXPECTED_LAYERS:
                    raise ValueError("BF16 prefill cache layer coverage differs")
                tensors = {}
                for layer_index, (key, value) in enumerate(layers):
                    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
                        raise ValueError("BF16 prefill cache storage differs")
                    tensors[f"layer_{layer_index:02d}_key"] = (
                        key.detach().to("cpu").contiguous()
                    )
                    tensors[f"layer_{layer_index:02d}_value"] = (
                        value.detach().to("cpu").contiguous()
                    )
                token = hashlib.sha256(
                    str(record["prompt_id"]).encode("utf-8")
                ).hexdigest()[:20]
                tensor_path = root / "records" / f"{token}.safetensors"
                _write_immutable_safetensors(tensor_path, tensors)
                metadata_body = {
                    "schema_version": PREFILL_RECORD_SCHEMA,
                    "cache_key": cache_key,
                    "prompt_id": record["prompt_id"],
                    "prompt_ids": list(prompt_ids),
                    "target_ids": list(target_ids),
                    "first_token_id": first_token,
                    "first_token_source": "sealed_continuation_token_0",
                    "scored_suffix_start_index": 1,
                    "prompt_tokens": len(prompt_ids),
                    "layers": EXPECTED_LAYERS,
                    "cache_tensors": path_identity(tensor_path),
                }
                metadata_path = root / "records" / f"{token}.json"
                write_immutable_json(metadata_path, metadata_body)
                records.append(path_identity(metadata_path))
                del output, input_ids, tensors
        body = {
            "schema_version": PREFILL_CACHE_SCHEMA,
            "cache_key": cache_key,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "heldout_manifest_hash": heldout["content_hash"],
            "prefill_dtype": "BF16",
            "serving_first_token_owner": "prefill",
            "evaluation_first_decode_input_source": (
                "teacher_forced_continuation_token_0"
            ),
            "evaluation_input_generated_by_bf16_model": False,
            "prefill_lm_head_executed": False,
            "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
            "record_count": len(records),
            "records": records,
        }
        write_immutable_json(index_path, body)
        index = load_immutable_json(index_path)
        _validate_prefill_index(
            index,
            heldout=heldout,
            cache_key=cache_key,
            token_records=token_records,
        )
        return index, _prefill_cache_binding(root, index_path, index)


def _prefill_cache_binding(
    root: Path, index_path: Path, index: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "cache_key": index["cache_key"],
        "index_content_hash": index["content_hash"],
        "index": path_identity(index_path),
        "artifact_tree": path_identity(root),
    }


def _validate_prefill_cache_binding(
    binding: Mapping[str, Any], *, heldout_manifest_hash: str
) -> dict[str, Any]:
    if set(binding) != {
        "cache_key",
        "index_content_hash",
        "index",
        "artifact_tree",
    }:
        raise ValueError("BF16 prefill cache binding fields differ")
    index_identity = binding.get("index")
    tree_identity = binding.get("artifact_tree")
    if not isinstance(index_identity, Mapping) or not isinstance(
        tree_identity, Mapping
    ):
        raise TypeError("BF16 prefill cache binding identities are invalid")
    index_path = _verify_identity(index_identity, label="BF16 prefill cache index")
    tree_path = _verify_identity(tree_identity, label="BF16 prefill cache tree")
    if index_path.parent != tree_path:
        raise ValueError("BF16 prefill cache index is outside its sealed tree")
    index = load_immutable_json(index_path)
    if (
        index.get("schema_version") != PREFILL_CACHE_SCHEMA
        or index.get("cache_key") != binding.get("cache_key")
        or index.get("content_hash") != binding.get("index_content_hash")
        or index.get("heldout_manifest_hash") != heldout_manifest_hash
        or index.get("serving_first_token_owner") != "prefill"
        or index.get("evaluation_first_decode_input_source")
        != "teacher_forced_continuation_token_0"
        or index.get("evaluation_input_generated_by_bf16_model") is not False
        or index.get("prefill_lm_head_executed") is not False
        or index.get("continuation_decode_contract")
        != _CONTINUATION_DECODE_CONTRACT
    ):
        raise ValueError("BF16 prefill cache binding differs from its index")
    return index


def _validate_prefill_index(
    index: Mapping[str, Any],
    *,
    heldout: Mapping[str, Any],
    cache_key: str,
    token_records: Sequence[Mapping[str, Any]],
) -> None:
    if (
        index.get("schema_version") != PREFILL_CACHE_SCHEMA
        or index.get("cache_key") != cache_key
        or index.get("model_name") != MODEL_NAME
        or index.get("model_revision") != MODEL_REVISION
        or index.get("heldout_manifest_hash") != heldout["content_hash"]
        or index.get("prefill_dtype") != "BF16"
        or index.get("serving_first_token_owner") != "prefill"
        or index.get("evaluation_first_decode_input_source")
        != "teacher_forced_continuation_token_0"
        or index.get("evaluation_input_generated_by_bf16_model") is not False
        or index.get("prefill_lm_head_executed") is not False
        or index.get("continuation_decode_contract")
        != _CONTINUATION_DECODE_CONTRACT
        or index.get("record_count") != len(heldout["records"])
        or len(token_records) != len(heldout["records"])
    ):
        raise ValueError("BF16 prefill cache index differs")
    records = index.get("records")
    if not isinstance(records, list) or len(records) != len(heldout["records"]):
        raise ValueError("BF16 prefill cache record coverage differs")
    for sealed, expected, identity in zip(
        heldout["records"], token_records, records
    ):
        if not isinstance(identity, Mapping):
            raise TypeError("BF16 prefill record identity is invalid")
        metadata_path = _verify_identity(identity, label="BF16 prefill record")
        metadata = load_immutable_json(metadata_path)
        tensors = metadata.get("cache_tensors")
        if (
            metadata.get("schema_version") != PREFILL_RECORD_SCHEMA
            or metadata.get("cache_key") != cache_key
            or metadata.get("prompt_id") != sealed["prompt_id"]
            or metadata.get("prompt_id") != expected["prompt_id"]
            or metadata.get("layers") != EXPECTED_LAYERS
            or metadata.get("prompt_ids") != list(expected["prompt_ids"])
            or metadata.get("target_ids") != list(expected["target_ids"])
            or len(metadata.get("target_ids", ())) < 2
            or metadata.get("first_token_id") != expected["target_ids"][0]
            or metadata.get("first_token_source")
            != "sealed_continuation_token_0"
            or metadata.get("scored_suffix_start_index") != 1
            or metadata.get("prompt_tokens") != len(expected["prompt_ids"])
            or not isinstance(tensors, Mapping)
        ):
            raise ValueError("BF16 prefill record metadata differs")
        _verify_identity(tensors, label="BF16 prefill cache tensors")


def _load_prefill_records(index: Mapping[str, Any]) -> tuple[HeldoutTokens, ...]:
    from safetensors.torch import load_file

    records = []
    for identity in index["records"]:
        metadata_path = _verify_identity(identity, label="BF16 prefill record")
        metadata = load_immutable_json(metadata_path)
        tensor_path = _verify_identity(
            metadata["cache_tensors"], label="BF16 prefill cache tensors"
        )
        tensors = load_file(str(tensor_path), device="cpu")
        target_ids = tuple(int(token) for token in metadata["target_ids"])
        first_token_id = int(metadata["first_token_id"])
        if len(target_ids) < 2 or first_token_id != target_ids[0]:
            raise ValueError("BF16 prefill record decode boundary differs")
        records.append(
            HeldoutTokens(
                prompt_id=str(metadata["prompt_id"]),
                prompt_ids=tuple(int(token) for token in metadata["prompt_ids"]),
                target_ids=target_ids,
                first_token_id=first_token_id,
                cache_tensors=tensors,
            )
        )
    return tuple(records)


def _build_body_bank(
    model: Any,
    *,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    device: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    from chop.passes.module.transforms.quantize.quantize import (
        install_phase_context_pre_hooks,
        quantize_module_transform_pass,
    )
    from decode_dse.software.decode_evaluator import (
        DecodeWeightBankIdentity,
        DecodeWeightQuantizationGuard,
        _validate_bank_structure,
        build_decode_binding_plan,
    )
    from decode_dse.software.precision_bindings import (
        build_decode_pass_args,
        decode_binding_expectations,
    )

    profile = DecodePrecisionProfile.from_dict(job["body_profile"])
    spec = profile_to_decode_quant_spec(profile)
    if spec is None or spec.gptq_weights or spec.use_rotation:
        raise ValueError("router driver body must use canonical RTN reconstruction")
    bank = job["body_bank"]
    if bank.get("materialization") != "rebuild_from_sealed_manifest_profile/v1":
        raise ValueError("router driver does not guess an unknown serialized bank ABI")
    current_mase_hash = _mase_tree_hash(_software_root(), config)
    if current_mase_hash != bank["source_mase_tree_sha256"]:
        raise ValueError("live MASE tree differs from the successful body-bank source")
    pass_args = build_decode_pass_args(MODEL_NAME, device, spec)
    pass_args["collapse_decode_banks"] = True
    variable = "MASE_PHASE_BANK_DEVICE"
    previous = os.environ.get(variable)
    try:
        os.environ[variable] = device
        model, _ = quantize_module_transform_pass(model, pass_args)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    model = model.to(device).eval()
    install_phase_context_pre_hooks(model)
    binding_plan = build_decode_binding_plan(model, pass_args)
    expected = decode_binding_expectations(dict(_TARGET_ARCHITECTURE))
    quantization_guard = DecodeWeightQuantizationGuard.capture(
        binding_plan, expected_modules=expected.sealed_weight_modules
    )
    _validate_bank_structure(
        model,
        binding_plan,
        len(quantization_guard.modules),
        _TARGET_ARCHITECTURE,
    )
    identity = DecodeWeightBankIdentity.capture(model)
    if identity.structure_fingerprint != bank[
        "source_weight_bank_structure_fingerprint"
    ]:
        raise ValueError("rebuilt body-bank structure differs from its successful source")
    receipt = {
        "materialization": bank["materialization"],
        "source_structure_fingerprint": bank[
            "source_weight_bank_structure_fingerprint"
        ],
        "rebuilt_structure_fingerprint": identity.structure_fingerprint,
        "source_process_identity_fingerprint": bank[
            "source_weight_bank_identity_fingerprint"
        ],
        "rebuilt_process_identity_fingerprint": identity.fingerprint,
        "process_identity_comparison": "not_comparable_across_processes",
        "mase_tree_sha256": current_mase_hash,
        "quantizer_provenance_hash": job["quantizer_provenance_hash"],
        "weight_quantization_events": quantization_guard.verify(),
    }
    return model, identity, quantization_guard, receipt


def _nonrouter_parameter_identity(model: Any) -> tuple[tuple[Any, ...], ...]:
    rows = []
    for name, parameter in model.named_parameters():
        if name.endswith(".mlp.gate.weight"):
            continue
        rows.append(
            (
                name,
                id(parameter),
                int(parameter.data_ptr()),
                int(getattr(parameter, "_version", 0)),
                tuple(parameter.shape),
                str(parameter.dtype),
                str(parameter.device),
            )
        )
    return tuple(rows)


def _build_router_candidates(
    model: Any,
    variant: Mapping[str, Any],
    *,
    expected_layers: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from chop.passes.module.transforms.quantize.quantize import (
        quantize_module_transform_pass,
    )
    from chop.nn.quantized.modules.qwen3_moe import Qwen3MoeTopKRouterMX
    from chop.nn.quantized.modules.qwen3_moe.router import (
        Qwen3MoeTopKRouterBF16,
    )

    parents = tuple(model.model.layers[index].mlp for index in range(expected_layers))
    baseline = tuple(parent.gate for parent in parents)
    if any(not isinstance(router, Qwen3MoeTopKRouterBF16) for router in baseline):
        raise TypeError("rebuilt body does not retain the BF16 router safety island")
    nonrouter_before = _nonrouter_parameter_identity(model)
    head_before = (id(model.lm_head), id(model.lm_head.weight), model.lm_head.weight.data_ptr())
    model, _ = quantize_module_transform_pass(
        model, build_router_variant_pass_args(variant)
    )
    candidates = tuple(parent.gate for parent in parents)
    if (
        len(candidates) != expected_layers
        or any(not isinstance(router, Qwen3MoeTopKRouterMX) for router in candidates)
        or any(
            router.router_precision_contract["decode_router_weight_format"]
            != variant["weight_format"]
            or router.router_precision_contract["decode_router_activation_format"]
            != variant["activation_format"]
            or router.router_precision_contract["matrix_mlen"]
            != variant["matrix_mlen"]
            for router in candidates
        )
        or _nonrouter_parameter_identity(model) != nonrouter_before
        or head_before
        != (id(model.lm_head), id(model.lm_head.weight), model.lm_head.weight.data_ptr())
    ):
        raise RuntimeError("router variant replacement escaped its gate-only boundary")
    for parent, router in zip(parents, baseline):
        parent.gate = router
    if _nonrouter_parameter_identity(model) != nonrouter_before:
        raise RuntimeError("restoring BF16 routers changed the model body")
    return baseline, candidates


def _install_routers(model: Any, routers: Sequence[Any]) -> None:
    if len(routers) != len(model.model.layers):
        raise ValueError("router replacement layer count differs")
    for layer, router in zip(model.model.layers, routers):
        layer.mlp.gate = router


def _admitted_cache(model: Any, record: HeldoutTokens, device: str) -> Any:
    from transformers import DynamicCache
    from chop.nn.quantized.functional.kvcache import kv_cache_mx

    cache = DynamicCache()
    for layer_index, layer in enumerate(model.model.layers):
        key_name = f"layer_{layer_index:02d}_key"
        value_name = f"layer_{layer_index:02d}_value"
        key = record.cache_tensors[key_name].to(device)
        value = record.cache_tensors[value_name].to(device)
        attention = layer.self_attn
        stage = attention._phase_stage_cfgs["decode"]
        if stage["kv_cache_bypass"]:
            raise ValueError("quantized body unexpectedly bypasses decode KV admission")
        key, value = kv_cache_mx(key, value, stage["kv_cache_config"])
        cache.update(key, value, layer_index)
    if cache.get_seq_length() != len(record.prompt_ids):
        raise ValueError("admitted KV cache length differs from its prompt")
    return cache


class _StreamingRouterComparator:
    def __init__(
        self,
        candidates: Sequence[Any],
        *,
        heldout_manifest_hash: str,
        layers: int,
        experts: int,
        top_k: int,
    ) -> None:
        self.candidates = tuple(candidates)
        self.accumulator = RouterAgreementAccumulator(
            heldout_manifest_hash=heldout_manifest_hash,
            shadow_router_input_hash="0" * 64,
            layers=layers,
            experts=experts,
            top_k=top_k,
        )
        self.digest = hashlib.sha256(SHADOW_ROUTER_INPUT_HASH_DOMAIN)
        self.handles: list[Any] = []

    def begin_step(self, prompt_id: str, step_index: int, input_token: int) -> None:
        payload = json.dumps(
            [prompt_id, step_index, input_token],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        self.digest.update(len(payload).to_bytes(8, "little"))
        self.digest.update(payload)

    def _hook(self, layer_index: int):
        import torch
        from chop.nn.quantized.modules.phase_context import force_runtime_phase

        def compare(module: Any, args: tuple[Any, ...], output: Any):
            if len(args) != 1:
                raise RuntimeError("router hook observed an unsupported input ABI")
            hidden = args[0].detach()
            raw = hidden.contiguous().view(torch.int16).cpu().numpy().tobytes()
            self.digest.update(layer_index.to_bytes(4, "little"))
            self.digest.update(len(raw).to_bytes(8, "little"))
            self.digest.update(raw)
            with force_runtime_phase("decode"):
                candidate = self.candidates[layer_index](hidden)
            baseline_probs, _, baseline_indices = output
            candidate_probs, _, candidate_indices = candidate
            self.accumulator.update(
                layer_index,
                baseline_probs,
                baseline_indices,
                candidate_probs,
                candidate_indices,
            )

        return compare

    def install(self, baseline: Sequence[Any]) -> None:
        if self.handles:
            raise RuntimeError("router comparison hooks are already installed")
        self.handles = [
            router.register_forward_hook(self._hook(index))
            for index, router in enumerate(baseline)
        ]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self) -> dict[str, Any]:
        if self.handles:
            raise RuntimeError("router comparison hooks must be removed first")
        self.accumulator.bind_shadow_router_input_hash(self.digest.hexdigest())
        value = self.accumulator.finalize()
        if value["schema_version"] != AGREEMENT_SCHEMA:
            raise AssertionError("router agreement schema differs")
        return value


def _score_decode_arm(
    model: Any,
    records: Sequence[HeldoutTokens],
    *,
    device: str,
    comparator: _StreamingRouterComparator | None,
) -> tuple[float, int, dict[str, str]]:
    import torch
    import torch.nn.functional as F

    nll_sum = 0.0
    token_count = 0
    token_stream = hashlib.sha256(TEACHER_FORCED_TOKEN_HASH_DOMAIN)
    first_decode_inputs = hashlib.sha256(FIRST_DECODE_INPUT_HASH_DOMAIN)
    scored_decode_suffix = hashlib.sha256(SCORED_DECODE_SUFFIX_HASH_DOMAIN)
    with torch.inference_mode():
        for record_index, record in enumerate(records):
            if len(record.target_ids) < 2:
                raise ValueError("held-out continuation lacks a scored decode suffix")
            if record.first_token_id != record.target_ids[0]:
                raise ValueError(
                    "prefill-owned decode input differs from continuation token 0"
                )
            first_input_payload = json.dumps(
                [record_index, record.prompt_id, record.first_token_id],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            first_decode_inputs.update(
                len(first_input_payload).to_bytes(8, "little")
            )
            first_decode_inputs.update(first_input_payload)
            suffix_payload = json.dumps(
                [record_index, record.prompt_id, list(record.target_ids[1:])],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            scored_decode_suffix.update(len(suffix_payload).to_bytes(8, "little"))
            scored_decode_suffix.update(suffix_payload)
            record_payload = json.dumps(
                [
                    record_index,
                    record.prompt_id,
                    list(record.prompt_ids),
                    record.first_token_id,
                    list(record.target_ids),
                ],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            token_stream.update(len(record_payload).to_bytes(8, "little"))
            token_stream.update(record_payload)
            cache = _admitted_cache(model, record, device)
            input_token = record.first_token_id
            prompt_tokens = len(record.prompt_ids)
            for step_index, target_token in enumerate(record.target_ids[1:]):
                step_payload = json.dumps(
                    [record_index, step_index, input_token, target_token],
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
                token_stream.update(len(step_payload).to_bytes(8, "little"))
                token_stream.update(step_payload)
                if comparator is not None:
                    comparator.begin_step(record.prompt_id, step_index, input_token)
                cache_position = prompt_tokens + step_index
                output = model(
                    input_ids=torch.tensor(
                        [[input_token]], dtype=torch.long, device=device
                    ),
                    attention_mask=torch.ones(
                        (1, cache_position + 1), dtype=torch.long, device=device
                    ),
                    position_ids=torch.tensor(
                        [[cache_position]], dtype=torch.long, device=device
                    ),
                    cache_position=torch.tensor(
                        [cache_position], dtype=torch.long, device=device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                if output.past_key_values.get_seq_length() != cache_position + 1:
                    raise RuntimeError("q_len=1 decode did not append exactly one KV row")
                logits = output.logits[0, -1].to(torch.float32)
                loss = F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([target_token], dtype=torch.long, device=device),
                    reduction="sum",
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("router ablation produced non-finite NLL")
                nll_sum += float(loss.item())
                token_count += 1
                input_token = target_token
                cache = output.past_key_values
            del cache
    if token_count <= 0:
        raise RuntimeError("router ablation scored no held-out tokens")
    return nll_sum, token_count, {
        "teacher_forced_token_stream_hash": token_stream.hexdigest(),
        "first_decode_input_stream_hash": first_decode_inputs.hexdigest(),
        "scored_decode_suffix_stream_hash": scored_decode_suffix.hexdigest(),
    }


def measure_variant(
    model: Any,
    records: Sequence[HeldoutTokens],
    *,
    job: Mapping[str, Any],
    heldout: Mapping[str, Any],
    device: str,
    expected_layers: int = EXPECTED_LAYERS,
    expected_experts: int = EXPECTED_EXPERTS,
    expected_top_k: int = EXPECTED_TOP_K,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline, candidates = _build_router_candidates(
        model, job["variant"], expected_layers=expected_layers
    )
    comparator = _StreamingRouterComparator(
        candidates,
        heldout_manifest_hash=heldout["content_hash"],
        layers=expected_layers,
        experts=expected_experts,
        top_k=expected_top_k,
    )
    comparator.install(baseline)
    try:
        baseline_sum, baseline_tokens, baseline_stream_hashes = _score_decode_arm(
            model, records, device=device, comparator=comparator
        )
    finally:
        comparator.remove()
    agreement = comparator.finalize()
    _install_routers(model, candidates)
    try:
        candidate_sum, candidate_tokens, candidate_stream_hashes = _score_decode_arm(
            model, records, device=device, comparator=None
        )
    finally:
        _install_routers(model, baseline)
    if (
        candidate_tokens != baseline_tokens
        or candidate_stream_hashes != baseline_stream_hashes
    ):
        raise RuntimeError("paired router arms scored different teacher-forced streams")
    baseline_nll = baseline_sum / baseline_tokens
    candidate_nll = candidate_sum / candidate_tokens
    end_to_end = {
        "paired_bf16_reexecuted": True,
        "baseline_router": dict(_BF16_ROUTER_CONTRACT),
        "heldout_manifest_hash": heldout["content_hash"],
        **baseline_stream_hashes,
        "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
        "bf16_router_baseline": {
            "mean_token_nll": baseline_nll,
            "token_count": baseline_tokens,
            **baseline_stream_hashes,
        },
        "mx_router_candidate": {
            "mean_token_nll": candidate_nll,
            "token_count": candidate_tokens,
            **candidate_stream_hashes,
        },
        "mean_token_nll_delta": candidate_nll - baseline_nll,
        "task_effects": {
            "status": "unsupported",
            "reason": (
                "router-only tasks are not duplicated here; IFEval and GSM8K "
                "remain in the separately sealed native accuracy campaign"
            ),
        },
    }
    return agreement, end_to_end


def _execution_receipt(
    *,
    request: Mapping[str, Any],
    job: Mapping[str, Any],
    heldout: Mapping[str, Any],
    snapshot: OfflineSnapshot,
    body_receipt: Mapping[str, Any],
    records: Sequence[HeldoutTokens],
    body_unchanged: bool,
    prefill_cache_binding: Mapping[str, Any],
    end_to_end: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    import transformers

    profile = DecodePrecisionProfile.from_dict(job["body_profile"])
    _validate_prefill_cache_binding(
        prefill_cache_binding, heldout_manifest_hash=heldout["content_hash"]
    )
    body = {
        "schema_version": DRIVER_RECEIPT_SCHEMA,
        "offline_local_files_only": True,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "fused_model_class": "Qwen3MoeForCausalLM",
        "model_revision": MODEL_REVISION,
        "model_snapshot_revision_verified": MODEL_REVISION in snapshot.snapshot_root.parts,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_snapshot_revision_verified": (
            MODEL_REVISION in snapshot.snapshot_root.parts
        ),
        "model_config": dict(snapshot.model_config),
        "tokenizer_json": dict(snapshot.tokenizer_json),
        "tokenizer_config": dict(snapshot.tokenizer_config),
        "weight_index": dict(snapshot.weight_index),
        "model_snapshot_content_seal": dict(snapshot.content_seal or {}),
        "weight_shard_count": snapshot.shard_count,
        "weight_shard_bytes": snapshot.weight_bytes,
        "driver_source": dict(request["driver"]),
        "chat_template": dict(heldout["chat_template"]),
        "heldout_manifest_hash": heldout["content_hash"],
        "heldout_counts": {
            "records": len(records),
            "continuation_tokens_per_arm": sum(
                len(row.target_ids) for row in records
            ),
            "prefill_owned_input_tokens_per_arm": len(records),
            "scored_tokens_per_arm": sum(
                len(row.target_ids) - 1 for row in records
            ),
        },
        "prefill_cache_binding": dict(prefill_cache_binding),
        "body_profile_id": profile.profile_id,
        "body_bank_ancestry_verified": True,
        "body_weight_bank_structure_verified": (
            body_receipt["source_structure_fingerprint"]
            == body_receipt["rebuilt_structure_fingerprint"]
        ),
        "body_nonrouter_parameters_unchanged": body_unchanged,
        "body_bank_rebuild": dict(body_receipt),
        "router_target_pattern": "model.layers.<index>.mlp.gate",
        "router_target_count": EXPECTED_LAYERS,
        "only_router_modules_replaced": True,
        "router_variant_id": job["variant"]["variant_id"],
        "prefill_owner": "external_bf16_source_model",
        "serving_first_token_owner": "prefill",
        "evaluation_first_decode_input_source": (
            "teacher_forced_continuation_token_0"
        ),
        "evaluation_input_generated_by_bf16_model": False,
        "prefill_lm_head_executed": False,
        "continuation_decode_contract": dict(_CONTINUATION_DECODE_CONTRACT),
        "first_decode_input_stream_hash": end_to_end[
            "first_decode_input_stream_hash"
        ],
        "scored_decode_suffix_stream_hash": end_to_end[
            "scored_decode_suffix_stream_hash"
        ],
        "teacher_forced_token_stream_hash": end_to_end[
            "teacher_forced_token_stream_hash"
        ],
        "decode_query_length": 1,
        "decode_cache_admission": "quantize_bf16_prefill_once_per_arm",
        "key_format": profile.key_format,
        "value_format": profile.value_format,
        "local_decode_head_unchanged": True,
        "task_effects": "unsupported_use_separate_native_accuracy_campaign",
        "classification": _classification(),
    }
    return body | {"content_hash": canonical_hash(body)}


def _success_result(
    request: Mapping[str, Any],
    *,
    agreement: Mapping[str, Any],
    end_to_end: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    job = request["job"]
    body = {
        "schema_version": RESULT_SCHEMA,
        "plan_hash": request["plan_hash"],
        "request_hash": request["content_hash"],
        "job_id": job["job_id"],
        "body_profile_id": job["body_profile_id"],
        "variant_id": job["variant"]["variant_id"],
        "status": "success",
        "measurements": {
            "execution_receipt": dict(receipt),
            "router_agreement": dict(agreement),
            "end_to_end": dict(end_to_end),
        },
        "failure": None,
        "classification": _classification(),
    }
    return body | {"content_hash": canonical_hash(body)}


def _failure_result(
    request: Mapping[str, Any], *, status: str, exc: BaseException, phase: str
) -> dict[str, Any]:
    job = request["job"]
    body = {
        "schema_version": RESULT_SCHEMA,
        "plan_hash": request["plan_hash"],
        "request_hash": request["content_hash"],
        "job_id": job["job_id"],
        "body_profile_id": job["body_profile_id"],
        "variant_id": job["variant"]["variant_id"],
        "status": status,
        "measurements": None,
        "failure": {
            "message": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
            "phase": phase,
        },
        "classification": _classification(),
    }
    return body | {"content_hash": canonical_hash(body)}


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).casefold()


def _write_terminal(path: Path, result: Mapping[str, Any]) -> None:
    write_immutable_json(path.resolve(), result)


def execute_requests(
    items: Sequence[tuple[Path, Path]],
    *,
    model_cache: Path,
    cache_root: Path,
    device: str,
    driver_path: Path,
) -> None:
    import torch

    requests: list[tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]] = []
    for request_path, output_path in items:
        request = load_immutable_json(request_path.resolve())
        job, config, heldout = validate_request(
            request, driver_path=driver_path.resolve()
        )
        if output_path.exists():
            existing = load_immutable_json(output_path)
            if existing.get("request_hash") != request["content_hash"]:
                raise FileExistsError("existing driver output belongs to another request")
            continue
        requests.append((request, output_path.resolve(), config, heldout))
    if not requests:
        return
    snapshot = _seal_offline_snapshot(
        _resolve_offline_snapshot(model_cache), cache_root
    )
    tokenizer = _load_tokenizer(snapshot, requests[0][3])
    token_records = _render_token_records(tokenizer, requests[0][3])
    groups: dict[str, list[tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in requests:
        request, _, config, heldout = item
        if config != requests[0][2] or heldout != requests[0][3]:
            raise ValueError("batched router requests do not share sealed inputs")
        groups[str(request["job"]["body_profile_id"])].append(item)
    shared_records: tuple[HeldoutTokens, ...] | None = None
    shared_prefill_binding: dict[str, Any] | None = None
    for profile_id in sorted(groups):
        group = sorted(groups[profile_id], key=lambda item: item[0]["job"]["variant_id"])
        model = None
        phase = "load_offline_model"
        try:
            model = _load_base_model(snapshot, device)
            phase = "materialize_bf16_prefill"
            if shared_records is None:
                prefill_index, shared_prefill_binding = _build_prefill_cache(
                    model=model,
                    token_records=token_records,
                    heldout=group[0][3],
                    snapshot=snapshot,
                    cache_root=cache_root,
                )
                shared_records = _load_prefill_records(prefill_index)
            records = shared_records
            if shared_prefill_binding is None:
                raise RuntimeError("BF16 prefill cache binding is unavailable")
            phase = "rebuild_body_bank"
            model, identity, quantization_guard, body_receipt = _build_body_bank(
                model,
                config=group[0][2],
                job=group[0][0]["job"],
                device=device,
            )
            nonrouter_identity = _nonrouter_parameter_identity(model)
            head_identity = (
                id(model.lm_head),
                id(model.lm_head.weight),
                model.lm_head.weight.data_ptr(),
            )
        except BaseException as exc:
            traceback.print_exc()
            status = "oom" if _is_oom(exc) else "failed"
            for request, output, _, _ in group:
                _write_terminal(
                    output,
                    _failure_result(request, status=status, exc=exc, phase=phase),
                )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        for request, output, _, heldout in group:
            phase = "paired_router_measurement"
            try:
                agreement, end_to_end = measure_variant(
                    model,
                    records,
                    job=request["job"],
                    heldout=heldout,
                    device=device,
                )
                body_unchanged = (
                    _nonrouter_parameter_identity(model) == nonrouter_identity
                    and head_identity
                    == (
                        id(model.lm_head),
                        id(model.lm_head.weight),
                        model.lm_head.weight.data_ptr(),
                    )
                    and identity.verify(model) == identity.fingerprint
                    and quantization_guard.verify()
                    == body_receipt["weight_quantization_events"]
                )
                if not body_unchanged:
                    raise RuntimeError("router measurement changed the body bank")
                receipt = _execution_receipt(
                    request=request,
                    job=request["job"],
                    heldout=heldout,
                    snapshot=snapshot,
                    body_receipt=body_receipt,
                    records=records,
                    body_unchanged=body_unchanged,
                    prefill_cache_binding=shared_prefill_binding,
                    end_to_end=end_to_end,
                )
                result = _success_result(
                    request,
                    agreement=agreement,
                    end_to_end=end_to_end,
                    receipt=receipt,
                )
            except BaseException as exc:
                traceback.print_exc()
                result = _failure_result(
                    request,
                    status="oom" if _is_oom(exc) else "failed",
                    exc=exc,
                    phase=phase,
                )
            _write_terminal(output, result)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_batch_manifest(path: Path) -> tuple[tuple[Path, Path], ...]:
    batch = load_immutable_json(path.resolve())
    body = {key: item for key, item in batch.items() if key != "content_hash"}
    if (
        batch.get("schema_version") != DRIVER_BATCH_SCHEMA
        or batch.get("content_hash") != canonical_hash(body)
        or batch.get("driver") != path_identity(Path(__file__).resolve())
    ):
        raise ValueError("router driver batch manifest differs")
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("router driver batch contains no jobs")
    items = []
    seen: set[str] = set()
    for row in jobs:
        if not isinstance(row, Mapping) or str(row.get("job_id", "")) in seen:
            raise ValueError("router driver batch job identity differs")
        request_path = _verify_identity(
            row.get("request", {}), label="router batch request"
        )
        output = Path(str(row.get("output", "")))
        if not output.is_absolute():
            raise ValueError("router batch output path must be absolute")
        request = load_immutable_json(request_path)
        if (
            request.get("job", {}).get("job_id") != row.get("job_id")
            or request.get("job", {}).get("body_profile_id")
            != row.get("body_profile_id")
            or request.get("plan_hash") != batch.get("plan_hash")
        ):
            raise ValueError("router batch row differs from its request")
        seen.add(str(row["job_id"]))
        items.append((request_path, output))
    return tuple(items)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=None,
        help="offline Hugging Face cache containing the pinned commit",
    )
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.request is not None:
        if args.output is None:
            parser.error("--output is required with --request")
        items = ((args.request, args.output),)
    else:
        if args.output is not None:
            parser.error("--output is not accepted with --batch-manifest")
        items = _load_batch_manifest(args.batch_manifest)
    model_cache = args.model_cache or (
        Path(os.environ["PLENA_ROUTER_MODEL_CACHE"])
        if "PLENA_ROUTER_MODEL_CACHE" in os.environ
        else None
    )
    if model_cache is None:
        raise ValueError(
            "supply --model-cache or PLENA_ROUTER_MODEL_CACHE; network fallback is forbidden"
        )
    cache_root = args.cache_root or Path(
        os.environ.get("PLENA_ROUTER_DRIVER_CACHE", "router_precision_driver_cache")
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    execute_requests(
        items,
        model_cache=model_cache,
        cache_root=cache_root,
        device=args.device,
        driver_path=Path(__file__).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
