"""Prepare, launch, and gate the staged exhaustive decode sweep.

Commands: `inputs` builds tokens and reusable BF16 prefill artifacts;
`compiler-trace-artifacts` builds compact native timing evidence; `stage`
drives one workspace stage; `shards` fans a stage across isolated devices;
`pipeline` runs every stage through its gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from decode_dse.legality import load_built_stack_validity
from decode_dse.manifest import (
    SweepManifest,
    build_exhaustive_manifest,
    load_manifest,
    validate_sweep_config,
    write_manifest,
)
from decode_dse.profiles import DECODE_FORMATS, declared_search_space
from decode_dse.software.cache_artifacts import (
    ArtifactProvenance,
    load_prefill_artifact,
    save_prefill_artifact,
)
from decode_dse.software.cached_decode import capture_bf16_prefill
from decode_dse.software.decode_evaluator import DecodeEvaluator
from decode_dse.software.runtime_environment import (
    RuntimeEnvironment,
    capture_runtime_environment,
    initialize_numerical_runtime,
    require_runtime_environment,
    run_launch_preflight,
)
from decode_dse.software.sweep_plan import (
    ExecutorContext,
    GPUBaselinePlan,
    HARDWARE_VALIDATION_SAMPLE_CONTRACT,
    NUMERICAL_SCREEN_SAMPLE_CONTRACT,
    PromptManifest,
    StageSampleContract,
    SweepRunPlan,
    _mase_tree_hash,
    _software_tree_hash,
    build_quantizer_provenance,
    build_run_plan,
    evaluate_preflight_gates,
    load_immutable_json,
    load_preflight_evidence,
    load_prompt_manifest,
    make_stage_manifest,
    profile_to_decode_quant_spec,
    resolve_bound_path,
    validate_run_plan,
    write_immutable_json,
)
from decode_dse.software.sweep_runner import (
    ExhaustiveSweepExecutor,
    ExhaustiveSweepRunner,
    SweepRunSummary,
)
from decode_dse.software.token_samples import (
    TokenSampleBundle,
    TokenizedSourceDocument,
    build_bundle_from_documents,
    load_sample_bundle,
    save_sample_bundle,
)

PREFILL_INDEX_SCHEMA = "decode-prefill-index"


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain a JSON object")
    return value


def _repository_hash() -> str:
    repository = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((repository / "decode_dse").rglob("*")):
        if (
            path.is_file()
            and path.suffix in {".py", ".json", ".sh"}
            and "__pycache__" not in path.parts
        ):
            relative = path.relative_to(repository).as_posix().encode()
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "little"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _artifact_directory(root: Path, document_id: str) -> Path:
    token = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return root / token


def _dataset_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("evaluation_data")
    if not isinstance(value, Mapping):
        raise ValueError("config.evaluation_data is required")
    required = ("dataset_name", "dataset_revision")
    missing = tuple(key for key in required if not value.get(key))
    if missing:
        raise ValueError(f"evaluation_data is missing {missing}")
    return value


def _is_wikitext_article_heading(text: str) -> bool:
    value = text.strip()
    return (
        value.startswith("= ")
        and value.endswith(" =")
        and not value.startswith("= =")
        and not value.endswith("= =")
    )


def _group_wikitext_documents(
    rows: Iterable[Any],
    *,
    split: str,
    separator: str,
) -> tuple[tuple[str, str], ...]:
    grouped: list[tuple[int, int, list[str]]] = []
    start: int | None = None
    content: list[str] = []
    last_index = -1
    for index, raw in enumerate(rows):
        text = str(raw)
        last_index = index
        if _is_wikitext_article_heading(text) and content:
            grouped.append((int(start), index - 1, content))
            start = index
            content = []
        if text.strip():
            if start is None:
                start = index
            content.append(text)
    if content:
        grouped.append((int(start), last_index, content))
    if not grouped:
        raise ValueError("held-out dataset contains no non-empty documents")
    result = []
    for ordinal, (first_row, last_row, parts) in enumerate(grouped):
        document_id = (
            f"wikitext-{split}-article-{ordinal:04d}-"
            f"rows-{first_row:04d}-{last_row:04d}"
        )
        result.append((document_id, separator.join(parts)))
    return tuple(result)


def build_sample_files(
    *,
    config: Mapping[str, Any],
    bundle_path: str | Path,
    prompt_manifest_path: str | Path,
) -> TokenSampleBundle:
    """Tokenize the pinned held-out split and persist its exact token windows."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "sample preparation requires datasets and transformers"
        ) from exc

    data = _dataset_config(config)
    model_name = str(config["model_name"])
    model_revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    cache_dir = config.get("hf_cache_dir")
    local_only = bool(config.get("local_files_only", True))
    trust_remote_code = bool(config.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=tokenizer_revision,
        cache_dir=cache_dir,
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    dataset = load_dataset(
        str(data["dataset_name"]),
        data.get("dataset_config"),
        split=str(data.get("split", "test")),
        revision=str(data["dataset_revision"]),
        cache_dir=data.get("cache_dir"),
    )
    text_column = str(data.get("text_column", "text"))
    if text_column not in dataset.column_names:
        raise ValueError(f"dataset split does not contain text column {text_column!r}")
    split = str(data.get("split", "test"))
    source_documents = []
    for document_id, text in _group_wikitext_documents(
        dataset[text_column],
        split=split,
        separator=str(data.get("document_separator", "\n\n")),
    ):
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if token_ids:
            source_documents.append(
                TokenizedSourceDocument(
                    document_id=document_id,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    token_ids=tuple(token_ids),
                )
            )
    bundle = build_bundle_from_documents(
        source_documents,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        dataset_name=str(data["dataset_name"]),
        dataset_revision=str(data["dataset_revision"]),
        seed=int(config.get("seed", 0)),
    )
    save_sample_bundle(bundle, bundle_path)
    write_immutable_json(prompt_manifest_path, bundle.prompt_manifest().to_dict())
    return bundle


def _verify_prefill(
    artifact: Any,
    *,
    sample: Any,
    bundle: TokenSampleBundle,
    provenance: ArtifactProvenance,
    runtime_environment: RuntimeEnvironment,
    model_architecture: Mapping[str, Any],
) -> None:
    if artifact.model_revision != bundle.model_revision:
        raise ValueError("prefill artifact model revision mismatch")
    if artifact.tokenizer_revision != bundle.tokenizer_revision:
        raise ValueError("prefill artifact tokenizer revision mismatch")
    if artifact.prompt_hash != sample.prompt_hash:
        raise ValueError("prefill artifact prompt mismatch")
    if artifact.first_token.selection != "greedy":
        raise ValueError("prefill artifact must own greedy first-token selection")
    metadata = dict(artifact.metadata)
    if metadata.get("document_id") != sample.document_id:
        raise ValueError("prefill artifact document mismatch")
    if metadata.get("sample_bundle_hash") != bundle.canonical_hash:
        raise ValueError("prefill artifact sample-bundle mismatch")
    if (
        not metadata.get("preparation_device_uuid")
        or metadata.get("preparation_device_name")
        != runtime_environment.logical["device_name"]
        or metadata.get("preparation_compute_capability")
        != runtime_environment.logical["compute_capability"]
    ):
        raise ValueError("prefill artifact device observation is invalid")
    if (
        artifact.provenance.producer != provenance.producer
        or artifact.provenance.code_revision != provenance.code_revision
        or artifact.provenance.parameters != provenance.parameters
    ):
        raise ValueError("prefill artifact producer contract mismatch")
    layer_count = int(model_architecture["num_hidden_layers"])
    if len(artifact.layers) != layer_count:
        raise ValueError(f"prefill requires exactly {layer_count} cache layers")
    expected_shape = (
        1,
        int(model_architecture["num_key_value_heads"]),
        len(sample.prompt_token_ids),
        int(model_architecture["head_dim"]),
    )
    if any(layer.key.shape != expected_shape for layer in artifact.layers):
        raise ValueError("prefill cache geometry differs from model_architecture")


def prepare_prefill_artifacts(
    *,
    config: Mapping[str, Any],
    bundle_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Capture each BF16 prompt cache once and resume from verified artifacts."""

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "prefill preparation requires torch and transformers"
        ) from exc

    bundle = load_sample_bundle(bundle_path)
    if bundle.model_revision != str(config["model_revision"]):
        raise ValueError("sample bundle model revision differs from config")
    if bundle.tokenizer_revision != str(config["tokenizer_revision"]):
        raise ValueError("sample bundle tokenizer revision differs from config")
    artifact_root = Path(artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    device = str(config.get("device", "cuda:0"))
    seed = int(config.get("seed", 0))
    initialize_numerical_runtime(seed)
    runtime_environment = capture_runtime_environment(device, seed=seed)
    preparation_device = {
        **dict(runtime_environment.observation),
        "device_name": runtime_environment.logical["device_name"],
        "compute_capability": runtime_environment.logical["compute_capability"],
    }
    code_revision = _repository_hash()
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    provenance = ArtifactProvenance(
        producer="packedkv-bf16-prefill",
        code_revision=code_revision,
        created_at_utc=created_at,
        parameters=(
            ("model_revision", bundle.model_revision),
            ("sample_bundle_hash", bundle.canonical_hash),
            ("tokenizer_revision", bundle.tokenizer_revision),
            (
                "runtime_environment_fingerprint",
                runtime_environment.logical_fingerprint,
            ),
        ),
    )

    artifact_ids: dict[str, str] = {}
    artifact_devices: dict[str, Mapping[str, Any]] = {}

    def artifact_device(artifact: Any) -> Mapping[str, Any]:
        metadata = dict(artifact.metadata)
        return {
            "device_index": int(metadata["preparation_device_index"]),
            "device_uuid": metadata["preparation_device_uuid"],
            "total_memory_bytes": int(metadata["preparation_total_memory_bytes"]),
            "device_name": metadata["preparation_device_name"],
            "compute_capability": metadata["preparation_compute_capability"],
        }

    preparation_metadata = {
        "preparation_device_index": str(preparation_device["device_index"]),
        "preparation_device_uuid": str(preparation_device["device_uuid"]),
        "preparation_total_memory_bytes": str(preparation_device["total_memory_bytes"]),
        "preparation_device_name": str(preparation_device["device_name"]),
        "preparation_compute_capability": str(preparation_device["compute_capability"]),
    }
    pending = []
    for sample in bundle.numerical_screen + bundle.hardware_validation:
        path = _artifact_directory(artifact_root, sample.document_id)
        if path.exists():
            artifact = load_prefill_artifact(path)
            _verify_prefill(
                artifact,
                sample=sample,
                bundle=bundle,
                provenance=provenance,
                runtime_environment=runtime_environment,
                model_architecture=config["model_architecture"],
            )
            artifact_ids[sample.document_id] = artifact.artifact_id
            artifact_devices[sample.document_id] = artifact_device(artifact)
            del artifact
        else:
            pending.append(sample)

    model = None
    try:
        if pending:
            dtype_name = str(config.get("dtype", "bfloat16")).lower()
            if dtype_name != "bfloat16":
                raise ValueError("prefill artifact capture requires bfloat16")
            load_kwargs = {
                "revision": bundle.model_revision,
                "torch_dtype": torch.bfloat16,
                "cache_dir": config.get("hf_cache_dir"),
                "local_files_only": bool(config.get("local_files_only", True)),
                "trust_remote_code": bool(config.get("trust_remote_code", False)),
                "attn_implementation": "eager",
                "low_cpu_mem_usage": True,
            }
            model = AutoModelForCausalLM.from_pretrained(
                str(config["model_name"]),
                **load_kwargs,
            )
            model = model.to(device).eval()
            for index, sample in enumerate(pending, start=1):
                input_ids = torch.tensor(
                    [sample.prompt_token_ids],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                artifact = capture_bf16_prefill(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    model_revision=bundle.model_revision,
                    tokenizer_revision=bundle.tokenizer_revision,
                    provenance=provenance,
                    metadata={
                        "document_id": sample.document_id,
                        "sample_bundle_hash": bundle.canonical_hash,
                        **preparation_metadata,
                    },
                )
                _verify_prefill(
                    artifact,
                    sample=sample,
                    bundle=bundle,
                    provenance=provenance,
                    runtime_environment=runtime_environment,
                    model_architecture=config["model_architecture"],
                )
                save_prefill_artifact(
                    artifact,
                    _artifact_directory(artifact_root, sample.document_id),
                )
                artifact_ids[sample.document_id] = artifact.artifact_id
                artifact_devices[sample.document_id] = dict(preparation_device)
                print(
                    f"[prefill {index}/{len(pending)}] {sample.document_id}",
                    flush=True,
                )
                del input_ids, attention_mask, artifact
    finally:
        if model is not None:
            del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = []
    for sample in bundle.numerical_screen + bundle.hardware_validation:
        artifact_id = artifact_ids.get(sample.document_id)
        if artifact_id is None:
            artifact = load_prefill_artifact(
                _artifact_directory(artifact_root, sample.document_id)
            )
            _verify_prefill(
                artifact,
                sample=sample,
                bundle=bundle,
                provenance=provenance,
                runtime_environment=runtime_environment,
                model_architecture=config["model_architecture"],
            )
            artifact_id = artifact.artifact_id
            artifact_devices[sample.document_id] = artifact_device(artifact)
            del artifact
        records.append(
            {
                "document_id": sample.document_id,
                "prompt_hash": sample.prompt_hash,
                "artifact_id": artifact_id,
                "preparation_device": dict(artifact_devices[sample.document_id]),
                "relative_path": _artifact_directory(
                    Path("."), sample.document_id
                ).as_posix(),
            }
        )
    index = {
        "schema_version": PREFILL_INDEX_SCHEMA,
        "model_revision": bundle.model_revision,
        "tokenizer_revision": bundle.tokenizer_revision,
        "sample_bundle_hash": bundle.canonical_hash,
        "code_revision": code_revision,
        "runtime_environment": runtime_environment.to_dict(),
        "preparation_devices": [
            json.loads(encoded)
            for encoded in sorted(
                {
                    json.dumps(
                        dict(device_observation),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for device_observation in artifact_devices.values()
                }
            )
        ],
        "records": records,
    }
    write_immutable_json(artifact_root / "index.json", index)
    return index


def _load_run_plan(path: Path) -> SweepRunPlan:
    value = load_immutable_json(path)
    value.pop("content_hash")
    return SweepRunPlan.from_dict(value)


def _load_prompts(path: Path) -> PromptManifest:
    value = load_immutable_json(path)
    value.pop("content_hash")
    return PromptManifest.from_dict(value)


def prepare_admission_artifacts(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Seal every cache representation consumed by both evaluation sets."""

    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    manifest = load_manifest(output_dir / "manifest.json")
    plan = _load_run_plan(output_dir / "run_plan.json")
    prompts = _load_prompts(output_dir / "prompt_manifest.json")
    validate_run_plan(plan, manifest)
    _validate_provenance(
        output_dir / "provenance.json",
        repository=Path(__file__).resolve().parents[2],
        config_path=config_path,
        manifest=manifest,
        plan=plan,
        prompts=prompts,
    )
    executor = DecodeEvaluator.for_admission_preparation(
        config,
        workspace_root=output_dir,
    )
    if executor.bundle.prompt_manifest() != prompts:
        raise ValueError("sample bundle differs from the workspace prompts")
    executor.seal_workspace_runtime_environment(output_dir)
    return executor.prepare_admission_catalog(
        workspace_root=output_dir,
        workspace_identity={
            "manifest_hash": manifest.canonical_hash,
            "run_plan_hash": plan.canonical_hash,
            "prompt_manifest_hash": prompts.canonical_hash,
        },
    )


def prepare_stack_validity_artifact(
    *,
    output_dir: str | Path,
    compiler_report_path: str | Path,
    emulator_report_path: str | Path,
    calibration_paths: tuple[Path, ...],
) -> Mapping[str, Any]:
    """Seal the workspace stack-validity artifact from measured stage reports.

    The artifact binds this workspace's run plan and manifest hashes, so it
    must be produced against the planned workspace it will gate.
    """

    from decode_dse.software.stack_validity import build_stack_validity_artifact

    output_dir = Path(output_dir).resolve()
    manifest = load_manifest(output_dir / "manifest.json")
    plan = _load_run_plan(output_dir / "run_plan.json")
    validate_run_plan(plan, manifest)
    destination = output_dir / "stack_validity.json"
    document = build_stack_validity_artifact(
        manifest=manifest,
        plan=plan,
        compiler_report_path=compiler_report_path,
        emulator_report_path=emulator_report_path,
        calibration_paths=calibration_paths,
        destination=destination,
    )
    load_built_stack_validity(
        destination,
        manifest=manifest,
        scope_profile_ids=plan.hardware_validation_profile_ids,
        required_stages=("compiler", "emulator"),
        scope_name="hardware-validation",
        run_plan_hash=plan.canonical_hash,
    )
    return {
        "stack_validity": str(destination),
        "profile_count": len(document["profiles"]),
        "calibration_ids": document["calibration_ids"],
        "run_plan_hash": document["run_plan_hash"],
        "manifest_hash": document["manifest_hash"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser("samples")
    samples.add_argument("--config", required=True)
    samples.add_argument("--sample-bundle", required=True)
    samples.add_argument("--prompt-manifest", required=True)
    prefill = commands.add_parser("prefill")
    prefill.add_argument("--config", required=True)
    prefill.add_argument("--sample-bundle", required=True)
    prefill.add_argument("--artifact-root", required=True)
    admission = commands.add_parser("admission")
    admission.add_argument("--config", required=True)
    admission.add_argument("--output-dir", required=True)
    validity = commands.add_parser("stack-validity")
    validity.add_argument("--config", required=True)
    validity.add_argument("--output-dir", required=True)
    validity.add_argument("--compiler-report", required=True)
    validity.add_argument("--emulator-report", required=True)
    validity.add_argument(
        "--calibration-artifact",
        action="append",
        required=True,
        help="retained emulator-calibration artifact to bind (repeatable)",
    )
    return parser


def sweep_inputs_main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    try:
        config = _load_config(args.config)
        if args.command == "samples":
            result = build_sample_files(
                config=config,
                bundle_path=args.sample_bundle,
                prompt_manifest_path=args.prompt_manifest,
            )
            payload = {
                "sample_bundle_hash": result.canonical_hash,
                "prompt_manifest_hash": result.prompt_manifest().canonical_hash,
            }
        elif args.command == "prefill":
            payload = prepare_prefill_artifacts(
                config=config,
                bundle_path=args.sample_bundle,
                artifact_root=args.artifact_root,
            )
        elif args.command == "stack-validity":
            payload = prepare_stack_validity_artifact(
                output_dir=args.output_dir,
                compiler_report_path=args.compiler_report,
                emulator_report_path=args.emulator_report,
                calibration_paths=tuple(
                    Path(path) for path in args.calibration_artifact
                ),
            )
        else:
            payload = prepare_admission_artifacts(
                config=config,
                config_path=args.config,
                output_dir=args.output_dir,
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


__all__ = [
    "PREFILL_INDEX_SCHEMA",
    "build_sample_files",
    "main",
    "prepare_admission_artifacts",
    "prepare_stack_validity_artifact",
    "prepare_prefill_artifacts",
]


PROVENANCE_SCHEMA = "decode-sweep-provenance"


STAGE_INVOCATION_SCHEMA = "decode-stage-invocation"


DEFAULT_EXECUTOR_FACTORY = "decode_dse.software.decode_evaluator:create_executor"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sweep_launcher_load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_manifest(
    config: Mapping[str, Any],
    repository: Path,
) -> SweepManifest:
    validate_sweep_config(config)
    return build_exhaustive_manifest(
        str(config["model_name"]),
        str(config["model_revision"]),
        dict(config["model_architecture"]),
        build_quantizer_provenance(repository, config),
        str(config["tokenizer_revision"]),
        search_space=declared_search_space(config.get("search", {})),
    )


def _load_plan(path: Path) -> SweepRunPlan:
    value = load_immutable_json(path)
    value.pop("content_hash")
    return SweepRunPlan.from_dict(value)


def _load_output_prompts(path: Path) -> PromptManifest:
    value = load_immutable_json(path)
    value.pop("content_hash")
    return PromptManifest.from_dict(value)


def _provenance(
    *,
    repository: Path,
    config_path: Path,
    manifest: SweepManifest,
    plan: SweepRunPlan,
    prompts: PromptManifest | None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    refinement = config.get("refinement", {})
    calibration = (
        refinement.get("calibration_data", {})
        if isinstance(refinement, Mapping)
        else {}
    )
    datasets = {
        "evaluation": {
            "name": config.get("evaluation_data", {}).get("dataset_name"),
            "config": config.get("evaluation_data", {}).get("dataset_config"),
            "revision": config.get("evaluation_data", {}).get("dataset_revision"),
            "split": config.get("evaluation_data", {}).get("split"),
        }
    }
    if calibration:
        calibration_descriptor = {
            "name": calibration.get("dataset_name"),
            "config": calibration.get("dataset_config"),
            "revision": calibration.get("dataset_revision"),
            "split": calibration.get("split"),
        }
        if any(value is None for value in calibration_descriptor.values()):
            raise ValueError(
                "refinement calibration provenance descriptor is incomplete"
            )
        datasets["refinement_calibration"] = calibration_descriptor
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "model": {
            "name": manifest.model_name,
            "revision": manifest.model_revision,
            "tokenizer_revision": manifest.tokenizer_revision,
            "dtype": str(config.get("dtype")),
            "architecture": dict(config.get("model_architecture", {})),
            "placement": dict(config.get("model_placement", {})),
        },
        "runtime_requirements": dict(config.get("runtime_requirements", {})),
        "artifact_policy": str(config.get("executor", {}).get("artifact_policy", "")),
        "datasets": datasets,
        "quantizer_provenance": manifest.quantizer_provenance.to_dict(),
        "quantizer_provenance_hash": manifest.quantizer_provenance.canonical_hash,
        "prompt_manifest_hash": (
            prompts.canonical_hash if prompts is not None else None
        ),
        "config_sha256": _sha256_file(config_path),
        "software_tree_sha256": _software_tree_hash(repository),
        "mase_tree_sha256": _mase_tree_hash(
            repository,
            config,
        ),
        "python_version": sys.version.splitlines()[0],
        "platform": platform.platform(),
    }


def _validate_provenance(
    provenance_path: Path,
    *,
    repository: Path,
    config_path: Path,
    manifest: SweepManifest,
    plan: SweepRunPlan,
    prompts: PromptManifest,
) -> None:
    recorded = load_immutable_json(provenance_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    current_quantizers = build_quantizer_provenance(repository, config)
    if current_quantizers != manifest.quantizer_provenance:
        raise RuntimeError(
            "quantizer sources differ from the immutable sweep manifest: "
            f"recorded={manifest.quantizer_provenance.canonical_hash}, "
            f"current={current_quantizers.canonical_hash}"
        )
    created_at_utc = recorded.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc.endswith("Z"):
        raise RuntimeError("provenance is missing its UTC creation timestamp")
    expected = _provenance(
        repository=repository,
        config_path=config_path,
        manifest=manifest,
        plan=plan,
        prompts=prompts,
        created_at_utc=created_at_utc,
    )
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise RuntimeError(
                f"provenance mismatch for {key}: "
                f"recorded={recorded.get(key)!r}, current={value!r}"
            )


class ExecutorFactory(Protocol):
    def __call__(self, context: ExecutorContext) -> ExhaustiveSweepExecutor: ...


def _load_executor_factory(spec: str) -> ExecutorFactory:
    if ":" not in spec:
        raise ValueError("executor factory must use module.path:callable syntax")
    module_name, attribute = spec.rsplit(":", 1)
    if not module_name or not attribute:
        raise ValueError("executor factory must use module.path:callable syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"executor factory is not callable: {spec}")
    return factory


def _stage_ids(plan: SweepRunPlan, stage: str) -> tuple[str, ...]:
    if stage == "preflight":
        return plan.preflight_profile_ids
    if stage == "validation-pilot":
        hardware_validation_ids = set(plan.hardware_validation_profile_ids)
        return tuple(
            profile_id
            for profile_id in plan.preflight_profile_ids
            if profile_id in hardware_validation_ids
        )
    if stage == "numerical-screen":
        return plan.numerical_screen_profile_ids
    if stage == "hardware-validation":
        return plan.hardware_validation_profile_ids
    raise ValueError(f"unsupported stage {stage!r}")


def _stage_contract(plan: SweepRunPlan, stage: str) -> StageSampleContract:
    if stage == "preflight":
        name = "numerical-screen"
    elif stage == "validation-pilot":
        name = "hardware-validation"
    else:
        name = stage
    contracts = {contract.name: contract for contract in plan.sample_contracts}
    return contracts[name]


def partition_stage_profile_ids(
    manifest: SweepManifest,
    profile_ids: Sequence[str],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[str, ...]:
    """Partition whole weight banks across deterministic execution shards."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    selected = set(profile_ids)
    if len(selected) != len(profile_ids):
        raise ValueError("stage schedule contains duplicate profile IDs")
    by_id = {entry.profile_id: entry for entry in manifest.entries}
    unknown = selected - set(by_id)
    if unknown:
        raise ValueError(f"stage schedule contains unknown profiles: {sorted(unknown)}")
    weight_order: list[str] = []
    for profile_id in profile_ids:
        weight_format = by_id[profile_id].profile.weight_format
        if weight_format not in weight_order:
            weight_order.append(weight_format)
    if shard_count > len(weight_order):
        raise ValueError(
            f"{shard_count} shards exceed {len(weight_order)} weight banks"
        )
    shard_by_weight = {
        weight_format: index % shard_count
        for index, weight_format in enumerate(weight_order)
    }
    result = tuple(
        profile_id
        for profile_id in profile_ids
        if shard_by_weight[by_id[profile_id].profile.weight_format] == shard_index
    )
    if not result:
        raise AssertionError("deterministic sharding produced an empty shard")
    return result


def _completion_rows(
    root: Path,
    stage_manifest: SweepManifest,
) -> dict[str, Mapping[str, Any]]:
    completed = root / "completed"
    rows: dict[str, Mapping[str, Any]] = {}
    if not completed.exists():
        return rows
    for path in sorted(completed.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        marker_hash = value.pop("marker_hash", None)
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if marker_hash != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"completion checksum mismatch: {path}")
        if value.get("manifest_hash") != stage_manifest.canonical_hash:
            raise ValueError(f"completion manifest mismatch: {path}")
        profile_id = str(value["profile_id"])
        if profile_id in rows:
            raise ValueError(f"duplicate completion marker for {profile_id}")
        rows[profile_id] = value
    return rows


def _require_stage_completion(
    root: Path,
    master_manifest: SweepManifest,
    *,
    expected_profile_ids: Iterable[str],
    required_success_ids: Iterable[str],
) -> None:
    expected_ids = set(expected_profile_ids)
    master_by_id = {
        entry.profile_id: entry.profile for entry in master_manifest.entries
    }
    if (root / "manifest.json").is_file():
        partitions = (root,)
    else:
        partitions = tuple(sorted(root.glob("part-*-of-*")))
    rows: dict[str, Mapping[str, Any]] = {}
    scheduled: set[str] = set()
    for partition in partitions:
        manifest_path = partition / "manifest.json"
        if not manifest_path.is_file():
            continue
        partition_manifest = load_manifest(manifest_path)
        invocation_path = partition / "invocation.json"
        if not invocation_path.is_file():
            raise ValueError(f"missing upstream invocation: {invocation_path}")
        invocation = load_immutable_json(invocation_path)
        if (
            invocation.get("master_manifest_hash") != master_manifest.canonical_hash
            or invocation.get("stage_manifest_hash")
            != partition_manifest.canonical_hash
            or invocation.get("profile_count") != len(partition_manifest.entries)
        ):
            raise ValueError(f"upstream invocation mismatch: {invocation_path}")
        for entry in partition_manifest.entries:
            if (
                entry.profile_id not in expected_ids
                or master_by_id.get(entry.profile_id) != entry.profile
            ):
                raise ValueError(
                    f"invalid upstream profile in {manifest_path}: "
                    f"{entry.profile_id}"
                )
            if entry.profile_id in scheduled:
                raise ValueError(
                    f"profile appears in multiple upstream partitions: "
                    f"{entry.profile_id}"
                )
            scheduled.add(entry.profile_id)
        for profile_id, value in _completion_rows(
            partition, partition_manifest
        ).items():
            if profile_id in rows:
                raise ValueError(f"duplicate upstream completion for {profile_id}")
            rows[profile_id] = value
    if scheduled != expected_ids:
        missing = sorted(expected_ids - scheduled)
        unexpected = sorted(scheduled - expected_ids)
        raise RuntimeError(
            f"upstream schedule is incomplete: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    if set(rows) != expected_ids:
        missing = sorted(expected_ids - set(rows))
        unexpected = sorted(set(rows) - expected_ids)
        raise RuntimeError(
            f"stage is incomplete: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    failed_required = sorted(
        profile_id
        for profile_id in required_success_ids
        if rows[profile_id]["state"] != "succeeded"
    )
    if failed_required:
        raise RuntimeError(
            "required upstream profiles failed: " + ", ".join(failed_required[:3])
        )


def _load_workspace(
    output_dir: Path,
) -> tuple[SweepManifest, SweepRunPlan, PromptManifest]:
    manifest = load_manifest(output_dir / "manifest.json")
    plan = _load_plan(output_dir / "run_plan.json")
    prompts = _load_output_prompts(output_dir / "prompt_manifest.json")
    validate_run_plan(plan, manifest)
    return manifest, plan, prompts


def _plan_summary(
    manifest: SweepManifest,
    plan: SweepRunPlan,
    prompts: PromptManifest | None,
) -> dict[str, Any]:
    return {
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "quantizer_provenance_hash": (manifest.quantizer_provenance.canonical_hash),
        "prompt_manifest_hash": (
            prompts.canonical_hash if prompts is not None else None
        ),
        "preflight_profiles": len(plan.preflight_profile_ids),
        "numerical_screen_profiles": len(plan.numerical_screen_profile_ids),
        "hardware_validation_profiles": len(plan.hardware_validation_profile_ids),
        "numerical_screen_workers": plan.numerical_screen_workers,
        "hardware_validation_workers": plan.hardware_validation_workers,
        "numerical_screen_microbatch_size": plan.numerical_screen_microbatch_size,
        "hardware_validation_microbatch_size": plan.hardware_validation_microbatch_size,
        "device_labels": list(plan.device_labels),
        "gpu_baseline_work_units": list(plan.gpu_baseline_work_units),
        "gpu_baseline_measurements": len(plan.gpu_baseline_work_units),
        "execution_ready": prompts is not None,
    }


_COMPILER_TRACE_PREFLIGHT_SCHEMA = "decode-compiler-trace-preflight/v1"
_COMPILER_TRACE_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "max_unique_family_artifacts",
        "max_unique_lowering_instantiations",
        "max_lazy_trace_instantiations",
        "max_projected_trace_bytes",
        "max_context_timing_resolutions",
        "max_joined_identities",
        "max_joined_bytes",
        "max_digest_updates",
        "projected_joined_row_bytes",
        "digest_updates_per_joined_identity",
    }
)
_COMPILER_TRACE_PREFLIGHT_CACHE: dict[str, dict[str, Any]] = {}


def _compiler_trace_execution_limits(
    config: Mapping[str, Any],
) -> dict[str, int | float | str]:
    value = config.get("compiler_trace_preflight")
    if not isinstance(value, Mapping):
        raise ValueError("config.compiler_trace_preflight is required")
    if set(value) != _COMPILER_TRACE_PREFLIGHT_FIELDS:
        raise ValueError("compiler_trace_preflight fields differ from its schema")
    if value.get("schema_version") != _COMPILER_TRACE_PREFLIGHT_SCHEMA:
        raise ValueError("unsupported compiler_trace_preflight schema_version")
    integer_fields = _COMPILER_TRACE_PREFLIGHT_FIELDS - {"schema_version"}
    normalized: dict[str, int | float | str] = {
        "schema_version": _COMPILER_TRACE_PREFLIGHT_SCHEMA
    }
    for name in sorted(integer_fields):
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(
                f"compiler_trace_preflight.{name} must be a positive integer"
            )
        normalized[name] = int(raw)
    return normalized


def _storage_token(token: str) -> tuple[str, int]:
    if token.startswith("MXINT") and token[5:].isdigit():
        return "mxint", int(token[5:])
    match = re.fullmatch(r"E([0-9]+)M([0-9]+)", token)
    if match is not None:
        return "mxfp", 1 + int(match.group(1)) + int(match.group(2))
    raise ValueError(f"unsupported compiler storage token {token!r}")


def _profile_storage_signature(profile: Any) -> tuple[Any, ...]:
    weight_family, weight_bits = _storage_token(str(profile.weight_format))
    key_family, key_bits = _storage_token(str(profile.key_format))
    value_family, value_bits = _storage_token(str(profile.value_format))
    return (
        weight_family,
        weight_bits,
        key_family,
        key_bits,
        value_family,
        value_bits,
        int(profile.block_size),
        int(profile.scale_bits),
    )


def _storage_signature_blockers(signature: tuple[Any, ...]) -> tuple[str, ...]:
    # MXINT/MXFP and distinct K/V formats are explicit frontend receipt
    # bindings. Their element widths remain structure-changing; their format
    # family does not alter the native ISA or logical request addresses.
    _ = signature
    return ()


def _lowering_storage_signature(signature: tuple[Any, ...]) -> tuple[int, ...]:
    """Return only physical widths that shape native trace generation."""

    return (
        int(signature[1]),
        int(signature[3]),
        int(signature[5]),
        int(signature[6]),
        int(signature[7]),
    )


def _sum_positive_ceiling_range(start: int, count: int, divisor: int) -> int:
    """Return sum(ceil(value/divisor)) without materializing context points."""

    def prefix(last: int) -> int:
        if last <= 0:
            return 0
        quotient, remainder = divmod(last, divisor)
        return (
            divisor * quotient * (quotient + 1) // 2
            + remainder * (quotient + 1)
        )

    return prefix(start + count - 1) - prefix(start - 1)


def _native_compiler_geometry_blockers(
    candidate: Any,
    architecture: Mapping[str, Any],
    block_sizes: set[int],
) -> tuple[str, ...]:
    """Return exact rank-local frontend blockers for one hardware candidate."""

    heads = int(architecture["num_attention_heads"])
    kv_heads = int(architecture["num_key_value_heads"])
    head_dim = int(architecture["head_dim"])
    blockers = []
    if heads % kv_heads or heads % candidate.tp or kv_heads % candidate.tp:
        blockers.append("invalid_grouped_query_attention_ratio")
        local_heads = heads
        local_kv_heads = kv_heads
    else:
        local_heads = heads // candidate.tp
        local_kv_heads = kv_heads // candidate.tp
    if candidate.vlen != candidate.mlen:
        blockers.append("native_frontend_requires_vlen_equal_mlen")
    if head_dim != candidate.hlen or candidate.mlen % candidate.hlen:
        blockers.append("packed_attention_head_geometry_unsupported")
    elif local_heads // local_kv_heads > candidate.mlen // candidate.hlen:
        blockers.append("grouped_query_ratio_exceeds_head_broadcast")
    if (
        local_kv_heads * head_dim > candidate.mlen
        or local_kv_heads > 16
        or candidate.mlen % head_dim
        or any(
            head_dim % block or candidate.mlen % block
            for block in block_sizes
        )
    ):
        blockers.append("packed_kv_layout_geometry_unsupported")
    return tuple(sorted(set(blockers)))


def _compiler_trace_generation_points(
    config: Mapping[str, Any],
    manifest: SweepManifest,
) -> tuple[Any, ...]:
    """Build one exact point for every native batch/storage lowering."""

    from decode_dse.hardware.design_space import ExactHardwareSpace
    from decode_dse.hardware.evaluation import precision_request
    from decode_dse.simulator_bridge import DecodeSimulator

    architecture = config.get("model_architecture")
    workload = config.get("reference_workload")
    output_head = config.get("output_head_contract")
    pipeline = config.get("publication_pipeline")
    if not isinstance(architecture, Mapping):
        raise ValueError("config is missing model_architecture")
    if not isinstance(workload, Mapping):
        raise ValueError("config is missing reference_workload")
    if not isinstance(output_head, Mapping):
        raise ValueError("config is missing output_head_contract")
    if not isinstance(pipeline, Mapping) or not isinstance(
        pipeline.get("resources"), Mapping
    ):
        raise ValueError("config is missing publication_pipeline.resources")
    resources = pipeline["resources"]

    storage_representatives: dict[tuple[int, ...], Any] = {}
    for entry in manifest.entries:
        if not entry.legality.hardware_candidate:
            continue
        signature = _lowering_storage_signature(
            _profile_storage_signature(entry.profile)
        )
        storage_representatives.setdefault(signature, entry)
    if not storage_representatives:
        raise ValueError("compiler artifact generation has no hardware profiles")

    block_sizes = {
        int(signature[3]) for signature in storage_representatives
    }
    candidate_representatives: dict[tuple[Any, ...], Any] = {}
    hardware_space = ExactHardwareSpace.from_study_config(config)
    for candidate in hardware_space.iter_candidates(
        int(architecture["hidden_size"])
    ):
        if _native_compiler_geometry_blockers(
            candidate,
            architecture,
            block_sizes,
        ):
            continue
        signature = (
            int(candidate.mlen),
            int(candidate.blen),
            int(candidate.vlen),
            int(candidate.hlen),
            int(candidate.batch),
            int(candidate.tp),
            int(candidate.kvp),
            bool(candidate.kv_head_reuse),
        )
        candidate_representatives.setdefault(signature, candidate)
    if not candidate_representatives:
        raise ValueError("compiler artifact generation has no supported hardware")

    simulator = DecodeSimulator(str(config["sim_model"]))
    precisions = {}
    for signature, entry in storage_representatives.items():
        request = precision_request(entry.profile)
        precisions[signature] = simulator.make_precision(
            attn_w=request.weight,
            ffn_w=request.weight,
            key=request.key,
            value=request.value,
            w_fmt=request.weight_family,
            key_fmt=request.key_family,
            value_fmt=request.value_family,
            block=request.block_size,
            act_w=request.activation,
            act_fmt=request.activation_family,
        )

    points = []
    for candidate_signature in sorted(candidate_representatives):
        candidate = candidate_representatives[candidate_signature]
        hbm = simulator.hbm_overrides(
            candidate.hbm_generation,
            candidate.hbm_channels,
        )
        overrides = {
            "MLEN": candidate.mlen,
            "BLEN": candidate.blen,
            "VLEN": candidate.vlen,
            "HLEN": candidate.hlen,
            "TP": candidate.tp,
            "KVP": candidate.kvp,
            "LINK_PORTS": candidate.link_ports,
            "SRAM_POLICY": candidate.sram_policy,
            "LINK_GENERATION": "nvlink4",
            **hbm,
        }
        if candidate.architecture_knobs_explicit:
            overrides.update(
                {
                    "KV_HEAD_REUSE": candidate.kv_head_reuse,
                    "DRAIN_OVERLAPPED": candidate.drain_overlapped,
                }
            )
        hardware = simulator.base_hw.model_copy(update=overrides)
        for storage_signature in sorted(precisions):
            points.append(
                simulator.compiler_trace_point(
                    precisions[storage_signature],
                    hardware=hardware,
                    overrides=overrides,
                    batch=int(candidate.batch),
                    input_seq=int(workload["input_seq"]),
                    output_seq=int(workload["output_seq"]),
                    stride=int(resources["stride"]),
                    n_chips=int(candidate.chip_count),
                    hbm_gen=str(candidate.hbm_generation),
                    hbm_channels=int(candidate.hbm_channels),
                    kv_layout="dense_selector",
                    runtime_hbm_reserve_bytes=int(
                        resources["runtime_hbm_reserve_bytes"]
                    ),
                    output_head_location=str(
                        output_head["headline_location"]
                    ),
                )
            )
    return tuple(points)


def compiler_trace_artifacts_main(
    argv: Iterable[str] | None = None,
) -> int:
    """Generate the complete compact compiler-trace artifact set."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    repository = Path(__file__).resolve().parents[2]
    try:
        config = _load_config(args.config.resolve())
        _, artifact_values, _ = _publication_pipeline_config(config)
        configured_output = _resolve_publication_pipeline_paths(
            artifact_values,
            repository=repository,
            output_dir=args.output_dir.resolve(),
        )["compiler_trace_artifacts"]
        if configured_output is None or args.output.resolve() != configured_output:
            raise ValueError(
                "compiler trace output differs from publication_pipeline.artifacts"
            )
        manifest = _build_manifest(config, repository)
        preflight = _compiler_trace_feasibility(config, manifest)
        _require_compiler_trace_feasible(preflight)
        points = _compiler_trace_generation_points(config, manifest)
        workload = config["reference_workload"]
        resources = config["publication_pipeline"]["resources"]
        contexts = range(
            int(workload["input_seq"]),
            int(workload["input_seq"]) + int(workload["output_seq"]),
            int(resources["stride"]),
        )
        from full_model_artifact_build import build_full_model_decode_artifact_set

        result = build_full_model_decode_artifact_set(
            ((point, contexts) for point in points),
            args.output.resolve(),
            dry_run=args.dry_run,
        )
        if not args.dry_run and not args.output.resolve().is_file():
            raise RuntimeError(
                "compiler trace builder did not create its declared artifact"
            )
        if callable(getattr(result, "to_dict", None)):
            payload = result.to_dict()
        elif isinstance(result, Mapping):
            payload = dict(result)
        else:
            payload = {"result": str(result)}
        print(
            json.dumps(
                {
                    "schema_version": "decode-compiler-trace-generation/v1",
                    "dry_run": args.dry_run,
                    "point_count": len(points),
                    "context_start": contexts.start,
                    "context_stop": contexts.stop,
                    "context_step": contexts.step,
                    "output": None if args.dry_run else str(args.output.resolve()),
                    "build": payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


def _compiler_trace_feasibility(
    config: Mapping[str, Any],
    manifest: SweepManifest,
) -> dict[str, Any]:
    """Factorize the full legal compiler/simulator/join workload exactly."""

    from decode_dse.hardware.design_space import ExactHardwareSpace

    cache_key = hashlib.sha256(
        json.dumps(
            {
                "schema_version": _COMPILER_TRACE_PREFLIGHT_SCHEMA,
                "config": dict(config),
                "manifest_hash": manifest.canonical_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    cached = _COMPILER_TRACE_PREFLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return json.loads(json.dumps(cached))

    limits = _compiler_trace_execution_limits(config)
    architecture = config.get("model_architecture")
    workload = config.get("reference_workload")
    pipeline = config.get("publication_pipeline")
    if not isinstance(architecture, Mapping):
        raise ValueError("config is missing model_architecture")
    if not isinstance(workload, Mapping):
        raise ValueError("config is missing reference_workload")
    if not isinstance(pipeline, Mapping) or not isinstance(
        pipeline.get("resources"),
        Mapping,
    ):
        raise ValueError("config is missing publication_pipeline.resources")
    resources = pipeline["resources"]
    stride = int(resources["stride"])
    input_tokens = int(workload["input_seq"])
    output_tokens = int(workload["output_seq"])
    if min(stride, input_tokens, output_tokens) <= 0:
        raise ValueError("compiler workload lengths and stride must be positive")

    hardware_entries = tuple(
        entry for entry in manifest.entries if entry.legality.hardware_candidate
    )
    storage_signatures = {
        _profile_storage_signature(entry.profile) for entry in hardware_entries
    }
    lowering_storage_signatures = {
        _lowering_storage_signature(signature) for signature in storage_signatures
    }
    family_signatures = {
        ("dense_selector", "format_bound_in_exact_point_receipt")
    }
    storage_blocker_sets = {
        signature: _storage_signature_blockers(signature)
        for signature in storage_signatures
    }
    supported_storage = {
        signature
        for signature, blockers in storage_blocker_sets.items()
        if not blockers
    }

    hidden = int(architecture["hidden_size"])
    heads = int(architecture["num_attention_heads"])
    block_sizes = {int(signature[6]) for signature in storage_signatures}
    hardware_space = ExactHardwareSpace.from_study_config(config)
    structural_candidates = 0
    batch_resolution_signatures: set[tuple[Any, ...]] = set()
    hardware_signature_blockers: dict[tuple[Any, ...], tuple[str, ...]] = {}
    hardware_signature_candidate_counts: dict[tuple[Any, ...], int] = {}
    for candidate in hardware_space.iter_candidates(hidden):
        structural_candidates += 1
        batch_resolution_signatures.add(
            (
                int(candidate.mlen),
                int(candidate.blen),
                int(candidate.vlen),
                int(candidate.hlen),
                int(candidate.batch),
                int(candidate.tp),
                int(candidate.kvp),
                bool(candidate.kv_head_reuse),
            )
        )
        # Independent serving batches are exact affine slab replications.
        # Native code generation is keyed only by the single-slab template;
        # the requested batch remains sealed in the resolution receipt.
        signature = (
            int(candidate.mlen),
            int(candidate.blen),
            int(candidate.vlen),
            int(candidate.hlen),
            int(candidate.tp),
            int(candidate.kvp),
            bool(candidate.kv_head_reuse),
        )
        hardware_signature_candidate_counts[signature] = (
            hardware_signature_candidate_counts.get(signature, 0) + 1
        )
        if signature in hardware_signature_blockers:
            continue
        hardware_signature_blockers[signature] = (
            _native_compiler_geometry_blockers(
                candidate,
                architecture,
                block_sizes,
            )
        )

    hardware_signatures = set(hardware_signature_blockers)
    structural_geometry_blockers = {
        "invalid_grouped_query_attention_ratio",
        "packed_attention_head_geometry_unsupported",
        "grouped_query_ratio_exceeds_head_broadcast",
        "packed_kv_layout_geometry_unsupported",
    }
    geometry_eligible_hardware = {
        signature
        for signature, blockers in hardware_signature_blockers.items()
        if not structural_geometry_blockers.intersection(blockers)
    }
    supported_hardware = {
        signature
        for signature, blockers in hardware_signature_blockers.items()
        if not blockers
    }
    model_blockers = []
    if int(architecture.get("num_local_experts", 1)) > 1:
        model_blockers.append("mixture_of_experts_not_lowered")
    if int(architecture.get("n_sliding", 0)) > 0:
        model_blockers.append("sliding_attention_not_lowered")

    batch_record_counts: dict[tuple[Any, ...], int] = {}
    for signature in batch_resolution_signatures:
        batch_free_signature = signature[:4] + signature[5:]
        batch_record_counts[batch_free_signature] = (
            batch_record_counts.get(batch_free_signature, 0) + 1
        )
    geometry_eligible_batch_record_count = sum(
        batch_record_counts[item] for item in geometry_eligible_hardware
    )
    geometry_eligible_candidate_count = sum(
        hardware_signature_candidate_counts[item]
        for item in geometry_eligible_hardware
    )
    lowering_instantiations = (
        geometry_eligible_batch_record_count * len(lowering_storage_signatures)
    )
    native_supported_lowerings = (
        0
        if model_blockers
        else sum(batch_record_counts[item] for item in supported_hardware)
        * len({_lowering_storage_signature(item) for item in supported_storage})
    )
    # The native recipe is batch-free, but exact records are not. Query tiling,
    # activation/cache addresses, shared weight traffic, and tails are compiled
    # for every requested batch. Context and storage parameters remain bound by
    # their separate exact algebraic resolvers.
    native_template_instantiations = geometry_eligible_batch_record_count
    raw_profile_candidate_pairs = structural_candidates * len(hardware_entries)
    raw_context_point_resolutions = raw_profile_candidate_pairs * output_tokens
    physical_signature_pairs = (
        geometry_eligible_candidate_count * len(lowering_storage_signatures)
    )
    # Every eligible candidate/storage factor produces one exact evaluator
    # outcome. Each outcome resolves its growing-context sequence through one
    # compact algebraic schedule rather than per-context compiler invocation.
    full_evaluator_calls = physical_signature_pairs
    timing_resolutions = full_evaluator_calls
    physical_context_step_outcomes = timing_resolutions * output_tokens
    joined_identities = physical_signature_pairs
    digest_updates = (
        joined_identities * int(limits["digest_updates_per_joined_identity"])
    )
    joined_bytes = joined_identities * int(limits["projected_joined_row_bytes"])

    layers = int(architecture["num_hidden_layers"])
    projected_trace_bytes = geometry_eligible_batch_record_count * (
        # Fixed decoder body plus one full-block and one masked-tail template
        # per layer/head. Context block multiplicity is an integer parameter,
        # so no per-token or per-block trace rows are serialized.
        64 * 1024 + layers * 96 * 1024 + layers * heads * 2 * 512
    )

    capability_counts: dict[str, int] = {}
    for blocker in model_blockers:
        capability_counts[blocker] = lowering_instantiations
    structural_rejection_signature_counts: dict[str, int] = {}
    structural_rejection_candidate_counts: dict[str, int] = {}
    for blocker in structural_geometry_blockers:
        rejected_signatures = {
            signature
            for signature, blockers in hardware_signature_blockers.items()
            if blocker in blockers
        }
        structural_rejection_signature_counts[blocker] = sum(
            batch_record_counts[signature] for signature in rejected_signatures
        )
        structural_rejection_candidate_counts[blocker] = sum(
            hardware_signature_candidate_counts[signature]
            for signature in rejected_signatures
        )
    for blocker in {
        item
        for blockers in hardware_signature_blockers.values()
        for item in blockers
    } - structural_geometry_blockers:
        capability_counts[blocker] = (
            sum(
                batch_record_counts[signature]
                for signature, blockers in hardware_signature_blockers.items()
                if blocker in blockers
            )
            * len(lowering_storage_signatures)
        )
    for blocker in {
        item for blockers in storage_blocker_sets.values() for item in blockers
    }:
        capability_counts[blocker] = (
            sum(blocker in blockers for blockers in storage_blocker_sets.values())
            * geometry_eligible_batch_record_count
        )

    checks = {
        "unique_family_artifacts": (
            len(family_signatures),
            int(limits["max_unique_family_artifacts"]),
        ),
        "unique_lowering_instantiations": (
            lowering_instantiations,
            int(limits["max_unique_lowering_instantiations"]),
        ),
        "lazy_trace_instantiations": (
            native_template_instantiations,
            int(limits["max_lazy_trace_instantiations"]),
        ),
        "projected_trace_bytes": (
            projected_trace_bytes,
            int(limits["max_projected_trace_bytes"]),
        ),
        "context_timing_resolutions": (
            timing_resolutions,
            int(limits["max_context_timing_resolutions"]),
        ),
        "joined_identities": (
            joined_identities,
            int(limits["max_joined_identities"]),
        ),
        "joined_bytes": (joined_bytes, int(limits["max_joined_bytes"])),
        "digest_updates": (digest_updates, int(limits["max_digest_updates"])),
    }
    blockers = [
        f"unsupported_native_lowering:{name}:{count}"
        for name, count in sorted(capability_counts.items())
        if count
    ]
    if stride != 1:
        blockers.append(
            "nonunit_context_stride_is_not_exact:"
            f"stride={stride}"
        )
    blockers.extend(
        f"execution_limit_exceeded:{name}:{observed}>{limit}"
        for name, (observed, limit) in checks.items()
        if observed > limit
    )
    result = {
        "schema_version": _COMPILER_TRACE_PREFLIGHT_SCHEMA,
        "execution_mode": "compiler_trace",
        "artifact_scope": "full_model_decode_step_independent_request_batch",
        "structurally_legal_hardware_candidates": structural_candidates,
        "compiler_geometry_eligible_hardware_candidates": (
            geometry_eligible_candidate_count
        ),
        "compiler_geometry_rejected_hardware_candidates": (
            structural_candidates - geometry_eligible_candidate_count
        ),
        "compiler_base_hardware_signatures": len(batch_resolution_signatures),
        "raw_batch_free_native_template_signatures": len(hardware_signatures),
        "batch_free_native_template_signatures": len(
            geometry_eligible_hardware
        ),
        "raw_exact_batch_record_signatures": len(batch_resolution_signatures),
        "exact_batch_record_signatures": geometry_eligible_batch_record_count,
        "batch_record_instantiation_factor": (
            geometry_eligible_batch_record_count
            / len(geometry_eligible_hardware)
        ),
        "hardware_timing_reuse_factor": (
            structural_candidates / len(batch_resolution_signatures)
        ),
        "hardware_relevant_precision_profiles": len(hardware_entries),
        "unique_storage_signatures": len(storage_signatures),
        "unique_structure_changing_storage_signatures": len(
            lowering_storage_signatures
        ),
        "unique_compiler_family_artifacts": len(family_signatures),
        "unique_compiler_lowering_instantiations": lowering_instantiations,
        "native_capability_supported_lowering_instantiations": (
            native_supported_lowerings
        ),
        "unique_lazy_trace_instantiations": native_template_instantiations,
        "projected_trace_generation_calls": native_template_instantiations,
        "projected_trace_bytes": projected_trace_bytes,
        "exact_contexts_per_evaluation": output_tokens,
        "configured_stride": stride,
        "projected_context_timing_resolutions": timing_resolutions,
        "physical_context_step_outcomes": physical_context_step_outcomes,
        "raw_profile_candidate_pairs": raw_profile_candidate_pairs,
        "raw_context_point_resolutions": raw_context_point_resolutions,
        "physical_signature_pairs": physical_signature_pairs,
        "projected_full_evaluator_calls": full_evaluator_calls,
        "simulator_priced_pairs": None,
        "simulator_priced_pairs_upper_bound": physical_signature_pairs,
        "projected_joined_identities": joined_identities,
        "joined_result_rows": None,
        "joined_result_rows_upper_bound": raw_profile_candidate_pairs,
        "conceptual_joined_result_rows": raw_profile_candidate_pairs,
        "projected_joined_bytes": joined_bytes,
        "projected_digest_updates": digest_updates,
        "capability_blocker_signature_counts": capability_counts,
        "structural_rejection_signature_counts": (
            structural_rejection_signature_counts
        ),
        "structural_rejection_candidate_counts": (
            structural_rejection_candidate_counts
        ),
        "projection_basis": {
            "trace_bytes": (
                "sum over exact-batch records of 64KiB + "
                "layers*96KiB + two loop/tail blocks*layers*heads*512B"
            ),
            "contexts": (
                "one compact schedule per candidate/storage factor represents "
                "every exact input_seq..input_seq+output_seq-1 context"
            ),
            "simulator_calls": (
                "one exact factor outcome per compiler-eligible hardware "
                "candidate and structure-changing storage signature"
            ),
            "joined_rows": (
                "factor rows are physical-signature*candidate; conceptual "
                "profile joins remain sealed separately and are not materialized"
            ),
            "native_template_resolution": (
                "the batch-free recipe is reused, but every batch has an exact "
                "compiled record; only storage and context loop/tail parameters "
                "bind outside native record generation"
            ),
            "elapsed_time_status": "awaiting_first_completed_profile",
        },
        "execution_limits": dict(limits),
        "compiler_trace_preflight_feasible": not blockers,
        "compiler_trace_preflight_blockers": blockers,
        "hard_gate": (
            "block_before launch preflight, remote GPU work, simulator pricing, "
            "or joined-result materialization"
        ),
    }
    _COMPILER_TRACE_PREFLIGHT_CACHE[cache_key] = result
    return json.loads(json.dumps(result))


def _require_compiler_trace_feasible(plan: Mapping[str, Any]) -> None:
    if plan.get("compiler_trace_preflight_feasible") is not True:
        blockers = plan.get("compiler_trace_preflight_blockers")
        if not isinstance(blockers, list):
            raise RuntimeError("compiler trace preflight did not return blockers")
        raise RuntimeError(
            "compiler trace production preflight failed: "
            + "; ".join(str(value) for value in blockers)
        )


def _cost_declaration(
    config: Mapping[str, Any],
    manifest: SweepManifest,
    plan: SweepRunPlan,
    launch_preflight: Any,
    compiler_trace_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare exact static work and measurement-dependent runtime cost."""

    from decode_dse.hardware.design_space import physical_cost_signature_id

    structural_candidates = int(
        compiler_trace_preflight["structurally_legal_hardware_candidates"]
    )
    hardware_entries = tuple(
        entry for entry in manifest.entries if entry.legality.hardware_candidate
    )
    physical_signatures = {
        physical_cost_signature_id(entry.profile)
        for entry in hardware_entries
    }
    tolerance = config.get("fp_ppl_tol")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0
    ):
        raise ValueError("fp_ppl_tol must be finite and positive")
    joint_hardware_search = {
        "schema": "plena-lossless-joint-search-declaration",
        "hardware_relevant_profiles": len(hardware_entries),
        "physical_cost_signatures_before_accuracy": len(physical_signatures),
        "structurally_legal_hardware_candidates": structural_candidates,
        "raw_precision_hardware_pairs": (
            len(hardware_entries) * structural_candidates
        ),
        "raw_signature_hardware_pairs": (
            len(physical_signatures) * structural_candidates
        ),
        "hard_accuracy_constraint": {
            "bf16_relative_perplexity_limit": 1.0 + float(tolerance),
            "formula": (
                "candidate_mean_nll <= reference_mean_nll + "
                "log(bf16_relative_perplexity_limit)"
            ),
        },
        "artifact_dependent_exact_counts": {
            "accuracy_passing_profiles": None,
            "accuracy_passing_physical_signatures": None,
            "hard_resource_passing_signature_pairs": None,
            "simulator_priced_pairs": None,
            "joined_result_rows": None,
            "resolution": "hardware_study_provenance_after_numerical_results",
        },
        "losslessness": (
            "The production study removes only hard accuracy, structural, "
            "matched-area, physical HBM/SRAM-capacity, and aggregate-resource "
            "failures; it performs no latency, bottleneck, or bandwidth-demand "
            "pruning and joins each cached cost result to every equivalent "
            "accuracy row."
        ),
    }

    pilot_profiles = len(_stage_ids(plan, "validation-pilot"))
    stage_profiles = {
        "preflight": len(plan.preflight_profile_ids),
        "validation_pilot": pilot_profiles,
        "numerical_screen": len(plan.numerical_screen_profile_ids),
        "hardware_validation": len(plan.hardware_validation_profile_ids),
    }
    measured = config.get("measured_trial_cost")
    trial_seconds = None
    source = None
    if isinstance(measured, Mapping):
        candidate = measured.get("seconds")
        source_candidate = measured.get("source_artifact")
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and math.isfinite(float(candidate))
            and float(candidate) > 0
            and isinstance(source_candidate, str)
            and source_candidate
        ):
            trial_seconds = float(candidate)
            source = source_candidate
    total_evaluations = sum(stage_profiles.values())
    gpu_baseline_measurements = len(plan.gpu_baseline_work_units)
    executor = config.get("executor", {})
    executor_map = executor if isinstance(executor, Mapping) else {}
    weight_bytes = (
        launch_preflight.memory_estimates[0].weight_bytes
        if launch_preflight.memory_estimates
        else 0
    )
    cpu_cache_bytes = int(float(executor_map.get("max_cpu_cache_gib", 0.0)) * (1 << 30))
    host_floor_bytes = int(
        float(executor_map.get("min_available_host_gib", 0.0)) * (1 << 30)
    )
    peak_host_bytes = max(
        host_floor_bytes,
        weight_bytes + int(config.get("max_parallel_points", 1)) * cpu_cache_bytes,
    )
    return {
        "manifest_profiles": len(manifest.entries),
        "stage_profile_evaluations": stage_profiles,
        "total_profile_evaluations": total_evaluations,
        "gpu_baseline_measurements": gpu_baseline_measurements,
        "gpu_baseline_prefill_runs": (
            gpu_baseline_measurements * plan.gpu_baseline.repetitions
        ),
        "gpu_baseline_decode_steps": (
            gpu_baseline_measurements
            * plan.gpu_baseline.repetitions
            * (
                plan.gpu_baseline.warmup_steps
                + plan.gpu_baseline.measured_steps
            )
        ),
        "total_work_units": total_evaluations + gpu_baseline_measurements,
        "joint_hardware_search": joint_hardware_search,
        "compiler_trace_preflight": dict(compiler_trace_preflight),
        "artifact_footprint": (
            launch_preflight.artifact_footprint.to_dict()
            if launch_preflight.artifact_footprint is not None
            else None
        ),
        "peak_host_memory_bytes": peak_host_bytes or None,
        "peak_device_memory_bytes": max(
            (estimate.required_bytes for estimate in launch_preflight.memory_estimates),
            default=None,
        ),
        "measured_trial_seconds": trial_seconds,
        "measured_trial_source": source,
        "projected_wall_clock_seconds": (
            total_evaluations * trial_seconds if trial_seconds is not None else None
        ),
        "projection_status": (
            "measured_trial_cost"
            if trial_seconds is not None
            else "awaiting_first_completed_profile"
        ),
        "maximum_projected_wall_clock_seconds": (plan.max_projected_hours * 3600.0),
    }


def create_workspace(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    device_labels: Sequence[str],
    prompt_manifest_path: str | Path | None = None,
    numerical_screen_workers: int | None = None,
    hardware_validation_workers: int | None = None,
    numerical_screen_microbatch_size: int | None = None,
    hardware_validation_microbatch_size: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and optionally persist one immutable sweep workspace."""

    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[2]
    config = _sweep_launcher_load_config(config_path)
    manifest = _build_manifest(config, repository)
    compiler_trace_preflight = _compiler_trace_feasibility(config, manifest)
    if not dry_run:
        _require_compiler_trace_feasible(compiler_trace_preflight)
    launch_preflight = run_launch_preflight(
        config,
        repository_root=repository,
        workspace_root=output_dir,
        device_labels=device_labels,
    )
    if not dry_run:
        launch_preflight.require_passed()
    executor_config = config.get("executor", {})
    microbatches = executor_config.get("decode_microbatch_size", {})
    if not isinstance(microbatches, Mapping):
        raise ValueError(
            "executor.decode_microbatch_size must map numerical_screen and hardware_validation"
        )
    configured_workers = int(config.get("max_parallel_points", 4))
    gpu_baseline_config = config.get("gpu_baseline")
    if not isinstance(gpu_baseline_config, Mapping):
        raise TypeError("config.gpu_baseline must be an explicit object")
    plan = build_run_plan(
        manifest,
        device_labels=device_labels,
        numerical_screen_workers=(
            configured_workers
            if numerical_screen_workers is None
            else numerical_screen_workers
        ),
        hardware_validation_workers=(
            configured_workers
            if hardware_validation_workers is None
            else hardware_validation_workers
        ),
        numerical_screen_microbatch_size=(
            int(microbatches.get("numerical_screen", 16))
            if numerical_screen_microbatch_size is None
            else numerical_screen_microbatch_size
        ),
        hardware_validation_microbatch_size=(
            int(microbatches.get("hardware_validation", 8))
            if hardware_validation_microbatch_size is None
            else hardware_validation_microbatch_size
        ),
        gpu_baseline=GPUBaselinePlan.from_config(gpu_baseline_config),
    )
    prompts = (
        load_prompt_manifest(prompt_manifest_path)
        if prompt_manifest_path is not None
        else None
    )
    summary = _plan_summary(manifest, plan, prompts)
    summary["compiler_trace_preflight"] = compiler_trace_preflight
    for name in (
        "unique_compiler_family_artifacts",
        "unique_lazy_trace_instantiations",
        "projected_trace_generation_calls",
        "projected_trace_bytes",
        "compiler_trace_preflight_feasible",
    ):
        summary[name] = compiler_trace_preflight[name]
    summary["launch_preflight"] = launch_preflight.to_dict()
    summary["cost_declaration"] = _cost_declaration(
        config,
        manifest,
        plan,
        launch_preflight,
        compiler_trace_preflight,
    )
    if dry_run:
        summary["manifest"] = manifest.to_dict()
        summary["run_plan"] = plan.to_dict()
        summary["provenance"] = _provenance(
            repository=repository,
            config_path=config_path,
            manifest=manifest,
            plan=plan,
            prompts=prompts,
            created_at_utc="1970-01-01T00:00:00Z",
        )
        return summary
    if prompts is None:
        raise ValueError("persisted workspaces require a versioned prompt manifest")

    write_manifest(output_dir / "manifest.json", manifest)
    write_immutable_json(output_dir / "run_plan.json", plan.to_dict())
    if prompts is not None:
        write_immutable_json(output_dir / "prompt_manifest.json", prompts.to_dict())
    provenance_path = output_dir / "provenance.json"
    existing_created_at = (
        load_immutable_json(provenance_path).get("created_at_utc")
        if provenance_path.is_file()
        else None
    )
    write_immutable_json(
        provenance_path,
        _provenance(
            repository=repository,
            config_path=config_path,
            manifest=manifest,
            plan=plan,
            prompts=prompts,
            created_at_utc=(
                str(existing_created_at) if existing_created_at is not None else None
            ),
        ),
    )
    return summary


def validate_workspace(
    *,
    output_dir: str | Path,
    evidence_path: str | Path,
) -> dict[str, Any]:
    """Validate preflight evidence against immutable workspace artifacts."""

    output_dir = Path(output_dir).resolve()
    manifest, plan, prompts = _load_workspace(output_dir)
    evidence = load_preflight_evidence(evidence_path)
    return evaluate_preflight_gates(manifest, plan, prompts, evidence).to_dict()


def launch_stage(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    stage: str,
    executor_factory: str = DEFAULT_EXECUTOR_FACTORY,
    evidence_path: str | Path | None = None,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    device_label: str | None = None,
) -> SweepRunSummary:
    """Launch one stage after all applicable immutable gates pass."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    repository = Path(__file__).resolve().parents[2]
    config = _sweep_launcher_load_config(config_path)
    validate_sweep_config(config)
    manifest, plan, prompts = _load_workspace(output_dir)
    if device_label is None:
        if len(plan.device_labels) != 1:
            raise ValueError(
                "device_label is required when the plan has multiple device types"
            )
        device_label = plan.device_labels[0]
    if device_label not in plan.device_labels:
        raise ValueError(f"device label {device_label!r} is outside the run plan")
    expected_shards = (
        plan.numerical_screen_workers
        if _stage_contract(plan, stage).name == "numerical-screen"
        else plan.hardware_validation_workers
    )
    if shard_count != expected_shards:
        raise ValueError(f"{stage} requires exactly {expected_shards} planned shards")
    _validate_provenance(
        output_dir / "provenance.json",
        repository=repository,
        config_path=config_path,
        manifest=manifest,
        plan=plan,
        prompts=prompts,
    )
    admission_receipt_path = output_dir / "admission_preparation.json"
    if executor_factory == DEFAULT_EXECUTOR_FACTORY:
        if not admission_receipt_path.is_file():
            raise FileNotFoundError(
                "decode stages require sealed admission preparation"
            )
        admission_receipt_sha256: str | None = _sha256_file(admission_receipt_path)
    else:
        admission_receipt_sha256 = None
    if stage not in {"preflight", "validation-pilot"}:
        if evidence_path is None:
            raise ValueError(f"{stage} requires validated preflight evidence")
        evidence = load_preflight_evidence(evidence_path)
        evaluate_preflight_gates(manifest, plan, prompts, evidence).require_passed()
        if executor_factory == DEFAULT_EXECUTOR_FACTORY:
            receipt = load_immutable_json(admission_receipt_path)
            if evidence.admission_preparation.admission_index_hash != receipt.get(
                "admission_index_hash"
            ) or evidence.admission_preparation.admission_contract_id != receipt.get(
                "admission_contract_id"
            ):
                raise ValueError(
                    "preflight evidence uses a different admission preparation"
                )
        evidence_sha256: str | None = _sha256_file(Path(evidence_path))
    else:
        evidence_sha256 = None

    stage_ids = _stage_ids(plan, stage)
    shard_ids = partition_stage_profile_ids(
        manifest,
        stage_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    stage_manifest = make_stage_manifest(manifest, shard_ids)
    if stage == "validation-pilot":
        _require_stage_completion(
            output_dir / "preflight",
            manifest,
            expected_profile_ids=plan.preflight_profile_ids,
            required_success_ids=plan.preflight_profile_ids,
        )
    elif stage == "numerical-screen":
        _require_stage_completion(
            output_dir / "preflight",
            manifest,
            expected_profile_ids=plan.preflight_profile_ids,
            required_success_ids=plan.preflight_profile_ids,
        )
    elif stage == "hardware-validation":
        _require_stage_completion(
            output_dir / "numerical-screen",
            manifest,
            expected_profile_ids=plan.numerical_screen_profile_ids,
            required_success_ids=plan.hardware_validation_profile_ids,
        )

    stage_root = output_dir / stage
    full_stage_manifest = make_stage_manifest(manifest, stage_ids)
    write_immutable_json(
        stage_root / "sharding.json",
        {
            "schema_version": "decode-stage-sharding",
            "stage": stage,
            "master_manifest_hash": manifest.canonical_hash,
            "full_stage_manifest_hash": full_stage_manifest.canonical_hash,
            "run_plan_hash": plan.canonical_hash,
            "shard_count": shard_count,
            "algorithm": "whole_weight_bank_round_robin/v1",
        },
    )
    stage_output = (
        stage_root
        if shard_count == 1
        else stage_root / f"part-{shard_index:04d}-of-{shard_count:04d}"
    )
    if limit == 0:
        return SweepRunSummary(
            attempts_written=0,
            succeeded=0,
            failed_terminal=0,
            pending=len(stage_manifest.entries),
            result_rows=0,
        )
    invocation = {
        "schema_version": STAGE_INVOCATION_SCHEMA,
        "stage": stage,
        "master_manifest_hash": manifest.canonical_hash,
        "stage_manifest_hash": stage_manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "prompt_manifest_hash": prompts.canonical_hash,
        "preflight_evidence_sha256": evidence_sha256,
        "admission_preparation_sha256": admission_receipt_sha256,
        "executor_factory": executor_factory,
        "device_label": device_label,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "profile_count": len(stage_manifest.entries),
        "sample_contract": _stage_contract(plan, stage).to_dict(),
        "decode_microbatch_size": (
            plan.numerical_screen_microbatch_size
            if _stage_contract(plan, stage).name == "numerical-screen"
            else plan.hardware_validation_microbatch_size
        ),
        "max_attempts": int(config.get("runtime", {}).get("max_attempts", 3)),
        "weight_bank_build_serialized": bool(
            config.get("executor", {}).get(
                "serialize_weight_bank_builds",
                True,
            )
        ),
    }
    write_immutable_json(stage_output / "invocation.json", invocation)

    context = ExecutorContext(
        stage=stage,
        workspace_root=output_dir,
        output_dir=stage_output,
        config=config,
        master_manifest=manifest,
        stage_manifest=stage_manifest,
        run_plan=plan,
        prompts=prompts,
        sample_contract=_stage_contract(plan, stage),
        shard_index=shard_index,
        shard_count=shard_count,
        device_label=device_label,
    )
    factory = _load_executor_factory(executor_factory)
    executor = factory(context)
    for method in ("open_weight_bank", "open_kv_admission_cache", "evaluate"):
        if not callable(getattr(executor, method, None)):
            raise TypeError(f"executor is missing callable {method}")
    runner = ExhaustiveSweepRunner(
        manifest=stage_manifest,
        output_dir=stage_output,
        executor=executor,
        max_attempts=int(config.get("runtime", {}).get("max_attempts", 3)),
        stage=stage,
    )
    return runner.run(limit=limit)


def _add_common_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)


def sweep_launcher_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan")
    _add_common_plan_arguments(plan_parser)
    plan_parser.add_argument("--device-label", action="append", required=True)
    plan_parser.add_argument("--prompt-manifest")
    plan_parser.add_argument("--numerical-screen-workers", type=int)
    plan_parser.add_argument("--hardware-validation-workers", type=int)
    plan_parser.add_argument("--numerical-screen-microbatch-size", type=int)
    plan_parser.add_argument("--hardware-validation-microbatch-size", type=int)
    plan_parser.add_argument("--dry-run", action="store_true")

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--output-dir", required=True)
    validate_parser.add_argument("--evidence", required=True)
    validate_parser.add_argument("--report")

    run_parser = commands.add_parser("run")
    _add_common_plan_arguments(run_parser)
    run_parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "validation-pilot",
            "numerical-screen",
            "hardware-validation",
        ),
        required=True,
    )
    run_parser.add_argument(
        "--executor-factory",
        default=DEFAULT_EXECUTOR_FACTORY,
    )
    run_parser.add_argument("--evidence")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--shard-count", type=int, default=1)
    run_parser.add_argument("--device-label")

    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.command == "plan":
            result = create_workspace(
                config_path=args.config,
                output_dir=args.output_dir,
                device_labels=args.device_label,
                prompt_manifest_path=args.prompt_manifest,
                numerical_screen_workers=args.numerical_screen_workers,
                hardware_validation_workers=args.hardware_validation_workers,
                numerical_screen_microbatch_size=args.numerical_screen_microbatch_size,
                hardware_validation_microbatch_size=args.hardware_validation_microbatch_size,
                dry_run=args.dry_run,
            )
        elif args.command == "validate":
            result = validate_workspace(
                output_dir=args.output_dir,
                evidence_path=args.evidence,
            )
            if args.report:
                write_immutable_json(args.report, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["passed"] else 2
        else:
            summary = launch_stage(
                config_path=args.config,
                output_dir=args.output_dir,
                stage=args.stage,
                executor_factory=args.executor_factory,
                evidence_path=args.evidence,
                limit=args.limit,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                device_label=args.device_label,
            )
            result = {
                "stage": args.stage,
                "attempts_written": summary.attempts_written,
                "succeeded": summary.succeeded,
                "failed_terminal": summary.failed_terminal,
                "pending": summary.pending,
                "result_rows": summary.result_rows,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


_GPU_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]+$")


_GATED_STAGES = {"numerical-screen", "hardware-validation"}


def parse_gpu_list(value: str) -> tuple[str, ...]:
    devices = tuple(token.strip() for token in value.split(",") if token.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("GPU identifiers must be non-empty and unique")
    if any(not _GPU_TOKEN.fullmatch(device) for device in devices):
        raise ValueError("GPU identifiers contain unsupported characters")
    return devices


def worker_command(
    *,
    config: Path,
    output_dir: Path,
    stage: str,
    device_label: str,
    shard_index: int,
    shard_count: int,
    evidence: Path | None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "decode_dse.software.sweep",
        "stage",
        "run",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--stage",
        stage,
        "--device-label",
        device_label,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]
    if evidence is not None:
        command.extend(("--evidence", str(evidence)))
    return tuple(command)


def launch(
    *,
    config: Path,
    output_dir: Path,
    stage: str,
    device_label: str,
    devices: tuple[str, ...],
    evidence: Path | None,
) -> int:
    if stage in _GATED_STAGES and evidence is None:
        raise ValueError(f"{stage} requires preflight evidence")
    if stage not in _GATED_STAGES and evidence is not None:
        raise ValueError(f"{stage} does not consume preflight evidence")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_root = output_dir / "logs" / f"{stage}-{timestamp}"
    log_root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []
    commands = []
    try:
        for index, device in enumerate(devices):
            command = worker_command(
                config=config,
                output_dir=output_dir,
                stage=stage,
                device_label=device_label,
                shard_index=index,
                shard_count=len(devices),
                evidence=evidence,
            )
            log_path = log_root / f"part-{index:04d}.log"
            handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            try:
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                handle.close()
                raise
            processes.append((process, handle, log_path))
            commands.append(
                {
                    "shard_index": index,
                    "cuda_visible_devices": device,
                    "command": list(command),
                    "log": str(log_path.resolve()),
                    "progress": str(
                        (
                            (
                                output_dir / stage
                                if len(devices) == 1
                                else output_dir
                                / stage
                                / f"part-{index:04d}-of-{len(devices):04d}"
                            )
                            / "progress.json"
                        ).resolve()
                    ),
                }
            )
    except BaseException:
        for process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            handle.close()
        raise
    return_codes = []
    reported_updates: dict[int, str] = {}

    def report_worker_progress() -> None:
        for index, command in enumerate(commands):
            progress_path = Path(command["progress"])
            if not progress_path.is_file():
                continue
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            identity = str(progress.get("updated_at", ""))
            if not identity or reported_updates.get(index) == identity:
                continue
            reported_updates[index] = identity
            eta = progress.get("estimated_remaining_seconds")
            eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
            print(
                f"{stage} shard {index}: event={progress.get('event')} "
                f"profiles={progress.get('completed_profiles')}/{progress.get('total_profiles')} "
                f"unique-banks={progress.get('unique_weight_banks_opened')}/"
                f"{progress.get('unique_weight_banks_required_this_invocation')} "
                f"remaining-eta={eta_text}",
                flush=True,
            )

    try:
        while any(process.poll() is None for process, _, _ in processes):
            report_worker_progress()
            time.sleep(1.0)
        report_worker_progress()
        return_codes.extend(process.wait() for process, _, _ in processes)
    except BaseException:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _, _ in processes:
            process.wait()
        raise
    finally:
        for _, handle, _ in processes:
            handle.close()
    summary = {
        "schema_version": "decode-shard-launch",
        "stage": stage,
        "device_label": device_label,
        "shard_count": len(devices),
        "config": str(config),
        "output_dir": str(output_dir),
        "evidence": str(evidence) if evidence is not None else None,
        "workers": [
            command | {"return_code": return_codes[index]}
            for index, command in enumerate(commands)
        ],
    }
    write_immutable_json(log_root / "summary.json", summary)
    failed = [index for index, return_code in enumerate(return_codes) if return_code]
    if failed:
        print(f"{stage} failed on shards {failed}", file=sys.stderr)
        return 1
    print(f"{stage} completed across {len(devices)} shards")
    return 0


def shards_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "validation-pilot",
            "numerical-screen",
            "hardware-validation",
        ),
        required=True,
    )
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        devices = parse_gpu_list(args.gpus)
        return launch(
            config=args.config.resolve(),
            output_dir=args.output_dir.resolve(),
            stage=args.stage,
            device_label=args.device_label,
            devices=devices,
            evidence=args.evidence.resolve() if args.evidence else None,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


_sweep_launcher_all__ = [
    "DEFAULT_EXECUTOR_FACTORY",
    "ExecutorContext",
    "ExecutorFactory",
    "create_workspace",
    "launch_stage",
    "sweep_launcher_main",
    "partition_stage_profile_ids",
    "profile_to_decode_quant_spec",
    "shards_main",
    "validate_workspace",
]


@dataclass(frozen=True)
class PipelineCommand:
    name: str
    argv: tuple[str, ...]
    first_gpu_only: bool = False
    outputs: tuple[Path, ...] = ()

    @property
    def command_id(self) -> str:
        body = {
            "name": self.name,
            "argv": list(self.argv),
            "first_gpu_only": self.first_gpu_only,
            "outputs": [str(path) for path in self.outputs],
        }
        return hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def _module_command(module: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *arguments)


PUBLICATION_PIPELINE_SCHEMA = "decode-publication-pipeline"

_PIPELINE_ARTIFACT_FIELDS = frozenset(
    {
        "admission_receipt",
        "timing_evidence",
        "compiler_trace_artifacts",
        "request_memory_calibration",
        "head_service_calibration",
        "handoff_artifact",
        "power_calibration",
        "area_config",
        "exact_dc_anchors",
        "hardware_study",
        "refined_hardware_study",
        "refinement_validity",
        "refinement_schedule",
        "refinement_promotion",
        "refinement_results",
        "publication_configurations",
        "publication_benchmarks",
        "publication_chat_template",
        "publication_contract",
        "publication_results",
        "final_selection",
        "packedkv_evidence",
        "decode_analysis",
        "figures",
    }
)
_OPTIONAL_PIPELINE_ARTIFACT_FIELDS = frozenset(
    {
        "handoff_artifact",
        "power_calibration",
        "area_config",
        "exact_dc_anchors",
        "refinement_validity",
        "packedkv_evidence",
        "decode_analysis",
    }
)
_PIPELINE_RESOURCE_FIELDS = frozenset(
    {
        "stride",
        "runtime_hbm_reserve_bytes",
        "rtl_source_tree_sha256",
        "refinement_enabled",
        "refinement_execution",
        "refinement_decode_microbatch_size",
        "bootstrap_replicates",
        "publication_enabled",
        "publication_executor",
        "publication_timing_tier",
        "study_parallel_workers",
        "figure_formats",
    }
)


def _hardware_timing_arguments(
    resources: Mapping[str, Any],
    required_path: Callable[[str], Path],
) -> list[str]:
    """Build the tier-matched timing arguments for the hardware evaluator."""

    timing_tier = str(resources["publication_timing_tier"])
    if timing_tier == "compiler_trace_request_calibrated":
        arguments = [
            "--execution-mode",
            "compiler_trace",
            "--compiler-trace-artifacts",
            str(required_path("compiler_trace_artifacts")),
            "--request-memory-calibration",
            str(required_path("request_memory_calibration")),
        ]
    else:
        arguments = ["--execution-mode", "legacy_aggregate_bandwidth"]
    arguments.extend(("--publication-timing-tier", timing_tier))
    return arguments


def _publication_pipeline_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate the declarative post-accuracy workflow without touching files."""

    pipeline = config.get("publication_pipeline")
    if not isinstance(pipeline, Mapping):
        raise ValueError("config.publication_pipeline is required")
    if pipeline.get("schema_version") != PUBLICATION_PIPELINE_SCHEMA:
        raise ValueError("unsupported publication_pipeline schema_version")
    if set(pipeline) != {"schema_version", "artifacts", "resources"}:
        raise ValueError("publication_pipeline fields differ from its schema")
    artifacts = pipeline.get("artifacts")
    resources = pipeline.get("resources")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _PIPELINE_ARTIFACT_FIELDS:
        raise ValueError("publication_pipeline.artifacts fields differ from its schema")
    if not isinstance(resources, Mapping) or set(resources) != _PIPELINE_RESOURCE_FIELDS:
        raise ValueError("publication_pipeline.resources fields differ from its schema")
    for name, value in artifacts.items():
        if name in _OPTIONAL_PIPELINE_ARTIFACT_FIELDS and value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"publication_pipeline.artifacts.{name} must be a non-empty path"
            )
    for name in (
        "stride",
        "refinement_decode_microbatch_size",
        "study_parallel_workers",
    ):
        value = resources.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"publication_pipeline.resources.{name} must be a positive integer"
            )
    if resources["study_parallel_workers"] > 256:
        raise ValueError(
            "publication_pipeline.resources.study_parallel_workers must be at most 256"
        )
    reserve = resources.get("runtime_hbm_reserve_bytes")
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError(
            "publication_pipeline.resources.runtime_hbm_reserve_bytes must be "
            "a non-negative integer"
        )
    for name in ("refinement_enabled", "publication_enabled"):
        if not isinstance(resources.get(name), bool):
            raise ValueError(
                f"publication_pipeline.resources.{name} must be a boolean"
            )
    if resources["publication_enabled"] and not resources["refinement_enabled"]:
        raise ValueError(
            "publication execution requires refined-profile evaluation and repricing"
        )
    if resources.get("refinement_execution") != "four_logical_shards":
        raise ValueError(
            "publication_pipeline.resources.refinement_execution must be "
            "four_logical_shards"
        )
    replicates = resources.get("bootstrap_replicates")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 100
    ):
        raise ValueError(
            "publication_pipeline.resources.bootstrap_replicates must be at least 100"
        )
    executor = resources.get("publication_executor")
    if not isinstance(executor, str) or not executor:
        raise ValueError(
            "publication_pipeline.resources.publication_executor must be non-empty"
        )
    from decode_dse.hardware.design_space import PUBLICATION_TIMING_TIERS

    if resources.get("publication_timing_tier") not in PUBLICATION_TIMING_TIERS:
        raise ValueError(
            "publication_pipeline.resources.publication_timing_tier must be "
            "one of " + ", ".join(sorted(PUBLICATION_TIMING_TIERS))
        )
    formats = resources.get("figure_formats")
    if (
        not isinstance(formats, list)
        or not formats
        or len(formats) != len(set(formats))
        or any(value not in {"png", "pdf", "svg"} for value in formats)
    ):
        raise ValueError(
            "publication_pipeline.resources.figure_formats must be unique PNG/PDF/SVG formats"
        )
    rtl_digest = resources.get("rtl_source_tree_sha256")
    if rtl_digest is not None and (
        not isinstance(rtl_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", rtl_digest)
    ):
        raise ValueError(
            "publication_pipeline.resources.rtl_source_tree_sha256 must be null or SHA-256"
        )
    if bool(artifacts["power_calibration"]) != bool(artifacts["area_config"]):
        raise ValueError(
            "publication pipeline power_calibration and area_config must be supplied together"
        )
    if bool(artifacts["exact_dc_anchors"]) != bool(rtl_digest):
        raise ValueError(
            "publication pipeline exact_dc_anchors and rtl_source_tree_sha256 "
            "must be supplied together"
        )
    if artifacts["exact_dc_anchors"] and not artifacts["power_calibration"]:
        raise ValueError("publication pipeline exact DC anchors require power calibration")
    return pipeline, artifacts, resources


def _resolve_publication_pipeline_paths(
    artifacts: Mapping[str, Any],
    *,
    repository: Path,
    output_dir: Path,
) -> dict[str, Path | None]:
    resolved: dict[str, Path | None] = {}
    for name, value in artifacts.items():
        if value is None:
            resolved[name] = None
            continue
        if value.startswith("simulator://"):
            suffix = value.removeprefix("simulator://")
            relative = Path(suffix)
            if not suffix or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid simulator-bound path: {value!r}")
            simulator = Path(
                os.environ.get(
                    "PLENA_SIMULATOR_PATH",
                    str(
                        Path(__file__).resolve().parents[3]
                        / "PLENA_Simulator"
                    ),
                )
            ).resolve()
            path = (simulator / relative).resolve()
            try:
                path.relative_to(simulator)
            except ValueError as exc:
                raise ValueError(
                    f"simulator-bound path escapes its root: {value!r}"
                ) from exc
            resolved[name] = path
            continue
        resolved[name] = resolve_bound_path(
            value,
            repository_root=repository,
            workspace_root=output_dir,
        )
    return resolved


def _stage_partition_roots(
    output_dir: Path,
    stage: str,
    count: int,
) -> tuple[Path, ...]:
    if count == 1:
        return (output_dir / stage,)
    return tuple(
        output_dir / stage / f"part-{index:04d}-of-{count:04d}"
        for index in range(count)
    )


def _partitioned_artifact(path: Path, index: int, count: int) -> Path:
    if count == 1:
        return path
    return path.with_name(
        f"{path.stem}.part-{index:04d}-of-{count:04d}{path.suffix}"
    )


def _path_identity(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    if path.is_dir():
        files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
        digest = hashlib.sha256()
        for item in files:
            relative = item.relative_to(path).as_posix().encode("utf-8")
            payload_hash = _sha256_file(item).encode("ascii")
            digest.update(len(relative).to_bytes(8, "little"))
            digest.update(relative)
            digest.update(payload_hash)
        return {
            "path": str(path.resolve()),
            "kind": "directory",
            "file_count": len(files),
            "tree_sha256": digest.hexdigest(),
        }
    raise FileNotFoundError(path)


def validate_publication_evidence(
    *,
    config: Path,
    output_dir: Path,
    report: Path,
) -> Mapping[str, Any]:
    """Fail before GPU work when required physical evidence is unavailable."""

    repository = Path(__file__).resolve().parents[2]
    value = _load_config(config)
    _, artifact_values, resources = _publication_pipeline_config(value)
    paths = _resolve_publication_pipeline_paths(
        artifact_values,
        repository=repository,
        output_dir=output_dir,
    )

    def require(name: str) -> Path:
        path = paths[name]
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"required publication artifact {name} is missing: {path}"
            )
        return path

    timing_path = require("timing_evidence")
    from decode_dse.simulator_bridge import _disagg

    timing = _disagg().TimingEvidence.load(timing_path)
    if timing.mode not in ("rtl_serialized", "emulator_serialized") or not timing.passed:
        raise ValueError(
            "timing_evidence must be passing serialized-contract evidence "
            "(rtl_serialized or emulator_serialized) for the production study"
        )

    head_path = require("head_service_calibration")
    architecture = value.get("model_architecture")
    space = value.get("hardware_space")
    if not isinstance(architecture, Mapping) or not isinstance(space, Mapping):
        raise ValueError("model_architecture and hardware_space are required")
    batches = space.get("BATCH")
    if not isinstance(batches, list):
        raise ValueError("hardware_space.BATCH must be a list")
    from decode_dse.hardware.lm_head_service import (
        load_bf16_head_service_artifact,
    )

    head = load_bf16_head_service_artifact(
        head_path,
        model_name=str(value["model_name"]),
        model_revision=str(value["model_revision"]),
        hidden_size=int(architecture["hidden_size"]),
        vocab_size=int(architecture["vocab_size"]),
        tie_embeddings=bool(architecture["tie_word_embeddings"]),
        required_batches=tuple(int(batch) for batch in batches),
    )
    if not head.passed:
        raise ValueError(
            "the external BF16 output-head artifact is not publication-rankable; "
            "it must contain repeated and holdout measurements from the dedicated "
            "prefill-chip endpoint, including numerical logits, remote-link timing, "
            "component dynamic energy, and leakage evidence: "
            + "; ".join(head.failures)
        )

    required_inputs = {
        "admission_receipt",
        "compiler_trace_artifacts",
        "request_memory_calibration",
    }
    # refinement_validity is produced by joint selection during the run, so
    # its identity is sealed by that stage's receipt and the promotion's
    # validity hash; the pre-run evidence ledger covers only external inputs.
    optional_inputs = {
        "handoff_artifact",
        "power_calibration",
        "area_config",
        "exact_dc_anchors",
        "packedkv_evidence",
        "decode_analysis",
    }
    identities = {
        "timing_evidence": _path_identity(timing_path),
        "head_service_calibration": _path_identity(head_path),
    }
    for name in sorted(required_inputs):
        identities[name] = _path_identity(require(name))
    for name in sorted(optional_inputs):
        path = paths[name]
        if path is not None:
            identities[name] = _path_identity(require(name))
    timing_tier = str(resources["publication_timing_tier"])
    body = {
        "schema_version": "decode-publication-evidence-gate",
        "model_name": value["model_name"],
        "model_revision": value["model_revision"],
        "timing_evidence_id": timing.evidence_id,
        "timing_mode": timing.mode,
        "timing_evidence_tier": timing.evidence_tier,
        "publication_timing_tier": timing_tier,
        "compiler_trace_artifacts_role": (
            "exhaustive_study_pricing"
            if timing_tier == "compiler_trace_request_calibrated"
            else "spot_check_only"
        ),
        "head_service": head.to_dict(),
        "artifacts": identities,
    }
    write_immutable_json(report, body)
    return body


def pipeline_evidence_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=validate_publication_evidence.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    result = validate_publication_evidence(
        config=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        report=args.report.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def publication_gate_main(argv: Iterable[str] | None = None) -> int:
    """Seal the benchmark-selected hardware deployment or fail closed."""

    parser = argparse.ArgumentParser(description=publication_gate_main.__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--hardware-artifact",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    from decode_dse.software.benchmark_runner import (
        PUBLICATION_REPORT_SCHEMA,
        PublicationContract,
    )

    contract_value = load_immutable_json(args.contract)
    contract_value.pop("content_hash", None)
    contract = PublicationContract.from_dict(contract_value)
    report = load_immutable_json(args.report)
    if report.get("schema_version") != PUBLICATION_REPORT_SCHEMA:
        raise ValueError("unsupported publication benchmark report")
    if report.get("contract_hash") != contract.canonical_hash:
        raise ValueError("publication report differs from its contract")
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("publication accuracy gates did not select a deployment")
    frontier_fallback = "report_pareto_frontier_without_near_lossless_claim"
    # The benchmark report may legitimately decline the near-lossless claim;
    # that outcome is sealed as an explicit no-selection artifact rather than
    # failing the run, and only for the report's own declared fallback action.
    frontier_only = (
        selection.get("selected") is False
        and selection.get("failure_action") == frontier_fallback
    )
    if not frontier_only and selection.get("selected") is not True:
        raise ValueError("publication accuracy gates did not select a deployment")
    passing_ids = selection.get("accuracy_configuration_ids")
    if (
        not isinstance(passing_ids, list)
        or (not passing_ids and not frontier_only)
        or any(not isinstance(value, str) or not value for value in passing_ids)
        or len(passing_ids) != len(set(passing_ids))
    ):
        raise ValueError("publication report accuracy-pass coverage is invalid")
    configuration_by_id = {
        item.configuration_id: item for item in contract.configurations
    }
    if any(
        configuration_id not in configuration_by_id
        or configuration_by_id[configuration_id].role == "bf16"
        for configuration_id in passing_ids
    ):
        raise ValueError("publication report selected an unknown accuracy configuration")
    contract_order = [
        item.configuration_id
        for item in contract.configurations
        if item.configuration_id in set(passing_ids)
    ]
    if passing_ids != contract_order:
        raise ValueError("publication accuracy-pass configurations are reordered")
    from decode_dse.hardware.design_space import (
        HARDWARE_STORAGE_REVISION,
        load_hardware_artifact,
    )

    artifacts_by_sha: dict[str, tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]] = {}
    component_artifacts = []
    for path in args.hardware_artifact:
        artifact_sha256 = _sha256_file(path)
        if artifact_sha256 in artifacts_by_sha:
            raise ValueError("publication hardware artifacts are duplicated")
        header, rows = load_hardware_artifact(path)
        if header.get("storage_revision") != HARDWARE_STORAGE_REVISION:
            raise ValueError("publication requires factorized exact hardware artifacts")
        artifacts_by_sha[artifact_sha256] = (header, rows)
        component_artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": artifact_sha256,
                "run_id": header["run_id"],
                "factor_evaluation_count": header["factor_evaluation_count"],
                "factor_evaluation_sha256": header[
                    "factor_evaluation_sha256"
                ],
                "ordered_membership_map_sha256": header[
                    "ordered_membership_map_sha256"
                ],
                "expansion_contract_sha256": header[
                    "expansion_contract_sha256"
                ],
                "conceptual_result_count": header[
                    "conceptual_result_count"
                ],
            }
        )
    referenced_artifacts = {
        alternative.hardware_artifact_sha256
        for alternative in contract.hardware_alternatives
    }
    if set(artifacts_by_sha) != referenced_artifacts:
        raise ValueError("publication hardware artifact coverage differs from contract")

    joined_alternatives = []
    for alternative in contract.hardware_alternatives:
        configuration = configuration_by_id[alternative.configuration_id]
        header, rows = artifacts_by_sha[alternative.hardware_artifact_sha256]
        matched = tuple(
            row for row in rows if row.get("record_hash") == alternative.record_hash
        )
        if len(matched) != 1:
            raise ValueError("publication hardware row is missing or duplicated")
        row = matched[0]
        labels = row.get("retention_labels")
        metrics = row.get("metrics")
        whole = metrics.get("whole_model") if isinstance(metrics, Mapping) else None
        energy = (
            whole.get("calibrated_energy")
            if isinstance(whole, Mapping)
            else None
        )
        source_profile = getattr(configuration.profile, "source_profile", None)
        observed_source_id = (
            source_profile.profile_id
            if source_profile is not None
            else configuration.profile.profile_id
        )
        if (
            not isinstance(labels, list)
            or "profile_frontier" not in labels
            or row.get("deployment_valid") is not True
            or row.get("profile_id") != alternative.profile_id
            or row.get("profile") != configuration.profile.to_dict()
            or row.get("candidate_id") != alternative.candidate_id
            or alternative.source_profile_id != observed_source_id
            or not isinstance(whole, Mapping)
            or whole.get("rankable") is not True
            or not isinstance(energy, Mapping)
            or energy.get("energy_tier") != alternative.energy_tier
            or not math.isclose(
                float(whole.get("tpot_ms", math.nan)),
                alternative.tpot_ms,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(energy.get("total_j", math.nan)),
                alternative.energy_per_token_j,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            raise ValueError("publication alternative differs from exact hardware evidence")
        if alternative.configuration_id in passing_ids:
            joined_alternatives.append((configuration, alternative, row, header))
    if frontier_only:
        result = {
            "schema_version": "decode-final-publication-selection",
            "contract_hash": contract.canonical_hash,
            "contract_sha256": _sha256_file(args.contract),
            "benchmark_report_sha256": _sha256_file(args.report),
            "accuracy_pass_configuration_ids": passing_ids,
            "hardware_artifacts": component_artifacts,
            "selection": {
                "selected": False,
                "failure_action": frontier_fallback,
            },
        }
        write_immutable_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not joined_alternatives:
        raise ValueError("no hardware alternatives cover passing accuracy configurations")

    selected_configuration, selected_alternative, selected_row, selected_header = min(
        joined_alternatives,
        key=lambda item: (
            0 if item[1].energy_tier == "dc_calibrated" else 1,
            item[1].energy_per_token_j,
            item[1].tpot_ms,
            item[0].configuration_id,
            item[1].profile_id,
            item[1].candidate_id,
            item[1].record_hash,
        ),
    )
    final_selection = {
        "configuration_id": selected_configuration.configuration_id,
        "role": selected_configuration.role,
        "alternative_id": selected_alternative.alternative_id,
        "profile_id": selected_alternative.profile_id,
        "source_profile_id": selected_alternative.source_profile_id,
        "candidate_id": selected_alternative.candidate_id,
        "hardware_record_hash": selected_alternative.record_hash,
        "hardware_artifact_sha256": (
            selected_alternative.hardware_artifact_sha256
        ),
        "tpot_ms": selected_alternative.tpot_ms,
        "energy_per_token_j": selected_alternative.energy_per_token_j,
        "energy_tier": selected_alternative.energy_tier,
        "retention_labels": list(selected_row["retention_labels"]),
        "factor_evaluation_sha256": selected_header[
            "factor_evaluation_sha256"
        ],
        "ordered_membership_map_sha256": selected_header[
            "ordered_membership_map_sha256"
        ],
        "expansion_contract_sha256": selected_header[
            "expansion_contract_sha256"
        ],
    }
    result = {
        "schema_version": "decode-final-publication-selection",
        "contract_hash": contract.canonical_hash,
        "contract_sha256": _sha256_file(args.contract),
        "benchmark_report_sha256": _sha256_file(args.report),
        "accuracy_pass_configuration_ids": passing_ids,
        "hardware_join": {
            "candidate_count": len(joined_alternatives),
            "alternative_ids": [
                alternative.alternative_id
                for _, alternative, _, _ in joined_alternatives
            ],
            "selection_order": [
                "energy_tier_dc_before_analytic",
                "energy_per_token_j",
                "tpot_ms",
                "configuration_id",
                "profile_id",
                "candidate_id",
                "record_hash",
            ],
            "accuracy_and_hardware_joined_after_accuracy_gates": True,
        },
        "hardware_artifacts": component_artifacts,
        "selection": final_selection,
    }
    write_immutable_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_pipeline(
    *,
    config: Path,
    output_dir: Path,
    device_label: str,
    gpus: tuple[str, ...],
    plan: SweepRunPlan,
    stack_validity: Path | None = None,
) -> tuple[PipelineCommand, ...]:
    """Build the restartable publication workflow from declared artifacts."""

    repository = Path(__file__).resolve().parents[2]
    workers = str(len(gpus))
    correctness = output_dir / "preflight_correctness.json"
    evidence = output_dir / "preflight_evidence.json"
    report = output_dir / "preflight_gate_report.json"
    publication_evidence = output_dir / "publication_evidence_gate.json"
    stack_validity = (
        output_dir / "stack_validity.json" if stack_validity is None else stack_validity
    )
    config_value = json.loads(config.read_text(encoding="utf-8"))
    _, artifact_values, resources = _publication_pipeline_config(config_value)
    paths = _resolve_publication_pipeline_paths(
        artifact_values,
        repository=repository,
        output_dir=output_dir,
    )

    def required_path(name: str) -> Path:
        value = paths[name]
        if value is None:
            raise AssertionError(f"required publication path {name} is null")
        return value

    executor = config_value.get("executor")
    if not isinstance(executor, Mapping):
        raise ValueError("config.executor is required")
    sample_bundle = resolve_bound_path(
        str(executor.get("sample_bundle", "")),
        repository_root=repository,
        workspace_root=output_dir,
    )
    refinement = config_value.get("refinement")
    if not isinstance(refinement, Mapping):
        raise ValueError("config.refinement is required")
    refinement_paths = {
        name: resolve_bound_path(
            str(refinement.get(name, "")),
            repository_root=repository,
            workspace_root=output_dir,
        )
        for name in (
            "sample_bundle",
            "prefill_artifact_root",
            "admission_artifact_root",
            "calibration_artifact",
            "calibration_receipt",
            "checkpoint_root",
        )
    }
    numerical_roots = _stage_partition_roots(
        output_dir,
        "numerical-screen",
        len(gpus),
    )
    validation_roots = _stage_partition_roots(
        output_dir,
        "hardware-validation",
        len(gpus),
    )
    hardware_studies = tuple(
        _partitioned_artifact(
            required_path("hardware_study"),
            index,
            len(numerical_roots),
        )
        for index in range(len(numerical_roots))
    )
    refinement_output_root = required_path("refinement_results")
    refinement_merge_receipt = refinement_output_root / "merged" / "merge.json"
    refinement_merged_results = (
        refinement_output_root / "merged" / "results.jsonl"
    )
    refined_hardware_study = required_path("refined_hardware_study")
    baseline_root = output_dir / "gpu_baseline"
    baseline_contract = baseline_root / "contract.json"
    baseline_report = baseline_root / "report.json"
    baseline_receipt = baseline_root / "stage_receipt.json"
    baseline_results = tuple(
        baseline_root / f"batch-{batch_size}.json"
        for batch_size in plan.gpu_baseline.batch_sizes
    )
    common_shards = (
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--device-label",
        device_label,
        "--gpus",
        ",".join(gpus),
    )
    baseline_commands = (
        PipelineCommand(
            "gpu-baseline-contract",
            _module_command(
                "decode_dse.software.gpu_baseline",
                "prepare",
                "--config",
                str(config),
                "--sample-bundle",
                str(sample_bundle),
                "--provenance",
                str(output_dir / "provenance.json"),
                "--output",
                str(baseline_contract),
                "--attention-implementation",
                plan.gpu_baseline.attention_implementation,
                "--warmup-steps",
                str(plan.gpu_baseline.warmup_steps),
                "--measured-steps",
                str(plan.gpu_baseline.measured_steps),
                "--repetitions",
                str(plan.gpu_baseline.repetitions),
                "--device-labels",
                device_label,
                "--batch-sizes",
                *(str(value) for value in plan.gpu_baseline.batch_sizes),
                "--energy-meter-priority",
                *plan.gpu_baseline.energy_meter_priority,
                "--power-trace-sample-interval-ms",
                str(plan.gpu_baseline.power_trace_sample_interval_ms),
            ),
            outputs=(baseline_contract,),
        ),
        *(
            PipelineCommand(
                f"gpu-baseline-batch-{batch_size}",
                _module_command(
                    "decode_dse.software.gpu_baseline",
                    "run",
                    "--config",
                    str(config),
                    "--sample-bundle",
                    str(sample_bundle),
                    "--contract",
                    str(baseline_contract),
                    "--device",
                    "cuda:0",
                    "--device-label",
                    device_label,
                    "--batch-size",
                    str(batch_size),
                    "--output",
                    str(result_path),
                ),
                first_gpu_only=True,
                outputs=(result_path,),
            )
            for batch_size, result_path in zip(
                plan.gpu_baseline.batch_sizes,
                baseline_results,
            )
        ),
        PipelineCommand(
            "gpu-baseline-report",
            _module_command(
                "decode_dse.software.gpu_baseline",
                "merge",
                "--contract",
                str(baseline_contract),
                "--results",
                *(str(path) for path in baseline_results),
                "--output",
                str(baseline_report),
            ),
            outputs=(baseline_report,),
        ),
        PipelineCommand(
            "gpu-baseline-receipt",
            _module_command(
                "decode_dse.software.gpu_baseline",
                "receipt",
                "--report",
                str(baseline_report),
                "--provenance",
                str(output_dir / "provenance.json"),
                "--output",
                str(baseline_receipt),
            ),
            outputs=(baseline_receipt,),
        ),
    )
    commands: list[PipelineCommand] = [
        PipelineCommand(
            "compiler-trace-artifacts",
            _module_command(
                "decode_dse.software.sweep",
                "compiler-trace-artifacts",
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
                "--output",
                str(required_path("compiler_trace_artifacts")),
            ),
            outputs=(required_path("compiler_trace_artifacts"),),
        ),
        PipelineCommand(
            "publication-evidence-gate",
            _module_command(
                "decode_dse.software.sweep",
                "pipeline-evidence",
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
                "--report",
                str(publication_evidence),
            ),
            outputs=(publication_evidence,),
        ),
        *(
            (
                PipelineCommand(
                    "publication-benchmark-manifest",
                    _module_command(
                        "decode_dse.software.benchmark_runner",
                        "manifest",
                        "--config",
                        str(config),
                        "--output",
                        str(required_path("publication_benchmarks")),
                    ),
                    outputs=(required_path("publication_benchmarks"),),
                ),
                PipelineCommand(
                    "publication-chat-template",
                    _module_command(
                        "decode_dse.software.benchmark_runner",
                        "chat-template",
                        "--config",
                        str(config),
                        "--output",
                        str(required_path("publication_chat_template")),
                    ),
                    outputs=(required_path("publication_chat_template"),),
                ),
            )
            if resources["publication_enabled"]
            else ()
        ),
        PipelineCommand(
            "preflight-numerical-screen-fidelity",
            _module_command(
                "decode_dse.software.sweep",
                "shards",
                *common_shards,
                "--stage",
                "preflight",
            ),
            outputs=_stage_partition_roots(output_dir, "preflight", len(gpus)),
        ),
        PipelineCommand(
            "validation-pilot-fidelity",
            _module_command(
                "decode_dse.software.sweep",
                "shards",
                *common_shards,
                "--stage",
                "validation-pilot",
            ),
            outputs=_stage_partition_roots(
                output_dir,
                "validation-pilot",
                len(gpus),
            ),
        ),
        PipelineCommand(
            "correctness",
            _module_command(
                "decode_dse.software.preflight",
                "check",
                "run",
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
                "--device-label",
                device_label,
                "--out",
                str(correctness),
            ),
            first_gpu_only=True,
            outputs=(correctness,),
        ),
        PipelineCommand(
            "build-preflight-evidence",
            _module_command(
                "decode_dse.software.preflight",
                "evidence",
                "--output-dir",
                str(output_dir),
                "--correctness",
                str(correctness),
                "--stack-validity",
                str(stack_validity),
                "--numerical-screen-workers",
                workers,
                "--hardware-validation-workers",
                workers,
                "--out",
                str(evidence),
            ),
            outputs=(evidence,),
        ),
        PipelineCommand(
            "validate-preflight",
            _module_command(
                "decode_dse.software.sweep",
                "stage",
                "validate",
                "--output-dir",
                str(output_dir),
                "--evidence",
                str(evidence),
                "--report",
                str(report),
            ),
            outputs=(report,),
        ),
        *baseline_commands,
        PipelineCommand(
            "numerical-screen",
            _module_command(
                "decode_dse.software.sweep",
                "shards",
                *common_shards,
                "--stage",
                "numerical-screen",
                "--evidence",
                str(evidence),
            ),
            outputs=numerical_roots,
        ),
        PipelineCommand(
            "hardware-validation",
            _module_command(
                "decode_dse.software.sweep",
                "shards",
                *common_shards,
                "--stage",
                "hardware-validation",
                "--evidence",
                str(evidence),
            ),
            outputs=validation_roots,
        ),
    ]

    # The study prices hardware candidates and gates them on relative
    # perplexity, so it needs the BF16 reference and one consistent sample
    # contract. The hardware-validation set is deliberately built from
    # hardware candidates and legal vector controls only and never contains
    # the reference; the numerical-screen shards carry it alongside every
    # accuracy row, and the study filters to hardware candidates itself.
    # Hardware-validation results reach selection and the figures directly.
    for index, (screen_root, hardware_study) in enumerate(
        zip(numerical_roots, hardware_studies)
    ):
        hardware_arguments = [
            "--manifest",
            str(screen_root / "manifest.json"),
            "--numerical-jsonl",
            str(screen_root),
            "--config",
            str(config),
            "--timing-evidence",
            str(required_path("timing_evidence")),
            *_hardware_timing_arguments(resources, required_path),
            "--stride",
            str(resources["stride"]),
            "--runtime-hbm-reserve-bytes",
            str(resources["runtime_hbm_reserve_bytes"]),
            "--head-service-calibration",
            str(required_path("head_service_calibration")),
            "--admission-receipt",
            str(required_path("admission_receipt")),
            "--parallel-workers",
            str(resources["study_parallel_workers"]),
            "--output",
            str(hardware_study),
        ]
        optional_hardware_paths = (
            ("handoff_artifact", "--handoff-artifact"),
            ("power_calibration", "--power-calibration"),
            ("area_config", "--area-config"),
            ("exact_dc_anchors", "--exact-dc-anchors"),
        )
        for name, flag in optional_hardware_paths:
            value = paths[name]
            if value is not None:
                hardware_arguments.extend((flag, str(value)))
        if resources["rtl_source_tree_sha256"] is not None:
            hardware_arguments.extend(
                (
                    "--rtl-source-tree-sha256",
                    str(resources["rtl_source_tree_sha256"]),
                )
            )
        commands.append(
            PipelineCommand(
                f"exact-hardware-study-part-{index:04d}",
                _module_command(
                    "decode_dse.hardware.evaluation",
                    *hardware_arguments,
                ),
                outputs=(
                    hardware_study,
                    hardware_study.with_name(f"{hardware_study.name}.meta.json"),
                ),
            )
        )

    schedule_arguments = [
        "--manifest",
        str(output_dir / "manifest.json"),
        "--run-plan",
        str(output_dir / "run_plan.json"),
    ]
    for path in numerical_roots:
        schedule_arguments.extend(("--numerical-screen-results", str(path)))
    for path in validation_roots:
        schedule_arguments.extend(("--hardware-validation-results", str(path)))
    for path in hardware_studies:
        schedule_arguments.extend(("--hardware-study", str(path)))
    if resources["refinement_enabled"]:
        if paths["refinement_validity"] is None:
            raise ValueError(
                "refinement requires artifacts.refinement_validity so joint "
                "selection can derive and record measured stack validity"
            )
        schedule_arguments.extend(
            (
                "--stack-validity",
                str(stack_validity),
                "--validity-output",
                str(paths["refinement_validity"]),
            )
        )
    accuracy_budgets = config.get("accuracy_budgets")
    if accuracy_budgets is not None:
        if not isinstance(accuracy_budgets, Mapping):
            raise ValueError("accuracy_budgets must be a mapping when present")
        schedule_arguments.extend(
            (
                "--strict-relative-perplexity",
                str(float(accuracy_budgets["strict_relative_perplexity"])),
                "--relaxed-relative-perplexity",
                str(float(accuracy_budgets["relaxed_relative_perplexity"])),
            )
        )
    schedule_arguments.extend(
        (
            "--schedule",
            str(required_path("refinement_schedule")),
            "--promotion",
            str(required_path("refinement_promotion")),
        )
    )
    commands.append(
        PipelineCommand(
            "joint-selection",
            _module_command(
                "decode_dse.software.refinement_schedule",
                *schedule_arguments,
            ),
            outputs=(
                required_path("refinement_schedule"),
                required_path("refinement_promotion"),
            )
            + (
                (paths["refinement_validity"],)
                if resources["refinement_enabled"]
                else ()
            ),
        )
    )

    if resources["refinement_enabled"]:
        commands.extend(
            (
                PipelineCommand(
                    "refinement-samples",
                    _module_command(
                        "decode_dse.software.refinement_runner",
                        "prepare",
                        "samples",
                        "--config",
                        str(config),
                        "--source-bundle",
                        str(sample_bundle),
                        "--output",
                        str(refinement_paths["sample_bundle"]),
                    ),
                    outputs=(refinement_paths["sample_bundle"],),
                ),
                PipelineCommand(
                    "refinement-prefill",
                    _module_command(
                        "decode_dse.software.refinement_runner",
                        "prepare",
                        "prefill",
                        "--config",
                        str(config),
                        "--sample-bundle",
                        str(refinement_paths["sample_bundle"]),
                        "--artifact-root",
                        str(refinement_paths["prefill_artifact_root"]),
                    ),
                    first_gpu_only=True,
                    outputs=(
                        refinement_paths["prefill_artifact_root"] / "index.json",
                    ),
                ),
                PipelineCommand(
                    "refinement-calibration",
                    _module_command(
                        "decode_dse.software.refinement_runner",
                        "prepare",
                        "calibration",
                        "--config",
                        str(config),
                        "--output",
                        str(refinement_paths["calibration_artifact"]),
                        "--receipt",
                        str(refinement_paths["calibration_receipt"]),
                    ),
                    first_gpu_only=True,
                    outputs=(
                        refinement_paths["calibration_artifact"],
                        refinement_paths["calibration_receipt"],
                    ),
                ),
                PipelineCommand(
                    "refinement-evaluation",
                    _module_command(
                        "decode_dse.software.refinement_runner",
                        "launch",
                        "--config",
                        str(config),
                        "--schedule",
                        str(required_path("refinement_schedule")),
                        "--sample-bundle",
                        str(refinement_paths["sample_bundle"]),
                        "--prefill-root",
                        str(refinement_paths["prefill_artifact_root"]),
                        "--admission-root",
                        str(refinement_paths["admission_artifact_root"]),
                        "--calibration",
                        str(refinement_paths["calibration_artifact"]),
                        "--calibration-receipt",
                        str(refinement_paths["calibration_receipt"]),
                        "--checkpoint-root",
                        str(refinement_paths["checkpoint_root"]),
                        "--output-root",
                        str(refinement_output_root),
                        "--work-root",
                        str(refinement_output_root.with_name("work")),
                        "--device-label",
                        device_label,
                        "--gpus",
                        ",".join(gpus),
                        "--decode-microbatch-size",
                        str(resources["refinement_decode_microbatch_size"]),
                        "--bootstrap-replicates",
                        str(resources["bootstrap_replicates"]),
                    ),
                    outputs=(refinement_output_root,),
                ),
            )
        )

        refined_hardware_arguments = [
            "--manifest",
            str(output_dir / "manifest.json"),
            "--refinement-schedule",
            str(required_path("refinement_schedule")),
            "--refinement-merge",
            str(refinement_merge_receipt),
            "--refinement-results",
            str(refinement_merged_results),
            "--config",
            str(config),
            "--timing-evidence",
            str(required_path("timing_evidence")),
            *_hardware_timing_arguments(resources, required_path),
            "--stride",
            str(resources["stride"]),
            "--runtime-hbm-reserve-bytes",
            str(resources["runtime_hbm_reserve_bytes"]),
            "--head-service-calibration",
            str(required_path("head_service_calibration")),
            "--admission-receipt",
            str(required_path("admission_receipt")),
            "--output",
            str(refined_hardware_study),
        ]
        for name, flag in (
            ("handoff_artifact", "--handoff-artifact"),
            ("power_calibration", "--power-calibration"),
            ("area_config", "--area-config"),
            ("exact_dc_anchors", "--exact-dc-anchors"),
        ):
            value = paths[name]
            if value is not None:
                refined_hardware_arguments.extend((flag, str(value)))
        if resources["rtl_source_tree_sha256"] is not None:
            refined_hardware_arguments.extend(
                (
                    "--rtl-source-tree-sha256",
                    str(resources["rtl_source_tree_sha256"]),
                )
            )
        commands.extend(
            (
                PipelineCommand(
                    "refined-hardware-study",
                    _module_command(
                        "decode_dse.hardware.evaluation",
                        *refined_hardware_arguments,
                    ),
                    outputs=(
                        refined_hardware_study,
                        refined_hardware_study.with_name(
                            f"{refined_hardware_study.name}.meta.json"
                        ),
                    ),
                ),
                PipelineCommand(
                    "publication-configurations",
                    _module_command(
                        "decode_dse.software.benchmark_runner",
                        "configurations",
                        "--manifest",
                        str(output_dir / "manifest.json"),
                        "--refinement-schedule",
                        str(required_path("refinement_schedule")),
                        "--source-selection",
                        str(required_path("refinement_promotion")),
                        "--refinement-merge",
                        str(refinement_merge_receipt),
                        "--refinement-results",
                        str(refinement_merged_results),
                        "--hardware-artifact",
                        str(refined_hardware_study),
                        "--publication-timing-tier",
                        str(resources["publication_timing_tier"]),
                        "--output",
                        str(required_path("publication_configurations")),
                    ),
                    outputs=(required_path("publication_configurations"),),
                ),
            )
        )

    if resources["publication_enabled"]:
        commands.extend(
            (
                PipelineCommand(
                    "publication-contract",
                    _module_command(
                        "decode_dse.software.benchmark_runner",
                        "contract",
                        "--config",
                        str(config),
                        "--configurations",
                        str(required_path("publication_configurations")),
                        "--benchmarks",
                        str(required_path("publication_benchmarks")),
                        "--chat-template",
                        str(required_path("publication_chat_template")),
                        "--output",
                        str(required_path("publication_contract")),
                    ),
                    outputs=(required_path("publication_contract"),),
                ),
                PipelineCommand(
                    "publication-benchmarks",
                    _module_command(
                        "decode_dse.software.benchmark_runner",
                        "run",
                        "--config",
                        str(config),
                        "--contract",
                        str(required_path("publication_contract")),
                        "--executor",
                        str(resources["publication_executor"]),
                        "--output-dir",
                        str(required_path("publication_results")),
                        "--bootstrap-replicates",
                        str(resources["bootstrap_replicates"]),
                    ),
                    first_gpu_only=True,
                    outputs=(
                        required_path("publication_results")
                        / "publication_report.json",
                    ),
                ),
                PipelineCommand(
                    "final-publication-selection",
                    _module_command(
                        "decode_dse.software.sweep",
                        "publication-gate",
                        "--contract",
                        str(required_path("publication_contract")),
                        "--report",
                        str(
                            required_path("publication_results")
                            / "publication_report.json"
                        ),
                        "--hardware-artifact",
                        str(refined_hardware_study),
                        "--output",
                        str(required_path("final_selection")),
                    ),
                    outputs=(required_path("final_selection"),),
                ),
            )
        )

    plot_arguments = [
        "--manifest",
        str(output_dir / "manifest.json"),
    ]
    for path in numerical_roots:
        plot_arguments.extend(("--numerical", str(path)))
    for path in validation_roots:
        plot_arguments.extend(("--validation-numerical", str(path)))
    for path in hardware_studies:
        plot_arguments.extend(("--hardware-artifact", str(path)))
    for name, flag in (
        ("packedkv_evidence", "--packedkv-evidence"),
        ("decode_analysis", "--decode-analysis"),
    ):
        if paths[name] is not None:
            plot_arguments.extend((flag, str(paths[name])))
    # The measured GPU baseline always exists, so the figure stage always
    # receives it: the analytic energy context and dual-accuracy envelopes
    # do not depend on the publication benchmark stages being enabled.
    plot_arguments.extend(
        (
            "--config",
            str(config),
            "--gpu-baseline-report",
            str(baseline_report),
            "--gpu-baseline-receipt",
            str(baseline_receipt),
        )
    )
    if resources["publication_enabled"]:
        plot_arguments.extend(
            (
                "--publication-contract",
                str(required_path("publication_contract")),
                "--publication-report",
                str(
                    required_path("publication_results")
                    / "publication_report.json"
                ),
                "--final-selection",
                str(required_path("final_selection")),
                "--refined-hardware-artifact",
                str(refined_hardware_study),
            )
        )
    plot_arguments.extend(
        (
            "--output-dir",
            str(required_path("figures")),
            "--formats",
            *(str(value) for value in resources["figure_formats"]),
        )
    )
    commands.append(
        PipelineCommand(
            "publication-figures",
            _module_command("decode_dse.plots", *plot_arguments),
            outputs=(required_path("figures"),),
        )
    )
    return tuple(commands)


def _resolve_config_path(
    repository: Path,
    output_dir: Path,
    value: str,
) -> Path:
    return resolve_bound_path(
        value,
        repository_root=repository,
        workspace_root=output_dir,
    )


def _validate_bound_launch_artifacts(
    *,
    repository: Path,
    config: Path,
    value: Mapping[str, object],
    output_dir: Path,
    plan: SweepRunPlan,
    validity_path: Path,
) -> None:
    """Validate immutable numerical, admission, and stack-evidence bindings."""

    validate_sweep_config(value)
    seed = int(value.get("seed", 0))
    initialize_numerical_runtime(seed)
    current_runtime = capture_runtime_environment(
        str(value.get("device", "cuda:0")),
        seed=seed,
    )
    require_runtime_environment(
        output_dir / "runtime_environment.json",
        current_runtime,
    )
    manifest = load_manifest(output_dir / "manifest.json")
    prompts = _load_prompts(output_dir / "prompt_manifest.json")
    validate_run_plan(plan, manifest)
    _validate_provenance(
        output_dir / "provenance.json",
        repository=repository,
        config_path=config,
        manifest=manifest,
        plan=plan,
        prompts=prompts,
    )

    executor = value.get("executor")
    if not isinstance(executor, Mapping):
        raise ValueError("config.executor is required")
    sample_path = _resolve_config_path(
        repository,
        output_dir,
        str(executor.get("sample_bundle", "")),
    )
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    bundle = load_sample_bundle(sample_path)
    if bundle.prompt_manifest() != prompts:
        raise ValueError("sample bundle differs from the immutable workspace")

    prefill_root = _resolve_config_path(
        repository,
        output_dir,
        str(executor.get("prefill_artifact_root", "")),
    )
    if not (prefill_root / "index.json").is_file():
        raise FileNotFoundError(prefill_root / "index.json")

    expected_documents = (
        NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count
        + HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
    )
    receipt = load_immutable_json(output_dir / "admission_preparation.json")
    if (
        receipt.get("schema_version") != "decode-admission-preparation"
        or receipt.get("manifest_hash") != manifest.canonical_hash
        or receipt.get("run_plan_hash") != plan.canonical_hash
        or receipt.get("prompt_manifest_hash") != prompts.canonical_hash
        or receipt.get("quantized_format_count") != len(DECODE_FORMATS)
        or receipt.get("document_count") != expected_documents
        or receipt.get("artifact_count")
        != expected_documents * (len(DECODE_FORMATS) + 1)
        or receipt.get("runtime_environment_fingerprint")
        != current_runtime.logical_fingerprint
    ):
        raise ValueError("admission preparation is not bound to the workspace")
    admission_index = Path(str(receipt.get("admission_index_path", ""))).resolve()
    if not admission_index.is_file():
        raise FileNotFoundError(admission_index)
    index = load_immutable_json(admission_index)
    if (
        index.get("content_hash") != receipt.get("admission_index_hash")
        or index.get("sample_bundle_hash") != bundle.canonical_hash
        or index.get("document_count") != expected_documents
        or index.get("artifact_count")
        != expected_documents * (len(DECODE_FORMATS) + 1)
    ):
        raise ValueError("admission index differs from its workspace receipt")

    load_built_stack_validity(
        validity_path,
        manifest=manifest,
        scope_profile_ids=plan.hardware_validation_profile_ids,
        required_stages=("compiler", "emulator"),
        scope_name="hardware-validation",
        run_plan_hash=plan.canonical_hash,
    )


def _validate_pipeline_shape(
    *,
    config: Path,
    output_dir: Path,
    device_label: str,
    gpus: tuple[str, ...],
) -> tuple[Mapping[str, Any], SweepRunPlan]:
    """Validate the CUDA-free pipeline identity and shard shape."""

    if not config.is_file():
        raise FileNotFoundError(config)
    value = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("config must contain a JSON object")
    architecture = value.get("model_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("the gated pipeline requires model_architecture")
    plan_path = output_dir / "run_plan.json"
    plan = _load_plan(plan_path)
    if plan.device_labels != (device_label,):
        raise ValueError("the pipeline requires one homogeneous planned GPU type")
    if plan.numerical_screen_workers != len(
        gpus
    ) or plan.hardware_validation_workers != len(gpus):
        raise ValueError(
            "GPU count must equal both immutable numerical screen and hardware validation worker counts"
        )
    return value, plan


def validate_inputs(
    *,
    repository: Path,
    config: Path,
    output_dir: Path,
    device_label: str,
    gpus: tuple[str, ...],
) -> SweepRunPlan:
    """Reject an incomplete or differently sharded immutable workspace."""

    value, plan = _validate_pipeline_shape(
        config=config,
        output_dir=output_dir,
        device_label=device_label,
        gpus=gpus,
    )
    compiler_trace_preflight = _compiler_trace_feasibility(
        value,
        _build_manifest(value, repository),
    )
    _require_compiler_trace_feasible(compiler_trace_preflight)
    for name in (
        "manifest.json",
        "prompt_manifest.json",
        "provenance.json",
        "admission_preparation.json",
        "runtime_environment.json",
    ):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
    executor = value.get("executor")
    if not isinstance(executor, Mapping):
        raise ValueError("config.executor is required")
    validity = executor.get("stack_validity_manifest")
    if not isinstance(validity, str) or not validity:
        raise ValueError("stack-validity artifact path is required")
    validity_path = _resolve_config_path(repository, output_dir, validity)
    if not validity_path.is_file():
        raise FileNotFoundError(validity_path)
    _validate_bound_launch_artifacts(
        repository=repository,
        config=config,
        value=value,
        output_dir=output_dir,
        plan=plan,
        validity_path=validity_path,
    )
    return plan


def _pipeline_output_identity(path: Path) -> dict[str, Any]:
    """Return a content identity for one declared pipeline output."""

    try:
        root_status = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"pipeline command did not create declared output: {path}"
        ) from exc
    if stat.S_ISLNK(root_status.st_mode):
        raise ValueError(f"pipeline output cannot be a symbolic link: {path}")
    if stat.S_ISREG(root_status.st_mode):
        return {
            "path": str(path.resolve()),
            "kind": "file",
            "size_bytes": root_status.st_size,
            "sha256": _sha256_file(path),
        }
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"pipeline output has an unsupported file type: {path}")

    digest = hashlib.sha256()
    digest.update(b"decode-pipeline-output-tree-v1\0")
    entry_count = 0
    file_count = 0
    directory_count = 0
    size_bytes = 0

    def visit(directory: Path, relative_directory: Path) -> None:
        nonlocal entry_count, file_count, directory_count, size_bytes
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            relative = relative_directory / entry.name
            relative_name = relative.as_posix()
            entry_status = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_status.st_mode):
                raise ValueError(
                    "pipeline output directory cannot contain symbolic links: "
                    f"{entry.path}"
                )
            if stat.S_ISREG(entry_status.st_mode):
                entry_path = Path(entry.path)
                file_hash = _sha256_file(entry_path)
                observed_status = entry_path.stat(follow_symlinks=False)
                before = (
                    entry_status.st_dev,
                    entry_status.st_ino,
                    entry_status.st_size,
                    entry_status.st_mtime_ns,
                    entry_status.st_ctime_ns,
                )
                after = (
                    observed_status.st_dev,
                    observed_status.st_ino,
                    observed_status.st_size,
                    observed_status.st_mtime_ns,
                    observed_status.st_ctime_ns,
                )
                if before != after:
                    raise RuntimeError(
                        f"pipeline output changed while it was hashed: {entry.path}"
                    )
                record = {
                    "path": relative_name,
                    "kind": "file",
                    "size_bytes": entry_status.st_size,
                    "sha256": file_hash,
                }
                file_count += 1
                size_bytes += entry_status.st_size
            elif stat.S_ISDIR(entry_status.st_mode):
                record = {
                    "path": relative_name,
                    "kind": "directory",
                }
                directory_count += 1
            else:
                raise ValueError(
                    "pipeline output directory contains an unsupported file type: "
                    f"{entry.path}"
                )
            digest.update(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            digest.update(b"\n")
            entry_count += 1
            if record["kind"] == "directory":
                visit(Path(entry.path), relative)

    visit(path, Path())
    return {
        "path": str(path.resolve()),
        "kind": "directory",
        "tree_schema": "decode-pipeline-output-tree-v1",
        "entry_count": entry_count,
        "file_count": file_count,
        "directory_count": directory_count,
        "size_bytes": size_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def run_pipeline(
    *,
    repository: Path,
    output_dir: Path,
    commands: Sequence[PipelineCommand],
    gpus: tuple[str, ...],
) -> None:
    """Execute commands synchronously with immutable command-level receipts."""

    pipeline_root = output_dir / "pipeline"
    completion_root = pipeline_root / "completed"
    contract_body = {
        "schema_version": "decode-publication-pipeline-contract",
        "commands": [
            {
                "ordinal": index,
                "name": command.name,
                "command_id": command.command_id,
                "argv": list(command.argv),
                "first_gpu_only": command.first_gpu_only,
                "outputs": [str(path) for path in command.outputs],
            }
            for index, command in enumerate(commands)
        ],
    }
    contract_path = write_immutable_json(
        pipeline_root / "contract.json",
        contract_body,
    )
    contract = load_immutable_json(contract_path)
    contract_hash = str(contract["content_hash"])

    completion_paths: list[Path] = []

    for index, command in enumerate(commands, start=1):
        completion_path = (
            completion_root / f"{index - 1:03d}-{command.name}.json"
        )
        if completion_path.is_file():
            completion = load_immutable_json(completion_path)
            if (
                completion.get("schema_version")
                != "decode-publication-pipeline-command"
                or completion.get("contract_hash") != contract_hash
                or completion.get("ordinal") != index - 1
                or completion.get("name") != command.name
                or completion.get("command_id") != command.command_id
            ):
                raise ValueError(
                    f"pipeline completion receipt changed for {command.name}"
                )
            observed = [
                _pipeline_output_identity(path) for path in command.outputs
            ]
            if completion.get("outputs") != observed:
                raise ValueError(
                    f"pipeline output identity changed for {command.name}"
                )
            completion_paths.append(completion_path)
            print(
                f"[{index}/{len(commands)}] {command.name} (receipt verified)",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(commands)}] {command.name}",
            flush=True,
        )
        environment = os.environ.copy()
        if command.first_gpu_only:
            environment["CUDA_VISIBLE_DEVICES"] = gpus[0]
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        subprocess.run(
            command.argv,
            cwd=repository,
            env=environment,
            check=True,
        )
        outputs = [
            _pipeline_output_identity(path) for path in command.outputs
        ]
        write_immutable_json(
            completion_path,
            {
                "schema_version": "decode-publication-pipeline-command",
                "contract_hash": contract_hash,
                "ordinal": index - 1,
                "name": command.name,
                "command_id": command.command_id,
                "outputs": outputs,
            },
        )
        completion_paths.append(completion_path)

    write_immutable_json(
        pipeline_root / "receipt.json",
        {
            "schema_version": "decode-publication-pipeline-receipt",
            "contract_hash": contract_hash,
            "command_count": len(commands),
            "completed": [
                {
                    "ordinal": index,
                    "name": commands[index].name,
                    "receipt_sha256": _sha256_file(path),
                }
                for index, path in enumerate(completion_paths)
            ],
        },
    )


def sweep_pipeline_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    repository = Path(__file__).resolve().parents[2]
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    try:
        gpus = parse_gpu_list(args.gpus)
        if args.dry_run:
            config_value, plan = _validate_pipeline_shape(
                config=config,
                output_dir=output_dir,
                device_label=args.device_label,
                gpus=gpus,
            )
            compiler_trace_preflight = _compiler_trace_feasibility(
                config_value,
                _build_manifest(config_value, repository),
            )
        else:
            plan = validate_inputs(
                repository=repository,
                config=config,
                output_dir=output_dir,
                device_label=args.device_label,
                gpus=gpus,
            )
        commands = build_pipeline(
            config=config,
            output_dir=output_dir,
            device_label=args.device_label,
            gpus=gpus,
            plan=plan,
            stack_validity=_resolve_config_path(
                repository,
                output_dir,
                str(
                    json.loads(config.read_text(encoding="utf-8"))["executor"][
                        "stack_validity_manifest"
                    ]
                ),
            ),
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "compiler_trace_preflight": compiler_trace_preflight,
                        "unique_compiler_family_artifacts": (
                            compiler_trace_preflight[
                                "unique_compiler_family_artifacts"
                            ]
                        ),
                        "unique_lazy_trace_instantiations": (
                            compiler_trace_preflight[
                                "unique_lazy_trace_instantiations"
                            ]
                        ),
                        "projected_trace_generation_calls": (
                            compiler_trace_preflight[
                                "projected_trace_generation_calls"
                            ]
                        ),
                        "projected_trace_bytes": compiler_trace_preflight[
                            "projected_trace_bytes"
                        ],
                        "compiler_trace_preflight_feasible": (
                            compiler_trace_preflight[
                                "compiler_trace_preflight_feasible"
                            ]
                        ),
                        "commands": [
                            {
                                "name": command.name,
                                "argv": list(command.argv),
                                "first_gpu_only": command.first_gpu_only,
                                "outputs": [str(path) for path in command.outputs],
                                "command_id": command.command_id,
                            }
                            for command in commands
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        run_pipeline(
            repository=repository,
            output_dir=output_dir,
            commands=commands,
            gpus=gpus,
        )
        return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(
            json.dumps(
                {
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


_sweep_pipeline_all__ = [
    "PipelineCommand",
    "PUBLICATION_PIPELINE_SCHEMA",
    "_validate_bound_launch_artifacts",
    "build_pipeline",
    "compiler_trace_artifacts_main",
    "pipeline_evidence_main",
    "publication_gate_main",
    "sweep_pipeline_main",
    "run_pipeline",
    "validate_publication_evidence",
    "validate_inputs",
]


def dispatch(argv: Sequence[str] | None = None) -> int:
    """Route to one of this module's commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "compiler-trace-artifacts": compiler_trace_artifacts_main,
        "inputs": sweep_inputs_main,
        "pipeline-evidence": pipeline_evidence_main,
        "publication-gate": publication_gate_main,
        "stage": sweep_launcher_main,
        "shards": shards_main,
        "pipeline": sweep_pipeline_main,
    }
    if not arguments or arguments[0] not in commands:
        raise SystemExit(
            "usage: <command> [options]; commands: " + ", ".join(sorted(commands))
        )
    return commands[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(dispatch())
