"""Preflight correctness checks and the fail-closed evidence they feed.

Checks measure fresh-bank, split-BF16, and cache-reuse behaviour; evidence
assembly then requires every profile to carry a verified terminal result."""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from decode_dse.manifest import SweepManifest, load_manifest
from decode_dse.profiles import DECODE_FORMATS, PROFILE_KIND_BF16_REFERENCE
from decode_dse.software.cached_decode import (
    ContinuationExample,
    TorchHFCachedDecodeBackend,
    capture_bf16_prefill,
    evaluate_teacher_forced_cached,
    evaluate_teacher_forced_cached_batched,
)
from decode_dse.software.sweep import (
    ExecutorContext,
    _load_config,
    _validate_provenance,
)
from decode_dse.software.decode_evaluator import (
    PACKED_CACHE_LAYOUT,
    DecodeEvaluator,
    _prefill_path,
)
from decode_dse.software.cache_artifacts import (
    ArtifactProvenance,
    BF16CacheConverter,
    admit_prefill_cache,
    load_prefill_artifact,
)
from decode_dse.software.sweep_plan import (
    PromptManifest,
    NUMERICAL_SCREEN_SAMPLE_CONTRACT,
    SweepRunPlan,
    load_immutable_json,
    make_stage_manifest,
    write_immutable_json,
)
import hashlib
import json
import math
from datetime import datetime
from decode_dse.legality import load_stack_validity
from decode_dse.software.sweep_runner import (
    COMPLETION_SCHEMA,
    RESULT_SCHEMA,
)
from decode_dse.software.sweep import STAGE_INVOCATION_SCHEMA
from decode_dse.software.sweep_plan import (
    PREFLIGHT_EVIDENCE_SCHEMA,
    PromptManifest,
    SweepRunPlan,
    load_immutable_json,
    write_immutable_json,
)


FRESH_BANK_SCHEMA = "decode-fresh-bank-check"


BF16_CHECK_SCHEMA = "decode-bf16-check"


CROSS_DEVICE_SCHEMA = "decode-cross-device-anchor"


def _load_plan(path: Path) -> SweepRunPlan:
    value = load_immutable_json(path)
    value.pop("content_hash")
    return SweepRunPlan.from_dict(value)


def _load_prompts(path: Path) -> PromptManifest:
    value = load_immutable_json(path)
    value.pop("content_hash")
    return PromptManifest.from_dict(value)


def _context(
    *,
    config_path: Path,
    output_dir: Path,
    device_label: str,
) -> ExecutorContext:
    config = _load_config(config_path)
    manifest = load_manifest(output_dir / "manifest.json")
    plan = _load_plan(output_dir / "run_plan.json")
    prompts = _load_prompts(output_dir / "prompt_manifest.json")
    if device_label not in plan.device_labels:
        raise ValueError("device label is outside the run plan")
    _validate_provenance(
        output_dir / "provenance.json",
        repository=Path(__file__).resolve().parents[2],
        config_path=config_path,
        manifest=manifest,
        plan=plan,
        prompts=prompts,
    )
    return ExecutorContext(
        stage="preflight",
        workspace_root=output_dir,
        output_dir=output_dir / "preflight-correctness",
        config=config,
        master_manifest=manifest,
        stage_manifest=make_stage_manifest(
            manifest,
            plan.preflight_profile_ids,
        ),
        run_plan=plan,
        prompts=prompts,
        sample_contract=NUMERICAL_SCREEN_SAMPLE_CONTRACT,
        shard_index=0,
        shard_count=1,
        device_label=device_label,
    )


def _entry(manifest: SweepManifest, profile_id: str):
    matches = tuple(
        entry for entry in manifest.entries if entry.profile_id == profile_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate profile {profile_id}")
    return matches[0]


def _run_fresh_bank_worker(
    *,
    config_path: Path,
    output_dir: Path,
    device_label: str,
    profile_id: str,
    result_path: Path,
) -> None:
    context = _context(
        config_path=config_path,
        output_dir=output_dir,
        device_label=device_label,
    )
    entry = _entry(context.stage_manifest, profile_id)
    executor = DecodeEvaluator(context)
    with executor.open_weight_bank(
        entry.profile.weight_format,
        (entry,),
    ) as bank:
        with executor.open_kv_admission_cache(
            entry.profile.kv_format
        ) as admitted:
            outcome = executor.evaluate(
                entry,
                weight_bank=bank,
                kv_admission_cache=admitted,
            )
    write_immutable_json(
        result_path,
        {
            "schema_version": FRESH_BANK_SCHEMA,
            "manifest_hash": context.master_manifest.canonical_hash,
            "run_plan_hash": context.run_plan.canonical_hash,
            "prompt_manifest_hash": context.prompts.canonical_hash,
            "profile_id": profile_id,
            "weight_format": entry.profile.weight_format,
            "mean_token_nll": outcome.metrics["mean_token_nll"],
        },
    )


def _run_device_anchor_worker(
    *,
    config_path: Path,
    output_dir: Path,
    device_label: str,
    result_path: Path,
) -> None:
    context = _context(
        config_path=config_path,
        output_dir=output_dir,
        device_label=device_label,
    )
    entries = tuple(
        entry
        for entry in context.stage_manifest.entries
        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
    )
    if len(entries) != 1:
        raise RuntimeError("preflight requires one BF16 cross-device anchor")
    entry = entries[0]
    executor = DecodeEvaluator(context)
    with executor.open_weight_bank("BF16", (entry,)) as bank:
        with executor.open_kv_admission_cache("BF16") as admitted:
            outcome = executor.evaluate(
                entry,
                weight_bank=bank,
                kv_admission_cache=admitted,
            )
    write_immutable_json(
        result_path,
        {
            "schema_version": CROSS_DEVICE_SCHEMA,
            "manifest_hash": context.master_manifest.canonical_hash,
            "run_plan_hash": context.run_plan.canonical_hash,
            "prompt_manifest_hash": context.prompts.canonical_hash,
            "device_label": device_label,
            "profile_nll": {
                entry.profile_id: outcome.metrics["mean_token_nll"],
            },
        },
    )


class _RecordingBackend(TorchHFCachedDecodeBackend):
    def __init__(
        self, *, device: Any, execution_batch_width: int | None = None
    ) -> None:
        super().__init__(
            device=device,
            native_append_format=True,
            execution_batch_width=execution_batch_width,
        )
        self.logits: list[Any] = []

    def decode_step(self, model: Any, **kwargs: Any):
        result = super().decode_step(model, **kwargs)
        self.logits.append(result.logits.detach().to("cpu"))
        return result

    def decode_step_batch(self, model: Any, **kwargs: Any):
        result = super().decode_step_batch(model, **kwargs)
        self.logits.append(result.logits.detach().to("cpu"))
        return result


def _cache_equal(left: Any, right: Any) -> bool:
    if len(left.layers) != len(right.layers):
        return False
    return all(
        fresh.key == saved.key and fresh.value == saved.value
        for fresh, saved in zip(left.layers, right.layers)
    )


def _evaluate_bf16(
    *,
    model: Any,
    device: Any,
    prefill: Any,
    continuation: tuple[int, ...],
    document_id: str,
    provenance: ArtifactProvenance,
    execution_batch_width: int | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    admitted = admit_prefill_cache(
        prefill,
        precision_id="BF16",
        layout_id=PACKED_CACHE_LAYOUT,
        converter=BF16CacheConverter(),
        provenance=provenance,
        metadata={"document_id": document_id},
    )
    backend = _RecordingBackend(
        device=device, execution_batch_width=execution_batch_width
    )
    result = evaluate_teacher_forced_cached(
        model,
        ContinuationExample(
            document_id=document_id,
            prefill=prefill,
            decode_cache=admitted,
            continuation_ids=continuation,
        ),
        backend,
    )
    return result, tuple(backend.logits)


def _evaluate_direct_hf(
    *,
    model: Any,
    device: Any,
    prompt_token_ids: tuple[int, ...],
    continuation: tuple[int, ...],
    execution_batch_width: int | None = None,
) -> tuple[float, tuple[Any, ...], tuple[int, ...]]:
    import torch

    input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=device,
    )[None, :]
    cache_position = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
    first_token = int(output.logits[0, -1].argmax(dim=-1).item())
    if first_token != continuation[0]:
        raise AssertionError("direct BF16 prefill selected a different first token")
    backend = _RecordingBackend(
        device=device, execution_batch_width=execution_batch_width
    )
    cache = backend.adopt_cache(output.past_key_values)
    losses: list[float] = []
    predicted = [first_token]
    for step_index, target_token in enumerate(continuation[1:]):
        previous_length = backend.cache_length(cache)
        expected_length = len(prompt_token_ids) + step_index
        if previous_length != expected_length:
            raise AssertionError("direct BF16 cache length is inconsistent")
        result = backend.decode_step(
            model,
            input_token_id=continuation[step_index],
            cache=cache,
            attention_mask=(1,) * (previous_length + 1),
            position_id=previous_length,
            cache_position=previous_length,
        )
        if backend.cache_length(result.cache) != previous_length + 1:
            raise AssertionError("direct BF16 decode did not append one entry")
        cache = result.cache
        losses.append(backend.token_nll(result.logits, target_token))
        predicted.append(int(result.logits[0, 0].argmax(dim=-1).item()))
    return (
        sum(losses) / len(losses),
        tuple(backend.logits),
        tuple(predicted),
    )


def _run_bf16_worker(
    *,
    config_path: Path,
    output_dir: Path,
    device_label: str,
    result_path: Path,
) -> None:
    import torch

    context = _context(
        config_path=config_path,
        output_dir=output_dir,
        device_label=device_label,
    )
    executor = DecodeEvaluator(context)
    model = executor._load_model().to(executor.device).eval()
    execution_batch_width = max(
        context.run_plan.numerical_screen_microbatch_size,
        context.run_plan.hardware_validation_microbatch_size,
    )
    provenance = ArtifactProvenance(
        producer="packedkv-bf16-preflight-check",
        code_revision=context.master_manifest.canonical_hash,
        created_at_utc="1970-01-01T00:00:00Z",
        parameters=(("manifest_hash", context.master_manifest.canonical_hash),),
    )
    split_checks: list[dict[str, Any]] = []
    cache_checks: list[dict[str, Any]] = []
    batched_examples: list[ContinuationExample] = []
    scalar_results: dict[str, Any] = {}
    scalar_logits: dict[str, tuple[Any, ...]] = {}
    try:
        for sample in executor.samples:
            input_ids = torch.tensor(
                [sample.prompt_token_ids],
                dtype=torch.long,
                device=executor.device,
            )
            mask = torch.ones_like(input_ids)
            fresh = capture_bf16_prefill(
                model,
                input_ids=input_ids,
                attention_mask=mask,
                model_revision=str(context.config["model_revision"]),
                tokenizer_revision=str(context.config["tokenizer_revision"]),
                provenance=provenance,
            )
            saved = load_prefill_artifact(
                _prefill_path(executor.prefill_root, sample.document_id)
            )
            continuation = (
                saved.first_token.token_ids[0],
                *sample.decode_target_ids[
                    : context.sample_contract.decode_steps
                ],
            )
            direct_nll, direct_logits, direct_tokens = _evaluate_direct_hf(
                model=model,
                device=executor.device,
                prompt_token_ids=sample.prompt_token_ids,
                continuation=continuation,
                execution_batch_width=execution_batch_width,
            )
            fresh_result, fresh_logits = _evaluate_bf16(
                model=model,
                device=executor.device,
                prefill=fresh,
                continuation=continuation,
                document_id=sample.document_id,
                provenance=provenance,
                execution_batch_width=execution_batch_width,
            )
            saved_result, saved_logits = _evaluate_bf16(
                model=model,
                device=executor.device,
                prefill=saved,
                continuation=continuation,
                document_id=sample.document_id,
                provenance=provenance,
                execution_batch_width=execution_batch_width,
            )
            if len(direct_logits) != len(saved_logits):
                raise AssertionError("BF16 decode produced different step counts")
            max_logit_error = max(
                (
                    float((left.float() - right.float()).abs().max().item())
                    for left, right in zip(direct_logits, saved_logits)
                ),
                default=0.0,
            )
            token_ids_equal = (
                direct_tokens[0] == saved.first_token.token_ids[0]
                and tuple(
                    int(logits[0, 0].argmax(dim=-1).item())
                    for logits in saved_logits
                )
                == direct_tokens[1:]
            )
            split_nll_error = abs(direct_nll - saved_result.mean_nll)
            cache_nll_error = abs(fresh_result.mean_nll - saved_result.mean_nll)
            source = str(result_path.resolve())
            split_checks.append(
                {
                    "document_id": sample.document_id,
                    "token_ids_equal": token_ids_equal,
                    "max_abs_logit_error": max_logit_error,
                    "mean_token_nll_abs_error": split_nll_error,
                    "source_artifact": source,
                }
            )
            cache_checks.append(
                {
                    "document_id": sample.document_id,
                    "cache_content_equal": _cache_equal(fresh, saved),
                    "mean_token_nll_abs_error": cache_nll_error,
                    "source_artifact": source,
                }
            )
            admitted = admit_prefill_cache(
                saved,
                precision_id="BF16",
                layout_id=PACKED_CACHE_LAYOUT,
                converter=BF16CacheConverter(),
                provenance=provenance,
                metadata={"document_id": sample.document_id},
            )
            batched_examples.append(
                ContinuationExample(
                    document_id=sample.document_id,
                    prefill=saved,
                    decode_cache=admitted,
                    continuation_ids=continuation,
                )
            )
            scalar_results[sample.document_id] = saved_result
            scalar_logits[sample.document_id] = saved_logits
            del (
                input_ids,
                mask,
                fresh,
                fresh_result,
                direct_logits,
                fresh_logits,
            )
        microbatch_checks = []
        for microbatch_size in sorted(
            {
                context.run_plan.numerical_screen_microbatch_size,
                context.run_plan.hardware_validation_microbatch_size,
            }
        ):
            selected = tuple(batched_examples[:microbatch_size])
            backend = _RecordingBackend(
                device=executor.device,
                execution_batch_width=execution_batch_width,
            )
            batched = evaluate_teacher_forced_cached_batched(
                model,
                selected,
                backend,
            )
            reversed_backend = _RecordingBackend(
                device=executor.device,
                execution_batch_width=execution_batch_width,
            )
            reversed_results = evaluate_teacher_forced_cached_batched(
                model,
                tuple(reversed(selected)),
                reversed_backend,
            )
            reversed_by_id = {
                result.document_id: result for result in reversed_results
            }
            max_nll_error = 0.0
            max_logit_error = 0.0
            permutation_error = 0.0
            cache_growth_exact = True
            for lane, (example, result) in enumerate(zip(selected, batched)):
                scalar = scalar_results[example.document_id]
                max_nll_error = max(
                    max_nll_error,
                    *(
                        abs(left - right)
                        for left, right in zip(
                            scalar.per_token_nll,
                            result.per_token_nll,
                        )
                    ),
                )
                permutation_error = max(
                    permutation_error,
                    *(
                        abs(left - right)
                        for left, right in zip(
                            result.per_token_nll,
                            reversed_by_id[
                                example.document_id
                            ].per_token_nll,
                        )
                    ),
                )
                cache_growth_exact &= (
                    result.final_cache_length
                    == result.initial_cache_length + result.token_count
                )
                for step, batch_logits in enumerate(backend.logits):
                    max_logit_error = max(
                        max_logit_error,
                        float(
                            (
                                scalar_logits[example.document_id][step][0]
                                .float()
                                - batch_logits[lane].float()
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
            microbatch_checks.append(
                {
                    "profile_id": next(
                        entry.profile_id
                        for entry in context.master_manifest.entries
                        if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
                    ),
                    "microbatch_size": microbatch_size,
                    "execution_batch_width": execution_batch_width,
                    "max_abs_logit_error": max_logit_error,
                    "max_abs_token_nll_error": max_nll_error,
                    "max_abs_permutation_nll_error": permutation_error,
                    "cache_growth_exact": cache_growth_exact,
                    "lane_isolation_checked": True,
                    "source_artifact": str(result_path.resolve()),
                }
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_immutable_json(
        result_path,
        {
            "schema_version": BF16_CHECK_SCHEMA,
            "manifest_hash": context.master_manifest.canonical_hash,
            "run_plan_hash": context.run_plan.canonical_hash,
            "prompt_manifest_hash": context.prompts.canonical_hash,
            "bf16_split_checks": split_checks,
            "cache_reuse_checks": cache_checks,
            "microbatch_checks": microbatch_checks,
        },
    )


def _run_subprocess(arguments: list[str]) -> None:
    completed = subprocess.run(arguments, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"preflight worker failed with exit code {completed.returncode}"
        )


def _observed_preflight(output_dir: Path) -> dict[str, tuple[Mapping[str, Any], str]]:
    return {
        str(row["profile_id"]): (row, source)
        for row, source in stage_rows(output_dir / "preflight")
    }


def _cross_device_checks(
    plan: SweepRunPlan,
    inputs: Iterable[str],
    *,
    prompt_manifest_hash: str,
) -> list[dict[str, Any]]:
    paths: dict[str, Path] = {}
    for value in inputs:
        if "=" not in value:
            raise ValueError("cross-device inputs must use label=path")
        label, path = value.split("=", 1)
        if label in paths:
            raise ValueError(f"duplicate cross-device label {label}")
        paths[label] = Path(path)
    if len(plan.device_labels) == 1:
        if paths:
            raise ValueError("cross-device inputs are invalid for one device type")
        return []
    if set(paths) != set(plan.device_labels):
        raise ValueError("cross-device inputs must cover every planned device type")
    by_label = {}
    for label, path in paths.items():
        value = load_immutable_json(path)
        if (
            value.get("schema_version") != CROSS_DEVICE_SCHEMA
            or value.get("manifest_hash") != plan.manifest_hash
            or value.get("run_plan_hash") != plan.canonical_hash
            or value.get("prompt_manifest_hash") != prompt_manifest_hash
            or value.get("device_label") != label
            or not isinstance(value.get("profile_nll"), Mapping)
        ):
            raise ValueError(f"invalid cross-device anchor artifact: {path}")
        by_label[label] = value
    common = set.intersection(
        *(
            set(value["profile_nll"])
            for value in by_label.values()
        )
    )
    if not common:
        raise ValueError("cross-device inputs have no common anchor profile")
    if len(common) != 1:
        raise ValueError("cross-device inputs require exactly one common anchor")
    anchor = next(iter(common))
    result: list[dict[str, Any]] = []
    labels = sorted(by_label)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            result.append(
                {
                    "profile_id": anchor,
                    "left_label": left,
                    "right_label": right,
                    "left_mean_token_nll": by_label[left]["profile_nll"][anchor],
                    "right_mean_token_nll": by_label[right]["profile_nll"][anchor],
                    "source_artifact": (
                        f"{paths[left].resolve()}::{paths[right].resolve()}"
                    ),
                }
            )
    return result


def _run_main(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    context = _context(
        config_path=config_path,
        output_dir=output_dir,
        device_label=args.device_label,
    )
    observed = _observed_preflight(output_dir)
    by_weight: dict[str, str] = {}
    for entry in context.stage_manifest.entries:
        weight = entry.profile.weight_format
        if weight in DECODE_FORMATS and weight not in by_weight:
            by_weight[weight] = entry.profile_id
    if set(by_weight) != set(DECODE_FORMATS):
        raise RuntimeError("preflight does not cover every decode weight format")
    root = output_dir / "preflight-correctness"
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    for weight_format in DECODE_FORMATS:
        profile_id = by_weight[weight_format]
        path = root / f"fresh-{weight_format}.json"
        if not path.exists():
            _run_subprocess(
                [
                    sys.executable,
                    "-m",
                    "decode_dse.software.preflight", "check",
                    "_weight",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--device-label",
                    args.device_label,
                    "--profile-id",
                    profile_id,
                    "--out",
                    str(path),
                ]
            )
        fresh = load_immutable_json(path)
        if (
            fresh.get("schema_version") != FRESH_BANK_SCHEMA
            or fresh.get("manifest_hash")
            != context.master_manifest.canonical_hash
            or fresh.get("run_plan_hash") != context.run_plan.canonical_hash
            or fresh.get("prompt_manifest_hash")
            != context.prompts.canonical_hash
            or fresh.get("profile_id") != profile_id
            or fresh.get("weight_format") != weight_format
        ):
            raise ValueError(f"invalid fresh-bank artifact: {path}")
        row, reused_source = observed[profile_id]
        checks.append(
            {
                "profile_id": profile_id,
                "left_label": "fresh_process",
                "right_label": "reused_bank",
                "left_mean_token_nll": fresh["mean_token_nll"],
                "right_mean_token_nll": row["result"]["mean_token_nll"],
                "source_artifact": f"{path.resolve()}::{reused_source}",
            }
        )
    bf16_path = root / "bf16.json"
    if not bf16_path.exists():
        _run_subprocess(
            [
                sys.executable,
                "-m",
                "decode_dse.software.preflight", "check",
                "_bf16",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--device-label",
                args.device_label,
                "--out",
                str(bf16_path),
            ]
        )
    bf16 = load_immutable_json(bf16_path)
    if (
        bf16.get("schema_version") != BF16_CHECK_SCHEMA
        or bf16.get("manifest_hash")
        != context.master_manifest.canonical_hash
        or bf16.get("run_plan_hash") != context.run_plan.canonical_hash
        or bf16.get("prompt_manifest_hash")
        != context.prompts.canonical_hash
    ):
        raise ValueError(f"invalid BF16 correctness artifact: {bf16_path}")
    result = {
        "schema_version": CORRECTNESS_SCHEMA,
        "manifest_hash": context.master_manifest.canonical_hash,
        "run_plan_hash": context.run_plan.canonical_hash,
        "prompt_manifest_hash": context.prompts.canonical_hash,
        "weight_bank_checks": checks,
        "cross_device_checks": _cross_device_checks(
            context.run_plan,
            args.cross_device,
            prompt_manifest_hash=context.prompts.canonical_hash,
        ),
        "bf16_split_checks": bf16["bf16_split_checks"],
        "cache_reuse_checks": bf16["cache_reuse_checks"],
        "microbatch_checks": bf16["microbatch_checks"],
    }
    write_immutable_json(args.out, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    worker = commands.add_parser("_weight")
    bf16 = commands.add_parser("_bf16")
    anchor = commands.add_parser("anchor")
    for command in (run, worker, bf16, anchor):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--device-label", required=True)
        command.add_argument("--out", type=Path, required=True)
    run.add_argument("--cross-device", action="append", default=[])
    worker.add_argument("--profile-id", required=True)
    return parser


def preflight_checks_main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    if args.command == "run":
        _run_main(args)
    elif args.command == "anchor":
        _run_device_anchor_worker(
            config_path=args.config.resolve(),
            output_dir=args.output_dir.resolve(),
            device_label=args.device_label,
            result_path=args.out.resolve(),
        )
    elif args.command == "_weight":
        _run_fresh_bank_worker(
            config_path=args.config.resolve(),
            output_dir=args.output_dir.resolve(),
            device_label=args.device_label,
            profile_id=args.profile_id,
            result_path=args.out.resolve(),
        )
    else:
        _run_bf16_worker(
            config_path=args.config.resolve(),
            output_dir=args.output_dir.resolve(),
            device_label=args.device_label,
            result_path=args.out.resolve(),
        )
    return 0


CORRECTNESS_SCHEMA = "decode-preflight-correctness"


def _partitions(root: Path) -> tuple[Path, ...]:
    if (root / "manifest.json").is_file():
        return (root,)
    values = tuple(sorted(root.glob("part-*-of-*")))
    if not values:
        raise FileNotFoundError(f"no stage partitions under {root}")
    return values


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: object) -> datetime:
    text = str(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("stack evidence timestamps must include a timezone")
    return parsed


def _stack_evidence_preparation(
    *,
    path: Path,
    manifest: SweepManifest,
    plan: SweepRunPlan,
) -> dict[str, Any]:
    load_stack_validity(
        path,
        scope_profile_ids=plan.hardware_validation_profile_ids,
        required_stages=("compiler", "emulator"),
        scope_name="hardware-validation",
        run_plan_hash=plan.canonical_hash,
    )
    value = load_immutable_json(path)
    source_reports = value.get("source_reports")
    if not isinstance(source_reports, Mapping):
        raise ValueError("stack validity has no source reports")
    intervals = []
    stage_seconds = {}
    stage_hashes = {}
    for stage in ("compiler", "emulator"):
        report = source_reports.get(stage)
        if not isinstance(report, Mapping):
            raise ValueError(f"stack validity lacks {stage} evidence")
        provenance = report.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{stage} evidence lacks provenance")
        started = _parse_utc(provenance.get("started_at_utc"))
        completed = _parse_utc(provenance.get("completed_at_utc"))
        elapsed = (completed - started).total_seconds()
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(
                f"{stage} evidence has no positive measured wall interval"
            )
        intervals.append((started, completed))
        stage_seconds[stage] = elapsed
        stage_hashes[stage] = str(report.get("content_hash", ""))
        if not stage_hashes[stage]:
            raise ValueError(f"{stage} evidence lacks a report hash")
    intervals.sort()
    merged = []
    for started, completed in intervals:
        if not merged or started >= merged[-1][1]:
            merged.append([started, completed])
        elif completed > merged[-1][1]:
            merged[-1][1] = completed
    critical_path_seconds = sum(
        (completed - started).total_seconds() for started, completed in merged
    )
    return {
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "required_stages": ["compiler", "emulator"],
        "compiler_seconds": stage_seconds["compiler"],
        "emulator_seconds": stage_seconds["emulator"],
        "critical_path_seconds": critical_path_seconds,
        "timing_basis": "union_of_measured_stage_wall_intervals",
        "stage_report_hashes": stage_hashes,
        "stack_validity_hash": str(value["content_hash"]),
        "source_artifact": str(path.resolve()),
    }


def _checked_row(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    path: Path,
) -> dict[str, Any]:
    row = dict(value)
    observed = row.pop(hash_field, None)
    if observed != _content_hash(row):
        raise ValueError(f"checksum mismatch: {path}")
    return row | {hash_field: observed}


def stage_rows(
    root: Path,
    *,
    expected_master_manifest_hash: str | None = None,
) -> tuple[tuple[Mapping[str, Any], str], ...]:
    """Return one verified terminal result row per profile in a stage.

    Each element is ``(row, source)`` where ``source`` is an absolute
    ``<shard>#<record_hash>`` reference to the exact measured line.
    """

    rows: dict[str, tuple[Mapping[str, Any], str]] = {}
    for partition in _partitions(root):
        manifest = load_manifest(partition / "manifest.json")
        expected = {entry.profile_id: entry for entry in manifest.entries}
        invocation_path = partition / "invocation.json"
        invocation = load_immutable_json(invocation_path)
        if (
            invocation.get("schema_version") != STAGE_INVOCATION_SCHEMA
            or invocation.get("stage") != root.name
            or invocation.get("stage_manifest_hash") != manifest.canonical_hash
            or invocation.get("profile_count") != len(manifest.entries)
            or (
                expected_master_manifest_hash is not None
                and invocation.get("master_manifest_hash")
                != expected_master_manifest_hash
            )
        ):
            raise ValueError(f"stage invocation mismatch: {invocation_path}")
        completion_root = partition / "completed"
        if not completion_root.is_dir():
            raise RuntimeError(f"incomplete stage partition {partition}")
        completions: dict[str, Mapping[str, Any]] = {}
        for path in sorted(completion_root.glob("*.json")):
            value = _checked_row(
                json.loads(path.read_text(encoding="utf-8")),
                hash_field="marker_hash",
                path=path,
            )
            profile_id = str(value["profile_id"])
            entry = expected.get(profile_id)
            if (
                value.get("schema_version") != COMPLETION_SCHEMA
                or entry is None
                or path.name != f"{profile_id}.json"
                or value.get("manifest_hash") != manifest.canonical_hash
                or int(value["ordinal"]) != entry.ordinal
                or int(value["attempt"]) < 1
                or value.get("state") not in {"succeeded", "failed"}
                or profile_id in completions
            ):
                raise ValueError(f"invalid completion marker: {path}")
            completions[profile_id] = value
        if set(completions) != set(expected):
            raise RuntimeError(f"incomplete completion coverage in {partition}")
        attempts: dict[tuple[str, int], tuple[Mapping[str, Any], str, str]] = {}
        for shard in sorted((partition / "shards").glob("*.jsonl")):
            for line_number, line in enumerate(shard.read_text().splitlines(), 1):
                value = _checked_row(
                    json.loads(line),
                    hash_field="record_hash",
                    path=Path(f"{shard}:{line_number}"),
                )
                record_hash = str(value["record_hash"])
                profile_id = str(value["profile_id"])
                entry = expected.get(profile_id)
                if (
                    value.get("schema_version") != RESULT_SCHEMA
                    or entry is None
                    or value.get("manifest_hash") != manifest.canonical_hash
                    or int(value["ordinal"]) != entry.ordinal
                    or value.get("profile") != entry.profile.to_dict()
                    or value.get("weight_format") != entry.profile.weight_format
                    or shard.name != f"{entry.profile.weight_format}.jsonl"
                    or int(value["attempt"]) < 1
                    or value.get("state") not in {"succeeded", "failed"}
                ):
                    raise ValueError(f"invalid result row at {shard}:{line_number}")
                key = (str(value["profile_id"]), int(value["attempt"]))
                if key in attempts:
                    raise ValueError(f"duplicate result at {shard}:{line_number}")
                attempts[key] = (
                    value,
                    f"{shard.resolve()}#{record_hash}",
                    f"shards/{shard.name}#{record_hash}",
                )
        for profile_id, completion in completions.items():
            if completion.get("state") != "succeeded":
                raise RuntimeError(f"preflight profile failed: {profile_id}")
            key = (profile_id, int(completion["attempt"]))
            if key not in attempts:
                raise RuntimeError(f"missing terminal result for {profile_id}")
            row, source, relative_source = attempts[key]
            if (
                row.get("state") != completion.get("state")
                or completion.get("result_path") != relative_source
            ):
                raise ValueError(f"terminal result mismatch for {profile_id}")
            if profile_id in rows:
                raise ValueError(
                    f"duplicate profile across partitions: {profile_id}"
                )
            rows[profile_id] = (row, source)
    return tuple(rows[key] for key in sorted(rows))


def _runtime_records(
    rows: Iterable[tuple[Mapping[str, Any], str]],
    *,
    stage: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    runtime: list[dict[str, Any]] = []
    rebinding: list[dict[str, Any]] = []
    gpu_memory: list[dict[str, Any]] = []
    append_validation: list[dict[str, Any]] = []
    for row, source in rows:
        metrics = row.get("result")
        if not isinstance(metrics, Mapping):
            raise ValueError("successful preflight row has no result metrics")
        binding = metrics.get("runtime_rebinding")
        if not isinstance(binding, Mapping):
            raise ValueError("preflight row has no rebinding telemetry")
        memory = metrics.get("gpu_memory")
        if not isinstance(memory, Mapping):
            raise ValueError("preflight row has no GPU-memory telemetry")
        append = metrics.get("native_append_validation")
        if not isinstance(append, Mapping):
            raise ValueError("preflight row has no append-validation telemetry")
        raw_runtime = float(row["runtime_seconds"])
        oracle_seconds = float(append["oracle_seconds"])
        hot_runtime = raw_runtime - oracle_seconds
        if not math.isfinite(hot_runtime) or hot_runtime <= 0:
            raise ValueError(
                "profile runtime does not exceed deep append-oracle time"
            )
        invocation = load_immutable_json(
            Path(source.split("#", 1)[0]).parents[1] / "invocation.json"
        )
        common = {
            "stage": stage,
            "profile_id": str(row["profile_id"]),
            "device_label": str(invocation["device_label"]),
            "source_artifact": source,
        }
        runtime.append(
            common
            | {
                "runtime_seconds": hot_runtime,
                "basis": (
                    "evaluation_per_profile_excluding_weight_bank_build_"
                    "and_deep_append_oracle"
                ),
            }
        )
        rebinding.append(
            common
            | {
                "binding_seconds": binding["seconds"],
                "performed": binding["performed"],
                "target_count": binding["target_count"],
                "used_cached_targets": binding["used_cached_targets"],
                "weight_requantizations": binding["weight_requantizations"],
                "sealed_weight_modules": binding["sealed_weight_modules"],
                "weight_quantization_events_before": binding[
                    "weight_quantization_events_before"
                ],
                "weight_quantization_events_after": binding[
                    "weight_quantization_events_after"
                ],
                "weight_identity_before": binding["weight_identity_before"],
                "weight_identity_after": binding["weight_identity_after"],
                "weight_structure_fingerprint": binding[
                    "weight_structure_fingerprint"
                ],
            }
        )
        append_validation.append(
            common
            | {
                "mode": append["mode"],
                "deep_oracle_enabled": append["deep_oracle_enabled"],
                "calls": append["calls"],
                "expected_calls": append["expected_calls"],
                "tensor_checks": append["tensor_checks"],
                "expected_tensor_checks": append["expected_tensor_checks"],
                "quantized_tensor_checks": append["quantized_tensor_checks"],
                "expected_quantized_tensor_checks": append[
                    "expected_quantized_tensor_checks"
                ],
                "oracle_seconds": oracle_seconds,
            }
        )
        gpu_memory.append(
            common
            | {
                "microbatch_size": memory["microbatch_size"],
                "peak_allocated_bytes": memory["peak_allocated_bytes"],
                "peak_reserved_bytes": memory["peak_reserved_bytes"],
                "total_device_bytes": memory["total_device_bytes"],
            }
        )
    return runtime, rebinding, gpu_memory, append_validation


def _weight_build_records(
    rows: Iterable[tuple[Mapping[str, Any], str]],
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row, source in rows:
        metrics = row["result"]
        bank = metrics.get("weight_bank")
        if not isinstance(bank, Mapping):
            raise ValueError("preflight row has no weight-bank telemetry")
        weight_format = str(bank["weight_format"])
        invocation = load_immutable_json(
            Path(source.split("#", 1)[0]).parents[1] / "invocation.json"
        )
        candidate = {
            "weight_format": weight_format,
            "device_label": str(invocation["device_label"]),
            "build_seconds": float(bank["build_seconds"]),
            "source_artifact": source,
        }
        previous = result.get(weight_format)
        if previous is not None and (
            previous["build_seconds"] != candidate["build_seconds"]
            or previous["device_label"] != candidate["device_label"]
        ):
            raise ValueError(f"inconsistent build telemetry for {weight_format}")
        result[weight_format] = candidate
    return [result[key] for key in sorted(result)]


def _weight_build_serialization(
    rows: Iterable[tuple[Mapping[str, Any], str]],
) -> bool:
    policies = set()
    for _, source in rows:
        invocation = load_immutable_json(
            Path(source.split("#", 1)[0]).parents[1] / "invocation.json"
        )
        policy = invocation.get("weight_bank_build_serialized")
        if not isinstance(policy, bool):
            raise TypeError("stage invocation lacks a weight-bank build policy")
        policies.add(policy)
    if len(policies) != 1:
        raise ValueError("preflight stages used different weight-bank policies")
    return policies.pop()


def build_evidence(
    *,
    output_dir: Path,
    correctness_path: Path,
    stack_validity_path: Path,
    numerical_screen_workers: int,
    hardware_validation_workers: int,
) -> dict[str, Any]:
    """Combine stage results, correctness checks, and stack evidence."""

    manifest = load_manifest(output_dir / "manifest.json")
    plan = _load_plan(output_dir / "run_plan.json")
    prompts = _load_prompts(output_dir / "prompt_manifest.json")
    if (
        numerical_screen_workers != plan.numerical_screen_workers
        or hardware_validation_workers != plan.hardware_validation_workers
    ):
        raise ValueError("worker counts differ from the immutable run plan")
    numerical_screen_rows = stage_rows(
        output_dir / "preflight",
        expected_master_manifest_hash=manifest.canonical_hash,
    )
    hardware_validation_rows = stage_rows(
        output_dir / "validation-pilot",
        expected_master_manifest_hash=manifest.canonical_hash,
    )
    (
        numerical_screen_runtime,
        numerical_screen_rebinding,
        numerical_screen_memory,
        numerical_screen_append,
    ) = _runtime_records(numerical_screen_rows, stage="numerical-screen")
    (
        hardware_validation_runtime,
        hardware_validation_rebinding,
        hardware_validation_memory,
        hardware_validation_append,
    ) = _runtime_records(hardware_validation_rows, stage="hardware-validation")
    correctness = load_immutable_json(correctness_path)
    if correctness.get("schema_version") != CORRECTNESS_SCHEMA:
        raise ValueError("unsupported correctness evidence schema")
    if correctness.get("manifest_hash") != manifest.canonical_hash:
        raise ValueError("correctness evidence manifest mismatch")
    if correctness.get("run_plan_hash") != plan.canonical_hash:
        raise ValueError("correctness evidence run-plan mismatch")
    if correctness.get("prompt_manifest_hash") != prompts.canonical_hash:
        raise ValueError("correctness evidence prompt-manifest mismatch")
    admission_path = output_dir / "admission_preparation.json"
    admission = load_immutable_json(admission_path)
    if (
        admission.get("manifest_hash") != manifest.canonical_hash
        or admission.get("run_plan_hash") != plan.canonical_hash
        or admission.get("prompt_manifest_hash") != prompts.canonical_hash
    ):
        raise ValueError("admission preparation receipt mismatch")
    admission_sha256 = _sha256_file(admission_path)
    for _, source in numerical_screen_rows + hardware_validation_rows:
        invocation = load_immutable_json(
            Path(source.split("#", 1)[0]).parents[1] / "invocation.json"
        )
        if invocation.get("admission_preparation_sha256") != admission_sha256:
            raise ValueError("stage invocation used a different admission receipt")
    admission.pop("content_hash")
    admission["source_artifact"] = str(admission_path.resolve())
    stack_preparation = _stack_evidence_preparation(
        path=stack_validity_path,
        manifest=manifest,
        plan=plan,
    )
    completed = tuple(str(row[0]["profile_id"]) for row in numerical_screen_rows)
    return {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA,
        "manifest_hash": manifest.canonical_hash,
        "run_plan_hash": plan.canonical_hash,
        "prompt_manifest_hash": prompts.canonical_hash,
        "completed_profile_ids": list(completed),
        "runtime_samples": numerical_screen_runtime + hardware_validation_runtime,
        "weight_bank_build_samples": _weight_build_records(numerical_screen_rows),
        "runtime_rebinding_samples": (
            numerical_screen_rebinding + hardware_validation_rebinding
        ),
        "native_append_validation_samples": (
            numerical_screen_append + hardware_validation_append
        ),
        "gpu_memory_samples": numerical_screen_memory + hardware_validation_memory,
        "admission_preparation": admission,
        "stack_evidence_preparation": stack_preparation,
        "parallel_workers": {
            "numerical_screen": plan.numerical_screen_workers,
            "hardware_validation": plan.hardware_validation_workers,
        },
        "weight_bank_build_serialized": _weight_build_serialization(
            numerical_screen_rows + hardware_validation_rows
        ),
        "weight_bank_checks": correctness["weight_bank_checks"],
        "cross_device_checks": correctness["cross_device_checks"],
        "bf16_split_checks": correctness["bf16_split_checks"],
        "cache_reuse_checks": correctness["cache_reuse_checks"],
        "microbatch_checks": correctness["microbatch_checks"],
    }


def preflight_evidence_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--stack-validity", type=Path, required=True)
    parser.add_argument("--numerical-screen-workers", type=int, required=True)
    parser.add_argument("--hardware-validation-workers", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    if args.numerical_screen_workers <= 0 or args.hardware_validation_workers <= 0:
        raise SystemExit("worker counts must be positive")
    value = build_evidence(
        output_dir=args.output_dir.resolve(),
        correctness_path=args.correctness.resolve(),
        stack_validity_path=args.stack_validity.resolve(),
        numerical_screen_workers=args.numerical_screen_workers,
        hardware_validation_workers=args.hardware_validation_workers,
    )
    write_immutable_json(args.out, value)
    print(args.out)
    return 0


__all__ = ["CORRECTNESS_SCHEMA", "build_evidence", "preflight_evidence_main", "stage_rows"]


def dispatch(argv: Sequence[str] | None = None) -> int:
    """Route to one of this module's commands."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "check": preflight_checks_main,
        "evidence": preflight_evidence_main,
    }
    if not arguments or arguments[0] not in commands:
        raise SystemExit(
            "usage: <command> [options]; commands: "
            + ", ".join(sorted(commands))
        )
    return commands[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(dispatch())
