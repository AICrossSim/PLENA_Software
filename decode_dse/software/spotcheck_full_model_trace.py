"""Cross-tier timing spot-check for the published decode configurations.

Prices each selected (profile, candidate) pair twice — once from full-model
compiler traces and once from the stage-calibrated analytic model — over a
sampled context axis, and writes a labelled comparison artifact. The scope is
decoder-step timing only: the BF16 output head is excluded identically on
both sides, so the delta isolates the timing source.

The compiler-trace pass raises the request-memory interpreter's runaway
guard (a resource valve, not a correctness gate) for the duration of this
process only; nothing on disk changes.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from decode_dse.hardware.design_space import (
    COMPILER_TRACE_EXECUTION_MODE,
    COMPILER_TRACE_TIMING_TIER,
    ExactHardwareSpace,
    LEGACY_AGGREGATE_BANDWIDTH_MODE,
    STAGE_CALIBRATED_ANALYTIC_TIMING_TIER,
)
from decode_dse.hardware.evaluation import (
    DecodeSimulatorBackend,
    HardwareWorkload,
)
from decode_dse.manifest import load_manifest
from decode_dse.software.sweep_plan import write_immutable_json

SPOTCHECK_SCHEMA = "decode-compiler-trace-spotcheck"
DEFAULT_DYNAMIC_INSTRUCTION_LIMIT = 2_000_000_000


def _raise_dynamic_instruction_limit(limit: int) -> None:
    import compiler.aten.plena.compiler as plena_compiler

    original = plena_compiler.build_request_memory_trace
    if isinstance(original, functools.partial):
        original = original.func
    plena_compiler.build_request_memory_trace = functools.partial(
        original,
        max_dynamic_instructions=limit,
    )


def _parse_pairs(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for value in values:
        profile_id, _, candidate_id = value.partition(":")
        if not profile_id or not candidate_id:
            raise ValueError("pairs must use PROFILE_ID:CANDIDATE_ID")
        pairs.append((profile_id, candidate_id))
    if not pairs:
        raise ValueError("at least one spot-check pair is required")
    return tuple(pairs)


def _candidate_index(
    space: ExactHardwareSpace,
    hidden_size: int,
    wanted: set[str],
) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for candidate in space.iter_candidates(hidden_size):
        if candidate.candidate_id in wanted:
            found[candidate.candidate_id] = candidate
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        raise ValueError(
            "candidates absent from the exact hardware space: "
            + ", ".join(sorted(missing))
        )
    return found


def run_spotcheck(
    *,
    config: Mapping[str, Any],
    workspace: Path,
    pairs: Sequence[tuple[str, str]],
    stride: int,
    dynamic_instruction_limit: int,
    output: Path,
) -> Mapping[str, Any]:
    manifest = load_manifest(workspace / "manifest.json")
    entries = {entry.profile_id: entry for entry in manifest.entries}
    reference = config["reference_workload"]
    resources = config["publication_pipeline"]["resources"]
    workload = HardwareWorkload(
        input_seq=int(reference["input_seq"]),
        output_seq=int(reference["output_seq"]),
        stride=int(stride),
        runtime_hbm_reserve_bytes=int(resources["runtime_hbm_reserve_bytes"]),
    )
    architecture = config["model_architecture"]
    space = ExactHardwareSpace.from_study_config(config)
    candidates = _candidate_index(
        space,
        int(architecture["hidden_size"]),
        {candidate_id for _, candidate_id in pairs},
    )

    timing_evidence = workspace / "external" / "decode_timing_evidence.json"
    trace_artifacts = workspace / "external" / "compiler_trace_artifacts.json"
    _raise_dynamic_instruction_limit(dynamic_instruction_limit)
    backend_common = {
        "model": str(config["sim_model"]),
        "model_lib": config.get("model_lib"),
        "settings_toml": None,
        "isa_path": None,
        "timing_evidence": timing_evidence,
    }
    trace_backend = DecodeSimulatorBackend(
        calibrated_bandwidth=False,
        execution_mode=COMPILER_TRACE_EXECUTION_MODE,
        compiler_trace_artifacts=trace_artifacts,
        **backend_common,
    )
    analytic_backend = DecodeSimulatorBackend(
        calibrated_bandwidth=True,
        execution_mode=LEGACY_AGGREGATE_BANDWIDTH_MODE,
        **backend_common,
    )

    rows = []
    for profile_id, candidate_id in pairs:
        entry = entries.get(profile_id)
        if entry is None:
            raise ValueError(f"profile {profile_id} is not in the manifest")
        candidate = candidates[candidate_id]
        trace = trace_backend.evaluate(entry, candidate, workload)
        analytic = analytic_backend.evaluate(entry, candidate, workload)
        if not (trace.timing_calibrated and analytic.timing_calibrated):
            raise ValueError(
                f"spot-check pair {profile_id}:{candidate_id} is not "
                "calibrated on both timing sources"
            )
        rows.append(
            {
                "profile_id": profile_id,
                "candidate_id": candidate_id,
                "batch": int(candidate.batch),
                "compiler_trace": {
                    "publication_timing_tier": COMPILER_TRACE_TIMING_TIER,
                    "tpot_ms": trace.tpot_ms,
                    "tps": trace.tps,
                    "timing_evidence_id": trace.timing_evidence_id,
                },
                "stage_calibrated_analytic": {
                    "publication_timing_tier": (
                        STAGE_CALIBRATED_ANALYTIC_TIMING_TIER
                    ),
                    "tpot_ms": analytic.tpot_ms,
                    "tps": analytic.tps,
                    "timing_evidence_id": analytic.timing_evidence_id,
                    "bandwidth_calibration_id": (
                        analytic.bandwidth_calibration_id
                    ),
                },
                "tpot_relative_delta": (
                    (trace.tpot_ms - analytic.tpot_ms) / analytic.tpot_ms
                ),
            }
        )

    sampled_contexts = len(
        range(
            workload.input_seq,
            workload.input_seq + workload.output_seq,
            workload.stride,
        )
    )
    body = {
        "schema_version": SPOTCHECK_SCHEMA,
        "scope": "decoder_step_timing_only_output_head_excluded_both_sides",
        "model_name": manifest.model_name,
        "model_revision": manifest.model_revision,
        "manifest_hash": manifest.canonical_hash,
        "context_stride": workload.stride,
        "sampled_contexts_per_pair": sampled_contexts,
        "dynamic_instruction_limit": dynamic_instruction_limit,
        "compiler_trace_artifact_set": str(
            trace_backend.compiler_trace_runtime.artifact_set.artifact_set_id
        ),
        "pairs": rows,
    }
    write_immutable_json(output, body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="PROFILE_ID:CANDIDATE_ID of one published configuration",
    )
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument(
        "--max-dynamic-instructions",
        type=int,
        default=DEFAULT_DYNAMIC_INSTRUCTION_LIMIT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stride < 1:
        raise ValueError("stride must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    body = run_spotcheck(
        config=config,
        workspace=args.output_dir.resolve(),
        pairs=_parse_pairs(args.pair),
        stride=args.stride,
        dynamic_instruction_limit=args.max_dynamic_instructions,
        output=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pair_count": len(body["pairs"]),
                "sampled_contexts_per_pair": body[
                    "sampled_contexts_per_pair"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
