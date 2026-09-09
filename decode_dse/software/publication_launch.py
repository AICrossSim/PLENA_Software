"""Bind generated publication artifacts and launch the benchmark contract."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from decode_dse.software import benchmark_runner


REQUIRED_PUBLICATION_BINDINGS = (
    "input_manifest",
    "prefill_artifact_root",
    "driver_output_root",
    "driver",
    "decode_banks",
)


def build_publication_execution_config(
    base_config: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an execution config only after every late-bound input is supplied.

    The exhaustive sweep config intentionally keeps publication disabled because
    bank identities and the audited external driver do not exist until after
    refinement.  This explicit overlay is the only supported transition to an
    executable publication run.
    """

    missing = tuple(key for key in REQUIRED_PUBLICATION_BINDINGS if key not in bindings)
    if missing:
        raise ValueError(f"publication bindings are incomplete: {missing}")
    unknown = tuple(sorted(set(bindings) - set(REQUIRED_PUBLICATION_BINDINGS)))
    if unknown:
        raise ValueError(f"publication bindings contain unknown keys: {unknown}")
    if not isinstance(bindings["driver"], Mapping):
        raise TypeError("publication driver binding must be a mapping")
    if not isinstance(bindings["decode_banks"], Mapping) or not bindings["decode_banks"]:
        raise TypeError("publication decode_banks must be a non-empty mapping")
    for key in ("input_manifest", "prefill_artifact_root", "driver_output_root"):
        if not isinstance(bindings[key], str) or not bindings[key]:
            raise TypeError(f"publication {key} must be a non-empty path string")

    config = copy.deepcopy(dict(base_config))
    publication = config.get("publication")
    pipeline = config.get("publication_pipeline")
    if not isinstance(publication, dict) or not isinstance(pipeline, dict):
        raise ValueError("base config lacks publication contracts")
    resources = pipeline.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("base config lacks publication pipeline resources")
    publication.update(copy.deepcopy(dict(bindings)))
    resources["publication_enabled"] = True
    config.pop("publication_launch_blocker", None)
    config["publication_execution_binding"] = {
        "schema_version": "decode-publication-execution-binding/v1",
        "late_bound_fields": list(REQUIRED_PUBLICATION_BINDINGS),
    }
    return config


def _load_mapping(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base-config", required=True)
    prepare.add_argument("--bindings", required=True)
    prepare.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--contract", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(tuple(argv) if argv is not None else None)
    if args.command == "prepare":
        output = _write_json(
            args.output,
            build_publication_execution_config(
                _load_mapping(args.base_config, "base config"),
                _load_mapping(args.bindings, "bindings"),
            ),
        )
        print(output)
        return 0
    return benchmark_runner.main(
        (
            "--config",
            args.config,
            "--contract",
            args.contract,
            "--output-dir",
            args.output_dir,
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

