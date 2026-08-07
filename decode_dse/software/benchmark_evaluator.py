"""Evaluate published benchmark suites over the split-cached decode path."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from decode_dse.software.benchmark_runner import (
    PublicationBenchmark,
    PublicationConfiguration,
    PublicationContract,
    PublicationEvaluation,
    PublicationItemMetric,
    PublicationProtocol,
    PublicationSplitEvidence,
    POST_HANDOFF_METRIC_ID,
    PUBLICATION_ROLES,
    STANDARD_WIKITEXT2_METRIC_ID,
    TASK_METRIC_ID,
)
from decode_dse.software.sweep_plan import (
    load_immutable_json,
    write_immutable_json,
)

PUBLICATION_INPUT_SCHEMA = "decode-publication-inputs"
PUBLICATION_DRIVER_REQUEST_SCHEMA = "decode-publication-driver-request"
PUBLICATION_DRIVER_RESULT_SCHEMA = "decode-publication-driver-result"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = tuple(sorted(value for value in path.rglob("*") if value.is_file()))
    if not files:
        raise ValueError(f"publication bank is empty: {path}")
    for value in files:
        name = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        payload = value.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _path_identity(path: Path) -> tuple[str, str]:
    if path.is_file():
        return "file", _sha256_file(path)
    if path.is_dir():
        return "directory", _sha256_tree(path)
    raise ValueError(f"publication path does not exist: {path}")


def _confined(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError("publication driver artifact escapes its output directory")
    return path


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_dependency_identities(
    value: Any,
    *,
    benchmark_names: Sequence[str],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("publication dependency identities are incomplete")
    required = {"transformers", "datasets", "ifeval", "gsm8k"}
    if "ruler" in benchmark_names:
        required.add("ruler")
    if set(value) != required:
        raise ValueError("publication dependency identities are incomplete")
    for name, identity in value.items():
        if not isinstance(identity, Mapping):
            raise TypeError("publication dependency identity must be a mapping")
        if name in {"transformers", "datasets"}:
            if (
                set(identity) != {"kind", "version", "distribution_sha256"}
                or identity.get("kind") != "package"
                or not str(identity.get("version", ""))
                or str(identity.get("version", "")).casefold()
                in {"main", "master", "latest"}
                or not _SHA256.fullmatch(str(identity.get("distribution_sha256", "")))
            ):
                raise ValueError("publication package identity is mutable")
        elif (
            set(identity) != {"kind", "revision", "tree_sha256"}
            or identity.get("kind") != "source"
            or not _REVISION.fullmatch(str(identity.get("revision", "")))
            or not _SHA256.fullmatch(str(identity.get("tree_sha256", "")))
        ):
            raise ValueError("publication source identity is mutable")


@dataclass(frozen=True)
class PublicationBankHandle:
    configuration_id: str
    bank_path: str
    bank_kind: str
    bank_sha256: str


class BenchmarkEvaluator:
    """Execute an audited external driver against immutable split artifacts."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        contract: PublicationContract,
    ) -> None:
        if contract.protocol.model_name != str(config.get("model_name", "")):
            raise ValueError("publication contract model differs from config")
        value = config.get("publication")
        if not isinstance(value, Mapping):
            raise ValueError(
                "config.publication is required; publication evaluation is not prepared"
            )
        self.config = dict(config)
        self.settings = dict(value)
        self.contract = contract
        self.input_manifest_path = Path(
            str(self.settings.get("input_manifest", ""))
        ).resolve()
        self.prefill_root = Path(
            str(self.settings.get("prefill_artifact_root", ""))
        ).resolve()
        self.output_root = Path(
            str(self.settings.get("driver_output_root", ""))
        ).resolve()
        self.input_manifest = self._validate_inputs()
        self.command, self.driver_identity, self.timeout_seconds = (
            self._validate_driver()
        )
        self.banks = self._validate_banks()

    def _validate_inputs(self) -> Mapping[str, Any]:
        if not self.input_manifest_path.is_file():
            raise ValueError(
                "full benchmark input manifest is missing; run publication evaluation preparation"
            )
        if not self.prefill_root.is_dir():
            raise ValueError("full benchmark BF16 prefill artifacts are missing")
        value = load_immutable_json(self.input_manifest_path)
        if value.get("schema_version") != PUBLICATION_INPUT_SCHEMA:
            raise ValueError("unsupported publication input manifest")
        if value.get("contract_hash") != self.contract.canonical_hash:
            raise ValueError("publication input manifest contract mismatch")
        if value.get("model_revision") != self.contract.protocol.model_revision:
            raise ValueError("publication input model revision mismatch")
        if value.get("tokenizer_revision") != self.contract.protocol.tokenizer_revision:
            raise ValueError("publication input tokenizer revision mismatch")
        if value.get("prefill_precision") != "BF16":
            raise ValueError("publication inputs require BF16 prefill")
        records = value.get("records")
        if not isinstance(records, list):
            raise ValueError("publication input records are missing")
        expected = {
            (benchmark.benchmark_id, item_id)
            for benchmark in self.contract.benchmarks
            for item_id in benchmark.item_ids
        }
        actual = {
            (str(record.get("benchmark_id")), str(record.get("item_id")))
            for record in records
        }
        if actual != expected or len(records) != len(expected):
            raise ValueError("publication input coverage is not full")
        benchmark_names = {
            benchmark.benchmark_id: benchmark.name
            for benchmark in self.contract.benchmarks
        }
        for record in records:
            if (
                record.get("prefill_precision") != "BF16"
                or record.get("first_token_owner") != "prefill"
                or not record.get("prompt_hash")
                or not record.get("prefill_artifact_id")
                or not _SHA256.fullmatch(str(record.get("prefill_tree_sha256", "")))
                or int(record.get("prefill_file_count", 0)) <= 0
                or int(record.get("prefill_size_bytes", 0)) <= 0
            ):
                raise ValueError("publication prefill record is incomplete")
            path = _confined(
                self.prefill_root,
                str(record.get("prefill_relative_path", "")),
            )
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("publication prefill artifact is missing")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_token = manifest.get("first_token", {})
            greedy_token_id = record.get("prefill_greedy_handoff_token_id")
            ground_truth_token_id = record.get("ground_truth_handoff_token_id")
            benchmark_name = benchmark_names[str(record["benchmark_id"])]
            if (
                manifest.get("schema") != "plena.prefill"
                or manifest.get("artifact_id") != record["prefill_artifact_id"]
                or manifest.get("prompt_hash") != record["prompt_hash"]
                or manifest.get("model_revision")
                != self.contract.protocol.model_revision
                or manifest.get("tokenizer_revision")
                != self.contract.protocol.tokenizer_revision
                or len(manifest.get("layers", ()))
                != int(self.config["model_architecture"]["num_hidden_layers"])
                or first_token.get("selection") != "greedy"
                or first_token.get("token_ids") != [greedy_token_id]
                or isinstance(greedy_token_id, bool)
                or not isinstance(greedy_token_id, int)
                or greedy_token_id < 0
                or (
                    benchmark_name == "wikitext2"
                    and (
                        isinstance(ground_truth_token_id, bool)
                        or not isinstance(ground_truth_token_id, int)
                        or ground_truth_token_id < 0
                    )
                )
                or (benchmark_name != "wikitext2" and ground_truth_token_id is not None)
            ):
                raise ValueError("publication prefill artifact identity mismatch")
            architecture = self.config["model_architecture"]
            expected_kv_heads = int(architecture["num_key_value_heads"])
            expected_head_dim = int(architecture["head_dim"])
            for layer in manifest["layers"]:
                for role in ("key", "value"):
                    tensor = layer[role]
                    shape = tuple(int(item) for item in tensor["shape"])
                    if (
                        tensor.get("dtype") != "bfloat16"
                        or len(shape) != 4
                        or shape[0] != 1
                        or shape[1] != expected_kv_heads
                        or shape[3] != expected_head_dim
                    ):
                        raise ValueError(
                            "publication prefill cache geometry is invalid"
                        )
        prefill_set_hash = _content_hash(
            [
                {
                    "benchmark_id": record["benchmark_id"],
                    "item_id": record["item_id"],
                    "prefill_artifact_id": record["prefill_artifact_id"],
                    "prefill_tree_sha256": record["prefill_tree_sha256"],
                }
                for record in records
            ]
        )
        if value.get("prefill_set_sha256") != prefill_set_hash:
            raise ValueError("publication prefill-set identity mismatch")
        _validate_dependency_identities(
            value.get("dependencies"),
            benchmark_names=tuple(
                benchmark.name for benchmark in self.contract.benchmarks
            ),
        )
        return value

    def _validate_driver(self) -> tuple[tuple[str, ...], dict[str, Any], float]:
        driver = self.settings.get("driver")
        if not isinstance(driver, Mapping):
            raise ValueError("config.publication.driver is required")
        command = driver.get("command")
        if (
            not isinstance(command, Sequence)
            or isinstance(command, (str, bytes))
            or not command
        ):
            raise TypeError("publication driver command must be an argument vector")
        command = tuple(str(item) for item in command)
        for token in ("{request}", "{result}", "{artifact_dir}"):
            if command.count(token) != 1:
                raise ValueError(
                    f"publication driver command requires one literal {token}"
                )
        source_root = Path(str(driver.get("source_root", ""))).resolve()
        if not source_root.is_dir():
            raise ValueError("publication driver source root is missing")
        if self.output_root == source_root or source_root in self.output_root.parents:
            raise ValueError(
                "publication outputs must be outside the driver source tree"
            )
        expected_tree_hash = str(driver.get("source_tree_sha256", ""))
        actual_tree_hash = _sha256_tree(source_root)
        if actual_tree_hash != expected_tree_hash:
            raise ValueError("publication driver source tree hash mismatch")
        source_path = Path(str(driver.get("source_path", ""))).resolve()
        if (
            not source_path.is_file()
            or source_path != source_root
            and source_root not in source_path.parents
        ):
            raise ValueError("publication driver source is missing")
        expected_hash = str(driver.get("source_sha256", ""))
        actual_hash = _sha256_file(source_path)
        if actual_hash != expected_hash:
            raise ValueError("publication driver source hash mismatch")
        executable = Path(str(driver.get("executable_path", ""))).resolve()
        if (
            not executable.is_file()
            or command[0] != str(executable)
            or _sha256_file(executable) != str(driver.get("executable_sha256", ""))
        ):
            raise ValueError("publication driver executable identity mismatch")
        runtime_path = Path(str(driver.get("runtime_receipt_path", ""))).resolve()
        if not runtime_path.is_file() or _sha256_file(runtime_path) != str(
            driver.get("runtime_receipt_sha256", "")
        ):
            raise ValueError("publication driver runtime receipt mismatch")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("schema_version") != "decode-publication-runtime"
            or runtime.get("interpreter_path") != str(executable)
            or runtime.get("interpreter_sha256") != _sha256_file(executable)
            or not str(runtime.get("python_version", ""))
            or not str(runtime.get("platform", ""))
        ):
            raise ValueError("publication driver runtime identity is invalid")
        packages = runtime.get("packages")
        if not isinstance(packages, Mapping) or not {
            "torch",
            "transformers",
            "datasets",
        }.issubset(packages):
            raise ValueError("publication driver runtime packages are incomplete")
        for identity in packages.values():
            if (
                not isinstance(identity, Mapping)
                or set(identity) != {"version", "source_sha256"}
                or not str(identity["version"])
                or not _SHA256.fullmatch(str(identity["source_sha256"]))
            ):
                raise ValueError("publication runtime package identity is mutable")
        for name in ("transformers", "datasets"):
            dependency = self.input_manifest["dependencies"][name]
            if (
                packages[name]["version"] != dependency["version"]
                or packages[name]["source_sha256"] != dependency["distribution_sha256"]
            ):
                raise ValueError(
                    "publication runtime differs from the sealed package identity"
                )
        timeout = float(driver.get("timeout_seconds", 0))
        if timeout <= 0:
            raise ValueError("publication driver timeout must be positive")
        tool_revision = str(driver.get("tool_revision", ""))
        if not _REVISION.fullmatch(tool_revision):
            raise ValueError("publication driver revision must be immutable")
        return (
            command,
            {
                "source_root": str(source_root),
                "source_tree_sha256": actual_tree_hash,
                "source_path": str(source_path),
                "source_sha256": actual_hash,
                "executable_path": str(executable),
                "executable_sha256": _sha256_file(executable),
                "runtime_receipt_path": str(runtime_path),
                "runtime_receipt_sha256": _sha256_file(runtime_path),
                "runtime_fingerprint_sha256": _content_hash(dict(runtime)),
                "tool_revision": tool_revision,
            },
            timeout,
        )

    def _validate_banks(self) -> dict[str, PublicationBankHandle]:
        values = self.settings.get("decode_banks")
        if not isinstance(values, Mapping):
            raise ValueError("config.publication.decode_banks is required")
        if set(values) != set(PUBLICATION_ROLES):
            raise ValueError("publication decode banks must cover all four roles")
        result = {}
        for configuration in self.contract.configurations:
            value = values[configuration.role]
            if not isinstance(value, Mapping):
                raise TypeError("publication bank binding must be a mapping")
            if value.get("profile_id") != configuration.profile.profile_id:
                raise ValueError("publication bank profile mismatch")
            path = Path(str(value.get("path", ""))).resolve()
            kind, digest = _path_identity(path)
            if value.get("sha256") != digest or value.get("kind") != kind:
                raise ValueError("publication bank file identity mismatch")
            result[configuration.configuration_id] = PublicationBankHandle(
                configuration_id=configuration.configuration_id,
                bank_path=str(path),
                bank_kind=kind,
                bank_sha256=digest,
            )
        return result

    @contextmanager
    def open_configuration(
        self,
        configuration: PublicationConfiguration,
        protocol: PublicationProtocol,
    ) -> Iterator[PublicationBankHandle]:
        if protocol != self.contract.protocol:
            raise ValueError("publication protocol changed")
        handle = self.banks.get(configuration.configuration_id)
        if handle is None:
            raise ValueError("publication configuration has no decode bank")
        path = Path(handle.bank_path)
        kind, digest = _path_identity(path)
        if kind != handle.bank_kind or digest != handle.bank_sha256:
            raise ValueError("publication decode bank changed before execution")
        yield handle

    def _records_for(
        self,
        benchmark: PublicationBenchmark,
    ) -> tuple[Mapping[str, Any], ...]:
        records = tuple(
            record
            for record in self.input_manifest["records"]
            if record["benchmark_id"] == benchmark.benchmark_id
        )
        if tuple(record["item_id"] for record in records) != benchmark.item_ids:
            raise ValueError("publication input record order differs from contract")
        return records

    def evaluate(
        self,
        configuration: PublicationConfiguration,
        benchmark: PublicationBenchmark,
        protocol: PublicationProtocol,
        *,
        configuration_handle: Any,
    ) -> PublicationEvaluation:
        if not isinstance(configuration_handle, PublicationBankHandle):
            raise TypeError("publication configuration handle is invalid")
        if configuration_handle.configuration_id != configuration.configuration_id:
            raise ValueError("publication driver received the wrong decode bank")
        benchmark_root = (
            self.output_root / configuration.configuration_id / benchmark.benchmark_id
        )
        benchmark_root.mkdir(parents=True, exist_ok=True)
        attempts = tuple(
            int(path.name.split("-", 1)[1])
            for path in benchmark_root.glob("attempt-*")
            if path.is_dir() and path.name.split("-", 1)[1].isdigit()
        )
        artifact_dir = benchmark_root / f"attempt-{max(attempts, default=0) + 1:03d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_body = {
            "schema_version": PUBLICATION_DRIVER_REQUEST_SCHEMA,
            "contract_hash": self.contract.canonical_hash,
            "configuration": configuration.to_dict(),
            "benchmark": benchmark.to_dict(),
            "protocol": protocol.to_dict(),
            "decode_bank": configuration_handle.__dict__,
            "input_manifest_path": str(self.input_manifest_path),
            "input_manifest_hash": self.input_manifest["content_hash"],
            "prefill_set_sha256": self.input_manifest["prefill_set_sha256"],
            "prefill_artifact_root": str(self.prefill_root),
            "records": [dict(record) for record in self._records_for(benchmark)],
            "driver": self.driver_identity,
            "required_execution": {
                "prefill_model_loaded_during_decode": False,
                "decode_query_length": 1,
                "cache_free_calls": 0,
                "prefill_precision": "BF16",
                "transferred_kv_precision": "BF16",
                "benchmark_metric_id": (
                    STANDARD_WIKITEXT2_METRIC_ID
                    if benchmark.name == "wikitext2"
                    else TASK_METRIC_ID
                ),
                "standard_handoff_token_source": (
                    "ground_truth" if benchmark.name == "wikitext2" else None
                ),
                "post_handoff_metric_id": (
                    POST_HANDOFF_METRIC_ID if benchmark.name == "wikitext2" else None
                ),
                "post_handoff_token_source": (
                    "prefill_greedy" if benchmark.name == "wikitext2" else None
                ),
            },
        }
        request_path = artifact_dir / "request.json"
        write_immutable_json(request_path, request_body)
        request = load_immutable_json(request_path)
        result_path = artifact_dir / "result.json"
        command = tuple(
            (
                str(request_path)
                if token == "{request}"
                else (
                    str(result_path)
                    if token == "{result}"
                    else str(artifact_dir) if token == "{artifact_dir}" else token
                )
            )
            for token in self.command
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            shell=False,
        )
        process_path = artifact_dir / "process.json"
        write_immutable_json(
            process_path,
            {
                "schema_version": "decode-publication-driver-process",
                "request_hash": request["content_hash"],
                "command": list(command),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"publication driver failed with exit {completed.returncode}"
            )
        if not result_path.is_file():
            raise ValueError("publication driver produced no result")
        result = load_immutable_json(result_path)
        if result.get("schema_version") != PUBLICATION_DRIVER_RESULT_SCHEMA:
            raise ValueError("unsupported publication driver result")
        if (
            result.get("request_hash") != request["content_hash"]
            or result.get("configuration_id") != configuration.configuration_id
            or result.get("benchmark_id") != benchmark.benchmark_id
        ):
            raise ValueError("publication driver result binding mismatch")
        items = tuple(PublicationItemMetric.from_dict(item) for item in result["items"])
        evidence = PublicationSplitEvidence.from_dict(result["split_execution"])
        post_handoff_raw = result.get("post_handoff_split_execution")
        post_handoff_evidence = (
            PublicationSplitEvidence.from_dict(post_handoff_raw)
            if isinstance(post_handoff_raw, Mapping)
            else None
        )
        expected_audit = {
            "input_manifest_hash": self.input_manifest["content_hash"],
            "prefill_set_sha256": self.input_manifest["prefill_set_sha256"],
            "decode_bank_sha256": configuration_handle.bank_sha256,
            "driver_source_tree_sha256": self.driver_identity["source_tree_sha256"],
            "runtime_fingerprint_sha256": self.driver_identity[
                "runtime_fingerprint_sha256"
            ],
            "enumeration_receipt_sha256": (benchmark.enumeration_receipt_sha256),
            "evaluated_item_count": benchmark.source_item_count,
            "score_mode": (
                "standard_teacher_forced_nll_plus_post_handoff_greedy_nll"
                if benchmark.name == "wikitext2"
                else "official_task_score"
            ),
        }
        if result.get("semantic_audit") != expected_audit:
            raise ValueError("publication driver semantic audit is incomplete")
        artifact_values = result.get("artifacts")
        if not isinstance(artifact_values, list) or not artifact_values:
            raise ValueError("publication driver returned no trace artifacts")
        artifacts = [str(request_path), str(result_path), str(process_path)]
        for value in artifact_values:
            if not isinstance(value, Mapping):
                raise TypeError("publication driver artifact is malformed")
            path = _confined(artifact_dir, str(value.get("relative_path", "")))
            if (
                not path.is_file()
                or path.stat().st_size <= 0
                or _sha256_file(path) != value.get("sha256")
            ):
                raise ValueError("publication driver artifact identity mismatch")
            artifacts.append(str(path))
        return PublicationEvaluation(
            items=items,
            evidence=evidence,
            artifacts=tuple(artifacts),
            post_handoff_evidence=post_handoff_evidence,
        )


def create_executor(
    *,
    config: Mapping[str, Any],
    contract: PublicationContract,
) -> BenchmarkEvaluator:
    """Create the built-in fail-closed publication bridge."""

    return BenchmarkEvaluator(config=config, contract=contract)


__all__ = [
    "PUBLICATION_DRIVER_REQUEST_SCHEMA",
    "PUBLICATION_DRIVER_RESULT_SCHEMA",
    "PUBLICATION_INPUT_SCHEMA",
    "PublicationBankHandle",
    "BenchmarkEvaluator",
    "create_executor",
]
