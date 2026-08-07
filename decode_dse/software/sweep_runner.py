"""Deterministic, restartable execution for the canonical decode sweep."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import time
import traceback
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from decode_dse.legality import constrain_stack_validity, merge_stack_validity
from decode_dse.legality import StackValidity
from decode_dse.manifest import (
    StatusJournal,
    SweepManifest,
    SweepManifestEntry,
    write_manifest,
)
from decode_dse.profiles import DECODE_FORMATS, VECTOR_FORMATS

RESULT_SCHEMA = "decode-sweep-result"
COMPLETION_SCHEMA = "decode-sweep-completion"
_SHARD_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
_DECODE_FORMAT_ORDER = {token: index for index, token in enumerate(DECODE_FORMATS)}
_VECTOR_FORMAT_ORDER = {token: index for index, token in enumerate(VECTOR_FORMATS)}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_shard_name(weight_format: str) -> str:
    if not _SHARD_TOKEN.fullmatch(weight_format):
        raise ValueError(f"unsafe weight format for result shard: {weight_format!r}")
    return f"{weight_format}.jsonl"


def _group_execution_key(entry: SweepManifestEntry) -> tuple[int, int, int, int]:
    """Keep each admitted KV representation hot across A and vector settings."""

    fallback = len(DECODE_FORMATS)
    return (
        _DECODE_FORMAT_ORDER.get(entry.profile.kv_format, fallback),
        _DECODE_FORMAT_ORDER.get(entry.profile.activation_format, fallback),
        _VECTOR_FORMAT_ORDER.get(entry.profile.vector_format, len(VECTOR_FORMATS)),
        entry.ordinal,
    )


def _sanitize_result(value: Any, path: str = "result") -> tuple[Any, tuple[str, ...]]:
    non_finite: list[str] = []

    def visit(item: Any, item_path: str) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if math.isfinite(item):
                return item
            non_finite.append(item_path)
            if math.isnan(item):
                token = "NaN"
            elif item > 0:
                token = "+Infinity"
            else:
                token = "-Infinity"
            return {"__nonfinite__": token}
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {
                str(key): visit(child, f"{item_path}.{key}")
                for key, child in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, (tuple, list)):
            return [
                visit(child, f"{item_path}[{index}]")
                for index, child in enumerate(item)
            ]
        if hasattr(item, "item"):
            return visit(item.item(), item_path)
        raise TypeError(f"{item_path} has unsupported type {type(item).__name__}")

    return visit(value, path), tuple(non_finite)


class NonFiniteResultError(FloatingPointError):
    """Raised when an evaluation returns a non-finite numeric metric."""

    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = tuple(paths)
        super().__init__("non-finite metrics at " + ", ".join(self.paths))


@dataclass(frozen=True)
class EvaluationOutcome:
    """Finite metrics and measured cross-stack validity for one profile."""

    metrics: Mapping[str, Any]
    validity: StackValidity = StackValidity()
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(
            self, "artifacts", tuple(str(artifact) for artifact in self.artifacts)
        )


class ExhaustiveSweepExecutor(Protocol):
    """Resource lifecycle implemented by the numerical evaluation stack."""

    def open_weight_bank(
        self,
        weight_format: str,
        entries: tuple[SweepManifestEntry, ...],
    ) -> AbstractContextManager[Any]: ...

    def open_kv_admission_cache(
        self,
        kv_format: str,
    ) -> AbstractContextManager[Any]: ...

    def evaluate(
        self,
        entry: SweepManifestEntry,
        *,
        weight_bank: Any,
        kv_admission_cache: Any,
    ) -> EvaluationOutcome: ...


@dataclass(frozen=True)
class ResultPointer:
    """Stable location and identity of one append-only result row."""

    relative_path: str
    record_hash: str
    record: Mapping[str, Any]

    @property
    def journal_path(self) -> str:
        return f"{self.relative_path}#{self.record_hash}"


class ResultShardStore:
    """Append-only result shards with torn-tail recovery and row checksums."""

    def __init__(self, root: Path, manifest: SweepManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.shard_root = root / "shards"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self._profile_ids = {entry.profile_id: entry for entry in manifest.entries}
        self._records: dict[tuple[str, int], ResultPointer] = {}
        self._load()

    def _repair_torn_tail(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.lseek(descriptor, 0, os.SEEK_END)
            if size == 0:
                return
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = os.read(descriptor, size)
            if payload.endswith(b"\n"):
                return
            end = payload.rfind(b"\n") + 1
            os.ftruncate(descriptor, end)
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load(self) -> None:
        for path in sorted(self.shard_root.glob("*.jsonl")):
            self._repair_torn_tail(path)
            relative = str(path.relative_to(self.root))
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    value = json.loads(line)
                    record_hash = value.pop("record_hash", None)
                    if record_hash != _content_hash(value):
                        raise ValueError(
                            f"result checksum mismatch at {path}:{line_number}"
                        )
                    if value.get("manifest_hash") != self.manifest.canonical_hash:
                        raise ValueError(
                            f"result manifest mismatch at {path}:{line_number}"
                        )
                    profile_id = str(value["profile_id"])
                    entry = self._profile_ids.get(profile_id)
                    if entry is None:
                        raise ValueError(f"unknown profile at {path}:{line_number}")
                    if int(value["ordinal"]) != entry.ordinal:
                        raise ValueError(
                            f"result ordinal mismatch at {path}:{line_number}"
                        )
                    if value.get("profile") != entry.profile.to_dict():
                        raise ValueError(
                            f"result profile mismatch at {path}:{line_number}"
                        )
                    if entry.profile.weight_format != value["weight_format"]:
                        raise ValueError(
                            f"result shard mismatch at {path}:{line_number}"
                        )
                    expected_shard = _safe_shard_name(entry.profile.weight_format)
                    if path.name != expected_shard:
                        raise ValueError(
                            f"result is in the wrong shard at {path}:{line_number}"
                        )
                    attempt = int(value["attempt"])
                    if attempt < 1 or value.get("state") not in {
                        "succeeded",
                        "failed",
                    }:
                        raise ValueError(
                            f"invalid result state at {path}:{line_number}"
                        )
                    StackValidity.from_dict(value.get("validity"))
                    key = (profile_id, attempt)
                    if key in self._records:
                        raise ValueError(f"duplicate result attempt for {profile_id}")
                    record = value | {"record_hash": record_hash}
                    self._records[key] = ResultPointer(
                        relative_path=relative,
                        record_hash=record_hash,
                        record=record,
                    )

    def get(self, profile_id: str, attempt: int) -> ResultPointer | None:
        return self._records.get((profile_id, attempt))

    @property
    def records(self) -> tuple[ResultPointer, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda pointer: (
                    int(pointer.record["ordinal"]),
                    int(pointer.record["attempt"]),
                ),
            )
        )

    def append(
        self,
        entry: SweepManifestEntry,
        *,
        attempt: int,
        state: str,
        validity: StackValidity,
        result: Any,
        artifacts: Sequence[str] = (),
        error: BaseException | None = None,
        traceback_text: str | None = None,
        runtime_seconds: float,
    ) -> ResultPointer:
        if state not in {"succeeded", "failed"}:
            raise ValueError(f"unsupported result state {state!r}")
        key = (entry.profile_id, attempt)
        if key in self._records:
            raise ValueError(
                f"result already exists for {entry.profile_id} attempt {attempt}"
            )
        safe_result, non_finite = _sanitize_result(result)
        if non_finite and state != "failed":
            raise NonFiniteResultError(non_finite)
        body: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "manifest_hash": self.manifest.canonical_hash,
            "ordinal": entry.ordinal,
            "profile_id": entry.profile_id,
            "profile": entry.profile.to_dict(),
            "weight_format": entry.profile.weight_format,
            "attempt": attempt,
            "state": state,
            "validity": validity.to_dict(),
            "result": safe_result,
            "artifacts": [str(path) for path in artifacts],
            "error_class": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
            "traceback": traceback_text,
            "runtime_seconds": float(runtime_seconds),
            "completed_at": _timestamp(),
        }
        record_hash = _content_hash(body)
        record = body | {"record_hash": record_hash}
        shard_path = self.shard_root / _safe_shard_name(entry.profile.weight_format)
        payload = _canonical_json(record)
        descriptor = os.open(shard_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("incomplete result journal append")
                offset += written
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        pointer = ResultPointer(
            relative_path=str(shard_path.relative_to(self.root)),
            record_hash=record_hash,
            record=record,
        )
        self._records[key] = pointer
        return pointer


class CompletionStore:
    """Atomic terminal markers derived from the status and result journals."""

    def __init__(self, root: Path, manifest: SweepManifest) -> None:
        self.root = root / "completed"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest

    def mark(
        self,
        entry: SweepManifestEntry,
        *,
        state: str,
        attempt: int,
        result: ResultPointer,
    ) -> Path:
        if state not in {"succeeded", "failed"}:
            raise ValueError("completion markers require a terminal state")
        body = {
            "schema_version": COMPLETION_SCHEMA,
            "manifest_hash": self.manifest.canonical_hash,
            "profile_id": entry.profile_id,
            "ordinal": entry.ordinal,
            "state": state,
            "attempt": attempt,
            "result_path": result.journal_path,
        }
        marker_hash = _content_hash(body)
        payload = _canonical_json(body | {"marker_hash": marker_hash})
        destination = self.root / f"{entry.profile_id}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != json.loads(payload):
                raise ValueError(f"conflicting completion marker: {destination}")
            return destination
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=f".{entry.profile_id}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.link(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination


@dataclass(frozen=True)
class SweepRunSummary:
    """Terminal and pending counts after one runner invocation."""

    attempts_written: int
    succeeded: int
    failed_terminal: int
    pending: int
    result_rows: int


class SweepProgressReporter:
    """Persist and print measured profile and weight-bank progress."""

    def __init__(
        self,
        *,
        output_dir: Path,
        stage: str,
        total_profiles: int,
        initial_succeeded: int,
        initial_failed_terminal: int,
        required_weight_banks: Sequence[str],
        prior_trial_seconds: Sequence[float] = (),
        emit: Callable[[str], None] = print,
    ) -> None:
        self.output_dir = output_dir
        self.stage = stage
        self.work_class = (
            "hardware-validation"
            if stage in {"hardware-validation", "validation-pilot"}
            else "numerical"
        )
        self.total_profiles = int(total_profiles)
        self.succeeded = int(initial_succeeded)
        self.failed_terminal = int(initial_failed_terminal)
        self.required_weight_banks = tuple(dict.fromkeys(required_weight_banks))
        self.opened_weight_banks: set[str] = set()
        self.trial_seconds = [
            float(value) for value in prior_trial_seconds if float(value) > 0
        ]
        self.bank_open_seconds: list[float] = []
        self.attempts_observed = 0
        self._terminal_reports = 0
        self._emit = emit

    @property
    def completed(self) -> int:
        return self.succeeded + self.failed_terminal

    def _mean(self, values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def _payload(
        self,
        *,
        event: str,
        last_trial_seconds: float | None = None,
        last_bank_seconds: float | None = None,
    ) -> dict[str, Any]:
        mean_trial = (
            sum(self.trial_seconds) / self.completed
            if self.trial_seconds and self.completed
            else None
        )
        mean_bank = self._mean(self.bank_open_seconds)
        profiles_remaining = max(0, self.total_profiles - self.completed)
        banks_remaining = max(
            0,
            len(self.required_weight_banks) - len(self.opened_weight_banks),
        )
        remaining_seconds = None
        if mean_trial is not None:
            remaining_seconds = profiles_remaining * mean_trial
            if mean_bank is not None:
                remaining_seconds += banks_remaining * mean_bank
        completion = (
            datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + remaining_seconds,
                tz=timezone.utc,
            )
            .isoformat()
            .replace("+00:00", "Z")
            if remaining_seconds is not None
            else None
        )
        return {
            "schema_version": "decode-sweep-progress",
            "event": event,
            "stage": self.stage,
            "work_class": self.work_class,
            "completed_profiles": self.completed,
            "succeeded_profiles": self.succeeded,
            "failed_terminal_profiles": self.failed_terminal,
            "total_profiles": self.total_profiles,
            "remaining_profiles": profiles_remaining,
            "attempts_observed_this_invocation": self.attempts_observed,
            "unique_weight_banks_opened": len(self.opened_weight_banks),
            "unique_weight_banks_required_this_invocation": len(
                self.required_weight_banks
            ),
            "unique_weight_banks_remaining": banks_remaining,
            "last_trial_seconds": last_trial_seconds,
            "mean_trial_seconds": mean_trial,
            "last_weight_bank_open_seconds": last_bank_seconds,
            "mean_weight_bank_open_seconds": mean_bank,
            "estimated_remaining_seconds": remaining_seconds,
            "estimated_completion_utc": completion,
            "updated_at": _timestamp(),
        }

    def _publish(self, payload: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / "progress.json"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=".progress.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(dict(payload), temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        eta = payload["estimated_remaining_seconds"]
        eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
        trial = payload["last_trial_seconds"]
        trial_text = "n/a" if trial is None else f"{float(trial):.3f}s"
        mean_trial = payload["mean_trial_seconds"]
        mean_trial_text = "n/a" if mean_trial is None else f"{float(mean_trial):.3f}s"
        bank = payload["last_weight_bank_open_seconds"]
        bank_text = "n/a" if bank is None else f"{float(bank):.3f}s"
        self._emit(
            "sweep-progress "
            f"event={payload['event']} stage={self.stage} class={self.work_class} "
            f"profiles={self.completed}/{self.total_profiles} "
            f"succeeded={self.succeeded} failed={self.failed_terminal} "
            f"unique-banks={len(self.opened_weight_banks)}/{len(self.required_weight_banks)} "
            f"attempt={trial_text} measured-per-completed-trial={mean_trial_text} "
            f"bank-open={bank_text} remaining-eta={eta_text}"
        )

    def start(self) -> None:
        self._publish(self._payload(event="resume" if self.completed else "start"))

    def record_weight_bank(self, weight_format: str, seconds: float) -> None:
        self.opened_weight_banks.add(weight_format)
        if seconds > 0:
            self.bank_open_seconds.append(float(seconds))
        self._publish(
            self._payload(event="weight-bank-opened", last_bank_seconds=seconds)
        )

    def record_weight_bank_failure(self, seconds: float) -> None:
        if seconds > 0:
            self.bank_open_seconds.append(float(seconds))
        self._publish(
            self._payload(event="weight-bank-failed", last_bank_seconds=seconds)
        )

    def record_attempt(
        self,
        *,
        state: str,
        terminal: bool,
        seconds: float,
    ) -> None:
        self.attempts_observed += 1
        if seconds > 0:
            self.trial_seconds.append(float(seconds))
        if terminal:
            if state == "succeeded":
                self.succeeded += 1
            elif state == "failed":
                self.failed_terminal += 1
            else:
                raise ValueError(f"unsupported terminal progress state {state!r}")
            self._terminal_reports += 1
        event = (
            "first-completion"
            if terminal and self._terminal_reports == 1
            else ("profile-completed" if terminal else "retry-required")
        )
        self._publish(self._payload(event=event, last_trial_seconds=float(seconds)))


class ExhaustiveSweepRunner:
    """Run manifest entries by weight bank with deterministic three-attempt resume."""

    def __init__(
        self,
        *,
        manifest: SweepManifest,
        output_dir: str | Path,
        executor: ExhaustiveSweepExecutor,
        max_attempts: int = 3,
        stage: str = "sweep",
        clock: Callable[[], float] = time.monotonic,
        emit_progress: Callable[[str], None] = print,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.executor = executor
        self.max_attempts = max_attempts
        self.stage = str(stage)
        self.clock = clock
        self.emit_progress = emit_progress
        self.entries = {entry.profile_id: entry for entry in manifest.entries}

    @contextmanager
    def _run_lock(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "runner.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another sweep runner owns {path}") from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _write_failure(
        self,
        *,
        entry: SweepManifestEntry,
        attempt: int,
        error: BaseException,
        validity: StackValidity,
        store: ResultShardStore,
        journal: StatusJournal,
        runtime_seconds: float,
        result: Any = None,
        traceback_text: str | None = None,
    ) -> ResultPointer:
        pointer = store.append(
            entry,
            attempt=attempt,
            state="failed",
            validity=validity,
            result=result,
            error=error,
            traceback_text=traceback_text,
            runtime_seconds=runtime_seconds,
        )
        journal.fail(entry.profile_id, error=error, validity=validity)
        return pointer

    def _recover(
        self,
        store: ResultShardStore,
        journal: StatusJournal,
        completions: CompletionStore,
    ) -> None:
        latest = journal.latest()
        for profile_id, status in latest.items():
            if status.state != "running":
                continue
            entry = self.entries[profile_id]
            pointer = store.get(profile_id, status.attempt)
            if pointer is None:
                error = RuntimeError("interrupted before a result row was committed")
                pointer = self._write_failure(
                    entry=entry,
                    attempt=status.attempt,
                    error=error,
                    validity=status.validity,
                    store=store,
                    journal=journal,
                    runtime_seconds=0.0,
                    result={"recovered": True},
                )
            elif pointer.record["state"] == "succeeded":
                validity = StackValidity.from_dict(pointer.record["validity"])
                journal.complete(
                    profile_id,
                    validity=validity,
                    result_path=pointer.journal_path,
                )
            else:
                validity = StackValidity.from_dict(pointer.record["validity"])
                error = (
                    f"{pointer.record['error_class']}: "
                    f"{pointer.record['error_message']}"
                )
                journal.fail(profile_id, error=error, validity=validity)

        latest = journal.latest()
        for profile_id, status in latest.items():
            terminal = status.state == "succeeded" or (
                status.state == "failed" and status.attempt >= self.max_attempts
            )
            if not terminal:
                continue
            pointer = store.get(profile_id, status.attempt)
            if pointer is None:
                raise ValueError(f"terminal status has no result row: {profile_id}")
            if (
                status.state == "succeeded"
                and status.result_path != pointer.journal_path
            ):
                raise ValueError(
                    f"terminal status points to a different result: {profile_id}"
                )
            if pointer.record["state"] != status.state:
                raise ValueError(f"terminal status/result disagreement: {profile_id}")
            completions.mark(
                self.entries[profile_id],
                state=status.state,
                attempt=status.attempt,
                result=pointer,
            )

    def _evaluate_entry(
        self,
        entry: SweepManifestEntry,
        *,
        weight_bank: Any,
        kv_cache: Any,
        store: ResultShardStore,
        journal: StatusJournal,
        completions: CompletionStore,
    ) -> tuple[str, bool, float]:
        running = journal.begin(entry.profile_id)
        started = self.clock()
        try:
            outcome = self.executor.evaluate(
                entry,
                weight_bank=weight_bank,
                kv_admission_cache=kv_cache,
            )
            if not isinstance(outcome, EvaluationOutcome):
                raise TypeError("executor.evaluate must return EvaluationOutcome")
            validity = constrain_stack_validity(
                entry.profile,
                merge_stack_validity(entry.validity, outcome.validity),
            )
            if validity.software_valid is False:
                raise ValueError("successful evaluation cannot be software-invalid")
            if validity.software_valid is None:
                validity = validity.updated(software_valid=True)
            safe_metrics, non_finite = _sanitize_result(outcome.metrics)
            if non_finite:
                raise NonFiniteResultError(non_finite)
        except Exception as error:
            validity = (
                merge_stack_validity(
                    entry.validity,
                    StackValidity(software_valid=False),
                )
                if isinstance(error, NonFiniteResultError)
                else entry.validity
            )
            result = None
            if "outcome" in locals() and isinstance(outcome, EvaluationOutcome):
                try:
                    result = _sanitize_result(outcome.metrics)[0]
                except Exception:
                    result = {"unserializable_result": repr(outcome.metrics)}
            elapsed = self.clock() - started
            pointer = self._write_failure(
                entry=entry,
                attempt=running.attempt,
                error=error,
                validity=validity,
                store=store,
                journal=journal,
                runtime_seconds=elapsed,
                result=result,
                traceback_text=traceback.format_exc(),
            )
            if running.attempt >= self.max_attempts:
                completions.mark(
                    entry,
                    state="failed",
                    attempt=running.attempt,
                    result=pointer,
                )
            return "failed", running.attempt >= self.max_attempts, elapsed

        elapsed = self.clock() - started
        pointer = store.append(
            entry,
            attempt=running.attempt,
            state="succeeded",
            validity=validity,
            result=safe_metrics,
            artifacts=outcome.artifacts,
            runtime_seconds=elapsed,
        )
        journal.complete(
            entry.profile_id,
            validity=validity,
            result_path=pointer.journal_path,
        )
        completions.mark(
            entry,
            state="succeeded",
            attempt=running.attempt,
            result=pointer,
        )
        return "succeeded", True, elapsed

    def _fail_unopened_group(
        self,
        entries: Sequence[SweepManifestEntry],
        error: BaseException,
        *,
        store: ResultShardStore,
        journal: StatusJournal,
        completions: CompletionStore,
    ) -> tuple[tuple[str, bool, float], ...]:
        results: list[tuple[str, bool, float]] = []
        for entry in entries:
            running = journal.begin(entry.profile_id)
            pointer = self._write_failure(
                entry=entry,
                attempt=running.attempt,
                error=error,
                validity=entry.validity,
                store=store,
                journal=journal,
                runtime_seconds=0.0,
                traceback_text=traceback.format_exc(),
            )
            if running.attempt >= self.max_attempts:
                completions.mark(
                    entry,
                    state="failed",
                    attempt=running.attempt,
                    result=pointer,
                )
            results.append(("failed", running.attempt >= self.max_attempts, 0.0))
        return tuple(results)

    def _summary(
        self, journal: StatusJournal, store: ResultShardStore, attempts: int
    ) -> SweepRunSummary:
        latest = journal.latest()
        succeeded = sum(record.state == "succeeded" for record in latest.values())
        failed_terminal = sum(
            record.state == "failed" and record.attempt >= self.max_attempts
            for record in latest.values()
        )
        return SweepRunSummary(
            attempts_written=attempts,
            succeeded=succeeded,
            failed_terminal=failed_terminal,
            pending=len(self.manifest.entries) - succeeded - failed_terminal,
            result_rows=len(store.records),
        )

    def run(self, *, limit: int | None = None) -> SweepRunSummary:
        """Execute or resume profiles until success or the attempt limit."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        with self._run_lock():
            write_manifest(self.output_dir / "manifest.json", self.manifest)
            journal = StatusJournal(self.output_dir / "status.jsonl", self.manifest)
            store = ResultShardStore(self.output_dir, self.manifest)
            completions = CompletionStore(self.output_dir, self.manifest)
            initial_rows = len(store.records)
            self._recover(store, journal, completions)
            remaining_limit = limit
            latest = journal.latest()
            initial_succeeded = sum(
                status.state == "succeeded" for status in latest.values()
            )
            initial_failed = sum(
                status.state == "failed" and status.attempt >= self.max_attempts
                for status in latest.values()
            )
            pending_ids = {
                profile_id
                for profile_id, status in latest.items()
                if not (
                    status.state == "succeeded"
                    or (
                        status.state == "failed" and status.attempt >= self.max_attempts
                    )
                )
            }
            required_weight_banks = tuple(
                dict.fromkeys(
                    entry.profile.weight_format
                    for entry in self.manifest.entries
                    if entry.profile_id in pending_ids
                )
            )
            progress = SweepProgressReporter(
                output_dir=self.output_dir,
                stage=self.stage,
                total_profiles=len(self.manifest.entries),
                initial_succeeded=initial_succeeded,
                initial_failed_terminal=initial_failed,
                required_weight_banks=required_weight_banks,
                prior_trial_seconds=tuple(
                    float(pointer.record["runtime_seconds"])
                    for pointer in store.records
                    if float(pointer.record["runtime_seconds"]) > 0
                ),
                emit=self.emit_progress,
            )
            progress.start()

            with ExitStack():
                while True:
                    pending = journal.next_entries(
                        max_attempts=self.max_attempts,
                        limit=remaining_limit,
                    )
                    if not pending:
                        break
                    groups: dict[str, list[SweepManifestEntry]] = {}
                    for entry in pending:
                        groups.setdefault(entry.profile.weight_format, []).append(entry)

                    for weight_format, group_values in groups.items():
                        group = tuple(sorted(group_values, key=_group_execution_key))
                        weight_stack = ExitStack()
                        bank_started = self.clock()
                        try:
                            weight_bank = weight_stack.enter_context(
                                self.executor.open_weight_bank(weight_format, group)
                            )
                        except Exception as error:
                            bank_seconds = self.clock() - bank_started
                            progress.record_weight_bank_failure(bank_seconds)
                            weight_stack.close()
                            outcomes = self._fail_unopened_group(
                                group,
                                error,
                                store=store,
                                journal=journal,
                                completions=completions,
                            )
                            for state, terminal, seconds in outcomes:
                                progress.record_attempt(
                                    state=state,
                                    terminal=terminal,
                                    seconds=seconds,
                                )
                            if remaining_limit is not None:
                                remaining_limit -= len(group)
                                if remaining_limit <= 0:
                                    return self._summary(
                                        journal,
                                        store,
                                        len(store.records) - initial_rows,
                                    )
                            continue

                        progress.record_weight_bank(
                            weight_format,
                            self.clock() - bank_started,
                        )
                        with weight_stack:
                            for kv_format, format_entries in groupby(
                                group,
                                key=lambda entry: entry.profile.kv_format,
                            ):
                                format_group = tuple(format_entries)
                                try:
                                    cache_context = (
                                        self.executor.open_kv_admission_cache(kv_format)
                                    )
                                    kv_cache = cache_context.__enter__()
                                except Exception as error:
                                    outcomes = self._fail_unopened_group(
                                        format_group,
                                        error,
                                        store=store,
                                        journal=journal,
                                        completions=completions,
                                    )
                                    for state, terminal, seconds in outcomes:
                                        progress.record_attempt(
                                            state=state,
                                            terminal=terminal,
                                            seconds=seconds,
                                        )
                                else:
                                    try:
                                        for entry in format_group:
                                            state, terminal, seconds = (
                                                self._evaluate_entry(
                                                    entry,
                                                    weight_bank=weight_bank,
                                                    kv_cache=kv_cache,
                                                    store=store,
                                                    journal=journal,
                                                    completions=completions,
                                                )
                                            )
                                            progress.record_attempt(
                                                state=state,
                                                terminal=terminal,
                                                seconds=seconds,
                                            )
                                    finally:
                                        cache_context.__exit__(None, None, None)
                                if remaining_limit is not None:
                                    remaining_limit -= len(format_group)
                                    if remaining_limit <= 0:
                                        return self._summary(
                                            journal,
                                            store,
                                            len(store.records) - initial_rows,
                                        )

            return self._summary(journal, store, len(store.records) - initial_rows)


__all__ = [
    "COMPLETION_SCHEMA",
    "EvaluationOutcome",
    "ExhaustiveSweepExecutor",
    "ExhaustiveSweepRunner",
    "NonFiniteResultError",
    "RESULT_SCHEMA",
    "ResultPointer",
    "ResultShardStore",
    "SweepRunSummary",
    "SweepProgressReporter",
]
