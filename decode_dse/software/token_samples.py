"""Immutable token sample bundles for sweep and refinement evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from decode_dse.software.cache_artifacts import compute_prompt_hash
from decode_dse.software.sweep_plan import (
    PromptManifest,
    PromptRecord,
    NUMERICAL_SCREEN_SAMPLE_CONTRACT,
    HARDWARE_VALIDATION_SAMPLE_CONTRACT,
    write_immutable_json,
)
from decode_dse.software.sweep_plan import write_immutable_json


SAMPLE_BUNDLE_SCHEMA = "decode-token-samples"


DOCUMENT_SELECTION_POLICY = "document_round_robin_nonoverlap"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prompt_hash(token_ids: Sequence[int]) -> str:
    ids = tuple(int(token) for token in token_ids)
    length = len(ids)
    return compute_prompt_hash(
        (ids,),
        ((1,) * length,),
        (tuple(range(length)),),
    )


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        [int(token) for token in token_ids],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SourceTokenSpan:
    """Exact source-document token interval used by one fixed window."""

    source_document_id: str
    token_start: int
    token_end: int
    source_content_hash: str

    def __post_init__(self) -> None:
        if not self.source_document_id:
            raise ValueError("source document ID is required")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("source token span is invalid")
        if len(self.source_content_hash) != 64:
            raise ValueError("source content hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_document_id": self.source_document_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "source_content_hash": self.source_content_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceTokenSpan":
        return cls(
            source_document_id=str(value["source_document_id"]),
            token_start=int(value["token_start"]),
            token_end=int(value["token_end"]),
            source_content_hash=str(value["source_content_hash"]),
        )


@dataclass(frozen=True)
class TokenizedSourceDocument:
    """One independently identified source document and its tokenization."""

    document_id: str
    content_hash: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        tokens = tuple(int(token) for token in self.token_ids)
        if not self.document_id or len(self.content_hash) != 64:
            raise ValueError("source document identity is invalid")
        if not tokens or any(token < 0 for token in tokens):
            raise ValueError("source document tokens are invalid")
        object.__setattr__(self, "token_ids", tokens)


@dataclass(frozen=True)
class DecodeTokenSample:
    """One fixed prompt and its post-first-token decode targets."""

    document_id: str
    source_cluster_id: str
    prompt_token_ids: tuple[int, ...]
    decode_target_ids: tuple[int, ...]
    prompt_hash: str
    prefill_reference_token_id: int
    source_spans: tuple[SourceTokenSpan, ...]
    selection_seed: int
    selection_policy: str
    window_token_hash: str

    def __post_init__(self) -> None:
        prompt = tuple(int(token) for token in self.prompt_token_ids)
        targets = tuple(int(token) for token in self.decode_target_ids)
        spans = tuple(self.source_spans)
        if not self.document_id or not self.source_cluster_id:
            raise ValueError("sample and source-cluster IDs are required")
        if len(prompt) != NUMERICAL_SCREEN_SAMPLE_CONTRACT.prefill_tokens:
            raise ValueError(
                f"prompt must contain {NUMERICAL_SCREEN_SAMPLE_CONTRACT.prefill_tokens} tokens"
            )
        if not targets:
            raise ValueError("decode_target_ids cannot be empty")
        if any(
            token < 0
            for token in (
                *prompt,
                int(self.prefill_reference_token_id),
                *targets,
            )
        ):
            raise ValueError("token IDs must be non-negative")
        if not spans or sum(
            span.token_end - span.token_start for span in spans
        ) != len(prompt) + 1 + len(targets):
            raise ValueError("source spans do not cover the exact token window")
        if self.selection_seed < 0:
            raise ValueError("selection seed must be non-negative")
        if self.selection_policy not in {
            DOCUMENT_SELECTION_POLICY,
            "contiguous_stream_test_only",
        }:
            raise ValueError("unsupported sample-selection policy")
        expected_hash = _prompt_hash(prompt)
        if self.prompt_hash != expected_hash:
            raise ValueError("prompt hash does not match token IDs")
        expected_window_hash = _token_hash(
            (
                *prompt,
                int(self.prefill_reference_token_id),
                *targets,
            )
        )
        if self.window_token_hash != expected_window_hash:
            raise ValueError("window token hash does not match token IDs")
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "decode_target_ids", targets)
        object.__setattr__(self, "source_spans", spans)

    @classmethod
    def create(
        cls,
        document_id: str,
        source_cluster_id: str,
        prompt_token_ids: Sequence[int],
        decode_target_ids: Sequence[int],
        *,
        prefill_reference_token_id: int,
        source_spans: Sequence[SourceTokenSpan],
        selection_seed: int,
        selection_policy: str,
    ) -> "DecodeTokenSample":
        prompt = tuple(int(token) for token in prompt_token_ids)
        targets = tuple(int(token) for token in decode_target_ids)
        return cls(
            document_id=document_id,
            source_cluster_id=source_cluster_id,
            prompt_token_ids=prompt,
            decode_target_ids=targets,
            prompt_hash=_prompt_hash(prompt),
            prefill_reference_token_id=int(prefill_reference_token_id),
            source_spans=tuple(source_spans),
            selection_seed=int(selection_seed),
            selection_policy=selection_policy,
            window_token_hash=_token_hash(
                (
                    *prompt,
                    int(prefill_reference_token_id),
                    *targets,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_cluster_id": self.source_cluster_id,
            "prompt_hash": self.prompt_hash,
            "prompt_token_ids": list(self.prompt_token_ids),
            "decode_target_ids": list(self.decode_target_ids),
            "prefill_reference_token_id": self.prefill_reference_token_id,
            "source_spans": [span.to_dict() for span in self.source_spans],
            "selection_seed": self.selection_seed,
            "selection_policy": self.selection_policy,
            "window_token_hash": self.window_token_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodeTokenSample":
        return cls(
            document_id=str(value["document_id"]),
            source_cluster_id=str(value["source_cluster_id"]),
            prompt_hash=str(value["prompt_hash"]),
            prompt_token_ids=tuple(value["prompt_token_ids"]),
            decode_target_ids=tuple(value["decode_target_ids"]),
            prefill_reference_token_id=int(
                value["prefill_reference_token_id"]
            ),
            source_spans=tuple(
                SourceTokenSpan.from_dict(span)
                for span in value["source_spans"]
            ),
            selection_seed=int(value["selection_seed"]),
            selection_policy=str(value["selection_policy"]),
            window_token_hash=str(value["window_token_hash"]),
        )


@dataclass(frozen=True)
class TokenSampleBundle:
    """Disjoint numerical screen and hardware validation token samples bound to pinned data revisions."""

    model_revision: str
    tokenizer_revision: str
    dataset_name: str
    dataset_revision: str
    numerical_screen: tuple[DecodeTokenSample, ...]
    hardware_validation: tuple[DecodeTokenSample, ...]
    schema_version: str = SAMPLE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SAMPLE_BUNDLE_SCHEMA:
            raise ValueError(
                f"unsupported sample schema {self.schema_version!r}"
            )
        for label, value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("dataset_name", self.dataset_name),
            ("dataset_revision", self.dataset_revision),
        ):
            if not value:
                raise ValueError(f"{label} must be pinned")
        numerical_screen = tuple(self.numerical_screen)
        hardware_validation = tuple(self.hardware_validation)
        if len(numerical_screen) != NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count:
            raise ValueError("numerical screen requires exactly 16 samples")
        if len(hardware_validation) != HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count:
            raise ValueError("hardware validation requires exactly 64 samples")
        if any(
            len(sample.decode_target_ids) < NUMERICAL_SCREEN_SAMPLE_CONTRACT.decode_steps
            for sample in numerical_screen
        ):
            raise ValueError("numerical screen samples do not contain enough decode targets")
        if any(
            len(sample.decode_target_ids) < HARDWARE_VALIDATION_SAMPLE_CONTRACT.decode_steps
            for sample in hardware_validation
        ):
            raise ValueError("hardware validation samples do not contain enough decode targets")
        all_samples = numerical_screen + hardware_validation
        document_ids = tuple(sample.document_id for sample in all_samples)
        prompt_hashes = tuple(sample.prompt_hash for sample in all_samples)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("sample document IDs must be unique")
        if len(prompt_hashes) != len(set(prompt_hashes)):
            raise ValueError("sample prompts must be content-disjoint")
        source_hashes: dict[str, str] = {}
        source_intervals: dict[str, list[tuple[int, int]]] = {}
        for sample in all_samples:
            for span in sample.source_spans:
                previous_hash = source_hashes.setdefault(
                    span.source_document_id,
                    span.source_content_hash,
                )
                if previous_hash != span.source_content_hash:
                    raise ValueError(
                        "one source document has conflicting content hashes"
                    )
                source_intervals.setdefault(
                    span.source_document_id, []
                ).append((span.token_start, span.token_end))
        for source_id, intervals in source_intervals.items():
            ordered_intervals = sorted(intervals)
            if any(
                right_start < left_end
                for (_, left_end), (right_start, _) in zip(
                    ordered_intervals,
                    ordered_intervals[1:],
                )
            ):
                raise ValueError(
                    f"sample token windows overlap within source {source_id!r}"
                )
        if len({sample.selection_seed for sample in all_samples}) != 1:
            raise ValueError("all samples must use one selection seed")
        object.__setattr__(self, "numerical_screen", numerical_screen)
        object.__setattr__(self, "hardware_validation", hardware_validation)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "numerical_screen": [sample.to_dict() for sample in self.numerical_screen],
            "hardware_validation": [sample.to_dict() for sample in self.hardware_validation],
        }

    @property
    def canonical_hash(self) -> str:
        return _canonical_hash(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {
            "sample_bundle_hash": self.canonical_hash
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenSampleBundle":
        bundle = cls(
            schema_version=str(value["schema_version"]),
            model_revision=str(value["model_revision"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            dataset_name=str(value["dataset_name"]),
            dataset_revision=str(value["dataset_revision"]),
            numerical_screen=tuple(
                DecodeTokenSample.from_dict(sample) for sample in value["numerical_screen"]
            ),
            hardware_validation=tuple(
                DecodeTokenSample.from_dict(sample) for sample in value["hardware_validation"]
            ),
        )
        if value.get("sample_bundle_hash") != bundle.canonical_hash:
            raise ValueError("sample-bundle content hash mismatch")
        return bundle

    def prompt_manifest(self) -> PromptManifest:
        return PromptManifest(
            dataset_name=self.dataset_name,
            dataset_revision=self.dataset_revision,
            numerical_screen=tuple(
                PromptRecord(sample.document_id, sample.prompt_hash)
                for sample in self.numerical_screen
            ),
            hardware_validation=tuple(
                PromptRecord(sample.document_id, sample.prompt_hash)
                for sample in self.hardware_validation
            ),
        )

    def samples_for_prompt_set(
        self,
        prompt_set: str,
    ) -> tuple[DecodeTokenSample, ...]:
        if prompt_set == "numerical_screen":
            return self.numerical_screen
        if prompt_set == "hardware_validation":
            return self.hardware_validation
        raise ValueError(f"unsupported prompt set {prompt_set!r}")


def build_bundle_from_documents(
    documents: Sequence[TokenizedSourceDocument],
    *,
    model_revision: str,
    tokenizer_revision: str,
    dataset_name: str,
    dataset_revision: str,
    seed: int,
) -> TokenSampleBundle:
    """Select the stage-contract window count maximizing source coverage."""

    if seed < 0:
        raise ValueError("selection seed must be non-negative")
    prefill = NUMERICAL_SCREEN_SAMPLE_CONTRACT.prefill_tokens
    targets = HARDWARE_VALIDATION_SAMPLE_CONTRACT.decode_steps
    window = prefill + 1 + targets
    count = (
        NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count + HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
    )
    ordered = tuple(sorted(documents, key=lambda item: item.document_id))
    if len({document.document_id for document in ordered}) != len(ordered):
        raise ValueError("source document IDs must be unique")

    candidates_by_document: list[
        tuple[TokenizedSourceDocument, tuple[int, ...]]
    ] = []
    for document in ordered:
        window_count = len(document.token_ids) // window
        if window_count == 0:
            continue
        unused_tokens = len(document.token_ids) - window_count * window
        phase_digest = hashlib.sha256(
            (
                f"{seed}:{document.document_id}:"
                f"{document.content_hash}:phase"
            ).encode("utf-8")
        ).digest()
        phase = int.from_bytes(phase_digest[:8], "big") % (
            unused_tokens + 1
        )
        starts = tuple(phase + index * window for index in range(window_count))
        ranked_starts = tuple(
            sorted(
                starts,
                key=lambda start: hashlib.sha256(
                    (
                        f"{seed}:{document.content_hash}:"
                        f"{start}:window"
                    ).encode("utf-8")
                ).digest(),
            )
        )
        candidates_by_document.append((document, ranked_starts))
    available_windows = sum(
        len(starts) for _, starts in candidates_by_document
    )
    if available_windows < count:
        raise ValueError(
            f"held-out documents cannot form {count} non-overlapping decode windows"
        )

    selected: list[tuple[TokenizedSourceDocument, int]] = []
    round_index = 0
    while len(selected) < count:
        available = [
            (document, starts[round_index])
            for document, starts in candidates_by_document
            if round_index < len(starts)
        ]
        if not available:
            raise AssertionError("window selection exhausted valid candidates")
        available.sort(
            key=lambda item: (
                len(item[0].token_ids),
                item[0].document_id,
            )
        )
        remaining = count - len(selected)
        if len(available) > remaining:
            selection_rng = random.Random(
                seed ^ 0x53545241 ^ round_index
            )
            chosen = []
            for stratum_index in range(remaining):
                lower = stratum_index * len(available) // remaining
                upper = (stratum_index + 1) * len(available) // remaining
                stratum = available[lower:upper]
                if not stratum:
                    raise AssertionError("document-length stratum is empty")
                chosen.append(
                    stratum[selection_rng.randrange(len(stratum))]
                )
            available = chosen
        selected.extend(available)
        round_index += 1

    built: list[tuple[int, int, DecodeTokenSample]] = []
    for document, start in selected:
        stop = start + window
        values = document.token_ids[start:stop]
        if len(values) != window:
            raise AssertionError("selected source window is incomplete")
        window_identity = _canonical_hash(
            {
                "source_document_id": document.document_id,
                "source_content_hash": document.content_hash,
                "token_start": start,
                "token_end": stop,
            }
        )
        sample = DecodeTokenSample.create(
            document_id=f"heldout-window-{window_identity[:24]}",
            source_cluster_id=document.document_id,
            prompt_token_ids=values[:prefill],
            prefill_reference_token_id=values[prefill],
            decode_target_ids=values[prefill + 1 :],
            source_spans=(
                SourceTokenSpan(
                    source_document_id=document.document_id,
                    token_start=start,
                    token_end=stop,
                    source_content_hash=document.content_hash,
                ),
            ),
            selection_seed=seed,
            selection_policy=DOCUMENT_SELECTION_POLICY,
        )
        built.append((len(document.token_ids), start, sample))

    built.sort(
        key=lambda item: (
            item[0],
            item[2].source_cluster_id,
            item[1],
        )
    )
    screen_count = NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count
    validation_count = HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
    if count % screen_count != 0:
        raise AssertionError("stage prompt counts must stratify evenly")
    stride = count // screen_count
    residue = stride // 2
    numerical_screen = [
        sample
        for index, (_, _, sample) in enumerate(built)
        if index % stride == residue
    ]
    hardware_validation = [
        sample
        for index, (_, _, sample) in enumerate(built)
        if index % stride != residue
    ]
    if (len(numerical_screen), len(hardware_validation)) != (
        screen_count,
        validation_count,
    ):
        raise AssertionError("stratified stage assignment has invalid counts")
    random.Random(seed ^ 0x53315F31).shuffle(numerical_screen)
    random.Random(seed ^ 0x53325F32).shuffle(hardware_validation)
    return TokenSampleBundle(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        numerical_screen=tuple(numerical_screen),
        hardware_validation=tuple(hardware_validation),
    )


def build_bundle_from_token_stream(
    token_ids: Iterable[int],
    *,
    model_revision: str,
    tokenizer_revision: str,
    dataset_name: str,
    dataset_revision: str,
) -> TokenSampleBundle:
    """Partition one held-out stream into deterministic non-overlapping samples."""

    tokens = tuple(int(token) for token in token_ids)
    prefill = NUMERICAL_SCREEN_SAMPLE_CONTRACT.prefill_tokens
    targets = HARDWARE_VALIDATION_SAMPLE_CONTRACT.decode_steps
    window = prefill + 1 + targets
    count = (
        NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count + HARDWARE_VALIDATION_SAMPLE_CONTRACT.prompt_count
    )
    required = window * count
    if len(tokens) < required:
        raise ValueError(
            f"token stream has {len(tokens)} tokens; {required} are required"
        )

    samples = []
    for index in range(count):
        start = index * window
        values = tokens[start : start + window]
        samples.append(
            DecodeTokenSample.create(
                document_id=f"heldout-window-{index:04d}",
                source_cluster_id=f"synthetic-window-{index:04d}",
                prompt_token_ids=values[:prefill],
                prefill_reference_token_id=values[prefill],
                decode_target_ids=values[prefill + 1 :],
                source_spans=(
                    SourceTokenSpan(
                        source_document_id=(
                            f"synthetic-window-{index:04d}"
                        ),
                        token_start=0,
                        token_end=window,
                        source_content_hash=_token_hash(values),
                    ),
                ),
                selection_seed=0,
                selection_policy="contiguous_stream_test_only",
            )
        )
    split = NUMERICAL_SCREEN_SAMPLE_CONTRACT.prompt_count
    return TokenSampleBundle(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        numerical_screen=tuple(samples[:split]),
        hardware_validation=tuple(samples[split:]),
    )


def save_sample_bundle(
    bundle: TokenSampleBundle,
    path: str | Path,
) -> Path:
    return write_immutable_json(path, bundle.to_dict())


def load_sample_bundle(path: str | Path) -> TokenSampleBundle:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value.pop("content_hash", None)
    return TokenSampleBundle.from_dict(value)


__all__ = [
    "DecodeTokenSample",
    "DOCUMENT_SELECTION_POLICY",
    "TokenSampleBundle",
    "SAMPLE_BUNDLE_SCHEMA",
    "SourceTokenSpan",
    "TokenizedSourceDocument",
    "build_bundle_from_documents",
    "build_bundle_from_token_stream",
    "load_sample_bundle",
    "save_sample_bundle",
]


REFINEMENT_SAMPLE_BUNDLE_SCHEMA = "decode-refinement-token-samples"


REFINEMENT_PROMPT_COUNT = 128


REFINEMENT_PREFILL_TOKENS = 512


REFINEMENT_DECODE_STEPS = 128


@dataclass(frozen=True)
class RefinementSampleBundle:
    """A standalone 128-document refinement sample contract."""

    model_revision: str
    tokenizer_revision: str
    dataset_name: str
    dataset_revision: str
    samples: tuple[DecodeTokenSample, ...]
    excluded_prompt_hashes: tuple[str, ...] = ()
    token_stream_start: int = 0
    selection_seed: int = 0
    schema_version: str = REFINEMENT_SAMPLE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFINEMENT_SAMPLE_BUNDLE_SCHEMA:
            raise ValueError(
                f"unsupported refinement sample schema {self.schema_version!r}"
            )
        for label, value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("dataset_name", self.dataset_name),
            ("dataset_revision", self.dataset_revision),
        ):
            if not value:
                raise ValueError(f"{label} must be pinned")
        samples = tuple(self.samples)
        if self.token_stream_start < 0:
            raise ValueError("token_stream_start must be non-negative")
        if self.selection_seed < 0:
            raise ValueError("selection_seed must be non-negative")
        if len(samples) != REFINEMENT_PROMPT_COUNT:
            raise ValueError(
                f"refinement requires exactly {REFINEMENT_PROMPT_COUNT} samples"
            )
        if any(
            len(sample.prompt_token_ids) != REFINEMENT_PREFILL_TOKENS
            for sample in samples
        ):
            raise ValueError(
                "refinement prompts must contain exactly "
                f"{REFINEMENT_PREFILL_TOKENS} tokens"
            )
        if any(
            len(sample.decode_target_ids) < REFINEMENT_DECODE_STEPS
            for sample in samples
        ):
            raise ValueError(
                "refinement samples require at least "
                f"{REFINEMENT_DECODE_STEPS} decode targets"
            )
        document_ids = tuple(sample.document_id for sample in samples)
        prompt_hashes = tuple(sample.prompt_hash for sample in samples)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("refinement document IDs must be unique")
        if len(prompt_hashes) != len(set(prompt_hashes)):
            raise ValueError("refinement prompts must be content-disjoint")
        source_hashes: dict[str, str] = {}
        source_intervals: dict[str, list[tuple[int, int]]] = {}
        for sample in samples:
            for span in sample.source_spans:
                previous_hash = source_hashes.setdefault(
                    span.source_document_id,
                    span.source_content_hash,
                )
                if previous_hash != span.source_content_hash:
                    raise ValueError(
                        "one refinement source has conflicting content hashes"
                    )
                source_intervals.setdefault(
                    span.source_document_id, []
                ).append((span.token_start, span.token_end))
        for source_id, intervals in source_intervals.items():
            ordered_intervals = sorted(intervals)
            if any(
                right_start < left_end
                for (_, left_end), (right_start, _) in zip(
                    ordered_intervals,
                    ordered_intervals[1:],
                )
            ):
                raise ValueError(
                    f"refinement windows overlap within source {source_id!r}"
                )
        excluded = tuple(sorted(set(str(value) for value in self.excluded_prompt_hashes)))
        if set(prompt_hashes) & set(excluded):
            raise ValueError("refinement prompts overlap an excluded prompt set")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "excluded_prompt_hashes", excluded)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "excluded_prompt_hashes": list(self.excluded_prompt_hashes),
            "token_stream_start": self.token_stream_start,
            "selection_seed": self.selection_seed,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @property
    def canonical_hash(self) -> str:
        return _canonical_hash(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict() | {"sample_bundle_hash": self.canonical_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefinementSampleBundle":
        bundle = cls(
            model_revision=str(value["model_revision"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            dataset_name=str(value["dataset_name"]),
            dataset_revision=str(value["dataset_revision"]),
            excluded_prompt_hashes=tuple(value.get("excluded_prompt_hashes", ())),
            token_stream_start=int(value.get("token_stream_start", 0)),
            selection_seed=int(value.get("selection_seed", 0)),
            samples=tuple(
                DecodeTokenSample.from_dict(sample) for sample in value["samples"]
            ),
            schema_version=str(value["schema_version"]),
        )
        if value.get("sample_bundle_hash") != bundle.canonical_hash:
            raise ValueError("refinement sample-bundle content hash mismatch")
        return bundle


def build_refinement_bundle_from_token_stream(
    token_ids: Iterable[int],
    *,
    model_revision: str,
    tokenizer_revision: str,
    dataset_name: str,
    dataset_revision: str,
    excluded_prompt_hashes: Sequence[str] = (),
    token_stream_start: int = 0,
) -> RefinementSampleBundle:
    """Build fixed windows without modifying numerical-screen inputs."""

    tokens = tuple(int(token) for token in token_ids)
    if token_stream_start < 0:
        raise ValueError("token_stream_start must be non-negative")
    window = REFINEMENT_PREFILL_TOKENS + 1 + REFINEMENT_DECODE_STEPS
    excluded = set(map(str, excluded_prompt_hashes))
    samples = []
    window_index = 0
    while len(samples) < REFINEMENT_PROMPT_COUNT:
        start = token_stream_start + window_index * window
        if start + window > len(tokens):
            raise ValueError(
                "token stream does not contain 128 prompts after exclusions"
            )
        values = tokens[start : start + window]
        source_id = f"refinement-synthetic-window-{window_index:04d}"
        candidate = DecodeTokenSample.create(
            document_id=f"refinement-heldout-window-{window_index:04d}",
            source_cluster_id=source_id,
            prompt_token_ids=values[:REFINEMENT_PREFILL_TOKENS],
            prefill_reference_token_id=values[REFINEMENT_PREFILL_TOKENS],
            decode_target_ids=values[REFINEMENT_PREFILL_TOKENS + 1 :],
            source_spans=(
                SourceTokenSpan(
                    source_document_id=source_id,
                    token_start=0,
                    token_end=window,
                    source_content_hash=hashlib.sha256(
                        json.dumps(
                            values,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                ),
            ),
            selection_seed=0,
            selection_policy="contiguous_stream_test_only",
        )
        if candidate.prompt_hash not in excluded:
            samples.append(candidate)
        window_index += 1
    return RefinementSampleBundle(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        samples=tuple(samples),
        excluded_prompt_hashes=tuple(excluded),
        token_stream_start=token_stream_start,
        selection_seed=0,
    )


def build_refinement_bundle_from_documents(
    documents: Sequence[TokenizedSourceDocument],
    *,
    model_revision: str,
    tokenizer_revision: str,
    dataset_name: str,
    dataset_revision: str,
    excluded_source_spans: Sequence[SourceTokenSpan],
    excluded_prompt_hashes: Sequence[str],
    selection_seed: int,
) -> RefinementSampleBundle:
    """Select document-stratified windows disjoint from screened spans."""

    if selection_seed < 0:
        raise ValueError("selection_seed must be non-negative")
    window = REFINEMENT_PREFILL_TOKENS + 1 + REFINEMENT_DECODE_STEPS
    blocked: dict[str, list[tuple[int, int]]] = {}
    for span in excluded_source_spans:
        blocked.setdefault(span.source_document_id, []).append(
            (span.token_start, span.token_end)
        )
    candidates: dict[
        str,
        list[tuple[str, TokenizedSourceDocument, int]]
    ] = {}
    for document in sorted(documents, key=lambda item: item.document_id):
        for start in range(0, len(document.token_ids) - window + 1, window):
            stop = start + window
            if any(
                start < prior_stop and prior_start < stop
                for prior_start, prior_stop in blocked.get(
                    document.document_id, ()
                )
            ):
                continue
            rank = hashlib.sha256(
                (
                    f"{selection_seed}:{document.content_hash}:"
                    f"{start}:{stop}"
                ).encode("utf-8")
            ).hexdigest()
            candidates.setdefault(document.document_id, []).append(
                (rank, document, start)
            )
    for values in candidates.values():
        values.sort()
    selected = []
    round_index = 0
    excluded = set(map(str, excluded_prompt_hashes))
    while len(selected) < REFINEMENT_PROMPT_COUNT:
        available = [
            (document_id, values)
            for document_id, values in candidates.items()
            if round_index < len(values)
        ]
        if not available:
            raise ValueError(
                "held-out documents cannot form 128 disjoint refinement windows"
            )
        available.sort(
            key=lambda item: hashlib.sha256(
                f"{selection_seed}:{round_index}:{item[0]}".encode("utf-8")
            ).hexdigest()
        )
        for _, values in available:
            if len(selected) == REFINEMENT_PROMPT_COUNT:
                break
            _, document, start = values[round_index]
            stop = start + window
            tokens = document.token_ids[start:stop]
            sample = DecodeTokenSample.create(
                document_id=(
                    f"refinement-{document.document_id}-"
                    f"{start:08d}-{document.content_hash[:12]}"
                ),
                source_cluster_id=document.document_id,
                prompt_token_ids=tokens[:REFINEMENT_PREFILL_TOKENS],
                prefill_reference_token_id=tokens[REFINEMENT_PREFILL_TOKENS],
                decode_target_ids=tokens[REFINEMENT_PREFILL_TOKENS + 1 :],
                source_spans=(
                    SourceTokenSpan(
                        source_document_id=document.document_id,
                        token_start=start,
                        token_end=stop,
                        source_content_hash=document.content_hash,
                    ),
                ),
                selection_seed=selection_seed,
                selection_policy=DOCUMENT_SELECTION_POLICY,
            )
            if sample.prompt_hash not in excluded:
                selected.append(sample)
        round_index += 1
    return RefinementSampleBundle(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        samples=tuple(selected),
        excluded_prompt_hashes=tuple(excluded),
        token_stream_start=0,
        selection_seed=selection_seed,
    )


def save_refinement_sample_bundle(
    bundle: RefinementSampleBundle, path: str | Path
) -> Path:
    return write_immutable_json(path, bundle.to_dict())


def load_refinement_sample_bundle(path: str | Path) -> RefinementSampleBundle:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value.pop("content_hash", None)
    return RefinementSampleBundle.from_dict(value)


_refinement_token_samples_all__ = [
    "REFINEMENT_DECODE_STEPS",
    "REFINEMENT_PREFILL_TOKENS",
    "REFINEMENT_PROMPT_COUNT",
    "REFINEMENT_SAMPLE_BUNDLE_SCHEMA",
    "RefinementSampleBundle",
    "build_refinement_bundle_from_token_stream",
    "build_refinement_bundle_from_documents",
    "load_refinement_sample_bundle",
    "save_refinement_sample_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
