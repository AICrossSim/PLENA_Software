"""Split-prefill capture and exact cached one-token decode scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from decode_dse.software.cache_artifacts import (
    ArtifactProvenance,
    DecodeCacheArtifact,
    FirstTokenResult,
    KVLayerPayload,
    PrefillArtifact,
    TensorPayload,
)


class PrefillDecodeView(Protocol):
    """Prompt metadata required after K/V admission has been verified."""

    artifact_id: str
    attention_mask: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    first_token: FirstTokenResult

    @property
    def batch_size(self) -> int: ...

    @property
    def cache_length(self) -> int: ...


@dataclass(frozen=True)
class ContinuationExample:
    """One independently cached document and its teacher-forced continuation."""

    document_id: str
    prefill: PrefillDecodeView
    decode_cache: DecodeCacheArtifact
    continuation_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        continuation = tuple(int(token) for token in self.continuation_ids)
        if not self.document_id:
            raise ValueError("document_id is required")
        if self.prefill.batch_size != 1 or self.decode_cache.batch_size != 1:
            raise ValueError("each continuation must own a batch-one cache")
        if any(mask != 1 for mask in self.prefill.attention_mask[0]):
            raise ValueError("cached documents must be stored without prompt padding")
        if self.decode_cache.source_artifact_id != self.prefill.artifact_id:
            raise ValueError("decode cache was not derived from this prefill artifact")
        if len(continuation) < 2:
            raise ValueError("at least two continuation tokens are required")
        if continuation[0] != self.prefill.first_token.token_ids[0]:
            raise ValueError("continuation must begin with the token selected by prefill")
        object.__setattr__(self, "continuation_ids", continuation)


@dataclass(frozen=True)
class DecodeStep:
    """Backend output for one cache-mutating model invocation."""

    logits: Any
    cache: Any


@dataclass(frozen=True)
class DocumentNLL:
    """Per-document likelihood result with first-token attribution explicit."""

    document_id: str
    nll_sum: float
    token_count: int
    per_token_nll: tuple[float, ...]
    first_token_id: int
    scored_token_ids: tuple[int, ...]
    initial_cache_length: int
    final_cache_length: int

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.token_count

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)


@dataclass(frozen=True)
class CorpusNLL:
    """Token-weighted aggregate retaining every document result."""

    documents: tuple[DocumentNLL, ...]

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        if not documents:
            raise ValueError("at least one document result is required")
        ids = [document.document_id for document in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("document IDs must be unique")
        object.__setattr__(self, "documents", documents)

    @property
    def nll_sum(self) -> float:
        return sum(document.nll_sum for document in self.documents)

    @property
    def token_count(self) -> int:
        return sum(document.token_count for document in self.documents)

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.token_count

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll)


class CachedDecodeBackend(Protocol):
    """Execution operations required by the phase-independent evaluator."""

    def materialize_cache(self, artifact: DecodeCacheArtifact) -> Any:
        ...

    def cache_length(self, cache: Any) -> int:
        ...

    def cache_batch_size(self, cache: Any) -> int:
        ...

    def decode_step(
        self,
        model: Any,
        *,
        input_token_id: int,
        cache: Any,
        attention_mask: tuple[int, ...],
        position_id: int,
        cache_position: int,
    ) -> DecodeStep:
        ...

    def commit_append(
        self,
        cache: Any,
        *,
        previous_length: int,
        artifact: DecodeCacheArtifact,
    ) -> Any:
        ...

    def token_nll(self, logits: Any, target_token_id: int) -> float:
        ...


def evaluate_teacher_forced_cached(
    model: Any,
    example: ContinuationExample,
    backend: CachedDecodeBackend,
) -> DocumentNLL:
    """Score only tokens predicted by sequential cached ``q_len=1`` calls."""
    cache = backend.materialize_cache(example.decode_cache)
    initial_length = example.prefill.cache_length
    if example.decode_cache.cache_length != initial_length:
        raise AssertionError("decode admission changed the cache length")
    if backend.cache_batch_size(cache) != 1:
        raise AssertionError("a document must have an independent batch-one cache")
    if backend.cache_length(cache) != initial_length:
        raise AssertionError("materialized cache length does not match prefill")

    attention_mask = example.prefill.attention_mask[0]
    next_position = example.prefill.position_ids[0][-1] + 1
    losses: list[float] = []
    for step_index, target_token in enumerate(example.continuation_ids[1:]):
        input_token = example.continuation_ids[step_index]
        previous_length = backend.cache_length(cache)
        expected_length = initial_length + step_index
        if previous_length != expected_length:
            raise AssertionError(
                f"cache length before step {step_index} is {previous_length}, "
                f"expected {expected_length}"
            )
        step_mask = attention_mask + (1,) * (step_index + 1)
        result = backend.decode_step(
            model,
            input_token_id=input_token,
            cache=cache,
            attention_mask=step_mask,
            position_id=next_position + step_index,
            cache_position=previous_length,
        )
        grown_length = backend.cache_length(result.cache)
        if grown_length != previous_length + 1:
            raise AssertionError(
                f"q_len=1 must grow the cache by one, observed "
                f"{previous_length}->{grown_length}"
            )
        cache = backend.commit_append(
            result.cache,
            previous_length=previous_length,
            artifact=example.decode_cache,
        )
        if backend.cache_length(cache) != grown_length:
            raise AssertionError("append conversion changed the logical cache length")
        loss = float(backend.token_nll(result.logits, target_token))
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"non-finite NLL for document {example.document_id!r}, "
                f"decode step {step_index}"
            )
        losses.append(loss)

    return DocumentNLL(
        document_id=example.document_id,
        nll_sum=sum(losses),
        token_count=len(losses),
        per_token_nll=tuple(losses),
        first_token_id=example.continuation_ids[0],
        scored_token_ids=example.continuation_ids[1:],
        initial_cache_length=initial_length,
        final_cache_length=backend.cache_length(cache),
    )


def evaluate_teacher_forced_cached_batched(
    model: Any,
    examples: Sequence[ContinuationExample],
    backend: Any,
) -> tuple[DocumentNLL, ...]:
    """Score independent equal-length caches in q_len=1 microbatches."""

    examples = tuple(examples)
    if not examples:
        raise ValueError("at least one cached document is required")
    document_ids = tuple(example.document_id for example in examples)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document IDs must be unique within a microbatch")
    initial_lengths = {example.prefill.cache_length for example in examples}
    step_counts = {len(example.continuation_ids) - 1 for example in examples}
    next_positions = {
        example.prefill.position_ids[0][-1] + 1 for example in examples
    }
    if len(initial_lengths) != 1:
        raise ValueError("microbatched caches must have equal prompt lengths")
    if len(step_counts) != 1:
        raise ValueError("microbatched continuations must have equal lengths")
    if len(next_positions) != 1:
        raise ValueError("microbatched caches must share the next position")
    initial_length = next(iter(initial_lengths))
    step_count = next(iter(step_counts))
    if any(
        example.decode_cache.cache_length != initial_length
        for example in examples
    ):
        raise AssertionError("decode admission changed a cache length")

    artifacts = tuple(example.decode_cache for example in examples)
    cache = backend.materialize_cache_batch(artifacts)
    if backend.cache_batch_size(cache) != len(examples):
        raise AssertionError("materialized cache batch size is inconsistent")
    if backend.cache_length(cache) != initial_length:
        raise AssertionError("materialized cache length does not match prefill")

    losses: list[list[float]] = [[] for _ in examples]
    next_position = next(iter(next_positions))
    for step_index in range(step_count):
        previous_length = backend.cache_length(cache)
        expected_length = initial_length + step_index
        if previous_length != expected_length:
            raise AssertionError(
                f"cache length before step {step_index} is {previous_length}, "
                f"expected {expected_length}"
            )
        masks = tuple(
            example.prefill.attention_mask[0] + (1,) * (step_index + 1)
            for example in examples
        )
        result = backend.decode_step_batch(
            model,
            input_token_ids=tuple(
                example.continuation_ids[step_index] for example in examples
            ),
            cache=cache,
            attention_masks=masks,
            position_ids=(next_position + step_index,) * len(examples),
            cache_position=previous_length,
        )
        grown_length = backend.cache_length(result.cache)
        if grown_length != previous_length + 1:
            raise AssertionError(
                "a q_len=1 microbatch must grow every cache by one entry"
            )
        cache = backend.commit_append_batch(
            result.cache,
            previous_length=previous_length,
            artifacts=artifacts,
        )
        if backend.cache_length(cache) != grown_length:
            raise AssertionError("append conversion changed the logical cache length")
        step_losses = backend.token_nll_batch(
            result.logits,
            tuple(
                example.continuation_ids[step_index + 1]
                for example in examples
            ),
        )
        if len(step_losses) != len(examples):
            raise AssertionError("backend returned the wrong number of token losses")
        for index, loss in enumerate(step_losses):
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite NLL for document {document_ids[index]!r}, "
                    f"decode step {step_index}"
                )
            losses[index].append(value)

    final_length = backend.cache_length(cache)
    return tuple(
        DocumentNLL(
            document_id=example.document_id,
            nll_sum=sum(document_losses),
            token_count=len(document_losses),
            per_token_nll=tuple(document_losses),
            first_token_id=example.continuation_ids[0],
            scored_token_ids=example.continuation_ids[1:],
            initial_cache_length=initial_length,
            final_cache_length=final_length,
        )
        for example, document_losses in zip(examples, losses)
    )


def evaluate_cached_documents(
    model: Any,
    examples: Sequence[ContinuationExample],
    backend: CachedDecodeBackend,
) -> CorpusNLL:
    """Evaluate documents independently and aggregate by scored-token count."""
    examples = tuple(examples)
    ids = [example.document_id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("document IDs must be unique")
    return CorpusNLL(
        tuple(
            evaluate_teacher_forced_cached(model, example, backend)
            for example in examples
        )
    )


def _cache_from_layers(legacy: tuple[tuple[Any, Any], ...]) -> Any:
    """Build a transformers Cache from per-layer (key, value) pairs.

    ``DynamicCache.from_legacy_cache`` no longer exists on current
    transformers; layer-wise ``update`` is the stable construction API.
    """

    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(legacy):
        cache.update(key, value, layer_idx)
    return cache


def _legacy_cache_layers(cache: Any) -> tuple[tuple[Any, Any], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache = tuple(zip(cache.key_cache, cache.value_cache))
    layers = tuple(cache)
    if not layers or any(len(layer) < 2 for layer in layers):
        raise ValueError("past_key_values does not contain K/V layers")
    return tuple((layer[0], layer[1]) for layer in layers)


def capture_bf16_prefill(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    model_revision: str,
    tokenizer_revision: str,
    provenance: ArtifactProvenance,
    expected_first_token_ids: Any | None = None,
    position_ids: Any | None = None,
    metadata: Sequence[tuple[str, str]] | dict[str, str] | None = None,
) -> PrefillArtifact:
    """Run one untouched BF16 prefill and capture its reusable cache."""
    import torch

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
    if input_ids.shape[1] <= 1:
        raise ValueError("prefill requires a prompt longer than one token")
    if position_ids is None:
        position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
        position_ids = position_ids.clamp_min(0)
    if position_ids.shape != input_ids.shape:
        raise ValueError("position_ids must match input_ids")
    device = input_ids.device
    cache_position = torch.arange(input_ids.shape[1], device=device)
    token_offsets = torch.arange(input_ids.shape[1], device=device)[None, :]
    last_indices = torch.where(
        attention_mask.to(torch.bool),
        token_offsets,
        torch.full_like(token_offsets, -1),
    ).amax(dim=-1)
    if torch.any(last_indices < 0):
        raise ValueError("each prompt must contain at least one active token")
    logit_positions, row_logit_indices = torch.unique(
        last_indices,
        sorted=True,
        return_inverse=True,
    )
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=logit_positions,
        )
    logits = output.logits
    expected_prefix = (input_ids.shape[0], logit_positions.numel())
    if logits.ndim != 3 or logits.shape[:2] != expected_prefix:
        raise AssertionError(
            "prefill logits must contain each requested final active position"
        )
    if logits.dtype != torch.bfloat16:
        raise TypeError("the BF16 prefill head must emit BF16 logits")
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    last_logits = logits[batch_indices, row_logit_indices]
    selection = "greedy"
    if expected_first_token_ids is None:
        first_tokens = last_logits.argmax(dim=-1)
    else:
        first_tokens = torch.as_tensor(
            expected_first_token_ids, dtype=torch.long, device=device
        ).reshape(-1)
        if first_tokens.shape[0] != input_ids.shape[0]:
            raise ValueError("expected_first_token_ids must match the prompt batch")
        selection = "teacher_forced"
    first_log_probs = torch.log_softmax(last_logits.float(), dim=-1).gather(
        -1, first_tokens[:, None]
    )

    layers = []
    for key, value in _legacy_cache_layers(output.past_key_values):
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise TypeError("prefill model must emit BF16 K/V tensors")
        layers.append(
            KVLayerPayload(
                key=TensorPayload.from_torch(key),
                value=TensorPayload.from_torch(value),
            )
        )
    expected_cache_length = input_ids.shape[1]
    if any(layer.key.shape[-2] != expected_cache_length for layer in layers):
        raise AssertionError("prefill cache length does not match the prompt")

    def rows(tensor: Any) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(value) for value in row)
            for row in tensor.detach().to("cpu").tolist()
        )

    return PrefillArtifact.create(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        input_ids=rows(input_ids),
        attention_mask=rows(attention_mask),
        position_ids=rows(position_ids),
        first_token=FirstTokenResult(
            token_ids=tuple(
                int(value) for value in first_tokens.detach().to("cpu").tolist()
            ),
            selection=selection,
            log_probabilities=tuple(
                float(value)
                for value in first_log_probs[:, 0].detach().to("cpu").tolist()
            ),
        ),
        layers=layers,
        provenance=provenance,
        metadata=metadata,
    )


AppendTransform = Callable[[Any, int, int, DecodeCacheArtifact], Any]
AppendValidator = Callable[[Any, int, int, DecodeCacheArtifact], None]


class TorchHFCachedDecodeBackend:
    """Torch/Hugging Face backend with explicit cache-append conversion."""

    def __init__(
        self,
        *,
        device: str | Any,
        cache_factory: Callable[[tuple[tuple[Any, Any], ...]], Any] | None = None,
        append_transform: AppendTransform | None = None,
        append_validator: AppendValidator | None = None,
        native_append_format: bool = False,
        execution_batch_width: int | None = None,
    ) -> None:
        # GPU GEMM kernel selection depends on the total batch count, so the
        # same lane rounds differently at different batch sizes. Padding every
        # forward to one fixed width keeps kernel dispatch identical across
        # microbatch sizes; lane isolation keeps pad lanes from affecting real
        # lanes. Only the plain-cache path supports it: append transforms and
        # cache factories assume the cache holds exactly the real lanes.
        if execution_batch_width is not None:
            if int(execution_batch_width) < 1:
                raise ValueError("execution_batch_width must be positive")
            if (
                cache_factory is not None
                or append_transform is not None
                or append_validator is not None
            ):
                raise ValueError(
                    "execution_batch_width requires the plain cache-append path"
                )
        self.device = device
        self.cache_factory = cache_factory
        self.append_transform = append_transform
        self.append_validator = append_validator
        self.native_append_format = native_append_format
        self.execution_batch_width = (
            None if execution_batch_width is None else int(execution_batch_width)
        )

    def _padded_legacy(
        self,
        legacy: tuple[tuple[Any, Any], ...],
        logical_batch: int,
    ) -> tuple[tuple[Any, Any], ...]:
        import torch

        width = self.execution_batch_width
        if width is None or width == logical_batch:
            return legacy
        if logical_batch > width:
            raise ValueError(
                f"cache batch {logical_batch} exceeds execution width {width}"
            )
        pad = width - logical_batch
        return tuple(
            (
                torch.cat((key, *(key[:1],) * pad), dim=0),
                torch.cat((value, *(value[:1],) * pad), dim=0),
            )
            for key, value in legacy
        )

    def _tag_logical_batch(self, cache: Any, logical_batch: int) -> Any:
        if self.execution_batch_width is not None:
            cache._decode_logical_batch = int(logical_batch)
        return cache

    def adopt_cache(self, cache: Any, *, logical_batch: int = 1) -> Any:
        """Pad a live model cache to the execution width for decode stepping."""
        if self.execution_batch_width is None:
            return cache
        legacy = self._padded_legacy(_legacy_cache_layers(cache), logical_batch)
        return self._tag_logical_batch(_cache_from_layers(legacy), logical_batch)

    def materialize_cache(self, artifact: DecodeCacheArtifact) -> Any:
        legacy = tuple(
            (
                layer.key.numerical_view.to_torch(self.device),
                layer.value.numerical_view.to_torch(self.device),
            )
            for layer in artifact.layers
        )
        if self.cache_factory is not None:
            return self.cache_factory(legacy)
        legacy = self._padded_legacy(legacy, 1)
        return self._tag_logical_batch(_cache_from_layers(legacy), 1)

    def materialize_cache_batch(
        self,
        artifacts: Sequence[DecodeCacheArtifact],
    ) -> Any:
        import torch

        artifacts = tuple(artifacts)
        if not artifacts:
            raise ValueError("at least one cache artifact is required")
        layer_count = len(artifacts[0].layers)
        reference = artifacts[0]
        for artifact in artifacts:
            if artifact.batch_size != 1:
                raise ValueError("microbatch inputs must be batch-one artifacts")
            if len(artifact.layers) != layer_count:
                raise ValueError("cache artifacts have different layer counts")
            if (
                artifact.cache_length != reference.cache_length
                or artifact.layout_id != reference.layout_id
                or artifact.key_format != reference.key_format
                or artifact.value_format != reference.value_format
            ):
                raise ValueError("cache artifacts have incompatible contracts")
        legacy = tuple(
            (
                torch.cat(
                    tuple(
                        artifact.layers[index].key.numerical_view.to_torch(
                            self.device
                        )
                        for artifact in artifacts
                    ),
                    dim=0,
                ),
                torch.cat(
                    tuple(
                        artifact.layers[index].value.numerical_view.to_torch(
                            self.device
                        )
                        for artifact in artifacts
                    ),
                    dim=0,
                ),
            )
            for index in range(layer_count)
        )
        if self.cache_factory is not None:
            return self.cache_factory(legacy)
        logical_batch = len(artifacts)
        legacy = self._padded_legacy(legacy, logical_batch)
        return self._tag_logical_batch(_cache_from_layers(legacy), logical_batch)

    def _first_key(self, cache: Any) -> Any:
        return _legacy_cache_layers(cache)[0][0]

    def cache_length(self, cache: Any) -> int:
        if hasattr(cache, "get_seq_length"):
            length = cache.get_seq_length()
            return int(length.item() if hasattr(length, "item") else length)
        return int(self._first_key(cache).shape[-2])

    def cache_batch_size(self, cache: Any) -> int:
        logical = getattr(cache, "_decode_logical_batch", None)
        if logical is not None:
            return int(logical)
        return int(self._first_key(cache).shape[0])

    def decode_step(
        self,
        model: Any,
        *,
        input_token_id: int,
        cache: Any,
        attention_mask: tuple[int, ...],
        position_id: int,
        cache_position: int,
    ) -> DecodeStep:
        import torch

        width = self.execution_batch_width or 1
        input_ids = torch.tensor(
            [[input_token_id]] * width, dtype=torch.long, device=self.device
        )
        if input_ids.shape[1] != 1:
            raise AssertionError("decode input must have q_len=1")
        mask = torch.tensor(
            [attention_mask] * width, dtype=torch.long, device=self.device
        )
        positions = torch.tensor(
            [[position_id]] * width, dtype=torch.long, device=self.device
        )
        cache_positions = torch.tensor(
            [cache_position], dtype=torch.long, device=self.device
        )
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=positions,
                cache_position=cache_positions,
                past_key_values=cache,
                use_cache=True,
            )
        if output.logits.ndim != 3 or output.logits.shape[:2] != (width, 1):
            raise AssertionError("decode logits must have shape [1, 1, vocabulary]")
        cache = self._tag_logical_batch(output.past_key_values, 1)
        return DecodeStep(logits=output.logits[:1], cache=cache)

    def decode_step_batch(
        self,
        model: Any,
        *,
        input_token_ids: Sequence[int],
        cache: Any,
        attention_masks: Sequence[Sequence[int]],
        position_ids: Sequence[int],
        cache_position: int,
    ) -> DecodeStep:
        import torch

        tokens = tuple(int(token) for token in input_token_ids)
        masks = tuple(tuple(int(value) for value in row) for row in attention_masks)
        positions_values = tuple(int(value) for value in position_ids)
        batch = len(tokens)
        if (
            batch == 0
            or len(masks) != batch
            or len(positions_values) != batch
        ):
            raise ValueError("batched decode inputs have inconsistent batch sizes")
        if len({len(row) for row in masks}) != 1:
            raise ValueError("batched attention masks must have equal lengths")
        width = batch
        if self.execution_batch_width is not None:
            width = self.execution_batch_width
            if batch > width:
                raise ValueError(
                    f"decode batch {batch} exceeds execution width {width}"
                )
            pad = width - batch
            tokens = tokens + (tokens[0],) * pad
            masks = masks + (masks[0],) * pad
            positions_values = positions_values + (positions_values[0],) * pad
        input_ids = torch.tensor(
            tokens,
            dtype=torch.long,
            device=self.device,
        )[:, None]
        if input_ids.shape[1] != 1:
            raise AssertionError("decode input must have q_len=1")
        mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        positions = torch.tensor(
            positions_values,
            dtype=torch.long,
            device=self.device,
        )[:, None]
        cache_positions = torch.tensor(
            [cache_position],
            dtype=torch.long,
            device=self.device,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=positions,
                cache_position=cache_positions,
                past_key_values=cache,
                use_cache=True,
            )
        if output.logits.ndim != 3 or output.logits.shape[:2] != (width, 1):
            raise AssertionError(
                "decode logits must have shape [batch, 1, vocabulary]"
            )
        cache = self._tag_logical_batch(output.past_key_values, batch)
        return DecodeStep(logits=output.logits[:batch], cache=cache)

    def commit_append(
        self,
        cache: Any,
        *,
        previous_length: int,
        artifact: DecodeCacheArtifact,
    ) -> Any:
        if self.append_validator is not None:
            self.append_validator(
                cache,
                previous_length,
                previous_length + 1,
                artifact,
            )
        if self.append_transform is not None:
            return self.append_transform(
                cache, previous_length, previous_length + 1, artifact
            )
        baseline = all(
            layer.key.format_id == "BF16" and layer.value.format_id == "BF16"
            for layer in artifact.layers
        )
        if not baseline and not self.native_append_format:
            raise RuntimeError(
                "quantized decode requires append_transform or native_append_format=True"
            )
        return cache

    def commit_append_batch(
        self,
        cache: Any,
        *,
        previous_length: int,
        artifacts: Sequence[DecodeCacheArtifact],
    ) -> Any:
        artifacts = tuple(artifacts)
        if not artifacts:
            raise ValueError("at least one cache artifact is required")
        reference = artifacts[0]
        if any(
            (
                artifact.layout_id,
                artifact.key_format,
                artifact.value_format,
            )
            != (
                reference.layout_id,
                reference.key_format,
                reference.value_format,
            )
            for artifact in artifacts
        ):
            raise ValueError("append artifacts have incompatible contracts")
        if self.append_validator is not None:
            self.append_validator(
                cache,
                previous_length,
                previous_length + 1,
                reference,
            )
        if self.append_transform is not None:
            raise RuntimeError(
                "batched append_transform requires a role-aware implementation"
            )
        baseline = all(
            layer.key.format_id == "BF16" and layer.value.format_id == "BF16"
            for layer in reference.layers
        )
        if not baseline and not self.native_append_format:
            raise RuntimeError(
                "quantized decode requires native_append_format=True"
            )
        return cache

    def token_nll(self, logits: Any, target_token_id: int) -> float:
        import torch

        vocabulary = logits.shape[-1]
        if not 0 <= target_token_id < vocabulary:
            raise ValueError(
                f"target token {target_token_id} is outside vocabulary {vocabulary}"
            )
        value = -torch.log_softmax(logits[0, 0].float(), dim=-1)[target_token_id]
        return float(value.detach().to("cpu"))

    def token_nll_batch(
        self,
        logits: Any,
        target_token_ids: Sequence[int],
    ) -> tuple[float, ...]:
        import torch

        targets = torch.tensor(
            tuple(int(token) for token in target_token_ids),
            dtype=torch.long,
            device=logits.device,
        )
        if targets.shape != (logits.shape[0],):
            raise ValueError("target batch size differs from logits")
        vocabulary = logits.shape[-1]
        if torch.any((targets < 0) | (targets >= vocabulary)):
            raise ValueError("a target token is outside the vocabulary")
        values = -torch.log_softmax(logits[:, 0].float(), dim=-1).gather(
            -1,
            targets[:, None],
        )[:, 0]
        return tuple(
            float(value)
            for value in values.detach().to("cpu").tolist()
        )
