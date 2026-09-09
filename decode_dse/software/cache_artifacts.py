"""Content-addressed artifacts for the prefill-to-decode boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence


_PREFILL_SCHEMA = 1
_DECODE_CACHE_SCHEMA = 2
_TENSOR_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _freeze_rows(rows: Sequence[Sequence[int]], name: str) -> tuple[tuple[int, ...], ...]:
    frozen = tuple(tuple(int(value) for value in row) for row in rows)
    if not frozen or any(not row for row in frozen):
        raise ValueError(f"{name} must contain at least one non-empty row")
    width = len(frozen[0])
    if any(len(row) != width for row in frozen):
        raise ValueError(f"{name} must be rectangular")
    return frozen


def _freeze_metadata(
    metadata: Sequence[tuple[str, str]] | dict[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    items = metadata.items() if isinstance(metadata, dict) else metadata or ()
    frozen = tuple(sorted((str(key), str(value)) for key, value in items))
    if len({key for key, _ in frozen}) != len(frozen):
        raise ValueError("metadata keys must be unique")
    return frozen


def compute_prompt_hash(
    input_ids: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]],
    position_ids: Sequence[Sequence[int]],
) -> str:
    """Return the canonical identity of the tokenized prompt batch."""
    payload = {
        "input_ids": [list(row) for row in input_ids],
        "attention_mask": [list(row) for row in attention_mask],
        "position_ids": [list(row) for row in position_ids],
    }
    return _sha256(_canonical_json(payload))


@dataclass(frozen=True)
class ArtifactProvenance:
    """Reproducibility fields embedded in an artifact identity."""

    producer: str
    code_revision: str
    created_at_utc: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.producer or not self.code_revision or not self.created_at_utc:
            raise ValueError("producer, code_revision, and created_at_utc are required")
        object.__setattr__(self, "parameters", _freeze_metadata(self.parameters))

    def descriptor(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "code_revision": self.code_revision,
            "created_at_utc": self.created_at_utc,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class TensorPayload:
    """An immutable tensor encoded as contiguous little-endian bytes."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes = field(repr=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        shape = tuple(int(size) for size in self.shape)
        if any(size < 0 for size in shape):
            raise ValueError("tensor dimensions must be non-negative")
        if self.dtype not in _TENSOR_BYTES:
            raise ValueError(f"unsupported tensor dtype {self.dtype!r}")
        data = bytes(self.data)
        elements = 1
        for size in shape:
            elements *= size
        expected = elements * _TENSOR_BYTES[self.dtype]
        if len(data) != expected:
            raise ValueError(
                f"{self.dtype}{shape} requires {expected} bytes, received {len(data)}"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sha256", _sha256(data))

    @classmethod
    def from_torch(cls, tensor: Any, *, dtype: str | None = None) -> "TensorPayload":
        """Copy a torch tensor into a framework-independent payload."""
        import torch

        dtype_name = dtype or str(tensor.dtype).removeprefix("torch.")
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }.get(dtype_name)
        if torch_dtype is None:
            raise ValueError(f"unsupported torch dtype {dtype_name!r}")
        value = tensor.detach().to(device="cpu", dtype=torch_dtype).contiguous()
        raw = value.view(torch.uint8).numpy().tobytes(order="C")
        return cls(dtype=dtype_name, shape=tuple(value.shape), data=raw)

    def to_torch(self, device: str | Any = "cpu") -> Any:
        """Materialize a writable torch tensor with independent storage."""
        import torch

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }[self.dtype]
        raw = torch.frombuffer(bytearray(self.data), dtype=torch.uint8)
        return raw.view(torch_dtype).reshape(self.shape).to(device=device)

    def descriptor(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape), "sha256": self.sha256}


@dataclass(frozen=True)
class KVLayerPayload:
    """The BF16 K/V state for one decoder layer."""

    key: TensorPayload
    value: TensorPayload

    def __post_init__(self) -> None:
        if self.key.dtype != "bfloat16" or self.value.dtype != "bfloat16":
            raise ValueError("prefill K/V tensors must be stored as bfloat16")
        if self.key.shape != self.value.shape:
            raise ValueError("key and value tensors must have identical shapes")
        if len(self.key.shape) != 4:
            raise ValueError("K/V tensors must use [batch, heads, sequence, head_dim]")

    def descriptor(self) -> dict[str, Any]:
        return {"key": self.key.descriptor(), "value": self.value.descriptor()}


@dataclass(frozen=True)
class FirstTokenResult:
    """The token selected by prefill and its optional log probability."""

    token_ids: tuple[int, ...]
    selection: Literal["greedy", "teacher_forced"]
    log_probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        tokens = tuple(int(token) for token in self.token_ids)
        if not tokens:
            raise ValueError("first-token result cannot be empty")
        object.__setattr__(self, "token_ids", tokens)
        if self.log_probabilities is not None:
            log_probs = tuple(float(value) for value in self.log_probabilities)
            if len(log_probs) != len(tokens):
                raise ValueError("first-token log probabilities must match the batch")
            object.__setattr__(self, "log_probabilities", log_probs)

    def descriptor(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "selection": self.selection,
            "log_probabilities": (
                list(self.log_probabilities)
                if self.log_probabilities is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PrefillArtifact:
    """Immutable BF16 output of a prefill-chip invocation."""

    model_revision: str
    tokenizer_revision: str
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    prompt_hash: str
    first_token: FirstTokenResult
    layers: tuple[KVLayerPayload, ...]
    provenance: ArtifactProvenance
    metadata: tuple[tuple[str, str], ...] = ()
    schema_version: int = _PREFILL_SCHEMA
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        input_ids = _freeze_rows(self.input_ids, "input_ids")
        attention_mask = _freeze_rows(self.attention_mask, "attention_mask")
        position_ids = _freeze_rows(self.position_ids, "position_ids")
        if not (len(input_ids) == len(attention_mask) == len(position_ids)):
            raise ValueError("prompt tensors must have the same batch size")
        if any(
            len(ids) != len(mask) or len(ids) != len(pos)
            for ids, mask, pos in zip(input_ids, attention_mask, position_ids)
        ):
            raise ValueError("prompt tensors must have identical sequence lengths")
        if any(value not in (0, 1) for row in attention_mask for value in row):
            raise ValueError("attention_mask values must be 0 or 1")
        if len(self.first_token.token_ids) != len(input_ids):
            raise ValueError("first-token batch size does not match the prompt")
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("prefill artifact must contain at least one K/V layer")
        batch_size = len(input_ids)
        cache_length = len(input_ids[0])
        for layer in layers:
            if layer.key.shape[0] != batch_size or layer.key.shape[-2] != cache_length:
                raise ValueError("K/V shape does not match the prompt batch")
        expected_prompt_hash = compute_prompt_hash(input_ids, attention_mask, position_ids)
        if self.prompt_hash != expected_prompt_hash:
            raise ValueError("prompt_hash does not match the tokenized prompt")
        if self.schema_version != _PREFILL_SCHEMA:
            raise ValueError(f"unsupported prefill schema {self.schema_version}")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "attention_mask", attention_mask)
        object.__setattr__(self, "position_ids", position_ids)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "artifact_id", _sha256(_canonical_json(self.descriptor())))

    @classmethod
    def create(
        cls,
        *,
        model_revision: str,
        tokenizer_revision: str,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        position_ids: Sequence[Sequence[int]],
        first_token: FirstTokenResult,
        layers: Sequence[KVLayerPayload],
        provenance: ArtifactProvenance,
        metadata: Sequence[tuple[str, str]] | dict[str, str] | None = None,
    ) -> "PrefillArtifact":
        frozen_ids = _freeze_rows(input_ids, "input_ids")
        frozen_mask = _freeze_rows(attention_mask, "attention_mask")
        frozen_positions = _freeze_rows(position_ids, "position_ids")
        return cls(
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            input_ids=frozen_ids,
            attention_mask=frozen_mask,
            position_ids=frozen_positions,
            prompt_hash=compute_prompt_hash(
                frozen_ids, frozen_mask, frozen_positions
            ),
            first_token=first_token,
            layers=tuple(layers),
            provenance=provenance,
            metadata=_freeze_metadata(metadata),
        )

    @property
    def batch_size(self) -> int:
        return len(self.input_ids)

    @property
    def cache_length(self) -> int:
        return len(self.input_ids[0])

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "plena.prefill",
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt_hash": self.prompt_hash,
            "input_ids": [list(row) for row in self.input_ids],
            "attention_mask": [list(row) for row in self.attention_mask],
            "position_ids": [list(row) for row in self.position_ids],
            "first_token": self.first_token.descriptor(),
            "layers": [layer.descriptor() for layer in self.layers],
            "provenance": self.provenance.descriptor(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrefillDecodeMetadata:
    """Verified manifest-only fields needed by cached decode.

    Decode consumes the separately admitted K/V artifact, so repeatedly
    loading the source BF16 tensor payloads cannot affect its arithmetic.  This
    view retains the content address and every prompt/first-token field used by
    :class:`ContinuationExample` while avoiding redundant BF16 K/V reads for
    every precision profile.
    """

    artifact_id: str
    prompt_hash: str
    attention_mask: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    first_token: FirstTokenResult
    layer_count: int
    _batch_size: int
    _cache_length: int

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cache_length(self) -> int:
        return self._cache_length


@dataclass(frozen=True)
class QuantizedTensorPayload:
    """Physical cache planes and the corresponding numerical tensor."""

    format_id: str
    block_size: int
    element_bits: int
    logical_shape: tuple[int, ...]
    element_plane: bytes = field(repr=False)
    scale_plane: bytes = field(repr=False)
    numerical_view: TensorPayload = field(repr=False)
    element_encoding: str
    scale_encoding: str
    element_sha256: str = field(init=False)
    scale_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        logical_shape = tuple(int(size) for size in self.logical_shape)
        if not self.format_id:
            raise ValueError("format_id is required")
        if not self.element_encoding or not self.scale_encoding:
            raise ValueError("element and scale encodings are required")
        if self.block_size <= 0 or self.element_bits <= 0:
            raise ValueError("block_size and element_bits must be positive")
        if logical_shape != self.numerical_view.shape:
            raise ValueError("numerical view must match the logical cache shape")
        if len(logical_shape) != 4:
            raise ValueError("cache tensors must use [batch, heads, sequence, head_dim]")
        element_plane = bytes(self.element_plane)
        scale_plane = bytes(self.scale_plane)
        if not element_plane:
            raise ValueError("element_plane cannot be empty")
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "element_plane", element_plane)
        object.__setattr__(self, "scale_plane", scale_plane)
        object.__setattr__(self, "element_sha256", _sha256(element_plane))
        object.__setattr__(self, "scale_sha256", _sha256(scale_plane))

    def descriptor(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "block_size": self.block_size,
            "element_bits": self.element_bits,
            "element_encoding": self.element_encoding,
            "scale_encoding": self.scale_encoding,
            "logical_shape": list(self.logical_shape),
            "element_plane": {
                "bytes": len(self.element_plane),
                "sha256": self.element_sha256,
            },
            "scale_plane": {
                "bytes": len(self.scale_plane),
                "sha256": self.scale_sha256,
            },
            "numerical_view": self.numerical_view.descriptor(),
        }


def _quantized_signature(
    tensor: QuantizedTensorPayload,
) -> tuple[str, int, int, str, str]:
    return (
        tensor.format_id,
        tensor.block_size,
        tensor.element_bits,
        tensor.element_encoding,
        tensor.scale_encoding,
    )


@dataclass(frozen=True)
class DecodeKVLayerPayload:
    """The admitted K/V cache representation for one layer."""

    key: QuantizedTensorPayload
    value: QuantizedTensorPayload

    def __post_init__(self) -> None:
        if self.key.logical_shape != self.value.logical_shape:
            raise ValueError("decode key and value shapes must match")

    def descriptor(self) -> dict[str, Any]:
        return {"key": self.key.descriptor(), "value": self.value.descriptor()}


@dataclass(frozen=True)
class DecodeCacheArtifact:
    """Immutable decode-ingress cache derived from one prefill artifact."""

    source_artifact_id: str
    precision_id: str
    layout_id: str
    layers: tuple[DecodeKVLayerPayload, ...]
    provenance: ArtifactProvenance
    metadata: tuple[tuple[str, str], ...] = ()
    schema_version: int = _DECODE_CACHE_SCHEMA
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.source_artifact_id or not self.precision_id or not self.layout_id:
            raise ValueError("source, precision, and layout identities are required")
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("decode cache must contain at least one K/V layer")
        reference_shape = layers[0].key.logical_shape
        key_signature = _quantized_signature(layers[0].key)
        value_signature = _quantized_signature(layers[0].value)
        for layer in layers:
            if layer.key.logical_shape[:1] != reference_shape[:1]:
                raise ValueError("decode cache layers must have a common batch size")
            if layer.key.logical_shape[-2] != reference_shape[-2]:
                raise ValueError("decode cache layers must have a common sequence length")
            if _quantized_signature(layer.key) != key_signature:
                raise ValueError("decode key format must be uniform across layers")
            if _quantized_signature(layer.value) != value_signature:
                raise ValueError("decode value format must be uniform across layers")
        if self.schema_version != _DECODE_CACHE_SCHEMA:
            raise ValueError(f"unsupported decode-cache schema {self.schema_version}")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "artifact_id", _sha256(_canonical_json(self.descriptor())))

    @property
    def batch_size(self) -> int:
        return self.layers[0].key.logical_shape[0]

    @property
    def cache_length(self) -> int:
        return self.layers[0].key.logical_shape[-2]

    @property
    def key_format(self) -> str:
        return self.layers[0].key.format_id

    @property
    def value_format(self) -> str:
        return self.layers[0].value.format_id

    @property
    def split_kv(self) -> bool:
        return _quantized_signature(self.layers[0].key) != _quantized_signature(
            self.layers[0].value
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "plena.decode_cache",
            "schema_version": self.schema_version,
            "source_artifact_id": self.source_artifact_id,
            "precision_id": self.precision_id,
            "layout_id": self.layout_id,
            "layers": [layer.descriptor() for layer in self.layers],
            "provenance": self.provenance.descriptor(),
            "metadata": dict(self.metadata),
        }


class CacheTensorConverter(Protocol):
    """Convert one BF16 K/V tensor to its physical decode representation."""

    def convert(
        self,
        tensor: TensorPayload,
        *,
        role: Literal["key", "value"],
        layer_index: int,
        precision_id: str,
        layout_id: str,
    ) -> QuantizedTensorPayload:
        ...


@dataclass(frozen=True)
class FunctionalCacheConverter:
    """Adapter for a quantize-and-pack function used during cache admission."""

    function: Callable[
        [TensorPayload, Literal["key", "value"], int, str, str],
        QuantizedTensorPayload,
    ]

    def convert(
        self,
        tensor: TensorPayload,
        *,
        role: Literal["key", "value"],
        layer_index: int,
        precision_id: str,
        layout_id: str,
    ) -> QuantizedTensorPayload:
        return self.function(tensor, role, layer_index, precision_id, layout_id)


@dataclass(frozen=True)
class SplitCacheConverter:
    """Dispatch admission conversion independently for K and V."""

    key_converter: CacheTensorConverter
    value_converter: CacheTensorConverter

    def convert(
        self,
        tensor: TensorPayload,
        *,
        role: Literal["key", "value"],
        layer_index: int,
        precision_id: str,
        layout_id: str,
    ) -> QuantizedTensorPayload:
        converter = (
            self.key_converter if role == "key" else self.value_converter
        )
        return converter.convert(
            tensor,
            role=role,
            layer_index=layer_index,
            precision_id=precision_id,
            layout_id=layout_id,
        )


@dataclass(frozen=True)
class BF16CacheConverter:
    """Lossless baseline converter with BF16 values as the element plane."""

    def convert(
        self,
        tensor: TensorPayload,
        *,
        role: Literal["key", "value"],
        layer_index: int,
        precision_id: str,
        layout_id: str,
    ) -> QuantizedTensorPayload:
        del role, layer_index, precision_id, layout_id
        return QuantizedTensorPayload(
            format_id="BF16",
            block_size=1,
            element_bits=16,
            logical_shape=tensor.shape,
            element_plane=tensor.data,
            scale_plane=b"",
            numerical_view=tensor,
            element_encoding="BF16_LE",
            scale_encoding="NONE",
        )


def admit_prefill_cache(
    prefill: PrefillArtifact,
    *,
    precision_id: str,
    layout_id: str,
    converter: CacheTensorConverter,
    provenance: ArtifactProvenance,
    metadata: Sequence[tuple[str, str]] | dict[str, str] | None = None,
) -> DecodeCacheArtifact:
    """Quantize and pack an immutable BF16 cache exactly once at admission."""
    layers: list[DecodeKVLayerPayload] = []
    for layer_index, layer in enumerate(prefill.layers):
        key = converter.convert(
            layer.key,
            role="key",
            layer_index=layer_index,
            precision_id=precision_id,
            layout_id=layout_id,
        )
        value = converter.convert(
            layer.value,
            role="value",
            layer_index=layer_index,
            precision_id=precision_id,
            layout_id=layout_id,
        )
        layers.append(DecodeKVLayerPayload(key=key, value=value))
    admitted = DecodeCacheArtifact(
        source_artifact_id=prefill.artifact_id,
        precision_id=precision_id,
        layout_id=layout_id,
        layers=tuple(layers),
        provenance=provenance,
        metadata=_freeze_metadata(metadata),
    )
    if admitted.batch_size != prefill.batch_size:
        raise ValueError("admission changed the cache batch size")
    if admitted.cache_length != prefill.cache_length:
        raise ValueError("admission changed the cache sequence length")
    return admitted


def admit_prefill_cache_split(
    prefill: PrefillArtifact,
    *,
    precision_id: str,
    layout_id: str,
    key_converter: CacheTensorConverter,
    value_converter: CacheTensorConverter,
    provenance: ArtifactProvenance,
    key_format: str,
    value_format: str,
    metadata: Sequence[tuple[str, str]] | dict[str, str] | None = None,
) -> DecodeCacheArtifact:
    """Admit BF16 prompt K/V into independently declared cache formats."""

    artifact = admit_prefill_cache(
        prefill,
        precision_id=precision_id,
        layout_id=layout_id,
        converter=SplitCacheConverter(key_converter, value_converter),
        provenance=provenance,
        metadata=metadata,
    )
    if artifact.key_format != key_format:
        raise ValueError("admitted key format differs from the refinement profile")
    if artifact.value_format != value_format:
        raise ValueError("admitted value format differs from the refinement profile")
    return artifact


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_directory_write(path: Path, writer: Callable[[Path], None]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"artifact path already exists: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        writer(temporary)
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def save_prefill_artifact(artifact: PrefillArtifact, path: str | Path) -> None:
    """Write a prefill artifact atomically without pickle serialization."""
    destination = Path(path)

    def writer(root: Path) -> None:
        for index, layer in enumerate(artifact.layers):
            _write_bytes(root / "layers" / f"{index:04d}.key.bin", layer.key.data)
            _write_bytes(root / "layers" / f"{index:04d}.value.bin", layer.value.data)
        manifest = artifact.descriptor() | {"artifact_id": artifact.artifact_id}
        _write_bytes(root / "manifest.json", _canonical_json(manifest))

    _atomic_directory_write(destination, writer)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_tensor(descriptor: dict[str, Any], path: Path) -> TensorPayload:
    payload = TensorPayload(
        dtype=descriptor["dtype"],
        shape=tuple(descriptor["shape"]),
        data=path.read_bytes(),
    )
    if payload.sha256 != descriptor["sha256"]:
        raise ValueError(f"tensor checksum mismatch: {path}")
    return payload


def _load_provenance(value: dict[str, Any]) -> ArtifactProvenance:
    return ArtifactProvenance(
        producer=value["producer"],
        code_revision=value["code_revision"],
        created_at_utc=value["created_at_utc"],
        parameters=value.get("parameters", {}),
    )


def load_prefill_artifact(path: str | Path) -> PrefillArtifact:
    """Load and verify a prefill artifact and all tensor checksums."""
    root = Path(path)
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != "plena.prefill":
        raise ValueError("not a PLENA prefill artifact")
    layers = []
    for index, layer_desc in enumerate(manifest["layers"]):
        layers.append(
            KVLayerPayload(
                key=_load_tensor(
                    layer_desc["key"], root / "layers" / f"{index:04d}.key.bin"
                ),
                value=_load_tensor(
                    layer_desc["value"], root / "layers" / f"{index:04d}.value.bin"
                ),
            )
        )
    first_desc = manifest["first_token"]
    artifact = PrefillArtifact(
        model_revision=manifest["model_revision"],
        tokenizer_revision=manifest["tokenizer_revision"],
        input_ids=tuple(tuple(row) for row in manifest["input_ids"]),
        attention_mask=tuple(tuple(row) for row in manifest["attention_mask"]),
        position_ids=tuple(tuple(row) for row in manifest["position_ids"]),
        prompt_hash=manifest["prompt_hash"],
        first_token=FirstTokenResult(
            token_ids=tuple(first_desc["token_ids"]),
            selection=first_desc["selection"],
            log_probabilities=(
                tuple(first_desc["log_probabilities"])
                if first_desc["log_probabilities"] is not None
                else None
            ),
        ),
        layers=tuple(layers),
        provenance=_load_provenance(manifest["provenance"]),
        metadata=manifest.get("metadata", {}),
        schema_version=manifest["schema_version"],
    )
    if artifact.artifact_id != manifest["artifact_id"]:
        raise ValueError("prefill artifact identity mismatch")
    return artifact


def load_prefill_decode_metadata(path: str | Path) -> PrefillDecodeMetadata:
    """Verify and load only the source-prefill fields consumed during decode.

    The complete prefill tensor planes are independently verified when the
    admission catalog is prepared.  Steady-state numerical evaluation reads
    K/V from that admitted artifact; reading the BF16 source planes again for
    every profile is therefore redundant.  The manifest content address still
    binds those planes through their shapes and SHA-256 descriptors.
    """

    root = Path(path)
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != "plena.prefill":
        raise ValueError("not a PLENA prefill artifact")
    artifact_id = str(manifest.get("artifact_id", ""))
    descriptor = dict(manifest)
    descriptor.pop("artifact_id", None)
    if not artifact_id or _sha256(_canonical_json(descriptor)) != artifact_id:
        raise ValueError("prefill manifest identity mismatch")
    if int(manifest.get("schema_version", -1)) != _PREFILL_SCHEMA:
        raise ValueError("unsupported prefill manifest schema")

    input_ids = _freeze_rows(manifest.get("input_ids", ()), "input_ids")
    attention_mask = _freeze_rows(
        manifest.get("attention_mask", ()),
        "attention_mask",
    )
    position_ids = _freeze_rows(
        manifest.get("position_ids", ()),
        "position_ids",
    )
    if not (len(input_ids) == len(attention_mask) == len(position_ids)):
        raise ValueError("prefill prompt tensors have different batch sizes")
    if any(
        len(ids) != len(mask) or len(ids) != len(position)
        for ids, mask, position in zip(input_ids, attention_mask, position_ids)
    ):
        raise ValueError("prefill prompt tensors have different sequence lengths")
    if any(value not in (0, 1) for row in attention_mask for value in row):
        raise ValueError("prefill attention mask is invalid")
    prompt_hash = str(manifest.get("prompt_hash", ""))
    if prompt_hash != compute_prompt_hash(input_ids, attention_mask, position_ids):
        raise ValueError("prefill prompt hash mismatch")

    first_desc = manifest.get("first_token")
    if not isinstance(first_desc, dict):
        raise ValueError("prefill manifest lacks first-token metadata")
    first = FirstTokenResult(
        token_ids=tuple(first_desc.get("token_ids", ())),
        selection=first_desc.get("selection"),
        log_probabilities=(
            tuple(first_desc["log_probabilities"])
            if first_desc.get("log_probabilities") is not None
            else None
        ),
    )
    if len(first.token_ids) != len(input_ids):
        raise ValueError("prefill first-token batch size mismatch")

    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("prefill manifest has no K/V layer descriptors")
    batch_size = len(input_ids)
    cache_length = len(input_ids[0])
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise TypeError(f"prefill layer {index} descriptor must be an object")
        key = layer.get("key")
        value = layer.get("value")
        if not isinstance(key, dict) or not isinstance(value, dict):
            raise ValueError(f"prefill layer {index} lacks K/V descriptors")
        key_shape = tuple(int(size) for size in key.get("shape", ()))
        value_shape = tuple(int(size) for size in value.get("shape", ()))
        if (
            key.get("dtype") != "bfloat16"
            or value.get("dtype") != "bfloat16"
            or key_shape != value_shape
            or len(key_shape) != 4
            or key_shape[0] != batch_size
            or key_shape[-2] != cache_length
        ):
            raise ValueError(f"prefill layer {index} K/V geometry mismatch")
    return PrefillDecodeMetadata(
        artifact_id=artifact_id,
        prompt_hash=prompt_hash,
        attention_mask=attention_mask,
        position_ids=position_ids,
        first_token=first,
        layer_count=len(layers),
        _batch_size=batch_size,
        _cache_length=cache_length,
    )


def save_decode_cache_artifact(
    artifact: DecodeCacheArtifact, path: str | Path
) -> None:
    """Write a decode-cache artifact atomically without pickle serialization."""
    destination = Path(path)

    def writer(root: Path) -> None:
        for index, layer in enumerate(artifact.layers):
            for role, tensor in (("key", layer.key), ("value", layer.value)):
                prefix = root / "layers" / f"{index:04d}.{role}"
                _write_bytes(Path(f"{prefix}.elements.bin"), tensor.element_plane)
                _write_bytes(Path(f"{prefix}.scales.bin"), tensor.scale_plane)
                _write_bytes(
                    Path(f"{prefix}.numerical.bin"), tensor.numerical_view.data
                )
        manifest = artifact.descriptor() | {"artifact_id": artifact.artifact_id}
        _write_bytes(root / "manifest.json", _canonical_json(manifest))

    _atomic_directory_write(destination, writer)


def _load_quantized_tensor(
    descriptor: dict[str, Any], root: Path, index: int, role: str
) -> QuantizedTensorPayload:
    prefix = root / "layers" / f"{index:04d}.{role}"
    elements = Path(f"{prefix}.elements.bin").read_bytes()
    scales = Path(f"{prefix}.scales.bin").read_bytes()
    if len(elements) != descriptor["element_plane"]["bytes"]:
        raise ValueError(f"element-plane length mismatch at layer {index} {role}")
    if len(scales) != descriptor["scale_plane"]["bytes"]:
        raise ValueError(f"scale-plane length mismatch at layer {index} {role}")
    if _sha256(elements) != descriptor["element_plane"]["sha256"]:
        raise ValueError(f"element-plane checksum mismatch at layer {index} {role}")
    if _sha256(scales) != descriptor["scale_plane"]["sha256"]:
        raise ValueError(f"scale-plane checksum mismatch at layer {index} {role}")
    numerical = _load_tensor(
        descriptor["numerical_view"], Path(f"{prefix}.numerical.bin")
    )
    return QuantizedTensorPayload(
        format_id=descriptor["format_id"],
        block_size=descriptor["block_size"],
        element_bits=descriptor["element_bits"],
        logical_shape=tuple(descriptor["logical_shape"]),
        element_plane=elements,
        scale_plane=scales,
        numerical_view=numerical,
        element_encoding=descriptor["element_encoding"],
        scale_encoding=descriptor["scale_encoding"],
    )


def load_decode_cache_artifact(path: str | Path) -> DecodeCacheArtifact:
    """Load and verify a decode-cache artifact and all physical planes."""
    root = Path(path)
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != "plena.decode_cache":
        raise ValueError("not a PLENA decode-cache artifact")
    layers = []
    for index, layer_desc in enumerate(manifest["layers"]):
        layers.append(
            DecodeKVLayerPayload(
                key=_load_quantized_tensor(layer_desc["key"], root, index, "key"),
                value=_load_quantized_tensor(
                    layer_desc["value"], root, index, "value"
                ),
            )
        )
    artifact = DecodeCacheArtifact(
        source_artifact_id=manifest["source_artifact_id"],
        precision_id=manifest["precision_id"],
        layout_id=manifest["layout_id"],
        layers=tuple(layers),
        provenance=_load_provenance(manifest["provenance"]),
        metadata=manifest.get("metadata", {}),
        schema_version=manifest["schema_version"],
    )
    if artifact.artifact_id != manifest["artifact_id"]:
        raise ValueError("decode-cache artifact identity mismatch")
    return artifact
