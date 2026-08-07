"""Deterministic decode event counts for dense transformer decoder models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

EVENT_MODEL = "dense-decode-events"
SELECTOR_SIGNATURE = "SELECTOR:PACKED_KV"


@dataclass(frozen=True)
class DenseDecoderShape:
    """Dimensions needed to count one cached q_len=1 decode workload."""

    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab_size: int

    def __post_init__(self) -> None:
        values = (
            self.hidden_size,
            self.intermediate_size,
            self.attention_heads,
            self.kv_heads,
            self.head_dim,
            self.layers,
            self.vocab_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError("decoder dimensions must be positive")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if self.attention_heads * self.head_dim < self.hidden_size:
            raise ValueError("attention width cannot be narrower than hidden size")

    @classmethod
    def from_mapping(cls, value: Mapping[str, int]) -> "DenseDecoderShape":
        return cls(
            hidden_size=int(value["hidden"]),
            intermediate_size=int(value["inter"]),
            attention_heads=int(value["heads"]),
            kv_heads=int(value["kv_heads"]),
            head_dim=int(value["head_dim"]),
            layers=int(value["layers"]),
            vocab_size=int(value["vocab"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "layers": self.layers,
            "vocab_size": self.vocab_size,
        }


@dataclass(frozen=True)
class DecodeEvent:
    """One operation signature and its complete workload count."""

    signature: str
    count: int
    mlen: int
    blen: int

    def __post_init__(self) -> None:
        if not self.signature:
            raise ValueError("event signature must be non-empty")
        if self.count < 0:
            raise ValueError("event count must be non-negative")
        if self.mlen <= 0 or self.blen <= 0 or self.mlen % self.blen:
            raise ValueError("event geometry is invalid")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "signature": self.signature,
            "count": self.count,
            "MLEN": self.mlen,
            "BLEN": self.blen,
        }


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _projection_events(
    shape: DenseDecoderShape,
    *,
    batch: int,
    mlen: int,
    blen: int,
) -> int:
    row_tiles = _ceil_div(batch, blen)
    reduction_tiles = _ceil_div(shape.hidden_size, mlen)
    query_width = shape.attention_heads * shape.head_dim
    kv_width = shape.kv_heads * shape.head_dim
    return row_tiles * reduction_tiles * (
        _ceil_div(query_width, blen) + 2 * _ceil_div(kv_width, blen)
    )


def _linear_events(
    shape: DenseDecoderShape,
    *,
    batch: int,
    mlen: int,
    blen: int,
) -> tuple[int, int]:
    rows = _ceil_div(batch, blen)
    attention_width = shape.attention_heads * shape.head_dim
    projection = _projection_events(
        shape,
        batch=batch,
        mlen=mlen,
        blen=blen,
    )
    output = (
        rows
        * _ceil_div(attention_width, mlen)
        * _ceil_div(shape.hidden_size, blen)
    )
    ffn = rows * (
        2
        * _ceil_div(shape.hidden_size, mlen)
        * _ceil_div(shape.intermediate_size, blen)
        + _ceil_div(shape.intermediate_size, mlen)
        * _ceil_div(shape.hidden_size, blen)
    )
    decoder = shape.layers * (projection + output + ffn)
    lm_head = (
        rows
        * _ceil_div(shape.hidden_size, mlen)
        * _ceil_div(shape.vocab_size, blen)
    )
    return decoder, lm_head


def _attention_events(
    shape: DenseDecoderShape,
    *,
    context: int,
    batch: int,
    mlen: int,
    blen: int,
    hlen: int,
) -> tuple[int, int]:
    if hlen <= 0 or mlen % hlen:
        raise ValueError("MLEN must be divisible by HLEN")
    query_group = shape.attention_heads // shape.kv_heads
    context_tiles = _ceil_div(context, mlen)
    selected_groups = _ceil_div(query_group, mlen // hlen)
    common = context_tiles * shape.kv_heads * batch * shape.layers
    qk = common * selected_groups
    pv = common * query_group * _ceil_div(shape.head_dim, blen)
    return qk, pv


def _vector_events(
    shape: DenseDecoderShape,
    *,
    context: int,
    batch: int,
    mlen: int,
    include_vocab_selection: bool,
) -> int:
    hidden_chunks = _ceil_div(shape.hidden_size, mlen)
    head_chunks = _ceil_div(shape.head_dim, mlen)
    intermediate_chunks = _ceil_div(shape.intermediate_size, mlen)
    vocab_chunks = _ceil_div(shape.vocab_size, mlen)
    query_group = shape.attention_heads // shape.kv_heads
    context_tiles = _ceil_div(context, mlen)

    rmsnorm = (2 * shape.layers + 1) * batch * hidden_chunks * 8
    qk_norm = (
        shape.layers
        * batch
        * (shape.attention_heads + shape.kv_heads)
        * head_chunks
        * 8
    )
    rope = (
        shape.layers
        * batch
        * (shape.attention_heads + shape.kv_heads)
        * head_chunks
    )
    attention = (
        shape.layers
        * batch
        * shape.kv_heads
        * query_group
        * context_tiles
        * 6
    )
    silu_gate = (
        shape.layers * batch * intermediate_chunks * 6
    )
    residual = shape.layers * batch * hidden_chunks * 2
    vocab_softmax = (
        batch * vocab_chunks * 6
        if include_vocab_selection
        else 0
    )
    return (
        rmsnorm
        + qk_norm
        + rope
        + attention
        + silu_gate
        + residual
        + vocab_softmax
    )


def count_decode_events(
    shape: DenseDecoderShape,
    *,
    input_seq: int,
    output_seq: int,
    batch: int,
    mlen: int,
    blen: int,
    hlen: int,
    linear_signature: str,
    qk_signature: str,
    pv_signature: str,
    vector_signature: str,
    stride: int = 1,
    lm_head_signature: str = "UNMODELED:LM_HEAD_BF16",
    include_output_head: bool = True,
) -> tuple[DecodeEvent, ...]:
    """Count the same strided q_len=1 workload used by the timing model."""

    positive = (input_seq, output_seq, batch, mlen, blen, hlen, stride)
    if any(value <= 0 for value in positive):
        raise ValueError("workload and geometry values must be positive")
    if mlen % blen:
        raise ValueError("MLEN must be divisible by BLEN")
    signatures = [
        linear_signature,
        qk_signature,
        pv_signature,
        vector_signature,
    ]
    if include_output_head:
        signatures.append(lm_head_signature)
    if any(not signature for signature in signatures):
        raise ValueError("event signatures must be non-empty")

    decoder_linear, lm_head = _linear_events(
        shape,
        batch=batch,
        mlen=mlen,
        blen=blen,
    )
    counts = {
        linear_signature: 0,
        qk_signature: 0,
        pv_signature: 0,
        vector_signature: 0,
        SELECTOR_SIGNATURE: 0,
    }
    if include_output_head:
        counts[lm_head_signature] = 0
    position = 0
    while position < output_seq:
        span = min(stride, output_seq - position)
        context = input_seq + position
        qk, pv = _attention_events(
            shape,
            context=context,
            batch=batch,
            mlen=mlen,
            blen=blen,
            hlen=hlen,
        )
        vector = _vector_events(
            shape,
            context=context,
            batch=batch,
            mlen=mlen,
            include_vocab_selection=include_output_head,
        )
        counts[linear_signature] += decoder_linear * span
        if include_output_head:
            counts[lm_head_signature] += lm_head * span
        counts[qk_signature] += qk * span
        counts[pv_signature] += pv * span
        counts[vector_signature] += vector * span
        counts[SELECTOR_SIGNATURE] += (qk + pv) * span
        position += stride
    return tuple(
        DecodeEvent(signature, count, mlen, blen)
        for signature, count in sorted(counts.items())
        if count
    )


def event_count_total(events: Iterable[DecodeEvent]) -> int:
    """Return a stable aggregate used by tests and diagnostics."""

    return sum(event.count for event in events)


__all__ = [
    "DecodeEvent",
    "DenseDecoderShape",
    "EVENT_MODEL",
    "SELECTOR_SIGNATURE",
    "count_decode_events",
    "event_count_total",
]
