"""Deterministic statistical helpers used by hardware result gates."""

from __future__ import annotations

import math
from typing import Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman_rank_correlation(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return tie-aware Spearman correlation for aligned finite observations."""

    if len(first) != len(second):
        raise ValueError("rank vectors must have equal length")
    if len(first) < 2:
        raise ValueError("at least two observations are required")
    left = [float(value) for value in first]
    right = [float(value) for value in second]
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("rank observations must be finite")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_rank, right_rank)
    )
    left_norm = sum((value - left_mean) ** 2 for value in left_rank)
    right_norm = sum((value - right_mean) ** 2 for value in right_rank)
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / math.sqrt(left_norm * right_norm)


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile from finite values."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["percentile", "spearman_rank_correlation"]
