"""Strategies for combining ranked results from multiple retrievers."""

from __future__ import annotations

from collections.abc import Sequence

from tessera.types import RetrievalResult


def _rank(combined: dict[str, float], seen: dict[str, RetrievalResult]) -> list[RetrievalResult]:
    """Materialize fused scores into results sorted best-first."""
    fused = [RetrievalResult(chunk=seen[cid].chunk, score=score) for cid, score in combined.items()]
    fused.sort(key=lambda result: result.score, reverse=True)
    return fused


def weighted_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    weights: Sequence[float] | None = None,
) -> list[RetrievalResult]:
    """Combine result lists by a weighted sum of per-list scores.

    Results are matched by chunk id; a chunk absent from a list contributes nothing
    from that list. The returned list is sorted by combined score, descending.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    combined: dict[str, float] = {}
    seen: dict[str, RetrievalResult] = {}
    for results, weight in zip(result_lists, weights):
        for result in results:
            cid = result.chunk.id
            combined[cid] = combined.get(cid, 0.0) + weight * result.score
            seen[cid] = result

    return _rank(combined, seen)


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Rank-based fusion that ignores raw score magnitudes (RRF).

    Each list contributes ``1 / (k + rank)`` for the items it ranks, which makes the
    method robust when retrievers report scores on different scales -- exactly the
    case when fusing a dense text index with an image index.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    combined: dict[str, float] = {}
    seen: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results):
            cid = result.chunk.id
            combined[cid] = combined.get(cid, 0.0) + 1.0 / (k + rank)
            seen[cid] = result

    return _rank(combined, seen)
