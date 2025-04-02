"""Strategies for combining ranked results from multiple retrievers."""

from __future__ import annotations

from collections.abc import Sequence

from tessera.types import RetrievalResult


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

    fused = [RetrievalResult(chunk=seen[cid].chunk, score=score) for cid, score in combined.items()]
    fused.sort(key=lambda result: result.score, reverse=True)
    return fused
