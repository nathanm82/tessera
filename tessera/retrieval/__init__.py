"""Retrieval turns a query into ranked, source-attributed chunks."""

from __future__ import annotations

from tessera.retrieval.dense import DenseRetriever
from tessera.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from tessera.retrieval.multimodal import MultimodalRetriever

__all__ = [
    "DenseRetriever",
    "MultimodalRetriever",
    "reciprocal_rank_fusion",
    "weighted_fusion",
]
