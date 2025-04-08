"""Retrieval turns a query into ranked, source-attributed chunks."""

from __future__ import annotations

from tessera.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from tessera.retrieval.multimodal import MultimodalRetriever
from tessera.retrieval.retriever import DenseRetriever

__all__ = [
    "DenseRetriever",
    "MultimodalRetriever",
    "reciprocal_rank_fusion",
    "weighted_fusion",
]
