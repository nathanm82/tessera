"""An optional FAISS-backed vector store for larger corpora.

Requires the ``faiss`` extra::

    pip install "tessera-rag[faiss]"

The numpy :class:`~tessera.stores.memory.InMemoryVectorStore` is the default and
needs no extra dependencies; reach for this only when brute-force search becomes the
bottleneck. Vectors are expected to be L2-normalized, so an inner-product index
ranks by cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tessera.exceptions import DimensionMismatchError, StoreError
from tessera.stores.base import SearchHit, VectorStore


class FaissVectorStore(VectorStore):
    """Inner-product FAISS index with parallel id and payload lists."""

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise StoreError(
                "FaissVectorStore requires the 'faiss' extra: pip install 'tessera-rag[faiss]'"
            ) from exc
        self._faiss = faiss
        self._index: Any = None
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []

    @property
    def dim(self) -> int | None:
        return None if self._index is None else int(self._index.d)

    def add(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if not len(ids) == matrix.shape[0] == len(payloads):
            raise ValueError("ids, vectors, and payloads must have equal length")
        if self._index is None:
            self._index = self._faiss.IndexFlatIP(matrix.shape[1])
        elif matrix.shape[1] != self._index.d:
            raise DimensionMismatchError(f"expected dim {self._index.d}, got {matrix.shape[1]}")
        self._index.add(matrix)
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query: NDArray[np.float32], top_k: int = 5) -> list[SearchHit]:
        if self._index is None:
            return []
        flat = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(flat, min(top_k, len(self._ids)))
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(
                SearchHit(id=self._ids[idx], score=float(score), payload=self._payloads[idx])
            )
        return hits

    def __len__(self) -> int:
        return len(self._ids)
