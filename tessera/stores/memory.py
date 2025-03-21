"""An in-memory vector store backed by a single numpy matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tessera.exceptions import DimensionMismatchError, StoreError
from tessera.similarity import cosine_similarity, top_k
from tessera.stores.base import SearchHit, VectorStore


class InMemoryVectorStore(VectorStore):
    """Brute-force store: ideal up to tens of thousands of vectors, zero extra deps.

    Vectors are kept in one contiguous matrix so search is a single matmul.
    """

    def __init__(self) -> None:
        self._vectors: NDArray[np.float32] | None = None
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []

    @property
    def dim(self) -> int | None:
        return None if self._vectors is None else int(self._vectors.shape[1])

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

        if self._vectors is None:
            self._vectors = matrix.copy()
        elif matrix.shape[1] != self._vectors.shape[1]:
            raise DimensionMismatchError(
                f"expected dim {self._vectors.shape[1]}, got {matrix.shape[1]}"
            )
        else:
            self._vectors = np.vstack([self._vectors, matrix])

        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query: NDArray[np.float32], top_k: int = 5) -> list[SearchHit]:
        if self._vectors is None:
            raise StoreError("cannot search an empty store")
        sims = cosine_similarity(query, self._vectors).ravel()
        indices, scores = top_k(sims, top_k)
        return [
            SearchHit(id=self._ids[i], score=float(score), payload=self._payloads[i])
            for i, score in zip(indices, scores)
        ]

    def __len__(self) -> int:
        return len(self._ids)
