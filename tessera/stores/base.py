"""Vector store interface.

A store keeps embedding vectors alongside a JSON-serializable payload (typically a
serialized chunk) and ranks stored items against a query vector. Keeping the payload
serializable is what lets an index round-trip to disk.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchHit:
    """One result from a vector store search."""

    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(abc.ABC):
    """Stores embedding vectors with attached payloads and ranks them by similarity."""

    @property
    @abc.abstractmethod
    def dim(self) -> int | None:
        """Embedding dimensionality, or ``None`` if nothing has been added yet."""

    @abc.abstractmethod
    def add(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """Add vectors and their payloads to the store."""

    @abc.abstractmethod
    def search(self, query: NDArray[np.float32], top_k: int = 5) -> list[SearchHit]:
        """Return the ``top_k`` most similar items to ``query``, best first."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of stored vectors."""
