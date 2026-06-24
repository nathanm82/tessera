"""Core data types shared across tessera."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# A row-major float32 matrix of embeddings, shape (n_items, dim).
Embedding = NDArray[np.float32]


class Modality(str, enum.Enum):
    """The kind of content a document or chunk carries."""

    TEXT = "text"
    IMAGE = "image"


@dataclass
class Document:
    """A single item in a corpus.

    A document may carry text, an image reference, or both. ``image`` is either a
    filesystem path or raw bytes; tessera never decodes it eagerly, which keeps the
    core dependency-light and lets image handling stay an optional concern.
    """

    id: str
    text: str | None = None
    image: Any | None = None  # path-like str or raw bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    def modalities(self) -> set[Modality]:
        """Return the set of modalities actually present on this document."""
        present: set[Modality] = set()
        if self.text:
            present.add(Modality.TEXT)
        if self.image is not None:
            present.add(Modality.IMAGE)
        return present


@dataclass
class Chunk:
    """A retrievable unit derived from a :class:`Document`.

    ``content`` holds the text of a text chunk, or a stable reference string for an
    image chunk (a path or synthetic id). The originating document id is kept so
    retrieval results can be traced back to their source.
    """

    id: str
    doc_id: str
    modality: Modality
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict (used as a store payload)."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "modality": self.modality.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=data["id"],
            doc_id=data["doc_id"],
            modality=Modality(data["modality"]),
            content=data["content"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievalResult:
    """A retrieved chunk paired with its similarity score."""

    chunk: Chunk
    score: float
