"""Core data types shared across tessera."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

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
    text: Optional[str] = None
    image: Optional[Any] = None  # path-like str or raw bytes
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
