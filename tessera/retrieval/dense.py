"""Dense retriever: encode chunks once, then rank them by vector similarity."""

from __future__ import annotations

from collections.abc import Sequence

from tessera.encoders.base import Encoder
from tessera.stores.base import VectorStore
from tessera.stores.memory import InMemoryVectorStore
from tessera.types import Chunk, Modality, RetrievalResult


class DenseRetriever:
    """Indexes chunks of one modality and retrieves them by similarity.

    Because an :class:`~tessera.encoders.base.Encoder` maps every modality into the
    same space, the query passed to :meth:`retrieve` need not share the indexed
    modality -- a text query can rank image chunks, which is the whole point of a
    multimodal index.
    """

    def __init__(
        self,
        encoder: Encoder,
        modality: Modality = Modality.TEXT,
        store: VectorStore | None = None,
    ) -> None:
        if not encoder.supports(modality):
            raise ValueError(f"encoder does not support {modality.value}")
        self.encoder = encoder
        self.modality = modality
        self.store: VectorStore = store if store is not None else InMemoryVectorStore()

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed and store ``chunks`` (all assumed to be of ``self.modality``)."""
        if not chunks:
            return
        vectors = self.encoder.encode([chunk.content for chunk in chunks], self.modality)
        self.store.add(
            [chunk.id for chunk in chunks],
            vectors,
            [chunk.to_dict() for chunk in chunks],
        )

    def retrieve(
        self,
        query: object,
        query_modality: Modality = Modality.TEXT,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        query_vec = self.encoder.encode([query], query_modality)
        hits = self.store.search(query_vec, top_k=top_k)
        return [
            RetrievalResult(chunk=Chunk.from_dict(hit.payload), score=hit.score) for hit in hits
        ]
