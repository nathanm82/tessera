"""A retriever that keeps separate text and image indexes over one encoder."""

from __future__ import annotations

from collections.abc import Sequence

from tessera.encoders.base import Encoder
from tessera.retrieval.dense import DenseRetriever
from tessera.retrieval.fusion import weighted_fusion
from tessera.types import Chunk, Modality, RetrievalResult


class MultimodalRetriever:
    """Routes chunks to a per-modality index and queries across all of them.

    Text and image chunks live in their own dense indexes, but because they share
    the encoder's embedding space a single query can reach both. Results are blended
    so that a caller sees one ranked list spanning modalities.
    """

    def __init__(
        self, encoder: Encoder, text_weight: float = 0.5, image_weight: float = 0.5
    ) -> None:
        self.encoder = encoder
        self.text_weight = text_weight
        self.image_weight = image_weight
        self._text = (
            DenseRetriever(encoder, Modality.TEXT) if encoder.supports(Modality.TEXT) else None
        )
        self._image = (
            DenseRetriever(encoder, Modality.IMAGE) if encoder.supports(Modality.IMAGE) else None
        )

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Partition ``chunks`` by modality and index each partition."""
        text_chunks = [c for c in chunks if c.modality is Modality.TEXT]
        image_chunks = [c for c in chunks if c.modality is Modality.IMAGE]
        if text_chunks and self._text is not None:
            self._text.index(text_chunks)
        if image_chunks and self._image is not None:
            self._image.index(image_chunks)

    def retrieve(
        self,
        query: object,
        query_modality: Modality = Modality.TEXT,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Query every populated index and fuse the hits into one ranked list."""
        result_lists: list[list[RetrievalResult]] = []
        weights: list[float] = []
        if self._text is not None and len(self._text.store) > 0:
            result_lists.append(self._text.retrieve(query, query_modality, top_k))
            weights.append(self.text_weight)
        if self._image is not None and len(self._image.store) > 0:
            result_lists.append(self._image.retrieve(query, query_modality, top_k))
            weights.append(self.image_weight)

        if not result_lists:
            return []
        fused = weighted_fusion(result_lists, weights)
        return fused[:top_k]
