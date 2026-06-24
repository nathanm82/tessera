"""A corpus is the collection of documents you index."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from tessera.chunking import chunk_text
from tessera.types import Chunk, Document, Modality


class Corpus:
    """An ordered collection of :class:`~tessera.types.Document` objects."""

    def __init__(self, documents: Sequence[Document] | None = None) -> None:
        self._documents: list[Document] = list(documents) if documents else []

    def add(self, document: Document) -> None:
        self._documents.append(document)

    def extend(self, documents: Iterable[Document]) -> None:
        self._documents.extend(documents)

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)

    def to_chunks(self, chunk_size: int = 256, overlap: int = 32) -> list[Chunk]:
        """Expand documents into indexable chunks.

        Text is split into overlapping windows; an image contributes a single chunk
        whose ``content`` is its path. Image documents carried as raw bytes are
        skipped here -- index those directly through a retriever instead.
        """
        chunks: list[Chunk] = []
        for document in self._documents:
            if document.text:
                pieces = chunk_text(document.text, chunk_size, overlap)
                for index, piece in enumerate(pieces):
                    chunks.append(
                        Chunk(
                            id=f"{document.id}::text::{index}",
                            doc_id=document.id,
                            modality=Modality.TEXT,
                            content=piece,
                            metadata=dict(document.metadata),
                        )
                    )
            if isinstance(document.image, str):
                chunks.append(
                    Chunk(
                        id=f"{document.id}::image",
                        doc_id=document.id,
                        modality=Modality.IMAGE,
                        content=document.image,
                        metadata=dict(document.metadata),
                    )
                )
        return chunks
