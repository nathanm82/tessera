"""A corpus is the collection of documents you index."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from tessera.types import Document


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
