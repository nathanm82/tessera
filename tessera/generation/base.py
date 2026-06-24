"""Generator interface and the grounded-answer data type."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass, field

from tessera.types import RetrievalResult


@dataclass
class GeneratedAnswer:
    """An answer plus the retrieved chunks that grounded it."""

    text: str
    sources: list[RetrievalResult] = field(default_factory=list)


class Generator(abc.ABC):
    """Produces an answer for a query given retrieved context."""

    @abc.abstractmethod
    def generate(self, query: str, context: Sequence[RetrievalResult]) -> GeneratedAnswer:
        """Generate an answer grounded in ``context``."""
