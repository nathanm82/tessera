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


def build_prompt(
    query: str,
    context: Sequence[RetrievalResult],
    *,
    max_chars: int = 500,
) -> str:
    """Assemble a grounded prompt: numbered context blocks, then the question.

    This is the exact text you would hand to an LLM. A generator backed by a real
    model should send this verbatim; an offline generator can summarize it instead.
    """
    lines = ["Use the following sources to answer the question.", ""]
    for index, result in enumerate(context, start=1):
        snippet = " ".join(result.chunk.content.split())
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "…"
        lines.append(f"[{index}] ({result.chunk.modality.value}) {snippet}")
    lines.extend(["", f"Question: {query}", "Answer:"])
    return "\n".join(lines)
