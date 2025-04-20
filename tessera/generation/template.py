"""A deterministic generator that needs no model -- ideal for tests and demos."""

from __future__ import annotations

from collections.abc import Sequence

from tessera.generation.base import GeneratedAnswer, Generator, build_prompt
from tessera.types import RetrievalResult


class TemplateGenerator(Generator):
    """Stitches retrieved context into a templated, citation-bearing answer.

    It performs no real language modeling -- it reports what was retrieved in a
    stable form. Use it as the default generator, in tests, or as a fallback when no
    LLM is configured. :meth:`prompt` exposes the exact text a real model would get,
    so swapping in an API-backed generator is a drop-in change.
    """

    def __init__(self, max_sources: int = 3) -> None:
        if max_sources <= 0:
            raise ValueError("max_sources must be positive")
        self.max_sources = max_sources

    def prompt(self, query: str, context: Sequence[RetrievalResult]) -> str:
        """Return the grounded prompt that an LLM-backed generator would send."""
        return build_prompt(query, context)

    def generate(self, query: str, context: Sequence[RetrievalResult]) -> GeneratedAnswer:
        used = list(context)[: self.max_sources]
        if not used:
            return GeneratedAnswer(text=f"No context was retrieved for: {query}", sources=[])
        citations = "; ".join(
            f"[{i}] {' '.join(result.chunk.content.split())}"
            for i, result in enumerate(used, start=1)
        )
        text = (
            f"Based on {len(used)} retrieved source(s), the most relevant context "
            f"for '{query}' is: {citations}"
        )
        return GeneratedAnswer(text=text, sources=used)
