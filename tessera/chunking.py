"""Whitespace-aware text chunking.

Chunk sizes are measured in whitespace tokens rather than characters. That keeps
the splitter dependency-free while staying close to how downstream encoders count
tokens, so a ``chunk_size`` here roughly tracks the encoder's context budget.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """Split ``text`` into windows of at most ``chunk_size`` tokens.

    ``overlap`` tokens are shared between consecutive windows so that context
    straddling a boundary is not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            # The window already reached the end; another step would only repeat it.
            break
    return chunks
