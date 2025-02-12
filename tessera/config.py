"""Configuration objects for assembling a pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Knobs for a :class:`tessera.pipeline.RagPipeline`.

    The defaults describe a small, fully offline pipeline: the deterministic
    hashing encoder, modest chunking, and an even blend of the text and image
    retrieval scores.
    """

    encoder: str = "hashing"
    chunk_size: int = 256
    chunk_overlap: int = 32
    top_k: int = 5
    text_weight: float = 0.5
    image_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
