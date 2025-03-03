"""Encoder interface and the shared embedding contract."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tessera.types import Modality


class Encoder(abc.ABC):
    """Maps text and/or images into a shared embedding space.

    Concrete encoders must return L2-normalized ``float32`` matrices so that a dot
    product equals cosine similarity. Crucially, every embedding from a single
    encoder shares one ``dim`` regardless of modality -- that shared space is what
    lets a text query retrieve image chunks, and vice versa.
    """

    #: Modalities this encoder can embed.
    supported_modalities: frozenset[Modality] = frozenset()

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding space."""

    def supports(self, modality: Modality) -> bool:
        """Whether this encoder can embed ``modality``."""
        return modality in self.supported_modalities

    def encode_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed a batch of text strings."""
        raise NotImplementedError(f"{type(self).__name__} does not support text")

    def encode_image(self, images: Sequence[object]) -> NDArray[np.float32]:
        """Embed a batch of images given as paths or raw bytes."""
        raise NotImplementedError(f"{type(self).__name__} does not support images")
