"""A deterministic, dependency-free encoder for tests and quick baselines.

It hashes tokens (for text) and byte windows (for images) into a fixed number of
buckets, producing stable embeddings without downloading any model weights. The
vectors are not semantically meaningful the way CLIP's are -- they exist so the
whole pipeline can run, and be tested, entirely offline. Swap in
:class:`tessera.encoders.clip.ClipEncoder` for real cross-modal semantics.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tessera.encoders.base import Encoder
from tessera.similarity import l2_normalize
from tessera.types import Modality

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bucket(token: bytes, n_features: int) -> tuple[int, float]:
    """Hash ``token`` to a (bucket index, sign) pair using a stable digest."""
    digest = hashlib.md5(token).digest()
    index = int.from_bytes(digest[:4], "big") % n_features
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


class HashingEncoder(Encoder):
    """Feature-hashing encoder over a fixed number of buckets."""

    supported_modalities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self, dim: int = 256, image_window: int = 8) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if image_window <= 0:
            raise ValueError("image_window must be positive")
        self._dim = dim
        self._image_window = image_window

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                index, sign = _bucket(token.encode("utf-8"), self._dim)
                out[row, index] += sign
        return l2_normalize(out)

    def encode_image(self, images: Sequence[object]) -> NDArray[np.float32]:
        out = np.zeros((len(images), self._dim), dtype=np.float32)
        window = self._image_window
        for row, image in enumerate(images):
            data = self._read(image)
            stop = max(len(data) - window, 1)
            for start in range(0, stop, window):
                index, sign = _bucket(data[start : start + window], self._dim)
                out[row, index] += sign
        return l2_normalize(out)

    @staticmethod
    def _read(image: object) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, str):
            with open(image, "rb") as handle:
                return handle.read()
        raise TypeError(f"cannot read image of type {type(image).__name__}")
