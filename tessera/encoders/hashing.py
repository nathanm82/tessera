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

    supported_modalities = frozenset({Modality.TEXT})

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                index, sign = _bucket(token.encode("utf-8"), self._dim)
                out[row, index] += sign
        return out
