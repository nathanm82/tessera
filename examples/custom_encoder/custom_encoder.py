"""Implement a custom encoder and plug it into the registry.

Any object that satisfies the ``Encoder`` interface can back retrieval. Here we add a
toy text encoder and register it so it is selectable by name, exactly like the
built-in ``hashing`` encoder.

Run with::

    python examples/custom_encoder/custom_encoder.py
"""

import numpy as np

from tessera.encoders import ENCODERS, Encoder
from tessera.types import Modality


class CharClassEncoder(Encoder):
    """Embeds text by counts of a few character classes -- intentionally simple."""

    supported_modalities = frozenset({Modality.TEXT})

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, 0] = sum(c.isalpha() for c in text)
            out[row, 1] = sum(c.isdigit() for c in text)
            out[row, 2] = sum(c.isspace() for c in text)
            out[row, 3] = len(text)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


def main() -> None:
    ENCODERS.register("charclass", CharClassEncoder)
    print("registered encoders:", ENCODERS.names())

    encoder = ENCODERS.create("charclass", dim=4)
    vectors = encoder.encode_text(["hello", "abc 123"])
    print("embedding shape:", vectors.shape)


if __name__ == "__main__":
    main()
