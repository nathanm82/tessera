"""Encoders map text and images into a shared embedding space."""

from __future__ import annotations

from tessera.encoders.base import Encoder
from tessera.encoders.hashing import HashingEncoder
from tessera.registry import Registry

#: Registry of encoder factories, keyed by name.
ENCODERS: Registry[Encoder] = Registry("encoder")
ENCODERS.register("hashing", HashingEncoder)


def get_encoder(name: str, **kwargs: object) -> Encoder:
    """Create a registered encoder by name (e.g. ``"hashing"``)."""
    return ENCODERS.create(name, **kwargs)


__all__ = ["ENCODERS", "Encoder", "HashingEncoder", "get_encoder"]
