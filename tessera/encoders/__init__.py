"""Encoders map text and images into a shared embedding space."""

from __future__ import annotations

from typing import Any

from tessera.encoders.base import Encoder
from tessera.encoders.hashing import HashingEncoder
from tessera.registry import Registry

#: Registry of encoder factories, keyed by name.
ENCODERS: Registry[Encoder] = Registry("encoder")
ENCODERS.register("hashing", HashingEncoder)


def _make_clip(**kwargs: object) -> Encoder:
    # Imported lazily so the optional torch/open_clip stack is never required
    # merely to enumerate the available encoders.
    from tessera.encoders.clip import ClipEncoder

    factory: Any = ClipEncoder
    return factory(**kwargs)


ENCODERS.register("clip", _make_clip)


def get_encoder(name: str, **kwargs: object) -> Encoder:
    """Create a registered encoder by name (e.g. ``"hashing"`` or ``"clip"``)."""
    return ENCODERS.create(name, **kwargs)


__all__ = ["ENCODERS", "Encoder", "HashingEncoder", "get_encoder"]
