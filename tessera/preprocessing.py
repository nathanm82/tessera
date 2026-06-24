"""Small helpers for turning raw inputs into documents."""

from __future__ import annotations

from pathlib import Path

from tessera.types import Document


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return " ".join(text.split())


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


def read_image_bytes(path: str | Path) -> bytes:
    """Read an image file as raw bytes."""
    return Path(path).read_bytes()


def text_document(doc_id: str, text: str, **metadata: object) -> Document:
    """Build a text :class:`~tessera.types.Document` with normalized text."""
    return Document(id=doc_id, text=normalize_whitespace(text), metadata=dict(metadata))


def image_document(doc_id: str, image_path: str, **metadata: object) -> Document:
    """Build an image-backed :class:`~tessera.types.Document` from a path."""
    return Document(id=doc_id, image=image_path, metadata=dict(metadata))
