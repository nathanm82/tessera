"""Reading corpora from disk."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tessera.types import Document

_RESERVED = {"id", "text", "image"}


def _to_document(record: dict[str, Any]) -> Document:
    if "id" not in record:
        raise ValueError("each record needs an 'id' field")
    metadata = {key: value for key, value in record.items() if key not in _RESERVED}
    return Document(
        id=str(record["id"]),
        text=record.get("text"),
        image=record.get("image"),
        metadata=metadata,
    )


def load_jsonl(path: str | Path) -> Iterator[Document]:
    """Yield :class:`~tessera.types.Document` objects from a JSON Lines file.

    Each line is a JSON object with an ``id`` and at least one of ``text`` or
    ``image``; remaining keys become document metadata. Blank lines are ignored.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            yield _to_document(record)
