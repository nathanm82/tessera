from pathlib import Path

from tessera.encoders.hashing import HashingEncoder
from tessera.retrieval import MultimodalRetriever
from tessera.types import Chunk, Modality


def _image_file(tmp_path: Path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_text_only_index_returns_text_hits() -> None:
    retriever = MultimodalRetriever(HashingEncoder(dim=256))
    retriever.index(
        [
            Chunk("t1", "d1", Modality.TEXT, "a photo of a red bus on a street"),
            Chunk("t2", "d2", Modality.TEXT, "a recipe for lemon cake"),
        ]
    )
    hits = retriever.retrieve("red bus", top_k=1)
    assert hits and hits[0].chunk.id == "t1"


def test_fuses_text_and_image_indexes(tmp_path: Path) -> None:
    retriever = MultimodalRetriever(HashingEncoder(dim=256))
    img = _image_file(tmp_path, "a.bin", bytes(range(32)))
    retriever.index(
        [
            Chunk("t1", "d1", Modality.TEXT, "a caption about mountains"),
            Chunk("i1", "d2", Modality.IMAGE, img),
        ]
    )
    hits = retriever.retrieve("mountains", query_modality=Modality.TEXT, top_k=5)
    ids = {h.chunk.id for h in hits}
    assert "t1" in ids
    assert len(hits) >= 1


def test_empty_retriever_returns_empty() -> None:
    retriever = MultimodalRetriever(HashingEncoder(dim=64))
    assert retriever.retrieve("anything") == []
