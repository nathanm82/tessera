from tessera.encoders.hashing import HashingEncoder
from tessera.retrieval import DenseRetriever
from tessera.types import Chunk, Modality


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=cid, modality=Modality.TEXT, content=text)


def test_retrieve_returns_ranked_results() -> None:
    retriever = DenseRetriever(HashingEncoder(dim=512), Modality.TEXT)
    retriever.index(
        [
            _chunk("1", "a fluffy cat sitting on a warm windowsill"),
            _chunk("2", "stock market closes higher after earnings"),
            _chunk("3", "kitten playing with a ball of yarn"),
        ]
    )
    results = retriever.retrieve("cat on a windowsill", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].chunk.id in {"1", "3"}


def test_indexing_empty_is_noop() -> None:
    retriever = DenseRetriever(HashingEncoder(dim=64), Modality.TEXT)
    retriever.index([])
    assert len(retriever.store) == 0


def test_unsupported_modality_rejected() -> None:
    class TextOnly(HashingEncoder):
        supported_modalities = frozenset({Modality.TEXT})

    try:
        DenseRetriever(TextOnly(dim=8), Modality.IMAGE)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported modality")
