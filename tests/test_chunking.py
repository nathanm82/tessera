import pytest

from tessera.chunking import chunk_text


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("hello world", chunk_size=8)
    assert chunks == ["hello world"]


def test_long_text_splits_into_multiple_chunks() -> None:
    words = " ".join(str(i) for i in range(20))
    chunks = chunk_text(words, chunk_size=8, overlap=0)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_all_words_are_covered() -> None:
    words = [str(i) for i in range(15)]
    chunks = chunk_text(" ".join(words), chunk_size=6, overlap=2)
    covered: set[str] = set()
    for chunk in chunks:
        covered.update(chunk.split())
    assert covered == set(words)
    assert chunks[0].split()[0] == "0"
    assert chunks[-1].split()[-1] == "14"


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=4, overlap=4)
