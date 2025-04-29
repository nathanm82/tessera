from tessera.corpus import Corpus
from tessera.types import Document, Modality


def test_add_and_len() -> None:
    corpus = Corpus()
    corpus.add(Document(id="a", text="hello"))
    corpus.extend([Document(id="b", text="world"), Document(id="c", text="!")])
    assert len(corpus) == 3
    assert [d.id for d in corpus] == ["a", "b", "c"]


def test_to_chunks_splits_text() -> None:
    text = " ".join(str(i) for i in range(20))
    corpus = Corpus([Document(id="doc", text=text)])
    chunks = corpus.to_chunks(chunk_size=8, overlap=0)
    assert len(chunks) > 1
    assert all(c.modality is Modality.TEXT for c in chunks)
    assert all(c.doc_id == "doc" for c in chunks)


def test_to_chunks_includes_image_path() -> None:
    corpus = Corpus([Document(id="doc", text="a caption", image="/tmp/x.png")])
    chunks = corpus.to_chunks(chunk_size=64)
    modalities = {c.modality for c in chunks}
    assert Modality.IMAGE in modalities
    image_chunk = next(c for c in chunks if c.modality is Modality.IMAGE)
    assert image_chunk.content == "/tmp/x.png"
