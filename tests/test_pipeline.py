from pathlib import Path

from tessera.config import PipelineConfig
from tessera.pipeline import RagPipeline
from tessera.types import Document, Modality


def _docs() -> list[Document]:
    return [
        Document(id="d1", text="the great barrier reef is a coral reef system off australia"),
        Document(id="d2", text="python is a high level programming language"),
        Document(id="d3", text="coral reefs host a quarter of all marine species"),
    ]


def test_pipeline_answers_with_sources() -> None:
    pipe = RagPipeline(PipelineConfig(chunk_size=64, top_k=2))
    pipe.add_documents(_docs())
    answer = pipe.answer("tell me about coral reefs")
    assert answer.sources
    assert len(answer.sources) <= 2
    top_ids = {result.chunk.doc_id for result in answer.sources}
    assert top_ids & {"d1", "d3"}


def test_retrieve_lazily_indexes() -> None:
    pipe = RagPipeline(PipelineConfig(chunk_size=64, top_k=1))
    pipe.add_documents(_docs())
    results = pipe.retrieve("programming language", query_modality=Modality.TEXT)
    assert results
    assert results[0].chunk.doc_id == "d2"


def test_adding_documents_invalidates_index() -> None:
    pipe = RagPipeline(PipelineConfig(chunk_size=64, top_k=3))
    pipe.add_documents([_docs()[0]])
    pipe.index()
    pipe.add_documents([_docs()[1]])
    # A fresh retrieve should see the newly added document after re-indexing.
    results = pipe.retrieve("programming language")
    assert any(r.chunk.doc_id == "d2" for r in results)


def test_from_jsonl_loads_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id": "d1", "text": "coral reefs host marine species"}\n'
        '{"id": "d2", "text": "python is a programming language"}\n',
        encoding="utf-8",
    )
    pipe = RagPipeline.from_jsonl(corpus, PipelineConfig(top_k=1))
    results = pipe.retrieve("marine species")
    assert results[0].chunk.doc_id == "d1"
