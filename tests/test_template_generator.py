from tessera.generation.template import TemplateGenerator
from tessera.types import Chunk, Modality, RetrievalResult


def _result(cid: str, content: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(id=cid, doc_id=cid, modality=Modality.TEXT, content=content),
        score=score,
    )


def test_generate_grounds_in_context() -> None:
    gen = TemplateGenerator(max_sources=2)
    context = [
        _result("1", "the eiffel tower is in paris", 0.9),
        _result("2", "paris is the capital of france", 0.7),
        _result("3", "ignored because of max_sources", 0.5),
    ]
    answer = gen.generate("where is the eiffel tower", context)
    assert len(answer.sources) == 2
    assert "eiffel tower" in answer.text
    assert "ignored" not in answer.text


def test_generate_with_no_context() -> None:
    gen = TemplateGenerator()
    answer = gen.generate("anything", [])
    assert answer.sources == []
    assert "No context" in answer.text


def test_prompt_contains_query_and_sources() -> None:
    gen = TemplateGenerator()
    prompt = gen.prompt("what is rag", [_result("1", "retrieval augmented generation", 0.9)])
    assert "Question: what is rag" in prompt
    assert "retrieval augmented generation" in prompt
