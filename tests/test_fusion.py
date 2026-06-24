import pytest

from tessera.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from tessera.types import Chunk, Modality, RetrievalResult


def _result(cid: str, score: float) -> RetrievalResult:
    chunk = Chunk(id=cid, doc_id=cid, modality=Modality.TEXT, content=cid)
    return RetrievalResult(chunk=chunk, score=score)


def test_weighted_fusion_sums_matching_ids() -> None:
    text = [_result("a", 0.9), _result("b", 0.2)]
    image = [_result("a", 0.4), _result("c", 0.7)]
    fused = weighted_fusion([text, image], weights=[1.0, 0.5])
    by_id = {r.chunk.id: r.score for r in fused}
    assert by_id["a"] == pytest.approx(0.9 + 0.5 * 0.4)
    assert fused[0].chunk.id == "a"


def test_weighted_fusion_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        weighted_fusion([[_result("a", 1.0)]], weights=[1.0, 2.0])


def test_rrf_prefers_consistently_top_ranked() -> None:
    list_one = [_result("a", 0.1), _result("b", 0.05)]
    list_two = [_result("a", 100.0), _result("c", 50.0)]
    fused = reciprocal_rank_fusion([list_one, list_two], k=60)
    assert fused[0].chunk.id == "a"


def test_rrf_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_result("a", 1.0)]], k=0)
