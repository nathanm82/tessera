import numpy as np

from tessera.similarity import cosine_similarity, l2_normalize, top_k


def test_l2_normalize_gives_unit_rows() -> None:
    m = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(m)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_cosine_similarity_identical_vectors_is_one() -> None:
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    sim = cosine_similarity(a, a)
    assert sim.shape == (1, 1)
    assert np.isclose(sim[0, 0], 1.0, atol=1e-5)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    assert np.isclose(cosine_similarity(a, b)[0, 0], 0.0, atol=1e-6)


def test_top_k_returns_sorted_indices() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    idx, vals = top_k(scores, 2)
    assert list(idx) == [1, 3]
    assert np.allclose(vals, [0.9, 0.7])


def test_top_k_clamps_to_length() -> None:
    scores = np.array([0.2, 0.4], dtype=np.float32)
    idx, vals = top_k(scores, 10)
    assert len(idx) == 2
