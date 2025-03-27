from pathlib import Path

import numpy as np
import pytest

from tessera.exceptions import DimensionMismatchError
from tessera.stores.memory import InMemoryVectorStore


def _vec(*values: float) -> np.ndarray:
    return np.array([values], dtype=np.float32)


def test_add_grows_length_and_sets_dim() -> None:
    store = InMemoryVectorStore()
    assert store.dim is None
    store.add(["a"], _vec(1.0, 0.0), [{"text": "a"}])
    store.add(["b"], _vec(0.0, 1.0), [{"text": "b"}])
    assert len(store) == 2
    assert store.dim == 2


def test_search_ranks_by_cosine() -> None:
    store = InMemoryVectorStore()
    store.add(
        ["a", "b", "c"],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
        [{"t": "a"}, {"t": "b"}, {"t": "c"}],
    )
    hits = store.search(_vec(1.0, 0.0), top_k=2)
    assert [h.id for h in hits] == ["a", "c"]
    assert hits[0].score >= hits[1].score


def test_mismatched_dim_raises() -> None:
    store = InMemoryVectorStore()
    store.add(["a"], _vec(1.0, 0.0), [{}])
    with pytest.raises(DimensionMismatchError):
        store.add(["b"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32), [{}])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    store.add(
        ["a", "b"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [{"text": "alpha"}, {"text": "beta"}],
    )
    store.save(tmp_path / "index")
    loaded = InMemoryVectorStore.load(tmp_path / "index")
    assert len(loaded) == 2
    assert loaded.dim == 2
    hits = loaded.search(_vec(0.0, 1.0), top_k=1)
    assert hits[0].payload == {"text": "beta"}
