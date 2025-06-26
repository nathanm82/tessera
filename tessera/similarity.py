"""Vector similarity helpers built on numpy.

These functions assume float32 row-major matrices and treat each row as one
embedding. Keeping them in one place means encoders, stores, and retrievers all
share the same normalization conventions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def l2_normalize(matrix: NDArray[np.float32], axis: int = -1) -> NDArray[np.float32]:
    """Return ``matrix`` with vectors scaled to unit L2 norm along ``axis``.

    Zero vectors are left as zeros rather than producing ``nan`` -- an empty text
    chunk or a degenerate embedding should not poison a whole similarity matrix.
    """
    norms = np.linalg.norm(matrix, axis=axis, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def cosine_similarity(
    queries: NDArray[np.float32], corpus: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Cosine similarity between every query row and every corpus row.

    Returns a matrix of shape ``(n_queries, n_corpus)``.
    """
    q = l2_normalize(np.atleast_2d(queries))
    c = l2_normalize(np.atleast_2d(corpus))
    return (q @ c.T).astype(np.float32)


def top_k(scores: NDArray[np.float32], k: int) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
    """Return ``(indices, scores)`` of the ``k`` largest entries, descending."""
    flat = np.asarray(scores, dtype=np.float32).ravel()
    k = min(k, flat.shape[0])
    if k <= 0:
        empty_idx = np.empty(0, dtype=np.intp)
        return empty_idx, np.empty(0, dtype=np.float32)
    # argpartition is O(n); a final argsort over k keeps the top slice ordered.
    part = np.argpartition(-flat, k - 1)[:k]
    order = part[np.argsort(-flat[part])]
    return order.astype(np.intp), flat[order]
