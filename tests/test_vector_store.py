"""Tests for :class:`rag.vector_store.FaissVectorStore`."""

from __future__ import annotations

import numpy as np
import pytest

from rag.vector_store import FaissVectorStore


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return (v / norm).astype(np.float32)


@pytest.fixture
def small_vectors() -> tuple[list[str], list[str], np.ndarray]:
    """4 toy vectors in 3-D, chosen so each cosine score is distinct.

    That matters for :func:`test_l2_topk_order_matches_cosine_on_unit_sphere`
    because FAISS's internal tie-breaking is index-specific; we want an
    unambiguous ranking in both metrics.
    """
    ids = ["a", "b", "c", "d"]
    texts = ["alpha", "beta", "gamma", "delta"]
    # Query will be [1, 0, 0]. Projections onto the x-axis (after
    # normalization): a ≈ 1.0, b ≈ 0.8, c ≈ 0.2, d ≈ 0.05 — strictly decreasing.
    vectors = np.array(
        [
            [1.00, 0.01, 0.01],  # nearly aligned with the query
            [0.80, 0.60, 0.00],
            [0.20, 0.98, 0.00],
            [0.05, 0.00, 1.00],
        ],
        dtype=np.float32,
    )
    return ids, texts, _normalize(vectors)


def test_store_empty_returns_no_hits():
    store = FaissVectorStore(dimension=4, metric="cosine")
    assert store.size == 0
    hits = store.search(np.zeros(4, dtype=np.float32), k=3)
    assert hits == []


def test_cosine_topk_order(small_vectors):
    ids, texts, vectors = small_vectors
    store = FaissVectorStore(dimension=3, metric="cosine")
    store.add(ids, texts, vectors)

    query = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))[0]
    hits = store.search(query, k=3)

    assert [h.doc_id for h in hits] == ["a", "b", "c"]
    # Scores monotonically decreasing for cosine.
    assert hits[0].score >= hits[1].score >= hits[2].score
    # Cosine scores live in [-1, 1].
    for h in hits:
        assert -1.0 <= h.score <= 1.0


def test_l2_topk_order_matches_cosine_on_unit_sphere(small_vectors):
    """On normalized vectors cosine and L2 produce the same ranking."""
    ids, texts, vectors = small_vectors

    store_cos = FaissVectorStore(dimension=3, metric="cosine")
    store_l2 = FaissVectorStore(dimension=3, metric="l2")
    store_cos.add(ids, texts, vectors)
    store_l2.add(ids, texts, vectors)

    query = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))[0]

    cos_order = [h.doc_id for h in store_cos.search(query, k=4)]
    l2_order = [h.doc_id for h in store_l2.search(query, k=4)]

    assert cos_order == l2_order


def test_dimension_mismatch_raises():
    store = FaissVectorStore(dimension=3, metric="cosine")
    with pytest.raises(ValueError):
        store.add(["x"], ["x"], np.zeros((1, 4), dtype=np.float32))


def test_length_mismatch_raises():
    store = FaissVectorStore(dimension=3, metric="cosine")
    with pytest.raises(ValueError):
        store.add(["x", "y"], ["x"], np.zeros((2, 3), dtype=np.float32))


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        FaissVectorStore(dimension=3, metric="manhattan")


def test_topk_larger_than_size_clamps(small_vectors):
    ids, texts, vectors = small_vectors
    store = FaissVectorStore(dimension=3, metric="cosine")
    store.add(ids, texts, vectors)

    query = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))[0]
    hits = store.search(query, k=100)
    assert len(hits) == 4
