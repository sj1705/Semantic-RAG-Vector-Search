"""Tests for the embedding adapters.

Covers only the adapters that don't need network / model downloads: the
``VertexGeckoEmbedder`` backed by our mock, plus the ``FakeEmbedder`` used
across the suite. The real sentence-transformers loader is validated in
``test_retriever.py::test_real_embedder_loads`` which is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import VertexGeckoEmbedder, _l2_normalize  # type: ignore[attr-defined]
from rag.vertex_mocks import TextEmbeddingModel


def test_l2_normalize_unit_length():
    vecs = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(vecs)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out[1]), 1.0, atol=1e-6)
    # Zero vectors should stay finite (not NaN).
    assert np.all(np.isfinite(out[2]))


def test_vertex_gecko_embedder_shapes():
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
    embedder = VertexGeckoEmbedder(model, dimension=model.dimension)

    docs = embedder.embed_documents(["hello world", "peak load autoscaling"])
    queries = embedder.embed_queries(["failover", "database"])

    assert docs.shape == (2, model.dimension)
    assert queries.shape == (2, model.dimension)
    assert docs.dtype == np.float32


def test_vertex_gecko_embedder_normalized():
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
    embedder = VertexGeckoEmbedder(model, dimension=model.dimension)
    out = embedder.embed_documents(["hello world"])
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_fake_embedder_shapes(fake_embedder):
    """Sanity check on the test-only embedder used throughout the suite."""
    q = fake_embedder.embed_queries(["peak load"])
    d = fake_embedder.embed_documents(["peak load"])
    assert q.shape == (1, fake_embedder.dimension)
    assert d.shape == (1, fake_embedder.dimension)
    # Query vectors are normalized.
    np.testing.assert_allclose(np.linalg.norm(q[0]), 1.0, atol=1e-5)
    # Unknown tokens -> zero vector, which gets normalized to an all-zero vector
    # (the implementation guards against divide-by-zero).
    zero = fake_embedder.embed_documents(["zzz_unknown_token_xyz"])
    assert np.all(zero == 0)
