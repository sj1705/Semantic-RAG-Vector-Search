"""Embedding adapters.

The pipeline depends on the :class:`Embedder` protocol so alternative backends (the
Vertex AI ``textembedding-gecko`` model, a fake for tests, etc.) can be plugged in
without touching the orchestrator.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Protocol every embedding backend must honour."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of document embeddings."""

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of query embeddings.

        Implementations MAY apply a model-specific query prefix or instruction here.
        """


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a small epsilon for numerical safety."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32, copy=False)


class SentenceTransformerEmbedder:
    """Local stand-in for Vertex AI ``textembedding-gecko``.

    Uses ``BAAI/bge-base-en-v1.5`` by default. Applies the BGE query instruction
    prefix to query inputs, and L2-normalizes all outputs so callers can safely use
    an inner-product index and treat scores as cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        # Imported lazily so test modules that monkeypatch this class don't pay the
        # ~500MB model download cost.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._dim = int(self._model.get_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for transparency
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"{self._query_prefix}{t}" for t in texts]
        return self._encode(prefixed)


class VertexGeckoEmbedder:
    """Adapter over a (mock or real) Vertex AI ``TextEmbeddingModel``.

    In production this wraps
    ``vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko")``.
    In tests it wraps :class:`rag.vertex_mocks.TextEmbeddingModel`. The interface is
    identical thanks to the :class:`Embedder` protocol, so the rest of the pipeline
    never needs to know which one is in play.
    """

    def __init__(self, vertex_model: "object", *, dimension: int) -> None:
        self._model = vertex_model
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self._model.get_embeddings(list(texts))
        matrix = np.asarray([e.values for e in embeddings], dtype=np.float32)
        return _l2_normalize(matrix)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


__all__ = ["Embedder", "SentenceTransformerEmbedder", "VertexGeckoEmbedder"]
