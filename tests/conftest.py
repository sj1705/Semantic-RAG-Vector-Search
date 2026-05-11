"""Shared pytest fixtures.

We deliberately avoid loading the real sentence-transformers model in unit
tests so the suite runs in <1s. Instead we use a small deterministic
``FakeEmbedder`` that embeds on a fixed vocabulary; this is enough to validate
all of the orchestration logic while staying offline.

For the end-to-end "Strategy A ≠ Strategy B" assertion we seed the fake so that
the mock generative model's rewrite demonstrably shifts the retrieved doc set.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

# Make ``src`` and the repo root importable without installing the package.
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Fake embedder — used across the suite
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32, copy=False)


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a fixed vocabulary.

    Tests can add vocabulary entries at construction time to steer which tokens
    map to which dimensions. Query and document text embeddings live in the
    same space so cosine similarity is meaningful.
    """

    def __init__(self, vocabulary: Sequence[str], query_prefix: str = "") -> None:
        self._vocab = {token: i for i, token in enumerate(vocabulary)}
        self._dim = len(vocabulary)
        self._query_prefix = query_prefix

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [f"{self._query_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                idx = self._vocab.get(token)
                if idx is not None:
                    matrix[row, idx] += 1.0
        return _l2_normalize(matrix)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TestDoc:
    id: str
    title: str
    text: str


@pytest.fixture
def tiny_corpus() -> list[_TestDoc]:
    """Three deliberately-contrastive docs. Each owns distinct vocab tokens.

    The keyword-map-driven mock in ``rag.vertex_mocks`` will inject the
    synonyms that cause Strategy B to pick a different top-1 than Strategy A.
    """
    return [
        _TestDoc(
            id="d-peak",
            title="peak",
            text=(
                "peak load traffic autoscaling horizontal pod autoscaler "
                "throughput burst capacity rate limiter load leveler"
            ),
        ),
        _TestDoc(
            id="d-db",
            title="db",
            text=(
                "database postgres streaming replication standby failover "
                "patroni availability zone disaster recovery"
            ),
        ),
        _TestDoc(
            id="d-billing",
            title="billing",
            text=(
                "billing invoicing usage warehouse rating ledger payments "
                "reconciliation portal PDF"
            ),
        ),
    ]


@pytest.fixture
def fake_embedder(tiny_corpus):
    """Embedder whose vocabulary is the union of the tiny corpus tokens plus
    the synonyms the mock generative model injects for our test query."""
    tokens: set[str] = set()
    for doc in tiny_corpus:
        tokens.update(_TOKEN_RE.findall(doc.text.lower()))
    # Include query-side tokens and synonyms emitted by the rewrite mock.
    tokens.update(
        _TOKEN_RE.findall(
            "how does the system handle peak load autoscaling horizontal pod "
            "autoscaler traffic spikes rate limiter throughput burst capacity "
            "load leveler context"
        )
    )
    return FakeEmbedder(vocabulary=sorted(tokens), query_prefix="query: ")


@pytest.fixture
def stub_vertex_module(monkeypatch):
    """Install ``rag.vertex_mocks`` as a stand-in for ``vertexai.language_models``.

    This proves the contract the assessment asks for: *any* code written against
    ``vertexai.language_models.TextEmbeddingModel`` / ``GenerativeModel`` will
    work against our mock because we expose the same import path.
    """
    import types

    from rag.vertex_mocks import GenerativeModel, TextEmbeddingModel

    fake_pkg = types.ModuleType("vertexai")
    fake_lang = types.ModuleType("vertexai.language_models")
    fake_lang.TextEmbeddingModel = TextEmbeddingModel
    fake_lang.GenerativeModel = GenerativeModel
    fake_pkg.language_models = fake_lang

    monkeypatch.setitem(sys.modules, "vertexai", fake_pkg)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", fake_lang)
    yield fake_lang
