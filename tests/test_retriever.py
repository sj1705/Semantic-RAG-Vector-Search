"""End-to-end tests for :class:`rag.retriever.RAGPipeline`.

The real sentence-transformers model is too heavy for unit tests, so we inject
the :class:`tests.conftest.FakeEmbedder` instead. One ``slow`` test exercises
the real embedder path to give us smoke coverage of the production stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.config import RagConfig
from rag.query_expansion import QueryExpander
from rag.retriever import ComparisonResult, RAGPipeline
from rag.vector_store import FaissVectorStore
from rag.vertex_mocks import GenerativeModel


def _build_pipeline(config: RagConfig, fake_embedder):
    store = FaissVectorStore(dimension=fake_embedder.dimension, metric=config.metric)
    expander = QueryExpander(
        GenerativeModel(config.generative_model_name),
        mode=config.expansion_mode,
    )
    return RAGPipeline(
        config=config,
        embedder=fake_embedder,
        vector_store=store,
        query_expander=expander,
    )


def test_ingest_populates_store(fake_embedder, tiny_corpus):
    pipeline = _build_pipeline(RagConfig(), fake_embedder)
    n = pipeline.ingest(tiny_corpus)
    assert n == len(tiny_corpus)
    assert pipeline.vector_store.size == len(tiny_corpus)


def test_ingest_empty_is_noop(fake_embedder):
    pipeline = _build_pipeline(RagConfig(), fake_embedder)
    assert pipeline.ingest([]) == 0
    assert pipeline.vector_store.size == 0


def test_strategy_a_returns_top_k(fake_embedder, tiny_corpus):
    pipeline = _build_pipeline(RagConfig(top_k=2), fake_embedder)
    pipeline.ingest(tiny_corpus)
    hits = pipeline.retrieve_raw("peak load", k=2)
    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[1].rank == 2
    # The "peak" doc must lead.
    assert hits[0].doc_id == "d-peak"


def test_strategy_b_rewrite_returns_expansion(fake_embedder, tiny_corpus):
    config = RagConfig(expansion_mode="rewrite")
    pipeline = _build_pipeline(config, fake_embedder)
    pipeline.ingest(tiny_corpus)
    hits, expansion = pipeline.retrieve_expanded("How does the system handle peak load?")
    assert len(hits) == config.top_k
    assert expansion.mode == "rewrite"
    assert len(expansion.expansions) == 1


def test_strategy_b_hyde_averages_embeddings(fake_embedder, tiny_corpus):
    config = RagConfig(expansion_mode="hyde", top_k=3)
    pipeline = _build_pipeline(config, fake_embedder)
    pipeline.ingest(tiny_corpus)
    hits, expansion = pipeline.retrieve_expanded("peak load")
    assert expansion.mode == "hyde"
    assert 1 <= len(expansion.expansions) <= 3
    assert len(hits) == 3


def test_retrieve_both_produces_comparison(fake_embedder, tiny_corpus):
    pipeline = _build_pipeline(RagConfig(), fake_embedder)
    pipeline.ingest(tiny_corpus)
    cmp: ComparisonResult = pipeline.retrieve_both(
        "How does the system handle peak load?"
    )
    assert cmp.query == "How does the system handle peak load?"
    assert len(cmp.strategy_a) == 3
    assert len(cmp.strategy_b) == 3
    assert 0 <= cmp.overlap <= 3
    assert set(cmp.a_vs_b_ids.keys()) == {"strategy_a", "strategy_b"}


def test_scores_in_cosine_range(fake_embedder, tiny_corpus):
    pipeline = _build_pipeline(RagConfig(), fake_embedder)
    pipeline.ingest(tiny_corpus)
    hits = pipeline.retrieve_raw("database failover")
    for h in hits:
        assert -1.0 <= h.score <= 1.0 + 1e-6


def test_strategy_b_can_differ_from_strategy_a(fake_embedder, tiny_corpus):
    """Core assessment assertion: query expansion must change at least *some*
    aspect of retrieval (either ordering or identity) on a targeted query.

    We assert that the Strategy-B ranked list is not identical to Strategy A on
    our peak-load query. That difference could be an ID swap, an ordering
    change, or different scores — any of which proves the expansion mechanism
    is actually influencing retrieval.
    """
    pipeline_a = _build_pipeline(RagConfig(), fake_embedder)
    pipeline_a.ingest(tiny_corpus)
    pipeline_b = _build_pipeline(
        RagConfig(expansion_mode="rewrite"), fake_embedder
    )
    pipeline_b.ingest(tiny_corpus)

    query = "How does the system handle peak load?"
    hits_a = pipeline_a.retrieve_raw(query, k=3)
    hits_b, _ = pipeline_b.retrieve_expanded(query, k=3)

    tuple_a = tuple((h.doc_id, round(h.score, 4)) for h in hits_a)
    tuple_b = tuple((h.doc_id, round(h.score, 4)) for h in hits_b)
    assert tuple_a != tuple_b, (
        "Strategy B should change the result set or scores at least once"
    )


def test_comparison_overlap_counts_correctly(fake_embedder, tiny_corpus):
    pipeline = _build_pipeline(RagConfig(), fake_embedder)
    pipeline.ingest(tiny_corpus)
    cmp = pipeline.retrieve_both("peak load")
    ids_a = {h.doc_id for h in cmp.strategy_a}
    ids_b = {h.doc_id for h in cmp.strategy_b}
    assert cmp.overlap == len(ids_a & ids_b)


@pytest.mark.slow
def test_default_pipeline_loads_real_embedder_and_runs(tmp_path):
    """Smoke test that the real sentence-transformers + FAISS stack works.

    Marked ``slow`` because it downloads the BGE model on the first run. Skip
    with ``pytest -m 'not slow'`` in CI if model downloads are a concern.
    """
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")

    from data.corpus import CORPUS

    pipeline = RAGPipeline(RagConfig())
    pipeline.ingest(CORPUS)
    hits = pipeline.retrieve_raw("How does the system handle peak load?", k=3)
    assert len(hits) == 3
    # The autoscaling doc should rank first for this query.
    assert hits[0].doc_id == "doc-01"
