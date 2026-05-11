"""End-to-end RAG orchestrator.

Ties the :class:`Embedder`, :class:`VectorStore`, and :class:`QueryExpander`
together behind a single class that the benchmark and the tests drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

from rag.config import RagConfig
from rag.embeddings import Embedder, SentenceTransformerEmbedder
from rag.query_expansion import ExpansionResult, QueryExpander
from rag.vector_store import FaissVectorStore, StoredHit, VectorStore


class _DocLike(Protocol):
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    """Single ranked retrieval result enriched with the source document text."""

    doc_id: str
    score: float
    text: str
    rank: int


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side outcome of Strategy A vs each Strategy B variant."""

    query: str
    strategy_a: list[RetrievalHit]
    strategy_b: list[RetrievalHit]
    expansion: ExpansionResult
    overlap: int = 0  # how many doc_ids are shared in the two top-k lists
    a_vs_b_ids: dict[str, list[str]] = field(default_factory=dict)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32, copy=False)


class RAGPipeline:
    """Orchestrator the assessment explicitly asks for.

    Usage::

        pipeline = RAGPipeline(RagConfig())
        pipeline.ingest(CORPUS)
        hits_a = pipeline.retrieve_raw("...")           # Strategy A
        hits_b = pipeline.retrieve_expanded("...")      # Strategy B
        cmp    = pipeline.retrieve_both("...")          # A vs B diff

    All heavy dependencies — the embedding model, the vector store, and the
    generative model powering query expansion — can be injected, which lets the
    pytest suite swap in fast fakes without monkeypatching the modules.
    """

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self.config = config or RagConfig()

        # Lazily construct the default embedder so tests that inject their own
        # fake never download the sentence-transformers model.
        self._embedder: Embedder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbedder(
                model_name=self.config.embedding_model,
                query_prefix=self.config.bge_query_prefix,
            )
        )

        self._store: VectorStore = (
            vector_store
            if vector_store is not None
            else FaissVectorStore(
                dimension=self._embedder.dimension,
                metric=self.config.metric,
            )
        )

        if query_expander is not None:
            self._expander: QueryExpander = query_expander
        else:
            # Default to the mock-backed expander so the pipeline is fully
            # self-contained and reproducible out of the box.
            from rag.vertex_mocks import GenerativeModel

            self._expander = QueryExpander(
                model=GenerativeModel(self.config.generative_model_name),
                mode=self.config.expansion_mode,
            )

    # -- accessors ---------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    @property
    def expander(self) -> QueryExpander:
        return self._expander

    # -- ingestion ---------------------------------------------------------

    def ingest(self, documents: Iterable[_DocLike]) -> int:
        """Embed and add a batch of documents. Returns the count inserted."""
        docs = list(documents)
        if not docs:
            return 0

        texts = [d.text for d in docs]
        ids = [d.id for d in docs]
        vectors = self._embedder.embed_documents(texts)
        self._store.add(ids, texts, vectors)
        return len(docs)

    # -- retrieval ---------------------------------------------------------

    def retrieve_raw(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Strategy A: embed the query as-is and return the top-k neighbours."""
        k = k or self.config.top_k
        vector = self._embedder.embed_queries([query])[0]
        raw_hits = self._store.search(vector, k)
        return _to_ranked(raw_hits)

    def retrieve_expanded(
        self, query: str, k: int | None = None
    ) -> tuple[list[RetrievalHit], ExpansionResult]:
        """Strategy B: expand the query via the mock LLM, then search."""
        k = k or self.config.top_k
        expansion = self._expander.expand(query)
        query_vector = self._embed_expansion(expansion.expansions)
        raw_hits = self._store.search(query_vector, k)
        return _to_ranked(raw_hits), expansion

    def retrieve_both(self, query: str, k: int | None = None) -> ComparisonResult:
        """Run A and B for the same query and return a comparison object."""
        k = k or self.config.top_k
        hits_a = self.retrieve_raw(query, k)
        hits_b, expansion = self.retrieve_expanded(query, k)

        ids_a = [h.doc_id for h in hits_a]
        ids_b = [h.doc_id for h in hits_b]
        overlap = len(set(ids_a) & set(ids_b))

        return ComparisonResult(
            query=query,
            strategy_a=hits_a,
            strategy_b=hits_b,
            expansion=expansion,
            overlap=overlap,
            a_vs_b_ids={"strategy_a": ids_a, "strategy_b": ids_b},
        )

    # -- internals ---------------------------------------------------------

    def _embed_expansion(self, expansions: Sequence[str]) -> np.ndarray:
        """Embed each expansion and average in vector space.

        A single expansion (``rewrite`` mode) is returned as-is. Multiple
        expansions (HyDE) are averaged and renormalized so the resulting vector
        still lives on the unit sphere, which keeps cosine-similarity scoring
        well-behaved.
        """
        vectors = self._embedder.embed_queries(list(expansions))
        if vectors.shape[0] == 1:
            return vectors[0]
        mean = vectors.mean(axis=0, keepdims=True)
        mean = _l2_normalize_rows(mean)
        return mean[0]


def _to_ranked(hits: list[StoredHit]) -> list[RetrievalHit]:
    return [
        RetrievalHit(doc_id=h.doc_id, score=h.score, text=h.text, rank=i + 1)
        for i, h in enumerate(hits)
    ]


__all__ = ["RAGPipeline", "RetrievalHit", "ComparisonResult"]
