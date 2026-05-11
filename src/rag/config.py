"""Runtime configuration for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Metric = Literal["cosine", "l2"]
ExpansionMode = Literal["rewrite", "hyde"]


@dataclass(frozen=True)
class RagConfig:
    """Single source of truth for pipeline knobs.

    The pipeline is built by injecting an :class:`RagConfig`; keeping it frozen prevents
    tests or benchmarks from mutating state between runs.
    """

    # Embedding model. We use BAAI/bge-base-en-v1.5 as a local stand-in for Vertex AI's
    # ``textembedding-gecko``. Output vectors are L2-normalized (768-dim) which makes
    # cosine similarity equivalent to an inner product.
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # BGE was trained with an instruction prefix for queries only.
    # https://huggingface.co/BAAI/bge-base-en-v1.5
    bge_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # Similarity metric. Cosine is the default because BGE outputs are normalized and
    # cosine scores are bounded in [-1, 1]. ``l2`` is kept for the comparison in docs.
    metric: Metric = "cosine"

    # Strategy B mode. ``rewrite`` -> single domain-enriched rewrite.
    # ``hyde``   -> three hypothetical answer passages whose embeddings are averaged.
    expansion_mode: ExpansionMode = "rewrite"

    # Default top-k returned by retrieve_* calls.
    top_k: int = 3

    # Name used by the mock generative model. Mirrors Vertex AI naming so the swap to
    # the real client is a single line change.
    generative_model_name: str = "gemini-1.5-pro-mock"
