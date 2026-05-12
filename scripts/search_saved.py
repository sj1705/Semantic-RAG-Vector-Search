"""Load the saved FAISS index and run a search — no re-embedding the corpus.

Demonstrates the payoff of :meth:`FaissVectorStore.save`: the 8 documents are
loaded from disk in milliseconds. Only the query needs to go through the
embedding model.

Usage::

    python scripts/search_saved.py "How does the system handle peak load?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag.config import RagConfig  # noqa: E402
from rag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a saved FAISS index.")
    parser.add_argument("query", help="The query string.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "saved_index",
        help="Directory containing index.faiss and metadata.json.",
    )
    parser.add_argument("-k", type=int, default=3, help="Number of hits.")
    args = parser.parse_args()

    t0 = time.time()
    store = FaissVectorStore.load(args.dir)
    print(f"Loaded {store.size} vectors from disk in {(time.time() - t0)*1000:.1f} ms")

    # We still need the embedder for the *query* (one vector).
    # Loading the model is slow once; every subsequent query is fast.
    t0 = time.time()
    config = RagConfig()
    embedder = SentenceTransformerEmbedder(
        model_name=config.embedding_model,
        query_prefix=config.bge_query_prefix,
    )
    print(f"Loaded embedder in {time.time() - t0:.1f}s")

    t0 = time.time()
    qvec = embedder.embed_queries([args.query])[0]
    hits = store.search(qvec, k=args.k)
    print(f"Search took {(time.time() - t0)*1000:.1f} ms")
    print()
    print(f"Query: {args.query!r}")
    print(f"Top {len(hits)} hits:")
    for rank, h in enumerate(hits, 1):
        snippet = " ".join(h.text.split())[:100] + "..."
        print(f"  {rank}. {h.doc_id}  score={h.score:.4f}")
        print(f"     {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
