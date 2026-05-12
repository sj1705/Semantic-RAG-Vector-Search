"""Build the FAISS index from the corpus and save it to disk.

Run once to create the index. After this, you can load the saved index in
milliseconds instead of re-embedding everything from scratch.

Usage::

    python scripts/save_index.py                # saves to ./saved_index/
    python scripts/save_index.py --out mydir    # custom directory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.corpus import CORPUS  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and save a FAISS index.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "saved_index",
        help="Directory to write index.faiss and metadata.json into.",
    )
    args = parser.parse_args()

    config = RagConfig()

    print(f"Loading embedding model: {config.embedding_model}")
    t0 = time.time()
    embedder = SentenceTransformerEmbedder(
        model_name=config.embedding_model,
        query_prefix=config.bge_query_prefix,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Embedding {len(CORPUS)} documents…")
    t0 = time.time()
    texts = [d.text for d in CORPUS]
    ids = [d.id for d in CORPUS]
    vectors = embedder.embed_documents(texts)
    print(f"  embedded in {time.time() - t0:.1f}s  (shape: {vectors.shape})")

    store = FaissVectorStore(dimension=embedder.dimension, metric=config.metric)
    store.add(ids, texts, vectors)

    args.out.mkdir(parents=True, exist_ok=True)
    store.save(args.out)

    index_size = (args.out / "index.faiss").stat().st_size
    meta_size = (args.out / "metadata.json").stat().st_size
    print()
    print(f"Saved to: {args.out}")
    print(f"  index.faiss    {index_size:>8} bytes   (the vectors)")
    print(f"  metadata.json  {meta_size:>8} bytes   (ids + texts)")
    print()
    print("Reload anytime with:")
    print(f"  FaissVectorStore.load({str(args.out)!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
