# FAISS Index Persistence

FAISS is an in-memory library by default — close the Python process and the
index is gone. For larger corpora you do not want to re-embed every startup,
so `FaissVectorStore` ships with explicit `save()` / `load()` helpers.

## Files written

Two files are written, and both are required to reload:

| File | Contents | Why it's needed |
| --- | --- | --- |
| `index.faiss` | The raw vectors (binary FAISS format) | Enables nearest-neighbor search |
| `metadata.json` | `doc_ids`, `texts`, `metric`, `dimension` | Maps FAISS row numbers back to real docs; without it, search returns numbers with nothing to look up |

## Build once, search many times

```bash
# Embed the corpus and write index.faiss + metadata.json to ./saved_index/
python scripts/save_index.py

# Load the saved index (milliseconds) and run a query
python scripts/search_saved.py "How does the system handle peak load?"
python scripts/search_saved.py "database failover"
```

## Programmatic use

```python
from rag.vector_store import FaissVectorStore

# Save
store.save("saved_index")

# Load later (raises FileNotFoundError if either file is missing)
store = FaissVectorStore.load("saved_index")
```

## Benchmark integration

`scripts/run_benchmark.py` automatically loads `./saved_index/` if it exists,
and falls back to embedding the corpus once (then saving it) if it does not.
That means 10 queries × 3 strategies = 30 pipeline builds all share the same
index — the corpus is embedded exactly once per benchmark run.

## Why it matters at scale

| Corpus size | Without persistence (re-embed each run) | With persistence (load from disk) |
| --- | --- | --- |
| 8 docs | ~1.7 s | ~150 ms |
| 100,000 docs | ~20 minutes | < 1 s |

For this assessment the corpus is tiny, but the mechanism is in place so the
same code scales.
