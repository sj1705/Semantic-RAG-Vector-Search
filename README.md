# Context-Aware Retrieval Engine

A local Retrieval-Augmented Generation (RAG) pipeline that benchmarks two retrieval
strategies over a small technical corpus:

- **Strategy A — Raw Vector Search:** embed the query as-is, FAISS cosine search.
- **Strategy B — AI-Enhanced Retrieval:** a (mocked) Vertex AI `GenerativeModel` rewrites
  or HyDE-expands the query before embedding + search.

The implementation uses `sentence-transformers` with `BAAI/bge-base-en-v1.5` (a stand-in
for Vertex AI `textembedding-gecko`), FAISS for the vector store, and a deterministic
mock of the `vertexai.language_models` SDK so Strategy B is reproducible and testable.

## Layout

```
src/rag/          core modules (config, embeddings, vector_store, vertex_mocks,
                  query_expansion, retriever, benchmark)
data/corpus.py    8 technical paragraphs used as the evaluation corpus
tests/            pytest suites (incl. GCP SDK mock contract tests)
scripts/          run_benchmark.py, save_index.py, search_saved.py
docs/DESIGN.md    similarity-metric rationale + Vertex AI Matching Engine migration
retrieval_benchmark.md   generated A-vs-B comparison report (committed as dev evidence)
saved_index/      optional persisted FAISS index (index.faiss + metadata.json)
```

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
# source .venv/bin/activate

pip install -r requirements.txt

# Run the tests
pytest -v

# Run the benchmark (prints tables, regenerates retrieval_benchmark.md)
python scripts/run_benchmark.py
```

## Persisting the index

FAISS is an in-memory library by default — close the Python process and the
index is gone. For larger corpora you do NOT want to re-embed every startup,
so `FaissVectorStore` ships with explicit `save()` / `load()` helpers.

Two files are written, and both are required to reload:

| File | Contents | Why it's needed |
| --- | --- | --- |
| `index.faiss` | The raw vectors (binary FAISS format) | Enables nearest-neighbor search |
| `metadata.json` | `doc_ids`, `texts`, `metric`, `dimension` | Maps FAISS row numbers back to real docs; without it, search returns numbers with nothing to look up |

### Build once, search many times

```bash
# Embed the corpus and write index.faiss + metadata.json to ./saved_index/
python scripts/save_index.py

# Load the saved index (milliseconds) and run a query
python scripts/search_saved.py "How does the system handle peak load?"
python scripts/search_saved.py "database failover"
```

### Programmatic use

```python
from rag.vector_store import FaissVectorStore

# Save
store.save("saved_index")

# Load later (raises FileNotFoundError if either file is missing)
store = FaissVectorStore.load("saved_index")
```

### Why it matters at scale

| Corpus size | Without persistence (re-embed each run) | With persistence (load from disk) |
| --- | --- | --- |
| 8 docs | ~1.7 s | ~150 ms |
| 100,000 docs | ~20 minutes | < 1 s |

For this assessment the corpus is tiny, but the mechanism is in place so the
same code scales.

## Strategies

Two strategies, per the assessment PDF:

- **Strategy A — Raw Vector Search.** Embed the query verbatim, search FAISS.
- **Strategy B — AI-Enhanced Retrieval.** A mocked `GenerativeModel` rewrites the
  query with domain synonyms before embedding and searching. This is the
  "Query Expansion" strategy named in the PDF.

**Bonus:** the benchmark also reports a third column, **HyDE**, which is an
alternative Strategy B implementation where the mock produces three hypothetical
answer passages and we average their embeddings before search. HyDE is not
required by the assessment — it's included to demonstrate a second expansion
technique for comparison.

## Design decisions

See `docs/DESIGN.md` for:
- Why cosine similarity (and when Euclidean is the right pick instead).
- How to migrate this pipeline to **Vertex AI Vector Search (Matching Engine)** in
  production without changing the `RAGPipeline` orchestrator.