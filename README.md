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
data/corpus.py    16 technical paragraphs used as the evaluation corpus
tests/            pytest suites (incl. GCP SDK mock contract tests)
scripts/          run_benchmark.py, save_index.py, search_saved.py
saved_index/      optional persisted FAISS index (index.faiss + metadata.json)
```

Docs:

- [`docs/DESIGN.md`](docs/DESIGN.md) — similarity-metric rationale + Vertex AI Matching Engine migration.
- [`docs/PERSISTENCE.md`](docs/PERSISTENCE.md) — how to save/load the FAISS index.
- [`retrieval_benchmark.md`](retrieval_benchmark.md) — generated A-vs-B comparison report (committed as dev evidence).

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

FAISS is in-memory by default. `FaissVectorStore.save()` / `.load()` write two
files (`index.faiss` + `metadata.json`) so the corpus is embedded only once.

See [`docs/PERSISTENCE.md`](docs/PERSISTENCE.md) for the full details, file format, and
programmatic / CLI usage.

## Strategies

Two strategies, per the assessment PDF:

- **Strategy A — Raw Vector Search.** Embed the query verbatim, search FAISS.
- **Strategy B — AI-Enhanced Retrieval.** A mocked `GenerativeModel` rewrites the
  query with domain synonyms before embedding and searching. This is the
  "Query Expansion" strategy named in the PDF.

**Extra:** the benchmark also reports a third column, **HyDE**, which is an
alternative Strategy B implementation where the mock produces three hypothetical
answer passages and we average their embeddings before search. HyDE is not
required by the assessment — it's included to demonstrate a second expansion
technique for comparison.

## Design decisions

See [`docs/DESIGN.md`](docs/DESIGN.md) for:
- Why cosine similarity (and when Euclidean is the right pick instead).
- How to migrate this pipeline to **Vertex AI Vector Search (Matching Engine)** in
  production without changing the `RAGPipeline` orchestrator.