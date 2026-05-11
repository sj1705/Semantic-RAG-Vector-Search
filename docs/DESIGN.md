# Design Notes

This document covers the two documentation requirements in the assessment:

1. **Why cosine similarity** was chosen over Euclidean distance.
2. **How to migrate** this local pipeline to **Vertex AI Vector Search
   (Matching Engine)** in production.

## 1. Similarity metric: cosine vs Euclidean

### What the code does

`FaissVectorStore` supports both metrics via a single `metric` flag
(`"cosine"` or `"l2"`). The default is `"cosine"`, backed by
`faiss.IndexFlatIP` over pre-normalized vectors.

### Why cosine

Modern sentence embeddings (BGE, Vertex AI `textembedding-gecko`, E5, OpenAI
`text-embedding-3-*`) emit **L2-normalized** vectors — every output lies on the
unit sphere. When vectors are unit-length:

```
‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·b = 2 − 2·cos(a, b)
```

Euclidean distance is a **monotonically decreasing** function of cosine
similarity, so the *ranking* produced by the two metrics is identical. (This is
also confirmed by `tests/test_vector_store.py::test_l2_topk_order_matches_cosine_on_unit_sphere`.)

Given that ranking is equivalent, cosine is preferred because:

- **Bounded range.** Cosine scores live in `[-1, 1]`, which is easy to threshold
  (“drop hits below 0.3”), log, alert on, and reason about. Euclidean distance
  is unbounded above and depends on vector norms.
- **Magnitude-invariance.** If future changes ever ship a backend that forgets
  to normalize (e.g. a fine-tuned model, a mis-configured pipeline, or an older
  embedder), cosine still compares *direction*. Euclidean would silently give
  different rankings depending on vector magnitude.
- **Infrastructure parity.** Vertex AI Matching Engine's
  `DOT_PRODUCT_DISTANCE` is exactly `-a·b` — equivalent to cosine for unit
  vectors. Choosing cosine here means the production index uses the same
  semantics, so benchmarks transfer.

### When Euclidean is the right choice

- **Non-normalized features** such as image descriptors, some older word
  embeddings (Word2Vec without post-normalization), or anything where the
  vector's *magnitude* carries signal.
- **Absolute-distance clustering.** Algorithms like k-means assume squared
  Euclidean; retrieval over k-means-style centroids (IVF coarse quantization)
  is happier in L2.
- **Mixed-domain spaces** where vectors from different encoders coexist and
  you don't want direction-only comparison.

Because all of our embeddings are unit-normalized, none of those conditions
apply.

### Implementation note

Both code paths use FAISS `Flat` indexes because the benchmark corpus has 8
documents; exact search is cheaper than building an approximate index. When
the corpus grows past ~10k vectors, swap `IndexFlatIP` for `IndexHNSWFlat`
(still inner-product, just approximate) without changing any caller.

---

## 2. Migration to Vertex AI Vector Search (Matching Engine) in production

The pipeline was designed so that swapping the local stack for Vertex AI is a
handful of small, low-risk changes. Here is the migration plan.

### 2.1 Dependencies and authentication

Add to `requirements.txt`:

```
google-cloud-aiplatform>=1.60.0
```

Provide credentials with either `GOOGLE_APPLICATION_CREDENTIALS` pointing at a
service-account key or Workload Identity in GKE. Call
`vertexai.init(project=PROJECT, location=REGION)` once during process start-up.

### 2.2 Swap the embedding backend

The code already ships a `VertexGeckoEmbedder` that honours the `Embedder`
protocol. In production we wire the real Vertex client behind it:

```python
from vertexai.language_models import TextEmbeddingModel
from rag.embeddings import VertexGeckoEmbedder

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embedder = VertexGeckoEmbedder(model, dimension=768)
```

No change elsewhere — the rest of the pipeline talks through the protocol.

### 2.3 Replace `FaissVectorStore` with a Matching Engine adapter

Create `MatchingEngineVectorStore` implementing the same `VectorStore`
protocol. Outline:

```python
from google.cloud import aiplatform

class MatchingEngineVectorStore:
    def __init__(self, index_endpoint: str, deployed_index_id: str):
        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint)
        self._deployed_index_id = deployed_index_id
        # keep a parallel metadata store (Firestore / Cloud SQL) for doc text
        ...

    def add(self, doc_ids, texts, vectors):
        # Upsert into the Index and into the metadata store in a single txn.
        self._metadata.upsert(doc_ids, texts)
        self._index.upsert_datapoints(
            [IndexDatapoint(datapoint_id=i, feature_vector=v.tolist())
             for i, v in zip(doc_ids, vectors)]
        )

    def search(self, query_vector, k):
        response = self._endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=[query_vector.tolist()],
            num_neighbors=k,
        )
        hits = []
        for n in response[0]:
            text = self._metadata.get(n.id)
            hits.append(StoredHit(doc_id=n.id, text=text,
                                  score=1.0 - n.distance))  # DOT_PRODUCT_DISTANCE -> similarity
        return hits
```

Wire it via `RagConfig` + dependency injection — `RAGPipeline` already accepts a
`vector_store` argument.

### 2.4 Index creation (one-time)

```bash
gcloud ai indexes create \
  --project=$PROJECT --region=$REGION \
  --display-name=rag-index-v1 \
  --metadata-file=index_metadata.json
```

`index_metadata.json`:

```json
{
  "contentsDeltaUri": "gs://<bucket>/deltas/initial/",
  "config": {
    "dimensions": 768,
    "approximateNeighborsCount": 100,
    "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
    "algorithmConfig": {
      "treeAhConfig": {
        "leafNodeEmbeddingCount": 1000,
        "leafNodesToSearchPercent": 10
      }
    }
  }
}
```

Key choices:

- `DOT_PRODUCT_DISTANCE` because BGE/gecko outputs are normalized — equivalent
  to cosine similarity, same ranking as our FAISS prototype.
- `treeAhConfig` (ScaNN) is the default for accuracy/latency balance; tune
  `leafNodesToSearchPercent` during perf tests.

Deploy the index to an `IndexEndpoint`:

```bash
gcloud ai index-endpoints create --display-name=rag-endpoint
gcloud ai index-endpoints deploy-index $ENDPOINT \
  --deployed-index-id=rag-v1 --index=$INDEX_ID \
  --min-replica-count=1 --max-replica-count=3
```

### 2.5 Swap the generative model

`QueryExpander` accepts any object with a `generate_content(prompt) -> resp`
method. The real Vertex AI class satisfies this directly:

```python
from vertexai.generative_models import GenerativeModel
expander = QueryExpander(GenerativeModel("gemini-1.5-pro"), mode="rewrite")
```

The `REWRITE:` / `HYDE:` prompt conventions used in the mock are legitimate
prompts for the real model — no code changes needed. In practice you'd
upgrade them with structured output instructions, temperature/top-p settings,
and timeouts wired through `RagConfig`.

### 2.6 Operational concerns

- **Metadata store.** Matching Engine stores only vectors + IDs; keep the
  original text in Firestore or Cloud SQL keyed by `doc_id` so `StoredHit.text`
  can be hydrated.
- **Observability.** Export Matching Engine query latency / QPS to Cloud
  Monitoring; emit a custom metric for A-vs-B top-k overlap so you can detect
  regressions after prompt or model upgrades.
- **Cost & scaling.** Matching Engine charges per deployed replica-hour plus
  per-query. The ScaNN algorithm scales to hundreds of millions of vectors;
  tune `approximateNeighborsCount` and `leafNodesToSearchPercent` to find
  your recall/latency/cost sweet spot.
- **CI/CD.** Keep the same pytest suite; monkey-patch
  `vertexai.language_models` with `rag.vertex_mocks` (already done via the
  `stub_vertex_module` fixture). Real Vertex calls stay off the unit-test
  path.

### 2.7 Recap

Because every seam in the pipeline — embedding, vector store, query expansion
— is a Protocol, migration to Vertex AI Vector Search is a **three-file
change**:

| File | Change |
| ---- | ------ |
| `requirements.txt` | add `google-cloud-aiplatform` |
| `src/rag/vector_store.py` | add `MatchingEngineVectorStore` alongside `FaissVectorStore` |
| `scripts/run_benchmark.py` (or a new `scripts/run_production.py`) | build the pipeline with Vertex adapters |

Everything else — the retrieval orchestrator, the expander, the benchmark
harness, and the tests — continues to work unchanged.
