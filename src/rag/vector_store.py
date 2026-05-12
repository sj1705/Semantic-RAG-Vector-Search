"""FAISS-backed vector store.

Thin wrapper around FAISS indexes that exposes a small, swap-friendly API:
``add`` and ``search``. The abstraction is deliberately minimal so that a future
``MatchingEngineVectorStore`` can be dropped in behind the same protocol without
touching callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class StoredHit:
    """A single retrieval result returned by a :class:`VectorStore`."""

    doc_id: str
    text: str
    score: float  # higher = better, regardless of underlying metric


class VectorStore(Protocol):
    """Minimal interface the pipeline relies on."""

    @property
    def size(self) -> int: ...

    def add(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
    ) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[StoredHit]: ...


class FaissVectorStore:
    """FAISS-backed store with cosine or L2 similarity.

    Cosine path: vectors are assumed already L2-normalized (the
    :class:`SentenceTransformerEmbedder` guarantees this) and we use an
    ``IndexFlatIP`` so scores are inner products in ``[-1, 1]``.

    L2 path: uses ``IndexFlatL2`` and returns ``-distance`` so larger scores still
    mean "more similar", giving callers a single consistent ordering rule.
    """

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        import faiss

        if metric not in {"cosine", "l2"}:
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")

        self._metric = metric
        self._dim = int(dimension)
        self._index = (
            faiss.IndexFlatIP(self._dim) if metric == "cosine"
            else faiss.IndexFlatL2(self._dim)
        )
        self._doc_ids: list[str] = []
        self._texts: list[str] = []

    # -- introspection --------------------------------------------------------

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    # -- mutation -------------------------------------------------------------

    def add(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        vectors: np.ndarray,
    ) -> None:
        if not (len(doc_ids) == len(texts) == vectors.shape[0]):
            raise ValueError(
                "doc_ids, texts, and vectors must have matching lengths: "
                f"{len(doc_ids)}, {len(texts)}, {vectors.shape[0]}"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != index dim {self._dim}"
            )

        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._doc_ids.extend(doc_ids)
        self._texts.extend(texts)

    # -- search ---------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int) -> list[StoredHit]:
        if self.size == 0:
            return []

        query = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        if query.shape[1] != self._dim:
            raise ValueError(
                f"query dim {query.shape[1]} != index dim {self._dim}"
            )

        k = min(k, self.size)
        distances, indices = self._index.search(query, k)

        hits: list[StoredHit] = []
        for raw_score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Normalize ordering: larger score = better similarity for both metrics.
            score = float(raw_score) if self._metric == "cosine" else -float(raw_score)
            hits.append(
                StoredHit(
                    doc_id=self._doc_ids[int(idx)],
                    text=self._texts[int(idx)],
                    score=score,
                )
            )
        return hits

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist the index and its metadata to disk.

        Writes two files into ``directory``:

        * ``index.faiss``   — the vector index (FAISS's binary format).
        * ``metadata.json`` — ``doc_ids``, ``texts``, ``metric``, and
          ``dimension`` so the store can be rebuilt exactly.

        The two files MUST be kept together — the FAISS index only knows row
        numbers; the metadata file is what maps those rows back to doc IDs and
        source text.
        """
        import faiss

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))

        metadata = {
            "metric": self._metric,
            "dimension": self._dim,
            "doc_ids": self._doc_ids,
            "texts": self._texts,
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        """Rebuild a :class:`FaissVectorStore` from files written by :meth:`save`.

        Raises :class:`FileNotFoundError` if either ``index.faiss`` or
        ``metadata.json`` is missing — they are a matched pair and partial
        state is meaningless.
        """
        import faiss

        path = Path(directory)
        index_path = path / "index.faiss"
        metadata_path = path / "metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Both index.faiss and metadata.json must exist in {path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        store = cls.__new__(cls)  # bypass __init__ (which would build a fresh index)
        store._metric = metadata["metric"]
        store._dim = int(metadata["dimension"])
        store._index = faiss.read_index(str(index_path))
        store._doc_ids = list(metadata["doc_ids"])
        store._texts = list(metadata["texts"])

        # Sanity check: the three collections must stay aligned.
        if store._index.ntotal != len(store._doc_ids):
            raise ValueError(
                f"Corrupt store: index has {store._index.ntotal} vectors but "
                f"{len(store._doc_ids)} doc_ids on disk"
            )
        return store


__all__ = ["StoredHit", "VectorStore", "FaissVectorStore"]
