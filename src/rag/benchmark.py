"""Benchmark harness: Strategy A vs Strategy B.

Runs the same set of queries against the pipeline twice per strategy B mode and
emits both a JSON-serializable summary and a markdown report suitable for
committing as dev evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from rag.config import RagConfig
from rag.retriever import ComparisonResult, RAGPipeline, RetrievalHit
from rag.query_expansion import QueryExpander
from rag.vertex_mocks import GenerativeModel


# Benchmark queries — the assessment mandates ≥3 complex queries; we include
# ten that span the corpus topics so Strategy A vs Strategy B deltas are
# visible across a variety of retrieval patterns.
DEFAULT_QUERIES: tuple[str, ...] = (
    "How does the system handle peak load?",
    "What happens when the primary database fails over?",
    "How is sensitive customer data protected in transit?",
    "Explain the observability stack used to diagnose incidents.",
    "How are user sessions and access tokens managed?",
    "Describe how secrets and credentials are rotated.",
    "How do feature flags support safe experimentation?",
    "What does the disaster recovery strategy cover?",
    "How is personal data handled when a user requests deletion?",
    "How are push notifications delivered to mobile devices?",
)


@dataclass
class StrategyRun:
    """Result of a single strategy against a single query."""

    label: str
    hits: list[RetrievalHit]
    expansion: str | None = None  # present only for Strategy B
    expansions_used: list[str] = field(default_factory=list)


@dataclass
class QueryReport:
    query: str
    runs: dict[str, StrategyRun]
    overlap: dict[str, int]  # label_b -> #common doc_ids with A

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "runs": {
                label: {
                    "label": run.label,
                    "expansion_mode": run.expansion,
                    "expansions_used": run.expansions_used,
                    "hits": [
                        {
                            "rank": h.rank,
                            "doc_id": h.doc_id,
                            "score": round(h.score, 6),
                            "snippet": _snippet(h.text),
                        }
                        for h in run.hits
                    ],
                }
                for label, run in self.runs.items()
            },
            "overlap_with_strategy_a": self.overlap,
        }


def run_benchmark(
    pipeline_factory,
    queries: Sequence[str] = DEFAULT_QUERIES,
    *,
    k: int = 3,
) -> list[QueryReport]:
    """Execute the benchmark and return per-query reports.

    ``pipeline_factory`` is a zero-arg callable returning a freshly ingested
    :class:`RAGPipeline`. We rebuild the pipeline per strategy B mode to cleanly
    reset the expander, while the index is rebuilt each time to avoid carrying
    state between runs.
    """
    reports: list[QueryReport] = []

    for query in queries:
        runs: dict[str, StrategyRun] = {}

        # Strategy A — raw vector search. Any pipeline works for this.
        pipeline_for_a = pipeline_factory()
        runs["strategy_a"] = StrategyRun(
            label="Strategy A (Raw Vector Search)",
            hits=pipeline_for_a.retrieve_raw(query, k),
        )

        # Strategy B — the query-rewrite mode described in the assessment PDF.
        pipeline_b_rewrite = pipeline_factory(expansion_mode="rewrite")
        comparison: ComparisonResult = pipeline_b_rewrite.retrieve_both(query, k)
        runs["strategy_b_rewrite"] = StrategyRun(
            label="Strategy B (AI-Enhanced Retrieval — Query Rewrite)",
            hits=comparison.strategy_b,
            expansion="rewrite",
            expansions_used=list(comparison.expansion.expansions),
        )

        # Bonus exploration: HyDE-style expansion. Not required by the PDF.
        pipeline_b_hyde = pipeline_factory(expansion_mode="hyde")
        comparison_hyde: ComparisonResult = pipeline_b_hyde.retrieve_both(query, k)
        runs["strategy_b_hyde"] = StrategyRun(
            label="Strategy B variant (HyDE Expansion)",
            hits=comparison_hyde.strategy_b,
            expansion="hyde",
            expansions_used=list(comparison_hyde.expansion.expansions),
        )

        ids_a = {h.doc_id for h in runs["strategy_a"].hits}
        overlap = {
            "strategy_b_rewrite": len(
                ids_a & {h.doc_id for h in runs["strategy_b_rewrite"].hits}
            ),
            "strategy_b_hyde": len(
                ids_a & {h.doc_id for h in runs["strategy_b_hyde"].hits}
            ),
        }

        reports.append(QueryReport(query=query, runs=runs, overlap=overlap))

    return reports


def make_default_pipeline_factory(corpus):
    """Return a zero-arg factory that builds a fresh pipeline per call.

    The returned factory accepts an optional ``expansion_mode`` override so the
    benchmark can flip between ``rewrite`` and ``hyde`` without the caller
    constructing configs by hand.
    """

    def _factory(*, expansion_mode: str = "rewrite") -> RAGPipeline:
        config = RagConfig(expansion_mode=expansion_mode)
        pipeline = RAGPipeline(config)
        pipeline.ingest(corpus)
        return pipeline

    return _factory


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def reports_to_markdown(reports: Iterable[QueryReport]) -> str:
    """Render the reports as a committable markdown document."""
    lines: list[str] = []
    lines.append("# Retrieval Benchmark: Strategy A vs Strategy B")
    lines.append("")
    lines.append(
        "This report is generated by `scripts/run_benchmark.py` and satisfies the"
    )
    lines.append(
        "assessment PDF's requirement to compare **Strategy A (Raw Vector Search)**"
    )
    lines.append(
        "against **Strategy B (AI-Enhanced Retrieval)** over the corpus in"
    )
    lines.append("`data/corpus.py`.")
    lines.append("")
    lines.append("- **Strategy A** — raw embedding similarity (baseline).")
    lines.append(
        "- **Strategy B** — a mocked `GenerativeModel` rewrites the query with"
    )
    lines.append(
        "  domain synonyms before embedding. *This is the \"Query Expansion\""
    )
    lines.append("  strategy described in the PDF.*")
    lines.append("")
    lines.append(
        "A third column, ** HyDE**, is included for depth — a different"
    )
    lines.append(
        "expansion technique where the mock produces three hypothetical answer"
    )
    lines.append(
        "passages whose embeddings are averaged and re-normalized before search."
    )
    lines.append(
        "HyDE is not required by the PDF; it's shown to demonstrate an alternative"
    )
    lines.append("Strategy B implementation.")
    lines.append("")
    lines.append(
        "Scores are cosine similarities in `[-1, 1]` (higher is better). "
        "`Overlap` is the number of top-3 doc IDs shared with Strategy A."
    )
    lines.append("")

    for report in reports:
        lines.append(f"## Query: {report.query!r}")
        lines.append("")

        for label, run in report.runs.items():
            lines.append(f"### {run.label}")
            if run.expansions_used:
                lines.append("")
                lines.append(f"**Expansion mode:** `{run.expansion}`")
                lines.append("")
                lines.append("**Expansions used:**")
                for e in run.expansions_used:
                    lines.append(f"- {e}")
                lines.append("")
            lines.append("| Rank | Doc ID | Score | Snippet |")
            lines.append("| ---: | :----- | ----: | :------ |")
            for hit in run.hits:
                lines.append(
                    f"| {hit.rank} | `{hit.doc_id}` | {hit.score:.4f} | "
                    f"{_snippet(hit.text)} |"
                )
            lines.append("")

        lines.append(
            f"**Overlap with Strategy A top-3:** "
            f"rewrite={report.overlap['strategy_b_rewrite']}, "
            f"hyde={report.overlap['strategy_b_hyde']}"
        )
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>JSON</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report.to_dict(), indent=2))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report_files(
    reports: list[QueryReport],
    markdown_path: Path,
    json_path: Path | None = None,
) -> None:
    """Persist the reports to disk (markdown mandatory, JSON optional)."""
    markdown_path.write_text(reports_to_markdown(reports), encoding="utf-8")
    if json_path is not None:
        json_path.write_text(
            json.dumps([r.to_dict() for r in reports], indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snippet(text: str, max_chars: int = 140) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


__all__ = [
    "DEFAULT_QUERIES",
    "StrategyRun",
    "QueryReport",
    "run_benchmark",
    "make_default_pipeline_factory",
    "reports_to_markdown",
    "write_report_files",
]
