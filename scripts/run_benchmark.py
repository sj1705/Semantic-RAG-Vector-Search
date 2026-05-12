"""CLI entrypoint for the Strategy A vs Strategy B benchmark.

Usage::

    python scripts/run_benchmark.py                 # default queries + corpus
    python scripts/run_benchmark.py -o report.md    # custom markdown path

Writes the report to ``retrieval_benchmark.md`` at the repo root by default and
also prints a compact side-by-side table to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure `src` and the repo root are importable when running this script
# directly (as opposed to `python -m ...`).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import time  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from data.corpus import CORPUS  # noqa: E402
from rag.benchmark import (  # noqa: E402
    DEFAULT_QUERIES,
    make_pipeline_factory_from_store,
    run_benchmark,
    write_report_files,
)
from rag.config import RagConfig  # noqa: E402
from rag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402


def _render_stdout_table(reports, console: Console) -> None:
    for report in reports:
        console.rule(f"[bold cyan]Query:[/] {report.query}")
        for label, run in report.runs.items():
            title = run.label
            if run.expansions_used:
                title += f" — expansions: {len(run.expansions_used)}"
            table = Table(title=title, show_lines=False)
            table.add_column("Rank", justify="right")
            table.add_column("Doc ID")
            table.add_column("Score", justify="right")
            table.add_column("Snippet")
            for hit in run.hits:
                snippet = " ".join(hit.text.split())
                if len(snippet) > 80:
                    snippet = snippet[:79] + "…"
                table.add_row(
                    str(hit.rank),
                    hit.doc_id,
                    f"{hit.score:.4f}",
                    snippet,
                )
            console.print(table)
        console.print(
            f"[dim]Overlap with Strategy A top-3: "
            f"rewrite={report.overlap['strategy_b_rewrite']}, "
            f"hyde={report.overlap['strategy_b_hyde']}[/]"
        )
        console.print()


def _load_or_build_store(
    index_dir: Path, console: Console
) -> tuple[FaissVectorStore, SentenceTransformerEmbedder]:
    """Load the saved FAISS index if it exists; otherwise build and save it.

    The index is embedded **once** per benchmark run. All 10 queries × 3
    strategies = 30 pipeline builds share this same store, instead of
    re-embedding the corpus 30 times.
    """
    config = RagConfig()

    t0 = time.time()
    embedder = SentenceTransformerEmbedder(
        model_name=config.embedding_model,
        query_prefix=config.bge_query_prefix,
    )
    console.print(
        f"[dim]Loaded embedding model in {time.time() - t0:.1f}s[/]"
    )

    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "metadata.json"

    if index_file.exists() and metadata_file.exists():
        t0 = time.time()
        store = FaissVectorStore.load(index_dir)
        console.print(
            f"[green]Loaded {store.size} vectors from {index_dir} "
            f"in {(time.time() - t0) * 1000:.0f}ms[/]"
        )
        return store, embedder

    # Fall back to building the index from the corpus, then persist it so
    # subsequent benchmark runs skip the embedding step entirely.
    console.print(
        f"[yellow]No saved index at {index_dir}; embedding corpus now…[/]"
    )
    t0 = time.time()
    docs = list(CORPUS)
    vectors = embedder.embed_documents([d.text for d in docs])
    store = FaissVectorStore(dimension=embedder.dimension, metric=config.metric)
    store.add([d.id for d in docs], [d.text for d in docs], vectors)
    console.print(
        f"[dim]Embedded {len(docs)} docs in {time.time() - t0:.1f}s[/]"
    )

    store.save(index_dir)
    console.print(f"[green]Saved index to {index_dir}[/]")
    return store, embedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG benchmark.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "retrieval_benchmark.md",
        help="Path to the generated markdown report.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to also emit the raw JSON report.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=list(DEFAULT_QUERIES),
        help="Override the benchmark queries.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "saved_index",
        help=(
            "Directory containing a saved FAISS index. If missing, the "
            "corpus is embedded once and saved there."
        ),
    )
    args = parser.parse_args()

    console = Console()
    store, embedder = _load_or_build_store(args.index_dir, console)
    factory = make_pipeline_factory_from_store(store, embedder)

    console.print(
        f"[bold]Running benchmark on {len(args.queries)} queries "
        f"(shared index of {store.size} vectors)…[/]"
    )
    t0 = time.time()
    reports = run_benchmark(factory, queries=args.queries, k=3)
    console.print(f"[dim]Benchmark took {time.time() - t0:.1f}s[/]")

    _render_stdout_table(reports, console)

    write_report_files(reports, args.output, args.json_output)
    console.print(f"[green]Wrote report to {args.output}[/]")
    if args.json_output:
        console.print(f"[green]Wrote JSON to {args.json_output}[/]")

    # Quick sanity summary so the exit code can be used in CI.
    any_change = any(
        report.overlap["strategy_b_rewrite"] != len(report.runs["strategy_a"].hits)
        or report.overlap["strategy_b_hyde"] != len(report.runs["strategy_a"].hits)
        for report in reports
    )
    if not any_change:
        console.print(
            "[yellow]Warning: Strategy B produced identical top-3 to "
            "Strategy A on every query.[/]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
