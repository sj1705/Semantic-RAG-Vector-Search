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

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from data.corpus import CORPUS  # noqa: E402
from rag.benchmark import (  # noqa: E402
    DEFAULT_QUERIES,
    make_default_pipeline_factory,
    reports_to_markdown,
    run_benchmark,
    write_report_files,
)


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
    args = parser.parse_args()

    console = Console()
    console.print("[bold]Building pipeline (loading embedding model)…[/]")
    factory = make_default_pipeline_factory(CORPUS)

    console.print(f"[bold]Running benchmark on {len(args.queries)} queries…[/]")
    reports = run_benchmark(factory, queries=args.queries, k=3)

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
