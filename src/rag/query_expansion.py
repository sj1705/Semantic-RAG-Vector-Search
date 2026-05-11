"""Query expansion — Strategy B's pre-retrieval step.

Given a user query, the :class:`QueryExpander` asks a (mocked) generative model to
produce either a single domain-enriched **rewrite** or three **HyDE**-style
hypothetical answer passages.

* ``mode="rewrite"`` returns a list of one string.
* ``mode="hyde"``    returns a list of up to three strings.

The caller then embeds each string, averages the vectors (renormalized), and
searches the index once. Averaging multiple HyDE embeddings is a well-known
technique to broaden recall for ambiguous queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GenerativeLike(Protocol):
    """Duck-typed interface the expander depends on.

    Both the mock in :mod:`rag.vertex_mocks` and the real
    ``vertexai.generative_models.GenerativeModel`` satisfy this protocol.
    """

    def generate_content(self, prompt: str): ...  # noqa: D401 - return is any response w/ .text


@dataclass(frozen=True)
class ExpansionResult:
    """Outcome of a single :meth:`QueryExpander.expand` call."""

    mode: str
    original_query: str
    expansions: list[str]


class QueryExpander:
    """Pre-retrieval query rewriter backed by a generative model.

    Parameters
    ----------
    model:
        Any object with a ``generate_content(prompt) -> response`` method where
        the response has a ``.text`` attribute. In tests and in the benchmark
        this is :class:`rag.vertex_mocks.GenerativeModel`; in production it is
        the real Vertex AI ``GenerativeModel``.
    mode:
        ``"rewrite"`` for a single synonym-enriched rewrite, or ``"hyde"`` for
        three hypothetical answer passages.
    """

    _HYDE_SPLIT = "\n---\n"

    def __init__(self, model: GenerativeLike, mode: str = "rewrite") -> None:
        if mode not in {"rewrite", "hyde"}:
            raise ValueError(f"mode must be 'rewrite' or 'hyde', got {mode!r}")
        self._model = model
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def expand(self, query: str) -> ExpansionResult:
        prompt = self._prompt_for(query)
        response = self._model.generate_content(prompt)
        text = (response.text or "").strip()
        if self._mode == "hyde":
            passages = [p.strip() for p in text.split(self._HYDE_SPLIT) if p.strip()]
            # Never return zero passages; fall back to the original query so the
            # downstream pipeline never has to branch on empty input.
            expansions = passages or [query]
        else:
            expansions = [text or query]
        return ExpansionResult(
            mode=self._mode, original_query=query, expansions=expansions
        )

    # -- internal -----------------------------------------------------------

    def _prompt_for(self, query: str) -> str:
        if self._mode == "rewrite":
            return f"REWRITE: {query}"
        return f"HYDE: {query}"


__all__ = ["QueryExpander", "ExpansionResult", "GenerativeLike"]
