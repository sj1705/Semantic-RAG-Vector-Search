"""Tests for :class:`rag.query_expansion.QueryExpander`."""

from __future__ import annotations

import pytest

from rag.query_expansion import ExpansionResult, QueryExpander
from rag.vertex_mocks import GenerativeModel


@pytest.fixture
def model() -> GenerativeModel:
    return GenerativeModel("gemini-1.5-pro-mock")


class TestRewriteMode:
    def test_returns_single_expansion(self, model):
        exp = QueryExpander(model, mode="rewrite")
        result = exp.expand("How does the system handle peak load?")
        assert isinstance(result, ExpansionResult)
        assert result.mode == "rewrite"
        assert len(result.expansions) == 1

    def test_expansion_includes_synonyms_for_peak_load(self, model):
        exp = QueryExpander(model, mode="rewrite")
        result = exp.expand("How does the system handle peak load?")
        lowered = result.expansions[0].lower()
        # At least one autoscaling-related synonym must appear.
        assert any(
            kw in lowered for kw in ("autoscaling", "throughput", "rate limiter")
        )

    def test_is_deterministic(self, model):
        exp = QueryExpander(model, mode="rewrite")
        r1 = exp.expand("What happens during database failover?")
        r2 = exp.expand("What happens during database failover?")
        assert r1.expansions == r2.expansions


class TestHydeMode:
    def test_returns_three_expansions(self, model):
        exp = QueryExpander(model, mode="hyde")
        result = exp.expand("How does the system handle peak load?")
        assert result.mode == "hyde"
        assert len(result.expansions) == 3

    def test_each_passage_non_empty(self, model):
        exp = QueryExpander(model, mode="hyde")
        result = exp.expand("How is sensitive customer data protected?")
        assert all(p.strip() for p in result.expansions)

    def test_falls_back_when_model_returns_empty(self, model):
        """Even with an empty LLM response we always produce at least one expansion."""

        class EmptyModel:
            def generate_content(self, prompt):
                from rag.vertex_mocks import GenerateContentResponse

                return GenerateContentResponse(text="")

        exp = QueryExpander(EmptyModel(), mode="hyde")
        result = exp.expand("anything")
        assert result.expansions == ["anything"]


def test_invalid_mode_raises(model):
    with pytest.raises(ValueError):
        QueryExpander(model, mode="wat")


def test_original_query_preserved(model):
    exp = QueryExpander(model, mode="rewrite")
    result = exp.expand("any query")
    assert result.original_query == "any query"
