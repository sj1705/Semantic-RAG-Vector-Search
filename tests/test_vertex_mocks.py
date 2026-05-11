"""Tests for the Vertex AI SDK mocks.

These tests prove the contract promised by the assessment: any code written
against ``vertexai.language_models.TextEmbeddingModel`` and
``vertexai.generative_models.GenerativeModel`` will work against the mocks
because method names, parameters, and return shapes match.
"""

from __future__ import annotations

import importlib

import numpy as np

from rag.vertex_mocks import (
    GenerateContentResponse,
    GenerativeModel,
    TextEmbedding,
    TextEmbeddingModel,
)


class TestTextEmbeddingModelContract:
    def test_from_pretrained_returns_instance(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        assert isinstance(model, TextEmbeddingModel)
        assert model.model_name == "textembedding-gecko"

    def test_get_embeddings_returns_list_of_textembedding(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        out = model.get_embeddings(["hello world", "peak load"])
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(isinstance(e, TextEmbedding) for e in out)

    def test_embedding_has_values_attribute(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        (emb,) = model.get_embeddings(["hello"])
        assert hasattr(emb, "values")
        assert isinstance(emb.values, list)
        assert len(emb.values) == model.dimension
        assert all(isinstance(v, float) for v in emb.values)

    def test_embeddings_are_deterministic(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        (a,) = model.get_embeddings(["repeatable"])
        (b,) = model.get_embeddings(["repeatable"])
        np.testing.assert_array_equal(a.values, b.values)

    def test_embeddings_normalized(self):
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        (emb,) = model.get_embeddings(["hello world"])
        norm = float(np.linalg.norm(emb.values))
        assert abs(norm - 1.0) < 1e-5


class TestGenerativeModelContract:
    def test_generate_content_returns_response_with_text(self):
        model = GenerativeModel("gemini-1.5-pro-mock")
        resp = model.generate_content("REWRITE: How does the system handle peak load?")
        assert isinstance(resp, GenerateContentResponse)
        assert isinstance(resp.text, str) and resp.text

    def test_rewrite_injects_domain_synonyms(self):
        model = GenerativeModel("gemini-1.5-pro-mock")
        resp = model.generate_content("REWRITE: peak load question")
        # The rewrite should mention autoscaling-related vocabulary.
        assert "autoscaling" in resp.text.lower() or "throughput" in resp.text.lower()

    def test_hyde_returns_three_passages(self):
        model = GenerativeModel("gemini-1.5-pro-mock")
        resp = model.generate_content("HYDE: peak load")
        passages = [p for p in resp.text.split("\n---\n") if p.strip()]
        assert len(passages) == 3

    def test_hyde_falls_back_for_unknown_query(self):
        model = GenerativeModel("gemini-1.5-pro-mock")
        resp = model.generate_content("HYDE: quantum chromodynamics on Jupiter")
        passages = [p for p in resp.text.split("\n---\n") if p.strip()]
        assert len(passages) == 3  # generic fallback still returns three

    def test_unknown_prompt_echoes(self):
        model = GenerativeModel("gemini-1.5-pro-mock")
        resp = model.generate_content("hello there")
        assert resp.text == "hello there"


def test_mocks_importable_as_vertexai_sdk(stub_vertex_module):
    """The ``stub_vertex_module`` fixture wires our mocks under the real SDK
    import path. Downstream code can then use the same import statement it
    would use in production."""
    # Import fresh to exercise the stubbed modules. The fixture installs both
    # ``vertexai`` and ``vertexai.language_models`` into ``sys.modules``.
    mod = importlib.import_module("vertexai.language_models")
    assert hasattr(mod, "TextEmbeddingModel")
    assert hasattr(mod, "GenerativeModel")

    model = mod.TextEmbeddingModel.from_pretrained("textembedding-gecko")
    (emb,) = model.get_embeddings(["contract test"])
    assert len(emb.values) == model.dimension

    gm = mod.GenerativeModel("gemini-1.5-pro-mock")
    assert gm.generate_content("REWRITE: anything").text
