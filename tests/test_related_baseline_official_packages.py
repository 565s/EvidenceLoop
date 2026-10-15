"""Optional package integration: run in each pinned backend environment.

Only the LLM and embeddings are mocked; official semantic processing executes.
"""
import importlib.metadata
import importlib.util
import json
from types import SimpleNamespace as NS

import pytest


def require_package(module, distribution, expected):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"run this test in the {distribution}=={expected} backend environment")
    assert importlib.metadata.version(distribution) == expected


def test_official_itext2kg_semantic_entity_and_relation_merging():
    require_package("itext2kg", "itext2kg", "0.0.7")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from evidenceloop.baselines.related.adapters import IText2KGAdapter

    def entities(a, b):
        return {"entities": [{"name": a, "label": "Person"}, {"name": b, "label": "Organization"}]}

    def relation(name):
        return {"relationships": [{"startNode": {"name": "Alice", "label": "Person"}, "name": name,
            "endNode": {"name": "Lab", "label": "Organization"}}]}

    responses = [{"facts": ["Alice leads Lab.", "Alicia manages Laboratory."]}, entities("Alice", "Lab"),
        relation("leads"), entities("Alicia", "Laboratory"), relation("manages")]
    llm = FakeListChatModel(responses=[json.dumps(row) for row in responses])

    class Embeddings:
        def embed_query(self, text):
            text = text.casefold()
            if text in ("alice", "alicia"):
                return [1., 0., 0., 0.]
            if text in ("lab", "laboratory"):
                return [0., 1., 0., 0.]
            if text == "person":
                return [0., 0., 1., 0.]
            if text == "organization":
                return [0., 0., 0., 1.]
            return [1., 1., 0., 0.]

        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

    adapter = IText2KGAdapter(llm, Embeddings())
    result = adapter.extract("d", "Alice leads Lab. Alicia manages Laboratory.")
    assert result.triples == [["alice", "leads", "lab"]]
    assert len(result.metadata["entities"]) == 2
    assert len([s for s in result.stages if s["stage"].startswith("llm_call")]) == 5


def test_official_itext2kg_rejects_partial_json_that_upstream_parser_repairs():
    require_package("itext2kg", "itext2kg", "0.0.7")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from evidenceloop.baselines.related.adapters import IText2KGAdapter
    from evidenceloop.baselines.related.common import ParseError
    llm = FakeListChatModel(responses=['{"facts": []'])
    adapter = IText2KGAdapter(llm, None)
    with pytest.raises(ParseError):
        adapter.extract("d", "No facts here.")


def test_official_kggen_clustering_validates_and_selects_representatives(monkeypatch):
    require_package("kg_gen", "kg-gen", "0.2.0")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import dspy
    from kg_gen import KGGen
    import kg_gen.steps._3_cluster_graph as clustering
    from evidenceloop.baselines.related.adapters import KGGenAdapter

    calls = []

    def predictor(signature):
        name = signature.__name__

        def call(**kwargs):
            calls.append((name, kwargs))
            if name == "TextEntities":
                return NS(entities=["Alice", "Alicia", "Lab"])
            if name == "ExtractTextRelations":
                return NS(relations=[NS(subject="Alice", predicate="leads", object="Lab"),
                    NS(subject="Alicia", predicate="manages", object="Lab")])
            if name == "ExtractCluster":
                items = set(kwargs["items"])
                cluster = items & {"Alice", "Alicia"} or items & {"leads", "manages"}
                return NS(cluster=list(cluster) if len(cluster) > 1 else [])
            if name == "ValidateCluster":
                return NS(validated_items=list(kwargs["cluster"]))
            if name == "CheckExistingClusters":
                return NS(cluster_reps_that_items_belong_to=[None] * len(kwargs["items"]))
            raise AssertionError(name)

        return call

    monkeypatch.setattr(dspy, "Predict", predictor)
    monkeypatch.setattr(dspy, "ChainOfThought", predictor)

    def choose_rep(**kwargs):
        calls.append(("ChooseRepresentative", kwargs))
        return NS(representative="Alice" if "Alice" in kwargs["cluster"] else "leads")

    monkeypatch.setattr(clustering, "choose_rep", choose_rep)
    adapter = KGGenAdapter(lambda: KGGen(model="openai/mock", api_key="mock", api_base="https://example.invalid/v1"))
    result = adapter.extract("d", "Alice leads Lab. Alicia manages Lab.")
    assert result.triples == [("Alice", "leads", "Lab")]
    assert result.metadata["entity_clusters"]["Alice"] == ["Alice", "Alicia"]
    assert result.metadata["edge_clusters"]["leads"] == ["leads", "manages"]
    assert sum(name == "ChooseRepresentative" for name, _ in calls) == 2
    assert sum(name == "ValidateCluster" for name, _ in calls) == 2


def test_official_kggen_empty_entities_are_a_successful_empty_graph(monkeypatch):
    require_package("kg_gen", "kg-gen", "0.2.0")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import dspy
    from kg_gen import KGGen
    import kg_gen.kg_gen as implementation
    from evidenceloop.baselines.related.adapters import KGGenAdapter
    original = implementation.get_relations
    calls = []

    def predictor(signature):
        def call(**kwargs):
            calls.append(signature.__name__)
            assert signature.__name__ == "TextEntities"
            return NS(entities=[])
        return call

    monkeypatch.setattr(dspy, "Predict", predictor)
    result = KGGenAdapter(lambda: KGGen(model="openai/mock", api_key="mock")).extract("d", "No facts.")
    assert result.triples == [] and result.metadata["entity_clusters"] == {}
    assert calls == ["TextEntities"]
    assert implementation.get_relations is original
