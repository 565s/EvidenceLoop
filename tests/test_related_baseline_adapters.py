"""No network, model weights or GPU are needed for these method-contract tests."""
import json
from types import SimpleNamespace as NS

import pytest

from evidenceloop.baselines.related.adapters import (
    GraphJudgeAdapter, IText2KGAdapter, KGGenAdapter, PiVeAdapter, RAKGAdapter,
)
from evidenceloop.baselines.related.common import ParseError, graph_rows, normalize_triples
from evidenceloop.baselines.related.prompts import pive_input


class SequenceModel:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def __call__(self, prompt):
        self.calls.append(prompt)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


def test_pive_feedback_reinjection_and_correct_early_stop():
    generator = SequenceModel(['[["Alice", "works at", "Lab"]]',
        '[["Alice", "works at", "Lab"], ["Lab", "located in", "Paris"]]', '[]'])
    verifier = SequenceModel([["['Lab', 'located in', 'Paris']"], ["Correct"], ["Correct."]])
    adapter = PiVeAdapter(generator, verifier)
    result = adapter.extract("d1", 'Alice works at Lab in "Paris".')
    assert len(result.triples) == 2 and len(result.extracted_triples) == 1
    assert result.metadata == {"revisions": 1, "stop_reason": "correct", "format_rejected": []}
    assert "Previous Semantic Graph" in generator.calls[1]
    assert "located in" in generator.calls[1] and "Paris" in generator.calls[1]
    assert 'Alice works at Lab in "Paris".' in generator.calls[1]
    assert " <S> " in verifier.calls[0][0] and '"' not in verifier.calls[0][0]
    assert "located in" in verifier.calls[1][0]
    assert [s["stage"] for s in result.stages] == ["initial_generation", "verifier_1", "revision_1", "verifier_2"]
    second = adapter.extract("d2", "No facts here.")
    assert second.triples == [] and "Alice" not in verifier.calls[2][0]
    assert [s["stage"] for s in second.stages] == ["initial_generation", "verifier_1"]


def test_pive_three_revisions_exclude_initial_generation():
    generator = SequenceModel(['[]'] * 4)
    verifier = SequenceModel([["Missing fact"]] * 3)
    result = PiVeAdapter(generator, verifier).extract("d", "Text.")
    assert len(generator.calls) == 4 and len(verifier.calls) == 3
    assert result.metadata["revisions"] == 3
    assert result.metadata["stop_reason"] == "iteration_limit"


def test_pive_python_graph_serialization():
    triples = [{"subject": "A", "predicate": "has", "object": "B"}]
    assert pive_input('"A" has B.', triples) == "'A' has B. <S> [['A', 'has', 'B']]"


def test_graphjudge_denoise_extract_filter_and_state_isolation():
    generator = SequenceModel(['["Alice", "Lab", "Paris"]', "Alice works at Lab.",
        '[["Alice", "works at", "Lab"], ["Lab", "located in", "Paris"]]', '[]'])
    verifier = SequenceModel([["Yes, this is true.", "No, this is not true."]])
    adapter = GraphJudgeAdapter(generator, verifier)
    result = adapter.extract("d", "Alice works at Lab. Unrelated original context.")
    assert len(result.triples) == 1 and len(result.extracted_triples) == 2
    assert "Alice works at Lab." in generator.calls[2]
    assert "Unrelated original context" in verifier.calls[0][0]
    assert "### Instruction:" in verifier.calls[0][0]
    assert [r["keep"] for r in result.metadata["judgments"]] == [True, False]
    assert adapter.extract("next", "No entities.").triples == []
    assert len(adapter.stages) == 1 and len(verifier.calls) == 1


@pytest.mark.parametrize("answer", ["Maybe", "", "Yes and false"])
def test_graphjudge_ambiguous_judgments_fail(answer):
    adapter = GraphJudgeAdapter(SequenceModel(['["A", "B"]', "A has B.", '[["A", "has", "B"]]']), SequenceModel([[answer]]))
    with pytest.raises(ParseError):
        adapter.extract("d", "A has B.")


class ToyEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_rakg_retrieval_disambiguation_attributes_and_empty_initial_graph():
    entity = {"entity1": {"name": "Alpha", "type": "company", "description": "Alpha org"}}
    generator = SequenceModel([json.dumps(entity), json.dumps(entity), '{"State": "none"}',
        '{"result": true}', json.dumps({"central_entity": {"name": "Alpha", "attributes": [{"key": "year", "value": 2000}],
            "relationships": [{"relation": "located in", "target_name": ["Paris", "France"]}]}}), '{"State": "none"}'])
    embeddings = ToyEmbeddings()
    adapter = RAKGAdapter(generator, embeddings, top_k=1)
    result = adapter.extract("d", "Alpha was founded. Alpha moved. Other text.")
    assert result.metadata["pre_entity_count"] == 2
    assert result.metadata["merged_entity_count"] == 1
    assert result.metadata["retrieval"][0]["retrieved_indices"] == [0]
    assert result.metadata["retrieval"][0]["sentence_indices"] == [0, 1]
    assert result.metadata["initial_graph"] == [] and result.metadata["graph_structure_candidates"] == 0
    assert "Alpha moved." in generator.calls[4] and "Other text." not in generator.calls[4]
    assert normalize_triples(result.triples)[0] == [
        {"subject": "Alpha", "predicate": "located in", "object": "Paris"},
        {"subject": "Alpha", "predicate": "located in", "object": "France"},
        {"subject": "Alpha", "predicate": "year", "object": "2000"}]
    assert adapter.extract("next", "Nothing.").triples == []


def test_rakg_threshold_prevents_unrelated_disambiguation():
    responses = [json.dumps({"e": {"name": name, "type": "type", "description": name}}) for name in ("Alpha", "Beta")]
    responses += [json.dumps({"central_entity": {"name": name, "relationships": []}}) for name in ("Alpha", "Beta")]
    result = RAKGAdapter(SequenceModel(responses), ToyEmbeddings()).extract("d", "Alpha. Beta.")
    assert result.metadata["merged_entity_count"] == 2
    assert not any(s["stage"].startswith("disambiguation") for s in result.stages)


def test_itext2kg_distiller_merger_contract_and_fresh_graph():
    builders, calls = [], []

    class Distiller:
        def __init__(self, **kwargs):
            pass

        def distill(self, **kwargs):
            calls.append(kwargs)
            return {"facts": ["Alice leads Lab.", "A. leads Laboratory."]}

    class Builder:
        def __init__(self, **kwargs):
            builders.append(self)
            assert kwargs["sleep_time"] == 0

        def build_graph(self, **kwargs):
            assert kwargs["existing_knowledge_graph"] is None
            assert kwargs["ent_threshold"] == kwargs["rel_threshold"] == 0.7
            assert len(kwargs["sections"]) == 2
            # Mock the package's semantic matcher and return its representative entities.
            a, b = NS(name="alice"), NS(name="lab")
            return NS(entities=[a, b], relationships=[NS(startEntity=a, name="leads", endEntity=b)])

    adapter = IText2KGAdapter(None, ToyEmbeddings(), distiller_factory=Distiller, graph_factory=Builder)
    assert adapter.extract("d1", "Source one.").triples == [["alice", "leads", "lab"]]
    assert adapter.extract("d2", "Source two.").triples == [["alice", "leads", "lab"]]
    assert len(builders) == 2 and builders[0] is not builders[1]
    assert calls[0]["documents"] == ["Source one."] and calls[1]["documents"] == ["Source two."]
    assert "explicit facts" in calls[0]["IE_query"]


def test_kggen_official_extraction_then_entity_and_relation_clustering():
    engines = []

    class Engine:
        def __init__(self):
            engines.append(self)
            self.lm = NS(history=[{"api_key": "sensitive", "messages": ["raw prompt"], "outputs": ["raw response"]}])

        def generate(self, **kwargs):
            assert kwargs == {"input_data": "A has B.", "chunk_size": None, "cluster": False}
            return NS(relations={("A", "has", "B"), ("Alias A", "owns", "B")})

        def cluster(self, graph, context):
            assert len(graph.relations) == 2 and context == "A has B."
            return NS(relations={("A", "has", "B")}, entity_clusters={"A": {"Alias A"}}, edge_clusters={"has": {"owns"}})

    adapter = KGGenAdapter(Engine)
    result = adapter.extract("d1", "A has B.")
    assert result.triples == [("A", "has", "B")] and len(result.extracted_triples) == 2
    assert result.metadata["entity_clusters"] == {"A": ["Alias A"]}
    assert result.metadata["edge_clusters"] == {"has": ["owns"]}
    assert "sensitive" not in str(result.stages)
    adapter.extract("d2", "A has B.")
    assert len(engines) == 2


@pytest.mark.parametrize("response", ["[]", '{"triples": []}', "```json\n[]\n```", "Semantic Graph: []", "[['A', 'has', 'B']]"])
def test_supported_graph_serializations(response):
    assert isinstance(graph_rows(response), list)


@pytest.mark.parametrize("response", ["", "nonsense", "{}", '{"triples": []', '[["A", "has", "B"]', '{"other": []}'])
def test_parse_failures_are_never_successful_empty_graphs(response):
    with pytest.raises(ParseError):
        graph_rows(response)


def test_format_only_normalization_and_deterministic_dedupe():
    result, rejected = normalize_triples([[" A ", " has  property ", " B "], ["A", "has property", "B"],
        ["A", "has\nproperty", "B"], ["", "has", "B"], ["A", "Has property", "B"]])
    assert len(result) == 2 and len(rejected) == 2
    assert [t["predicate"] for t in result] == ["has property", "Has property"]
