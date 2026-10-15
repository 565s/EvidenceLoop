"""Independent method implementations, with injectable runtimes for offline tests."""

from __future__ import annotations

import math
import re

from .common import Extraction, ParseError, graph_rows, normalize_triples, parse_structure
from .prompts import (
    denoise_prompt,
    entities_prompt,
    entity_graph_prompt,
    extraction_prompt,
    graphjudge_prompt,
    pive_prompt,
    rakg_prompts,
    revision_prompt,
)
from .runtime import ChatGenerator, Embeddings, LocalVerifier, json_safe, redact, traced_call


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("embedding dimensions are invalid")
    denominator = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(x) ** 2 for x in b))
    return sum(float(x) * float(y) for x, y in zip(a, b)) / denominator if denominator else 0.0


def graph_list(triples):
    return [[t["subject"], t["predicate"], t["object"]] for t in triples]


class PiVeAdapter:
    def __init__(self, generator, verifier, *, rounds=3, instruction=None):
        self.generator, self.verifier, self.rounds, self.instruction = (
            generator,
            verifier,
            rounds,
            instruction,
        )
        self.stages = []

    def extract(self, document_id, text):
        self.stages = []
        response = traced_call(
            self.stages, "initial_generation", self.generator, extraction_prompt(text)
        )
        triples, rejected = normalize_triples(graph_rows(response))
        initial = list(triples)
        revisions, stop_reason = 0, "iteration_limit"
        for iteration in range(1, self.rounds + 1):
            prompt = pive_prompt(
                text, triples, **({"instruction": self.instruction} if self.instruction else {})
            )
            responses = traced_call(self.stages, f"verifier_{iteration}", self.verifier, [prompt])
            if (
                not isinstance(responses, list)
                or len(responses) != 1
                or not isinstance(responses[0], str)
            ):
                raise ParseError("PiVe verifier must return one textual feedback")
            feedback = responses[0].strip()
            if feedback.rstrip(".").casefold() == "correct":
                stop_reason = "correct"
                break
            if not feedback:
                raise ParseError("PiVe verifier returned empty feedback")
            response = traced_call(
                self.stages,
                f"revision_{iteration}",
                self.generator,
                revision_prompt(text, graph_list(triples), feedback),
            )
            triples, invalid = normalize_triples(graph_rows(response))
            rejected.extend(invalid)
            revisions += 1
        return Extraction(
            triples,
            self.stages,
            initial,
            {"revisions": revisions, "stop_reason": stop_reason, "format_rejected": rejected},
        )


def judgment(response: str) -> bool:
    text = response.strip().casefold()
    if re.match(r"^(no\b|false\b)", text):
        return False
    if re.match(r"^(yes\b|true\b)", text) and "not true" not in text and "false" not in text:
        return True
    raise ParseError(f"GraphJudge returned an unrecognized judgment: {response!r}")


class GraphJudgeAdapter:
    def __init__(self, generator, verifier, *, batch_size=4):
        self.generator, self.verifier, self.batch_size = generator, verifier, batch_size
        self.stages = []

    def extract(self, document_id, text):
        self.stages = []
        entities = parse_structure(
            traced_call(self.stages, "entity_extraction", self.generator, entities_prompt(text))
        )
        if not isinstance(entities, list) or any(
            not isinstance(e, str) or not e.strip() for e in entities
        ):
            raise ParseError("GraphJudge entities must be a JSON list of names")
        entities = list(dict.fromkeys(entities))
        if not entities:
            return Extraction([], self.stages, [], {"entities": []})
        denoised = traced_call(
            self.stages, "entity_centric_denoising", self.generator, denoise_prompt(text, entities)
        )
        if not isinstance(denoised, str) or not denoised.strip():
            raise ParseError("GraphJudge denoising returned empty text")
        candidates, rejected = normalize_triples(
            graph_rows(
                traced_call(
                    self.stages,
                    "relation_extraction",
                    self.generator,
                    entity_graph_prompt(denoised, entities),
                )
            )
        )
        final, decisions = [], []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            # The published context adapters judge against the original source document.
            prompts = [graphjudge_prompt(text, t) for t in batch]
            responses = traced_call(
                self.stages,
                f"graph_judgment_{start // self.batch_size + 1}",
                self.verifier,
                prompts,
            )
            if not isinstance(responses, list) or len(responses) != len(batch):
                raise ParseError("GraphJudge verifier response count mismatch")
            for triple, response in zip(batch, responses):
                keep = judgment(response)
                decisions.append({"triple": triple, "keep": keep, "response": response})
                if keep:
                    final.append(triple)
        return Extraction(
            final,
            self.stages,
            candidates,
            {
                "entities": entities,
                "denoised_text": denoised,
                "judgments": decisions,
                "format_rejected": rejected,
            },
        )


class RAKGAdapter:
    def __init__(self, generator, embeddings, *, top_k=5, threshold=0.6):
        self.generator, self.embeddings, self.top_k, self.threshold = (
            generator,
            embeddings,
            top_k,
            threshold,
        )
        self.prompts, self.stages = rakg_prompts(), []

    def extract(self, document_id, text):
        self.stages = []
        # Keep duplicate sentence occurrences distinct instead of upstream's text-keyed IDs.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        vectors = self.embeddings.embed_documents(sentences)
        if len(vectors) != len(sentences):
            raise ValueError("embedding result count mismatch")
        entities = []
        for index, sentence in enumerate(sentences):
            response = traced_call(
                self.stages,
                f"sentence_entities_{index}",
                self.generator,
                self.prompts["text2entity_en"].format(text=sentence),
            )
            value = parse_structure(response)
            if not isinstance(value, dict):
                raise ParseError("RAKG entity response must be an object")
            if "State" in value:
                continue
            for entity in value.values():
                if not isinstance(entity, dict) or any(
                    not isinstance(entity.get(k), str) for k in ("name", "type", "description")
                ):
                    raise ParseError("RAKG entities require name, type and description")
                if not entity["name"].strip():
                    raise ParseError("RAKG entity name is empty")
                entities.append({**entity, "sentence_indices": [index]})
        count = len(entities)
        parent = list(range(count))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        entity_vectors = (
            self.embeddings.embed_documents(
                [f"{e['name']} {e['type']} {e['description']}" for e in entities]
            )
            if entities
            else []
        )
        if len(entity_vectors) != count:
            raise ValueError("entity embedding result count mismatch")
        for i in range(count):
            for j in range(i + 1, count):
                if cosine(entity_vectors[i], entity_vectors[j]) <= self.threshold:
                    continue
                prompt = self.prompts["judge_sim_entity_en"].format(
                    entity1=entities[i], entity2=entities[j]
                )
                value = parse_structure(
                    traced_call(self.stages, f"disambiguation_{i}_{j}", self.generator, prompt)
                )
                if not isinstance(value, dict) or not isinstance(value.get("result"), bool):
                    raise ParseError("RAKG disambiguation requires a boolean result")
                if value["result"]:
                    left, right = find(i), find(j)
                    parent[max(left, right)] = min(left, right)
        merged = {}
        for i, entity in enumerate(entities):
            root = find(i)
            if root not in merged:
                merged[root] = {**entity, "sentence_indices": [], "descriptions": []}
            merged[root]["sentence_indices"].extend(entity["sentence_indices"])
            merged[root]["descriptions"].append(entity["description"])
        triples, subgraphs, retrieval = [], [], []
        for root, entity in merged.items():
            query = self.embeddings.embed_query(entity["name"])
            nearest = sorted(range(len(sentences)), key=lambda i: (-cosine(query, vectors[i]), i))[
                : self.top_k
            ]
            indices = list(dict.fromkeys(entity["sentence_indices"] + nearest))
            context = ", ".join(sentences[i] for i in indices)
            prompt = self.prompts["extract_entiry_centric_kg_en_v2"].format(
                text=context, target_entity=entity["name"], related_kg="none"
            )
            graph = parse_structure(
                traced_call(self.stages, f"relation_network_{root}", self.generator, prompt)
            )
            central = graph.get("central_entity") if isinstance(graph, dict) else None
            if not isinstance(central, dict) or not isinstance(central.get("name"), str):
                raise ParseError("RAKG subgraph requires central_entity.name")
            if any(
                not isinstance(central.get(k, []), list) for k in ("relationships", "attributes")
            ):
                raise ParseError("RAKG relationships and attributes must be lists")
            for relation in central.get("relationships", []):
                if not isinstance(relation, dict):
                    raise ParseError("RAKG relationship must be an object")
                targets = relation.get("target_name")
                for target in targets if isinstance(targets, list) else [targets]:
                    triples.append([central["name"], relation.get("relation"), target])
            for attribute in central.get("attributes", []):
                if not isinstance(attribute, dict) or "value" not in attribute:
                    raise ParseError("RAKG attribute requires key and value")
                triples.append([central["name"], attribute.get("key"), str(attribute["value"])])
            subgraphs.append(graph)
            retrieval.append(
                {
                    "entity": entity["name"],
                    "sentence_indices": indices,
                    "retrieved_indices": nearest,
                }
            )
        return Extraction(
            triples,
            self.stages,
            metadata={
                "sentence_count": len(sentences),
                "pre_entity_count": count,
                "merged_entity_count": len(merged),
                "retrieval": retrieval,
                "subgraphs": subgraphs,
                "initial_graph": [],
                "graph_structure_candidates": 0,
                "attribute_export": "key/value properties are exported as SPO facts; entity types are not exported",
            },
        )


def attach_langchain_trace(llm, adapter):
    import time

    from langchain_core.callbacks import BaseCallbackHandler

    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.pending = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self.pending[run_id] = (
                time.perf_counter(),
                [[str(m.content) for m in group] for group in messages],
            )

        def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
            self.pending[run_id] = (time.perf_counter(), prompts)

        def on_llm_end(self, response, *, run_id, **kwargs):
            started, prompts = self.pending.pop(run_id, (time.perf_counter(), []))
            generations = [g.text for batch in response.generations for g in batch]
            adapter.stages.append(
                redact(
                    {
                        "stage": f"llm_call_{len(adapter.stages)}",
                        "prompt": prompts,
                        "response": generations,
                        "usage": json_safe(response.llm_output),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            )
            for batch in response.generations:
                for generation in batch:
                    try:
                        if not isinstance(parse_structure(generation.text), dict):
                            raise ParseError("iText2KG response must be a JSON object")
                    except ParseError as exc:
                        adapter.parse_error = str(exc)
                    info = getattr(generation, "generation_info", None) or {}
                    if info.get("finish_reason") == "length":
                        adapter.truncated = True

        def on_llm_error(self, error, *, run_id, **kwargs):
            started, prompts = self.pending.pop(run_id, (time.perf_counter(), []))
            adapter.stages.append(
                redact(
                    {
                        "stage": f"llm_call_{len(adapter.stages)}",
                        "prompt": prompts,
                        "error": str(error),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            )

    llm.callbacks = [Recorder()]


class IText2KGAdapter:
    def __init__(
        self,
        llm,
        embeddings,
        *,
        distiller_factory=None,
        graph_factory=None,
        entity_threshold=0.7,
        relation_threshold=0.7,
    ):
        self.llm, self.embeddings = llm, embeddings
        self.distiller_factory, self.graph_factory = distiller_factory, graph_factory
        self.entity_threshold, self.relation_threshold = entity_threshold, relation_threshold
        self.stages, self.truncated, self.parse_error = [], False, ""
        if distiller_factory is None or graph_factory is None:
            from itext2kg import DocumentsDistiller, iText2KG

            self.distiller_factory, self.graph_factory = DocumentsDistiller, iText2KG
            attach_langchain_trace(llm, self)

    def extract(self, document_id, text):
        from pydantic import BaseModel, Field

        class Facts(BaseModel):
            facts: list[str] = Field(
                description="Complete factual sentences preserving the document's entities and relationships."
            )

        self.stages, self.truncated, self.parse_error = [], False, ""
        distiller = self.distiller_factory(llm_model=self.llm)
        query = "Organize all explicit facts into complete factual sentences. Preserve proper names, domain terminology and relationships."
        distilled = traced_call(
            self.stages,
            "document_distillation",
            lambda _: distiller.distill(
                documents=[text], output_data_structure=Facts, IE_query=query
            ),
            text,
        )
        if self.truncated:
            raise ParseError("iText2KG generation reached max_tokens")
        if self.parse_error:
            raise ParseError(self.parse_error)
        if (
            not isinstance(distilled, dict)
            or not isinstance(distilled.get("facts"), list)
            or any(not isinstance(s, str) or not s.strip() for s in distilled["facts"])
        ):
            raise ParseError("iText2KG distillation requires a facts list of sentences")
        if not distilled["facts"]:
            return Extraction([], self.stages, metadata={"distilled": distilled})
        # A fresh constructor and explicit None prevent the incremental graph leaking across documents.
        builder = self.graph_factory(
            llm_model=self.llm, embeddings_model=self.embeddings, sleep_time=0
        )
        graph = traced_call(
            self.stages,
            "incremental_graph",
            lambda _: builder.build_graph(
                sections=distilled["facts"],
                existing_knowledge_graph=None,
                ent_threshold=self.entity_threshold,
                rel_threshold=self.relation_threshold,
            ),
            distilled["facts"],
        )
        if self.truncated:
            raise ParseError("iText2KG generation reached max_tokens")
        if self.parse_error:
            raise ParseError(self.parse_error)
        if not hasattr(graph, "relationships"):
            raise ParseError("iText2KG did not return a knowledge graph")
        triples = [[r.startEntity.name, r.name, r.endEntity.name] for r in graph.relationships]
        return Extraction(
            triples,
            self.stages,
            metadata={"distilled": distilled, "entities": [e.name for e in graph.entities]},
        )


def kggen_generate(engine, text):
    generate = engine.generate
    function = getattr(generate, "__func__", None)
    if function is not None and function.__module__ == "kg_gen.kg_gen":
        # 0.2.0 constructs Literal[()] before relation extraction when NER returns [].
        # Clone the function namespace, so this empty-list guard changes no global state.
        from types import FunctionType

        namespace = dict(function.__globals__)
        original = namespace["get_relations"]

        def get_relations(dspy, input_data, entities, **kwargs):
            return original(dspy, input_data, entities, **kwargs) if entities else []

        namespace["get_relations"] = get_relations
        guarded = FunctionType(
            function.__code__,
            namespace,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        guarded.__kwdefaults__ = function.__kwdefaults__
        generate = guarded.__get__(engine, type(engine))
    return generate(input_data=text, chunk_size=None, cluster=False)


class KGGenAdapter:
    def __init__(self, engine_factory):
        self.engine_factory, self.stages = engine_factory, []

    def extract(self, document_id, text):
        self.stages = []
        engine = self.engine_factory()
        try:
            graph = traced_call(
                self.stages,
                "entity_relation_extraction",
                lambda _: kggen_generate(engine, text),
                text,
            )
            if not hasattr(graph, "relations"):
                raise ParseError("KGGen did not return a graph")
            initial = sorted(graph.relations)
            # Explicitly run both entity and edge clustering; the upstream default is disabled.
            final = traced_call(
                self.stages,
                "entity_relation_clustering",
                lambda _: engine.cluster(graph=graph, context=text),
                json_safe(graph),
            )
            if not hasattr(final, "relations"):
                raise ParseError("KGGen clustering did not return a graph")
        finally:
            for index, call in enumerate(getattr(getattr(engine, "lm", None), "history", [])):
                self.stages.append(
                    redact(
                        {
                            "stage": f"dspy_call_{index}",
                            "response": json_safe(call),
                            "elapsed_seconds": 0.0,
                        }
                    )
                )

        def truncated(value):
            if isinstance(value, dict):
                return value.get("finish_reason") == "length" or any(
                    truncated(v) for v in value.values()
                )
            return isinstance(value, list) and any(truncated(v) for v in value)

        if any(truncated(s.get("response")) for s in self.stages):
            raise ParseError("KGGen generation reached max_tokens")
        return Extraction(
            sorted(final.relations),
            self.stages,
            initial,
            {
                "cluster": True,
                "chunk_size": None,
                "entity_clusters": json_safe(final.entity_clusters),
                "edge_clusters": json_safe(final.edge_clusters),
            },
        )


def create_adapter(runtime):
    method = runtime["method"]
    if method == "kggen":
        import os

        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import dspy
        from kg_gen import KGGen

        def factory():
            engine = KGGen(
                model="openai/" + runtime["model"],
                temperature=runtime["temperature"],
                api_key=os.environ["SIFLOW_API_KEY"],
                api_base=runtime["base_url"],
            )
            engine.lm = dspy.LM(
                model="openai/" + runtime["model"],
                temperature=runtime["temperature"],
                api_key=os.environ["SIFLOW_API_KEY"],
                api_base=runtime["base_url"],
                max_tokens=runtime["max_tokens"],
                timeout=runtime["request_timeout"],
                num_retries=0,
                cache=False,
                extra_body={"enable_thinking": runtime["enable_thinking"]},
            )
            dspy.configure(lm=engine.lm)
            return engine

        return KGGenAdapter(factory)
    if method in ("rakg", "itext2kg"):
        embeddings = Embeddings(runtime["models"]["embedding"]["path"], runtime["device"])
        if method == "rakg":
            return RAKGAdapter(
                ChatGenerator(runtime),
                embeddings,
                top_k=runtime["top_k"],
                threshold=runtime["disambiguation_threshold"],
            )
        from .runtime import make_chat

        return IText2KGAdapter(
            make_chat(runtime),
            embeddings,
            entity_threshold=runtime["entity_threshold"],
            relation_threshold=runtime["relation_threshold"],
        )
    generator, verifier = ChatGenerator(runtime), LocalVerifier(runtime)
    if method == "pive":
        return PiVeAdapter(
            generator,
            verifier,
            rounds=runtime["pive_rounds"],
            instruction=runtime["pive_instruction"],
        )
    if method == "graphjudge":
        return GraphJudgeAdapter(generator, verifier, batch_size=runtime["verifier_batch_size"])
    raise ValueError(f"unknown baseline method: {method}")
