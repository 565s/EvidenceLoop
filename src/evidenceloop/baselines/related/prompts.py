"""Prompt formats, including the exact separators used by the published verifiers."""

from __future__ import annotations

import json
from pathlib import Path

from .common import PIVE_INSTRUCTION

PROMPT_VERSION = "related-baselines-v1"
GRAPH_EXAMPLE = """Text: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.
Semantic Graph: [["Shotgate Thickets", "instance of", "Nature reserve"], ["Shotgate Thickets", "country", "United Kingdom"], ["Shotgate Thickets", "operator", "Essex Wildlife Trust"]"""


def extraction_prompt(text: str) -> str:
    return f"Transform the text into a semantic graph (a list of subject, predicate, object triples).\nExample:\n{GRAPH_EXAMPLE}\nRefer to the example and here is the text:\nText: {text}\nSemantic Graph:"


def revision_prompt(text: str, graph, feedback: str) -> str:
    return (
        "Transform the text into a semantic graph and also add the given triples to the generated semantic graph.\n"
        f"Example:\n{GRAPH_EXAMPLE}\nText: {text}\nPrevious Semantic Graph: {json.dumps(graph, ensure_ascii=False)}\n"
        f"Given triples (verifier feedback): {feedback}\nSemantic Graph:"
    )


def pive_input(text: str, triples) -> str:
    graph = [[t["subject"], t["predicate"], t["object"]] for t in triples]
    # PiVe's notebook serializes the graph as a Python list and replaces double quotes.
    return text.replace('"', "'") + " <S> " + str(graph).replace('"', "'")


def pive_prompt(text: str, triples, instruction=PIVE_INSTRUCTION) -> str:
    return instruction + " " + pive_input(text, triples)


def graphjudge_prompt(text: str, triple) -> str:
    # Context adapters were trained with the source context inside the instruction.
    # GraphJudge paper, Appendix I / Figure 8, plus the official Alpaca wrapper.
    instruction = f"According to the original document: {text}\nIs this true: {triple['subject']} {triple['predicate']} {triple['object']}?"
    return (
        "\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"### Instruction:\n{instruction}\n### Response:\n"
    )


def entities_prompt(text: str) -> str:
    return f"Transform the text into a list of entities. Return only a JSON list of entity names.\nText: {text}\nList of entities:"


def denoise_prompt(text: str, entities) -> str:
    return (
        "Denoise the raw text with the given entities, which means remove unrelated text and make it more formatted. "
        "Retain the facts about the given entities. Return the denoised text only.\n"
        f"Raw text: {text}\nEntities: {json.dumps(entities, ensure_ascii=False)}\nDenoised text:"
    )


def entity_graph_prompt(text: str, entities) -> str:
    return (
        "Transform the text into a semantic graph (a list of triples) with the given text and entities. "
        "Find relations between the given entities with the given text.\n"
        f"Example:\n{GRAPH_EXAMPLE}\nText: {text}\nEntity List: {json.dumps(entities, ensure_ascii=False)}\nSemantic graph:"
    )


def rakg_prompts() -> dict:
    return json.loads(
        (Path(__file__).parent / "resources/rakg_prompts.json").read_text(encoding="utf-8")
    )
