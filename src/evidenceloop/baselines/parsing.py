"""Historical direct-baseline JSON parsing, separate from agent parsing."""

import json
import re
from typing import Dict, Mapping

from evidenceloop.types import Triple
from evidenceloop.validation import clean_predicate, predicate_validation_error


def _extract_json_obj(text: str) -> Dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    raw = match.group(0)
    raw = re.sub(r'""([^"]+)":', r'"\\"\1\\":', raw)
    raw = re.sub(r'\[""([^"]+)":', r'["\\"\1\\":', raw)
    raw = re.sub(r"[\x00-\x1F\x7F]", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_gpt_response(response: str, document_id: str, text: str) -> Dict:
    data = _extract_json_obj(response)
    relations = data.get("relations", []) if data else []
    if not isinstance(relations, list):
        relations = []

    triples = []
    rejected_predicates = []
    for item in relations:
        if not isinstance(item, Mapping):
            continue
        triple = Triple.from_dict(dict(item), source_text=text, document_id=document_id)
        if not (triple.subject and triple.predicate and triple.object):
            continue
        predicate_error = predicate_validation_error(triple.predicate)
        if predicate_error:
            rejected_predicates.append(
                {
                    "predicate": triple.predicate,
                    "error": predicate_error,
                }
            )
            continue
        triple.predicate = clean_predicate(triple.predicate)
        triples.append(triple)

    return {
        "triples": triples,
        "raw_response": response or "",
        "parse_error": "" if data else "GPT response did not contain valid JSON.",
        "rejected_predicates": rejected_predicates,
    }
