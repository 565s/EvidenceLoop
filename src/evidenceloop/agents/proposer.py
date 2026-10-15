import json
from typing import Dict

from evidenceloop.agents.base import BaseAgent
from evidenceloop.logging import logger
from evidenceloop.resources.prompt_loader import (
    load_proposer_extract_call_prompt,
    load_proposer_system_prompt,
)
from evidenceloop.types import Triple
from evidenceloop.validation import clean_predicate, predicate_validation_error


class ProposerAgent(BaseAgent):
    def __init__(self, name="Proposer", role="Proposer", *, llm=None):
        self._llm_override = llm
        super().__init__(name, role)

    def _get_system_prompt(self) -> str:
        return load_proposer_system_prompt()

    def _init_llm(self):
        if self._llm_override is None:
            raise ValueError(
                "ProposerAgent requires llm; use Pipeline for default runtime construction"
            )
        return self._llm_override

    def _extract_json_obj(self, response: str) -> Dict:
        raw = self.extract_json(response or "")
        if not raw:
            return {}
        try:
            return json.loads(self.fix_unescaped_quotes(raw))
        except json.JSONDecodeError:
            logger.warning(f"Proposer JSON parse failed: {raw[:200]}...")
            return {}

    def propose_triples(self, document_id: str, text: str) -> Dict:
        prompt = load_proposer_extract_call_prompt().format(text=text)
        response = self.call_llm(prompt)
        data = self._extract_json_obj(response)
        relations = data.get("relations", [])
        if not isinstance(relations, list):
            relations = []

        triples = []
        rejected_predicates = []
        for item in relations:
            if not isinstance(item, dict):
                continue
            triple = Triple.from_dict(
                item,
                source_text=text,
                document_id=document_id,
            )
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

        if rejected_predicates:
            logger.warning(
                f"[document {document_id}] discarded {len(rejected_predicates)} relation(s) "
                "with invalid open predicates"
            )

        print(f"[document {document_id}] extracted {len(triples)} triples")
        return {
            "triples": triples,
            "raw_response": response,
            "parse_error": "" if data else "Proposer response did not contain valid JSON.",
            "rejected_predicates": rejected_predicates,
        }
