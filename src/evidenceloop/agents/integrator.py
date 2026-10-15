import json
from typing import Dict, List

from evidenceloop.agents.base import BaseAgent
from evidenceloop.logging import logger
from evidenceloop.resources.prompt_loader import (
    load_integrator_call_prompt_batch,
    load_integrator_system_prompts,
)
from evidenceloop.types import Triple


class IntegratorAgent(BaseAgent):
    def __init__(self, name: str, role: str, *, llm=None):
        self._llm_override = llm
        super().__init__(name=name, role=role)

    def _get_system_prompt(self) -> str:
        return load_integrator_system_prompts()

    def _init_llm(self):
        if self._llm_override is not None:
            return self._llm_override
        raise ValueError("IntegratorAgent requires llm")

    def _public_triple_dict(self, triple: Triple) -> Dict:
        return triple.to_spo_dict()

    def _extract_json_obj(self, response: str) -> Dict:
        raw = self.extract_json(response or "")
        if not raw:
            return {}
        try:
            return json.loads(self.fix_unescaped_quotes(raw))
        except json.JSONDecodeError:
            logger.warning(f"Integrator JSON parse failed: {raw[:200]}...")
            return {}

    def audit_coverage(
        self,
        source_text: str,
        refined_triples: List[Triple],
    ) -> Dict:
        prompt = load_integrator_call_prompt_batch().format(
            source_text=source_text,
            refined_triples=json.dumps(
                [self._public_triple_dict(triple) for triple in refined_triples],
                ensure_ascii=False,
                indent=2,
            ),
        )
        response = self.call_llm(prompt)
        data = self._extract_json_obj(response)
        additions = data.get("additions", [])
        if not isinstance(additions, list):
            additions = []
        return {
            "coverage_complete": bool(data.get("coverage_complete", False)),
            "summary": str(data.get("summary", "") or ""),
            "additions": [item for item in additions if isinstance(item, dict)],
            "raw_response": response,
            "parse_error": "" if data else "Integrator response did not contain valid JSON.",
        }
