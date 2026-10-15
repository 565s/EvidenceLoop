import json
import unittest
from pathlib import Path


ROOT = Path.cwd()
from importlib.resources import files
PROMPTS_DIR = files("evidenceloop.resources").joinpath("prompts")


def read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


class ProposerPromptTests(unittest.TestCase):
    def test_extract_prompt_formats_entity_first_open_spo_schema(self):
        template = read_prompt("proposer_extract_call_prompt.txt")
        rendered = template.format(text="Ada was born in London.")

        final_schema = rendered.rsplit("exact top-level structure:", 1)[1]
        self.assertLess(final_schema.index('"entities"'), final_schema.index('"relations"'))
        self.assertIn('"name": "..."', final_schema)
        for field in ("subject", "predicate", "object"):
            self.assertIn(f'"{field}"', final_schema)
        for removed in ("type", "subject_type", "object_type", "qualifiers"):
            self.assertNotIn(f'"{removed}"', rendered)
        self.assertIn("Few-shot Example 1", rendered)
        self.assertIn("Few-shot Example 2", rendered)
        self.assertIn("Ada was born in London.", rendered)
        self.assertIn("natural-language predicate phrase", rendered)
        self.assertIn("no more than 80 characters", rendered)

    def test_role_prompts_do_not_reference_candidate_constraints(self):
        prompts = [
            read_prompt("proposer_system_prompt.txt"),
            read_prompt("proposer_extract_call_prompt.txt"),
            read_prompt("critic_system_prompt.txt"),
            read_prompt("critic_call_prompt_batch.txt"),
            read_prompt("integrator_system_prompt.txt"),
            read_prompt("integrator_call_prompt_batch.txt"),
        ]
        removed_terms = ("Predicate Candidates", "allowlist", "Top5", "Top20")
        for prompt in prompts:
            for term in removed_terms:
                self.assertNotIn(term, prompt)
        self.assertFalse((PROMPTS_DIR / "proposer_response_to_critic_call_batch.txt").exists())

    def test_critic_prompts_constrain_evidence_for_json_safety(self):
        prompts = [
            read_prompt("critic_system_prompt.txt"),
            read_prompt("critic_call_prompt_batch.txt"),
        ]
        combined = "\n".join(prompts)

        self.assertIn("160 characters", combined)
        self.assertIn("single-line", combined)
        self.assertIn("JSON-escape", combined)
        self.assertIn("unescaped quotes", combined)
        self.assertIn("raw newlines", combined)
        self.assertIn("shortest evidence span", combined)

    def test_legacy_type_and_qualifier_fields_are_ignored(self):
        from evidenceloop.agents.proposer import ProposerAgent

        proposer = object.__new__(ProposerAgent)
        proposer.call_llm = lambda prompt: json.dumps({
            "entities": [
                {"name": "Ada", "type": "person"},
                {"name": "London", "type": "city"},
            ],
            "relations": [{
                "subject": "Ada",
                "subject_type": "person",
                "predicate": "place of birth",
                "object": "London",
                "object_type": "city",
                "qualifiers": [{"predicate": "point in time", "object": "1815"}],
            }],
        })

        result = proposer.propose_triples("doc-1", "Ada was born in London.")
        triples = result["triples"]

        self.assertEqual(len(triples), 1)
        self.assertEqual(
            triples[0].to_spo_dict(),
            {"subject": "Ada", "predicate": "place of birth", "object": "London"},
        )
        for removed in ("subject_type", "object_type", "qualifiers"):
            self.assertFalse(hasattr(triples[0], removed))

    def test_arbitrary_valid_predicates_are_kept_and_whitespace_is_folded(self):
        from evidenceloop.agents.proposer import ProposerAgent

        proposer = object.__new__(ProposerAgent)
        proposer.call_llm = lambda prompt: json.dumps({
            "entities": [],
            "relations": [
                {"subject": "Ada", "predicate": "  pioneered   analytical engines ", "object": "Engine"},
                {"subject": "Ada", "predicate": "occupation", "object": "mathematician"},
            ],
        })

        result = proposer.propose_triples("doc-1", "Ada pioneered analytical engines.")

        self.assertEqual(
            [triple.predicate for triple in result["triples"]],
            ["pioneered analytical engines", "occupation"],
        )
        self.assertEqual(result["rejected_predicates"], [])

    def test_invalid_predicate_formats_are_discarded(self):
        from evidenceloop.agents.proposer import ProposerAgent

        proposer = object.__new__(ProposerAgent)
        proposer.call_llm = lambda prompt: json.dumps({
            "relations": [
                {"subject": "A", "predicate": "line one\nline two", "object": "B"},
                {"subject": "A", "predicate": "x" * 81, "object": "C"},
                {"subject": "A", "predicate": "valid relation", "object": "D"},
            ],
        })

        result = proposer.propose_triples("doc-1", "A relates to B, C, and D.")

        self.assertEqual([triple.predicate for triple in result["triples"]], ["valid relation"])
        self.assertEqual(len(result["rejected_predicates"]), 2)
        errors = [item["error"] for item in result["rejected_predicates"]]
        self.assertTrue(any("single line" in error for error in errors))
        self.assertTrue(any("exceeds 80 characters" in error for error in errors))

    def test_extraction_runs_without_candidate_input(self):
        from evidenceloop.agents.proposer import ProposerAgent

        proposer = object.__new__(ProposerAgent)
        captured = {}

        def call_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps({"entities": [], "relations": []})

        proposer.call_llm = call_llm
        result = proposer.propose_triples("doc-1", "Ada was born in London.")

        self.assertEqual(result["triples"], [])
        self.assertIn("Ada was born in London.", captured["prompt"])
        self.assertNotIn("Predicate Candidates", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
