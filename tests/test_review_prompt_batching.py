import json
import unittest
from types import SimpleNamespace

from evidenceloop.agents.critic import CriticAgent
from evidenceloop.agents.integrator import IntegratorAgent
from evidenceloop.types import Triple
from evidenceloop.runtime.direct_critic import DirectGenerationResult


SOURCE_TEXT = "x" * 1250 + " UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts."


def make_triples():
    return [
        Triple("SubjectA", "related_to", "ObjectA", source_text=SOURCE_TEXT, document_id="doc-1"),
        Triple("SubjectB", "located_in", "ObjectB", source_text=SOURCE_TEXT, document_id="doc-1"),
    ]


def make_many_triples(count):
    return [
        Triple(
            f"Subject{index}",
            "related_to",
            f"Object{index}",
            source_text=SOURCE_TEXT,
            document_id="doc-1",
        )
        for index in range(count)
    ]


def triples_info_from_prompt(prompt):
    value = prompt.split("Triples:\n", 1)[1].split("\n\nOutput rules:", 1)[0]
    return json.loads(value)


class FakeChatModel:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=json.dumps({"results": [{
            "triple_idx": 0,
            "action": "KEEP",
            "reason": "Supported.",
            "evidence": "UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts.",
            "triple": None,
        }]}))


class FakeDirectModel:
    def __init__(self, *, fail_prompt=None):
        self.calls = []
        self.fail_prompt = fail_prompt

    def generate_batch(self, message_batches, *, batch_size):
        self.calls.append((message_batches, batch_size))
        output = []
        for prompt_index, messages in enumerate(message_batches):
            prompt = messages[-1][1]
            info = triples_info_from_prompt(prompt)
            if prompt_index == self.fail_prompt:
                output.append(DirectGenerationResult(error="simulated CUDA OOM"))
                continue
            output.append(DirectGenerationResult(content=json.dumps({"results": [{
                "triple_idx": item["triple_idx"],
                "action": "KEEP",
                "reason": "Supported.",
                "evidence": "UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts.",
                "triple": None,
            } for item in info]})))
        return output


class ReviewPromptBatchingTests(unittest.TestCase):
    def test_critic_prompt_contains_shared_source_stage_and_items(self):
        critic = object.__new__(CriticAgent)
        captured = {}

        def call_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps({"results": [
                {
                    "triple_idx": idx,
                    "action": "KEEP",
                    "reason": "Supported.",
                    "evidence": "UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts.",
                    "triple": None,
                }
                for idx in range(2)
            ]})

        critic.call_llm = call_llm
        payload = critic.review_triples_batch(
            make_triples(),
            stage="initial_review",
            origins=["proposer", "proposer"],
        )

        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(captured["prompt"].count(SOURCE_TEXT), 1)
        self.assertIn("initial_review", captured["prompt"])
        self.assertIn('"related_to"', captured["prompt"])
        self.assertIn('"origin": "proposer"', captured["prompt"])
        self.assertNotIn('"source_text"', captured["prompt"])
        self.assertNotIn("Predicate Candidates", captured["prompt"])

    def test_integrator_prompt_supports_empty_refined_graph(self):
        integrator = object.__new__(IntegratorAgent)
        captured = {}

        def call_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps({"coverage_complete": True, "summary": "Complete.", "additions": []})

        integrator.call_llm = call_llm
        payload = integrator.audit_coverage(SOURCE_TEXT, [])

        self.assertTrue(payload["coverage_complete"])
        self.assertEqual(captured["prompt"].count(SOURCE_TEXT), 1)
        self.assertIn("Refined Knowledge Graph:\n[]", captured["prompt"])
        self.assertNotIn("Predicate Candidates", captured["prompt"])

    def test_empty_critic_batch_does_not_call_llm(self):
        critic = object.__new__(CriticAgent)
        critic.call_llm = lambda prompt: self.fail("LLM should not be called")
        payload = critic.review_triples_batch([], stage="initial_review")
        self.assertEqual(payload["results"], [])

    def test_critic_chunks_and_remaps_document_indices(self):
        for count, expected_calls in ((1, 1), (4, 1), (5, 2), (10, 3)):
            with self.subTest(count=count):
                critic = object.__new__(CriticAgent)
                critic.max_batch_size = 4
                prompts = []

                def call_llm(prompt):
                    prompts.append(prompt)
                    info = triples_info_from_prompt(prompt)
                    return json.dumps({"results": [{
                        "triple_idx": item["triple_idx"],
                        "action": "KEEP",
                        "reason": "Supported.",
                        "evidence": "UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts.",
                        "triple": None,
                    } for item in info]})

                critic.call_llm = call_llm
                origins = ["proposer"] * count
                if count:
                    origins[-1] = "integrator"
                payload = critic.review_triples_batch(
                    make_many_triples(count),
                    stage="initial_review",
                    origins=origins,
                )

                self.assertEqual(len(prompts), expected_calls)
                self.assertTrue(all(len(triples_info_from_prompt(prompt)) <= 4 for prompt in prompts))
                self.assertEqual(
                    [item["triple_idx"] for item in payload["results"]],
                    list(range(count)),
                )
                self.assertIn('"origin": "integrator"', prompts[-1])
                if expected_calls > 1:
                    chunks = json.loads(payload["raw_response"])
                    self.assertEqual(len(chunks), expected_calls)

    def test_critic_accepts_explicit_candidate_set_boundaries(self):
        critic = object.__new__(CriticAgent)
        critic.max_batch_size = 4
        prompts = []

        def call_llm(prompt):
            prompts.append(prompt)
            info = triples_info_from_prompt(prompt)
            return json.dumps({"results": [{
                "triple_idx": item["triple_idx"],
                "action": "KEEP",
                "reason": "Supported.",
                "evidence": "UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts.",
                "triple": None,
            } for item in info]})

        critic.call_llm = call_llm
        payload = critic.review_triples_batch(
            make_many_triples(5),
            stage="initial_review",
            chunk_sizes=[2, 2, 1],
        )

        self.assertEqual(
            [len(triples_info_from_prompt(prompt)) for prompt in prompts],
            [2, 2, 1],
        )
        self.assertEqual(
            [item["triple_idx"] for item in payload["results"]],
            list(range(5)),
        )

    def test_invalid_chunk_does_not_discard_other_chunk_results(self):
        critic = object.__new__(CriticAgent)
        critic.max_batch_size = 4
        call_number = 0

        def call_llm(prompt):
            nonlocal call_number
            call_number += 1
            if call_number == 2:
                return "not json"
            info = triples_info_from_prompt(prompt)
            return json.dumps({"results": [{
                "triple_idx": item["triple_idx"],
                "action": "KEEP",
                "reason": "Supported.",
                "evidence": "UNIQUE_TAIL_SOURCE_TEXT_MARKER supports both facts.",
                "triple": None,
            } for item in info]})

        critic.call_llm = call_llm
        payload = critic.review_triples_batch(
            make_many_triples(10),
            stage="initial_review",
        )
        self.assertEqual(
            [item["triple_idx"] for item in payload["results"]],
            [0, 1, 2, 3, 8, 9],
        )
        self.assertIn("indices 4-7", payload["parse_error"])

    def test_initial_and_final_stages_share_the_adapter_client(self):
        adapter = FakeChatModel("adapter")
        critic = CriticAgent(
            name="test",
            role="Critic",
            llm=adapter,
            runtime_settings={"batch_size": 4},
        )
        triple = make_many_triples(1)

        critic.review_triples_batch(triple, stage="initial_review")
        critic.review_triples_batch(triple, stage="final_validation")

        self.assertEqual(len(adapter.calls), 2)
        prompts = [messages[-1][1] for messages in adapter.calls]
        self.assertIn("initial_review", prompts[0])
        self.assertIn("final_validation", prompts[1])

    def test_direct_model_batches_chunks_across_documents_and_remaps_indices(self):
        direct = FakeDirectModel()
        critic = CriticAgent(
            name="test",
            role="Critic",
            llm=direct,
            runtime_settings={"batch_size": 4, "prompt_batch_size": 4},
        )

        outputs = critic.review_documents_batch([
            {
                "triples": make_many_triples(5),
                "stage": "initial_review",
                "origins": ["proposer"] * 5,
            },
            {
                "triples": make_many_triples(3),
                "stage": "initial_review",
                "origins": ["proposer"] * 3,
            },
        ])

        self.assertEqual(len(direct.calls), 1)
        message_batches, batch_size = direct.calls[0]
        self.assertEqual(batch_size, 4)
        self.assertEqual(len(message_batches), 3)
        self.assertEqual(
            [item["triple_idx"] for item in outputs[0]["payload"]["results"]],
            list(range(5)),
        )
        self.assertEqual(
            [item["triple_idx"] for item in outputs[1]["payload"]["results"]],
            list(range(3)),
        )
        self.assertEqual([item["prompt_count"] for item in outputs], [2, 1])

    def test_direct_generation_failure_is_isolated_to_its_chunk(self):
        direct = FakeDirectModel(fail_prompt=1)
        critic = CriticAgent(
            name="test",
            role="Critic",
            llm=direct,
            runtime_settings={"batch_size": 4, "prompt_batch_size": 4},
        )

        outputs = critic.review_documents_batch([
            {
                "triples": make_many_triples(5),
                "stage": "initial_review",
                "origins": ["proposer"] * 5,
            },
            {
                "triples": make_many_triples(2),
                "stage": "initial_review",
                "origins": ["proposer"] * 2,
            },
        ])

        self.assertEqual(
            [item["triple_idx"] for item in outputs[0]["payload"]["results"]],
            [0, 1, 2, 3],
        )
        self.assertIn("indices 4-4: simulated CUDA OOM", outputs[0]["payload"]["parse_error"])
        self.assertEqual(len(outputs[1]["payload"]["results"]), 2)


if __name__ == "__main__":
    unittest.main()
