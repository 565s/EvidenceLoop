import json
import os
import tempfile
import unittest

from evidenceloop.pipeline.core import Pipeline
from evidenceloop.types import DocumentPipelineRecord, Triple, TripleReviewRecord


SOURCE = (
    "KeepMe is related to Target. FixMe is located in Paris. "
    "FixMe is located in Paris Center. RewriteMe is related to the item. "
    "Added works at Company."
)
class MockProposer:
    def propose_triples(self, document_id, text):
        base = {"source_text": text, "document_id": document_id}
        return {
            "triples": [
                Triple("KeepMe", "related_to", "Target", **base),
                Triple("DeleteMe", "unrelated_to", "Noise", **base),
                Triple("FixMe", "wrong_relation", "Paris", **base),
                Triple("RewriteMe", "related_to", "the item", **base),
                Triple("KeepMe", "related_to", "Target", **base),
            ],
            "raw_response": "proposer raw",
            "parse_error": "",
            "rejected_predicates": [],
        }


class MockCritic:
    def review_triples_batch(self, triples, stage, origins=None):
        results = []
        for idx, triple in enumerate(triples):
            if stage == "initial_review" and triple.subject == "DeleteMe":
                item = {"triple_idx": idx, "action": "DELETE", "reason": "Unsupported.", "evidence": "", "triple": None}
            elif stage == "initial_review" and triple.subject == "FixMe":
                item = {
                    "triple_idx": idx,
                    "action": "FIX",
                    "reason": "Correct relation.",
                    "evidence": "FixMe is located in Paris.",
                    "triple": {"subject": "FixMe", "predicate": "  located   in  ", "object": "Paris"},
                }
            elif stage == "initial_review" and triple.subject == "RewriteMe":
                item = {
                    "triple_idx": idx,
                    "action": "REWRITE",
                    "reason": "Normalize object.",
                    "evidence": "RewriteMe is related to the item.",
                    "triple": {"subject": "RewriteMe", "predicate": "related_to", "object": "Item"},
                }
            elif stage == "final_validation" and triple.subject == "RewriteMe":
                item = {"triple_idx": idx, "action": "DELETE", "reason": "Remove in final validation.", "evidence": "", "triple": None}
            elif stage == "final_validation" and triple.subject == "FixMe":
                item = {
                    "triple_idx": idx,
                    "action": "FIX",
                    "reason": "Use precise place.",
                    "evidence": "FixMe is located in Paris Center.",
                    "triple": {"subject": "FixMe", "predicate": "located within", "object": "Paris Center"},
                }
            else:
                evidence = "Added works at Company." if triple.subject == "Added" else "KeepMe is related to Target."
                item = {"triple_idx": idx, "action": "KEEP", "reason": "Supported.", "evidence": evidence, "triple": None}
            results.append(item)
        return {"results": results, "raw_response": f"critic {stage}", "parse_error": ""}


class CapturingCritic(MockCritic):
    def __init__(self):
        self.calls = []

    def review_triples_batch(self, triples, stage, origins=None):
        self.calls.append({
            "triples": list(triples),
            "stage": stage,
            "origins": list(origins or []),
        })
        return super().review_triples_batch(triples, stage, origins)


class MockIntegrator:
    def __init__(self):
        self.calls = 0

    def audit_coverage(self, source_text, refined_triples):
        self.calls += 1
        return {
            "coverage_complete": False,
            "summary": "One supported fact is missing.",
            "additions": [
                {
                    "triple": {"subject": "Added", "predicate": "employed by", "object": "Company"},
                    "reason": "Missing employment fact.",
                    "evidence": "Added works at Company.",
                },
                {
                    "triple": {"subject": "KeepMe", "predicate": "related_to", "object": "Target"},
                    "reason": "Duplicate.",
                    "evidence": "KeepMe is related to Target.",
                },
                {
                    "triple": {"subject": "Added", "predicate": "x" * 81, "object": "Engineer"},
                    "reason": "Invalid predicate format.",
                    "evidence": "Added works at Company.",
                },
                {
                    "triple": {"subject": "Added", "predicate": "unsupported role", "object": "Engineer"},
                    "reason": "Missing evidence.",
                    "evidence": "",
                },
                {
                    "triple": {"subject": "", "predicate": "employed by", "object": "Company"},
                    "reason": "Empty subject.",
                    "evidence": "Added works at Company.",
                },
            ],
            "raw_response": "integrator raw",
            "parse_error": "",
        }


class EmptyProposer:
    def propose_triples(self, document_id, text):
        return {"triples": [], "raw_response": "empty", "parse_error": "", "rejected_predicates": []}


class AdditionOnlyIntegrator(MockIntegrator):
    def audit_coverage(self, source_text, refined_triples):
        self.calls += 1
        return {
            "coverage_complete": False,
            "summary": "Missing fact.",
            "additions": [{
                "triple": {"subject": "Added", "predicate": "employed by", "object": "Company"},
                "reason": "Missing.",
                "evidence": "Added works at Company.",
            }],
            "raw_response": "integrator raw",
            "parse_error": "",
        }


class MissingFinalCritic:
    def review_triples_batch(self, triples, stage, origins=None):
        return {"results": [], "raw_response": stage, "parse_error": ""}


class InvalidInitialPredicateCritic(MockCritic):
    def review_triples_batch(self, triples, stage, origins=None):
        if stage == "initial_review":
            return {
                "results": [{
                    "triple_idx": idx,
                    "action": "FIX",
                    "reason": "Invalid attempted correction.",
                    "evidence": "KeepMe is related to Target.",
                    "triple": {"subject": triple.subject, "predicate": "x" * 81, "object": triple.object},
                } for idx, triple in enumerate(triples)],
                "raw_response": stage,
                "parse_error": "",
            }
        return super().review_triples_batch(triples, stage, origins)


class PipelineFlowTests(unittest.TestCase):
    def make_controller(
        self,
        proposer=None,
        critic=None,
        integrator=None,
        pipeline_variant="full",
    ):
        return Pipeline(
            proposer_agent=proposer or MockProposer(),
            critic_agent=critic or MockCritic(),
            integrator_agent=integrator or MockIntegrator(),
            pipeline_variant=pipeline_variant,
        )

    def run_document(self, controller, document_id="doc-1"):
        return controller.process_document(document_id, SOURCE)

    def test_full_pipeline_actions_evidence_additions_and_dedupe(self):
        critic = CapturingCritic()
        controller = self.make_controller(critic=critic)
        record = self.run_document(controller)

        keys = [triple.to_spo_dict() for triple in record.final_triples]
        self.assertEqual(keys, [
            {"subject": "KeepMe", "predicate": "related_to", "object": "Target"},
            {"subject": "FixMe", "predicate": "located within", "object": "Paris Center"},
            {"subject": "Added", "predicate": "employed by", "object": "Company"},
        ])
        self.assertEqual([review.action for review in record.initial_reviews], ["KEEP", "DELETE", "FIX", "REWRITE", "KEEP"])
        fixed = record.initial_reviews[2]
        self.assertEqual(fixed.output_triple.predicate, "located in")
        self.assertEqual(fixed.evidence_start, SOURCE.index("FixMe is located in Paris."))
        self.assertEqual(len(record.coverage_audit["additions"]), 5)
        self.assertEqual(len(record.added_triples), 1)
        self.assertIn("exceeds 80 characters", record.coverage_audit["additions"][2]["validation_error"])
        self.assertIn("evidence is required", record.coverage_audit["additions"][3]["validation_error"])
        self.assertIn("empty fields", record.coverage_audit["additions"][4]["validation_error"])
        self.assertEqual(set(record.raw_responses), {"proposer", "initial_critic", "integrator", "final_critic"})
        self.assertEqual([call["stage"] for call in critic.calls], ["initial_review", "final_validation"])
        final_call = critic.calls[1]
        self.assertEqual(final_call["triples"], record.refined_triples + record.added_triples)
        self.assertEqual(
            final_call["origins"],
            ["proposer"] * len(record.refined_triples)
            + ["integrator"] * len(record.added_triples),
        )

    def test_invalid_critic_predicate_correction_keeps_original_fact(self):
        controller = self.make_controller(
            critic=InvalidInitialPredicateCritic(),
            integrator=AdditionOnlyIntegrator(),
        )
        record = self.run_document(controller)

        self.assertFalse(record.initial_reviews[0].valid)
        self.assertEqual(record.initial_reviews[0].action, "KEEP")
        self.assertEqual(record.initial_reviews[0].output_triple.predicate, "related_to")
        self.assertIn("exceeds 80 characters", record.initial_reviews[0].validation_error)

    def test_proposer_critic_once_skips_integrator_and_final_review(self):
        critic = CapturingCritic()
        integrator = MockIntegrator()
        controller = self.make_controller(
            critic=critic,
            integrator=integrator,
            pipeline_variant="proposer_critic_once",
        )

        record = self.run_document(controller)

        self.assertEqual(integrator.calls, 0)
        self.assertEqual([call["stage"] for call in critic.calls], ["initial_review"])
        self.assertEqual(record.final_reviews, [])
        self.assertEqual(record.added_triples, [])
        self.assertTrue(record.coverage_audit["skipped"])
        self.assertEqual(
            [triple.to_spo_dict() for triple in record.final_triples],
            [
                {"subject": "KeepMe", "predicate": "related_to", "object": "Target"},
                {"subject": "FixMe", "predicate": "located in", "object": "Paris"},
                {"subject": "RewriteMe", "predicate": "related_to", "object": "Item"},
            ],
        )
        self.assertEqual(record.final_triples, controller._dedupe_document(record.refined_triples))
        self.assertEqual(set(record.raw_responses), {"proposer", "initial_critic"})
        self.assertEqual(set(record.stage_timings_seconds), {"proposer", "initial_critic"})

    def test_injected_proposal_skips_internal_proposer_call(self):
        class NeverProposer:
            def propose_triples(self, document_id, text):
                raise AssertionError("internal Proposer must not be called")

        critic = CapturingCritic()
        controller = self.make_controller(
            proposer=NeverProposer(),
            critic=critic,
            pipeline_variant="proposer_critic_once",
        )
        triple = Triple(
            "KeepMe",
            "related_to",
            "Target",
            source_text=SOURCE,
            document_id="doc-1",
        )
        payload = {
            "triples": [triple],
            "raw_response": "cached proposer",
            "parse_error": "",
            "rejected_predicates": [],
        }

        record = controller.process_document(
            "doc-1",
            SOURCE,
            proposer_payload=payload,
            proposer_elapsed_seconds=1.25,
        )

        self.assertEqual(record.stage_timings_seconds["proposer"], 1.25)
        self.assertEqual(record.raw_responses["proposer"], "cached proposer")
        self.assertEqual([call["stage"] for call in critic.calls], ["initial_review"])

    def test_injected_initial_review_skips_internal_critic_call(self):
        class NeverCritic:
            def review_triples_batch(self, triples, stage, origins=None):
                raise AssertionError("internal Critic must not be called")

        controller = self.make_controller(
            critic=NeverCritic(),
            pipeline_variant="proposer_critic_once",
        )
        triple = Triple(
            "KeepMe",
            "related_to",
            "Target",
            source_text=SOURCE,
            document_id="doc-1",
        )
        proposal = {
            "triples": [triple],
            "raw_response": "cached proposer",
            "parse_error": "",
            "rejected_predicates": [],
        }
        review = {
            "results": [{
                "triple_idx": 0,
                "action": "KEEP",
                "reason": "Supported.",
                "evidence": "KeepMe is related to Target",
                "triple": None,
            }],
            "raw_response": "cached critic",
            "parse_error": "",
        }

        record = controller.process_document(
            "doc-1",
            SOURCE,
            proposer_payload=proposal,
            proposer_elapsed_seconds=1.25,
            initial_critic_payload=review,
            initial_critic_elapsed_seconds=2.5,
        )

        self.assertEqual(record.stage_timings_seconds["initial_critic"], 2.5)
        self.assertEqual(record.raw_responses["initial_critic"], "cached critic")
        self.assertEqual(record.final_triples, [triple])

    def test_invalid_pipeline_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline_variant"):
            self.make_controller(pipeline_variant="unknown")

    def test_empty_extraction_still_audits_and_unverified_addition_is_discarded(self):
        integrator = AdditionOnlyIntegrator()
        controller = self.make_controller(
            proposer=EmptyProposer(),
            critic=MissingFinalCritic(),
            integrator=integrator,
        )
        record = self.run_document(controller)

        self.assertEqual(integrator.calls, 1)
        self.assertEqual(len(record.added_triples), 1)
        self.assertEqual(record.final_triples, [])
        self.assertFalse(record.final_reviews[0].valid)
        self.assertEqual(record.final_reviews[0].action, "DELETE")

    def test_missing_final_reviews_keep_refined_triples_and_drop_additions(self):
        controller = self.make_controller(
            critic=MissingFinalCritic(),
            integrator=AdditionOnlyIntegrator(),
        )
        record = self.run_document(controller)

        self.assertNotIn("Added", [triple.subject for triple in record.final_triples])
        proposer_reviews = [
            review for review in record.final_reviews if review.origin == "proposer"
        ]
        addition_reviews = [
            review for review in record.final_reviews if review.origin == "integrator"
        ]
        self.assertTrue(proposer_reviews)
        self.assertTrue(all(not review.valid and review.action == "KEEP" for review in proposer_reviews))
        self.assertEqual(len(addition_reviews), 1)
        self.assertFalse(addition_reviews[0].valid)
        self.assertEqual(addition_reviews[0].action, "DELETE")

    def test_same_triple_is_retained_for_different_documents(self):
        controller = self.make_controller()
        records = [self.run_document(controller, "doc-1"), self.run_document(controller, "doc-2")]

        repeated = [triple for record in records for triple in record.final_triples if triple.subject == "KeepMe"]
        self.assertEqual([triple.document_id for triple in repeated], ["doc-1", "doc-2"])

    def test_pipeline_outputs_stream_per_document(self):
        controller = self.make_controller()
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            from evidenceloop import Document
            from evidenceloop.pipeline.run import RunManager
            run_dir = RunManager(controller).run([Document("doc-1", SOURCE)], Path(tmpdir) / "run")
            record = self.run_document(controller)
            kg_rows = [json.loads(line) for line in (run_dir / "kg.jsonl").read_text(encoding="utf-8").splitlines()]
            pipeline_rows = [json.loads(line) for line in (run_dir / "pipeline_records.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(kg_rows), len(record.final_triples))
        self.assertEqual(len(pipeline_rows), 1)
        self.assertEqual(pipeline_rows[0]["document_id"], "doc-1")
        self.assertIn("coverage_audit", pipeline_rows[0])
        self.assertEqual(
            set(pipeline_rows[0]["stage_timings_seconds"]),
            {"proposer", "initial_critic", "integrator", "final_critic"},
        )

    def test_pipeline_dataclasses_serialize_stage_provenance(self):
        triple = Triple("S", "p", "O", source_text="S p O", document_id="doc-1")
        review = TripleReviewRecord(
            input_triple=triple,
            stage="initial_review",
            origin="proposer",
            action="KEEP",
            output_triple=triple,
            reason="Supported.",
            evidence="S p O",
            evidence_start=0,
            evidence_end=5,
        )
        record = DocumentPipelineRecord(
            document_id="doc-1",
            source_text="S p O",
            initial_reviews=[review],
            final_triples=[triple],
        )
        payload = record.to_dict()
        self.assertEqual(payload["initial_reviews"][0]["origin"], "proposer")
        self.assertEqual(payload["final_triples"][0]["document_id"], "doc-1")
        self.assertNotIn("proposer_predicates", payload)
        self.assertNotIn("review_predicates", payload)


if __name__ == "__main__":
    unittest.main()
