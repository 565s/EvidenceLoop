import json
import unittest

from evidenceloop.data.builders.candidate_sets import (
    _record_from_items,
    _sizes_for_variant,
)
from evidenceloop.evaluation.candidate_sets import (
    NOMINAL_TRAINING_SET_SIZE,
    _bootstrap_delta,
)
from evidenceloop.experiments.checkpoints import critic_chunk_sizes


def candidate_items(count=8):
    source_text = " ".join(
        "Subject {0} supports Object {0}.".format(index)
        for index in range(count)
    )
    return [
        {
            "document_id": "redocred_test_000000",
            "source_document_index": 0,
            "source_text": source_text,
            "action": "KEEP",
            "candidate": {
                "subject": "Subject {0}".format(index),
                "predicate": "supports",
                "object": "Object {0}".format(index),
            },
            "gold": {
                "subject": "Subject {0}".format(index),
                "predicate": "supports",
                "object": "Object {0}".format(index),
            },
            "evidence": "Subject {0} supports Object {0}.".format(index),
            "corruption": "none",
        }
        for index in range(count)
    ]


class ReDocREDCandidateSetTests(unittest.TestCase):
    def test_candidate_set_sizes_are_deterministic(self):
        self.assertEqual(_sizes_for_variant("fixed1", 5), [1, 1, 1, 1, 1])
        self.assertEqual(_sizes_for_variant("fixed2", 5), [2, 2, 1])
        self.assertEqual(_sizes_for_variant("fixed4", 5), [4, 1])
        first = _sizes_for_variant("variable", 17, __import__("random").Random("seed"))
        second = _sizes_for_variant("variable", 17, __import__("random").Random("seed"))
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 17)
        self.assertTrue(all(1 <= value <= 4 for value in first))

    def test_records_keep_document_boundaries_and_reindex(self):
        items = candidate_items(4)
        record = _record_from_items(items[1:3], "fixed2", "test", 0)
        self.assertEqual(record["source_document_id"], "redocred_test_000000")
        self.assertEqual([item["triple_idx"] for item in record["triples_info"]], [0, 1])
        self.assertEqual(
            [item["triple_idx"] for item in record["target"]["results"]],
            [0, 1],
        )
        self.assertNotIn("prompt", json.dumps(record))

    def test_cross_document_groups_are_rejected(self):
        items = candidate_items(2)
        items[1] = dict(items[1], document_id="redocred_test_000001")
        with self.assertRaisesRegex(RuntimeError, "document boundary"):
            _record_from_items(items, "fixed2", "test", 0)

    def test_runtime_candidate_chunking_preserves_all_indices(self):
        for variant, expected in {
            "fixed1": [1] * 5,
            "fixed2": [2, 2, 1],
            "fixed4": [4, 1],
        }.items():
            sizes = critic_chunk_sizes(5, variant, document_id="doc-1")
            self.assertEqual(sizes, expected)
            self.assertEqual(sum(sizes), 5)
        first = critic_chunk_sizes(19, "variable", seed=42, document_id="doc-1")
        second = critic_chunk_sizes(19, "variable", seed=42, document_id="doc-1")
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 19)
        self.assertTrue(all(1 <= size <= 4 for size in first))

    def test_bootstrap_delta_is_paired_by_document_id(self):
        result = _bootstrap_delta(
            {"a": 0.4, "b": 0.8},
            {"a": 0.2, "b": 0.5, "ignored": 1.0},
            seed=42,
            samples=100,
        )
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["delta"], 0.25)
        self.assertEqual(len(result["ci95"]), 2)
        self.assertEqual(NOMINAL_TRAINING_SET_SIZE["variable"], "1-4")


if __name__ == "__main__":
    unittest.main()
