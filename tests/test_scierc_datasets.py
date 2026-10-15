import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from evidenceloop.experiments.checkpoints import load_canonical_rebel
from evidenceloop.data.builders.scierc_sft import (
    _make_fix,
    build_dataset as build_sft,
)
from evidenceloop.data.builders.scierc_documents import build_dataset as build_pipeline
from evidenceloop.data.sft import iter_jsonl, validate_critic_sft_jsonl
from evidenceloop.data.scierc import (
    convert_document,
    load_relation_map,
    predicate_aware_triple_key,
    symmetric_predicates,
)


@contextmanager
def workspace_temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"scierc_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


RELATIONS = {
    "COMPARE": "is compared with",
    "CONJUNCTION": "is conjoined with",
    "EVALUATE-FOR": "is used to evaluate",
    "FEATURE-OF": "is a feature of",
    "HYPONYM-OF": "is a hyponym of",
    "PART-OF": "is part of",
    "USED-FOR": "is used for",
}
SYMMETRIC = ("is compared with", "is conjoined with")


def document(doc_key, include_unknown=False, relationless=False):
    relations = [
        [
            [0, 1, 3, 4, "USED-FOR"],
            [3, 4, 0, 1, "COMPARE"],
            [0, 1, 3, 4, "COMPARE"],
        ],
        [[6, 6, 9, 9, "EVALUATE-FOR"]],
    ]
    if include_unknown:
        relations[0].append([0, 1, 3, 4, "UNKNOWN"])
    if relationless:
        relations = [[], []]
    return {
        "doc_key": doc_key,
        "sentences": [
            ["Graph", "models", "use", "neural", "features", "."],
            ["Accuracy", "evaluates", "the", "parser", "."],
        ],
        "ner": [
            [[0, 1, "Method"], [3, 4, "Task"]],
            [[6, 6, "Metric"], [9, 9, "Method"]],
        ],
        "relations": relations,
        "clusters": [[[0, 1], [9, 9]]],
    }


def write_sources(root, documents_per_split=8, relationless_test=False):
    relation_map = root / "relations.json"
    relation_map.write_text(json.dumps(RELATIONS), encoding="utf-8")
    source_files = {}
    for split_name, file_name in (
        ("train", "train.json"),
        ("validation", "dev.json"),
        ("test", "test.json"),
    ):
        path = root / file_name
        rows = [
            document(
                f"{split_name}-{index}",
                include_unknown=index == 0,
                relationless=(
                    relationless_test
                    and split_name == "test"
                    and index == documents_per_split - 1
                ),
            )
            for index in range(documents_per_split)
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        source_files[split_name] = path
    return source_files, relation_map


class ReverseShuffleRandom:
    def shuffle(self, values):
        values.reverse()

    def choice(self, values):
        return values[0]


class SciERCDatasetTests(unittest.TestCase):
    def test_adapter_reconstructs_spans_maps_relations_and_deduplicates_symmetric_edges(self):
        with workspace_temporary_directory() as root:
            relation_path = root / "relations.json"
            relation_path.write_text(json.dumps(RELATIONS), encoding="utf-8")
            converted, stats = convert_document(
                document("fixture", include_unknown=True),
                "test",
                load_relation_map(relation_path),
            )

            self.assertEqual(converted["document_id"], "scierc_test_fixture")
            self.assertEqual(
                converted["text"],
                "Graph models use neural features .\nAccuracy evaluates the parser .",
            )
            self.assertEqual(stats["raw_relations"], 5)
            self.assertEqual(stats["valid_triples"], 3)
            self.assertEqual(stats["duplicate_triples"], 1)
            self.assertEqual(stats["unknown_relations"], 1)
            self.assertEqual(stats["coreference_clusters"], 1)
            self.assertNotEqual(stats["valid_triples"], 4)
            for item in converted["gold_items"]:
                self.assertIn(item["evidence"], converted["text"])
                self.assertNotIn("\n", item["evidence"])
                self.assertIn(item["triple"]["subject"], item["evidence"])
                self.assertIn(item["triple"]["object"], item["evidence"])

    def test_fix_direction_is_available_only_for_asymmetric_predicates(self):
        entities = ["Graph models", "neural features", "Accuracy", "parser"]
        predicates = list(RELATIONS.values())
        asymmetric = {
            "triple": {
                "subject": "Graph models",
                "predicate": "is used for",
                "object": "neural features",
            }
        }
        symmetric = {
            "triple": {
                "subject": "Graph models",
                "predicate": "is compared with",
                "object": "neural features",
            }
        }
        rng = ReverseShuffleRandom()
        value = _make_fix(
            asymmetric, set(), entities, predicates, entities, predicates, SYMMETRIC, rng
        )
        self.assertEqual(value[1], "direction")
        symmetric_gold = {predicate_aware_triple_key(symmetric["triple"], SYMMETRIC)}
        value = _make_fix(
            symmetric,
            symmetric_gold,
            entities,
            predicates,
            entities,
            predicates,
            SYMMETRIC,
            rng,
        )
        self.assertNotEqual(value[1], "direction")

    def test_sft_uses_all_unique_gold_and_is_reproducible(self):
        with workspace_temporary_directory() as root:
            sources, relation_map = write_sources(root)
            first = root / "first"
            second = root / "second"
            metadata = build_sft(
                sources["train"],
                sources["validation"],
                sources["test"],
                relation_map,
                first,
                seed=42,
            )
            build_sft(
                sources["train"],
                sources["validation"],
                sources["test"],
                relation_map,
                second,
                seed=42,
            )

            for split in ("train", "validation", "test"):
                rows = list(iter_jsonl(first / f"{split}.jsonl"))
                summary = validate_critic_sft_jsonl(first / f"{split}.jsonl")
                expected = metadata["conversion"][split]["counts"]["valid_triples"]
                self.assertEqual(summary["candidates"], expected)
                self.assertTrue(all(1 <= len(row["triples_info"]) <= 4 for row in rows))
                self.assertTrue(all(
                    row["source_document_id"].startswith(f"scierc_{split}_")
                    for row in rows
                ))
                self.assertEqual(
                    (first / f"{split}.jsonl").read_bytes(),
                    (second / f"{split}.jsonl").read_bytes(),
                )
                for row in rows:
                    for item, result in zip(row["triples_info"], row["target"]["results"]):
                        if item.get("corruption") == "direction":
                            self.assertNotIn(
                                result["triple"]["predicate"], SYMMETRIC
                            )
            self.assertEqual(
                set(metadata["splits"]["train"]["action_counts"]),
                {"KEEP", "DELETE", "FIX", "REWRITE"},
            )

    def test_pipeline_uses_every_test_document_and_canonical_loader(self):
        with workspace_temporary_directory() as root:
            sources, relation_map = write_sources(
                root, documents_per_split=3, relationless_test=True
            )
            output = root / "pipeline"
            metadata = build_pipeline(sources["test"], relation_map, output)
            documents = list(iter_jsonl(output / "documents.jsonl"))
            gold = list(iter_jsonl(output / "gold_triples.jsonl"))
            loaded = load_canonical_rebel(output, limit=3)

            self.assertEqual(len(documents), 3)
            self.assertEqual(len(gold), 3)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(
                [row["document_id"] for row in documents],
                [row["document_id"] for row in gold],
            )
            self.assertEqual(metadata["statistics"]["documents"], 3)
            self.assertEqual(metadata["statistics"]["valid_gold_triples"], 6)
            self.assertEqual(loaded[2]["gold_triples"], [])
            self.assertEqual(set(documents[0]), {"document_id", "text"})
            self.assertEqual(
                tuple(metadata["symmetric_predicates"]),
                symmetric_predicates(load_relation_map(relation_map)),
            )


if __name__ == "__main__":
    unittest.main()
