import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from evidenceloop.experiments.checkpoints import load_canonical_rebel
from evidenceloop.data.builders.redocred_sft import build_dataset as build_sft
from evidenceloop.data.builders.redocred_documents import build_dataset as build_pipeline
from evidenceloop.data.sft import iter_jsonl, validate_critic_sft_jsonl
from evidenceloop.data.redocred import convert_document, load_relation_map


@contextmanager
def workspace_temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"redocred_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


RELATIONS = {
    "P1": "works with",
    "P2": "located in",
    "P3": "member of",
    "P4": "place of birth",
}


def article(prefix, include_unknown=False):
    sents = [
        [f"{prefix}Alpha", "Researcher", "joined", f"{prefix}Beta", "Labs", "."],
        ["The", "researcher", "works", "in", f"{prefix}Paris", "with", f"{prefix}Gamma", "Team", "."],
        [f"{prefix}Delta", "Institute", "supports", f"{prefix}Beta", "Labs", "."],
    ]
    vertex_set = [
        [
            {"name": f"{prefix}Alpha Researcher", "sent_id": 0, "pos": [0, 2], "type": "PER"},
            {"name": "The researcher", "sent_id": 1, "pos": [0, 2], "type": "PER"},
        ],
        [{"name": f"{prefix}Beta Labs", "sent_id": 0, "pos": [3, 5], "type": "ORG"}],
        [{"name": f"{prefix}Paris", "sent_id": 1, "pos": [4, 5], "type": "LOC"}],
        [{"name": f"{prefix}Gamma Team", "sent_id": 1, "pos": [6, 8], "type": "ORG"}],
        [{"name": f"{prefix}Delta Institute", "sent_id": 2, "pos": [0, 2], "type": "ORG"}],
    ]
    labels = [
        {"h": 0, "t": 1, "r": "P1", "evidence": [0]},
        {"h": 0, "t": 2, "r": "P2", "evidence": [1]},
        {"h": 0, "t": 3, "r": "P3", "evidence": [1]},
        {"h": 1, "t": 2, "r": "P2", "evidence": [0, 1]},
        {"h": 4, "t": 1, "r": "P1", "evidence": [2]},
        {"h": 4, "t": 3, "r": "P3", "evidence": []},
    ]
    if include_unknown:
        labels.append({"h": 0, "t": 4, "r": "P999", "evidence": [0, 2]})
    return {
        "title": f"{prefix} document",
        "sents": sents,
        "vertexSet": vertex_set,
        "labels": labels,
    }


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_sources(root, documents_per_split=8):
    relation_map = root / "rel_info.json"
    write_json(relation_map, RELATIONS)
    paths = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.json"
        write_json(
            path,
            [article(f"{split}{index}", include_unknown=index == 0) for index in range(documents_per_split)],
        )
        paths[split] = path
    return paths, relation_map


class ReDocREDDatasetTests(unittest.TestCase):
    def test_adapter_maps_relations_and_builds_verbatim_cross_sentence_evidence(self):
        with workspace_temporary_directory() as root:
            relation_path = root / "relations.json"
            write_json(relation_path, RELATIONS)
            relation_map = load_relation_map(relation_path)
            document, stats = convert_document(
                article("sample", include_unknown=True), "test", 0, relation_map
            )

            self.assertEqual(document["document_id"], "redocred_test_000000")
            self.assertIn(".\nThe researcher", document["text"])
            self.assertEqual(stats["raw_labels"], 7)
            self.assertEqual(stats["valid_triples"], 6)
            self.assertEqual(stats["unknown_or_invalid_relations"], 1)
            self.assertEqual(document["gold_items"][0]["triple"]["predicate"], "works with")
            for item in document["gold_items"]:
                self.assertIn(item["evidence"], document["text"])
                self.assertIn(item["triple"]["subject"], item["evidence"])
                self.assertIn(item["triple"]["object"], item["evidence"])

    def test_sft_uses_every_valid_triple_in_batches_of_at_most_four(self):
        with workspace_temporary_directory() as root:
            paths, relation_map = write_sources(root)
            first = root / "first"
            second = root / "second"
            metadata = build_sft(
                paths["train"],
                paths["validation"],
                paths["test"],
                relation_map,
                first,
                seed=42,
            )
            build_sft(
                paths["train"],
                paths["validation"],
                paths["test"],
                relation_map,
                second,
                seed=42,
            )

            for split in ("train", "validation", "test"):
                rows = list(iter_jsonl(first / f"{split}.jsonl"))
                summary = validate_critic_sft_jsonl(first / f"{split}.jsonl")
                expected_candidates = metadata["conversion"][split]["valid_triples"]
                self.assertEqual(summary["candidates"], expected_candidates)
                self.assertEqual(metadata["splits"][split]["candidates"], expected_candidates)
                self.assertTrue(all(1 <= len(row["triples_info"]) <= 4 for row in rows))
                self.assertTrue(all(row["source_document_id"].startswith(f"redocred_{split}_") for row in rows))
                self.assertEqual(len({row["document_id"] for row in rows}), len(rows))
                self.assertEqual(
                    (first / f"{split}.jsonl").read_bytes(),
                    (second / f"{split}.jsonl").read_bytes(),
                )
            self.assertEqual(metadata["relation_map"]["relations"], 4)

    def test_pipeline_uses_every_test_document_and_all_gold(self):
        with workspace_temporary_directory() as root:
            paths, relation_map = write_sources(root, documents_per_split=3)
            relationless = article("empty")
            relationless["labels"] = []
            write_json(paths["test"], [article("test0"), article("test1"), relationless])
            output = root / "pipeline"
            metadata = build_pipeline(paths["test"], relation_map, output)
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
            self.assertEqual(metadata["statistics"]["valid_gold_triples"], 12)
            self.assertEqual(len(loaded[0]["gold_triples"]), 6)
            self.assertEqual(loaded[2]["gold_triples"], [])
            self.assertEqual(metadata["validation"]["nonempty_gold_documents"], 2)
            self.assertEqual(set(documents[0]), {"document_id", "text"})
            with self.assertRaises(FileExistsError):
                build_pipeline(paths["test"], relation_map, output)


if __name__ == "__main__":
    unittest.main()
