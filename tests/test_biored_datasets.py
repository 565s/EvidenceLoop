import json
import shutil
import unittest
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from evidenceloop.experiments.checkpoints import load_canonical_rebel
from evidenceloop.data.builders.biored_sft import build_dataset as build_sft
from evidenceloop.data.builders.biored_documents import build_dataset as build_pipeline
from evidenceloop.data.biored import convert_document, load_relation_map
from evidenceloop.data.sft import iter_jsonl, validate_critic_sft_jsonl


@contextmanager
def workspace_temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"biored_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


RELATIONS = {
    "Association": "associated with",
    "Bind": "binds",
    "Comparison": "compared with",
    "Cotreatment": "co-treated with",
    "Negative_Correlation": "negatively correlated with",
    "Positive_Correlation": "positively correlated with",
}


def _annotation(identifier, entity_type, surface, offset):
    return {
        "id": f"A{offset}",
        "infons": {"identifier": identifier, "type": entity_type},
        "text": surface,
        "locations": [{"offset": offset, "length": len(surface)}],
    }


def document(prefix, include_unknown=False, relationless=False):
    title = f"{prefix} Alpha family and Beta study."
    abstract = (
        "Alpha activates Beta. Gamma is discussed separately. "
        "Later, Delta is linked to Alpha and Beta with Gamma."
    )
    text = title + "\n" + abstract
    abstract_offset = len(title) + 1
    title_annotations = [
        _annotation(f"{prefix}A,{prefix}X", "GeneOrGeneProduct", "Alpha family", text.index("Alpha family")),
        _annotation(f"{prefix}B", "ChemicalEntity", "Beta", text.index("Beta")),
    ]
    abstract_annotations = []
    for identifier, entity_type, surface in (
        (f"{prefix}A", "GeneOrGeneProduct", "Alpha"),
        (f"{prefix}B", "ChemicalEntity", "Beta"),
        (f"{prefix}C", "DiseaseOrPhenotypicFeature", "Gamma"),
        (f"{prefix}D", "SequenceVariant", "Delta"),
    ):
        start = abstract_offset
        while True:
            found = text.find(surface, start)
            if found < 0:
                break
            abstract_annotations.append(
                _annotation(identifier, entity_type, surface, found)
            )
            start = found + len(surface)
    relation_specs = [] if relationless else [
        ("Association", f"{prefix}A", f"{prefix}B", "No"),
        ("Positive_Correlation", f"{prefix}A", f"{prefix}C", "Novel"),
        ("Negative_Correlation", f"{prefix}B", f"{prefix}C", "Novel"),
        ("Bind", f"{prefix}A", f"{prefix}D", "Novel"),
        ("Comparison", f"{prefix}B", f"{prefix}D", "No"),
        ("Cotreatment", f"{prefix}C", f"{prefix}D", "Novel"),
    ]
    if include_unknown:
        relation_specs.append(("Unknown", f"{prefix}A", f"{prefix}D", "Novel"))
    relations = [
        {
            "id": f"R{index}",
            "infons": {
                "entity1": entity1,
                "entity2": entity2,
                "type": relation_type,
                "novel": novelty,
            },
        }
        for index, (relation_type, entity1, entity2, novelty) in enumerate(relation_specs)
    ]
    return {
        "id": prefix,
        "passages": [
            {"offset": 0, "text": title, "annotations": title_annotations},
            {
                "offset": abstract_offset,
                "text": abstract,
                "annotations": abstract_annotations,
            },
        ],
        "relations": relations,
    }


def write_sources(root, documents_per_split=8, relationless_test=False):
    relation_map = root / "relations.json"
    relation_map.write_text(json.dumps(RELATIONS), encoding="utf-8")
    source_zip = root / "BIORED.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for split, member in (
            ("train", "BioRED/Train.BioC.JSON"),
            ("validation", "BioRED/Dev.BioC.JSON"),
            ("test", "BioRED/Test.BioC.JSON"),
        ):
            documents = [
                document(
                    f"{split}{index}",
                    include_unknown=index == 0,
                    relationless=relationless_test and split == "test" and index == documents_per_split - 1,
                )
                for index in range(documents_per_split)
            ]
            archive.writestr(member, json.dumps({"documents": documents}))
    return source_zip, relation_map


class BioREDDatasetTests(unittest.TestCase):
    def test_adapter_reconstructs_offsets_prefers_specific_mentions_and_maps_relations(self):
        with workspace_temporary_directory() as root:
            relation_path = root / "relations.json"
            relation_path.write_text(json.dumps(RELATIONS), encoding="utf-8")
            converted, stats = convert_document(
                document("fixture", include_unknown=True),
                "test",
                load_relation_map(relation_path),
            )

            self.assertEqual(converted["document_id"], "biored_test_fixture")
            self.assertIn("study.\nAlpha activates", converted["text"])
            self.assertEqual(stats["raw_relations"], 7)
            self.assertEqual(stats["valid_triples"], 6)
            self.assertEqual(stats["unknown_or_invalid_relations"], 1)
            association = next(
                item
                for item in converted["gold_items"]
                if item["triple"]["predicate"] == "associated with"
            )
            self.assertEqual(association["triple"]["subject"], "Alpha")
            for item in converted["gold_items"]:
                self.assertIn(item["evidence"], converted["text"])
                self.assertIn(item["triple"]["subject"], item["evidence"])
                self.assertIn(item["triple"]["object"], item["evidence"])

    def test_sft_uses_all_valid_triples_without_direction_corruption(self):
        with workspace_temporary_directory() as root:
            source_zip, relation_map = write_sources(root)
            first = root / "first"
            second = root / "second"
            metadata = build_sft(source_zip, relation_map, first, seed=42)
            build_sft(source_zip, relation_map, second, seed=42)

            for split in ("train", "validation", "test"):
                rows = list(iter_jsonl(first / f"{split}.jsonl"))
                summary = validate_critic_sft_jsonl(first / f"{split}.jsonl")
                expected = metadata["conversion"][split]["valid_triples"]
                self.assertEqual(summary["candidates"], expected)
                self.assertTrue(all(1 <= len(row["triples_info"]) <= 4 for row in rows))
                self.assertTrue(all(
                    row["source_document_id"].startswith(f"biored_{split}_")
                    for row in rows
                ))
                self.assertFalse(any(
                    item.get("corruption") == "direction"
                    for row in rows
                    for item in row["triples_info"]
                ))
                self.assertNotIn("novelty", json.dumps(rows).lower())
                self.assertEqual(
                    (first / f"{split}.jsonl").read_bytes(),
                    (second / f"{split}.jsonl").read_bytes(),
                )

    def test_pipeline_uses_every_test_document_and_canonical_loader(self):
        with workspace_temporary_directory() as root:
            source_zip, relation_map = write_sources(
                root, documents_per_split=3, relationless_test=True
            )
            output = root / "pipeline"
            metadata = build_pipeline(source_zip, relation_map, output)
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
            self.assertEqual(loaded[2]["gold_triples"], [])
            self.assertEqual(set(documents[0]), {"document_id", "text"})


if __name__ == "__main__":
    unittest.main()
