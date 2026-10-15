import json
import shutil
import unittest
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from evidenceloop.data.builders.legacy_phase1 import DatasetBuildError
from evidenceloop.data.builders.rebel_documents import build_dataset


@contextmanager
def workspace_temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rebel_pipeline_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def entity(text, surface):
    start = text.index(surface)
    return {
        "boundaries": [start, start + len(surface)],
        "surfaceform": surface,
    }


def article(document_id, facts=2):
    values = [
        (f"Alpha {document_id}", "works with", f"Beta {document_id}"),
        (f"Gamma {document_id}", "located in", f"Paris {document_id}"),
    ][:facts]
    text = " ".join(f"{subject} relates to {obj}." for subject, _, obj in values)
    triples = []
    for subject, predicate, obj in values:
        triples.append({
            "subject": entity(text, subject),
            "predicate": {"surfaceform": predicate},
            "object": entity(text, obj),
        })
    return {"docid": document_id, "text": text, "triples": triples, "entities": []}


def write_zip(path, articles):
    payload = "".join(json.dumps(value) + "\n" for value in articles)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("en_test.jsonl", payload)


def write_sft_test(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document_id, text in rows:
            handle.write(json.dumps({
                "document_id": f"rebel_{document_id}",
                "source_text": text,
                "review_stage": "initial_review",
                "triples_info": [{"generation_type": "DELETE", "corruption": "object"}],
                "target": {"results": [{"action": "DELETE"}]},
            }) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class RebelPipelineDatasetTests(unittest.TestCase):
    def test_build_preserves_sft_order_and_uses_all_original_gold(self):
        with workspace_temporary_directory() as root:
            articles = [article("doc-a", 2), article("doc-b", 1), article("doc-c", 2)]
            source_zip = root / "rebel.zip"
            write_zip(source_zip, articles)
            critic_test = root / "test.jsonl"
            write_sft_test(critic_test, [
                ("doc-b", articles[1]["text"]),
                ("doc-a", articles[0]["text"]),
            ])
            first = root / "first"
            second = root / "second"

            metadata = build_dataset(source_zip, critic_test, first)
            build_dataset(source_zip, critic_test, second)
            documents = read_jsonl(first / "documents.jsonl")
            gold = read_jsonl(first / "gold_triples.jsonl")

            self.assertEqual(
                [row["document_id"] for row in documents],
                ["rebel_doc-b", "rebel_doc-a"],
            )
            self.assertEqual(set(documents[0]), {"document_id", "text"})
            self.assertEqual(documents[1]["text"], articles[0]["text"])
            self.assertEqual([len(row["triples"]) for row in gold], [1, 2])
            self.assertEqual(gold[1]["triples"][1]["predicate"], "located in")
            self.assertEqual(metadata["statistics"]["documents"], 2)
            self.assertEqual(metadata["statistics"]["valid_gold_triples"], 3)
            for name in ("documents.jsonl", "gold_triples.jsonl", "metadata.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with self.assertRaises(FileExistsError):
                build_dataset(source_zip, critic_test, first)
            build_dataset(source_zip, critic_test, first, overwrite=True)

    def test_missing_duplicate_mismatched_and_goldless_documents_fail(self):
        with workspace_temporary_directory() as root:
            valid = article("doc-a", 1)
            goldless = {"docid": "doc-empty", "text": "No relation here.", "triples": []}
            source_zip = root / "rebel.zip"
            write_zip(source_zip, [valid, goldless])

            cases = {
                "missing": [("doc-missing", "Missing text.")],
                "duplicate": [("doc-a", valid["text"]), ("doc-a", valid["text"])],
                "mismatch": [("doc-a", valid["text"] + " changed")],
                "goldless": [("doc-empty", goldless["text"])],
            }
            for name, rows in cases.items():
                with self.subTest(name=name):
                    critic_test = root / f"{name}.jsonl"
                    write_sft_test(critic_test, rows)
                    with self.assertRaises(DatasetBuildError):
                        build_dataset(source_zip, critic_test, root / f"output-{name}")


if __name__ == "__main__":
    unittest.main()
