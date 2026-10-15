import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidenceloop.types import Triple
from tests.reference import direct as gpt_runner


class FakeExtractor:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def extract(self, document_id, text):
        self.calls.append(document_id)
        values = self.responses[document_id]
        value = values.pop(0) if isinstance(values, list) else values
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str):
            return gpt_runner.parse_gpt_response(value, document_id, text)
        triples = [
            Triple(
                subject=item["subject"],
                predicate=item["predicate"],
                object=item["object"],
                source_text=text,
                document_id=document_id,
            )
            for item in value
        ]
        return {
            "triples": triples,
            "raw_response": json.dumps({"relations": value}),
            "parse_error": "",
            "rejected_predicates": [],
        }


def write_dataset(root: Path, rows):
    documents = []
    gold = []
    for row in rows:
        documents.append({
            "document_id": row["document_id"],
            "text": row["text"],
        })
        gold.append({
            "document_id": row["document_id"],
            "triples": row["gold_triples"],
        })
    (root / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in documents),
        encoding="utf-8",
    )
    (root / "gold_triples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gold),
        encoding="utf-8",
    )


def one_doc_args(tmpdir: Path, *, skip_soft_metrics=True):
    args = gpt_runner.parse_args([
        "--dataset",
        "rebel_sub",
        "--models",
        "gpt-4o",
        "--limit",
        "1",
    ])
    args.model = "gpt-4o"
    args.skip_soft_metrics = skip_soft_metrics
    return args


class GPTBaselineExperimentTests(unittest.TestCase):
    def test_model_alias_normalizes_gpt_4omini(self):
        self.assertEqual(gpt_runner.normalize_model_name("gpt-4omini"), "gpt-4o-mini")

    def test_ofox_model_id_uses_provider_prefix(self):
        self.assertEqual(gpt_runner.ofox_model_id("gpt-4o"), "openai/gpt-4o")
        self.assertEqual(
            gpt_runner.ofox_model_id("gpt-4omini"),
            "openai/gpt-4o-mini",
        )

    def test_missing_ofox_api_key_has_clear_error(self):
        with patch.dict(
            os.environ,
            {
                "OFOX_API_KEY": "",
                "OFOX_BASE_URL": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "OFOX_API_KEY"):
                gpt_runner.GPTBaselineExtractor("gpt-4o")

    def test_successful_run_writes_kg_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            write_dataset(data_dir, [{
                "document_id": "doc-1",
                "text": "Ada was born in London.",
                "gold_triples": [{
                    "subject": "Ada",
                    "predicate": "place of birth",
                    "object": "London",
                }],
            }])
            preset = gpt_runner.DatasetPreset("rebel_sub", "REBEL_Sub", str(data_dir))
            fake = FakeExtractor({
                "doc-1": [[{
                    "subject": "Ada",
                    "predicate": "place of birth",
                    "object": "London",
                }]],
            })

            with patch.dict(gpt_runner.DATASETS, {"rebel_sub": preset}), patch.object(gpt_runner, "ROOT", Path(tmpdir)):
                run_dir = gpt_runner.run_experiment(
                    one_doc_args(data_dir),
                    extractor=fake,
                )

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            kg_rows = [
                json.loads(line)
                for line in (run_dir / "kg.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["samples_completed"], 1)
        self.assertEqual(summary["exact_match_f1"], 1.0)
        self.assertEqual(kg_rows[0]["subject"], "Ada")
        self.assertNotIn("api_key", json.dumps(state).lower())

    def test_invalid_json_retries_and_writes_failed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            rows = [{
                "document_id": "doc-1",
                "text": "No valid output.",
                "gold_triples": [{
                    "subject": "A",
                    "predicate": "related to",
                    "object": "B",
                }],
            }]
            write_dataset(data_dir, rows)
            preset = gpt_runner.DatasetPreset("rebel_sub", "REBEL_Sub", str(data_dir))
            fake = FakeExtractor({"doc-1": ["not json", "", "still not json"]})

            with patch.dict(gpt_runner.DATASETS, {"rebel_sub": preset}), patch.object(gpt_runner, "ROOT", Path(tmpdir)):
                run_dir = gpt_runner.run_experiment(
                    one_doc_args(data_dir),
                    extractor=fake,
                )

            checkpoint = json.loads(
                (run_dir / "checkpoints" / "000000.json").read_text(encoding="utf-8")
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(fake.calls, ["doc-1", "doc-1", "doc-1"])
        self.assertEqual(checkpoint["status"], "failed")
        self.assertEqual(checkpoint["attempts"], 3)
        self.assertEqual(summary["samples_failed"], 1)

    def test_resume_skips_success_and_retries_failed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            rows = [
                {
                    "document_id": "doc-1",
                    "text": "Ada was born in London.",
                    "gold_triples": [{
                        "subject": "Ada",
                        "predicate": "place of birth",
                        "object": "London",
                    }],
                },
                {
                    "document_id": "doc-2",
                    "text": "Grace worked at Navy.",
                    "gold_triples": [{
                        "subject": "Grace",
                        "predicate": "employer",
                        "object": "Navy",
                    }],
                },
            ]
            write_dataset(data_dir, rows)
            preset = gpt_runner.DatasetPreset("rebel_sub", "REBEL_Sub", str(data_dir))
            first = FakeExtractor({
                "doc-1": [[{
                    "subject": "Ada",
                    "predicate": "place of birth",
                    "object": "London",
                }]],
                "doc-2": ["not json", "not json", "not json"],
            })
            args = gpt_runner.parse_args([
                "--dataset",
                "rebel_sub",
                "--models",
                "gpt-4o",
                "--limit",
                "2",
                "--skip-soft-metrics",
            ])
            args.model = "gpt-4o"

            with patch.dict(gpt_runner.DATASETS, {"rebel_sub": preset}), patch.object(gpt_runner, "ROOT", Path(tmpdir)):
                run_dir = gpt_runner.run_experiment(args, extractor=first)
                resumed = FakeExtractor({
                    "doc-1": AssertionError("successful checkpoint should be skipped"),
                    "doc-2": [[{
                        "subject": "Grace",
                        "predicate": "employer",
                        "object": "Navy",
                    }]],
                })
                resume_args = argparse.Namespace(**vars(args))
                resume_args.resume_run_dir = str(run_dir)
                gpt_runner.run_experiment(resume_args, extractor=resumed)

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            checkpoint = json.loads(
                (run_dir / "checkpoints" / "000001.json").read_text(encoding="utf-8")
            )

        self.assertEqual(resumed.calls, ["doc-2"])
        self.assertEqual(checkpoint["attempts"], 4)
        self.assertEqual(summary["samples_completed"], 2)

    def test_soft_metrics_receives_dataset_entity_order_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            write_dataset(data_dir, [{
                "document_id": "doc-1",
                "text": "A is compared with B.",
                "gold_triples": [{
                    "subject": "A",
                    "predicate": "is compared with",
                    "object": "B",
                }],
            }])
            preset = gpt_runner.DatasetPreset(
                "scierc",
                "SciERC",
                str(data_dir),
                "predicate_aware",
                gpt_runner.SCIERC_SYMMETRIC_PREDICATES,
            )
            fake = FakeExtractor({
                "doc-1": [[{
                    "subject": "B",
                    "predicate": "is compared with",
                    "object": "A",
                }]],
            })
            args = gpt_runner.parse_args([
                "--dataset",
                "scierc",
                "--models",
                "gpt-4omini",
                "--limit",
                "1",
            ])
            args.model = "gpt-4omini"

            with patch.dict(gpt_runner.DATASETS, {"scierc": preset}), patch.object(gpt_runner, "ROOT", Path(tmpdir)):
                with patch("tests.reference.direct.evaluate_run", return_value={"metrics": {}}) as mocked:
                    gpt_runner.run_experiment(args, extractor=fake)

            kwargs = mocked.call_args.kwargs

        self.assertEqual(kwargs["entity_order_mode"], "predicate_aware")
        self.assertEqual(tuple(kwargs["symmetric_predicates"]), gpt_runner.SCIERC_SYMMETRIC_PREDICATES)

    def test_skip_soft_metrics_does_not_call_evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            write_dataset(data_dir, [{
                "document_id": "doc-1",
                "text": "Ada was born in London.",
                "gold_triples": [{
                    "subject": "Ada",
                    "predicate": "place of birth",
                    "object": "London",
                }],
            }])
            preset = gpt_runner.DatasetPreset("biored", "BioRED", str(data_dir), "undirected")
            fake = FakeExtractor({
                "doc-1": [[{
                    "subject": "London",
                    "predicate": "place of birth",
                    "object": "Ada",
                }]],
            })
            args = gpt_runner.parse_args([
                "--dataset",
                "biored",
                "--models",
                "gpt-4o",
                "--limit",
                "1",
                "--skip-soft-metrics",
            ])
            args.model = "gpt-4o"

            with patch.dict(gpt_runner.DATASETS, {"biored": preset}), patch.object(gpt_runner, "ROOT", Path(tmpdir)):
                with patch("tests.reference.direct.evaluate_run") as mocked:
                    run_dir = gpt_runner.run_experiment(args, extractor=fake)

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        mocked.assert_not_called()
        self.assertEqual(summary["exact_match_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
