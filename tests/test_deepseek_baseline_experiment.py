import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidenceloop.types import Triple
from tests.reference import deepseek as deepseek_runner


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
            parsed = deepseek_runner.shared.parse_gpt_response(
                value,
                document_id,
                text,
            )
            if parsed.get("parse_error"):
                parsed["parse_error"] = (
                    "DeepSeek response did not contain valid JSON."
                )
            return parsed
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


def make_args(dataset="rebel_sub", limit=1, skip_soft_metrics=True):
    args = deepseek_runner.parse_args([
        "--dataset",
        dataset,
        "--limit",
        str(limit),
    ])
    args.skip_soft_metrics = skip_soft_metrics
    return args


class DeepSeekBaselineExperimentTests(unittest.TestCase):
    def test_default_model_matches_configured_proposer(self):
        args = deepseek_runner.parse_args([])

        self.assertEqual(
            args.model,
            "deepseek-ai/DeepSeek-V4-Flash",
        )
        self.assertEqual(
            deepseek_runner.model_slug(args.model),
            "deepseek-v4-flash",
        )

    def test_missing_siflow_api_key_has_clear_error(self):
        with patch.dict(
            os.environ,
            {"SIFLOW_API_KEY": "", "SIFLOW_BASE_URL": deepseek_runner.DEFAULT_BASE_URL},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "SIFLOW_API_KEY"):
                deepseek_runner.DeepSeekBaselineExtractor(
                    "deepseek-ai/DeepSeek-V4-Flash"
                )

    def test_chat_openai_receives_siliconflow_configuration(self):
        api_key = "test-siflow-key"
        base_url = "https://siliconflow.example/v1"
        with patch.dict(
            os.environ,
            {"SIFLOW_API_KEY": api_key, "SIFLOW_BASE_URL": base_url},
            clear=False,
        ), patch("langchain_openai.ChatOpenAI") as chat_openai:
            deepseek_runner.DeepSeekBaselineExtractor(
                "deepseek-ai/DeepSeek-V4-Flash",
                temperature=0.2,
                max_tokens=1024,
                request_timeout=45,
            )

        kwargs = chat_openai.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(kwargs["base_url"], base_url)
        self.assertEqual(kwargs["api_key"], api_key)
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 1024)
        self.assertEqual(kwargs["timeout"], 45)
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})

    def test_successful_run_writes_outputs_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "dataset"
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
            preset = deepseek_runner.shared.DatasetPreset(
                "rebel_sub",
                "REBEL_Sub",
                str(data_dir),
            )
            fake = FakeExtractor({
                "doc-1": [[{
                    "subject": "Ada",
                    "predicate": "place of birth",
                    "object": "London",
                }]],
            })

            with patch.dict(
                deepseek_runner.DATASETS,
                {"rebel_sub": preset},
            ), patch.object(deepseek_runner, "ROOT", root), patch.dict(
                os.environ,
                {"SIFLOW_API_KEY": "secret-not-for-output"},
                clear=False,
            ):
                run_dir = deepseek_runner.run_experiment(
                    make_args(),
                    extractor=fake,
                )

            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            state_text = (run_dir / "run_state.json").read_text(encoding="utf-8")
            record = json.loads(
                (run_dir / "pipeline_records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            required = {
                "kg.jsonl",
                "kg.json",
                "gold_triples.jsonl",
                "sample_results.jsonl",
                "pipeline_records.jsonl",
                "pipeline_records.json",
                "summary.json",
                "progress.json",
                "run_state.json",
            }
            output_names = {path.name for path in run_dir.iterdir()}

        self.assertEqual(summary["samples_completed"], 1)
        self.assertEqual(summary["exact_match_f1"], 1.0)
        self.assertTrue(required.issubset(output_names))
        self.assertNotIn("secret-not-for-output", state_text)
        self.assertEqual(
            list(record["stage_timings_seconds"]),
            [deepseek_runner.STAGE_NAME],
        )

    def test_invalid_json_retries_and_writes_failed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "dataset"
            data_dir.mkdir()
            write_dataset(data_dir, [{
                "document_id": "doc-1",
                "text": "No valid output.",
                "gold_triples": [],
            }])
            preset = deepseek_runner.shared.DatasetPreset(
                "rebel_sub",
                "REBEL_Sub",
                str(data_dir),
            )
            fake = FakeExtractor({
                "doc-1": ["not json", "", "still not json"],
            })

            with patch.dict(
                deepseek_runner.DATASETS,
                {"rebel_sub": preset},
            ), patch.object(deepseek_runner, "ROOT", root), patch.object(
                deepseek_runner.shared,
                "PROPOSER_RETRY_DELAYS",
                (0.0, 0.0),
            ):
                run_dir = deepseek_runner.run_experiment(
                    make_args(),
                    extractor=fake,
                )

            checkpoint = json.loads(
                (run_dir / "checkpoints" / "000000.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(fake.calls, ["doc-1", "doc-1", "doc-1"])
        self.assertEqual(checkpoint["status"], "failed")
        self.assertEqual(checkpoint["attempts"], 3)
        self.assertIn("DeepSeek response", checkpoint["error"])

    def test_resume_skips_success_and_retries_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "dataset"
            data_dir.mkdir()
            rows = [
                {
                    "document_id": "doc-1",
                    "text": "A relates to B.",
                    "gold_triples": [{
                        "subject": "A",
                        "predicate": "relates to",
                        "object": "B",
                    }],
                },
                {
                    "document_id": "doc-2",
                    "text": "C relates to D.",
                    "gold_triples": [{
                        "subject": "C",
                        "predicate": "relates to",
                        "object": "D",
                    }],
                },
            ]
            write_dataset(data_dir, rows)
            preset = deepseek_runner.shared.DatasetPreset(
                "rebel_sub",
                "REBEL_Sub",
                str(data_dir),
            )
            first = FakeExtractor({
                "doc-1": [[{
                    "subject": "A",
                    "predicate": "relates to",
                    "object": "B",
                }]],
                "doc-2": ["bad", "bad", "bad"],
            })
            args = make_args(limit=2)

            with patch.dict(
                deepseek_runner.DATASETS,
                {"rebel_sub": preset},
            ), patch.object(deepseek_runner, "ROOT", root), patch.object(
                deepseek_runner.shared,
                "PROPOSER_RETRY_DELAYS",
                (0.0, 0.0),
            ):
                run_dir = deepseek_runner.run_experiment(args, extractor=first)
                resumed = FakeExtractor({
                    "doc-1": AssertionError("successful checkpoint was rerun"),
                    "doc-2": [[{
                        "subject": "C",
                        "predicate": "relates to",
                        "object": "D",
                    }]],
                })
                resume_args = argparse.Namespace(**vars(args))
                resume_args.resume_run_dir = str(run_dir)
                deepseek_runner.run_experiment(
                    resume_args,
                    extractor=resumed,
                )

            checkpoint = json.loads(
                (run_dir / "checkpoints" / "000001.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(resumed.calls, ["doc-2"])
        self.assertEqual(checkpoint["attempts"], 4)
        self.assertEqual(summary["samples_completed"], 2)

    def test_dataset_entity_order_reaches_soft_metrics(self):
        cases = [
            ("biored", "undirected", ()),
            (
                "scierc",
                "predicate_aware",
                deepseek_runner.shared.SCIERC_SYMMETRIC_PREDICATES,
            ),
        ]
        for dataset_key, order_mode, symmetric in cases:
            with self.subTest(dataset=dataset_key), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                data_dir = root / "dataset"
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
                preset = deepseek_runner.shared.DatasetPreset(
                    dataset_key,
                    dataset_key,
                    str(data_dir),
                    order_mode,
                    symmetric,
                )
                fake = FakeExtractor({
                    "doc-1": [[{
                        "subject": "B",
                        "predicate": "is compared with",
                        "object": "A",
                    }]],
                })
                args = make_args(
                    dataset=dataset_key,
                    skip_soft_metrics=False,
                )

                with patch.dict(
                    deepseek_runner.DATASETS,
                    {dataset_key: preset},
                ), patch.object(deepseek_runner, "ROOT", root), patch.object(
                    deepseek_runner.shared,
                    "evaluate_soft_metrics",
                    return_value={},
                ) as evaluator:
                    run_dir = deepseek_runner.run_experiment(
                        args,
                        extractor=fake,
                    )

                evaluated_dataset = evaluator.call_args.args[1]
                summary = json.loads(
                    (run_dir / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(evaluated_dataset.entity_order_mode, order_mode)
                self.assertEqual(
                    tuple(evaluated_dataset.symmetric_predicates),
                    tuple(symmetric),
                )
                self.assertEqual(summary["exact_match_f1"], 1.0)

    def test_skip_soft_metrics_does_not_call_evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "dataset"
            data_dir.mkdir()
            write_dataset(data_dir, [{
                "document_id": "doc-1",
                "text": "A relates to B.",
                "gold_triples": [],
            }])
            preset = deepseek_runner.shared.DatasetPreset(
                "rebel_sub",
                "REBEL_Sub",
                str(data_dir),
            )
            fake = FakeExtractor({"doc-1": [[]]})

            with patch.dict(
                deepseek_runner.DATASETS,
                {"rebel_sub": preset},
            ), patch.object(deepseek_runner, "ROOT", root), patch.object(
                deepseek_runner.shared,
                "evaluate_soft_metrics",
            ) as evaluator:
                deepseek_runner.run_experiment(
                    make_args(skip_soft_metrics=True),
                    extractor=fake,
                )

        evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
