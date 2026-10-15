import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from evidenceloop.evaluation.cs_fs import (
    PROTOCOL,
    build_factualness_prompt,
    compute_cs,
    compute_fs,
    encode_elements,
    evaluate_run,
    load_documents,
    parse_factualness_output,
)
from evidenceloop.evaluation.graphjudge import EvaluationInputError


@contextmanager
def temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"cs_fs_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class FakeEncoder:
    device = "fake-device"
    dtype = "float32"

    def __init__(self, vectors, *, fail=False):
        self.vectors = vectors
        self.fail = fail
        self.calls = []
        self.closed = False

    def encode(self, texts, batch_size):
        if self.fail:
            raise AssertionError("embedding cache should have prevented model inference")
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]

    def close(self):
        self.closed = True


class FakeJudge:
    device = "fake-device"
    dtype = "float32"
    context_limit = 4096

    def __init__(self, outputs=None, *, fail=False, error=None):
        self.outputs = list(outputs or [])
        self.fail = fail
        self.error = error
        self.calls = []
        self.closed = False

    def generate(self, prompts, max_new_tokens):
        if self.error is not None:
            raise self.error
        if self.fail:
            raise AssertionError("FS cache should have prevented model inference")
        self.calls.append((list(prompts), max_new_tokens))
        result = self.outputs[:len(prompts)]
        self.outputs = self.outputs[len(prompts):]
        return result

    def close(self):
        self.closed = True


class CsFsMetricTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {"OPENAI_BASE_URL": "https://example.invalid/v1", "OPENAI_API_KEY": "test-only", "EMBEDDING_MODEL": "test-embedding", "FS_MODEL": "test-judge"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_cs_uses_element_sum_reusable_matching_and_threshold(self):
        gold = {
            "doc": [
                ("gold-a", "rel-a", "tail-a"),
                ("gold-b", "rel-b", "tail-b"),
            ]
        }
        predicted = {"doc": [("pred", "rel-p", "tail-p")]}
        embeddings = {
            "gold-a": [1.0, 0.0],
            "rel-a": [0.0, 0.0],
            "tail-a": [0.0, 0.0],
            "gold-b": [2.0, 0.0],
            "rel-b": [0.0, 0.0],
            "tail-b": [0.0, 0.0],
            "pred": [0.95, (1.0 - 0.95**2) ** 0.5],
            "rel-p": [0.0, 0.0],
            "tail-p": [0.0, 0.0],
        }

        scores, details = compute_cs(gold, predicted, embeddings, threshold=0.95)

        self.assertAlmostEqual(scores["doc"], 1.0)
        self.assertEqual([row["best_prediction_index"] for row in details], [0, 0])
        self.assertTrue(all(row["matched"] for row in details))

    def test_cs_scores_empty_predictions_as_zero(self):
        gold = {"doc": [("a", "r", "b")]}
        embeddings = {"a": [1.0, 0.0], "r": [0.0, 1.0], "b": [0.0, 0.0]}

        scores, details = compute_cs(gold, {"doc": []}, embeddings, threshold=0.95)

        self.assertEqual(scores, {"doc": 0.0})
        self.assertIsNone(details[0]["best_prediction"])
        self.assertFalse(details[0]["matched"])

    def test_factualness_parser_preserves_permissive_true_precedence(self):
        self.assertTrue(parse_factualness_output(" TRUE \n"))
        self.assertFalse(parse_factualness_output("false"))
        for accepted in ("true.", "The answer is true", "false true"):
            self.assertTrue(parse_factualness_output(accepted))
        for invalid in ("", "unknown"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_factualness_output(invalid)

    def test_prompt_contains_full_document_and_triple(self):
        prompt = build_factualness_prompt(
            "Ada was born in London.", ("Ada", "born in", "London")
        )
        self.assertIn("Ada was born in London.", prompt)
        self.assertIn('["Ada", "born in", "London"]', prompt)
        self.assertTrue(prompt.endswith("Factualness:"))

    def test_compute_fs_uses_cache_and_scores_empty_prediction_as_zero(self):
        predictions = {
            "doc-1": [("a", "r", "b"), ("c", "s", "d")],
            "doc-2": [],
        }
        documents = {"doc-1": "Source one.", "doc-2": "Source two."}
        with temporary_directory() as root:
            cache_path = root / "fs_judgments.jsonl"
            judge = FakeJudge(["true", "false"])
            scores, counts, supported, total = compute_fs(
                predictions,
                documents,
                model_id="model-a",
                judge=judge,
                batch_size=2,
                max_new_tokens=8,
                cache_path=cache_path,
            )
            cached_judge = FakeJudge(fail=True)
            cached = compute_fs(
                predictions,
                documents,
                model_id="model-a",
                judge=cached_judge,
                batch_size=2,
                max_new_tokens=8,
                cache_path=cache_path,
            )

        self.assertEqual(scores, {"doc-1": 0.5, "doc-2": 0.0})
        self.assertEqual(counts, {"doc-1": 1, "doc-2": 0})
        self.assertEqual((supported, total), (1, 2))
        self.assertEqual(cached[:4], (scores, counts, supported, total))
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(cached_judge.calls, [])

    def test_duplicate_predictions_share_judgment_but_remain_in_fs_denominator(self):
        triple = ("a", "r", "b")
        predictions = {"doc": [triple, triple]}
        documents = {"doc": "a r b"}
        with temporary_directory() as root:
            judge = FakeJudge(["true"])
            scores, counts, supported, total = compute_fs(
                predictions,
                documents,
                model_id="model",
                judge=judge,
                batch_size=2,
                max_new_tokens=8,
                cache_path=root / "fs.jsonl",
            )

        self.assertEqual(len(judge.calls[0][0]), 1)
        self.assertEqual(scores, {"doc": 1.0})
        self.assertEqual(counts, {"doc": 2})
        self.assertEqual((supported, total), (2, 2))

    def test_embedding_cache_is_invalidated_by_model_fingerprint(self):
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        with temporary_directory() as root:
            cache_path = root / "embeddings.jsonl"
            first = FakeEncoder(vectors)
            encode_elements(
                ["a", "b"],
                model_id="model-a",
                encoder=first,
                batch_size=2,
                cache_path=cache_path,
            )
            same = FakeEncoder(vectors, fail=True)
            encode_elements(
                ["a", "b"],
                model_id="model-a",
                encoder=same,
                batch_size=2,
                cache_path=cache_path,
            )
            changed = FakeEncoder(vectors)
            encode_elements(
                ["a", "b"],
                model_id="model-b",
                encoder=changed,
                batch_size=2,
                cache_path=cache_path,
            )

        self.assertEqual(first.calls, [["a", "b"]])
        self.assertEqual(same.calls, [])
        self.assertEqual(changed.calls, [["a", "b"]])

    def test_evaluate_run_writes_macro_micro_and_protocol_metadata(self):
        source_text = "SECRET_SOURCE_TEXT: Ada was born in London."
        with temporary_directory() as root:
            run_dir = root / "run"
            run_dir.mkdir()
            embedding_model = root / "embedding-model"
            fs_model = root / "fs-model"
            embedding_model.mkdir()
            fs_model.mkdir()
            (embedding_model / "config.json").write_text("{}", encoding="utf-8")
            (fs_model / "config.json").write_text("{}", encoding="utf-8")
            write_jsonl(run_dir / "gold_triples.jsonl", [
                {
                    "document_id": "doc-1",
                    "triples": [["Ada", "born in", "London"]],
                },
                {
                    "document_id": "doc-2",
                    "triples": [["Grace", "worked at", "Navy"]],
                },
                {"document_id": "doc-empty", "triples": []},
            ])
            write_jsonl(run_dir / "documents.jsonl", [
                {"document_id": "doc-1", "text": source_text},
                {"document_id": "doc-2", "text": "Grace worked at the Navy."},
                {"document_id": "doc-empty", "text": "No relation."},
            ])
            write_jsonl(run_dir / "kg.jsonl", [{
                "document_id": "doc-1",
                "subject": "Ada",
                "predicate": "born in",
                "object": "London",
            }])
            vectors = {
                "Ada": [1.0, 0.0],
                "born in": [0.0, 1.0],
                "London": [0.0, 0.0],
                "Grace": [0.0, 1.0],
                "worked at": [1.0, 0.0],
                "Navy": [1.0, 1.0],
            }
            encoder = FakeEncoder(vectors)
            judge = FakeJudge(["true"])

            summary = evaluate_run(
                run_dir,
                metrics="both",
                embedding_model="test-embedding",
                fs_model="test-judge",
                embedding_encoder=encoder,
                factualness_judge=judge,
            )

            output_text = (run_dir / "cs_fs_metrics.json").read_text(encoding="utf-8")
            cache_text = (run_dir / "cs_fs_cache" / "fs_judgments.jsonl").read_text(
                encoding="utf-8"
            )
            per_sample = [
                json.loads(line)
                for line in (run_dir / "cs_fs_per_sample.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary["protocol"], PROTOCOL)
        self.assertAlmostEqual(summary["metrics"]["CS"], 0.5)
        self.assertAlmostEqual(summary["metrics"]["FS"], 0.5)
        self.assertAlmostEqual(summary["metrics"]["FS_micro"], 1.0)
        self.assertEqual(summary["samples"]["skipped_empty_gold"], 1)
        self.assertEqual(summary["samples"]["missing_predictions"], 1)
        self.assertEqual(per_sample[1]["predicted_count"], 0)
        self.assertNotIn(source_text, output_text)
        self.assertNotIn(source_text, cache_text)
        self.assertTrue(encoder.closed)
        self.assertTrue(judge.closed)

    def test_metric_selection_requires_and_loads_only_relevant_model(self):
        with temporary_directory() as root:
            run_dir = root / "run"
            run_dir.mkdir()
            model_dir = root / "model"
            model_dir.mkdir()
            write_jsonl(run_dir / "gold_triples.jsonl", [{
                "document_id": "doc",
                "triples": [["a", "r", "b"]],
            }])
            write_jsonl(run_dir / "documents.jsonl", [{
                "document_id": "doc",
                "text": "a r b",
            }])
            write_jsonl(run_dir / "kg.jsonl", [{
                "document_id": "doc",
                "subject": "a",
                "predicate": "r",
                "object": "b",
            }])
            vectors = {"a": [1.0, 0.0], "r": [0.0, 1.0], "b": [0.0, 0.0]}

            cs_summary = evaluate_run(
                run_dir,
                metrics="cs",
                embedding_model="test-embedding",
                embedding_encoder=FakeEncoder(vectors),
                cache_dir=root / "cs-cache",
                output_path=root / "cs.json",
                per_sample_output_path=root / "cs-samples.jsonl",
                cs_details_output_path=root / "cs-details.jsonl",
            )
            fs_summary = evaluate_run(
                run_dir,
                metrics="fs",
                fs_model="test-judge",
                factualness_judge=FakeJudge(["false"]),
                cache_dir=root / "fs-cache",
                output_path=root / "fs.json",
                per_sample_output_path=root / "fs-samples.jsonl",
            )

        self.assertEqual(set(cs_summary["metrics"]), {"CS"})
        self.assertEqual(set(fs_summary["metrics"]), {"FS", "FS_micro"})

    def test_document_loader_rejects_duplicates_and_missing_documents(self):
        with temporary_directory() as root:
            duplicate = root / "duplicate.jsonl"
            write_jsonl(duplicate, [
                {"document_id": "doc", "text": "one"},
                {"document_id": "doc", "text": "two"},
            ])
            with self.assertRaisesRegex(EvaluationInputError, "duplicate document id"):
                load_documents(duplicate, ["doc"])

            missing = root / "missing.jsonl"
            write_jsonl(missing, [{"document_id": "other", "text": "text"}])
            with self.assertRaisesRegex(EvaluationInputError, "missing 1 required"):
                load_documents(missing, ["doc"])

    def test_invalid_fs_output_is_false_and_runtime_failure_aborts(self):
        predictions = {"doc": [("a", "r", "b")]}
        documents = {"doc": "a r b"}
        with temporary_directory() as root:
            scores, counts, supported, total = compute_fs(
                predictions, documents, model_id="model", judge=FakeJudge(["unknown"]),
                batch_size=1, max_new_tokens=8, cache_path=root / "invalid.jsonl",
            )
            self.assertEqual((scores, counts, supported, total), ({"doc": 0.0}, {"doc": 0}, 0, 1))
            with self.assertRaisesRegex(RuntimeError, "context window"):
                compute_fs(
                    predictions,
                    documents,
                    model_id="model",
                    judge=FakeJudge(error=RuntimeError("context window exceeded")),
                    batch_size=1,
                    max_new_tokens=8,
                    cache_path=root / "context.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
