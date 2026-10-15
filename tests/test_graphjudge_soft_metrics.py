from contextlib import contextmanager
import json
import shutil
import unittest
from unittest.mock import patch
import uuid
from pathlib import Path

import numpy as np

from evidenceloop.evaluation.graphjudge import (
    EvaluationInputError,
    METRIC_NAMES,
    aggregate_metrics,
    aligned_scores,
    compute_bert_metrics,
    evaluate_run,
    load_gold_graphs,
    load_gold_graphs_with_metadata,
    load_prediction_graphs,
    normalize_entity_order,
    official_edge_text,
    reusable_scores,
    resolve_prediction_path,
)


@contextmanager
def temporary_directory():
    root = Path.cwd() / ".test_tmp"
    path = root / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            root.rmdir()
        except OSError:
            pass


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class GraphJudgeSoftMetricTests(unittest.TestCase):
    def test_aligned_scores_use_optimal_one_to_one_matching(self):
        accuracy, recall, f1, matched_sum = aligned_scores([
            [0.9, 0.8],
            [0.85, 0.1],
            [0.2, 0.7],
        ])

        self.assertAlmostEqual(matched_sum, 1.65)
        self.assertAlmostEqual(accuracy, 1.65 / 3)
        self.assertAlmostEqual(recall, 1.65 / 2)
        self.assertAlmostEqual(f1, 0.66)

    def test_reusable_scores_allow_gold_and_predictions_to_match_more_than_once(self):
        accuracy, recall, f1, accuracy_sum, recall_sum = reusable_scores([
            [0.9, 0.8],
            [0.85, 0.1],
            [0.2, 0.7],
        ])

        self.assertAlmostEqual(accuracy_sum, 2.45)
        self.assertAlmostEqual(recall_sum, 1.7)
        self.assertAlmostEqual(accuracy, 2.45 / 3)
        self.assertAlmostEqual(recall, 1.7 / 2)
        self.assertAlmostEqual(f1, 2 * (2.45 / 3) * (1.7 / 2) / ((2.45 / 3) + (1.7 / 2)))

    def test_aggregate_metrics_average_per_sample_f1(self):
        first = {name: 0.0 for name in METRIC_NAMES}
        second = {name: 1.0 for name in METRIC_NAMES}
        first.update({"G-BS-Acc": 0.2, "G-BS-Recall": 0.8, "G-BS-F1": 0.32})
        second.update({"G-BS-Acc": 0.8, "G-BS-Recall": 0.2, "G-BS-F1": 0.32})

        result = aggregate_metrics([first, second])

        self.assertAlmostEqual(result["G-BS-Acc"], 0.5)
        self.assertAlmostEqual(result["G-BS-Recall"], 0.5)
        self.assertAlmostEqual(result["G-BS-F1"], 0.32)

    def test_loaders_accept_id_aliases_and_both_prediction_formats(self):
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold_triples.jsonl"
            json_path = root / "kg.json"
            jsonl_path = root / "kg.jsonl"
            write_jsonl(gold_path, [{
                "document_id": "doc-1",
                "triples": [["Ada", "born in", "London"]],
            }])
            prediction = {
                "sample_id": "doc-1",
                "subject": "Ada",
                "predicate": "born in",
                "object": "London",
            }
            json_path.write_text(json.dumps([prediction]), encoding="utf-8")
            write_jsonl(jsonl_path, [prediction])

            gold = load_gold_graphs(gold_path)
            from_json, missing_json = load_prediction_graphs(json_path, list(gold))
            from_jsonl, missing_jsonl = load_prediction_graphs(jsonl_path, list(gold))

        self.assertEqual(from_json, from_jsonl)
        self.assertEqual(from_json["doc-1"], [("Ada", "born in", "London")])
        self.assertEqual(missing_json, [])
        self.assertEqual(missing_jsonl, [])

    def test_missing_prediction_gets_official_null_triple(self):
        with temporary_directory() as tmpdir:
            path = Path(tmpdir) / "kg.json"
            path.write_text("[]", encoding="utf-8")
            graphs, missing = load_prediction_graphs(path, ["doc-1"])

        self.assertEqual(graphs["doc-1"], [("Null", "Null", "Null")])
        self.assertEqual(missing, ["doc-1"])

    def test_unknown_prediction_id_is_rejected(self):
        with temporary_directory() as tmpdir:
            path = Path(tmpdir) / "kg.json"
            path.write_text(json.dumps([{
                "sample_id": "other",
                "subject": "Ada",
                "predicate": "born in",
                "object": "London",
            }]), encoding="utf-8")

            with self.assertRaises(EvaluationInputError):
                load_prediction_graphs(path, ["doc-1"])

    def test_empty_gold_and_malformed_prediction_are_rejected(self):
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold_triples.jsonl"
            prediction_path = root / "kg.json"
            write_jsonl(gold_path, [{"sample_id": "doc-1", "triples": []}])
            prediction_path.write_text(json.dumps([{
                "sample_id": "doc-1",
                "subject": "Ada",
                "predicate": "",
                "object": "London",
            }]), encoding="utf-8")

            with self.assertRaises(EvaluationInputError):
                load_gold_graphs(gold_path)
            with self.assertRaises(EvaluationInputError):
                load_prediction_graphs(prediction_path, ["doc-1"])

    def test_empty_gold_can_be_skipped_and_predictions_for_it_are_ignored(self):
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold_triples.jsonl"
            prediction_path = root / "kg.json"
            write_jsonl(gold_path, [
                {
                    "document_id": "doc-1",
                    "triples": [["Ada", "born in", "London"]],
                },
                {
                    "document_id": "doc-empty",
                    "triples": [],
                },
            ])
            prediction_path.write_text(json.dumps([
                {
                    "sample_id": "doc-1",
                    "subject": "Ada",
                    "predicate": "born in",
                    "object": "London",
                },
                {
                    "sample_id": "doc-empty",
                    "subject": "Extra",
                    "predicate": "ignored for",
                    "object": "Soft metrics",
                },
            ]), encoding="utf-8")

            gold, skipped_ids, total_rows = load_gold_graphs_with_metadata(
                gold_path,
                empty_gold_policy="skip",
            )
            predictions, missing = load_prediction_graphs(
                prediction_path,
                list(gold),
                ignored_ids=skipped_ids,
            )

        self.assertEqual(total_rows, 2)
        self.assertEqual(skipped_ids, ["doc-empty"])
        self.assertEqual(list(gold), ["doc-1"])
        self.assertEqual(list(predictions), ["doc-1"])
        self.assertEqual(missing, [])

    def test_resolve_prediction_path_prefers_json(self):
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            (root / "kg.json").write_text("[]", encoding="utf-8")
            (root / "kg.jsonl").write_text("", encoding="utf-8")

            selected = resolve_prediction_path(root)

        self.assertEqual(selected.name, "kg.json")

    def test_bert_backend_is_mockable_without_model_download(self):
        def fake_score(**kwargs):
            self.assertEqual(kwargs["model_type"], "bert-base-uncased")
            self.assertEqual(len(kwargs["cands"]), 6)
            return None, None, np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.7])

        result = compute_bert_metrics(
            gold_graphs=[[('g1', 'r', 'o'), ('g2', 'r', 'o')]],
            predicted_graphs=[[('p1', 'r', 'o'), ('p2', 'r', 'o'), ('p3', 'r', 'o')]],
            model_type="bert-base-uncased",
            device="cpu",
            batch_size=4,
            score_backend=fake_score,
        )[0]

        self.assertAlmostEqual(result["G-BS-Matched-Sum"], 1.65)
        self.assertAlmostEqual(result["G-BS-Acc"], 1.65 / 3)
        self.assertAlmostEqual(result["G-BS-Recall"], 1.65 / 2)
        self.assertAlmostEqual(result["G-BS-F1"], 0.66)

    def test_bert_reusable_matching_uses_directional_best_scores(self):
        def fake_score(**kwargs):
            return None, None, np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.7])

        result = compute_bert_metrics(
            gold_graphs=[[('g1', 'r', 'o'), ('g2', 'r', 'o')]],
            predicted_graphs=[[('p1', 'r', 'o'), ('p2', 'r', 'o'), ('p3', 'r', 'o')]],
            model_type="bert-base-uncased",
            device="cpu",
            batch_size=4,
            score_backend=fake_score,
            matching_mode="reusable",
        )[0]

        self.assertAlmostEqual(result["G-BS-Accuracy-Sum"], 2.45)
        self.assertAlmostEqual(result["G-BS-Recall-Sum"], 1.7)
        self.assertAlmostEqual(result["G-BS-Acc"], 2.45 / 3)
        self.assertAlmostEqual(result["G-BS-Recall"], 1.7 / 2)

    def test_official_edge_text_preserves_character_level_join(self):
        text = official_edge_text(("Ada", "born in", "London"))

        self.assertTrue(text.startswith("[;';a;d;a;';,"))
        self.assertIn("b;o;r;n; ;i;n", text)

    def test_undirected_entity_order_canonicalizes_reversed_triples(self):
        forward = normalize_entity_order(("Alpha", "binds", "Beta"), "undirected")
        reverse = normalize_entity_order(("Beta", "binds", "Alpha"), "undirected")

        self.assertEqual(forward, reverse)
        self.assertNotEqual(
            normalize_entity_order(("Alpha", "binds", "Beta"), "directed"),
            normalize_entity_order(("Beta", "binds", "Alpha"), "directed"),
        )

    def test_evaluate_run_writes_summary_and_per_sample_outputs(self):
        bert_result = [{
            "G-BS-Acc": 0.7,
            "G-BS-Recall": 0.8,
            "G-BS-F1": 0.7466666667,
            "G-BS-Matched-Sum": 0.8,
        }]
        lexical_result = [{
            "G-BL-Acc": 0.4,
            "G-BL-Recall": 0.5,
            "G-BL-F1": 0.4444444444,
            "G-BL-Matched-Sum": 0.5,
            "G-RO-Acc": 0.3,
            "G-RO-Recall": 0.4,
            "G-RO-F1": 0.3428571429,
            "G-RO-Matched-Sum": 0.4,
        }]
        with temporary_directory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold_triples.jsonl"
            prediction_path = root / "kg.json"
            output_path = root / "metrics.json"
            per_sample_path = root / "per_sample.jsonl"
            write_jsonl(gold_path, [
                {
                    "document_id": "doc-1",
                    "triples": [["Ada", "born in", "London"]],
                },
                {
                    "document_id": "doc-empty",
                    "triples": [],
                },
            ])
            prediction_path.write_text(json.dumps([
                {
                    "sample_id": "doc-1",
                    "subject": "Ada",
                    "predicate": "born in",
                    "object": "London",
                },
                {
                    "sample_id": "doc-empty",
                    "subject": "Ignored",
                    "predicate": "has",
                    "object": "Prediction",
                },
            ]), encoding="utf-8")

            with (
                patch("evidenceloop.evaluation.graphjudge.ensure_runtime_dependencies"),
                patch("evidenceloop.evaluation.graphjudge.resolve_device", return_value="cpu"),
                patch(
                    "evidenceloop.evaluation.graphjudge.compute_bert_metrics",
                    return_value=bert_result,
                ),
                patch(
                    "evidenceloop.evaluation.graphjudge.compute_bleu_rouge_metrics",
                    return_value=lexical_result,
                ),
            ):
                summary = evaluate_run(
                    run_dir=root,
                    gold_path=gold_path,
                    prediction_path=prediction_path,
                    output_path=output_path,
                    per_sample_output_path=per_sample_path,
                    empty_gold_policy="skip",
                )

            saved_summary = json.loads(output_path.read_text(encoding="utf-8"))
            sample_rows = [json.loads(line) for line in per_sample_path.read_text().splitlines()]

        self.assertEqual(summary["protocol"], "graphjudge_official_v1")
        self.assertEqual(summary["samples_total_input"], 2)
        self.assertEqual(summary["samples_evaluated"], 1)
        self.assertEqual(summary["samples_skipped_empty_gold"], 1)
        self.assertEqual(summary["skipped_empty_gold_ids"], ["doc-empty"])
        self.assertEqual(saved_summary["metrics"]["G-BS-Acc"], 0.7)
        self.assertEqual(len(sample_rows), 1)
        self.assertEqual(sample_rows[0]["document_id"], "doc-1")
        self.assertFalse(sample_rows[0]["prediction_missing"])


if __name__ == "__main__":
    unittest.main()
