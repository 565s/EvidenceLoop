import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from evidenceloop.reports.variants import build_comparison


@contextmanager
def temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"pipeline_compare_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def summary(variant, *, f1, kg, seconds):
    return {
        "runtime_config": {"pipeline_variant": variant},
        "kg_triples": kg,
        "samples_failed": 0,
        "exact_match_precision": f1,
        "exact_match_recall": f1,
        "exact_match_f1": f1,
        "initial_reviews_fallback": 0,
        "critic_parse_error_documents": 0,
        "stage_errors_total": 0,
        "predicted_spo_total": kg,
        "integrator_additions_accepted": 0,
        "initial_critic_actions": {"KEEP": 1},
        "final_critic_actions": {},
        "stage_elapsed_seconds_mean": {
            "proposer": seconds / 2,
            "initial_critic": seconds / 2,
            "integrator": 0,
            "final_critic": 0,
        },
        "samples": [{
            "document_id": "rebel_1",
            "status": "ok",
            "predicted_count": kg,
            "exact_match_count": 1,
        }],
    }


class PipelineVariantComparisonTests(unittest.TestCase):
    def test_comparison_uses_once_minus_full_and_includes_soft_metrics(self):
        with temporary_directory() as root:
            once_dir = root / "once"
            full_dir = root / "full"
            once_dir.mkdir()
            full_dir.mkdir()
            (once_dir / "summary.json").write_text(
                json.dumps(summary("proposer_critic_once", f1=0.6, kg=2, seconds=3)),
                encoding="utf-8",
            )
            (full_dir / "summary.json").write_text(
                json.dumps(summary("full", f1=0.5, kg=3, seconds=8)),
                encoding="utf-8",
            )
            (once_dir / "graphjudge_metrics.json").write_text(
                json.dumps({"metrics": {"G-BS-F1": 0.7}}), encoding="utf-8"
            )
            (full_dir / "graphjudge_metrics.json").write_text(
                json.dumps({"metrics": {"G-BS-F1": 0.6}}), encoding="utf-8"
            )

            value = build_comparison(once_dir, full_dir)

        self.assertEqual(value["delta_definition"], "once_critic - full")
        self.assertEqual(value["delta_once_minus_full"]["kg_triples"], -1.0)
        self.assertEqual(value["delta_once_minus_full"]["exact_match_f1"], 0.1)
        self.assertEqual(value["delta_once_minus_full"]["mean_pipeline_seconds"], -5.0)
        self.assertEqual(
            value["soft_metrics"]["delta_once_minus_full"]["G-BS-F1"],
            0.1,
        )
        self.assertEqual(len(value["per_sample"]), 1)
        self.assertIn("sampling variance", value["interpretation_warning"])

    def test_comparison_rejects_mismatched_ids_or_variants(self):
        with temporary_directory() as root:
            once_dir = root / "once"
            full_dir = root / "full"
            once_dir.mkdir()
            full_dir.mkdir()
            once = summary("full", f1=0.5, kg=1, seconds=1)
            full = summary("full", f1=0.5, kg=1, seconds=1)
            (once_dir / "summary.json").write_text(json.dumps(once), encoding="utf-8")
            (full_dir / "summary.json").write_text(json.dumps(full), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "once-run-dir"):
                build_comparison(once_dir, full_dir)

            once["runtime_config"]["pipeline_variant"] = "proposer_critic_once"
            once["samples"][0]["document_id"] = "different"
            (once_dir / "summary.json").write_text(json.dumps(once), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same ordered document IDs"):
                build_comparison(once_dir, full_dir)


if __name__ == "__main__":
    unittest.main()
