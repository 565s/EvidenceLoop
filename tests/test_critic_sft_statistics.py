from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evidenceloop.reports.sft_statistics import (
    collect_split_statistics,
    collect_statistics,
    render_markdown,
)


def _triple(subject: str, predicate: str, obj: str) -> dict:
    return {"subject": subject, "predicate": predicate, "object": obj}


def _record(
    actions=("KEEP", "DELETE", "FIX", "REWRITE"), *, bad_evidence=False, document_id="doc-1"
) -> dict:
    triples = []
    results = []
    for index, action in enumerate(actions):
        candidate = _triple(f"Alice{index}", "relates to", f"Bob{index}")
        triples.append({
            "triple_idx": index,
            "origin": "proposer",
            "triple": candidate,
            "generation_type": action,
            "corruption": "none" if action == "KEEP" else "subject",
        })
        evidence = "Alice met Bob."
        if action == "DELETE":
            evidence = ""
        elif bad_evidence and index == 0:
            evidence = "not in source"
        corrected = (
            _triple(f"Alice{index}", "relates to", f"Bob{index}")
            if action in {"FIX", "REWRITE"}
            else None
        )
        results.append({
            "triple_idx": index,
            "action": action,
            "reason": "Supported by the source.",
            "evidence": evidence,
            "triple": corrected,
        })
    return {
        "document_id": document_id,
        "source_text": "Alice met Bob.",
        "review_stage": "initial_review",
        "triples_info": triples,
        "target": {"results": results},
    }


class CriticSFTStatisticsTests(unittest.TestCase):
    def test_counts_evidence_and_missing_candidate_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            rows = [
                _record(
                    actions=("KEEP", "DELETE", "FIX", "REWRITE")[:size],
                    document_id=f"doc-{size}",
                )
                for size in range(1, 5)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            stats = collect_split_statistics(path)

        self.assertEqual(stats["records"], 4)
        self.assertEqual(stats["candidates"], 10)
        self.assertEqual(stats["candidate_set_counts"], {"1": 1, "2": 1, "3": 1, "4": 1, "other": 0})
        self.assertEqual(stats["action_counts"], {
            "KEEP": 4,
            "DELETE": 3,
            "FIX": 2,
            "REWRITE": 1,
        })
        self.assertEqual(stats["evidence"]["required_candidates"], 7)
        self.assertEqual(stats["evidence"]["nonempty_and_locatable"], 7)
        self.assertEqual(stats["evidence"]["rate"], 1.0)
        self.assertEqual(stats["evidence"]["delete_empty"], 3)
        self.assertEqual(stats["validation"]["failed_records"], 0)

    def test_validation_failure_counts_record_and_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            path.write_text(
                json.dumps(_record(actions=("KEEP",), bad_evidence=True))
                + "\n{not json}\n",
                encoding="utf-8",
            )

            stats = collect_split_statistics(path)

        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["candidates"], 1)
        self.assertEqual(stats["validation"]["failed_records"], 2)
        self.assertEqual(stats["validation"]["failed_candidates"], 1)
        self.assertGreaterEqual(len(stats["validation"]["error_types"]), 1)

    def test_duplicate_document_ids_are_counted_as_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            row = _record(actions=("KEEP",))
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )

            stats = collect_split_statistics(path)

        self.assertEqual(stats["validation"]["failed_records"], 1)
        self.assertEqual(stats["validation"]["failed_candidates"], 1)
        self.assertEqual(stats["validation"]["final_invalid_records"], 1)

    def test_collect_statistics_reads_seed_split_policy_and_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "dataset": "tonytan48/Re-DocRED",
                "purpose": "controlled Critic candidate-set ablation",
                "seed": 42,
                "variants": ["fixed1", "fixed2", "fixed4", "variable"],
                "candidate_range": [1, 4],
                "common_document_policy": "valid gold triple count >= 4",
                "source_files": {
                    "train": {"path": "train_revised.json"},
                    "validation": {"path": "dev_revised.json"},
                    "test": {"path": "test_revised.json"},
                },
                "conversion": {
                    "train": {"raw_labels": 10, "valid_triples": 9, "duplicate_triples": 1},
                    "validation": {"raw_labels": 2, "valid_triples": 2},
                    "test": {"raw_labels": 2, "valid_triples": 2},
                },
                "splits": {
                    split: {
                        "source_documents": 1,
                        "eligible_documents": 1,
                        "excluded_documents_lt4": 0,
                        "excluded_gold_triples": 0,
                        "dropped_tail_gold_triples": 0,
                    }
                    for split in ("train", "validation", "test")
                },
                "candidate_manifest": {"path": str(root / "candidate_manifest.jsonl")},
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            manifest_row = {
                "split": "train",
                "document_id": "doc-1",
                "source_text": "Alice met Bob.",
                "action": "KEEP",
                "evidence": "Alice met Bob.",
                "corruption": "none",
            }
            (root / "candidate_manifest.jsonl").write_text(
                json.dumps(manifest_row) + "\n", encoding="utf-8"
            )
            for variant in ("fixed1", "fixed2", "fixed4", "variable"):
                for split in ("train", "validation", "test"):
                    output = root / variant / f"{split}.jsonl"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(_record(actions=("KEEP",))) + "\n", encoding="utf-8")

            report = collect_statistics(root)
            markdown = render_markdown(report)

        self.assertEqual(report["seed"], 42)
        self.assertEqual(report["source_split_files"]["validation"], "dev_revised.json")
        self.assertEqual(report["variants"], ["fixed1", "fixed2", "fixed4", "variable"])
        self.assertEqual(report["shared_candidate_manifest"]["candidates"], 1)
        self.assertEqual(
            report["shared_candidate_manifest"]["splits"]["train"]["action_counts"]["KEEP"],
            1,
        )
        self.assertEqual(report["source_filtering"]["train"]["conversion"]["removed_total"], 1)
        self.assertIn("候选集合统计报告", markdown)
        self.assertIn("fixed1", markdown)


if __name__ == "__main__":
    unittest.main()
