from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evidenceloop.reports.phase2_statistics import (
    DATASETS,
    collect_phase2_dataset_statistics,
    render_phase2_markdown,
)


SPLITS = ("train", "validation", "test")
CORRUPTIONS = ("none", "subject", "object", "direction")


def _record(document_id: str) -> dict:
    actions = ("KEEP", "DELETE", "FIX", "REWRITE")
    source_text = " ".join(f"Subject{i} relation{i} Object{i}." for i in range(4))
    triples_info = []
    results = []
    for index, action in enumerate(actions):
        triple = {
            "subject": f"Subject{index}",
            "predicate": f"relation{index}",
            "object": f"Object{index}",
        }
        triples_info.append({
            "triple_idx": index,
            "origin": "proposer",
            "triple": triple,
            "generation_type": action,
            "corruption": CORRUPTIONS[index],
        })
        results.append({
            "triple_idx": index,
            "action": action,
            "reason": "Supported by the source.",
            "evidence": "" if action == "DELETE" else source_text,
            "triple": triple if action in {"FIX", "REWRITE"} else None,
        })
    return {
        "document_id": document_id,
        "source_text": source_text,
        "review_stage": "initial_review",
        "triples_info": triples_info,
        "target": {"results": results},
    }


class CriticSFTPhase2StatisticsTests(unittest.TestCase):
    def test_all_four_datasets_read_sources_and_count_final_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            for dataset in DATASETS:
                root = data_root / dataset
                root.mkdir()
                metadata = {
                    "dataset": dataset,
                    "seed": 42,
                    "candidate_range": [1, 4],
                    "batching": "at most four candidates",
                    "conversion": {
                        "train": {"raw_labels": 5, "valid_triples": 4, "duplicate_triples": 1},
                        "validation": {"counts": {
                            "raw_relations": 6,
                            "valid_triples": 4,
                            "duplicate_triples": 2,
                        }},
                        "test": {"raw_relations": 4, "valid_triples": 4},
                    },
                    "splits": {
                        "train": {"records": 1, "candidates": 4, "rejected": {"no_valid_gold": 2}},
                        "validation": {"records": 1, "candidates": 4, "skipped_source_documents": 1},
                        "test": {"records": 1, "candidates": 4},
                    },
                }
                if dataset in {"rebel_phase2", "biored_phase2"}:
                    metadata["source_members"] = {
                        "train": "train.member",
                        "validation": "dev.member",
                        "test": "test.member",
                    }
                else:
                    metadata["source_files"] = {
                        "train": {"path": "train.source"},
                        "validation": {"path": "dev.source"},
                        "test": {"path": "test.source"},
                    }
                (root / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                for split in SPLITS:
                    row = _record(f"{dataset}-{split}")
                    if split == "validation":
                        row["target"]["results"][0]["evidence"] = "missing evidence"
                    (root / f"{split}.jsonl").write_text(
                        json.dumps(row) + "\n", encoding="utf-8"
                    )

            reports = [
                collect_phase2_dataset_statistics(dataset, data_root)
                for dataset in DATASETS
            ]
            report = reports[0]
            validation = report["splits_statistics"]["validation"]
            markdown = render_phase2_markdown(reports)

        self.assertEqual(report["seed"], 42)
        self.assertEqual(report["official_split_policy"].startswith("保留来源数据集"), True)
        self.assertIn("确定性随机流", report["randomness_policy"])
        self.assertEqual(
            [item["splits_statistics"]["train"]["official_source"] for item in reports],
            ["train.member", "train.source", "train.source", "train.member"],
        )
        self.assertEqual(report["splits_statistics"]["train"]["records"], 1)
        self.assertEqual(report["splits_statistics"]["train"]["candidate_set_counts"], {
            "1": 0, "2": 0, "3": 0, "4": 1, "other": 0,
        })
        self.assertEqual(report["splits_statistics"]["train"]["action_counts"], {
            "KEEP": 1, "DELETE": 1, "FIX": 1, "REWRITE": 1,
        })
        self.assertEqual(report["splits_statistics"]["train"]["evidence"]["rate"], 1.0)
        self.assertEqual(validation["evidence"]["required_candidates"], 3)
        self.assertEqual(validation["evidence"]["rate"], 2 / 3)
        self.assertEqual(validation["validation"]["failed_records"], 1)
        self.assertEqual(validation["validation"]["failed_candidates"], 4)
        self.assertEqual(
            report["splits_statistics"]["train"]["filtering"]["generation_rejections"]["rejected_rows"],
            2,
        )
        self.assertEqual(
            report["splits_statistics"]["validation"]["filtering"]["source_conversion"]["removed_or_invalid_total"],
            2,
        )
        self.assertIn("train.source", markdown)


if __name__ == "__main__":
    unittest.main()
