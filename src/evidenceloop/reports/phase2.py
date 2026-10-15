from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evidenceloop.reports.phase2_statistics import (  # noqa: E402
    DATA_ROOT,
    DATASETS,
    collect_phase2_dataset_statistics,
    render_phase2_markdown,
)

ROOT = Path.cwd()


DEFAULT_JSON_OUTPUT = ROOT / "reports" / "critic_sft_phase2_statistics.json"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "reports" / "critic_sft_phase2_statistics.md"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report actual Critic SFT phase-two statistics for the four datasets."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        reports = [
            collect_phase2_dataset_statistics(dataset, args.data_root) for dataset in args.datasets
        ]
        report = {
            "title": "Critic SFT Phase 2 statistics",
            "datasets": reports,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.output_markdown.write_text(
            render_phase2_markdown(reports),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "json": str(args.output_json.resolve()),
                "markdown": str(args.output_markdown.resolve()),
                "datasets": [
                    {
                        "dataset": item["dataset_key"],
                        "seed": item["seed"],
                        "records": {
                            split: item["splits_statistics"][split]["records"]
                            for split in item["splits"]
                        },
                        "validation_failures": {
                            split: item["splits_statistics"][split]["validation"]["failed_records"]
                            for split in item["splits"]
                        },
                    }
                    for item in reports
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
