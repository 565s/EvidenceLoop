from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evidenceloop.reports.sft_statistics import collect_statistics, render_markdown  # noqa: E402

ROOT = Path.cwd()


DEFAULT_DATA_ROOT = ROOT / "data" / "critic_sft" / "re_docred_candidate_set_ablation"
DEFAULT_JSON_OUTPUT = ROOT / "reports" / "critic_sft_statistics.json"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "reports" / "critic_sft_statistics.md"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report actual Re-DocRED Critic candidate-set SFT statistics."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = collect_statistics(args.data_root)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.output_markdown.write_text(
            render_markdown(report),
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
                "seed": report.get("seed"),
                "variants": report.get("variants", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
