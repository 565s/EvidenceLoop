from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evidenceloop.reports.plot_sft import (
    ACTIONS,
    ERROR_TYPES,
    SPLITS,
    VARIANTS,
    PlotDataError,
    aggregate_shared_counts,
    create_figure,
    load_report,
    main,
    save_figure,
)


def _report() -> dict:
    # Counts differ by split so the test can detect split aggregation. All four
    # variants share the same manifest and therefore have the same totals.
    variants_statistics = {}
    for variant in VARIANTS:
        variants_statistics[variant] = {}
        for split_index, split in enumerate(SPLITS, start=1):
            variants_statistics[variant][split] = {
                "error_type_counts": {
                    error_type: (error_index + 1) * split_index
                    for error_index, error_type in enumerate(ERROR_TYPES)
                },
                "action_counts": {
                    action: action_value * split_index
                    for action, action_value in zip(ACTIONS, (2, 3, 4, 12))
                },
            }
    return {
        "dataset": "tonytan48/Re-DocRED",
        "variants": list(VARIANTS),
        "splits": list(SPLITS),
        "variants_statistics": variants_statistics,
    }


class PlotCriticSFTStatisticsTests(unittest.TestCase):
    def test_aggregate_sums_three_splits_once(self):
        totals = aggregate_shared_counts(_report())
        self.assertEqual(
            totals["error_type_counts"],
            {error_type: (error_index + 1) * 6 for error_index, error_type in enumerate(ERROR_TYPES)},
        )
        self.assertEqual(
            totals["action_counts"],
            {action: action_value * 6 for action, action_value in zip(ACTIONS, (2, 3, 4, 12))},
        )

    def test_create_figure_has_one_grouped_axis_with_six_bars_and_action_line(self):
        figure, axis = create_figure(_report())
        try:
            self.assertEqual(len(figure.axes), 1)
            self.assertEqual(len(axis.patches), len(ERROR_TYPES))
            self.assertEqual(len(axis.lines), 2)  # action trend and section divider
            action_line = axis.lines[0]
            self.assertEqual(len(action_line.get_xdata()), len(ACTIONS))
            self.assertEqual(
                [label.get_text() for label in axis.get_xticklabels()],
                [*ERROR_TYPES, *ACTIONS],
            )
            self.assertTrue(any("ERROR TYPES" in item.get_text() for item in axis.texts))
            self.assertTrue(any("CRITIC ACTIONS" in item.get_text() for item in axis.texts))
            self.assertEqual(len(axis.get_legend().get_texts()), 2)
        finally:
            figure.clf()

    def test_save_outputs_png_and_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            figure, _ = create_figure(_report())
            try:
                png_path, svg_path = save_figure(
                    figure, output_dir, "chart", dpi=80, write_svg=True
                )
            finally:
                figure.clf()

            self.assertTrue(png_path.is_file())
            self.assertTrue(svg_path.is_file())
            self.assertGreater(png_path.stat().st_size, 1000)
            self.assertGreater(svg_path.stat().st_size, 1000)

    def test_cli_no_svg_writes_merged_chart_and_prints_markdown_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "statistics.json"
            report_path.write_text(json.dumps(_report()), encoding="utf-8")
            output_dir = root / "output"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                result = main([
                    "--report-json", str(report_path),
                    "--output-dir", str(output_dir),
                    "--output-stem", "merged",
                    "--dpi", "80",
                    "--no-svg",
                ])

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / "merged.png").is_file())
            self.assertFalse((output_dir / "merged.svg").exists())
            self.assertIn("![Critic SFT statistics](merged.png)", stdout.getvalue())

    def test_variant_total_mismatch_is_rejected(self):
        report = _report()
        report["variants_statistics"]["fixed4"]["test"]["error_type_counts"]["none"] += 1
        with self.assertRaisesRegex(PlotDataError, "shared candidate manifest"):
            aggregate_shared_counts(report)

    def test_missing_required_variant_and_missing_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = _report()
            report["variants"] = list(VARIANTS[:-1])
            path = root / "incomplete.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaises(PlotDataError):
                load_report(path)
            with self.assertRaises(FileNotFoundError):
                load_report(root / "missing.json")

    def test_missing_field_and_invalid_count_are_rejected(self):
        missing = _report()
        del missing["variants_statistics"]["fixed1"]["train"]["error_type_counts"]["none"]
        invalid = _report()
        invalid["variants_statistics"]["fixed1"]["train"]["action_counts"]["KEEP"] = float("nan")
        for index, report in enumerate((missing, invalid)):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "malformed.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(PlotDataError):
                    load_report(path)


if __name__ == "__main__":
    unittest.main()
