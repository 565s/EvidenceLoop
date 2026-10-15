from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


ROOT = Path.cwd()
DEFAULT_REPORT_JSON = ROOT / "reports" / "critic_sft_statistics.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports"
DEFAULT_OUTPUT_STEM = "critic_sft_statistics_chart"

VARIANTS = ("fixed1", "fixed2", "fixed4", "variable")
SPLITS = ("train", "validation", "test")
ERROR_TYPES = ("none", "subject", "object", "predicate", "direction", "predicate+object")
ACTIONS = ("KEEP", "DELETE", "FIX", "REWRITE")

# The first four colors follow the supplied reference image.  Purple and brown
# provide two additional distinct colors for the six error categories.
ERROR_COLORS = {
    "none": "#4C78A8",
    "subject": "#F58518",
    "object": "#54A24B",
    "predicate": "#E45756",
    "direction": "#B279A2",
    "predicate+object": "#9D755D",
}
ACTION_COLORS = {
    "KEEP": "#24558A",
    "DELETE": "#B65D0B",
    "FIX": "#2F7130",
    "REWRITE": "#A83232",
}


class PlotDataError(ValueError):
    """Raised when the statistics report cannot support the requested chart."""


def _require_mapping(value, context: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise PlotDataError(f"{context}: expected a JSON object")
    return value


def _count(mapping: Mapping, key: str, context: str) -> int:
    if key not in mapping:
        raise PlotDataError(f"{context}.{key}: required count is missing")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotDataError(f"{context}.{key}: expected a numeric count")
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise PlotDataError(f"{context}.{key}: expected a non-negative integer")
    return int(value)


def normalize_variants(report):
    """Accept historical pointwise reports without changing the caller's data."""
    report = dict(report)
    statistics = dict(_require_mapping(report.get("variants_statistics"), "variants_statistics"))
    if "pointwise" in statistics:
        legacy = statistics.pop("pointwise")
        if "fixed1" in statistics and statistics["fixed1"] != legacy:
            raise PlotDataError("Conflicting pointwise/fixed1 statistics")
        statistics["fixed1"] = legacy
    report["variants_statistics"] = statistics
    if isinstance(report.get("variants"), list):
        report["variants"] = list(
            dict.fromkeys("fixed1" if x == "pointwise" else x for x in report["variants"])
        )
    return report


def load_report(path: Path) -> Dict:
    """Load and validate the machine-readable counterpart of the Markdown report."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"statistics report not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlotDataError(f"invalid JSON report {path}: {exc}") from exc
    report = dict(_require_mapping(report, str(path)))

    report = normalize_variants(report)
    configured_variants = report.get("variants")
    if configured_variants is not None:
        if not isinstance(configured_variants, list):
            raise PlotDataError("variants: expected a list")
        missing = [variant for variant in VARIANTS if variant not in configured_variants]
        unknown = [variant for variant in configured_variants if variant not in VARIANTS]
        if missing:
            raise PlotDataError(f"variants: missing required variants {missing}")
        if unknown:
            raise PlotDataError(f"variants: unknown variants {unknown}")

    variants_statistics = _require_mapping(report.get("variants_statistics"), "variants_statistics")
    unknown_statistics = [variant for variant in variants_statistics if variant not in VARIANTS]
    if unknown_statistics:
        raise PlotDataError(f"variants_statistics: unknown variants {unknown_statistics}")
    for variant in VARIANTS:
        variant_data = _require_mapping(
            variants_statistics.get(variant), f"variants_statistics.{variant}"
        )
        for split in SPLITS:
            split_data = _require_mapping(
                variant_data.get(split), f"variants_statistics.{variant}.{split}"
            )
            _require_mapping(
                split_data.get("error_type_counts"),
                f"variants_statistics.{variant}.{split}.error_type_counts",
            )
            _require_mapping(
                split_data.get("action_counts"),
                f"variants_statistics.{variant}.{split}.action_counts",
            )
            # Validate every plotted key up front so a partially malformed report
            # cannot produce a plausible-looking but incomplete chart.
            for error_type in ERROR_TYPES:
                _count(
                    split_data["error_type_counts"],
                    error_type,
                    f"variants_statistics.{variant}.{split}.error_type_counts",
                )
            for action in ACTIONS:
                _count(
                    split_data["action_counts"],
                    action,
                    f"variants_statistics.{variant}.{split}.action_counts",
                )
    return report


def aggregate_shared_counts(report: Mapping) -> Dict[str, Dict[str, int]]:
    """Sum the official splits once and ensure variants still share counts."""

    report = normalize_variants(report)
    variants_statistics = _require_mapping(report.get("variants_statistics"), "variants_statistics")
    totals_by_variant = {}
    for variant in VARIANTS:
        variant_data = _require_mapping(
            variants_statistics.get(variant), f"variants_statistics.{variant}"
        )
        error_totals = {key: 0 for key in ERROR_TYPES}
        action_totals = {key: 0 for key in ACTIONS}
        for split in SPLITS:
            split_data = _require_mapping(
                variant_data.get(split), f"variants_statistics.{variant}.{split}"
            )
            error_counts = _require_mapping(
                split_data.get("error_type_counts"),
                f"variants_statistics.{variant}.{split}.error_type_counts",
            )
            action_counts = _require_mapping(
                split_data.get("action_counts"),
                f"variants_statistics.{variant}.{split}.action_counts",
            )
            for error_type in ERROR_TYPES:
                error_totals[error_type] += _count(
                    error_counts,
                    error_type,
                    f"variants_statistics.{variant}.{split}.error_type_counts",
                )
            for action in ACTIONS:
                action_totals[action] += _count(
                    action_counts,
                    action,
                    f"variants_statistics.{variant}.{split}.action_counts",
                )

        if sum(error_totals.values()) != sum(action_totals.values()):
            raise PlotDataError(
                f"{variant}: aggregated error and action candidate totals differ; "
                "the shared candidate manifest may be inconsistent"
            )
        totals_by_variant[variant] = {
            "error_type_counts": error_totals,
            "action_counts": action_totals,
        }

    reference = totals_by_variant["fixed1"]
    for variant in VARIANTS[1:]:
        differences = []
        for field in ("error_type_counts", "action_counts"):
            if totals_by_variant[variant][field] != reference[field]:
                differences.append(field)
        if differences:
            raise PlotDataError(
                f"{variant} aggregated counts differ from pointwise for {differences}; "
                "the variants are expected to share one candidate manifest"
            )
    return reference


def create_figure(report: Mapping) -> Tuple[Figure, Axes]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    """Create a single merged bar/line figure from final candidate counts."""

    totals = aggregate_shared_counts(report)
    error_counts = totals["error_type_counts"]
    action_counts = totals["action_counts"]

    figure, axis = plt.subplots(figsize=(14.2, 7.2))
    error_x = np.arange(len(ERROR_TYPES), dtype=float)
    action_start = len(ERROR_TYPES) + 1.1
    action_x = action_start + np.arange(len(ACTIONS), dtype=float)

    bars = axis.bar(
        error_x,
        [error_counts[key] for key in ERROR_TYPES],
        width=0.68,
        color=[ERROR_COLORS[key] for key in ERROR_TYPES],
        alpha=0.86,
        edgecolor="white",
        linewidth=0.8,
        label="Error type count",
        zorder=3,
    )
    for bar, error_type in zip(bars, ERROR_TYPES):
        count = error_counts[error_type]
        axis.annotate(
            f"{count:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    # One connected line shows the actual action-count changes from one action
    # category to the next. Markers use the reference palette for easy reading.
    action_values = [action_counts[key] for key in ACTIONS]
    axis.plot(
        action_x,
        action_values,
        color="#3D4652",
        linewidth=2.5,
        zorder=5,
        label="Action count",
    )
    for x_value, action, count in zip(action_x, ACTIONS, action_values):
        axis.scatter(
            [x_value],
            [count],
            s=100,
            color=ACTION_COLORS[action],
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
        )
        axis.annotate(
            f"{count:,}",
            (x_value, count),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=ACTION_COLORS[action],
        )

    axis.axvline(len(ERROR_TYPES) - 0.45, color="#9AA0A6", linestyle="--", linewidth=1.0)
    axis.set_xticks([*error_x, *action_x])
    axis.set_xticklabels(
        [*ERROR_TYPES, *ACTIONS],
        rotation=17,
        ha="right",
        fontsize=10,
    )
    axis.set_xlim(-0.65, action_x[-1] + 0.65)
    axis.set_ylim(0, max(max(error_counts.values()), max(action_values)) * 1.16)
    axis.set_ylabel("Candidate count", fontsize=11)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    axis.set_axisbelow(True)

    axis.text(
        0.27,
        1.045,
        "ERROR TYPES · BAR CHART",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#4A5560",
    )
    axis.text(
        0.79,
        1.045,
        "CRITIC ACTIONS · LINE",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#4A5560",
    )
    axis.set_title(
        "Re-DocRED Critic SFT · Merged train / validation / test",
        fontsize=17,
        pad=38,
    )
    axis.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=10)
    figure.text(
        0.5,
        0.015,
        "Final JSONL counts; official splits are summed once. Candidate-set variants share one manifest, "
        "so their counts are not added repeatedly.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    return figure, axis


def save_figure(
    figure: Figure,
    output_dir: Path,
    output_stem: str,
    *,
    dpi: int = 300,
    write_svg: bool = True,
) -> Tuple[Path, Path | None]:
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_stem or Path(output_stem).name != output_stem:
        raise PlotDataError("output-stem must be a simple file name without path separators")
    if dpi <= 0:
        raise PlotDataError("dpi must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg" if write_svg else None
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if svg_path is not None:
        figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    return png_path, svg_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Re-DocRED Critic SFT error and action statistics."
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-svg", action="store_true", help="only write the PNG output")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = load_report(args.report_json)
        figure, _ = create_figure(report)
        png_path, svg_path = save_figure(
            figure,
            args.output_dir,
            args.output_stem,
            dpi=args.dpi,
            write_svg=not args.no_svg,
        )
        import matplotlib.pyplot as plt

        plt.close(figure)
    except (OSError, PlotDataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"PNG: {png_path}")
    if svg_path is not None:
        print(f"SVG: {svg_path}")
    print(f"Markdown: ![Critic SFT statistics]({png_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
