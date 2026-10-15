"""Plot supplied measurements; no experiment numbers are embedded in the code."""

import argparse
import math
from pathlib import Path

from evidenceloop.data.registry import candidate_strategy
from evidenceloop.io import read_json_object


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop report plot")
    parser.add_argument("--kind", choices=("tradeoff", "distribution"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    data = read_json_object(args.input)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    if args.kind == "tradeoff":
        rows = data.get("measurements", [])
        if not rows:
            raise ValueError("measurements must contain variant, tokens_per_document and f1")
        for row in rows:
            x, y = float(row["tokens_per_document"]), float(row["f1"])
            if not math.isfinite(x) or x < 0 or not math.isfinite(y) or not 0 <= y <= 1:
                raise ValueError("Invalid measurement")
            ax.scatter(x / 1000, y, label=candidate_strategy(row["variant"]), s=80)
        ax.set_xlabel("Average tokens per document (K)")
        ax.set_ylabel(data.get("metric", "F1"))
        ax.legend(title="Variant")
    else:
        errors, actions = data.get("error_counts", {}), data.get("action_counts", {})
        if not errors or not actions:
            raise ValueError("error_counts and action_counts must be nonempty mappings")
        if any(type(v) is not int or v < 0 for v in [*errors.values(), *actions.values()]):
            raise ValueError("Counts must be nonnegative integers")
        ax.bar(range(len(errors)), list(errors.values()), label="Error type", alpha=0.8)
        ax.plot(
            range(len(errors), len(errors) + len(actions)),
            list(actions.values()),
            "o-",
            label="Review action",
        )
        ax.set_xticks(
            range(len(errors) + len(actions)), [*errors, *actions], rotation=25, ha="right"
        )
        ax.axvline(len(errors) - 0.5, linestyle=":", color="gray")
        ax.set_ylabel("Count")
        ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)
    return 0
