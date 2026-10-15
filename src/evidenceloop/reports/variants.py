from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from evidenceloop.reports.runs import load_summary

ROOT = Path.cwd()
EXACT_FIELDS = (
    "kg_triples",
    "samples_failed",
    "exact_match_precision",
    "exact_match_recall",
    "exact_match_f1",
    "initial_reviews_fallback",
    "critic_parse_error_documents",
    "stage_errors_total",
)


def resolve_run_dir(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def pipeline_variant(summary: Mapping) -> str:
    runtime = summary.get("runtime_config") or {}
    return str(runtime.get("pipeline_variant") or "full")


def mean_pipeline_seconds(summary: Mapping) -> float:
    timings = summary.get("stage_elapsed_seconds_mean") or {}
    return round(sum(float(value or 0.0) for value in timings.values()), 3)


def compact(run_dir: Path, summary: Mapping) -> Dict:
    return {
        "run_dir": str(run_dir),
        "pipeline_variant": pipeline_variant(summary),
        **{
            field: summary.get(field) if field.startswith("exact_match_") else summary.get(field, 0)
            for field in EXACT_FIELDS
        },
        "predicted_spo_total": summary.get("predicted_spo_total", 0),
        "integrator_additions_accepted": summary.get("integrator_additions_accepted", 0),
        "initial_critic_actions": summary.get("initial_critic_actions", {}),
        "final_critic_actions": summary.get("final_critic_actions", {}),
        "stage_elapsed_seconds_mean": summary.get("stage_elapsed_seconds_mean", {}),
        "mean_pipeline_seconds": mean_pipeline_seconds(summary),
    }


def _ordered_ids(summary: Mapping) -> list:
    return [str(row.get("document_id") or "") for row in summary.get("samples", [])]


def _soft_metrics(run_dir: Path) -> Optional[Dict]:
    path = run_dir / "graphjudge_metrics.json"
    if not path.exists():
        return None
    value = read_json(path)
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: metrics must be an object")
    return metrics


def _numeric_delta(left: Mapping, right: Mapping, fields: Sequence[str]) -> Dict:
    return {
        field: (
            round(float(left[field]) - float(right[field]), 4)
            if left.get(field) is not None and right.get(field) is not None
            else None
        )
        for field in fields
    }


def build_comparison(once_run_dir: Path, full_run_dir: Path) -> Dict:
    once_summary = load_summary(once_run_dir)
    full_summary = load_summary(full_run_dir)
    if pipeline_variant(once_summary) != "proposer_critic_once":
        raise ValueError("--once-run-dir is not a proposer_critic_once run")
    if pipeline_variant(full_summary) != "full":
        raise ValueError("--full-run-dir is not a full pipeline run")

    once_ids = _ordered_ids(once_summary)
    full_ids = _ordered_ids(full_summary)
    if not once_ids or once_ids != full_ids:
        raise ValueError("pipeline variant runs must contain the same ordered document IDs")

    once_compact = compact(once_run_dir, once_summary)
    full_compact = compact(full_run_dir, full_summary)
    delta_fields = (*EXACT_FIELDS, "predicted_spo_total", "mean_pipeline_seconds")
    once_soft = _soft_metrics(once_run_dir)
    full_soft = _soft_metrics(full_run_dir)
    soft_comparison = None
    if once_soft is not None and full_soft is not None:
        common_fields = sorted(set(once_soft) & set(full_soft))
        soft_comparison = {
            "once_critic": once_soft,
            "full": full_soft,
            "delta_once_minus_full": _numeric_delta(once_soft, full_soft, common_fields),
        }

    once_samples = {row["document_id"]: row for row in once_summary["samples"]}
    full_samples = {row["document_id"]: row for row in full_summary["samples"]}
    per_sample = []
    for document_id in once_ids:
        once = once_samples[document_id]
        full = full_samples[document_id]
        per_sample.append(
            {
                "document_id": document_id,
                "once_critic": {
                    "status": once.get("status"),
                    "predicted": once.get("predicted_count", 0),
                    "exact": once.get("exact_match_count"),
                },
                "full": {
                    "status": full.get("status"),
                    "predicted": full.get("predicted_count", 0),
                    "exact": full.get("exact_match_count"),
                },
                "delta_once_minus_full": {
                    "predicted": once.get("predicted_count", 0) - full.get("predicted_count", 0),
                    "exact": (
                        once["exact_match_count"] - full["exact_match_count"]
                        if "exact_match_count" in once and "exact_match_count" in full
                        else None
                    ),
                },
            }
        )

    return {
        "comparison_design": "independent_rerun",
        "interpretation_warning": (
            "Independent runs may include Proposer sampling variance; verify proposal "
            "fingerprints before interpreting differences as a shared-prefix causal ablation."
        ),
        "delta_definition": "once_critic - full",
        "documents": len(once_ids),
        "once_critic": once_compact,
        "full": full_compact,
        "delta_once_minus_full": _numeric_delta(once_compact, full_compact, delta_fields),
        "soft_metrics": soft_comparison,
        "per_sample": per_sample,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare independent proposer+one-Critic and full pipeline runs."
    )
    parser.add_argument("--once-run-dir", required=True)
    parser.add_argument("--full-run-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        value = build_comparison(
            resolve_run_dir(args.once_run_dir),
            resolve_run_dir(args.full_run_dir),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    text = json.dumps(value, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
