from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path.cwd()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def summary_sample_map(summary: Dict) -> Dict[str, Dict]:
    return {row.get("document_id", ""): row for row in summary.get("samples", [])}


def per_sample_comparison(baseline: Dict, lora: Dict, limit: int = 10) -> List[Dict]:
    baseline_samples = summary_sample_map(baseline)
    lora_samples = summary_sample_map(lora)
    document_ids = list(dict.fromkeys(list(baseline_samples.keys()) + list(lora_samples.keys())))
    output = []
    for document_id in document_ids[:limit]:
        b = baseline_samples.get(document_id, {})
        adapted = lora_samples.get(document_id, {})
        output.append(
            {
                "document_id": document_id,
                "baseline": {
                    "gold": b.get("gold_count"),
                    "predicted": b.get("predicted_count"),
                    "exact": b.get("exact_match_count"),
                },
                "lora": {
                    "gold": adapted.get("gold_count"),
                    "predicted": adapted.get("predicted_count"),
                    "exact": adapted.get("exact_match_count"),
                },
            }
        )
    return output


def compact_summary(run_dir: Path, summary: Dict) -> Dict:
    return {
        "run_dir": str(run_dir),
        "kg_triples": summary.get("kg_triples", 0),
        "pipeline_records": summary.get("pipeline_records", 0),
        "extracted_triples": summary.get("extracted_triples", 0),
        "refined_triples": summary.get("refined_triples", 0),
        "integrator_additions_proposed": summary.get("integrator_additions_proposed", 0),
        "integrator_additions_accepted": summary.get("integrator_additions_accepted", 0),
        "exact_match_precision": summary.get("exact_match_precision"),
        "exact_match_recall": summary.get("exact_match_recall"),
        "exact_match_f1": summary.get("exact_match_f1"),
        "initial_critic_actions": summary.get("initial_critic_actions", {}),
        "final_critic_actions": summary.get("final_critic_actions", {}),
        "initial_reviews_valid": summary.get("initial_reviews_valid", 0),
        "initial_reviews_fallback": summary.get("initial_reviews_fallback", 0),
        "final_reviews_valid": summary.get("final_reviews_valid", 0),
        "final_reviews_fallback": summary.get("final_reviews_fallback", 0),
        "critic_parse_error_documents": summary.get("critic_parse_error_documents", 0),
        "stage_errors_total": summary.get("stage_errors_total", 0),
        "runtime_config": summary.get("runtime_config", {}),
    }


def load_summary(run_dir: Path) -> Dict:
    summary = read_json(run_dir / "summary.json")
    exact_path = run_dir / "exact_metrics.json"
    if exact_path.is_file():
        exact = read_json(exact_path)
        if exact.get("protocol") == "evidenceloop_exact_v1":
            summary.update({"exact_match_" + k: exact[k] for k in ("precision", "recall", "f1")})
            summary["gold_triples_total"] = exact["gold_triples"]
            summary["exact_match_spo_total"] = exact["matches"]
            by_id = {row["document_id"]: row for row in exact.get("documents", [])}
            for sample in summary.get("samples", []):
                row = by_id.get(sample.get("document_id"))
                if row:
                    sample.update(
                        gold_count=row["gold"],
                        predicted_count=row["predicted"],
                        exact_match_count=row["matched"],
                    )
    return summary


def metric_delta(baseline: Dict, adapted: Dict, field: str):
    before, after = baseline.get(field), adapted.get(field)
    return round(after - before, 4) if before is not None and after is not None else None


def build_comparison(baseline_run_dir: Path, lora_run_dir: Path) -> Dict:
    baseline_summary = load_summary(baseline_run_dir)
    lora_summary = load_summary(lora_run_dir)
    return {
        "baseline": compact_summary(baseline_run_dir, baseline_summary),
        "lora": compact_summary(lora_run_dir, lora_summary),
        "delta": {
            "kg_triples": lora_summary.get("kg_triples", 0) - baseline_summary.get("kg_triples", 0),
            "exact_match_precision": metric_delta(
                baseline_summary, lora_summary, "exact_match_precision"
            ),
            "exact_match_recall": metric_delta(
                baseline_summary, lora_summary, "exact_match_recall"
            ),
            "exact_match_f1": metric_delta(baseline_summary, lora_summary, "exact_match_f1"),
            "initial_reviews_fallback": (
                lora_summary.get("initial_reviews_fallback", 0)
                - baseline_summary.get("initial_reviews_fallback", 0)
            ),
            "critic_parse_error_documents": (
                lora_summary.get("critic_parse_error_documents", 0)
                - baseline_summary.get("critic_parse_error_documents", 0)
            ),
            "stage_errors_total": (
                lora_summary.get("stage_errors_total", 0)
                - baseline_summary.get("stage_errors_total", 0)
            ),
        },
        "first_10_samples": per_sample_comparison(baseline_summary, lora_summary, limit=10),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare two GraphJudge REBEL experiment runs.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--lora-run-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    baseline_run_dir = resolve_run_dir(args.baseline_run_dir)
    lora_run_dir = resolve_run_dir(args.lora_run_dir)
    comparison = build_comparison(baseline_run_dir, lora_run_dir)
    text = json.dumps(comparison, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        if output_path.exists():
            raise FileExistsError(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
