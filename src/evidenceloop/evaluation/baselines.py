"""Evaluate finalized related-work baseline runs; this is the only entry reading gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

from evidenceloop.baselines.related.common import (
    DATASETS,
    DEFAULT_OUTPUT_ROOT,
    METHODS,
    digest,
    manifest,
    read_json,
    read_jsonl,
    resolve_path,
    validate_run,
)
from evidenceloop.data.documents import load_documents
from evidenceloop.evaluation.graphjudge import METRIC_NAMES, evaluate_run, normalize_entity_order
from evidenceloop.io import _atomic_write_text, atomic_write_json, atomic_write_jsonl

ROOT = Path.cwd()


def aligned_gold(state, selected, gold_file=None):
    data_dir = Path(state["data"]["data_dir"])
    path = resolve_path(gold_file) if gold_file else data_dir / "gold_triples.jsonl"
    documents_path = data_dir / "documents.jsonl"
    # Verify that the local gold directory still describes the source cohort that was inferred.
    if documents_path.exists():
        documents = load_documents(str(documents_path))
        source = {d.document_id: d for d in documents}
        if len(source) != len(documents):
            raise ValueError("gold dataset contains duplicate source document IDs")
        try:
            current = [source[row["document_id"]] for row in selected]
        except KeyError as exc:
            raise ValueError("gold dataset is missing a selected source document") from exc
        if manifest(current) != selected:
            raise ValueError("gold dataset source text changed since inference")
    by_id = {}
    for row in read_jsonl(path):
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or document_id in by_id:
            raise ValueError(f"{path}: missing or duplicate gold document ID")
        if not isinstance(row.get("triples"), list):
            raise ValueError(f"{path}: triples must be a list")
        by_id[document_id] = row
    missing = [r["document_id"] for r in selected if r["document_id"] not in by_id]
    if missing:
        raise ValueError(f"missing gold for selected documents: {missing[:5]}")
    return [
        {"document_id": r["document_id"], "triples": by_id[r["document_id"]]["triples"]}
        for r in selected
    ], path


def exact_metrics(gold, checkpoints, dataset):
    preset = DATASETS[dataset]

    def key(triple):
        if isinstance(triple, dict):
            parts = (triple.get("subject"), triple.get("predicate"), triple.get("object"))
        elif isinstance(triple, (list, tuple)) and len(triple) == 3:
            parts = triple
        else:
            raise ValueError("invalid gold triple")
        if any(not isinstance(p, str) or not p.strip() for p in parts):
            raise ValueError("gold triples require three non-empty strings")
        normalized = tuple(" ".join(p.strip().casefold().split()) for p in parts)
        return normalize_entity_order(
            normalized, preset.entity_order_mode, preset.symmetric_predicates
        )

    gold_total = pred_total = exact_total = 0
    for index, row in enumerate(gold):
        gold_set = {key(t) for t in row["triples"]}
        pred_set = {key(t) for t in checkpoints[index]["final_triples"]}
        gold_total += len(gold_set)
        pred_total += len(pred_set)
        exact_total += len(gold_set & pred_set)
    precision = exact_total / pred_total if pred_total else 0.0
    recall = exact_total / gold_total if gold_total else 0.0
    return {
        "gold_triples_total": gold_total,
        "predicted_spo_total": pred_total,
        "exact_match_spo_total": exact_total,
        "exact_match_precision": precision,
        "exact_match_recall": recall,
        "exact_match_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def evaluate_target(
    run_dir,
    *,
    gold_file=None,
    bert_model="bert-base-uncased",
    device="auto",
    batch_size=64,
    force=False,
    metric_fn=evaluate_run,
):
    run_dir = resolve_path(run_dir)
    state, selected, checkpoints = validate_run(run_dir)
    runtime = state["runtime"]
    dataset = runtime["dataset"]
    preset = DATASETS[dataset]
    expected = [cp["pipeline_record"] for cp in checkpoints.values()]
    if read_jsonl(run_dir / "pipeline_records.jsonl") != expected:
        raise ValueError("pipeline records do not match successful checkpoints")
    flat = [t for cp in checkpoints.values() for t in cp["final_triples"]]
    if read_jsonl(run_dir / "kg.jsonl") != flat:
        raise ValueError("KG export does not match successful checkpoints")
    snapshot = load_documents(str(run_dir / "source_documents.jsonl"))
    if manifest(snapshot) != selected:
        raise ValueError("saved source snapshot does not match document manifest")
    gold, gold_path = aligned_gold(state, selected, gold_file)
    exact = exact_metrics(gold, checkpoints, dataset)
    signature = {
        "manifest_sha256": state["data"]["manifest_sha256"],
        "predictions_sha256": digest(flat),
        "gold_sha256": digest(gold),
        "runtime_sha256": digest(runtime),
        "bert_model": bert_model,
        "device": device,
        "batch_size": batch_size,
        "matching_mode": "hungarian",
        "entity_order_mode": preset.entity_order_mode,
        "symmetric_predicates": list(preset.symmetric_predicates),
        "empty_gold_policy": preset.empty_gold_policy,
        "evaluator_sha256": hashlib.sha256(
            Path(
                __import__("evidenceloop.evaluation.graphjudge", fromlist=["__file__"]).__file__
            ).read_bytes()
        ).hexdigest(),
    }
    cache_path = run_dir / "related_baseline_evaluation.json"
    if not force and cache_path.exists() and (run_dir / "graphjudge_metrics.json").exists():
        cached = read_json(cache_path)
        if cached.get("signature") == signature:
            return cached["result"]
    # This selected gold snapshot is created only by offline evaluation.
    atomic_write_jsonl(run_dir / "gold_triples.jsonl", gold)
    metrics = metric_fn(
        run_dir=run_dir,
        gold_path=run_dir / "gold_triples.jsonl",
        prediction_path=run_dir / "kg.jsonl",
        bert_model=bert_model,
        requested_device=device,
        batch_size=batch_size,
        matching_mode="hungarian",
        entity_order_mode=preset.entity_order_mode,
        symmetric_predicates=preset.symmetric_predicates,
        empty_gold_policy=preset.empty_gold_policy,
    )
    if any(name not in metrics.get("metrics", {}) for name in METRIC_NAMES):
        raise ValueError("semantic evaluator did not return all nine GraphJudge metrics")
    verifier = runtime.get("models", {}).get("verifier", {})
    summary = read_json(run_dir / "summary.json")
    result = {
        "method": runtime["method"],
        "dataset": dataset,
        "model": runtime["model"],
        "run_dir": str(run_dir),
        "runtime_sha256": digest(runtime),
        "manifest_sha256": signature["manifest_sha256"],
        "source": runtime["source"],
        "verifier": verifier,
        "verifier_training_domain": verifier.get("training_domain", ""),
        "verifier_transfer": verifier.get("transfer", False),
        "documents_total": len(selected),
        "documents_evaluated": metrics["samples_evaluated"],
        "documents_skipped_empty_gold": metrics.get("samples_skipped_empty_gold", 0),
        "triples_total": len(flat),
        "triples_per_document": len(flat) / len(selected),
        "active_wall_seconds": summary["active_wall_seconds"],
        "protocol": metrics["protocol"],
        "bert_model": bert_model,
        "gold_source": str(gold_path),
        "entity_order_mode": preset.entity_order_mode,
        "empty_gold_policy": preset.empty_gold_policy,
        **exact,
        **metrics["metrics"],
    }
    atomic_write_json(cache_path, {"signature": signature, "result": result})
    return result


def discover_runs(input_root: Path, method="all", dataset="all", model=None):
    groups = {}
    for state_path in sorted(input_root.rglob("run_state.json")):
        state = read_json(state_path)
        if state.get("kind") != "related_baseline":
            continue
        runtime = state["runtime"]
        if method != "all" and method != runtime["method"]:
            continue
        if dataset != "all" and dataset != runtime["dataset"]:
            continue
        if model is not None and model != runtime["model"]:
            continue
        progress_path = state_path.parent / "progress.json"
        if not progress_path.exists() or not read_json(progress_path).get("complete"):
            continue
        # Keep different method parameters/checkpoints in separate comparison groups.
        group = (
            runtime["method"],
            runtime["dataset"],
            runtime["model"],
            digest(runtime),
            state["data"]["manifest_sha256"],
        )
        candidate = (state["created_at"], str(state_path.parent))
        if group not in groups or candidate > groups[group][0]:
            groups[group] = (candidate, state_path.parent)
    return [groups[key][1] for key in sorted(groups)]


def write_summary(results, output_dir):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A single comparison group must have an identical document cohort across methods.
    cohorts = {}
    for result in results:
        cohorts.setdefault(result["dataset"], set()).add(result["manifest_sha256"])
    mismatched = sorted(key for key, hashes in cohorts.items() if len(hashes) > 1)
    atomic_write_json(
        output_dir / "summary.json",
        {"runs": results, "datasets_with_different_cohorts": mismatched},
    )
    columns = [
        "method",
        "dataset",
        "model",
        "documents_total",
        "documents_evaluated",
        "documents_skipped_empty_gold",
        "triples_total",
        "active_wall_seconds",
        "exact_match_precision",
        "exact_match_recall",
        "exact_match_f1",
        *METRIC_NAMES,
        "protocol",
        "verifier_training_domain",
        "verifier_transfer",
        "runtime_sha256",
        "manifest_sha256",
        "run_dir",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    _atomic_write_text(output_dir / "summary.csv", buffer.getvalue())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", action="append", help="repeat for explicitly selected complete runs"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "evaluation"))
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model")
    parser.add_argument("--gold-file", help="explicit canonical gold file for one run")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        targets = (
            [resolve_path(p) for p in args.run_dir]
            if args.run_dir
            else discover_runs(resolve_path(args.input_root), args.method, args.dataset, args.model)
        )
        if not targets:
            raise ValueError("no complete successful related-baseline runs found")
        if args.gold_file and len(targets) != 1:
            raise ValueError("--gold-file requires exactly one run")
        results = [
            evaluate_target(
                path,
                gold_file=args.gold_file,
                bert_model=args.bert_model,
                device=args.device,
                batch_size=args.batch_size,
                force=args.force,
            )
            for path in targets
        ]
        write_summary(results, args.output_dir)
        print(
            f"Evaluated {len(results)} runs. Summary: {resolve_path(args.output_dir) / 'summary.csv'}"
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
