"""Exact matching uses each dataset's existing endpoint-direction policy."""

import argparse
from pathlib import Path

from evidenceloop.data.registry import DATASETS, canonical_dataset
from evidenceloop.evaluation.graphjudge import (
    load_gold_graphs_with_metadata,
    load_prediction_graphs,
)
from evidenceloop.experiments.checkpoints import triple_key
from evidenceloop.io import atomic_write_json


def evaluate(gold_path, prediction_path, *, dataset="rebel_sub", empty_gold_policy="error"):
    spec = DATASETS[canonical_dataset(dataset)]
    gold, skipped, total = load_gold_graphs_with_metadata(
        Path(gold_path), empty_gold_policy=empty_gold_policy
    )
    predicted, missing = load_prediction_graphs(Path(prediction_path), list(gold), skipped)
    metadata = {"skipped_empty_gold": skipped, "total_rows": total}
    for document_id in missing:
        predicted[document_id] = []

    def keys(triples):
        return {triple_key(*t, spec.entity_order_mode, spec.symmetric_predicates) for t in triples}

    matches = gold_total = pred_total = 0
    rows = []
    for document_id, triples in gold.items():
        g = keys(triples)
        p = keys(predicted.get(document_id, []))
        count = len(g & p)
        matches += count
        gold_total += len(g)
        pred_total += len(p)
        rows.append(
            {"document_id": document_id, "gold": len(g), "predicted": len(p), "matched": count}
        )
    precision = matches / pred_total if pred_total else 0.0
    recall = matches / gold_total if gold_total else 0.0
    return {
        "protocol": "evidenceloop_exact_v1",
        "dataset": spec.key,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matches": matches,
        "gold_triples": gold_total,
        "predicted_triples": pred_total,
        "gold_metadata": metadata,
        "documents": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop evaluate exact")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default="rebel_sub")
    parser.add_argument("--empty-gold-policy", choices=("error", "skip"), default="error")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = evaluate(
        args.gold, args.predictions, dataset=args.dataset, empty_gold_policy=args.empty_gold_policy
    )
    atomic_write_json(target, result)
    print(result["f1"])
    return 0
