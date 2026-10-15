from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.evaluation.direction import (
    normalize_symmetric_predicates,
    validate_entity_order_config,
)

PROTOCOL = "graphjudge_official_v1"
REUSABLE_PROTOCOL = "graphjudge_reusable_gold_v1"
DEFAULT_BERT_MODEL = "bert-base-uncased"
NULL_TRIPLE = ("Null", "Null", "Null")
MATCHING_MODES = ("hungarian", "reusable")
ENTITY_ORDER_MODES = ("directed", "undirected", "predicate_aware")
EMPTY_GOLD_POLICIES = ("error", "skip")
METRIC_NAMES = (
    "G-BS-Acc",
    "G-BS-Recall",
    "G-BS-F1",
    "G-BL-Acc",
    "G-BL-Recall",
    "G-BL-F1",
    "G-RO-Acc",
    "G-RO-Recall",
    "G-RO-F1",
)

Triple = Tuple[str, str, str]


class EvaluationInputError(ValueError):
    pass


def normalize_entity_order(
    triple: Triple,
    entity_order_mode: str,
    symmetric_predicates: Sequence[str] = (),
) -> Triple:
    if entity_order_mode not in ENTITY_ORDER_MODES:
        raise ValueError(f"entity_order_mode must be one of {ENTITY_ORDER_MODES}")
    if entity_order_mode == "directed" or triple == NULL_TRIPLE:
        return triple
    subject, predicate, obj = triple
    if entity_order_mode == "predicate_aware" and " ".join(
        predicate.strip().casefold().split()
    ) not in normalize_symmetric_predicates(symmetric_predicates):
        return triple
    left, right = sorted((subject, obj), key=lambda value: value.casefold())
    return left, predicate, right


def missing_runtime_dependencies() -> List[str]:
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "nltk": "nltk",
        "torch": "torch",
        "bert-score": "bert_score",
        "rouge-score": "rouge_score",
        "spacy": "spacy",
    }
    return [
        package for package, module in packages.items() if importlib.util.find_spec(module) is None
    ]


def ensure_runtime_dependencies() -> None:
    missing = missing_runtime_dependencies()
    if not missing:
        return
    install = "python -m pip install " + " ".join(missing)
    raise RuntimeError(
        "Missing GraphJudge evaluation dependencies: "
        + ", ".join(missing)
        + f". Install them manually with: {install}"
    )


def _read_jsonl(path: Path) -> List[Mapping]:
    rows: List[Mapping] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise EvaluationInputError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _read_prediction_rows(path: Path) -> List[Mapping]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationInputError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise EvaluationInputError(f"{path}: expected a JSON array")
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise EvaluationInputError(f"{path}: item {index} is not a JSON object")
    return value


def _record_id(row: Mapping, context: str) -> str:
    document_id = str(row.get("document_id") or "").strip()
    sample_id = str(row.get("sample_id") or "").strip()
    if document_id and sample_id and document_id != sample_id:
        raise EvaluationInputError(
            f"{context}: document_id {document_id!r} conflicts with sample_id {sample_id!r}"
        )
    record_id = document_id or sample_id
    if not record_id:
        raise EvaluationInputError(f"{context}: missing document_id/sample_id")
    return record_id


def _triple_from_value(value, context: str) -> Triple:
    if isinstance(value, Mapping):
        parts = (value.get("subject"), value.get("predicate"), value.get("object"))
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        parts = value[:3]
    else:
        raise EvaluationInputError(f"{context}: expected a triple object or a three-item sequence")

    triple = tuple(str(part or "").strip() for part in parts)
    if any(not part for part in triple):
        raise EvaluationInputError(f"{context}: subject, predicate, and object must be non-empty")
    return triple  # type: ignore[return-value]


def load_gold_graphs_with_metadata(
    path: Path,
    empty_gold_policy: str = "error",
) -> Tuple[OrderedDict[str, List[Triple]], List[str], int]:
    if empty_gold_policy not in EMPTY_GOLD_POLICIES:
        raise ValueError(f"empty_gold_policy must be one of {EMPTY_GOLD_POLICIES}")
    graphs: OrderedDict[str, List[Triple]] = OrderedDict()
    skipped_empty_gold_ids: List[str] = []
    total_rows = 0
    for index, row in enumerate(_read_jsonl(path), start=1):
        total_rows += 1
        record_id = _record_id(row, f"{path}: row {index}")
        if record_id in graphs or record_id in skipped_empty_gold_ids:
            raise EvaluationInputError(f"{path}: duplicate gold document id {record_id!r}")
        triples = row.get("triples")
        if not isinstance(triples, list):
            raise EvaluationInputError(f"{path}: row {index}: triples must be a list")
        graph = [
            _triple_from_value(item, f"{path}: row {index}, triple {triple_index}")
            for triple_index, item in enumerate(triples, start=1)
        ]
        if not graph:
            if empty_gold_policy == "skip":
                skipped_empty_gold_ids.append(record_id)
                continue
            raise EvaluationInputError(f"{path}: row {index}: gold graph must not be empty")
        graphs[record_id] = graph
    if not graphs:
        raise EvaluationInputError(f"{path}: no gold graphs found")
    return graphs, skipped_empty_gold_ids, total_rows


def load_gold_graphs(
    path: Path,
    empty_gold_policy: str = "error",
) -> OrderedDict[str, List[Triple]]:
    graphs, _, _ = load_gold_graphs_with_metadata(path, empty_gold_policy)
    return graphs


def load_prediction_graphs(
    path: Path,
    gold_ids: Sequence[str],
    ignored_ids: Sequence[str] = (),
) -> Tuple[OrderedDict[str, List[Triple]], List[str]]:
    gold_id_set = set(gold_ids)
    ignored_id_set = set(ignored_ids)
    grouped: Dict[str, List[Triple]] = {}
    for index, row in enumerate(_read_prediction_rows(path), start=1):
        record_id = _record_id(row, f"{path}: prediction {index}")
        if record_id in ignored_id_set:
            continue
        if record_id not in gold_id_set:
            raise EvaluationInputError(f"{path}: unknown prediction document id {record_id!r}")
        grouped.setdefault(record_id, []).append(
            _triple_from_value(row, f"{path}: prediction {index}")
        )

    missing_ids = [record_id for record_id in gold_ids if record_id not in grouped]
    ordered: OrderedDict[str, List[Triple]] = OrderedDict()
    for record_id in gold_ids:
        ordered[record_id] = grouped.get(record_id, [NULL_TRIPLE])
    return ordered, missing_ids


def resolve_prediction_path(run_dir: Path, explicit_path: Optional[Path] = None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(explicit_path)
        return explicit_path
    for name in ("kg.json", "kg.jsonl"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no kg.json or kg.jsonl found under {run_dir}")


def official_edge_text(triple: Triple) -> str:
    # This intentionally preserves GraphJudge's published character-level join.
    return ";".join(str(list(triple))).lower().strip()


def build_official_tokens(
    graphs: Sequence[Sequence[Triple]],
) -> Tuple[List[List[List[str]]], List[List[str]]]:
    from spacy.lang.en import English
    from spacy.tokenizer import Tokenizer

    tokenizer = Tokenizer(English().vocab, infix_finditer=re.compile(r"[;]").finditer)
    all_tokens: List[List[List[str]]] = []
    all_sentences: List[List[str]] = []
    for graph in graphs:
        sentences = [official_edge_text(triple) for triple in graph]
        tokens = [[token.text for token in sample] for sample in tokenizer.pipe(sentences)]
        all_sentences.append(sentences)
        all_tokens.append(tokens)
    return all_tokens, all_sentences


def aligned_scores(score_matrix) -> Tuple[float, float, float, float]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("score matrix must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("score matrix contains a non-finite value")
    row_indices, column_indices = linear_sum_assignment(matrix, maximize=True)
    matched_sum = float(matrix[row_indices, column_indices].sum())
    accuracy = matched_sum / matrix.shape[0]
    recall = matched_sum / matrix.shape[1]
    f1 = 2 * accuracy * recall / (accuracy + recall) if accuracy + recall else 0.0
    return accuracy, recall, f1, matched_sum


def reusable_scores(score_matrix) -> Tuple[float, float, float, float, float]:
    """Score both directions independently so the same edge may be reused."""
    import numpy as np

    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("score matrix must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("score matrix contains a non-finite value")
    accuracy_sum = float(matrix.max(axis=1).sum())
    recall_sum = float(matrix.max(axis=0).sum())
    accuracy = accuracy_sum / matrix.shape[0]
    recall = recall_sum / matrix.shape[1]
    f1 = 2 * accuracy * recall / (accuracy + recall) if accuracy + recall else 0.0
    return accuracy, recall, f1, accuracy_sum, recall_sum


def score_matrix(score_matrix, matching_mode: str) -> Tuple[float, float, float, float, float]:
    if matching_mode == "hungarian":
        accuracy, recall, f1, matched_sum = aligned_scores(score_matrix)
        return accuracy, recall, f1, matched_sum, matched_sum
    if matching_mode == "reusable":
        return reusable_scores(score_matrix)
    raise ValueError(f"unsupported matching mode: {matching_mode!r}")


def compute_bleu_rouge_metrics(
    gold_graphs: Sequence[Sequence[Triple]],
    predicted_graphs: Sequence[Sequence[Triple]],
    matching_mode: str = "hungarian",
) -> List[Dict[str, float]]:
    import numpy as np
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    gold_tokens, gold_sentences = build_official_tokens(gold_graphs)
    predicted_tokens, predicted_sentences = build_official_tokens(predicted_graphs)
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rouge3", "rougeL"],
        use_stemmer=True,
    )
    smoothing = SmoothingFunction().method1
    results: List[Dict[str, float]] = []

    for graph_index in range(len(gold_graphs)):
        bleu_matrix = np.zeros(
            (len(predicted_graphs[graph_index]), len(gold_graphs[graph_index])),
            dtype=float,
        )
        rouge_matrix = np.zeros_like(bleu_matrix)
        for pred_index in range(bleu_matrix.shape[0]):
            for gold_index in range(bleu_matrix.shape[1]):
                bleu_matrix[pred_index, gold_index] = sentence_bleu(
                    [gold_tokens[graph_index][gold_index]],
                    predicted_tokens[graph_index][pred_index],
                    smoothing_function=smoothing,
                )
                rouge_matrix[pred_index, gold_index] = rouge.score(
                    gold_sentences[graph_index][gold_index],
                    predicted_sentences[graph_index][pred_index],
                )["rouge2"].precision

        bl_acc, bl_recall, bl_f1, bl_acc_sum, bl_recall_sum = score_matrix(
            bleu_matrix, matching_mode
        )
        ro_acc, ro_recall, ro_f1, ro_acc_sum, ro_recall_sum = score_matrix(
            rouge_matrix, matching_mode
        )
        results.append(
            {
                "G-BL-Acc": bl_acc,
                "G-BL-Recall": bl_recall,
                "G-BL-F1": bl_f1,
                "G-BL-Matched-Sum": bl_acc_sum,
                "G-BL-Accuracy-Sum": bl_acc_sum,
                "G-BL-Recall-Sum": bl_recall_sum,
                "G-RO-Acc": ro_acc,
                "G-RO-Recall": ro_recall,
                "G-RO-F1": ro_f1,
                "G-RO-Matched-Sum": ro_acc_sum,
                "G-RO-Accuracy-Sum": ro_acc_sum,
                "G-RO-Recall-Sum": ro_recall_sum,
            }
        )
    return results


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_bert_metrics(
    gold_graphs: Sequence[Sequence[Triple]],
    predicted_graphs: Sequence[Sequence[Triple]],
    model_type: str,
    device: str,
    batch_size: int,
    score_backend: Optional[Callable] = None,
    matching_mode: str = "hungarian",
) -> List[Dict[str, float]]:
    import numpy as np

    if score_backend is None:
        from bert_score import score as score_backend

    references: List[str] = []
    candidates: List[str] = []
    matrix_shapes: List[Tuple[int, int]] = []
    for gold_graph, predicted_graph in zip(gold_graphs, predicted_graphs):
        matrix_shapes.append((len(predicted_graph), len(gold_graph)))
        for predicted_triple in predicted_graph:
            for gold_triple in gold_graph:
                candidates.append(official_edge_text(predicted_triple))
                references.append(official_edge_text(gold_triple))

    _, _, pair_f1 = score_backend(
        cands=candidates,
        refs=references,
        model_type=model_type,
        lang="en",
        idf=False,
        device=device,
        batch_size=batch_size,
    )
    pair_scores = np.asarray(
        pair_f1.detach().cpu() if hasattr(pair_f1, "detach") else pair_f1, dtype=float
    )
    if pair_scores.size != len(candidates):
        raise RuntimeError(
            f"BERTScore returned {pair_scores.size} scores for {len(candidates)} triple pairs"
        )

    results: List[Dict[str, float]] = []
    offset = 0
    for rows, columns in matrix_shapes:
        size = rows * columns
        matrix = pair_scores[offset : offset + size].reshape(rows, columns)
        offset += size
        bs_acc, bs_recall, bs_f1, bs_acc_sum, bs_recall_sum = score_matrix(matrix, matching_mode)
        results.append(
            {
                "G-BS-Acc": bs_acc,
                "G-BS-Recall": bs_recall,
                "G-BS-F1": bs_f1,
                "G-BS-Matched-Sum": bs_acc_sum,
                "G-BS-Accuracy-Sum": bs_acc_sum,
                "G-BS-Recall-Sum": bs_recall_sum,
            }
        )
    return results


def aggregate_metrics(per_sample: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not per_sample:
        raise ValueError("cannot aggregate an empty sample list")
    metrics = {
        name: sum(float(sample[name]) for sample in per_sample) / len(per_sample)
        for name in METRIC_NAMES
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("aggregate metrics contain a non-finite value")
    return metrics


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_run(
    run_dir: Path,
    gold_path: Optional[Path] = None,
    prediction_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    per_sample_output_path: Optional[Path] = None,
    bert_model: str = DEFAULT_BERT_MODEL,
    requested_device: str = "auto",
    batch_size: int = 64,
    matching_mode: str = "hungarian",
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
    empty_gold_policy: str = "error",
) -> Dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if matching_mode not in MATCHING_MODES:
        raise ValueError(f"matching_mode must be one of {MATCHING_MODES}")
    symmetric_predicates = validate_entity_order_config(entity_order_mode, symmetric_predicates)
    if empty_gold_policy not in EMPTY_GOLD_POLICIES:
        raise ValueError(f"empty_gold_policy must be one of {EMPTY_GOLD_POLICIES}")
    run_dir = run_dir.resolve()
    gold_path = (gold_path or run_dir / "gold_triples.jsonl").resolve()
    prediction_path = resolve_prediction_path(
        run_dir,
        prediction_path.resolve() if prediction_path else None,
    ).resolve()
    output_name = (
        "graphjudge_metrics.json"
        if matching_mode == "hungarian"
        else "graphjudge_metrics_reusable.json"
    )
    per_sample_name = (
        "graphjudge_metrics_per_sample.jsonl"
        if matching_mode == "hungarian"
        else "graphjudge_metrics_reusable_per_sample.jsonl"
    )
    output_path = (output_path or run_dir / output_name).resolve()
    per_sample_output_path = (per_sample_output_path or run_dir / per_sample_name).resolve()

    if not gold_path.exists():
        raise FileNotFoundError(gold_path)
    ensure_runtime_dependencies()

    gold_by_id, skipped_empty_gold_ids, samples_total_input = load_gold_graphs_with_metadata(
        gold_path,
        empty_gold_policy,
    )
    predicted_by_id, missing_ids = load_prediction_graphs(
        prediction_path,
        list(gold_by_id),
        ignored_ids=skipped_empty_gold_ids,
    )
    document_ids = list(gold_by_id)
    gold_graphs = [gold_by_id[document_id] for document_id in document_ids]
    predicted_graphs = [predicted_by_id[document_id] for document_id in document_ids]
    gold_graphs = [
        [
            normalize_entity_order(triple, entity_order_mode, symmetric_predicates)
            for triple in graph
        ]
        for graph in gold_graphs
    ]
    predicted_graphs = [
        [
            normalize_entity_order(triple, entity_order_mode, symmetric_predicates)
            for triple in graph
        ]
        for graph in predicted_graphs
    ]
    device = resolve_device(requested_device)

    lexical_results = compute_bleu_rouge_metrics(
        gold_graphs, predicted_graphs, matching_mode=matching_mode
    )
    bert_results = compute_bert_metrics(
        gold_graphs,
        predicted_graphs,
        model_type=bert_model,
        device=device,
        batch_size=batch_size,
        matching_mode=matching_mode,
    )

    per_sample: List[Dict] = []
    missing_id_set = set(missing_ids)
    for index, document_id in enumerate(document_ids):
        row = {
            "document_id": document_id,
            "gold_count": len(gold_graphs[index]),
            "predicted_count": len(predicted_graphs[index]),
            "prediction_missing": document_id in missing_id_set,
            **bert_results[index],
            **lexical_results[index],
        }
        per_sample.append(row)

    metrics = aggregate_metrics(per_sample)
    summary = {
        "protocol": PROTOCOL if matching_mode == "hungarian" else REUSABLE_PROTOCOL,
        "matching_mode": matching_mode,
        "entity_order_mode": entity_order_mode,
        "symmetric_predicates": list(symmetric_predicates),
        "run_dir": str(run_dir),
        "gold_file": str(gold_path),
        "prediction_file": str(prediction_path),
        "bert_model": bert_model,
        "device": device,
        "batch_size": batch_size,
        "samples_total": len(document_ids),
        "samples_total_input": samples_total_input,
        "samples_evaluated": len(document_ids),
        "samples_skipped_empty_gold": len(skipped_empty_gold_ids),
        "skipped_empty_gold_ids": skipped_empty_gold_ids,
        "samples_with_predictions": len(document_ids) - len(missing_ids),
        "samples_missing_predictions": len(missing_ids),
        "missing_prediction_ids": missing_ids,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(per_sample_output_path, per_sample)
    return summary


def print_summary(summary: Mapping) -> None:
    metrics = summary["metrics"]
    print(f"Protocol: {summary['protocol']}")
    print(
        f"Samples: {summary['samples_evaluated']} "
        f"(input: {summary['samples_total_input']}, "
        f"skipped empty gold: {summary['samples_skipped_empty_gold']}, "
        f"missing predictions: {summary['samples_missing_predictions']})"
    )
    print("Metric       Acc     Recall  F1")
    for prefix in ("G-BS", "G-BL", "G-RO"):
        print(
            f"{prefix:<10} "
            f"{metrics[prefix + '-Acc']:.4f}  "
            f"{metrics[prefix + '-Recall']:.4f}  "
            f"{metrics[prefix + '-F1']:.4f}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate REBEL-Sub predictions with GraphJudge's official soft graph metrics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--pred-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-sample-output", type=Path)
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--matching-mode",
        choices=MATCHING_MODES,
        default="hungarian",
        help="hungarian uses official one-to-one matching; reusable scores best matches independently",
    )
    parser.add_argument(
        "--entity-order-mode",
        choices=ENTITY_ORDER_MODES,
        default="directed",
    )
    parser.add_argument(
        "--symmetric-predicate",
        action="append",
        default=None,
        help="repeat for each predicate whose subject/object order is symmetric",
    )
    parser.add_argument(
        "--empty-gold-policy",
        choices=EMPTY_GOLD_POLICIES,
        default="error",
        help="error rejects empty gold graphs; skip excludes them from soft metrics",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = evaluate_run(
            run_dir=args.run_dir,
            gold_path=args.gold_file,
            prediction_path=args.pred_file,
            output_path=args.output,
            per_sample_output_path=args.per_sample_output,
            bert_model=args.bert_model,
            requested_device=args.device,
            batch_size=args.batch_size,
            matching_mode=args.matching_mode,
            entity_order_mode=args.entity_order_mode,
            symmetric_predicates=args.symmetric_predicate or (),
            empty_gold_policy=args.empty_gold_policy,
        )
    except (EvaluationInputError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
