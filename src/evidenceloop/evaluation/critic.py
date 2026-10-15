"""Evaluate completed GPT + Critic-Once runs without invoking any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.registry import DATASETS as DATASET_SPECS
from evidenceloop.data.registry import AliasDict
from evidenceloop.evaluation.graphjudge import evaluate_run  # noqa: E402
from evidenceloop.experiments.checkpoints import (  # noqa: E402
    gold_keys,
    normalize_symmetric_predicates,
    pred_keys,
    validate_entity_order_config,
)
from evidenceloop.types import Triple  # noqa: E402

ROOT = Path.cwd()


DEFAULT_TARGET_ROOT = ROOT / "data" / "experiments" / "gpt_proposer_critic_once"
DEFAULT_BASELINE_ROOT = ROOT / "data" / "experiments" / "gpt_baselines"
DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_METRIC_DEVICE = "auto"
DEFAULT_METRIC_BATCH_SIZE = 64
DATASET_ORDER = ("rebel_sub", "redocred", "biored", "scierc")
DATASET_DIRECTORY_ALIASES = {
    "redocred": ("redocred", "re_docred", "re_record"),
}

DATASET_CONFIG = AliasDict(
    {
        key: (spec.entity_order_mode, spec.symmetric_predicates, spec.empty_gold_policy)
        for key, spec in DATASET_SPECS.items()
    }
)


def _read_json(path: Path) -> Dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: line {line_number}: expected a JSON object")
        rows.append(dict(value))
    return rows


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _record_id(row: Mapping, context: str) -> str:
    value = str(row.get("document_id") or row.get("sample_id") or "").strip()
    if not value:
        raise ValueError(f"{context}: missing document_id")
    return value


def _dataset_config(dataset: str) -> Tuple[str, Tuple[str, ...], str]:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"unsupported dataset: {dataset}")
    mode, predicates, empty_policy = DATASET_CONFIG[dataset]
    normalized = normalize_symmetric_predicates(predicates)
    validate_entity_order_config(mode, normalized)
    return mode, normalized, empty_policy


def _source_baseline_dir(
    target_dir: Path,
    state: Mapping,
    baseline_root: Path,
) -> Path:
    source = state.get("source") or {}
    source_path = Path(str(source.get("run_dir") or ""))
    if source_path.is_dir():
        return source_path
    dataset = str(source.get("dataset") or "")
    model = str(source.get("model") or "")
    timestamp = str(source.get("timestamp") or "") or source_path.name
    candidate = baseline_root / dataset / model / timestamp
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"{target_dir}: source baseline is unavailable; checked {source_path} and {candidate}"
    )


def _validate_complete_target(target_dir: Path, state: Mapping, progress: Mapping) -> None:
    total = int(progress.get("documents_total", 0) or 0)
    completed = int(progress.get("completed", 0) or 0)
    failed = int(progress.get("failed", 0) or 0)
    pending = int(progress.get("pending", 0) or 0)
    if not progress.get("finalized") or completed != total or failed or pending:
        raise ValueError(
            f"{target_dir}: run is incomplete: total={total}, completed={completed}, "
            f"failed={failed}, pending={pending}, finalized={progress.get('finalized')}"
        )
    if str((state.get("runtime") or {}).get("pipeline_variant")) != "proposer_critic_once":
        raise ValueError(f"{target_dir}: not a proposer_critic_once run")
    for name in ("kg.jsonl", "documents.jsonl"):
        if not (target_dir / name).is_file():
            raise FileNotFoundError(target_dir / name)


def _prepare_gold(
    target_dir: Path, source_dir: Path, target_documents: Sequence[Mapping]
) -> List[Dict]:
    source_gold_path = source_dir / "gold_triples.jsonl"
    if not source_gold_path.is_file():
        raise FileNotFoundError(source_gold_path)
    gold_rows = _read_jsonl(source_gold_path)
    target_ids = [
        _record_id(row, f"{target_dir / 'documents.jsonl'} row {i}")
        for i, row in enumerate(target_documents)
    ]
    gold_ids = [_record_id(row, f"{source_gold_path} row {i}") for i, row in enumerate(gold_rows)]
    if target_ids != gold_ids:
        raise ValueError(f"{target_dir}: target document IDs do not match source gold order")
    for index, row in enumerate(gold_rows):
        if not isinstance(row.get("triples"), list):
            raise ValueError(f"{source_gold_path}: row {index}: triples must be a list")
    _write_jsonl(target_dir / "gold_triples.jsonl", gold_rows)
    return gold_rows


def _exact_metrics(
    gold_rows: Sequence[Mapping],
    prediction_rows: Sequence[Mapping],
    entity_order_mode: str,
    symmetric_predicates: Sequence[str],
) -> Dict:
    gold_by_id = {_record_id(row, "gold"): row for row in gold_rows}
    predicted_by_id: Dict[str, List[Triple]] = {document_id: [] for document_id in gold_by_id}
    for index, row in enumerate(prediction_rows):
        document_id = _record_id(row, f"prediction row {index}")
        if document_id not in gold_by_id:
            raise ValueError(f"prediction row {index}: unknown document_id {document_id}")
        predicted_by_id[document_id].append(Triple.from_dict(dict(row), document_id=document_id))

    gold_total = sum(len(row.get("triples") or []) for row in gold_rows)
    predicted_total = sum(len(values) for values in predicted_by_id.values())
    exact_total = 0
    per_sample = []
    for row in gold_rows:
        document_id = _record_id(row, "gold")
        gold_triples = row.get("triples") or []
        predicted_triples = predicted_by_id[document_id]
        matches = gold_keys(gold_triples, entity_order_mode, symmetric_predicates) & pred_keys(
            predicted_triples, entity_order_mode, symmetric_predicates
        )
        exact_total += len(matches)
        per_sample.append(
            {
                "document_id": document_id,
                "gold_count": len(gold_triples),
                "predicted_count": len(predicted_triples),
                "exact_match_count": len(matches),
            }
        )
    precision = exact_total / predicted_total if predicted_total else 0.0
    recall = exact_total / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_triples_total": gold_total,
        "predicted_triples_total": predicted_total,
        "exact_match_spo_total": exact_total,
        "exact_match_precision": round(precision, 4),
        "exact_match_recall": round(recall, 4),
        "exact_match_f1": round(f1, 4),
        "samples": per_sample,
    }


def evaluate_target(
    target_dir: Path,
    baseline_root: Path,
    bert_model: str,
    metric_device: str,
    metric_batch_size: int,
) -> Dict:
    state = _read_json(target_dir / "run_state.json")
    progress = _read_json(target_dir / "progress.json")
    _validate_complete_target(target_dir, state, progress)
    dataset = str((state.get("source") or {}).get("dataset") or "")
    entity_order_mode, symmetric_predicates, empty_gold_policy = _dataset_config(dataset)
    source_dir = _source_baseline_dir(target_dir, state, baseline_root)
    target_documents = _read_jsonl(target_dir / "documents.jsonl")
    gold_rows = _prepare_gold(target_dir, source_dir, target_documents)
    prediction_rows = _read_jsonl(target_dir / "kg.jsonl")
    exact = _exact_metrics(gold_rows, prediction_rows, entity_order_mode, symmetric_predicates)
    graphjudge = evaluate_run(
        run_dir=target_dir,
        gold_path=target_dir / "gold_triples.jsonl",
        prediction_path=target_dir / "kg.jsonl",
        output_path=target_dir / "graphjudge_metrics.json",
        per_sample_output_path=target_dir / "graphjudge_metrics_per_sample.jsonl",
        bert_model=bert_model,
        requested_device=metric_device,
        batch_size=metric_batch_size,
        matching_mode="hungarian",
        entity_order_mode=entity_order_mode,
        symmetric_predicates=symmetric_predicates,
        empty_gold_policy=empty_gold_policy,
    )
    summary = {
        "run_dir": str(target_dir),
        "source_run_dir": str(source_dir),
        "dataset": dataset,
        "gpt_model": (state.get("source") or {}).get("model"),
        "pipeline_variant": "proposer_critic_once",
        "documents_total": len(gold_rows),
        "documents_completed": progress.get("completed"),
        "documents_failed": progress.get("failed"),
        "entity_order_mode": entity_order_mode,
        "symmetric_predicates": list(symmetric_predicates),
        "empty_gold_policy": empty_gold_policy,
        **{key: value for key, value in exact.items() if key != "samples"},
        "graphjudge_metrics": graphjudge.get("metrics", {}),
        "graphjudge_samples_evaluated": graphjudge.get("samples_evaluated"),
        "graphjudge_samples_skipped_empty_gold": graphjudge.get("samples_skipped_empty_gold"),
        "graphjudge_metrics_file": str(target_dir / "graphjudge_metrics.json"),
        "exact_match_per_sample": exact["samples"],
    }
    _write_json(target_dir / "evaluation_summary.json", summary)
    return summary


def discover_targets(
    target_root: Path, dataset: Optional[str], models: Sequence[str]
) -> List[Path]:
    wanted_datasets = DATASET_ORDER if dataset in (None, "all") else (dataset,)
    wanted_models = {str(value).strip().casefold().replace("openai/", "") for value in models}
    targets = []
    for dataset_key in wanted_datasets:
        directory_names = DATASET_DIRECTORY_ALIASES.get(dataset_key, (dataset_key,))
        for directory_name in directory_names:
            root = target_root / directory_name
            if not root.is_dir():
                continue
            for target_dir in sorted(root.glob("*/20*")):
                if (
                    target_dir.parent.name.casefold() in wanted_models
                    and (target_dir / "run_state.json").is_file()
                ):
                    targets.append(target_dir)
    return targets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate completed GPT + Critic-Once runs offline."
    )
    parser.add_argument("--dataset", choices=(*DATASET_ORDER, "all"), default="all")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-4o-mini"])
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--metric-device", default=DEFAULT_METRIC_DEVICE)
    parser.add_argument("--metric-batch-size", type=int, default=DEFAULT_METRIC_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    target_root = Path(args.target_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    targets = discover_targets(target_root, args.dataset, args.models)
    if not targets:
        raise SystemExit("No completed GPT + Critic-Once runs found.")
    for target_dir in targets:
        summary = evaluate_target(
            target_dir,
            baseline_root,
            args.bert_model,
            args.metric_device,
            args.metric_batch_size,
        )
        print(
            f"[done] {summary['dataset']}/{summary['gpt_model']} "
            f"P/R/F1={summary['exact_match_precision']}/"
            f"{summary['exact_match_recall']}/{summary['exact_match_f1']} "
            f"G-BS-F1={summary['graphjudge_metrics'].get('G-BS-F1')} "
            f"G-BL-F1={summary['graphjudge_metrics'].get('G-BL-F1')} "
            f"G-RO-F1={summary['graphjudge_metrics'].get('G-RO-F1')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
