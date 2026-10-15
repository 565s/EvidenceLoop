"""Evaluate DeepSeek ME ablation stages without invoking inference models."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.paired import DATASETS, load_dataset_rows  # noqa: E402
from evidenceloop.data.registry import DATASETS as DATASET_SPECS
from evidenceloop.data.registry import AliasDict
from evidenceloop.evaluation.graphjudge import METRIC_NAMES, evaluate_run  # noqa: E402
from evidenceloop.experiments.checkpoints import (  # noqa: E402
    gold_keys,
    normalize_symmetric_predicates,
    pred_keys,
    validate_entity_order_config,
)
from evidenceloop.io import atomic_write_json, atomic_write_jsonl
from evidenceloop.types import Triple  # noqa: E402

ROOT = Path.cwd()


DEFAULT_OUTPUT_DIR = ROOT / "data" / "experiments" / "ablation_stage_evaluation"
DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_METRIC_DEVICE = "auto"
DEFAULT_METRIC_BATCH_SIZE = 64
DEFAULT_BASELINE_ROOT = ROOT / "data" / "experiments" / "deepseek_baselines"
DEFAULT_INITIAL_CRITIC_ROOT = ROOT / "data" / "experiments" / "deepseek_proposer_critic"
DEFAULT_INTEGRATOR_ROOT = ROOT / "data" / "experiments" / "deepseek_proposer_critic_integrator"
DEFAULT_FINAL_CRITIC_ROOT = (
    ROOT / "data" / "experiments" / "deepseek_proposer_critic_integrator_critic2"
)
DATASET_ORDER = ("rebel_sub", "redocred", "biored", "scierc")
STAGE_ORDER = ("proposer", "initial_critic", "integrator", "final_critic")
DATASET_LABELS = {
    "rebel_sub": "REBEL_Sub",
    "redocred": "Re-DocRED",
    "biored": "BioRED",
    "scierc": "SciERC",
}
STAGE_LABELS = {
    "proposer": "Proposer",
    "initial_critic": "+ Initial Critic",
    "integrator": "+ Integrator",
    "final_critic": "+ Final Critic",
}
DATASET_CONFIG = AliasDict(
    {
        key: (spec.entity_order_mode, spec.symmetric_predicates, spec.empty_gold_policy)
        for key, spec in DATASET_SPECS.items()
    }
)


@dataclass(frozen=True)
class StageTarget:
    dataset: str
    stage: str
    run_dir: Path


def _read_json(path: Path) -> Dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}: line {line_number}: expected a JSON object")
            rows.append(dict(value))
    return rows


def _record_id(row: Mapping, context: str) -> str:
    document_id = str(row.get("document_id") or row.get("sample_id") or "").strip()
    if not document_id:
        raise ValueError(f"{context}: missing document_id")
    return document_id


def _dataset_config(dataset: str) -> Tuple[str, Tuple[str, ...], str]:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"unsupported dataset: {dataset}")
    entity_order_mode, predicates, empty_gold_policy = DATASET_CONFIG[dataset]
    symmetric_predicates = normalize_symmetric_predicates(predicates)
    validate_entity_order_config(entity_order_mode, symmetric_predicates)
    return entity_order_mode, symmetric_predicates, empty_gold_policy


def _latest_timestamp_dir(root: Path) -> Path:
    if root.is_dir() and (root / "run_state.json").is_file():
        return root
    candidates = [
        path for path in root.glob("20*") if path.is_dir() and (path / "run_state.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no timestamped run with run_state.json under {root}")
    return sorted(candidates)[-1]


def _stage_root(
    dataset: str,
    stage: str,
    *,
    baseline_root: Path,
    initial_critic_root: Path,
    integrator_root: Path,
    final_critic_root: Path,
) -> Path:
    if stage == "proposer":
        return baseline_root / dataset / "deepseek-v4-flash"
    if stage == "initial_critic":
        return initial_critic_root / dataset / "deepseek-v4-flash"
    if stage == "integrator":
        return integrator_root / dataset
    if stage == "final_critic":
        return final_critic_root / dataset
    raise ValueError(f"unsupported stage: {stage}")


def discover_targets(
    datasets: Sequence[str],
    stages: Sequence[str],
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    initial_critic_root: Path = DEFAULT_INITIAL_CRITIC_ROOT,
    integrator_root: Path = DEFAULT_INTEGRATOR_ROOT,
    final_critic_root: Path = DEFAULT_FINAL_CRITIC_ROOT,
) -> List[StageTarget]:
    targets: List[StageTarget] = []
    for dataset in datasets:
        for stage in stages:
            root = _stage_root(
                dataset,
                stage,
                baseline_root=baseline_root,
                initial_critic_root=initial_critic_root,
                integrator_root=integrator_root,
                final_critic_root=final_critic_root,
            )
            targets.append(StageTarget(dataset, stage, _latest_timestamp_dir(root)))
    return targets


def _validate_complete_run(target: StageTarget) -> Dict:
    progress_path = target.run_dir / "progress.json"
    kg_path = target.run_dir / "kg.jsonl"
    if not progress_path.is_file():
        raise FileNotFoundError(progress_path)
    if not kg_path.is_file():
        raise FileNotFoundError(kg_path)
    progress = _read_json(progress_path)
    total = int(progress.get("documents_total") or 0)
    completed = int(progress.get("completed") or 0)
    failed = int(progress.get("failed") or 0)
    pending = int(progress.get("pending") or 0)
    if not progress.get("finalized") or completed != total or failed or pending:
        raise ValueError(
            f"{target.run_dir}: incomplete run: total={total}, completed={completed}, "
            f"failed={failed}, pending={pending}, finalized={progress.get('finalized')}"
        )
    return progress


def _gold_rows_for_dataset(dataset: str, expected_limit: int) -> List[Dict]:
    preset = DATASETS[dataset]
    rows, selected_limit = load_dataset_rows(ROOT / preset.data_dir, expected_limit)
    if selected_limit != expected_limit:
        raise ValueError(
            f"{dataset}: selected limit {selected_limit} differs from expected {expected_limit}"
        )
    return [
        {
            "document_id": row["document_id"],
            "text": row["text"],
            "triples": row["gold_triples"],
        }
        for row in rows
    ]


def _prediction_rows(run_dir: Path) -> List[Dict]:
    rows = _read_jsonl(run_dir / "kg.jsonl")
    for index, row in enumerate(rows, 1):
        for field in ("subject", "predicate", "object"):
            if not str(row.get(field) or "").strip():
                raise ValueError(f"{run_dir / 'kg.jsonl'}: row {index}: missing {field}")
    return rows


def _validate_document_alignment(
    target: StageTarget,
    gold_rows: Sequence[Mapping],
    prediction_rows: Sequence[Mapping],
) -> None:
    gold_ids = [
        _record_id(row, f"{target.dataset} gold row {i}") for i, row in enumerate(gold_rows, 1)
    ]
    predicted_ids = [
        _record_id(row, f"{target.run_dir / 'kg.jsonl'} row {i}")
        for i, row in enumerate(prediction_rows, 1)
    ]
    unknown = sorted(set(predicted_ids) - set(gold_ids))
    if unknown:
        raise ValueError(
            f"{target.run_dir}: predictions contain unknown document ids: {unknown[:5]}"
        )
    documents_path = target.run_dir / "documents.jsonl"
    if documents_path.is_file():
        document_ids = [
            _record_id(row, f"{documents_path} row {i}")
            for i, row in enumerate(_read_jsonl(documents_path), 1)
        ]
        if document_ids != gold_ids:
            raise ValueError(f"{target.run_dir}: documents.jsonl order does not match gold")


def _exact_metrics(
    gold_rows: Sequence[Mapping],
    prediction_rows: Sequence[Mapping],
    entity_order_mode: str,
    symmetric_predicates: Sequence[str],
) -> Dict:
    gold_by_id = {_record_id(row, "gold"): row for row in gold_rows}
    predicted_by_id: Dict[str, List[Triple]] = {document_id: [] for document_id in gold_by_id}
    for index, row in enumerate(prediction_rows, 1):
        document_id = _record_id(row, f"prediction row {index}")
        if document_id not in predicted_by_id:
            raise ValueError(f"prediction row {index}: unknown document_id {document_id}")
        predicted_by_id[document_id].append(Triple.from_dict(dict(row), document_id=document_id))

    gold_total = sum(len(row.get("triples") or []) for row in gold_rows)
    predicted_total = sum(len(values) for values in predicted_by_id.values())
    exact_total = 0
    for row in gold_rows:
        document_id = _record_id(row, "gold")
        matches = gold_keys(
            row.get("triples") or [], entity_order_mode, symmetric_predicates
        ) & pred_keys(predicted_by_id[document_id], entity_order_mode, symmetric_predicates)
        exact_total += len(matches)
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
    }


def _load_or_evaluate_graphjudge(
    target: StageTarget,
    entity_order_mode: str,
    symmetric_predicates: Sequence[str],
    empty_gold_policy: str,
    bert_model: str,
    metric_device: str,
    metric_batch_size: int,
    force: bool,
) -> Dict:
    output_path = target.run_dir / "graphjudge_metrics.json"
    per_sample_path = target.run_dir / "graphjudge_metrics_per_sample.jsonl"
    if output_path.is_file() and per_sample_path.is_file() and not force:
        return _read_json(output_path)
    return evaluate_run(
        run_dir=target.run_dir,
        gold_path=target.run_dir / "gold_triples.jsonl",
        prediction_path=target.run_dir / "kg.jsonl",
        output_path=output_path,
        per_sample_output_path=per_sample_path,
        bert_model=bert_model,
        requested_device=metric_device,
        batch_size=metric_batch_size,
        matching_mode="hungarian",
        entity_order_mode=entity_order_mode,
        symmetric_predicates=symmetric_predicates,
        empty_gold_policy=empty_gold_policy,
    )


def evaluate_target(
    target: StageTarget,
    *,
    bert_model: str,
    metric_device: str,
    metric_batch_size: int,
    force: bool,
) -> Dict:
    progress = _validate_complete_run(target)
    completed = int(progress.get("completed") or 0)
    entity_order_mode, symmetric_predicates, empty_gold_policy = _dataset_config(target.dataset)
    gold_rows = _gold_rows_for_dataset(target.dataset, completed)
    prediction_rows = _prediction_rows(target.run_dir)
    _validate_document_alignment(target, gold_rows, prediction_rows)
    atomic_write_jsonl(target.run_dir / "gold_triples.jsonl", gold_rows)
    exact = _exact_metrics(gold_rows, prediction_rows, entity_order_mode, symmetric_predicates)
    graphjudge = _load_or_evaluate_graphjudge(
        target,
        entity_order_mode,
        symmetric_predicates,
        empty_gold_policy,
        bert_model,
        metric_device,
        metric_batch_size,
        force,
    )
    metrics = graphjudge.get("metrics") or {}
    missing_metrics = [name for name in METRIC_NAMES if name not in metrics]
    if missing_metrics:
        raise ValueError(f"{target.run_dir}: missing GraphJudge metrics: {missing_metrics}")
    triples_total = len(prediction_rows)
    triples_per_doc = triples_total / completed if completed else 0.0
    result = {
        "dataset": target.dataset,
        "dataset_label": DATASET_LABELS[target.dataset],
        "stage": target.stage,
        "variant": STAGE_LABELS[target.stage],
        "run_dir": str(target.run_dir),
        "documents_total": completed,
        "triples_total": triples_total,
        "triples_per_doc": round(triples_per_doc, 4),
        "entity_order_mode": entity_order_mode,
        "symmetric_predicates": list(symmetric_predicates),
        "empty_gold_policy": empty_gold_policy,
        **exact,
        **{name: round(float(metrics[name]), 4) for name in METRIC_NAMES},
        "graphjudge_samples_evaluated": graphjudge.get("samples_evaluated"),
        "graphjudge_samples_skipped_empty_gold": graphjudge.get("samples_skipped_empty_gold"),
    }
    atomic_write_json(target.run_dir / "ablation_evaluation_summary.json", result)
    return result


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    atomic_write_jsonl(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    fieldnames = [
        "dataset_label",
        "variant",
        "documents_total",
        "triples_total",
        "triples_per_doc",
        "G-BS-F1",
        "G-BL-F1",
        "G-RO-F1",
        "exact_match_precision",
        "exact_match_recall",
        "exact_match_f1",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _markdown_table(rows: Sequence[Mapping]) -> str:
    lines = [
        "| Dataset | Variant | Triples/Doc | G-BS-F1 | G-BL-F1 | G-RO-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {variant} | {triples} | {gbs:.4f} | {gbl:.4f} | {gro:.4f} |".format(
                dataset=row["dataset_label"],
                variant=row["variant"],
                triples=int(row["triples_total"]),
                gbs=float(row["G-BS-F1"]),
                gbl=float(row["G-BL-F1"]),
                gro=float(row["G-RO-F1"]),
            )
        )
    return "\n".join(lines) + "\n"


def _latex_table(rows: Sequence[Mapping]) -> str:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Variant & Triples/Doc & G-BS-F1 & G-BL-F1 & G-RO-F1 \\\\",
        "\\midrule",
    ]
    previous_dataset: Optional[str] = None
    for row in rows:
        dataset = str(row["dataset_label"])
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append("\\midrule")
        previous_dataset = dataset
        lines.append(
            "{dataset} & {variant} & {triples} & {gbs:.4f} & {gbl:.4f} & {gro:.4f} \\\\".format(
                dataset=dataset.replace("_", "\\_"),
                variant=str(row["variant"]).replace("+", "$+$").replace("_", "\\_"),
                triples=int(row["triples_total"]),
                gbs=float(row["G-BS-F1"]),
                gbl=float(row["G-BL-F1"]),
                gro=float(row["G-RO-F1"]),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_outputs(rows: Sequence[Mapping], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "deepseek_me_ablation_summary.json"
    atomic_write_json(summary_path, {"rows": list(rows)})
    _write_jsonl(output_dir / "deepseek_me_ablation_summary.jsonl", rows)
    _write_csv(output_dir / "deepseek_me_ablation_summary.csv", rows)
    (output_dir / "deepseek_me_ablation_table.md").write_text(
        _markdown_table(rows),
        encoding="utf-8",
    )
    (output_dir / "deepseek_me_ablation_table.tex").write_text(
        _latex_table(rows),
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASET_ORDER, "all"), default="all")
    parser.add_argument("--stage", choices=(*STAGE_ORDER, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument(
        "--initial-critic-root",
        type=Path,
        default=DEFAULT_INITIAL_CRITIC_ROOT,
    )
    parser.add_argument("--integrator-root", type=Path, default=DEFAULT_INTEGRATOR_ROOT)
    parser.add_argument(
        "--final-critic-root",
        type=Path,
        default=DEFAULT_FINAL_CRITIC_ROOT,
    )
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--metric-device", default=DEFAULT_METRIC_DEVICE)
    parser.add_argument("--metric-batch-size", type=int, default=DEFAULT_METRIC_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="recompute existing GraphJudge files")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate target discovery without writing metrics"
    )
    return parser.parse_args(argv)


def _selected(value: str, order: Sequence[str]) -> Tuple[str, ...]:
    return tuple(order if value == "all" else (value,))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    datasets = _selected(args.dataset, DATASET_ORDER)
    stages = _selected(args.stage, STAGE_ORDER)
    targets = discover_targets(
        datasets,
        stages,
        baseline_root=args.baseline_root.resolve(),
        initial_critic_root=args.initial_critic_root.resolve(),
        integrator_root=args.integrator_root.resolve(),
        final_critic_root=args.final_critic_root.resolve(),
    )
    if args.dry_run:
        for target in targets:
            progress = _validate_complete_run(target)
            print(
                f"[dry-run] {DATASET_LABELS[target.dataset]} {STAGE_LABELS[target.stage]} "
                f"docs={progress.get('completed')} run={target.run_dir}"
            )
        return 0

    rows: List[Dict] = []
    for target in targets:
        print(
            f"[eval] {DATASET_LABELS[target.dataset]} {STAGE_LABELS[target.stage]}: {target.run_dir}"
        )
        rows.append(
            evaluate_target(
                target,
                bert_model=args.bert_model,
                metric_device=args.metric_device,
                metric_batch_size=args.metric_batch_size,
                force=args.force,
            )
        )
    write_outputs(rows, args.output_dir.resolve())
    print(f"[done] wrote summary to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
