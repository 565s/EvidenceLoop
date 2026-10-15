from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path.cwd()

from evidenceloop.types import DocumentPipelineRecord, Triple  # noqa: E402
from evidenceloop.evaluation.graphjudge import evaluate_run  # noqa: E402
from tests.reference.checkpoints import (  # noqa: E402
    CHECKPOINT_VERSION,
    PROPOSER_MAX_ATTEMPTS,
    PROPOSER_RETRY_DELAYS,
    _read_json_object,
    atomic_write_json,
    atomic_write_jsonl,
    checkpoint_path,
    document_ids_sha256,
    gold_keys,
    load_dotenv_file,
    load_graphjudge_rebel,
    normalize_symmetric_predicates,
    pred_keys,
    progress_payload,
    validate_entity_order_config,
)
from evidenceloop.resources.prompt_loader import (  # noqa: E402
    load_proposer_extract_call_prompt,
    load_proposer_system_prompt,
)
from evidenceloop.validation import clean_predicate, predicate_validation_error  # noqa: E402


DEFAULT_BASE_URL = ""
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RUN_NAME = "gpt_baselines"
DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_METRIC_DEVICE = "auto"
DEFAULT_METRIC_BATCH_SIZE = 64
MODEL_ALIASES = {
    "gpt-4omini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "openai/gpt-4o": "gpt-4o",
}
OFOX_MODEL_IDS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}
DEFAULT_MODELS = ("gpt-4o", "gpt-4o-mini")
DATASET_CHOICES = ("rebel_sub", "redocred", "biored", "scierc", "all")
SCIERC_SYMMETRIC_PREDICATES = ("is compared with", "is conjoined with")


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    label: str
    data_dir: str
    entity_order_mode: str = "directed"
    symmetric_predicates: Tuple[str, ...] = ()


from evidenceloop.data.registry import DATASETS as DATASET_SPECS, AliasDict
DATASETS = AliasDict({key: DatasetPreset(key, spec.label, "data/pipeline/" + spec.directory,
    spec.entity_order_mode, spec.symmetric_predicates) for key, spec in DATASET_SPECS.items()})


def _resolve_repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_model_name(value: str) -> str:
    model = str(value or "").strip()
    normalized = MODEL_ALIASES.get(model, model)
    if normalized not in {"gpt-4o", "gpt-4o-mini"}:
        raise ValueError("model must be gpt-4o, gpt-4o-mini, or gpt-4omini")
    return normalized


def ofox_model_id(value: str) -> str:
    return OFOX_MODEL_IDS[normalize_model_name(value)]


def selected_dataset_keys(value: str) -> List[str]:
    if value == "all":
        return ["rebel_sub", "redocred", "biored", "scierc"]
    if value not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}")
    return [value]


def count_dataset_records(data_dir: Path) -> int:
    documents_path = data_dir / "documents.jsonl"
    if documents_path.exists():
        return sum(1 for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip())
    source_path = data_dir / "test.source"
    if source_path.exists():
        return sum(1 for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip())
    raise FileNotFoundError(f"missing documents.jsonl or test.source under {data_dir}")


def load_dataset_rows(data_dir: Path, limit: Optional[int]) -> Tuple[List[Dict], int]:
    selected_limit = limit if limit is not None else count_dataset_records(data_dir)
    if not isinstance(selected_limit, int) or selected_limit <= 0:
        raise ValueError("limit must be positive")
    return load_graphjudge_rebel(data_dir, selected_limit), selected_limit


from evidenceloop.baselines.parsing import _extract_json_obj, parse_gpt_response


class GPTBaselineExtractor:
    def __init__(self, model_name: str, llm=None, *, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model_name = normalize_model_name(model_name)
        self.system_prompt = load_proposer_system_prompt()
        self.call_prompt = load_proposer_extract_call_prompt()
        self.llm = llm or self._build_llm(temperature=temperature, max_tokens=max_tokens)

    def _build_llm(self, *, temperature: float, max_tokens: int):
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OFOX_API_KEY")
        if not api_key:
            raise RuntimeError("OFOX_API_KEY is required in .env or the environment")
        return ChatOpenAI(
            model=ofox_model_id(self.model_name),
            base_url=os.environ.get("OFOX_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=0,
        )

    def extract(self, document_id: str, text: str) -> Dict:
        response = self.llm.invoke([
            ("system", self.system_prompt),
            ("user", self.call_prompt.format(text=text)),
        ])
        content = getattr(response, "content", response)
        return parse_gpt_response(str(content or "").strip(), document_id, text)


def serialize_payload(payload: Mapping) -> Dict:
    return {
        "triples": [
            item.to_dict() if isinstance(item, Triple) else dict(item)
            for item in payload.get("triples", [])
        ],
        "raw_response": str(payload.get("raw_response") or ""),
        "parse_error": str(payload.get("parse_error") or ""),
        "rejected_predicates": list(payload.get("rejected_predicates") or []),
    }


def triples_from_checkpoint(value: Mapping, row: Mapping) -> List[Triple]:
    return [
        Triple.from_dict(dict(item), source_text=row["text"], document_id=row["document_id"])
        for item in value.get("final_triples", [])
        if isinstance(item, Mapping)
    ]


def sample_result(
    row: Mapping,
    triples: Sequence[Triple],
    *,
    status: str,
    error: str,
    elapsed: float,
    rejected_predicates: Sequence[Mapping] = (),
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
) -> Dict:
    matches = gold_keys(
        row["gold_triples"], entity_order_mode, symmetric_predicates
    ) & pred_keys(
        triples, entity_order_mode, symmetric_predicates
    )
    return {
        "document_id": row["document_id"],
        "status": status,
        "error": error,
        "gold_count": len(row["gold_triples"]),
        "predicted_count": len(triples),
        "exact_match_count": len(matches),
        "rejected_predicates": len(rejected_predicates),
        "elapsed_seconds": round(elapsed, 3),
    }


def validate_checkpoint(value: Mapping, index: int, row: Mapping, path: Path) -> Dict:
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version")
    if value.get("index") != index or value.get("document_id") != row["document_id"]:
        raise ValueError(f"{path}: checkpoint does not match selected dataset order")
    if value.get("status") not in {"ok", "failed"}:
        raise ValueError(f"{path}: checkpoint status must be ok or failed")
    if not isinstance(value.get("attempts"), int) or value["attempts"] <= 0:
        raise ValueError(f"{path}: checkpoint attempts must be positive")
    if value.get("status") == "ok" and not isinstance(value.get("final_triples"), list):
        raise ValueError(f"{path}: successful checkpoint requires final_triples")
    return dict(value)


def load_checkpoints(checkpoint_dir: Path, rows: Sequence[Mapping]) -> Dict[int, Dict]:
    checkpoints = {}
    if not checkpoint_dir.exists():
        return checkpoints
    for index, row in enumerate(rows):
        path = checkpoint_path(checkpoint_dir, index)
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: checkpoint must be a JSON object")
        checkpoints[index] = validate_checkpoint(value, index, row, path)
    return checkpoints


def build_record(
    row: Mapping,
    triples: Sequence[Triple],
    raw_response: str,
    elapsed: float,
    *,
    stage_name: str = "gpt_extraction",
) -> DocumentPipelineRecord:
    return DocumentPipelineRecord(
        document_id=row["document_id"],
        source_text=row["text"],
        extracted_triples=list(triples),
        refined_triples=list(triples),
        final_triples=list(triples),
        stage_timings_seconds={stage_name: round(elapsed, 6)},
        raw_responses={stage_name: raw_response},
    )


def process_rows_with_checkpoints(
    extractor: GPTBaselineExtractor,
    rows: Sequence[Mapping],
    checkpoint_dir: Path,
    progress_path: Path,
    *,
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
    stage_name: str = "gpt_extraction",
    sleep_fn=time.sleep,
    console_print=print,
) -> Dict[int, Dict]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = load_checkpoints(checkpoint_dir, rows)
    update_progress(progress_path, progress_payload(rows, checkpoints))

    for index, row in enumerate(rows):
        previous = checkpoints.get(index)
        if previous and previous["status"] == "ok":
            console_print(f"[{index + 1}/{len(rows)}] {row['document_id']} checkpoint ok; skipped")
            continue

        attempts = int(previous.get("attempts", 0)) if previous else 0
        status = "failed"
        error = ""
        payload = None
        elapsed = 0.0
        started = time.perf_counter()
        for local_attempt in range(1, PROPOSER_MAX_ATTEMPTS + 1):
            attempts += 1
            console_print(f"[{index + 1}/{len(rows)}] {row['document_id']} start (attempt {attempts})")
            try:
                candidate = extractor.extract(row["document_id"], row["text"])
                parse_error = str(candidate.get("parse_error") or "")
                if parse_error:
                    error = parse_error
                else:
                    payload = serialize_payload(candidate)
                    status = "ok"
                    error = ""
                    break
            except Exception as exc:
                error = str(exc)
                traceback.print_exc()
            if local_attempt < PROPOSER_MAX_ATTEMPTS:
                sleep_fn(PROPOSER_RETRY_DELAYS[local_attempt - 1])
        elapsed = time.perf_counter() - started

        triples = [
            Triple.from_dict(item, source_text=row["text"], document_id=row["document_id"])
            for item in (payload or {}).get("triples", [])
        ] if payload else []
        result = sample_result(
            row,
            triples,
            status=status,
            error=error,
            elapsed=elapsed,
            rejected_predicates=(payload or {}).get("rejected_predicates", []),
            entity_order_mode=entity_order_mode,
            symmetric_predicates=symmetric_predicates,
        )
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "index": index,
            "document_id": row["document_id"],
            "attempts": attempts,
            "status": status,
            "error": error,
            "sample_result": result,
            "payload": payload,
            "pipeline_record": build_record(
                row,
                triples,
                (payload or {}).get("raw_response", ""),
                elapsed,
                stage_name=stage_name,
            ).to_dict() if status == "ok" else None,
            "final_triples": [triple.to_dict() for triple in triples],
        }
        atomic_write_json(checkpoint_path(checkpoint_dir, index), checkpoint)
        checkpoints[index] = checkpoint
        update_progress(progress_path, progress_payload(rows, checkpoints))
        console_print(
            f"[{index + 1}/{len(rows)}] {row['document_id']} {status}; "
            f"pred={len(triples)} exact={result['exact_match_count']}"
        )
    return checkpoints


def update_progress(path: Path, value: Mapping) -> None:
    merged = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, Mapping):
            merged.update(existing)
    merged.update(value)
    atomic_write_json(path, merged)


def build_runtime_signature(
    *,
    dataset: DatasetPreset,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Dict:
    return {
        "baseline": "gpt_direct_extraction",
        "dataset": dataset.key,
        "model": normalize_model_name(model_name),
        "base_url": os.environ.get("OFOX_BASE_URL") or DEFAULT_BASE_URL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_attempts": PROPOSER_MAX_ATTEMPTS,
        "entity_order_mode": dataset.entity_order_mode,
        "symmetric_predicates": list(normalize_symmetric_predicates(dataset.symmetric_predicates)),
    }


def build_run_state(
    *,
    data_dir: Path,
    requested_limit: int,
    rows: Sequence[Mapping],
    run_name: str,
    runtime: Mapping,
) -> Dict:
    return {
        "version": CHECKPOINT_VERSION,
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data": {
            "data_dir": str(data_dir.resolve()),
            "requested_limit": requested_limit,
            "selected_documents": len(rows),
            "document_ids_sha256": document_ids_sha256(rows),
        },
        "runtime": dict(runtime),
    }


def validate_resume_state(saved: Mapping, expected: Mapping) -> None:
    if saved.get("version") != CHECKPOINT_VERSION:
        raise ValueError("run_state.json has an unsupported version")
    if saved.get("data") != expected.get("data"):
        raise ValueError("resume data directory, limit, document count, or ID order has changed")
    if dict(saved.get("runtime") or {}) != dict(expected.get("runtime") or {}):
        raise ValueError("resume dataset, model, or runtime configuration has changed")


def build_summary(
    rows: Sequence[Mapping],
    checkpoints: Mapping[int, Mapping],
    run_dir: Path,
    experiment_config: Mapping,
    *,
    active_wall_seconds: float,
) -> Dict:
    sample_results = [dict(checkpoints[index]["sample_result"]) for index in range(len(rows))]
    kg_triples = sum(item.get("predicted_count", 0) for item in sample_results if item.get("status") == "ok")
    gold_total = sum(item.get("gold_count", 0) for item in sample_results)
    pred_total = sum(item.get("predicted_count", 0) for item in sample_results)
    exact_total = sum(item.get("exact_match_count", 0) for item in sample_results)
    precision = exact_total / pred_total if pred_total else 0.0
    recall = exact_total / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    completed = sum(1 for item in sample_results if item.get("status") == "ok")
    return {
        "output_dir": str(run_dir),
        "samples_requested": len(sample_results),
        "samples_completed": completed,
        "samples_failed": len(sample_results) - completed,
        "kg_triples": kg_triples,
        "pipeline_records": completed,
        "gold_triples_total": gold_total,
        "predicted_spo_total": pred_total,
        "exact_match_spo_total": exact_total,
        "exact_match_precision": round(precision, 4),
        "exact_match_recall": round(recall, 4),
        "exact_match_f1": round(f1, 4),
        "rejected_predicates_total": sum(item.get("rejected_predicates", 0) for item in sample_results),
        "runtime_config": dict(experiment_config),
        "active_wall_seconds": round(active_wall_seconds, 3),
        "documents_per_minute": round(completed * 60 / active_wall_seconds, 3) if active_wall_seconds else 0.0,
        "samples": sample_results,
    }


def materialize_outputs(
    rows: Sequence[Mapping],
    checkpoint_dir: Path,
    run_dir: Path,
    experiment_config: Mapping,
    *,
    active_wall_seconds: float,
    stage_name: str = "gpt_extraction",
) -> Dict:
    checkpoints = load_checkpoints(checkpoint_dir, rows)
    if len(checkpoints) != len(rows):
        missing = [index for index in range(len(rows)) if index not in checkpoints]
        raise ValueError(f"cannot finalize run with missing checkpoints: {missing[:5]}")

    sample_results = []
    record_dicts = []
    kg_dicts = []
    for index, row in enumerate(rows):
        checkpoint = checkpoints[index]
        sample_results.append(dict(checkpoint["sample_result"]))
        if checkpoint["status"] != "ok":
            continue
        triples = triples_from_checkpoint(checkpoint, row)
        kg_dicts.extend(triple.to_dict() for triple in triples)
        record = build_record(
            row,
            triples,
            str((checkpoint.get("payload") or {}).get("raw_response") or ""),
            float(checkpoint["sample_result"].get("elapsed_seconds", 0.0) or 0.0),
            stage_name=stage_name,
        )
        record_dicts.append(record.to_dict())

    atomic_write_jsonl(run_dir / "gold_triples.jsonl", ({
        "document_id": row["document_id"],
        "text": row["text"],
        "triples": row["gold_triples"],
    } for row in rows))
    atomic_write_jsonl(run_dir / "sample_results.jsonl", sample_results)
    atomic_write_jsonl(run_dir / "pipeline_records.jsonl", record_dicts)
    atomic_write_jsonl(run_dir / "kg.jsonl", kg_dicts)
    atomic_write_json(run_dir / "pipeline_records.json", record_dicts)
    atomic_write_json(run_dir / "kg.json", kg_dicts)

    summary = build_summary(
        rows,
        checkpoints,
        run_dir,
        experiment_config,
        active_wall_seconds=active_wall_seconds,
    )
    atomic_write_json(run_dir / "summary.json", summary)
    progress = progress_payload(rows, checkpoints)
    progress["finalized"] = True
    progress["active_wall_seconds"] = round(active_wall_seconds, 3)
    atomic_write_json(run_dir / "progress.json", progress)
    return summary


def evaluate_soft_metrics(
    run_dir: Path,
    dataset: DatasetPreset,
    *,
    bert_model: str,
    metric_device: str,
    metric_batch_size: int,
) -> Dict:
    return evaluate_run(
        run_dir=run_dir,
        gold_path=run_dir / "gold_triples.jsonl",
        prediction_path=run_dir / "kg.jsonl",
        output_path=run_dir / "graphjudge_metrics.json",
        per_sample_output_path=run_dir / "graphjudge_metrics_per_sample.jsonl",
        bert_model=bert_model,
        requested_device=metric_device,
        batch_size=metric_batch_size,
        matching_mode="hungarian",
        entity_order_mode=dataset.entity_order_mode,
        symmetric_predicates=dataset.symmetric_predicates,
    )


def _dataset_from_runtime(runtime: Mapping) -> DatasetPreset:
    key = str(runtime.get("dataset") or "")
    if key not in DATASETS:
        raise ValueError(f"saved run_state contains unknown dataset {key!r}")
    return DATASETS[key]


def run_experiment(args, *, extractor: Optional[GPTBaselineExtractor] = None) -> Path:
    load_dotenv_file(ROOT / ".env")

    saved_state = None
    if args.resume_run_dir:
        run_dir = _resolve_repo_path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        saved_state = _read_json_object(run_dir / "run_state.json")
        saved_runtime = saved_state.get("runtime") or {}
        dataset = _dataset_from_runtime(saved_runtime)
        data_dir = Path((saved_state.get("data") or {}).get("data_dir", ""))
        requested_limit = int((saved_state.get("data") or {}).get("requested_limit", 0))
        model_name = normalize_model_name(saved_runtime.get("model"))
        temperature = float(saved_runtime.get("temperature", DEFAULT_TEMPERATURE))
        max_tokens = int(saved_runtime.get("max_tokens", DEFAULT_MAX_TOKENS))
        run_name = str(saved_state.get("run_name") or DEFAULT_RUN_NAME)
    else:
        dataset = DATASETS[args.dataset]
        data_dir = _resolve_repo_path(dataset.data_dir).resolve()
        raw_model = getattr(args, "model", None)
        if raw_model is None:
            models = list(getattr(args, "models", []) or [])
            if len(models) != 1:
                raise ValueError("run_experiment requires exactly one model")
            raw_model = models[0]
        model_name = normalize_model_name(raw_model)
        temperature = float(args.temperature)
        max_tokens = int(args.max_tokens)
        run_name = args.run_name or DEFAULT_RUN_NAME
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "data" / "experiments" / "gpt_baselines" / dataset.key / model_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        requested_limit = args.limit if args.limit is not None else count_dataset_records(data_dir)

    validate_entity_order_config(dataset.entity_order_mode, dataset.symmetric_predicates)
    rows, selected_limit = load_dataset_rows(data_dir, requested_limit)
    runtime = build_runtime_signature(
        dataset=dataset,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    expected_state = build_run_state(
        data_dir=data_dir,
        requested_limit=selected_limit,
        rows=rows,
        run_name=run_name,
        runtime=runtime,
    )
    if saved_state is not None:
        validate_resume_state(saved_state, expected_state)
    else:
        atomic_write_json(run_dir / "run_state.json", expected_state)

    console = sys.__stdout__

    def console_print(message: str) -> None:
        print(message, file=console, flush=True)

    console_print(f"[setup] loaded {len(rows)} {dataset.label} samples for {model_name}")
    console_print(f"[setup] output dir: {run_dir}")
    if saved_state is not None:
        console_print("[setup] resume state validated")

    log_mode = "a" if saved_state is not None else "w"
    progress_path = run_dir / "progress.json"
    previous_progress = _read_json_object(progress_path) if progress_path.exists() else {}
    previous_active_seconds = float(previous_progress.get("active_wall_seconds", 0.0) or 0.0)
    processing_started = time.perf_counter()
    with (run_dir / "run.log").open(log_mode, encoding="utf-8", buffering=1) as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        print(
            f"{dataset.label} GPT baseline run {'resumed' if saved_state else 'started'} at "
            f"{datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        print(f"Runtime config: {json.dumps(runtime, ensure_ascii=False, indent=2)}", flush=True)
        active_wall_seconds = previous_active_seconds
        try:
            extractor = extractor or GPTBaselineExtractor(
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            process_rows_with_checkpoints(
                extractor,
                rows,
                run_dir / "checkpoints",
                progress_path,
                entity_order_mode=dataset.entity_order_mode,
                symmetric_predicates=dataset.symmetric_predicates,
                console_print=console_print,
            )
        finally:
            active_wall_seconds = round(
                previous_active_seconds + time.perf_counter() - processing_started,
                3,
            )
            if progress_path.exists():
                progress = _read_json_object(progress_path)
                progress["active_wall_seconds"] = active_wall_seconds
                atomic_write_json(progress_path, progress)

        summary = materialize_outputs(
            rows,
            run_dir / "checkpoints",
            run_dir,
            runtime,
            active_wall_seconds=active_wall_seconds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if not args.skip_soft_metrics:
        evaluate_soft_metrics(
            run_dir,
            dataset,
            bert_model=args.bert_model,
            metric_device=args.metric_device,
            metric_batch_size=args.metric_batch_size,
        )

    summary = _read_json_object(run_dir / "summary.json")
    console_print(
        "[done] "
        f"completed={summary['samples_completed']}/{summary['samples_requested']} "
        f"kg={summary['kg_triples']} "
        f"exact={summary['exact_match_spo_total']}/{summary['gold_triples_total']} "
        f"p/r/f1={summary['exact_match_precision']}/{summary['exact_match_recall']}/{summary['exact_match_f1']}"
    )
    if not args.skip_soft_metrics:
        console_print(f"[done] soft metrics: {run_dir / 'graphjudge_metrics.json'}")
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-4o and GPT-4o-mini direct extraction baselines."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--skip-soft-metrics", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--metric-device", default=DEFAULT_METRIC_DEVICE)
    parser.add_argument("--metric-batch-size", type=int, default=DEFAULT_METRIC_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.resume_run_dir:
        run_experiment(args)
        return 0

    run_dirs = []
    for dataset_key in selected_dataset_keys(args.dataset):
        for model in args.models:
            child_args = argparse.Namespace(**vars(args))
            child_args.dataset = dataset_key
            child_args.model = normalize_model_name(model)
            run_dirs.append(run_experiment(child_args))

    print("Completed GPT baseline runs:")
    for run_dir in run_dirs:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
