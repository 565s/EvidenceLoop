"""Shared checkpoint and source validation for offline pipeline stages."""

from __future__ import annotations

from evidenceloop.serialization import pipeline_record_from_dict
from evidenceloop.serialization import _review_from_dict
from evidenceloop.serialization import _triple_from_dict
from evidenceloop.io import load_dotenv_file
from evidenceloop.io import atomic_write_jsonl
from evidenceloop.io import atomic_write_json
from evidenceloop.io import _atomic_write_text

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.types import DocumentPipelineRecord, Triple, TripleReviewRecord


ROOT = Path.cwd()
CHECKPOINT_VERSION = 1
RUN_STATE_VERSION = 1
DATASET_CHOICES = ("rebel_sub", "redocred", "biored", "scierc", "all")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    limit: int
    source_root: Path


DATASETS: Dict[str, DatasetSpec] = {
    "rebel_sub": DatasetSpec(
        "rebel_sub",
        1000,
        ROOT / "data" / "experiments" / "rebel_phase2_sft_test_1000_once_critic_direct_p4_b4",
    ),
    "redocred": DatasetSpec(
        "redocred",
        500,
        ROOT / "data" / "experiments" / "re_docred_phase2_once_critic",
    ),
    "biored": DatasetSpec(
        "biored",
        100,
        ROOT / "data" / "experiments" / "biored_phase2_once_500",
    ),
    "scierc": DatasetSpec(
        "scierc",
        100,
        ROOT / "data" / "experiments" / "scierc_phase2_once",
    ),
}

from evidenceloop.data.registry import AliasDict
DATASETS = AliasDict(DATASETS)


@dataclass(frozen=True)
class OfflineSourceRun:
    path: Path
    dataset: str
    timestamp: str
    document_count: int
    document_ids_sha256: str
    source_records_sha256: str
    critic_runtime: Dict
    source_kind: str


@dataclass(frozen=True)
class SourceDocument:
    index: int
    document_id: str
    text: str
    pipeline_record: Dict


def resolve_path(value) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def selected_dataset_keys(value: str) -> List[str]:
    if value == "all":
        return list(DATASETS)
    if value not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}")
    return [value]


def read_json_object(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)










def document_ids_sha256(document_ids: Sequence[str]) -> str:
    payload = "".join(f"{document_id}\n" for document_id in document_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_sha256(value: Mapping) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_from_legacy_state(state: Mapping, expected_dataset: Optional[str]) -> str:
    data = dict(state.get("data") or {})
    haystack = " ".join(
        (
            str(data.get("data_dir") or ""),
            str(state.get("run_name") or ""),
        )
    ).replace("\\", "/").casefold()
    markers = {
        "rebel_sub": ("rebel_phase2", "rebel_sub"),
        "redocred": ("redocred", "redocred"),
        "biored": ("biored",),
        "scierc": ("scierc",),
    }
    matches = [
        dataset
        for dataset, values in markers.items()
        if any(marker in haystack for marker in values)
    ]
    dataset = expected_dataset or (matches[0] if len(matches) == 1 else "")
    if dataset not in DATASETS:
        raise ValueError("could not identify the source dataset from run_state.json")
    if matches and dataset not in matches:
        raise ValueError(
            f"source dataset is {matches[0]!r}, not requested dataset {dataset!r}"
        )
    return dataset


def _critic1_source_identity(
    state: Mapping,
    expected_dataset: Optional[str],
) -> Tuple[str, int, int, str, str]:
    runtime = dict(state.get("runtime") or {})
    source = dict(state.get("source") or {})
    if runtime.get("offline_proposals") is True and source:
        dataset = str(source.get("dataset") or "")
        if dataset not in DATASETS:
            raise ValueError("could not identify the source dataset from run_state.json")
        if expected_dataset and dataset != expected_dataset:
            raise ValueError(
                f"source dataset is {dataset!r}, not requested dataset {expected_dataset!r}"
            )
        documents = int(source.get("documents", 0) or 0)
        fingerprint = str(source.get("document_ids_sha256") or "")
        timestamp = str(source.get("timestamp") or "")
        return dataset, documents, documents, fingerprint, timestamp

    dataset = _dataset_from_legacy_state(state, expected_dataset)
    data = dict(state.get("data") or {})
    selected = int(data.get("selected_documents", 0) or 0)
    requested = int(data.get("requested_limit", 0) or 0)
    fingerprint = str(data.get("document_ids_sha256") or "")
    return dataset, requested, selected, fingerprint, ""


def _validate_triple_list(
    values,
    *,
    path: Path,
    field: str,
    document_id: str,
    source_text: str,
) -> List[Dict]:
    if not isinstance(values, list):
        raise ValueError(f"{path}: {field} must be a list")
    triples = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: {field}[{index}] must be an object")
        triple = dict(item)
        if str(triple.get("document_id") or "") != document_id:
            raise ValueError(f"{path}: {field}[{index}] has the wrong document_id")
        if str(triple.get("source_text") or "") != source_text:
            raise ValueError(f"{path}: {field}[{index}] has the wrong source_text")
        if any(
            not str(triple.get(name) or "").strip()
            for name in ("subject", "predicate", "object")
        ):
            raise ValueError(f"{path}: {field}[{index}] has empty fields")
        triples.append(triple)
    return triples


def _validate_pipeline_record(
    record,
    *,
    path: Path,
    document_id: str,
    require_integrator: bool,
) -> Dict:
    if not isinstance(record, Mapping):
        raise ValueError(f"{path}: successful checkpoint requires pipeline_record")
    output = dict(record)
    if str(output.get("document_id") or "") != document_id:
        raise ValueError(f"{path}: pipeline record document_id does not match")
    source_text = str(output.get("source_text") or "")
    if not source_text.strip():
        raise ValueError(f"{path}: source_text is required")
    for field in (
        "extracted_triples",
        "refined_triples",
        "added_triples",
        "final_triples",
    ):
        _validate_triple_list(
            output.get(field),
            path=path,
            field=field,
            document_id=document_id,
            source_text=source_text,
        )
    for field in ("initial_reviews", "final_reviews"):
        if not isinstance(output.get(field), list):
            raise ValueError(f"{path}: {field} must be a list")
    if not isinstance(output.get("coverage_audit"), Mapping):
        raise ValueError(f"{path}: coverage_audit must be an object")

    audit = dict(output["coverage_audit"])
    if require_integrator:
        if audit.get("skipped") is True:
            raise ValueError(f"{path}: Integrator stage is marked skipped")
        if not isinstance(audit.get("additions"), list):
            raise ValueError(f"{path}: coverage_audit.additions must be a list")
        if output.get("final_reviews"):
            raise ValueError(f"{path}: Integrator source must not contain final reviews")
        accepted = [
            dict(item.get("triple") or {})
            for item in audit["additions"]
            if isinstance(item, Mapping) and item.get("accepted") is True
        ]
        if accepted != output.get("added_triples"):
            raise ValueError(
                f"{path}: accepted coverage additions do not match added_triples"
            )
    else:
        if audit.get("skipped") is not True:
            raise ValueError(f"{path}: source run did not stop after Critic1")
        if output.get("added_triples") or output.get("final_reviews"):
            raise ValueError(f"{path}: Critic1 source already contains later-stage output")
        seen = set()
        deduped = []
        for triple in output.get("refined_triples") or []:
            key = tuple(
                " ".join(str(triple.get(field) or "").casefold().split())
                for field in ("subject", "predicate", "object")
            )
            if key not in seen:
                seen.add(key)
                deduped.append(triple)
        if output.get("final_triples") != deduped:
            raise ValueError(
                f"{path}: Critic1 final_triples do not match deduped refined_triples"
            )
    return output


def _validate_source_checkpoint(
    path: Path,
    index: int,
    *,
    require_integrator: bool,
) -> Tuple[str, Dict, str]:
    value = read_json_object(path)
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version")
    if value.get("index") != index:
        raise ValueError(f"{path}: checkpoint index does not match filename")
    document_id = str(value.get("document_id") or "").strip()
    if not document_id:
        raise ValueError(f"{path}: document_id is required")
    if value.get("status") != "ok":
        raise ValueError(f"{path}: source checkpoint is not successful")
    record = _validate_pipeline_record(
        value.get("pipeline_record"),
        path=path,
        document_id=document_id,
        require_integrator=require_integrator,
    )
    if value.get("final_triples") != record.get("final_triples"):
        raise ValueError(f"{path}: checkpoint final_triples do not match pipeline record")
    digest = canonical_sha256(
        {"index": index, "document_id": document_id, "pipeline_record": record}
    )
    return document_id, record, digest


def _validate_complete_progress(path: Path, progress: Mapping, limit: int) -> None:
    completed = int(progress.get("completed", 0) or 0)
    failed = int(progress.get("failed", 0) or 0)
    pending = int(progress.get("pending", 0) or 0)
    if progress.get("finalized") is not True:
        raise ValueError(f"{path}: source run is not finalized")
    if completed != limit or failed or pending:
        raise ValueError(
            f"{path}: source run is incomplete "
            f"(completed={completed}, failed={failed}, pending={pending})"
        )


def inspect_critic1_run(
    run_dir: Path,
    expected_dataset: Optional[str] = None,
) -> OfflineSourceRun:
    run_dir = run_dir.resolve()
    state = read_json_object(run_dir / "run_state.json")
    progress = read_json_object(run_dir / "progress.json")
    dataset, requested, selected, recorded_fingerprint, source_timestamp = (
        _critic1_source_identity(state, expected_dataset)
    )
    spec = DATASETS[dataset]
    runtime = dict(state.get("runtime") or {})
    if selected != spec.limit or requested != spec.limit:
        raise ValueError(
            f"{run_dir}: expected a formal {spec.limit}-document run, "
            f"found requested={requested}, selected={selected}"
        )
    if runtime.get("pipeline_variant") != "proposer_critic_once":
        raise ValueError(f"{run_dir}: source is not a proposer + Critic1 run")
    critic_runtime = runtime.get("critic_runtime")
    if not isinstance(critic_runtime, Mapping):
        raise ValueError(f"{run_dir}: Critic1 runtime is missing")
    _validate_complete_progress(run_dir, progress, spec.limit)

    document_ids = []
    source_hasher = hashlib.sha256()
    checkpoint_dir = run_dir / "checkpoints"
    for index in range(spec.limit):
        document_id, _, digest = _validate_source_checkpoint(
            checkpoint_dir / f"{index:06d}.json",
            index,
            require_integrator=False,
        )
        document_ids.append(document_id)
        source_hasher.update(f"{digest}\n".encode("ascii"))
    fingerprint = document_ids_sha256(document_ids)
    if fingerprint != recorded_fingerprint:
        raise ValueError(f"{run_dir}: checkpoint document order does not match run_state")
    return OfflineSourceRun(
        path=run_dir,
        dataset=dataset,
        timestamp=source_timestamp or run_dir.name,
        document_count=spec.limit,
        document_ids_sha256=fingerprint,
        source_records_sha256=source_hasher.hexdigest(),
        critic_runtime=dict(critic_runtime),
        source_kind="critic1",
    )


def inspect_integrator_run(
    run_dir: Path,
    expected_dataset: Optional[str] = None,
) -> OfflineSourceRun:
    run_dir = run_dir.resolve()
    state = read_json_object(run_dir / "run_state.json")
    progress = read_json_object(run_dir / "progress.json")
    source_state = dict(state.get("source") or {})
    runtime = dict(state.get("runtime") or {})
    dataset = str(source_state.get("dataset") or "")
    if dataset not in DATASETS:
        raise ValueError(f"{run_dir}: unknown Integrator source dataset {dataset!r}")
    if expected_dataset and dataset != expected_dataset:
        raise ValueError(
            f"{run_dir}: source dataset is {dataset!r}, not {expected_dataset!r}"
        )
    if runtime.get("stage") != "integrator":
        raise ValueError(f"{run_dir}: source is not an offline Integrator run")
    spec = DATASETS[dataset]
    documents = int(source_state.get("documents", 0) or 0)
    if documents != spec.limit:
        raise ValueError(
            f"{run_dir}: expected a formal {spec.limit}-document run, found {documents}"
        )
    critic_runtime = source_state.get("critic1_runtime")
    if not isinstance(critic_runtime, Mapping):
        raise ValueError(f"{run_dir}: original Critic1 runtime is missing")
    _validate_complete_progress(run_dir, progress, spec.limit)

    document_ids = []
    source_hasher = hashlib.sha256()
    checkpoint_dir = run_dir / "checkpoints"
    for index in range(spec.limit):
        document_id, _, digest = _validate_source_checkpoint(
            checkpoint_dir / f"{index:06d}.json",
            index,
            require_integrator=True,
        )
        document_ids.append(document_id)
        source_hasher.update(f"{digest}\n".encode("ascii"))
    fingerprint = document_ids_sha256(document_ids)
    if fingerprint != str(source_state.get("document_ids_sha256") or ""):
        raise ValueError(f"{run_dir}: checkpoint document order does not match run_state")
    recorded_records = str(progress.get("output_records_sha256") or "")
    if not recorded_records:
        raise ValueError(f"{run_dir}: Integrator output fingerprint is missing")
    if source_hasher.hexdigest() != recorded_records:
        raise ValueError(f"{run_dir}: Integrator checkpoint content does not match run_state")
    return OfflineSourceRun(
        path=run_dir,
        dataset=dataset,
        timestamp=str(source_state.get("timestamp") or run_dir.name),
        document_count=spec.limit,
        document_ids_sha256=fingerprint,
        source_records_sha256=source_hasher.hexdigest(),
        critic_runtime=dict(critic_runtime),
        source_kind="integrator",
    )


def _candidate_runs(root: Path, *, nested_model_dirs: bool) -> List[Path]:
    if not root.is_dir():
        return []
    direct = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "run_state.json").is_file()
    ]
    if not nested_model_dirs:
        return sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
    nested = []
    for model_root in root.iterdir():
        if not model_root.is_dir():
            continue
        nested.extend(
            path
            for path in model_root.iterdir()
            if path.is_dir() and (path / "run_state.json").is_file()
        )
    return sorted(direct + nested, key=lambda path: path.name, reverse=True)


def discover_latest_runs(
    roots: Mapping[str, Path],
    dataset_keys: Sequence[str],
    inspector,
    *,
    nested_model_dirs: bool = False,
) -> Tuple[List[OfflineSourceRun], List[str]]:
    selected = []
    skipped = []
    for dataset in dataset_keys:
        root = Path(roots[dataset])
        candidates = _candidate_runs(root, nested_model_dirs=nested_model_dirs)
        errors = []
        for candidate in candidates:
            try:
                selected.append(inspector(candidate, dataset))
                break
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
        else:
            detail = errors[0] if errors else f"no timestamp runs under {root}"
            skipped.append(f"{dataset}: {detail}")
    return selected, skipped


def load_source_documents(source: OfflineSourceRun) -> List[SourceDocument]:
    require_integrator = source.source_kind == "integrator"
    rows = []
    document_ids = []
    source_hasher = hashlib.sha256()
    for index in range(source.document_count):
        path = source.path / "checkpoints" / f"{index:06d}.json"
        document_id, record, digest = _validate_source_checkpoint(
            path,
            index,
            require_integrator=require_integrator,
        )
        rows.append(SourceDocument(
            index=index,
            document_id=document_id,
            text=str(record["source_text"]),
            pipeline_record=record,
        ))
        document_ids.append(document_id)
        source_hasher.update(f"{digest}\n".encode("ascii"))
    if document_ids_sha256(document_ids) != source.document_ids_sha256:
        raise ValueError(f"{source.path}: source checkpoint order changed during loading")
    if source_hasher.hexdigest() != source.source_records_sha256:
        raise ValueError(f"{source.path}: source checkpoints changed during loading")
    return rows








def checkpoint_path(run_dir: Path, index: int) -> Path:
    return run_dir / "checkpoints" / f"{index:06d}.json"


def validate_output_checkpoint(
    value: Mapping,
    row: SourceDocument,
    path: Path,
) -> Dict:
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported output checkpoint version")
    if value.get("index") != row.index or value.get("document_id") != row.document_id:
        raise ValueError(f"{path}: output checkpoint does not match source order")
    if value.get("status") not in {"ok", "failed"}:
        raise ValueError(f"{path}: checkpoint status must be ok or failed")
    if not isinstance(value.get("attempts"), int) or value["attempts"] <= 0:
        raise ValueError(f"{path}: attempts must be positive")
    result = value.get("processing_result")
    if not isinstance(result, Mapping) or result.get("document_id") != row.document_id:
        raise ValueError(f"{path}: processing_result is missing or mismatched")
    if result.get("status") != value.get("status"):
        raise ValueError(f"{path}: processing_result status does not match")
    if value.get("status") == "ok":
        record = value.get("pipeline_record")
        triples = value.get("final_triples")
        if not isinstance(record, Mapping) or not isinstance(triples, list):
            raise ValueError(f"{path}: successful checkpoint requires record and triples")
        if record.get("document_id") != row.document_id:
            raise ValueError(f"{path}: pipeline record document_id does not match")
        if record.get("final_triples") != triples:
            raise ValueError(f"{path}: final triples do not match pipeline record")
    return dict(value)


def load_output_checkpoints(
    run_dir: Path,
    rows: Sequence[SourceDocument],
) -> Dict[int, Dict]:
    checkpoints = {}
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return checkpoints
    for row in rows:
        path = checkpoint_path(run_dir, row.index)
        if path.exists():
            checkpoints[row.index] = validate_output_checkpoint(
                read_json_object(path), row, path
            )
    unexpected = sorted(
        path.name
        for path in checkpoint_dir.glob("*.json")
        if path.stem.isdigit() and int(path.stem) >= len(rows)
    )
    if unexpected:
        raise ValueError(f"{checkpoint_dir}: out-of-range checkpoints: {unexpected[:5]}")
    return checkpoints


def progress_payload(
    rows: Sequence[SourceDocument],
    checkpoints: Mapping[int, Mapping],
    **extra,
) -> Dict:
    completed = sum(value.get("status") == "ok" for value in checkpoints.values())
    failed = sum(value.get("status") == "failed" for value in checkpoints.values())
    return {
        "documents_total": len(rows),
        "completed": completed,
        "failed": failed,
        "pending": len(rows) - completed - failed,
        "attempts_total": sum(
            int(value.get("attempts", 0)) for value in checkpoints.values()
        ),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }


def validate_resume_state(saved: Mapping, expected: Mapping) -> None:
    if saved.get("version") != RUN_STATE_VERSION:
        raise ValueError("run_state.json has an unsupported version")
    if dict(saved.get("source") or {}) != dict(expected.get("source") or {}):
        raise ValueError("resume source path, content, or document order has changed")
    if dict(saved.get("runtime") or {}) != dict(expected.get("runtime") or {}):
        raise ValueError("resume runtime or batching configuration has changed")


def initialize_run_state(run_dir: Path, expected_state: Mapping) -> None:
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        validate_resume_state(read_json_object(state_path), expected_state)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            f"{run_dir}: non-empty output directory has no compatible run_state.json"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_path, expected_state)


def source_state(source: OfflineSourceRun) -> Dict:
    return {
        "run_dir": str(source.path),
        "dataset": source.dataset,
        "timestamp": source.timestamp,
        "documents": source.document_count,
        "document_ids_sha256": source.document_ids_sha256,
        "source_records_sha256": source.source_records_sha256,
        "critic1_runtime": dict(source.critic_runtime),
    }


def materialize_outputs(
    source: OfflineSourceRun,
    rows: Sequence[SourceDocument],
    run_dir: Path,
    *,
    stage: str,
    runtime_summary: Mapping,
    active_wall_seconds: float,
) -> Dict:
    checkpoints = load_output_checkpoints(run_dir, rows)
    previous_progress = (
        read_json_object(run_dir / "progress.json")
        if (run_dir / "progress.json").is_file()
        else {}
    )
    progress = progress_payload(rows, checkpoints)
    complete = (
        len(checkpoints) == len(rows)
        and progress["completed"] == len(rows)
        and progress["failed"] == 0
    )
    records = []
    kg = []
    processing = []
    output_hasher = hashlib.sha256()
    for row in rows:
        checkpoint = checkpoints.get(row.index)
        if checkpoint is None:
            processing.append({
                "document_id": row.document_id,
                "status": "pending",
                "error": "",
            })
            continue
        processing.append(dict(checkpoint["processing_result"]))
        if checkpoint["status"] != "ok":
            continue
        record = dict(checkpoint["pipeline_record"])
        records.append(record)
        kg.extend(dict(item) for item in checkpoint["final_triples"])
        output_hasher.update(
            f"{canonical_sha256({'index': row.index, 'document_id': row.document_id, 'pipeline_record': record})}\n".encode(
                "ascii"
            )
        )

    atomic_write_jsonl(
        run_dir / "documents.jsonl",
        ({"document_id": row.document_id, "text": row.text} for row in rows),
    )
    atomic_write_jsonl(run_dir / "processing_results.jsonl", processing)
    atomic_write_jsonl(run_dir / "pipeline_records.jsonl", records)
    atomic_write_jsonl(run_dir / "kg.jsonl", kg)
    atomic_write_json(run_dir / "pipeline_records.json", records)
    atomic_write_json(run_dir / "kg.json", kg)

    summary = {
        "stage": stage,
        "output_dir": str(run_dir),
        "dataset": source.dataset,
        "source_run_dir": str(source.path),
        "source_timestamp": source.timestamp,
        "documents_total": len(rows),
        "documents_completed": progress["completed"],
        "documents_failed": progress["failed"],
        "input_triples": sum(
            len(row.pipeline_record.get("refined_triples") or []) for row in rows
        ),
        "output_triples": len(kg),
        "active_wall_seconds": round(float(active_wall_seconds), 3),
        "runtime": dict(runtime_summary),
        "finalized": complete,
    }
    if stage == "integrator":
        summary["integrator_additions_accepted"] = sum(
            len(record.get("added_triples") or []) for record in records
        )
    atomic_write_json(run_dir / "summary.json", summary)
    progress = {**previous_progress, **progress, "finalized": complete}
    if complete:
        progress["output_records_sha256"] = output_hasher.hexdigest()
    atomic_write_json(run_dir / "progress.json", progress)
    return summary


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
