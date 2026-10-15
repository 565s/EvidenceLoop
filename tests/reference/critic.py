"""Apply the local dataset-specific Critic to completed direct baseline runs.

The script never invokes an online proposer and never computes benchmark metrics.
Each source run is processed in a child process so only one dataset adapter is
resident on the GPU at a time.
"""

from __future__ import annotations

from evidenceloop.io import atomic_write_jsonl
from evidenceloop.io import atomic_write_json
from evidenceloop.io import _atomic_write_text

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path.cwd()


CHECKPOINT_VERSION = 1
RUN_STATE_VERSION = 1
DEFAULT_MODELS = ("gpt-4o", "gpt-4o-mini")
DEFAULT_BASELINE_ROOT = ROOT / "data" / "experiments" / "gpt_baselines"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "gpt_proposer_critic_once"
SUPPORTED_BASELINES = {"gpt_direct_extraction", "deepseek_direct_extraction"}
DEFAULT_CRITIC_DOCUMENT_BATCH_SIZE = 8
DEFAULT_CRITIC_PROMPT_BATCH_SIZE = 4
DEFAULT_MAX_RECOVERY_ROUNDS = 3
DATASET_CHOICES = ("rebel_sub", "redocred", "biored", "scierc", "all")
CRITIC_PATH_ENV_KEYS = ("CRITIC_BASE_MODEL", "CRITIC_ADAPTER_PATH")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    limit: int
    runner_module: str


@dataclass(frozen=True)
class BaselineRun:
    path: Path
    dataset: str
    model: str
    timestamp: str
    document_count: int
    document_ids_sha256: str
    source_payloads_sha256: str


@dataclass(frozen=True)
class SourceDocument:
    index: int
    document_id: str
    text: str
    proposer_payload: Dict
    proposer_elapsed_seconds: float


DATASETS: Dict[str, DatasetSpec] = {
    "rebel_sub": DatasetSpec("rebel_sub", 1000, "tests.reference.checkpoints"),
    "redocred": DatasetSpec("redocred", 500, "evidenceloop.data.registry"),
    "biored": DatasetSpec("biored", 100, "evidenceloop.data.registry"),
    "scierc": DatasetSpec("scierc", 100, "evidenceloop.data.registry"),
}

from evidenceloop.data.registry import AliasDict
DATASETS = AliasDict(DATASETS)


def _resolve_path(value) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json_object(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return dict(value)








def normalize_model_name(value: str) -> str:
    model = str(value or "").strip().casefold()
    aliases = {
        "gpt-4omini": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "openai/gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "openai/gpt-4o": "gpt-4o",
        "deepseek-v4-flash": "deepseek-v4-flash",
        "deepseek-ai/deepseek-v4-flash": "deepseek-v4-flash",
    }
    if model not in aliases:
        raise ValueError("model must be gpt-4o, gpt-4o-mini, or deepseek-v4-flash")
    return aliases[model]


def selected_dataset_keys(value: str) -> List[str]:
    if value == "all":
        return list(DATASETS)
    if value not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}")
    return [value]


def normalize_models(values: Sequence[str]) -> List[str]:
    output = []
    for value in values:
        model = normalize_model_name(value)
        if model not in output:
            output.append(model)
    if not output:
        raise ValueError("at least one baseline model is required")
    return output


def document_ids_sha256(document_ids: Sequence[str]) -> str:
    payload = "".join(f"{document_id}\n" for document_id in document_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _baseline_identity(run_dir: Path) -> Tuple[DatasetSpec, str, Dict, Dict]:
    state = _read_json_object(run_dir / "run_state.json")
    progress = _read_json_object(run_dir / "progress.json")
    runtime = dict(state.get("runtime") or {})
    dataset_key = str(runtime.get("dataset") or runtime.get("dataset_key") or "").strip()
    if dataset_key not in DATASETS:
        raise ValueError(f"{run_dir}: unknown baseline dataset {dataset_key!r}")
    baseline = str(runtime.get("baseline") or "").strip()
    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(f"{run_dir}: source is not a supported direct-extraction baseline")
    model = normalize_model_name(runtime.get("model"))
    return DATASETS[dataset_key], model, state, progress


def _validate_source_checkpoint(path: Path, index: int) -> Tuple[str, str, str]:
    value = _read_json_object(path)
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version")
    if value.get("index") != index:
        raise ValueError(f"{path}: checkpoint index does not match its filename")
    document_id = str(value.get("document_id") or "").strip()
    if not document_id:
        raise ValueError(f"{path}: document_id is required")
    if value.get("status") != "ok":
        raise ValueError(f"{path}: baseline checkpoint is not successful")
    payload = value.get("payload")
    record = value.get("pipeline_record")
    if not isinstance(payload, Mapping) or not isinstance(record, Mapping):
        raise ValueError(f"{path}: successful baseline requires payload and pipeline_record")
    if str(record.get("document_id") or "") != document_id:
        raise ValueError(f"{path}: pipeline record document_id does not match")
    source_text = str(record.get("source_text") or "")
    if not source_text.strip():
        raise ValueError(f"{path}: source_text is required")
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise ValueError(f"{path}: payload.triples must be a list")
    if str(payload.get("parse_error") or ""):
        raise ValueError(f"{path}: successful baseline payload contains a parse error")
    if value.get("final_triples") != triples:
        raise ValueError(f"{path}: payload triples do not match final_triples")
    if record.get("extracted_triples") != triples or record.get("final_triples") != triples:
        raise ValueError(f"{path}: pipeline record does not preserve baseline triples")
    for triple_index, triple in enumerate(triples):
        if not isinstance(triple, Mapping):
            raise ValueError(f"{path}: triple {triple_index} must be an object")
        if str(triple.get("document_id") or document_id) != document_id:
            raise ValueError(f"{path}: triple {triple_index} has the wrong document_id")
        triple_source = str(triple.get("source_text") or source_text)
        if triple_source != source_text:
            raise ValueError(f"{path}: triple {triple_index} has the wrong source_text")
        if any(not str(triple.get(field) or "").strip() for field in ("subject", "predicate", "object")):
            raise ValueError(f"{path}: triple {triple_index} has empty fields")
    source_payload = {
        "index": index,
        "document_id": document_id,
        "source_text": source_text,
        "payload": payload,
    }
    payload_digest = hashlib.sha256(
        json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return document_id, source_text, payload_digest


def inspect_baseline_run(run_dir: Path) -> BaselineRun:
    run_dir = run_dir.resolve()
    spec, model, state, progress = _baseline_identity(run_dir)
    data = dict(state.get("data") or {})
    selected = int(data.get("selected_documents", 0) or 0)
    requested = int(data.get("requested_limit", 0) or 0)
    if selected != spec.limit or requested != spec.limit:
        raise ValueError(
            f"{run_dir}: expected a formal {spec.limit}-document run, "
            f"found requested={requested}, selected={selected}"
        )
    if progress.get("finalized") is not True:
        raise ValueError(f"{run_dir}: baseline run is not finalized")
    completed = int(progress.get("completed", 0) or 0)
    failed = int(progress.get("failed", 0) or 0)
    pending = int(progress.get("pending", 0) or 0)
    if completed != spec.limit or failed or pending:
        raise ValueError(
            f"{run_dir}: baseline is incomplete "
            f"(completed={completed}, failed={failed}, pending={pending})"
        )

    checkpoint_dir = run_dir / "checkpoints"
    document_ids = []
    source_hasher = hashlib.sha256()
    for index in range(spec.limit):
        document_id, _, payload_digest = _validate_source_checkpoint(
            checkpoint_dir / f"{index:06d}.json", index
        )
        document_ids.append(document_id)
        source_hasher.update(f"{payload_digest}\n".encode("ascii"))
    unexpected = sorted(
        path.name
        for path in checkpoint_dir.glob("*.json")
        if path.stem.isdigit() and int(path.stem) >= spec.limit
    )
    if unexpected:
        raise ValueError(f"{run_dir}: out-of-range checkpoints: {unexpected[:5]}")

    fingerprint = document_ids_sha256(document_ids)
    recorded_fingerprint = str(data.get("document_ids_sha256") or "")
    if fingerprint != recorded_fingerprint:
        raise ValueError(f"{run_dir}: checkpoint document order does not match run_state")
    return BaselineRun(
        path=run_dir,
        dataset=spec.key,
        model=model,
        timestamp=run_dir.name,
        document_count=spec.limit,
        document_ids_sha256=fingerprint,
        source_payloads_sha256=source_hasher.hexdigest(),
    )


def discover_baseline_runs(
    baseline_root: Path,
    dataset_keys: Sequence[str],
    models: Sequence[str],
) -> Tuple[List[BaselineRun], List[str]]:
    selected = []
    skipped = []
    for dataset_key in dataset_keys:
        for model in models:
            model_root = baseline_root / dataset_key / model
            candidates = sorted(
                (path for path in model_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            ) if model_root.is_dir() else []
            errors = []
            for candidate in candidates:
                try:
                    selected.append(inspect_baseline_run(candidate))
                    break
                except (FileNotFoundError, ValueError) as exc:
                    errors.append(str(exc))
            else:
                detail = errors[0] if errors else f"no runs under {model_root}"
                skipped.append(f"{dataset_key}/{model}: {detail}")
    return selected, skipped


def load_source_documents(source: BaselineRun) -> List[SourceDocument]:
    rows = []
    document_ids = []
    source_hasher = hashlib.sha256()
    for index in range(source.document_count):
        path = source.path / "checkpoints" / f"{index:06d}.json"
        value = _read_json_object(path)
        document_id, source_text, payload_digest = _validate_source_checkpoint(path, index)
        payload = dict(value["payload"])
        sample_result = dict(value.get("sample_result") or {})
        rows.append(SourceDocument(
            index=index,
            document_id=document_id,
            text=source_text,
            proposer_payload={
                "triples": [dict(triple) for triple in payload.get("triples", [])],
                "raw_response": str(payload.get("raw_response") or ""),
                "parse_error": "",
                "rejected_predicates": list(payload.get("rejected_predicates") or []),
            },
            proposer_elapsed_seconds=float(
                sample_result.get("elapsed_seconds", 0.0) or 0.0
            ),
        ))
        document_ids.append(document_id)
        source_hasher.update(f"{payload_digest}\n".encode("ascii"))
    if document_ids_sha256(document_ids) != source.document_ids_sha256:
        raise ValueError(f"{source.path}: source checkpoints changed during loading")
    if source_hasher.hexdigest() != source.source_payloads_sha256:
        raise ValueError(f"{source.path}: source payloads changed during loading")
    return rows


def _critic_defaults(spec: DatasetSpec) -> Dict[str, str]:
    from evidenceloop.data.registry import dataset_config
    config = dataset_config(spec.key)
    return {
        "CRITIC_MODEL_NAME": config.critic_model_name,
        "CRITIC_BASE_MODEL": config.critic_base_model_path,
        "CRITIC_ADAPTER_PATH": config.critic_adapter_path,
        "CRITIC_DEVICE": config.critic_device,
    }


def critic_child_env(
    spec: DatasetSpec,
    prompt_batch_size: int,
    runtime_overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    for key, default in _critic_defaults(spec).items():
        env[key] = str(os.environ.get(key) or default)
    runtime_overrides = dict(runtime_overrides or {})
    unexpected = sorted(set(runtime_overrides) - set(CRITIC_PATH_ENV_KEYS))
    if unexpected:
        raise ValueError(f"unsupported Critic runtime overrides: {unexpected}")
    for key, value in runtime_overrides.items():
        env[key] = str(value)
    env["CRITIC_PROMPT_BATCH_SIZE"] = str(prompt_batch_size)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _output_checkpoint_path(run_dir: Path, index: int) -> Path:
    return run_dir / "checkpoints" / f"{index:06d}.json"


def _validate_output_checkpoint(value: Mapping, row: SourceDocument, path: Path) -> Dict:
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported output checkpoint version")
    if value.get("index") != row.index or value.get("document_id") != row.document_id:
        raise ValueError(f"{path}: output checkpoint does not match source order")
    if value.get("status") not in {"ok", "failed"}:
        raise ValueError(f"{path}: output checkpoint status must be ok or failed")
    if not isinstance(value.get("attempts"), int) or value["attempts"] <= 0:
        raise ValueError(f"{path}: attempts must be positive")
    result = value.get("processing_result")
    if not isinstance(result, Mapping) or result.get("document_id") != row.document_id:
        raise ValueError(f"{path}: processing_result is missing or mismatched")
    if result.get("status") != value.get("status"):
        raise ValueError(f"{path}: processing_result status does not match")
    if value.get("status") == "ok":
        record = value.get("pipeline_record")
        final_triples = value.get("final_triples")
        if not isinstance(record, Mapping) or not isinstance(final_triples, list):
            raise ValueError(f"{path}: successful output requires record and triples")
        if record.get("document_id") != row.document_id:
            raise ValueError(f"{path}: pipeline record document_id does not match")
        if record.get("final_triples") != final_triples:
            raise ValueError(f"{path}: final triples do not match pipeline record")
    return dict(value)


def load_output_checkpoints(run_dir: Path, rows: Sequence[SourceDocument]) -> Dict[int, Dict]:
    checkpoints = {}
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return checkpoints
    for row in rows:
        path = _output_checkpoint_path(run_dir, row.index)
        if path.exists():
            checkpoints[row.index] = _validate_output_checkpoint(
                _read_json_object(path), row, path
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
) -> Dict:
    completed = sum(value.get("status") == "ok" for value in checkpoints.values())
    failed = sum(value.get("status") == "failed" for value in checkpoints.values())
    return {
        "documents_total": len(rows),
        "completed": completed,
        "failed": failed,
        "pending": len(rows) - completed - failed,
        "attempts_total": sum(int(value.get("attempts", 0)) for value in checkpoints.values()),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _build_run_state(
    source: BaselineRun,
    critic_runtime: Mapping,
    document_batch_size: int,
    prompt_batch_size: int,
) -> Dict:
    return {
        "version": RUN_STATE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "run_dir": str(source.path),
            "dataset": source.dataset,
            "model": source.model,
            "timestamp": source.timestamp,
            "documents": source.document_count,
            "document_ids_sha256": source.document_ids_sha256,
            "source_payloads_sha256": source.source_payloads_sha256,
        },
        "runtime": {
            "pipeline_variant": "proposer_critic_once",
            "offline_proposals": True,
            "critic_document_batch_size": document_batch_size,
            "critic_prompt_batch_size": prompt_batch_size,
            "critic_runtime": dict(critic_runtime),
        },
    }


def _validate_resume_state(saved: Mapping, expected: Mapping) -> None:
    if saved.get("version") != RUN_STATE_VERSION:
        raise ValueError("run_state.json has an unsupported version")
    if dict(saved.get("source") or {}) != dict(expected.get("source") or {}):
        raise ValueError("resume source run, model, document count, or ID order has changed")
    if dict(saved.get("runtime") or {}) != dict(expected.get("runtime") or {}):
        raise ValueError("resume Critic runtime or batching configuration has changed")


def _triple_objects(row: SourceDocument):
    from evidenceloop.types import Triple

    return [
        Triple.from_dict(
            dict(value),
            source_text=row.text,
            document_id=row.document_id,
        )
        for value in row.proposer_payload.get("triples", [])
    ]


class _UnavailableProposer:
    def propose_triples(self, document_id, text):
        raise AssertionError("offline Critic runner must never invoke a proposer")


def _processing_result(row, record, *, status: str, error: str, elapsed: float) -> Dict:
    action_counts = Counter(
        review.action for review in (record.initial_reviews if record else [])
    )
    return {
        "document_id": row.document_id,
        "status": status,
        "error": error,
        "input_triples": len(row.proposer_payload.get("triples", [])),
        "output_triples": len(record.final_triples) if record else 0,
        "critic_actions": dict(sorted(action_counts.items())),
        "invalid_reviews": sum(
            not review.valid for review in (record.initial_reviews if record else [])
        ),
        "stage_errors": len(record.stage_errors) if record else 0,
        "elapsed_seconds": round(elapsed, 3),
    }


def process_documents(
    controller,
    rows: Sequence[SourceDocument],
    run_dir: Path,
    document_batch_size: int,
    console_print=print,
) -> Tuple[Dict[int, Dict], Dict]:
    from evidenceloop.types import Triple
    from tests.reference.checkpoints import pipeline_record_from_dict

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoints = load_output_checkpoints(run_dir, rows)
    controller.kg = []
    controller.pipeline_records = []
    for index in sorted(checkpoints):
        checkpoint = checkpoints[index]
        if checkpoint["status"] != "ok":
            continue
        record = pipeline_record_from_dict(checkpoint["pipeline_record"])
        controller.pipeline_records.append(record)
        controller.kg.extend(
            Triple.from_dict(dict(value)) for value in checkpoint["final_triples"]
        )

    progress_path = run_dir / "progress.json"
    previous_progress = _read_json_object(progress_path) if progress_path.exists() else {}
    value = progress_payload(rows, checkpoints)
    value["active_wall_seconds"] = float(previous_progress.get("active_wall_seconds", 0.0) or 0.0)
    value["critic_metrics"] = dict(previous_progress.get("critic_metrics") or {})
    atomic_write_json(progress_path, value)

    pending_rows = [
        row for row in rows
        if not (checkpoints.get(row.index) and checkpoints[row.index]["status"] == "ok")
    ]
    for batch_start in range(0, len(pending_rows), document_batch_size):
        batch = pending_rows[batch_start:batch_start + document_batch_size]
        prepared = []
        for row in batch:
            triples = _triple_objects(row)
            prepared.append({
                "row": row,
                "triples": triples,
                "payload": {
                    **row.proposer_payload,
                    "triples": triples,
                },
            })

        try:
            review_outputs = controller.critic.review_documents_batch([{
                "triples": item["triples"],
                "stage": "initial_review",
                "origins": ["proposer"] * len(item["triples"]),
            } for item in prepared])
            if len(review_outputs) != len(prepared):
                raise RuntimeError("Critic document batch output count does not match input count")
            batch_error = ""
        except Exception as exc:
            traceback.print_exc()
            review_outputs = [None] * len(prepared)
            batch_error = str(exc)

        for item, review_output in zip(prepared, review_outputs):
            row = item["row"]
            previous = checkpoints.get(row.index)
            attempts = int(previous.get("attempts", 0)) + 1 if previous else 1
            started = time.perf_counter()
            kg_start = len(controller.kg)
            record_start = len(controller.pipeline_records)
            record = None
            error = batch_error
            status = "failed" if batch_error else "ok"
            critic_elapsed = 0.0
            if not batch_error:
                try:
                    critic_elapsed = float(review_output.get("elapsed_seconds", 0.0) or 0.0)
                    record = controller.process_document(
                        row.document_id,
                        row.text,
                        proposer_payload=item["payload"],
                        proposer_elapsed_seconds=row.proposer_elapsed_seconds,
                        initial_critic_payload=review_output["payload"],
                        initial_critic_elapsed_seconds=critic_elapsed,
                    )
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    del controller.kg[kg_start:]
                    del controller.pipeline_records[record_start:]
                    traceback.print_exc()

            elapsed = critic_elapsed + time.perf_counter() - started
            result = _processing_result(
                row,
                record,
                status=status,
                error=error,
                elapsed=elapsed,
            )
            checkpoint = {
                "version": CHECKPOINT_VERSION,
                "index": row.index,
                "document_id": row.document_id,
                "attempts": attempts,
                "status": status,
                "error": error,
                "processing_result": result,
                "pipeline_record": record.to_dict() if record else None,
                "final_triples": [
                    triple.to_dict() for triple in (record.final_triples if record else [])
                ],
            }
            atomic_write_json(_output_checkpoint_path(run_dir, row.index), checkpoint)
            checkpoints[row.index] = checkpoint
            current = progress_payload(rows, checkpoints)
            current["active_wall_seconds"] = value["active_wall_seconds"]
            current["critic_metrics"] = value["critic_metrics"]
            atomic_write_json(progress_path, current)
            console_print(
                f"[{row.index + 1}/{len(rows)}] {row.document_id} {status} | "
                f"input={result['input_triples']} output={result['output_triples']}"
            )
    return checkpoints, previous_progress


def _combine_critic_metrics(previous: Mapping, current: Mapping) -> Dict:
    additive = (
        "model_load_seconds",
        "critic_prompt_count",
        "critic_generation_batches",
        "critic_input_tokens",
        "critic_output_tokens",
        "critic_generation_seconds",
    )
    output = {
        "backend": "direct",
        "adapter_merged": bool(current.get("adapter_merged", False)),
    }
    for key in additive:
        total = float(previous.get(key, 0) or 0) + float(current.get(key, 0) or 0)
        output[key] = round(total, 3) if key.endswith("seconds") else int(total)
    output["critic_peak_gpu_memory_bytes"] = max(
        int(previous.get("critic_peak_gpu_memory_bytes", 0) or 0),
        int(current.get("critic_peak_gpu_memory_bytes", 0) or 0),
    )
    seconds = float(output["critic_generation_seconds"])
    output["critic_tokens_per_second"] = round(
        output["critic_output_tokens"] / seconds, 3
    ) if seconds else 0.0
    return output


def materialize_outputs(
    source: BaselineRun,
    rows: Sequence[SourceDocument],
    run_dir: Path,
    critic_runtime: Mapping,
    critic_metrics: Mapping,
    active_wall_seconds: float,
) -> Dict:
    checkpoints = load_output_checkpoints(run_dir, rows)
    if len(checkpoints) != len(rows):
        missing = [row.index for row in rows if row.index not in checkpoints]
        raise ValueError(f"cannot materialize run with missing checkpoints: {missing[:5]}")

    processing_results = []
    record_dicts = []
    kg_dicts = []
    action_counts = Counter()
    for row in rows:
        checkpoint = checkpoints[row.index]
        result = dict(checkpoint["processing_result"])
        processing_results.append(result)
        action_counts.update(result.get("critic_actions") or {})
        if checkpoint["status"] != "ok":
            continue
        record_dicts.append(dict(checkpoint["pipeline_record"]))
        kg_dicts.extend(dict(value) for value in checkpoint["final_triples"])

    atomic_write_jsonl(run_dir / "documents.jsonl", ({
        "document_id": row.document_id,
        "text": row.text,
    } for row in rows))
    atomic_write_jsonl(run_dir / "processing_results.jsonl", processing_results)
    atomic_write_jsonl(run_dir / "pipeline_records.jsonl", record_dicts)
    atomic_write_jsonl(run_dir / "kg.jsonl", kg_dicts)
    atomic_write_json(run_dir / "pipeline_records.json", record_dicts)
    atomic_write_json(run_dir / "kg.json", kg_dicts)

    completed = sum(item["status"] == "ok" for item in processing_results)
    failed = len(processing_results) - completed
    summary = {
        "output_dir": str(run_dir),
        "source_run_dir": str(source.path),
        "dataset": source.dataset,
        "source_model": source.model,
        "gpt_model": source.model,
        "documents_total": len(rows),
        "documents_completed": completed,
        "documents_failed": failed,
        "input_triples_total": sum(item["input_triples"] for item in processing_results),
        "output_triples_total": len(kg_dicts),
        "critic_actions": dict(sorted(action_counts.items())),
        "invalid_reviews": sum(item["invalid_reviews"] for item in processing_results),
        "stage_errors": sum(item["stage_errors"] for item in processing_results),
        "active_wall_seconds": round(active_wall_seconds, 3),
        "critic_runtime": dict(critic_runtime),
        "critic_metrics": dict(critic_metrics),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    progress = progress_payload(rows, checkpoints)
    progress["finalized"] = failed == 0 and progress["pending"] == 0
    progress["active_wall_seconds"] = round(active_wall_seconds, 3)
    progress["critic_metrics"] = dict(critic_metrics)
    atomic_write_json(run_dir / "progress.json", progress)
    return summary


def run_offline_source(
    source_run_dir: Path,
    output_run_dir: Path,
    *,
    document_batch_size: int,
    prompt_batch_size: int,
) -> Path:
    source = inspect_baseline_run(source_run_dir)
    rows = load_source_documents(source)
    output_run_dir = output_run_dir.resolve()

    from evidenceloop.agents.critic import CriticAgent
    from evidenceloop.pipeline.core import Pipeline
    from evidenceloop.runtime.direct_critic import DirectCriticEngine
    from evidenceloop.settings import agent_config
    from evidenceloop.runtime.settings import (
        public_critic_runtime,
        resolve_critic_runtime,
        validate_direct_runtime_paths,
    )

    runtime_settings = resolve_critic_runtime(agent_config)
    public_runtime = public_critic_runtime(runtime_settings)
    expected_state = _build_run_state(
        source,
        public_runtime,
        document_batch_size,
        prompt_batch_size,
    )
    state_path = output_run_dir / "run_state.json"
    if state_path.exists():
        _validate_resume_state(_read_json_object(state_path), expected_state)
    else:
        existing = [path for path in output_run_dir.iterdir()] if output_run_dir.exists() else []
        if existing:
            raise ValueError(f"{output_run_dir}: non-empty output directory has no run_state.json")
        validate_direct_runtime_paths(runtime_settings)
        output_run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_path, expected_state)

    console = sys.__stdout__

    def console_print(message: str) -> None:
        print(message, file=console, flush=True)

    previous_progress = (
        _read_json_object(output_run_dir / "progress.json")
        if (output_run_dir / "progress.json").exists()
        else {}
    )
    previous_active_seconds = float(previous_progress.get("active_wall_seconds", 0.0) or 0.0)
    existing_checkpoints = load_output_checkpoints(output_run_dir, rows)
    if len(existing_checkpoints) == len(rows) and all(
        checkpoint["status"] == "ok" for checkpoint in existing_checkpoints.values()
    ):
        materialize_outputs(
            source,
            rows,
            output_run_dir,
            public_runtime,
            previous_progress.get("critic_metrics") or {},
            previous_active_seconds,
        )
        console_print(
            f"[done] {source.dataset}/{source.model} already complete; "
            f"output={output_run_dir}"
        )
        return output_run_dir

    log_mode = "a" if previous_progress else "w"
    started = time.perf_counter()
    engine = DirectCriticEngine(runtime_settings)
    critic = CriticAgent(
        name="Offline-Critic",
        role="Critic",
        llm=engine,
        runtime_settings=runtime_settings,
    )
    critic.validate_runtime()
    controller = Pipeline(
        proposer_agent=_UnavailableProposer(),
        critic_agent=critic,
        integrator_agent=None,
        pipeline_variant="proposer_critic_once",
    )

    with (output_run_dir / "run.log").open(
        log_mode, encoding="utf-8", buffering=1
    ) as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        print(
            f"Offline Critic run started at {datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        print(json.dumps(expected_state["runtime"], ensure_ascii=False, indent=2), flush=True)
        checkpoints, prior_progress = process_documents(
            controller,
            rows,
            output_run_dir,
            document_batch_size,
            console_print=console_print,
        )
        active_wall_seconds = previous_active_seconds + time.perf_counter() - started
        cumulative_metrics = _combine_critic_metrics(
            prior_progress.get("critic_metrics") or {},
            engine.metrics(),
        )
        summary = materialize_outputs(
            source,
            rows,
            output_run_dir,
            public_runtime,
            cumulative_metrics,
            active_wall_seconds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    progress = progress_payload(rows, checkpoints)
    console_print(
        f"[done] {source.dataset}/{source.model} "
        f"completed={progress['completed']}/{progress['documents_total']} "
        f"failed={progress['failed']} output={output_run_dir}"
    )
    return output_run_dir


def _source_from_output_run(run_dir: Path) -> Path:
    state = _read_json_object(run_dir / "run_state.json")
    source = dict(state.get("source") or {})
    source_path = str(source.get("run_dir") or "").strip()
    if not source_path:
        raise ValueError(f"{run_dir}: run_state does not contain source.run_dir")
    return Path(source_path).resolve()


def _saved_critic_path_overrides(run_dir: Path) -> Dict[str, str]:
    state = _read_json_object(run_dir / "run_state.json")
    runtime = dict(state.get("runtime") or {})
    critic_runtime = dict(runtime.get("critic_runtime") or {})
    field_map = {
        "CRITIC_BASE_MODEL": "base_model_path",
        "CRITIC_ADAPTER_PATH": "adapter_path",
    }
    overrides = {}
    for env_key, state_key in field_map.items():
        value = str(critic_runtime.get(state_key) or "").strip()
        if not value:
            raise ValueError(
                f"{run_dir}: run_state does not contain runtime.critic_runtime.{state_key}"
            )
        overrides[env_key] = value
    return overrides


def _cli_critic_path_overrides(args: argparse.Namespace) -> Dict[str, str]:
    field_map = {
        "CRITIC_BASE_MODEL": args.critic_base_model,
        "CRITIC_ADAPTER_PATH": args.critic_adapter_path,
    }
    return {
        env_key: str(_resolve_path(value))
        for env_key, value in field_map.items()
        if value is not None
    }


def _run_worker(
    source: BaselineRun,
    output_run_dir: Path,
    args: argparse.Namespace,
    runtime_overrides: Optional[Mapping[str, str]] = None,
) -> None:
    env = critic_child_env(
        DATASETS[source.dataset],
        args.critic_prompt_batch_size,
        runtime_overrides,
    )
    command = [
        sys.executable,
        "-m", "tests.reference.critic",
        "--worker-source-run-dir", str(source.path),
        "--worker-output-run-dir", str(output_run_dir),
        "--critic-document-batch-size", str(args.critic_document_batch_size),
        "--critic-prompt-batch-size", str(args.critic_prompt_batch_size),
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"offline Critic worker failed for {source.dataset}/{source.model} "
            f"with exit code {result.returncode}"
        )


def complete_source_run(
    source: BaselineRun,
    output_run_dir: Path,
    args: argparse.Namespace,
    runtime_overrides: Optional[Mapping[str, str]] = None,
) -> Path:
    for recovery_round in range(args.max_recovery_rounds + 1):
        _run_worker(source, output_run_dir, args, runtime_overrides)
        progress = _read_json_object(output_run_dir / "progress.json")
        if int(progress.get("failed", 0) or 0) == 0 and int(progress.get("pending", 0) or 0) == 0:
            return output_run_dir
        if recovery_round >= args.max_recovery_rounds:
            raise RuntimeError(
                f"{source.dataset}/{source.model} remains incomplete: "
                f"completed={progress.get('completed')} failed={progress.get('failed')} "
                f"pending={progress.get('pending')}"
            )
    raise AssertionError("unreachable")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the local Qwen LoRA Critic to completed direct baseline outputs."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--source-run-dir")
    parser.add_argument(
        "--output-run-dir",
        help="Exact output directory for one explicit --source-run-dir.",
    )
    parser.add_argument("--resume-run-dir")
    parser.add_argument(
        "--critic-base-model",
        help="Local base-model directory for one explicit source or resumed run.",
    )
    parser.add_argument(
        "--critic-adapter-path",
        help="Local PEFT adapter directory for one explicit source or resumed run.",
    )
    parser.add_argument(
        "--critic-document-batch-size",
        type=int,
        default=DEFAULT_CRITIC_DOCUMENT_BATCH_SIZE,
    )
    parser.add_argument(
        "--critic-prompt-batch-size",
        type=int,
        default=DEFAULT_CRITIC_PROMPT_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-recovery-rounds",
        type=int,
        default=DEFAULT_MAX_RECOVERY_ROUNDS,
    )
    parser.add_argument("--worker-source-run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-run-dir", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.critic_document_batch_size <= 64:
        raise ValueError("critic-document-batch-size must be between 1 and 64")
    if not 1 <= args.critic_prompt_batch_size <= 4:
        raise ValueError("critic-prompt-batch-size must be between 1 and 4")
    if args.max_recovery_rounds < 0:
        raise ValueError("max-recovery-rounds must not be negative")
    if args.source_run_dir and args.resume_run_dir:
        raise ValueError("--source-run-dir and --resume-run-dir are mutually exclusive")
    custom_runtime = any(
        value is not None
        for value in (args.critic_base_model, args.critic_adapter_path)
    )
    for option, value in (
        ("--critic-base-model", args.critic_base_model),
        ("--critic-adapter-path", args.critic_adapter_path),
        ("--output-run-dir", args.output_run_dir),
    ):
        if value is not None and not str(value).strip():
            raise ValueError(f"{option} must not be empty")
    if custom_runtime and not (args.source_run_dir or args.resume_run_dir):
        raise ValueError(
            "custom Critic model paths require --source-run-dir or --resume-run-dir"
        )
    if custom_runtime and args.source_run_dir and not args.output_run_dir:
        raise ValueError(
            "custom Critic model paths require --output-run-dir for a new source run"
        )
    if args.output_run_dir and not args.source_run_dir:
        raise ValueError("--output-run-dir is only valid with --source-run-dir")
    if bool(args.worker_source_run_dir) != bool(args.worker_output_run_dir):
        raise ValueError("worker source and output directories must be provided together")


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_dotenv_file(ROOT / ".env")
    args = parse_args(argv)
    _validate_args(args)
    cli_runtime_overrides = _cli_critic_path_overrides(args)
    if args.worker_source_run_dir:
        run_offline_source(
            _resolve_path(args.worker_source_run_dir),
            _resolve_path(args.worker_output_run_dir),
            document_batch_size=args.critic_document_batch_size,
            prompt_batch_size=args.critic_prompt_batch_size,
        )
        return 0

    output_root = _resolve_path(args.output_root)
    if args.resume_run_dir:
        output_run_dir = _resolve_path(args.resume_run_dir)
        source = inspect_baseline_run(_source_from_output_run(output_run_dir))
        runtime_overrides = _saved_critic_path_overrides(output_run_dir)
        runtime_overrides.update(cli_runtime_overrides)
        complete_source_run(source, output_run_dir, args, runtime_overrides)
        print(output_run_dir)
        return 0

    if args.source_run_dir:
        sources = [inspect_baseline_run(_resolve_path(args.source_run_dir))]
        skipped = []
    else:
        sources, skipped = discover_baseline_runs(
            _resolve_path(args.baseline_root),
            selected_dataset_keys(args.dataset),
            normalize_models(args.models),
        )

    for message in skipped:
        print(f"[skip] {message}")
    if not sources:
        print("No complete direct baseline runs were found.")
        return 0

    results = []
    for source in sources:
        output_run_dir = (
            _resolve_path(args.output_run_dir)
            if args.output_run_dir
            else output_root / source.dataset / source.model / source.timestamp
        )
        results.append(
            complete_source_run(source, output_run_dir, args, cli_runtime_overrides)
        )
    print("Completed offline baseline + Critic-Once runs:")
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
