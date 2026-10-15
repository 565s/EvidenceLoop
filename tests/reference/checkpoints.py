from __future__ import annotations

from evidenceloop.evaluation.direction import normalize_symmetric_predicates, validate_entity_order_config

from evidenceloop.serialization import pipeline_record_from_dict
from evidenceloop.serialization import _review_from_dict
from evidenceloop.serialization import _triple_from_dict
from evidenceloop.io import atomic_write_jsonl
from evidenceloop.io import atomic_write_json
from evidenceloop.io import _atomic_write_text
from evidenceloop.io import load_dotenv_file

import argparse
import ast
import contextlib
import hashlib
import json
import os
import random
import sys
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path.cwd()

from evidenceloop.data.documents import DocumentInputError, normalize_document  # noqa: E402
from evidenceloop.types import DocumentPipelineRecord, Triple, TripleReviewRecord  # noqa: E402


CHECKPOINT_VERSION = 1
DEFAULT_DATA_DIR = "data/graphjudge/rebel_sub"
DEFAULT_LIMIT = 10
DEFAULT_RUN_NAME = "graphjudge_rebel_10"
DEFAULT_PROPOSER_WORKERS = 1
MAX_PROPOSER_WORKERS = 16
CRITIC_CANDIDATE_SET_SIZES = ("max4", "fixed1", "fixed2", "fixed4", "variable")
ENTITY_ORDER_MODES = ("directed", "undirected", "predicate_aware")
PROPOSER_MAX_ATTEMPTS = 3
PROPOSER_RETRY_DELAYS = (1.0, 2.0)
CRITIC_ADDITIVE_METRICS = (
    "model_load_seconds",
    "critic_prompt_count",
    "critic_generation_batches",
    "critic_input_tokens",
    "critic_output_tokens",
    "critic_generation_seconds",
)




def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _looks_like_triples(line: str) -> bool:
    try:
        value = ast.literal_eval(line)
    except Exception:
        return False
    return isinstance(value, list)


def parse_gold_triples(line: str) -> List[Dict[str, str]]:
    value = ast.literal_eval(line)
    if not isinstance(value, list):
        raise ValueError("gold triple line is not a list")

    triples = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        subject, predicate, obj = item[:3]
        triples.append({
            "subject": str(subject or "").strip(),
            "predicate": str(predicate or "").strip(),
            "object": str(obj or "").strip(),
        })
    return [
        triple
        for triple in triples
        if triple["subject"] and triple["predicate"] and triple["object"]
    ]


def _read_jsonl_objects(path: Path) -> List[Mapping]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records found")
    return rows


def _gold_triple(value, context: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a triple object")
    triple = {
        field: str(value.get(field) or "").strip()
        for field in ("subject", "predicate", "object")
    }
    if any(not part for part in triple.values()):
        raise ValueError(f"{context}: subject, predicate, and object must be non-empty")
    return triple


def load_canonical_rebel(data_dir: Path, limit: int) -> List[Dict]:
    documents_path = data_dir / "documents.jsonl"
    gold_path = data_dir / "gold_triples.jsonl"
    if not documents_path.exists() or not gold_path.exists():
        raise FileNotFoundError(
            f"canonical REBEL data requires documents.jsonl and gold_triples.jsonl under {data_dir}"
        )

    documents = []
    document_ids = []
    seen_document_ids = set()
    for index, row in enumerate(_read_jsonl_objects(documents_path), start=1):
        try:
            document = normalize_document(dict(row), f"{documents_path}:record {index}")
        except DocumentInputError as exc:
            raise ValueError(str(exc)) from exc
        if document.document_id in seen_document_ids:
            raise ValueError(
                f"{documents_path}: duplicate document_id {document.document_id!r}"
            )
        seen_document_ids.add(document.document_id)
        document_ids.append(document.document_id)
        documents.append(document)

    gold_by_id = {}
    gold_ids = []
    for index, row in enumerate(_read_jsonl_objects(gold_path), start=1):
        document_id = str(row.get("document_id") or "").strip()
        if not document_id:
            raise ValueError(f"{gold_path}:record {index}: document_id is required")
        if document_id in gold_by_id:
            raise ValueError(f"{gold_path}: duplicate document_id {document_id!r}")
        values = row.get("triples")
        if not isinstance(values, list):
            raise ValueError(f"{gold_path}:record {index}: triples must be a list")
        triples = [
            _gold_triple(value, f"{gold_path}:record {index}, triple {triple_index}")
            for triple_index, value in enumerate(values, start=1)
        ]
        gold_ids.append(document_id)
        gold_by_id[document_id] = triples

    if set(document_ids) != set(gold_ids):
        missing_gold = [value for value in document_ids if value not in gold_by_id]
        missing_documents = [value for value in gold_ids if value not in seen_document_ids]
        raise ValueError(
            "canonical document/gold ID sets differ: "
            f"missing_gold={missing_gold[:5]}, missing_documents={missing_documents[:5]}"
        )
    if document_ids != gold_ids:
        raise ValueError("canonical documents and gold triples must use the same document order")

    return [
        {
            "document_id": document.document_id,
            "text": document.text,
            "gold_triples": gold_by_id[document.document_id],
        }
        for document in documents[:limit]
    ]


def load_graphjudge_rebel(data_dir: Path, limit: int) -> List[Dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    canonical_paths = (data_dir / "documents.jsonl", data_dir / "gold_triples.jsonl")
    if any(path.exists() for path in canonical_paths):
        return load_canonical_rebel(data_dir, limit)

    source_path = data_dir / "test.source"
    target_path = data_dir / "test.target"
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"missing GraphJudge files under {data_dir}")

    source_lines = _read_lines(source_path)
    target_lines = _read_lines(target_path)
    if len(source_lines) != len(target_lines):
        raise ValueError(f"line count mismatch: source={len(source_lines)} target={len(target_lines)}")
    if not source_lines:
        raise ValueError("GraphJudge REBEL subset is empty")

    if _looks_like_triples(source_lines[0]) and not _looks_like_triples(target_lines[0]):
        raise ValueError(
            "GraphJudge files still look reversed: test.source contains triples and test.target contains text."
        )
    if _looks_like_triples(source_lines[0]):
        raise ValueError("test.source should contain source text, not triples")
    if not _looks_like_triples(target_lines[0]):
        raise ValueError("test.target should contain gold triples")

    rows = []
    for idx, (text, gold_line) in enumerate(zip(source_lines[:limit], target_lines[:limit])):
        rows.append({
            "document_id": f"graphjudge_rebel_{idx:04d}",
            "text": text.strip(),
            "gold_triples": parse_gold_triples(gold_line),
        })
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")








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


def document_ids_sha256(rows: Sequence[Mapping]) -> str:
    payload = "".join(f"{row['document_id']}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()








def checkpoint_path(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"{index:06d}.json"


def proposal_checkpoint_path(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"{index:06d}.json"


def _serialize_proposer_payload(payload: Mapping) -> Dict:
    triples = []
    for triple in payload.get("triples", []):
        if isinstance(triple, Triple):
            triples.append(triple.to_dict())
        elif isinstance(triple, Mapping):
            triples.append(dict(triple))
        else:
            raise ValueError("proposer payload contains an invalid triple")
    return {
        "triples": triples,
        "raw_response": str(payload.get("raw_response") or ""),
        "parse_error": str(payload.get("parse_error") or ""),
        "rejected_predicates": list(payload.get("rejected_predicates") or []),
    }


def _deserialize_proposer_payload(value: Mapping, row: Mapping) -> Dict:
    triples = []
    for item in value.get("triples", []):
        if not isinstance(item, Mapping):
            raise ValueError("cached proposer triple must be an object")
        triples.append(Triple.from_dict(
            dict(item),
            source_text=row["text"],
            document_id=row["document_id"],
        ))
    return {
        "triples": triples,
        "raw_response": str(value.get("raw_response") or ""),
        "parse_error": str(value.get("parse_error") or ""),
        "rejected_predicates": list(value.get("rejected_predicates") or []),
    }


def _validate_proposal_checkpoint(
    value: Mapping,
    index: int,
    row: Mapping,
    path: Path,
) -> Dict:
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported proposal checkpoint version")
    if value.get("index") != index or value.get("document_id") != row["document_id"]:
        raise ValueError(f"{path}: proposal checkpoint does not match dataset order")
    if value.get("status") not in {"ok", "failed"}:
        raise ValueError(f"{path}: proposal checkpoint status must be ok or failed")
    if not isinstance(value.get("attempts"), int) or value["attempts"] <= 0:
        raise ValueError(f"{path}: proposal attempts must be positive")
    if not isinstance(value.get("api_elapsed_seconds"), (int, float)):
        raise ValueError(f"{path}: proposal api_elapsed_seconds is required")
    if value["status"] == "ok":
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: successful proposal requires payload")
        parsed = _deserialize_proposer_payload(payload, row)
        if parsed["parse_error"]:
            raise ValueError(f"{path}: successful proposal cannot contain parse_error")
    return dict(value)


def load_proposal_checkpoints(
    checkpoint_dir: Path,
    rows: Sequence[Mapping],
) -> Dict[int, Dict]:
    checkpoints = {}
    if not checkpoint_dir.exists():
        return checkpoints
    for index, row in enumerate(rows):
        path = proposal_checkpoint_path(checkpoint_dir, index)
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid proposal checkpoint JSON: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: proposal checkpoint must be an object")
        checkpoints[index] = _validate_proposal_checkpoint(value, index, row, path)
    return checkpoints


def validate_proposal_cache_state(
    cache_dir: Path,
    rows: Sequence[Mapping],
    data_dir: Optional[Path] = None,
) -> None:
    """Validate optional cache metadata before reusing cached proposals."""
    state_path = Path(cache_dir) / "cache_state.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{state_path}: invalid cache state JSON: {exc.msg}") from exc
    if not isinstance(state, Mapping):
        raise ValueError(f"{state_path}: cache state must be a JSON object")
    expected_ids = document_ids_sha256(rows)
    try:
        cache_limit = int(state.get("limit", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{state_path}: cache limit must be an integer") from exc
    if cache_limit != len(rows):
        raise ValueError(f"{state_path}: cache limit does not match selected data")
    if str(state.get("document_ids_sha256") or "") != expected_ids:
        raise ValueError(f"{state_path}: cache document ID order does not match selected data")
    saved_data_dir = state.get("data_dir")
    if data_dir is not None and saved_data_dir:
        if Path(str(saved_data_dir)).resolve() != Path(data_dir).resolve():
            raise ValueError(f"{state_path}: cache data_dir does not match selected data")


def build_proposal_checkpoint(
    proposer,
    row: Mapping,
    index: int,
    path: Path,
    previous_attempts: int = 0,
    sleep_fn=time.sleep,
) -> Dict:
    started = time.perf_counter()
    payload = None
    error = ""
    attempts = previous_attempts
    for local_attempt in range(1, PROPOSER_MAX_ATTEMPTS + 1):
        attempts += 1
        try:
            candidate = proposer.propose_triples(row["document_id"], row["text"])
            if not isinstance(candidate, Mapping):
                error = "Proposer returned a non-object payload."
            else:
                error = str(candidate.get("parse_error") or "")
                if not error:
                    payload = _serialize_proposer_payload(candidate)
                    break
        except Exception as exc:
            error = str(exc)
        if local_attempt < PROPOSER_MAX_ATTEMPTS:
            sleep_fn(PROPOSER_RETRY_DELAYS[local_attempt - 1])

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "index": index,
        "document_id": row["document_id"],
        "attempts": attempts,
        "status": "ok" if payload is not None else "failed",
        "error": "" if payload is not None else error or "Proposer failed.",
        "api_elapsed_seconds": round(time.perf_counter() - started, 6),
        "payload": payload,
    }
    atomic_write_json(path, checkpoint)
    return checkpoint


def proposal_metrics(
    checkpoints: Mapping[int, Mapping],
    *,
    cache_hits: int = 0,
    max_in_flight: int = 0,
) -> Dict:
    attempts = sum(int(value.get("attempts", 0)) for value in checkpoints.values())
    return {
        "proposal_completed": sum(value.get("status") == "ok" for value in checkpoints.values()),
        "proposal_failed": sum(value.get("status") == "failed" for value in checkpoints.values()),
        "proposal_api_attempts": attempts,
        "proposal_retries": max(0, attempts - len(checkpoints)),
        "proposal_cache_hits": cache_hits,
        "proposal_max_in_flight": max_in_flight,
    }


def critic_chunk_sizes(
    count: int,
    variant: str,
    *,
    seed: int = 42,
    document_id: str = "",
) -> Optional[List[int]]:
    """Return deterministic candidate-set boundaries for a Critic request."""
    if count <= 0:
        return []
    variant = {"pointwise": "fixed1"}.get(variant, str(variant or "max4"))
    if variant == "max4":
        return None
    if variant == "fixed1":
        return [1] * count
    if variant == "fixed2":
        sizes = [2] * (count // 2)
        if count % 2:
            sizes.append(1)
        return sizes
    if variant == "fixed4":
        sizes = [4] * (count // 4)
        remainder = count % 4
        if remainder:
            sizes.append(remainder)
        return sizes
    if variant == "variable":
        rng = random.Random(f"{seed}:{document_id}:critic_variable")
        sizes = []
        remaining = count
        while remaining:
            size = rng.choice(list(range(1, min(4, remaining) + 1)))
            sizes.append(size)
            remaining -= size
        return sizes
    raise ValueError(
        "critic_candidate_set_size must be one of "
        f"{CRITIC_CANDIDATE_SET_SIZES}"
    )


def cumulative_critic_metrics(previous: Mapping, current: Mapping) -> Dict:
    output = {
        "backend": "direct",
        "adapter_merged": bool(current.get("adapter_merged", False)),
    }
    for key in CRITIC_ADDITIVE_METRICS:
        output[key] = round(
            float(previous.get(key, 0) or 0) + float(current.get(key, 0) or 0),
            3,
        )
        if key not in {"model_load_seconds", "critic_generation_seconds"}:
            output[key] = int(output[key])
    output["critic_peak_gpu_memory_bytes"] = max(
        int(previous.get("critic_peak_gpu_memory_bytes", 0) or 0),
        int(current.get("critic_peak_gpu_memory_bytes", 0) or 0),
    )
    seconds = float(output["critic_generation_seconds"])
    output["critic_tokens_per_second"] = round(
        output["critic_output_tokens"] / seconds, 3
    ) if seconds else 0.0
    return output


def _validate_checkpoint(value: Mapping, index: int, row: Mapping, path: Path) -> Dict:
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version")
    if value.get("index") != index or value.get("document_id") != row["document_id"]:
        raise ValueError(f"{path}: checkpoint does not match selected dataset order")
    if value.get("status") not in {"ok", "failed"}:
        raise ValueError(f"{path}: checkpoint status must be ok or failed")
    if not isinstance(value.get("attempts"), int) or value["attempts"] <= 0:
        raise ValueError(f"{path}: checkpoint attempts must be positive")
    if not isinstance(value.get("sample_result"), Mapping):
        raise ValueError(f"{path}: sample_result is required")
    if value["sample_result"].get("document_id") != row["document_id"]:
        raise ValueError(f"{path}: sample_result document_id does not match")
    if value["sample_result"].get("status") != value["status"]:
        raise ValueError(f"{path}: sample_result status does not match")
    if value["status"] == "ok":
        if not isinstance(value.get("pipeline_record"), Mapping):
            raise ValueError(f"{path}: successful checkpoint requires pipeline_record")
        if not isinstance(value.get("final_triples"), list):
            raise ValueError(f"{path}: successful checkpoint requires final_triples")
        if value["pipeline_record"].get("document_id") != row["document_id"]:
            raise ValueError(f"{path}: pipeline_record document_id does not match")
        if value["pipeline_record"].get("final_triples") != value["final_triples"]:
            raise ValueError(f"{path}: checkpoint final triples do not match pipeline_record")
    return dict(value)


def load_checkpoints(checkpoint_dir: Path, rows: Sequence[Mapping]) -> Dict[int, Dict]:
    checkpoints = {}
    for index, row in enumerate(rows):
        path = checkpoint_path(checkpoint_dir, index)
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid checkpoint JSON: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: checkpoint must be a JSON object")
        checkpoints[index] = _validate_checkpoint(value, index, row, path)
    unexpected = sorted(
        path.name
        for path in checkpoint_dir.glob("*.json")
        if path.stem.isdigit() and int(path.stem) >= len(rows)
    )
    if unexpected:
        raise ValueError(f"checkpoint directory contains out-of-range files: {unexpected[:5]}")
    return checkpoints


def progress_payload(rows: Sequence[Mapping], checkpoints: Mapping[int, Mapping]) -> Dict:
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


def _sample_result(
    row: Mapping,
    pipeline_record,
    triples,
    status,
    error,
    elapsed,
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
        "extracted_count": len(pipeline_record.extracted_triples) if pipeline_record else 0,
        "refined_count": len(pipeline_record.refined_triples) if pipeline_record else 0,
        "additions_proposed": (
            len(pipeline_record.coverage_audit.get("additions", [])) if pipeline_record else 0
        ),
        "additions_accepted": len(pipeline_record.added_triples) if pipeline_record else 0,
        "elapsed_seconds": round(elapsed, 3),
    }


def process_rows_with_checkpoints(
    controller,
    rows: Sequence[Mapping],
    checkpoint_dir: Path,
    progress_path: Path,
    console_print=print,
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
) -> Dict[int, Dict]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = load_checkpoints(checkpoint_dir, rows)

    controller.kg = []
    controller.pipeline_records = []
    for index in sorted(checkpoints):
        checkpoint = checkpoints[index]
        if checkpoint["status"] != "ok":
            continue
        record = pipeline_record_from_dict(checkpoint["pipeline_record"])
        controller.pipeline_records.append(record)
        controller.kg.extend(_triple_from_dict(item) for item in checkpoint["final_triples"])

    update_progress(progress_path, progress_payload(rows, checkpoints))
    for index, row in enumerate(rows):
        previous = checkpoints.get(index)
        if previous and previous["status"] == "ok":
            console_print(f"[{index + 1}/{len(rows)}] {row['document_id']} checkpoint ok; skipped")
            continue

        attempts = int(previous.get("attempts", 0)) + 1 if previous else 1
        console_print(f"[{index + 1}/{len(rows)}] {row['document_id']} start (attempt {attempts})")
        started = time.time()
        kg_start = len(controller.kg)
        record_start = len(controller.pipeline_records)
        pipeline_record = None
        error = ""
        status = "ok"
        try:
            pipeline_record = controller.process_document(
                document_id=row["document_id"],
                text=row["text"],
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            del controller.kg[kg_start:]
            del controller.pipeline_records[record_start:]
            traceback.print_exc()

        if status == "ok" and len(controller.pipeline_records) == record_start:
            controller.pipeline_records.append(pipeline_record)
            controller.kg.extend(pipeline_record.final_triples)
        final_triples = list(pipeline_record.final_triples) if status == "ok" else []
        sample_result = _sample_result(
            row,
            pipeline_record,
            final_triples,
            status,
            error,
            time.time() - started,
            entity_order_mode,
            symmetric_predicates,
        )
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "index": index,
            "document_id": row["document_id"],
            "attempts": attempts,
            "status": status,
            "error": error,
            "sample_result": sample_result,
            "pipeline_record": pipeline_record.to_dict() if pipeline_record else None,
            "final_triples": [triple.to_dict() for triple in final_triples],
        }
        atomic_write_json(checkpoint_path(checkpoint_dir, index), checkpoint)
        checkpoints[index] = checkpoint
        update_progress(progress_path, progress_payload(rows, checkpoints))
        console_print(
            f"[{index + 1}/{len(rows)}] {row['document_id']} {status} | "
            f"pred={sample_result['predicted_count']} gold={sample_result['gold_count']} "
            f"exact={sample_result['exact_match_count']}"
        )
    return checkpoints


def process_rows_with_concurrent_proposals(
    controller,
    proposer,
    rows: Sequence[Mapping],
    checkpoint_dir: Path,
    proposal_checkpoint_dir: Path,
    progress_path: Path,
    *,
    proposer_workers: int,
    prefetch_window: int,
    console_print=print,
    sleep_fn=time.sleep,
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
    critic_candidate_set_size: str = "max4",
    critic_candidate_set_seed: int = 42,
    proposal_cache_source: Optional[Path] = None,
    proposal_cache_data_dir: Optional[Path] = None,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    proposal_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = load_checkpoints(checkpoint_dir, rows)
    proposals = load_proposal_checkpoints(proposal_checkpoint_dir, rows)
    if proposal_cache_source is not None:
        source = Path(proposal_cache_source).expanduser().resolve()
        validate_proposal_cache_state(source, rows, proposal_cache_data_dir)
        cached = load_proposal_checkpoints(source, rows)
        missing = [index for index in range(len(rows)) if index not in cached]
        missing.extend(
            index for index, value in cached.items() if value.get("status") != "ok"
        )
        if missing:
            raise ValueError(
                f"proposal cache is incomplete; missing or failed indices: {missing[:5]}"
            )
        for index, checkpoint in cached.items():
            target = proposal_checkpoint_path(proposal_checkpoint_dir, index)
            atomic_write_json(target, checkpoint)
            proposals[index] = checkpoint

    controller.kg = []
    controller.pipeline_records = []
    for index in sorted(checkpoints):
        checkpoint = checkpoints[index]
        if checkpoint["status"] != "ok":
            continue
        record = pipeline_record_from_dict(checkpoint["pipeline_record"])
        controller.pipeline_records.append(record)
        controller.kg.extend(_triple_from_dict(item) for item in checkpoint["final_triples"])

    cache_hits = 0
    max_in_flight = 0

    def write_progress():
        value = progress_payload(rows, checkpoints)
        value.update(proposal_metrics(
            proposals,
            cache_hits=cache_hits,
            max_in_flight=max_in_flight,
        ))
        update_progress(progress_path, value)

    write_progress()
    futures: Dict[int, Future] = {}
    next_submit = 0
    executor = ThreadPoolExecutor(max_workers=proposer_workers)

    def fill_prefetch():
        nonlocal next_submit, max_in_flight
        while next_submit < len(rows) and len(futures) < prefetch_window:
            index = next_submit
            next_submit += 1
            final_checkpoint = checkpoints.get(index)
            if final_checkpoint and final_checkpoint["status"] == "ok":
                continue
            proposal_checkpoint = proposals.get(index)
            if proposal_checkpoint and proposal_checkpoint["status"] == "ok":
                continue
            if proposal_cache_source is not None:
                raise RuntimeError(
                    f"proposal cache did not contain successful index {index}"
                )
            previous_attempts = int(proposal_checkpoint.get("attempts", 0)) if proposal_checkpoint else 0
            futures[index] = executor.submit(
                build_proposal_checkpoint,
                proposer,
                rows[index],
                index,
                proposal_checkpoint_path(proposal_checkpoint_dir, index),
                previous_attempts,
                sleep_fn,
            )
            max_in_flight = max(max_in_flight, len(futures))

    try:
        fill_prefetch()
        for window_start in range(0, len(rows), prefetch_window):
            window_end = min(window_start + prefetch_window, len(rows))
            ready = []
            for index in range(window_start, window_end):
                row = rows[index]
                previous = checkpoints.get(index)
                if previous and previous["status"] == "ok":
                    console_print(
                        f"[{index + 1}/{len(rows)}] {row['document_id']} "
                        "checkpoint ok; skipped"
                    )
                    fill_prefetch()
                    continue

                proposal_checkpoint = proposals.get(index)
                if proposal_checkpoint and proposal_checkpoint["status"] == "ok":
                    cache_hits += 1
                else:
                    future = futures.pop(index, None)
                    if future is None:
                        raise RuntimeError(f"proposal {index} was not scheduled")
                    proposal_checkpoint = future.result()
                    proposals[index] = proposal_checkpoint
                    fill_prefetch()

                attempts = int(previous.get("attempts", 0)) + 1 if previous else 1
                if proposal_checkpoint["status"] != "ok":
                    error = f"Proposer failed: {proposal_checkpoint['error']}"
                    sample_result = _sample_result(
                        row,
                        None,
                        [],
                        "failed",
                        error,
                        float(proposal_checkpoint["api_elapsed_seconds"]),
                        entity_order_mode,
                        symmetric_predicates,
                    )
                    checkpoint = {
                        "version": CHECKPOINT_VERSION,
                        "index": index,
                        "document_id": row["document_id"],
                        "attempts": attempts,
                        "status": "failed",
                        "error": error,
                        "sample_result": sample_result,
                        "pipeline_record": None,
                        "final_triples": [],
                    }
                    atomic_write_json(checkpoint_path(checkpoint_dir, index), checkpoint)
                    checkpoints[index] = checkpoint
                    write_progress()
                    console_print(
                        f"[{index + 1}/{len(rows)}] {row['document_id']} failed | {error}"
                    )
                    continue

                payload = _deserialize_proposer_payload(
                    proposal_checkpoint["payload"], row
                )
                ready.append({
                    "index": index,
                    "row": row,
                    "attempts": attempts,
                    "proposal_checkpoint": proposal_checkpoint,
                    "proposal_payload": payload,
                })

            batch_reviewer = getattr(
                getattr(controller, "critic", None),
                "review_documents_batch",
                None,
            )
            if ready and callable(batch_reviewer):
                review_outputs = batch_reviewer([{
                    "triples": item["proposal_payload"]["triples"],
                    "stage": "initial_review",
                    "origins": ["proposer"] * len(item["proposal_payload"]["triples"]),
                    "chunk_sizes": critic_chunk_sizes(
                        len(item["proposal_payload"]["triples"]),
                        critic_candidate_set_size,
                        seed=critic_candidate_set_seed,
                        document_id=item["row"]["document_id"],
                    ),
                } for item in ready])
                if len(review_outputs) != len(ready):
                    raise RuntimeError(
                        "Critic document batch output count does not match input count"
                    )
            else:
                review_outputs = [None] * len(ready)

            for item, review_output in zip(ready, review_outputs):
                index = item["index"]
                row = item["row"]
                attempts = item["attempts"]
                proposal_checkpoint = item["proposal_checkpoint"]
                console_print(
                    f"[{index + 1}/{len(rows)}] {row['document_id']} critic apply "
                    f"(pipeline attempt {attempts})"
                )
                pipeline_started = time.time()
                kg_start = len(controller.kg)
                record_start = len(controller.pipeline_records)
                pipeline_record = None
                error = ""
                status = "ok"
                critic_elapsed = 0.0
                try:
                    kwargs = {}
                    if review_output is not None:
                        critic_elapsed = float(
                            review_output.get("elapsed_seconds", 0.0) or 0.0
                        )
                        kwargs = {
                            "initial_critic_payload": review_output["payload"],
                            "initial_critic_elapsed_seconds": critic_elapsed,
                        }
                    pipeline_record = controller.process_document(
                        document_id=row["document_id"],
                        text=row["text"],
                        proposer_payload=item["proposal_payload"],
                        proposer_elapsed_seconds=(
                            proposal_checkpoint["api_elapsed_seconds"]
                        ),
                        **kwargs,
                    )
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    del controller.kg[kg_start:]
                    del controller.pipeline_records[record_start:]
                    traceback.print_exc()

                if status == "ok" and len(controller.pipeline_records) == record_start:
                    controller.pipeline_records.append(pipeline_record)
                    controller.kg.extend(pipeline_record.final_triples)
                final_triples = list(pipeline_record.final_triples) if status == "ok" else []
                elapsed = (
                    float(proposal_checkpoint["api_elapsed_seconds"])
                    + critic_elapsed
                    + time.time() - pipeline_started
                )
                sample_result = _sample_result(
                    row,
                    pipeline_record,
                    final_triples,
                    status,
                    error,
                    elapsed,
                    entity_order_mode,
                    symmetric_predicates,
                )
                sample_result["critic_candidate_set_sizes"] = critic_chunk_sizes(
                    len(item["proposal_payload"]["triples"]),
                    critic_candidate_set_size,
                    seed=critic_candidate_set_seed,
                    document_id=row["document_id"],
                )
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "index": index,
                    "document_id": row["document_id"],
                    "attempts": attempts,
                    "status": status,
                    "error": error,
                    "sample_result": sample_result,
                    "pipeline_record": (
                        pipeline_record.to_dict() if pipeline_record else None
                    ),
                    "final_triples": [triple.to_dict() for triple in final_triples],
                }
                atomic_write_json(checkpoint_path(checkpoint_dir, index), checkpoint)
                checkpoints[index] = checkpoint
                write_progress()
                console_print(
                    f"[{index + 1}/{len(rows)}] {row['document_id']} {status} | "
                    f"pred={sample_result['predicted_count']} "
                    f"gold={sample_result['gold_count']} "
                    f"exact={sample_result['exact_match_count']}"
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    return checkpoints, proposals, proposal_metrics(
        proposals,
        cache_hits=cache_hits,
        max_in_flight=max_in_flight,
    )


def runtime_signature(
    config: Mapping,
    critic_runtime: Mapping,
    pipeline_variant: str = "full",
    proposer_workers: int = DEFAULT_PROPOSER_WORKERS,
    proposer_prefetch_window: int = 1,
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
    critic_candidate_set_size: str = "max4",
    critic_candidate_set_seed: int = 42,
    proposal_cache_dir: Optional[str] = None,
    require_proposal_cache: bool = False,
) -> Dict:
    public_critic_runtime = {
        key: value
        for key, value in critic_runtime.items()
        if "api_key" not in str(key).lower()
    }
    signature = {
        "pipeline_variant": pipeline_variant,
        "proposer_workers": proposer_workers,
        "proposer_prefetch_window": proposer_prefetch_window,
        "proposer_max_attempts": PROPOSER_MAX_ATTEMPTS,
        "entity_order_mode": entity_order_mode,
        "symmetric_predicates": list(
            normalize_symmetric_predicates(symmetric_predicates)
        ),
        "critic_candidate_set_size": critic_candidate_set_size,
        "critic_candidate_set_seed": int(critic_candidate_set_seed),
        "proposal_cache_dir": proposal_cache_dir,
        "require_proposal_cache": bool(require_proposal_cache),
        "proposer_model": os.environ.get(
            "PROPOSER_MODEL_NAME", config.get("proposer_model_name")
        ),
        "proposer_endpoint": os.environ.get(
            "PROPOSER_BASE_URL", (os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL"))
        ),
        "proposer_temperature": float(
            os.environ.get("PROPOSER_TEMPERATURE", "0.5")
        ),
        "integrator_model": config.get("integrator_model_name"),
        "critic_runtime": public_critic_runtime,
    }
    dataset_key = os.environ.get("EXPERIMENT_DATASET_KEY")
    if dataset_key:
        signature["dataset_key"] = dataset_key
    return signature


def build_run_state(
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
    saved_runtime = dict(saved.get("runtime") or {})
    expected_runtime = dict(expected.get("runtime") or {})
    saved_runtime.setdefault("pipeline_variant", "full")
    expected_runtime.setdefault("pipeline_variant", "full")
    saved_runtime.setdefault("proposer_workers", DEFAULT_PROPOSER_WORKERS)
    expected_runtime.setdefault("proposer_workers", DEFAULT_PROPOSER_WORKERS)
    saved_runtime.setdefault("proposer_prefetch_window", 1)
    expected_runtime.setdefault("proposer_prefetch_window", 1)
    saved_runtime.setdefault("proposer_max_attempts", PROPOSER_MAX_ATTEMPTS)
    expected_runtime.setdefault("proposer_max_attempts", PROPOSER_MAX_ATTEMPTS)
    saved_runtime.setdefault("entity_order_mode", "directed")
    expected_runtime.setdefault("entity_order_mode", "directed")
    saved_runtime.setdefault("symmetric_predicates", [])
    expected_runtime.setdefault("symmetric_predicates", [])
    # Candidate-set controls were added after the original direct-runtime
    # state format. Treat their absence as the legacy max-four behaviour.
    saved_runtime.setdefault("critic_candidate_set_size", "max4")
    expected_runtime.setdefault("critic_candidate_set_size", "max4")
    saved_runtime.setdefault("critic_candidate_set_seed", 42)
    expected_runtime.setdefault("critic_candidate_set_seed", 42)
    saved_runtime.setdefault("proposal_cache_dir", None)
    expected_runtime.setdefault("proposal_cache_dir", None)
    saved_runtime.setdefault("require_proposal_cache", False)
    expected_runtime.setdefault("require_proposal_cache", False)
    if saved_runtime != expected_runtime:
        raise ValueError("resume model routing or Critic runtime configuration has changed")






def triple_key(
    subject: str,
    predicate: str,
    obj: str,
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
) -> Tuple[str, str, str]:
    if entity_order_mode not in ENTITY_ORDER_MODES:
        raise ValueError(f"unsupported entity order mode: {entity_order_mode!r}")
    normalized_subject = str(subject or "").strip().lower()
    normalized_object = str(obj or "").strip().lower()
    normalized_predicate = str(predicate or "").strip().lower()
    if entity_order_mode == "undirected" or (
        entity_order_mode == "predicate_aware"
        and normalized_predicate in normalize_symmetric_predicates(symmetric_predicates)
    ):
        normalized_subject, normalized_object = sorted(
            (normalized_subject, normalized_object)
        )
    return (
        normalized_subject,
        normalized_predicate,
        normalized_object,
    )


def gold_keys(
    gold_triples: Sequence[Dict],
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
) -> set:
    return {
        triple_key(
            item.get("subject"),
            item.get("predicate"),
            item.get("object"),
            entity_order_mode,
            symmetric_predicates,
        )
        for item in gold_triples
    }


def pred_keys(
    pred_triples: Sequence,
    entity_order_mode: str = "directed",
    symmetric_predicates: Sequence[str] = (),
) -> set:
    return {
        triple_key(
            item.subject,
            item.predicate,
            item.object,
            entity_order_mode,
            symmetric_predicates,
        )
        for item in pred_triples
    }


def configure_controller_jsonl(controller, run_dir: Path) -> None:
    controller.output_dir = str(run_dir)
    controller.kg_jsonl_path = str(run_dir / "kg.jsonl")
    controller.pipeline_jsonl_path = str(run_dir / "pipeline_records.jsonl")
    Path(controller.kg_jsonl_path).write_text("", encoding="utf-8")
    Path(controller.pipeline_jsonl_path).write_text("", encoding="utf-8")


def runtime_config(config: Dict, args, critic_runtime: Dict) -> Dict:
    pipeline_variant = getattr(args, "pipeline_variant", None) or "full"
    dataset_label = str(getattr(args, "dataset_label", None) or "REBEL")
    proposer_workers = int(
        getattr(args, "proposer_workers", None) or DEFAULT_PROPOSER_WORKERS
    )
    critic_candidate_set_size = str(
        getattr(args, "critic_candidate_set_size", None) or "max4"
    )
    if critic_candidate_set_size not in CRITIC_CANDIDATE_SET_SIZES:
        raise ValueError(
            "critic_candidate_set_size must be one of "
            f"{CRITIC_CANDIDATE_SET_SIZES}"
        )
    if (
        critic_candidate_set_size != "max4"
        and pipeline_variant != "proposer_critic_once"
    ):
        raise ValueError(
            "candidate-set ablations require pipeline_variant=proposer_critic_once"
        )
    candidate_seed = getattr(args, "critic_candidate_set_seed", None)
    if candidate_seed is None:
        candidate_seed = getattr(args, "seed", 42)
    return {
        "run_name": args.run_name,
        "dataset": dataset_label,
        "data_dir": str(args.data_dir),
        "limit": args.limit,
        "pipeline_variant": pipeline_variant,
        "proposer_workers": proposer_workers,
        "proposer_prefetch_window": 1 if proposer_workers == 1 else proposer_workers * 2,
        "proposer_max_attempts": PROPOSER_MAX_ATTEMPTS,
        "critic_prompt_batch_size": critic_runtime.get("prompt_batch_size"),
        "critic_candidate_set_size": critic_candidate_set_size,
        "critic_candidate_set_seed": int(candidate_seed),
        "proposal_cache_dir": getattr(args, "proposal_cache_dir", None),
        "require_proposal_cache": bool(getattr(args, "require_proposal_cache", False)),
        "predicate_mode": "open",
        "entity_order_mode": getattr(args, "entity_order_mode", "directed"),
        "symmetric_predicates": list(normalize_symmetric_predicates(
            getattr(args, "symmetric_predicate", None) or ()
        )),
        "proposer_model": os.environ.get(
            "PROPOSER_MODEL_NAME", config.get("proposer_model_name")
        ),
        "proposer_endpoint": os.environ.get(
            "PROPOSER_BASE_URL", (os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL"))
        ),
        "proposer_temperature": float(
            os.environ.get("PROPOSER_TEMPERATURE", "0.5")
        ),
        "critic_model": critic_runtime.get("model_name"),
        "integrator_model": config.get("integrator_model_name"),
        "integrator_enabled": pipeline_variant == "full",
        "final_critic_enabled": pipeline_variant == "full",
        "critic_runtime": dict(critic_runtime),
    }


def build_summary(
    controller,
    sample_results: List[Dict],
    run_dir: Path,
    runtime_config: Dict,
) -> Dict:
    initial_actions = Counter(
        review.action
        for record in controller.pipeline_records
        for review in record.initial_reviews
    )
    final_actions = Counter(
        review.action
        for record in controller.pipeline_records
        for review in record.final_reviews
    )
    extracted_total = sum(len(record.extracted_triples) for record in controller.pipeline_records)
    refined_total = sum(len(record.refined_triples) for record in controller.pipeline_records)
    additions_proposed = sum(
        len(record.coverage_audit.get("additions", []))
        for record in controller.pipeline_records
    )
    additions_accepted = sum(len(record.added_triples) for record in controller.pipeline_records)
    initial_reviews = [
        review
        for record in controller.pipeline_records
        for review in record.initial_reviews
    ]
    final_reviews = [
        review
        for record in controller.pipeline_records
        for review in record.final_reviews
    ]
    initial_parse_error_documents = sum(
        any(error.startswith("initial_review:") for error in record.stage_errors)
        for record in controller.pipeline_records
    )
    final_parse_error_documents = sum(
        any(error.startswith("final_validation:") for error in record.stage_errors)
        for record in controller.pipeline_records
    )
    critic_parse_error_documents = sum(
        any(
            error.startswith(("initial_review:", "final_validation:"))
            for error in record.stage_errors
        )
        for record in controller.pipeline_records
    )
    stage_errors_total = sum(
        len(record.stage_errors) for record in controller.pipeline_records
    )
    stage_names = ("proposer", "initial_critic", "integrator", "final_critic")
    stage_elapsed_totals = {
        stage: round(sum(
            record.stage_timings_seconds.get(stage, 0.0)
            for record in controller.pipeline_records
        ), 3)
        for stage in stage_names
    }
    record_count = len(controller.pipeline_records)
    stage_elapsed_means = {
        stage: round(total / record_count, 3) if record_count else 0.0
        for stage, total in stage_elapsed_totals.items()
    }

    gold_total = sum(item.get("gold_count", 0) for item in sample_results)
    pred_total = sum(item.get("predicted_count", 0) for item in sample_results)
    exact_total = sum(item.get("exact_match_count", 0) for item in sample_results)
    precision = exact_total / pred_total if pred_total else 0.0
    recall = exact_total / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    candidate_set_counts = Counter()
    for item in sample_results:
        for size in item.get("critic_candidate_set_sizes") or []:
            candidate_set_counts[str(size)] += 1

    return {
        "output_dir": str(run_dir),
        "samples_requested": len(sample_results),
        "samples_completed": sum(1 for item in sample_results if item.get("status") == "ok"),
        "samples_failed": sum(1 for item in sample_results if item.get("status") != "ok"),
        "kg_triples": len(controller.kg),
        "pipeline_records": len(controller.pipeline_records),
        "extracted_triples": extracted_total,
        "refined_triples": refined_total,
        "integrator_additions_proposed": additions_proposed,
        "integrator_additions_accepted": additions_accepted,
        "gold_triples_total": gold_total,
        "predicted_spo_total": pred_total,
        "exact_match_spo_total": exact_total,
        "exact_match_precision": round(precision, 4),
        "exact_match_recall": round(recall, 4),
        "exact_match_f1": round(f1, 4),
        "initial_critic_actions": dict(initial_actions),
        "final_critic_actions": dict(final_actions),
        "initial_reviews_valid": sum(review.valid for review in initial_reviews),
        "initial_reviews_fallback": sum(not review.valid for review in initial_reviews),
        "final_reviews_valid": sum(review.valid for review in final_reviews),
        "final_reviews_fallback": sum(not review.valid for review in final_reviews),
        "initial_critic_parse_error_documents": initial_parse_error_documents,
        "final_critic_parse_error_documents": final_parse_error_documents,
        "critic_parse_error_documents": critic_parse_error_documents,
        "stage_errors_total": stage_errors_total,
        "stage_elapsed_seconds_total": stage_elapsed_totals,
        "stage_elapsed_seconds_mean": stage_elapsed_means,
        "critic_candidate_set_size_counts": dict(candidate_set_counts),
        "runtime_config": runtime_config,
        "samples": sample_results,
    }


def materialize_run_outputs(
    controller,
    rows: Sequence[Mapping],
    checkpoint_dir: Path,
    run_dir: Path,
    experiment_config: Mapping,
    summary_extra: Optional[Mapping] = None,
) -> Dict:
    checkpoints = load_checkpoints(checkpoint_dir, rows)
    if len(checkpoints) != len(rows):
        missing = [index for index in range(len(rows)) if index not in checkpoints]
        raise ValueError(f"cannot finalize run with missing checkpoints: {missing[:5]}")

    sample_results = []
    record_dicts = []
    kg_dicts = []
    pipeline_records = []
    kg = []
    for index in range(len(rows)):
        checkpoint = checkpoints[index]
        sample_results.append(dict(checkpoint["sample_result"]))
        if checkpoint["status"] != "ok":
            continue
        record_dict = dict(checkpoint["pipeline_record"])
        triple_dicts = [dict(item) for item in checkpoint["final_triples"]]
        record_dicts.append(record_dict)
        kg_dicts.extend(triple_dicts)
        pipeline_records.append(pipeline_record_from_dict(record_dict))
        kg.extend(_triple_from_dict(item) for item in triple_dicts)

    controller.pipeline_records = pipeline_records
    controller.kg = kg
    gold_rows = [
        {
            "document_id": row["document_id"],
            "text": row["text"],
            "triples": row["gold_triples"],
        }
        for row in rows
    ]
    atomic_write_jsonl(run_dir / "gold_triples.jsonl", gold_rows)
    atomic_write_jsonl(run_dir / "sample_results.jsonl", sample_results)
    atomic_write_jsonl(run_dir / "pipeline_records.jsonl", record_dicts)
    atomic_write_jsonl(run_dir / "kg.jsonl", kg_dicts)
    _atomic_write_text(
        run_dir / "pipeline_records.json",
        json.dumps(record_dicts, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        run_dir / "kg.json",
        json.dumps(kg_dicts, ensure_ascii=False, indent=2) + "\n",
    )
    summary = build_summary(
        controller,
        sample_results,
        run_dir,
        dict(experiment_config),
    )
    if summary_extra:
        summary.update(dict(summary_extra))
    atomic_write_json(run_dir / "summary.json", summary)
    progress = progress_payload(rows, checkpoints)
    if summary_extra:
        progress.update(dict(summary_extra))
    progress["finalized"] = True
    atomic_write_json(run_dir / "progress.json", progress)
    return summary


def _resolve_repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json_object(path: Path) -> Dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def run_experiment(args) -> Path:
    load_dotenv_file(ROOT / ".env")
    dataset_label = str(getattr(args, "dataset_label", None) or "REBEL")

    saved_state = None
    if args.resume_run_dir:
        run_dir = _resolve_repo_path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        state_path = run_dir / "run_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        saved_state = _read_json_object(state_path)
        saved_data = saved_state.get("data") or {}
        data_dir = _resolve_repo_path(args.data_dir).resolve() if args.data_dir else Path(saved_data.get("data_dir", ""))
        limit = args.limit if args.limit is not None else saved_data.get("requested_limit")
        run_name = args.run_name or str(saved_state.get("run_name") or "")
        saved_runtime = saved_state.get("runtime") or {}
        saved_variant = str(saved_runtime.get("pipeline_variant") or "full")
        pipeline_variant = args.pipeline_variant or saved_variant
        saved_entity_order_mode = str(
            saved_runtime.get("entity_order_mode") or "directed"
        )
        entity_order_mode = (
            getattr(args, "entity_order_mode", None) or saved_entity_order_mode
        )
        requested_symmetric_predicates = getattr(args, "symmetric_predicate", None)
        symmetric_predicates = normalize_symmetric_predicates(
            requested_symmetric_predicates
            if requested_symmetric_predicates is not None
            else saved_runtime.get("symmetric_predicates") or ()
        )
        saved_workers = int(saved_runtime.get("proposer_workers") or DEFAULT_PROPOSER_WORKERS)
        proposer_workers = (
            args.proposer_workers
            if args.proposer_workers is not None
            else saved_workers
        )
        saved_critic_runtime = saved_runtime.get("critic_runtime") or {}
        if saved_critic_runtime.get("backend") != "direct":
            raise ValueError(
                "legacy vLLM runs cannot be resumed with the direct Critic backend"
            )
        saved_prompt_batch_size = int(
            saved_critic_runtime.get("prompt_batch_size") or 4
        )
        critic_prompt_batch_size = (
            args.critic_prompt_batch_size
            if args.critic_prompt_batch_size is not None
            else saved_prompt_batch_size
        )
        saved_candidate_set_size = str(
            saved_runtime.get("critic_candidate_set_size") or "max4"
        )
        critic_candidate_set_size = (
            args.critic_candidate_set_size
            if args.critic_candidate_set_size is not None
            else saved_candidate_set_size
        )
        saved_candidate_seed = int(
            saved_runtime.get("critic_candidate_set_seed", 42)
        )
        critic_candidate_set_seed = (
            int(args.critic_candidate_set_seed)
            if args.critic_candidate_set_seed != 42
            else saved_candidate_seed
        )
        saved_cache_dir = saved_runtime.get("proposal_cache_dir")
        proposal_cache_dir = (
            str(args.proposal_cache_dir)
            if args.proposal_cache_dir is not None
            else saved_cache_dir
        )
        require_proposal_cache = bool(
            args.require_proposal_cache
            or saved_runtime.get("require_proposal_cache", False)
        )
        if args.run_name and args.run_name != saved_state.get("run_name"):
            raise ValueError("--run-name does not match the resumed run")
    else:
        data_dir = _resolve_repo_path(args.data_dir or DEFAULT_DATA_DIR).resolve()
        limit = args.limit if args.limit is not None else DEFAULT_LIMIT
        run_name = args.run_name or DEFAULT_RUN_NAME
        pipeline_variant = args.pipeline_variant or "full"
        entity_order_mode = getattr(args, "entity_order_mode", None) or "directed"
        symmetric_predicates = normalize_symmetric_predicates(
            getattr(args, "symmetric_predicate", None) or ()
        )
        proposer_workers = (
            args.proposer_workers
            if args.proposer_workers is not None
            else DEFAULT_PROPOSER_WORKERS
        )
        critic_prompt_batch_size = (
            args.critic_prompt_batch_size
            if args.critic_prompt_batch_size is not None
            else 4
        )
        critic_candidate_set_size = args.critic_candidate_set_size or "max4"
        critic_candidate_set_seed = int(args.critic_candidate_set_seed)
        proposal_cache_dir = args.proposal_cache_dir
        require_proposal_cache = bool(args.require_proposal_cache)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "data" / "experiments" / run_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be positive")
    symmetric_predicates = validate_entity_order_config(
        entity_order_mode, symmetric_predicates
    )
    if not 1 <= int(proposer_workers) <= MAX_PROPOSER_WORKERS:
        raise ValueError(f"proposer_workers must be between 1 and {MAX_PROPOSER_WORKERS}")
    proposer_workers = int(proposer_workers)
    if proposer_workers > 1 and pipeline_variant != "proposer_critic_once":
        raise ValueError(
            "concurrent Proposer is only supported for proposer_critic_once"
        )
    proposer_prefetch_window = 1 if proposer_workers == 1 else proposer_workers * 2
    if not 1 <= int(critic_prompt_batch_size) <= 4:
        raise ValueError("critic_prompt_batch_size must be between 1 and 4")
    critic_prompt_batch_size = int(critic_prompt_batch_size)
    if critic_candidate_set_size not in CRITIC_CANDIDATE_SET_SIZES:
        raise ValueError(
            "critic_candidate_set_size must be one of "
            f"{CRITIC_CANDIDATE_SET_SIZES}"
        )
    if int(critic_candidate_set_seed) < 0:
        raise ValueError("critic_candidate_set_seed must be non-negative")
    if require_proposal_cache and not proposal_cache_dir:
        raise ValueError(
            "--require-proposal-cache requires --proposal-cache-dir"
        )
    if proposal_cache_dir:
        proposal_cache_dir = str(_resolve_repo_path(proposal_cache_dir).resolve())
        if not Path(proposal_cache_dir).is_dir():
            raise FileNotFoundError(proposal_cache_dir)
    os.environ["CRITIC_PROMPT_BATCH_SIZE"] = str(critic_prompt_batch_size)
    rows = load_graphjudge_rebel(data_dir, limit)

    console = sys.__stdout__

    def console_print(message: str) -> None:
        print(message, file=console, flush=True)

    atomic_write_jsonl(
        run_dir / "gold_triples.jsonl",
        ({
            "document_id": row["document_id"],
            "text": row["text"],
            "triples": row["gold_triples"],
        } for row in rows),
    )

    args.data_dir = str(data_dir)
    args.limit = limit
    args.run_name = run_name
    args.pipeline_variant = pipeline_variant
    args.proposer_workers = proposer_workers
    args.critic_prompt_batch_size = critic_prompt_batch_size
    args.critic_candidate_set_size = critic_candidate_set_size
    args.critic_candidate_set_seed = int(critic_candidate_set_seed)
    args.proposal_cache_dir = proposal_cache_dir
    args.require_proposal_cache = require_proposal_cache
    args.dataset_label = dataset_label
    args.entity_order_mode = entity_order_mode
    args.symmetric_predicate = list(symmetric_predicates)

    from evidenceloop.runtime.settings import public_critic_runtime
    critic_runtime = public_critic_runtime(controller.critic.runtime_settings)
    from evidenceloop.settings import agent_config

    experiment_config = runtime_config(agent_config, args, critic_runtime)
    expected_state = build_run_state(
        data_dir,
        limit,
        rows,
        run_name,
        runtime_signature(
            agent_config,
            critic_runtime,
            pipeline_variant,
            proposer_workers,
            proposer_prefetch_window,
            entity_order_mode,
            symmetric_predicates,
            critic_candidate_set_size,
            critic_candidate_set_seed,
            proposal_cache_dir,
            require_proposal_cache,
        ),
    )
    if saved_state is not None:
        validate_resume_state(saved_state, expected_state)
    else:
        atomic_write_json(run_dir / "run_state.json", expected_state)

    log_path = run_dir / "run.log"
    console_print(f"[setup] loaded {len(rows)} {dataset_label} samples")
    console_print(f"[setup] output dir: {run_dir}")
    if saved_state is not None:
        console_print("[setup] resume state validated")

    log_mode = "a" if saved_state is not None else "w"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        event = "resumed" if saved_state is not None else "started"
        print(
            f"{dataset_label} pipeline run {event} at "
            f"{datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )

        from evidenceloop.pipeline.core import Pipeline

        print(f"Runtime config: {json.dumps(experiment_config, ensure_ascii=False, indent=2)}", flush=True)

        controller = Pipeline(pipeline_variant=pipeline_variant)
        progress_path = run_dir / "progress.json"
        previous_active_seconds = 0.0
        previous_progress = {}
        if progress_path.exists():
            previous_progress = _read_json_object(progress_path)
            previous_active_seconds = float(
                previous_progress.get("active_wall_seconds", 0.0) or 0.0
            )
        processing_started = time.perf_counter()
        proposal_run_metrics = proposal_metrics({})
        critic_run_metrics = {}
        try:
            if (
                proposer_workers > 1
                or proposal_cache_dir
                or critic_candidate_set_size != "max4"
            ):
                checkpoints, proposals, proposal_run_metrics = (
                    process_rows_with_concurrent_proposals(
                        controller,
                        controller.proposer,
                        rows,
                        run_dir / "checkpoints",
                        run_dir / "proposal_checkpoints",
                        progress_path,
                        proposer_workers=proposer_workers,
                        prefetch_window=proposer_prefetch_window,
                        console_print=console_print,
                        entity_order_mode=entity_order_mode,
                        symmetric_predicates=symmetric_predicates,
                        critic_candidate_set_size=critic_candidate_set_size,
                        critic_candidate_set_seed=critic_candidate_set_seed,
                        proposal_cache_source=proposal_cache_dir,
                        proposal_cache_data_dir=data_dir,
                    )
                )
            else:
                checkpoints = process_rows_with_checkpoints(
                    controller,
                    rows,
                    run_dir / "checkpoints",
                    progress_path,
                    console_print=console_print,
                    entity_order_mode=entity_order_mode,
                    symmetric_predicates=symmetric_predicates,
                )
        finally:
            active_wall_seconds = round(
                previous_active_seconds + time.perf_counter() - processing_started,
                3,
            )
            if progress_path.exists():
                progress = _read_json_object(progress_path)
                progress["active_wall_seconds"] = active_wall_seconds
                critic_engine = getattr(controller.critic, "llm", None)
                if hasattr(critic_engine, "metrics"):
                    critic_run_metrics = cumulative_critic_metrics(
                        previous_progress,
                        critic_engine.metrics(),
                    )
                    progress.update(critic_run_metrics)
                atomic_write_json(progress_path, progress)

        completed = sum(
            checkpoint.get("status") == "ok" for checkpoint in checkpoints.values()
        )
        execution_metrics = {
            "proposer_workers": proposer_workers,
            "proposer_prefetch_window": proposer_prefetch_window,
            "proposer_max_attempts": PROPOSER_MAX_ATTEMPTS,
            **proposal_run_metrics,
            "active_wall_seconds": active_wall_seconds,
            "documents_per_minute": round(
                completed * 60 / active_wall_seconds, 3
            ) if active_wall_seconds else 0.0,
        }
        execution_metrics.update(critic_run_metrics)
        summary = materialize_run_outputs(
            controller,
            rows,
            run_dir / "checkpoints",
            run_dir,
            experiment_config,
            execution_metrics,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    console_print(
        "[done] "
        f"completed={summary['samples_completed']}/{summary['samples_requested']} "
        f"kg={summary['kg_triples']} records={summary['pipeline_records']} "
        f"exact={summary['exact_match_spo_total']}/{summary['gold_triples_total']} "
        f"p/r/f1={summary['exact_match_precision']}/{summary['exact_match_recall']}/{summary['exact_match_f1']}"
    )
    console_print(f"[done] detailed log: {log_path}")
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the GraphJudge REBEL pipeline with the required Qwen LoRA Critic."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument(
        "--pipeline-variant",
        choices=("full", "proposer_critic_once"),
        default=None,
    )
    parser.add_argument("--proposer-workers", type=int, default=None)
    parser.add_argument("--critic-prompt-batch-size", type=int, default=None)
    parser.add_argument(
        "--critic-candidate-set-size",
        choices=CRITIC_CANDIDATE_SET_SIZES,
        default=None,
        help="candidate grouping sent inside each Critic prompt",
    )
    parser.add_argument("--critic-candidate-set-seed", type=int, default=42)
    parser.add_argument("--proposal-cache-dir", default=None)
    parser.add_argument("--require-proposal-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--entity-order-mode",
        choices=ENTITY_ORDER_MODES,
        default=None,
    )
    parser.add_argument(
        "--symmetric-predicate",
        action="append",
        default=None,
        help="repeat for each predicate whose subject/object order is symmetric",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
