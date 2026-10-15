"""Version-one codecs and unchanged proposal retries/candidate grouping."""

from __future__ import annotations

import ast
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.documents import DocumentInputError, normalize_document
from evidenceloop.evaluation.direction import (
    ENTITY_ORDER_MODES,
    normalize_symmetric_predicates,
)
from evidenceloop.evaluation.direction import (
    validate_entity_order_config as validate_entity_order_config,
)
from evidenceloop.io import atomic_write_json
from evidenceloop.serialization import pipeline_record_from_dict as pipeline_record_from_dict
from evidenceloop.types import Triple

ROOT = Path.cwd()
CHECKPOINT_VERSION = 1
PROPOSER_MAX_ATTEMPTS = 3
PROPOSER_RETRY_DELAYS = (1.0, 2.0)
CRITIC_CANDIDATE_SET_SIZES = ("max4", "fixed1", "fixed2", "fixed4", "variable")
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
        triples.append(
            {
                "subject": str(subject or "").strip(),
                "predicate": str(predicate or "").strip(),
                "object": str(obj or "").strip(),
            }
        )
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
        field: str(value.get(field) or "").strip() for field in ("subject", "predicate", "object")
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
            raise ValueError(f"{documents_path}: duplicate document_id {document.document_id!r}")
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
        raise ValueError(
            f"line count mismatch: source={len(source_lines)} target={len(target_lines)}"
        )
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
        rows.append(
            {
                "document_id": f"graphjudge_rebel_{idx:04d}",
                "text": text.strip(),
                "gold_triples": parse_gold_triples(gold_line),
            }
        )
    return rows


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
        triples.append(
            Triple.from_dict(
                dict(item),
                source_text=row["text"],
                document_id=row["document_id"],
            )
        )
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
    raise ValueError(f"critic_candidate_set_size must be one of {CRITIC_CANDIDATE_SET_SIZES}")


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
    output["critic_tokens_per_second"] = (
        round(output["critic_output_tokens"] / seconds, 3) if seconds else 0.0
    )
    return output


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
        normalized_subject, normalized_object = sorted((normalized_subject, normalized_object))
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


def _resolve_repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
