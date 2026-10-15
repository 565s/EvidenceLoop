"""Pure-Python contracts and checkpointing shared by the five baselines."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evidenceloop.data.documents import Document, load_documents
from evidenceloop.data.registry import DATASETS as DATASET_SPECS
from evidenceloop.data.registry import AliasDict
from evidenceloop.io import atomic_write_json, atomic_write_jsonl
from evidenceloop.types import DocumentPipelineRecord, Triple
from evidenceloop.validation import clean_predicate, predicate_validation_error

ROOT = Path.cwd()
METHODS = ("rakg", "itext2kg", "pive", "kggen", "graphjudge")
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_OUTPUT_ROOT = ROOT / "data/experiments/related_baselines"
DEFAULT_MODEL_ROOT = ROOT / "models/related_baselines"
VERSION = 1
PIVE_BASE = "philschmid/flan-t5-xxl-sharded-fp16"
PIVE_ADAPTER = "Jiuzhouh/flan-t5-xxl-lora-verifier"
GJ_BASE = "NousResearch/Llama-2-7b-hf"
GJ_ADAPTER = "HaoyuHuang2/graphjudger"
EMBEDDING_MODEL = "BAAI/bge-m3"
PIVE_INSTRUCTION = "Predict the missing triple given the text and graph for KELM dataset."
SOURCES = {
    "rakg": {
        "repository": "https://github.com/KnowledgeXLab/RAKG",
        "revision": "7cdd1d230be91ac9a2e6661ae9d60f1d28e7391f",
    },
    "itext2kg": {
        "repository": "https://github.com/AuvaLab/itext2kg",
        "package": "itext2kg",
        "version": "0.0.7",
    },
    "pive": {
        "repository": "https://github.com/Jiuzhouh/PiVe",
        "revision": "3c1ec7e864ea8fdcadee9e9170ede795df4a1d48",
    },
    "kggen": {
        "repository": "https://github.com/stair-lab/kg-gen",
        "package": "kg-gen",
        "version": "0.2.0",
    },
    "graphjudge": {
        "repository": "https://github.com/hhy-huang/GraphJudge",
        "revision": "0ae5913d49d8b530c254cbc0cefa71a57d107bb5",
    },
}


@dataclass(frozen=True)
class Dataset:
    directory: str
    limit: int
    entity_order_mode: str = "directed"
    symmetric_predicates: tuple[str, ...] = ()
    empty_gold_policy: str = "error"


DATASETS = AliasDict(
    {
        key: Dataset(
            spec.directory,
            spec.limit,
            spec.entity_order_mode,
            spec.symmetric_predicates,
            spec.empty_gold_policy,
        )
        for key, spec in DATASET_SPECS.items()
    }
)


def resolve_path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(value)
    return rows


def load_cohort(dataset: str, data_dir=None, limit=None) -> tuple[Path, list[Document]]:
    preset = DATASETS[dataset]
    directory = resolve_path(data_dir) if data_dir else ROOT / "data/pipeline" / preset.directory
    selected_limit = preset.limit if limit is None else limit
    if selected_limit <= 0:
        raise ValueError("limit must be positive")
    documents = load_documents(str(directory / "documents.jsonl"))
    if len({d.document_id for d in documents}) != len(documents):
        raise ValueError("duplicate document IDs in documents.jsonl")
    if selected_limit > len(documents):
        raise ValueError(f"requested {selected_limit} documents, but only {len(documents)} exist")
    return directory.resolve(), documents[:selected_limit]


def manifest(documents: Sequence[Document]) -> list[dict]:
    return [
        {
            "index": i,
            "document_id": d.document_id,
            "text_sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest(),
        }
        for i, d in enumerate(documents)
    ]


class ParseError(ValueError):
    pass


def parse_structure(response: str):
    """Accept JSON, fenced JSON and the Python-list representation used upstream."""
    text = str(response).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(text)
            if isinstance(value, (dict, list)):
                return value
        except (ValueError, SyntaxError, TypeError):
            pass
    # Only inspect top-level balanced payloads; do not mistake a nested [] for a graph.
    decoder = json.JSONDecoder()
    match = re.search(r"[\[{]", text)
    if match:
        try:
            value, end = decoder.raw_decode(text[match.start() :])
        except ValueError:
            value = None
        if isinstance(value, (dict, list)) and not text[match.start() + end :].strip().startswith(
            ("]", "}")
        ):
            return value
    raise ParseError("response does not contain a valid JSON object or graph list")


def graph_rows(response: str) -> list:
    value = parse_structure(response)
    if isinstance(value, dict):
        for key in ("triples", "relations", "graph", "semantic_graph"):
            if key in value:
                value = value[key]
                break
        else:
            raise ParseError(
                "graph object must contain triples, relations, graph or semantic_graph"
            )
    if not isinstance(value, list):
        raise ParseError("graph must be a list")
    return value


def normalize_triples(values: Sequence) -> tuple[list[dict], list[dict]]:
    triples, rejected, seen = [], [], set()
    for index, value in enumerate(values):
        if isinstance(value, Triple):
            parts = (value.subject, value.predicate, value.object)
        elif isinstance(value, Mapping):
            parts = (value.get("subject"), value.get("predicate"), value.get("object"))
        elif isinstance(value, (tuple, list)) and len(value) == 3:
            parts = tuple(value)
        else:
            raise ParseError(f"triple {index}: expected three fields")
        if any(not isinstance(part, str) for part in parts):
            raise ParseError(f"triple {index}: subject, predicate and object must be strings")
        subject, predicate, obj = parts
        error = predicate_validation_error(predicate)
        if not subject.strip() or not obj.strip():
            error = "subject and object must be non-empty"
        if error:
            rejected.append({"index": index, "triple": list(parts), "reason": error})
            continue
        key = (subject.strip(), clean_predicate(predicate), obj.strip())
        if key not in seen:
            seen.add(key)
            triples.append(dict(zip(("subject", "predicate", "object"), key)))
    return triples, rejected


@dataclass
class Extraction:
    triples: list
    stages: list[dict] = field(default_factory=list)
    extracted_triples: list | None = None
    metadata: dict = field(default_factory=dict)


def checkpoint_path(run_dir: Path, index: int) -> Path:
    return run_dir / "checkpoints" / f"{index:06d}.json"


def load_checkpoints(run_dir: Path, selected: Sequence[dict]) -> dict[int, dict]:
    checkpoints = {}
    for row in selected:
        index = row["index"]
        path = checkpoint_path(run_dir, index)
        if not path.exists():
            continue
        value = read_json(path)
        if (
            value.get("version") != VERSION
            or value.get("document") != row
            or value.get("status") not in ("ok", "failed")
            or not isinstance(value.get("attempts"), int)
            or value["attempts"] <= 0
        ):
            raise ValueError(f"{path}: checkpoint identity or status mismatch")
        if value["status"] == "ok":
            if not isinstance(value.get("final_triples"), list) or not isinstance(
                value.get("pipeline_record"), dict
            ):
                raise ValueError(f"{path}: successful checkpoint has no output")
            record = value["pipeline_record"]
            if (
                record.get("document_id") != row["document_id"]
                or record.get("final_triples") != value["final_triples"]
                or hashlib.sha256(record.get("source_text", "").encode("utf-8")).hexdigest()
                != row["text_sha256"]
            ):
                raise ValueError(f"{path}: checkpoint output identity mismatch")
            for triple in value["final_triples"]:
                if (
                    not isinstance(triple, dict)
                    or triple.get("document_id") != row["document_id"]
                    or any(
                        not isinstance(triple.get(k), str) or not triple[k].strip()
                        for k in ("subject", "predicate", "object")
                    )
                ):
                    raise ValueError(f"{path}: invalid checkpoint triple")
        checkpoints[index] = value
    return checkpoints


def progress_payload(selected, checkpoints, *, finalized=False, active_wall_seconds=0.0):
    completed = sum(cp["status"] == "ok" for cp in checkpoints.values())
    failed = sum(cp["status"] == "failed" for cp in checkpoints.values())
    return {
        "total": len(selected),
        "completed": completed,
        "failed": failed,
        "pending": len(selected) - len(checkpoints),
        "finalized": finalized,
        "complete": finalized and completed == len(selected),
        "active_wall_seconds": active_wall_seconds,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def validate_run(
    run_dir: Path, *, require_complete=True
) -> tuple[dict, list[dict], dict[int, dict]]:
    state = read_json(run_dir / "run_state.json")
    selected = read_jsonl(run_dir / "document_manifest.jsonl")
    if (
        state.get("version") != VERSION
        or state.get("kind") != "related_baseline"
        or digest(selected) != state.get("data", {}).get("manifest_sha256")
        or len(selected) != state["data"]["selected_documents"]
        or [r["index"] for r in selected] != list(range(len(selected)))
        or len({r["document_id"] for r in selected}) != len(selected)
    ):
        raise ValueError(f"{run_dir}: invalid run state or document manifest")
    checkpoints = load_checkpoints(run_dir, selected)
    if require_complete:
        progress = read_json(run_dir / "progress.json")
        if (
            not progress.get("complete")
            or len(checkpoints) != len(selected)
            or any(cp["status"] != "ok" for cp in checkpoints.values())
        ):
            raise ValueError(f"{run_dir}: evaluation requires a complete successful run")
    return state, selected, checkpoints


def process_documents(
    adapter,
    documents,
    run_dir: Path,
    runtime: Mapping,
    *,
    sleep_fn=time.sleep,
    report: Callable = print,
):
    selected = manifest(documents)
    checkpoints = load_checkpoints(run_dir, selected)
    old_progress = (
        read_json(run_dir / "progress.json") if (run_dir / "progress.json").exists() else {}
    )
    previous_wall = float(old_progress.get("active_wall_seconds", 0.0))
    started = time.perf_counter()
    atomic_write_json(
        run_dir / "progress.json",
        progress_payload(selected, checkpoints, active_wall_seconds=previous_wall),
    )
    try:
        for index, document in enumerate(documents):
            previous = checkpoints.get(index, {})
            if previous.get("status") == "ok":
                report(f"[{index + 1}/{len(documents)}] {document.document_id}: skipped")
                continue
            attempts = previous.get("attempts", 0)
            checkpoint = {"version": VERSION, "document": selected[index], "status": "failed"}
            document_started = time.perf_counter()
            for attempt in range(runtime["max_attempts"]):
                attempts += 1
                try:
                    result = adapter.extract(document.document_id, document.text)
                    triples, rejected = normalize_triples(result.triples)
                    extracted, _ = normalize_triples(
                        result.extracted_triples
                        if result.extracted_triples is not None
                        else triples
                    )
                    final = [
                        Triple.from_dict(
                            t, source_text=document.text, document_id=document.document_id
                        )
                        for t in triples
                    ]
                    record = DocumentPipelineRecord(
                        document_id=document.document_id,
                        source_text=document.text,
                        extracted_triples=[
                            Triple.from_dict(
                                t, source_text=document.text, document_id=document.document_id
                            )
                            for t in extracted
                        ],
                        refined_triples=final,
                        final_triples=final,
                        stage_timings_seconds={
                            s["stage"]: s.get("elapsed_seconds", 0.0) for s in result.stages
                        },
                        raw_responses={
                            s["stage"]: str(s.get("response", "")) for s in result.stages
                        },
                    ).to_dict()
                    checkpoint.update(
                        status="ok",
                        error="",
                        final_triples=[t.to_dict() for t in final],
                        pipeline_record=record,
                        stages=result.stages,
                        metadata=result.metadata,
                        rejected_triples=rejected,
                    )
                    break
                except Exception as exc:
                    # A partial method result is never promoted to a successful baseline graph.
                    from .runtime import redact

                    checkpoint.update(
                        error=redact(str(exc)),
                        final_triples=[],
                        pipeline_record=None,
                        stages=getattr(adapter, "stages", []),
                    )
                    if attempt + 1 < runtime["max_attempts"]:
                        sleep_fn(runtime["retry_delay"] * (attempt + 1))
            checkpoint.update(
                attempts=attempts, elapsed_seconds=round(time.perf_counter() - document_started, 6)
            )
            atomic_write_json(checkpoint_path(run_dir, index), checkpoint)
            checkpoints[index] = checkpoint
            atomic_write_json(
                run_dir / "progress.json",
                progress_payload(
                    selected,
                    checkpoints,
                    active_wall_seconds=previous_wall + time.perf_counter() - started,
                ),
            )
            report(
                f"[{index + 1}/{len(documents)}] {document.document_id}: {checkpoint['status']}; triples={len(checkpoint['final_triples'])}"
            )
    finally:
        atomic_write_json(
            run_dir / "progress.json",
            progress_payload(
                selected,
                checkpoints,
                active_wall_seconds=previous_wall + time.perf_counter() - started,
            ),
        )
    wall = previous_wall + time.perf_counter() - started
    records = [
        checkpoints[i]["pipeline_record"]
        for i in range(len(documents))
        if checkpoints[i]["status"] == "ok"
    ]
    triples = [t for record in records for t in record["final_triples"]]
    for name, values in (("kg", triples), ("pipeline_records", records)):
        atomic_write_json(run_dir / f"{name}.json", values)
        atomic_write_jsonl(run_dir / f"{name}.jsonl", values)
    atomic_write_jsonl(
        run_dir / "sample_results.jsonl",
        (
            {
                "document_id": d.document_id,
                "status": checkpoints[i]["status"],
                "error": checkpoints[i]["error"],
                "predicted_count": len(checkpoints[i]["final_triples"]),
                "elapsed_seconds": checkpoints[i]["elapsed_seconds"],
            }
            for i, d in enumerate(documents)
        ),
    )
    summary = {
        "method": runtime["method"],
        "dataset": runtime["dataset"],
        "runtime_config": dict(runtime),
        "samples_requested": len(documents),
        "samples_completed": len(records),
        "samples_failed": len(documents) - len(records),
        "kg_triples": len(triples),
        "active_wall_seconds": round(wall, 6),
        "inference_reads_gold": False,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(
        run_dir / "progress.json",
        progress_payload(selected, checkpoints, finalized=True, active_wall_seconds=wall),
    )
    return summary
