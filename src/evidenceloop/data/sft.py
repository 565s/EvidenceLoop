from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from evidenceloop.validation import predicate_validation_error

VALID_ACTIONS = ("KEEP", "DELETE", "FIX", "REWRITE")


class CriticSFTDataError(ValueError):
    """Raised when a Critic SFT record violates the training contract."""


def _require_string(value, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise CriticSFTDataError(f"{context}: expected a string")
    if not allow_empty and not value.strip():
        raise CriticSFTDataError(f"{context}: value must not be empty")
    return value


def _validate_triple(value, context: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise CriticSFTDataError(f"{context}: expected a triple object")
    expected = {"subject", "predicate", "object"}
    if set(value) != expected:
        raise CriticSFTDataError(
            f"{context}: expected exactly {sorted(expected)}, got {sorted(value)}"
        )
    triple = {
        field: _require_string(value.get(field), f"{context}.{field}").strip()
        for field in ("subject", "predicate", "object")
    }
    predicate_error = predicate_validation_error(triple["predicate"])
    if predicate_error:
        raise CriticSFTDataError(f"{context}.predicate: {predicate_error}")
    return triple


def validate_critic_sft_record(record, context: str = "record") -> Dict[str, int]:
    if not isinstance(record, Mapping):
        raise CriticSFTDataError(f"{context}: expected a JSON object")

    _require_string(record.get("document_id"), f"{context}.document_id")
    source_text = _require_string(record.get("source_text"), f"{context}.source_text")
    stage = _require_string(record.get("review_stage"), f"{context}.review_stage")
    if stage not in {"initial_review", "final_validation"}:
        raise CriticSFTDataError(f"{context}.review_stage: unsupported stage {stage!r}")

    triples_info = record.get("triples_info")
    if not isinstance(triples_info, list) or not triples_info:
        raise CriticSFTDataError(f"{context}.triples_info: expected a non-empty list")

    target = record.get("target")
    if not isinstance(target, Mapping):
        raise CriticSFTDataError(f"{context}.target: expected an object")
    results = target.get("results")
    if not isinstance(results, list) or len(results) != len(triples_info):
        raise CriticSFTDataError(f"{context}.target.results: expected {len(triples_info)} results")

    actions = Counter()
    expected_indices = list(range(len(triples_info)))
    input_by_idx = {}
    for position, item in enumerate(triples_info):
        item_context = f"{context}.triples_info[{position}]"
        if not isinstance(item, Mapping):
            raise CriticSFTDataError(f"{item_context}: expected an object")
        index = item.get("triple_idx")
        if index != position:
            raise CriticSFTDataError(
                f"{item_context}.triple_idx: expected {position}, got {index!r}"
            )
        _require_string(item.get("origin"), f"{item_context}.origin")
        input_by_idx[index] = _validate_triple(item.get("triple"), f"{item_context}.triple")
        generation_type = item.get("generation_type")
        if generation_type is not None and generation_type not in VALID_ACTIONS:
            raise CriticSFTDataError(
                f"{item_context}.generation_type: invalid value {generation_type!r}"
            )

    result_indices = []
    for position, result in enumerate(results):
        result_context = f"{context}.target.results[{position}]"
        if not isinstance(result, Mapping):
            raise CriticSFTDataError(f"{result_context}: expected an object")
        index = result.get("triple_idx")
        if not isinstance(index, int) or index not in input_by_idx:
            raise CriticSFTDataError(f"{result_context}.triple_idx: invalid index {index!r}")
        result_indices.append(index)

        action = _require_string(result.get("action"), f"{result_context}.action").upper()
        if action not in VALID_ACTIONS:
            raise CriticSFTDataError(f"{result_context}.action: invalid action {action!r}")
        actions[action] += 1

        reason = _require_string(result.get("reason"), f"{result_context}.reason")
        if len(reason.split()) > 30:
            raise CriticSFTDataError(f"{result_context}.reason: exceeds 30 words")
        evidence = _require_string(
            result.get("evidence"),
            f"{result_context}.evidence",
            allow_empty=action == "DELETE",
        )
        if evidence and evidence not in source_text:
            raise CriticSFTDataError(f"{result_context}.evidence: value is not a source substring")

        output_triple = result.get("triple")
        if action in {"FIX", "REWRITE"}:
            _validate_triple(output_triple, f"{result_context}.triple")
        elif output_triple is not None:
            raise CriticSFTDataError(f"{result_context}.triple: must be null for action {action}")

        declared_type = triples_info[index].get("generation_type")
        if declared_type is not None and declared_type != action:
            raise CriticSFTDataError(
                f"{result_context}.action: does not match generation_type {declared_type!r}"
            )

    if sorted(result_indices) != expected_indices:
        raise CriticSFTDataError(
            f"{context}.target.results: indices must be exactly {expected_indices}"
        )
    return dict(actions)


def public_triples_info(record: Mapping) -> List[Dict]:
    """Return only fields that are available to the Critic at inference time."""
    return [
        {
            "triple_idx": item["triple_idx"],
            "origin": item["origin"],
            "triple": dict(item["triple"]),
        }
        for item in record["triples_info"]
    ]


def format_critic_sft_parts(
    record: Mapping,
    system_prompt: str,
    call_prompt_template: str,
) -> Dict[str, str]:
    validate_critic_sft_record(record)
    user_prompt = call_prompt_template.format(
        review_stage=record["review_stage"],
        source_text=record["source_text"],
        triples_info=json.dumps(public_triples_info(record), ensure_ascii=False, indent=2),
    )
    return {
        "system": system_prompt,
        "user": user_prompt,
        "assistant": json.dumps(record["target"], ensure_ascii=False, separators=(",", ":")),
    }


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CriticSFTDataError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise CriticSFTDataError(f"{path}:{line_number}: expected a JSON object")
            yield value


def validate_critic_sft_jsonl(path: Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    action_counts = Counter()
    records = 0
    candidates = 0
    document_ids = set()
    for records, record in enumerate(iter_jsonl(path), start=1):
        location = f"{path}:{records}"
        counts = validate_critic_sft_record(record, location)
        action_counts.update(counts)
        candidates += sum(counts.values())
        document_id = record["document_id"]
        if document_id in document_ids:
            raise CriticSFTDataError(f"{location}: duplicate document_id {document_id!r}")
        document_ids.add(document_id)
    if not records:
        raise CriticSFTDataError(f"{path}: no records found")
    return {
        "path": str(path),
        "records": records,
        "candidates": candidates,
        "documents": len(document_ids),
        "action_counts": dict(action_counts),
    }


def assert_disjoint_document_ids(paths: Sequence[Path]) -> None:
    owners = {}
    for path in paths:
        for line_number, record in enumerate(iter_jsonl(Path(path)), start=1):
            document_id = record.get("document_id")
            previous = owners.get(document_id)
            if previous is not None:
                raise CriticSFTDataError(
                    f"{path}:{line_number}: document_id {document_id!r} also appears in {previous}"
                )
            owners[document_id] = str(path)
