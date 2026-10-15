from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.sft import (  # noqa: E402
    VALID_ACTIONS,
    assert_disjoint_document_ids,
    validate_critic_sft_jsonl,
    validate_critic_sft_record,
)
from evidenceloop.validation import clean_predicate, predicate_validation_error  # noqa: E402

ROOT = Path.cwd()


DEFAULT_OUTPUT_DIR = ROOT / "data" / "critic_sft" / "rebel_phase1"
EXPECTED_ZIP_MEMBERS = {"en_train.jsonl", "en_val.jsonl", "en_test.jsonl"}
TOKEN_PATTERN = re.compile(r"\S+")
TERMINATORS = ".!?\n"


class DatasetBuildError(RuntimeError):
    pass


def default_rebel_zip() -> Optional[Path]:
    configured = os.getenv("REBEL_DATASET_ZIP")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate

    hub_root = (
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--Babelscape--rebel-dataset"
    )
    matches = sorted(hub_root.glob("snapshots/*/rebel_dataset.zip"))
    return matches[-1] if matches else None


def normalized(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def triple_key(triple: Mapping[str, str]) -> Tuple[str, str, str]:
    return (
        normalized(triple.get("subject")),
        normalized(triple.get("predicate")),
        normalized(triple.get("object")),
    )


def entity_value(entity: Mapping, text: str) -> Optional[Dict]:
    boundaries = entity.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 2:
        return None
    start, end = boundaries
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not (0 <= start < end <= len(text))
    ):
        return None
    surface = str(entity.get("surfaceform") or "").strip()
    source_surface = text[start:end].strip()
    if not source_surface:
        return None
    if not surface or normalized(surface) != normalized(source_surface):
        surface = source_surface
    return {"surface": surface, "start": start, "end": end}


def evidence_span(text: str, subject: Mapping, obj: Mapping) -> str:
    left = min(subject["start"], obj["start"])
    right = max(subject["end"], obj["end"])
    start = 0
    for marker in TERMINATORS:
        position = text.rfind(marker, 0, left)
        if position >= start:
            start = position + 1
    while start < len(text) and text[start].isspace():
        start += 1

    end_positions = [
        position
        for marker in TERMINATORS
        for position in [text.find(marker, right)]
        if position >= 0
    ]
    end = min(end_positions) + 1 if end_positions else len(text)
    evidence = text[start:end].strip()
    if not evidence or evidence not in text:
        return ""
    if text[subject["start"] : subject["end"]].strip() not in evidence:
        return ""
    if text[obj["start"] : obj["end"]].strip() not in evidence:
        return ""
    return evidence


def valid_gold_triples(article: Mapping) -> List[Dict]:
    text = str(article.get("text") or "")
    output = []
    seen = set()
    for raw in article.get("triples") or []:
        if not isinstance(raw, Mapping):
            continue
        subject = entity_value(raw.get("subject") or {}, text)
        obj = entity_value(raw.get("object") or {}, text)
        predicate_data = raw.get("predicate") or {}
        predicate = clean_predicate(predicate_data.get("surfaceform"))
        if not subject or not obj or predicate_validation_error(predicate):
            continue
        triple = {
            "subject": subject["surface"],
            "predicate": predicate,
            "object": obj["surface"],
        }
        key = triple_key(triple)
        evidence = evidence_span(text, subject, obj)
        if key in seen or not evidence:
            continue
        seen.add(key)
        output.append(
            {
                "triple": triple,
                "subject_span": subject,
                "object_span": obj,
                "evidence": evidence,
            }
        )
    return output


def article_entities(article: Mapping) -> List[str]:
    text = str(article.get("text") or "")
    entities = []
    seen = set()
    for raw in article.get("entities") or []:
        value = entity_value(raw, text) if isinstance(raw, Mapping) else None
        if value and normalized(value["surface"]) not in seen:
            seen.add(normalized(value["surface"]))
            entities.append(value["surface"])
    return entities


def corrupt_mention(text: str, span: Mapping, rng: random.Random) -> Optional[str]:
    start, end = span["start"], span["end"]
    sentence_start = max(text.rfind(marker, 0, start) for marker in TERMINATORS) + 1
    following = [text.find(marker, end) for marker in TERMINATORS]
    following = [position for position in following if position >= 0]
    sentence_end = min(following) + 1 if following else len(text)
    tokens = [match.span() for match in TOKEN_PATTERN.finditer(text, sentence_start, sentence_end)]
    containing = [index for index, (a, b) in enumerate(tokens) if a <= start and end <= b]
    if not containing:
        overlapping = [index for index, (a, b) in enumerate(tokens) if a < end and b > start]
        if not overlapping:
            return None
        first, last = overlapping[0], overlapping[-1]
    else:
        first = last = containing[0]

    candidates = []
    if first > 0:
        candidates.append(text[tokens[first - 1][0] : end].strip())
    if last + 1 < len(tokens):
        candidates.append(text[start : tokens[last + 1][1]].strip())

    mention_tokens = list(TOKEN_PATTERN.finditer(text[start:end]))
    if len(mention_tokens) > 1:
        candidates.append(text[start + mention_tokens[1].start() : end].strip())
        candidates.append(text[start : start + mention_tokens[-2].end()].strip())

    original = normalized(text[start:end])
    candidates = [
        candidate
        for candidate in candidates
        if candidate and normalized(candidate) != original and "\n" not in candidate
    ]
    return rng.choice(candidates) if candidates else None


def result_item(index: int, action: str, evidence: str, corrected: Optional[Dict]) -> Dict:
    reasons = {
        "KEEP": "Supported by the source.",
        "DELETE": "Unsupported by the source.",
        "FIX": "Corrected to match the source evidence.",
        "REWRITE": "Normalized to match the source mention.",
    }
    return {
        "triple_idx": index,
        "action": action,
        "reason": reasons[action],
        "evidence": evidence,
        "triple": corrected,
    }


def make_delete(
    base: Mapping,
    gold_keys: set,
    entities: Sequence[str],
    predicates: Sequence[str],
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    rng: random.Random,
) -> Optional[Dict]:
    original = base["triple"]
    relation_pool = [
        value for value in predicates if normalized(value) != normalized(original["predicate"])
    ]
    object_pool = [
        value for value in entities if normalized(value) != normalized(original["object"])
    ]
    if not relation_pool:
        relation_pool = [
            value
            for value in fallback_predicates
            if normalized(value) != normalized(original["predicate"])
        ]
    if not object_pool:
        object_pool = [
            value
            for value in fallback_entities
            if normalized(value) != normalized(original["object"])
        ]
    rng.shuffle(relation_pool)
    rng.shuffle(object_pool)
    for predicate in relation_pool[:20]:
        if predicate_validation_error(predicate):
            continue
        for obj in object_pool[:20]:
            candidate = {
                "subject": original["subject"],
                "predicate": predicate,
                "object": obj,
            }
            if triple_key(candidate) not in gold_keys:
                return candidate
    return None


def make_fix(
    base: Mapping,
    gold_keys: set,
    entities: Sequence[str],
    predicates: Sequence[str],
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    rng: random.Random,
) -> Optional[Tuple[Dict, str]]:
    original = base["triple"]
    operations = ["subject", "object", "predicate", "direction"]
    rng.shuffle(operations)
    for operation in operations:
        candidate = dict(original)
        if operation == "direction":
            candidate["subject"], candidate["object"] = candidate["object"], candidate["subject"]
        elif operation == "predicate":
            pool = [
                value
                for value in predicates
                if normalized(value) != normalized(original["predicate"])
            ]
            if not pool:
                pool = [
                    value
                    for value in fallback_predicates
                    if normalized(value) != normalized(original["predicate"])
                ]
            if not pool:
                continue
            candidate["predicate"] = rng.choice(pool)
        else:
            original_value = original[operation]
            pool = [value for value in entities if normalized(value) != normalized(original_value)]
            if not pool:
                pool = [
                    value
                    for value in fallback_entities
                    if normalized(value) != normalized(original_value)
                ]
            if not pool:
                continue
            candidate[operation] = rng.choice(pool)
        if (
            not predicate_validation_error(candidate["predicate"])
            and triple_key(candidate) not in gold_keys
        ):
            return candidate, operation
    return None


def make_rewrite(
    text: str,
    gold_items: Sequence[Mapping],
    gold_keys: set,
    rng: random.Random,
) -> Optional[Tuple[Mapping, Dict, str]]:
    choices = list(gold_items)
    rng.shuffle(choices)
    for base in choices:
        slots = [("subject", base["subject_span"]), ("object", base["object_span"])]
        rng.shuffle(slots)
        for field, span in slots:
            corrupted = corrupt_mention(text, span, rng)
            if not corrupted:
                continue
            candidate = dict(base["triple"])
            candidate[field] = corrupted
            if triple_key(candidate) not in gold_keys:
                return base, candidate, field
    return None


def build_record(
    article: Mapping,
    split_name: str,
    sequence: int,
    rng: random.Random,
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
) -> Tuple[Optional[Dict], str]:
    text = str(article.get("text") or "").strip()
    if not text:
        return None, "empty_text"
    gold_items = valid_gold_triples(article)
    if not gold_items:
        return None, "no_valid_gold"
    gold_keys = {triple_key(item["triple"]) for item in gold_items}
    entities = article_entities(article)
    predicates = list(dict.fromkeys(item["triple"]["predicate"] for item in gold_items))

    keep_base = rng.choice(gold_items)
    delete_base = rng.choice(gold_items)
    delete_candidate = make_delete(
        delete_base,
        gold_keys,
        entities,
        predicates,
        fallback_entities,
        fallback_predicates,
        rng,
    )
    if delete_candidate is None:
        return None, "cannot_generate_delete"

    fix_base = rng.choice(gold_items)
    fix_value = make_fix(
        fix_base,
        gold_keys,
        entities,
        predicates,
        fallback_entities,
        fallback_predicates,
        rng,
    )
    if fix_value is None:
        return None, "cannot_generate_fix"
    fix_candidate, fix_operation = fix_value

    rewrite_value = make_rewrite(text, gold_items, gold_keys, rng)
    if rewrite_value is None:
        return None, "cannot_generate_rewrite"
    rewrite_base, rewrite_candidate, rewrite_operation = rewrite_value

    generated = [
        ("KEEP", keep_base["triple"], keep_base, None),
        ("DELETE", delete_candidate, delete_base, None),
        ("FIX", fix_candidate, fix_base, fix_operation),
        ("REWRITE", rewrite_candidate, rewrite_base, rewrite_operation),
    ]
    triples_info = []
    results = []
    for index, (action, candidate, base, corruption) in enumerate(generated):
        item = {
            "triple_idx": index,
            "origin": "proposer",
            "triple": candidate,
            "generation_type": action,
        }
        if corruption:
            item["corruption"] = corruption
        triples_info.append(item)
        corrected = dict(base["triple"]) if action in {"FIX", "REWRITE"} else None
        evidence = "" if action == "DELETE" else base["evidence"]
        results.append(result_item(index, action, evidence, corrected))

    raw_source_id = article.get("docid") or article.get("uri")
    source_id = str(raw_source_id).strip() if raw_source_id else f"{split_name}_{sequence:06d}"
    record = {
        "document_id": f"rebel_{source_id}",
        "source_text": text,
        "review_stage": "initial_review",
        "triples_info": triples_info,
        "target": {"results": results},
    }
    validate_critic_sft_record(record, record["document_id"])
    return record, ""


def iter_zip_jsonl(archive: zipfile.ZipFile, member: str) -> Iterable[Mapping]:
    with archive.open(member) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetBuildError(f"{member}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, Mapping):
                yield value


def build_split(
    archive: zipfile.ZipFile,
    member: str,
    split_name: str,
    target_count: int,
    output_path: Path,
    seed: int,
    seen_source_ids: set,
    skip_accepted: int = 0,
) -> Dict:
    rng = random.Random(f"{seed}:{split_name}")
    fallback_entities: List[str] = []
    fallback_predicates: List[str] = []
    fallback_entity_seen = set()
    fallback_predicate_seen = set()
    rejected = Counter()
    scanned = 0
    accepted_seen = 0
    written = 0
    action_counts = Counter()
    corruption_counts = Counter()

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for sequence, article in enumerate(iter_zip_jsonl(archive, member)):
            scanned += 1
            source_id = str(article.get("docid") or article.get("uri") or sequence).strip()
            if source_id in seen_source_ids:
                rejected["duplicate_document_id"] += 1
                continue

            record, error = build_record(
                article,
                split_name,
                sequence,
                rng,
                fallback_entities,
                fallback_predicates,
            )
            for entity in article_entities(article):
                key = normalized(entity)
                if key not in fallback_entity_seen:
                    fallback_entity_seen.add(key)
                    fallback_entities.append(entity)
            for item in valid_gold_triples(article):
                predicate = item["triple"]["predicate"]
                key = normalized(predicate)
                if key not in fallback_predicate_seen:
                    fallback_predicate_seen.add(key)
                    fallback_predicates.append(predicate)
            if record is None:
                rejected[error] += 1
                continue
            if accepted_seen < skip_accepted:
                accepted_seen += 1
                continue

            seen_source_ids.add(source_id)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            action_counts.update(result["action"] for result in record["target"]["results"])
            corruption_counts.update(
                item.get("corruption", "none") for item in record["triples_info"]
            )
            if written >= target_count:
                break

    if written != target_count:
        raise DatasetBuildError(
            f"{split_name}: generated {written}/{target_count} records after scanning {scanned} rows"
        )
    return {
        "member": member,
        "records": written,
        "candidates": written * len(VALID_ACTIONS),
        "rows_scanned": scanned,
        "accepted_rows_skipped": skip_accepted,
        "action_counts": dict(action_counts),
        "corruption_counts": dict(corruption_counts),
        "rejected": dict(rejected),
    }


def file_digest(path: Path) -> Dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def build_dataset(
    source_zip: Path,
    output_dir: Path,
    train_docs: int,
    validation_docs: int,
    test_docs: int,
    seed: int,
    overwrite: bool = False,
) -> Dict:
    source_zip = Path(source_zip).expanduser().resolve()
    output_dir = Path(output_dir).resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    for name, value in {
        "train_docs": train_docs,
        "validation_docs": validation_docs,
        "test_docs": test_docs,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    expected_outputs = [
        output_dir / name
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json")
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".critic_sft_build_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(source_zip) as archive:
            members = set(archive.namelist())
            missing = sorted(EXPECTED_ZIP_MEMBERS - members)
            if missing:
                raise DatasetBuildError(f"source ZIP is missing members: {missing}")

            seen_source_ids = set()
            split_stats = {}
            split_stats["train"] = build_split(
                archive,
                "en_train.jsonl",
                "train",
                train_docs,
                temp_dir / "train.jsonl",
                seed,
                seen_source_ids,
            )
            split_stats["validation"] = build_split(
                archive,
                "en_val.jsonl",
                "validation",
                validation_docs,
                temp_dir / "validation.jsonl",
                seed,
                seen_source_ids,
            )
            split_stats["test"] = build_split(
                archive,
                "en_val.jsonl",
                "test",
                test_docs,
                temp_dir / "test.jsonl",
                seed,
                seen_source_ids,
            )

        validation = {
            split: validate_critic_sft_jsonl(temp_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        for split, summary in validation.items():
            summary["path"] = str(output_dir / f"{split}.jsonl")
        assert_disjoint_document_ids(
            [
                temp_dir / "train.jsonl",
                temp_dir / "validation.jsonl",
                temp_dir / "test.jsonl",
            ]
        )
        files = {
            f"{split}.jsonl": file_digest(temp_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        metadata = {
            "dataset": "Babelscape/rebel-dataset",
            "source_zip": str(source_zip),
            "source_members": {
                "train": "en_train.jsonl",
                "validation": "en_val.jsonl",
                "test": "en_val.jsonl",
            },
            "uses_graphjudge_test_files": False,
            "seed": seed,
            "actions": list(VALID_ACTIONS),
            "review_stage": "initial_review",
            "origin": "proposer",
            "splits": split_stats,
            "validation": validation,
            "files": files,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"):
            os.replace(temp_dir / name, output_dir / name)
        return metadata
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    detected_zip = default_rebel_zip()
    parser = argparse.ArgumentParser(
        description="Build evidence-grounded KEEP/DELETE/FIX/REWRITE Critic SFT data from REBEL."
    )
    parser.add_argument("--source-zip", type=Path, default=detected_zip)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-docs", type=int, default=20_000)
    parser.add_argument("--validation-docs", type=int, default=2_000)
    parser.add_argument("--test-docs", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.source_zip is None:
        print(
            "error: REBEL ZIP was not found; pass --source-zip or set REBEL_DATASET_ZIP",
            file=sys.stderr,
        )
        return 2
    try:
        metadata = build_dataset(
            source_zip=args.source_zip,
            output_dir=args.output_dir,
            train_docs=args.train_docs,
            validation_docs=args.validation_docs,
            test_docs=args.test_docs,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (
        DatasetBuildError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
