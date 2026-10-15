from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.builders.legacy_phase1 import (  # noqa: E402
    DatasetBuildError,
    article_entities,
    default_rebel_zip,
    file_digest,
    iter_zip_jsonl,
    make_delete,
    make_fix,
    make_rewrite,
    normalized,
    result_item,
    triple_key,
    valid_gold_triples,
)
from evidenceloop.data.sft import (  # noqa: E402
    VALID_ACTIONS,
    assert_disjoint_document_ids,
    validate_critic_sft_jsonl,
    validate_critic_sft_record,
)

ROOT = Path.cwd()


DEFAULT_OUTPUT_DIR = ROOT / "data" / "critic_sft" / "rebel_phase2"
EXPECTED_ZIP_MEMBERS = {"en_train.jsonl", "en_val.jsonl", "en_test.jsonl"}
SOURCE_MEMBERS = {
    "train": "en_train.jsonl",
    "validation": "en_val.jsonl",
    "test": "en_test.jsonl",
}
ACTION_RATIOS = {
    "KEEP": 0.50,
    "DELETE": 0.15,
    "FIX": 0.20,
    "REWRITE": 0.15,
}
MAX_CANDIDATES = 4
RATIO_TOLERANCE = 0.005
NMI_THRESHOLD = 0.02


class ActionScheduler:
    """Keep cumulative action counts close to the requested global ratios."""

    def __init__(self, ratios: Mapping[str, float]):
        if set(ratios) != set(VALID_ACTIONS):
            raise ValueError("action ratios must define every Critic action")
        if abs(sum(ratios.values()) - 1.0) > 1e-9:
            raise ValueError("action ratios must sum to 1")
        self.ratios = dict(ratios)
        self.counts: Counter = Counter()
        self.total = 0

    def ranked_actions(self, rng: random.Random) -> List[str]:
        projected_total = self.total + 1
        tie_breakers = {action: rng.random() for action in VALID_ACTIONS}
        return sorted(
            VALID_ACTIONS,
            key=lambda action: (
                self.ratios[action] * projected_total - self.counts[action],
                tie_breakers[action],
            ),
            reverse=True,
        )

    def commit(self, action: str) -> None:
        self.counts[action] += 1
        self.total += 1


def _candidate_for_action(
    action: str,
    base: Mapping,
    text: str,
    gold_items: Sequence[Mapping],
    gold_keys: set,
    entities: Sequence[str],
    predicates: Sequence[str],
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    used_candidate_keys: set,
    rng: random.Random,
) -> Optional[Tuple[Dict, Mapping, Optional[str]]]:
    for _ in range(12):
        corruption = None
        if action == "KEEP":
            candidate = dict(base["triple"])
        elif action == "DELETE":
            candidate = make_delete(
                base,
                gold_keys,
                entities,
                predicates,
                fallback_entities,
                fallback_predicates,
                rng,
            )
            corruption = "predicate+object"
        elif action == "FIX":
            value = make_fix(
                base,
                gold_keys,
                entities,
                predicates,
                fallback_entities,
                fallback_predicates,
                rng,
            )
            if value is None:
                candidate = None
            else:
                candidate, corruption = value
        elif action == "REWRITE":
            value = make_rewrite(text, [base], gold_keys, rng)
            if value is None:
                candidate = None
            else:
                _, candidate, corruption = value
        else:  # pragma: no cover - guarded by ActionScheduler
            raise ValueError(f"unknown action: {action}")

        if candidate is None:
            continue
        key = triple_key(candidate)
        if key in used_candidate_keys:
            continue
        if action != "KEEP" and key in gold_keys:
            continue
        return dict(candidate), base, corruption
    return None


def validate_phase2_record(record: Mapping, context: str = "record") -> None:
    validate_critic_sft_record(record, context)
    triples_info = record["triples_info"]
    if not 1 <= len(triples_info) <= MAX_CANDIDATES:
        raise DatasetBuildError(f"{context}: candidate count must be between 1 and 4")
    keys = [triple_key(item["triple"]) for item in triples_info]
    if len(keys) != len(set(keys)):
        raise DatasetBuildError(f"{context}: duplicate candidate triple")

    results = {item["triple_idx"]: item for item in record["target"]["results"]}
    for item in triples_info:
        index = item["triple_idx"]
        result = results[index]
        action = result["action"]
        if action == "DELETE":
            if result["evidence"]:
                raise DatasetBuildError(f"{context}: DELETE evidence must be empty")
            continue
        expected = item["triple"] if action == "KEEP" else result["triple"]
        evidence = result["evidence"]
        if expected["subject"] not in evidence or expected["object"] not in evidence:
            raise DatasetBuildError(
                f"{context}: evidence does not cover corrected subject and object"
            )


def build_phase2_record(
    article: Mapping,
    split_name: str,
    sequence: int,
    rng: random.Random,
    scheduler: ActionScheduler,
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
) -> Tuple[Optional[Dict], str]:
    text = str(article.get("text") or "").strip()
    if not text:
        return None, "empty_text"
    gold_items = valid_gold_triples(article)
    if not gold_items:
        return None, "no_valid_gold"

    batch_size = min(len(gold_items), MAX_CANDIDATES)
    gold_keys = {triple_key(item["triple"]) for item in gold_items}
    entities = article_entities(article)
    predicates = list(dict.fromkeys(item["triple"]["predicate"] for item in gold_items))
    unused_bases = list(gold_items)
    used_candidate_keys = set()
    generated = []

    for _ in range(batch_size):
        built = None
        for action in scheduler.ranked_actions(rng):
            base_order = list(unused_bases)
            rng.shuffle(base_order)
            for base in base_order:
                value = _candidate_for_action(
                    action,
                    base,
                    text,
                    gold_items,
                    gold_keys,
                    entities,
                    predicates,
                    fallback_entities,
                    fallback_predicates,
                    used_candidate_keys,
                    rng,
                )
                if value is not None:
                    candidate, selected_base, corruption = value
                    built = (action, candidate, selected_base, corruption)
                    break
            if built is not None:
                break

        if built is None:
            return None, "cannot_fill_batch"
        action, candidate, base, corruption = built
        scheduler.commit(action)
        unused_bases.remove(base)
        used_candidate_keys.add(triple_key(candidate))
        generated.append((action, candidate, base, corruption))

    rng.shuffle(generated)
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
    validate_phase2_record(record, record["document_id"])
    return record, ""


def normalized_mutual_information(position_action: Mapping[int, Mapping[str, int]]) -> float:
    joint = {
        (int(position), action): count
        for position, actions in position_action.items()
        for action, count in actions.items()
        if count
    }
    total = sum(joint.values())
    if not total:
        return 0.0
    position_counts = Counter()
    action_counts = Counter()
    for (position, action), count in joint.items():
        position_counts[position] += count
        action_counts[action] += count

    mutual_information = 0.0
    for (position, action), count in joint.items():
        p_xy = count / total
        p_x = position_counts[position] / total
        p_y = action_counts[action] / total
        mutual_information += p_xy * math.log(p_xy / (p_x * p_y))
    h_position = -sum(
        (count / total) * math.log(count / total) for count in position_counts.values()
    )
    h_action = -sum((count / total) * math.log(count / total) for count in action_counts.values())
    denominator = math.sqrt(h_position * h_action)
    return mutual_information / denominator if denominator else 0.0


def _validate_split_distribution(stats: Mapping) -> None:
    if stats["records"] < 1_000:
        return
    if set(stats["batch_size_counts"]) != {"1", "2", "3", "4"}:
        raise DatasetBuildError("full split does not contain every batch size from 1 to 4")
    if stats["max_abs_action_ratio_error"] > RATIO_TOLERANCE:
        raise DatasetBuildError(
            "action ratio error exceeds "
            f"{RATIO_TOLERANCE:.3f}: {stats['max_abs_action_ratio_error']:.6f}"
        )
    if stats["position_action_nmi"] >= NMI_THRESHOLD:
        raise DatasetBuildError(
            f"position/action NMI {stats['position_action_nmi']:.6f} exceeds {NMI_THRESHOLD}"
        )


def build_split(
    archive: zipfile.ZipFile,
    member: str,
    split_name: str,
    target_count: int,
    output_path: Path,
    seed: int,
    seen_source_ids: set,
) -> Dict:
    rng = random.Random(f"{seed}:{split_name}:phase2")
    scheduler = ActionScheduler(ACTION_RATIOS)
    fallback_entities: List[str] = []
    fallback_predicates: List[str] = []
    fallback_entity_seen = set()
    fallback_predicate_seen = set()
    rejected = Counter()
    batch_size_counts = Counter()
    corruption_counts = Counter()
    position_action = defaultdict(Counter)
    scanned = 0
    written = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for sequence, article in enumerate(iter_zip_jsonl(archive, member)):
            scanned += 1
            source_id = str(article.get("docid") or article.get("uri") or sequence).strip()
            if source_id in seen_source_ids:
                rejected["duplicate_document_id"] += 1
                continue

            record, error = build_phase2_record(
                article,
                split_name,
                sequence,
                rng,
                scheduler,
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

            seen_source_ids.add(source_id)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            batch_size_counts[len(record["triples_info"])] += 1
            for item, result in zip(record["triples_info"], record["target"]["results"]):
                position_action[item["triple_idx"]][result["action"]] += 1
                corruption_counts[item.get("corruption", "none")] += 1
            if written >= target_count:
                break

    if written != target_count:
        raise DatasetBuildError(
            f"{split_name}: generated {written}/{target_count} records after scanning {scanned} rows"
        )

    candidates = scheduler.total
    action_counts = {action: scheduler.counts[action] for action in VALID_ACTIONS}
    action_ratios = {action: action_counts[action] / candidates for action in VALID_ACTIONS}
    ratio_errors = {
        action: abs(action_ratios[action] - ACTION_RATIOS[action]) for action in VALID_ACTIONS
    }
    position_action_dict = {
        str(position): {action: counts[action] for action in VALID_ACTIONS}
        for position, counts in sorted(position_action.items())
    }
    stats = {
        "member": member,
        "records": written,
        "candidates": candidates,
        "rows_scanned": scanned,
        "batch_size_counts": {
            str(size): batch_size_counts[size] for size in range(1, MAX_CANDIDATES + 1)
        },
        "action_counts": action_counts,
        "action_ratios": action_ratios,
        "action_ratio_errors": ratio_errors,
        "max_abs_action_ratio_error": max(ratio_errors.values()),
        "position_action_counts": position_action_dict,
        "position_action_nmi": normalized_mutual_information(position_action),
        "corruption_counts": dict(corruption_counts),
        "rejected": dict(rejected),
    }
    _validate_split_distribution(stats)
    return stats


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

    output_names = ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json")
    expected_outputs = [output_dir / name for name in output_names]
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
            for split_name, target_count in (
                ("train", train_docs),
                ("validation", validation_docs),
                ("test", test_docs),
            ):
                split_stats[split_name] = build_split(
                    archive,
                    SOURCE_MEMBERS[split_name],
                    split_name,
                    target_count,
                    temp_dir / f"{split_name}.jsonl",
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
            [temp_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
        )
        files = {
            f"{split}.jsonl": file_digest(temp_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        metadata = {
            "dataset": "Babelscape/rebel-dataset",
            "phase": 2,
            "source_zip": str(source_zip),
            "source_members": dict(SOURCE_MEMBERS),
            "uses_graphjudge_test_files": False,
            "seed": seed,
            "candidate_range": [1, MAX_CANDIDATES],
            "batch_size_rule": "min(valid_gold_triples, 4)",
            "target_action_ratios": dict(ACTION_RATIOS),
            "ratio_tolerance": RATIO_TOLERANCE,
            "position_action_nmi_threshold": NMI_THRESHOLD,
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
        for name in output_names:
            os.replace(temp_dir / name, output_dir / name)
        return metadata
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build variable-batch phase-two Critic SFT data from REBEL."
    )
    parser.add_argument("--source-zip", type=Path, default=default_rebel_zip())
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
