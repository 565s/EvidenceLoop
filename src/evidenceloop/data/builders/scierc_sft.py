from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.builders.legacy_phase1 import (  # noqa: E402
    DatasetBuildError,
    file_digest,
    make_delete,
    make_rewrite,
    normalized,
    result_item,
)
from evidenceloop.data.builders.rebel_sft import (  # noqa: E402
    ACTION_RATIOS,
    MAX_CANDIDATES,
    ActionScheduler,
    normalized_mutual_information,
    validate_phase2_record,
)
from evidenceloop.data.scierc import (  # noqa: E402
    SCIERC_GUIDELINE_URL,
    SCIERC_PROJECT_URL,
    SCIERC_SOURCE_URL,
    SciERCError,
    load_documents,
    load_relation_map,
    predicate_aware_triple_key,
    symmetric_predicates,
)
from evidenceloop.data.sft import (  # noqa: E402
    VALID_ACTIONS,
    assert_disjoint_document_ids,
    validate_critic_sft_jsonl,
)
from evidenceloop.validation import predicate_validation_error  # noqa: E402

ROOT = Path.cwd()


DEFAULT_OUTPUT_DIR = ROOT / "data" / "critic_sft" / "scierc_phase2"
DEFAULT_RELATION_MAP = __import__(
    "evidenceloop.paths", fromlist=["relation_map_path"]
).relation_map_path("scierc_relation_info.json")
OUTPUT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json")
SPLITS = ("train", "validation", "test")
FILTER_COUNT_KEYS = (
    "invalid_ner_fields",
    "invalid_entity_mentions",
    "invalid_entity_spans",
    "empty_entity_surfaces",
    "invalid_coreference_fields",
    "invalid_relation_fields",
    "invalid_relation_structures",
    "unknown_relations",
    "invalid_relation_spans",
    "cross_sentence_or_misaligned_relations",
    "relation_endpoints_missing_ner",
    "empty_triple_fields",
    "invalid_evidence",
    "duplicate_triples",
)


def _stats_payload(stats: Mapping[str, int]) -> Dict:
    relation_distribution = {}
    entity_type_distribution = {}
    counts = {}
    for key, value in sorted(stats.items()):
        if key.startswith("relation_label:"):
            relation_distribution[key.split(":", 1)[1]] = value
        elif key.startswith("entity_type:"):
            entity_type_distribution[key.split(":", 1)[1]] = value
        else:
            counts[key] = value
    for key in FILTER_COUNT_KEYS:
        counts.setdefault(key, 0)
    return {
        "counts": counts,
        "relation_distribution": relation_distribution,
        "entity_type_distribution": entity_type_distribution,
    }


def _make_fix(
    base: Mapping,
    gold_keys: set,
    entities: Sequence[str],
    predicates: Sequence[str],
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    symmetric: Sequence[str],
    rng: random.Random,
) -> Optional[Tuple[Dict, str]]:
    original = base["triple"]
    operations = ["subject", "object", "predicate"]
    if normalized(original["predicate"]) not in {normalized(value) for value in symmetric}:
        operations.append("direction")
    rng.shuffle(operations)
    for operation in operations:
        candidate = dict(original)
        if operation == "direction":
            candidate["subject"], candidate["object"] = (
                candidate["object"],
                candidate["subject"],
            )
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
            pool = [
                value for value in entities if normalized(value) != normalized(original[operation])
            ]
            if not pool:
                pool = [
                    value
                    for value in fallback_entities
                    if normalized(value) != normalized(original[operation])
                ]
            if not pool:
                continue
            candidate[operation] = rng.choice(pool)
        if (
            not predicate_validation_error(candidate["predicate"])
            and predicate_aware_triple_key(candidate, symmetric) not in gold_keys
        ):
            return candidate, operation
    return None


def _candidate_for_action(
    action: str,
    base: Mapping,
    text: str,
    gold_keys: set,
    entities: Sequence[str],
    predicates: Sequence[str],
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    symmetric: Sequence[str],
    used_candidate_keys: set,
    rng: random.Random,
):
    for _ in range(16):
        corruption = None
        if action == "KEEP":
            candidate = dict(base["triple"])
        elif action == "DELETE":
            candidate = make_delete(
                base,
                set(),
                entities,
                predicates,
                fallback_entities,
                fallback_predicates,
                rng,
            )
            corruption = "predicate+object"
        elif action == "FIX":
            value = _make_fix(
                base,
                gold_keys,
                entities,
                predicates,
                fallback_entities,
                fallback_predicates,
                symmetric,
                rng,
            )
            candidate, corruption = value if value is not None else (None, None)
        elif action == "REWRITE":
            value = make_rewrite(text, [base], set(), rng)
            if value is None:
                candidate = None
            else:
                _, candidate, corruption = value
        else:  # pragma: no cover
            raise ValueError(f"unknown action: {action}")
        if candidate is None:
            continue
        key = predicate_aware_triple_key(candidate, symmetric)
        if key in used_candidate_keys or (action != "KEEP" and key in gold_keys):
            continue
        return dict(candidate), base, corruption
    return None


def _record_for_batch(
    document: Mapping,
    batch_index: int,
    bases: Sequence[Mapping],
    scheduler: ActionScheduler,
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    symmetric: Sequence[str],
    rng: random.Random,
) -> Dict:
    gold_keys = {
        predicate_aware_triple_key(item["triple"], symmetric) for item in document["gold_items"]
    }
    used_candidate_keys = set()
    generated = []
    for base in bases:
        built = None
        for action in scheduler.ranked_actions(rng):
            value = _candidate_for_action(
                action,
                base,
                document["text"],
                gold_keys,
                document["entities"],
                document["predicates"],
                fallback_entities,
                fallback_predicates,
                symmetric,
                used_candidate_keys,
                rng,
            )
            if value is not None:
                built = (action, *value)
                break
        if built is None:
            raise DatasetBuildError(
                f"{document['document_id']}: cannot generate candidate for batch {batch_index}"
            )
        action, candidate, selected_base, corruption = built
        scheduler.commit(action)
        used_candidate_keys.add(predicate_aware_triple_key(candidate, symmetric))
        generated.append((action, candidate, selected_base, corruption))

    rng.shuffle(generated)
    triples_info = []
    results = []
    for triple_index, (action, candidate, base, corruption) in enumerate(generated):
        item = {
            "triple_idx": triple_index,
            "origin": "proposer",
            "triple": dict(candidate),
            "generation_type": action,
        }
        if corruption:
            item["corruption"] = corruption
        triples_info.append(item)
        corrected = dict(base["triple"]) if action in {"FIX", "REWRITE"} else None
        evidence = "" if action == "DELETE" else base["evidence"]
        results.append(result_item(triple_index, action, evidence, corrected))

    record = {
        "document_id": f"{document['document_id']}_batch_{batch_index:03d}",
        "source_document_id": document["document_id"],
        "source_text": document["text"],
        "review_stage": "initial_review",
        "triples_info": triples_info,
        "target": {"results": results},
    }
    validate_phase2_record(record, record["document_id"])
    keys = [predicate_aware_triple_key(item["triple"], symmetric) for item in triples_info]
    if len(keys) != len(set(keys)):
        raise DatasetBuildError(f"{record['document_id']}: duplicate predicate-aware candidate")
    return record


def _build_split(
    documents: Sequence[Mapping],
    split_name: str,
    output_path: Path,
    seed: int,
    symmetric: Sequence[str],
) -> Dict:
    rng = random.Random(f"{seed}:{split_name}:scierc_phase2")
    scheduler = ActionScheduler(ACTION_RATIOS)
    fallback_entities = list(
        dict.fromkeys(entity for document in documents for entity in document["entities"])
    )
    fallback_predicates = list(
        dict.fromkeys(predicate for document in documents for predicate in document["predicates"])
    )
    records = []
    batch_size_counts = Counter()
    position_action = defaultdict(Counter)
    corruption_counts = Counter()
    skipped_documents = 0
    for document in documents:
        bases = list(document["gold_items"])
        if not bases:
            skipped_documents += 1
            continue
        rng.shuffle(bases)
        for batch_index, start in enumerate(range(0, len(bases), MAX_CANDIDATES)):
            record = _record_for_batch(
                document,
                batch_index,
                bases[start : start + MAX_CANDIDATES],
                scheduler,
                fallback_entities,
                fallback_predicates,
                symmetric,
                rng,
            )
            records.append(record)
            batch_size_counts[len(record["triples_info"])] += 1
            for item, result in zip(record["triples_info"], record["target"]["results"]):
                position_action[item["triple_idx"]][result["action"]] += 1
                corruption_counts[item.get("corruption", "none")] += 1
    if not records:
        raise DatasetBuildError(f"{split_name}: no SFT records were generated")
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    candidates = scheduler.total
    action_counts = {action: scheduler.counts[action] for action in VALID_ACTIONS}
    action_ratios = {action: action_counts[action] / candidates for action in VALID_ACTIONS}
    return {
        "source_documents": len(documents),
        "skipped_source_documents": skipped_documents,
        "records": len(records),
        "candidates": candidates,
        "batch_size_counts": {
            str(size): batch_size_counts[size] for size in range(1, MAX_CANDIDATES + 1)
        },
        "action_counts": action_counts,
        "action_ratios": action_ratios,
        "max_abs_action_ratio_error": max(
            abs(action_ratios[action] - ACTION_RATIOS[action]) for action in VALID_ACTIONS
        ),
        "position_action_nmi": normalized_mutual_information(position_action),
        "corruption_counts": dict(corruption_counts),
    }


def build_dataset(
    train_file: Path,
    validation_file: Path,
    test_file: Path,
    relation_map_file: Path = DEFAULT_RELATION_MAP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict:
    source_files = {
        "train": Path(train_file).expanduser().resolve(),
        "validation": Path(validation_file).expanduser().resolve(),
        "test": Path(test_file).expanduser().resolve(),
    }
    relation_map_file = Path(relation_map_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (*source_files.values(), relation_map_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    relation_map = load_relation_map(relation_map_file)
    symmetric = symmetric_predicates(relation_map)
    documents = {}
    conversion = {}
    for split_name in SPLITS:
        documents[split_name], stats = load_documents(
            source_files[split_name], split_name, relation_map
        )
        conversion[split_name] = _stats_payload(stats)
    split_id_sets = {
        split_name: {document["doc_key"] for document in documents[split_name]}
        for split_name in SPLITS
    }
    if any(
        split_id_sets[left] & split_id_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise DatasetBuildError("SciERC doc_key splits overlap")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".scierc_sft_build_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        split_stats = {
            split_name: _build_split(
                documents[split_name],
                split_name,
                temp_dir / f"{split_name}.jsonl",
                seed,
                symmetric,
            )
            for split_name in SPLITS
        }
        validation = {
            split_name: validate_critic_sft_jsonl(temp_dir / f"{split_name}.jsonl")
            for split_name in SPLITS
        }
        for split_name, summary in validation.items():
            summary["path"] = str(output_dir / f"{split_name}.jsonl")
        assert_disjoint_document_ids([temp_dir / f"{split_name}.jsonl" for split_name in SPLITS])
        files = {
            f"{split_name}.jsonl": file_digest(temp_dir / f"{split_name}.jsonl")
            for split_name in SPLITS
        }
        metadata = {
            "dataset": "SciERC",
            "phase": 2,
            "source_url": SCIERC_SOURCE_URL,
            "project_url": SCIERC_PROJECT_URL,
            "annotation_guideline_url": SCIERC_GUIDELINE_URL,
            "source_files": {
                split_name: {"path": str(path), **file_digest(path)}
                for split_name, path in source_files.items()
            },
            "relation_map": {
                "path": str(relation_map_file),
                "relations": len(relation_map),
                "mapping": dict(relation_map),
                **file_digest(relation_map_file),
            },
            "coreference_policy": "statistics only; no derived relation triples",
            "triple_deduplication": "case-folded subject/predicate/object; endpoints additionally sorted for symmetric predicates",
            "entity_order_mode": "predicate_aware",
            "symmetric_predicates": list(symmetric),
            "evidence_policy": "complete annotated relation sentence as a verbatim substring",
            "seed": seed,
            "candidate_range": [1, MAX_CANDIDATES],
            "batching": "all valid unique gold triples, shuffled per document and partitioned into batches of at most four",
            "target_action_ratios": dict(ACTION_RATIOS),
            "review_stage": "initial_review",
            "origin": "proposer",
            "conversion": conversion,
            "splits": split_stats,
            "validation": validation,
            "files": files,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_FILES:
            os.replace(temp_dir / name, output_dir / name)
        return metadata
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build variable-batch Critic SFT data from official SciERC processed JSONL."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--relation-map", type=Path, default=DEFAULT_RELATION_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        metadata = build_dataset(
            args.train_file,
            args.validation_file,
            args.test_file,
            args.relation_map,
            args.output_dir,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (
        DatasetBuildError,
        SciERCError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
