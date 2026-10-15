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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.builders.legacy_phase1 import (  # noqa: E402
    DatasetBuildError,
    file_digest,
    normalized,
    result_item,
    triple_key,
)
from evidenceloop.data.builders.rebel_sft import (  # noqa: E402
    ACTION_RATIOS,
    ActionScheduler,
    _candidate_for_action,
    normalized_mutual_information,
    validate_phase2_record,
)
from evidenceloop.data.redocred import (  # noqa: E402
    REDOCRED_SOURCE_URLS,
    load_documents,
    load_relation_map,
)
from evidenceloop.data.sft import (  # noqa: E402
    VALID_ACTIONS,
    assert_disjoint_document_ids,
    validate_critic_sft_jsonl,
)

ROOT = Path.cwd()


VARIANTS = ("fixed1", "fixed2", "fixed4", "variable")
MAX_CANDIDATES = 4
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "critic_sft" / "re_docred_candidate_set_ablation"
DEFAULT_RELATION_MAP = __import__(
    "evidenceloop.paths", fromlist=["relation_map_path"]
).relation_map_path("docred_rel_info.json")
DEFAULT_TRAIN_FILE = (
    ROOT / "tmp" / "re_docred_source" / "Re-DocRED-main" / "data" / "train_revised.json"
)
DEFAULT_VALIDATION_FILE = (
    ROOT / "tmp" / "re_docred_source" / "Re-DocRED-main" / "data" / "dev_revised.json"
)
DEFAULT_TEST_FILE = (
    ROOT / "tmp" / "re_docred_source" / "Re-DocRED-main" / "data" / "test_revised.json"
)


def _sizes_for_variant(
    variant: str,
    count: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    variant = {"pointwise": "fixed1"}.get(variant, variant)
    if variant == "fixed1":
        return [1] * count
    if variant == "fixed2":
        sizes = [2] * (count // 2)
        if count % 2:
            sizes.append(1)
        return sizes
    if variant == "fixed4":
        sizes = [4] * (count // 4)
        if count % 4:
            sizes.append(count % 4)
        return sizes
    if variant != "variable":
        raise ValueError(f"unknown candidate-set variant: {variant}")
    if rng is None:
        raise ValueError("variable grouping requires a random generator")
    sizes = []
    remaining = count
    while remaining:
        size = rng.choice(list(range(1, min(MAX_CANDIDATES, remaining) + 1)))
        sizes.append(size)
        remaining -= size
    return sizes


def _fallback_pools(documents: Sequence[Mapping]) -> Tuple[List[str], List[str]]:
    entities = []
    predicates = []
    seen_entities = set()
    seen_predicates = set()
    for document in documents:
        for value in document["entities"]:
            key = normalized(value)
            if key and key not in seen_entities:
                seen_entities.add(key)
                entities.append(value)
        for value in document["predicates"]:
            key = normalized(value)
            if key and key not in seen_predicates:
                seen_predicates.add(key)
                predicates.append(value)
    return entities, predicates


def _candidate_manifest_for_split(
    documents: Sequence[Mapping],
    split_name: str,
    seed: int,
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(f"{seed}:{split_name}:redocred_ablation_candidates")
    scheduler = ActionScheduler(ACTION_RATIOS)
    manifest = []
    stats = Counter()
    eligible_document_ids = []
    dropped_tail_by_document = []
    excluded_documents = []
    position_action = defaultdict(Counter)
    corruption_counts = Counter()

    for document_index, document in enumerate(documents):
        # load_documents already performed the authoritative conversion. Reuse its
        # gold items and only apply the common multiple-of-four policy here.
        gold_items = list(document["gold_items"])
        if len(gold_items) < MAX_CANDIDATES:
            stats["excluded_documents_lt4"] += 1
            stats["excluded_gold_triples"] += len(gold_items)
            excluded_documents.append(
                {
                    "document_id": document["document_id"],
                    "valid_gold_triples": len(gold_items),
                }
            )
            continue

        usable_count = (len(gold_items) // MAX_CANDIDATES) * MAX_CANDIDATES
        stats["eligible_documents"] += 1
        eligible_document_ids.append(document["document_id"])
        stats["usable_gold_bases"] += usable_count
        stats["dropped_tail_gold_triples"] += len(gold_items) - usable_count
        if len(gold_items) != usable_count:
            dropped_tail_by_document.append(
                {
                    "document_id": document["document_id"],
                    "valid_gold_triples": len(gold_items),
                    "usable_gold_bases": usable_count,
                    "dropped_tail_count": len(gold_items) - usable_count,
                }
            )
        rng.shuffle(gold_items)
        gold_keys = {triple_key(item["triple"]) for item in gold_items}
        entities = list(document["entities"])
        predicates = list(document["predicates"])
        unused_bases = list(gold_items[:usable_count])
        used_candidate_keys = set()
        generated = []

        while unused_bases:
            built = None
            for action in scheduler.ranked_actions(rng):
                base_order = list(unused_bases)
                rng.shuffle(base_order)
                for base in base_order:
                    value = _candidate_for_action(
                        action,
                        base,
                        document["text"],
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
                        built = action, candidate, selected_base, corruption
                        break
                if built is not None:
                    break
            if built is None:
                raise DatasetBuildError(
                    f"{document['document_id']}: unable to create a shared candidate"
                )
            action, candidate, base, corruption = built
            scheduler.commit(action)
            unused_bases.remove(base)
            used_candidate_keys.add(triple_key(candidate))
            generated.append(
                {
                    "document_id": document["document_id"],
                    "source_document_index": document_index,
                    "source_text": document["text"],
                    "action": action,
                    "candidate": dict(candidate),
                    "gold": dict(base["triple"]),
                    "evidence": base["evidence"] if action != "DELETE" else "",
                    "corruption": corruption or "none",
                }
            )

        rng.shuffle(generated)
        for position, item in enumerate(generated):
            item["canonical_position"] = position
            position_action[position % MAX_CANDIDATES][item["action"]] += 1
            corruption_counts[item["corruption"]] += 1
            manifest.append(item)

    total = scheduler.total
    action_counts = {action: scheduler.counts[action] for action in VALID_ACTIONS}
    action_ratios = {
        action: action_counts[action] / total if total else 0.0 for action in VALID_ACTIONS
    }
    ratio_errors = {
        action: abs(action_ratios[action] - ACTION_RATIOS[action]) for action in VALID_ACTIONS
    }
    stats["candidates"] = total
    stats["eligible_document_ids"] = eligible_document_ids
    stats["dropped_tail_by_document"] = dropped_tail_by_document
    stats["excluded_documents"] = excluded_documents
    stats["action_counts"] = action_counts
    stats["action_ratios"] = action_ratios
    stats["max_abs_action_ratio_error"] = max(ratio_errors.values()) if ratio_errors else 0.0
    stats["position_action_nmi"] = normalized_mutual_information(position_action)
    stats["corruption_counts"] = dict(corruption_counts)
    return manifest, dict(stats)


def _record_from_items(
    items: Sequence[Mapping],
    variant: str,
    split_name: str,
    batch_index: int,
) -> Dict:
    if not 1 <= len(items) <= MAX_CANDIDATES:
        raise DatasetBuildError("candidate set must contain between 1 and 4 items")
    document_id = str(items[0]["document_id"])
    if any(str(item["document_id"]) != document_id for item in items):
        raise DatasetBuildError("candidate set crossed a document boundary")
    triples_info = []
    results = []
    for index, item in enumerate(items):
        action = str(item["action"])
        triples_info.append(
            {
                "triple_idx": index,
                "origin": "proposer",
                "triple": dict(item["candidate"]),
                "generation_type": action,
                "corruption": item["corruption"],
            }
        )
        corrected = dict(item["gold"]) if action in {"FIX", "REWRITE"} else None
        results.append(result_item(index, action, item["evidence"], corrected))

    record = {
        "document_id": (
            f"redocred_{split_name}_{int(items[0]['source_document_index']):06d}_"
            f"{variant}_batch_{batch_index:05d}"
        ),
        "source_document_id": document_id,
        "source_text": items[0]["source_text"],
        "review_stage": "initial_review",
        "triples_info": triples_info,
        "target": {"results": results},
    }
    validate_phase2_record(record, record["document_id"])
    return record


def _write_variant(
    variant: str,
    split_name: str,
    manifest: Sequence[Mapping],
    output_path: Path,
    seed: int,
) -> Dict:
    by_document = defaultdict(list)
    for item in manifest:
        by_document[item["document_id"]].append(item)

    records = []
    batch_size_counts = Counter()
    position_action = defaultdict(Counter)
    variable_rng = random.Random(f"{seed}:{split_name}:redocred_ablation:variable")
    for document_id, items in by_document.items():
        sizes = _sizes_for_variant(
            variant,
            len(items),
            variable_rng if variant == "variable" else None,
        )
        cursor = 0
        for batch_index, size in enumerate(sizes):
            group = items[cursor : cursor + size]
            cursor += size
            record = _record_from_items(group, variant, split_name, batch_index)
            records.append(record)
            batch_size_counts[size] += 1
            for item in record["target"]["results"]:
                position_action[item["triple_idx"]][item["action"]] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    total = sum(batch_size * count for batch_size, count in batch_size_counts.items())
    action_counts = Counter(
        item["action"] for record in records for item in record["target"]["results"]
    )
    action_ratios = {
        action: action_counts[action] / total if total else 0.0 for action in VALID_ACTIONS
    }
    ratio_errors = {
        action: abs(action_ratios[action] - ACTION_RATIOS[action]) for action in VALID_ACTIONS
    }
    stats = {
        "records": len(records),
        "candidates": total,
        "batch_size_counts": {
            str(size): batch_size_counts[size] for size in range(1, MAX_CANDIDATES + 1)
        },
        "action_counts": dict(action_counts),
        "action_ratios": action_ratios,
        "max_abs_action_ratio_error": max(ratio_errors.values()) if ratio_errors else 0.0,
        "position_action_nmi": normalized_mutual_information(position_action),
        "validation": validate_critic_sft_jsonl(output_path),
    }
    return stats


def build_dataset(
    train_file: Path,
    validation_file: Path,
    test_file: Path,
    relation_map_file: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict:
    source_paths = {
        "train": Path(train_file).expanduser().resolve(),
        "validation": Path(validation_file).expanduser().resolve(),
        "test": Path(test_file).expanduser().resolve(),
    }
    relation_map_file = Path(relation_map_file).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    for path in [*source_paths.values(), relation_map_file]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output root is not empty: {output_root}; pass --overwrite to replace it"
        )

    relation_map = load_relation_map(relation_map_file)
    documents_by_split = {}
    conversion = {}
    for split_name, path in source_paths.items():
        documents_by_split[split_name], stats = load_documents(path, split_name, relation_map)
        conversion[split_name] = dict(stats)

    all_documents = [
        document for documents in documents_by_split.values() for document in documents
    ]
    fallback_entities, fallback_predicates = _fallback_pools(all_documents)
    temp_root = output_root.parent / f".redocred_ablation_build_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = temp_root / "candidate_manifest.jsonl"
        manifest_rows = []
        split_stats = {}
        variant_stats = {variant: {} for variant in VARIANTS}
        for split_name in ("train", "validation", "test"):
            manifest, stats = _candidate_manifest_for_split(
                documents_by_split[split_name],
                split_name,
                seed,
                fallback_entities,
                fallback_predicates,
            )
            manifest_rows.extend({"split": split_name, **item} for item in manifest)
            split_stats[split_name] = {
                "source_documents": len(documents_by_split[split_name]),
                **stats,
            }

            for variant in VARIANTS:
                variant_dir = temp_root / variant
                variant_stats[variant][split_name] = _write_variant(
                    variant,
                    split_name,
                    manifest,
                    variant_dir / f"{split_name}.jsonl",
                    seed,
                )

        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_digest = file_digest(manifest_path)

        variant_validation = {}
        for variant in VARIANTS:
            paths = [
                temp_root / variant / f"{split}.jsonl" for split in ("train", "validation", "test")
            ]
            assert_disjoint_document_ids(paths)
            variant_validation[variant] = {}
            for split in ("train", "validation", "test"):
                generated = dict(variant_stats[variant][split])
                generated["path"] = str(output_root / variant / f"{split}.jsonl")
                generated["validation"] = validate_critic_sft_jsonl(
                    temp_root / variant / f"{split}.jsonl"
                )
                generated["validation"]["path"] = str(output_root / variant / f"{split}.jsonl")
                variant_validation[variant][split] = generated
            (temp_root / variant / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset": "tonytan48/Re-DocRED",
                        "purpose": "candidate-set ablation",
                        "variant": variant,
                        "seed": seed,
                        "candidate_set": variant,
                        "common_document_policy": "valid gold triple count >= 4; truncate each document to a multiple of 4",
                        "actions": list(VALID_ACTIONS),
                        "target_action_ratios": dict(ACTION_RATIOS),
                        "candidate_manifest": {
                            "path": str(output_root / "candidate_manifest.jsonl"),
                            **manifest_digest,
                        },
                        "splits": variant_validation[variant],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        metadata = {
            "dataset": "tonytan48/Re-DocRED",
            "purpose": "controlled Critic candidate-set ablation",
            "source_urls": dict(REDOCRED_SOURCE_URLS),
            "source_files": {
                split: {"path": str(path), **file_digest(path)}
                for split, path in source_paths.items()
            },
            "relation_map": {"path": str(relation_map_file), **file_digest(relation_map_file)},
            "seed": seed,
            "variants": list(VARIANTS),
            "candidate_range": [1, MAX_CANDIDATES],
            "common_document_policy": "valid gold triple count >= 4; truncate each document to a multiple of 4",
            "pipeline_test_policy": "full canonical 500-document test is retained; SFT test is only for standalone Critic checks",
            "target_action_ratios": dict(ACTION_RATIOS),
            "conversion": conversion,
            "splits": split_stats,
            "variant_validation": variant_validation,
            "candidate_manifest": {
                "path": str(output_root / "candidate_manifest.jsonl"),
                **manifest_digest,
            },
        }
        (temp_root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temp_root, output_root)
        return metadata
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired Re-DocRED Critic SFT data for Pointwise/Fixed-2/Fixed-4/Variable-set ablations."
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--relation-map", type=Path, default=DEFAULT_RELATION_MAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
            args.output_root,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (DatasetBuildError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
