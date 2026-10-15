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
from typing import Dict, Mapping, Optional, Sequence

from evidenceloop.data.builders.legacy_phase1 import (  # noqa: E402
    DatasetBuildError,
    file_digest,
    result_item,
    triple_key,
)
from evidenceloop.data.builders.rebel_sft import (  # noqa: E402
    ACTION_RATIOS,
    MAX_CANDIDATES,
    ActionScheduler,
    _candidate_for_action,
    normalized_mutual_information,
    validate_phase2_record,
)
from evidenceloop.data.redocred import (  # noqa: E402
    REDOCRED_SOURCE_URLS,
    ReDocREDError,
    load_documents,
    load_relation_map,
)
from evidenceloop.data.sft import (  # noqa: E402
    VALID_ACTIONS,
    assert_disjoint_document_ids,
    validate_critic_sft_jsonl,
)

ROOT = Path.cwd()


DEFAULT_OUTPUT_DIR = ROOT / "data" / "critic_sft" / "re_docred_phase2"
OUTPUT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json")


def _record_for_batch(
    document: Mapping,
    split_name: str,
    document_index: int,
    batch_index: int,
    bases: Sequence[Mapping],
    scheduler: ActionScheduler,
    fallback_entities: Sequence[str],
    fallback_predicates: Sequence[str],
    rng: random.Random,
) -> Dict:
    gold_items = document["gold_items"]
    gold_keys = {triple_key(item["triple"]) for item in gold_items}
    used_candidate_keys = set()
    generated = []

    for base in bases:
        built = None
        for action in scheduler.ranked_actions(rng):
            value = _candidate_for_action(
                action,
                base,
                document["text"],
                gold_items,
                gold_keys,
                document["entities"],
                document["predicates"],
                fallback_entities,
                fallback_predicates,
                used_candidate_keys,
                rng,
            )
            if value is not None:
                candidate, selected_base, corruption = value
                built = (action, candidate, selected_base, corruption)
                break
        if built is None:
            raise DatasetBuildError(
                f"{document['document_id']}: cannot generate candidate for batch {batch_index}"
            )
        action, candidate, selected_base, corruption = built
        scheduler.commit(action)
        used_candidate_keys.add(triple_key(candidate))
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
        "document_id": (f"redocred_{split_name}_{document_index:06d}_batch_{batch_index:03d}"),
        "source_document_id": document["document_id"],
        "source_text": document["text"],
        "review_stage": "initial_review",
        "triples_info": triples_info,
        "target": {"results": results},
    }
    validate_phase2_record(record, record["document_id"])
    return record


def _build_split(
    documents: Sequence[Mapping],
    split_name: str,
    output_path: Path,
    seed: int,
) -> Dict:
    rng = random.Random(f"{seed}:{split_name}:redocred_phase2")
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

    for document_index, document in enumerate(documents):
        bases = list(document["gold_items"])
        if not bases:
            skipped_documents += 1
            continue
        rng.shuffle(bases)
        for batch_index, start in enumerate(range(0, len(bases), MAX_CANDIDATES)):
            record = _record_for_batch(
                document,
                split_name,
                document_index,
                batch_index,
                bases[start : start + MAX_CANDIDATES],
                scheduler,
                fallback_entities,
                fallback_predicates,
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
    relation_map_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
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
    output_dir = Path(output_dir).expanduser().resolve()
    for path in [*source_paths.values(), relation_map_file]:
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    relation_map = load_relation_map(relation_map_file)
    documents = {}
    conversion_stats = {}
    for split_name, path in source_paths.items():
        documents[split_name], stats = load_documents(path, split_name, relation_map)
        conversion_stats[split_name] = dict(stats)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".redocred_sft_build_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        split_stats = {
            split_name: _build_split(
                documents[split_name],
                split_name,
                temp_dir / f"{split_name}.jsonl",
                seed,
            )
            for split_name in ("train", "validation", "test")
        }
        validation = {
            split_name: validate_critic_sft_jsonl(temp_dir / f"{split_name}.jsonl")
            for split_name in ("train", "validation", "test")
        }
        for split_name, summary in validation.items():
            summary["path"] = str(output_dir / f"{split_name}.jsonl")
        assert_disjoint_document_ids(
            [temp_dir / f"{split_name}.jsonl" for split_name in ("train", "validation", "test")]
        )
        files = {
            f"{split_name}.jsonl": file_digest(temp_dir / f"{split_name}.jsonl")
            for split_name in ("train", "validation", "test")
        }
        metadata = {
            "dataset": "tonytan48/Re-DocRED",
            "phase": 2,
            "source_urls": dict(REDOCRED_SOURCE_URLS),
            "source_files": {
                split_name: {
                    "path": str(path),
                    **file_digest(path),
                }
                for split_name, path in source_paths.items()
            },
            "relation_map": {
                "path": str(relation_map_file),
                "relations": len(relation_map),
                **file_digest(relation_map_file),
            },
            "seed": seed,
            "candidate_range": [1, MAX_CANDIDATES],
            "batching": "all valid gold triples, shuffled and partitioned into batches of at most four",
            "target_action_ratios": dict(ACTION_RATIOS),
            "review_stage": "initial_review",
            "origin": "proposer",
            "conversion": conversion_stats,
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
        description="Build variable-batch Critic SFT data from Re-DocRED."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--relation-map", type=Path, required=True)
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
        ReDocREDError,
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
