from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from evidenceloop.data.biored import (  # noqa: E402
    BIORED_GUIDELINE_URL,
    BIORED_SOURCE_URL,
    SOURCE_MEMBERS,
    BioREDError,
    load_documents,
    load_relation_map,
    undirected_triple_key,
)
from evidenceloop.data.builders.biored_sft import DEFAULT_RELATION_MAP  # noqa: E402
from evidenceloop.data.builders.legacy_phase1 import DatasetBuildError, file_digest  # noqa: E402

ROOT = Path.cwd()


DEFAULT_OUTPUT_DIR = ROOT / "data" / "pipeline" / "biored_phase2_sft_test"
OUTPUT_FILES = ("documents.jsonl", "gold_triples.jsonl", "metadata.json")


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _validate_gold(document_id: str, triples: Sequence[Mapping]) -> None:
    seen = set()
    for index, triple in enumerate(triples):
        if set(triple) != {"subject", "predicate", "object"}:
            raise DatasetBuildError(
                f"{document_id}: gold triple {index} must contain only subject/predicate/object"
            )
        if any(not isinstance(value, str) or not value.strip() for value in triple.values()):
            raise DatasetBuildError(f"{document_id}: gold triple {index} has an empty field")
        key = undirected_triple_key(triple)
        if key in seen:
            raise DatasetBuildError(f"{document_id}: duplicate undirected gold triple {index}")
        seen.add(key)


def build_dataset(
    source_zip: Path,
    relation_map_file: Path = DEFAULT_RELATION_MAP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    overwrite: bool = False,
) -> Dict:
    source_zip = Path(source_zip).expanduser().resolve()
    relation_map_file = Path(relation_map_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (source_zip, relation_map_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    relation_map = load_relation_map(relation_map_file)
    converted, conversion_stats = load_documents(source_zip, "test", relation_map)
    documents = []
    gold_rows = []
    for document in converted:
        triples = [dict(item["triple"]) for item in document["gold_items"]]
        _validate_gold(document["document_id"], triples)
        documents.append({"document_id": document["document_id"], "text": document["text"]})
        gold_rows.append({"document_id": document["document_id"], "triples": triples})
    gold_counts = [len(row["triples"]) for row in gold_rows]
    total_gold = sum(gold_counts)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".biored_pipeline_build_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(temp_dir / "documents.jsonl", documents)
        _write_jsonl(temp_dir / "gold_triples.jsonl", gold_rows)
        files = {
            name: file_digest(temp_dir / name) for name in ("documents.jsonl", "gold_triples.jsonl")
        }
        metadata = {
            "dataset": "NCBI/BioRED",
            "purpose": "pipeline evaluation on the official BioRED test split",
            "source_url": BIORED_SOURCE_URL,
            "annotation_guideline_url": BIORED_GUIDELINE_URL,
            "source_file": {"path": str(source_zip), **file_digest(source_zip)},
            "source_member": SOURCE_MEMBERS["test"],
            "relation_map": {
                "path": str(relation_map_file),
                "relations": len(relation_map),
                **file_digest(relation_map_file),
            },
            "novelty_policy": "Novel/No excluded from canonical triples and retained in conversion statistics",
            "entity_order_mode": "undirected",
            "statistics": {
                "documents": len(documents),
                "valid_gold_triples": total_gold,
                "min_gold_triples_per_document": min(gold_counts),
                "max_gold_triples_per_document": max(gold_counts),
                "mean_gold_triples_per_document": round(total_gold / len(documents), 6),
                "conversion": dict(conversion_stats),
            },
            "validation": {
                "unique_document_ids": len({row["document_id"] for row in documents})
                == len(documents),
                "document_gold_order_match": [row["document_id"] for row in documents]
                == [row["document_id"] for row in gold_rows],
                "nonempty_gold_documents": sum(bool(row["triples"]) for row in gold_rows),
                "schema_errors": 0,
            },
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
        description="Build canonical pipeline evaluation data from BioRED test."
    )
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--relation-map", type=Path, default=DEFAULT_RELATION_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        metadata = build_dataset(
            args.source_zip,
            args.relation_map,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (
        DatasetBuildError,
        BioREDError,
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
