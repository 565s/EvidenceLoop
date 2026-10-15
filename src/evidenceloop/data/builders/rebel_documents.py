from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from evidenceloop.data.builders.legacy_phase1 import (  # noqa: E402
    DatasetBuildError,
    default_rebel_zip,
    file_digest,
    iter_zip_jsonl,
    triple_key,
    valid_gold_triples,
)

ROOT = Path.cwd()


DEFAULT_CRITIC_TEST_FILE = ROOT / "data" / "critic_sft" / "rebel_phase2" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "pipeline" / "rebel_phase2_sft_test"
SOURCE_MEMBER = "en_test.jsonl"
OUTPUT_FILES = ("documents.jsonl", "gold_triples.jsonl", "metadata.json")


def _read_jsonl(path: Path) -> List[Mapping]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise DatasetBuildError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise DatasetBuildError(f"{path}: no records found")
    return rows


def _load_selected_documents(path: Path) -> List[Dict[str, str]]:
    selected = []
    seen_ids = set()
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        document_id = str(row.get("document_id") or "").strip()
        source_text = row.get("source_text")
        if not document_id:
            raise DatasetBuildError(f"{path}:{line_number}: document_id is required")
        if document_id in seen_ids:
            raise DatasetBuildError(f"{path}:{line_number}: duplicate document_id {document_id!r}")
        if not isinstance(source_text, str) or not source_text.strip():
            raise DatasetBuildError(f"{path}:{line_number}: source_text is required")
        seen_ids.add(document_id)
        selected.append({"document_id": document_id, "text": source_text})
    return selected


def _raw_document_id(article: Mapping, sequence: int) -> str:
    raw_source_id = article.get("docid") or article.get("uri")
    source_id = str(raw_source_id).strip() if raw_source_id else f"test_{sequence:06d}"
    return f"rebel_{source_id}"


def _validate_gold_triples(document_id: str, triples: List[Mapping]) -> None:
    if not triples:
        raise DatasetBuildError(f"{document_id}: no valid gold triples")
    seen = set()
    for index, triple in enumerate(triples):
        if not isinstance(triple, Mapping):
            raise DatasetBuildError(f"{document_id}: gold triple {index} is not an object")
        if set(triple) != {"subject", "predicate", "object"}:
            raise DatasetBuildError(
                f"{document_id}: gold triple {index} must contain only subject/predicate/object"
            )
        if any(not isinstance(triple[field], str) or not triple[field].strip() for field in triple):
            raise DatasetBuildError(f"{document_id}: gold triple {index} has an empty field")
        key = triple_key(triple)
        if key in seen:
            raise DatasetBuildError(f"{document_id}: duplicate gold triple at index {index}")
        seen.add(key)


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(
    source_zip: Path,
    critic_test_file: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Dict:
    source_zip = Path(source_zip).expanduser().resolve()
    critic_test_file = Path(critic_test_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if not critic_test_file.is_file():
        raise FileNotFoundError(critic_test_file)

    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    selected = _load_selected_documents(critic_test_file)
    selected_by_id = {row["document_id"]: row for row in selected}
    matched: Dict[str, Dict] = {}
    raw_triple_count = 0
    valid_gold_count = 0
    rows_scanned = 0

    with zipfile.ZipFile(source_zip) as archive:
        if SOURCE_MEMBER not in set(archive.namelist()):
            raise DatasetBuildError(f"source ZIP is missing member: {SOURCE_MEMBER}")
        for sequence, article in enumerate(iter_zip_jsonl(archive, SOURCE_MEMBER)):
            rows_scanned += 1
            document_id = _raw_document_id(article, sequence)
            if document_id not in selected_by_id:
                continue
            if document_id in matched:
                raise DatasetBuildError(
                    f"{SOURCE_MEMBER}: duplicate selected document_id {document_id!r}"
                )

            raw_text = str(article.get("text") or "").strip()
            expected_text = selected_by_id[document_id]["text"]
            if raw_text != expected_text:
                raise DatasetBuildError(
                    f"{document_id}: source text differs between SFT test and {SOURCE_MEMBER}"
                )

            raw_triples = article.get("triples") or []
            raw_triple_count += len(raw_triples) if isinstance(raw_triples, list) else 0
            triples = [dict(item["triple"]) for item in valid_gold_triples(article)]
            _validate_gold_triples(document_id, triples)
            valid_gold_count += len(triples)
            matched[document_id] = {
                "document": {"document_id": document_id, "text": expected_text},
                "gold": {"document_id": document_id, "triples": triples},
            }

    missing = [row["document_id"] for row in selected if row["document_id"] not in matched]
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        raise DatasetBuildError(
            f"{len(missing)} selected document(s) were not found in {SOURCE_MEMBER}: {preview}"
        )

    documents = [matched[row["document_id"]]["document"] for row in selected]
    gold_rows = [matched[row["document_id"]]["gold"] for row in selected]
    gold_counts = [len(row["triples"]) for row in gold_rows]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".rebel_pipeline_build_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(temp_dir / "documents.jsonl", documents)
        _write_jsonl(temp_dir / "gold_triples.jsonl", gold_rows)
        files = {
            name: file_digest(temp_dir / name) for name in ("documents.jsonl", "gold_triples.jsonl")
        }
        metadata = {
            "dataset": "Babelscape/rebel-dataset",
            "purpose": "pipeline evaluation on the documents selected for phase2 Critic SFT test",
            "source_zip": str(source_zip),
            "source_member": SOURCE_MEMBER,
            "critic_sft_test_file": str(critic_test_file),
            "critic_sft_test_digest": file_digest(critic_test_file),
            "selection": "document IDs and order from phase2 Critic SFT test.jsonl",
            "statistics": {
                "documents": len(documents),
                "rows_scanned": rows_scanned,
                "raw_triples": raw_triple_count,
                "valid_gold_triples": valid_gold_count,
                "filtered_raw_triples": raw_triple_count - valid_gold_count,
                "min_gold_triples_per_document": min(gold_counts),
                "max_gold_triples_per_document": max(gold_counts),
                "mean_gold_triples_per_document": round(valid_gold_count / len(documents), 6),
            },
            "validation": {
                "unique_document_ids": True,
                "all_selected_ids_found": True,
                "exact_source_text_matches": len(documents),
                "document_gold_order_match": True,
                "nonempty_gold_documents": len(documents),
                "schema_errors": 0,
            },
            "files": files,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
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
        description="Build a pipeline dataset from the original REBEL documents selected for phase2 Critic test."
    )
    parser.add_argument("--source-zip", type=Path, default=default_rebel_zip())
    parser.add_argument("--critic-test-file", type=Path, default=DEFAULT_CRITIC_TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.source_zip is None:
        print(
            "error: rebel_dataset.zip was not found; pass --source-zip or set REBEL_DATASET_ZIP",
            file=sys.stderr,
        )
        return 1
    try:
        metadata = build_dataset(
            args.source_zip,
            args.critic_test_file,
            args.output_dir,
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
