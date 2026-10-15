"""Evaluation-only loading of canonical documents/gold or historical source/target pairs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evidenceloop.data.registry import DATASETS as DATASET_SPECS
from evidenceloop.data.registry import AliasDict
from evidenceloop.experiments.checkpoints import load_graphjudge_rebel

ROOT = Path.cwd()


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    label: str
    data_dir: str
    entity_order_mode: str = "directed"
    symmetric_predicates: Tuple[str, ...] = ()


def count_dataset_records(data_dir: Path) -> int:
    documents_path = data_dir / "documents.jsonl"
    if documents_path.exists():
        return sum(
            1 for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    source_path = data_dir / "test.source"
    if source_path.exists():
        return sum(
            1 for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    raise FileNotFoundError(f"missing documents.jsonl or test.source under {data_dir}")


def load_dataset_rows(data_dir: Path, limit: Optional[int]) -> Tuple[List[Dict], int]:
    selected_limit = limit if limit is not None else count_dataset_records(data_dir)
    if not isinstance(selected_limit, int) or selected_limit <= 0:
        raise ValueError("limit must be positive")
    return load_graphjudge_rebel(data_dir, selected_limit), selected_limit


DATASETS = AliasDict(
    {
        key: DatasetPreset(
            key,
            spec.label,
            "data/pipeline/" + spec.directory,
            spec.entity_order_mode,
            spec.symmetric_predicates,
        )
        for key, spec in DATASET_SPECS.items()
    }
)
