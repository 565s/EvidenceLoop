from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from evidenceloop.data.builders.rebel_sft import validate_phase2_record
from evidenceloop.data.sft import VALID_ACTIONS
from evidenceloop.reports.sft_statistics import (
    CANDIDATE_SIZES,
    ERROR_TYPES,
    SPLITS,
    collect_split_statistics,
)

DATASETS = ("rebel_phase2", "re_docred_phase2", "scierc_phase2", "biored_phase2")

DATA_ROOT = Path.cwd() / "data" / "critic_sft"


def _dig(mapping: Mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _raw_rejected(metadata: Mapping, split: str) -> Dict:
    split_data = _dig(metadata, "splits", split, default={})
    split_data = split_data if isinstance(split_data, Mapping) else {}
    rejected = split_data.get("rejected", {})
    rejected = rejected if isinstance(rejected, Mapping) else {}
    return {
        "rows_scanned": int(split_data.get("rows_scanned", 0) or 0),
        "rejected_rows_by_reason": {str(key): int(value) for key, value in rejected.items()},
        "rejected_rows": sum(int(value) for value in rejected.values()),
        "skipped_source_documents": int(split_data.get("skipped_source_documents", 0) or 0),
    }


def _conversion_filtering(metadata: Mapping, split: str) -> Dict:
    conversion = _dig(metadata, "conversion", split, default={})
    conversion = conversion if isinstance(conversion, Mapping) else {}
    counts = conversion.get("counts")
    counts = counts if isinstance(counts, Mapping) else conversion
    raw = counts.get("raw_relations", counts.get("raw_labels"))
    valid = counts.get("valid_triples")
    removed = max(int(raw) - int(valid), 0) if raw is not None and valid is not None else None
    excluded = {}
    for key, value in counts.items():
        if not isinstance(value, (int, float)) or key in {
            "raw_relations",
            "raw_labels",
            "valid_triples",
            "raw_documents",
            "documents",
            "raw_sentences",
            "raw_tokens",
            "raw_entity_mentions",
            "valid_entity_mentions",
            "coreference_cluster_mentions",
            "coreference_clusters",
        }:
            continue
        # These conversion counters describe actual excluded/invalid material.
        if any(
            token in str(key).lower()
            for token in (
                "invalid",
                "duplicate",
                "unusable",
                "unrepresentable",
                "unknown",
                "missing",
                "empty",
                "cross_sentence",
                "misaligned",
                "endpoint",
            )
        ):
            if int(value) != 0:
                excluded[str(key)] = int(value)
    if removed is None and excluded:
        # BioRED stores total relations as raw_relations and includes annotation
        # counts that are not mutually exclusive; report explicit exclusion
        # counters rather than manufacturing a subtraction from valid annotations.
        removed = sum(excluded.values())
    return {
        "raw_relation_count": int(raw) if raw is not None else None,
        "valid_triples": int(valid) if valid is not None else None,
        "removed_or_invalid_total": removed,
        "breakdown": dict(sorted(excluded.items())),
        "interpretation": (
            "raw relations minus valid triples"
            if raw is not None and valid is not None
            else "sum of explicit conversion rejection counters; overlapping categories may occur"
        ),
    }


def _source_reference(metadata: Mapping, split: str) -> str:
    source_files = metadata.get("source_files")
    if isinstance(source_files, Mapping):
        value = source_files.get(split)
        if isinstance(value, Mapping):
            return str(value.get("path", ""))
    members = metadata.get("source_members")
    if isinstance(members, Mapping) and split in members:
        return str(members[split])
    if split == "validation":
        return "dev/validation"
    return split


def collect_phase2_dataset_statistics(dataset: str, data_root: Path = DATA_ROOT) -> Dict:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported Critic phase-two dataset: {dataset}")
    root = Path(data_root).expanduser().resolve() / dataset
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{metadata_path}: expected JSON object")

    splits = {}
    for split in SPLITS:
        file_path = root / f"{split}.jsonl"
        stats = collect_split_statistics(file_path, validator=validate_phase2_record)
        recorded = _dig(metadata, "splits", split, default={})
        recorded = recorded if isinstance(recorded, Mapping) else {}
        generation_rejections = _raw_rejected(metadata, split)
        source_documents = int(recorded.get("source_documents", 0) or 0)
        if not source_documents:
            source_documents = generation_rejections["rows_scanned"]
        skipped_source_documents = int(recorded.get("skipped_source_documents", 0) or 0)
        if not skipped_source_documents:
            skipped_source_documents = generation_rejections["rejected_rows_by_reason"].get(
                "no_valid_gold", 0
            )
        stats["generation_metadata"] = {
            "records": int(recorded.get("records", 0) or 0),
            "candidates": int(recorded.get("candidates", 0) or 0),
            "batch_size_counts": {
                str(size): int(_dig(recorded, "batch_size_counts", str(size), default=0) or 0)
                for size in CANDIDATE_SIZES
            },
            "action_counts": {
                action: int(_dig(recorded, "action_counts", action, default=0) or 0)
                for action in VALID_ACTIONS
            },
            "corruption_counts": dict(recorded.get("corruption_counts", {}))
            if isinstance(recorded.get("corruption_counts"), Mapping)
            else {},
            "source_documents": source_documents,
            "skipped_source_documents": skipped_source_documents,
        }
        stats["filtering"] = {
            "source_conversion": _conversion_filtering(metadata, split),
            "generation_rejections": generation_rejections,
            "candidate_strategy": {
                "source_documents": source_documents,
                "skipped_source_documents": skipped_source_documents,
                "excluded_documents_lt4": (
                    int(recorded["excluded_documents_lt4"])
                    if recorded.get("excluded_documents_lt4") is not None
                    else None
                ),
                "dropped_tail_candidates": (
                    int(recorded["dropped_tail_gold_triples"])
                    if recorded.get("dropped_tail_gold_triples") is not None
                    else None
                ),
            },
        }
        stats["official_source"] = _source_reference(metadata, split)
        splits[split] = stats

    return {
        "dataset_key": dataset,
        "dataset": metadata.get("dataset", dataset),
        "phase": metadata.get("phase", 2),
        "data_root": str(root),
        "metadata_path": str(metadata_path),
        "seed": metadata.get("seed"),
        "splits": list(SPLITS),
        "official_split_policy": (
            "保留来源数据集提供的 train/validation(test)/test 划分；未在 Critic SFT 构建阶段重新随机划分。"
        ),
        "randomness_policy": (
            "seed 与 split 名称共同初始化每个划分独立的确定性随机流，用于候选打乱和/或动作采样；"
            "不改变来源数据集的 train/dev/test 文档归属。"
        ),
        "source_members": metadata.get("source_members", {}),
        "source_files": metadata.get("source_files", {}),
        "candidate_range": metadata.get("candidate_range", [1, 4]),
        "batching_rule": metadata.get("batching", metadata.get("batch_size_rule", "")),
        "target_action_ratios_for_context": metadata.get("target_action_ratios", {}),
        "splits_statistics": splits,
        "notes": [
            "动作数、动作比例、集合规模、错误类型、证据质量及自动校验结果直接来自最终 JSONL。",
            "最终 JSONL 自动校验失败记录/候选是最终文件内检出的无效样本数；构建元数据中记录的跳过/拒绝数单独列出。",
            "转换阶段计数口径因数据集而异；显式无效/重复计数按来源元数据原样报告，可能不是互斥类别。",
            "若构建元数据没有记录截断/排除三元组数量，则保留为未记录，不根据最终 JSONL 推算。",
            "不能从现有元数据恢复的历史候选生成尝试失败数不作推算。",
        ],
    }


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_phase2_markdown(reports: Sequence[Mapping]) -> str:
    lines = [
        "# 四类 Critic SFT Phase 2 数据统计",
        "",
        "以下动作数与比例均从各数据集最终 JSONL 逐条统计；目标比例只作为构造背景，不用于计算实际比例。",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {report.get('dataset_key')}（{report.get('dataset')}）",
                "",
                f"- 随机种子：`{report.get('seed')}`",
                f"- 划分方式：{report.get('official_split_policy')}",
                f"- 随机种子用途：{report.get('randomness_policy')}",
                f"- 候选范围：`{report.get('candidate_range')}`；构造规则：{report.get('batching_rule')}",
                "",
                "| 划分 | 来源文件/成员 | 记录数 | 候选数 | 集合 1 | 集合 2 | 集合 3 | 集合 4 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            stats = report["splits_statistics"][split]
            sizes = stats["candidate_set_counts"]
            lines.append(
                f"| {split} | {stats['official_source']} | {stats['records']} | {stats['candidates']} | "
                + " | ".join(str(sizes.get(str(size), 0)) for size in CANDIDATE_SIZES)
                + " |"
            )
        lines.extend(
            [
                "",
                "### 动作数量与实际比例",
                "",
                "| 划分 | KEEP | DELETE | FIX | REWRITE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            stats = report["splits_statistics"][split]
            counts, ratios = stats["action_counts"], stats["action_ratios"]
            lines.append(
                f"| {split} | "
                + " | ".join(
                    f"{counts[action]} ({_percent(ratios[action])})" for action in VALID_ACTIONS
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### 错误类型数量（候选级）",
                "",
                "| 划分 | none | subject | object | predicate | direction | predicate+object |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            errors = report["splits_statistics"][split]["error_type_counts"]
            lines.append(
                f"| {split} | "
                + " | ".join(str(errors.get(error, 0)) for error in ERROR_TYPES)
                + " |"
            )
        lines.extend(
            [
                "",
                "### 证据质量与自动校验",
                "",
                "| 划分 | 需证据候选 | 非空 | 可定位 | 非空且可定位 | 比例 | DELETE 数 | DELETE 空证据 | 校验失败记录 | 校验失败候选 | 最终文件无效记录 | 最终文件无效候选 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            stats = report["splits_statistics"][split]
            evidence, validation = stats["evidence"], stats["validation"]
            lines.append(
                f"| {split} | {evidence['required_candidates']} | {evidence['nonempty_candidates']} | "
                f"{evidence['locatable_candidates']} | {evidence['nonempty_and_locatable']} | "
                f"{_percent(evidence['rate'])} | {evidence['delete_candidates']} | {evidence['delete_empty']} | "
                f"{validation['failed_records']} | {validation['failed_candidates']} | "
                f"{validation['final_invalid_records']} | {validation['final_invalid_candidates']} |"
            )
        lines.extend(
            [
                "",
                "### 转换过滤与生成阶段跳过",
                "",
                "| 划分 | 源文档 | 跳过源文档 | 少于 4 个三元组文档 | 截断尾部候选 | 转换删除/无效总数 | 转换原因计数 | 扫描源行 | 生成拒绝/跳过数 | 生成拒绝原因 |",
                "|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for split in SPLITS:
            filtering = report["splits_statistics"][split]["filtering"]
            conversion = filtering["source_conversion"]
            rejected = filtering["generation_rejections"]
            strategy = filtering["candidate_strategy"]
            lines.append(
                f"| {split} | {strategy['source_documents']} | {strategy['skipped_source_documents']} | "
                f"{strategy['excluded_documents_lt4'] if strategy['excluded_documents_lt4'] is not None else '未记录'} | "
                f"{strategy['dropped_tail_candidates'] if strategy['dropped_tail_candidates'] is not None else '未记录'} | "
                f"{conversion['removed_or_invalid_total'] if conversion['removed_or_invalid_total'] is not None else '未记录'} | "
                f"{json.dumps(conversion['breakdown'], ensure_ascii=False, sort_keys=True)} | "
                f"{rejected['rows_scanned']} | "
                f"{rejected['rejected_rows'] + rejected['skipped_source_documents']} | "
                f"{json.dumps(rejected['rejected_rows_by_reason'], ensure_ascii=False, sort_keys=True)}"
                + (
                    f"；跳过源文档={rejected['skipped_source_documents']}"
                    if rejected["skipped_source_documents"]
                    else ""
                )
                + " |"
            )
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## 统计口径",
            "",
            "- 证据可定位率分母为 KEEP/FIX/REWRITE；DELETE 的空证据要求单列。",
            "- `none` 表示无错误的 KEEP 基线；其他 corruption 值是构造的错误类型。",
            "- “自动校验失败”逐条复核最终 JSONL；失败记录/候选数即最终文件中检出的无效样本数。",
            "- 转换阶段和构建阶段跳过数按元数据原有口径单列，不与最终校验失败相加。",
            "- 截断或排除数量只报告元数据明确持久化的值；缺失时标为未记录。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DATASETS",
    "DATA_ROOT",
    "collect_phase2_dataset_statistics",
    "render_phase2_markdown",
]
