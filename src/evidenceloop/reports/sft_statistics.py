from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.registry import candidate_strategy, variant_directory
from evidenceloop.data.sft import VALID_ACTIONS, validate_critic_sft_record

"""Statistics for the materialized Critic SFT datasets.

The dataset builders deliberately keep generation metadata small and focused on
building the data.  This module performs a second, read-only pass over the
materialized JSONL files so the reported numbers always describe the files that
will actually be consumed by training.
"""


SPLITS = ("train", "validation", "test")
VARIANTS = ("fixed1", "fixed2", "fixed4", "variable")
CANDIDATE_SIZES = (1, 2, 3, 4)
ERROR_TYPES = ("none", "subject", "object", "predicate", "direction", "predicate+object")
EVIDENCE_ACTIONS = ("KEEP", "FIX", "REWRITE")


class CriticSFTStatisticsError(ValueError):
    """Raised when a statistics input is missing or has an invalid shape."""


def _counter_with_keys(counter: Counter, keys: Sequence[str]) -> Dict[str, int]:
    """Return a deterministic counter mapping with zeroes for missing keys."""

    result = {key: int(counter.get(key, 0)) for key in keys}
    result.update(
        {str(key): int(value) for key, value in sorted(counter.items()) if str(key) not in result}
    )
    return result


def _ratios(counts: Mapping[str, int], denominator: int, keys: Sequence[str]) -> Dict[str, float]:
    denominator = int(denominator)
    return {key: (int(counts.get(key, 0)) / denominator if denominator else 0.0) for key in keys}


def _error_name(exc: BaseException) -> str:
    return type(exc).__name__


def _iter_jsonl_rows(
    path: Path,
) -> Iterator[Tuple[int, Optional[Mapping], Optional[BaseException]]]:
    """Yield JSONL rows while retaining malformed lines for QC accounting."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                yield line_number, None, exc
                continue
            if not isinstance(value, Mapping):
                yield line_number, None, CriticSFTStatisticsError("JSONL row must be an object")
                continue
            yield line_number, value, None


def _failure_detail(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 240:
        message = message[:237] + "..."
    return message


def _new_evidence_stats() -> Dict[str, int | float]:
    return {
        "required_candidates": 0,
        "nonempty_candidates": 0,
        "locatable_candidates": 0,
        "nonempty_and_locatable": 0,
        "empty_or_unlocatable": 0,
        "rate": 0.0,
        "delete_candidates": 0,
        "delete_empty": 0,
        "delete_nonempty": 0,
        "delete_nonempty_and_locatable": 0,
    }


def _update_evidence_stats(
    stats: Dict[str, int | float], action: str, evidence, source_text: str
) -> None:
    evidence_is_string = isinstance(evidence, str)
    nonempty = evidence_is_string and bool(evidence.strip())
    locatable = nonempty and evidence in source_text
    if action in EVIDENCE_ACTIONS:
        stats["required_candidates"] += 1
        stats["nonempty_candidates"] += int(nonempty)
        stats["locatable_candidates"] += int(locatable)
        stats["nonempty_and_locatable"] += int(nonempty and locatable)
        stats["empty_or_unlocatable"] += int(not (nonempty and locatable))
    elif action == "DELETE":
        stats["delete_candidates"] += 1
        stats["delete_empty"] += int(not nonempty)
        stats["delete_nonempty"] += int(nonempty)
        stats["delete_nonempty_and_locatable"] += int(locatable)


def collect_split_statistics(
    path: Path,
    *,
    validator: Optional[Callable[[Mapping, str], object]] = None,
) -> Dict:
    """Collect actual candidate and quality counts from one materialized JSONL."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    action_counts = Counter()
    size_counts = Counter()
    error_counts = Counter()
    validation_error_types = Counter()
    validation_error_messages = Counter()
    evidence = _new_evidence_stats()
    records = 0
    candidates = 0
    failed_records = 0
    failed_candidates = 0
    seen_document_ids = set()

    for line_number, record, parse_error in _iter_jsonl_rows(path):
        records += 1
        if parse_error is not None:
            failed_records += 1
            validation_error_types[_error_name(parse_error)] += 1
            validation_error_messages[_failure_detail(parse_error)] += 1
            continue

        assert record is not None
        triples_info = record.get("triples_info")
        row_candidates = len(triples_info) if isinstance(triples_info, list) else 0
        candidates += row_candidates
        if isinstance(triples_info, list):
            size_counts[len(triples_info)] += 1

        try:
            if validator is None:
                validate_critic_sft_record(record, f"{path}:{line_number}")
            else:
                validator(record, f"{path}:{line_number}")
        except Exception as exc:  # the report must account for every bad row
            failed_records += 1
            validation_error_types[_error_name(exc)] += 1
            validation_error_messages[_failure_detail(exc)] += 1
            failed_candidates += row_candidates

        document_id = record.get("document_id")
        if isinstance(document_id, str):
            if document_id in seen_document_ids:
                failed_records += 1
                failed_candidates += row_candidates
                validation_error_types["DuplicateDocumentId"] += 1
                validation_error_messages[f"duplicate document_id: {document_id}"] += 1
            else:
                seen_document_ids.add(document_id)

        target = record.get("target")
        results = target.get("results") if isinstance(target, Mapping) else None
        if not isinstance(results, list):
            continue
        source_text = record.get("source_text")
        source_text = source_text if isinstance(source_text, str) else ""
        for index, result in enumerate(results):
            if not isinstance(result, Mapping):
                continue
            action = str(result.get("action") or "").upper()
            if action in VALID_ACTIONS:
                action_counts[action] += 1
            evidence_value = result.get("evidence", "")
            _update_evidence_stats(evidence, action, evidence_value, source_text)
            if isinstance(triples_info, list) and index < len(triples_info):
                item = triples_info[index]
                if isinstance(item, Mapping):
                    error_counts[str(item.get("corruption") or "none")] += 1

    evidence["rate"] = (
        evidence["nonempty_and_locatable"] / evidence["required_candidates"]
        if evidence["required_candidates"]
        else 0.0
    )
    action_counts = _counter_with_keys(action_counts, VALID_ACTIONS)
    error_counts = _counter_with_keys(error_counts, ERROR_TYPES)
    raw_size_counts = size_counts.copy()
    size_counts = {str(size): int(raw_size_counts.get(size, 0)) for size in CANDIDATE_SIZES}
    size_counts["other"] = sum(
        int(count) for size, count in raw_size_counts.items() if size not in CANDIDATE_SIZES
    )
    return {
        "path": str(path),
        "records": records,
        "candidates": candidates,
        "candidate_set_counts": size_counts,
        "action_counts": action_counts,
        "action_ratios": _ratios(action_counts, candidates, VALID_ACTIONS),
        "error_type_counts": error_counts,
        "evidence": evidence,
        "validation": {
            "failed_records": failed_records,
            "failed_candidates": failed_candidates,
            "final_invalid_records": failed_records,
            "final_invalid_candidates": failed_candidates,
            "error_types": dict(sorted(validation_error_types.items())),
            "error_messages": dict(sorted(validation_error_messages.items())),
        },
    }


def _manifest_path(root: Path, metadata: Mapping) -> Path:
    raw = metadata.get("candidate_manifest")
    raw_path = raw.get("path") if isinstance(raw, Mapping) else None
    if raw_path:
        candidate = Path(str(raw_path))
        if candidate.is_file():
            return candidate
    fallback = root / "candidate_manifest.jsonl"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(fallback)


def collect_manifest_statistics(path: Path) -> Dict:
    """Collect the shared candidate statistics once, independent of grouping."""

    action_counts = Counter()
    error_counts = Counter()
    split_candidates = Counter()
    split_actions = {split: Counter() for split in SPLITS}
    split_errors = {split: Counter() for split in SPLITS}
    evidence = _new_evidence_stats()
    split_evidence = {split: _new_evidence_stats() for split in SPLITS}
    rows = 0
    documents = set()
    for _, row, parse_error in _iter_jsonl_rows(Path(path)):
        if parse_error is not None or row is None:
            continue
        rows += 1
        split = str(row.get("split") or "unknown")
        split_candidates[split] += 1
        document_id = row.get("document_id")
        if document_id is not None:
            documents.add((split, str(document_id)))
        action = str(row.get("action") or "").upper()
        if action in VALID_ACTIONS:
            action_counts[action] += 1
            if split in split_actions:
                split_actions[split][action] += 1
        corruption = str(row.get("corruption") or "none")
        error_counts[corruption] += 1
        if split in split_errors:
            split_errors[split][corruption] += 1
        source_text = str(row.get("source_text") or "")
        _update_evidence_stats(evidence, action, row.get("evidence", ""), source_text)
        if split in split_evidence:
            _update_evidence_stats(
                split_evidence[split], action, row.get("evidence", ""), source_text
            )
    evidence["rate"] = (
        evidence["nonempty_and_locatable"] / evidence["required_candidates"]
        if evidence["required_candidates"]
        else 0.0
    )
    for split_stats in split_evidence.values():
        split_stats["rate"] = (
            split_stats["nonempty_and_locatable"] / split_stats["required_candidates"]
            if split_stats["required_candidates"]
            else 0.0
        )
    action_counts = _counter_with_keys(action_counts, VALID_ACTIONS)
    error_counts = _counter_with_keys(error_counts, ERROR_TYPES)
    per_split = {}
    for split in SPLITS:
        split_count = int(split_candidates.get(split, 0))
        per_split[split] = {
            "candidates": split_count,
            "action_counts": _counter_with_keys(split_actions[split], VALID_ACTIONS),
            "action_ratios": _ratios(split_actions[split], split_count, VALID_ACTIONS),
            "error_type_counts": _counter_with_keys(split_errors[split], ERROR_TYPES),
            "evidence": split_evidence[split],
        }
    return {
        "path": str(path),
        "rows": rows,
        "documents": len(documents),
        "candidates": rows,
        "split_candidate_counts": {split: int(split_candidates.get(split, 0)) for split in SPLITS},
        "action_counts": action_counts,
        "action_ratios": _ratios(action_counts, rows, VALID_ACTIONS),
        "error_type_counts": error_counts,
        "evidence": evidence,
        "splits": per_split,
    }


def _source_filtering(metadata: Mapping) -> Dict:
    conversion = metadata.get("conversion")
    split_stats = metadata.get("splits")
    conversion = conversion if isinstance(conversion, Mapping) else {}
    split_stats = split_stats if isinstance(split_stats, Mapping) else {}
    result = {}
    for split in SPLITS:
        converted = conversion.get(split, {})
        policy = split_stats.get(split, {})
        converted = converted if isinstance(converted, Mapping) else {}
        policy = policy if isinstance(policy, Mapping) else {}
        raw_labels = int(converted.get("raw_labels", 0) or 0)
        valid_triples = int(converted.get("valid_triples", 0) or 0)
        removed = max(raw_labels - valid_triples, 0)
        explicit = {
            str(key): int(value)
            for key, value in converted.items()
            if key not in {"raw_labels", "valid_triples", "documents"}
            and isinstance(value, (int, float))
        }
        explicit_total = sum(explicit.values())
        result[split] = {
            "conversion": {
                "raw_labels": raw_labels,
                "valid_triples": valid_triples,
                "removed_total": removed,
                "breakdown": explicit,
                "unattributed_removed": max(removed - explicit_total, 0),
            },
            "candidate_policy": {
                "source_documents": int(policy.get("source_documents", 0) or 0),
                "eligible_documents": int(policy.get("eligible_documents", 0) or 0),
                "excluded_documents_lt4": int(policy.get("excluded_documents_lt4", 0) or 0),
                "excluded_gold_triples": int(policy.get("excluded_gold_triples", 0) or 0),
                "dropped_tail_gold_triples": int(policy.get("dropped_tail_gold_triples", 0) or 0),
                "policy_excluded_triples": int(policy.get("excluded_gold_triples", 0) or 0)
                + int(policy.get("dropped_tail_gold_triples", 0) or 0),
            },
        }
    return result


def _grouping_rules() -> Dict[str, str]:
    return {
        "fixed1": "每条候选单独成组（1）",
        "fixed2": "每 2 条候选成组",
        "fixed4": "每 4 条候选成组",
        "variable": "使用独立确定性随机流，将候选分成 1–4 条的组",
    }


def collect_statistics(data_root: Path) -> Dict:
    """Collect the complete report for the Re-DocRED candidate ablation."""

    root = Path(data_root).expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise CriticSFTStatisticsError(f"{metadata_path}: expected an object")

    configured_variants = tuple(
        candidate_strategy(v) for v in (metadata.get("variants") or VARIANTS)
    )
    variants = [variant for variant in VARIANTS if variant in configured_variants]
    if not variants:
        raise CriticSFTStatisticsError("no supported candidate-set variants found")

    variant_stats = {}
    for variant in variants:
        variant_stats[variant] = {
            split: collect_split_statistics(variant_directory(root, variant) / f"{split}.jsonl")
            for split in SPLITS
        }

    manifest = collect_manifest_statistics(_manifest_path(root, metadata))
    source_files = metadata.get("source_files")
    source_files = source_files if isinstance(source_files, Mapping) else {}
    source_split_files = {
        split: str((source_files.get(split) or {}).get("path", ""))
        if isinstance(source_files.get(split), Mapping)
        else ""
        for split in SPLITS
    }
    return {
        "dataset": metadata.get("dataset", "tonytan48/Re-DocRED"),
        "purpose": metadata.get("purpose", "controlled Critic candidate-set ablation"),
        "data_root": str(root),
        "metadata_path": str(metadata_path),
        "seed": metadata.get("seed"),
        "variants": variants,
        "splits": list(SPLITS),
        "source_split_files": source_split_files,
        "candidate_range": metadata.get("candidate_range", list(CANDIDATE_SIZES)),
        "common_document_policy": metadata.get("common_document_policy", ""),
        "pipeline_test_policy": metadata.get("pipeline_test_policy", ""),
        "split_method": "保留官方 train/dev/test 划分，不重新随机划分文档；候选与分组使用按 seed 派生的确定性随机流。",
        "grouping_rules": _grouping_rules(),
        "target_action_ratios": metadata.get("target_action_ratios", {}),
        "source_filtering": _source_filtering(metadata),
        "shared_candidate_manifest": manifest,
        "variants_statistics": variant_stats,
        "final_jsonl_quality": {
            variant: {
                split: {
                    "failed_records": variant_stats[variant][split]["validation"]["failed_records"],
                    "failed_candidates": variant_stats[variant][split]["validation"][
                        "failed_candidates"
                    ],
                    "final_invalid_records": variant_stats[variant][split]["validation"][
                        "failed_records"
                    ],
                    "final_invalid_candidates": variant_stats[variant][split]["validation"][
                        "failed_candidates"
                    ],
                }
                for split in SPLITS
            }
            for variant in variants
        },
        "notes": [
            "所有动作比例的分母均为对应最终 JSONL 中的实际候选数。",
            "证据比例的分母为 KEEP/FIX/REWRITE 候选；DELETE 的空证据契约单独统计。",
            "共享 candidate manifest 只在 shared_candidate_manifest 中作为总量统计一次；变体统计按各自最终 JSONL 展示，不跨变体求和。",
            "历史构建过程中未持久化的失败尝试无法从最终 JSONL 反推。",
        ],
    }


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping) -> str:
    """Render the machine-readable report as a compact reviewable Markdown file."""

    lines = [
        "# Re-DocRED Critic 候选集合统计报告",
        "",
        f"- 数据集：`{report.get('dataset', '')}`",
        f"- 随机种子：`{report.get('seed')}`",
        f"- 候选规模范围：`{report.get('candidate_range')}`",
        f"- 文档划分：{report.get('split_method', '')}",
        f"- 公共文档策略：{report.get('common_document_policy', '')}",
        "",
        "## 官方划分与数据过滤",
        "",
        "| 划分 | 官方源文件 | 源文档数 | 可用文档数 | 少于 4 个三元组的文档 | 排除三元组 | 截断尾部三元组 | 转换删除总数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    source_filtering = report.get("source_filtering", {})
    source_files = report.get("source_split_files", {})
    for split in SPLITS:
        value = source_filtering.get(split, {})
        conversion = value.get("conversion", {})
        policy = value.get("candidate_policy", {})
        lines.append(
            "| "
            + " | ".join(
                _cell(item)
                for item in (
                    split,
                    source_files.get(split, ""),
                    policy.get("source_documents", 0),
                    policy.get("eligible_documents", 0),
                    policy.get("excluded_documents_lt4", 0),
                    policy.get("excluded_gold_triples", 0),
                    policy.get("dropped_tail_gold_triples", 0),
                    conversion.get("removed_total", 0),
                )
            )
            + " |"
        )

    lines.extend(["", "## 共享 candidate manifest", ""])
    manifest = report.get("shared_candidate_manifest", {})
    lines.extend(
        [
            f"候选总数：**{manifest.get('candidates', 0)}**；文档数：**{manifest.get('documents', 0)}**。",
            "",
            "| 划分 | 候选数 | KEEP | DELETE | FIX | REWRITE | 需证据候选 | 非空且可定位 | 比例 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        split_stats = manifest.get("splits", {}).get(split, {})
        count = split_stats.get("candidates", 0)
        actions = split_stats.get("action_counts", {})
        evidence = split_stats.get("evidence", {})
        lines.append(
            f"| {split} | {count} | "
            + " | ".join(str(actions.get(action, 0)) for action in VALID_ACTIONS)
            + f" | {evidence.get('required_candidates', 0)} | "
            + f"{evidence.get('nonempty_and_locatable', 0)} | "
            + f"{_pct(float(evidence.get('rate', 0.0)))} |"
        )
    lines.extend(["", "## 变体分组规则", ""])
    for variant, rule in report.get("grouping_rules", {}).items():
        lines.append(f"- `{variant}`：{rule}")

    for variant in report.get("variants", []):
        lines.extend(["", f"## 变体：`{variant}`", ""])
        lines.extend(
            [
                "| 划分 | 记录数 | 候选数 | 集合 1 | 集合 2 | 集合 3 | 集合 4 | 自动校验失败记录 | 自动校验失败候选 | 最终文件无效记录 | 最终文件无效候选 | 证据可定位率 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            stats = report["variants_statistics"][variant][split]
            validation = stats["validation"]
            evidence = stats["evidence"]
            sizes = stats["candidate_set_counts"]
            lines.append(
                f"| {split} | {stats['records']} | {stats['candidates']} | "
                f"{sizes.get('1', 0)} | {sizes.get('2', 0)} | {sizes.get('3', 0)} | {sizes.get('4', 0)} | "
                f"{validation['failed_records']} | {validation['failed_candidates']} | "
                f"{validation['failed_records']} | {validation['failed_candidates']} | "
                f"{_pct(float(evidence['rate']))} |"
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
            stats = report["variants_statistics"][variant][split]
            counts = stats["action_counts"]
            ratios = stats["action_ratios"]
            cells = [
                f"{counts[action]} ({_pct(float(ratios[action]))})" for action in VALID_ACTIONS
            ]
            lines.append(f"| {split} | " + " | ".join(cells) + " |")

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
            errors = report["variants_statistics"][variant][split]["error_type_counts"]
            lines.append(
                f"| {split} | "
                + " | ".join(str(errors.get(error, 0)) for error in ERROR_TYPES)
                + " |"
            )

        lines.extend(
            [
                "",
                "### 证据与最终质量检查",
                "",
                "| 划分 | 需证据候选 | 非空 | 可定位 | 非空且可定位 | 非空且可定位比例 | DELETE 数 | DELETE 空证据 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split in SPLITS:
            evidence = report["variants_statistics"][variant][split]["evidence"]
            lines.append(
                f"| {split} | {evidence['required_candidates']} | {evidence['nonempty_candidates']} | "
                f"{evidence['locatable_candidates']} | {evidence['nonempty_and_locatable']} | "
                f"{_pct(float(evidence['rate']))} | {evidence['delete_candidates']} | {evidence['delete_empty']} |"
            )

    lines.extend(["", "## 统计口径", ""])
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CANDIDATE_SIZES",
    "CriticSFTStatisticsError",
    "ERROR_TYPES",
    "SPLITS",
    "VARIANTS",
    "collect_manifest_statistics",
    "collect_split_statistics",
    "collect_statistics",
    "render_markdown",
]
