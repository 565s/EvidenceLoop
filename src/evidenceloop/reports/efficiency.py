"""Offline Critic efficiency report; standard library only, with no model imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from evidenceloop.data.registry import candidate_strategy, variant_directory

ROOT = Path.cwd()
VARIANTS = ("fixed1", "fixed2", "fixed4", "variable")
DEFAULT_RUN_ROOT = ROOT / "data/experiments/re_docred_candidate_set_ablation"
DEFAULT_OUTPUT_PREFIX = ROOT / "reports/redocred_critic_efficiency"
REQUIRED_FILES = (
    "run_state.json",
    "progress.json",
    "summary.json",
    "sample_results.jsonl",
    "pipeline_records.jsonl",
)
RUNTIME_METRICS = (
    "critic_prompt_count",
    "critic_generation_batches",
    "critic_input_tokens",
    "critic_output_tokens",
    "critic_generation_seconds",
    "active_wall_seconds",
    "model_load_seconds",
    "critic_peak_gpu_memory_bytes",
)
STAGES = ("proposer", "initial_critic", "integrator", "final_critic")


def resolve_path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path: Path) -> Dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def index_rows(rows: Iterable[Mapping], source: Path) -> Dict:
    result = {}
    for row in rows:
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{source}: missing document_id")
        if document_id in result:
            raise ValueError(f"{source}: duplicate document_id {document_id}")
        result[document_id] = row
    return result


def ratio(numerator, denominator) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def distribution(sizes) -> Dict:
    return {str(size): count for size, count in sorted(Counter(sizes).items())}


def diagnostic(items, code, message, document_ids=()) -> None:
    ids = list(document_ids)
    items.append(
        {
            "code": code,
            "message": message,
            "document_count": len(ids),
            "example_document_ids": ids[:5],
        }
    )


def latest_complete_run(variant_root: Path) -> Path:
    """Select the newest finalized run with no pending or failed documents."""
    if not variant_root.is_dir():
        raise FileNotFoundError(variant_root)
    for run_dir in sorted(variant_root.iterdir(), reverse=True):
        if not run_dir.is_dir() or not all((run_dir / f).is_file() for f in REQUIRED_FILES):
            continue
        progress = read_json(run_dir / "progress.json")
        summary = read_json(run_dir / "summary.json")
        if (
            progress.get("finalized") is True
            and progress.get("pending") == 0
            and progress.get("failed") == 0
            and summary.get("samples_failed") == 0
            and summary.get("samples_completed", 0) > 0
            and summary.get("samples_completed") == summary.get("samples_requested")
        ):
            return run_dir
    raise ValueError(f"{variant_root}: no complete finalized run found")


def select_runs(run_root: Path, explicit: Sequence[str] = ()) -> Dict[str, Path]:
    runs = {}
    for value in explicit:
        if "=" not in value:
            raise ValueError("--run must use VARIANT=RUN_DIR")
        variant, path = value.split("=", 1)
        variant = candidate_strategy(variant)
        if variant not in VARIANTS or variant in runs:
            raise ValueError(f"unknown or duplicate --run variant: {variant}")
        runs[variant] = resolve_path(path)
    return {
        variant: runs[variant]
        if variant in runs
        else latest_complete_run(variant_directory(run_root, variant))
        for variant in VARIANTS
    }


def proposal_signature(triples: Sequence[Mapping], source_text: str) -> str:
    """Hash ordered SPOs and exact source text; ignore incidental triple metadata."""
    value = [
        source_text,
        [
            [triple.get("subject"), triple.get("predicate"), triple.get("object")]
            for triple in triples
        ],
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_proposal_cache(cache_dir: Path) -> Dict:
    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)
    rows = {}
    for path in sorted(cache_dir.glob("*.json")):
        if not path.stem.isdecimal():
            continue
        item = read_json(path)
        document_id = item.get("document_id")
        if not document_id or document_id in rows:
            raise ValueError(f"{path}: missing or duplicate cache document_id")
        triples = (item.get("payload") or {}).get("triples")
        if item.get("status") != "ok" or not isinstance(triples, list):
            rows[document_id] = {"status": "failed"}
            continue
        text = triples[0].get("source_text", "") if triples else ""
        rows[document_id] = {
            "status": "ok",
            "triple_count": len(triples),
            "signature": proposal_signature(triples, text),
        }
    if not rows:
        raise ValueError(f"{cache_dir}: empty proposal cache")
    return rows


def raw_chunks(raw, candidate_count: int):
    """Return observed sizes, count, and boundary validity; never use result count.

    Multi-prompt responses have saved index envelopes. A single prompt's response
    may be invalid JSON, empty, or omit results and still represents a generation.
    Missing response data for a nonempty request remains unknown.
    """
    if raw is None:
        return ([] if candidate_count == 0 else None), (0 if candidate_count == 0 else None), True
    if not isinstance(raw, str):
        return None, None, False
    if not raw:
        if candidate_count == 0:
            return [], 0, True
        if candidate_count <= 4:
            return [candidate_count], 1, True
        return None, None, False
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, dict) and "response" in item for item in value)
    ):
        sizes = []
        cursor = 0
        valid = True
        for item in value:
            start, end = item.get("start_idx"), item.get("end_idx_exclusive")
            if type(start) is not int or type(end) is not int or end <= start:
                return None, len(value), False
            sizes.append(end - start)
            valid = valid and start == cursor and 1 <= end - start <= 4
            cursor = end
        return sizes, len(value), valid and cursor == candidate_count
    # This is the model's single response, not the saved multi-prompt envelope.
    return [candidate_count], 1, 0 < candidate_count <= 4


def timing_statistics(values: Sequence[float], absent: int = 0) -> Dict:
    ordered = sorted(values)
    total = round(sum(values), 6) if not absent else None
    return {
        "seconds_total": total,
        "seconds_known_total": round(sum(values), 6),
        "documents_timed": len(values),
        "documents_missing_time": absent,
        "seconds_mean": ratio(total, len(values)),
        "seconds_median": statistics.median(values) if values else None,
        "seconds_p95": ordered[math.ceil(len(values) * 0.95) - 1] if values else None,
        "seconds_min": min(values) if values else None,
        "seconds_max": max(values) if values else None,
    }


def saved_metric(summary, progress, key, diagnostics):
    left, right = summary.get(key), progress.get(key)
    if left is not None and right is not None and left != right:
        diagnostic(
            diagnostics,
            "summary_progress_mismatch",
            f"{key}: summary={left}, progress={right}; use summary",
        )
    value = left if left is not None else right
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"invalid nonnegative metric {key}: {value!r}")
    if key in {
        "critic_prompt_count",
        "critic_generation_batches",
        "critic_input_tokens",
        "critic_output_tokens",
        "critic_peak_gpu_memory_bytes",
    }:
        if int(value) != value:
            raise ValueError(f"{key}: expected an integer")
        return int(value)
    return value


def collect_run(variant: str, run_dir: Path, cache: Mapping) -> Dict:
    for filename in REQUIRED_FILES:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)
    summary = read_json(run_dir / "summary.json")
    progress = read_json(run_dir / "progress.json")
    state = read_json(run_dir / "run_state.json")
    if progress.get("finalized") is not True or progress.get("pending") != 0:
        raise ValueError(f"{run_dir}: run is not finalized or has pending documents")
    samples = index_rows(iter_jsonl(run_dir / "sample_results.jsonl"), run_dir)
    if not samples:
        raise ValueError(f"{run_dir}: empty sample_results.jsonl")
    if any(sample.get("status") not in {"ok", "failed"} for sample in samples.values()):
        raise ValueError(f"{run_dir}: unknown sample status")
    completed_ids = [key for key, sample in samples.items() if sample["status"] == "ok"]
    completed_set = set(completed_ids)
    failed = len(samples) - len(completed_ids)
    diagnostics = []
    if failed:
        diagnostic(
            diagnostics,
            "failed_documents",
            "存在失败文档；运行累计 token/时间包含尝试成本，每篇成本按成功文档摊销。",
            [key for key in samples if key not in completed_set],
        )
    for key, actual in (
        ("samples_requested", len(samples)),
        ("samples_completed", len(completed_ids)),
        ("samples_failed", failed),
    ):
        if summary.get(key) != actual:
            diagnostic(
                diagnostics,
                "sample_count_mismatch",
                f"{key}: summary={summary.get(key)}, sample_results={actual}",
            )
    runtime = dict(summary.get("runtime_config") or {})
    state_runtime = dict(state.get("runtime") or {})
    configured = runtime.get(
        "critic_candidate_set_size", state_runtime.get("critic_candidate_set_size")
    )
    if configured != variant:
        diagnostic(
            diagnostics,
            "variant_config_mismatch",
            f"目录标签 {variant}，运行配置 {configured}；保留目录标签。",
        )
    if (
        state_runtime.get("critic_candidate_set_size") is not None
        and state_runtime["critic_candidate_set_size"] != configured
    ):
        diagnostic(
            diagnostics, "state_config_mismatch", "summary 与 run_state 的候选集配置不一致。"
        )

    planned_sizes = {}
    planned_unknown = []
    for key, sample in samples.items():
        sizes = sample.get("critic_candidate_set_sizes")
        if sizes is None:
            planned_unknown.append(key)
        elif not isinstance(sizes, list) or any(
            type(size) is not int or not 1 <= size <= 4 for size in sizes
        ):
            raise ValueError(f"{run_dir}: {key}: invalid critic_candidate_set_sizes")
        else:
            planned_sizes[key] = sizes
    planned_distribution = distribution(size for sizes in planned_sizes.values() for size in sizes)
    if planned_unknown:
        diagnostic(
            diagnostics,
            "missing_planned_groups",
            "候选集计划分组缺失；完整计划 Prompt 数记为 null。",
            planned_unknown,
        )
    if summary.get("critic_candidate_set_size_counts") != planned_distribution:
        diagnostic(
            diagnostics,
            "planned_distribution_mismatch",
            "summary 候选集分布与 sample_results 分组汇总不一致。",
        )

    stage_acc = {
        stage: {
            "times": [],
            "missing_time": 0,
            "document_calls": 0,
            "known_prompts": 0,
            "missing_prompts": [],
            "sizes": [],
        }
        for stage in STAGES
    }
    documents = []
    seen = set()
    grouping_mismatches, invalid_boundaries, input_mismatches = [], [], []
    planned_coverage_mismatches = []
    missing_cache = [key for key in samples if cache.get(key, {}).get("status") != "ok"]
    record_statistics = {
        "initial_reviews_valid": 0,
        "initial_reviews_fallback": 0,
        "final_reviews_valid": 0,
        "final_reviews_fallback": 0,
        "initial_critic_parse_error_documents": 0,
        "final_critic_parse_error_documents": 0,
        "critic_parse_error_documents": 0,
        "stage_errors_total": 0,
    }
    extracted_total = 0
    for record in iter_jsonl(run_dir / "pipeline_records.jsonl"):
        key = record.get("document_id")
        if key in seen or key not in completed_set:
            raise ValueError(f"{run_dir}: duplicate or unexpected pipeline document {key}")
        seen.add(key)
        extracted = record.get("extracted_triples") or []
        extracted_total += len(extracted)
        if (
            key in planned_sizes
            and sum(planned_sizes[key]) != len(extracted)
            or samples[key].get("extracted_count", len(extracted)) != len(extracted)
        ):
            planned_coverage_mismatches.append(key)
        cached = cache.get(key, {})
        if cached.get("status") == "ok" and (
            cached["triple_count"] != len(extracted)
            or (
                extracted
                and cached["signature"]
                != proposal_signature(extracted, record.get("source_text", ""))
            )
        ):
            input_mismatches.append(key)
        raw = record.get("raw_responses") or {}
        timings = record.get("stage_timings_seconds") or {}
        doc = {
            "document_id": key,
            "input_triples": len(extracted),
            "planned_candidate_set_sizes": planned_sizes.get(key),
            "stages": {},
        }
        for stage in STAGES:
            acc = stage_acc[stage]
            candidate_count = (
                len(extracted)
                if stage == "initial_critic"
                else len(record.get("refined_triples") or [])
                + len(record.get("added_triples") or [])
            )
            enabled = (
                stage != "final_critic"
                or runtime.get("pipeline_variant", state_runtime.get("pipeline_variant"))
                != "proposer_critic_once"
            )
            if stage in {"initial_critic", "final_critic"}:
                if not enabled:
                    sizes, count, valid = [], 0, True
                    if raw.get(stage) or record.get("final_reviews"):
                        diagnostic(
                            diagnostics,
                            "disabled_stage_has_output",
                            f"{key}: final_critic 被配置为禁用，但包含输出；按输出统计。",
                        )
                        sizes, count, valid = raw_chunks(raw.get(stage), candidate_count)
                else:
                    sizes, count, valid = raw_chunks(raw.get(stage), candidate_count)
                active = bool(raw.get(stage)) or (enabled and candidate_count > 0)
                if count is None:
                    acc["missing_prompts"].append(key)
                else:
                    acc["known_prompts"] += count
                if sizes is not None:
                    acc["sizes"].extend(sizes)
                if not valid:
                    invalid_boundaries.append(f"{key}:{stage}")
                if (
                    stage == "initial_critic"
                    and key in planned_sizes
                    and sizes != planned_sizes[key]
                ):
                    grouping_mismatches.append(key)
                doc["stages"][stage] = {
                    "observed_prompt_count": count,
                    "observed_candidate_set_sizes": sizes,
                }
            else:
                active = bool(raw.get(stage))
                doc["stages"][stage] = {}
            acc["document_calls"] += int(active)
            seconds = timings.get(stage)
            if seconds is not None:
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (float, int))
                    or seconds < 0
                    or not math.isfinite(seconds)
                ):
                    raise ValueError(f"{run_dir}: {key}: invalid {stage} time")
                if active:
                    acc["times"].append(seconds)
            elif active:
                acc["missing_time"] += 1
            doc["stages"][stage]["elapsed_seconds"] = seconds
        for prefix in ("initial", "final"):
            reviews = record.get(f"{prefix}_reviews") or []
            record_statistics[f"{prefix}_reviews_valid"] += sum(
                item.get("valid") is True for item in reviews
            )
            record_statistics[f"{prefix}_reviews_fallback"] += sum(
                item.get("valid") is not True for item in reviews
            )
            error_prefix = "initial_review:" if prefix == "initial" else "final_validation:"
            record_statistics[f"{prefix}_critic_parse_error_documents"] += int(
                any(
                    str(error).startswith(error_prefix)
                    for error in record.get("stage_errors") or []
                )
            )
        record_statistics["stage_errors_total"] += len(record.get("stage_errors") or [])
        record_statistics["critic_parse_error_documents"] += int(
            any(
                str(error).startswith(("initial_review:", "final_validation:"))
                for error in record.get("stage_errors") or []
            )
        )
        documents.append(doc)

    absent_records = [key for key in completed_ids if key not in seen]
    if absent_records:
        diagnostic(
            diagnostics,
            "missing_pipeline_records",
            "成功文档缺少 pipeline record；观测 Prompt 总数不完整。",
            absent_records,
        )
        stage_acc["initial_critic"]["missing_prompts"].extend(absent_records)
        stage_acc["initial_critic"]["missing_time"] += len(absent_records)
        if (
            runtime.get("pipeline_variant", state_runtime.get("pipeline_variant"))
            != "proposer_critic_once"
        ):
            stage_acc["final_critic"]["missing_prompts"].extend(absent_records)
            stage_acc["final_critic"]["missing_time"] += len(absent_records)
    if planned_coverage_mismatches:
        diagnostic(
            diagnostics,
            "planned_candidate_coverage_mismatch",
            "候选集计划或 sample 的 extracted_count 与原始三元组数量不一致。",
            planned_coverage_mismatches,
        )
    if grouping_mismatches:
        diagnostic(
            diagnostics,
            "planned_observed_grouping_mismatch",
            "计划候选分组与原始响应区间不一致；不能将计划 Prompt 数作为实际调用数。",
            grouping_mismatches,
        )
    if invalid_boundaries:
        diagnostic(
            diagnostics,
            "invalid_response_boundaries",
            "原始响应区间存在缺口、越界或分组信息不可恢复。",
            invalid_boundaries,
        )
    if missing_cache:
        diagnostic(
            diagnostics, "missing_cache_documents", "共享缓存缺少成功 proposal。", missing_cache
        )
    if input_mismatches:
        diagnostic(
            diagnostics,
            "proposal_input_mismatch",
            "输入三元组或原文与共享缓存不一致。",
            input_mismatches,
        )
    for field, actual in record_statistics.items():
        if summary.get(field) is not None and summary[field] != actual:
            diagnostic(
                diagnostics,
                "stability_summary_mismatch",
                f"{field}: summary={summary[field]}, records={actual}",
            )

    metrics = {key: saved_metric(summary, progress, key, diagnostics) for key in RUNTIME_METRICS}
    missing_metrics = [key for key in RUNTIME_METRICS[:6] if metrics[key] is None]
    if missing_metrics:
        diagnostic(
            diagnostics,
            "missing_runtime_metrics",
            f"运行指标缺失：{', '.join(missing_metrics)}；不估算缺失 token/时间。",
        )
    stages = {}
    for stage, acc in stage_acc.items():
        timing = timing_statistics(acc["times"], acc["missing_time"])
        if acc["missing_time"]:
            diagnostic(
                diagnostics,
                "missing_stage_time",
                f"{stage}: {acc['missing_time']} 个文档缺少阶段时间。",
            )
        observed = None if acc["missing_prompts"] else acc["known_prompts"]
        stages[stage] = {
            "document_calls": acc["document_calls"],
            "observed_prompt_count": observed if "critic" in stage else None,
            "observed_prompt_count_known": acc["known_prompts"] if "critic" in stage else None,
            "documents_missing_prompt_data": len(acc["missing_prompts"]),
            "observed_candidate_set_size_counts": distribution(acc["sizes"]),
            **timing,
        }
        recorded_total = (summary.get("stage_elapsed_seconds_total") or {}).get(stage)
        if (
            recorded_total is not None
            and timing["seconds_total"] is not None
            and abs(recorded_total - timing["seconds_total"]) > 0.002
        ):
            diagnostic(
                diagnostics,
                "stage_time_summary_mismatch",
                f"{stage}: summary={recorded_total}, records={timing['seconds_total']}",
            )
    initial_count = stages["initial_critic"]["observed_prompt_count"]
    final_count = stages["final_critic"]["observed_prompt_count"]
    observed = (
        initial_count + final_count
        if initial_count is not None and final_count is not None
        else None
    )
    runtime_count = metrics["critic_prompt_count"]
    if observed is not None and runtime_count is not None and observed != runtime_count:
        diagnostic(
            diagnostics,
            "runtime_observed_prompt_mismatch",
            f"运行累计 Prompt={runtime_count}，现存原始响应 Prompt={observed}；可能含重试/恢复或缺失记录，不强行合并。",
        )
    planned_count = sum(map(len, planned_sizes.values())) if not planned_unknown else None
    if (
        planned_count is not None
        and final_count == 0
        and runtime_count is not None
        and planned_count != runtime_count
    ):
        diagnostic(
            diagnostics,
            "planned_runtime_prompt_mismatch",
            f"计划 Prompt={planned_count}，运行累计 Prompt={runtime_count}。",
        )
    if (
        stages["initial_critic"]["documents_missing_prompt_data"]
        or stages["final_critic"]["documents_missing_prompt_data"]
    ):
        diagnostic(
            diagnostics,
            "missing_observed_prompts",
            "部分 Critic 原始响应缺失；观测总数记为 null，并保留已知数量。",
        )

    cache_hits = summary.get("proposal_cache_hits", progress.get("proposal_cache_hits"))
    cache_only = runtime.get(
        "require_proposal_cache", state_runtime.get("require_proposal_cache")
    ) is True or cache_hits == len(samples)
    stages["proposer"]["api_calls_current_run"] = 0 if cache_only else None
    stages["proposer"]["cached_documents"] = cache_hits
    stages["proposer"]["historical_proposal_api_attempts"] = summary.get(
        "proposal_api_attempts", progress.get("proposal_api_attempts")
    )
    stages["proposer"]["timing_scope"] = (
        "historical_cached_proposals" if cache_only else "saved_records"
    )
    stages["integrator"]["api_calls_in_saved_records"] = stages["integrator"]["document_calls"]
    stages["initial_critic"]["generation_batches"] = (
        metrics["critic_generation_batches"] if final_count == 0 else None
    )
    stages["final_critic"]["generation_batches"] = 0 if final_count == 0 else None
    for stage in ("initial_critic", "final_critic"):
        stats = stages[stage]
        stats["prompts_per_document"] = ratio(stats["observed_prompt_count"], len(completed_ids))
        stats["prompts_per_stage_second"] = ratio(
            stats["observed_prompt_count"], stats["seconds_total"]
        )
        stats["seconds_per_prompt_amortized"] = ratio(
            stats["seconds_total"], stats["observed_prompt_count"]
        )

    input_tokens, output_tokens = metrics["critic_input_tokens"], metrics["critic_output_tokens"]
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    generation_seconds = metrics["critic_generation_seconds"]
    all_critic_stage_seconds = stages["initial_critic"]["seconds_total"]
    final_seconds = stages["final_critic"]["seconds_total"]
    all_critic_stage_seconds = (
        all_critic_stage_seconds + final_seconds
        if all_critic_stage_seconds is not None and final_seconds is not None
        else None
    )
    row = {
        "variant": variant,
        "run_dir": str(run_dir),
        "configured_variant": configured,
        "pipeline_variant": runtime.get("pipeline_variant", state_runtime.get("pipeline_variant")),
        "adapter_path": (
            runtime.get("critic_runtime") or state_runtime.get("critic_runtime") or {}
        ).get("adapter_path"),
        "documents_total": len(samples),
        "documents_completed": len(completed_ids),
        "documents_failed": failed,
        "pipeline_records": len(seen),
        "input_triples": extracted_total,
        "shared_input_verified": not (
            missing_cache or input_mismatches or absent_records or failed
        ),
        "document_ids": list(samples),
        "planned_prompt_count": planned_count,
        "planned_prompt_count_known": sum(map(len, planned_sizes.values())),
        "planned_prompts_per_document": ratio(planned_count, len(samples)),
        "planned_candidate_set_size_counts": planned_distribution,
        "observed_prompt_count": observed,
        "observed_prompts_per_document": ratio(observed, len(completed_ids)),
        "observed_candidate_set_size_counts": distribution(
            stage_acc["initial_critic"]["sizes"] + stage_acc["final_critic"]["sizes"]
        ),
        "runtime_prompt_count": runtime_count,
        "generation_batches": metrics["critic_generation_batches"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_per_document": ratio(input_tokens, len(completed_ids)),
        "output_tokens_per_document": ratio(output_tokens, len(completed_ids)),
        "total_tokens_per_document": ratio(total_tokens, len(completed_ids)),
        "input_tokens_per_runtime_prompt": ratio(input_tokens, runtime_count),
        "output_tokens_per_runtime_prompt": ratio(output_tokens, runtime_count),
        "generation_seconds": generation_seconds,
        "generation_seconds_per_document": ratio(generation_seconds, len(completed_ids)),
        "generation_seconds_per_prompt_amortized": ratio(generation_seconds, runtime_count),
        "generation_seconds_per_batch_mean": ratio(
            generation_seconds, metrics["critic_generation_batches"]
        ),
        "critic_stage_seconds_total": all_critic_stage_seconds,
        "critic_stage_seconds_per_completed_document": ratio(
            all_critic_stage_seconds, len(completed_ids)
        ),
        "active_wall_seconds": metrics["active_wall_seconds"],
        "active_seconds_per_completed_document": ratio(
            metrics["active_wall_seconds"], len(completed_ids)
        ),
        "model_load_seconds": metrics["model_load_seconds"],
        "runtime_prompts_per_generation_second": ratio(runtime_count, generation_seconds),
        "output_tokens_per_generation_second": ratio(output_tokens, generation_seconds),
        "total_tokens_per_generation_second": ratio(total_tokens, generation_seconds),
        "documents_per_active_minute": ratio(
            len(completed_ids) * 60, metrics["active_wall_seconds"]
        ),
        "input_triples_per_critic_stage_second": ratio(extracted_total, all_critic_stage_seconds),
        "input_triples_per_generation_second": ratio(extracted_total, generation_seconds),
        "peak_gpu_memory_bytes": metrics["critic_peak_gpu_memory_bytes"],
        "stages": stages,
        "stability": record_statistics,
        "diagnostics": diagnostics,
        "documents": documents,
    }
    row["stability"]["initial_fallback_rate"] = ratio(
        record_statistics["initial_reviews_fallback"],
        record_statistics["initial_reviews_valid"] + record_statistics["initial_reviews_fallback"],
    )
    row["stability"]["initial_parse_error_document_rate"] = ratio(
        record_statistics["initial_critic_parse_error_documents"], len(completed_ids)
    )
    row["stability"]["final_fallback_rate"] = ratio(
        record_statistics["final_reviews_fallback"],
        record_statistics["final_reviews_valid"] + record_statistics["final_reviews_fallback"],
    )
    row["stability"]["final_parse_error_document_rate"] = ratio(
        record_statistics["final_critic_parse_error_documents"], len(completed_ids)
    )
    return row


def collect_report(runs: Mapping[str, Path], proposal_cache_dir: Path) -> Dict:
    if set(runs) != set(VARIANTS):
        raise ValueError(f"exactly these variants are required: {VARIANTS}")
    cache = read_proposal_cache(proposal_cache_dir)
    rows = [collect_run(variant, runs[variant], cache) for variant in VARIANTS]
    same_ids = all(row["document_ids"] == rows[0]["document_ids"] for row in rows)
    diagnostics = []
    if not same_ids:
        diagnostic(
            diagnostics,
            "variant_document_ids_mismatch",
            "四个运行的文档 ID 或顺序不同，不能视为同一输入比较。",
        )
    verified = same_ids and all(row["shared_input_verified"] for row in rows)
    return {
        "schema_version": 1,
        "dataset": "Re-DocRED",
        "proposal_cache_dir": str(proposal_cache_dir),
        "shared_input_verified": verified,
        "shared_documents": len(rows[0]["document_ids"]) if verified else None,
        "shared_input_triples": rows[0]["input_triples"] if verified else None,
        "definitions": {
            "planned_prompt_count": "sample_results 中记录的初始 Critic 候选分组总数；不根据目录标签改写。",
            "observed_prompt_count": "成功文档原始响应中的初始/最终 Critic Prompt 数；包括无效响应，不按有效 review 数计数。",
            "runtime_metrics": "运行保存的累计 token、生成批次、生成时间；可能包含重试/恢复成本。缺失项用 null，不重新估算。",
            "per_document": "计划均值按所有选中文档；实际观测和运行成本均值按成功文档。失败/缺失时查看诊断与已知部分。",
            "token_count": "本地 tokenizer 的实际输入（含聊天模板，不含 padding）及生成输出计数；输出沿用运行时 pad/EOS 处理规则。",
            "generation_time": "本地模型 generate 与 CUDA 同步时间，排除加载、prompt 构建、tokenization、解码与验证。",
            "stage_time": "逐文档阶段计时；跨文档 batch 的时间可能按 prompt 数分摊，不能作为独立请求延迟。",
            "latency": "seconds_per_prompt_amortized 是批量运行总时间/Prompt 数，非实测单 Prompt 响应延迟；批次均值为生成时间/生成批数。",
            "throughput": "Prompt 与 token 吞吐使用累计运行计数/纯生成时间；文档吞吐使用成功文档/active_wall_seconds。",
            "stage_calls": "Critic 分别报告文档级请求、Prompt 数及 GPU 生成批次；GPU 批次不等于 Prompt 数。Integrator 调用为保存记录数量，未恢复 API 重试。",
            "proposer": "缓存复用时当前 Proposer API 调用为 0；缓存中的原始 Proposer 时间/尝试数是历史成本，排除于 Critic 效率。",
            "limitations": "原始响应区间只能证明实际观测分组，不能证明历史运行忽略配置的原因；不同 adapter 的结果不证明候选集大小的因果效果。",
        },
        "diagnostics": diagnostics,
        "variants": rows,
    }


def flatten_row(row: Mapping) -> Dict:
    excluded = {"document_ids", "documents", "stages", "stability", "diagnostics"}
    flat = {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, dict)
        else value
        for key, value in row.items()
        if key not in excluded
    }
    for stage, values in row["stages"].items():
        for key, value in values.items():
            if not isinstance(value, dict):
                flat[f"{stage}_{key}"] = value
    flat.update({f"stability_{key}": value for key, value in row["stability"].items()})
    flat["diagnostics"] = json.dumps(row["diagnostics"], ensure_ascii=False)
    return flat


def format_value(value) -> str:
    if value is None:
        return "—"
    if type(value) is int:
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers, values) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(format_value(value) for value in row) + " |" for row in values),
        ]
    )


def render_markdown(report: Mapping) -> str:
    rows = report["variants"]
    lines = ["# Re-DocRED Critic 效率统计", ""]
    if report["shared_input_verified"]:
        lines.append(
            f"四个运行已核验共享输入：{report['shared_documents']:,} 篇文档、{report['shared_input_triples']:,} 个原始三元组。变体名称沿用目录标签。"
        )
    else:
        lines.append("共享输入未通过完整核验，详见诊断。变体名称沿用目录标签。")
    lines.extend(
        [
            "",
            "## Prompt 与 Token",
            "",
            "计划 Prompt 数来自保存的候选分组，实际 Prompt 数来自原始响应区间，两者分别报告。每篇计划值按选中文档计算，其余每篇成本按成功文档摊销。",
            "",
            markdown_table(
                [
                    "变体",
                    "记录配置",
                    "成功/失败文档",
                    "计划 Prompt",
                    "计划/篇",
                    "实际 Prompt",
                    "实际/篇",
                    "GPU 批次",
                    "输入 Token",
                    "输出 Token",
                    "总 Token",
                ],
                [
                    [
                        row["variant"],
                        row["configured_variant"],
                        f"{row['documents_completed']}/{row['documents_failed']}",
                        row["planned_prompt_count"],
                        row["planned_prompts_per_document"],
                        row["observed_prompt_count"],
                        row["observed_prompts_per_document"],
                        row["generation_batches"],
                        row["input_tokens"],
                        row["output_tokens"],
                        row["total_tokens"],
                    ]
                    for row in rows
                ],
            ),
            "",
            markdown_table(
                ["变体", "输入 Token/篇", "输出 Token/篇", "总 Token/篇"],
                [
                    [
                        row["variant"],
                        row["input_tokens_per_document"],
                        row["output_tokens_per_document"],
                        row["total_tokens_per_document"],
                    ]
                    for row in rows
                ],
            ),
            "",
            "## 推理时间与吞吐",
            "",
            "纯生成时间排除模型加载；阶段时间来自逐文档 initial_critic/final_critic。平均秒/Prompt 为摊销成本，不是实测单请求延迟。Token/s 为输出 Token/纯生成秒；篇/分钟为成功文档/活动墙钟时间。",
            "",
            markdown_table(
                [
                    "变体",
                    "纯生成秒",
                    "Critic 阶段秒",
                    "阶段秒/篇",
                    "生成秒/篇",
                    "生成秒/Prompt（摊销）",
                    "生成秒/批次",
                    "Prompt/s",
                    "输出 Token/s",
                    "总 Token/s",
                    "篇/分钟",
                    "输入三元组/阶段秒",
                ],
                [
                    [
                        row["variant"],
                        row["generation_seconds"],
                        row["critic_stage_seconds_total"],
                        row["critic_stage_seconds_per_completed_document"],
                        row["generation_seconds_per_document"],
                        row["generation_seconds_per_prompt_amortized"],
                        row["generation_seconds_per_batch_mean"],
                        row["runtime_prompts_per_generation_second"],
                        row["output_tokens_per_generation_second"],
                        row["total_tokens_per_generation_second"],
                        row["documents_per_active_minute"],
                        row["input_triples_per_critic_stage_second"],
                    ]
                    for row in rows
                ],
            ),
            "",
            markdown_table(
                ["变体", "活动墙钟秒", "模型加载秒", "初始阶段中位秒/篇", "初始阶段 P95 秒/篇"],
                [
                    [
                        row["variant"],
                        row["active_wall_seconds"],
                        row["model_load_seconds"],
                        row["stages"]["initial_critic"]["seconds_median"],
                        row["stages"]["initial_critic"]["seconds_p95"],
                    ]
                    for row in rows
                ],
            ),
            "",
            "## 各阶段调用",
            "",
            "本地 Critic 的一个 Prompt 可处理多个三元组，一个 GPU 批次可生成多个 Prompt。Proposer 历史缓存成本不计入当前 Critic 运行。Integrator 数字是保存的请求记录数，API 重试没有逐次记录。",
            "",
            markdown_table(
                [
                    "变体",
                    "当前 Proposer API",
                    "缓存文档",
                    "初始 Critic 文档请求",
                    "初始 Critic Prompt",
                    "Integrator 请求记录",
                    "最终 Critic 文档请求",
                    "最终 Critic Prompt",
                ],
                [
                    [
                        row["variant"],
                        row["stages"]["proposer"]["api_calls_current_run"],
                        row["stages"]["proposer"]["cached_documents"],
                        row["stages"]["initial_critic"]["document_calls"],
                        row["stages"]["initial_critic"]["observed_prompt_count"],
                        row["stages"]["integrator"]["document_calls"],
                        row["stages"]["final_critic"]["document_calls"],
                        row["stages"]["final_critic"]["observed_prompt_count"],
                    ]
                    for row in rows
                ],
            ),
            "",
            "## 稳定性",
            "",
            markdown_table(
                [
                    "变体",
                    "初始解析失败文档",
                    "有效 Review",
                    "Fallback",
                    "Fallback 比例",
                    "阶段错误数",
                ],
                [
                    [
                        row["variant"],
                        row["stability"]["initial_critic_parse_error_documents"],
                        row["stability"]["initial_reviews_valid"],
                        row["stability"]["initial_reviews_fallback"],
                        f"{row['stability']['initial_fallback_rate'] * 100:.2f}%"
                        if row["stability"]["initial_fallback_rate"] is not None
                        else None,
                        row["stability"]["stage_errors_total"],
                    ]
                    for row in rows
                ],
            ),
            "",
            "## 一致性核验",
            "",
        ]
    )
    for row in rows:
        planned = ", ".join(
            f"size-{size} × {count:,}"
            for size, count in row["planned_candidate_set_size_counts"].items()
        )
        observed = ", ".join(
            f"size-{size} × {count:,}"
            for size, count in row["observed_candidate_set_size_counts"].items()
        )
        lines.extend(
            [
                f"### {row['variant']}",
                "",
                f"- 计划分组：{planned or '缺失'}。",
                f"- 实际观测分组：{observed or '缺失'}。",
            ]
        )
        for item in row["diagnostics"]:
            lines.append(
                f"- `{item['code']}`：{item['message']}"
                + (
                    f" 涉及 {item['document_count']} 篇；示例：{', '.join(item['example_document_ids'])}。"
                    if item["document_count"]
                    else ""
                )
            )
        if not row["diagnostics"]:
            lines.append("- 未发现数据不一致。")
        lines.append("")
    for item in report["diagnostics"]:
        lines.append(f"- `{item['code']}`：{item['message']}")
    lines.extend(
        [
            "",
            "原始响应区间与运行时累计 Prompt 交叉核验；与计划不同的运行不能用于证明候选集大小带来的效率提升。这里只描述已保存运行，无法由现有记录确定历史运行为何没有采用计划分组。缺失指标显示为 —（JSON 中为 null），不重新执行推理或估算 token。",
            "",
            "## 数据来源与统计口径",
            "",
            f"- 共享缓存：`{report['proposal_cache_dir']}`。",
        ]
    )
    lines.extend(f"- {row['variant']}：`{row['run_dir']}`。" for row in rows)
    lines.append("")
    lines.extend(f"- **{key}**：{value}" for key, value in report["definitions"].items())
    return "\n".join(lines) + "\n"


def write_report(report: Mapping, output_prefix: Path) -> Sequence[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(output_prefix) + suffix) for suffix in (".json", ".csv", ".md")]
    paths[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    flat_rows = [
        {
            **flatten_row(row),
            "shared_input_verified_across_variants": report["shared_input_verified"],
        }
        for row in report["variants"]
    ]
    fieldnames = list(dict.fromkeys(key for row in flat_rows for key in row))
    with paths[1].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    paths[2].write_text(render_markdown(report), encoding="utf-8")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report saved Re-DocRED Critic efficiency without running models or reading gold triples."
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--run", action="append", default=[], help="VARIANT=RUN_DIR; override any or all four runs"
    )
    parser.add_argument("--proposal-cache-dir", help="default: RUN_ROOT/proposal_cache")
    parser.add_argument(
        "--output-prefix", default=str(DEFAULT_OUTPUT_PREFIX), help="output stem for .json/.csv/.md"
    )
    args = parser.parse_args(argv)
    try:
        run_root = resolve_path(args.run_root)
        runs = select_runs(run_root, args.run)
        cache_dir = (
            resolve_path(args.proposal_cache_dir)
            if args.proposal_cache_dir
            else run_root / "proposal_cache"
        )
        report = collect_report(runs, cache_dir)
        paths = write_report(report, resolve_path(args.output_prefix))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for row in report["variants"]:
        print(
            f"{row['variant']}: planned={row['planned_prompt_count']} observed={row['observed_prompt_count']} diagnostics={len(row['diagnostics'])}"
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
