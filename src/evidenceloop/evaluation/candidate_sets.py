from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.documents import load_documents
from evidenceloop.data.registry import candidate_strategy, variant_directory
from evidenceloop.experiments.checkpoints import (  # noqa: E402
    _deserialize_proposer_payload,
    _resolve_repo_path,
    critic_chunk_sizes,
    load_graphjudge_rebel,
    load_proposal_checkpoints,
    pipeline_record_from_dict,
)
from evidenceloop.reports.runs import load_summary

ROOT = Path.cwd()


VARIANTS = ("fixed1", "fixed2", "fixed4", "variable")
SOFT_FIELDS = (
    "G-BS-Acc",
    "G-BS-Recall",
    "G-BS-F1",
    "G-BL-Acc",
    "G-BL-Recall",
    "G-BL-F1",
    "G-RO-Acc",
    "G-RO-Recall",
    "G-RO-F1",
)
NOMINAL_TRAINING_SET_SIZE = {
    "fixed1": "1",
    "fixed2": "2",
    "fixed4": "4",
    "variable": "1-4",
}


def read_json(path: Path) -> Dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                rows.append(value)
    return rows


def _bootstrap_delta(
    left: Mapping[str, float], right: Mapping[str, float], *, seed: int = 42, samples: int = 2000
) -> Dict:
    ids = [document_id for document_id in left if document_id in right]
    if not ids:
        return {"n": 0, "delta": 0.0, "ci95": [0.0, 0.0]}
    differences = [float(left[item]) - float(right[item]) for item in ids]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    means.sort()
    return {
        "n": len(differences),
        "delta": sum(differences) / len(differences),
        "ci95": [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]],
    }


def _summary_row(variant: str, run_dir: Path) -> Tuple[Dict, Dict[str, float]]:
    summary = load_summary(run_dir)
    soft = {}
    soft_path = run_dir / "graphjudge_metrics.json"
    if soft_path.exists():
        soft = dict(read_json(soft_path).get("metrics") or {})
    missing_metrics = [name for name in SOFT_FIELDS if name not in soft]
    if missing_metrics:
        raise ValueError(f"{soft_path}: missing GraphJudge metrics: {', '.join(missing_metrics)}")
    non_finite = [name for name in SOFT_FIELDS if not math.isfinite(float(soft[name]))]
    if non_finite:
        raise ValueError(f"{soft_path}: non-finite GraphJudge metrics: {non_finite}")
    sizes = dict(summary.get("critic_candidate_set_size_counts") or {})
    completed = int(summary.get("samples_completed", 0) or 0)
    prompt_count = int(
        (summary.get("critic_generation") or {}).get("prompt_count", 0)
        or summary.get("critic_prompt_count", 0)
        or 0
    )
    if not prompt_count:
        prompt_count = sum(int(value) for value in sizes.values())
    row = {
        "variant": variant,
        "run_dir": str(run_dir),
        "training_set_size": NOMINAL_TRAINING_SET_SIZE[variant],
        "training_set_records": None,
        "inference_set_size": variant,
        "effective_size_distribution": dict(sizes),
        **{name: soft.get(name) for name in SOFT_FIELDS},
        "Exact-F1": summary.get("exact_match_f1"),
        "Exact-Precision": summary.get("exact_match_precision"),
        "Exact-Recall": summary.get("exact_match_recall"),
        "order_consistency": None,
        "order_check_prompts": 0,
        "prompts_per_doc": prompt_count / completed if completed else 0.0,
        "samples_completed": completed,
        "samples_failed": int(summary.get("samples_failed", 0) or 0),
        "predicted_triples": int(summary.get("predicted_spo_total", 0) or 0),
        "gold_triples": int(summary.get("gold_triples_total", 0) or 0),
        "critic_parse_error_documents": int(summary.get("critic_parse_error_documents", 0) or 0),
        "stage_errors_total": int(summary.get("stage_errors_total", 0) or 0),
        "critic_tokens_per_second": (summary.get("critic_tokens_per_second") or 0.0),
    }
    return row, soft


def _decision(review) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    output = getattr(review, "output_triple", None)
    if output is None:
        return str(getattr(review, "action", "")), None
    return (
        str(getattr(review, "action", "")),
        (
            str(output.subject),
            str(output.predicate),
            str(output.object),
        ),
    )


def order_consistency(
    run_dir: Path, proposal_cache_dir: Optional[Path], batch_documents: int = 8
) -> Dict:
    state = read_json(run_dir / "run_state.json")
    runtime = dict(state.get("runtime") or {})
    variant = str(
        runtime.get("candidate_grouping")
        or runtime.get("candidate_set")
        or runtime.get("critic_candidate_set_size")
        or "max4"
    )
    candidate_seed = int(runtime.get("seed", runtime.get("critic_candidate_set_seed", 42)) or 42)
    data = dict(state.get("data") or {})
    if state.get("schema_version") == 2:
        rows = [d.__dict__ for d in load_documents(str(run_dir / "documents.jsonl"))]
    else:
        rows = load_graphjudge_rebel(
            _resolve_repo_path(data["data_dir"]), int(data["requested_limit"])
        )
    cache_value = proposal_cache_dir or runtime.get("proposal_cache_dir")
    if not cache_value:
        raise ValueError(
            "order consistency requires --proposal-cache-dir or a saved proposal cache"
        )
    proposals = load_proposal_checkpoints(_resolve_repo_path(cache_value), rows)
    missing = [index for index in range(len(rows)) if index not in proposals]
    failed = [index for index, value in proposals.items() if value.get("status") != "ok"]
    if missing or failed:
        raise ValueError(
            "order consistency proposal cache is incomplete; "
            f"missing={missing[:5]} failed={failed[:5]}"
        )
    pipeline_rows = {
        row["document_id"]: pipeline_record_from_dict(row)
        for row in read_jsonl(run_dir / "pipeline_records.jsonl")
    }

    from evidenceloop.agents.critic import CriticAgent
    from evidenceloop.pipeline.core import Pipeline
    from evidenceloop.runtime.direct_critic import DirectCriticEngine

    critic_runtime = dict(runtime.get("critic_runtime") or {})
    engine = DirectCriticEngine(critic_runtime)
    critic = CriticAgent(
        name="OrderCheckCritic",
        role="Critic",
        llm=engine,
        runtime_settings=critic_runtime,
    )
    controller = Pipeline(
        proposer_agent=object(),
        critic_agent=critic,
        pipeline_variant="proposer_critic_once",
    )

    total = 0
    consistent = 0
    prompt_count = 0
    for start in range(0, len(rows), batch_documents):
        batch = rows[start : start + batch_documents]
        requests = []
        triples_by_id = {}
        for row in batch:
            proposal = _deserialize_proposer_payload(proposals[rows.index(row)]["payload"], row)
            triples = list(proposal["triples"])
            triples_by_id[row["document_id"]] = triples
            chunk_sizes = critic_chunk_sizes(
                len(triples), variant, seed=candidate_seed, document_id=row["document_id"]
            )
            prompt_count += (
                len(chunk_sizes) if chunk_sizes is not None else ((len(triples) + 3) // 4)
            )
            requests.append(
                {
                    "triples": list(reversed(triples)),
                    "stage": "initial_review",
                    "origins": ["proposer"] * len(triples),
                    "chunk_sizes": chunk_sizes,
                }
            )
        outputs = critic.review_documents_batch(requests)
        for row, output in zip(batch, outputs):
            record = pipeline_rows.get(row["document_id"])
            if record is None:
                continue
            errors = []
            reversed_reviews = controller._review_records(
                list(reversed(triples_by_id[row["document_id"]])),
                ["proposer"] * len(triples_by_id[row["document_id"]]),
                output["payload"],
                "initial_review",
                errors,
            )
            original_reviews = list(record.initial_reviews)
            for reversed_index, review in enumerate(reversed_reviews):
                original_index = len(original_reviews) - 1 - reversed_index
                if original_index < 0:
                    continue
                total += 1
                if _decision(review) == _decision(original_reviews[original_index]):
                    consistent += 1
    return {
        "consistent": consistent,
        "total": total,
        "rate": consistent / total if total else 0.0,
        "order_check_prompts": prompt_count,
    }


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--run must use VARIANT=RUN_DIR")
    variant, path = value.split("=", 1)
    variant = candidate_strategy(variant)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return variant, _resolve_repo_path(path).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Re-DocRED candidate-set ablation results."
    )
    parser.add_argument(
        "--run", action="append", required=True, help="VARIANT=RUN_DIR; repeat four times"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--order-check", action="store_true")
    parser.add_argument("--proposal-cache-dir", default=None)
    parser.add_argument(
        "--sft-root",
        default="data/critic_sft/re_docred_candidate_set_ablation",
        help="root containing the four variant metadata.json files",
    )
    args = parser.parse_args(argv)
    try:
        runs = dict(parse_run(value) for value in args.run)
        if set(runs) != set(VARIANTS):
            raise ValueError(f"exactly these variants are required: {VARIANTS}")
        rows = []
        soft_by_variant = {}
        sft_root = _resolve_repo_path(args.sft_root)
        for variant in VARIANTS:
            row, soft = _summary_row(variant, runs[variant])
            sft_metadata_path = variant_directory(sft_root, variant) / "metadata.json"
            if sft_metadata_path.exists():
                sft_metadata = read_json(sft_metadata_path)
                train_stats = dict((sft_metadata.get("splits") or {}).get("train") or {})
                row["training_set_records"] = train_stats.get("records")
                row["training_candidates"] = train_stats.get("candidates")
            if args.order_check:
                consistency = order_consistency(
                    runs[variant],
                    _resolve_repo_path(args.proposal_cache_dir)
                    if args.proposal_cache_dir
                    else None,
                )
                row["order_consistency"] = consistency["rate"]
                row["order_check_prompts"] = consistency["order_check_prompts"]
                row["order_consistency_detail"] = consistency
            rows.append(row)
            per_sample_path = runs[variant] / "graphjudge_metrics_per_sample.jsonl"
            if per_sample_path.exists():
                soft_by_variant[variant] = {
                    item["document_id"]: float(item.get("G-BS-F1", 0.0) or 0.0)
                    for item in read_jsonl(per_sample_path)
                }

        variable = soft_by_variant.get("variable", {})
        bootstrap = {
            variant: _bootstrap_delta(soft_by_variant.get(variant, {}), variable)
            for variant in VARIANTS
            if variant != "variable"
        }
        result = {
            "dataset": "Re-DocRED",
            "evaluation": "candidate-set ablation",
            "variants": rows,
            "paired_bootstrap_gbs_f1_vs_variable": bootstrap,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        print(text, end="")
        if args.output:
            output = _resolve_repo_path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
