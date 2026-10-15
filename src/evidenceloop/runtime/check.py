from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from evidenceloop.data.sft import iter_jsonl  # noqa: E402
from evidenceloop.training.evaluate import validate_prediction_target  # noqa: E402
from evidenceloop.types import Triple  # noqa: E402

ROOT = Path.cwd()


DEFAULT_TEST_FILE = ROOT / "data" / "critic_sft" / "rebel_phase2" / "test.jsonl"


def first_record(path: Path):
    try:
        return next(iter(iter_jsonl(path)))
    except StopIteration as exc:
        raise ValueError(f"test file contains no records: {path}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the merged direct Critic and validate one real SFT record."
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt-batch-size", type=int, default=None)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.base_model:
        os.environ["CRITIC_BASE_MODEL"] = args.base_model
    if args.adapter_path:
        os.environ["CRITIC_ADAPTER_PATH"] = args.adapter_path
    if args.device:
        os.environ["CRITIC_DEVICE"] = args.device
    if args.prompt_batch_size is not None:
        os.environ["CRITIC_PROMPT_BATCH_SIZE"] = str(args.prompt_batch_size)

    try:
        from evidenceloop import Pipeline

        critic = Pipeline().critic

        runtime = critic.validate_runtime()
        record = first_record(args.test_file)
        triples = [
            Triple.from_dict(
                item["triple"],
                source_text=record["source_text"],
                document_id=record["document_id"],
            )
            for item in record["triples_info"]
        ]
        origins = [item.get("origin", "proposer") for item in record["triples_info"]]
        started = time.perf_counter()
        payload = critic.review_triples_batch(
            triples,
            stage=record["review_stage"],
            origins=origins,
        )
        target = {"results": payload["results"]}
        validate_prediction_target(record, target)
        if len(target["results"]) != len(triples):
            raise ValueError("direct Critic result count does not match candidate count")
        expected_indices = list(range(len(triples)))
        actual_indices = sorted(item["triple_idx"] for item in target["results"])
        if actual_indices != expected_indices:
            raise ValueError(
                f"direct Critic indices are incomplete: expected={expected_indices}, "
                f"actual={actual_indices}"
            )
        metrics = critic.llm.metrics()
        summary = {
            "ok": True,
            "runtime": runtime,
            "document_id": record["document_id"],
            "candidate_count": len(triples),
            "result_count": len(target["results"]),
            "thinking_enabled": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "metrics": metrics,
        }
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
