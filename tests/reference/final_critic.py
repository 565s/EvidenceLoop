"""Run only the final local Critic over completed offline Integrator runs.

The input candidates are the Critic1-refined proposer triples followed by accepted
Integrator additions. This script never calls online agents or evaluation code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path.cwd()

from tests.reference import offline as common  # noqa: E402


DEFAULT_INPUT_ROOT = ROOT / "data" / "experiments" / "proposer_critic_integrator"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "data" / "experiments" / "proposer_critic_integrator_critic2"
)
DEFAULT_DOCUMENT_BATCH_SIZE = 8
DEFAULT_PROMPT_BATCH_SIZE = 4
DEFAULT_MAX_RECOVERY_ROUNDS = 3
CRITIC_ENV_FIELDS = {
    "CRITIC_MODEL_NAME": "model_name",
    "CRITIC_BASE_MODEL": "base_model_path",
    "CRITIC_ADAPTER_PATH": "adapter_path",
    "CRITIC_DEVICE": "device",
}


class _UnavailableStage:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, method):
        raise AssertionError(f"offline final Critic runner called {self.name}.{method}")


def normalize_critic_runtime(
    source: common.OfflineSourceRun,
    prompt_batch_size: int,
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict:
    runtime = dict(source.critic_runtime)
    override_values = dict(overrides or {})
    unexpected = sorted(set(override_values) - set(CRITIC_ENV_FIELDS))
    if unexpected:
        raise ValueError(f"unsupported Critic runtime overrides: {unexpected}")
    for env_key, field in CRITIC_ENV_FIELDS.items():
        if env_key in override_values:
            runtime[field] = str(override_values[env_key])
    for field in ("base_model_path", "adapter_path"):
        value = str(runtime.get(field) or "").strip()
        if not value:
            raise ValueError(f"Critic1 runtime has no {field}")
        runtime[field] = str(common.resolve_path(value))
    runtime["prompt_batch_size"] = int(prompt_batch_size)
    runtime["backend"] = "direct"
    runtime.setdefault("batch_size", 4)
    runtime.setdefault("max_tokens", 1024)
    runtime.setdefault("max_seq_length", 8192)
    runtime.setdefault("model_name", "critic-phase2")
    runtime.setdefault("device", "cuda:0")
    runtime.setdefault("dtype", "bfloat16")
    runtime.setdefault("attention_implementation", "sdpa")
    runtime.setdefault("merge_adapter", True)
    runtime.setdefault("thinking_enabled", False)
    runtime.setdefault("initial_model", runtime["model_name"])
    runtime.setdefault("final_model", runtime["model_name"])
    return runtime


def build_run_state(
    source: common.OfflineSourceRun,
    critic_runtime: Mapping,
    document_batch_size: int,
    prompt_batch_size: int,
) -> Dict:
    return {
        "version": common.RUN_STATE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": common.source_state(source),
        "runtime": {
            "stage": "final_critic",
            "critic_document_batch_size": int(document_batch_size),
            "critic_prompt_batch_size": int(prompt_batch_size),
            "critic_runtime": dict(critic_runtime),
        },
    }


def build_controller(runtime_settings: Mapping):
    from evidenceloop.agents.critic import CriticAgent
    from evidenceloop.pipeline.core import Pipeline
    from evidenceloop.runtime.direct_critic import DirectCriticEngine

    engine = DirectCriticEngine(runtime_settings)
    critic = CriticAgent(
        name="Offline-Final-Critic",
        role="Critic",
        llm=engine,
        runtime_settings=runtime_settings,
    )
    critic.validate_runtime()
    controller = Pipeline(
        proposer_agent=_UnavailableStage("Proposer"),
        critic_agent=critic,
        integrator_agent=_UnavailableStage("Integrator"),
        pipeline_variant="full",
    )
    return controller, engine


def _processing_result(row, record, status: str, error: str, elapsed: float) -> Dict:
    reviews = record.final_reviews if record else []
    action_counts = Counter(review.action for review in reviews)
    origin_counts = Counter(review.origin for review in reviews)
    input_count = len(row.pipeline_record.get("refined_triples") or []) + len(
        row.pipeline_record.get("added_triples") or []
    )
    return {
        "document_id": row.document_id,
        "status": status,
        "error": error,
        "input_triples": input_count,
        "proposer_candidates": len(row.pipeline_record.get("refined_triples") or []),
        "integrator_candidates": len(row.pipeline_record.get("added_triples") or []),
        "output_triples": len(record.final_triples) if record else 0,
        "critic_actions": dict(sorted(action_counts.items())),
        "review_origins": dict(sorted(origin_counts.items())),
        "invalid_reviews": sum(not review.valid for review in reviews),
        "stage_errors": len(record.stage_errors) if record else 0,
        "elapsed_seconds": round(float(elapsed), 6),
    }


def _write_progress(run_dir: Path, rows, checkpoints, previous: Mapping) -> None:
    value = common.progress_payload(rows, checkpoints)
    value["active_wall_seconds"] = float(
        previous.get("active_wall_seconds", 0.0) or 0.0
    )
    value["critic_metrics"] = dict(previous.get("critic_metrics") or {})
    value["finalized"] = False
    common.atomic_write_json(run_dir / "progress.json", value)


def process_documents(
    controller,
    rows,
    run_dir: Path,
    *,
    document_batch_size: int = DEFAULT_DOCUMENT_BATCH_SIZE,
    console_print=print,
) -> Tuple[Dict[int, Dict], Dict]:
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoints = common.load_output_checkpoints(run_dir, rows)
    progress_path = run_dir / "progress.json"
    previous_progress = (
        common.read_json_object(progress_path) if progress_path.is_file() else {}
    )
    _write_progress(run_dir, rows, checkpoints, previous_progress)
    pending = [
        row
        for row in rows
        if checkpoints.get(row.index, {}).get("status") != "ok"
    ]
    for batch_start in range(0, len(pending), document_batch_size):
        batch = pending[batch_start:batch_start + document_batch_size]
        prepared = []
        requests = []
        request_positions = []
        outputs = [None] * len(batch)
        errors = [""] * len(batch)
        for position, row in enumerate(batch):
            record = common.pipeline_record_from_dict(row.pipeline_record)
            triples = record.refined_triples + record.added_triples
            origins = (
                ["proposer"] * len(record.refined_triples)
                + ["integrator"] * len(record.added_triples)
            )
            prepared.append((row, record, triples, origins))
            if triples:
                request_positions.append(position)
                requests.append({
                    "triples": triples,
                    "stage": "final_validation",
                    "origins": origins,
                })
            else:
                outputs[position] = {
                    "payload": {
                        "results": [],
                        "raw_response": "",
                        "parse_error": "",
                    },
                    "elapsed_seconds": 0.0,
                    "prompt_count": 0,
                }

        if requests:
            try:
                generated = controller.critic.review_documents_batch(requests)
                if len(generated) != len(requests):
                    raise RuntimeError(
                        "Critic document batch output count does not match input count"
                    )
                for position, output in zip(request_positions, generated):
                    outputs[position] = output
            except Exception as exc:
                traceback.print_exc()
                error = f"{type(exc).__name__}: {exc}"
                for position in request_positions:
                    errors[position] = error

        for (row, record, _, _), output, batch_error in zip(
            prepared, outputs, errors
        ):
            previous = checkpoints.get(row.index) or {}
            attempts = int(previous.get("attempts", 0) or 0) + 1
            status = "failed" if batch_error else "ok"
            error = batch_error
            critic_elapsed = 0.0
            if error:
                record = None
            else:
                try:
                    critic_elapsed = float(output.get("elapsed_seconds", 0.0) or 0.0)
                    controller.run_final_critic_stage(
                        record,
                        final_critic_payload=output["payload"],
                        final_critic_elapsed_seconds=critic_elapsed,
                    )
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    record = None
                    traceback.print_exc()
            result = _processing_result(
                row,
                record,
                status=status,
                error=error,
                elapsed=critic_elapsed,
            )
            checkpoint = {
                "version": common.CHECKPOINT_VERSION,
                "index": row.index,
                "document_id": row.document_id,
                "attempts": attempts,
                "status": status,
                "error": error,
                "processing_result": result,
                "pipeline_record": record.to_dict() if record else None,
                "final_triples": (
                    [triple.to_dict() for triple in record.final_triples]
                    if record
                    else None
                ),
            }
            common.atomic_write_json(
                common.checkpoint_path(run_dir, row.index), checkpoint
            )
            checkpoints[row.index] = checkpoint
            _write_progress(run_dir, rows, checkpoints, previous_progress)
            console_print(
                f"[{row.index + 1}/{len(rows)}] {row.document_id} {status} | "
                f"input={result['input_triples']} output={result['output_triples']}"
            )
    return checkpoints, previous_progress


def _combine_critic_metrics(previous: Mapping, current: Mapping) -> Dict:
    additive = (
        "model_load_seconds",
        "critic_prompt_count",
        "critic_generation_batches",
        "critic_input_tokens",
        "critic_output_tokens",
        "critic_generation_seconds",
    )
    output = {
        "backend": "direct",
        "adapter_merged": bool(current.get("adapter_merged", False)),
    }
    for key in additive:
        value = float(previous.get(key, 0) or 0) + float(current.get(key, 0) or 0)
        output[key] = round(value, 3) if key.endswith("seconds") else int(value)
    output["critic_peak_gpu_memory_bytes"] = max(
        int(previous.get("critic_peak_gpu_memory_bytes", 0) or 0),
        int(current.get("critic_peak_gpu_memory_bytes", 0) or 0),
    )
    seconds = float(output["critic_generation_seconds"])
    output["critic_tokens_per_second"] = (
        round(output["critic_output_tokens"] / seconds, 3) if seconds else 0.0
    )
    return output


def _augment_summary(run_dir: Path, summary: Dict, rows, checkpoints) -> Dict:
    results = [
        dict(checkpoints[index]["processing_result"])
        for index in sorted(checkpoints)
    ]
    actions = Counter()
    origins = Counter()
    for result in results:
        actions.update(result.get("critic_actions") or {})
        origins.update(result.get("review_origins") or {})
    summary.update({
        "input_triples": sum(item.get("input_triples", 0) for item in results),
        "critic_actions": dict(sorted(actions.items())),
        "review_origins": dict(sorted(origins.items())),
        "invalid_reviews": sum(item.get("invalid_reviews", 0) for item in results),
        "stage_errors": sum(item.get("stage_errors", 0) for item in results),
    })
    common.atomic_write_json(run_dir / "summary.json", summary)
    return summary


def run_offline_source(
    source_run_dir: Path,
    output_run_dir: Path,
    *,
    document_batch_size: int,
    prompt_batch_size: int,
    runtime_overrides: Optional[Mapping[str, str]] = None,
    controller=None,
    engine=None,
) -> Path:
    source = common.inspect_integrator_run(source_run_dir)
    rows = common.load_source_documents(source)
    output_run_dir = output_run_dir.resolve()
    runtime = normalize_critic_runtime(source, prompt_batch_size, runtime_overrides)
    state = build_run_state(
        source,
        runtime,
        document_batch_size,
        prompt_batch_size,
    )
    state_path = output_run_dir / "run_state.json"
    existing = common.load_output_checkpoints(output_run_dir, rows)
    complete = (
        len(existing) == len(rows)
        and all(value.get("status") == "ok" for value in existing.values())
    )
    if state_path.exists():
        common.validate_resume_state(common.read_json_object(state_path), state)
    else:
        from evidenceloop.runtime.settings import validate_direct_runtime_paths

        validate_direct_runtime_paths(runtime)
        common.initialize_run_state(output_run_dir, state)

    previous_progress = (
        common.read_json_object(output_run_dir / "progress.json")
        if (output_run_dir / "progress.json").is_file()
        else {}
    )
    previous_seconds = float(previous_progress.get("active_wall_seconds", 0.0) or 0.0)
    if complete:
        summary = common.materialize_outputs(
            source,
            rows,
            output_run_dir,
            stage="final_critic",
            runtime_summary={
                "critic_document_batch_size": document_batch_size,
                "critic_prompt_batch_size": prompt_batch_size,
                "critic_runtime": runtime,
                "critic_metrics": dict(previous_progress.get("critic_metrics") or {}),
            },
            active_wall_seconds=previous_seconds,
        )
        _augment_summary(output_run_dir, summary, rows, existing)
        print(f"[done] {source.dataset} already complete; output={output_run_dir}")
        return output_run_dir

    started = time.perf_counter()
    if controller is None:
        controller, engine = build_controller(runtime)
    console = sys.__stdout__

    def console_print(message: str) -> None:
        print(message, file=console, flush=True)
        print(message, flush=True)

    log_mode = "a" if previous_progress else "w"
    with (output_run_dir / "run.log").open(
        log_mode, encoding="utf-8", buffering=1
    ) as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        print(
            f"Offline final Critic started at "
            f"{datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        print(json.dumps(state["runtime"], ensure_ascii=False, indent=2), flush=True)
        checkpoints, prior_progress = process_documents(
            controller,
            rows,
            output_run_dir,
            document_batch_size=document_batch_size,
            console_print=console_print,
        )
        active_seconds = previous_seconds + time.perf_counter() - started
        current_metrics = engine.metrics() if engine is not None else {}
        cumulative_metrics = _combine_critic_metrics(
            prior_progress.get("critic_metrics") or {}, current_metrics
        )
        summary = common.materialize_outputs(
            source,
            rows,
            output_run_dir,
            stage="final_critic",
            runtime_summary={
                "critic_document_batch_size": document_batch_size,
                "critic_prompt_batch_size": prompt_batch_size,
                "critic_runtime": runtime,
                "critic_metrics": cumulative_metrics,
            },
            active_wall_seconds=active_seconds,
        )
        summary = _augment_summary(output_run_dir, summary, rows, checkpoints)
        progress = common.read_json_object(output_run_dir / "progress.json")
        progress["active_wall_seconds"] = round(active_seconds, 3)
        progress["critic_metrics"] = cumulative_metrics
        common.atomic_write_json(output_run_dir / "progress.json", progress)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    value = common.progress_payload(rows, checkpoints)
    console_print(
        f"[done] {source.dataset} completed={value['completed']}/"
        f"{value['documents_total']} failed={value['failed']} "
        f"output={output_run_dir}"
    )
    return output_run_dir


def critic_child_env(
    source: common.OfflineSourceRun,
    prompt_batch_size: int,
    runtime_overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    runtime = normalize_critic_runtime(source, prompt_batch_size, runtime_overrides)
    for env_key, field in CRITIC_ENV_FIELDS.items():
        env[env_key] = str(runtime[field])
    env["CRITIC_PROMPT_BATCH_SIZE"] = str(prompt_batch_size)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _worker_overrides_from_environment() -> Dict[str, str]:
    return {
        key: str(os.environ[key])
        for key in CRITIC_ENV_FIELDS
        if str(os.environ.get(key) or "").strip()
    }


def _run_worker(
    source: common.OfflineSourceRun,
    output_run_dir: Path,
    args,
    runtime_overrides: Optional[Mapping[str, str]] = None,
) -> None:
    env = critic_child_env(
        source,
        args.critic_prompt_batch_size,
        runtime_overrides,
    )
    command = [
        sys.executable,
        "-m", "tests.reference.final_critic",
        "--worker-source-run-dir",
        str(source.path),
        "--worker-output-run-dir",
        str(output_run_dir),
        "--critic-document-batch-size",
        str(args.critic_document_batch_size),
        "--critic-prompt-batch-size",
        str(args.critic_prompt_batch_size),
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"offline final Critic worker failed for {source.dataset} "
            f"with exit code {result.returncode}"
        )


def complete_source_run(
    source: common.OfflineSourceRun,
    output_run_dir: Path,
    args,
    runtime_overrides: Optional[Mapping[str, str]] = None,
) -> Path:
    for recovery_round in range(args.max_recovery_rounds + 1):
        _run_worker(source, output_run_dir, args, runtime_overrides)
        progress = common.read_json_object(output_run_dir / "progress.json")
        if (
            int(progress.get("failed", 0) or 0) == 0
            and int(progress.get("pending", 0) or 0) == 0
        ):
            return output_run_dir
        if recovery_round >= args.max_recovery_rounds:
            raise RuntimeError(
                f"{source.dataset} remains incomplete: "
                f"completed={progress.get('completed')} failed={progress.get('failed')} "
                f"pending={progress.get('pending')}"
            )
    raise AssertionError("unreachable")


def _cli_runtime_overrides(args) -> Dict[str, str]:
    values = {
        "CRITIC_BASE_MODEL": args.critic_base_model,
        "CRITIC_ADAPTER_PATH": args.critic_adapter_path,
        "CRITIC_DEVICE": args.critic_device,
    }
    return {
        key: (
            str(common.resolve_path(value))
            if key in {"CRITIC_BASE_MODEL", "CRITIC_ADAPTER_PATH"}
            else str(value)
        )
        for key, value in values.items()
        if value is not None
    }


def _saved_runtime_overrides(run_dir: Path) -> Dict[str, str]:
    state = common.read_json_object(run_dir / "run_state.json")
    runtime = dict((state.get("runtime") or {}).get("critic_runtime") or {})
    output = {}
    for env_key, field in CRITIC_ENV_FIELDS.items():
        value = str(runtime.get(field) or "").strip()
        if not value:
            raise ValueError(f"{run_dir}: saved Critic runtime has no {field}")
        output[env_key] = value
    return output


def _source_from_output_run(run_dir: Path) -> common.OfflineSourceRun:
    state = common.read_json_object(run_dir / "run_state.json")
    source_path = str((state.get("source") or {}).get("run_dir") or "").strip()
    if not source_path:
        raise ValueError(f"{run_dir}: run_state.json has no source run")
    return common.inspect_integrator_run(Path(source_path))


def _validate_args(args) -> None:
    if args.source_run_dir and args.resume_run_dir:
        raise ValueError("--source-run-dir and --resume-run-dir are mutually exclusive")
    if args.output_run_dir and not args.source_run_dir:
        raise ValueError("--output-run-dir is only valid with --source-run-dir")
    if not 1 <= args.critic_document_batch_size <= 64:
        raise ValueError("--critic-document-batch-size must be between 1 and 64")
    if not 1 <= args.critic_prompt_batch_size <= 4:
        raise ValueError("--critic-prompt-batch-size must be between 1 and 4")
    if args.max_recovery_rounds < 0:
        raise ValueError("--max-recovery-rounds must not be negative")
    custom_paths = args.critic_base_model or args.critic_adapter_path
    if custom_paths and not (args.source_run_dir or args.resume_run_dir):
        raise ValueError(
            "custom Critic model paths require --source-run-dir or --resume-run-dir"
        )
    if custom_paths and args.source_run_dir and not args.output_run_dir:
        raise ValueError(
            "custom Critic model paths require --output-run-dir for a new source run"
        )
    if bool(args.worker_source_run_dir) != bool(args.worker_output_run_dir):
        raise ValueError("worker source and output directories must be provided together")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run the dataset-specific local final Critic over Integrator output."
    )
    parser.add_argument("--dataset", choices=common.DATASET_CHOICES, default="all")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-run-dir")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--critic-base-model")
    parser.add_argument("--critic-adapter-path")
    parser.add_argument("--critic-device")
    parser.add_argument(
        "--critic-document-batch-size", type=int, default=DEFAULT_DOCUMENT_BATCH_SIZE
    )
    parser.add_argument(
        "--critic-prompt-batch-size", type=int, default=DEFAULT_PROMPT_BATCH_SIZE
    )
    parser.add_argument(
        "--max-recovery-rounds", type=int, default=DEFAULT_MAX_RECOVERY_ROUNDS
    )
    parser.add_argument("--worker-source-run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-run-dir", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    common.load_dotenv_file(ROOT / ".env")
    args = parse_args(argv)
    _validate_args(args)
    if args.worker_source_run_dir:
        run_offline_source(
            common.resolve_path(args.worker_source_run_dir),
            common.resolve_path(args.worker_output_run_dir),
            document_batch_size=args.critic_document_batch_size,
            prompt_batch_size=args.critic_prompt_batch_size,
            runtime_overrides=_worker_overrides_from_environment(),
        )
        return 0

    cli_overrides = _cli_runtime_overrides(args)
    if args.resume_run_dir:
        output_run_dir = common.resolve_path(args.resume_run_dir)
        source = _source_from_output_run(output_run_dir)
        overrides = _saved_runtime_overrides(output_run_dir)
        overrides.update(cli_overrides)
        complete_source_run(source, output_run_dir, args, overrides)
        return 0

    if args.source_run_dir:
        sources = [
            common.inspect_integrator_run(common.resolve_path(args.source_run_dir))
        ]
        skipped = []
    else:
        input_root = common.resolve_path(args.input_root)
        roots = {
            key: input_root / key for key in common.selected_dataset_keys(args.dataset)
        }
        sources, skipped = common.discover_latest_runs(
            roots,
            common.selected_dataset_keys(args.dataset),
            common.inspect_integrator_run,
        )
    for message in skipped:
        print(f"[skip] {message}", flush=True)

    output_root = common.resolve_path(args.output_root)
    failures = []
    for source in sources:
        output_run_dir = (
            common.resolve_path(args.output_run_dir)
            if args.output_run_dir
            else output_root / source.dataset / source.timestamp
        )
        try:
            complete_source_run(source, output_run_dir, args, cli_overrides)
        except Exception as exc:
            failures.append(f"{source.dataset}: {exc}")
            traceback.print_exc()
    if failures:
        raise RuntimeError("offline final Critic failures: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
