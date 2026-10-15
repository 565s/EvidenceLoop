"""Run only the Integrator over completed proposer + Critic1 experiments.

This script never calls the Proposer or Critic and never reads benchmark gold or
computes evaluation metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path.cwd()

from evidenceloop.agents.integrator import IntegratorAgent  # noqa: E402
from evidenceloop.pipeline.core import Pipeline  # noqa: E402
from tests.reference import offline as common  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "proposer_critic_integrator"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_BASE_URL = ""
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 2048
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_WORKERS = 4
DEFAULT_MAX_ATTEMPTS = 3
MAX_WORKERS = 16


class _UnavailableStage:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, method):
        raise AssertionError(f"offline Integrator runner called {self.name}.{method}")


def _runtime_signature(args) -> Dict:
    return {
        "stage": "integrator",
        "provider": "siliconflow",
        "model": str(args.integrator_model),
        "base_url": (os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL")) or DEFAULT_BASE_URL,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "request_timeout": float(args.request_timeout),
        "enable_thinking": False,
        "integrator_workers": int(args.integrator_workers),
        "prefetch_window": int(args.integrator_workers) * 2,
        "max_attempts": int(args.max_attempts),
    }


def build_run_state(source: common.OfflineSourceRun, runtime: Mapping) -> Dict:
    return {
        "version": common.RUN_STATE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": common.source_state(source),
        "runtime": dict(runtime),
    }


def build_integrator(args) -> IntegratorAgent:
    api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("SIFLOW_API_KEY"))
    if not api_key:
        raise RuntimeError("SIFLOW_API_KEY is required in .env or the environment")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=str(args.integrator_model),
        base_url=(os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL")) or DEFAULT_BASE_URL,
        api_key=api_key,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout=float(args.request_timeout),
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    return IntegratorAgent(name="Integrator-Offline", role="Integrator", llm=llm)


def build_controller(integrator) -> Pipeline:
    return Pipeline(
        proposer_agent=_UnavailableStage("Proposer"),
        critic_agent=_UnavailableStage("Critic1"),
        integrator_agent=integrator,
        pipeline_variant="full",
    )


def request_integrator(
    integrator,
    row: common.SourceDocument,
    *,
    max_attempts: int,
) -> Tuple[Optional[Dict], float, int, str]:
    record = common.pipeline_record_from_dict(row.pipeline_record)
    elapsed_total = 0.0
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            payload = integrator.audit_coverage(
                record.source_text,
                record.refined_triples,
            )
            elapsed_total += time.perf_counter() - started
            if not isinstance(payload, Mapping):
                last_error = "Integrator returned a non-object payload"
            else:
                payload = dict(payload)
                last_error = str(payload.get("parse_error") or "")
                if not last_error:
                    return payload, elapsed_total, attempt, ""
        except Exception as exc:
            elapsed_total += time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < max_attempts:
            time.sleep(float(attempt))
    return None, elapsed_total, max_attempts, last_error or "Integrator request failed"


def _write_progress(
    run_dir: Path,
    rows,
    checkpoints,
    *,
    workers: int,
    max_in_flight: int,
) -> None:
    api_attempts = sum(
        int((checkpoint.get("processing_result") or {}).get("api_attempts", 0) or 0)
        for checkpoint in checkpoints.values()
    )
    common.atomic_write_json(
        run_dir / "progress.json",
        common.progress_payload(
            rows,
            checkpoints,
            integrator_workers=workers,
            prefetch_window=workers * 2,
            integrator_api_attempts=api_attempts,
            integrator_retries=max(0, api_attempts - len(checkpoints)),
            max_in_flight=max_in_flight,
            finalized=False,
        ),
    )


def _append_log(run_dir: Path, message: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().isoformat(timespec='seconds')} {message.rstrip()}\n"
        )


def process_documents(
    controller,
    integrator,
    rows,
    run_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    console_print=print,
) -> Tuple[Dict[int, Dict], Dict]:
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoints = common.load_output_checkpoints(run_dir, rows)
    pending = [
        row
        for row in rows
        if checkpoints.get(row.index, {}).get("status") != "ok"
    ]
    max_in_flight = 0
    _write_progress(
        run_dir,
        rows,
        checkpoints,
        workers=workers,
        max_in_flight=max_in_flight,
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        window_size = workers * 2
        for window_start in range(0, len(pending), window_size):
            window = pending[window_start:window_start + window_size]
            futures = [
                (
                    row,
                    executor.submit(
                        request_integrator,
                        integrator,
                        row,
                        max_attempts=max_attempts,
                    ),
                )
                for row in window
            ]
            max_in_flight = max(max_in_flight, len(futures))
            for row, future in futures:
                previous = checkpoints.get(row.index) or {}
                previous_attempts = int(previous.get("attempts", 0) or 0)
                previous_api_attempts = int(
                    (previous.get("processing_result") or {}).get(
                        "api_attempts", 0
                    ) or 0
                )
                payload, elapsed, api_attempts, error = future.result()
                attempts = previous_attempts + 1
                processing_result = {
                    "document_id": row.document_id,
                    "status": "failed" if error else "ok",
                    "error": error,
                    "api_attempts": previous_api_attempts + api_attempts,
                    "elapsed_seconds": round(elapsed, 6),
                    "refined_count": len(
                        row.pipeline_record.get("refined_triples") or []
                    ),
                    "additions_proposed": 0,
                    "additions_accepted": 0,
                    "predicted_count": 0,
                }
                checkpoint = {
                    "version": common.CHECKPOINT_VERSION,
                    "index": row.index,
                    "document_id": row.document_id,
                    "attempts": attempts,
                    "status": processing_result["status"],
                    "error": error,
                    "processing_result": processing_result,
                    "pipeline_record": None,
                    "final_triples": None,
                }
                if not error:
                    try:
                        record = common.pipeline_record_from_dict(row.pipeline_record)
                        controller.run_integrator_stage(
                            record,
                            coverage_payload=payload,
                            integrator_elapsed_seconds=elapsed,
                        )
                        record_dict = record.to_dict()
                        proposed = len((payload or {}).get("additions") or [])
                        accepted = len(record.added_triples)
                        processing_result.update({
                            "additions_proposed": proposed,
                            "additions_accepted": accepted,
                            "predicted_count": len(record.final_triples),
                        })
                        checkpoint["pipeline_record"] = record_dict
                        checkpoint["final_triples"] = record_dict["final_triples"]
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        processing_result["status"] = "failed"
                        processing_result["error"] = error
                        checkpoint["status"] = "failed"
                        checkpoint["error"] = error
                        checkpoint["pipeline_record"] = None
                        checkpoint["final_triples"] = None

                common.atomic_write_json(
                    common.checkpoint_path(run_dir, row.index), checkpoint
                )
                checkpoints[row.index] = checkpoint
                _write_progress(
                    run_dir,
                    rows,
                    checkpoints,
                    workers=workers,
                    max_in_flight=max_in_flight,
                )
                label = "ok" if checkpoint["status"] == "ok" else f"failed: {error}"
                console_print(
                    f"[{row.index + 1}/{len(rows)}] {row.document_id} {label} | "
                    f"input={processing_result['refined_count']} "
                    f"added={processing_result['additions_accepted']} "
                    f"output={processing_result['predicted_count']}"
                )
    stats = {
        "integrator_workers": workers,
        "prefetch_window": workers * 2,
        "max_in_flight": max_in_flight,
    }
    return checkpoints, stats


def run_integrator_source(
    source: common.OfflineSourceRun,
    run_dir: Path,
    args,
    *,
    integrator=None,
    controller=None,
) -> Dict:
    rows = common.load_source_documents(source)
    runtime = _runtime_signature(args)
    state = build_run_state(source, runtime)
    state_path = run_dir / "run_state.json"
    if not state_path.exists() and integrator is None:
        integrator = build_integrator(args)
    common.initialize_run_state(run_dir, state)
    _append_log(
        run_dir,
        f"stage=integrator dataset={source.dataset} source={source.path} "
        f"model={runtime['model']} workers={runtime['integrator_workers']}",
    )

    checkpoints = common.load_output_checkpoints(run_dir, rows)
    complete = (
        len(checkpoints) == len(rows)
        and all(value.get("status") == "ok" for value in checkpoints.values())
    )
    progress = (
        common.read_json_object(run_dir / "progress.json")
        if (run_dir / "progress.json").is_file()
        else {}
    )
    previous_seconds = float(progress.get("active_wall_seconds", 0.0) or 0.0)
    active_seconds = previous_seconds
    stats = {
        "integrator_workers": args.integrator_workers,
        "prefetch_window": args.integrator_workers * 2,
        "max_in_flight": 0,
    }
    if not complete:
        if integrator is None:
            integrator = build_integrator(args)
        controller = controller or build_controller(integrator)
        started = time.perf_counter()
        try:
            def console_print(message: str) -> None:
                print(message, flush=True)
                _append_log(run_dir, message)

            checkpoints, stats = process_documents(
                controller,
                integrator,
                rows,
                run_dir,
                workers=args.integrator_workers,
                max_attempts=args.max_attempts,
                console_print=console_print,
            )
        finally:
            active_seconds = previous_seconds + time.perf_counter() - started

    summary = common.materialize_outputs(
        source,
        rows,
        run_dir,
        stage="integrator",
        runtime_summary={**runtime, **stats},
        active_wall_seconds=active_seconds,
    )
    progress = common.read_json_object(run_dir / "progress.json")
    progress["active_wall_seconds"] = round(active_seconds, 3)
    common.atomic_write_json(run_dir / "progress.json", progress)
    print(
        f"[done] {source.dataset} completed={summary['documents_completed']}/"
        f"{summary['documents_total']} failed={summary['documents_failed']} "
        f"output={run_dir}",
        flush=True,
    )
    _append_log(
        run_dir,
        f"completed={summary['documents_completed']}/{summary['documents_total']} "
        f"failed={summary['documents_failed']} finalized={summary['finalized']}",
    )
    if not summary["finalized"]:
        raise RuntimeError(
            f"{source.dataset} Integrator run remains incomplete: "
            f"completed={summary['documents_completed']} "
            f"failed={summary['documents_failed']}"
        )
    return summary


def _validate_args(args) -> None:
    if args.source_run_dir and args.resume_run_dir:
        raise ValueError("--source-run-dir and --resume-run-dir are mutually exclusive")
    if args.output_run_dir and not args.source_run_dir:
        raise ValueError("--output-run-dir is only valid with --source-run-dir")
    if args.input_root and (args.source_run_dir or args.resume_run_dir):
        raise ValueError("--input-root cannot be combined with --source-run-dir or --resume-run-dir")
    if args.input_root is not None and not str(args.input_root).strip():
        raise ValueError("--input-root must not be empty")
    if not str(args.integrator_model or "").strip():
        raise ValueError("--integrator-model must not be empty")
    if not 1 <= args.integrator_workers <= MAX_WORKERS:
        raise ValueError(f"--integrator-workers must be between 1 and {MAX_WORKERS}")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run DeepSeek Integrator over completed proposer + Critic1 runs."
    )
    parser.add_argument("--dataset", choices=common.DATASET_CHOICES, default="all")
    parser.add_argument("--source-run-dir")
    parser.add_argument(
        "--input-root",
        help=(
            "Root containing offline proposer + Critic1 outputs. Supports "
            "<dataset>/<model>/<timestamp> layouts."
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-run-dir")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--integrator-model", default=DEFAULT_MODEL)
    parser.add_argument("--integrator-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args(argv)


def _resume_source(run_dir: Path) -> common.OfflineSourceRun:
    state = common.read_json_object(run_dir / "run_state.json")
    source_path = str((state.get("source") or {}).get("run_dir") or "")
    if not source_path:
        raise ValueError(f"{run_dir}: run_state.json has no source run")
    return common.inspect_critic1_run(Path(source_path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    common.load_dotenv_file(ROOT / ".env")

    if args.resume_run_dir:
        run_dir = common.resolve_path(args.resume_run_dir)
        source = _resume_source(run_dir)
        run_integrator_source(source, run_dir, args)
        return 0

    if args.source_run_dir:
        source = common.inspect_critic1_run(common.resolve_path(args.source_run_dir))
        run_dir = (
            common.resolve_path(args.output_run_dir)
            if args.output_run_dir
            else common.resolve_path(args.output_root) / source.dataset / source.timestamp
        )
        run_integrator_source(source, run_dir, args)
        return 0

    dataset_keys = common.selected_dataset_keys(args.dataset)
    if args.input_root:
        input_root = common.resolve_path(args.input_root)
        roots = {key: input_root / key for key in dataset_keys}
        nested_model_dirs = True
    else:
        roots = {key: spec.source_root for key, spec in common.DATASETS.items()}
        nested_model_dirs = False
    sources, skipped = common.discover_latest_runs(
        roots,
        dataset_keys,
        common.inspect_critic1_run,
        nested_model_dirs=nested_model_dirs,
    )
    for message in skipped:
        print(f"[skip] {message}", flush=True)
    output_root = common.resolve_path(args.output_root)
    failures = []
    for source in sources:
        run_dir = output_root / source.dataset / source.timestamp
        try:
            run_integrator_source(source, run_dir, args)
        except Exception as exc:
            failures.append(f"{source.dataset}: {exc}")
            if (run_dir / "run_state.json").is_file():
                _append_log(run_dir, f"failed={type(exc).__name__}: {exc}")
            traceback.print_exc()
    if failures:
        raise RuntimeError("offline Integrator failures: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
