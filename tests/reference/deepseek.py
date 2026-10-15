from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


ROOT = Path.cwd()

from tests.reference import direct as shared  # noqa: E402
from evidenceloop.settings import agent_config  # noqa: E402
from evidenceloop.resources.prompt_loader import (  # noqa: E402
    load_proposer_extract_call_prompt,
    load_proposer_system_prompt,
)


DEFAULT_BASE_URL = ""
FALLBACK_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_MODEL = str(agent_config.get("proposer_model_name") or FALLBACK_MODEL)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_ENABLE_THINKING = False
DEFAULT_RUN_NAME = "deepseek_baselines"
DEFAULT_BERT_MODEL = shared.DEFAULT_BERT_MODEL
DEFAULT_METRIC_DEVICE = shared.DEFAULT_METRIC_DEVICE
DEFAULT_METRIC_BATCH_SIZE = shared.DEFAULT_METRIC_BATCH_SIZE
DATASET_CHOICES = shared.DATASET_CHOICES
DATASETS = shared.DATASETS

from evidenceloop.data.registry import AliasDict
DATASETS = AliasDict(DATASETS)
STAGE_NAME = "deepseek_extraction"


def normalize_model_name(value: str) -> str:
    model_name = str(value or "").strip()
    if not model_name:
        raise ValueError("model must be a non-empty SiliconFlow model identifier")
    return model_name


def model_slug(value: str) -> str:
    model_name = normalize_model_name(value)
    if model_name.casefold() == FALLBACK_MODEL.casefold():
        return "deepseek-v4-flash"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-._").lower()
    if not slug:
        raise ValueError("model identifier cannot be converted to an output-directory name")
    return slug


class DeepSeekBaselineExtractor:
    def __init__(
        self,
        model_name: str,
        llm=None,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.model_name = normalize_model_name(model_name)
        self.system_prompt = load_proposer_system_prompt()
        self.call_prompt = load_proposer_extract_call_prompt()
        self.llm = llm or self._build_llm(
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )

    def _build_llm(
        self,
        *,
        temperature: float,
        max_tokens: int,
        request_timeout: float,
    ):
        from langchain_openai import ChatOpenAI

        api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("SIFLOW_API_KEY"))
        if not api_key:
            raise RuntimeError("SIFLOW_API_KEY is required in .env or the environment")
        return ChatOpenAI(
            model=self.model_name,
            base_url=(os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL")) or DEFAULT_BASE_URL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            max_retries=0,
            extra_body={"enable_thinking": DEFAULT_ENABLE_THINKING},
        )

    def extract(self, document_id: str, text: str) -> Dict:
        response = self.llm.invoke([
            ("system", self.system_prompt),
            ("user", self.call_prompt.format(text=text)),
        ])
        content = getattr(response, "content", response)
        parsed = shared.parse_gpt_response(str(content or "").strip(), document_id, text)
        if parsed.get("parse_error"):
            parsed["parse_error"] = "DeepSeek response did not contain valid JSON."
        return parsed


def build_runtime_signature(
    *,
    dataset: shared.DatasetPreset,
    model_name: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
) -> Dict:
    return {
        "baseline": "deepseek_direct_extraction",
        "provider": "siliconflow",
        "dataset": dataset.key,
        "model": normalize_model_name(model_name),
        "base_url": (os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL")) or DEFAULT_BASE_URL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_timeout": request_timeout,
        "enable_thinking": DEFAULT_ENABLE_THINKING,
        "max_attempts": shared.PROPOSER_MAX_ATTEMPTS,
        "entity_order_mode": dataset.entity_order_mode,
        "symmetric_predicates": list(
            shared.normalize_symmetric_predicates(dataset.symmetric_predicates)
        ),
    }


def _dataset_from_runtime(runtime: Mapping) -> shared.DatasetPreset:
    key = str(runtime.get("dataset") or "")
    if key not in DATASETS:
        raise ValueError(f"saved run_state contains unknown dataset {key!r}")
    return DATASETS[key]


def _resolve_repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_experiment(
    args,
    *,
    extractor: Optional[DeepSeekBaselineExtractor] = None,
) -> Path:
    shared.load_dotenv_file(ROOT / ".env")

    saved_state = None
    if args.resume_run_dir:
        run_dir = _resolve_repo_path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        saved_state = shared._read_json_object(run_dir / "run_state.json")
        saved_runtime = saved_state.get("runtime") or {}
        dataset = _dataset_from_runtime(saved_runtime)
        data_dir = Path((saved_state.get("data") or {}).get("data_dir", ""))
        requested_limit = int((saved_state.get("data") or {}).get("requested_limit", 0))
        model_name = normalize_model_name(saved_runtime.get("model"))
        temperature = float(saved_runtime.get("temperature", DEFAULT_TEMPERATURE))
        max_tokens = int(saved_runtime.get("max_tokens", DEFAULT_MAX_TOKENS))
        request_timeout = float(
            saved_runtime.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)
        )
        run_name = str(saved_state.get("run_name") or DEFAULT_RUN_NAME)
    else:
        dataset = DATASETS[args.dataset]
        data_dir = _resolve_repo_path(dataset.data_dir).resolve()
        model_name = normalize_model_name(args.model)
        temperature = float(args.temperature)
        max_tokens = int(args.max_tokens)
        request_timeout = float(args.request_timeout)
        run_name = args.run_name or DEFAULT_RUN_NAME
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            ROOT
            / "data"
            / "experiments"
            / "deepseek_baselines"
            / dataset.key
            / model_slug(model_name)
            / timestamp
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        requested_limit = (
            args.limit
            if args.limit is not None
            else shared.count_dataset_records(data_dir)
        )

    if request_timeout <= 0:
        raise ValueError("request-timeout must be positive")
    shared.validate_entity_order_config(
        dataset.entity_order_mode,
        dataset.symmetric_predicates,
    )
    rows, selected_limit = shared.load_dataset_rows(data_dir, requested_limit)
    runtime = build_runtime_signature(
        dataset=dataset,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )
    expected_state = shared.build_run_state(
        data_dir=data_dir,
        requested_limit=selected_limit,
        rows=rows,
        run_name=run_name,
        runtime=runtime,
    )
    if saved_state is not None:
        shared.validate_resume_state(saved_state, expected_state)
    else:
        shared.atomic_write_json(run_dir / "run_state.json", expected_state)

    console = sys.__stdout__

    def console_print(message: str) -> None:
        print(message, file=console, flush=True)

    console_print(
        f"[setup] loaded {len(rows)} {dataset.label} samples for {model_name}"
    )
    console_print(f"[setup] output dir: {run_dir}")
    if saved_state is not None:
        console_print("[setup] resume state validated")

    log_mode = "a" if saved_state is not None else "w"
    progress_path = run_dir / "progress.json"
    previous_progress = (
        shared._read_json_object(progress_path) if progress_path.exists() else {}
    )
    previous_active_seconds = float(
        previous_progress.get("active_wall_seconds", 0.0) or 0.0
    )
    processing_started = time.perf_counter()
    with (
        (run_dir / "run.log").open(
            log_mode,
            encoding="utf-8",
            buffering=1,
        ) as log_f,
        contextlib.redirect_stdout(log_f),
        contextlib.redirect_stderr(log_f),
    ):
        print(
            f"{dataset.label} DeepSeek baseline run "
            f"{'resumed' if saved_state else 'started'} at "
            f"{datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        print(
            f"Runtime config: {json.dumps(runtime, ensure_ascii=False, indent=2)}",
            flush=True,
        )
        active_wall_seconds = previous_active_seconds
        try:
            extractor = extractor or DeepSeekBaselineExtractor(
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            shared.process_rows_with_checkpoints(
                extractor,
                rows,
                run_dir / "checkpoints",
                progress_path,
                entity_order_mode=dataset.entity_order_mode,
                symmetric_predicates=dataset.symmetric_predicates,
                stage_name=STAGE_NAME,
                console_print=console_print,
            )
        finally:
            active_wall_seconds = round(
                previous_active_seconds + time.perf_counter() - processing_started,
                3,
            )
            if progress_path.exists():
                progress = shared._read_json_object(progress_path)
                progress["active_wall_seconds"] = active_wall_seconds
                shared.atomic_write_json(progress_path, progress)

        summary = shared.materialize_outputs(
            rows,
            run_dir / "checkpoints",
            run_dir,
            runtime,
            active_wall_seconds=active_wall_seconds,
            stage_name=STAGE_NAME,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if not args.skip_soft_metrics:
        shared.evaluate_soft_metrics(
            run_dir,
            dataset,
            bert_model=args.bert_model,
            metric_device=args.metric_device,
            metric_batch_size=args.metric_batch_size,
        )

    summary = shared._read_json_object(run_dir / "summary.json")
    console_print(
        "[done] "
        f"completed={summary['samples_completed']}/{summary['samples_requested']} "
        f"kg={summary['kg_triples']} "
        f"exact={summary['exact_match_spo_total']}/{summary['gold_triples_total']} "
        f"p/r/f1={summary['exact_match_precision']}/"
        f"{summary['exact_match_recall']}/{summary['exact_match_f1']}"
    )
    if not args.skip_soft_metrics:
        console_print(f"[done] soft metrics: {run_dir / 'graphjudge_metrics.json'}")
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DeepSeek-V4-Flash direct extraction baseline."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--skip-soft-metrics", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
    )
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--metric-device", default=DEFAULT_METRIC_DEVICE)
    parser.add_argument("--metric-batch-size", type=int, default=DEFAULT_METRIC_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.resume_run_dir:
        run_experiment(args)
        return 0

    run_dirs = []
    for dataset_key in shared.selected_dataset_keys(args.dataset):
        child_args = argparse.Namespace(**vars(args))
        child_args.dataset = dataset_key
        run_dirs.append(run_experiment(child_args))

    print("Completed DeepSeek baseline runs:")
    for run_dir in run_dirs:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
