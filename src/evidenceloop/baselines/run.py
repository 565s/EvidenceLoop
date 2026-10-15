"""Run related-work baselines without loading any gold triples during inference."""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from evidenceloop.baselines.related.common import (
    DATASETS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    METHODS,
    PIVE_INSTRUCTION,
    SOURCES,
    VERSION,
    digest,
    load_checkpoints,
    load_cohort,
    manifest,
    process_documents,
    read_json,
    read_jsonl,
    resolve_path,
)
from evidenceloop.baselines.related.runtime import dependency_versions, model_locations, redact
from evidenceloop.io import atomic_write_json, atomic_write_jsonl, load_dotenv_file

ROOT = Path.cwd()


DEFAULTS = {
    "model": DEFAULT_MODEL,
    "temperature": 0.0,
    "max_tokens": 2048,
    "request_timeout": 120.0,
    "enable_thinking": False,
    "device": "auto",
    "max_attempts": 3,
    "retry_delay": 2.0,
    "seed": 42,
}
METHOD_DEFAULTS = {
    "rakg": {"top_k": 5, "disambiguation_threshold": 0.6},
    "itext2kg": {"entity_threshold": 0.7, "relation_threshold": 0.7},
    "pive": {
        "pive_rounds": 3,
        "pive_instruction": PIVE_INSTRUCTION,
        "load_in_8bit": False,
        "verifier_max_input_tokens": 8192,
    },
    "kggen": {},
    "graphjudge": {
        "verifier_batch_size": 4,
        "load_in_8bit": False,
        "verifier_max_input_tokens": 8192,
    },
}
MODEL_OPTIONS = ("model_root", "embedding_model", "verifier_base_model", "verifier_adapter")


def implementation_digest():
    files = list((Path(__file__).parent / "related").glob("*.py"))
    files += list((Path(__file__).parent / "related/resources").glob("*.json"))
    files += [Path(__file__)]
    return digest(
        {
            p.relative_to(Path(__file__).parent).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(files)
        }
    )


def build_runtime(args, method, dataset, *, saved=None, check_environment=True):
    saved = saved or {}
    defaults = {**DEFAULTS, **METHOD_DEFAULTS[method]}
    runtime = {
        "method": method,
        "dataset": dataset,
        "source": SOURCES[method],
        "implementation_sha256": implementation_digest(),
    }
    for key, default in defaults.items():
        supplied = getattr(args, key, None)
        runtime[key] = supplied if supplied is not None else saved.get(key, default)
    runtime["base_url"] = (
        args.base_url
        or saved.get("base_url")
        or (os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL"))
        or ""
    )
    for key in (
        "max_tokens",
        "max_attempts",
        "request_timeout",
        "top_k",
        "verifier_batch_size",
        "verifier_max_input_tokens",
    ):
        if key in runtime and runtime[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if runtime["retry_delay"] < 0 or runtime.get("pive_rounds", 0) < 0:
        raise ValueError("retry_delay and pive_rounds must not be negative")
    for key in ("disambiguation_threshold", "entity_threshold", "relation_threshold"):
        if key in runtime and not 0 <= runtime[key] <= 1:
            raise ValueError(f"{key} must be between zero and one")
    if not str(runtime["model"]).strip() or not str(runtime["base_url"]).strip():
        raise ValueError("model and base_url must not be empty")
    options = {
        key: getattr(args, key, None) or saved.get("model_options", {}).get(key)
        for key in MODEL_OPTIONS
    }
    options["model_root"] = str(resolve_path(options["model_root"] or DEFAULT_MODEL_ROOT))
    for key in MODEL_OPTIONS[1:]:
        if options[key] and Path(options[key]).expanduser().is_dir():
            options[key] = str(Path(options[key]).expanduser().resolve())
    runtime["model_options"] = options
    if check_environment:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("SIFLOW_API_KEY")):
            raise RuntimeError("OPENAI_API_KEY is required in the environment")
        runtime["package_versions"] = dependency_versions(
            method, runtime.get("load_in_8bit", False)
        )
        runtime["models"] = model_locations(method, dataset, options)
    else:
        runtime["package_versions"], runtime["models"] = (
            saved.get("package_versions", {}),
            saved.get("models", {}),
        )
    runtime["python_version"] = sys.version.split()[0]
    return runtime


def run_experiment(args, *, adapter=None):
    load_dotenv_file(ROOT / ".env")
    resume_dir = resolve_path(args.resume_run_dir) if args.resume_run_dir else None
    state = read_json(resume_dir / "run_state.json") if resume_dir else None
    if state and (state.get("version") != VERSION or state.get("kind") != "related_baseline"):
        raise ValueError("resume directory is not a supported related-baseline run")
    saved_runtime = state["runtime"] if state else {}
    method = args.method or saved_runtime.get("method")
    dataset = args.dataset or saved_runtime.get("dataset")
    if method not in METHODS or dataset not in DATASETS:
        raise ValueError("one explicit method and dataset are required for an individual run")
    runtime = build_runtime(
        args, method, dataset, saved=saved_runtime, check_environment=adapter is None
    )
    data_dir = args.data_dir or (state["data"]["data_dir"] if state else None)
    limit = (
        args.limit
        if args.limit is not None
        else (state["data"]["selected_documents"] if state else None)
    )
    directory, documents = load_cohort(dataset, data_dir, limit)
    selected = manifest(documents)
    identity = {
        "data_dir": str(directory),
        "selected_documents": len(documents),
        "manifest_sha256": digest(selected),
    }
    if state:
        if (
            state["data"] != identity
            or state["runtime"] != runtime
            or read_jsonl(resume_dir / "document_manifest.jsonl") != selected
        ):
            raise ValueError(
                "resume document content, order, model, implementation or runtime parameters changed"
            )
        if read_jsonl(resume_dir / "source_documents.jsonl") != [
            {"document_id": d.document_id, "text": d.text} for d in documents
        ]:
            raise ValueError("resume source document snapshot changed")
    if args.preflight:
        print(f"Preflight OK: {method}/{dataset}, documents={len(documents)}")
        return None
    random.seed(runtime["seed"])
    if resume_dir:
        run_dir = resume_dir
        complete = load_checkpoints(run_dir, selected)
        need_adapter = len(complete) != len(selected) or any(
            c["status"] != "ok" for c in complete.values()
        )
    else:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", runtime["model"]).strip("-._").lower()
        if runtime["model"] == DEFAULT_MODEL:
            slug = "deepseek-v4-flash"
        if not slug:
            raise ValueError("model does not yield a valid output directory name")
        run_dir = (
            resolve_path(args.output_run_dir)
            if args.output_run_dir
            else resolve_path(args.output_root)
            / method
            / dataset
            / slug
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(
                "output directory is not empty; use --resume-run-dir to recover an existing run"
            )
        need_adapter = True
    if adapter is None and need_adapter:
        from evidenceloop.baselines.related.adapters import create_adapter

        adapter = create_adapter(runtime)
    if not state:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_dir / "run_state.json",
            {
                "version": VERSION,
                "kind": "related_baseline",
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "data": identity,
                "runtime": runtime,
            },
        )
        atomic_write_jsonl(run_dir / "document_manifest.jsonl", selected)
        atomic_write_jsonl(
            run_dir / "source_documents.jsonl",
            ({"document_id": d.document_id, "text": d.text} for d in documents),
        )
    summary = process_documents(adapter, documents, run_dir, runtime)
    print(
        f"Results: {run_dir}; completed={summary['samples_completed']}; failed={summary['samples_failed']}"
    )
    return run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=(*METHODS, "all"), help="default: all methods; inherited on resume"
    )
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), help="default: all datasets; inherited on resume"
    )
    parser.add_argument("--data-dir", help="one dataset directory containing documents.jsonl")
    parser.add_argument("--limit", type=int, help="override the fixed dataset cohort size")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-run-dir", help="explicit output for one method/dataset")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--model-root")
    parser.add_argument(
        "--embedding-model", help="prepared local directory or an already cached Hugging Face model"
    )
    parser.add_argument("--verifier-base-model")
    parser.add_argument("--verifier-adapter", help="local PEFT adapter directory, not its parent")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device")
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verifier-max-input-tokens", type=int)
    parser.add_argument("--verifier-batch-size", type=int)
    parser.add_argument("--pive-rounds", type=int)
    parser.add_argument("--pive-instruction")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--disambiguation-threshold", type=float)
    parser.add_argument("--entity-threshold", type=float)
    parser.add_argument("--relation-threshold", type=float)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--retry-delay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--backend-python",
        action="append",
        default=[],
        metavar="[METHOD=]PYTHON",
        help="repeat to map each method to its own Python environment",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check dependencies, local models and data without inference or downloads",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def backend_pythons(values):
    mapping = {method: sys.executable for method in METHODS}
    for value in values:
        if "=" in value:
            method, executable = value.split("=", 1)
            if method not in METHODS or not executable:
                raise ValueError("--backend-python must be PYTHON or METHOD=PYTHON")
            mapping[method] = executable
        else:
            if not value:
                raise ValueError("--backend-python must not be empty")
            mapping.update({method: value for method in METHODS})
    return mapping


def child_command(args, method, dataset, executable, *, preflight=False):
    command = [
        executable,
        "-m",
        "evidenceloop.baselines.run",
        "--_worker",
        "--method",
        method,
        "--dataset",
        dataset,
    ]
    excluded = {"method", "dataset", "backend_python", "preflight", "_worker"}
    for key, value in vars(args).items():
        if key in excluded or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            command.extend((flag, str(value)))
    if preflight:
        command.append("--preflight")
    return command


def main(argv=None):
    args = parse_args(argv)
    try:
        if args._worker:
            run_dir = run_experiment(args)
            return 1 if run_dir and read_json(run_dir / "summary.json")["samples_failed"] else 0
        if args.resume_run_dir:
            state = read_json(resolve_path(args.resume_run_dir) / "run_state.json")
            method = args.method or state["runtime"]["method"]
            dataset = args.dataset or state["runtime"]["dataset"]
            if method == "all" or dataset == "all":
                raise ValueError("resume accepts one saved method/dataset")
            pairs = [(method, dataset)]
        else:
            methods = METHODS if args.method in (None, "all") else (args.method,)
            datasets = tuple(DATASETS) if args.dataset in (None, "all") else (args.dataset,)
            if len(datasets) != 1 and args.data_dir:
                raise ValueError("--data-dir requires one --dataset")
            if len(methods) != 1 and (args.verifier_base_model or args.verifier_adapter):
                raise ValueError("custom verifier paths require one --method")
            pairs = [(m, d) for m in methods for d in datasets]
            if len(pairs) != 1 and args.output_run_dir:
                raise ValueError("--output-run-dir requires one method and dataset")
        pythons = backend_pythons(args.backend_python)
        child_env = dict(os.environ)
        seed = args.seed if args.seed is not None else 42
        if args.resume_run_dir and args.seed is None:
            seed = state["runtime"]["seed"]
        child_env["PYTHONHASHSEED"] = str(seed)
        # Preflight every selected environment before spending any inference calls.
        failed = False
        for method, dataset in pairs:
            result = subprocess.run(
                child_command(args, method, dataset, pythons[method], preflight=True),
                cwd=ROOT,
                env=child_env,
            )
            failed |= result.returncode != 0
        if failed or args.preflight:
            return 2 if failed else 0
        failed = False
        for method, dataset in pairs:
            result = subprocess.run(
                child_command(args, method, dataset, pythons[method]), cwd=ROOT, env=child_env
            )
            failed |= result.returncode != 0
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("Interrupted; resume the saved run directory to continue.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(redact(f"Error: {exc}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
