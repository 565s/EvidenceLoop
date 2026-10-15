"""Lazy command routing for the installed package."""

from __future__ import annotations

import argparse
import importlib
import sys

ROUTES = {
    ("offline", "critic"): "experiments.stages",
    ("offline", "integrator"): "experiments.stages",
    ("offline", "final-critic"): "experiments.stages",
    ("data", "rebel-sft"): "data.builders.rebel_sft",
    ("data", "redocred-sft"): "data.builders.redocred_sft",
    ("data", "biored-sft"): "data.builders.biored_sft",
    ("data", "scierc-sft"): "data.builders.scierc_sft",
    ("data", "rebel-documents"): "data.builders.rebel_documents",
    ("data", "redocred-documents"): "data.builders.redocred_documents",
    ("data", "biored-documents"): "data.builders.biored_documents",
    ("data", "scierc-documents"): "data.builders.scierc_documents",
    ("data", "candidate-sets"): "data.builders.candidate_sets",
    ("data", "legacy-phase1"): "data.builders.legacy_phase1",
    ("data", "proposal-cache"): "experiments.proposal_cache",
    ("critic", "train"): "training.train",
    ("critic", "evaluate"): "training.evaluate",
    ("critic", "check"): "runtime.check",
    ("baseline", "direct"): "baselines.extract",
    ("baseline", "deepseek"): "baselines.extract",
    ("baseline", "prepare"): "baselines.prepare",
    ("baseline", "run"): "baselines.run",
    ("baseline", "train"): "baselines.prepare",
    ("evaluate", "graphjudge"): "evaluation.graphjudge",
    ("evaluate", "cs-fs"): "evaluation.cs_fs",
    ("evaluate", "baselines"): "evaluation.baselines",
    ("evaluate", "critic"): "evaluation.critic",
    ("evaluate", "stages"): "evaluation.stages",
    ("evaluate", "candidate-sets"): "evaluation.candidate_sets",
    ("report", "runs"): "reports.runs",
    ("report", "variants"): "reports.variants",
    ("report", "efficiency"): "reports.efficiency",
    ("report", "costs"): "reports.costs",
    ("report", "sft"): "reports.sft",
    ("report", "phase2"): "reports.phase2",
    ("report", "plot"): "reports.plots",
    ("report", "plot-sft"): "reports.plot_sft",
}


def run_command(argv):
    parser = argparse.ArgumentParser(
        prog="evidenceloop run",
        description="Run question-free documents; gold labels are read only by evaluate.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--pipeline-variant", choices=("full", "proposer_critic_once"))
    parser.add_argument("--output-run-dir")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--proposer-workers", type=int)
    parser.add_argument("--critic-candidate-set-size")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--proposal-cache-dir")
    parser.add_argument("--require-proposal-cache", action="store_true", default=None)
    parser.add_argument("--env-file")
    args = parser.parse_args(argv)
    from pathlib import Path

    from evidenceloop import Pipeline, PipelineConfig
    from evidenceloop.data.documents import load_documents
    from evidenceloop.io import load_dotenv_file, read_json_object
    from evidenceloop.logging import configure_logging
    from evidenceloop.pipeline.run import RunManager

    if args.env_file:
        load_dotenv_file(Path(args.env_file))
    configure_logging()
    runtime = {}
    if args.resume_run_dir:
        state = read_json_object(Path(args.resume_run_dir) / "run_state.json")
        if state.get("schema_version") != 2:
            raise ValueError("Historical checkpoints require migrate-run")
        runtime = state["runtime"]
    config = (
        PipelineConfig.from_file(args.config)
        if args.config
        else PipelineConfig(**runtime["config"])
        if runtime
        else __import__("evidenceloop.data.registry", fromlist=["dataset_config"]).dataset_config(
            args.dataset or "rebel_sub"
        )
    )
    for arg_name, runtime_name, default in (
        ("dataset", "dataset", "rebel_sub"),
        ("proposer_workers", "proposer_workers", 1),
        ("critic_candidate_set_size", "candidate_grouping", None),
        ("seed", "seed", 42),
        ("proposal_cache_dir", "proposal_cache_dir", None),
        ("require_proposal_cache", "require_proposal_cache", False),
    ):
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, runtime.get(runtime_name, default))
    if args.pipeline_variant:
        from dataclasses import replace

        config = replace(config, pipeline_variant=args.pipeline_variant)
    documents = load_documents(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        documents = documents[: args.limit]
    manager = RunManager(
        Pipeline(config=config),
        dataset=args.dataset,
        proposer_workers=args.proposer_workers,
        candidate_set=args.critic_candidate_set_size,
        seed=args.seed,
        proposal_cache_dir=args.proposal_cache_dir,
        require_proposal_cache=args.require_proposal_cache,
    )
    path = manager.run(documents, args.output_run_dir, resume_run_dir=args.resume_run_dir)
    print(path)
    return int(read_json_object(path / "progress.json")["failed"] > 0)


def validate_command(argv):
    parser = argparse.ArgumentParser(prog="evidenceloop data validate")
    parser.add_argument("path")
    parser.add_argument("--kind", choices=("documents", "sft"), default="documents")
    args = parser.parse_args(argv)
    if args.kind == "sft":
        from pathlib import Path

        from evidenceloop.data.sft import validate_critic_sft_jsonl

        print(validate_critic_sft_jsonl(Path(args.path)))
    else:
        from evidenceloop.data.documents import load_documents

        documents = load_documents(args.path)
        if len({d.document_id for d in documents}) != len(documents):
            raise ValueError("duplicate document ids")
        print(f"Valid documents: {len(documents)}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="evidenceloop", description="EvidenceLoop research reproduction toolkit"
    )
    parser.add_argument("--version", action="version", version="EvidenceLoop 0.1.0")
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "run",
            "offline",
            "data",
            "critic",
            "baseline",
            "evaluate",
            "report",
            "migrate-run",
        ),
    )
    if not argv or argv[0] in {"--help", "-h", "--version"}:
        parser.parse_args(argv)
        if not argv:
            parser.print_help()
        return 0
    command = argv[0]
    if command not in {
        "run",
        "offline",
        "data",
        "critic",
        "baseline",
        "evaluate",
        "report",
        "migrate-run",
    }:
        parser.error(f"Unknown command: {command}")
    if command == "run":
        return run_command(argv[1:])
    if command == "migrate-run":
        from evidenceloop.migration import main as migrate

        return migrate(argv[1:])
    if (
        command == "offline"
        and len(argv) > 1
        and argv[1] in {"critic", "integrator", "final-critic"}
    ):
        from evidenceloop.experiments.stages import main as offline

        return offline(argv[1], argv[2:])
    if argv[:2] == ["data", "validate"]:
        return validate_command(argv[2:])
    if argv[:2] == ["evaluate", "exact"]:
        from evidenceloop.evaluation.exact import main as exact

        return exact(argv[2:])
    choices = sorted(sub for group, sub in ROUTES if group == command)
    if command == "data":
        choices.append("validate")
    if command == "evaluate":
        choices.append("exact")
    if len(argv) < 2 or argv[1] in {"--help", "-h"}:
        group_parser = argparse.ArgumentParser(prog=f"evidenceloop {command}")
        group_parser.add_argument("action", choices=choices)
        group_parser.print_help()
        return 0
    route = ROUTES.get(tuple(argv[:2]))
    if route is None:
        parser.error(f"Unknown command; {command} actions: {', '.join(choices)}")
    if (
        argv[:2] == ["baseline", "train"]
        and not any(a in {"-h", "--help"} for a in argv[2:])
        and "--train-pive-data" not in argv[2:]
    ):
        parser.error("baseline train requires --train-pive-data DIR")
    result = importlib.import_module("evidenceloop." + route).main(argv[2:])
    return result if isinstance(result, int) else 0


def entrypoint():
    try:
        return main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"evidenceloop: {exc}", file=sys.stderr)
        return 2
