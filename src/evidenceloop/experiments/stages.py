"""Offline stages reuse source candidates and never load gold labels."""

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from evidenceloop import Pipeline, PipelineConfig
from evidenceloop.experiments.history import read_completed_run
from evidenceloop.io import load_dotenv_file, read_json_object
from evidenceloop.pipeline.run import RunManager


def main(stage, argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop offline " + stage)
    parser.add_argument("--source-run-dir")
    parser.add_argument("--input-root", action="append", default=[])
    parser.add_argument("--output-root")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-run-dir")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--config")
    parser.add_argument("--dataset")
    parser.add_argument("--proposer-workers", type=int)
    parser.add_argument("--critic-candidate-set-size")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--critic-base-model")
    parser.add_argument("--critic-adapter-path")
    parser.add_argument("--critic-device")
    parser.add_argument("--env-file")
    args = parser.parse_args(argv)
    if args.input_root:
        return run_discovered(stage, args)
    if args.output_root or args.datasets or args.models:
        raise ValueError("--output-root/--datasets/--models require --input-root")
    if args.env_file:
        load_dotenv_file(Path(args.env_file))
    saved = (
        read_json_object(Path(args.resume_run_dir) / "run_state.json")
        if args.resume_run_dir
        else None
    )
    if saved and saved.get("schema_version") != 2:
        raise ValueError("Historical checkpoints require migrate-run")
    source_path = args.source_run_dir or (saved or {}).get("runtime", {}).get("source_run_dir")
    if not source_path:
        raise ValueError("--source-run-dir is required")
    source_state, documents, records = read_completed_run(source_path)
    source_runtime = source_state.get("runtime", {})
    stage = stage.replace("-", "_")
    if stage == "critic":
        if any(
            r.get("initial_reviews") or r.get("added_triples") or r.get("final_reviews")
            for r in records.values()
        ):
            raise ValueError("Critic1 requires direct extraction results")
    elif stage == "integrator":
        if any(
            not r.get("coverage_audit", {}).get("skipped")
            or r.get("added_triples")
            or r.get("final_reviews")
            for r in records.values()
        ):
            raise ValueError("Integrator requires results stopped after Critic1")
    else:
        if any(
            not r.get("coverage_audit")
            or r.get("coverage_audit", {}).get("skipped")
            or r.get("final_reviews")
            for r in records.values()
        ):
            raise ValueError("Final Critic requires Integrator output before final review")
    inherited_critic = None
    if saved:
        inherited_critic = saved["runtime"]["critic_runtime"]
        config = PipelineConfig(**saved["runtime"]["config"])
        dataset = args.dataset or saved["runtime"]["dataset"]
        if args.config:
            config = PipelineConfig.from_file(args.config)
    else:
        config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
        dataset = (
            args.dataset
            or source_runtime.get("dataset")
            or source_runtime.get("dataset_key")
            or (source_state.get("source") or {}).get("dataset")
            or "rebel_sub"
        )
        if not args.config:
            from evidenceloop.data.registry import dataset_config

            config = dataset_config(dataset)
        critic = (
            source_runtime.get("critic_runtime")
            or (source_state.get("source") or {}).get("critic1_runtime")
            or {}
        )
        if stage != "critic" and critic:
            inherited_critic = critic
            config = replace(
                config,
                **{
                    field: critic[key]
                    for field, key in {
                        "critic_base_model_path": "base_model_path",
                        "critic_adapter_path": "adapter_path",
                        "critic_device": "device",
                        "critic_model_name": "model_name",
                        "critic_batch_size": "batch_size",
                        "critic_prompt_batch_size": "prompt_batch_size",
                        "critic_max_tokens": "max_tokens",
                        "critic_max_seq_length": "max_seq_length",
                    }.items()
                    if key in critic
                },
            )
    overrides = {
        name: value
        for name, value in {
            "critic_base_model_path": args.critic_base_model,
            "critic_adapter_path": args.critic_adapter_path,
            "critic_device": args.critic_device,
        }.items()
        if value
    }
    config = replace(
        config,
        pipeline_variant="proposer_critic_once" if stage == "critic" else "full",
        **overrides,
    )
    runtime = saved["runtime"] if saved else {}
    workers = (
        args.proposer_workers
        if args.proposer_workers is not None
        else runtime.get("proposer_workers", 4 if stage == "integrator" else 1)
    )
    strategy = args.critic_candidate_set_size or runtime.get(
        "candidate_grouping", runtime.get("candidate_set")
    )
    seed = args.seed if args.seed is not None else runtime.get("seed", 42)
    from evidenceloop.runtime.settings import resolve_critic_runtime

    effective_runtime = resolve_critic_runtime(
        config.to_dict(), environ={} if inherited_critic else None
    )
    manager = RunManager(
        Pipeline(config=config, critic_runtime=effective_runtime),
        dataset=dataset,
        stage=stage,
        source_records=records,
        source_run_dir=source_path,
        proposer_workers=workers,
        candidate_set=strategy,
        seed=seed,
    )
    if args.dry_run:
        print(f"Validated {stage}: {source_path}, documents={len(documents)}")
        return 0
    if stage == "integrator":
        from evidenceloop.agents.integrator import IntegratorAgent
        from evidenceloop.runtime.factory import LazyRuntime, create_chat

        manager.pipeline.integrator = LazyRuntime(
            lambda: IntegratorAgent(
                "Integrator",
                "Integrator",
                llm=LazyRuntime(lambda: create_chat(config, "integrator", offline=True)),
            )
        )
    path = manager.run(documents, args.output_run_dir, resume_run_dir=args.resume_run_dir)
    print(path)
    return int(read_json_object(path / "progress.json")["failed"] > 0)


def run_discovered(stage, args):
    """Select newest complete formal cohorts, then isolate each GPU run in a process."""
    from evidenceloop.data.registry import DATASETS, canonical_dataset

    if args.source_run_dir or args.resume_run_dir or args.output_run_dir:
        raise ValueError("Discovery cannot be combined with an explicit source/output/resume run")
    if not args.output_root and not args.dry_run:
        raise ValueError("Discovery requires --output-root")
    wanted = {canonical_dataset(d) for d in args.datasets} if args.datasets else set(DATASETS)
    selected = {}
    for root in args.input_root:
        directory = Path(root).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for state_path in directory.rglob("run_state.json"):
            state = read_json_object(state_path)
            runtime = state.get("runtime") or {}
            source = state.get("source") or {}
            dataset = runtime.get("dataset") or runtime.get("dataset_key") or source.get("dataset")
            if not dataset:
                continue
            dataset = canonical_dataset(dataset)
            if dataset not in wanted:
                continue
            model = str(
                runtime.get("model")
                or runtime.get("proposer_model")
                or source.get("model")
                or (runtime.get("config") or {}).get("proposer_model_name")
                or "default"
            )
            if (
                args.models
                and model not in args.models
                and state_path.parent.parent.name not in args.models
            ):
                continue
            progress_path = state_path.parent / "progress.json"
            if not progress_path.is_file():
                continue
            progress = read_json_object(progress_path)
            if (
                not progress.get("finalized")
                or progress.get("pending")
                or progress.get("failed")
                or progress.get("completed") != DATASETS[dataset].limit
            ):
                continue
            kind = state.get("kind") or runtime.get("stage")
            variant = runtime.get("pipeline_variant") or (runtime.get("config") or {}).get(
                "pipeline_variant"
            )
            eligible = (
                (
                    kind in {"direct", "gpt_baseline", "deepseek_baseline"}
                    or runtime.get("offline_proposals") is False
                )
                if stage == "critic"
                else (kind == "critic" or variant == "proposer_critic_once")
                if stage == "integrator"
                else kind == "integrator"
            )
            # Historical baseline states have no stage, but record a model and data cohort.
            if stage == "critic" and kind is None and runtime.get("model") and state.get("data"):
                eligible = True
            if not eligible:
                continue
            key = (dataset, model) if stage == "critic" else (dataset,)
            path = state_path.parent
            if key not in selected or (path.name, str(path)) > (
                selected[key].name,
                str(selected[key]),
            ):
                selected[key] = path
    if not selected:
        raise ValueError(
            "No complete formal source runs match the filters; use --source-run-dir for small cohorts"
        )
    failures = 0
    for key, source_path in sorted(selected.items()):
        command = [
            sys.executable,
            "-m",
            "evidenceloop",
            "offline",
            stage,
            "--source-run-dir",
            str(source_path),
            "--dataset",
            key[0],
        ]
        if args.dry_run:
            command.append("--dry-run")
        else:
            import re

            slug = (
                re.sub(r"[^A-Za-z0-9._-]+", "-", key[1])
                if len(key) > 1
                else source_path.parent.name
            )
            output = Path(args.output_root) / key[0] / slug / source_path.name
            command.extend(["--output-run-dir", str(output)])
        for flag, value in {
            "--config": args.config,
            "--env-file": args.env_file,
            "--proposer-workers": args.proposer_workers,
            "--critic-candidate-set-size": args.critic_candidate_set_size,
            "--seed": args.seed,
            "--critic-base-model": args.critic_base_model,
            "--critic-adapter-path": args.critic_adapter_path,
            "--critic-device": args.critic_device,
        }.items():
            if value is not None:
                command.extend([flag, str(value)])
        failures += subprocess.run(command, check=False).returncode != 0
    return int(failures > 0)
