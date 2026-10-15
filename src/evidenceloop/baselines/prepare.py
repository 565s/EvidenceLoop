"""Explicit downloads/checks for baseline models, plus optional PiVe data/training."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from evidenceloop.baselines.related.common import (
    DATASETS,
    DEFAULT_MODEL_ROOT,
    EMBEDDING_MODEL,
    GJ_ADAPTER,
    GJ_BASE,
    METHODS,
    PIVE_ADAPTER,
    PIVE_BASE,
    read_json,
    resolve_path,
)
from evidenceloop.baselines.related.runtime import model_signature, validate_adapter
from evidenceloop.io import atomic_write_json

ROOT = Path.cwd()


def artifact_tasks(method="all", dataset="all"):
    methods = METHODS if method == "all" else (method,)
    tasks = {}
    if any(m in methods for m in ("rakg", "itext2kg")):
        tasks["embedding"] = {"repository": EMBEDDING_MODEL, "relative_dir": "embedding"}
    if "pive" in methods:
        tasks["pive_base"] = {"repository": PIVE_BASE, "relative_dir": "pive/base"}
        tasks["pive_adapter"] = {
            "repository": PIVE_ADAPTER,
            "relative_dir": "pive/adapter",
            "adapter": True,
        }
    if "graphjudge" in methods:
        tasks["graphjudge_base"] = {"repository": GJ_BASE, "relative_dir": "graphjudge/base"}
        datasets = tuple(DATASETS) if dataset == "all" else (dataset,)
        for domain in dict.fromkeys(
            {"rebel_sub": "rebel-sub", "scierc": "scierc"}.get(d, "genwiki") for d in datasets
        ):
            tasks[f"graphjudge_{domain}"] = {
                "repository": GJ_ADAPTER,
                "relative_dir": "graphjudge/adapters",
                "subfolder": f"llama2-7b-lora-{domain}-context",
                "adapter": True,
            }
    return tasks


def download_artifact(task, root, *, revision="main", api=None, downloader=None):
    if api is None or downloader is None:
        from huggingface_hub import HfApi, snapshot_download

        api = api or HfApi()
        downloader = downloader or snapshot_download
    info = api.model_info(task["repository"], revision=revision)
    subfolder = task.get("subfolder", "")
    prefix = subfolder + "/" if subfolder else ""
    files = [s.rfilename for s in info.siblings if s.rfilename.startswith(prefix)]
    # Avoid downloading duplicate TensorFlow/Flax/PyTorch representations of large bases.
    suffix = ".safetensors" if any(p.endswith(".safetensors") for p in files) else ".bin"
    selected = [
        p
        for p in files
        if p.endswith((suffix, ".json", ".model", ".txt")) and not p.startswith(".cache/")
    ]
    selected = [
        p for p in selected if not p.endswith(".index.json") or p.endswith(suffix + ".index.json")
    ]
    if not any(p.endswith(suffix) for p in selected):
        raise ValueError(f"{task['repository']}/{subfolder}: no model weights found")
    directory = root / task["relative_dir"]
    downloader(
        repo_id=task["repository"],
        revision=info.sha,
        allow_patterns=selected,
        local_dir=str(directory),
    )
    path = directory / subfolder if subfolder else directory
    return {
        "repository": task["repository"],
        "revision": info.sha,
        "subfolder": subfolder,
        "path": str(path.resolve()),
        "signature": model_signature(path, adapter=task.get("adapter", False)),
    }


def prepare_models(args, *, api=None, downloader=None):
    root = resolve_path(args.model_root)
    path = root / "model_manifest.json"
    saved = read_json(path) if path.exists() else {"version": 1, "artifacts": {}}
    artifacts = dict(saved["artifacts"])
    tasks = artifact_tasks(args.method, args.dataset)
    revisions = {}
    for item in args.revision:
        name, separator, revision = item.partition("=")
        if not separator or name not in tasks or not revision:
            raise ValueError(
                "--revision requires an artifact name and revision, e.g. pive_adapter=COMMIT"
            )
        revisions[name] = revision
    for name, task in tasks.items():
        if args.download:
            entry = download_artifact(
                task, root, revision=revisions.get(name, "main"), api=api, downloader=downloader
            )
        else:
            directory = (
                Path(artifacts[name]["path"])
                if name in artifacts
                else root / task["relative_dir"] / task.get("subfolder", "")
            )
            entry = {
                **artifacts.get(name, {}),
                "path": str(directory.resolve()),
                "signature": model_signature(directory, adapter=task.get("adapter", False)),
            }
        artifacts[name] = entry
        print(f"{'Prepared' if args.download else 'Checked'}: {name}: {entry['path']}")
    if "pive" in (METHODS if args.method == "all" else (args.method,)):
        validate_adapter(
            Path(artifacts["pive_base"]["path"]), Path(artifacts["pive_adapter"]["path"]), "pive"
        )
    for name in tasks:
        if name.startswith("graphjudge_") and name != "graphjudge_base":
            validate_adapter(
                Path(artifacts["graphjudge_base"]["path"]),
                Path(artifacts[name]["path"]),
                "graphjudge",
            )
    if args.download:
        atomic_write_json(
            path,
            {
                "version": 1,
                "prepared_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "artifacts": artifacts,
            },
        )
    return artifacts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--download",
        action="store_true",
        help="explicitly download weights; otherwise only check local files",
    )
    action.add_argument(
        "--convert-pive-data",
        metavar="FILE",
        help="official training JSON/JSONL or training-data ZIP",
    )
    action.add_argument(
        "--train-pive-data",
        metavar="DIR",
        help="directory containing converted train.jsonl and validation.jsonl",
    )
    parser.add_argument("--revision", action="append", default=[], metavar="ARTIFACT=REVISION")
    parser.add_argument("--training-output-dir", default=str(ROOT / "outputs/pive_verifier"))
    parser.add_argument("--base-model", help="local prepared PiVe base for optional training")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--training-resume-checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.convert_pive_data:
            from evidenceloop.baselines.related.training import convert_training_data

            convert_training_data(
                resolve_path(args.convert_pive_data),
                resolve_path(args.training_output_dir),
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
        elif args.train_pive_data:
            from evidenceloop.baselines.related.training import train_pive

            train_pive(args)
        else:
            prepare_models(args)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
