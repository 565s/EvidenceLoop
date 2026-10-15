"""Report recorded costs for explicit current or historical run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceloop.io import _atomic_write_text, read_json_object


def summarize_run(path):
    path = Path(path).resolve()
    state = read_json_object(path / "run_state.json")
    summary = read_json_object(path / "summary.json")
    progress = read_json_object(path / "progress.json")
    runtime = state.get("runtime", {})
    completed = int(progress.get("completed", summary.get("samples_completed", 0)))
    wall = summary.get("active_wall_seconds", progress.get("active_wall_seconds"))
    generation = summary.get("critic_generation") or {}

    def metric(key, old_key):
        return summary.get(key, generation.get(old_key))

    inputs = metric("critic_input_tokens", "input_tokens")
    outputs = metric("critic_output_tokens", "output_tokens")
    return {
        "run_dir": str(path),
        "stage": state.get("kind", runtime.get("stage", "pipeline")),
        "dataset": runtime.get("dataset", runtime.get("dataset_key")),
        "completed": completed,
        "failed": int(progress.get("failed", 0)),
        "finalized": bool(progress.get("finalized")),
        "active_wall_seconds": wall,
        "documents_per_minute": completed * 60 / wall if wall else None,
        "critic_prompt_count": metric("critic_prompt_count", "prompt_count"),
        "critic_input_tokens": inputs,
        "critic_output_tokens": outputs,
        "critic_tokens_per_document": (inputs + outputs) / completed
        if inputs is not None and outputs is not None and completed
        else None,
        "critic_generation_seconds": metric("critic_generation_seconds", "generation_seconds"),
        "proposal_cache_dir": runtime.get("proposal_cache_dir"),
        "candidate_set": runtime.get("candidate_set", runtime.get("critic_candidate_set_size")),
        "api_tokens": None,
    }


def render_report(rows):
    columns = (
        "dataset",
        "stage",
        "completed",
        "failed",
        "active_wall_seconds",
        "documents_per_minute",
        "critic_prompt_count",
        "critic_input_tokens",
        "critic_output_tokens",
        "critic_tokens_per_document",
    )

    def cell(value):
        if value is None:
            return "unavailable"
        return f"{value:.3f}" if isinstance(value, float) else str(value).replace("|", "\\|")

    lines = [
        "# Recorded inference costs",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(cell(row[key]) for key in columns) + " |" for row in rows]
    lines += [
        "",
        "API token usage was not persisted by the historical extraction agents; missing values are unavailable.",
        "Wall time and throughput describe the recorded stage. Cached proposals exclude their original API cost.",
        "GPU generation totals and document stage timers have different scopes; concurrent timers are not wall time.",
        "Partial or failed runs are listed as recorded. No monetary costs or missing measurements are estimated.",
        "",
        "## Sources",
        "",
    ]
    lines += [
        f"- `{row['run_dir']}` (finalized={row['finalized']}, cache={row['proposal_cache_dir']})"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop report costs")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    for target in (args.output, args.json_output):
        if target and target.exists():
            raise FileExistsError(target)
    rows = [summarize_run(path) for path in args.run_dir]
    _atomic_write_text(args.output, render_report(rows))
    if args.json_output:
        _atomic_write_text(args.json_output, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
    return 0
