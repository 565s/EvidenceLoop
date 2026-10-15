import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from evidenceloop.baselines import run as runner
from evidenceloop.evaluation import baselines as evaluator
from evidenceloop.io import atomic_write_json, atomic_write_jsonl
from evidenceloop.baselines.related.common import Extraction, ParseError, read_json, read_jsonl, validate_run


class FakeAdapter:
    def __init__(self, responses=None):
        self.responses, self.calls, self.stages = responses or {}, [], []

    def extract(self, document_id, text):
        self.calls.append(document_id)
        self.stages = [{"stage": "mock", "prompt": text, "response": "mock", "elapsed_seconds": 0.0}]
        result = self.responses.get(document_id, [["A", "has", "B"]])
        if isinstance(result, BaseException):
            raise result
        return Extraction(result, self.stages)


def write_data(directory, documents=None, gold=None):
    documents = documents or [{"document_id": "d1", "text": "A has B."}, {"document_id": "d2", "text": "Another document."}]
    atomic_write_jsonl(directory / "documents.jsonl", documents)
    if gold is None:
        gold = [{"document_id": d["document_id"], "triples": [["A", "has", "B"]]} for d in reversed(documents)]
    atomic_write_jsonl(directory / "gold_triples.jsonl", gold)


def args_for(tmp_path, **kwargs):
    args = runner.parse_args(["--method", "pive", "--dataset", "rebel_sub", "--data-dir", str(tmp_path / "data"),
        "--limit", "2", "--output-run-dir", str(tmp_path / "run"), "--max-attempts", "1", "--retry-delay", "0", "--base-url", "https://example.invalid/v1"])
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def run_mock(tmp_path, **kwargs):
    write_data(tmp_path / "data")
    return runner.run_experiment(args_for(tmp_path, **kwargs), adapter=FakeAdapter())


def test_inference_does_not_read_gold_and_exports_empty_document(tmp_path, monkeypatch):
    write_data(tmp_path / "data")
    (tmp_path / "data/gold_triples.jsonl").write_text("invalid gold that inference must never open", encoding="utf-8")
    original_open = Path.open

    def forbid_gold(path, *args, **kwargs):
        if path.name == "gold_triples.jsonl":
            raise AssertionError("inference opened gold")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_gold)
    run_dir = runner.run_experiment(args_for(tmp_path), adapter=FakeAdapter({"d2": []}))
    state, selected, checkpoints = validate_run(run_dir)
    assert state["runtime"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert [r["document_id"] for r in selected] == ["d1", "d2"]
    assert checkpoints[1]["status"] == "ok" and checkpoints[1]["final_triples"] == []
    assert len(read_jsonl(run_dir / "pipeline_records.jsonl")) == 2
    assert read_json(run_dir / "kg.json") == read_jsonl(run_dir / "kg.jsonl")
    assert not (run_dir / "gold_triples.jsonl").exists()


def test_failed_parse_and_resume_retries_only_failed_document(tmp_path):
    write_data(tmp_path / "data")
    run_dir = runner.run_experiment(args_for(tmp_path), adapter=FakeAdapter({"d2": ParseError("bad response")}))
    assert read_json(run_dir / "progress.json")["complete"] is False
    with pytest.raises(ValueError, match="complete successful"):
        evaluator.evaluate_target(run_dir)
    retry = FakeAdapter({"d2": []})
    args = runner.parse_args(["--resume-run-dir", str(run_dir)])
    assert runner.run_experiment(args, adapter=retry) == run_dir
    assert retry.calls == ["d2"]
    assert read_json(run_dir / "checkpoints/000001.json")["attempts"] == 2
    assert read_json(run_dir / "progress.json")["complete"] is True
    assert runner.run_experiment(args, adapter=FakeAdapter()) == run_dir


@pytest.mark.parametrize("change", ["text", "order", "model", "temperature", "limit", "pive_rounds", "implementation"])
def test_resume_refuses_changed_experiment(tmp_path, change, monkeypatch):
    run_dir = run_mock(tmp_path)
    args = runner.parse_args(["--resume-run-dir", str(run_dir)])
    if change in ("text", "order"):
        documents = read_jsonl(tmp_path / "data/documents.jsonl")
        if change == "text":
            documents[0]["text"] += " changed"
        else:
            documents.reverse()
        atomic_write_jsonl(tmp_path / "data/documents.jsonl", documents)
    elif change == "implementation":
        monkeypatch.setattr(runner, "implementation_digest", lambda: "changed")
    else:
        setattr(args, change, {"model": "other", "temperature": 0.1, "limit": 1, "pive_rounds": 4}[change])
    with pytest.raises(ValueError, match="changed"):
        runner.run_experiment(args, adapter=FakeAdapter())


def test_keyboard_interrupt_preserves_atomic_successes(tmp_path):
    write_data(tmp_path / "data")
    with pytest.raises(KeyboardInterrupt):
        runner.run_experiment(args_for(tmp_path), adapter=FakeAdapter({"d2": KeyboardInterrupt()}))
    run_dir = tmp_path / "run"
    assert (run_dir / "checkpoints/000000.json").exists()
    assert not (run_dir / "checkpoints/000001.json").exists()
    assert read_json(run_dir / "progress.json")["pending"] == 1
    adapter = FakeAdapter()
    runner.run_experiment(runner.parse_args(["--resume-run-dir", str(run_dir)]), adapter=adapter)
    assert adapter.calls == ["d2"]


def test_checkpoint_identity_tampering_is_detected(tmp_path):
    run_dir = run_mock(tmp_path)
    path = run_dir / "checkpoints/000000.json"
    checkpoint = read_json(path)
    checkpoint["pipeline_record"]["document_id"] = "wrong"
    atomic_write_json(path, checkpoint)
    with pytest.raises(ValueError, match="identity"):
        validate_run(run_dir)


def test_parent_preflights_every_environment_before_inference(monkeypatch):
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return NS(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", launch)
    assert runner.main(["--method", "all", "--dataset", "biored", "--limit", "1",
        "--backend-python", "pive=C:/Python With Space/python.exe"]) == 0
    assert len(calls) == 10
    assert all("--preflight" in cmd for cmd, _ in calls[:5])
    assert all("--preflight" not in cmd for cmd, _ in calls[5:])
    assert calls[2][0][0] == "C:/Python With Space/python.exe"
    assert all(kwargs["env"]["PYTHONHASHSEED"] == "42" and "shell" not in kwargs for _, kwargs in calls)


def test_parent_aborts_inference_if_any_backend_is_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: (calls.append(command) or NS(returncode=2)))
    assert runner.main(["--method", "all", "--dataset", "scierc"]) == 2
    assert len(calls) == 5 and all("--preflight" in c for c in calls)


@pytest.mark.parametrize("script", ["run_related_baselines", "prepare_related_baseline_models", "evaluate_related_baselines"])
def test_help_is_lazy_and_works_without_credentials(script):
    code = f'''
import sys, socket, importlib
def forbidden(*args, **kwargs): raise AssertionError("network during import/help")
socket.socket = forbidden
module = importlib.import_module({{"run_related_baselines": "evidenceloop.baselines.run", "prepare_related_baseline_models": "evidenceloop.baselines.prepare", "evaluate_related_baselines": "evidenceloop.evaluation.baselines"}}["{script}"])
assert "evidenceloop.runtime.factory" not in sys.modules
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
try: module.parse_args(["--help"])
except SystemExit as e: assert e.code == 0
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=runner.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def fake_metrics(calls):
    def evaluate(**kwargs):
        calls.append(kwargs)
        gold = read_jsonl(kwargs["gold_path"])
        skipped = sum(not r["triples"] for r in gold) if kwargs["empty_gold_policy"] == "skip" else 0
        metrics = {name: 0.5 for name in evaluator.METRIC_NAMES}
        result = {"metrics": metrics, "samples_evaluated": len(gold) - skipped,
            "samples_skipped_empty_gold": skipped, "protocol": "graphjudge_hungarian_v1"}
        atomic_write_json(kwargs["run_dir"] / "graphjudge_metrics.json", result)
        return result
    return evaluate


def test_offline_evaluation_aligns_gold_by_manifest_and_caches(tmp_path):
    run_dir = run_mock(tmp_path)
    calls = []
    result = evaluator.evaluate_target(run_dir, metric_fn=fake_metrics(calls))
    assert result["exact_match_f1"] == 1 and result["documents_evaluated"] == 2
    assert all(result[name] == 0.5 for name in evaluator.METRIC_NAMES)
    assert [r["document_id"] for r in read_jsonl(run_dir / "gold_triples.jsonl")] == ["d1", "d2"]
    assert calls[0]["entity_order_mode"] == "directed"
    assert evaluator.evaluate_target(run_dir, metric_fn=fake_metrics(calls)) == result
    assert len(calls) == 1
    gold = read_jsonl(tmp_path / "data/gold_triples.jsonl")
    gold[0]["triples"] = [["different", "has", "B"]]
    atomic_write_jsonl(tmp_path / "data/gold_triples.jsonl", gold)
    result = evaluator.evaluate_target(run_dir, metric_fn=fake_metrics(calls))
    assert len(calls) == 2 and result["exact_match_f1"] == 0.5
    evaluator.write_summary([result], tmp_path / "summary")
    assert "G-BS-F1" in (tmp_path / "summary/summary.csv").read_text(encoding="utf-8")


@pytest.mark.parametrize("dataset,expected", [("rebel_sub", 0), ("redocred", 0), ("biored", 1), ("scierc", 0)])
def test_exact_match_entity_direction_protocol(dataset, expected):
    gold = [{"document_id": "d", "triples": [["A", "has", "B"]]}]
    checkpoint = {0: {"final_triples": [{"subject": "B", "predicate": "has", "object": "A"}]}}
    assert evaluator.exact_metrics(gold, checkpoint, dataset)["exact_match_f1"] == expected


@pytest.mark.parametrize("predicate,expected", [("is compared with", 1), ("is conjoined with", 1), ("is used for", 0)])
def test_scierc_only_designated_predicates_are_symmetric(predicate, expected):
    gold = [{"document_id": "d", "triples": [["A", predicate, "B"]]}]
    checkpoint = {0: {"final_triples": [{"subject": "B", "predicate": predicate, "object": "A"}]}}
    assert evaluator.exact_metrics(gold, checkpoint, "scierc")["exact_match_f1"] == expected


def test_redocred_empty_gold_skip_reports_actual_semantic_count(tmp_path):
    run_dir = run_mock(tmp_path, dataset="redocred")
    atomic_write_jsonl(tmp_path / "data/gold_triples.jsonl", [
        {"document_id": "d1", "triples": [["A", "has", "B"]]}, {"document_id": "d2", "triples": []}])
    calls = []
    result = evaluator.evaluate_target(run_dir, metric_fn=fake_metrics(calls))
    assert result["documents_total"] == 2 and result["documents_evaluated"] == 1
    assert result["documents_skipped_empty_gold"] == 1
    assert calls[0]["empty_gold_policy"] == "skip"
    assert result["exact_match_precision"] == 0.5


@pytest.mark.parametrize("target", ["kg.jsonl", "pipeline_records.jsonl", "source_documents.jsonl"])
def test_offline_evaluation_rejects_tampered_exports(tmp_path, target):
    run_dir = run_mock(tmp_path)
    atomic_write_jsonl(run_dir / target, [])
    with pytest.raises(ValueError, match="match"):
        evaluator.evaluate_target(run_dir, metric_fn=fake_metrics([]))


def test_discovery_keeps_configurations_and_ignores_incomplete_runs(tmp_path):
    run_dir = run_mock(tmp_path)
    args = args_for(tmp_path, output_run_dir=str(tmp_path / "variant"), pive_rounds=1)
    variant = runner.run_experiment(args, adapter=FakeAdapter())
    args = args_for(tmp_path, output_run_dir=str(tmp_path / "failed"))
    runner.run_experiment(args, adapter=FakeAdapter({"d1": ParseError("invalid")}))
    assert set(evaluator.discover_runs(tmp_path)) == {run_dir, variant}


def test_real_existing_evaluator_accepts_exports_including_successful_empty_graph(tmp_path, monkeypatch):
    import evidenceloop.evaluation.graphjudge as existing
    write_data(tmp_path / "data")
    run_dir = runner.run_experiment(args_for(tmp_path), adapter=FakeAdapter({"d2": []}))
    observed = []
    monkeypatch.setattr(existing, "ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(existing, "resolve_device", lambda device: "cpu")

    def score(gold, predictions, **kwargs):
        observed.append(predictions)
        return [{name: 0.0 for name in existing.METRIC_NAMES} for _ in gold]

    monkeypatch.setattr(existing, "compute_bert_metrics", score)
    monkeypatch.setattr(existing, "compute_bleu_rouge_metrics", score)
    result = evaluator.evaluate_target(run_dir)
    assert result["documents_evaluated"] == 2 and result["protocol"] == existing.PROTOCOL
    assert observed[0][1] == [existing.NULL_TRIPLE]
    samples = read_jsonl(run_dir / "graphjudge_metrics_per_sample.jsonl")
    assert [s["document_id"] for s in samples] == ["d1", "d2"]
    assert read_json(run_dir / "checkpoints/000001.json")["status"] == "ok"


def test_discovery_and_summary_distinguish_different_document_cohorts(tmp_path):
    run_dir = run_mock(tmp_path)
    shorter = runner.run_experiment(args_for(tmp_path, limit=1, output_run_dir=str(tmp_path / "shorter")), adapter=FakeAdapter())
    assert set(evaluator.discover_runs(tmp_path)) == {run_dir, shorter}
    results = [evaluator.evaluate_target(p, metric_fn=fake_metrics([])) for p in (run_dir, shorter)]
    evaluator.write_summary(results, tmp_path / "summary")
    assert read_json(tmp_path / "summary/summary.json")["datasets_with_different_cohorts"] == ["rebel_sub"]
