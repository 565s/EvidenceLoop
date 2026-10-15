import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evidenceloop import Document, Pipeline, PipelineConfig, Triple
from evidenceloop.experiments.history import read_completed_run
from evidenceloop.io import atomic_write_json, atomic_write_jsonl, read_json_object
from evidenceloop.migration import migrate_run
from evidenceloop.pipeline.run import RunManager, digest
from evidenceloop.runtime.settings import resolve_critic_runtime


class Proposer:
    def __init__(self):
        self.calls = []

    def propose_triples(self, document_id, text):
        self.calls.append(document_id)
        return {"triples": [Triple("A", "works at", "B", text, document_id)],
                "raw_response": "proposal", "parse_error": ""}


class Critic:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def review_documents_batch(self, requests):
        self.calls.extend(requests)
        if self.fail:
            raise RuntimeError("simulated GPU error")
        return [{"payload": self.review_triples_batch(**{k: v for k, v in r.items() if k != "chunk_sizes"}),
                 "elapsed_seconds": 0.0} for r in requests]

    def review_triples_batch(self, triples, stage, origins=None):
        return {"results": [{"triple_idx": i, "action": "KEEP", "evidence": "A works at B.",
                             "triple": None} for i, _ in enumerate(triples)],
                "parse_error": "", "raw_response": "reviews"}


class Integrator:
    def audit_coverage(self, source_text, refined_triples):
        return {"coverage_complete": True, "summary": "complete", "additions": [],
                "parse_error": "", "raw_response": "audit"}


def pipeline(critic=None, config=None):
    return Pipeline(Proposer(), critic or Critic(), Integrator(),
                    config=config or PipelineConfig(pipeline_variant="proposer_critic_once"))


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_resume_retries_failure_reuses_proposals_and_checks_content(tmp_path):
    docs = [Document("one", "A works at B."), Document("two", "A works at B.")]
    first = pipeline(Critic(fail=True))
    run = RunManager(first).run(docs, tmp_path / "run")
    assert read_json_object(run / "progress.json")["failed"] == 2
    second = pipeline()
    RunManager(second).run(docs, resume_run_dir=run)
    assert second.proposer.calls == []
    assert read_json_object(run / "checkpoints/000000.json")["attempts"] == 2
    assert read_json_object(run / "progress.json")["completed"] == 2
    third = pipeline()
    RunManager(third).run(docs, resume_run_dir=run)
    assert not third.critic.calls and not third.proposer.calls
    with pytest.raises(ValueError, match="documents mismatch"):
        RunManager(third).run([replace(docs[0], text="changed"), docs[1]], resume_run_dir=run)
    with pytest.raises(ValueError, match="runtime mismatch"):
        RunManager(third, seed=43).run(docs, resume_run_dir=run)
    with pytest.raises(FileExistsError):
        RunManager(third).run(docs, run)


def test_cache_checks_model_and_content_and_ordered_prefetch(tmp_path):
    docs = [Document(str(i), "A works at B.") for i in range(11)]
    first = pipeline()
    RunManager(first, proposer_workers=4, proposal_cache_dir=tmp_path / "cache").run(docs, tmp_path / "first")
    _, restored, records = read_completed_run(tmp_path / "first")
    assert restored == docs and list(records) == [d.document_id for d in docs]
    second = pipeline()
    RunManager(second, proposer_workers=4, proposal_cache_dir=tmp_path / "cache",
               require_proposal_cache=True).run(docs, tmp_path / "second")
    assert second.proposer.calls == []
    mismatch = pipeline(config=replace(second.config, proposer_model_name="different"))
    with pytest.raises(ValueError, match="cache document/runtime mismatch"):
        RunManager(mismatch, proposal_cache_dir=tmp_path / "cache").run(docs, tmp_path / "bad")


def test_historical_migration_is_read_only_and_resumable(tmp_path):
    docs = [Document("one", "A works at B."), Document("two", "A works at B.")]
    input_dir = tmp_path / "inputs"
    atomic_write_jsonl(input_dir / "documents.jsonl", [d.__dict__ for d in docs])
    old = tmp_path / "old"
    config = PipelineConfig(pipeline_variant="proposer_critic_once")
    state = {"version": 1, "data": {"data_dir": "/old/data", "selected_documents": 2,
        "document_ids_sha256": hashlib.sha256(b"one\ntwo\n").hexdigest()},
        "runtime": {"pipeline_variant": "proposer_critic_once", "critic_runtime": resolve_critic_runtime(config.to_dict(), environ={}),
                    "critic_candidate_set_size": "pointwise"}}
    atomic_write_json(old / "run_state.json", state)
    record = pipeline().process(docs[0]).to_dict()
    for i, status in enumerate(("ok", "failed")):
        atomic_write_json(old / "checkpoints" / f"{i:06d}.json", {
            "version": 1, "index": i, "document_id": docs[i].document_id,
            "status": status, "attempts": 1, "pipeline_record": record if i == 0 else None,
            "final_triples": record["final_triples"] if i == 0 else [],
            "sample_result": {"document_id": docs[i].document_id, "status": status}})
    before = snapshot(old)
    migrated = migrate_run(old, tmp_path / "new", input_dir / "documents.jsonl",
                           dataset="rebel_sub", path_mappings=[("/old/data", str(input_dir))])
    manager = RunManager(pipeline(config=config), candidate_set="fixed1")
    assert read_json_object(migrated / "run_state.json")["runtime"] == manager.runtime_signature()
    manager.run(docs, resume_run_dir=migrated)
    assert manager.pipeline.proposer.calls == ["two"]
    assert read_json_object(migrated / "checkpoints/000001.json")["attempts"] == 2
    assert snapshot(old) == before
    changed = tmp_path / "changed.jsonl"
    atomic_write_jsonl(changed, [replace(d, text="changed").__dict__ for d in docs])
    with pytest.raises(ValueError, match="content mismatch"):
        migrate_run(old, tmp_path / "rejected", changed, dataset="rebel_sub",
                    path_mappings=[("/old/data", str(input_dir))])
    assert not (tmp_path / "rejected").exists()


def test_completed_history_continues_through_independent_stages(tmp_path):
    docs = [Document("one", "A works at B.")]
    direct = RunManager(pipeline(), stage="direct").run(docs, tmp_path / "direct")
    before = snapshot(direct)
    _, restored, records = read_completed_run(direct)
    critic = RunManager(pipeline(), stage="critic", source_records=records, source_run_dir=direct).run(restored, tmp_path / "critic")
    assert snapshot(direct) == before
    _, restored, records = read_completed_run(critic)
    full_config = PipelineConfig()
    integrated = RunManager(pipeline(config=full_config), stage="integrator", proposer_workers=4,
                            source_records=records, source_run_dir=critic).run(restored, tmp_path / "integrator")
    _, restored, records = read_completed_run(integrated)
    final = RunManager(pipeline(config=full_config), stage="final_critic",
                      source_records=records, source_run_dir=integrated).run(restored, tmp_path / "final")
    _, _, records = read_completed_run(final)
    assert records["one"]["final_reviews"][0]["origin"] == "proposer"
    cp = read_json_object(final / "checkpoints/000000.json")
    cp["pipeline_record"]["source_text"] = "tampered"
    atomic_write_json(final / "checkpoints/000000.json", cp)
    with pytest.raises(ValueError, match="manifest mismatch"):
        read_completed_run(final)
