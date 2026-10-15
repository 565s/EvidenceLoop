import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evidenceloop.pipeline.core import Pipeline
from tests.reference import offline as common
from tests.reference import final_critic as final_runner
from tests.reference import integrator as integrator_runner
from evidenceloop.runtime.settings import CriticRuntimeError


def make_assets(root: Path):
    base = root / "base"
    adapter = root / "adapter"
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (base / name).write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    return base, adapter


def triple(document_id, source_text, subject="A", predicate="works at", obj="B"):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source_text": source_text,
        "document_id": document_id,
    }


def critic1_record(index: int, *, with_triple=True):
    document_id = f"doc-{index}"
    source_text = "A works at B. C lives in D." if with_triple else "No relation stated."
    triples = [triple(document_id, source_text)] if with_triple else []
    return {
        "document_id": document_id,
        "source_text": source_text,
        "extracted_triples": triples,
        "initial_reviews": [{
            "stage": "initial_review",
            "origin": "proposer",
            "input_triple": triples[0],
            "action": "KEEP",
            "output_triple": triples[0],
            "reason": "supported",
            "evidence": "A works at B.",
            "evidence_start": 0,
            "evidence_end": 13,
            "valid": True,
            "validation_error": "",
            "raw_result": {"triple_idx": 0, "action": "KEEP"},
        }] if triples else [],
        "refined_triples": triples,
        "coverage_audit": {
            "skipped": True,
            "pipeline_variant": "proposer_critic_once",
            "summary": "later stages disabled",
            "additions": [],
            "parse_error": "",
        },
        "added_triples": [],
        "final_reviews": [],
        "final_triples": triples,
        "stage_errors": [],
        "stage_timings_seconds": {"proposer": 0.1, "initial_critic": 0.2},
        "raw_responses": {
            "proposer": f"raw-proposer-{index}",
            "initial_critic": f"raw-critic1-{index}",
        },
    }


def make_critic1_run(
    root: Path,
    *,
    count=2,
    selected=None,
    timestamp="20260101_000000",
    finalized=True,
    runtime_paths=None,
):
    run_dir = root / timestamp
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    selected = count if selected is None else selected
    records = [critic1_record(index, with_triple=index != count - 1) for index in range(count)]
    ids = [record["document_id"] for record in records]
    base, adapter = runtime_paths or (root / "base", root / "adapter")
    state = {
        "version": 1,
        "run_name": "biored_phase2_once_500",
        "created_at": "2026-01-01T00:00:00",
        "data": {
            "data_dir": str(root / "data" / "pipeline" / "biored_phase2_sft_test"),
            "requested_limit": selected,
            "selected_documents": selected,
            "document_ids_sha256": common.document_ids_sha256(ids[:selected]),
        },
        "runtime": {
            "pipeline_variant": "proposer_critic_once",
            "critic_runtime": {
                "backend": "direct",
                "batch_size": 4,
                "prompt_batch_size": 4,
                "max_tokens": 1024,
                "max_seq_length": 8192,
                "model_name": "critic-biored-phase2",
                "base_model_path": str(base),
                "adapter_path": str(adapter),
                "device": "cuda:0",
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
                "merge_adapter": True,
                "thinking_enabled": False,
                "initial_model": "critic-biored-phase2",
                "final_model": "critic-biored-phase2",
            },
        },
    }
    progress = {
        "documents_total": selected,
        "completed": selected if finalized else max(0, selected - 1),
        "failed": 0,
        "pending": 0 if finalized else 1,
        "finalized": finalized,
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    for index, record in enumerate(records[:selected]):
        checkpoint = {
            "version": 1,
            "index": index,
            "document_id": record["document_id"],
            "attempts": 1,
            "status": "ok",
            "error": "",
            "processing_result": {
                "document_id": record["document_id"],
                "status": "ok",
            },
            "pipeline_record": record,
            "final_triples": record["final_triples"],
        }
        (checkpoint_dir / f"{index:06d}.json").write_text(
            json.dumps(checkpoint), encoding="utf-8"
        )
    return run_dir


def make_offline_baseline_critic_run(
    root: Path,
    *,
    dataset="biored",
    model="deepseek-v4-flash",
    count=2,
    timestamp="20260101_000000",
    finalized=True,
):
    run_dir = root / dataset / model / timestamp
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    records = [critic1_record(index, with_triple=index != count - 1) for index in range(count)]
    ids = [record["document_id"] for record in records]
    state = {
        "version": 1,
        "created_at": "2026-01-01T00:00:00",
        "source": {
            "run_dir": str(root / "deepseek_baselines" / dataset / model / timestamp),
            "dataset": dataset,
            "model": model,
            "timestamp": timestamp,
            "documents": count,
            "document_ids_sha256": common.document_ids_sha256(ids),
            "source_payloads_sha256": "baseline-payloads",
        },
        "runtime": {
            "pipeline_variant": "proposer_critic_once",
            "offline_proposals": True,
            "critic_runtime": {
                "backend": "direct",
                "model_name": f"critic-{dataset}-phase2",
                "base_model_path": str(root / "base"),
                "adapter_path": str(root / "adapter"),
                "device": "cuda:0",
            },
        },
    }
    progress = {
        "documents_total": count,
        "completed": count if finalized else max(0, count - 1),
        "failed": 0,
        "pending": 0 if finalized else 1,
        "finalized": finalized,
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    for index, record in enumerate(records):
        checkpoint = {
            "version": 1,
            "index": index,
            "document_id": record["document_id"],
            "attempts": 1,
            "status": "ok",
            "error": "",
            "processing_result": {
                "document_id": record["document_id"],
                "status": "ok",
            },
            "pipeline_record": record,
            "final_triples": record["final_triples"],
        }
        (checkpoint_dir / f"{index:06d}.json").write_text(
            json.dumps(checkpoint), encoding="utf-8"
        )
    return run_dir


class FakeIntegrator:
    def __init__(self):
        self.calls = 0

    def audit_coverage(self, source_text, refined_triples):
        self.calls += 1
        additions = []
        if refined_triples:
            additions = [
                {
                    "triple": {"subject": "C", "predicate": "lives in", "object": "D"},
                    "reason": "missing supported fact",
                    "evidence": "C lives in D.",
                },
                {
                    "triple": {"subject": "A", "predicate": "works at", "object": "B"},
                    "reason": "duplicate",
                    "evidence": "A works at B.",
                },
                {
                    "triple": {"subject": "X", "predicate": "near", "object": "Y"},
                    "reason": "unsupported",
                    "evidence": "not in source",
                },
            ]
        return {
            "coverage_complete": not additions,
            "summary": "checked",
            "additions": additions,
            "raw_response": "raw-integrator",
            "parse_error": "",
        }


class SequenceIntegrator:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def audit_coverage(self, source_text, refined_triples):
        return self.payloads.pop(0)


class BarrierIntegrator(FakeIntegrator):
    def __init__(self, parties):
        super().__init__()
        self.barrier = threading.Barrier(parties)

    def audit_coverage(self, source_text, refined_triples):
        self.barrier.wait(timeout=3)
        return super().audit_coverage(source_text, refined_triples)


class FakeCritic:
    def __init__(self, fail=False, reject=False):
        self.fail = fail
        self.reject = reject
        self.calls = []

    def review_documents_batch(self, requests):
        if self.reject:
            raise AssertionError("Critic should not be called")
        self.calls.append(requests)
        if self.fail:
            raise RuntimeError("critic failed")
        outputs = []
        for request in requests:
            outputs.append({
                "payload": {
                    "results": [
                        {
                            "triple_idx": index,
                            "action": "KEEP",
                            "reason": "supported",
                            "evidence": item.source_text,
                        }
                        for index, item in enumerate(request["triples"])
                    ],
                    "raw_response": "raw-critic2",
                    "parse_error": "",
                },
                "elapsed_seconds": 0.5,
                "prompt_count": 1,
            })
        return outputs


class FakeEngine:
    def metrics(self):
        return {
            "backend": "direct",
            "adapter_merged": True,
            "critic_prompt_count": 1,
            "critic_generation_batches": 1,
            "critic_input_tokens": 10,
            "critic_output_tokens": 5,
            "critic_generation_seconds": 0.5,
        }


def final_controller(critic):
    return Pipeline(
        proposer_agent=final_runner._UnavailableStage("Proposer"),
        critic_agent=critic,
        integrator_agent=final_runner._UnavailableStage("Integrator"),
        pipeline_variant="full",
    )


class OfflineIntegratorFinalCriticTests(unittest.TestCase):
    def dataset_patch(self, root: Path, count: int):
        return patch.dict(
            common.DATASETS,
            {"biored": common.DatasetSpec("biored", count, root)},
            clear=True,
        )

    def make_integrator_output(self, root: Path, count=2):
        base, adapter = make_assets(root / "assets")
        source_dir = make_critic1_run(
            root / "sources",
            count=count,
            runtime_paths=(base, adapter),
        )
        source = common.inspect_critic1_run(source_dir, "biored")
        output_dir = root / "integrator" / "biored" / source.timestamp
        agent = FakeIntegrator()
        args = integrator_runner.parse_args([
            "--dataset", "biored",
            "--integrator-workers", "2",
        ])
        integrator_runner.run_integrator_source(
            source,
            output_dir,
            args,
            integrator=agent,
            controller=integrator_runner.build_controller(agent),
        )
        return source_dir, output_dir, base, adapter

    def test_discovery_selects_latest_complete_formal_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root, 2):
                complete = make_critic1_run(root, count=2, timestamp="20260101_000000")
                make_critic1_run(
                    root, count=2, selected=1, timestamp="20260102_000000"
                )
                make_critic1_run(
                    root,
                    count=2,
                    timestamp="20260103_000000",
                    finalized=False,
                )
                sources, skipped = common.discover_latest_runs(
                    {"biored": root}, ["biored"], common.inspect_critic1_run
                )
        self.assertEqual([item.path for item in sources], [complete.resolve()])
        self.assertEqual(skipped, [])

    def test_offline_baseline_critic_output_is_valid_integrator_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root / "unused", 2):
                run_dir = make_offline_baseline_critic_run(root, count=2)
                source = common.inspect_critic1_run(run_dir, "biored")
                rows = common.load_source_documents(source)

        self.assertEqual(source.path, run_dir.resolve())
        self.assertEqual(source.dataset, "biored")
        self.assertEqual(source.timestamp, "20260101_000000")
        self.assertEqual(source.source_kind, "critic1")
        self.assertEqual([row.document_id for row in rows], ["doc-0", "doc-1"])

    def test_integrator_input_root_scans_nested_model_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "deepseek_proposer_critic"
            output_root = root / "deepseek_proposer_critic_integrator"
            with self.dataset_patch(root / "unused", 2):
                selected = make_offline_baseline_critic_run(
                    input_root,
                    count=2,
                    timestamp="20260102_000000",
                )
                make_offline_baseline_critic_run(
                    input_root,
                    count=2,
                    timestamp="20260103_000000",
                    finalized=False,
                )
                with patch.object(
                    integrator_runner,
                    "run_integrator_source",
                    return_value={},
                ) as run_source:
                    result = integrator_runner.main([
                        "--dataset",
                        "biored",
                        "--input-root",
                        str(input_root),
                        "--output-root",
                        str(output_root),
                    ])

        self.assertEqual(result, 0)
        source_arg, output_arg, _ = run_source.call_args.args
        self.assertEqual(source_arg.path, selected.resolve())
        self.assertEqual(output_arg, output_root.resolve() / "biored" / "20260102_000000")

    def test_integrator_preserves_critic1_and_rejects_invalid_additions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root / "sources", 2):
                _, output_dir, _, _ = self.make_integrator_output(root)
                output = common.inspect_integrator_run(output_dir, "biored")
                rows = common.load_source_documents(output)
                first = rows[0].pipeline_record
                names = {path.name for path in output_dir.iterdir()}

        self.assertEqual(first["raw_responses"]["proposer"], "raw-proposer-0")
        self.assertEqual(first["raw_responses"]["initial_critic"], "raw-critic1-0")
        self.assertEqual(first["raw_responses"]["integrator"], "raw-integrator")
        self.assertEqual(len(first["added_triples"]), 1)
        self.assertEqual(len(first["coverage_audit"]["additions"]), 3)
        self.assertFalse(first["coverage_audit"]["additions"][1]["accepted"])
        self.assertFalse(first["coverage_audit"]["additions"][2]["accepted"])
        self.assertEqual(first["final_reviews"], [])
        self.assertIn("documents.jsonl", names)
        self.assertIn("run.log", names)
        self.assertNotIn("gold_triples.jsonl", names)
        self.assertFalse(any("metric" in name.casefold() for name in names))

    def test_integrator_retries_invalid_json(self):
        row = common.SourceDocument(0, "doc-0", "text", critic1_record(0))
        agent = SequenceIntegrator([
            {"parse_error": "bad JSON"},
            {
                "coverage_complete": True,
                "summary": "done",
                "additions": [],
                "raw_response": "{}",
                "parse_error": "",
            },
        ])
        with patch.object(integrator_runner.time, "sleep"):
            payload, _, attempts, error = integrator_runner.request_integrator(
                agent, row, max_attempts=3
            )
        self.assertEqual(attempts, 2)
        self.assertEqual(error, "")
        self.assertTrue(payload["coverage_complete"])

    def test_integrator_module_import_does_not_initialize_online_factory(self):
        command = [
            sys.executable,
            "-c",
            (
                "import sys; import evidenceloop.agents.integrator as module; "
                "print(hasattr(module, 'integrator')); "
                "print('evidenceloop.runtime.factory' in sys.modules)"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=common.ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["False", "False"])

    def test_integrator_uses_four_workers_with_bounded_ordered_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root / "sources", 4):
                source_dir = make_critic1_run(root / "sources", count=4)
                source = common.inspect_critic1_run(source_dir, "biored")
                rows = common.load_source_documents(source)
                agent = BarrierIntegrator(4)
                run_dir = root / "out"
                checkpoints, stats = integrator_runner.process_documents(
                    integrator_runner.build_controller(agent),
                    agent,
                    rows,
                    run_dir,
                    workers=4,
                    console_print=lambda message: None,
                )
        self.assertEqual(stats["prefetch_window"], 8)
        self.assertLessEqual(stats["max_in_flight"], 8)
        self.assertEqual(list(sorted(checkpoints)), [0, 1, 2, 3])

    def test_final_critic_preserves_origins_and_skips_empty_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root / "sources", 2):
                _, integrator_dir, _, adapter = self.make_integrator_output(root)
                critic = FakeCritic()
                output_dir = root / "final"
                final_runner.run_offline_source(
                    integrator_dir,
                    output_dir,
                    document_batch_size=8,
                    prompt_batch_size=4,
                    controller=final_controller(critic),
                    engine=FakeEngine(),
                )
                records = json.loads(
                    (output_dir / "pipeline_records.json").read_text(encoding="utf-8")
                )
                state = json.loads(
                    (output_dir / "run_state.json").read_text(encoding="utf-8")
                )
                names = {path.name for path in output_dir.iterdir()}

        self.assertEqual(len(critic.calls), 1)
        self.assertEqual(len(critic.calls[0]), 1)
        self.assertEqual(critic.calls[0][0]["origins"], ["proposer", "integrator"])
        self.assertEqual(
            [item["origin"] for item in records[0]["final_reviews"]],
            ["proposer", "integrator"],
        )
        self.assertEqual(records[1]["final_triples"], [])
        self.assertEqual(
            Path(state["runtime"]["critic_runtime"]["adapter_path"]),
            adapter.resolve(),
        )
        self.assertNotIn("gold_triples.jsonl", names)
        self.assertFalse(any("graphjudge" in name.casefold() for name in names))

    def test_final_critic_failure_is_checkpointed_retried_and_resume_skips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root / "sources", 2):
                _, integrator_dir, _, _ = self.make_integrator_output(root)
                source = common.inspect_integrator_run(integrator_dir)
                rows = common.load_source_documents(source)
                run_dir = root / "checkpoints-only"
                failed, _ = final_runner.process_documents(
                    final_controller(FakeCritic(fail=True)),
                    rows,
                    run_dir,
                    console_print=lambda message: None,
                )
                recovered_critic = FakeCritic()
                recovered, _ = final_runner.process_documents(
                    final_controller(recovered_critic),
                    rows,
                    run_dir,
                    console_print=lambda message: None,
                )
                skipped, _ = final_runner.process_documents(
                    final_controller(FakeCritic(reject=True)),
                    rows,
                    run_dir,
                    console_print=lambda message: None,
                )
        self.assertEqual(failed[0]["status"], "failed")
        self.assertTrue(all(item["status"] == "ok" for item in recovered.values()))
        self.assertEqual(recovered[0]["attempts"], 2)
        self.assertEqual(skipped[0]["attempts"], 2)

    def test_final_critic_uses_exact_dataset_adapters_in_child_environments(self):
        expected = {
            "rebel_sub": "outputs/critic_qwen3_8b_lora_phase2/adapter",
            "redocred": "outputs/critic_qwen3_8b_lora_redocred_phase2/adapter",
            "biored": "outputs/critic_qwen3_8b_lora_biored_phase2_500/adapter",
            "scierc": "outputs/critic_qwen3_8b_lora_scierc_phase2_500/adapter",
        }
        for dataset, adapter in expected.items():
            source = common.OfflineSourceRun(
                path=Path("source") / dataset,
                dataset=dataset,
                timestamp="stamp",
                document_count=1,
                document_ids_sha256="ids",
                source_records_sha256="records",
                critic_runtime={
                    "model_name": f"critic-{dataset}",
                    "base_model_path": "models/Qwen3-8B",
                    "adapter_path": adapter,
                    "device": "cuda:0",
                },
                source_kind="integrator",
            )
            env = final_runner.critic_child_env(source, 4)
            self.assertEqual(
                Path(env["CRITIC_ADAPTER_PATH"]), common.resolve_path(adapter)
            )

    def test_missing_assets_fail_before_run_state_is_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.dataset_patch(root / "sources", 2):
                _, integrator_dir, _, _ = self.make_integrator_output(root)
                output_dir = root / "missing-assets-output"
                with self.assertRaises(CriticRuntimeError):
                    final_runner.run_offline_source(
                        integrator_dir,
                        output_dir,
                        document_batch_size=8,
                        prompt_batch_size=4,
                        runtime_overrides={
                            "CRITIC_BASE_MODEL": str(root / "missing-base"),
                            "CRITIC_ADAPTER_PATH": str(root / "missing-adapter"),
                        },
                    )
                self.assertFalse((output_dir / "run_state.json").exists())

    def test_resume_state_rejects_source_runtime_or_batch_changes(self):
        source = common.OfflineSourceRun(
            path=Path("source"),
            dataset="biored",
            timestamp="stamp",
            document_count=2,
            document_ids_sha256="ids",
            source_records_sha256="records",
            critic_runtime={},
            source_kind="integrator",
        )
        one = final_runner.build_run_state(source, {"adapter_path": "one"}, 8, 4)
        two = final_runner.build_run_state(source, {"adapter_path": "two"}, 8, 4)
        with self.assertRaisesRegex(ValueError, "runtime"):
            common.validate_resume_state(one, two)

    def test_final_critic_launches_a_child_process(self):
        source = common.OfflineSourceRun(
            path=Path("source"),
            dataset="biored",
            timestamp="stamp",
            document_count=1,
            document_ids_sha256="ids",
            source_records_sha256="records",
            critic_runtime={
                "model_name": "critic-biored",
                "base_model_path": "base",
                "adapter_path": "adapter",
                "device": "cuda:0",
            },
            source_kind="integrator",
        )
        args = final_runner.parse_args([])
        with patch.object(
            final_runner.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            final_runner._run_worker(source, Path("output"), args)
        command = run.call_args.args[0]
        self.assertIn("--worker-source-run-dir", command)
        self.assertEqual(
            run.call_args.kwargs["env"]["CRITIC_ADAPTER_PATH"],
            str(common.resolve_path("adapter")),
        )


if __name__ == "__main__":
    unittest.main()
