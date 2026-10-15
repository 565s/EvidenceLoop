import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from evidenceloop.reports.runs import build_comparison
from tests.reference.checkpoints import (
    build_proposal_checkpoint,
    build_run_state,
    build_summary,
    cumulative_critic_metrics,
    document_ids_sha256,
    load_graphjudge_rebel,
    materialize_run_outputs,
    parse_gold_triples,
    process_rows_with_concurrent_proposals,
    process_rows_with_checkpoints,
    runtime_config,
    runtime_signature,
    validate_resume_state,
)
from evidenceloop.types import DocumentPipelineRecord, Triple


class FakePipelineController:
    def __init__(self, *, interrupt_id=None, fail_id=None):
        self.interrupt_id = interrupt_id
        self.fail_id = fail_id
        self.calls = []
        self.kg = []
        self.pipeline_records = []

    def process_document(self, document_id, text):
        self.calls.append(document_id)
        if document_id == self.interrupt_id:
            raise KeyboardInterrupt("simulated interruption")
        if document_id == self.fail_id:
            raise RuntimeError("simulated failure")
        triple = Triple(
            subject=document_id,
            predicate="supported by",
            object="gold",
            source_text=text,
            document_id=document_id,
        )
        record = DocumentPipelineRecord(
            document_id=document_id,
            source_text=text,
            final_triples=[triple],
            stage_timings_seconds={
                "proposer": 0.1,
                "initial_critic": 0.2,
                "integrator": 0.1,
                "final_critic": 0.2,
            },
        )
        self.kg.append(triple)
        self.pipeline_records.append(record)
        return record


class ConcurrentFakeProposer:
    def __init__(self, *, invalid=False, reject_calls=False):
        self.invalid = invalid
        self.reject_calls = reject_calls
        self.calls = []
        self.completion_order = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def propose_triples(self, document_id, text):
        if self.reject_calls:
            raise AssertionError("cached proposal should have been reused")
        with self.lock:
            self.calls.append(document_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        index = int(document_id.rsplit("_", 1)[1])
        time.sleep(0.04 if index == 0 else 0.005)
        with self.lock:
            self.active -= 1
            self.completion_order.append(document_id)
        if self.invalid:
            return {
                "triples": [],
                "raw_response": "invalid",
                "parse_error": "Proposer response did not contain valid JSON.",
                "rejected_predicates": [],
            }
        return {
            "triples": [Triple(
                subject=document_id,
                predicate="supported by",
                object="gold",
                source_text=text,
                document_id=document_id,
            )],
            "raw_response": "valid",
            "parse_error": "",
            "rejected_predicates": [],
        }


class ConcurrentFakeController:
    def __init__(self, critic=None):
        self.calls = []
        self.kg = []
        self.pipeline_records = []
        self.critic = critic
        self.injected_reviews = []

    def process_document(
        self,
        document_id,
        text,
        *,
        proposer_payload=None,
        proposer_elapsed_seconds=None,
        initial_critic_payload=None,
        initial_critic_elapsed_seconds=None,
    ):
        self.calls.append(document_id)
        self.injected_reviews.append(initial_critic_payload)
        triples = list(proposer_payload["triples"])
        record = DocumentPipelineRecord(
            document_id=document_id,
            source_text=text,
            extracted_triples=triples,
            refined_triples=triples,
            final_triples=triples,
            stage_timings_seconds={
                "proposer": proposer_elapsed_seconds,
                "initial_critic": initial_critic_elapsed_seconds or 0.01,
            },
        )
        self.kg.extend(triples)
        self.pipeline_records.append(record)
        return record


class FakeDocumentBatchCritic:
    def __init__(self):
        self.calls = []

    def review_documents_batch(self, requests):
        self.calls.append(requests)
        return [{
            "payload": {
                "results": [{
                    "triple_idx": index,
                    "action": "KEEP",
                    "reason": "Supported.",
                    "evidence": triple.source_text,
                    "triple": None,
                } for index, triple in enumerate(request["triples"])],
                "raw_response": "direct batch",
                "parse_error": "",
            },
            "elapsed_seconds": 0.25,
            "prompt_count": 1 if request["triples"] else 0,
        } for request in requests]


def checkpoint_rows(count=3):
    return [
        {
            "document_id": f"rebel_{index}",
            "text": f"Text {index}.",
            "gold_triples": [{
                "subject": f"rebel_{index}",
                "predicate": "supported by",
                "object": "gold",
            }],
        }
        for index in range(count)
    ]


class GraphJudgeExperimentScriptTests(unittest.TestCase):
    def test_load_graphjudge_rebel_source_and_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "test.source").write_text("Ada was born in London.\n", encoding="utf-8")
            (data_dir / "test.target").write_text("[('Ada', 'place of birth', 'London')]\n", encoding="utf-8")

            rows = load_graphjudge_rebel(data_dir, limit=1)

        self.assertEqual(rows[0]["document_id"], "graphjudge_rebel_0000")
        self.assertEqual(rows[0]["text"], "Ada was born in London.")
        self.assertNotIn("title", rows[0])
        self.assertNotIn("question", rows[0])
        self.assertEqual(rows[0]["gold_triples"][0]["predicate"], "place of birth")

    def test_load_canonical_rebel_preserves_ids_text_and_applies_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            documents = [
                {"document_id": "rebel_2", "text": "Line one.\n\nLine two."},
                {"document_id": "rebel_1", "text": "Another document."},
            ]
            gold = [
                {"document_id": "rebel_2", "triples": [
                    {"subject": "Line one", "predicate": "followed by", "object": "Line two"},
                ]},
                {"document_id": "rebel_1", "triples": [
                    {"subject": "Another", "predicate": "type", "object": "document"},
                ]},
            ]
            (data_dir / "documents.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
            )
            (data_dir / "gold_triples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in gold), encoding="utf-8"
            )

            rows = load_graphjudge_rebel(data_dir, limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["document_id"], "rebel_2")
        self.assertEqual(rows[0]["text"], "Line one.\n\nLine two.")
        self.assertEqual(rows[0]["gold_triples"], gold[0]["triples"])

    def test_canonical_rebel_rejects_order_and_schema_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "documents.jsonl").write_text(
                json.dumps({"document_id": "rebel_1", "text": "Text."}) + "\n",
                encoding="utf-8",
            )
            (data_dir / "gold_triples.jsonl").write_text(
                json.dumps({
                    "document_id": "rebel_2",
                    "triples": [{"subject": "S", "predicate": "p", "object": "O"}],
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ID sets differ"):
                load_graphjudge_rebel(data_dir, limit=1)

            (data_dir / "documents.jsonl").write_text(
                "".join([
                    json.dumps({"document_id": "rebel_1", "text": "Text."}) + "\n",
                    json.dumps({"document_id": "rebel_2", "text": "Text 2."}) + "\n",
                ]),
                encoding="utf-8",
            )
            (data_dir / "gold_triples.jsonl").write_text(
                "".join([
                    json.dumps({"document_id": "rebel_2", "triples": [{"subject": "S", "predicate": "p", "object": "O"}]}) + "\n",
                    json.dumps({"document_id": "rebel_1", "triples": [{"subject": "S", "predicate": "p", "object": "O"}]}) + "\n",
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "same document order"):
                load_graphjudge_rebel(data_dir, limit=1)

    def test_parse_gold_triples_filters_incomplete_items(self):
        triples = parse_gold_triples("[('Ada', 'place of birth', 'London'), ('bad', '', 'row')]")
        self.assertEqual(triples, [{
            "subject": "Ada",
            "predicate": "place of birth",
            "object": "London",
        }])

    def test_compare_runs_reports_pipeline_and_exact_metric_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_dir = root / "baseline"
            lora_dir = root / "lora"
            baseline_dir.mkdir()
            lora_dir.mkdir()
            summary = {
                "kg_triples": 1,
                "pipeline_records": 1,
                "extracted_triples": 1,
                "refined_triples": 1,
                "integrator_additions_proposed": 0,
                "integrator_additions_accepted": 0,
                "exact_match_precision": 0.0,
                "exact_match_recall": 0.0,
                "exact_match_f1": 0.0,
                "initial_reviews_fallback": 2,
                "critic_parse_error_documents": 1,
                "stage_errors_total": 3,
                "samples": [{
                    "document_id": "graphjudge_rebel_0000",
                    "gold_count": 1,
                    "predicted_count": 1,
                    "exact_match_count": 0,
                }],
            }
            (baseline_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            lora_summary = dict(summary)
            lora_summary["exact_match_f1"] = 1.0
            lora_summary["kg_triples"] = 3
            lora_summary["initial_reviews_fallback"] = 0
            lora_summary["critic_parse_error_documents"] = 0
            lora_summary["stage_errors_total"] = 1
            (lora_dir / "summary.json").write_text(json.dumps(lora_summary), encoding="utf-8")

            comparison = build_comparison(baseline_dir, lora_dir)

        self.assertEqual(comparison["delta"]["exact_match_f1"], 1.0)
        self.assertEqual(comparison["delta"]["kg_triples"], 2)
        self.assertEqual(comparison["delta"]["initial_reviews_fallback"], -2)
        self.assertEqual(comparison["delta"]["critic_parse_error_documents"], -1)
        self.assertEqual(comparison["delta"]["stage_errors_total"], -2)
        self.assertEqual(comparison["first_10_samples"][0]["document_id"], "graphjudge_rebel_0000")
        self.assertNotIn("candidate_gold_predicate_top5", comparison)

    def test_runtime_config_records_critic_stage_routing_without_api_key(self):
        args = SimpleNamespace(
            run_name="adapter-run",
            data_dir="data/graphjudge/rebel_sub",
            limit=10,
            pipeline_variant="proposer_critic_once",
            proposer_workers=4,
        )
        critic = {
            "backend": "direct",
            "batch_size": 4,
            "prompt_batch_size": 4,
            "max_tokens": 1024,
            "model_name": "critic-rebel-phase2",
            "initial_model": "critic-rebel-phase2",
            "final_model": "critic-rebel-phase2",
            "base_model_path": "/models/Qwen3-8B",
            "adapter_path": "/outputs/adapter",
        }
        value = runtime_config(
            {
                "proposer_model_name": "proposer",
                "integrator_model_name": "integrator",
            },
            args,
            critic,
        )

        self.assertEqual(value["critic_runtime"]["initial_model"], "critic-rebel-phase2")
        self.assertEqual(value["critic_runtime"]["final_model"], "critic-rebel-phase2")
        self.assertEqual(value["critic_model"], "critic-rebel-phase2")
        self.assertEqual(value["critic_runtime"]["backend"], "direct")
        self.assertEqual(value["critic_prompt_batch_size"], 4)
        self.assertEqual(value["pipeline_variant"], "proposer_critic_once")
        self.assertEqual(value["proposer_workers"], 4)
        self.assertEqual(value["proposer_prefetch_window"], 8)
        self.assertEqual(value["critic_candidate_set_size"], "max4")
        self.assertEqual(value["critic_candidate_set_seed"], 42)
        self.assertFalse(value["integrator_enabled"])
        self.assertFalse(value["final_critic_enabled"])
        self.assertNotIn("api_key", json.dumps(value))

    def test_summary_separates_critic_errors_and_stage_timings(self):
        record = SimpleNamespace(
            initial_reviews=[SimpleNamespace(action="KEEP", valid=True)],
            final_reviews=[SimpleNamespace(action="KEEP", valid=False)],
            extracted_triples=[],
            refined_triples=[],
            coverage_audit={"additions": []},
            added_triples=[],
            stage_errors=[
                "initial_review: indices 0-3: invalid JSON",
                "final_validation: indices 0-3: invalid JSON",
            ],
            stage_timings_seconds={
                "proposer": 1.0,
                "initial_critic": 2.0,
                "integrator": 3.0,
                "final_critic": 4.0,
            },
        )
        controller = SimpleNamespace(pipeline_records=[record], kg=[])

        summary = build_summary(controller, [], Path("unused"), {})

        self.assertEqual(summary["initial_critic_parse_error_documents"], 1)
        self.assertEqual(summary["final_critic_parse_error_documents"], 1)
        self.assertEqual(summary["stage_elapsed_seconds_total"]["initial_critic"], 2.0)
        self.assertEqual(summary["stage_elapsed_seconds_mean"]["final_critic"], 4.0)

    def test_direct_critic_metrics_accumulate_across_resume(self):
        merged = cumulative_critic_metrics(
            {
                "model_load_seconds": 4.0,
                "critic_prompt_count": 10,
                "critic_generation_batches": 3,
                "critic_input_tokens": 100,
                "critic_output_tokens": 40,
                "critic_generation_seconds": 2.0,
                "critic_peak_gpu_memory_bytes": 1000,
            },
            {
                "model_load_seconds": 5.0,
                "adapter_merged": True,
                "critic_prompt_count": 6,
                "critic_generation_batches": 2,
                "critic_input_tokens": 80,
                "critic_output_tokens": 20,
                "critic_generation_seconds": 1.0,
                "critic_peak_gpu_memory_bytes": 900,
            },
        )

        self.assertEqual(merged["model_load_seconds"], 9.0)
        self.assertEqual(merged["critic_prompt_count"], 16)
        self.assertEqual(merged["critic_generation_batches"], 5)
        self.assertEqual(merged["critic_peak_gpu_memory_bytes"], 1000)
        self.assertEqual(merged["critic_tokens_per_second"], 20.0)

    def test_checkpoint_resume_skips_successes_and_materializes_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rows = checkpoint_rows()
            first = FakePipelineController(interrupt_id="rebel_2")
            with self.assertRaises(KeyboardInterrupt):
                process_rows_with_checkpoints(
                    first,
                    rows,
                    run_dir / "checkpoints",
                    run_dir / "progress.json",
                    console_print=lambda message: None,
                )
            self.assertEqual(first.calls, ["rebel_0", "rebel_1", "rebel_2"])

            resumed = FakePipelineController()
            checkpoints = process_rows_with_checkpoints(
                resumed,
                rows,
                run_dir / "checkpoints",
                run_dir / "progress.json",
                console_print=lambda message: None,
            )
            summary = materialize_run_outputs(
                resumed,
                rows,
                run_dir / "checkpoints",
                run_dir,
                {},
            )

            self.assertEqual(resumed.calls, ["rebel_2"])
            self.assertEqual([checkpoints[index]["attempts"] for index in range(3)], [1, 1, 1])
            pipeline_rows = [
                json.loads(line)
                for line in (run_dir / "pipeline_records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["document_id"] for row in pipeline_rows],
                ["rebel_0", "rebel_1", "rebel_2"],
            )
            self.assertEqual(summary["samples_completed"], 3)
            self.assertEqual(summary["exact_match_f1"], 1.0)
            self.assertTrue(json.loads((run_dir / "progress.json").read_text())["finalized"])

    def test_failed_checkpoint_is_retried_and_attempt_count_increases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rows = checkpoint_rows(1)
            failed = FakePipelineController(fail_id="rebel_0")
            checkpoints = process_rows_with_checkpoints(
                failed,
                rows,
                run_dir / "checkpoints",
                run_dir / "progress.json",
                console_print=lambda message: None,
            )
            self.assertEqual(checkpoints[0]["status"], "failed")
            self.assertEqual(checkpoints[0]["attempts"], 1)

            resumed = FakePipelineController()
            checkpoints = process_rows_with_checkpoints(
                resumed,
                rows,
                run_dir / "checkpoints",
                run_dir / "progress.json",
                console_print=lambda message: None,
            )
            self.assertEqual(resumed.calls, ["rebel_0"])
            self.assertEqual(checkpoints[0]["status"], "ok")
            self.assertEqual(checkpoints[0]["attempts"], 2)

    def test_concurrent_proposer_is_bounded_and_critic_order_is_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rows = checkpoint_rows(8)
            proposer = ConcurrentFakeProposer()
            controller = ConcurrentFakeController()
            (run_dir / "progress.json").write_text(
                json.dumps({"active_wall_seconds": 3.5}),
                encoding="utf-8",
            )

            checkpoints, proposals, metrics = process_rows_with_concurrent_proposals(
                controller,
                proposer,
                rows,
                run_dir / "checkpoints",
                run_dir / "proposal_checkpoints",
                run_dir / "progress.json",
                proposer_workers=4,
                prefetch_window=8,
                console_print=lambda message: None,
                sleep_fn=lambda seconds: None,
            )

            self.assertGreater(proposer.max_active, 1)
            self.assertLessEqual(proposer.max_active, 4)
            self.assertEqual(metrics["proposal_max_in_flight"], 8)
            self.assertEqual(controller.calls, [row["document_id"] for row in rows])
            self.assertNotEqual(proposer.completion_order, controller.calls)
            self.assertEqual(len(proposals), 8)
            self.assertTrue(all(value["status"] == "ok" for value in checkpoints.values()))

            (run_dir / "checkpoints" / "000000.json").unlink()
            cached_proposer = ConcurrentFakeProposer(reject_calls=True)
            resumed = ConcurrentFakeController()
            _, _, resume_metrics = process_rows_with_concurrent_proposals(
                resumed,
                cached_proposer,
                rows,
                run_dir / "checkpoints",
                run_dir / "proposal_checkpoints",
                run_dir / "progress.json",
                proposer_workers=4,
                prefetch_window=8,
                console_print=lambda message: None,
                sleep_fn=lambda seconds: None,
            )
            self.assertEqual(resumed.calls, ["rebel_0"])
            self.assertEqual(cached_proposer.calls, [])
            self.assertEqual(resume_metrics["proposal_cache_hits"], 1)
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["proposal_completed"], 8)
            self.assertEqual(progress["proposal_failed"], 0)
            self.assertEqual(progress["active_wall_seconds"], 3.5)

    def test_concurrent_runner_batches_ordered_documents_and_injects_reviews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rows = checkpoint_rows(8)
            critic = FakeDocumentBatchCritic()
            controller = ConcurrentFakeController(critic=critic)

            process_rows_with_concurrent_proposals(
                controller,
                ConcurrentFakeProposer(),
                rows,
                run_dir / "checkpoints",
                run_dir / "proposal_checkpoints",
                run_dir / "progress.json",
                proposer_workers=4,
                prefetch_window=8,
                console_print=lambda message: None,
                sleep_fn=lambda seconds: None,
            )

        self.assertEqual(len(critic.calls), 1)
        self.assertEqual(len(critic.calls[0]), 8)
        self.assertEqual(controller.calls, [row["document_id"] for row in rows])
        self.assertTrue(all(value is not None for value in controller.injected_reviews))
        self.assertTrue(all(
            record.stage_timings_seconds["initial_critic"] == 0.25
            for record in controller.pipeline_records
        ))

    def test_valid_empty_proposal_is_cached_without_retry(self):
        class EmptyProposer:
            def __init__(self):
                self.calls = 0

            def propose_triples(self, document_id, text):
                self.calls += 1
                return {
                    "triples": [],
                    "raw_response": '{"relations": []}',
                    "parse_error": "",
                    "rejected_predicates": [],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            proposer = EmptyProposer()
            checkpoint = build_proposal_checkpoint(
                proposer,
                checkpoint_rows(1)[0],
                0,
                Path(tmpdir) / "000000.json",
                sleep_fn=lambda seconds: self.fail("valid empty output must not retry"),
            )

        self.assertEqual(proposer.calls, 1)
        self.assertEqual(checkpoint["status"], "ok")
        self.assertEqual(checkpoint["payload"]["triples"], [])

    def test_invalid_proposal_retries_three_times_then_recovers_on_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rows = checkpoint_rows(1)
            invalid = ConcurrentFakeProposer(invalid=True)
            controller = ConcurrentFakeController()
            checkpoints, proposals, metrics = process_rows_with_concurrent_proposals(
                controller,
                invalid,
                rows,
                run_dir / "checkpoints",
                run_dir / "proposal_checkpoints",
                run_dir / "progress.json",
                proposer_workers=2,
                prefetch_window=4,
                console_print=lambda message: None,
                sleep_fn=lambda seconds: None,
            )
            self.assertEqual(len(invalid.calls), 3)
            self.assertEqual(controller.calls, [])
            self.assertEqual(proposals[0]["status"], "failed")
            self.assertEqual(checkpoints[0]["status"], "failed")
            self.assertEqual(metrics["proposal_retries"], 2)

            valid = ConcurrentFakeProposer()
            resumed = ConcurrentFakeController()
            checkpoints, proposals, _ = process_rows_with_concurrent_proposals(
                resumed,
                valid,
                rows,
                run_dir / "checkpoints",
                run_dir / "proposal_checkpoints",
                run_dir / "progress.json",
                proposer_workers=2,
                prefetch_window=4,
                console_print=lambda message: None,
                sleep_fn=lambda seconds: None,
            )
            self.assertEqual(len(valid.calls), 1)
            self.assertEqual(proposals[0]["attempts"], 4)
            self.assertEqual(checkpoints[0]["status"], "ok")
            self.assertEqual(resumed.calls, ["rebel_0"])

    def test_resume_state_rejects_data_or_runtime_changes_and_hides_api_key(self):
        rows = checkpoint_rows(2)
        runtime = runtime_signature(
            {"proposer_model_name": "p", "integrator_model_name": "i"},
            {
                "backend": "direct",
                "model_name": "critic-rebel-phase2",
                "base_model_path": "/models/Qwen3-8B",
                "adapter_path": "/outputs/adapter",
                "prompt_batch_size": 4,
                "batch_size": 4,
            },
        )
        state = build_run_state(Path("data"), 2, rows, "run", runtime)
        self.assertEqual(document_ids_sha256(rows), state["data"]["document_ids_sha256"])
        self.assertNotIn("api_key", json.dumps(state))
        self.assertNotIn("base_url", json.dumps(state))
        validate_resume_state(state, dict(state))

        changed_data = json.loads(json.dumps(state))
        changed_data["data"]["requested_limit"] = 1
        with self.assertRaisesRegex(ValueError, "data directory, limit"):
            validate_resume_state(state, changed_data)
        changed_runtime = json.loads(json.dumps(state))
        changed_runtime["runtime"]["critic_runtime"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "model routing"):
            validate_resume_state(state, changed_runtime)
        changed_variant = json.loads(json.dumps(state))
        changed_variant["runtime"]["pipeline_variant"] = "proposer_critic_once"
        with self.assertRaisesRegex(ValueError, "model routing"):
            validate_resume_state(state, changed_variant)

        changed_workers = json.loads(json.dumps(state))
        changed_workers["runtime"]["proposer_workers"] = 4
        changed_workers["runtime"]["proposer_prefetch_window"] = 8
        with self.assertRaisesRegex(ValueError, "model routing"):
            validate_resume_state(state, changed_workers)

        legacy_state = json.loads(json.dumps(state))
        for key in (
            "proposer_workers",
            "proposer_prefetch_window",
            "proposer_max_attempts",
        ):
            legacy_state["runtime"].pop(key)
        validate_resume_state(legacy_state, state)

        vllm_state = json.loads(json.dumps(state))
        vllm_state["runtime"]["critic_runtime"] = {
            "adapter_model": "critic-rebel-phase2",
            "adapter_base_url": "http://127.0.0.1:8000/v1",
        }
        with self.assertRaisesRegex(ValueError, "model routing"):
            validate_resume_state(vllm_state, state)


if __name__ == "__main__":
    unittest.main()
