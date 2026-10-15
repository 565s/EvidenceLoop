import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evidenceloop.pipeline.core import Pipeline
from tests.reference import critic as runner
from evidenceloop.runtime.settings import CriticRuntimeError


def triple_dict(document_id: str, text: str):
    return {
        "subject": "A",
        "predicate": "relates to",
        "object": "B",
        "source_text": text,
        "document_id": document_id,
    }


def source_checkpoint(index: int, *, with_triple: bool = True):
    document_id = f"doc-{index}"
    text = f"A relates to B in document {index}."
    triples = [triple_dict(document_id, text)] if with_triple else []
    payload = {
        "triples": triples,
        "raw_response": f"raw-gpt-{index}",
        "parse_error": "",
        "rejected_predicates": [],
    }
    record = {
        "document_id": document_id,
        "source_text": text,
        "extracted_triples": triples,
        "initial_reviews": [],
        "refined_triples": triples,
        "coverage_audit": {},
        "added_triples": [],
        "final_reviews": [],
        "final_triples": triples,
        "stage_errors": [],
        "stage_timings_seconds": {"gpt_extraction": 0.25},
        "raw_responses": {"gpt_extraction": f"raw-gpt-{index}"},
    }
    return {
        "version": runner.CHECKPOINT_VERSION,
        "index": index,
        "document_id": document_id,
        "attempts": 1,
        "status": "ok",
        "error": "",
        "sample_result": {"document_id": document_id, "elapsed_seconds": 0.25},
        "payload": payload,
        "pipeline_record": record,
        "final_triples": triples,
    }


def write_baseline(
    baseline_root: Path,
    *,
    dataset: str = "biored",
    model: str = "gpt-4o",
    runtime_model: str = None,
    baseline: str = "gpt_direct_extraction",
    timestamp: str = "20260101_000000",
    count: int = 2,
    finalized: bool = True,
    completed: int = None,
):
    run_dir = baseline_root / dataset / model / timestamp
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    completed = count if completed is None else completed
    ids = [f"doc-{index}" for index in range(count)]
    state = {
        "version": 1,
        "data": {
            "requested_limit": count,
            "selected_documents": count,
            "document_ids_sha256": runner.document_ids_sha256(ids),
        },
        "runtime": {
            "baseline": baseline,
            "dataset": dataset,
            "model": runtime_model or model,
        },
    }
    progress = {
        "documents_total": count,
        "completed": completed,
        "failed": 0,
        "pending": count - completed,
        "finalized": finalized,
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    for index in range(completed):
        (checkpoint_dir / f"{index:06d}.json").write_text(
            json.dumps(source_checkpoint(index)), encoding="utf-8"
        )
    return run_dir


def source_rows():
    first = source_checkpoint(0)
    second = source_checkpoint(1, with_triple=False)
    return [
        runner.SourceDocument(
            index=value["index"],
            document_id=value["document_id"],
            text=value["pipeline_record"]["source_text"],
            proposer_payload=value["payload"],
            proposer_elapsed_seconds=0.25,
        )
        for value in (first, second)
    ]


class NeverProposer:
    def propose_triples(self, document_id, text):
        raise AssertionError("offline runner called the proposer")


class KeepBatchCritic:
    def __init__(self, *, reject_calls=False, fail_calls=False):
        self.calls = 0
        self.reject_calls = reject_calls
        self.fail_calls = fail_calls

    def review_documents_batch(self, requests):
        self.calls += 1
        if self.reject_calls:
            raise AssertionError("completed checkpoints should skip the Critic")
        if self.fail_calls:
            raise RuntimeError("critic failed")
        output = []
        for request in requests:
            results = [{
                "triple_idx": index,
                "action": "KEEP",
                "reason": "Supported by the source.",
                "evidence": triple.source_text,
            } for index, triple in enumerate(request["triples"])]
            output.append({
                "payload": {
                    "results": results,
                    "raw_response": "critic-response",
                    "parse_error": "",
                },
                "elapsed_seconds": 0.5 if results else 0.0,
                "prompt_count": 1 if results else 0,
            })
        return output


def controller(critic):
    return Pipeline(
        proposer_agent=NeverProposer(),
        critic_agent=critic,
        integrator_agent=None,
        pipeline_variant="proposer_critic_once",
    )


class GPTProposerCriticOfflineTests(unittest.TestCase):
    def test_defaults_and_cli_are_inference_only(self):
        args = runner.parse_args([])
        self.assertEqual(args.models, ["gpt-4o", "gpt-4o-mini"])
        self.assertEqual(args.critic_document_batch_size, 8)
        self.assertEqual(args.critic_prompt_batch_size, 4)
        self.assertIsNone(args.critic_base_model)
        self.assertIsNone(args.critic_adapter_path)
        self.assertIsNone(args.output_run_dir)
        self.assertEqual(
            runner.selected_dataset_keys("all"),
            ["rebel_sub", "redocred", "biored", "scierc"],
        )
        for removed in (
            "proposer_model",
            "temperature",
            "proposer_workers",
            "bert_model",
            "metric_device",
            "metric_batch_size",
        ):
            self.assertFalse(hasattr(args, removed))

    def test_custom_runtime_cli_requires_one_source_and_exact_output(self):
        invalid_cases = (
            (
                ["--critic-base-model", "models/base"],
                "require --source-run-dir or --resume-run-dir",
            ),
            (
                [
                    "--source-run-dir", "baseline/run",
                    "--critic-adapter-path", "outputs/adapter",
                ],
                "require --output-run-dir",
            ),
            (["--output-run-dir", "custom/output"], "only valid with --source-run-dir"),
        )
        for argv, message in invalid_cases:
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, message):
                runner._validate_args(runner.parse_args(argv))

        args = runner.parse_args([
            "--source-run-dir", "baseline/run",
            "--critic-base-model", "models/base",
            "--critic-adapter-path", "outputs/adapter",
            "--output-run-dir", "custom/output",
        ])
        runner._validate_args(args)
        self.assertEqual(
            runner._cli_critic_path_overrides(args),
            {
                "CRITIC_BASE_MODEL": str((runner.ROOT / "models/base").resolve()),
                "CRITIC_ADAPTER_PATH": str((runner.ROOT / "outputs/adapter").resolve()),
            },
        )

    def test_model_aliases_normalize_to_baseline_directory_names(self):
        self.assertEqual(runner.normalize_model_name("openai/gpt-4o"), "gpt-4o")
        self.assertEqual(
            runner.normalize_model_name("openai/gpt-4o-mini"), "gpt-4o-mini"
        )
        self.assertEqual(
            runner.normalize_model_name("deepseek-ai/DeepSeek-V4-Flash"),
            "deepseek-v4-flash",
        )
        self.assertEqual(
            runner.normalize_models([
                "gpt-4o",
                "openai/gpt-4o",
                "gpt-4o-mini",
                "deepseek-ai/DeepSeek-V4-Flash",
            ]),
            ["gpt-4o", "gpt-4o-mini", "deepseek-v4-flash"],
        )

    def test_complete_baseline_is_validated_and_loaded_without_metric_files(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            run_dir = write_baseline(Path(tmpdir), count=2)
            source = runner.inspect_baseline_run(run_dir)
            rows = runner.load_source_documents(source)

        self.assertEqual(source.document_count, 2)
        self.assertEqual([row.document_id for row in rows], ["doc-0", "doc-1"])
        self.assertEqual(rows[0].proposer_payload["raw_response"], "raw-gpt-0")
        self.assertEqual(rows[0].proposer_payload["triples"][0]["subject"], "A")

    def test_deepseek_direct_baseline_is_validated_and_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            run_dir = write_baseline(
                Path(tmpdir),
                count=2,
                model="deepseek-v4-flash",
                runtime_model="deepseek-ai/DeepSeek-V4-Flash",
                baseline="deepseek_direct_extraction",
            )
            source = runner.inspect_baseline_run(run_dir)
            rows = runner.load_source_documents(source)

        self.assertEqual(source.model, "deepseek-v4-flash")
        self.assertEqual(source.document_count, 2)
        self.assertEqual(rows[0].proposer_payload["triples"][0]["subject"], "A")

    def test_smoke_and_incomplete_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            root = Path(tmpdir)
            smoke = write_baseline(root, timestamp="20260101_000000", count=1)
            incomplete = write_baseline(
                root,
                timestamp="20260102_000000",
                count=2,
                finalized=False,
                completed=1,
            )
            with self.assertRaisesRegex(ValueError, "formal 2-document"):
                runner.inspect_baseline_run(smoke)
            with self.assertRaisesRegex(ValueError, "not finalized"):
                runner.inspect_baseline_run(incomplete)

    def test_source_payload_fingerprint_detects_changes_after_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            run_dir = write_baseline(Path(tmpdir), count=2)
            source = runner.inspect_baseline_run(run_dir)
            checkpoint_path = run_dir / "checkpoints" / "000001.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["payload"]["raw_response"] = "changed after selection"
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source payloads changed"):
                runner.load_source_documents(source)

    def test_discovery_uses_latest_complete_run_and_reports_missing_combinations(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            root = Path(tmpdir)
            complete = write_baseline(root, timestamp="20260101_000000", count=2)
            write_baseline(
                root,
                timestamp="20260102_000000",
                count=2,
                finalized=False,
                completed=1,
            )
            sources, skipped = runner.discover_baseline_runs(
                root,
                ["biored"],
                ["gpt-4o", "gpt-4o-mini"],
            )

        self.assertEqual([source.path for source in sources], [complete.resolve()])
        self.assertEqual(len(skipped), 1)
        self.assertIn("biored/gpt-4o-mini", skipped[0])

    def test_discovery_selects_deepseek_baseline_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            root = Path(tmpdir)
            complete = write_baseline(
                root,
                count=2,
                model="deepseek-v4-flash",
                runtime_model="deepseek-ai/DeepSeek-V4-Flash",
                baseline="deepseek_direct_extraction",
            )
            sources, skipped = runner.discover_baseline_runs(
                root,
                ["biored"],
                ["deepseek-v4-flash"],
            )

        self.assertEqual([source.path for source in sources], [complete.resolve()])
        self.assertEqual(skipped, [])

    def test_dataset_child_environment_uses_dataset_adapter_and_no_online_args(self):
        with patch.dict(os.environ, {}, clear=True):
            env = runner.critic_child_env(runner.DATASETS["biored"], 3)
        self.assertEqual(env["CRITIC_MODEL_NAME"], "critic-biored-phase2")
        self.assertIn("critic_qwen3_8b_lora_biored_phase2", env["CRITIC_ADAPTER_PATH"])
        self.assertEqual(env["CRITIC_PROMPT_BATCH_SIZE"], "3")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

        overrides = {
            "CRITIC_BASE_MODEL": "/custom/base",
            "CRITIC_ADAPTER_PATH": "/custom/adapter",
        }
        with patch.dict(os.environ, {
            "CRITIC_BASE_MODEL": "/environment/base",
            "CRITIC_ADAPTER_PATH": "/environment/adapter",
        }, clear=True):
            custom_env = runner.critic_child_env(
                runner.DATASETS["biored"], 3, overrides
            )
        self.assertEqual(custom_env["CRITIC_BASE_MODEL"], "/custom/base")
        self.assertEqual(custom_env["CRITIC_ADAPTER_PATH"], "/custom/adapter")

        source = runner.BaselineRun(
            path=Path("D:/source"),
            dataset="biored",
            model="gpt-4o",
            timestamp="stamp",
            document_count=100,
            document_ids_sha256="hash",
            source_payloads_sha256="payload-hash",
        )
        args = runner.parse_args([])
        with patch.object(
            runner,
            "critic_child_env",
            return_value={"CRITIC_ADAPTER_PATH": "adapter"},
        ) as child_env, patch.object(
            runner.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            runner._run_worker(source, Path("D:/output"), args, overrides)
        command = run.call_args.args[0]
        command_text = " ".join(str(item) for item in command)
        self.assertNotIn("api-key", command_text.casefold())
        self.assertNotIn("--proposer-model", command)
        self.assertNotIn("--proposer-workers", command)
        self.assertNotIn("--temperature", command)
        self.assertIn("--worker-source-run-dir", command)
        self.assertEqual(run.call_args.kwargs["env"]["CRITIC_ADAPTER_PATH"], "adapter")
        child_env.assert_called_once_with(runner.DATASETS["biored"], 4, overrides)

    def test_explicit_source_uses_exact_output_and_custom_runtime(self):
        source = runner.BaselineRun(
            path=Path("D:/source"),
            dataset="biored",
            model="gpt-4o",
            timestamp="stamp",
            document_count=100,
            document_ids_sha256="hash",
            source_payloads_sha256="payload-hash",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "custom-output"
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            with patch.object(runner, "_load_dotenv_file"), \
                 patch.object(runner, "inspect_baseline_run", return_value=source), \
                 patch.object(runner, "complete_source_run", return_value=output_dir) as complete:
                result = runner.main([
                    "--source-run-dir", str(root / "source"),
                    "--critic-base-model", str(base_dir),
                    "--critic-adapter-path", str(adapter_dir),
                    "--output-run-dir", str(output_dir),
                ])

        self.assertEqual(result, 0)
        call = complete.call_args
        self.assertEqual(call.args[1], output_dir.resolve())
        self.assertEqual(call.args[3], {
            "CRITIC_BASE_MODEL": str(base_dir.resolve()),
            "CRITIC_ADAPTER_PATH": str(adapter_dir.resolve()),
        })

    def test_resume_reuses_saved_critic_paths(self):
        source = runner.BaselineRun(
            path=Path("D:/source"),
            dataset="biored",
            model="gpt-4o",
            timestamp="stamp",
            document_count=100,
            document_ids_sha256="hash",
            source_payloads_sha256="payload-hash",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            state = {
                "source": {"run_dir": str(run_dir / "baseline")},
                "runtime": {
                    "critic_runtime": {
                        "base_model_path": "/saved/base",
                        "adapter_path": "/saved/adapter",
                    }
                },
            }
            (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
            overrides = runner._saved_critic_path_overrides(run_dir)
            with patch.object(runner, "_load_dotenv_file"), \
                 patch.object(runner, "inspect_baseline_run", return_value=source), \
                 patch.object(runner, "complete_source_run", return_value=run_dir) as complete:
                result = runner.main(["--resume-run-dir", str(run_dir)])

        self.assertEqual(overrides, {
            "CRITIC_BASE_MODEL": "/saved/base",
            "CRITIC_ADAPTER_PATH": "/saved/adapter",
        })
        self.assertEqual(result, 0)
        self.assertEqual(complete.call_args.args[3], overrides)

    def test_invalid_local_assets_fail_before_output_state_is_created(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            runner.DATASETS,
            {"biored": runner.DatasetSpec("biored", 2, "evidenceloop.experiments.biored")},
        ):
            root = Path(tmpdir)
            source_dir = write_baseline(root / "baselines", count=2)
            output_dir = root / "new-output"
            with patch.dict(os.environ, {
                "CRITIC_BASE_MODEL": str(root / "missing-base"),
                "CRITIC_ADAPTER_PATH": str(root / "missing-adapter"),
            }, clear=False):
                with self.assertRaises(CriticRuntimeError):
                    runner.run_offline_source(
                        source_dir,
                        output_dir,
                        document_batch_size=8,
                        prompt_batch_size=4,
                    )

            self.assertFalse(output_dir.exists())

    def test_processing_uses_cached_payload_batches_and_resume_skips_successes(self):
        rows = source_rows()
        critic = KeepBatchCritic()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoints, _ = runner.process_documents(
                controller(critic), rows, run_dir, document_batch_size=8,
                console_print=lambda message: None,
            )
            resumed_critic = KeepBatchCritic(reject_calls=True)
            resumed, _ = runner.process_documents(
                controller(resumed_critic), rows, run_dir, document_batch_size=8,
                console_print=lambda message: None,
            )

        self.assertEqual(critic.calls, 1)
        self.assertEqual(resumed_critic.calls, 0)
        self.assertTrue(all(value["status"] == "ok" for value in checkpoints.values()))
        self.assertEqual(resumed[0]["pipeline_record"]["raw_responses"]["proposer"], "raw-gpt-0")
        self.assertEqual(resumed[1]["final_triples"], [])

    def test_failed_critic_batch_is_checkpointed_and_retried(self):
        rows = source_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            failed, _ = runner.process_documents(
                controller(KeepBatchCritic(fail_calls=True)),
                rows,
                run_dir,
                document_batch_size=8,
                console_print=lambda message: None,
            )
            recovered, _ = runner.process_documents(
                controller(KeepBatchCritic()),
                rows,
                run_dir,
                document_batch_size=8,
                console_print=lambda message: None,
            )

        self.assertTrue(all(value["status"] == "failed" for value in failed.values()))
        self.assertTrue(all(value["status"] == "ok" for value in recovered.values()))
        self.assertTrue(all(value["attempts"] == 2 for value in recovered.values()))

    def test_materialized_outputs_contain_no_automatic_evaluation_artifacts(self):
        rows = source_rows()
        source = runner.BaselineRun(
            path=Path("D:/baseline"),
            dataset="biored",
            model="gpt-4o",
            timestamp="stamp",
            document_count=2,
            document_ids_sha256=runner.document_ids_sha256([row.document_id for row in rows]),
            source_payloads_sha256="payload-hash",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            runner.process_documents(
                controller(KeepBatchCritic()), rows, run_dir, document_batch_size=8,
                console_print=lambda message: None,
            )
            summary = runner.materialize_outputs(
                source,
                rows,
                run_dir,
                {"backend": "direct"},
                {"critic_prompt_count": 1},
                2.0,
            )
            names = {path.name for path in run_dir.iterdir()}

        self.assertIn("documents.jsonl", names)
        self.assertIn("processing_results.jsonl", names)
        self.assertIn("kg.jsonl", names)
        self.assertNotIn("gold_triples.jsonl", names)
        self.assertNotIn("graphjudge_metrics.json", names)
        self.assertEqual(summary["source_model"], "gpt-4o")
        serialized = json.dumps(summary).casefold()
        for forbidden in ("exact_match", "precision", "recall", "bert", "graphjudge"):
            self.assertNotIn(forbidden, serialized)

    def test_resume_state_rejects_source_or_runtime_changes(self):
        source = runner.BaselineRun(
            path=Path("D:/baseline"),
            dataset="biored",
            model="gpt-4o",
            timestamp="stamp",
            document_count=2,
            document_ids_sha256="hash",
            source_payloads_sha256="payload-hash",
        )
        state = runner._build_run_state(source, {"adapter_path": "one"}, 8, 4)
        changed = runner._build_run_state(source, {"adapter_path": "two"}, 8, 4)
        with self.assertRaisesRegex(ValueError, "runtime"):
            runner._validate_resume_state(state, changed)

    def test_critic_module_import_does_not_initialize_online_factory(self):
        command = [
            sys.executable,
            "-c",
            (
                "import sys; import evidenceloop.agents.critic as module; "
                "print(hasattr(module, 'critic')); "
                "print('evidenceloop.runtime.factory' in sys.modules)"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["False", "False"])


if __name__ == "__main__":
    unittest.main()
