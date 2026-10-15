from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from evidenceloop.reports.efficiency import (
    VARIANTS,
    collect_report,
    collect_run,
    latest_complete_run,
    main,
    raw_chunks,
    read_proposal_cache,
    select_runs,
    write_report,
)


GROUPS = {
    "fixed1": [[1] * 5, [1] * 4],
    "fixed2": [[2, 2, 1], [2, 2]],
    "fixed4": [[4, 1], [4]],
    "variable": [[2, 3], [1, 3]],
}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def triples(count):
    return [{"subject": f"A{index}", "predicate": "met", "object": f"B{index}",
             "source_text": "A met B."} for index in range(count)]


def response(groups):
    payload = json.dumps({"results": []})
    if len(groups) == 1:
        return payload
    chunks = []
    cursor = 0
    for size in groups:
        chunks.append({"start_idx": cursor, "end_idx_exclusive": cursor + size,
                       "response": payload})
        cursor += size
    return json.dumps(chunks)


def fixture(root: Path):
    cache_dir = root / "proposal_cache"
    cache_dir.mkdir()
    for index, count in enumerate((5, 4)):
        write_json(cache_dir / f"{index:06d}.json", {
            "document_id": f"doc-{index}", "status": "ok",
            "payload": {"triples": triples(count)},
        })
    runs = {variant: make_run(root, variant) for variant in VARIANTS}
    # These must never be read by an efficiency-only report.
    for run in runs.values():
        (run / "gold_triples.jsonl").write_text("INVALID GOLD", encoding="utf-8")
        (run / "graphjudge_metrics.json").write_text("INVALID METRICS", encoding="utf-8")
    return runs, cache_dir


def make_run(root, variant, timestamp="20261011_120000"):
    run = root / variant / timestamp
    run.mkdir(parents=True)
    groups_by_doc = GROUPS[variant]
    runtime = {"critic_candidate_set_size": variant,
               "pipeline_variant": "proposer_critic_once", "require_proposal_cache": True}
    write_json(run / "run_state.json", {"runtime": runtime})
    write_json(run / "progress.json", {"finalized": True, "pending": 0, "failed": 0})
    samples, records = [], []
    for index, groups in enumerate(groups_by_doc):
        count = sum(groups)
        samples.append({"document_id": f"doc-{index}", "status": "ok",
                        "extracted_count": count, "critic_candidate_set_sizes": groups})
        records.append({
            "document_id": f"doc-{index}", "source_text": "A met B.",
            "extracted_triples": triples(count), "refined_triples": triples(count),
            "added_triples": [], "initial_reviews": [{"valid": True}] * count,
            "final_reviews": [], "stage_errors": [],
            "stage_timings_seconds": {"proposer": 100, "initial_critic": 12 + index * 2},
            "raw_responses": {"proposer": "cached proposal", "initial_critic": response(groups)},
        })
    write_jsonl(run / "sample_results.jsonl", samples)
    write_jsonl(run / "pipeline_records.jsonl", records)
    counts = {}
    for groups in groups_by_doc:
        for size in groups:
            counts[str(size)] = counts.get(str(size), 0) + 1
    write_json(run / "summary.json", {
        "samples_requested": 2, "samples_completed": 2, "samples_failed": 0,
        "runtime_config": runtime, "critic_candidate_set_size_counts": counts,
        "critic_prompt_count": sum(map(len, groups_by_doc)),
        "critic_generation_batches": sum((len(groups) + 3) // 4 for groups in groups_by_doc),
        "critic_input_tokens": 100, "critic_output_tokens": 40,
        "critic_generation_seconds": 20.0, "active_wall_seconds": 30,
        "proposal_cache_hits": 2, "proposal_api_attempts": 2,
        "stage_elapsed_seconds_total": {"proposer": 200, "initial_critic": 26, "integrator": 0, "final_critic": 0},
    })
    return run


def change_json(path, **updates):
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(updates)
    write_json(path, value)


def codes(row):
    return {item["code"] for item in row["diagnostics"]}


class ReDocREDCriticEfficiencyTests(unittest.TestCase):
    def test_four_variants_prompt_counts_and_shared_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            report = collect_report(runs, cache_dir)
        self.assertTrue(report["shared_input_verified"])
        self.assertEqual(report["shared_input_triples"], 9)
        self.assertEqual(report["shared_documents"], 2)
        for row in report["variants"]:
            expected = sum(map(len, GROUPS[row["variant"]]))
            self.assertEqual(row["planned_prompt_count"], expected)
            self.assertEqual(row["observed_prompt_count"], expected)
            self.assertEqual(row["observed_prompts_per_document"], expected / 2)
            self.assertEqual(row["planned_candidate_set_size_counts"], row["observed_candidate_set_size_counts"])
            self.assertEqual(row["diagnostics"], [])

    def test_tokens_times_throughput_and_batch_amortization(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            row = collect_run("fixed4", runs["fixed4"], read_proposal_cache(cache_dir))
        self.assertEqual((row["input_tokens"], row["output_tokens"], row["total_tokens"]), (100, 40, 140))
        self.assertEqual(row["input_tokens_per_document"], 50)
        self.assertEqual(row["output_tokens_per_document"], 20)
        self.assertEqual(row["total_tokens_per_document"], 70)
        self.assertEqual(row["generation_seconds_per_document"], 10)
        self.assertAlmostEqual(row["generation_seconds_per_prompt_amortized"], 20 / 3, places=6)
        self.assertEqual(row["generation_seconds_per_batch_mean"], 10)
        self.assertEqual(row["critic_stage_seconds_total"], 26)
        self.assertEqual(row["critic_stage_seconds_per_completed_document"], 13)
        self.assertEqual(row["runtime_prompts_per_generation_second"], 0.15)
        self.assertEqual(row["output_tokens_per_generation_second"], 2)
        self.assertEqual(row["total_tokens_per_generation_second"], 7)
        self.assertEqual(row["documents_per_active_minute"], 4)
        self.assertEqual(row["stages"]["initial_critic"]["seconds_median"], 13)
        self.assertEqual(row["stages"]["initial_critic"]["seconds_p95"], 14)

    def test_cached_proposer_and_disabled_stages_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            row = collect_run("fixed1", runs["fixed1"], read_proposal_cache(cache_dir))
        self.assertEqual(row["stages"]["proposer"]["api_calls_current_run"], 0)
        self.assertEqual(row["stages"]["proposer"]["historical_proposal_api_attempts"], 2)
        self.assertEqual(row["stages"]["initial_critic"]["document_calls"], 2)
        self.assertEqual(row["stages"]["initial_critic"]["observed_prompt_count"], 9)
        self.assertEqual(row["stages"]["initial_critic"]["generation_batches"], 3)
        self.assertEqual(row["stages"]["integrator"]["document_calls"], 0)
        self.assertEqual(row["stages"]["final_critic"]["observed_prompt_count"], 0)

    def test_full_pipeline_reports_final_and_integrator_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed4"]
            records = [json.loads(line) for line in (run / "pipeline_records.jsonl").read_text().splitlines()]
            for index, record in enumerate(records):
                record["final_reviews"] = [{"valid": True}] * len(record["refined_triples"])
                record["raw_responses"]["final_critic"] = response(GROUPS["fixed4"][index])
                record["raw_responses"]["integrator"] = "invalid integrator JSON"
                record["stage_timings_seconds"].update(final_critic=10, integrator=3)
            write_jsonl(run / "pipeline_records.jsonl", records)
            summary = json.loads((run / "summary.json").read_text())
            summary["runtime_config"]["pipeline_variant"] = "full"
            summary["critic_prompt_count"] = 6
            summary["critic_generation_batches"] = 4
            summary["stage_elapsed_seconds_total"].update(final_critic=20, integrator=6)
            write_json(run / "summary.json", summary)
            row = collect_run("fixed4", run, read_proposal_cache(cache_dir))
        self.assertEqual(row["observed_prompt_count"], 6)
        self.assertEqual(row["stages"]["final_critic"]["observed_prompt_count"], 3)
        self.assertEqual(row["stages"]["integrator"]["document_calls"], 2)
        self.assertEqual(row["critic_stage_seconds_total"], 46)
        self.assertIsNone(row["stages"]["final_critic"]["generation_batches"])

    def test_directory_labels_and_observed_groups_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed2"]
            summary = json.loads((run / "summary.json").read_text())
            summary["runtime_config"]["critic_candidate_set_size"] = "fixed1"
            summary["critic_candidate_set_size_counts"] = {"1": 9}
            summary["critic_prompt_count"] = 3
            write_json(run / "summary.json", summary)
            write_json(run / "run_state.json", {"runtime": summary["runtime_config"]})
            samples = [json.loads(line) for line in (run / "sample_results.jsonl").read_text().splitlines()]
            for sample in samples:
                sample["critic_candidate_set_sizes"] = [1] * sample["extracted_count"]
            write_jsonl(run / "sample_results.jsonl", samples)
            records = [json.loads(line) for line in (run / "pipeline_records.jsonl").read_text().splitlines()]
            for index, record in enumerate(records):
                record["raw_responses"]["initial_critic"] = response(GROUPS["fixed4"][index])
            write_jsonl(run / "pipeline_records.jsonl", records)
            row = collect_run("fixed2", run, read_proposal_cache(cache_dir))
        self.assertEqual(row["variant"], "fixed2")
        self.assertEqual(row["configured_variant"], "fixed1")
        self.assertEqual(row["planned_prompt_count"], 9)
        self.assertEqual(row["observed_prompt_count"], 3)
        self.assertEqual(row["planned_candidate_set_size_counts"], {"1": 9})
        self.assertEqual(row["observed_candidate_set_size_counts"], {"1": 1, "4": 2})
        self.assertIn("variant_config_mismatch", codes(row))
        self.assertIn("planned_observed_grouping_mismatch", codes(row))
        self.assertIn("planned_runtime_prompt_mismatch", codes(row))

    def test_saved_cumulative_metrics_remain_separate_and_diagnosed(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed4"]
            change_json(run / "summary.json", critic_prompt_count=10,
                        critic_candidate_set_size_counts={"4": 3})
            change_json(run / "progress.json", critic_input_tokens=200)
            row = collect_run("fixed4", run, read_proposal_cache(cache_dir))
        self.assertEqual(row["observed_prompt_count"], 3)
        self.assertEqual(row["runtime_prompt_count"], 10)
        self.assertEqual(row["runtime_prompts_per_generation_second"], 0.5)
        self.assertEqual(row["input_tokens"], 100)
        self.assertIn("runtime_observed_prompt_mismatch", codes(row))
        self.assertIn("summary_progress_mismatch", codes(row))
        self.assertIn("planned_distribution_mismatch", codes(row))

    def test_invalid_responses_still_count_and_stability_uses_records(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed4"]
            records = [json.loads(line) for line in (run / "pipeline_records.jsonl").read_text().splitlines()]
            records[1]["raw_responses"]["initial_critic"] = "{invalid JSON"
            records[1]["initial_reviews"][0] = {"valid": False}
            records[1]["stage_errors"] = ["initial_review: bad JSON", "initial_review[0]: missing"]
            write_jsonl(run / "pipeline_records.jsonl", records)
            row = collect_run("fixed4", run, read_proposal_cache(cache_dir))
        self.assertEqual(row["observed_prompt_count"], 3)
        self.assertEqual(row["stability"]["initial_reviews_fallback"], 1)
        self.assertEqual(row["stability"]["initial_critic_parse_error_documents"], 1)
        self.assertEqual(row["stability"]["stage_errors_total"], 2)
        self.assertAlmostEqual(row["stability"]["initial_fallback_rate"], 1 / 9, places=6)

    def test_missing_data_is_null_and_does_not_invent_tokens_or_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed4"]
            summary = json.loads((run / "summary.json").read_text())
            del summary["critic_input_tokens"]
            write_json(run / "summary.json", summary)
            records = [json.loads(line) for line in (run / "pipeline_records.jsonl").read_text().splitlines()]
            del records[0]["raw_responses"]["initial_critic"]
            del records[0]["stage_timings_seconds"]["initial_critic"]
            write_jsonl(run / "pipeline_records.jsonl", records)
            row = collect_run("fixed4", run, read_proposal_cache(cache_dir))
        self.assertIsNone(row["input_tokens"])
        self.assertIsNone(row["total_tokens"])
        self.assertIsNone(row["observed_prompt_count"])
        self.assertEqual(row["stages"]["initial_critic"]["observed_prompt_count_known"], 1)
        self.assertIsNone(row["critic_stage_seconds_total"])
        self.assertIn("missing_runtime_metrics", codes(row))
        self.assertIn("missing_observed_prompts", codes(row))

    def test_failed_documents_keep_attempt_cost_and_have_no_fabricated_records(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed4"]
            samples = [json.loads(line) for line in (run / "sample_results.jsonl").read_text().splitlines()]
            samples[1]["status"] = "failed"
            write_jsonl(run / "sample_results.jsonl", samples)
            record = json.loads((run / "pipeline_records.jsonl").read_text().splitlines()[0])
            write_jsonl(run / "pipeline_records.jsonl", [record])
            change_json(run / "summary.json", samples_completed=1, samples_failed=1)
            change_json(run / "progress.json", failed=1)
            row = collect_run("fixed4", run, read_proposal_cache(cache_dir))
        self.assertEqual(row["documents_failed"], 1)
        self.assertEqual(row["observed_prompt_count"], 2)
        self.assertEqual(row["input_tokens_per_document"], 100)
        self.assertIn("failed_documents", codes(row))

    def test_cache_compares_actual_ordered_triples_and_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            run = runs["fixed2"]
            records = [json.loads(line) for line in (run / "pipeline_records.jsonl").read_text().splitlines()]
            records[0]["extracted_triples"].reverse()
            write_jsonl(run / "pipeline_records.jsonl", records)
            report = collect_report(runs, cache_dir)
        self.assertFalse(report["shared_input_verified"])
        self.assertIn("proposal_input_mismatch", codes(report["variants"][1]))

    def test_latest_complete_selection_skips_incomplete_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, _ = fixture(root)
            newest = make_run(root, "fixed4", "20261011_130000")
            change_json(newest / "progress.json", finalized=False, pending=1)
            failed = make_run(root, "fixed4", "20261011_140000")
            change_json(failed / "summary.json", samples_failed=1)
            self.assertEqual(latest_complete_run(root / "fixed4"), runs["fixed4"])
            self.assertEqual(select_runs(root)["fixed4"], runs["fixed4"])
            self.assertEqual(select_runs(root, [f"fixed4={failed}"])["fixed4"], failed)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                select_runs(root, [f"fixed4={failed}", f"fixed4={newest}"])

    def test_missing_files_empty_runs_and_pending_runs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, cache_dir = fixture(root)
            cache = read_proposal_cache(cache_dir)
            with self.assertRaisesRegex(ValueError, "no complete"):
                latest_complete_run(cache_dir)
            run = runs["fixed4"]
            (run / "pipeline_records.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                collect_run("fixed4", run, cache)
            write_jsonl(run / "pipeline_records.jsonl", [])
            write_jsonl(run / "sample_results.jsonl", [])
            with self.assertRaisesRegex(ValueError, "empty sample"):
                collect_run("fixed4", run, cache)
            change_json(runs["fixed1"] / "progress.json", pending=1)
            with self.assertRaisesRegex(ValueError, "pending"):
                collect_run("fixed1", runs["fixed1"], cache)

    def test_raw_boundaries_validate_coverage_and_count_generated_not_valid_results(self):
        self.assertEqual(raw_chunks(response([2, 2, 1]), 5), ([2, 2, 1], 3, True))
        self.assertEqual(raw_chunks("", 4), ([4], 1, True))
        self.assertEqual(raw_chunks(None, 5), (None, None, True))
        self.assertEqual(raw_chunks(None, 0), ([], 0, True))
        self.assertEqual(raw_chunks(json.dumps({"results": []}), 4), ([4], 1, True))
        bad = json.dumps([{"start_idx": 1, "end_idx_exclusive": 5, "response": "bad"}])
        self.assertFalse(raw_chunks(bad, 5)[2])

    def test_json_csv_markdown_and_cli_are_offline_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, cache_dir = fixture(root)
            report = collect_report(runs, cache_dir)
            paths = write_report(report, root / "reports" / "efficiency")
            self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8"))["shared_input_triples"], 9)
            with paths[1].open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["observed_prompt_count"], "9")
            self.assertEqual(rows[0]["proposer_api_calls_current_run"], "0")
            self.assertIn("摊销", paths[2].read_text(encoding="utf-8"))
            self.assertIn("fixed2", paths[2].read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--run-root", str(root), "--output-prefix", str(root / "cli_report")])
            self.assertEqual(code, 0)
            self.assertTrue((root / "cli_report.md").exists())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--run-root", str(root / "missing")]), 1)

    def test_zero_generation_time_has_no_infinite_or_fake_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, cache_dir = fixture(Path(directory))
            change_json(runs["fixed4"] / "summary.json", critic_generation_seconds=0)
            row = collect_run("fixed4", runs["fixed4"], read_proposal_cache(cache_dir))
        self.assertIsNone(row["output_tokens_per_generation_second"])
        self.assertIsNone(row["runtime_prompts_per_generation_second"])
        self.assertEqual(row["generation_seconds_per_prompt_amortized"], 0)


if __name__ == "__main__":
    unittest.main()
