"""Inference statistics, deliberately independent of gold labels and metrics."""

from collections import Counter, defaultdict


def run_statistics(checkpoints):
    records = [v["pipeline_record"] for _, v in sorted(checkpoints.items()) if v["status"] == "ok"]
    timings = defaultdict(float)
    for record in records:
        for stage, elapsed in record.get("stage_timings_seconds", {}).items():
            timings[stage] += float(elapsed)
    result = {
        "samples_requested": len(checkpoints),
        "samples_completed": len(records),
        "samples_failed": len(checkpoints) - len(records),
        "pipeline_records": len(records),
        "kg_triples": sum(len(r["final_triples"]) for r in records),
        "extracted_triples": sum(len(r["extracted_triples"]) for r in records),
        "refined_triples": sum(len(r["refined_triples"]) for r in records),
        "integrator_additions_proposed": sum(
            len(r.get("coverage_audit", {}).get("additions", [])) for r in records
        ),
        "integrator_additions_accepted": sum(len(r.get("added_triples", [])) for r in records),
        "stage_errors_total": sum(len(r.get("stage_errors", [])) for r in records),
        "stage_elapsed_seconds_total": dict(timings),
        "stage_elapsed_seconds_mean": {s: t / len(records) for s, t in timings.items()}
        if records
        else {},
        "samples": [
            v.get(
                "sample_result",
                {"document_id": v["manifest"]["document_id"], "status": v["status"]},
            )
            for _, v in sorted(checkpoints.items())
        ],
    }
    for prefix, field in (("initial", "initial_reviews"), ("final", "final_reviews")):
        reviews = [review for r in records for review in r[field]]
        result[prefix + "_critic_actions"] = dict(Counter(r["action"] for r in reviews))
        result[prefix + "_reviews_valid"] = sum(bool(r["valid"]) for r in reviews)
        result[prefix + "_reviews_fallback"] = sum(not r["valid"] for r in reviews)
    result["critic_parse_error_documents"] = sum(
        any(e.startswith(("initial_review:", "final_validation:")) for e in r["stage_errors"])
        for r in records
    )
    result["predicted_spo_total"] = result["kg_triples"]
    result["critic_candidate_set_size_counts"] = dict(
        Counter(
            str(size)
            for row in result["samples"]
            for size in (row.get("critic_candidate_set_sizes") or [])
        )
    )
    return result
