from typing import Mapping

from evidenceloop.types import DocumentPipelineRecord, Triple, TripleReviewRecord


def _triple_from_dict(value: Mapping) -> Triple:
    return Triple.from_dict(
        dict(value),
        source_text=str(value.get("source_text") or ""),
        document_id=str(value.get("document_id") or ""),
    )


def _review_from_dict(value: Mapping) -> TripleReviewRecord:
    output = value.get("output_triple")
    return TripleReviewRecord(
        input_triple=_triple_from_dict(value.get("input_triple") or {}),
        stage=str(value.get("stage") or ""),
        origin=str(value.get("origin") or ""),
        action=str(value.get("action") or ""),
        output_triple=_triple_from_dict(output) if isinstance(output, Mapping) else None,
        reason=str(value.get("reason") or ""),
        evidence=str(value.get("evidence") or ""),
        evidence_start=int(value.get("evidence_start", -1)),
        evidence_end=int(value.get("evidence_end", -1)),
        valid=bool(value.get("valid", False)),
        validation_error=str(value.get("validation_error") or ""),
        raw_result=(
            dict(value["raw_result"]) if isinstance(value.get("raw_result"), Mapping) else None
        ),
    )


def pipeline_record_from_dict(value: Mapping) -> DocumentPipelineRecord:
    return DocumentPipelineRecord(
        document_id=str(value.get("document_id") or ""),
        source_text=str(value.get("source_text") or ""),
        extracted_triples=[_triple_from_dict(item) for item in value.get("extracted_triples", [])],
        initial_reviews=[_review_from_dict(item) for item in value.get("initial_reviews", [])],
        refined_triples=[_triple_from_dict(item) for item in value.get("refined_triples", [])],
        coverage_audit=dict(value.get("coverage_audit") or {}),
        added_triples=[_triple_from_dict(item) for item in value.get("added_triples", [])],
        final_reviews=[_review_from_dict(item) for item in value.get("final_reviews", [])],
        final_triples=[_triple_from_dict(item) for item in value.get("final_triples", [])],
        stage_errors=list(value.get("stage_errors") or []),
        stage_timings_seconds=dict(value.get("stage_timings_seconds") or {}),
        raw_responses=dict(value.get("raw_responses") or {}),
    )
