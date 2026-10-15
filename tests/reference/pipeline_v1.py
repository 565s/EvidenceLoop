"""Original document algorithm snapshot; external I/O replaced for offline comparison."""
import time
from typing import Dict, List, Optional, Tuple
from evidenceloop.types import DocumentPipelineRecord, Triple, TripleReviewRecord
from evidenceloop.validation import clean_predicate, predicate_validation_error
VALID_REVIEW_ACTIONS = {"KEEP", "DELETE", "FIX", "REWRITE"}
class PipelineV1:
    def __init__(self, proposer, critic, integrator, variant="full"):
        self.proposer, self.critic, self.integrator = proposer, critic, integrator
        self.pipeline_variant = variant
        self.kg = []
        self.pipeline_records = []
    def _append_document(self, record):
        pass
    def _triple_key(self, triple: Triple) -> Tuple[str, str, str]:
        return (
            " ".join((triple.subject or "").lower().split()),
            " ".join((triple.predicate or "").lower().split()),
            " ".join((triple.object or "").lower().split()),
        )

    def _evidence_position(self, source_text: str, evidence: str) -> Tuple[int, int]:
        evidence = str(evidence or "")
        if not evidence:
            return -1, -1
        start = source_text.find(evidence)
        return (start, start + len(evidence)) if start >= 0 else (-1, -1)

    def _fallback_review(
        self,
        triple: Triple,
        stage: str,
        origin: str,
        error: str,
        raw_result: Optional[Dict],
    ) -> TripleReviewRecord:
        discard_addition = stage == "final_validation" and origin == "integrator"
        return TripleReviewRecord(
            input_triple=triple,
            stage=stage,
            origin=origin,
            action="DELETE" if discard_addition else "KEEP",
            output_triple=None if discard_addition else triple,
            reason="Unverified Integrator addition discarded." if discard_addition else "Fallback kept reviewed triple.",
            valid=False,
            validation_error=error,
            raw_result=raw_result,
        )

    def _review_records(
        self,
        triples: List[Triple],
        origins: List[str],
        payload: Dict,
        stage: str,
        stage_errors: List[str],
    ) -> List[TripleReviewRecord]:
        parse_error = str(payload.get("parse_error", "") or "")
        if parse_error:
            stage_errors.append(f"{stage}: {parse_error}")
        results = payload.get("results", [])
        if not isinstance(results, list):
            results = []
        by_idx = {
            result.get("triple_idx"): result
            for result in results
            if isinstance(result, dict) and isinstance(result.get("triple_idx"), int)
        }
        records = []
        for index, triple in enumerate(triples):
            origin = origins[index] if index < len(origins) else "proposer"
            raw = by_idx.get(index)
            if not isinstance(raw, dict):
                error = "Critic result is missing."
                stage_errors.append(f"{stage}[{index}]: {error}")
                records.append(self._fallback_review(triple, stage, origin, error, raw))
                continue

            action = str(raw.get("action", "") or "").strip().upper()
            reason = str(raw.get("reason", "") or "").strip()
            evidence = str(raw.get("evidence", "") or "")
            evidence_start, evidence_end = self._evidence_position(triple.source_text, evidence)
            errors = []
            if action not in VALID_REVIEW_ACTIONS:
                errors.append(f"invalid action {action!r}")
            if not reason:
                errors.append("reason is required")
            if action in {"KEEP", "FIX", "REWRITE"} and not evidence:
                errors.append("evidence is required")
            if evidence and evidence_start < 0:
                errors.append("evidence is not a source substring")

            output_triple = triple if action == "KEEP" else None
            if action in {"FIX", "REWRITE"}:
                triple_data = raw.get("triple")
                if not isinstance(triple_data, dict):
                    errors.append("corrected triple is required")
                else:
                    candidate = Triple.from_dict(
                        triple_data,
                        source_text=triple.source_text,
                        document_id=triple.document_id,
                    )
                    predicate_error = predicate_validation_error(candidate.predicate)
                    if not (candidate.subject and candidate.predicate and candidate.object):
                        errors.append("corrected triple has empty fields")
                    elif predicate_error:
                        errors.append(predicate_error)
                    else:
                        candidate.predicate = clean_predicate(candidate.predicate)
                        output_triple = candidate

            if errors:
                error = "; ".join(errors)
                stage_errors.append(f"{stage}[{index}]: {error}")
                records.append(self._fallback_review(triple, stage, origin, error, raw))
                continue

            records.append(TripleReviewRecord(
                input_triple=triple,
                stage=stage,
                origin=origin,
                action=action,
                output_triple=output_triple,
                reason=reason,
                evidence=evidence,
                evidence_start=evidence_start,
                evidence_end=evidence_end,
                raw_result=raw,
            ))
        return records

    def _apply_reviews(self, reviews: List[TripleReviewRecord]) -> List[Triple]:
        return [review.output_triple for review in reviews if review.output_triple is not None]

    def _validate_coverage_additions(
        self,
        document_id: str,
        source_text: str,
        refined_triples: List[Triple],
        payload: Dict,
        stage_errors: List[str],
    ) -> Tuple[Dict, List[Triple]]:
        parse_error = str(payload.get("parse_error", "") or "")
        if parse_error:
            stage_errors.append(f"coverage_audit: {parse_error}")
        seen = {self._triple_key(triple) for triple in refined_triples}
        accepted = []
        addition_records = []
        additions = payload.get("additions", [])
        if not isinstance(additions, list):
            additions = []

        for index, item in enumerate(additions):
            raw = item if isinstance(item, dict) else {}
            triple_data = raw.get("triple")
            reason = str(raw.get("reason", "") or "").strip()
            evidence = str(raw.get("evidence", "") or "")
            evidence_start, evidence_end = self._evidence_position(source_text, evidence)
            errors = []
            candidate = None
            if not isinstance(triple_data, dict):
                errors.append("addition triple is required")
            else:
                candidate = Triple.from_dict(triple_data, source_text=source_text, document_id=document_id)
                predicate_error = predicate_validation_error(candidate.predicate)
                if not (candidate.subject and candidate.predicate and candidate.object):
                    errors.append("addition triple has empty fields")
                elif predicate_error:
                    errors.append(predicate_error)
                else:
                    candidate.predicate = clean_predicate(candidate.predicate)
                    if self._triple_key(candidate) in seen:
                        errors.append("addition duplicates an existing document triple")
            if not reason:
                errors.append("reason is required")
            if not evidence:
                errors.append("evidence is required")
            elif evidence_start < 0:
                errors.append("evidence is not a source substring")

            valid = not errors
            if valid:
                seen.add(self._triple_key(candidate))
                accepted.append(candidate)
            else:
                stage_errors.append(f"coverage_audit[{index}]: {'; '.join(errors)}")
            addition_records.append({
                "triple": candidate.to_dict() if candidate else triple_data,
                "reason": reason,
                "evidence": evidence,
                "evidence_start": evidence_start,
                "evidence_end": evidence_end,
                "accepted": valid,
                "validation_error": "; ".join(errors),
                "raw_result": raw,
            })

        audit = {
            "coverage_complete": bool(payload.get("coverage_complete", False)),
            "summary": str(payload.get("summary", "") or ""),
            "additions": addition_records,
            "parse_error": parse_error,
        }
        return audit, accepted

    def _dedupe_document(self, triples: List[Triple]) -> List[Triple]:
        seen = set()
        output = []
        for triple in triples:
            key = self._triple_key(triple)
            if key not in seen:
                seen.add(key)
                output.append(triple)
        return output

    def run_integrator_stage(
        self,
        record: DocumentPipelineRecord,
        *,
        coverage_payload: Optional[Dict] = None,
        integrator_elapsed_seconds: Optional[float] = None,
    ) -> DocumentPipelineRecord:
        if not isinstance(record, DocumentPipelineRecord):
            raise TypeError("record must be a DocumentPipelineRecord")
        if coverage_payload is None:
            if self.integrator is None:
                raise RuntimeError("Integrator is unavailable")
            started = time.perf_counter()
            coverage_payload = self.integrator.audit_coverage(
                record.source_text,
                record.refined_triples,
            )
            integrator_elapsed_seconds = time.perf_counter() - started
        elif not isinstance(coverage_payload, dict):
            raise TypeError("coverage_payload must be a dictionary")

        record.stage_timings_seconds["integrator"] = round(
            float(integrator_elapsed_seconds or 0.0),
            6,
        )
        record.raw_responses["integrator"] = str(
            coverage_payload.get("raw_response", "") or ""
        )
        record.coverage_audit, record.added_triples = self._validate_coverage_additions(
            record.document_id,
            record.source_text,
            record.refined_triples,
            coverage_payload,
            record.stage_errors,
        )
        record.final_reviews = []
        record.final_triples = self._dedupe_document(
            record.refined_triples + record.added_triples
        )
        return record

    def run_final_critic_stage(
        self,
        record: DocumentPipelineRecord,
        *,
        final_critic_payload: Optional[Dict] = None,
        final_critic_elapsed_seconds: Optional[float] = None,
    ) -> DocumentPipelineRecord:
        if not isinstance(record, DocumentPipelineRecord):
            raise TypeError("record must be a DocumentPipelineRecord")
        combined = record.refined_triples + record.added_triples
        origins = (
            ["proposer"] * len(record.refined_triples)
            + ["integrator"] * len(record.added_triples)
        )
        if final_critic_payload is None:
            started = time.perf_counter()
            final_critic_payload = self.critic.review_triples_batch(
                combined,
                stage="final_validation",
                origins=origins,
            )
            final_critic_elapsed_seconds = time.perf_counter() - started
        elif not isinstance(final_critic_payload, dict):
            raise TypeError("final_critic_payload must be a dictionary")

        record.stage_timings_seconds["final_critic"] = round(
            float(final_critic_elapsed_seconds or 0.0),
            6,
        )
        record.raw_responses["final_critic"] = str(
            final_critic_payload.get("raw_response", "") or ""
        )
        record.final_reviews = self._review_records(
            combined,
            origins,
            final_critic_payload,
            "final_validation",
            record.stage_errors,
        )
        record.final_triples = self._dedupe_document(
            self._apply_reviews(record.final_reviews)
        )
        return record

    def process_document(
        self,
        document_id: str,
        text: str,
        *,
        proposer_payload: Optional[Dict] = None,
        proposer_elapsed_seconds: Optional[float] = None,
        initial_critic_payload: Optional[Dict] = None,
        initial_critic_elapsed_seconds: Optional[float] = None,
    ) -> DocumentPipelineRecord:
        print(f"\n--- processing document: {document_id} ---")
        record = DocumentPipelineRecord(
            document_id=document_id,
            source_text=text,
        )

        if proposer_payload is None:
            started = time.perf_counter()
            proposer_payload = self.proposer.propose_triples(document_id, text)
            proposer_elapsed_seconds = time.perf_counter() - started
        elif not isinstance(proposer_payload, dict):
            raise TypeError("proposer_payload must be a dictionary")
        record.stage_timings_seconds["proposer"] = round(
            float(proposer_elapsed_seconds or 0.0), 6
        )
        record.raw_responses["proposer"] = str(proposer_payload.get("raw_response", "") or "")
        proposer_error = str(proposer_payload.get("parse_error", "") or "")
        if proposer_error:
            record.stage_errors.append(f"proposer: {proposer_error}")
        record.extracted_triples = list(proposer_payload.get("triples", []) or [])

        if initial_critic_payload is None:
            started = time.perf_counter()
            initial_payload = self.critic.review_triples_batch(
                record.extracted_triples,
                stage="initial_review",
                origins=["proposer"] * len(record.extracted_triples),
            )
            initial_critic_elapsed_seconds = time.perf_counter() - started
        elif not isinstance(initial_critic_payload, dict):
            raise TypeError("initial_critic_payload must be a dictionary")
        else:
            initial_payload = initial_critic_payload
        record.stage_timings_seconds["initial_critic"] = round(
            float(initial_critic_elapsed_seconds or 0.0), 6
        )
        record.raw_responses["initial_critic"] = str(initial_payload.get("raw_response", "") or "")
        record.initial_reviews = self._review_records(
            record.extracted_triples,
            ["proposer"] * len(record.extracted_triples),
            initial_payload,
            "initial_review",
            record.stage_errors,
        )
        record.refined_triples = self._apply_reviews(record.initial_reviews)

        if self.pipeline_variant == "proposer_critic_once":
            record.coverage_audit = {
                "skipped": True,
                "pipeline_variant": self.pipeline_variant,
                "summary": "Integrator and final Critic are disabled for this experiment.",
                "additions": [],
                "parse_error": "",
            }
            record.final_triples = self._dedupe_document(record.refined_triples)
            self.kg.extend(record.final_triples)
            self.pipeline_records.append(record)
            self._append_document(record)
            return record

        self.run_integrator_stage(record)
        self.run_final_critic_stage(record)

        self.kg.extend(record.final_triples)
        self.pipeline_records.append(record)
        self._append_document(record)
        return record
