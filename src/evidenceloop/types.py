from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    source_text: str = ""
    document_id: str = ""

    @staticmethod
    def from_dict(
        data: Optional[dict],
        *,
        source_text: str = "",
        document_id: str = "",
    ):
        data = data or {}
        return Triple(
            subject=str(data.get("subject", "") or ""),
            predicate=str(data.get("predicate", "") or ""),
            object=str(data.get("object", "") or ""),
            source_text=source_text,
            document_id=document_id,
        )

    def to_spo_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }

    def to_dict(self) -> Dict:
        return {
            **self.to_spo_dict(),
            "source_text": self.source_text,
            "document_id": self.document_id,
        }


@dataclass
class TripleReviewRecord:
    input_triple: Triple
    stage: str
    origin: str
    action: str
    output_triple: Optional[Triple] = None
    reason: str = ""
    evidence: str = ""
    evidence_start: int = -1
    evidence_end: int = -1
    valid: bool = True
    validation_error: str = ""
    raw_result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "origin": self.origin,
            "input_triple": self.input_triple.to_dict(),
            "action": self.action,
            "output_triple": self.output_triple.to_dict() if self.output_triple else None,
            "reason": self.reason,
            "evidence": self.evidence,
            "evidence_start": self.evidence_start,
            "evidence_end": self.evidence_end,
            "valid": self.valid,
            "validation_error": self.validation_error,
            "raw_result": self.raw_result,
        }


@dataclass
class DocumentPipelineRecord:
    document_id: str
    source_text: str
    extracted_triples: List[Triple] = field(default_factory=list)
    initial_reviews: List[TripleReviewRecord] = field(default_factory=list)
    refined_triples: List[Triple] = field(default_factory=list)
    coverage_audit: Dict = field(default_factory=dict)
    added_triples: List[Triple] = field(default_factory=list)
    final_reviews: List[TripleReviewRecord] = field(default_factory=list)
    final_triples: List[Triple] = field(default_factory=list)
    stage_errors: List[str] = field(default_factory=list)
    stage_timings_seconds: Dict[str, float] = field(default_factory=dict)
    raw_responses: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "source_text": self.source_text,
            "extracted_triples": [triple.to_dict() for triple in self.extracted_triples],
            "initial_reviews": [review.to_dict() for review in self.initial_reviews],
            "refined_triples": [triple.to_dict() for triple in self.refined_triples],
            "coverage_audit": self.coverage_audit,
            "added_triples": [triple.to_dict() for triple in self.added_triples],
            "final_reviews": [review.to_dict() for review in self.final_reviews],
            "final_triples": [triple.to_dict() for triple in self.final_triples],
            "stage_errors": list(self.stage_errors),
            "stage_timings_seconds": dict(self.stage_timings_seconds),
            "raw_responses": dict(self.raw_responses),
        }
