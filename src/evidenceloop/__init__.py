"""EvidenceLoop: evidence-grounded document knowledge graph extraction."""

from evidenceloop.data.documents import Document
from evidenceloop.pipeline.core import Pipeline
from evidenceloop.settings import PipelineConfig
from evidenceloop.types import DocumentPipelineRecord, Triple

__all__ = ["Pipeline", "PipelineConfig", "Document", "Triple", "DocumentPipelineRecord"]
__version__ = "0.1.0"
