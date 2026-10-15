import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DOCUMENT_ID_FIELDS = ("document_id", "article_id", "sample_id", "source_id", "id", "_id")
TEXT_FIELDS = ("text", "article", "source_text", "context")


class DocumentInputError(ValueError):
    """Raised when an input record cannot be normalized into a document."""


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


def _first_nonempty_string(record: Dict, fields) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_nonempty_identifier(record: Dict, fields) -> str:
    for field in fields:
        value = record.get(field)
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return ""


def normalize_document(record: Dict, location: str) -> Document:
    if not isinstance(record, dict):
        raise DocumentInputError(f"{location}: document record must be a JSON object")

    document_id = _first_nonempty_identifier(record, DOCUMENT_ID_FIELDS)
    if not document_id:
        fields = ", ".join(DOCUMENT_ID_FIELDS)
        raise DocumentInputError(f"{location}: missing non-empty document id ({fields})")

    text = _first_nonempty_string(record, TEXT_FIELDS)
    if not text:
        fields = ", ".join(TEXT_FIELDS)
        raise DocumentInputError(f"{location}: missing non-empty document text ({fields})")

    return Document(document_id=document_id, text=text)


def load_documents(input_path: str) -> List[Document]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Document input file not found: {path}")

    if path.suffix.lower() == ".source":
        documents = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                text = raw_line.strip()
                if not text:
                    raise DocumentInputError(f"{path}:{line_number}: empty source document")
                documents.append(
                    Document(
                        document_id=f"graphjudge_rebel_{line_number - 1:04d}",
                        text=text,
                    )
                )
        return documents

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        documents = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DocumentInputError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                documents.append(normalize_document(record, f"{path}:{line_number}"))
        return documents

    try:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except json.JSONDecodeError as exc:
        raise DocumentInputError(f"{path}: invalid JSON: {exc.msg}") from exc

    if not isinstance(rows, list):
        raise DocumentInputError(f"{path}: JSON input must be an array of document records")

    return [
        normalize_document(record, f"{path}:record {index}")
        for index, record in enumerate(rows, start=1)
    ]
