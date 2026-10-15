import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

from evidenceloop.data.documents import DocumentInputError, load_documents, normalize_document


class DocumentLoaderTests(unittest.TestCase):
    def test_load_graphjudge_source_uses_stable_line_ids(self):
        tmpdir = Path.cwd() / ".test_tmp" / f"source_loader_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True)
        try:
            path = tmpdir / "test.source"
            path.write_text("First document.\nSecond document.\n", encoding="utf-8")
            documents = load_documents(str(path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(
            [(document.document_id, document.text) for document in documents],
            [
                ("graphjudge_rebel_0000", "First document."),
                ("graphjudge_rebel_0001", "Second document."),
            ],
        )

    def test_load_json_normalizes_existing_article_fields_and_ignores_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "documents.json"
            path.write_text(json.dumps([{
                "article_id": "article-1",
                "title": "Ignored title",
                "article": "Ada was born in London.",
            }]), encoding="utf-8")
            documents = load_documents(str(path))

        self.assertEqual(documents[0].document_id, "article-1")
        self.assertEqual(documents[0].text, "Ada was born in London.")
        self.assertFalse(hasattr(documents[0], "title"))

    def test_load_jsonl_reports_physical_line_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "documents.jsonl"
            path.write_text(
                '{"document_id":"doc-1","text":"valid"}\n'
                '{"document_id":"doc-2","text":""}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DocumentInputError, r":2: missing non-empty document text"):
                load_documents(str(path))

    def test_normalize_document_rejects_missing_id(self):
        with self.assertRaisesRegex(DocumentInputError, "record 3"):
            normalize_document({"text": "content"}, "records.json:record 3")

    def test_normalize_document_accepts_numeric_source_id(self):
        document = normalize_document({"source_id": 7, "article": "content"}, "records.json:record 1")
        self.assertEqual(document.document_id, "7")


if __name__ == "__main__":
    unittest.main()
