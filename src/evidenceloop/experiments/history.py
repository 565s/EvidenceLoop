"""Read completed historical and current runs without modifying their files."""

from pathlib import Path

from evidenceloop import Document
from evidenceloop.io import read_json_object
from evidenceloop.pipeline.run import digest, document_manifest


def read_completed_run(run_dir):
    path = Path(run_dir).resolve()
    state = read_json_object(path / "run_state.json")
    if state.get("schema_version") != 2 and state.get("version") != 1:
        raise ValueError("Unsupported source run version")
    related_schema = state.get("kind") == "related_baseline"
    if related_schema:
        from evidenceloop.baselines.related.common import validate_run

        validate_run(path)
    progress = read_json_object(path / "progress.json")
    if not progress.get("finalized") or progress.get("failed", 0) or progress.get("pending", 0):
        raise ValueError("Offline continuation requires a complete finalized source run")
    records = {}
    documents = []
    checkpoints = sorted((path / "checkpoints").glob("*.json"))
    count = int(progress.get("documents_total", progress.get("completed", 0)))
    if not count or len(checkpoints) != count:
        raise ValueError("Source checkpoint count mismatch")
    for i, checkpoint_path in enumerate(checkpoints):
        value = read_json_object(checkpoint_path)
        if (
            checkpoint_path.stem != f"{i:06d}"
            or (value.get("document", {}).get("index") if related_schema else value.get("index"))
            != i
            or value.get("status") != "ok"
        ):
            raise ValueError("Source checkpoint order/status mismatch")
        record = value.get("pipeline_record") or {}
        if value.get("final_triples") != record.get("final_triples"):
            raise ValueError("Source checkpoint final triples mismatch")
        document_id = record.get("document_id")
        if not document_id or not record.get("source_text") or document_id in records:
            raise ValueError("Source document identity/content invalid")
        if state.get("schema_version") == 2:
            manifest = document_manifest([Document(document_id, record["source_text"])])[0]
            manifest["index"] = i
            if value.get("schema_version") != 2 or value.get("manifest") != manifest:
                raise ValueError("Source checkpoint manifest mismatch")
            if value.get("record_sha256") != digest(record):
                raise ValueError("Source record checksum mismatch")
        elif (
            value.get("version") != 1
            or (
                value.get("document", {}).get("document_id")
                if related_schema
                else value.get("document_id")
            )
            != document_id
        ):
            raise ValueError("Unsupported historical checkpoint")
        for field in ("extracted_triples", "refined_triples", "added_triples", "final_triples"):
            triples = record.get(field)
            if not isinstance(triples, list):
                raise ValueError(f"Source record requires {field}")
            for triple in triples:
                if (
                    not isinstance(triple, dict)
                    or any(
                        not isinstance(triple.get(key), str) or not triple[key].strip()
                        for key in ("subject", "predicate", "object")
                    )
                    or triple.get("document_id", document_id) != document_id
                    or triple.get("source_text", record["source_text"]) != record["source_text"]
                ):
                    raise ValueError("Source triple identity/content mismatch")
        records[document_id] = record
        documents.append(Document(document_id, record["source_text"]))
    if state.get("schema_version") == 2:
        if state.get("documents") != document_manifest(documents):
            raise ValueError("Source manifest order/content mismatch")
    elif not related_schema:
        expected = (state.get("data") or {}).get("document_ids_sha256") or (
            state.get("source") or {}
        ).get("document_ids_sha256")
        actual = (
            __import__("hashlib")
            .sha256("".join(d.document_id + "\n" for d in documents).encode("utf-8"))
            .hexdigest()
        )
        if not expected or actual != expected:
            raise ValueError("Historical source document order fingerprint mismatch")
    return state, documents, records
