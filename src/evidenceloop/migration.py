"""Copy compatible historical direct-Critic checkpoints into a versioned run."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from evidenceloop import Pipeline, PipelineConfig
from evidenceloop.data.documents import load_documents
from evidenceloop.data.registry import canonical_dataset
from evidenceloop.io import atomic_write_json, atomic_write_jsonl, read_json_object
from evidenceloop.pipeline.run import RunManager, digest, document_manifest
from evidenceloop.runtime.settings import resolve_critic_runtime


def remap_path(value, mappings):
    normalized = str(value).replace("\\", "/").rstrip("/")
    if not normalized or ".." in normalized.split("/"):
        raise ValueError("Historical paths must be non-empty and cannot contain parent traversal")
    for old, new in sorted(mappings, key=lambda item: -len(item[0])):
        prefix = old.replace("\\", "/").rstrip("/")
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return str(Path(new).joinpath(*normalized[len(prefix) :].lstrip("/").split("/")))
    if normalized.startswith("/") or (len(normalized) > 2 and normalized[1] == ":"):
        raise ValueError(f"Historical absolute path requires --path-map: {value}")
    return normalized


def migrate_run(source, destination, input_path, *, dataset, path_mappings=(), config=None):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directories")
    if destination.exists():
        raise FileExistsError(destination)
    state = read_json_object(source / "run_state.json")
    if state.get("version") != 1:
        raise ValueError(
            "Only version-1 direct-Critic checkpoints can be migrated; completed results remain readable"
        )
    runtime = state.get("runtime") or {}
    old_critic = runtime.get("critic_runtime") or {}
    if old_critic.get("backend") != "direct":
        raise ValueError(
            "Only direct-Critic checkpoints are resumable; completed results remain readable"
        )
    stage = runtime.get("stage") or ("critic" if runtime.get("offline_proposals") else "pipeline")
    if stage not in {"pipeline", "critic", "final_critic"}:
        raise ValueError("Only pipeline and direct-Critic stage checkpoints are compatible")
    documents = load_documents(str(input_path))
    identities = document_manifest(documents)
    if len({d.document_id for d in documents}) != len(documents):
        raise ValueError("Duplicate document ids")
    data = state.get("data") or state.get("source") or {}
    count = int(data.get("selected_documents", data.get("documents", 0)))
    expected_hash = hashlib.sha256(
        "".join(d.document_id + "\n" for d in documents).encode("utf-8")
    ).hexdigest()
    if count != len(documents) or data.get("document_ids_sha256") != expected_hash:
        raise ValueError("Historical document count/order fingerprint mismatch")
    source_records = None
    source_run_dir = None
    if stage != "pipeline":
        from evidenceloop.experiments.history import read_completed_run

        source_run_dir = remap_path(data.get("run_dir", ""), path_mappings)
        _, source_documents, source_records = read_completed_run(source_run_dir)
        if document_manifest(source_documents) != identities:
            raise ValueError("Offline source document order/content mismatch")
        field = "source_payloads_sha256" if stage == "critic" else "source_records_sha256"
        hasher = hashlib.sha256()
        for i, document in enumerate(documents):
            checkpoint = read_json_object(Path(source_run_dir) / "checkpoints" / f"{i:06d}.json")
            if stage == "critic":
                payload = {
                    "index": i,
                    "document_id": document.document_id,
                    "source_text": document.text,
                    "payload": checkpoint.get("payload"),
                }
            else:
                payload = {
                    "index": i,
                    "document_id": document.document_id,
                    "pipeline_record": source_records[document.document_id],
                }
            hasher.update((digest(payload) + "\n").encode("ascii"))
        if data.get(field) != hasher.hexdigest():
            raise ValueError("Offline source content fingerprint mismatch")
    else:
        # Legacy manifests only hashed IDs. Check the historical input too so
        # pending/failed documents cannot silently resume with different text.
        historical_input = Path(remap_path(data.get("data_dir", ""), path_mappings))
        if historical_input.is_dir():
            historical_input = next(
                (
                    historical_input / name
                    for name in ("documents.jsonl", "test.source")
                    if (historical_input / name).is_file()
                ),
                historical_input / "documents.jsonl",
            )
        historical_documents = load_documents(str(historical_input))[: len(documents)]
        if document_manifest(historical_documents) != identities:
            raise ValueError("Historical input document order/content mismatch")
    variant = runtime.get(
        "pipeline_variant", "proposer_critic_once" if stage == "critic" else "full"
    )
    strategy = runtime.get("critic_candidate_set_size", "max4")
    settings = dict(old_critic)
    for key in ("base_model_path", "adapter_path"):
        settings[key] = remap_path(settings.get(key, ""), path_mappings)
    saved_dataset = (
        runtime.get("dataset_key") or runtime.get("dataset") or data.get("dataset") or dataset
    )
    if canonical_dataset(saved_dataset) != canonical_dataset(dataset):
        raise ValueError("Historical dataset mismatch")
    inferred = PipelineConfig(
        pipeline_variant=variant,
        proposer_model_name=runtime.get("proposer_model") or PipelineConfig().proposer_model_name,
        integrator_model_name=runtime.get("integrator_model")
        or PipelineConfig().integrator_model_name,
        proposer_temperature=float(runtime.get("proposer_temperature", 0.5)),
        critic_model_name=settings["model_name"],
        critic_base_model_path=settings["base_model_path"],
        critic_adapter_path=settings["adapter_path"],
        critic_device=settings["device"],
        critic_batch_size=int(settings["batch_size"]),
        critic_prompt_batch_size=int(settings.get("prompt_batch_size", 4)),
        critic_max_tokens=int(settings["max_tokens"]),
        critic_max_seq_length=int(settings["max_seq_length"]),
    )
    config = config or inferred
    if config.to_dict() != inferred.to_dict():
        raise ValueError("Configuration does not match historical run parameters")
    resolved = resolve_critic_runtime(config.to_dict(), environ={})
    for key in (
        "dtype",
        "attention_implementation",
        "merge_adapter",
        "thinking_enabled",
        "backend",
    ):
        if settings.get(key) != resolved[key]:
            raise ValueError(f"Incompatible historical Critic runtime: {key}")
    external_cache = (
        Path(remap_path(runtime["proposal_cache_dir"], path_mappings))
        if runtime.get("proposal_cache_dir")
        else None
    )
    manager = RunManager(
        Pipeline(config=config, critic_runtime=resolved),
        dataset=dataset,
        proposer_workers=int(runtime.get("proposer_workers", 1)),
        candidate_set=strategy,
        seed=int(runtime.get("critic_candidate_set_seed", 42)),
        stage=stage,
        source_records=source_records,
        source_run_dir=source_run_dir,
        proposal_cache_dir=destination / "proposals" if external_cache else None,
        require_proposal_cache=bool(runtime.get("require_proposal_cache", False)),
    )
    new_runtime = manager.runtime_signature()
    new_runtime["proposer_endpoint"] = (
        runtime.get("proposer_endpoint") if stage == "pipeline" else None
    )
    if stage == "pipeline" and variant == "full":
        new_runtime["integrator_endpoint"] = runtime.get("integrator_endpoint") or runtime.get(
            "proposer_endpoint"
        )
    migrated = {}
    checkpoints = source / "checkpoints"
    for path in checkpoints.glob("*.json"):
        if not path.stem.isdigit() or not 0 <= int(path.stem) < count:
            raise ValueError(f"Unexpected historical checkpoint: {path.name}")
        index = int(path.stem)
        if index in migrated:
            raise ValueError("Duplicate historical checkpoint index")
        document = documents[index]
        value = read_json_object(path)
        if (
            value.get("version") != 1
            or value.get("index") != index
            or value.get("document_id") != document.document_id
        ):
            raise ValueError("Historical checkpoint order mismatch")
        status = value.get("status")
        if status not in {"ok", "failed"} or int(value.get("attempts", 0)) < 1:
            raise ValueError("Invalid historical checkpoint status/attempts")
        result = value.get("sample_result") or value.get("processing_result") or {}
        if result.get("document_id") != document.document_id or result.get("status") != status:
            raise ValueError("Historical processing result mismatch")
        record = value.get("pipeline_record")
        if status == "ok":
            if (
                not isinstance(record, dict)
                or record.get("source_text") != document.text
                or record.get("document_id") != document.document_id
            ):
                raise ValueError("Historical checkpoint document content mismatch")
            if record.get("final_triples") != value.get("final_triples"):
                raise ValueError("Historical checkpoint final triples mismatch")
        # Failed checkpoints may have no source text. The supplied canonical input is
        # authoritative; every available record is checked before migration.
        if record and record.get("source_text") != document.text:
            raise ValueError("Historical failed checkpoint content mismatch")
        migrated[index] = {
            "schema_version": 2,
            "index": index,
            "manifest": identities[index],
            "attempts": value["attempts"],
            "status": status,
            "error": value.get("error", ""),
            "pipeline_record": record if status == "ok" else None,
            "final_triples": value["final_triples"] if status == "ok" else [],
            "record_sha256": digest(record) if status == "ok" else None,
        }
    proposal_checkpoints = {}
    if stage == "pipeline":
        from evidenceloop.experiments.checkpoints import _validate_proposal_checkpoint

        rows = [{"document_id": d.document_id, "text": d.text} for d in documents]
        proposal_dir = external_cache or source / "proposal_checkpoints"
        if external_cache:
            from evidenceloop.experiments.checkpoints import validate_proposal_cache_state

            if not (external_cache / "cache_state.json").is_file():
                raise ValueError("External historical proposal cache requires cache_state.json")
            validate_proposal_cache_state(external_cache, rows)
        elif not proposal_dir.exists():
            proposal_dir = source / "proposals"
        for path in proposal_dir.glob("*.json"):
            if path.name == "cache_state.json":
                continue
            if not path.stem.isdigit() or not 0 <= int(path.stem) < count:
                raise ValueError("Unexpected historical proposal checkpoint")
            i = int(path.stem)
            if i in proposal_checkpoints:
                raise ValueError("Duplicate historical proposal checkpoint index")
            value = _validate_proposal_checkpoint(read_json_object(path), i, rows[i], path)
            for triple in (value.get("payload") or {}).get("triples", []):
                if (
                    triple.get("source_text") != documents[i].text
                    or triple.get("document_id") != documents[i].document_id
                ):
                    raise ValueError("Historical proposal content mismatch")
            value.update(manifest=identities[i], payload_sha256=digest(value.get("payload")))
            proposal_checkpoints[i] = value
    # Mutations start only after all source validation succeeds.
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        destination / "run_state.json",
        {
            "schema_version": 2,
            "kind": stage,
            "documents": identities,
            "runtime": new_runtime,
            "migration": {
                "source_run_dir": str(source),
                "source_state_sha256": digest(state),
                "path_mappings": [list(item) for item in path_mappings],
            },
        },
    )
    atomic_write_jsonl(destination / "documents.jsonl", [d.__dict__ for d in documents])
    if external_cache:
        atomic_write_json(
            destination / "proposals" / "proposal_state.json",
            {
                "schema_version": 2,
                "documents": identities,
                "parser": "agent",
                "model": config.proposer_model_name,
                "temperature": config.proposer_temperature,
                "endpoint": new_runtime["proposer_endpoint"],
            },
        )
    for index, value in proposal_checkpoints.items():
        atomic_write_json(destination / "proposals" / f"{index:06d}.json", value)
    for index, value in migrated.items():
        atomic_write_json(destination / "checkpoints" / f"{index:06d}.json", value)
    manager._export(destination, migrated)
    manager._progress(destination, documents, migrated)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evidenceloop migrate-run")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    mappings = []
    for value in args.path_map:
        if "=" not in value:
            raise ValueError("--path-map must be OLD=NEW")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError("--path-map requires both paths")
        mappings.append((old, new))
    path = migrate_run(
        args.source_run_dir,
        args.output_run_dir,
        args.input,
        dataset=args.dataset,
        path_mappings=mappings,
        config=PipelineConfig.from_file(args.config) if args.config else None,
    )
    print(path)
    return 0
