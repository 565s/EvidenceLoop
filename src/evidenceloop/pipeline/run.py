"""Versioned batch runs, bounded proposal prefetch, and atomic recovery."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from evidenceloop import Document, Pipeline
from evidenceloop.data.registry import candidate_strategy, canonical_dataset
from evidenceloop.io import atomic_write_json, atomic_write_jsonl, read_json_object
from evidenceloop.serialization import pipeline_record_from_dict

SCHEMA_VERSION = 2
logger = logging.getLogger("evidenceloop")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def document_manifest(documents):
    return [
        {
            "index": i,
            "document_id": d.document_id,
            "text_sha256": hashlib.sha256(d.text.encode("utf-8")).hexdigest(),
        }
        for i, d in enumerate(documents)
    ]


class RunManager:
    def __init__(
        self,
        pipeline: Pipeline,
        *,
        dataset="rebel_sub",
        proposer_workers=1,
        candidate_set=None,
        seed=42,
        proposal_cache_dir=None,
        require_proposal_cache=False,
        stage="pipeline",
        source_records=None,
        source_run_dir=None,
    ):
        self.pipeline = pipeline
        self.stage = stage
        self.source_records = source_records or {}
        self.source_run_dir = str(Path(source_run_dir).resolve()) if source_run_dir else None
        if stage not in {"pipeline", "critic", "integrator", "final_critic", "direct"}:
            raise ValueError("Invalid run stage")
        self.dataset = canonical_dataset(dataset)
        self.workers = int(proposer_workers)
        self.candidate_set = candidate_strategy(candidate_set or "fixed4")
        self.candidate_grouping = "max4" if candidate_set in (None, "max4") else self.candidate_set
        self.seed = seed
        self.cache_dir = Path(proposal_cache_dir) if proposal_cache_dir else None
        self.require_cache = require_proposal_cache
        if not 1 <= self.workers <= 16:
            raise ValueError("proposer_workers must be between 1 and 16")
        if stage == "pipeline" and pipeline.pipeline_variant == "full" and self.workers != 1:
            raise ValueError("full pipeline requires proposer_workers=1")
        if pipeline.pipeline_variant == "full" and self.candidate_set != "fixed4":
            raise ValueError("candidate-set ablations require proposer_critic_once")
        if self.require_cache and self.cache_dir is None:
            raise ValueError("require_proposal_cache requires proposal_cache_dir")

    def _critic_metrics(self):
        from evidenceloop.runtime.factory import LazyRuntime

        critic = self.pipeline.critic
        if isinstance(critic, LazyRuntime):
            critic = critic._instance
        engine = vars(critic).get("llm") if critic is not None else None
        return dict(engine.metrics()) if engine is not None and hasattr(engine, "metrics") else {}

    def runtime_signature(self):

        config = self.pipeline.config.to_dict()
        config["pipeline_variant"] = self.pipeline.pipeline_variant
        return {
            "stage": self.stage,
            "source_run_dir": self.source_run_dir,
            "source_records_sha256": digest(self.source_records) if self.source_records else None,
            "config": config,
            "critic_runtime": self.pipeline.critic_runtime,
            "dataset": self.dataset,
            "proposer_workers": self.workers,
            "candidate_set": self.candidate_set,
            "candidate_grouping": self.candidate_grouping,
            "seed": self.seed,
            "proposer_endpoint": (
                os.environ.get(config["base_url_env"])
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("SIFLOW_BASE_URL")
                or None
            )
            if self.stage in {"pipeline", "direct"}
            else None,
            "integrator_endpoint": (
                os.environ.get(config["base_url_env"])
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("SIFLOW_BASE_URL")
                or None
            )
            if self.stage == "integrator"
            or (self.stage == "pipeline" and self.pipeline.pipeline_variant == "full")
            else None,
            "proposal_cache_dir": str(self.cache_dir.resolve()) if self.cache_dir else None,
            "require_proposal_cache": self.require_cache,
        }

    def _validate_checkpoint(self, value, index, document, manifest):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint version; use migrate-run")
        if value.get("manifest") != manifest or value.get("index") != index:
            raise ValueError("Checkpoint document order/content mismatch")
        if value.get("status") not in {"ok", "failed"} or int(value.get("attempts", 0)) < 1:
            raise ValueError("Invalid checkpoint status or attempts")
        if value["status"] == "ok":
            record = value.get("pipeline_record") or {}
            if (
                record.get("document_id") != document.document_id
                or record.get("source_text") != document.text
            ):
                raise ValueError("Checkpoint record content mismatch")
            if record.get("final_triples") != value.get("final_triples"):
                raise ValueError("Checkpoint final triples mismatch")
            if digest(record) != value.get("record_sha256"):
                raise ValueError("Checkpoint record checksum mismatch")
        return value

    def _prepare_cache(self, documents):
        if self.cache_dir is None or self.stage not in {"pipeline", "direct"}:
            return
        state_path = self.cache_dir / "proposal_state.json"
        expected = {
            "schema_version": SCHEMA_VERSION,
            "documents": document_manifest(documents),
            "parser": "direct_baseline" if self.stage == "direct" else "agent",
            "model": self.pipeline.config.proposer_model_name,
            "temperature": self.pipeline.config.proposer_temperature,
            "endpoint": self.runtime_signature()["proposer_endpoint"],
        }
        if state_path.exists():
            if read_json_object(state_path) != expected:
                raise ValueError("Proposal cache document/runtime mismatch")
        elif self.require_cache:
            raise ValueError("Proposal cache manifest is missing")
        else:
            if self.cache_dir.exists() and any(self.cache_dir.iterdir()):
                raise FileExistsError("Refusing a cache directory without a compatible manifest")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(state_path, expected)

    def _propose(self, index, document):
        if self.stage == "integrator":
            record = pipeline_record_from_dict(self.source_records[document.document_id])
            elapsed = 0.0
            error = ""
            for attempt in range(1, 4):
                started = time.perf_counter()
                try:
                    payload = self.pipeline.integrator.audit_coverage(
                        record.source_text, record.refined_triples
                    )
                    if not isinstance(payload, dict):
                        error = "Integrator returned a non-object payload"
                    else:
                        error = str(payload.get("parse_error") or "")
                        if not error:
                            return payload, elapsed + time.perf_counter() - started
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed += time.perf_counter() - started
                if attempt < 3:
                    time.sleep(float(attempt))
            raise RuntimeError(error or "Integrator request failed")
        if self.stage in {"critic", "final_critic"}:
            from evidenceloop.serialization import _triple_from_dict

            record = self.source_records[document.document_id]
            return {
                "triples": [_triple_from_dict(t) for t in record["extracted_triples"]],
                "raw_response": record.get("raw_responses", {}).get(
                    "proposer", record.get("raw_responses", {}).get("gpt_extraction", "")
                ),
                "parse_error": "",
            }, record.get("stage_timings_seconds", {}).get(
                "proposer", record.get("stage_timings_seconds", {}).get("gpt_extraction", 0)
            )
        from evidenceloop.experiments.checkpoints import (
            _deserialize_proposer_payload,
            build_proposal_checkpoint,
        )

        manifest = document_manifest([document])[0]
        manifest["index"] = index
        path = (
            self.cache_dir if self.cache_dir else self._run_dir / "proposals"
        ) / f"{index:06d}.json"
        if path and path.exists():
            cached = read_json_object(path)
            if cached.get("manifest") != manifest:
                raise ValueError("Proposal checkpoint document mismatch")
            if cached.get("status") == "ok":
                if digest(cached.get("payload")) != cached.get("payload_sha256"):
                    raise ValueError("Proposal cache payload checksum mismatch")
                return _deserialize_proposer_payload(
                    cached["payload"], {"document_id": document.document_id, "text": document.text}
                ), float(cached.get("api_elapsed_seconds", 0))
        if self.require_cache:
            raise ValueError(f"Required proposal cache missing/failed for {document.document_id}")
        # Reuse the historical three-attempt proposer retry and response serialization.
        previous = read_json_object(path) if path and path.exists() else None
        target = path
        checkpoint = build_proposal_checkpoint(
            self.pipeline.proposer,
            {"document_id": document.document_id, "text": document.text},
            index,
            target,
            previous_attempts=int((previous or {}).get("attempts", 0)),
        )
        checkpoint["manifest"] = manifest
        checkpoint["payload_sha256"] = digest(checkpoint.get("payload"))
        atomic_write_json(target, checkpoint)
        if checkpoint["status"] != "ok":
            raise RuntimeError(checkpoint["error"])
        return _deserialize_proposer_payload(
            checkpoint["payload"], {"document_id": document.document_id, "text": document.text}
        ), checkpoint["api_elapsed_seconds"]

    def _progress(self, run_dir, documents, checkpoints):
        ok = sum(v["status"] == "ok" for v in checkpoints.values())
        failed = sum(v["status"] == "failed" for v in checkpoints.values())
        value = {
            "schema_version": SCHEMA_VERSION,
            "documents_total": len(documents),
            "completed": ok,
            "failed": failed,
            "pending": len(documents) - ok - failed,
            "finalized": len(checkpoints) == len(documents),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(run_dir / "progress.json", value)
        return value

    def _export(self, run_dir, checkpoints):
        records = [
            v["pipeline_record"] for _, v in sorted(checkpoints.items()) if v["status"] == "ok"
        ]
        triples = [triple for record in records for triple in record["final_triples"]]
        for name, values in (("pipeline_records", records), ("kg", triples)):
            atomic_write_json(run_dir / f"{name}.json", values)
            atomic_write_jsonl(run_dir / f"{name}.jsonl", values)
        processing = [
            {
                "document_id": v["manifest"]["document_id"],
                "status": v["status"],
                "error": v.get("error", ""),
                "attempts": v["attempts"],
            }
            for _, v in sorted(checkpoints.items())
        ]
        atomic_write_jsonl(run_dir / "processing_results.jsonl", processing)
        atomic_write_jsonl(
            run_dir / "sample_results.jsonl",
            [
                v.get("sample_result", r)
                for (_, v), r in zip(sorted(checkpoints.items()), processing)
            ],
        )

    def run(self, documents, output_dir=None, *, resume_run_dir=None):
        documents = list(documents)
        if not documents or not all(
            isinstance(d, Document) and d.document_id.strip() and d.text.strip() for d in documents
        ):
            raise ValueError("A run requires non-empty Documents")
        if len({d.document_id for d in documents}) != len(documents):
            raise ValueError("Duplicate document IDs")
        if output_dir is not None and resume_run_dir is not None:
            raise ValueError("Choose output_dir or resume_run_dir")
        manifest = document_manifest(documents)
        runtime = self.runtime_signature()
        run_dir = Path(
            resume_run_dir
            or output_dir
            or (Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        ).resolve()
        state = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.stage,
            "documents": manifest,
            "runtime": runtime,
        }
        if resume_run_dir:
            saved = read_json_object(run_dir / "run_state.json")
            if saved.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    "Historical checkpoints require migrate-run; completed results remain readable"
                )
            for key in ("kind", "documents", "runtime"):
                if saved.get(key) != state[key]:
                    raise ValueError(f"Resume {key} mismatch")
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            atomic_write_json(run_dir / "run_state.json", state)
            atomic_write_jsonl(run_dir / "documents.jsonl", [d.__dict__ for d in documents])
        metrics_before = self._critic_metrics()
        started_run = time.perf_counter()
        self._run_dir = run_dir
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        lock = run_dir / ".run.lock"
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        validated = False
        previous_summary = (
            read_json_object(run_dir / "summary.json")
            if (run_dir / "summary.json").is_file()
            else {}
        )
        checkpoints = {}
        try:
            for path in checkpoint_dir.glob("*.json"):
                if not path.stem.isdigit() or not 0 <= int(path.stem) < len(documents):
                    raise ValueError(f"Unexpected checkpoint {path.name}")
                i = int(path.stem)
                if i in checkpoints or path.stem != f"{i:06d}":
                    raise ValueError("Duplicate or non-canonical checkpoint index")
                checkpoints[i] = self._validate_checkpoint(
                    read_json_object(path), i, documents[i], manifest[i]
                )
            self._prepare_cache(documents)
            self._progress(run_dir, documents, checkpoints)
            validated = True
            pending = [
                i for i in range(len(documents)) if checkpoints.get(i, {}).get("status") != "ok"
            ]
            # Ordered bounded prefetch: at most twice the worker count is in flight.
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {}
                cursor = 0

                def fill():
                    nonlocal cursor
                    while cursor < len(pending) and len(futures) < 2 * self.workers:
                        i = pending[cursor]
                        cursor += 1
                        futures[i] = executor.submit(self._propose, i, documents[i])

                fill()
                for start in range(
                    0,
                    len(pending),
                    8
                    if self.pipeline.pipeline_variant == "proposer_critic_once"
                    or self.stage == "final_critic"
                    else 1,
                ):
                    indices = pending[
                        start : start
                        + (
                            8
                            if self.pipeline.pipeline_variant == "proposer_critic_once"
                            or self.stage == "final_critic"
                            else 1
                        )
                    ]
                    proposals = {}
                    errors = {}
                    for i in indices:
                        try:
                            proposals[i] = futures.pop(i).result()
                        except Exception as exc:
                            errors[i] = str(exc)
                        fill()
                    reviews = {}
                    if (
                        (
                            self.pipeline.pipeline_variant == "proposer_critic_once"
                            and self.stage in {"pipeline", "critic"}
                        )
                        or self.stage == "final_critic"
                    ) and proposals:
                        from evidenceloop.experiments.checkpoints import critic_chunk_sizes

                        requests = []
                        request_indices = []
                        for i, payload in proposals.items():
                            if self.stage == "final_critic":
                                source = pipeline_record_from_dict(
                                    self.source_records[documents[i].document_id]
                                )
                                triples = source.refined_triples + source.added_triples
                                origins = ["proposer"] * len(source.refined_triples) + [
                                    "integrator"
                                ] * len(source.added_triples)
                                request = {
                                    "triples": triples,
                                    "stage": "final_validation",
                                    "origins": origins,
                                }
                            else:
                                triples = payload[0]["triples"]
                                request = {
                                    "triples": triples,
                                    "stage": "initial_review",
                                    "origins": ["proposer"] * len(triples),
                                    "chunk_sizes": critic_chunk_sizes(
                                        len(triples),
                                        self.candidate_grouping,
                                        seed=self.seed,
                                        document_id=documents[i].document_id,
                                    ),
                                }
                            if triples:
                                requests.append(request)
                                request_indices.append(i)
                            else:
                                reviews[i] = {
                                    "payload": {
                                        "results": [],
                                        "raw_response": "",
                                        "parse_error": "",
                                    },
                                    "elapsed_seconds": 0.0,
                                }
                        if requests:
                            try:
                                results = self.pipeline.critic.review_documents_batch(requests)
                                if len(results) != len(requests):
                                    raise RuntimeError("Critic document response count mismatch")
                                reviews.update(zip(request_indices, results))
                            except Exception as exc:
                                errors.update({i: str(exc) for i in request_indices})
                    for i in indices:
                        previous = checkpoints.get(i, {})
                        value = {
                            "schema_version": SCHEMA_VERSION,
                            "index": i,
                            "manifest": manifest[i],
                            "attempts": int(previous.get("attempts", 0)) + 1,
                            "status": "failed",
                            "error": errors.get(i, ""),
                            "pipeline_record": None,
                            "final_triples": [],
                        }
                        try:
                            if i in errors:
                                raise RuntimeError(errors[i])
                            kwargs = {
                                "proposer_payload": proposals[i][0],
                                "proposer_elapsed_seconds": proposals[i][1],
                            }
                            if i in reviews:
                                kwargs.update(
                                    initial_critic_payload=reviews[i]["payload"],
                                    initial_critic_elapsed_seconds=reviews[i].get(
                                        "elapsed_seconds", 0
                                    ),
                                )
                            if self.stage in {"integrator", "final_critic"}:
                                import copy

                                record = pipeline_record_from_dict(
                                    copy.deepcopy(self.source_records[documents[i].document_id])
                                )
                                if self.stage == "integrator":
                                    record = self.pipeline.run_integrator_stage(
                                        record,
                                        coverage_payload=proposals[i][0],
                                        integrator_elapsed_seconds=proposals[i][1],
                                    )
                                else:
                                    record = self.pipeline.run_final_critic_stage(
                                        record,
                                        final_critic_payload=reviews[i]["payload"],
                                        final_critic_elapsed_seconds=reviews[i].get(
                                            "elapsed_seconds", 0
                                        ),
                                    )
                            elif self.stage == "direct":
                                from evidenceloop.types import DocumentPipelineRecord

                                payload, elapsed = proposals[i]
                                triples = list(payload["triples"])
                                record = DocumentPipelineRecord(
                                    documents[i].document_id,
                                    documents[i].text,
                                    extracted_triples=triples,
                                    final_triples=triples,
                                    stage_timings_seconds={"proposer": elapsed},
                                    raw_responses={"proposer": payload.get("raw_response", "")},
                                )
                            else:
                                record = self.pipeline.process_document(
                                    documents[i].document_id, documents[i].text, **kwargs
                                )
                            value.update(
                                status="ok",
                                pipeline_record=record.to_dict(),
                                final_triples=[t.to_dict() for t in record.final_triples],
                                record_sha256=digest(record.to_dict()),
                            )
                        except Exception as exc:
                            value["error"] = str(exc)
                            logger.error("%s: %s", documents[i].document_id, exc)
                        value["sample_result"] = {
                            "document_id": documents[i].document_id,
                            "status": value["status"],
                            "error": value.get("error", ""),
                            "predicted_count": len(value["final_triples"]),
                            "critic_candidate_set_sizes": (
                                critic_chunk_sizes(
                                    len(proposals[i][0]["triples"]),
                                    self.candidate_grouping,
                                    seed=self.seed,
                                    document_id=documents[i].document_id,
                                )
                                if self.stage in {"pipeline", "critic"}
                                and self.pipeline.pipeline_variant == "proposer_critic_once"
                                and i in proposals
                                else None
                            ),
                        }
                        atomic_write_json(checkpoint_dir / f"{i:06d}.json", value)
                        checkpoints[i] = value
                        self._progress(run_dir, documents, checkpoints)
                        self._export(run_dir, checkpoints)
            return run_dir
        finally:
            try:
                if validated:
                    from evidenceloop.experiments.checkpoints import cumulative_critic_metrics
                    from evidenceloop.reports.statistics import run_statistics

                    current_metrics = self._critic_metrics()
                    from evidenceloop.experiments.checkpoints import CRITIC_ADDITIVE_METRICS

                    for key in CRITIC_ADDITIVE_METRICS:
                        if key in current_metrics:
                            current_metrics[key] -= metrics_before.get(key, 0)
                    previous_metrics = {
                        key: previous_summary[key]
                        for key in current_metrics
                        if key in previous_summary
                    }
                    progress = self._progress(run_dir, documents, checkpoints)
                    atomic_write_json(
                        run_dir / "summary.json",
                        {
                            **progress,
                            **run_statistics(checkpoints),
                            "samples_requested": len(documents),
                            **(
                                cumulative_critic_metrics(previous_metrics, current_metrics)
                                if current_metrics
                                else {}
                            ),
                            "runtime": runtime,
                            "runtime_config": {
                                **runtime,
                                "pipeline_variant": self.pipeline.pipeline_variant,
                                "critic_candidate_set_size": self.candidate_set,
                            },
                            "active_wall_seconds": round(
                                float(previous_summary.get("active_wall_seconds", 0))
                                + time.perf_counter()
                                - started_run,
                                6,
                            ),
                        },
                    )
                    self._export(run_dir, checkpoints)
            finally:
                os.close(lock_fd)
                lock.unlink(missing_ok=True)
