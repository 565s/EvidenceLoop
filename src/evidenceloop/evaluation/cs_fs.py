"""Evaluate CS and source-grounded FS with explicitly configured API backends.

OPENAI_BASE_URL and OPENAI_API_KEY configure the compatible endpoint; historical
SIFLOW environment names remain readable. Importing creates no client.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from evidenceloop.evaluation.graphjudge import (  # noqa: E402
    EvaluationInputError,
    Triple,
    load_gold_graphs_with_metadata,
    load_prediction_graphs,
    resolve_prediction_path,
)
from evidenceloop.io import _atomic_write_text

ROOT = Path.cwd()


PROTOCOL = "genres_api_models_v1"
CS_COMPOSITION = "element_sum"
FS_PROMPT_VERSION = "genres_fact_checker_v1"
METRIC_CHOICES = ("cs", "fs", "both")
EMPTY_GOLD_POLICIES = ("skip", "error")
DEFAULT_CS_THRESHOLD = 0.95
DEFAULT_FS_MAX_WORKERS = 8

FACT_CHECKER_PROMPT = """Evaluate the factualness of an extracted relationship (triplet) based on the given source text. Indicate whether the relationship accurately reflects the information in the source text by responding with \"true\" or \"false\". You should only output \"true\" or \"false\" with no additional information.

Example 1:
Source Text: The Great Barrier Reef, located off the coast of Australia, is the world's largest coral reef system. It has been severely affected by climate change, leading to coral bleaching.
Relationship: [\"Great Barrier Reef\", \"affected by\", \"climate change\"]
Factualness: true

Example 2:
Source Text: The Eiffel Tower was constructed in 1889 and is located in Paris, France. It is one of the most recognizable structures in the world.
Relationship: [\"Eiffel Tower\", \"located in\", \"London\"]
Factualness: false

Example 3:
Source Text: The novel \"Moby-Dick\" by Herman Melville features a ship named Pequod. The narrative follows the ship and its crew in their pursuit of a giant white sperm whale.
Relationship: [\"Moby-Dick\", \"is about\", \"a whale named Pequod\"]
Factualness: false


Source Text: $TEXT$
Relationship: $TRIPLE$
Factualness:"""


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #
def _load_env(env_path: Optional[Path]) -> None:
    if env_path is not None:
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"environment variable {name!r} is not set; add it to your .env file")
    return value


def _resolve_siliconflow() -> Tuple[str, str]:
    """Read the shared OpenAI-compatible endpoint credentials."""
    base_url = (os.environ.get("OPENAI_BASE_URL") or _require_env("SIFLOW_BASE_URL")).rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY") or _require_env("SIFLOW_API_KEY")
    return base_url, api_key


# --------------------------------------------------------------------------- #
# hashing / cache keys
# --------------------------------------------------------------------------- #
def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_hash() -> str:
    return _sha256_text(FS_PROMPT_VERSION + "\n" + FACT_CHECKER_PROMPT)


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, Mapping):
                raise EvaluationInputError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _record_id(row: Mapping[str, Any], context: str) -> str:
    document_id = str(row.get("document_id") or "").strip()
    sample_id = str(row.get("sample_id") or "").strip()
    if document_id and sample_id and document_id != sample_id:
        raise EvaluationInputError(
            f"{context}: document_id {document_id!r} conflicts with sample_id {sample_id!r}"
        )
    value = document_id or sample_id
    if not value:
        raise EvaluationInputError(f"{context}: missing document_id/sample_id")
    return value


def load_documents(path: Path, required_ids: Sequence[str]) -> OrderedDict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = set(required_ids)
    loaded: Dict[str, str] = {}
    seen: set[str] = set()
    for index, row in enumerate(_read_jsonl(path), 1):
        document_id = _record_id(row, f"{path}: row {index}")
        if document_id in seen:
            raise EvaluationInputError(f"{path}: duplicate document id {document_id!r}")
        seen.add(document_id)
        if document_id not in required:
            continue
        text = str(
            row.get("text")
            or row.get("article")
            or row.get("source_text")
            or row.get("context")
            or ""
        ).strip()
        if not text:
            raise EvaluationInputError(f"{path}: row {index}: missing document text")
        loaded[document_id] = text
    missing = [document_id for document_id in required_ids if document_id not in loaded]
    if missing:
        raise EvaluationInputError(
            f"{path}: missing {len(missing)} required documents: {missing[:5]}"
        )
    return OrderedDict((document_id, loaded[document_id]) for document_id in required_ids)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    _atomic_write_text(path, content)


# --------------------------------------------------------------------------- #
# API model fingerprint & cache keys
# --------------------------------------------------------------------------- #
def api_model_fingerprint(kind: str, base_url: str, model: str) -> str:
    """Stable fingerprint for an API-backed model.

    Bakes in the provider base URL and model name so that swapping providers
    or versions automatically invalidates old caches.
    """
    digest = hashlib.sha256(kind.encode("utf-8"))
    digest.update(base_url.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    return digest.hexdigest()


def _embedding_key(model_id: str, text: str) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "composition": CS_COMPOSITION,
                "model": model_id,
                "text": text,
            }
        )
    )


def _judgment_key(
    model_id: str,
    document_text: str,
    triple: Triple,
    max_new_tokens: int,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "prompt_hash": _prompt_hash(),
                "model": model_id,
                "document_hash": _sha256_text(document_text),
                "triple": list(triple),
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
        )
    )


# --------------------------------------------------------------------------- #
# embedding cache
# --------------------------------------------------------------------------- #
def load_embedding_cache(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    cache: Dict[str, List[float]] = {}
    for index, row in enumerate(_read_jsonl(path), 1):
        key = str(row.get("key") or "")
        vector = row.get("embedding")
        if not key or not isinstance(vector, list) or not vector:
            raise EvaluationInputError(f"{path}: row {index}: invalid embedding cache entry")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EvaluationInputError(
                f"{path}: row {index}: embedding must contain numbers"
            ) from exc
        if any(not math.isfinite(value) for value in values):
            raise EvaluationInputError(
                f"{path}: row {index}: embedding contains a non-finite value"
            )
        cache[key] = values
    return cache


def save_embedding_cache(path: Path, cache: Mapping[str, Sequence[float]]) -> None:
    rows = (
        {"key": key, "embedding": [float(value) for value in cache[key]]} for key in sorted(cache)
    )
    _atomic_write_jsonl(path, rows)


def load_judgment_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(path), 1):
        key = str(row.get("key") or "")
        supported = row.get("supported")
        if not key or not isinstance(supported, bool):
            raise EvaluationInputError(f"{path}: row {index}: invalid FS cache entry")
        cache[key] = dict(row)
    return cache


def save_judgment_cache(path: Path, cache: Mapping[str, Mapping[str, Any]]) -> None:
    _atomic_write_jsonl(path, (cache[key] for key in sorted(cache)))


def _dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in ("openai", "python-dotenv", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


# --------------------------------------------------------------------------- #
# API backends (drop-in replacements for the local encoder / judge)
# --------------------------------------------------------------------------- #
class APIEmbeddingEncoder:
    """OpenAI-compatible embedding API client.

    Mirrors the subset of SentenceTransformerEncoder used by this script:
        .encode(texts, batch_size) -> np.ndarray (L2-normalized)
        .close()
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: Optional[int] = None,
        timeout: float = 60.0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:
        import numpy as np

        results: List[List[float]] = []
        for offset in range(0, len(texts), batch_size):
            batch = list(texts[offset : offset + batch_size])
            kwargs: Dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)
            vectors = [item.embedding for item in resp.data]
            # L2-normalize to match normalize_embeddings=True in the local variant.
            arr = np.asarray(vectors, dtype=float)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arr = arr / norms
            results.extend(arr.tolist())
        return results

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


class APIFactualnessJudge:
    """OpenAI-compatible chat-completions factualness judge.

    Mirrors LocalFactualnessJudge:
        .generate(prompts, max_new_tokens) -> List[str]
        .close()
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_workers: int = DEFAULT_FS_MAX_WORKERS,
        timeout: float = 120.0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_workers = max_workers

    def _one(self, prompt: str, max_new_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        prompts = list(prompts)
        if self.max_workers <= 1 or len(prompts) <= 1:
            return [self._one(p, max_new_tokens) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self._one(p, max_new_tokens), prompts))

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def _release_backend(backend: Any) -> None:
    if backend is not None and hasattr(backend, "close"):
        backend.close()
    del backend


# --------------------------------------------------------------------------- #
# prompt / parsing
# --------------------------------------------------------------------------- #
def build_factualness_prompt(document_text: str, triple: Triple) -> str:
    return FACT_CHECKER_PROMPT.replace("$TEXT$", document_text).replace(
        "$TRIPLE$", json.dumps(list(triple), ensure_ascii=False)
    )


def parse_factualness_output(value: str) -> bool:
    normalized = str(value or "").strip().casefold()
    # 从输出里找第一个true/false，忽略后面所有多余文字
    if "true" in normalized:
        return True
    elif "false" in normalized:
        return False
    raise ValueError(
        f"FS judge must output exactly 'true' or 'false'; received {str(value)[:120]!r}"
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    value = float(np.dot(left, right) / denominator)
    return max(-1.0, min(1.0, value))


def build_triple_embedding(
    triple: Triple,
    element_embeddings: Mapping[str, Sequence[float]],
) -> Any:
    import numpy as np

    vectors = [np.asarray(element_embeddings[element], dtype=float) for element in triple]
    dimensions = {vector.shape for vector in vectors}
    if len(dimensions) != 1 or len(next(iter(dimensions))) != 1:
        raise ValueError(f"inconsistent element embedding dimensions for triple {triple!r}")
    return vectors[0] + vectors[1] + vectors[2]


def compute_cs(
    gold_by_id: Mapping[str, Sequence[Triple]],
    predicted_by_id: Mapping[str, Sequence[Triple]],
    element_embeddings: Mapping[str, Sequence[float]],
    threshold: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    scores: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []
    for document_id, gold_triples in gold_by_id.items():
        predicted_triples = list(predicted_by_id[document_id])
        predicted_vectors = [
            build_triple_embedding(triple, element_embeddings) for triple in predicted_triples
        ]
        matched = 0
        for gold_index, gold_triple in enumerate(gold_triples):
            gold_vector = build_triple_embedding(gold_triple, element_embeddings)
            similarities = [_cosine(gold_vector, vector) for vector in predicted_vectors]
            if similarities:
                best_index = max(range(len(similarities)), key=similarities.__getitem__)
                best_similarity = similarities[best_index]
                best_prediction: Optional[Triple] = predicted_triples[best_index]
            else:
                best_index = None
                best_similarity = 0.0
                best_prediction = None
            is_matched = best_prediction is not None and best_similarity >= threshold
            matched += int(is_matched)
            details.append(
                {
                    "document_id": document_id,
                    "gold_index": gold_index,
                    "gold_triple": list(gold_triple),
                    "best_prediction_index": best_index,
                    "best_prediction": list(best_prediction) if best_prediction else None,
                    "best_similarity": best_similarity,
                    "matched": is_matched,
                }
            )
        scores[document_id] = matched / len(gold_triples) if gold_triples else 0.0
    return scores, details


def _unique_elements(
    gold_by_id: Mapping[str, Sequence[Triple]],
    predicted_by_id: Mapping[str, Sequence[Triple]],
) -> List[str]:
    return sorted(
        {
            element
            for graphs in (gold_by_id, predicted_by_id)
            for triples in graphs.values()
            for triple in triples
            for element in triple
        }
    )


def encode_elements(
    elements: Sequence[str],
    *,
    model_id: str,
    encoder: Any,
    batch_size: int,
    cache_path: Path,
) -> Dict[str, List[float]]:
    cache = load_embedding_cache(cache_path)
    missing = [text for text in elements if _embedding_key(model_id, text) not in cache]
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        encoded = encoder.encode(batch, batch_size=batch_size)
        if len(encoded) != len(batch):
            raise RuntimeError(
                f"embedding API returned {len(encoded)} vectors for {len(batch)} inputs"
            )
        for text, vector in zip(batch, encoded):
            values = [float(value) for value in vector]
            if not values or any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"embedding API returned an invalid vector for {text!r}")
            cache[_embedding_key(model_id, text)] = values
        save_embedding_cache(cache_path, cache)
    return {text: cache[_embedding_key(model_id, text)] for text in elements}


def compute_fs(
    predicted_by_id: Mapping[str, Sequence[Triple]],
    documents_by_id: Mapping[str, str],
    *,
    model_id: str,
    judge: Any,
    batch_size: int,
    max_new_tokens: int,
    cache_path: Path,
) -> Tuple[Dict[str, float], Dict[str, int], int, int]:
    cache = load_judgment_cache(cache_path)
    pending: List[Tuple[str, int, Triple, str, str]] = []
    pending_keys: set[str] = set()
    for document_id, triples in predicted_by_id.items():
        document_text = documents_by_id[document_id]
        for triple_index, triple in enumerate(triples):
            key = _judgment_key(model_id, document_text, triple, max_new_tokens)
            if key not in cache and key not in pending_keys:
                pending.append((document_id, triple_index, triple, document_text, key))
                pending_keys.add(key)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        prompts = [build_factualness_prompt(item[3], item[2]) for item in batch]
        outputs = judge.generate(prompts, max_new_tokens=max_new_tokens)
        if len(outputs) != len(batch):
            raise RuntimeError(f"FS judge returned {len(outputs)} outputs for {len(batch)} prompts")
        for item, raw_output in zip(batch, outputs):
            document_id, triple_index, triple, document_text, key = item
            try:
                supported = parse_factualness_output(raw_output)
            except ValueError as exc:
                print(f"WARNING parse failed, treat as false: {exc}")
                supported = False
            cache[key] = {
                "key": key,
                "document_id": document_id,
                "triple_index": triple_index,
                "triple": list(triple),
                "document_hash": _sha256_text(document_text),
                "model_fingerprint": model_id,
                "prompt_hash": _prompt_hash(),
                "supported": supported,
                "raw_output": str(raw_output).strip(),
            }
        save_judgment_cache(cache_path, cache)

    scores: Dict[str, float] = {}
    supported_counts: Dict[str, int] = {}
    supported_total = 0
    predicted_total = 0
    for document_id, triples in predicted_by_id.items():
        document_text = documents_by_id[document_id]
        count = sum(
            int(cache[_judgment_key(model_id, document_text, triple, max_new_tokens)]["supported"])
            for triple in triples
        )
        supported_counts[document_id] = count
        scores[document_id] = count / len(triples) if triples else 0.0
        supported_total += count
        predicted_total += len(triples)
    return scores, supported_counts, supported_total, predicted_total


# --------------------------------------------------------------------------- #
# env-backed model config
# --------------------------------------------------------------------------- #
def _resolve_embedding_config(cli_model: Optional[str]) -> Tuple[str, str, str]:
    base_url, api_key = _resolve_siliconflow()
    model = cli_model or os.environ.get("EMBEDDING_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "embedding model is not set; pass --embedding-model or set EMBEDDING_MODEL in .env"
        )
    return base_url, api_key, model


def _resolve_fs_config(cli_model: Optional[str]) -> Tuple[str, str, str, int]:
    base_url, api_key = _resolve_siliconflow()
    model = cli_model or os.environ.get("FS_MODEL", "").strip()
    if not model:
        raise RuntimeError("factualness model is not set; pass --fs-model or set FS_MODEL in .env")
    max_workers = int(os.environ.get("FS_MAX_WORKERS", str(DEFAULT_FS_MAX_WORKERS)))
    return base_url, api_key, model, max_workers


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def evaluate_run(
    run_dir: Path,
    *,
    metrics: str = "both",
    gold_path: Optional[Path] = None,
    documents_path: Optional[Path] = None,
    prediction_path: Optional[Path] = None,
    embedding_model: Optional[str] = None,
    fs_model: Optional[str] = None,
    cs_threshold: float = DEFAULT_CS_THRESHOLD,
    embedding_batch_size: int = 64,
    fs_batch_size: int = 8,
    fs_max_new_tokens: int = 8,
    empty_gold_policy: str = "skip",
    output_path: Optional[Path] = None,
    per_sample_output_path: Optional[Path] = None,
    cs_details_output_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    env_path: Optional[Path] = None,
    embedding_encoder: Any = None,
    factualness_judge: Any = None,
) -> Dict[str, Any]:
    _load_env(env_path)

    if metrics not in METRIC_CHOICES:
        raise ValueError(f"metrics must be one of {METRIC_CHOICES}")
    if empty_gold_policy not in EMPTY_GOLD_POLICIES:
        raise ValueError(f"empty_gold_policy must be one of {EMPTY_GOLD_POLICIES}")
    if not 0.0 <= cs_threshold <= 1.0:
        raise ValueError("cs_threshold must be between 0 and 1")
    if embedding_batch_size <= 0 or fs_batch_size <= 0 or fs_max_new_tokens <= 0:
        raise ValueError("batch sizes and fs_max_new_tokens must be positive")

    run_dir = run_dir.expanduser().resolve()
    gold_path = (gold_path or run_dir / "gold_triples.jsonl").expanduser().resolve()
    if not gold_path.is_file():
        raise FileNotFoundError(gold_path)
    prediction_path = resolve_prediction_path(
        run_dir,
        prediction_path.expanduser().resolve() if prediction_path else None,
    ).resolve()
    output_path = (output_path or run_dir / "cs_fs_metrics.json").expanduser().resolve()
    per_sample_output_path = (
        (per_sample_output_path or run_dir / "cs_fs_per_sample.jsonl").expanduser().resolve()
    )
    cs_details_output_path = (
        (cs_details_output_path or run_dir / "cs_details.jsonl").expanduser().resolve()
    )
    cache_dir = (cache_dir or run_dir / "cs_fs_cache").expanduser().resolve()

    gold_by_id, skipped_gold_ids, samples_total_input = load_gold_graphs_with_metadata(
        gold_path, empty_gold_policy
    )
    document_ids = list(gold_by_id)
    predicted_by_id, missing_prediction_ids = load_prediction_graphs(
        prediction_path,
        document_ids,
        ignored_ids=skipped_gold_ids,
    )
    for document_id in missing_prediction_ids:
        predicted_by_id[document_id] = []

    documents_by_id: Optional[OrderedDict[str, str]] = None
    if metrics in {"fs", "both"}:
        documents_path = (documents_path or run_dir / "documents.jsonl").expanduser().resolve()
        documents_by_id = load_documents(documents_path, document_ids)
    elif documents_path is not None:
        documents_path = documents_path.expanduser().resolve()

    embedding_runtime: Dict[str, Any] = {}
    fs_runtime: Dict[str, Any] = {}
    cs_scores: Dict[str, float] = {}
    cs_details: List[Dict[str, Any]] = []
    fs_scores: Dict[str, float] = {}
    supported_counts: Dict[str, int] = {}
    supported_total = 0
    fs_prediction_total = 0
    embedding_fingerprint: Optional[str] = None
    fs_fingerprint: Optional[str] = None
    embedding_meta: Optional[Dict[str, Any]] = None
    fs_meta: Optional[Dict[str, Any]] = None

    if metrics in {"cs", "both"}:
        emb_base, emb_key, emb_model = _resolve_embedding_config(embedding_model)
        embedding_fingerprint = api_model_fingerprint("embedding", emb_base, emb_model)
        encoder = embedding_encoder
        if encoder is None:
            encoder = APIEmbeddingEncoder(base_url=emb_base, api_key=emb_key, model=emb_model)
        embedding_runtime = {"transport": "openai-compatible-http"}
        embedding_meta = {"base_url": emb_base, "model": emb_model}
        try:
            elements = _unique_elements(gold_by_id, predicted_by_id)
            embeddings = encode_elements(
                elements,
                model_id=embedding_fingerprint,
                encoder=encoder,
                batch_size=embedding_batch_size,
                cache_path=cache_dir / "cs_embeddings.jsonl",
            )
            cs_scores, cs_details = compute_cs(
                gold_by_id,
                predicted_by_id,
                embeddings,
                cs_threshold,
            )
        finally:
            _release_backend(encoder)

    if metrics in {"fs", "both"}:
        fs_base, fs_key, fs_model_name, fs_workers = _resolve_fs_config(fs_model)
        fs_fingerprint = api_model_fingerprint("fs", fs_base, fs_model_name)
        judge = factualness_judge
        if judge is None:
            judge = APIFactualnessJudge(
                base_url=fs_base,
                api_key=fs_key,
                model=fs_model_name,
                max_workers=fs_workers,
            )
        fs_runtime = {
            "transport": "openai-compatible-http",
            "max_workers": fs_workers,
        }
        fs_meta = {"base_url": fs_base, "model": fs_model_name}
        try:
            assert documents_by_id is not None
            fs_scores, supported_counts, supported_total, fs_prediction_total = compute_fs(
                predicted_by_id,
                documents_by_id,
                model_id=fs_fingerprint,
                judge=judge,
                batch_size=fs_batch_size,
                max_new_tokens=fs_max_new_tokens,
                cache_path=cache_dir / "fs_judgments.jsonl",
            )
        finally:
            _release_backend(judge)

    per_sample: List[Dict[str, Any]] = []
    for document_id in document_ids:
        row: Dict[str, Any] = {
            "document_id": document_id,
            "gold_count": len(gold_by_id[document_id]),
            "predicted_count": len(predicted_by_id[document_id]),
            "prediction_missing": document_id in set(missing_prediction_ids),
        }
        if metrics in {"cs", "both"}:
            row["CS"] = cs_scores[document_id]
        if metrics in {"fs", "both"}:
            row["FS"] = fs_scores[document_id]
            row["supported_count"] = supported_counts[document_id]
        per_sample.append(row)

    metric_values: Dict[str, float] = {}
    if metrics in {"cs", "both"}:
        metric_values["CS"] = sum(cs_scores.values()) / len(cs_scores)
    if metrics in {"fs", "both"}:
        metric_values["FS"] = sum(fs_scores.values()) / len(fs_scores)
        metric_values["FS_micro"] = (
            supported_total / fs_prediction_total if fs_prediction_total else 0.0
        )

    summary: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_note": (
            "GenRES score definitions with OpenAI-compatible API "
            "backends; credentials loaded from .env (OPENAI_API_KEY / "
            "OPENAI_BASE_URL). Not a numerically identical reproduction of the "
            "original OpenAI evaluator."
        ),
        "metrics_requested": metrics,
        "inputs": {
            "run_dir": str(run_dir),
            "gold_file": str(gold_path),
            "documents_file": str(documents_path) if documents_path else None,
            "prediction_file": str(prediction_path),
        },
        "models": {
            "embedding": (
                {
                    **(embedding_meta or {}),
                    "fingerprint": embedding_fingerprint,
                    **embedding_runtime,
                }
                if embedding_meta
                else None
            ),
            "factualness": (
                {
                    **(fs_meta or {}),
                    "fingerprint": fs_fingerprint,
                    **fs_runtime,
                }
                if fs_meta
                else None
            ),
        },
        "cs_config": (
            {
                "triple_composition": CS_COMPOSITION,
                "element_embeddings_normalized": True,
                "matching": "reusable_best_match_per_gold",
                "threshold": cs_threshold,
                "aggregation": "document_macro_mean",
                "embedding_batch_size": embedding_batch_size,
            }
            if metrics in {"cs", "both"}
            else None
        ),
        "fs_config": (
            {
                "prompt_version": FS_PROMPT_VERSION,
                "prompt_hash": _prompt_hash(),
                "do_sample": False,
                "aggregation": "document_macro_mean",
                "micro_aggregation": "supported_triples_over_all_predictions",
                "batch_size": fs_batch_size,
                "max_workers": fs_workers if metrics in {"fs", "both"} else None,
                "max_new_tokens": fs_max_new_tokens,
                "empty_prediction_score": 0.0,
                "source_truncation": False,
            }
            if metrics in {"fs", "both"}
            else None
        ),
        "empty_gold_policy": empty_gold_policy,
        "samples": {
            "total_input": samples_total_input,
            "evaluated": len(document_ids),
            "skipped_empty_gold": len(skipped_gold_ids),
            "skipped_empty_gold_ids": skipped_gold_ids,
            "missing_predictions": len(missing_prediction_ids),
            "missing_prediction_ids": missing_prediction_ids,
            "gold_triples": sum(len(values) for values in gold_by_id.values()),
            "predicted_triples": sum(len(values) for values in predicted_by_id.values()),
            "supported_predictions": supported_total if metrics in {"fs", "both"} else None,
        },
        "dependencies": _dependency_versions(),
        "metrics": metric_values,
    }
    _atomic_write_json(output_path, summary)
    _atomic_write_jsonl(per_sample_output_path, per_sample)
    if metrics in {"cs", "both"}:
        _atomic_write_jsonl(cs_details_output_path, cs_details)
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"Protocol: {summary['protocol']}")
    print(
        f"Samples: {summary['samples']['evaluated']} "
        f"(skipped empty gold: {summary['samples']['skipped_empty_gold']}, "
        f"missing predictions: {summary['samples']['missing_predictions']})"
    )
    for name, value in summary["metrics"].items():
        print(f"{name}: {float(value):.4f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a complete experiment KG with GenRES-style CS and FS "
            "using OpenAI-compatible API backends configured via .env."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics", choices=METRIC_CHOICES, default="both")
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--documents-file", type=Path)
    parser.add_argument("--pred-file", type=Path)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="override EMBEDDING_MODEL from .env",
    )
    parser.add_argument(
        "--fs-model",
        type=str,
        default=None,
        help="override FS_MODEL from .env",
    )
    parser.add_argument("--cs-threshold", type=float, default=DEFAULT_CS_THRESHOLD)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--fs-batch-size",
        type=int,
        default=8,
        help="number of FS prompts submitted per HTTP round (each prompt is submitted independently)",
    )
    parser.add_argument("--fs-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--empty-gold-policy",
        choices=EMPTY_GOLD_POLICIES,
        default="skip",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-sample-output", type=Path)
    parser.add_argument("--cs-details-output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="path to .env file (default: auto-discover via python-dotenv)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = evaluate_run(
            run_dir=args.run_dir,
            metrics=args.metrics,
            gold_path=args.gold_file,
            documents_path=args.documents_file,
            prediction_path=args.pred_file,
            embedding_model=args.embedding_model,
            fs_model=args.fs_model,
            cs_threshold=args.cs_threshold,
            embedding_batch_size=args.embedding_batch_size,
            fs_batch_size=args.fs_batch_size,
            fs_max_new_tokens=args.fs_max_new_tokens,
            empty_gold_policy=args.empty_gold_policy,
            output_path=args.output,
            per_sample_output_path=args.per_sample_output,
            cs_details_output_path=args.cs_details_output,
            cache_dir=args.cache_dir,
            env_path=args.env_file,
        )
    except (
        EvaluationInputError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
