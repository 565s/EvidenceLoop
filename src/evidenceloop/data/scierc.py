from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from evidenceloop.validation import clean_predicate, predicate_validation_error

SCIERC_SOURCE_URL = "https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
SCIERC_PROJECT_URL = "https://nlp.cs.washington.edu/sciIE/"
SCIERC_GUIDELINE_URL = "https://nlp.cs.washington.edu/sciIE/annotation_guideline.pdf"
SYMMETRIC_RELATION_LABELS = frozenset({"COMPARE", "CONJUNCTION"})


class SciERCError(RuntimeError):
    pass


def normalized(value) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def load_relation_map(path: Path) -> Dict[str, str]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SciERCError(f"{path}: invalid relation-map JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping) or not value:
        raise SciERCError(f"{path}: expected a non-empty relation mapping")
    output = {}
    for raw_key, raw_predicate in value.items():
        key = str(raw_key or "").strip()
        predicate = clean_predicate(raw_predicate)
        if not key or predicate_validation_error(predicate):
            raise SciERCError(f"{path}: invalid relation mapping {raw_key!r}: {raw_predicate!r}")
        output[key] = predicate
    return output


def symmetric_predicates(relation_map: Mapping[str, str]) -> Tuple[str, ...]:
    missing = SYMMETRIC_RELATION_LABELS - set(relation_map)
    if missing:
        raise SciERCError("relation map is missing symmetric labels: " + ", ".join(sorted(missing)))
    return tuple(
        sorted(
            (relation_map[label] for label in SYMMETRIC_RELATION_LABELS),
            key=str.casefold,
        )
    )


def predicate_aware_triple_key(
    triple: Mapping[str, str],
    symmetric: Sequence[str],
) -> Tuple[str, str, str]:
    subject = normalized(triple.get("subject"))
    predicate = normalized(triple.get("predicate"))
    obj = normalized(triple.get("object"))
    if predicate in {normalized(value) for value in symmetric}:
        subject, obj = sorted((subject, obj))
    return subject, predicate, obj


def _safe_doc_key(value, context: str) -> Tuple[str, str]:
    original = str(value or "").strip()
    if not original:
        raise SciERCError(f"{context}: missing doc_key")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", original).strip("_")
    if not safe:
        raise SciERCError(f"{context}: doc_key has no usable characters")
    return original, safe


def _reconstruct_document(
    sentences,
    context: str,
) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    if not isinstance(sentences, list) or not sentences:
        raise SciERCError(f"{context}: sentences must be a non-empty list")
    chunks: List[str] = []
    token_char_spans: List[Tuple[int, int]] = []
    sentence_token_spans: List[Tuple[int, int]] = []
    sentence_char_spans: List[Tuple[int, int]] = []
    cursor = 0
    token_cursor = 0
    for sentence_index, raw_tokens in enumerate(sentences):
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise SciERCError(f"{context}: sentence {sentence_index} is empty or invalid")
        if sentence_index:
            chunks.append("\n")
            cursor += 1
        sentence_char_start = cursor
        sentence_token_start = token_cursor
        for token_index, raw_token in enumerate(raw_tokens):
            if not isinstance(raw_token, str) or not raw_token or "\n" in raw_token:
                raise SciERCError(
                    f"{context}: invalid token at sentence {sentence_index}, index {token_index}"
                )
            if token_index:
                chunks.append(" ")
                cursor += 1
            start = cursor
            chunks.append(raw_token)
            cursor += len(raw_token)
            token_char_spans.append((start, cursor))
            token_cursor += 1
        sentence_token_spans.append((sentence_token_start, token_cursor - 1))
        sentence_char_spans.append((sentence_char_start, cursor))
    return "".join(chunks), token_char_spans, sentence_token_spans, sentence_char_spans


def _span_surface(
    text: str,
    token_char_spans: Sequence[Tuple[int, int]],
    start: int,
    end: int,
) -> Tuple[str, Dict]:
    char_start = token_char_spans[start][0]
    char_end = token_char_spans[end][1]
    surface = text[char_start:char_end]
    return surface, {"surface": surface, "start": char_start, "end": char_end}


def convert_document(
    raw: Mapping,
    split_name: str,
    relation_map: Mapping[str, str],
    *,
    line_number: int = 0,
) -> Tuple[Dict, Counter]:
    context = f"{split_name} line {line_number}" if line_number else split_name
    if not isinstance(raw, Mapping):
        raise SciERCError(f"{context}: expected a JSON object")
    original_doc_key, safe_doc_key = _safe_doc_key(raw.get("doc_key"), context)
    text, token_char_spans, sentence_token_spans, sentence_char_spans = _reconstruct_document(
        raw.get("sentences"), context
    )
    raw_ner = raw.get("ner")
    raw_relations = raw.get("relations")
    if not isinstance(raw_ner, list) or len(raw_ner) != len(sentence_token_spans):
        raise SciERCError(f"{context}: ner must align with sentences")
    if not isinstance(raw_relations, list) or len(raw_relations) != len(sentence_token_spans):
        raise SciERCError(f"{context}: relations must align with sentences")

    stats = Counter(
        {
            "raw_documents": 1,
            "raw_sentences": len(sentence_token_spans),
            "raw_tokens": len(token_char_spans),
        }
    )
    valid_ner_spans = set()
    entities = []
    seen_entities = set()
    for sentence_index, mentions in enumerate(raw_ner):
        if not isinstance(mentions, list):
            stats["invalid_ner_fields"] += 1
            continue
        sentence_start, sentence_end = sentence_token_spans[sentence_index]
        for mention in mentions:
            stats["raw_entity_mentions"] += 1
            if (
                not isinstance(mention, list)
                or len(mention) != 3
                or not isinstance(mention[0], int)
                or not isinstance(mention[1], int)
            ):
                stats["invalid_entity_mentions"] += 1
                continue
            start, end, entity_type = mention
            if not sentence_start <= start <= end <= sentence_end:
                stats["invalid_entity_spans"] += 1
                continue
            surface, _ = _span_surface(text, token_char_spans, start, end)
            if not surface.strip():
                stats["empty_entity_surfaces"] += 1
                continue
            valid_ner_spans.add((start, end))
            stats[f"entity_type:{str(entity_type or '').strip() or '<empty>'}"] += 1
            key = normalized(surface)
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append(surface)
            stats["valid_entity_mentions"] += 1

    clusters = raw.get("clusters")
    if isinstance(clusters, list):
        stats["coreference_clusters"] += len(clusters)
        stats["coreference_cluster_mentions"] += sum(
            len(cluster) for cluster in clusters if isinstance(cluster, list)
        )
    else:
        stats["invalid_coreference_fields"] += 1

    symmetric = symmetric_predicates(relation_map)
    gold_items = []
    seen_triples = set()
    for sentence_index, relations in enumerate(raw_relations):
        if not isinstance(relations, list):
            stats["invalid_relation_fields"] += 1
            continue
        sentence_start, sentence_end = sentence_token_spans[sentence_index]
        evidence_start, evidence_end = sentence_char_spans[sentence_index]
        evidence = text[evidence_start:evidence_end]
        for relation in relations:
            stats["raw_relations"] += 1
            if (
                not isinstance(relation, list)
                or len(relation) != 5
                or any(not isinstance(value, int) for value in relation[:4])
            ):
                stats["invalid_relation_structures"] += 1
                continue
            head_start, head_end, tail_start, tail_end, raw_label = relation
            label = str(raw_label or "").strip()
            stats[f"relation_label:{label or '<empty>'}"] += 1
            predicate = relation_map.get(label)
            if not predicate:
                stats["unknown_relations"] += 1
                continue
            if not (
                0 <= head_start <= head_end < len(token_char_spans)
                and 0 <= tail_start <= tail_end < len(token_char_spans)
            ):
                stats["invalid_relation_spans"] += 1
                continue
            if not (
                sentence_start <= head_start <= head_end <= sentence_end
                and sentence_start <= tail_start <= tail_end <= sentence_end
            ):
                stats["cross_sentence_or_misaligned_relations"] += 1
                continue
            if (head_start, head_end) not in valid_ner_spans or (
                tail_start,
                tail_end,
            ) not in valid_ner_spans:
                stats["relation_endpoints_missing_ner"] += 1
                continue
            subject, subject_span = _span_surface(text, token_char_spans, head_start, head_end)
            obj, object_span = _span_surface(text, token_char_spans, tail_start, tail_end)
            triple = {"subject": subject, "predicate": predicate, "object": obj}
            if any(not str(value).strip() for value in triple.values()):
                stats["empty_triple_fields"] += 1
                continue
            if subject not in evidence or obj not in evidence or evidence not in text:
                stats["invalid_evidence"] += 1
                continue
            key = predicate_aware_triple_key(triple, symmetric)
            if key in seen_triples:
                stats["duplicate_triples"] += 1
                continue
            seen_triples.add(key)
            gold_items.append(
                {
                    "triple": triple,
                    "subject_span": subject_span,
                    "object_span": object_span,
                    "evidence": evidence,
                    "relation_label": label,
                    "sentence_index": sentence_index,
                }
            )
            stats["valid_triples"] += 1

    document = {
        "document_id": f"scierc_{split_name}_{safe_doc_key}",
        "doc_key": original_doc_key,
        "text": text,
        "entities": entities,
        "predicates": list(dict.fromkeys(item["triple"]["predicate"] for item in gold_items)),
        "gold_items": gold_items,
    }
    return document, stats


def load_documents(
    path: Path,
    split_name: str,
    relation_map: Mapping[str, str],
) -> Tuple[List[Dict], Counter]:
    path = Path(path)
    documents = []
    total_stats = Counter()
    seen_doc_keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                total_stats["blank_lines"] += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SciERCError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            document, stats = convert_document(
                raw, split_name, relation_map, line_number=line_number
            )
            if document["doc_key"] in seen_doc_keys:
                raise SciERCError(
                    f"{path}:{line_number}: duplicate doc_key {document['doc_key']!r}"
                )
            seen_doc_keys.add(document["doc_key"])
            documents.append(document)
            total_stats.update(stats)
    if not documents:
        raise SciERCError(f"{path}: no documents found")
    return documents, total_stats
