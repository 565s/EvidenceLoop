from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from evidenceloop.validation import clean_predicate, predicate_validation_error

REDOCRED_SOURCE_URLS = {
    "train": "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/train_revised.json",
    "validation": "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/dev_revised.json",
    "test": "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/test_revised.json",
    "relation_map": "https://huggingface.co/datasets/thunlp/docred/resolve/main/data/rel_info.json.gz",
}


class ReDocREDError(ValueError):
    pass


def _normalized(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _triple_key(triple: Mapping[str, str]) -> Tuple[str, str, str]:
    return tuple(_normalized(triple.get(field)) for field in ("subject", "predicate", "object"))


def _read_json(path: Path):
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_relation_map(path: Path) -> Dict[str, str]:
    value = _read_json(Path(path))
    if not isinstance(value, Mapping) or not value:
        raise ReDocREDError(f"{path}: expected a non-empty relation mapping")
    output = {}
    for raw_key, raw_label in value.items():
        key = str(raw_key or "").strip()
        label = clean_predicate(raw_label)
        if not key or predicate_validation_error(label):
            raise ReDocREDError(f"{path}: invalid relation mapping {raw_key!r}: {raw_label!r}")
        output[key] = label
    return output


def load_split(path: Path) -> List[Mapping]:
    value = _read_json(Path(path))
    if not isinstance(value, list):
        raise ReDocREDError(f"{path}: expected a JSON array")
    if not value:
        raise ReDocREDError(f"{path}: dataset split is empty")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ReDocREDError(f"{path}: record {index} is not an object")
    return value


def source_document_id(split_name: str, index: int) -> str:
    return f"redocred_{split_name}_{index:06d}"


def _sentence_layout(raw_sents) -> Tuple[List[List[str]], str, List[Tuple[int, int]]]:
    if not isinstance(raw_sents, list) or not raw_sents:
        raise ReDocREDError("document.sents: expected a non-empty list")

    sentences: List[List[str]] = []
    sentence_texts = []
    spans = []
    cursor = 0
    for sentence_index, raw_sentence in enumerate(raw_sents):
        if not isinstance(raw_sentence, list):
            raise ReDocREDError(f"document.sents[{sentence_index}]: expected a list")
        tokens = [str(token) for token in raw_sentence]
        if any(not token for token in tokens):
            raise ReDocREDError(f"document.sents[{sentence_index}]: empty token")
        sentence = " ".join(tokens)
        sentences.append(tokens)
        sentence_texts.append(sentence)
        spans.append((cursor, cursor + len(sentence)))
        cursor += len(sentence) + 1
    return sentences, "\n".join(sentence_texts), spans


def _mention_value(
    raw: Mapping,
    sentences: Sequence[Sequence[str]],
    sentence_spans: Sequence[Tuple[int, int]],
) -> Dict | None:
    sent_id = raw.get("sent_id")
    pos = raw.get("pos")
    if not isinstance(sent_id, int) or not (0 <= sent_id < len(sentences)):
        return None
    if not isinstance(pos, list) or len(pos) != 2:
        return None
    start_token, end_token = pos
    tokens = sentences[sent_id]
    if (
        not isinstance(start_token, int)
        or not isinstance(end_token, int)
        or not (0 <= start_token < end_token <= len(tokens))
    ):
        return None

    surface = " ".join(tokens[start_token:end_token]).strip()
    if not surface:
        return None
    relative_start = sum(len(token) + 1 for token in tokens[:start_token])
    relative_end = relative_start + len(surface)
    sentence_start = sentence_spans[sent_id][0]
    return {
        "surface": surface,
        "start": sentence_start + relative_start,
        "end": sentence_start + relative_end,
        "sent_id": sent_id,
    }


def _entity_clusters(
    raw_vertex_set,
    sentences: Sequence[Sequence[str]],
    sentence_spans: Sequence[Tuple[int, int]],
) -> Tuple[List[List[Dict]], Counter]:
    stats = Counter()
    if not isinstance(raw_vertex_set, list):
        return [], Counter({"invalid_vertex_set": 1})
    clusters = []
    for raw_cluster in raw_vertex_set:
        mentions = []
        if isinstance(raw_cluster, list):
            for raw_mention in raw_cluster:
                mention = (
                    _mention_value(raw_mention, sentences, sentence_spans)
                    if isinstance(raw_mention, Mapping)
                    else None
                )
                if mention is None:
                    stats["invalid_mentions"] += 1
                else:
                    mentions.append(mention)
        if not mentions:
            stats["empty_entity_clusters"] += 1
        clusters.append(mentions)
    return clusters, stats


def _valid_evidence_ids(value, sentence_count: int) -> List[int]:
    if not isinstance(value, list):
        return []
    return sorted({item for item in value if isinstance(item, int) and 0 <= item < sentence_count})


def _select_mention_pair(
    heads: Sequence[Mapping],
    tails: Sequence[Mapping],
    evidence_ids: Sequence[int],
) -> Tuple[Mapping, Mapping]:
    def pair_key(pair):
        head, tail = pair
        sentence_ids = list(evidence_ids) + [head["sent_id"], tail["sent_id"]]
        return (
            max(sentence_ids) - min(sentence_ids),
            abs(head["sent_id"] - tail["sent_id"]),
            head["start"],
            tail["start"],
        )

    return min(((head, tail) for head in heads for tail in tails), key=pair_key)


def _evidence_span(
    text: str,
    sentence_spans: Sequence[Tuple[int, int]],
    head: Mapping,
    tail: Mapping,
    evidence_ids: Sequence[int],
) -> str:
    sentence_ids = list(evidence_ids) + [head["sent_id"], tail["sent_id"]]
    first = min(sentence_ids)
    last = max(sentence_ids)
    evidence = text[sentence_spans[first][0] : sentence_spans[last][1]]
    if head["surface"] not in evidence or tail["surface"] not in evidence:
        return ""
    return evidence


def convert_document(
    article: Mapping,
    split_name: str,
    index: int,
    relation_map: Mapping[str, str],
) -> Tuple[Dict, Counter]:
    sentences, text, sentence_spans = _sentence_layout(article.get("sents"))
    clusters, stats = _entity_clusters(article.get("vertexSet"), sentences, sentence_spans)
    raw_labels = article.get("labels")
    if not isinstance(raw_labels, list):
        raw_labels = []
        stats["invalid_labels_field"] += 1

    gold_items = []
    seen = set()
    stats["raw_labels"] += len(raw_labels)
    for label in raw_labels:
        if not isinstance(label, Mapping):
            stats["invalid_labels"] += 1
            continue
        head_index = label.get("h")
        tail_index = label.get("t")
        relation_id = str(label.get("r") or "").strip()
        if (
            not isinstance(head_index, int)
            or not isinstance(tail_index, int)
            or not (0 <= head_index < len(clusters))
            or not (0 <= tail_index < len(clusters))
            or head_index == tail_index
        ):
            stats["invalid_entity_references"] += 1
            continue
        if not clusters[head_index] or not clusters[tail_index]:
            stats["unusable_entity_clusters"] += 1
            continue
        predicate = clean_predicate(relation_map.get(relation_id))
        if predicate_validation_error(predicate):
            stats["unknown_or_invalid_relations"] += 1
            continue

        evidence_ids = _valid_evidence_ids(label.get("evidence"), len(sentences))
        head, tail = _select_mention_pair(clusters[head_index], clusters[tail_index], evidence_ids)
        triple = {
            "subject": head["surface"],
            "predicate": predicate,
            "object": tail["surface"],
        }
        key = _triple_key(triple)
        if key in seen:
            stats["duplicate_triples"] += 1
            continue
        evidence = _evidence_span(text, sentence_spans, head, tail, evidence_ids)
        if not evidence:
            stats["missing_evidence"] += 1
            continue
        seen.add(key)
        gold_items.append(
            {
                "triple": triple,
                "subject_span": dict(head),
                "object_span": dict(tail),
                "evidence": evidence,
                "relation_id": relation_id,
            }
        )

    entities = []
    entity_seen = set()
    for cluster in clusters:
        for mention in cluster:
            key = _normalized(mention["surface"])
            if key and key not in entity_seen:
                entity_seen.add(key)
                entities.append(mention["surface"])
    predicates = list(dict.fromkeys(item["triple"]["predicate"] for item in gold_items))
    stats["valid_triples"] += len(gold_items)
    document = {
        "document_id": source_document_id(split_name, index),
        "title": str(article.get("title") or ""),
        "text": text,
        "gold_items": gold_items,
        "entities": entities,
        "predicates": predicates,
    }
    return document, stats


def load_documents(
    path: Path,
    split_name: str,
    relation_map: Mapping[str, str],
) -> Tuple[List[Dict], Counter]:
    documents = []
    totals = Counter()
    for index, article in enumerate(load_split(path)):
        document, stats = convert_document(article, split_name, index, relation_map)
        documents.append(document)
        totals.update(stats)
    totals["documents"] = len(documents)
    return documents, totals
