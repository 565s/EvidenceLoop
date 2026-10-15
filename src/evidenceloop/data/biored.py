from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from evidenceloop.validation import clean_predicate, predicate_validation_error

BIORED_SOURCE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
BIORED_GUIDELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BioRED_Annotation_Guideline.pdf"
SOURCE_MEMBERS = {
    "train": "BioRED/Train.BioC.JSON",
    "validation": "BioRED/Dev.BioC.JSON",
    "test": "BioRED/Test.BioC.JSON",
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


class BioREDError(RuntimeError):
    pass


def _normalized(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def undirected_triple_key(triple: Mapping[str, str]) -> Tuple[str, str, str]:
    endpoints = sorted((_normalized(triple.get("subject")), _normalized(triple.get("object"))))
    return endpoints[0], _normalized(triple.get("predicate")), endpoints[1]


def load_relation_map(path: Path) -> Dict[str, str]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BioREDError(f"{path}: invalid relation-map JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping) or not value:
        raise BioREDError(f"{path}: expected a non-empty relation mapping")
    output = {}
    for raw_key, raw_label in value.items():
        key = str(raw_key or "").strip()
        label = clean_predicate(raw_label)
        if not key or predicate_validation_error(label):
            raise BioREDError(f"{path}: invalid relation mapping {raw_key!r}: {raw_label!r}")
        output[key] = label
    return output


def load_collection(archive: zipfile.ZipFile, member: str) -> List[Mapping]:
    try:
        with archive.open(member) as handle:
            value = json.load(handle)
    except KeyError as exc:
        raise BioREDError(f"archive is missing {member}") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("documents"), list):
        raise BioREDError(f"{member}: expected a BioC collection with documents")
    documents = value["documents"]
    if not documents:
        raise BioREDError(f"{member}: collection is empty")
    return documents


def _document_text(raw_passages) -> Tuple[str, List[Tuple[int, int]], Counter]:
    stats = Counter()
    if not isinstance(raw_passages, list) or not raw_passages:
        raise BioREDError("document.passages: expected a non-empty list")
    passages = []
    for index, passage in enumerate(raw_passages):
        if not isinstance(passage, Mapping):
            raise BioREDError(f"document.passages[{index}]: expected an object")
        offset = passage.get("offset")
        text = passage.get("text")
        if not isinstance(offset, int) or offset < 0 or not isinstance(text, str):
            raise BioREDError(f"document.passages[{index}]: invalid offset or text")
        passages.append((offset, text, passage))
    passages.sort(key=lambda item: item[0])
    if passages[0][0] != 0:
        raise BioREDError("document.passages: first passage must start at offset 0")

    chunks = []
    spans = []
    cursor = 0
    for offset, passage_text, _ in passages:
        if offset < cursor:
            raise BioREDError("document.passages: overlapping passage offsets")
        gap = offset - cursor
        if gap:
            chunks.append("\n" if gap == 1 else " " * gap)
            if gap != 1:
                stats["nonstandard_passage_gaps"] += 1
        chunks.append(passage_text)
        spans.append((offset, offset + len(passage_text)))
        cursor = offset + len(passage_text)
    return "".join(chunks), spans, stats


def _sentence_spans(text: str, passage_spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    spans = []
    for passage_start, passage_end in passage_spans:
        passage_text = text[passage_start:passage_end]
        relative_start = 0
        for match in SENTENCE_BOUNDARY.finditer(passage_text):
            relative_end = match.start()
            start = passage_start + relative_start
            end = passage_start + relative_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
            relative_start = match.end()
        start = passage_start + relative_start
        end = passage_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _identifier_parts(value) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _mentions_by_identifier(
    raw_passages,
    text: str,
) -> Tuple[Dict[str, List[Dict]], Counter]:
    mentions = defaultdict(list)
    stats = Counter()
    for passage in raw_passages:
        annotations = passage.get("annotations") if isinstance(passage, Mapping) else None
        if not isinstance(annotations, list):
            stats["invalid_annotations_fields"] += 1
            continue
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                stats["invalid_annotations"] += 1
                continue
            infons = annotation.get("infons")
            locations = annotation.get("locations")
            identifiers = _identifier_parts(
                infons.get("identifier") if isinstance(infons, Mapping) else None
            )
            if not identifiers or not isinstance(locations, list) or len(locations) != 1:
                stats["invalid_annotations"] += 1
                continue
            location = locations[0]
            if not isinstance(location, Mapping):
                stats["invalid_annotations"] += 1
                continue
            start = location.get("offset")
            length = location.get("length")
            if (
                not isinstance(start, int)
                or not isinstance(length, int)
                or length <= 0
                or not (0 <= start < start + length <= len(text))
            ):
                stats["invalid_annotation_offsets"] += 1
                continue
            end = start + length
            surface = text[start:end]
            annotated_surface = str(annotation.get("text") or "")
            if surface != annotated_surface or not surface.strip():
                stats["annotation_text_mismatches"] += 1
                continue
            mention = {
                "surface": surface,
                "start": start,
                "end": end,
                "entity_type": str(infons.get("type") or "").strip(),
                "identifier_count": len(identifiers),
            }
            for identifier in identifiers:
                mentions[identifier].append(mention)
            stats["valid_annotations"] += 1
    return dict(mentions), stats


def _sentence_index(
    mention: Mapping,
    sentence_spans: Sequence[Tuple[int, int]],
) -> int | None:
    for index, (start, end) in enumerate(sentence_spans):
        if start <= mention["start"] and mention["end"] <= end:
            return index
    return None


def _mention_pair_and_evidence(
    heads: Sequence[Mapping],
    tails: Sequence[Mapping],
    text: str,
    sentence_spans: Sequence[Tuple[int, int]],
    predicate: str,
    seen_triple_keys: set,
) -> Tuple[Mapping, Mapping, str]:
    choices = []
    for head in heads:
        for tail in tails:
            head_sentence = _sentence_index(head, sentence_spans)
            tail_sentence = _sentence_index(tail, sentence_spans)
            if head_sentence is not None and tail_sentence is not None:
                first_sentence = min(head_sentence, tail_sentence)
                last_sentence = max(head_sentence, tail_sentence)
                evidence_start = sentence_spans[first_sentence][0]
                evidence_end = sentence_spans[last_sentence][1]
                sentence_distance = last_sentence - first_sentence
            else:
                evidence_start = min(head["start"], tail["start"])
                evidence_end = max(head["end"], tail["end"])
                sentence_distance = len(sentence_spans) + 1
            choices.append(
                (
                    (
                        head.get("identifier_count", 1) + tail.get("identifier_count", 1),
                        sentence_distance,
                        evidence_end - evidence_start,
                        abs(head["start"] - tail["start"]),
                        head["start"],
                        tail["start"],
                    ),
                    head,
                    tail,
                    evidence_start,
                    evidence_end,
                )
            )
    if not choices:
        raise BioREDError("cannot select mentions for relation")
    for _, head, tail, start, end in sorted(choices, key=lambda item: item[0]):
        candidate_key = undirected_triple_key(
            {
                "subject": head["surface"],
                "predicate": predicate,
                "object": tail["surface"],
            }
        )
        if candidate_key in seen_triple_keys:
            continue
        evidence = text[start:end].strip()
        if (
            evidence
            and evidence in text
            and head["surface"] in evidence
            and tail["surface"] in evidence
        ):
            return head, tail, evidence
    raise BioREDError("cannot construct a unique verbatim relation evidence span")


def source_document_id(split_name: str, pmid: str) -> str:
    return f"biored_{split_name}_{pmid}"


def convert_document(
    raw_document: Mapping,
    split_name: str,
    relation_map: Mapping[str, str],
) -> Tuple[Dict, Counter]:
    if not isinstance(raw_document, Mapping):
        raise BioREDError("document: expected an object")
    pmid = str(raw_document.get("id") or "").strip()
    if not pmid:
        raise BioREDError("document.id: expected a non-empty PMID")
    raw_passages = raw_document.get("passages")
    text, passage_spans, stats = _document_text(raw_passages)
    sentence_spans = _sentence_spans(text, passage_spans)
    mentions_by_id, mention_stats = _mentions_by_identifier(raw_passages, text)
    stats.update(mention_stats)

    raw_relations = raw_document.get("relations")
    if not isinstance(raw_relations, list):
        raw_relations = []
        stats["invalid_relations_field"] += 1
    stats["raw_relations"] += len(raw_relations)

    def relation_choice_count(index_and_relation) -> Tuple[int, int]:
        index, raw_relation = index_and_relation
        infons = raw_relation.get("infons") if isinstance(raw_relation, Mapping) else None
        if not isinstance(infons, Mapping):
            return 10**9, index
        entity1 = str(infons.get("entity1") or "").strip()
        entity2 = str(infons.get("entity2") or "").strip()
        predicate = clean_predicate(relation_map.get(str(infons.get("type") or "").strip()))
        if (
            predicate_validation_error(predicate)
            or entity1 not in mentions_by_id
            or entity2 not in mentions_by_id
        ):
            return 10**9, index
        choices = {
            undirected_triple_key(
                {
                    "subject": head["surface"],
                    "predicate": predicate,
                    "object": tail["surface"],
                }
            )
            for head in mentions_by_id[entity1]
            for tail in mentions_by_id[entity2]
        }
        return len(choices), index

    ordered_relations = [
        relation for _, relation in sorted(enumerate(raw_relations), key=relation_choice_count)
    ]
    gold_items = []
    seen = set()
    for raw_relation in ordered_relations:
        if not isinstance(raw_relation, Mapping) or not isinstance(
            raw_relation.get("infons"), Mapping
        ):
            stats["invalid_relations"] += 1
            continue
        infons = raw_relation["infons"]
        entity1 = str(infons.get("entity1") or "").strip()
        entity2 = str(infons.get("entity2") or "").strip()
        relation_type = str(infons.get("type") or "").strip()
        novelty = str(infons.get("novel") or "").strip()
        predicate = clean_predicate(relation_map.get(relation_type))
        if not entity1 or not entity2 or entity1 == entity2:
            stats["invalid_relation_entities"] += 1
            continue
        if predicate_validation_error(predicate):
            stats["unknown_or_invalid_relations"] += 1
            continue
        if entity1 not in mentions_by_id or entity2 not in mentions_by_id:
            stats["unresolved_relation_entities"] += 1
            continue
        try:
            head, tail, evidence = _mention_pair_and_evidence(
                mentions_by_id[entity1],
                mentions_by_id[entity2],
                text,
                sentence_spans,
                predicate,
                seen,
            )
        except BioREDError:
            stats["unrepresentable_duplicate_or_missing_evidence"] += 1
            continue
        triple = {
            "subject": head["surface"],
            "predicate": predicate,
            "object": tail["surface"],
        }
        key = undirected_triple_key(triple)
        seen.add(key)
        stats[f"relation_type_{relation_type}"] += 1
        stats[f"novelty_{novelty or 'missing'}"] += 1
        gold_items.append(
            {
                "triple": triple,
                "subject_span": dict(head),
                "object_span": dict(tail),
                "evidence": evidence,
                "entity1_id": entity1,
                "entity2_id": entity2,
                "relation_type": relation_type,
                "novelty": novelty,
            }
        )

    entities = []
    entity_seen = set()
    for identifier in mentions_by_id:
        for mention in mentions_by_id[identifier]:
            key = _normalized(mention["surface"])
            if key and key not in entity_seen:
                entity_seen.add(key)
                entities.append(mention["surface"])
    predicates = list(dict.fromkeys(item["triple"]["predicate"] for item in gold_items))
    stats["valid_triples"] += len(gold_items)
    return {
        "document_id": source_document_id(split_name, pmid),
        "pmid": pmid,
        "text": text,
        "gold_items": gold_items,
        "entities": entities,
        "predicates": predicates,
    }, stats


def load_documents(
    source_zip: Path,
    split_name: str,
    relation_map: Mapping[str, str],
) -> Tuple[List[Dict], Counter]:
    if split_name not in SOURCE_MEMBERS:
        raise ValueError(f"unknown BioRED split: {split_name}")
    documents = []
    totals = Counter()
    seen_pmids = set()
    with zipfile.ZipFile(Path(source_zip)) as archive:
        for raw_document in load_collection(archive, SOURCE_MEMBERS[split_name]):
            document, stats = convert_document(raw_document, split_name, relation_map)
            if document["pmid"] in seen_pmids:
                raise BioREDError(
                    f"{SOURCE_MEMBERS[split_name]}: duplicate PMID {document['pmid']}"
                )
            seen_pmids.add(document["pmid"])
            documents.append(document)
            totals.update(stats)
    totals["documents"] = len(documents)
    return documents, totals
