import json
import io
import shutil
import unittest
import uuid
import zipfile
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from evidenceloop.data.builders.legacy_phase1 import triple_key, valid_gold_triples
from evidenceloop.data.builders.rebel_sft import (
    ACTION_RATIOS,
    NMI_THRESHOLD,
    build_dataset,
    normalized_mutual_information,
    validate_phase2_record,
)
from evidenceloop.training.evaluate import (
    _EvaluationProgress,
    _format_duration,
    evaluate_prediction_rows,
    parse_json_object,
    prediction_row_from_raw,
)
from evidenceloop.training.train import encode_training_record, main as training_main
from evidenceloop.data.sft import format_critic_sft_parts, iter_jsonl


def entity(text, surface):
    start = text.index(surface)
    return {
        "uri": f"entity:{surface}",
        "boundaries": [start, start + len(surface)],
        "surfaceform": surface,
    }


def variable_article(split, index):
    triple_count = index % 4 + 1
    facts = []
    triple_values = []
    entity_surfaces = []
    for fact_index in range(triple_count):
        subject = f"Alpha Entity {split} {index} {fact_index}"
        obj = f"Beta Organization {split} {index} {fact_index}"
        predicate = f"relation type {index} {fact_index}"
        facts.append(f"{subject} works with {obj}.")
        triple_values.append((subject, predicate, obj))
        entity_surfaces.extend((subject, obj))
    text = " ".join(facts)
    entities = {surface: entity(text, surface) for surface in entity_surfaces}
    triples = [
        {
            "subject": entities[subject],
            "predicate": {"surfaceform": predicate},
            "object": entities[obj],
        }
        for subject, predicate, obj in triple_values
    ]
    return {
        "docid": f"doc-{split}-{index}",
        "title": f"Document {index}",
        "uri": f"uri:{split}:{index}",
        "text": text,
        "entities": list(entities.values()),
        "triples": triples,
    }


def write_variable_zip(path, count=64):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, split in (
            ("en_train.jsonl", "train"),
            ("en_val.jsonl", "validation"),
            ("en_test.jsonl", "test"),
        ):
            payload = "".join(
                json.dumps(variable_article(split, index)) + "\n"
                for index in range(count)
            )
            archive.writestr(member, payload)


@contextmanager
def workspace_temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"critic_sft_phase2_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakeTokenizer:
    eos_token_id = 3

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.enable_thinking = enable_thinking
        value = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            value += "<assistant>"
        return [ord(character) for character in value]

    def encode(self, value, *, add_special_tokens):
        self.add_special_tokens = add_special_tokens
        return [ord(character) for character in value]


class CriticSFTPhase2Tests(unittest.TestCase):
    def test_evaluation_progress_tracks_records_and_can_be_disabled(self):
        progress = _EvaluationProgress(5, enabled=False)
        progress.update(4)
        progress.update(1)
        self.assertEqual(progress.completed, 0)

        output = io.StringIO()
        with redirect_stderr(output):
            progress = _EvaluationProgress(5, enabled=True)
            progress.update(2)
            progress.update(2)
            progress.update(1)
            progress.close()
        self.assertEqual(progress.completed, 5)
        self.assertEqual(_format_duration(3661), "1h01m01s")
        self.assertIn("5/5", output.getvalue())

    def test_evaluation_progress_falls_back_without_tqdm(self):
        real_import = __import__

        def import_without_tqdm(name, *args, **kwargs):
            if name.startswith("tqdm"):
                raise ImportError("tqdm unavailable")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with patch("builtins.__import__", side_effect=import_without_tqdm):
            with redirect_stderr(output):
                progress = _EvaluationProgress(3)
                progress.update(2)
                progress.update(1)
                progress.close()
        self.assertEqual(progress.completed, 3)
        self.assertIn("[eval] 2/3 records", output.getvalue())
        self.assertIn("[eval] 3/3 records", output.getvalue())

    def test_variable_batches_are_reproducible_and_use_official_splits(self):
        with workspace_temporary_directory() as root:
            source_zip = root / "rebel.zip"
            write_variable_zip(source_zip)
            first = root / "first"
            second = root / "second"
            different_seed = root / "different_seed"

            metadata = build_dataset(source_zip, first, 40, 8, 8, seed=42)
            build_dataset(source_zip, second, 40, 8, 8, seed=42)
            build_dataset(source_zip, different_seed, 40, 8, 8, seed=43)

            self.assertEqual(metadata["source_members"]["test"], "en_test.jsonl")
            self.assertEqual(metadata["candidate_range"], [1, 4])
            self.assertLess(metadata["splits"]["train"]["max_abs_action_ratio_error"], 0.015)
            self.assertEqual(
                set(metadata["splits"]["train"]["batch_size_counts"]),
                {"1", "2", "3", "4"},
            )
            self.assertLess(metadata["splits"]["train"]["position_action_nmi"], 0.10)

            for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertNotEqual(
                (first / "train.jsonl").read_bytes(),
                (different_seed / "train.jsonl").read_bytes(),
            )

            split_ids = []
            for split in ("train", "validation", "test"):
                rows = list(iter_jsonl(first / f"{split}.jsonl"))
                split_ids.append({row["document_id"] for row in rows})
                for row in rows:
                    validate_phase2_record(row)
                    self.assertIn(len(row["triples_info"]), {1, 2, 3, 4})
                    keys = [triple_key(item["triple"]) for item in row["triples_info"]]
                    self.assertEqual(len(keys), len(set(keys)))
                expected_split = "validation" if split == "validation" else split
                self.assertTrue(
                    all(f"doc-{expected_split}-" in document_id for document_id in split_ids[-1])
                )
            self.assertFalse(split_ids[0] & split_ids[1])
            self.assertFalse(split_ids[0] & split_ids[2])
            self.assertFalse(split_ids[1] & split_ids[2])

    def test_delete_candidates_are_not_gold_and_positions_do_not_encode_actions(self):
        with workspace_temporary_directory() as root:
            source_zip = root / "rebel.zip"
            write_variable_zip(source_zip)
            output = root / "output"
            metadata = build_dataset(source_zip, output, 40, 8, 8, seed=42)

            articles = {
                f"rebel_{article['docid']}": article
                for index in range(40)
                for article in [variable_article("train", index)]
            }
            for record in iter_jsonl(output / "train.jsonl"):
                gold_keys = {
                    triple_key(item["triple"])
                    for item in valid_gold_triples(articles[record["document_id"]])
                }
                for candidate, result in zip(
                    record["triples_info"], record["target"]["results"]
                ):
                    if result["action"] == "DELETE":
                        self.assertNotIn(triple_key(candidate["triple"]), gold_keys)

            by_position = metadata["splits"]["train"]["position_action_counts"]
            self.assertGreater(
                len([count for count in by_position["0"].values() if count]),
                1,
            )
            balanced = {
                position: {action: 10 for action in ACTION_RATIOS}
                for position in range(4)
            }
            self.assertLess(normalized_mutual_information(balanced), NMI_THRESHOLD)

    def test_variable_batch_prompt_and_assistant_only_mask(self):
        with workspace_temporary_directory() as root:
            source_zip = root / "rebel.zip"
            write_variable_zip(source_zip)
            output = root / "output"
            build_dataset(source_zip, output, 4, 4, 4, seed=42)
            records = list(iter_jsonl(output / "train.jsonl"))
            record = next(item for item in records if len(item["triples_info"]) == 3)

            parts = format_critic_sft_parts(
                record,
                "SYSTEM",
                "Stage={review_stage}\nText={source_text}\nTriples={triples_info}",
            )
            self.assertNotIn("generation_type", parts["user"])
            self.assertNotIn("corruption", parts["user"])

            tokenizer = FakeTokenizer()
            encoded = encode_training_record(
                record,
                tokenizer,
                "SYSTEM",
                "Stage={review_stage}\nText={source_text}\nTriples={triples_info}",
                max_seq_length=100_000,
            )
            first_target = next(
                index for index, label in enumerate(encoded["labels"]) if label != -100
            )
            self.assertTrue(all(label == -100 for label in encoded["labels"][:first_target]))
            self.assertEqual(encoded["labels"][first_target:], encoded["input_ids"][first_target:])
            self.assertFalse(tokenizer.enable_thinking)

            self.assertEqual(
                training_main(
                    [
                        "--train-file",
                        str(output / "train.jsonl"),
                        "--validation-file",
                        str(output / "validation.jsonl"),
                        "--validate-only",
                    ]
                ),
                0,
            )

    def test_offline_evaluation_metrics(self):
        with workspace_temporary_directory() as root:
            source_zip = root / "rebel.zip"
            write_variable_zip(source_zip)
            output = root / "output"
            build_dataset(source_zip, output, 4, 4, 4, seed=42)
            records = list(iter_jsonl(output / "test.jsonl"))
            prediction_rows = [
                {
                    "document_id": record["document_id"],
                    "parsed_target": record["target"],
                }
                for record in records
            ]
            summary = evaluate_prediction_rows(records, prediction_rows)
            self.assertEqual(summary["json_valid_rate"], 1.0)
            self.assertEqual(summary["schema_valid_rate"], 1.0)
            self.assertEqual(summary["action"]["accuracy"], 1.0)
            self.assertEqual(summary["action"]["macro_f1"], 1.0)
            self.assertEqual(summary["corrected_triple_exact_match"]["rate"], 1.0)
            self.assertEqual(summary["evidence_validity"]["rate"], 1.0)
            self.assertEqual(summary["keep_overcorrection"]["rate"], 0.0)

            raw = "prefix\n" + json.dumps(records[0]["target"]) + "\nsuffix"
            self.assertIn("results", parse_json_object(raw))
            parsed_row = prediction_row_from_raw(records[0], raw)
            self.assertTrue(parsed_row["json_valid"])
            self.assertTrue(parsed_row["schema_valid"])

            invalid_rows = list(prediction_rows)
            invalid_rows[0] = {
                "document_id": records[0]["document_id"],
                "raw_output": "not json",
            }
            invalid_summary = evaluate_prediction_rows(records, invalid_rows)
            self.assertLess(invalid_summary["json_valid_rate"], 1.0)
            self.assertLess(invalid_summary["action"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
