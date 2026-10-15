import json
import shutil
import unittest
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from evidenceloop.data.builders.legacy_phase1 import (
    build_dataset,
    build_record,
    triple_key,
    valid_gold_triples,
)
from evidenceloop.training.train import (
    DEFAULT_MODEL_PATH,
    TrainingSetupError,
    encode_training_record,
    main as training_main,
    model_parameter_stats,
    parse_args as parse_training_args,
    validate_local_model_path,
)
from evidenceloop.data.sft import (
    format_critic_sft_parts,
    iter_jsonl,
    validate_critic_sft_jsonl,
    validate_critic_sft_record,
)


def entity(text, surface):
    start = text.index(surface)
    return {
        "uri": f"entity:{surface}",
        "boundaries": [start, start + len(surface)],
        "surfaceform": surface,
    }


def article(split, index):
    alpha = f"Alpha Entity {index}"
    beta = f"Beta Organization {index}"
    gamma = f"Gamma Person {index}"
    delta = f"Delta Company {index}"
    text = f"{alpha} works for {beta}. {gamma} founded {delta}."
    alpha_value = entity(text, alpha)
    beta_value = entity(text, beta)
    gamma_value = entity(text, gamma)
    delta_value = entity(text, delta)
    return {
        "docid": f"doc-{split}-{index}",
        "title": f"Document {index}",
        "uri": f"uri:{split}:{index}",
        "text": text,
        "entities": [alpha_value, beta_value, gamma_value, delta_value],
        "triples": [
            {
                "subject": alpha_value,
                "predicate": {"surfaceform": "employer"},
                "object": beta_value,
            },
            {
                "subject": gamma_value,
                "predicate": {"surfaceform": "founded"},
                "object": delta_value,
            },
        ],
    }


def write_fake_zip(path):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, split, count in (
            ("en_train.jsonl", "train", 8),
            ("en_val.jsonl", "validation", 8),
            ("en_test.jsonl", "test", 2),
        ):
            payload = "".join(
                json.dumps(article(split, index)) + "\n"
                for index in range(count)
            )
            archive.writestr(member, payload)


@contextmanager
def workspace_temporary_directory():
    root = Path.cwd() / ".test_runtime_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"critic_sft_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakeTokenizer:
    eos_token_id = 3

    def __init__(self, return_mapping=False):
        self.return_mapping = return_mapping

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.last_enable_thinking = enable_thinking
        value = ""
        for message in messages:
            value += f"<{message['role']}>{message['content']}</{message['role']}>"
        if add_generation_prompt:
            value += "<assistant>"
        input_ids = [ord(char) for char in value]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)} if self.return_mapping else input_ids

    def encode(self, value, *, add_special_tokens):
        self.last_add_special_tokens = add_special_tokens
        return [ord(char) for char in value]


class FakeParameter:
    def __init__(self, size, requires_grad):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class FakeModel:
    def parameters(self):
        return [FakeParameter(90, False), FakeParameter(10, True)]


class CriticSFTDatasetTests(unittest.TestCase):
    def test_record_contains_four_valid_actions_and_delete_is_not_gold(self):
        source = article("train", 0)
        record, error = build_record(
            source,
            "train",
            0,
            __import__("random").Random(42),
            [],
            [],
        )

        self.assertEqual(error, "")
        self.assertIsNotNone(record)
        validate_critic_sft_record(record)
        actions = [item["action"] for item in record["target"]["results"]]
        self.assertEqual(actions, ["KEEP", "DELETE", "FIX", "REWRITE"])

        gold_keys = {triple_key(item["triple"]) for item in valid_gold_triples(source)}
        delete_triple = record["triples_info"][1]["triple"]
        self.assertNotIn(triple_key(delete_triple), gold_keys)
        for result in record["target"]["results"]:
            if result["action"] != "DELETE":
                self.assertIn(result["evidence"], record["source_text"])

    def test_full_build_is_reproducible_and_splits_are_disjoint(self):
        with workspace_temporary_directory() as root:
            source_zip = root / "rebel.zip"
            write_fake_zip(source_zip)
            first = root / "first"
            second = root / "second"

            build_dataset(source_zip, first, 2, 1, 1, seed=42)
            build_dataset(source_zip, second, 2, 1, 1, seed=42)

            for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )
                summary = validate_critic_sft_jsonl(first / name)
                self.assertEqual(summary["candidates"], summary["records"] * 4)

            split_ids = []
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                ids = {row["document_id"] for row in iter_jsonl(first / name)}
                split_ids.append(ids)
            self.assertFalse(split_ids[0] & split_ids[1])
            self.assertFalse(split_ids[0] & split_ids[2])
            self.assertFalse(split_ids[1] & split_ids[2])

            metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["uses_graphjudge_test_files"])
            self.assertEqual(metadata["splits"]["train"]["records"], 2)
            self.assertEqual(metadata["validation"]["train"]["path"], str(first / "train.jsonl"))

    def test_training_prompt_hides_generation_metadata(self):
        record, _ = build_record(
            article("train", 1),
            "train",
            1,
            __import__("random").Random(7),
            [],
            [],
        )
        parts = format_critic_sft_parts(
            record,
            "SYSTEM",
            "Stage={review_stage}\nText={source_text}\nTriples={triples_info}",
        )

        self.assertNotIn("generation_type", parts["user"])
        self.assertNotIn("corruption", parts["user"])
        self.assertIn('"results"', parts["assistant"])

    def test_assistant_only_mask_and_validate_only_need_no_training_packages(self):
        record, _ = build_record(
            article("train", 2),
            "train",
            2,
            __import__("random").Random(9),
            [],
            [],
        )
        tokenizer = FakeTokenizer()
        encoded = encode_training_record(
            record,
            tokenizer,
            "SYSTEM",
            "Stage={review_stage}\nText={source_text}\nTriples={triples_info}",
            max_seq_length=100_000,
        )

        first_target = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
        self.assertTrue(all(label == -100 for label in encoded["labels"][:first_target]))
        self.assertEqual(encoded["labels"][first_target:], encoded["input_ids"][first_target:])
        self.assertFalse(tokenizer.last_enable_thinking)
        self.assertFalse(tokenizer.last_add_special_tokens)
        self.assertEqual(encoded["input_ids"][-1], tokenizer.eos_token_id)

        mapping_tokenizer = FakeTokenizer(return_mapping=True)
        mapping_encoded = encode_training_record(
            record,
            mapping_tokenizer,
            "SYSTEM",
            "Stage={review_stage}\nText={source_text}\nTriples={triples_info}",
            max_seq_length=100_000,
        )
        self.assertEqual(mapping_encoded["input_ids"], encoded["input_ids"])

        with workspace_temporary_directory() as root:
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            line = json.dumps(record, ensure_ascii=False) + "\n"
            train_path.write_text(line, encoding="utf-8")
            validation_path.write_text(line.replace("rebel_train_", "rebel_validation_"), encoding="utf-8")
            exit_code = training_main([
                "--train-file", str(train_path),
                "--validation-file", str(validation_path),
                "--validate-only",
            ])
            self.assertEqual(exit_code, 0)

    def test_training_defaults_to_server_model_and_requires_local_assets(self):
        args = parse_training_args([])
        self.assertEqual(args.model_name_or_path, DEFAULT_MODEL_PATH)
        self.assertEqual(DEFAULT_MODEL_PATH.as_posix(), "models/Qwen3-8B")

        with workspace_temporary_directory() as model_dir:
            with self.assertRaises(TrainingSetupError):
                validate_local_model_path(model_dir)
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "model-00001-of-00001.safetensors",
            ):
                (model_dir / name).write_text("{}", encoding="utf-8")
            validate_local_model_path(model_dir)

    def test_training_and_evaluation_use_standard_bf16_lora(self):
        train_source = Path(__import__("evidenceloop.training.train", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
        eval_source = Path(__import__("evidenceloop.training.evaluate", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
        readme = Path("docs/reproduction.md").read_text(encoding="utf-8")
        for source in (train_source, eval_source):
            self.assertIn("torch_dtype=torch.bfloat16", source)
            for forbidden in (
                "BitsAndBytesConfig",
                "load_in_4bit",
                "bnb_4bit",
                "prepare_model_for_kbit_training",
                "paged_adamw_8bit",
            ):
                self.assertNotIn(forbidden, source)
        self.assertIn('"optim": "adamw_torch"', train_source)
        self.assertIn("model.enable_input_require_grads()", train_source)
        self.assertNotIn('"bitsandbytes>=', readme)

        stats = model_parameter_stats(FakeModel())
        self.assertEqual(stats["trainable"], 10)
        self.assertEqual(stats["total"], 100)
        self.assertEqual(stats["trainable_percent"], 10.0)


if __name__ == "__main__":
    unittest.main()
