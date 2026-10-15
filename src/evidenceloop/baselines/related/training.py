"""Optional PiVe preparation. No training executes during baseline inference."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from pathlib import Path

from evidenceloop.io import atomic_write_json, atomic_write_jsonl

from .common import ROOT, SOURCES, digest, read_json, read_jsonl, resolve_path
from .runtime import model_signature


def validate_training_rows(rows):
    unique = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"training row {index}: expected an object")
        required = ("instruction", "input", "output")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError(f"training row {index}: instruction, input and output are required")
        if "<S>" not in row["input"]:
            raise ValueError(f"training row {index}: missing PiVe <S> text/graph separator")
        cleaned = {k: row[k] for k in required}
        key = (cleaned["instruction"], cleaned["input"])
        if key in unique and unique[key]["output"] != cleaned["output"]:
            raise ValueError("conflicting labels for the same verifier input")
        unique[key] = cleaned
    return list(unique.values())


def load_official_training(path):
    # Pipeline directories contain evaluation sources, never verifier training data.
    pipeline = (ROOT / "data/pipeline").resolve()
    if path.resolve().is_relative_to(pipeline) or path.name in (
        "test.source",
        "test.target",
        "gold_triples.jsonl",
    ):
        raise ValueError("benchmark test data cannot be used as PiVe training data")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            json_names = [
                n
                for n in names
                if n.endswith(("combined_verifier_data.json", "train.json", "train.jsonl"))
            ]
            rows = []
            if json_names:
                for name in json_names:
                    text = archive.read(name).decode("utf-8")
                    rows.extend(
                        [json.loads(line) for line in text.splitlines() if line.strip()]
                        if name.endswith(".jsonl")
                        else json.loads(text)
                    )
            else:
                # Read paired training files in memory; no archive paths are extracted.
                for name in names:
                    if not name.endswith("train.source"):
                        continue
                    target = name[: -len(".source")] + ".target"
                    if target not in names:
                        raise ValueError(f"archive is missing {target}")
                    sources = archive.read(name).decode("utf-8").splitlines()
                    outputs = archive.read(target).decode("utf-8").splitlines()
                    if len(sources) != len(outputs):
                        raise ValueError("training source/target length mismatch")
                    domain = (
                        "WebNLG"
                        if "webnlg" in name.lower()
                        else ("GenWiki" if "genwiki" in name.lower() else "KELM")
                    )
                    instruction = (
                        f"Predict the missing triple given the text and graph for {domain} dataset."
                    )
                    rows.extend(
                        {"instruction": instruction, "input": x, "output": y}
                        for x, y in zip(sources, outputs)
                    )
            if not rows:
                raise ValueError("archive contains no recognized official training files")
    elif path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError("official training JSON must be an array")
    return validate_training_rows(rows)


def convert_training_data(source: Path, output: Path, *, validation_fraction=0.05, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    if output.exists() and any(output.iterdir()):
        raise ValueError("training data output directory must be empty")
    rows = load_official_training(source)
    # Group corruptions of the same source document to avoid train/validation leakage.
    groups = {}
    for row in rows:
        context = row["input"].split("<S>", 1)[0].strip()
        groups.setdefault(context, []).append(row)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(
            "at least two distinct source documents are required for a validation split"
        )
    random.Random(seed).shuffle(keys)
    validation_count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:validation_count])
    train = [row for key in keys if key not in validation_keys for row in groups[key]]
    validation = [row for key in keys if key in validation_keys for row in groups[key]]
    atomic_write_jsonl(output / "train.jsonl", train)
    atomic_write_jsonl(output / "validation.jsonl", validation)
    atomic_write_json(
        output / "training_data_manifest.json",
        {
            "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "upstream": SOURCES["pive"],
            "seed": seed,
            "validation_fraction": validation_fraction,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_sha256": digest(train),
            "validation_sha256": digest(validation),
            "split_unit": "source_document",
        },
    )
    print(f"PiVe training data: {output}; train={len(train)}, validation={len(validation)}")
    return train, validation


def train_pive(args):
    import torch
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    directory = resolve_path(args.train_pive_data)
    manifest = read_json(directory / "training_data_manifest.json")
    train, validation = (
        validate_training_rows(read_jsonl(directory / "train.jsonl")),
        validate_training_rows(read_jsonl(directory / "validation.jsonl")),
    )
    if (
        digest(train) != manifest["train_sha256"]
        or digest(validation) != manifest["validation_sha256"]
    ):
        raise ValueError("converted PiVe training data changed since preparation")
    train_contexts = {r["input"].split("<S>", 1)[0].strip() for r in train}
    if train_contexts & {r["input"].split("<S>", 1)[0].strip() for r in validation}:
        raise ValueError("PiVe training and validation source documents overlap")
    for name in (
        "epochs",
        "learning_rate",
        "train_batch_size",
        "gradient_accumulation",
        "max_source_length",
        "max_target_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    root = resolve_path(args.model_root)
    model_manifest = (
        read_json(root / "model_manifest.json") if (root / "model_manifest.json").exists() else {}
    )
    base = (
        resolve_path(args.base_model)
        if args.base_model
        else Path(
            model_manifest.get("artifacts", {}).get("pive_base", {}).get("path", root / "pive/base")
        )
    )
    base_signature = model_signature(base)
    if read_json(base / "config.json").get("model_type") != "t5":
        raise ValueError("PiVe training requires a T5 base model")
    output = resolve_path(args.training_output_dir)
    if output.exists() and any(output.iterdir()) and not args.training_resume_checkpoint:
        raise ValueError(
            "training output is not empty; use a new output directory or --training-resume-checkpoint"
        )
    set_seed(args.seed)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    kwargs = {"local_files_only": True, "trust_remote_code": False, "torch_dtype": dtype}
    if args.load_in_8bit:
        if not cuda:
            raise ValueError("8-bit PiVe training requires CUDA")
        kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": torch.cuda.current_device()},
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(str(base), **kwargs)
    tokenizer.add_tokens(["<S>"])
    model.resize_token_embeddings(len(tokenizer))
    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        ),
    )
    model.config.use_cache = False

    def tokenize(batch):
        prompts = [
            instruction + " " + text
            for instruction, text in zip(batch["instruction"], batch["input"])
        ]
        inputs = tokenizer(prompts, max_length=args.max_source_length, truncation=True)
        inputs["labels"] = tokenizer(
            text_target=batch["output"], max_length=args.max_target_length, truncation=True
        )["input_ids"]
        return inputs

    data = DatasetDict(train=Dataset.from_list(train), validation=Dataset.from_list(validation))
    tokenized = data.map(tokenize, batched=True, remove_columns=["instruction", "input", "output"])
    settings = Seq2SeqTrainingArguments(
        output_dir=str(output),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16,
        fp16=cuda and not bf16,
        seed=args.seed,
        report_to=[],
        push_to_hub=False,
        gradient_checkpointing=True,
        remove_unused_columns=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=settings,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    training_state = {
        "base": base_signature,
        "data_manifest": manifest,
        "upstream": SOURCES["pive"],
        "settings": settings.to_dict(),
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }
    if args.training_resume_checkpoint:
        if (
            not (output / "training_runtime.json").exists()
            or read_json(output / "training_runtime.json") != training_state
        ):
            raise ValueError("training resume data, model or settings changed")
    atomic_write_json(output / "training_runtime.json", training_state)
    trainer.train(resume_from_checkpoint=args.training_resume_checkpoint or None)
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_state()
    print(f"PiVe adapter: {output / 'adapter'}; pass this directory to --verifier-adapter")
