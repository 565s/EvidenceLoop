from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import random
import sys
from importlib.resources import files
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from evidenceloop.data.sft import (  # noqa: E402
    CriticSFTDataError,
    format_critic_sft_parts,
    validate_critic_sft_jsonl,
)

ROOT = Path.cwd()


DEFAULT_MODEL_PATH = Path("models/Qwen3-8B")
DEFAULT_TRAIN_FILE = ROOT / "data" / "critic_sft" / "rebel_phase1" / "train.jsonl"
DEFAULT_VALIDATION_FILE = ROOT / "data" / "critic_sft" / "rebel_phase1" / "validation.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "critic_qwen3_8b_lora"
DEFAULT_SYSTEM_PROMPT = Path(
    str(files("evidenceloop.resources") / "prompts" / "critic_system_prompt.txt")
)
DEFAULT_CALL_PROMPT = Path(
    str(files("evidenceloop.resources") / "prompts" / "critic_call_prompt_batch.txt")
)
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TrainingSetupError(RuntimeError):
    pass


def read_prompt(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise TrainingSetupError(f"prompt file is empty: {path}")
    return value


def validate_local_model_path(path: Path) -> None:
    path = Path(path)
    if not path.is_dir():
        raise TrainingSetupError(f"local model directory does not exist: {path}")
    required = [path / "config.json", path / "tokenizer_config.json"]
    missing = [str(item) for item in required if not item.is_file()]
    tokenizer_files = [
        candidate
        for candidate in (path / "tokenizer.json", path / "vocab.json", path / "vocab.txt")
        if candidate.is_file()
    ]
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not tokenizer_files:
        missing.append("tokenizer.json or vocab file")
    if not weight_files:
        missing.append("*.safetensors or pytorch_model*.bin")
    if missing:
        raise TrainingSetupError(
            f"local model directory is incomplete ({path}): missing {', '.join(missing)}"
        )


def validate_data_files(train_file: Path, validation_file: Path) -> Dict:
    train_summary = validate_critic_sft_jsonl(Path(train_file))
    validation_summary = validate_critic_sft_jsonl(Path(validation_file))
    return {"train": train_summary, "validation": validation_summary}


def training_parts(record: Mapping, system_prompt: str, call_prompt: str) -> Dict[str, str]:
    return format_critic_sft_parts(record, system_prompt, call_prompt)


def chat_template_input_ids(value) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TrainingSetupError("chat template output does not contain input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise TrainingSetupError("chat template unexpectedly returned a batch")
        value = list(value[0])
    if not value or not all(isinstance(token_id, int) for token_id in value):
        raise TrainingSetupError("chat template returned invalid input_ids")
    return value


def encode_training_record(
    record: Mapping,
    tokenizer,
    system_prompt: str,
    call_prompt: str,
    max_seq_length: int,
) -> Dict:
    parts = training_parts(record, system_prompt, call_prompt)
    prompt_messages = [
        {"role": "system", "content": parts["system"]},
        {"role": "user", "content": parts["user"]},
    ]
    template_kwargs = {"enable_thinking": False}
    prompt_output = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    prompt_ids = chat_template_input_ids(prompt_output)
    assistant_ids = list(tokenizer.encode(parts["assistant"], add_special_tokens=False))
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise TrainingSetupError("tokenizer does not define eos_token_id")
    assistant_ids.append(int(eos_token_id))
    if len(assistant_ids) <= 1:
        raise TrainingSetupError("tokenizer produced no assistant JSON target tokens")
    full_ids = prompt_ids + assistant_ids
    assistant_tokens = len(assistant_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
        "sequence_length": len(full_ids),
        "assistant_tokens": assistant_tokens,
        "overlength": len(full_ids) > max_seq_length,
    }


def build_training_arguments(
    training_arguments_class,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    eval_steps: int,
    max_steps: int,
):
    parameters = inspect.signature(training_arguments_class.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 1.0,
        "max_steps": max_steps,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": eval_steps,
        "logging_steps": 1 if args.smoke_test else 10,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "steps"
    else:
        raise TrainingSetupError(
            "TrainingArguments supports neither eval_strategy nor evaluation_strategy"
        )

    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in parameters:
        kwargs["warmup_steps"] = 0.03
    else:
        raise TrainingSetupError("TrainingArguments supports neither warmup_ratio nor warmup_steps")

    optional_values = {
        "tf32": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "data_seed": args.seed,
    }
    for name, value in optional_values.items():
        if name in parameters:
            kwargs[name] = value

    unsupported = sorted(name for name in kwargs if name not in parameters)
    if unsupported:
        version = dependency_versions(["transformers"])["transformers"]
        raise TrainingSetupError(
            f"transformers {version} has incompatible TrainingArguments fields: {unsupported}"
        )
    return training_arguments_class(**kwargs)


def dependency_versions(packages: Sequence[str]) -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def model_parameter_stats(model) -> Dict[str, float]:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }


def _select_limit(dataset, limit: Optional[int]):
    if limit is None:
        return dataset
    return dataset.select(range(min(len(dataset), int(limit))))


def run_training(args: argparse.Namespace) -> Dict:
    validate_local_model_path(args.model_name_or_path)
    if not args.train_file.is_file():
        raise FileNotFoundError(args.train_file)
    if not args.validation_file.is_file():
        raise FileNotFoundError(args.validation_file)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite_output_dir and not args.resume_from_checkpoint:
            raise TrainingSetupError(
                f"output directory is not empty: {output_dir}; use --overwrite-output-dir or resume"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise TrainingSetupError(
            "training dependencies are missing; follow training/README.md on the GPU server"
        ) from exc

    set_seed(args.seed)
    random.seed(args.seed)
    system_prompt = read_prompt(args.system_prompt)
    call_prompt = read_prompt(args.call_prompt)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {
        "train": str(args.train_file.resolve()),
        "validation": str(args.validation_file.resolve()),
    }
    datasets = load_dataset("json", data_files=data_files)
    train_dataset = _select_limit(datasets["train"], args.max_train_samples)
    eval_dataset = _select_limit(datasets["validation"], args.max_eval_samples)
    if args.smoke_test:
        train_dataset = _select_limit(train_dataset, 32)
        eval_dataset = _select_limit(eval_dataset, 16)

    def tokenize_record(record):
        return encode_training_record(
            record,
            tokenizer,
            system_prompt,
            call_prompt,
            args.max_seq_length,
        )

    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(tokenize_record, remove_columns=original_columns)
    eval_dataset = eval_dataset.map(tokenize_record, remove_columns=eval_dataset.column_names)
    train_before = len(train_dataset)
    eval_before = len(eval_dataset)
    train_dataset = train_dataset.filter(lambda row: not row["overlength"])
    eval_dataset = eval_dataset.filter(lambda row: not row["overlength"])
    train_overlength = train_before - len(train_dataset)
    eval_overlength = eval_before - len(eval_dataset)
    if not len(train_dataset) or not len(eval_dataset):
        raise TrainingSetupError("no in-range samples remain after token-length filtering")
    train_dataset = train_dataset.remove_columns(
        ["sequence_length", "assistant_tokens", "overlength"]
    )
    eval_dataset = eval_dataset.remove_columns(
        ["sequence_length", "assistant_tokens", "overlength"]
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_name_or_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, lora_config)
    parameter_stats = model_parameter_stats(model)

    eval_steps = 1 if args.smoke_test else 100
    max_steps = 2 if args.smoke_test else args.max_steps
    training_args = build_training_arguments(
        TrainingArguments,
        args=args,
        output_dir=output_dir,
        eval_steps=eval_steps,
        max_steps=max_steps,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    run_manifest = {
        "training_method": "lora",
        "base_dtype": "bfloat16",
        "optimizer": "adamw_torch",
        "model_name_or_path": str(args.model_name_or_path),
        "train_file": str(args.train_file.resolve()),
        "validation_file": str(args.validation_file.resolve()),
        "output_dir": str(output_dir),
        "smoke_test": args.smoke_test,
        "max_steps": max_steps,
        "candidate_set_variant": args.candidate_set_variant,
        "seed": args.seed,
        "max_seq_length": args.max_seq_length,
        "train_samples": len(train_dataset),
        "validation_samples": len(eval_dataset),
        "train_overlength_skipped": train_overlength,
        "validation_overlength_skipped": eval_overlength,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": list(TARGET_MODULES),
        },
        "parameters": parameter_stats,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    versions = dependency_versions(["torch", "transformers", "peft", "datasets", "accelerate"])
    (output_dir / "versions.json").write_text(
        json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    resume = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume)
    eval_metrics = trainer.evaluate()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        "train": train_result.metrics,
        "validation": eval_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"run_config": run_manifest, "versions": versions, "metrics": metrics}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BF16 LoRA fine-tuning for the evidence-grounded KG Critic."
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--call-prompt", type=Path, default=DEFAULT_CALL_PROMPT)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--candidate-set-variant",
        default="unspecified",
        choices=("fixed1", "fixed2", "fixed4", "variable", "unspecified"),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Resume from the latest checkpoint or from the supplied checkpoint path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.max_seq_length <= 0:
            raise ValueError("--max-seq-length must be positive")
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        data_summary = validate_data_files(args.train_file, args.validation_file)
        if args.validate_only:
            print(json.dumps({"valid": True, "data": data_summary}, ensure_ascii=False, indent=2))
            return 0
        result = run_training(args)
    except (
        CriticSFTDataError,
        FileNotFoundError,
        TrainingSetupError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
