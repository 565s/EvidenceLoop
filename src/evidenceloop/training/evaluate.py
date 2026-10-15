from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evidenceloop.data.sft import (  # noqa: E402
    VALID_ACTIONS,
    CriticSFTDataError,
    format_critic_sft_parts,
    iter_jsonl,
    public_triples_info,
    validate_critic_sft_jsonl,
    validate_critic_sft_record,
)
from evidenceloop.training.train import (  # noqa: E402
    DEFAULT_CALL_PROMPT,
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    TrainingSetupError,
    dependency_versions,
    read_prompt,
    validate_local_model_path,
)

ROOT = Path.cwd()


DEFAULT_TEST_FILE = ROOT / "data" / "critic_sft" / "rebel_phase2" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "critic_qwen3_8b_lora_phase2_eval"
INVALID_ACTION = "INVALID"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class _EvaluationProgress:
    """Report inference progress without making tqdm a hard dependency."""

    def __init__(self, total: int, enabled: bool = True):
        self.total = max(0, int(total))
        self.completed = 0
        self.started = time.perf_counter()
        self._bar = None
        self._enabled = bool(enabled)
        if not self._enabled:
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return
        self._bar = tqdm(
            total=self.total,
            desc="Evaluating Critic",
            unit="record",
            file=sys.stderr,
            dynamic_ncols=True,
        )

    def update(self, count: int) -> None:
        if not self._enabled:
            return
        count = max(0, int(count))
        self.completed = min(self.total, self.completed + count)
        if self._bar is not None:
            self._bar.update(count)
            return

        elapsed = max(0.0, time.perf_counter() - self.started)
        rate = self.completed / elapsed if elapsed and self.completed else 0.0
        eta = (self.total - self.completed) / rate if rate else 0.0
        print(
            f"[eval] {self.completed}/{self.total} records | "
            f"{rate:.2f} records/s | elapsed={_format_duration(elapsed)} | "
            f"eta={_format_duration(eta)}",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def parse_json_object(value: str) -> Mapping:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model output is empty")
    stripped = value.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for position, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError("model output does not contain a JSON object")


def validate_prediction_target(record: Mapping, target: Mapping) -> None:
    prediction_record = {
        "document_id": record["document_id"],
        "source_text": record["source_text"],
        "review_stage": record["review_stage"],
        "triples_info": public_triples_info(record),
        "target": target,
    }
    validate_critic_sft_record(prediction_record, f"prediction:{record['document_id']}")


def normalized_triple(value) -> Optional[Tuple[str, str, str]]:
    if not isinstance(value, Mapping):
        return None
    fields = []
    for name in ("subject", "predicate", "object"):
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            return None
        fields.append(" ".join(item.casefold().split()))
    return tuple(fields)


def classification_metrics(pairs: Iterable[Tuple[str, str]]) -> Dict:
    pairs = list(pairs)
    confusion = {
        gold: {predicted: 0 for predicted in (*VALID_ACTIONS, INVALID_ACTION)}
        for gold in VALID_ACTIONS
    }
    correct = 0
    for gold, predicted in pairs:
        predicted = predicted if predicted in VALID_ACTIONS else INVALID_ACTION
        confusion[gold][predicted] += 1
        correct += int(gold == predicted)

    per_action = {}
    for action in VALID_ACTIONS:
        true_positive = confusion[action][action]
        false_positive = sum(confusion[gold][action] for gold in VALID_ACTIONS if gold != action)
        false_negative = sum(
            confusion[action][predicted]
            for predicted in (*VALID_ACTIONS, INVALID_ACTION)
            if predicted != action
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_action[action] = {
            "support": sum(confusion[action].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total": len(pairs),
        "accuracy": correct / len(pairs) if pairs else 0.0,
        "macro_f1": sum(per_action[action]["f1"] for action in VALID_ACTIONS) / len(VALID_ACTIONS),
        "per_action": per_action,
        "confusion_matrix": confusion,
    }


def prediction_row_from_raw(record: Mapping, raw_output: str) -> Dict:
    row = {
        "document_id": record["document_id"],
        "raw_output": raw_output,
        "parsed_target": None,
        "json_valid": False,
        "schema_valid": False,
        "error": None,
    }
    try:
        target = parse_json_object(raw_output)
        row["json_valid"] = True
        row["parsed_target"] = target
        validate_prediction_target(record, target)
        row["schema_valid"] = True
    except (CriticSFTDataError, ValueError) as exc:
        row["error"] = str(exc)
    return row


def _target_from_prediction_row(
    record: Mapping, row: Mapping
) -> Tuple[Optional[Mapping], bool, bool]:
    target = None
    json_valid = False
    schema_valid = False
    if isinstance(row.get("parsed_target"), Mapping):
        target = row["parsed_target"]
        json_valid = True
    elif isinstance(row.get("target"), Mapping):
        target = row["target"]
        json_valid = True
    elif isinstance(row.get("results"), list):
        target = row
        json_valid = True
    elif isinstance(row.get("raw_output"), str):
        try:
            target = parse_json_object(row["raw_output"])
            json_valid = True
        except ValueError:
            return None, False, False

    if target is not None:
        try:
            validate_prediction_target(record, target)
            schema_valid = True
        except CriticSFTDataError:
            schema_valid = False
    return target, json_valid, schema_valid


def evaluate_prediction_rows(
    records: Sequence[Mapping], prediction_rows: Sequence[Mapping]
) -> Dict:
    rows_by_id = {}
    duplicate_prediction_ids = []
    for row in prediction_rows:
        document_id = row.get("document_id")
        if not isinstance(document_id, str):
            continue
        if document_id in rows_by_id:
            duplicate_prediction_ids.append(document_id)
        rows_by_id[document_id] = row
    if duplicate_prediction_ids:
        raise ValueError(
            "duplicate prediction document_id values: "
            + ", ".join(sorted(set(duplicate_prediction_ids))[:5])
        )

    all_pairs = []
    by_batch_size = defaultdict(list)
    by_position = defaultdict(list)
    json_valid_records = 0
    schema_valid_records = 0
    missing_records = 0
    corrected_total = 0
    corrected_exact = 0
    evidence_total = 0
    evidence_valid = 0
    keep_total = 0
    keep_overcorrected = 0

    for record in records:
        row = rows_by_id.get(record["document_id"])
        if row is None:
            missing_records += 1
            target = None
            json_valid = schema_valid = False
        else:
            target, json_valid, schema_valid = _target_from_prediction_row(record, row)
        json_valid_records += int(json_valid)
        schema_valid_records += int(schema_valid)
        predicted_by_index = (
            {result["triple_idx"]: result for result in target["results"]} if schema_valid else {}
        )

        batch_size = len(record["triples_info"])
        gold_by_index = {result["triple_idx"]: result for result in record["target"]["results"]}
        for index in range(batch_size):
            gold = gold_by_index[index]
            predicted = predicted_by_index.get(index)
            predicted_action = predicted["action"] if predicted else INVALID_ACTION
            pair = (gold["action"], predicted_action)
            all_pairs.append(pair)
            by_batch_size[batch_size].append(pair)
            by_position[index].append(pair)

            if gold["action"] in {"FIX", "REWRITE"}:
                corrected_total += 1
                if (
                    predicted
                    and predicted_action == gold["action"]
                    and normalized_triple(predicted.get("triple"))
                    == normalized_triple(gold.get("triple"))
                ):
                    corrected_exact += 1

            if gold["action"] != "DELETE":
                evidence_total += 1
                evidence = predicted.get("evidence") if predicted else None
                if isinstance(evidence, str) and evidence and evidence in record["source_text"]:
                    evidence_valid += 1

            if gold["action"] == "KEEP":
                keep_total += 1
                if predicted_action != "KEEP":
                    keep_overcorrected += 1

    expected_ids = {record["document_id"] for record in records}
    extra_prediction_ids = sorted(set(rows_by_id) - expected_ids)
    record_count = len(records)
    return {
        "records": record_count,
        "candidates": len(all_pairs),
        "json_valid_records": json_valid_records,
        "json_valid_rate": json_valid_records / record_count if record_count else 0.0,
        "schema_valid_records": schema_valid_records,
        "schema_valid_rate": schema_valid_records / record_count if record_count else 0.0,
        "missing_prediction_records": missing_records,
        "extra_prediction_records": len(extra_prediction_ids),
        "extra_prediction_document_ids": extra_prediction_ids[:20],
        "action": classification_metrics(all_pairs),
        "corrected_triple_exact_match": {
            "correct": corrected_exact,
            "total": corrected_total,
            "rate": corrected_exact / corrected_total if corrected_total else 0.0,
        },
        "evidence_validity": {
            "valid": evidence_valid,
            "total": evidence_total,
            "rate": evidence_valid / evidence_total if evidence_total else 0.0,
        },
        "keep_overcorrection": {
            "overcorrected": keep_overcorrected,
            "total": keep_total,
            "rate": keep_overcorrected / keep_total if keep_total else 0.0,
        },
        "by_batch_size": {
            str(size): classification_metrics(pairs)
            for size, pairs in sorted(by_batch_size.items())
        },
        "by_triple_idx": {
            str(position): classification_metrics(pairs)
            for position, pairs in sorted(by_position.items())
        },
    }


def validate_local_adapter_path(path: Path) -> None:
    path = Path(path)
    if not path.is_dir():
        raise TrainingSetupError(f"local adapter directory does not exist: {path}")
    required = [path / "adapter_config.json"]
    weights = list(path.glob("adapter_model*.safetensors")) + list(path.glob("adapter_model*.bin"))
    missing = [str(item) for item in required if not item.is_file()]
    if not weights:
        missing.append("adapter_model*.safetensors or adapter_model*.bin")
    if missing:
        raise TrainingSetupError(
            f"local adapter directory is incomplete ({path}): missing {', '.join(missing)}"
        )


def _batched(values: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_inference(args: argparse.Namespace, records: Sequence[Mapping]) -> List[Dict]:
    validate_local_model_path(args.model_name_or_path)
    validate_local_adapter_path(args.adapter_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise TrainingSetupError(
            "evaluation dependencies are missing; follow training/README.md on the GPU server"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_name_or_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
    model.eval()
    system_prompt = read_prompt(args.system_prompt)
    call_prompt = read_prompt(args.call_prompt)
    rows = []
    progress = _EvaluationProgress(
        len(records), enabled=not bool(getattr(args, "no_progress", False))
    )
    try:
        for batch in _batched(records, args.batch_size):
            prompts = []
            for record in batch:
                parts = format_critic_sft_parts(record, system_prompt, call_prompt)
                prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": parts["system"]},
                            {"role": "user", "content": parts["user"]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {name: value.to(model.device) for name, value in inputs.items()}
            prompt_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_texts = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            rows.extend(
                prediction_row_from_raw(record, output_text)
                for record, output_text in zip(batch, output_texts)
            )
            progress.update(len(batch))
    finally:
        progress.close()
    return rows


def load_prediction_rows(path: Path) -> List[Dict]:
    return list(iter_jsonl(Path(path)))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Critic LoRA adapter on Critic SFT JSONL data."
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--predictions-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--call-prompt", type=Path, default=DEFAULT_CALL_PROMPT)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable inference progress output.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.batch_size <= 0 or args.max_new_tokens <= 0:
            raise ValueError("--batch-size and --max-new-tokens must be positive")
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        validate_critic_sft_jsonl(args.test_file)
        records = list(iter_jsonl(args.test_file))
        if args.max_samples is not None:
            records = records[: args.max_samples]

        output_dir = args.output_dir.resolve()
        prediction_path = output_dir / "predictions.jsonl"
        summary_path = output_dir / "summary.json"
        if not args.overwrite and any(path.exists() for path in (prediction_path, summary_path)):
            raise FileExistsError(
                f"evaluation outputs already exist under {output_dir}; pass --overwrite"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.predictions_file is not None:
            prediction_rows = load_prediction_rows(args.predictions_file)
        else:
            if args.adapter_path is None:
                raise ValueError("--adapter-path is required unless --predictions-file is used")
            prediction_rows = run_inference(args, records)

        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary = evaluate_prediction_rows(records, prediction_rows)
        summary["evaluation_method"] = "lora_bfloat16"
        summary["base_dtype"] = "bfloat16"
        summary["model_name_or_path"] = str(args.model_name_or_path)
        summary["adapter_path"] = str(args.adapter_path) if args.adapter_path else None
        summary["test_file"] = str(args.test_file.resolve())
        summary["predictions_file"] = str(prediction_path)
        summary["versions"] = dependency_versions(["torch", "transformers", "peft", "accelerate"])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except (
        CriticSFTDataError,
        FileExistsError,
        FileNotFoundError,
        TrainingSetupError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
