from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

DEFAULT_BASE_MODEL = "models/Qwen3-8B"
DEFAULT_ADAPTER_PATH = "outputs/critic_qwen3_8b_lora_phase2/adapter"


class CriticRuntimeError(RuntimeError):
    pass


def _positive_int(value, name: str, *, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CriticRuntimeError(f"{name} must be an integer") from exc
    if parsed <= 0 or (maximum is not None and parsed > maximum):
        suffix = f" between 1 and {maximum}" if maximum is not None else " positive"
        raise CriticRuntimeError(f"{name} must be{suffix}")
    return parsed


def _local_path(value, name: str) -> str:
    path = str(value or "").strip()
    if not path:
        raise CriticRuntimeError(f"{name} must not be empty")
    return str(Path(path).expanduser())


def _cuda_device(value) -> str:
    device = str(value or "cuda:0").strip().lower()
    if not re.fullmatch(r"cuda(?::\d+)?", device):
        raise CriticRuntimeError("critic_device must be cuda or cuda:N")
    return device


def resolve_critic_runtime(
    config: Mapping,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict:
    environ = os.environ if environ is None else environ
    batch_size = _positive_int(
        config.get("critic_batch_size", 4),
        "critic_batch_size",
        maximum=4,
    )
    prompt_batch_size = _positive_int(
        environ.get("CRITIC_PROMPT_BATCH_SIZE") or config.get("critic_prompt_batch_size", 4),
        "critic_prompt_batch_size",
        maximum=4,
    )
    max_tokens = _positive_int(
        config.get("critic_max_tokens", 1024),
        "critic_max_tokens",
    )
    max_seq_length = _positive_int(
        config.get("critic_max_seq_length", 8192),
        "critic_max_seq_length",
    )
    if max_seq_length <= max_tokens:
        raise CriticRuntimeError("critic_max_seq_length must be greater than critic_max_tokens")

    model_name = str(
        environ.get("CRITIC_MODEL_NAME")
        or config.get("critic_model_name")
        or config.get("critic_adapter_model_name")
        or "critic-rebel-phase2"
    ).strip()
    if not model_name:
        raise CriticRuntimeError("critic_model_name must not be empty")

    base_model_path = _local_path(
        environ.get("CRITIC_BASE_MODEL")
        or config.get("critic_base_model_path")
        or DEFAULT_BASE_MODEL,
        "critic_base_model_path",
    )
    adapter_path = _local_path(
        environ.get("CRITIC_ADAPTER_PATH")
        or config.get("critic_adapter_path")
        or DEFAULT_ADAPTER_PATH,
        "critic_adapter_path",
    )
    device = _cuda_device(environ.get("CRITIC_DEVICE") or config.get("critic_device") or "cuda:0")

    return {
        "backend": "direct",
        "batch_size": batch_size,
        "prompt_batch_size": prompt_batch_size,
        "max_tokens": max_tokens,
        "max_seq_length": max_seq_length,
        "model_name": model_name,
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "device": device,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "merge_adapter": True,
        "thinking_enabled": False,
        "initial_model": model_name,
        "final_model": model_name,
    }


def public_critic_runtime(runtime: Mapping) -> Dict:
    return {
        key: runtime[key]
        for key in (
            "backend",
            "batch_size",
            "prompt_batch_size",
            "max_tokens",
            "max_seq_length",
            "model_name",
            "base_model_path",
            "adapter_path",
            "device",
            "dtype",
            "attention_implementation",
            "merge_adapter",
            "thinking_enabled",
            "initial_model",
            "final_model",
        )
    }


def validate_direct_runtime_paths(runtime: Mapping) -> Dict:
    base_path = Path(runtime["base_model_path"])
    adapter_path = Path(runtime["adapter_path"])
    missing = []
    for path in (base_path / "config.json", base_path / "tokenizer_config.json"):
        if not path.is_file():
            missing.append(str(path))
    if not any(
        path.is_file()
        for path in (
            base_path / "tokenizer.json",
            base_path / "vocab.json",
            base_path / "vocab.txt",
        )
    ):
        missing.append(f"{base_path}/tokenizer.json or vocab file")
    if not (list(base_path.glob("*.safetensors")) or list(base_path.glob("pytorch_model*.bin"))):
        missing.append(f"{base_path}/*.safetensors or pytorch_model*.bin")
    if not (adapter_path / "adapter_config.json").is_file():
        missing.append(str(adapter_path / "adapter_config.json"))
    if not any(
        path.is_file()
        for path in (
            adapter_path / "adapter_model.safetensors",
            adapter_path / "adapter_model.bin",
        )
    ):
        missing.append(f"{adapter_path}/adapter_model.safetensors or adapter_model.bin")
    if missing:
        raise CriticRuntimeError(
            "direct Critic assets are incomplete; missing " + ", ".join(missing)
        )
    return {
        "checked": True,
        "backend": "direct",
        "base_model_path": str(base_path),
        "adapter_path": str(adapter_path),
        "device": runtime["device"],
    }
