from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence

from evidenceloop.runtime.settings import CriticRuntimeError, validate_direct_runtime_paths


@dataclass
class DirectGenerationResult:
    content: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class DirectCriticEngine:
    def __init__(self, runtime: Mapping, *, dependency_loader=None):
        self.runtime = dict(runtime)
        self._dependency_loader = dependency_loader
        self._lock = threading.RLock()
        self._loaded = False
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._stats = {
            "backend": "direct",
            "model_load_seconds": 0.0,
            "adapter_merged": False,
            "critic_prompt_count": 0,
            "critic_generation_batches": 0,
            "critic_input_tokens": 0,
            "critic_output_tokens": 0,
            "critic_generation_seconds": 0.0,
            "critic_peak_gpu_memory_bytes": 0,
        }

    def _load_dependencies(self):
        if self._dependency_loader is not None:
            return self._dependency_loader()
        try:
            import torch
            import transformers
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise CriticRuntimeError(
                "direct Critic dependencies are missing; install torch, transformers, "
                "peft, accelerate, and safetensors"
            ) from exc
        return {
            "torch": torch,
            "transformers_version": transformers.__version__,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "PeftModel": PeftModel,
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            validate_direct_runtime_paths(self.runtime)
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            deps = self._load_dependencies()
            torch = deps["torch"]
            if not torch.cuda.is_available():
                raise CriticRuntimeError("direct Critic requires an available CUDA device")
            if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                raise CriticRuntimeError("direct Critic requires CUDA BF16 support")

            device = torch.device(self.runtime["device"])
            device_index = device.index if device.index is not None else 0
            if device_index >= torch.cuda.device_count():
                raise CriticRuntimeError(
                    f"direct Critic device {self.runtime['device']} is unavailable"
                )
            torch.cuda.set_device(device)
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats(device)

            started = time.perf_counter()
            tokenizer = deps["AutoTokenizer"].from_pretrained(
                self.runtime["base_model_path"],
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            version = str(deps.get("transformers_version") or "4")
            try:
                major_version = int(version.split(".", 1)[0])
            except ValueError:
                major_version = 4
            dtype_key = "dtype" if major_version >= 5 else "torch_dtype"
            model_kwargs = {
                "local_files_only": True,
                "trust_remote_code": False,
                "device_map": {"": str(device)},
                "attn_implementation": self.runtime["attention_implementation"],
                dtype_key: torch.bfloat16,
            }
            base_model = deps["AutoModelForCausalLM"].from_pretrained(
                self.runtime["base_model_path"],
                **model_kwargs,
            )
            peft_model = deps["PeftModel"].from_pretrained(
                base_model,
                self.runtime["adapter_path"],
                is_trainable=False,
                local_files_only=True,
            )
            model = peft_model.merge_and_unload(safe_merge=True)
            model.eval()
            if getattr(model, "config", None) is not None:
                model.config.use_cache = True
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize(device)

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._loaded = True
            self._stats["model_load_seconds"] = round(time.perf_counter() - started, 3)
            self._stats["adapter_merged"] = True
            self._update_peak_memory()

    def validate_runtime(self) -> Dict:
        self._ensure_loaded()
        return {
            "checked": True,
            "backend": "direct",
            "model": self.runtime["model_name"],
            "base_model_path": self.runtime["base_model_path"],
            "adapter_path": self.runtime["adapter_path"],
            "device": self.runtime["device"],
            "dtype": self.runtime["dtype"],
            "adapter_merged": True,
            "thinking_enabled": False,
        }

    def _update_peak_memory(self) -> None:
        if self._torch is None or self._device is None:
            return
        if hasattr(self._torch.cuda, "max_memory_allocated"):
            peak = int(self._torch.cuda.max_memory_allocated(self._device))
            self._stats["critic_peak_gpu_memory_bytes"] = max(
                self._stats["critic_peak_gpu_memory_bytes"], peak
            )

    def _messages_for_template(self, messages: Sequence) -> List[Dict[str, str]]:
        output = []
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "")
                content = str(message.get("content") or "")
            elif isinstance(message, (list, tuple)) and len(message) == 2:
                role, content = str(message[0]), str(message[1])
            else:
                raise CriticRuntimeError("direct Critic received an invalid chat message")
            if role not in {"system", "user", "assistant"}:
                raise CriticRuntimeError(f"unsupported direct Critic chat role: {role!r}")
            output.append({"role": role, "content": content})
        return output

    def _render_prompt(self, messages: Sequence) -> str:
        return self._tokenizer.apply_chat_template(
            self._messages_for_template(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        oom_type = getattr(self._torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
        return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()

    def _generate_group_once(self, prompts: Sequence[str]) -> List[DirectGenerationResult]:
        tokenizer = self._tokenizer
        torch = self._torch
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
        max_seq_length = int(self.runtime["max_seq_length"])
        invalid = [index for index, length in enumerate(lengths) if length >= max_seq_length]
        if invalid:
            if len(prompts) == 1:
                return [
                    DirectGenerationResult(
                        error=(
                            f"Critic prompt has {lengths[0]} tokens, exceeding the "
                            f"{max_seq_length - 1}-token input limit"
                        ),
                        input_tokens=lengths[0],
                    )
                ]
            midpoint = len(prompts) // 2
            return self._generate_group(prompts[:midpoint]) + self._generate_group(
                prompts[midpoint:]
            )

        encoded = {name: value.to(self._device) for name, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        max_new_tokens = min(
            int(self.runtime["max_tokens"]),
            max_seq_length - max(lengths),
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize(self._device)

        elapsed = time.perf_counter() - started
        output_ids = generated[:, prompt_width:]
        contents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output_lengths = []
        for row in output_ids:
            if tokenizer.pad_token_id is None:
                output_lengths.append(int(row.shape[0]))
            else:
                output_lengths.append(int((row != tokenizer.pad_token_id).sum().item()))
        self._stats["critic_generation_batches"] += 1
        self._stats["critic_generation_seconds"] += elapsed
        self._stats["critic_input_tokens"] += sum(lengths)
        self._stats["critic_output_tokens"] += sum(output_lengths)
        self._update_peak_memory()
        return [
            DirectGenerationResult(
                content=str(content or "").strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            for content, input_tokens, output_tokens in zip(contents, lengths, output_lengths)
        ]

    def _generate_group(self, prompts: Sequence[str]) -> List[DirectGenerationResult]:
        try:
            return self._generate_group_once(prompts)
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            if hasattr(self._torch.cuda, "empty_cache"):
                self._torch.cuda.empty_cache()
            if len(prompts) == 1:
                return [
                    DirectGenerationResult(
                        error=f"CUDA out of memory while generating Critic prompt: {exc}"
                    )
                ]
            midpoint = len(prompts) // 2
            return self._generate_group(prompts[:midpoint]) + self._generate_group(
                prompts[midpoint:]
            )

    def generate_batch(
        self,
        message_batches: Sequence[Sequence],
        *,
        batch_size: Optional[int] = None,
    ) -> List[DirectGenerationResult]:
        self._ensure_loaded()
        size = int(batch_size or self.runtime["prompt_batch_size"])
        if not 1 <= size <= int(self.runtime["prompt_batch_size"]):
            raise CriticRuntimeError(
                f"direct Critic batch_size must be between 1 and "
                f"{self.runtime['prompt_batch_size']}"
            )
        with self._lock:
            prompts = [self._render_prompt(messages) for messages in message_batches]
            self._stats["critic_prompt_count"] += len(prompts)
            output = []
            for start in range(0, len(prompts), size):
                output.extend(self._generate_group(prompts[start : start + size]))
            return output

    def invoke(self, messages: Sequence):
        result = self.generate_batch([messages], batch_size=1)[0]
        if result.error:
            raise CriticRuntimeError(result.error)
        return SimpleNamespace(content=result.content)

    def metrics(self) -> Dict:
        with self._lock:
            output = dict(self._stats)
        seconds = float(output["critic_generation_seconds"])
        output["critic_generation_seconds"] = round(seconds, 3)
        output["critic_tokens_per_second"] = (
            round(output["critic_output_tokens"] / seconds, 3) if seconds else 0.0
        )
        return output
