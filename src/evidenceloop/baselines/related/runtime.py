"""Explicit dependencies, cached model resolution, and lazy generation clients."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path

from .common import (
    DEFAULT_MODEL_ROOT,
    EMBEDDING_MODEL,
    GJ_ADAPTER,
    GJ_BASE,
    PIVE_ADAPTER,
    PIVE_BASE,
    ParseError,
    digest,
)

DEPENDENCIES = {
    "rakg": {
        "langchain_openai": "langchain-openai",
        "sentence_transformers": "sentence-transformers",
        "numpy": "numpy",
    },
    "itext2kg": {
        "itext2kg": "itext2kg",
        "langchain_openai": "langchain-openai",
        "sentence_transformers": "sentence-transformers",
        "numpy": "numpy",
    },
    "pive": {
        "langchain_openai": "langchain-openai",
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "sentencepiece": "sentencepiece",
    },
    "kggen": {"kg_gen": "kg-gen", "dspy": "dspy", "openai": "openai"},
    "graphjudge": {
        "langchain_openai": "langchain-openai",
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "sentencepiece": "sentencepiece",
    },
}
EXACT_VERSIONS = {"itext2kg": ("itext2kg", "0.0.7"), "kggen": ("kg-gen", "0.2.0")}


def redact(value):
    """Never persist the configured API key, including in SDK exception strings."""
    if isinstance(value, dict):
        return {
            k: redact(v)
            for k, v in value.items()
            if k.lower() not in {"api_key", "authorization", "headers", "api-key"}
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        secret = os.environ.get("OPENAI_API_KEY") or os.environ.get("SIFLOW_API_KEY")
        return value.replace(secret, "[REDACTED]") if secret else value
    return value


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if str(k) != "embeddings"}
    if isinstance(value, set):
        return [json_safe(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def dependency_versions(method: str, load_in_8bit=False) -> dict:
    required = dict(DEPENDENCIES[method])
    if method == "kggen":
        required.update(litellm="litellm", pydantic="pydantic", json_repair="json-repair")
    else:
        required.update(
            langchain_core="langchain-core",
            openai="openai",
            httpx="httpx",
            torch="torch",
            transformers="transformers",
            huggingface_hub="huggingface-hub",
            numpy="numpy",
        )
        if method == "itext2kg":
            required.update(langchain="langchain", sklearn="scikit-learn", pydantic="pydantic")
    if load_in_8bit:
        required["bitsandbytes"] = "bitsandbytes"
    missing, versions = [], {}
    for module, distribution in required.items():
        try:
            available = importlib.util.find_spec(module) is not None
            version = importlib.metadata.version(distribution) if available else None
        except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
            available, version = False, None
        if not available:
            missing.append(distribution)
        else:
            versions[distribution] = version
    if missing:
        raise RuntimeError(
            f"{method}: missing dependencies: {', '.join(missing)}; install src/evidenceloop/baselines/related/requirements/{method}.txt in its backend environment"
        )
    if method in EXACT_VERSIONS:
        distribution, expected = EXACT_VERSIONS[method]
        if versions[distribution] != expected:
            raise RuntimeError(
                f"{method}: requires {distribution}=={expected}, found {versions[distribution]}"
            )
    return versions


def cached_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or value.startswith((".", "models/", "models\\")):
        raise FileNotFoundError(f"model directory does not exist: {value}")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=value, local_files_only=True)).resolve()
    except Exception as exc:
        raise FileNotFoundError(
            f"model {value!r} is not prepared locally; run experiment/prepare_related_baseline_models.py --download"
        ) from exc


def model_signature(path: Path, *, adapter=False) -> dict:
    config_name = "adapter_config.json" if adapter else "config.json"
    if not (path / config_name).is_file():
        raise ValueError(f"{path}: missing {config_name}")
    weights = sorted([*path.glob("*.safetensors"), *path.glob("*.bin")])
    if not weights:
        raise ValueError(f"{path}: no model weights found")
    if not adapter and not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "spiece.model", "tokenizer.model", "vocab.txt")
    ):
        raise ValueError(f"{path}: missing tokenizer files")
    suffix = ".safetensors" if any(p.suffix == ".safetensors" for p in weights) else ".bin"
    for index in path.glob("*" + suffix + ".index.json"):
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not mapping or any(not (path / name).is_file() for name in set(mapping.values())):
            raise ValueError(f"{path}: incomplete model weight shards")
    config = json.loads((path / config_name).read_text(encoding="utf-8"))
    artifacts = {}
    for file in sorted(path.rglob("*")):
        if (
            not file.is_file()
            or ".cache" in file.relative_to(path).parts
            or file.suffix not in (".json", ".model", ".txt", ".bin", ".safetensors")
        ):
            continue
        stat = file.stat()
        item = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if file.suffix in (".json", ".model", ".txt") or adapter:
            sha = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    sha.update(block)
            item["sha256"] = sha.hexdigest()
        artifacts[file.relative_to(path).as_posix()] = item
    return {"path": str(path), "config_sha256": digest(config), "artifacts": artifacts}


def validate_adapter(base: Path, adapter: Path, method: str):
    base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    task = "SEQ_2_SEQ_LM" if method == "pive" else "CAUSAL_LM"
    architecture = "t5" if method == "pive" else "llama"
    if config.get("task_type") != task or base_config.get("model_type") != architecture:
        raise ValueError(
            f"{method}: adapter/base-model architecture mismatch (expected {task}/{architecture})"
        )
    expected_base = PIVE_BASE if method == "pive" else GJ_BASE
    declared = str(config.get("base_model_name_or_path", ""))
    # Local training writes an absolute base path; published adapters declare their repository.
    if declared != expected_base and not Path(declared).is_dir():
        raise ValueError(f"{method}: unrecognized adapter base_model_name_or_path {declared!r}")
    expected_dimensions = (
        {"d_model": 4096, "d_ff": 10240, "num_layers": 24, "num_heads": 64, "vocab_size": 32128}
        if method == "pive"
        else {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }
    )
    if declared == expected_base and any(
        k in base_config and base_config[k] != v for k, v in expected_dimensions.items()
    ):
        raise ValueError(f"{method}: published adapter does not match base-model dimensions")
    if Path(declared).is_dir() and Path(declared).resolve() != base.resolve():
        declared_config = json.loads((Path(declared) / "config.json").read_text(encoding="utf-8"))
        if digest(declared_config) != digest(base_config):
            raise ValueError(f"{method}: local adapter was trained with a different base model")
    return config


def model_locations(method: str, dataset: str, options: dict) -> dict:
    root = Path(options.get("model_root") or DEFAULT_MODEL_ROOT)
    manifest_path = root / "model_manifest.json"
    prepared = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    artifacts = prepared.get("artifacts", {})

    def locate(option, artifact, relative, repo):
        explicit = options.get(option)
        saved = artifacts.get(artifact, {}).get("path")
        value = explicit or saved or (str(root / relative) if (root / relative).exists() else repo)
        return cached_directory(str(value))

    result = {}
    if method in ("rakg", "itext2kg"):
        embedding = locate("embedding_model", "embedding", "embedding", EMBEDDING_MODEL)
        result["embedding"] = model_signature(embedding)
    if method == "pive":
        base = locate("verifier_base_model", "pive_base", "pive/base", PIVE_BASE)
        adapter = locate("verifier_adapter", "pive_adapter", "pive/adapter", PIVE_ADAPTER)
        validate_adapter(base, adapter, method)
        result["verifier"] = {
            "base": model_signature(base),
            "adapter": model_signature(adapter, adapter=True),
            "repository": PIVE_ADAPTER if not options.get("verifier_adapter") else None,
            "custom_adapter": bool(options.get("verifier_adapter")),
            "transfer": True if not options.get("verifier_adapter") else None,
            "training_domain": "KELM/WebNLG/GenWiki"
            if not options.get("verifier_adapter")
            else "custom/unspecified",
        }
    if method == "graphjudge":
        domain = {"rebel_sub": "rebel-sub", "scierc": "scierc"}.get(dataset, "genwiki")
        folder = f"llama2-7b-lora-{domain}-context"
        base = locate("verifier_base_model", "graphjudge_base", "graphjudge/base", GJ_BASE)
        if options.get("verifier_adapter"):
            adapter = cached_directory(str(options["verifier_adapter"]))
        elif artifacts.get(f"graphjudge_{domain}"):
            adapter = cached_directory(artifacts[f"graphjudge_{domain}"]["path"])
        elif (root / "graphjudge/adapters" / folder).is_dir():
            adapter = root / "graphjudge/adapters" / folder
        else:
            adapter = cached_directory(GJ_ADAPTER) / folder
        validate_adapter(base, adapter, method)
        result["verifier"] = {
            "base": model_signature(base),
            "adapter": model_signature(adapter, adapter=True),
            "repository": GJ_ADAPTER if not options.get("verifier_adapter") else None,
            "subfolder": folder if not options.get("verifier_adapter") else None,
            "transfer": domain == "genwiki" if not options.get("verifier_adapter") else None,
            "training_domain": domain
            if not options.get("verifier_adapter")
            else "custom/unspecified",
            "custom_adapter": bool(options.get("verifier_adapter")),
        }
    return result


def make_chat(runtime):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=runtime["model"],
        base_url=runtime["base_url"],
        api_key=os.environ["SIFLOW_API_KEY"],
        temperature=runtime["temperature"],
        max_tokens=runtime["max_tokens"],
        timeout=runtime["request_timeout"],
        max_retries=0,
        extra_body={"enable_thinking": runtime["enable_thinking"]},
    )


class ChatGenerator:
    def __init__(self, runtime):
        self.llm = make_chat(runtime)

    def __call__(self, prompt: str):
        response = self.llm.invoke([("user", prompt)])
        metadata = getattr(response, "response_metadata", {})
        if metadata.get("finish_reason") == "length":
            raise ParseError("generation reached max_tokens; increase --max-tokens")
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            raise ParseError("generation did not return text")
        return content, getattr(response, "usage_metadata", None)


class Embeddings:
    def __init__(self, path: str, device="auto"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            path,
            device=None if device == "auto" else device,
            local_files_only=True,
            trust_remote_code=False,
        )

    def embed_documents(self, texts):
        return self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class LocalVerifier:
    def __init__(self, runtime):
        import torch
        from peft import LoraConfig, PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch, self.runtime = torch, runtime
        self.seq2seq = runtime["method"] == "pive"
        paths = runtime["models"]["verifier"]
        base, adapter = paths["base"]["path"], paths["adapter"]["path"]
        adapter_tokenizer = (Path(adapter) / "tokenizer_config.json").is_file()
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter if adapter_tokenizer else base, local_files_only=True, trust_remote_code=False
        )
        if self.seq2seq:
            self.tokenizer.add_tokens(["<S>"])
        else:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        device = runtime["device"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.float32
            if device == "cpu"
            else (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        )
        kwargs = {"local_files_only": True, "trust_remote_code": False, "torch_dtype": dtype}
        if runtime["load_in_8bit"]:
            if device == "cpu":
                raise RuntimeError("--load-in-8bit requires a CUDA device")
            kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": device}
            )
        loader = AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        model = loader.from_pretrained(base, **kwargs)
        if not runtime["load_in_8bit"]:
            model = model.to(device)
        if self.seq2seq and (
            adapter_tokenizer or len(self.tokenizer) > model.get_input_embeddings().num_embeddings
        ):
            model.resize_token_embeddings(len(self.tokenizer))
        config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        # The 2023 PiVe checkpoint contains fields removed in modern PEFT.
        for legacy in ("enable_lora", "merge_weights"):
            config.pop(legacy, None)
        peft_config = LoraConfig(
            **{k: v for k, v in config.items() if k in LoraConfig.__dataclass_fields__}
        )
        self.model = PeftModel.from_pretrained(
            model, adapter, config=peft_config, local_files_only=True
        ).eval()

    def __call__(self, prompts: list[str]) -> list[str]:
        lengths = [len(self.tokenizer(p)["input_ids"]) for p in prompts]
        if max(lengths, default=0) > self.runtime["verifier_max_input_tokens"]:
            raise ValueError(
                "verifier input exceeds --verifier-max-input-tokens; source text was not truncated"
            )
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        encoded = {
            k: v.to(self.model.get_input_embeddings().weight.device) for k, v in encoded.items()
        }
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=4,
                max_new_tokens=256 if self.seq2seq else 64,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if not self.seq2seq:
            output = output[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def traced_call(stages: list, stage: str, fn, prompt):
    started = time.perf_counter()
    try:
        value = fn(prompt)
        response, usage = value if isinstance(value, tuple) and len(value) == 2 else (value, None)
        stages.append(
            redact(
                {
                    "stage": stage,
                    "prompt": prompt,
                    "response": json_safe(response),
                    "usage": json_safe(usage),
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
            )
        )
        return response
    except Exception as exc:
        stages.append(
            redact(
                {
                    "stage": stage,
                    "prompt": prompt,
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
            )
        )
        raise
