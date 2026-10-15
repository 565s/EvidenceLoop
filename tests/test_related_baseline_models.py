import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from evidenceloop.baselines import prepare as prep
from evidenceloop.baselines.related import runtime
from evidenceloop.baselines.related.common import GJ_BASE, PIVE_BASE, PIVE_INSTRUCTION, read_json, read_jsonl
from evidenceloop.baselines.related.training import convert_training_data, load_official_training


def base_model(path, model_type="t5", **config):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type, **config}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fake model")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def adapter_model(path, method="pive", **config):
    path.mkdir(parents=True, exist_ok=True)
    data = {"task_type": "SEQ_2_SEQ_LM" if method == "pive" else "CAUSAL_LM",
        "base_model_name_or_path": PIVE_BASE if method == "pive" else GJ_BASE, **config}
    (path / "adapter_config.json").write_text(json.dumps(data), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"fake adapter")
    return path


def test_dependency_check_reports_absent_packages_without_importing_models():
    with patch.object(runtime.importlib.util, "find_spec", return_value=None):
        with pytest.raises(RuntimeError, match="missing dependencies.*peft"):
            runtime.dependency_versions("pive")


@pytest.mark.parametrize("method,version", [("kggen", "0.4.0"), ("itext2kg", "0.1.0")])
def test_official_package_version_must_be_pinned(method, version):
    with patch.object(runtime.importlib.util, "find_spec", return_value=object()), patch.object(runtime.importlib.metadata, "version", return_value=version):
        with pytest.raises(RuntimeError, match="requires"):
            runtime.dependency_versions(method)


def test_model_compatibility_and_adapter_signature_changes(tmp_path):
    base = base_model(tmp_path / "base")
    adapter = adapter_model(tmp_path / "adapter")
    assert runtime.validate_adapter(base, adapter, "pive")["task_type"] == "SEQ_2_SEQ_LM"
    signature = runtime.model_signature(adapter, adapter=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed adapter")
    assert runtime.model_signature(adapter, adapter=True) != signature
    with pytest.raises(ValueError, match="architecture mismatch"):
        runtime.validate_adapter(base, adapter, "graphjudge")


def test_model_preflight_checks_dimensions_and_complete_shards(tmp_path):
    base = base_model(tmp_path / "base", d_model=1024)
    adapter = adapter_model(tmp_path / "adapter")
    with pytest.raises(ValueError, match="dimensions"):
        runtime.validate_adapter(base, adapter, "pive")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "missing-shard.safetensors"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete model weight shards"):
        runtime.model_signature(base)


def test_model_preflight_missing_tokenizer_and_nested_embedding_settings(tmp_path):
    base = base_model(tmp_path / "embedding", model_type="xlm-roberta")
    directory = base / "1_Pooling"
    directory.mkdir()
    (directory / "config.json").write_text('{"pool": true}', encoding="utf-8")
    before = runtime.model_signature(base)
    (directory / "config.json").write_text('{"pool": false}', encoding="utf-8")
    assert runtime.model_signature(base) != before
    (base / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="missing tokenizer"):
        runtime.model_signature(base)


@pytest.mark.parametrize("dataset,domain,transfer", [("rebel_sub", "rebel-sub", False), ("scierc", "scierc", False),
    ("redocred", "genwiki", True), ("biored", "genwiki", True)])
def test_graphjudge_public_adapter_selection_and_transfer_metadata(tmp_path, dataset, domain, transfer):
    base_model(tmp_path / "graphjudge/base", "llama")
    adapter_model(tmp_path / f"graphjudge/adapters/llama2-7b-lora-{domain}-context", "graphjudge")
    result = runtime.model_locations("graphjudge", dataset, {"model_root": str(tmp_path)})
    assert result["verifier"]["training_domain"] == domain and result["verifier"]["transfer"] is transfer
    custom = runtime.model_locations("graphjudge", dataset, {"model_root": str(tmp_path),
        "verifier_adapter": result["verifier"]["adapter"]["path"]})
    assert custom["verifier"]["training_domain"] == "custom/unspecified"


def test_cached_model_lookup_never_downloads_missing_weights():
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not cached")

    with patch.dict(sys.modules, {"huggingface_hub": NS(snapshot_download=snapshot)}):
        with pytest.raises(FileNotFoundError, match="not prepared locally"):
            runtime.cached_directory("example/missing-model")
    assert calls == [{"repo_id": "example/missing-model", "local_files_only": True}]


def test_explicit_download_pins_sha_and_avoids_duplicate_weight_formats(tmp_path):
    calls = []

    class API:
        def model_info(self, repo, revision):
            assert repo == "example/model" and revision == "main"
            return NS(sha="pinned-sha", siblings=[NS(rfilename=name) for name in
                ("model.safetensors", "pytorch_model.bin", "config.json", "tokenizer.json", "pytorch_model.bin.index.json")])

    def download(**kwargs):
        calls.append(kwargs)
        base_model(Path(kwargs["local_dir"]))

    result = prep.download_artifact({"repository": "example/model", "relative_dir": "base"}, tmp_path, api=API(), downloader=download)
    assert result["revision"] == "pinned-sha"
    assert calls[0]["revision"] == "pinned-sha"
    assert "pytorch_model.bin" not in calls[0]["allow_patterns"]
    assert "pytorch_model.bin.index.json" not in calls[0]["allow_patterns"]


def test_prepare_check_only_has_no_download_or_manifest_mutation(tmp_path):
    base_model(tmp_path / "pive/base")
    adapter_model(tmp_path / "pive/adapter")
    args = prep.parse_args(["--method", "pive", "--model-root", str(tmp_path)])

    def forbidden(**kwargs):
        raise AssertionError("check-only downloaded models")

    artifacts = prep.prepare_models(args, downloader=forbidden)
    assert set(artifacts) == {"pive_base", "pive_adapter"}
    assert not (tmp_path / "model_manifest.json").exists()


def test_model_task_list_reuses_one_bge_and_all_three_graphjudge_adapters():
    tasks = prep.artifact_tasks()
    assert len(tasks) == 7
    assert set(n for n in tasks if n.startswith("graphjudge_")) == {
        "graphjudge_base", "graphjudge_rebel-sub", "graphjudge_scierc", "graphjudge_genwiki"}


def test_credentials_are_redacted_in_nested_responses(monkeypatch):
    monkeypatch.setenv("SIFLOW_API_KEY", "test-secret")
    assert runtime.redact({"api_key": "test-secret", "error": "key=test-secret", "nested": [{"authorization": "bearer", "text": "test-secret"}]}) == {
        "error": "key=[REDACTED]", "nested": [{"text": "[REDACTED]"}]}


def test_pive_training_conversion_groups_sources_without_test_leakage(tmp_path):
    rows = [{"instruction": PIVE_INSTRUCTION, "input": f"Context {doc} <S> [['A','has','{variant}']]", "output": "Correct"}
        for doc in range(4) for variant in range(2)]
    source = tmp_path / "combined_verifier_data.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    train, validation = convert_training_data(source, tmp_path / "converted", validation_fraction=0.25)
    contexts = lambda values: {r["input"].split("<S>")[0] for r in values}
    assert not contexts(train) & contexts(validation)
    assert len(train) == 6 and len(validation) == 2
    assert read_json(tmp_path / "converted/training_data_manifest.json")["split_unit"] == "source_document"
    assert read_jsonl(tmp_path / "converted/train.jsonl") == train


def test_official_training_zip_conversion_ignores_test_files(tmp_path):
    source = tmp_path / "official.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("only_one_error_webnlg/train.source", "Context 1 <S> []\nContext 2 <S> []\n")
        archive.writestr("only_one_error_webnlg/train.target", "Correct\n['A', 'has', 'B']\n")
        archive.writestr("only_one_error_webnlg/test.source", "Not a training input")
    rows = load_official_training(source)
    assert len(rows) == 2 and "WebNLG" in rows[0]["instruction"]
    assert all("Not a training input" not in row["input"] for row in rows)


@pytest.mark.parametrize("filename", ["test.source", "test.target", "gold_triples.jsonl"])
def test_training_entry_rejects_benchmark_test_inputs(tmp_path, filename):
    path = tmp_path / filename
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="test data"):
        load_official_training(path)


@pytest.mark.parametrize("saved_tokenizer", [False, True])
def test_local_pive_verifier_legacy_config_and_saved_tokenizer_compatibility(tmp_path, saved_tokenizer):
    base, adapter = base_model(tmp_path / "base"), adapter_model(tmp_path / "adapter", enable_lora=None, merge_weights=False)
    if saved_tokenizer:
        (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    calls = []

    class Tokenizer:
        def add_tokens(self, values):
            assert values == ["<S>"]

        def __len__(self):
            return 32101

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

        def get_input_embeddings(self):
            return NS(num_embeddings=32128)

        def resize_token_embeddings(self, size):
            calls.append(("resize", size))

        def eval(self):
            return self

    class LoraConfig:
        __dataclass_fields__ = {k: None for k in ("task_type", "base_model_name_or_path")}

        def __init__(self, **kwargs):
            assert "enable_lora" not in kwargs and "merge_weights" not in kwargs

    def tokenizer_loader(path, **kwargs):
        calls.append(("tokenizer", path))
        assert kwargs["local_files_only"] is True
        return Tokenizer()

    def base_loader(path, **kwargs):
        assert kwargs["local_files_only"] is True
        return Model()

    def adapter_loader(model, path, **kwargs):
        assert path == str(adapter) and kwargs["local_files_only"] is True
        return model

    fake_torch = NS(float32="fp32", cuda=NS(is_available=lambda: False))
    transformers = NS(AutoTokenizer=NS(from_pretrained=tokenizer_loader),
        AutoModelForSeq2SeqLM=NS(from_pretrained=base_loader), AutoModelForCausalLM=object(), BitsAndBytesConfig=object())
    peft = NS(LoraConfig=LoraConfig, PeftModel=NS(from_pretrained=adapter_loader))
    config = {"method": "pive", "models": {"verifier": {"base": {"path": str(base)}, "adapter": {"path": str(adapter)}}},
        "device": "auto", "load_in_8bit": False}
    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": transformers, "peft": peft}):
        runtime.LocalVerifier(config)
    assert calls[0] == ("tokenizer", str(adapter if saved_tokenizer else base))
    assert (("resize", 32101) in calls) is saved_tokenizer
