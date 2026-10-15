import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evidenceloop.runtime.direct_critic import DirectCriticEngine, DirectGenerationResult
from evidenceloop.runtime.settings import (
    CriticRuntimeError,
    public_critic_runtime,
    resolve_critic_runtime,
    validate_direct_runtime_paths,
)


def make_assets(root: Path):
    base = root / "base"
    adapter = root / "adapter"
    base.mkdir()
    adapter.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (base / name).write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return base, adapter


def runtime_config(base, adapter):
    return {
        "critic_batch_size": 4,
        "critic_prompt_batch_size": 4,
        "critic_max_tokens": 1024,
        "critic_max_seq_length": 8192,
        "critic_model_name": "critic-rebel-phase2",
        "critic_base_model_path": str(base),
        "critic_adapter_path": str(adapter),
        "critic_device": "cuda:0",
    }


class FakeDevice:
    def __init__(self, value):
        self.value = value
        self.index = int(value.split(":", 1)[1]) if ":" in value else None

    def __str__(self):
        return self.value


class FakeCuda:
    def is_available(self):
        return True

    def is_bf16_supported(self):
        return True

    def device_count(self):
        return 1

    def set_device(self, device):
        self.device = device

    def reset_peak_memory_stats(self, device):
        pass

    def max_memory_allocated(self, device):
        return 1234

    def synchronize(self, device):
        pass


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return json.dumps(messages)


class FakeTokenizerFactory:
    calls = []
    tokenizer = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        cls.tokenizer = FakeTokenizer()
        return cls.tokenizer


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=False)
        self.eval_called = False

    def eval(self):
        self.eval_called = True


class FakeModelFactory:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return FakeModel()


class FakePeftModel:
    def __init__(self, base):
        self.base = base
        self.safe_merge = None

    def merge_and_unload(self, *, safe_merge):
        self.safe_merge = safe_merge
        return self.base


class FakePeftFactory:
    calls = []
    instance = None

    @classmethod
    def from_pretrained(cls, base, path, **kwargs):
        cls.calls.append((base, path, kwargs))
        cls.instance = FakePeftModel(base)
        return cls.instance


class CriticRuntimeTests(unittest.TestCase):
    def setUp(self):
        FakeTokenizerFactory.calls = []
        FakeModelFactory.calls = []
        FakePeftFactory.calls = []

    def test_environment_overrides_paths_and_public_runtime_has_no_endpoint_or_key(self):
        runtime = resolve_critic_runtime(runtime_config("yaml-base", "yaml-adapter"), {
            "CRITIC_MODEL_NAME": "critic-re-docred-phase2",
            "CRITIC_BASE_MODEL": "/models/Qwen3-8B",
            "CRITIC_ADAPTER_PATH": "/outputs/adapter",
            "CRITIC_DEVICE": "cuda:1",
            "CRITIC_PROMPT_BATCH_SIZE": "2",
        })

        self.assertEqual(runtime["backend"], "direct")
        self.assertEqual(runtime["model_name"], "critic-re-docred-phase2")
        self.assertEqual(runtime["base_model_path"], str(Path("/models/Qwen3-8B")))
        self.assertEqual(runtime["adapter_path"], str(Path("/outputs/adapter")))
        self.assertEqual(runtime["device"], "cuda:1")
        self.assertEqual(runtime["prompt_batch_size"], 2)
        public = public_critic_runtime(runtime)
        serialized = json.dumps(public)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("base_url", serialized)

    def test_invalid_runtime_values_fail_fast(self):
        base = runtime_config("base", "adapter")
        cases = (
            {**base, "critic_batch_size": 0},
            {**base, "critic_batch_size": 5},
            {**base, "critic_prompt_batch_size": 5},
            {**base, "critic_max_seq_length": 1024},
            {**base, "critic_device": "cpu"},
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(CriticRuntimeError):
                    resolve_critic_runtime(config, {})

    def test_path_validation_requires_complete_local_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base, adapter = make_assets(Path(tmpdir))
            runtime = resolve_critic_runtime(runtime_config(base, adapter), {})
            self.assertTrue(validate_direct_runtime_paths(runtime)["checked"])
            (adapter / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(CriticRuntimeError, "adapter_model"):
                validate_direct_runtime_paths(runtime)

    def test_direct_load_is_offline_bf16_sdpa_safe_merged_and_non_thinking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base, adapter = make_assets(Path(tmpdir))
            runtime = resolve_critic_runtime(runtime_config(base, adapter), {})
            fake_torch = SimpleNamespace(
                cuda=FakeCuda(),
                bfloat16="bf16",
                device=FakeDevice,
            )
            engine = DirectCriticEngine(runtime, dependency_loader=lambda: {
                "torch": fake_torch,
                "transformers_version": "5.0.0",
                "AutoTokenizer": FakeTokenizerFactory,
                "AutoModelForCausalLM": FakeModelFactory,
                "PeftModel": FakePeftFactory,
            })

            result = engine.validate_runtime()
            rendered = engine._render_prompt([
                ("system", "SYSTEM"),
                ("user", "USER"),
            ])

        self.assertEqual(result["backend"], "direct")
        self.assertTrue(result["adapter_merged"])
        self.assertFalse(result["thinking_enabled"])
        self.assertTrue(FakeTokenizerFactory.calls[0][1]["local_files_only"])
        model_kwargs = FakeModelFactory.calls[0][1]
        self.assertEqual(model_kwargs["dtype"], "bf16")
        self.assertEqual(model_kwargs["attn_implementation"], "sdpa")
        self.assertTrue(FakePeftFactory.instance.safe_merge)
        self.assertTrue(FakePeftFactory.calls[0][2]["local_files_only"])
        self.assertFalse(FakeTokenizerFactory.tokenizer.template_kwargs["enable_thinking"])
        self.assertIn("SYSTEM", rendered)
        self.assertEqual(engine.metrics()["critic_peak_gpu_memory_bytes"], 1234)

    def test_direct_generation_batches_at_four_and_bisects_cuda_oom(self):
        class OOMCuda:
            def __init__(self):
                self.empty_cache_calls = 0

            def empty_cache(self):
                self.empty_cache_calls += 1

        class OOMEngine(DirectCriticEngine):
            def __init__(self):
                super().__init__({"prompt_batch_size": 4})
                self._loaded = True
                self._torch = SimpleNamespace(cuda=OOMCuda(), OutOfMemoryError=RuntimeError)
                self.group_sizes = []

            def _render_prompt(self, messages):
                return messages[-1][1]

            def _generate_group_once(self, prompts):
                self.group_sizes.append(len(prompts))
                if len(prompts) > 1:
                    raise RuntimeError("CUDA out of memory")
                return [DirectGenerationResult(content=prompts[0])]

        engine = OOMEngine()
        results = engine.generate_batch(
            [[("user", f"prompt-{index}")] for index in range(5)],
            batch_size=4,
        )

        self.assertEqual([item.content for item in results], [f"prompt-{i}" for i in range(5)])
        self.assertEqual(engine.group_sizes, [4, 2, 1, 1, 2, 1, 1, 1])
        self.assertEqual(engine._torch.cuda.empty_cache_calls, 3)


if __name__ == "__main__":
    unittest.main()
