"""Explicit configuration; importing this module performs no I/O."""

from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_AGENT_CONFIG = {
    "critic_model_name": "critic-rebel-phase2",
    "proposer_model_name": "deepseek-ai/DeepSeek-V4-Flash",
    "integrator_model_name": "deepseek-ai/DeepSeek-V4-Flash",
    "critic_batch_size": 4,
    "critic_prompt_batch_size": 4,
    "critic_max_tokens": 1024,
    "critic_max_seq_length": 8192,
    "critic_base_model_path": "models/Qwen3-8B",
    "critic_adapter_path": "outputs/critic_qwen3_8b_lora_phase2/adapter",
    "critic_device": "cuda:0",
}
# Research commands receive a fresh copy before applying command-line overrides.
agent_config = DEFAULT_AGENT_CONFIG.copy()


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_variant: str = "full"
    proposer_model_name: str = "deepseek-ai/DeepSeek-V4-Flash"
    integrator_model_name: str = "deepseek-ai/DeepSeek-V4-Flash"
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    proposer_temperature: float = 0.5
    integrator_temperature: float = 0.5
    offline_integrator_max_tokens: int = 2048
    offline_integrator_request_timeout: float = 120.0
    critic_model_name: str = "critic-rebel-phase2"
    critic_base_model_path: str = "models/Qwen3-8B"
    critic_adapter_path: str = "outputs/critic_qwen3_8b_lora_phase2/adapter"
    critic_device: str = "cuda:0"
    critic_batch_size: int = 4
    critic_prompt_batch_size: int = 4
    critic_max_tokens: int = 1024
    critic_max_seq_length: int = 8192

    def __post_init__(self):
        if self.pipeline_variant not in {"full", "proposer_critic_once"}:
            raise ValueError("pipeline_variant must be full or proposer_critic_once")
        from evidenceloop.runtime.settings import resolve_critic_runtime

        resolve_critic_runtime(asdict(self), environ={})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls, path):
        import yaml

        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Pipeline config must be a mapping")
        return cls(**value)


def load_agent_config(config_path, encoding="utf-8"):
    import yaml

    return yaml.safe_load(Path(config_path).read_text(encoding=encoding))
