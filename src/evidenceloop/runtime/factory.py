"""Per-pipeline runtime factories. No process-global clients or models."""

import os
import threading


class LazyRuntime:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None
        self._lock = threading.RLock()

    def _get(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)


def create_chat(config, role, *, offline=False):
    from langchain_openai import ChatOpenAI

    base_url = os.environ.get(config.base_url_env) or (
        os.environ.get("OPENAI_BASE_URL") or os.environ.get("SIFLOW_BASE_URL")
    )
    api_key = os.environ.get(config.api_key_env) or (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("SIFLOW_API_KEY")
    )
    if not base_url or not api_key:
        raise ValueError(
            f"Set {config.base_url_env} and {config.api_key_env} before running API stages"
        )
    options = {}
    if offline and role == "integrator":
        options = {
            "max_tokens": config.offline_integrator_max_tokens,
            "timeout": config.offline_integrator_request_timeout,
            "max_retries": 0,
            "extra_body": {"enable_thinking": False},
        }
    return ChatOpenAI(
        model=getattr(config, role + "_model_name"),
        base_url=base_url,
        api_key=api_key,
        temperature=getattr(config, role + "_temperature"),
        **options,
    )


def create_agents(config, runtime_settings=None):
    from evidenceloop.agents.critic import CriticAgent
    from evidenceloop.agents.integrator import IntegratorAgent
    from evidenceloop.agents.proposer import ProposerAgent
    from evidenceloop.runtime.direct_critic import DirectCriticEngine
    from evidenceloop.runtime.settings import resolve_critic_runtime

    settings = runtime_settings or resolve_critic_runtime(config.to_dict())
    engine = DirectCriticEngine(settings)
    proposer = LazyRuntime(
        lambda: ProposerAgent(
            "Proposer", "Proposer", llm=LazyRuntime(lambda: create_chat(config, "proposer"))
        )
    )
    critic = LazyRuntime(
        lambda: CriticAgent("Critic", "Critic", llm=engine, runtime_settings=settings)
    )
    integrator = LazyRuntime(
        lambda: IntegratorAgent(
            "Integrator", "Integrator", llm=LazyRuntime(lambda: create_chat(config, "integrator"))
        )
    )
    return proposer, critic, integrator
