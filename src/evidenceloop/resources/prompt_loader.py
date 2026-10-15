"""Prompt resources retain the original wording and parser contracts."""

from importlib.resources import files


def _load_prompt(name):
    return (
        files("evidenceloop.resources")
        .joinpath("prompts", name + ".txt")
        .read_text(encoding="utf-8")
    )


def load_proposer_system_prompt():
    return _load_prompt("proposer_system_prompt")


def load_proposer_extract_call_prompt():
    return _load_prompt("proposer_extract_call_prompt")


def load_critic_system_prompt():
    return _load_prompt("critic_system_prompt")


def load_critic_call_prompt_batch():
    return _load_prompt("critic_call_prompt_batch")


def load_integrator_system_prompts():
    return _load_prompt("integrator_system_prompt")


def load_integrator_call_prompt_batch():
    return _load_prompt("integrator_call_prompt_batch")
