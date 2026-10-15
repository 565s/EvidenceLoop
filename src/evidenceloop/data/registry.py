"""Canonical dataset names and evaluation conventions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    entity_order_mode: str = "directed"
    symmetric_predicates: tuple[str, ...] = ()
    directory: str = ""
    limit: int = 100
    critic_model: str = ""
    critic_adapter: str = ""
    empty_gold_policy: str = "error"


DATASETS = {
    "rebel_sub": DatasetSpec(
        "rebel_sub",
        "REBEL_Sub",
        directory="rebel_phase2_sft_test",
        limit=1000,
        critic_model="critic-rebel-phase2",
        critic_adapter="outputs/critic_qwen3_8b_lora_phase2/adapter",
    ),
    "redocred": DatasetSpec(
        "redocred",
        "Re-DocRED",
        directory="re_docred_phase2_sft_test",
        limit=500,
        critic_model="critic-re-docred-phase2",
        critic_adapter="outputs/critic_qwen3_8b_lora_redocred_phase2/adapter",
        empty_gold_policy="skip",
    ),
    "biored": DatasetSpec(
        "biored",
        "BioRED",
        "undirected",
        directory="biored_phase2_sft_test",
        critic_model="critic-biored-phase2",
        critic_adapter="outputs/critic_qwen3_8b_lora_biored_phase2/adapter",
    ),
    "scierc": DatasetSpec(
        "scierc",
        "SciERC",
        "predicate_aware",
        ("is compared with", "is conjoined with"),
        directory="scierc_phase2_sft_test",
        critic_model="critic-scierc-phase2",
        critic_adapter="outputs/critic_qwen3_8b_lora_scierc_phase2/adapter",
    ),
}


def dataset_config(value):
    from evidenceloop.settings import PipelineConfig

    spec = DATASETS[canonical_dataset(value)]
    return PipelineConfig(
        critic_model_name=spec.critic_model, critic_adapter_path=spec.critic_adapter
    )


def canonical_dataset(value):
    value = {"re_docred": "redocred"}.get(value, value)
    if value not in DATASETS:
        raise ValueError(f"Unknown dataset {value!r}; choose {tuple(DATASETS)}")
    return value


def candidate_strategy(value):
    value = {"pointwise": "fixed1", "max4": "fixed4"}.get(value, value)
    if value not in {"fixed1", "fixed2", "fixed4", "variable"}:
        raise ValueError(f"Unknown candidate strategy: {value}")
    return value


class AliasDict(dict):
    """Store canonical keys while accepting legacy lookup spellings."""

    def __getitem__(self, key):
        return super().__getitem__({"re_docred": "redocred", "pointwise": "fixed1"}.get(key, key))

    def __contains__(self, key):
        return super().__contains__({"re_docred": "redocred", "pointwise": "fixed1"}.get(key, key))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def variant_directory(root, value):
    from pathlib import Path

    variant = candidate_strategy(value)
    canonical = Path(root) / variant
    if canonical.exists() or variant != "fixed1":
        return canonical
    return Path(root) / "pointwise"


def dataset_aliases(value):
    dataset = canonical_dataset(value)
    return (dataset, "re_docred", "re_record") if dataset == "redocred" else (dataset,)
