import os
from unittest.mock import patch
from evidenceloop.data.registry import DATASETS, dataset_config


def test_scierc_profile_is_explicit_and_preserves_direction():
    with patch.dict(os.environ, {}, clear=True):
        config = dataset_config("scierc")
        assert "scierc" in config.critic_adapter_path
        assert not os.environ
    assert DATASETS["scierc"].entity_order_mode == 'predicate_aware'
