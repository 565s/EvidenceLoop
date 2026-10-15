import os
from unittest.mock import patch
from evidenceloop.data.registry import DATASETS, dataset_config


def test_biored_profile_is_explicit_and_preserves_direction():
    with patch.dict(os.environ, {}, clear=True):
        config = dataset_config("biored")
        assert "biored" in config.critic_adapter_path
        assert not os.environ
    assert DATASETS["biored"].entity_order_mode == 'undirected'
