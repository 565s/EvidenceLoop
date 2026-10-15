import os
from unittest.mock import patch
from evidenceloop.data.registry import DATASETS, dataset_config


def test_redocred_profile_is_explicit_and_preserves_direction():
    with patch.dict(os.environ, {}, clear=True):
        config = dataset_config("redocred")
        assert "redocred" in config.critic_adapter_path
        assert not os.environ
    assert DATASETS["redocred"].entity_order_mode == 'directed'
