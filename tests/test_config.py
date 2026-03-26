"""Basic tests for configuration."""

import pytest
from llm_ft.config import FineTuningConfig


def test_default_config():
    """Test default configuration creation."""
    cfg = FineTuningConfig()
    assert cfg.method == "lora"
    assert cfg.rank == 8
    assert cfg.batch_size == 1


def test_qlora_config():
    """Test QLoRA configuration."""
    cfg = FineTuningConfig(method="qlora", load_in_4bit=True, rank=8)
    assert cfg.load_in_4bit is True
    assert cfg.rank == 8


def test_config_validation():
    """Test configuration validation."""
    cfg = FineTuningConfig()
    cfg.validate()  # Should not raise


def test_invalid_rank():
    """Test invalid rank raises error."""
    with pytest.raises(ValueError):
        cfg = FineTuningConfig(rank=0)
        cfg.validate()


def test_config_from_dict():
    """Test updating config from dict."""
    cfg = FineTuningConfig()
    cfg.update(rank=16, alpha=32)
    assert cfg.rank == 16
    assert cfg.alpha == 32
