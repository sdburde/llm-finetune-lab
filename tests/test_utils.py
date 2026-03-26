"""Tests for utility functions."""

import torch
from llm_ft.utils import detect_gpu, qlora_bnb_config


def test_detect_gpu():
    """Test GPU detection."""
    device, vram, bf16, dtype = detect_gpu()
    assert device in ["cuda", "cpu"]
    if device == "cuda":
        assert vram > 0
        assert dtype in [torch.float16, torch.bfloat16]


def test_qlora_config():
    """Test QLoRA BitsAndBytesConfig creation."""
    dtype = torch.float16
    config = qlora_bnb_config(dtype)
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype == dtype
