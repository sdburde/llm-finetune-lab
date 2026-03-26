"""
LLM Fine-Tuning Toolkit
=======================

A production-grade toolkit for fine-tuning Large Language Models using
state-of-the-art parameter-efficient methods.

Supports: LoRA, QLoRA, SFT, DPO, and RLHF (GRPO/PPO)
"""

__version__ = "1.0.0"
__author__ = "sdburde"

from .utils import (
    pip_install,
    install_all,
    detect_gpu,
    qlora_bnb_config,
    prepare_for_training,
    vram_snapshot,
    merge_and_save,
    convert_to_gguf,
    write_modelfile,
    register_ollama,
    plot_loss,
)
from .config import FineTuningConfig
from .data import load_alpaca_dataset, format_instruction
from .models import load_model, load_tokenizer, setup_peft_model

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Utils
    "pip_install",
    "install_all",
    "detect_gpu",
    "qlora_bnb_config",
    "prepare_for_training",
    "vram_snapshot",
    "merge_and_save",
    "convert_to_gguf",
    "write_modelfile",
    "register_ollama",
    "plot_loss",
    # Config
    "FineTuningConfig",
    # Data
    "load_alpaca_dataset",
    "format_instruction",
    # Models
    "load_model",
    "load_tokenizer",
    "setup_peft_model",
]
