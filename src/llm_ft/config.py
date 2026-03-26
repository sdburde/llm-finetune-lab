"""
Configuration management for LLM fine-tuning.

Provides a unified configuration class for all fine-tuning methods.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import yaml


@dataclass
class FineTuningConfig:
    """
    Unified configuration for LLM fine-tuning.
    
    Supports LoRA, QLoRA, SFT, DPO, and RLHF methods.
    """
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    
    # Method settings
    method: str = "lora"  # lora, qlora, sft, dpo, rlhf
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization (for QLoRA)
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Dataset settings
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    num_samples: int = 500
    max_length: int = 256
    packing: bool = False
    
    # Training settings
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    gradient_clip: float = 1.0
    
    # Optimizer settings
    optim: str = "adamw_torch"
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Output settings
    output_dir: str = "./output"
    adapter_dir: str = "./adapter"
    merged_dir: str = "./merged"
    gguf_path: str = "./model.gguf"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    num_ctx: int = 2048
    
    # System prompt for Ollama
    system_prompt: str = "You are a helpful AI assistant."
    stop_words: List[str] = field(default_factory=lambda: ["<|im_end|>", "<|eot_id|>"])
    
    # Hardware settings
    bf16: Optional[bool] = None  # Auto-detect
    fp16: Optional[bool] = None  # Auto-detect
    
    @classmethod
    def from_yaml(cls, path: str) -> "FineTuningConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.method not in ["lora", "qlora", "sft", "dpo", "rlhf"]:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.rank < 1 or self.rank > 256:
            raise ValueError(f"Rank must be between 1 and 256, got {self.rank}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {self.batch_size}")
        
        print("✅ Configuration validated")


# Default configurations for different methods
LORA_CONFIG = FineTuningConfig(
    method="lora",
    rank=16,
    alpha=32,
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_epochs=3,
)

QLORA_CONFIG = FineTuningConfig(
    method="qlora",
    load_in_4bit=True,
    rank=8,
    alpha=16,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_epochs=2,
    max_length=256,
)

SFT_CONFIG = FineTuningConfig(
    method="sft",
    rank=16,
    alpha=32,
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_epochs=3,
    max_length=512,
)

DPO_CONFIG = FineTuningConfig(
    method="dpo",
    rank=16,
    alpha=32,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    num_epochs=2,
    beta=0.1,  # DPO-specific
)

RLHF_CONFIG = FineTuningConfig(
    method="rlhf",
    rank=16,
    alpha=32,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_epochs=2,
)
