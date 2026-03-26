"""Data loading and preprocessing for LLM fine-tuning."""

from typing import Dict, Any
from datasets import load_dataset, Dataset


def load_alpaca_dataset(num_samples: int = 500, split: str = "train", seed: int = 42) -> Dataset:
    """Load the Alpaca dataset with optional subsampling."""
    dataset = load_dataset("tatsu-lab/alpaca", split=split)
    if num_samples and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    print(f"Loaded Alpaca dataset: {len(dataset)} samples")
    return dataset


def load_custom_dataset(path: str, split: str = "train", **kwargs) -> Dataset:
    """Load a custom dataset from local file or HuggingFace Hub."""
    dataset = load_dataset(path, split=split, **kwargs)
    print(f"Loaded custom dataset: {len(dataset)} samples")
    return dataset


def format_instruction(example: Dict[str, Any]) -> Dict[str, str]:
    """Format Alpaca-style instruction data."""
    if example.get("input") and example["input"].strip():
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}
