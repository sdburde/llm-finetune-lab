#!/usr/bin/env python3
"""
Training script with CLI interface.
Optimized for 8GB VRAM and 8GB RAM systems.

Usage:
    python scripts/train.py --method qlora --vram 8 --ram 8
    python scripts/train.py --config configs/qlora_8gb.yaml
"""

import argparse
import os
import sys
import gc

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Fine-Tuning Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # QLoRA for 8GB VRAM
  python scripts/train.py --method qlora --vram 8 --ram 8

  # LoRA for 12GB VRAM
  python scripts/train.py --method lora --vram 12 --model mistralai/Mistral-7B-Instruct-v0.3

  # Use config file
  python scripts/train.py --config configs/qlora_8gb.yaml
        """
    )
    
    # Config file (overrides other args)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Method
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["lora", "qlora", "sft", "dpo", "rlhf"],
        default="qlora",
        help="Fine-tuning method (default: qlora)"
    )
    
    # Model
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    
    # Hardware limits
    parser.add_argument(
        "--vram", 
        type=int, 
        default=8,
        help="Available VRAM in GB (default: 8)"
    )
    parser.add_argument(
        "--ram", 
        type=int, 
        default=8,
        help="Available RAM in GB (default: 8)"
    )
    
    # Dataset
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="tatsu-lab/alpaca",
        help="Dataset name (default: tatsu-lab/alpaca)"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=500,
        help="Number of training samples (default: 500)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=None,
        help="Max sequence length (auto-set based on VRAM)"
    )
    
    # Training params
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--name", type=str, default="model", help="Model name for export")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_settings_for_vram(vram_gb: int, ram_gb: int, method: str) -> dict:
    """
    Auto-configure settings based on available VRAM and RAM.
    
    Returns optimized settings to prevent OOM errors.
    """
    settings = {
        "max_length": 256,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "rank": 8,
        "learning_rate": 2e-4,
        "num_epochs": 2,
        "gradient_checkpointing": True,
    }
    
    # VRAM-based adjustments
    if vram_gb <= 4:
        # Ultra low memory mode
        settings.update({
            "max_length": 128,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "rank": 4,
            "learning_rate": 1e-4,
        })
    elif vram_gb <= 6:
        # Low memory mode
        settings.update({
            "max_length": 192,
            "batch_size": 1,
            "gradient_accumulation_steps": 12,
            "rank": 8,
        })
    elif vram_gb <= 8:
        # 8GB mode (default)
        settings.update({
            "max_length": 256,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "rank": 8,
        })
    elif vram_gb <= 12:
        # 12GB mode
        settings.update({
            "max_length": 512,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "rank": 16,
        })
    elif vram_gb <= 16:
        # 16GB mode
        settings.update({
            "max_length": 512,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "rank": 32,
        })
    elif vram_gb >= 24:
        # High-end GPU
        settings.update({
            "max_length": 1024,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "rank": 64,
        })
    
    # RAM-based adjustments (for dataset loading)
    if ram_gb <= 4:
        settings["num_proc"] = 1
        settings["preprocessing_num_workers"] = 1
    elif ram_gb <= 8:
        settings["num_proc"] = 2
        settings["preprocessing_num_workers"] = 2
    else:
        settings["num_proc"] = 4
        settings["preprocessing_num_workers"] = 4
    
    # Method-specific adjustments
    if method == "dpo":
        settings["learning_rate"] = 5e-7
        settings["beta"] = 0.1
    elif method == "rlhf":
        settings["learning_rate"] = 1e-6
        settings["grpo_group_size"] = 8
    
    return settings


def print_system_info():
    """Print system information."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # CPU RAM
    import psutil
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_available_gb = ram.available / (1024**3)
    print(f"RAM: {ram_available_gb:.1f}GB available / {ram_total_gb:.1f}GB total")
    
    # GPU VRAM
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_total_gb:.1f}GB total, {vram_allocated:.2f}GB allocated")
        print(f"Compute: bfloat16={torch.cuda.is_bf16_supported()}")
    else:
        print("GPU: Not detected (CPU mode)")
    
    # PyTorch version
    print(f"PyTorch: {torch.__version__}")
    print("="*60 + "\n")


def main():
    args = parse_args()
    
    # Load config from file if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        config = {}
    
    # Merge CLI args with config (CLI takes precedence)
    method = args.method or config.get("method", "qlora")
    model_name = args.model or config.get("model_name", args.model)
    vram_limit = args.vram
    ram_limit = args.ram
    
    # Auto-configure based on VRAM/RAM
    auto_settings = get_settings_for_vram(vram_limit, ram_limit, method)
    
    # Override with CLI args if provided
    if args.max_length:
        auto_settings["max_length"] = args.max_length
    if args.rank:
        auto_settings["rank"] = args.rank
    if args.epochs:
        auto_settings["num_epochs"] = args.epochs
    if args.lr:
        auto_settings["learning_rate"] = args.lr
    if args.batch_size:
        auto_settings["batch_size"] = args.batch_size
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Method: {method}")
    print(f"Model: {model_name}")
    print(f"VRAM Limit: {vram_limit}GB | RAM Limit: {ram_limit}GB")
    print(f"Max Length: {auto_settings['max_length']}")
    print(f"Batch Size: {auto_settings['batch_size']}")
    print(f"Gradient Accumulation: {auto_settings['gradient_accumulation_steps']}")
    print(f"LoRA Rank: {auto_settings['rank']}")
    print(f"Learning Rate: {auto_settings['learning_rate']}")
    print(f"Epochs: {auto_settings['num_epochs']}")
    print("="*60 + "\n")
    
    # Print system info
    print_system_info()
    
    # Import training modules
    print("Loading training modules...")
    from llm_ft import (
        detect_gpu,
        qlora_bnb_config,
        load_model,
        load_tokenizer,
        setup_peft_model,
        load_alpaca_dataset,
        format_instruction,
    )
    from llm_ft.trainers import create_lora_trainer, get_trainer_args
    from llm_ft.config import FineTuningConfig
    
    # Detect GPU
    device, vram, bf16, dtype = detect_gpu()
    
    if device == "cpu":
        print("⚠️  WARNING: Running on CPU! Training will be very slow.")
        print("   For GPU acceleration, install CUDA and PyTorch with GPU support.")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_alpaca_dataset(num_samples=args.num_samples)
    dataset = dataset.map(format_instruction)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    
    # Load model with quantization if needed
    load_in_4bit = (method == "qlora") or (vram_limit <= 8)
    print(f"\nLoading model: {model_name} (4-bit={load_in_4bit})")
    model = load_model(model_name, load_in_4bit=load_in_4bit, compute_dtype=dtype)
    
    # Setup PEFT
    print("\nSetting up LoRA adapters...")
    model = setup_peft_model(
        model,
        rank=auto_settings["rank"],
        alpha=auto_settings["rank"] * 2,
        dropout=0.05,
    )
    
    # Create trainer
    print("\nCreating trainer...")
    cfg = FineTuningConfig(
        method=method,
        model_name=model_name,
        batch_size=auto_settings["batch_size"],
        gradient_accumulation_steps=auto_settings["gradient_accumulation_steps"],
        learning_rate=auto_settings["learning_rate"],
        num_epochs=auto_settings["num_epochs"],
        max_length=auto_settings["max_length"],
        output_dir=args.output_dir,
    )
    
    trainer = create_lora_trainer(model, tokenizer, dataset, cfg)
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    trainer.train()
    
    # Save model
    adapter_dir = os.path.join(args.output_dir, f"{args.name}_adapter")
    print(f"\nSaving adapter to: {adapter_dir}")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Adapter saved: {adapter_dir}")
    print(f"\nNext steps:")
    print(f"  1. Merge adapter: python scripts/merge.py --adapter {adapter_dir}")
    print(f"  2. Convert to GGUF: python scripts/convert.py --model {adapter_dir}")
    print(f"  3. Run with Ollama: ollama run {args.name}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
