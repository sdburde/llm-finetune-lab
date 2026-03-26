#!/usr/bin/env python3
"""
Unified Fine-Tuning Script - All Methods (LoRA, QLoRA, SFT, DPO, RLHF)
Optimized for 8GB VRAM with proper model storage

Usage:
    python scripts/finetune.py --method qlora --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/finetune.py --method qlora --vram 8 --ram 8
    python scripts/finetune.py --config configs/qlora_8gb.yaml
"""

import argparse
import os
import sys
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml


# ============================================================================
# CONFIGURATION
# ============================================================================

class FineTuningConfig:
    """Configuration manager for fine-tuning."""
    
    def __init__(self, **kwargs):
        # Model
        self.model_name = kwargs.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        self.trust_remote_code = kwargs.get("trust_remote_code", True)
        
        # Method
        self.method = kwargs.get("method", "qlora")
        
        # LoRA
        self.rank = kwargs.get("rank", 8)
        self.alpha = kwargs.get("alpha", 16)
        self.dropout = kwargs.get("dropout", 0.05)
        self.target_modules = kwargs.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        
        # Quantization
        self.load_in_4bit = kwargs.get("load_in_4bit", None)
        self.bnb_4bit_quant_type = kwargs.get("bnb_4bit_quant_type", "nf4")
        self.bnb_4bit_use_double_quant = kwargs.get("bnb_4bit_use_double_quant", True)
        
        # Dataset
        self.dataset_name = kwargs.get("dataset_name", "tatsu-lab/alpaca")
        self.num_samples = kwargs.get("num_samples", 500)
        self.max_length = kwargs.get("max_length", 256)
        
        # Training
        self.batch_size = kwargs.get("batch_size", 1)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 8)
        self.learning_rate = kwargs.get("learning_rate", 2e-4)
        self.num_epochs = kwargs.get("num_epochs", 2)
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.05)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.lr_scheduler_type = kwargs.get("lr_scheduler_type", "cosine")
        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", True)
        self.gradient_clip = kwargs.get("gradient_clip", 1.0)
        
        # DPO specific
        self.beta = kwargs.get("beta", 0.1)
        
        # RLHF specific
        self.grpo_group_size = kwargs.get("grpo_group_size", 8)
        
        # Output
        self.output_base = kwargs.get("output_base", "./models")
        self.run_name = kwargs.get("run_name", None)
        
        # Hardware
        self.bf16 = kwargs.get("bf16", None)
        self.fp16 = kwargs.get("fp16", None)
    
    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def get_output_dirs(self):
        """Get properly organized output directories with unique names."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Auto-generate run name based on parameters if not provided
        if self.run_name is None:
            # Extract model name (last part of path)
            model_short = self.model_name.split('/')[-1].replace('-Instruct', '').replace('-instruct', '')
            
            # Get dataset name
            if hasattr(self, 'dataset_name') and self.dataset_name:
                if '/' in self.dataset_name:
                    dataset_short = self.dataset_name.split('/')[-1]
                elif '.' in self.dataset_name:
                    # Local file
                    dataset_short = Path(self.dataset_name).stem
                else:
                    dataset_short = self.dataset_name
            else:
                dataset_short = "custom"
            
            # Create descriptive name
            self.run_name = f"{self.method}_{model_short}_{dataset_short}_{timestamp}"
        
        return {
            "adapter": f"{self.output_base}/adapters/{self.run_name}",
            "merged": f"{self.output_base}/merged/{self.run_name}",
            "gguf": f"{self.output_base}/gguf/{self.run_name}.gguf",
            "checkpoints": f"{self.output_base}/checkpoints/{self.run_name}",
            "logs": f"{self.output_base}/logs/{self.run_name}.json",
        }
    
    def auto_configure_for_vram(self, vram_gb: int):
        """Auto-configure settings based on available VRAM."""
        # Auto-enable 4-bit for low VRAM
        if self.load_in_4bit is None:
            self.load_in_4bit = vram_gb <= 8
        
        # Configure based on VRAM
        if vram_gb <= 4:
            self.max_length = 128
            self.batch_size = 1
            self.gradient_accumulation_steps = 16
            self.rank = 4
            self.alpha = 8
            self.learning_rate = 1e-4
        elif vram_gb <= 6:
            self.max_length = 192
            self.batch_size = 1
            self.gradient_accumulation_steps = 12
            self.rank = 8
            self.alpha = 16
        elif vram_gb <= 8:
            self.max_length = 256
            self.batch_size = 1
            self.gradient_accumulation_steps = 8
            self.rank = 8
            self.alpha = 32
        elif vram_gb <= 12:
            self.max_length = 512
            self.batch_size = 2
            self.gradient_accumulation_steps = 4
            self.rank = 16
            self.alpha = 32
        elif vram_gb <= 16:
            self.max_length = 512
            self.batch_size = 2
            self.gradient_accumulation_steps = 4
            self.rank = 32
            self.alpha = 64
        else:
            self.max_length = 1024
            self.batch_size = 4
            self.gradient_accumulation_steps = 2
            self.rank = 64
            self.alpha = 128
        
        # Method-specific defaults
        if self.method == "dpo":
            self.learning_rate = 5e-7
        elif self.method == "rlhf":
            self.learning_rate = 1e-6
        
        print(f"✅ Auto-configured for {vram_gb}GB VRAM")
    
    def save(self, path):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    def __repr__(self):
        return f"FineTuningConfig(method={self.method}, model={self.model_name})"


# ============================================================================
# MODEL STORAGE MANAGER
# ============================================================================

class ModelStorageManager:
    """Manages model storage directories and organization."""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self._create_directories()
    
    def _create_directories(self):
        """Create organized directory structure."""
        dirs = [
            "downloads",      # HuggingFace cache
            "adapters",       # LoRA adapters
            "merged",         # Merged models
            "gguf",           # GGUF exports
            "checkpoints",    # Training checkpoints
            "logs",           # Training logs
            "ollama",         # Ollama exports
        ]
        
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore
        gitignore = self.base_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("""# Model files
*.bin
*.pt
*.pth
*.ckpt
*.gguf
*.safetensors
""")
    
    def get_adapter_path(self, run_name: str) -> Path:
        return self.base_dir / "adapters" / run_name
    
    def get_merged_path(self, run_name: str) -> Path:
        return self.base_dir / "merged" / run_name
    
    def get_gguf_path(self, run_name: str) -> Path:
        return self.base_dir / "gguf" / f"{run_name}.gguf"
    
    def save_training_log(self, run_name: str, metrics: dict):
        """Save training metrics."""
        log_path = self.base_dir / "logs" / f"{run_name}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"📊 Logs saved: {log_path}")


# ============================================================================
# FINE-TUNING ENGINE
# ============================================================================

class FineTuningEngine:
    """Main fine-tuning engine supporting all methods."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.storage = ModelStorageManager(config.output_base)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set environment for better memory management
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    def setup_environment(self):
        """Setup environment and check GPU."""
        print("\n" + "="*60)
        print("ENVIRONMENT SETUP")
        print("="*60)
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if bf16 else torch.float16
            
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {vram:.1f} GB")
            print(f"Compute: {'bfloat16' if bf16 else 'float16'}")
            
            # Auto-configure if not set
            if self.config.bf16 is None:
                self.config.bf16 = bf16
            if self.config.fp16 is None and not bf16:
                self.config.fp16 = True
        else:
            print("⚠️  No GPU detected - using CPU (slow)")
            dtype = torch.float32
        
        return torch.cuda.is_available()
    
    def load_tokenizer(self):
        """Load tokenizer."""
        from transformers import AutoTokenizer
        
        print(f"\nLoading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded (vocab: {len(self.tokenizer)})")
        return self.tokenizer
    
    def load_model(self):
        """Load model with optional quantization."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"\nLoading model: {self.config.model_name}")
        
        # Determine compute dtype
        if self.config.bf16:
            dtype = torch.bfloat16
        elif self.config.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Build quantization config for QLoRA
        bnb_config = None
        if self.config.load_in_4bit:
            print("Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=dtype,
            )
        
        # Load model
        load_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )
        
        print(f"✅ Model loaded ({self.model.num_parameters()/1e6:.1f}M params)")
        
        # VRAM check
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"VRAM used: {vram_used:.2f} GB")
        
        return self.model
    
    def setup_peft(self):
        """Setup PEFT (LoRA/QLoRA) adapters."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        print(f"\nSetting up LoRA (rank={self.config.rank})")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ LoRA setup: {trainable}/{total} params ({100*trainable/total:.3f}%)")
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
        return self.model
    
    def load_dataset(self):
        """Load and format dataset. Supports HF datasets and custom files."""
        from datasets import load_dataset
        
        print(f"\nLoading dataset: {self.config.dataset_name}")
        
        # Check if it's a local file
        dataset_path = self.config.dataset_name
        is_local_file = Path(dataset_path).exists()
        
        if is_local_file:
            file_ext = Path(dataset_path).suffix.lower()
            print(f"Loading local file: {dataset_path} ({file_ext})")
            
            if file_ext == '.json':
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif file_ext == '.jsonl':
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif file_ext == '.csv':
                dataset = load_dataset('csv', data_files=dataset_path, split='train')
            elif file_ext == '.parquet':
                dataset = load_dataset('parquet', data_files=dataset_path, split='train')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            print(f"Loaded {len(dataset)} samples from file")
        else:
            # Load from HuggingFace Hub
            dataset = load_dataset(self.config.dataset_name, split="train")
            print(f"Loaded from Hub: {len(dataset)} samples")
        
        # Subsample if needed
        if self.config.num_samples and self.config.num_samples < len(dataset):
            print(f"Subsampling to {self.config.num_samples} samples")
            dataset = dataset.shuffle(seed=42).select(range(self.config.num_samples))
        
        # Format for instruction tuning
        def format_example(example):
            if example.get("input") and example["input"].strip():
                text = (
                    f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Input:\n{example['input']}\n\n"
                    f"### Response:\n{example['output']}"
                )
            else:
                text = (
                    f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Response:\n{example['output']}"
                )
            return {"text": text}
        
        dataset = dataset.map(format_example)
        print(f"✅ Dataset ready: {len(dataset)} samples")
        
        return dataset
    
    def create_trainer(self, dataset):
        """Create appropriate trainer based on method."""
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainingArguments
        
        print(f"\nCreating {self.config.method.upper()} trainer")
        
        # Training arguments
        output_dirs = self.config.get_output_dirs()
        
        # New TRL 0.29+ uses SFTConfig
        training_args = SFTConfig(
            output_dir=output_dirs["adapter"],
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_grad_norm=self.config.gradient_clip,
            logging_steps=self.config.batch_size * self.config.gradient_accumulation_steps,
            save_steps=100,
            eval_strategy="no",
            bf16=self.config.bf16 or False,
            fp16=self.config.fp16 or False,
            report_to="none",
            dataset_text_field="text",
            max_length=self.config.max_length,
            dataset_num_proc=2,
            packing=False,
        )
        
        # Create trainer with new API
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        
        print(f"✅ Trainer created")
        return self.trainer
    
    def train(self):
        """Run training."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Train
        result = self.trainer.train()
        
        # Save adapter
        output_dirs = self.config.get_output_dirs()
        self.trainer.save_model(output_dirs["adapter"])
        self.tokenizer.save_pretrained(output_dirs["adapter"])
        
        print(f"\n✅ Training complete!")
        print(f"Adapter saved: {output_dirs['adapter']}")
        
        # Save metrics
        metrics = {
            "config": self.config.__dict__,
            "training_result": str(result),
            "final_loss": result.training_loss,
        }
        run_name = os.path.basename(output_dirs["adapter"])
        self.storage.save_training_log(run_name, metrics)
        
        return result
    
    def merge_adapter(self):
        """Merge LoRA adapter with base model."""
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        
        output_dirs = self.config.get_output_dirs()
        
        print(f"\nMerging adapter into base model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load and merge adapter
        model = PeftModel.from_pretrained(base_model, output_dirs["adapter"])
        merged = model.merge_and_unload()
        
        # Save merged
        merged.save_pretrained(output_dirs["merged"], safe_serialization=True)
        self.tokenizer.save_pretrained(output_dirs["merged"])
        
        print(f"✅ Merged model saved: {output_dirs['merged']}")
        
        # Cleanup
        del merged, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_dirs["merged"]
    
    def convert_to_gguf(self, quant_type: str = "q4_k_m"):
        """Convert to GGUF format."""
        import subprocess
        
        output_dirs = self.config.get_output_dirs()
        
        print(f"\nConverting to GGUF ({quant_type})...")
        
        # Clone llama.cpp if needed
        if not os.path.exists("llama.cpp"):
            print("Cloning llama.cpp...")
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git",
                "--depth=1", "--quiet"
            ], check=True)
        
        # Convert
        result = subprocess.run([
            sys.executable, "llama.cpp/convert_hf_to_gguf.py",
            output_dirs["merged"],
            "--outfile", output_dirs["gguf"],
            "--outtype", quant_type
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            size_gb = os.path.getsize(output_dirs["gguf"]) / 1e9
            print(f"✅ GGUF created: {output_dirs['gguf']} ({size_gb:.2f} GB)")
            return output_dirs["gguf"]
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return None
    
    def run_full_pipeline(self, do_merge: bool = True, do_convert: bool = True):
        """Run complete fine-tuning pipeline."""
        self.setup_environment()
        self.load_tokenizer()
        self.load_model()
        self.setup_peft()
        dataset = self.load_dataset()
        self.create_trainer(dataset)
        self.train()
        
        if do_merge:
            self.merge_adapter()
        
        if do_convert:
            self.convert_to_gguf()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        output_dirs = self.config.get_output_dirs()
        run_name = os.path.basename(output_dirs["adapter"])
        
        print(f"""
Results:
  Adapter:  {output_dirs['adapter']}
  Merged:   {output_dirs['merged'] if do_merge else 'N/A'}
  GGUF:     {output_dirs['gguf'] if do_convert else 'N/A'}
  Logs:     {self.storage.base_dir}/logs/{run_name}.json

Next steps:
  1. Test: python scripts/infer.py --model {output_dirs['merged']}
  2. Ollama: ollama create {run_name} -f Modelfile
  3. Run: ollama run {run_name}
""")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Fine-Tuning - All Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # QLoRA for 8GB VRAM
  python scripts/finetune.py --method qlora --vram 8

  # LoRA with custom model
  python scripts/finetune.py --method lora --model mistralai/Mistral-7B-Instruct-v0.3

  # Use config file
  python scripts/finetune.py --config configs/qlora_8gb.yaml

  # DPO fine-tuning
  python scripts/finetune.py --method dpo --beta 0.1
        """
    )
    
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--method", type=str, 
                       choices=["lora", "qlora", "sft", "dpo", "rlhf"],
                       default="qlora", help="Fine-tuning method")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--vram", type=int, default=None, help="Available VRAM (GB)")
    parser.add_argument("--ram", type=int, default=8, help="Available RAM (GB)")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--output", type=str, default="./models", help="Output base directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging")
    parser.add_argument("--no-convert", action="store_true", help="Skip GGUF conversion")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        print(f"Loading config: {args.config}")
        config = FineTuningConfig.from_yaml(args.config)
    else:
        config = FineTuningConfig()
    
    # Override with CLI args
    if args.method:
        config.method = args.method
    if args.model:
        config.model_name = args.model
    if args.output:
        config.output_base = args.output
    if args.name:
        config.run_name = args.name
    if args.rank:
        config.rank = args.rank
        config.alpha = args.rank * 2
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_length:
        config.max_length = args.max_length
    if args.dataset:
        config.dataset_name = args.dataset
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.beta:
        config.beta = args.beta
    
    # Auto-configure for VRAM
    if args.vram:
        config.auto_configure_for_vram(args.vram)
    elif torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        config.auto_configure_for_vram(int(vram))
    else:
        config.auto_configure_for_vram(8)  # Default to 8GB
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Method: {config.method}")
    print(f"Model: {config.model_name}")
    print(f"VRAM: Auto-configured")
    print(f"Rank: {config.rank}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Length: {config.max_length}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Output: {config.output_base}")
    print("="*60)
    
    # Run pipeline
    engine = FineTuningEngine(config)
    engine.run_full_pipeline(
        do_merge=not args.no_merge,
        do_convert=not args.no_convert,
    )


if __name__ == "__main__":
    main()
