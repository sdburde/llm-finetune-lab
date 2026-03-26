#!/usr/bin/env python3
"""
DPO Fine-Tuning Test Script for 8GB VRAM
Uses UltraFeedback dataset for preferences
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import os

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_BASE = "./models"
NUM_SAMPLES = 200  # Smaller dataset for DPO
MAX_LENGTH = 256
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
BETA = 0.1

# Auto-generate unique output directory name
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"dpo_{MODEL_NAME.split('/')[-1].replace('-Instruct', '')}_{timestamp}"
OUTPUT_DIR = f"{OUTPUT_BASE}/adapters/{RUN_NAME}"
LOG_PATH = f"{OUTPUT_BASE}/logs/{RUN_NAME}.json"

print("="*60)
print("DPO FINE-TUNING TEST - 8GB VRAM")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB)")
else:
    print("No GPU detected!")
    exit(1)

# Load tokenizer
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization for 8GB VRAM
print(f"\nLoading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
print(f"Model loaded: {model.num_parameters()/1e6:.1f}M params")

# Load reference model (same architecture, frozen)
print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
ref_model.eval()

# Setup LoRA
print("\nSetting up LoRA adapters...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA: {trainable}/{total} params ({100*trainable/total:.3f}%)")

# Load preference dataset
print(f"\nLoading preference dataset (UltraFeedback)...")
try:
    # Load UltraFeedback binarized dataset
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    print(f"Loaded {len(dataset)} preference pairs")
except Exception as e:
    print(f"Could not load UltraFeedback: {e}")
    print("Creating synthetic preference data for testing...")
    
    # Create synthetic preference data from Alpaca
    alpaca = load_dataset("tatsu-lab/alpaca", split="train").select(range(NUM_SAMPLES))
    
    def create_preference(example):
        # For testing, use same response as both chosen and rejected
        # In real use, you'd have human preferences
        return {
            "prompt": f"### Instruction:\n{example['instruction']}\n\n### Response:\n",
            "chosen": example['output'],
            "rejected": example['output'][:len(example['output'])//2] if len(example['output']) > 20 else example['output'],
        }
    
    dataset = alpaca.map(create_preference, remove_columns=alpaca.column_names)
    print(f"Created {len(dataset)} synthetic preference pairs")

# DPO Training
print("\n" + "="*60)
print("STARTING DPO TRAINING")
print("="*60)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=50,
    eval_strategy="no",
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
    beta=BETA,
    max_length=MAX_LENGTH,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save
print(f"\nSaving DPO adapter to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Log results
import json
from datetime import datetime

log_data = {
    "method": "dpo",
    "model": MODEL_NAME,
    "run_name": RUN_NAME,
    "timestamp": datetime.now().isoformat(),
    "training_result": str(trainer.state.log_history[-1]) if trainer.state.log_history else "N/A",
    "config": {
        "rank": 8,
        "beta": BETA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
    }
}

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "w") as f:
    json.dump(log_data, f, indent=2)

print(f"\n✅ DPO Training Complete!")
print(f"Logs saved: {LOG_PATH}")
print(f"Adapter saved: {OUTPUT_DIR}")
