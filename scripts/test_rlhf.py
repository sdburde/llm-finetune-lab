#!/usr/bin/env python3
"""
RLHF/GRPO Fine-Tuning Test Script for 8GB VRAM
Simplified version with correct TRL 0.29 API
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
import os
import json
from datetime import datetime

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_BASE = "./models"
NUM_SAMPLES = 50
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-6
NUM_EPOCHS = 1
NUM_GENERATIONS = 2

# Auto-generate unique output directory name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"rlhf_{MODEL_NAME.split('/')[-1].replace('-Instruct', '')}_{timestamp}"
OUTPUT_DIR = f"{OUTPUT_BASE}/adapters/{RUN_NAME}"
LOG_PATH = f"{OUTPUT_BASE}/logs/{RUN_NAME}.json"

print("="*60)
print("RLHF/GRPO FINE-TUNING TEST - 8GB VRAM")
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

# Load model with 4-bit quantization
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

# Load dataset
print(f"\nLoading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train").select(range(NUM_SAMPLES))

def format_prompt(example):
    return {
        "prompt": f"### Instruction:\n{example['instruction']}\n\n### Response:\n",
    }

dataset = dataset.map(format_prompt, remove_columns=["instruction", "input", "output"])
print(f"Loaded {len(dataset)} samples")

# Define reward function
def reward_length(completions, **kwargs):
    """Reward based on completion length."""
    return [len(c) for c in completions]

# GRPO Training
print("\n" + "="*60)
print("STARTING GRPO TRAINING")
print("="*60)

training_args = GRPOConfig(
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
    logging_steps=5,
    save_steps=20,
    eval_strategy="no",
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
    num_generations=2,  # Must be small for 8GB VRAM
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_length,
)

# Train
trainer.train()

# Save
print(f"\nSaving RLHF adapter to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Log results
log_data = {
    "method": "rlhf_grpo",
    "model": MODEL_NAME,
    "run_name": RUN_NAME,
    "timestamp": datetime.now().isoformat(),
    "training_result": str(trainer.state.log_history[-1]) if trainer.state.log_history else "N/A",
    "config": {
        "rank": 8,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "num_generations": NUM_GENERATIONS,
    }
}

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "w") as f:
    json.dump(log_data, f, indent=2)

print(f"\n✅ RLHF/GRPO Training Complete!")
print(f"Logs saved: {LOG_PATH}")
print(f"Adapter saved: {OUTPUT_DIR}")
