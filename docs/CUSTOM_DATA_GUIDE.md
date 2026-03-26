# 📚 Custom Data Fine-Tuning Guide

## Overview

This guide shows you how to fine-tune LLMs with your own custom datasets, including proper data formats, ideal dataset sizes, and automatic model naming to prevent overwriting.

---

## 🎯 Quick Start

### 1. Prepare Your Data

Create a JSON file with your training data:

```json
[
  {
    "instruction": "What is your product?",
    "input": "",
    "output": "Our product is an AI-powered analytics tool that helps businesses make data-driven decisions."
  },
  {
    "instruction": "How do I get started?",
    "input": "",
    "output": "To get started, sign up for a free account at our website and follow the onboarding tutorial."
  }
]
```

### 2. Run Fine-Tuning

```bash
# With custom name
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json \
    --name my_custom_model

# Auto-generated name (recommended)
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json
```

### 3. Check Results

```bash
# Your model will be saved with unique name:
# adapters/qlora_Qwen2.5-0.5B_my_data_20260326_123456/

# Test inference
python scripts/infer.py --model ./models/adapters/qlora_Qwen2.5-0.5B_my_data_20260326_123456 \
    --prompt "What is your product?"
```

---

## 📁 Supported Data Formats

### 1. JSON Format (Recommended)

**File:** `my_data.json`

```json
[
  {
    "instruction": "Question or task",
    "input": "Optional context",
    "output": "Expected response"
  }
]
```

**Best for:** Datasets up to 10,000 samples

### 2. JSONL Format

**File:** `my_data.jsonl`

```jsonl
{"instruction": "Question 1", "input": "", "output": "Answer 1"}
{"instruction": "Question 2", "input": "", "output": "Answer 2"}
{"instruction": "Question 3", "input": "Context", "output": "Answer 3"}
```

**Best for:** Large datasets (>10K samples), streaming

### 3. CSV Format

**File:** `my_data.csv`

```csv
instruction,input,output
"What is Python?","","Python is a programming language"
"Explain AI","","AI is artificial intelligence"
```

**Best for:** Data exported from spreadsheets

### 4. Parquet Format

**File:** `my_data.parquet`

```python
# Create with pandas
import pandas as pd
df = pd.DataFrame({
    "instruction": ["What is...?", "Explain..."],
    "input": ["", ""],
    "output": ["Answer 1", "Answer 2"]
})
df.to_parquet("my_data.parquet")
```

**Best for:** Very large datasets, efficient storage

---

## 📊 Ideal Dataset Sizes

### By Model Size

| Model Size | Minimum | Recommended | Maximum | Training Time (8GB VRAM) |
|------------|---------|-------------|---------|--------------------------|
| **0.5B** | 50 | 500-1,000 | 5,000 | 5-30 min |
| **1.5B** | 100 | 1,000-5,000 | 10,000 | 10-60 min |
| **3B** | 200 | 2,000-10,000 | 20,000 | 20-120 min |
| **7B** | 500 | 5,000-20,000 | 50,000 | 1-4 hours |

### By Use Case

| Use Case | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **Testing** | 5-10 | 50 | Verify pipeline works |
| **Proof of Concept** | 50 | 200-500 | Demo purposes |
| **Production** | 500 | 2,000-5,000 | Good quality |
| **High Quality** | 2,000 | 10,000+ | Best results |

### Quality vs Quantity

```
More samples ≠ Better quality

100 high-quality examples > 1,000 noisy examples

Focus on:
✅ Diverse coverage of topics
✅ Clear, accurate responses
✅ Consistent formatting
✅ No duplicates or errors
```

---

## 🏷️ Automatic Model Naming

### How It Works

When you don't specify `--name`, the script auto-generates a unique name:

```bash
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/customer_support.json
```

**Auto-generated name:**
```
qlora_Qwen2.5-0.5B_customer_support_20260326_172841
│       │            │           │
│       │            │           └─ Timestamp (unique)
│       │            └─ Dataset name
│       └─ Model name (shortened)
└─ Method
```

### Naming Pattern

```
{method}_{model}_{dataset}_{timestamp}

Examples:
- qlora_Qwen2.5-0.5B_custom_data_20260326_172841
- lora_Mistral-7B_medical_20260326_143022
- dpo_Phi-3-mini_preferences_20260326_091530
```

### Benefits

✅ **No overwriting** - Each run gets unique timestamp  
✅ **Descriptive** - Name includes method, model, dataset  
✅ **Organized** - Easy to find specific runs  
✅ **Reproducible** - Can identify exact training run  

### Custom Naming

Override auto-naming with `--name`:

```bash
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --name support_bot_v1
```

---

## 📝 Data Quality Guidelines

### DO ✅

```json
[
  {
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing is a type of computing that uses quantum mechanical phenomena like superposition and entanglement to perform calculations. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can exist in multiple states simultaneously."
  },
  {
    "instruction": "Calculate the tip",
    "input": "Bill: $50, Tip rate: 15%",
    "output": "A 15% tip on $50 is $7.50, making the total $57.50."
  }
]
```

- ✅ Specific, clear instructions
- ✅ Accurate, well-written responses
- ✅ Appropriate input context when needed
- ✅ Consistent tone and style

### DON'T ❌

```json
[
  {
    "instruction": "Tell me stuff",
    "input": "",
    "output": "idk maybe google it"
  },
  {
    "instruction": "What is AI?",
    "input": "",
    "output": "AI is artificial intelligence"
  }
]
```

- ❌ Vague instructions
- ❌ Incorrect or low-quality responses
- ❌ Too brief (model learns to give short answers)
- ❌ Inconsistent formatting

---

## 🔧 Commands

### Test with Tiny Dataset

```bash
# Verify everything works (2 minutes)
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/custom_data_tiny.json \
    --num-samples 5 \
    --epochs 1
```

### Full Training

```bash
# Production training (30-60 minutes)
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_full_dataset.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --rank 8 \
    --epochs 2 \
    --lr 2e-4
```

### Multiple Runs (No Overwriting)

```bash
# Run 1 - auto name: qlora_Qwen2.5-0.5B_data_20260326_120000
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json

# Run 2 - auto name: qlora_Qwen2.5-0.5B_data_20260326_123000
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json

# Both saved separately - no overwriting!
```

### Compare Different Models

```bash
# Test with 0.5B model
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --name support_0.5B

# Test with 1.5B model
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --name support_1.5B

# Compare results
```

---

## 📂 Output Structure

After training with custom data:

```
models/
├── adapters/
│   ├── qlora_Qwen2.5-0.5B_custom_data_20260326_172841/
│   │   ├── adapter_config.json      # LoRA config
│   │   ├── adapter_model.safetensors # Adapter weights (8-20 MB)
│   │   ├── tokenizer.json           # Tokenizer
│   │   └── ...
│   ├── qlora_Qwen2.5-0.5B_custom_data_20260326_173512/  # Second run
│   └── ...
├── merged/
│   └── qlora_Qwen2.5-0.5B_custom_data_20260326_172841/  # Merged model
├── logs/
│   └── qlora_Qwen2.5-0.5B_custom_data_20260326_172841.json  # Training log
└── gguf/
    └── qlora_Qwen2.5-0.5B_custom_data_20260326_172841.gguf  # GGUF export
```

---

## 🎓 Example Datasets

### Customer Support Bot

**File:** `data/customer_support.json`

```json
[
  {
    "instruction": "What are your business hours?",
    "input": "",
    "output": "Our customer support is available Monday-Friday, 9 AM - 6 PM EST. We respond to all inquiries within 24 hours."
  },
  {
    "instruction": "How do I return a product?",
    "input": "",
    "output": "To return a product: 1) Log into your account, 2) Go to Order History, 3) Select the item and click 'Return', 4) Print the prepaid shipping label, 5) Drop off at any shipping location."
  },
  {
    "instruction": "Check order status",
    "input": "Order #12345",
    "output": "Order #12345 has been shipped and is expected to arrive on March 28. You can track it using tracking number ABC123456789."
  }
]
```

### Medical Assistant

**File:** `data/medical.json`

```json
[
  {
    "instruction": "What are the symptoms of flu?",
    "input": "",
    "output": "Common flu symptoms include: fever (100-102°F), cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Symptoms usually appear 1-4 days after exposure."
  },
  {
    "instruction": "Recommend treatment for headache",
    "input": "Patient has mild tension headache, no allergies",
    "output": "For mild tension headaches: 1) Rest in a quiet, dark room, 2) Apply a warm or cold compress to your forehead, 3) Stay hydrated, 4) Consider OTC pain relievers like ibuprofen (400mg) or acetaminophen (500mg). If headaches persist, consult a doctor."
  }
]
```

### Code Assistant

**File:** `data/coding.json`

```json
[
  {
    "instruction": "Write a Python function",
    "input": "Calculate factorial of n",
    "output": "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"
  },
  {
    "instruction": "Explain this code",
    "input": "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x < arr[len(arr)//2] for x in arr]) + quicksort([x >= arr[len(arr)//2] for x in arr])",
    "output": "This is a concise implementation of the quicksort algorithm. It: 1) Returns the array if it has 0-1 elements (base case), 2) Otherwise, picks the middle element as pivot, 3) Recursively sorts elements less than pivot and greater/equal to pivot, 4) Concatenates the results."
  }
]
```

---

## 🚀 Best Practices

### 1. Start Small

```bash
# Test with 10 samples first
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json \
    --num-samples 10 \
    --epochs 1
```

### 2. Iterate

```bash
# Run 1: Test pipeline
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/test_10.json --name test_run

# Run 2: Full dataset
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/full_1000.json --name production_v1

# Run 3: Improved data
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/improved_2000.json --name production_v2
```

### 3. Evaluate

```bash
# Test your model
python scripts/infer.py --model ./models/adapters/production_v1 \
    --prompt "Your test question"

# Compare with other runs
python scripts/infer.py --model ./models/adapters/production_v2 \
    --prompt "Same test question"
```

### 4. Version Control

```bash
# Keep data versions
data/
├── customer_support_v1.json
├── customer_support_v2.json
└── customer_support_final.json

# Use descriptive names
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support_v2.json \
    --name support_bot_v2_20260326
```

---

## 📈 Troubleshooting

### Problem: "CUDA out of memory"

**Solution:** Reduce dataset size or max_length
```bash
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json \
    --num-samples 100 \
    --max-length 128
```

### Problem: "File not found"

**Solution:** Use absolute path
```bash
python scripts/finetune.py --method qlora --vram 8 \
    --dataset /home/user/project/data/my_data.json
```

### Problem: Model overwrites previous run

**Solution:** Don't use --name, let it auto-generate
```bash
# Auto-generates unique name with timestamp
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json
```

---

## 📚 Next Steps

1. **Start with tiny dataset** (`data/custom_data_tiny.json`)
2. **Verify training works** (2 minutes)
3. **Prepare your full dataset** (500-1000 samples)
4. **Run production training** (30-60 minutes)
5. **Test and evaluate** your fine-tuned model
6. **Iterate** with improved data

---

**Tested on:** RTX 3070 Ti Laptop (8GB VRAM)  
**Auto-naming:** ✅ Prevents overwriting  
**Custom data:** ✅ JSON, JSONL, CSV, Parquet supported
