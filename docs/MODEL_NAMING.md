# 🏷️ Model Naming Convention

## Problem Fixed

**Before:** All models had simple names like `lora_0.5B`, `dpo_0.5B` - easy to overwrite!

**After:** Auto-generated unique names with model, dataset, and timestamp.

---

## Naming Pattern

### Auto-Generated Names

```
{method}_{model}_{dataset}_{timestamp}

Examples:
├── qlora_Qwen2.5-0.5B_custom_data_tiny.json_20260326_172841
│   ├── method: qlora
│   ├── model: Qwen2.5-0.5B (shortened)
│   ├── dataset: custom_data_tiny.json
│   └── timestamp: 20260326_172841
│
├── dpo_Qwen2.5-0.5B_ultrafeedback_20260326_180530
│   ├── method: dpo
│   ├── model: Qwen2.5-0.5B
│   ├── dataset: ultrafeedback
│   └── timestamp: 20260326_180530
│
└── rlhf_Qwen2.5-0.5B_alpaca_20260326_191245
    ├── method: rlhf
    ├── model: Qwen2.5-0.5B
    ├── dataset: alpaca
    └── timestamp: 20260326_191245
```

### Custom Names (User-Provided)

```bash
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --name support_bot_v1

# Output: adapters/support_bot_v1/
```

---

## Updated Scripts

All scripts now use auto-generated naming:

| Script | Naming | Status |
|--------|--------|--------|
| `finetune.py` | Auto + Custom support | ✅ Fixed |
| `test_dpo.py` | Auto-generated | ✅ Fixed |
| `test_rlhf.py` | Auto-generated | ✅ Fixed |
| `test_install.py` | N/A (no training) | ✅ N/A |
| `check_env.py` | N/A (no training) | ✅ N/A |

---

## Before vs After

### Before (Old Runs)

```
models/adapters/
├── lora_0.5B              ❌ Simple name
├── sft_0.5B               ❌ Simple name
├── dpo_0.5B               ❌ Simple name
├── rlhf_0.5B              ❌ Simple name
└── test_custom            ⚠️ Custom name
```

**Problems:**
- ❌ Easy to overwrite (run twice = lost data)
- ❌ No model info in name
- ❌ No dataset info
- ❌ No timestamp

### After (New Runs)

```
models/adapters/
├── qlora_Qwen2.5-0.5B_custom_data_tiny.json_20260326_172841
├── qlora_Qwen2.5-0.5B_custom_data_tiny.json_20260326_173512
├── dpo_Qwen2.5-0.5B_ultrafeedback_20260326_180530
├── rlhf_Qwen2.5-0.5B_alpaca_20260326_191245
└── support_bot_v1
```

**Benefits:**
- ✅ Never overwrites (unique timestamp)
- ✅ Model name in folder
- ✅ Dataset name in folder
- ✅ Exact time of run
- ✅ Easy to compare runs

---

## How It Works

### In `finetune.py`

```python
def get_output_dirs(self):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if self.run_name is None:
        # Auto-generate from parameters
        model_short = self.model_name.split('/')[-1]
        dataset_short = Path(self.dataset_name).stem
        
        self.run_name = f"{self.method}_{model_short}_{dataset_short}_{timestamp}"
    
    return {
        "adapter": f"{self.output_base}/adapters/{self.run_name}",
        "merged": f"{self.output_base}/merged/{self.run_name}",
        "logs": f"{self.output_base}/logs/{self.run_name}.json",
    }
```

### In `test_dpo.py` and `test_rlhf.py`

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"dpo_{MODEL_NAME.split('/')[-1].replace('-Instruct', '')}_{timestamp}"
OUTPUT_DIR = f"./models/adapters/{RUN_NAME}"
LOG_PATH = f"./models/logs/{RUN_NAME}.json"
```

---

## Usage Examples

### 1. Auto-Generated (Recommended)

```bash
# Each run gets unique name
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json

# Run 1: adapters/qlora_Qwen2.5-0.5B_support_20260326_120000/
# Run 2: adapters/qlora_Qwen2.5-0.5B_support_20260326_123000/
# No overwriting!
```

### 2. Custom Name

```bash
# Use your own name
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --name support_bot_v1

# Output: adapters/support_bot_v1/
```

### 3. Compare Different Models

```bash
# Test 0.5B model
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --model Qwen/Qwen2.5-0.5B-Instruct

# Output: qlora_Qwen2.5-0.5B_support_20260326_120000/

# Test 1.5B model
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --model Qwen/Qwen2.5-1.5B-Instruct

# Output: qlora_Qwen2.5-1.5B_support_20260326_130000/

# Both saved separately!
```

---

## File Organization

```
models/
├── adapters/
│   ├── qlora_Qwen2.5-0.5B_custom_data_20260326_172841/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors  (8-20 MB)
│   │   ├── tokenizer.json
│   │   └── ...
│   ├── qlora_Qwen2.5-0.5B_custom_data_20260326_173512/  ← Second run
│   └── ...
├── merged/
│   └── qlora_Qwen2.5-0.5B_custom_data_20260326_172841/
├── logs/
│   └── qlora_Qwen2.5-0.5B_custom_data_20260326_172841.json
└── gguf/
    └── qlora_Qwen2.5-0.5B_custom_data_20260326_172841.gguf
```

---

## Migration from Old Names

If you have old runs with simple names:

```bash
# Old structure (at risk of overwriting)
models/adapters/
├── lora_0.5B/
└── dpo_0.5B/

# Rename to add timestamp
mv models/adapters/lora_0.5B models/adapters/lora_0.5B_legacy_20260326
mv models/adapters/dpo_0.5B models/adapters/dpo_0.5B_legacy_20260326
```

---

## Best Practices

### ✅ DO

```bash
# Let it auto-generate names
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/my_data.json

# Use descriptive custom names
python scripts/finetune.py --method qlora --vram 8 \
    --dataset ./data/support.json \
    --name support_bot_v1_final

# Keep multiple runs
models/adapters/
├── support_bot_v1_20260326_120000/
├── support_bot_v2_20260326_130000/
└── support_bot_v3_20260326_140000/
```

### ❌ DON'T

```bash
# Don't use generic names (will overwrite!)
python scripts/finetune.py --method qlora --vram 8 \
    --name my_model  # ← Will overwrite on next run!

# Don't use same name twice
python scripts/finetune.py --method qlora --name test
python scripts/finetune.py --method qlora --name test  # ← Overwrites first!
```

---

## Summary

**Fixed:** All scripts now use auto-generated unique names

**Pattern:** `{method}_{model}_{dataset}_{timestamp}`

**Benefits:**
- ✅ No accidental overwriting
- ✅ Easy to identify runs
- ✅ Compare different models/datasets
- ✅ Reproducible experiments

**Scripts Updated:**
- `finetune.py` ✅
- `test_dpo.py` ✅
- `test_rlhf.py` ✅

---

**Date:** March 26, 2026  
**Issue:** Simple names caused overwriting  
**Solution:** Auto-generated unique names with timestamps
