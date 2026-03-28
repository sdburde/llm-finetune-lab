# 💾 VRAM & RAM Management Guide

**Complete guide to managing GPU VRAM and System RAM for LLM fine-tuning**

---

## 📊 Quick Reference

### VRAM Requirements by Model Size

| Model Size | FP16 | QLoRA (4-bit) | LoRA (8-bit) | Recommended GPU |
|------------|------|---------------|--------------|-----------------|
| **0.5B** | 1 GB | 0.5 GB | 1 GB | GTX 1060 (6GB) |
| **1.5B** | 3 GB | 1 GB | 2 GB | RTX 3060 (8GB) ⭐ |
| **3B** | 6 GB | 2 GB | 4 GB | RTX 3070 (8GB) |
| **7B** | 14 GB | 4 GB | 8 GB | RTX 3080 (12GB) |
| **13B** | 26 GB | 8 GB | 16 GB | RTX 4090 (24GB) |
| **30B** | 60 GB | 16 GB | 32 GB | A100 (40GB) |
| **70B** | 140 GB | 35 GB | 70 GB | 2×A100 (80GB) |

### RAM Requirements

| Model Size | Minimum RAM | Recommended RAM |
|------------|-------------|-----------------|
| 0.5B | 4 GB | 8 GB |
| 1.5B | 8 GB | 16 GB |
| 3B | 12 GB | 24 GB |
| 7B | 16 GB | 32 GB |
| 13B | 32 GB | 64 GB |
| 30B | 64 GB | 128 GB |
| 70B | 128 GB | 256 GB |

---

## 🎯 VRAM Optimization Techniques

### 1. Quantization (Biggest Savings)

| Method | VRAM Saved | Quality Loss | Speed Impact |
|--------|------------|--------------|--------------|
| **FP32 → FP16** | 50% | None | None |
| **FP16 → INT8** | 50% | ~1% | None |
| **FP16 → INT4** | 75% | ~2-5% | Minimal |
| **FP16 → NF4** | 75% | ~2% | Minimal ⭐ |

**Recommendation:** Always use NF4 for QLoRA

```python
# Enable 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Best quality
    bnb_4bit_use_double_quant=True,  # Extra 0.5GB savings
    bnb_4bit_compute_dtype=torch.float16,
)
```

### 2. Gradient Checkpointing

**VRAM Saved:** 60%  
**Speed Cost:** 30% slower

```python
# Enable in config
gradient_checkpointing = True
```

**How it works:**
- Without: Store all activations → Fast, high VRAM
- With: Recompute activations → Slower, low VRAM

### 3. Batch Size Reduction

**VRAM Saved:** 50% per halving

```python
# High VRAM usage
batch_size = 4  # Uses 4× VRAM

# Low VRAM usage
batch_size = 1  # Uses 1× VRAM

# Compensate with gradient accumulation
gradient_accumulation_steps = 4  # Same effective batch!
```

### 4. Sequence Length Reduction

**VRAM Saved:** Quadratic reduction (attention is O(n²))

| Max Length | VRAM Usage | Relative |
|------------|------------|----------|
| 1024 | 4.0 GB | 4× |
| 512 | 1.0 GB | 1× |
| 256 | 0.25 GB | 0.25× |
| 128 | 0.06 GB | 0.06× |

**Recommendation:** Start with 256, increase if VRAM allows

### 5. LoRA Rank Reduction

**VRAM Saved:** Linear with rank

| Rank | VRAM (LoRA) | Quality |
|------|-------------|---------|
| 64 | 2.0 GB | Best |
| 32 | 1.0 GB | Very Good |
| 16 | 0.5 GB | Good ⭐ |
| 8 | 0.25 GB | Okay |
| 4 | 0.12 GB | Basic |

---

## 📋 VRAM Management by GPU

### For 4GB VRAM (GTX 1060, MX450)

```yaml
# Ultra Low Memory Mode
model: <= 0.5B
method: QLoRA
rank: 4
max_length: 128
batch_size: 1
gradient_accumulation: 16
gradient_checkpointing: true
```

**Expected VRAM Usage:**
```
Model (4-bit):    0.25 GB
Activations:      0.50 GB
Gradients:        0.10 GB
Optimizer:        0.05 GB
Overhead:         0.50 GB
─────────────────────────
Total:            1.40 GB  ✅ (2.6 GB free)
```

### For 6GB VRAM (GTX 1060 6GB, RTX 2060)

```yaml
# Low Memory Mode
model: <= 1.5B
method: QLoRA
rank: 8
max_length: 192
batch_size: 1
gradient_accumulation: 12
gradient_checkpointing: true
```

**Expected VRAM Usage:**
```
Model (4-bit):    0.75 GB
Activations:      0.80 GB
Gradients:        0.30 GB
Optimizer:        0.20 GB
Overhead:         0.50 GB
─────────────────────────
Total:            2.55 GB  ✅ (3.45 GB free)
```

### For 8GB VRAM (RTX 3060, 3070 Ti Laptop) ⭐

```yaml
# Standard Mode (Tested!)
model: 0.5B - 1.5B
method: QLoRA
rank: 8
max_length: 256
batch_size: 1
gradient_accumulation: 8
gradient_checkpointing: true
```

**Expected VRAM Usage:**
```
Model (4-bit):    0.75 GB
Activations:      0.80 GB
Gradients:        0.30 GB
Optimizer:        0.20 GB
Overhead:         0.50 GB
Buffer:           0.65 GB
─────────────────────────
Total:            3.20 GB  ✅ (4.8 GB free!)
```

**For 7B Models:**
```yaml
model: 7B
method: QLoRA
rank: 8
max_length: 256
batch_size: 1
gradient_accumulation: 8
```

**Expected VRAM Usage:**
```
Model (4-bit):    3.50 GB
Activations:      1.00 GB
Gradients:        0.50 GB
Optimizer:        0.30 GB
Overhead:         0.50 GB
─────────────────────────
Total:            5.80 GB  ✅ (2.2 GB free)
```

### For 12GB VRAM (RTX 3080, 4070)

```yaml
# Enhanced Mode
model: <= 7B
method: LoRA+
rank: 16
max_length: 512
batch_size: 2
gradient_accumulation: 4
gradient_checkpointing: false  # Can disable!
```

**Expected VRAM Usage:**
```
Model (FP16):     14.0 GB  ❌ Too high!
# Use QLoRA instead:
Model (4-bit):    3.50 GB
Activations:      1.50 GB
Gradients:        0.60 GB
Optimizer:        0.40 GB
Overhead:         0.50 GB
─────────────────────────
Total:            6.50 GB  ✅ (5.5 GB free)
```

### For 16GB VRAM (RTX 4080)

```yaml
# High-End Mode
model: <= 13B
method: DoRA
rank: 32
max_length: 512
batch_size: 2
gradient_accumulation: 4
```

### For 24GB VRAM (RTX 4090, A10)

```yaml
# Enthusiast Mode
model: <= 30B
method: Full FT (or GaLore)
rank: 64
max_length: 1024
batch_size: 4
gradient_accumulation: 2
```

### For 40GB+ VRAM (A100)

```yaml
# Professional Mode
model: <= 70B
method: Full FT
rank: N/A (full)
max_length: 2048
batch_size: 8
gradient_accumulation: 1
```

---

## 🧠 System RAM Management

### Why RAM Matters

1. **Dataset Loading:** Large datasets need RAM
2. **Model Caching:** HuggingFace caches models in RAM
3. **Data Preprocessing:** Tokenization happens in RAM
4. **System Stability:** OS needs RAM too

### RAM Optimization

#### 1. Dataset Streaming

```python
# Instead of loading full dataset
dataset = load_dataset("alpaca")  # Loads all into RAM

# Stream from disk
dataset = load_dataset("alpaca", streaming=True)  # Loads on demand
```

**RAM Saved:** 90%+ for large datasets

#### 2. Reduce Workers

```python
# High RAM usage
dataset.map(process, num_proc=8)  # 8 processes

# Low RAM usage
dataset.map(process, num_proc=2)  # 2 processes
```

**RAM Saved:** ~1 GB per process

#### 3. Clear Cache

```python
import gc
import torch

# After each training run
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### 4. Monitor RAM Usage

```python
import psutil

ram = psutil.virtual_memory()
print(f"RAM: {ram.available / 1e9:.1f} GB available")
print(f"RAM: {ram.percent}% used")

# Warning if low
if ram.percent > 90:
    print("⚠️  Low RAM! Close other applications.")
```

---

## 🔧 Troubleshooting

### OOM (Out of Memory) Errors

#### GPU OOM

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Fixes (try in order):**

1. **Reduce batch size:**
   ```python
   batch_size = 1  # Was 2 or 4
   ```

2. **Reduce max length:**
   ```python
   max_length = 128  # Was 256 or 512
   ```

3. **Enable gradient checkpointing:**
   ```python
   gradient_checkpointing = True
   ```

4. **Use 4-bit quantization:**
   ```python
   load_in_4bit = True
   bnb_4bit_quant_type = "nf4"
   ```

5. **Reduce LoRA rank:**
   ```python
   rank = 8  # Was 16 or 32
   ```

6. **Increase gradient accumulation:**
   ```python
   gradient_accumulation_steps = 16  # Was 4
   # Same effective batch, less VRAM!
   ```

#### System RAM OOM

**Error:**
```
Killed (OOM)
```

**Fixes:**

1. **Close other applications**
2. **Reduce dataset workers:**
   ```python
   num_proc = 1  # Was 4 or 8
   ```
3. **Use streaming:**
   ```python
   dataset = load_dataset("...", streaming=True)
   ```
4. **Add swap space (Linux):**
   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## 📊 Monitoring Tools

### Real-Time VRAM Monitoring

```bash
# Watch VRAM usage
watch -n 1 nvidia-smi

# One-time check
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Python Monitoring

```python
import torch

def print_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Peak:      {peak:.2f} GB")
```

### Training with Monitoring

```python
# In training loop
for step, batch in enumerate(dataloader):
    # ... training code ...
    
    # Log VRAM every 10 steps
    if step % 10 == 0:
        vram = torch.cuda.memory_allocated(0) / 1e9
        print(f"Step {step}: VRAM = {vram:.2f} GB")
```

---

## 🎯 Best Practices

### Before Training

1. **Check available VRAM:**
   ```bash
   nvidia-smi
   ```

2. **Close other GPU applications:**
   - Browsers with hardware acceleration
   - Other ML training jobs
   - GPU mining software

3. **Clear GPU cache:**
   ```python
   torch.cuda.empty_cache()
   ```

### During Training

1. **Monitor VRAM:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Watch for spikes:**
   - Model loading: Brief spike (normal)
   - Checkpoint saving: Brief spike (normal)
   - Continuous high usage: May OOM soon

3. **Adjust on the fly:**
   - If VRAM > 90%: Reduce batch size
   - If VRAM < 50%: Can increase batch size

### After Training

1. **Clear memory:**
   ```python
   import gc
   import torch
   
   gc.collect()
   torch.cuda.empty_cache()
   ```

2. **Save and unload:**
   ```python
   model.save_pretrained("./output")
   del model
   torch.cuda.empty_cache()
   ```

---

## 📈 VRAM Budget Calculator

### Quick Calculation

```
Total VRAM Needed = Model + Activations + Gradients + Optimizer + Overhead

For QLoRA 1.5B:
  Model (4-bit):     1.5B × 0.5 bytes = 0.75 GB
  Activations:       Batch × Seq² × Hidden = 0.80 GB
  Gradients:         Params × 4 bytes = 0.30 GB
  Optimizer:         Params × 8 bytes = 0.20 GB
  Overhead:          CUDA + buffers = 0.50 GB
  Buffer:            Safety margin = 0.65 GB
  ─────────────────────────────────────────────
  Total:             3.20 GB
```

### Formula Reference

```
Model VRAM (FP16)  = Parameters × 2 bytes
Model VRAM (INT4)  = Parameters × 0.5 bytes
Activations        = Batch × Sequence_Length² × Hidden_Size × 4 bytes
Gradients          = Trainable_Params × 4 bytes
Optimizer (Adam)   = Trainable_Params × 8 bytes
LoRA Params        = 2 × Rank × Hidden_Size × Num_Layers
```

---

## 🎓 Quick Decision Tree

```
Start: How much VRAM do you have?
│
├─ 4 GB?
│  └─→ QLoRA, 0.5B model, r=4, seq=128
│
├─ 6 GB?
│  └─→ QLoRA, 1.5B model, r=8, seq=192
│
├─ 8 GB?  ⭐ (Most common)
│  └─→ QLoRA, 1.5B-7B model, r=8, seq=256
│
├─ 12 GB?
│  └─→ LoRA+, 7B model, r=16, seq=512
│
├─ 16 GB?
│  └─→ DoRA, 13B model, r=32, seq=512
│
└─ 24 GB+?
   └─→ Full FT, 30B+ model, r=64, seq=1024
```

---

## 📚 Additional Resources

- [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface)
- [HuggingFace Memory Guide](https://huggingface.co/docs/transformers/memory)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

---

**Last Updated:** March 28, 2026  
**Tested On:** RTX 3070 Ti Laptop (8GB VRAM)  
**Maintainer:** sdburde
