# LLM Fine-Tuning Documentation

## Quick Links

- [README](../README.md) - Main project overview with real test results
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step getting started
- [ALL_METHODS_COMPARISON.md](../ALL_METHODS_COMPARISON.md) - All 5 methods compared
- [VRAM_ANALYSIS.md](../VRAM_ANALYSIS.md) - Why we only use 3.2GB of 8GB
- [Installation](#installation) - Setup guide
- [Training Guide](#training-guide) - How to fine-tune
- [Troubleshooting](#troubleshooting) - Common issues

---

## Installation

### Prerequisites

- Python 3.9+
- GPU with 8GB+ VRAM (recommended)
- 8GB+ RAM

### Step-by-Step

```bash
# Clone repository
git clone https://github.com/sdburde/llm-fine-tuning.git
cd llm-fine-tuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## Training Guide

### Method 1: Quick Start (Recommended)

```bash
# QLoRA fine-tuning for 8GB VRAM
python ollama-finetuning/02_QLoRA_Finetuning.py
```

### Method 2: CLI Interface

```bash
# Auto-configured for your hardware
python scripts/train.py --method qlora --vram 8 --ram 8

# Custom configuration
python scripts/train.py \
    --method qlora \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --rank 8 \
    --epochs 2 \
    --lr 2e-4
```

### Method 3: Config File

```bash
# Use predefined config
python scripts/train.py --config configs/qlora_8gb.yaml

# Create your own config
cp configs/qlora_8gb.yaml configs/my_config.yaml
# Edit my_config.yaml, then:
python scripts/train.py --config configs/my_config.yaml
```

### Memory Settings

| VRAM | Model Size | Max Length | Batch | Rank |
|------|------------|------------|-------|------|
| 4GB | 0.5B | 128 | 1 | 4 |
| 6GB | 1.5B | 192 | 1 | 8 |
| 8GB | 1.5B | 256 | 1 | 8 |
| 12GB | 7B | 512 | 2 | 16 |
| 16GB | 7B | 512 | 4 | 32 |
| 24GB | 13B | 1024 | 4 | 64 |

---

## Inference Guide

### Using Ollama (Recommended)

```bash
# After training, your model is auto-registered
ollama run qwen-qlora "What is machine learning?"

# Or use Python API
python scripts/infer.py --ollama qwen-qlora --prompt "Explain AI"
```

### Local Inference

```bash
# Merge adapter with base model
python scripts/merge.py --adapter ./output/adapter --output ./merged

# Run inference
python scripts/infer.py --model ./merged --prompt "Hello!"
```

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./merged")
tokenizer = AutoTokenizer.from_pretrained("./merged")

messages = [{"role": "user", "content": "What is AI?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

## Troubleshooting

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `--max-length`: `--max-length 128`
2. Reduce batch size: `--batch-size 1`
3. Use QLoRA: `--method qlora`
4. Enable gradient checkpointing (default)

### Model Not Learning

**Problem:** Loss stays flat or increases

**Solutions:**
1. Lower learning rate: `--lr 1e-4`
2. Increase warmup: Edit config, set `warmup_ratio: 0.1`
3. Check data quality
4. Train longer: `--epochs 3`

### Slow Training

**Problem:** Training is very slow

**Solutions:**
1. Verify GPU is being used: Check `torch.cuda.is_available()`
2. Use smaller model: `--model Qwen/Qwen2.5-0.5B-Instruct`
3. Reduce max_length: `--max-length 128`
4. Use fewer samples: `--num-samples 200`

### Ollama Not Working

**Problem:** `ollama: command not found`

**Solutions:**
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Start server: `ollama serve`
3. Check: `ollama list`

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'xxx'`

**Solutions:**
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Advanced Topics

### Custom Datasets

```python
from datasets import load_dataset

# Load custom dataset
dataset = load_dataset("json", data_files="my_data.jsonl")
dataset = load_dataset("csv", data_files="my_data.csv")
```

### Multi-GPU Training

```bash
# Using accelerate
accelerate launch scripts/train.py --method qlora
```

### Export Formats

- **GGUF**: For Ollama, llama.cpp
- **Safetensors**: For HuggingFace Hub
- **PyTorch**: Standard .bin format

---

## Resources

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Ollama](https://ollama.ai)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
