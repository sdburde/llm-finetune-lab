# Quickstart Guide

Get started with LLM fine-tuning in 5 minutes!

---

## Step 1: Check Your System

```bash
# Run environment check
python scripts/check_env.py
```

This will tell you:
- GPU VRAM available
- System RAM
- Required packages installed
- **Recommended fine-tuning method for your hardware**

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6 GB | 12+ GB |
| RAM | 8 GB | 16+ GB |
| Storage | 20 GB | 50+ GB SSD |

---

## Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python scripts/test_install.py
```

---

## Step 3: Run Your First Fine-Tuning

### Option A: Quick Test (All Hardware)

```bash
# QLoRA on small model - works on 6-8GB VRAM
python scripts/finetune.py --method qlora --model Qwen/Qwen2.5-0.5B-Instruct --vram 6
```

### Option B: Standard Fine-Tuning (8GB+ VRAM)

```bash
# QLoRA on 1.5B model - recommended for 8GB VRAM
python scripts/finetune.py --method qlora --vram 8
```

### Option C: Use Pre-made Script

```bash
# Run the complete QLoRA script
python ollama-finetuning/02_QLoRA_Finetuning.py
```

---

## Step 4: Test Your Model

### Via Python
```bash
python scripts/infer.py --model ./models/merged/YOUR_MODEL --prompt "What is AI?"
```

### Via Ollama (Recommended)

After training, your model is automatically converted to GGUF. Register it:

```bash
# Create Modelfile (if not auto-created)
echo "FROM ./models/gguf/YOUR_MODEL.gguf

SYSTEM \"\"\"You are a helpful AI assistant.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048" > Modelfile

# Register with Ollama
ollama create my-model -f Modelfile

# Run your model!
ollama run my-model "Explain machine learning"
```

---

## Step 5: Experiment with Different Methods

### LoRA (12GB+ VRAM)
```bash
python scripts/finetune.py --method lora --vram 12 --rank 16
```

### DPO - Alignment (16GB+ VRAM)
```bash
python scripts/finetune.py --method dpo --beta 0.1 --vram 16
```

### Custom Configuration
```bash
# Edit config file
nano configs/my_config.yaml

# Run with config
python scripts/finetune.py --config configs/my_config.yaml
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce VRAM usage
python scripts/finetune.py --vram 4 --max-length 128 --rank 4
```

### Slow Training

1. Verify GPU is being used: `nvidia-smi`
2. Use smaller model: `--model Qwen/Qwen2.5-0.5B-Instruct`
3. Reduce batch size: `--batch-size 1`

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Next Steps

1. **Read the docs**: `docs/README.md`
2. **Try notebooks**: `notebooks/01_QLoRA_Tutorial.ipynb`
3. **Custom datasets**: See docs for JSON/CSV loading
4. **Deploy to production**: Export to GGUF + Ollama

---

## Command Reference

| Command | Description |
|---------|-------------|
| `python scripts/check_env.py` | Check system compatibility |
| `python scripts/test_install.py` | Test installation |
| `python scripts/finetune.py --method qlora` | QLoRA fine-tuning |
| `python scripts/finetune.py --method lora` | LoRA fine-tuning |
| `python scripts/finetune.py --method dpo` | DPO fine-tuning |
| `python scripts/merge.py --adapter PATH` | Merge adapter |
| `python scripts/convert.py --model PATH` | Convert to GGUF |
| `python scripts/infer.py --model PATH` | Run inference |

---

## Getting Help

- **Documentation**: `docs/README.md`
- **Examples**: `notebooks/`, `examples/`
- **Configs**: `configs/`
- **Issues**: https://github.com/sdburde/llm-fine-tuning/issues
