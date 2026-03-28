# 🌐 Web UI Guide - LLM Fine-Tuning Toolkit

**Interactive Gradio Dashboard for Training, Evaluation, and Inference**

---

## 📦 Installation

### Option 1: Install Gradio Only

```bash
pip install gradio>=4.0.0
```

### Option 2: Install All Dependencies

```bash
cd llm-fine-tuning
pip install -r requirements.txt
pip install gradio>=4.0.0
```

### Option 3: Use Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at: http://localhost:7860
```

---

## 🚀 Quick Start

### Start Web UI

```bash
cd llm-fine-tuning
python app/gradio_app.py
```

### Access in Browser

```
http://localhost:7860
```

**Default Port:** 7860  
**Custom Port:** `python app/gradio_app.py --port 8080`

---

## 🎯 Features

### 1. Training Dashboard

**Location:** Training Tab

**Features:**
- Select fine-tuning method (QLoRA, LoRA, SFT, DPO)
- Choose base model from dropdown
- Upload or select dataset
- Configure hyperparameters (rank, epochs, learning rate)
- Set VRAM limit
- Real-time training status
- Live loss monitoring

**Steps:**
1. Select method (e.g., QLoRA)
2. Choose model (e.g., Qwen/Qwen2.5-0.5B-Instruct)
3. Select dataset (or upload custom)
4. Adjust settings:
   - Rank: 8 (for 8GB VRAM)
   - Epochs: 2
   - Learning Rate: 0.0002
   - VRAM Limit: 8 GB
5. Click "🚀 Start Training"
6. Monitor progress in Status box

---

### 2. Inference Playground

**Location:** Inference Tab

**Features:**
- Select trained model
- Enter prompt
- Adjust generation settings
- View response
- Example prompts included

**Steps:**
1. Click "🔄 Refresh Models" to load available models
2. Select your trained model
3. Enter prompt or click example
4. Adjust settings:
   - Max Tokens: 256
   - Temperature: 0.7
5. Click "🚀 Generate"
6. View response in output box

**Example Prompts:**
```
- What is machine learning?
- Explain quantum computing in simple terms
- Write a Python function to calculate factorial
```

---

### 3. Model Evaluation

**Location:** Evaluation Tab

**Features:**
- Perplexity scoring
- BLEU/ROUGE metrics
- Model comparison
- Export reports

**Steps:**
1. Select model to evaluate
2. Choose evaluation dataset
3. Click "📊 Evaluate"
4. View metrics in results box

---

### 4. Model Browser

**Location:** Models Tab

**Features:**
- List all trained models
- View model details (size, date, method)
- Compare models
- Delete unused models

---

## ⚙️ Configuration

### VRAM Management

| VRAM Limit | Max Model | Recommended Method |
|------------|-----------|-------------------|
| 4 GB | 0.5B | QLoRA (r=4) |
| 6 GB | 1.5B | QLoRA (r=8) |
| **8 GB** | **7B** | **QLoRA (r=8)** ⭐ |
| 12 GB | 13B | LoRA+ (r=16) |
| 16 GB | 30B | DoRA (r=32) |
| 24 GB | 70B | Full FT |

### Settings Recommendations

#### For 8GB VRAM (RTX 3070 Ti Laptop)

```yaml
Method: QLoRA
Model: Qwen/Qwen2.5-0.5B-Instruct or 1.5B-Instruct
Rank: 8
Alpha: 32
Batch Size: 1
Gradient Accumulation: 8
Max Length: 256
Epochs: 2
Learning Rate: 0.0002
VRAM Limit: 8 GB
```

#### For 12GB VRAM (RTX 3060/4070)

```yaml
Method: LoRA+
Model: Mistral-7B-Instruct
Rank: 16
Alpha: 32
Batch Size: 2
Gradient Accumulation: 4
Max Length: 512
Epochs: 3
Learning Rate: 0.0002
VRAM Limit: 12 GB
```

#### For 16GB VRAM (RTX 4080)

```yaml
Method: DoRA
Model: Llama-3-8B-Instruct
Rank: 32
Alpha: 64
Batch Size: 2
Gradient Accumulation: 4
Max Length: 512
Epochs: 3
Learning Rate: 0.0002
VRAM Limit: 16 GB
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'gradio'"

**Solution:**
```bash
pip install gradio>=4.0.0
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce VRAM limit in UI (8 → 6 → 4 GB)
2. Reduce Max Length (256 → 128)
3. Reduce Rank (16 → 8 → 4)
4. Use QLoRA instead of LoRA

### Issue: "No models found"

**Solution:**
1. Train a model first
2. Click "🔄 Refresh Models" button
3. Check models/adapters/ directory

### Issue: "Training stuck at 0%"

**Solutions:**
1. Check GPU is detected: `nvidia-smi`
2. Reduce batch size to 1
3. Enable gradient checkpointing
4. Restart Web UI

### Issue: "Web UI not loading"

**Solutions:**
```bash
# Check if port is in use
lsof -i :7860

# Kill process if needed
kill -9 <PID>

# Restart Web UI
python app/gradio_app.py
```

---

## 🌐 Remote Access

### Access from Another Device

```bash
# Start with public URL
python app/gradio_app.py --share

# Or specify host
python app/gradio_app.py --host 0.0.0.0 --port 7860
```

**Public URL:** Generated automatically (expires in 72 hours)  
**Local Network:** `http://<YOUR_IP>:7860`

### Deploy on Server

```bash
# Use nohup for background
nohup python app/gradio_app.py &

# Or use screen/tmux
screen -S llm-ui
python app/gradio_app.py
# Ctrl+A, D to detach
```

---

## 📊 Monitoring

### Real-Time Metrics

**During Training:**
- Progress percentage
- Current loss value
- Loss trend (last 5 values)
- GPU info
- VRAM usage

**After Training:**
- Final loss
- Training time
- Samples per second
- Steps per second

### Logs Location

```
models/logs/
└── <RUN_NAME>.json
```

**Contents:**
- Training configuration
- Loss history
- Training time
- Final metrics

---

## 🎨 UI Customization

### Change Theme

```python
# In app/gradio_app.py
with gr.Blocks(theme=gr.themes.Soft()) as ui:
# Change to:
with gr.Blocks(theme=gr.themes.Base()) as ui:
```

**Available Themes:**
- `gr.themes.Soft()` - Default (rounded, soft colors)
- `gr.themes.Base()` - Minimal
- `gr.themes.Monochrome()` - Black & white
- `gr.themes.Glass()` - Glassmorphism

### Change Port

```bash
python app/gradio_app.py --port 8080
```

### Enable Authentication

```python
# Add to app/gradio_app.py
ui.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("admin", "password"),  # Add auth
    auth_message="LLM Fine-Tuning Toolkit"
)
```

---

## 📱 Mobile Access

The Web UI is mobile-responsive! Access from phone/tablet:

1. Start Web UI on computer
2. Find computer's IP: `ip addr show` (Linux) or `ipconfig` (Windows)
3. Open browser on mobile: `http://<COMPUTER_IP>:7860`

**Note:** Both devices must be on same network

---

## 🚀 Production Deployment

### With Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/llm-finetuning
server {
    listen 80;
    server_name llm.yourdomain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### With Docker

```bash
# Build
docker build -t llm-finetuning .

# Run
docker run --gpus all -p 7860:7860 \
  -v ./models:/app/models \
  -v ./data:/app/data \
  llm-finetuning
```

### With Docker Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📚 API Endpoints

The Web UI also exposes REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/generate` | POST | Generate text |
| `/v1/train` | POST | Start training |
| `/v1/status` | GET | Get training status |
| `/v1/evaluate` | POST | Evaluate model |

**Example API Call:**
```bash
curl http://localhost:7860/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test_qlora",
    "prompt": "Hello!",
    "max_tokens": 100
  }'
```

---

## 💡 Tips & Tricks

### 1. Save Configuration Presets

```yaml
# configs/webui_presets.yaml
8gb_qlora:
  method: QLoRA
  rank: 8
  max_length: 256
  vram_limit: 8

12gb_lora:
  method: LoRA
  rank: 16
  max_length: 512
  vram_limit: 12
```

### 2. Batch Training

Queue multiple training runs:
1. Train model 1
2. While training, prepare dataset for model 2
3. Start model 2 immediately after model 1 finishes

### 3. Model Comparison

Train same dataset with different methods:
- QLoRA (fast, low VRAM)
- LoRA (balanced)
- DoRA (best quality)

Compare results in Models tab.

### 4. Export Models

After training:
1. Go to Models tab
2. Select model
3. Click "Export to GGUF"
4. Use with Ollama

---

## 🎓 Learning Path

### Beginner (First Time)

1. Start with QLoRA preset
2. Use tiny dataset (5 samples)
3. Train for 1 epoch
4. Test inference
5. Understand the flow

### Intermediate

1. Try different methods (LoRA, DoRA)
2. Use custom dataset (500 samples)
3. Tune hyperparameters
4. Evaluate with metrics

### Advanced

1. Train multiple models
2. Compare methods
3. Export to production
4. Deploy with Docker

---

## 📞 Support

### Common Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| `CUDA OOM` | Out of VRAM | Reduce VRAM limit |
| `No models` | No trained models | Train a model first |
| `Import Error` | Missing dependency | `pip install gradio` |
| `Port in use` | 7860 busy | Use different port |

### Getting Help

1. Check logs: `models/logs/*.json`
2. Check GPU: `nvidia-smi`
3. Check processes: `ps aux | grep gradio`
4. Restart Web UI

---

## 🔐 Security

### Best Practices

1. **Don't expose to public internet** without auth
2. **Use authentication** for remote access
3. **Limit file uploads** to trusted users
4. **Monitor resource usage** during training

### Enable HTTPS

```bash
# With Let's Encrypt
certbot --nginx -d llm.yourdomain.com

# Or use Cloudflare Tunnel
cloudflared tunnel --url http://localhost:7860
```

---

**Last Updated:** March 28, 2026  
**Maintainer:** sdburde  
**Version:** 1.0.0
