# 🐳 Docker GPU Troubleshooting Guide

## ⚠️ GPU Not Detected in Docker

If you see this message:
```
⚠️  No GPU detected - using CPU (slow)
```

It means Docker cannot access your GPU. Here's how to fix it.

---

## ✅ Step 1: Verify NVIDIA Docker Toolkit Installation

### Check if NVIDIA Docker is installed:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage | GPU-Util  Compute M.   |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8     7W /  45W |    123MiB /  8192MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**If you get an error:** NVIDIA Docker Toolkit is not installed.

### Install NVIDIA Docker Toolkit:

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

---

## ✅ Step 2: Verify GPU Access

### Test GPU access in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

**If this works:** GPU access is configured correctly  
**If this fails:** GPU access is not working (see Step 1)

---

## ✅ Step 3: Run Docker Compose with GPU

### Start with GPU:

```bash
docker compose up
```

**Important:** Use `docker compose` (v2), NOT `docker-compose` (v1)

### Verify GPU in container:

```bash
docker exec llm-finetuning nvidia-smi
```

**Expected:** Should show your GPU info

---

## ✅ Step 4: Alternative - Use Docker Run Directly

If docker compose still doesn't detect GPU, try direct docker run:

```bash
docker run --rm -it \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  llm-fine-tuning:latest \
  python app/gradio_app.py
```

---

## 🔍 Common Issues

### Issue 1: "runtime: nvidia" not found

**Error:**
```
error during connect: unknown or invalid runtime: nvidia
```

**Fix:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Issue 2: Permission denied

**Error:**
```
docker: permission denied while trying to connect to the Docker daemon socket
```

**Fix:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login again, or:
newgrp docker
```

### Issue 3: GPU works in docker run but not compose

**Fix:** Add `--gpus all` to compose command:

```bash
docker compose --gpu all up
```

Or edit docker-compose.yml to use explicit device_ids:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Explicit GPU ID
          capabilities: [gpu]
```

### Issue 4: Multiple GPUs

**Select specific GPU:**

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # First GPU
```

**Use all GPUs:**

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## 📊 Verify GPU is Working

### Check in container:

```bash
# Exec into container
docker exec -it llm-finetuning bash

# Check GPU
nvidia-smi

# Or run Python check
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

### Expected output in Web UI:

```
✅ NVIDIA GeForce RTX 3070 Ti Laptop GPU | 8.2 GB total | 0.00 GB used | bf16 ✓
```

**NOT:**
```
❌ No GPU detected — running on CPU (very slow)
```

---

## 🚀 Quick Test

Run this one-liner to verify everything works:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base bash -c "nvidia-smi && echo '✅ GPU working in Docker!'"
```

**If you see GPU info + "✅ GPU working in Docker!":** Everything is configured correctly!

---

## 📝 Notes

- Docker Compose v2 (`docker compose`) is recommended over v1 (`docker-compose`)
- The `version: '3.8'` attribute is obsolete in Compose v2 (can be removed)
- GPU access requires NVIDIA Driver 450.80.02 or later
- Docker Desktop for Windows/Mac has different GPU passthrough requirements

---

**Last Updated:** March 29, 2026  
**Tested On:** Ubuntu 22.04, RTX 3070 Ti Laptop
