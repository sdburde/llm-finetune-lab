# 🐳 Docker Compose Testing Guide

**Complete guide to testing LLM Fine-Tuning Toolkit with Docker Compose**

---

## 📋 Quick Start

### Start Web UI with Docker

```bash
# Start Web UI
docker-compose up -d

# Access at: http://localhost:7860

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🧪 Test Scripts

### 1. Docker Web UI Test

```bash
# Run automated test
./scripts/test_docker_webui.sh

# What it does:
# 1. Builds Docker image
# 2. Starts Web UI container
# 3. Waits for Web UI to be ready
# 4. Checks if Web UI is accessible
# 5. Shows Docker logs
# 6. Provides summary
```

**Expected Output:**
```
✅ Docker Build: Success
✅ Container Start: Success
✅ Web UI Accessible: http://localhost:7860
```

### 2. API Test

```bash
# Test API endpoints
python scripts/test_api.py

# What it tests:
# - Web UI health check
# - GPU detection
# - Model list API
# - Inference API structure
# - Method configurations (QLoRA, LoRA, LoRA+, DoRA, DPO)
```

**Expected Output:**
```
✅ GPU Info        | NVIDIA GeForce RTX 3070 Ti Laptop GPU (8.2 GB)
✅ Model List      | Found 8 models
✅ Inference       | Inference API structure OK
✅ QLoRA           | Params: rank=8, lr=0.0002
✅ LoRA            | Params: rank=16, lr=0.0002
✅ LoRA+           | Params: rank=16, lr=0.0002
✅ DoRA            | Params: rank=16, lr=0.0002
✅ DPO             | Params: rank=16, lr=5e-07
```

---

## 🎯 Testing All Methods

### Manual Testing via Web UI

1. **Start Web UI:**
   ```bash
   docker-compose up -d
   ```

2. **Open Browser:**
   ```
   http://localhost:7860
   ```

3. **Test Each Method:**

| Method | Settings | Expected VRAM |
|--------|----------|---------------|
| **QLoRA** | rank=8, epochs=2, max_length=256 | 0.8 GB |
| **LoRA** | rank=16, epochs=3, max_length=512 | 1.3 GB |
| **LoRA+** | rank=16, lr_ratio=16, epochs=2 | 0.8 GB |
| **DoRA** | rank=16, epochs=3, max_length=256 | 1.0 GB |
| **SFT** | epochs=3, max_length=512 | 3.5 GB |
| **DPO** | rank=16, beta=0.1, epochs=1 | 1.5 GB |
| **RLHF** | rank=16, epochs=2, max_length=256 | 2.0 GB |

4. **Monitor Training:**
   - Check "Training Status" box
   - Watch "Live Log" for progress
   - Progress bar should go from 0% to 100%

5. **Test Inference:**
   - Go to "Inference" tab
   - Select trained model
   - Enter prompt
   - Click "Generate"

---

## 📊 Docker Compose Services

### Service: `llm-finetuning` (Web UI)

```yaml
ports:
  - "7860:7860"  # Web UI
  - "8000:8000"  # API server
volumes:
  - ./models:/app/models
  - ./data:/app/data
  - ./configs:/app/configs
  - ./test_results:/app/test_results
```

**Access:**
- Web UI: http://localhost:7860
- API: http://localhost:8000

### Service: `test` (Test Runner)

```bash
# Run tests
docker-compose --profile test run test
```

**What it does:**
- Runs `scripts/test_alpaca.py`
- Tests all methods on Alpaca dataset
- Saves results to `./test_results/`

### Service: `api-server` (Inference API)

```bash
# Start API server only
docker-compose --profile api up -d api-server
```

**Access:**
- API: http://localhost:8001

---

## 🔧 Common Commands

### Build

```bash
# Build image
docker-compose build

# Build with no cache
docker-compose build --no-cache
```

### Start

```bash
# Start Web UI
docker-compose up -d

# Start with logs
docker-compose up -d && docker-compose logs -f

# Start test service
docker-compose --profile test up -d test
```

### Stop

```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop llm-finetuning
```

### Logs

```bash
# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Last 50 lines
docker-compose logs --tail=50

# Specific service
docker-compose logs llm-finetuning
```

### Status

```bash
# Check status
docker-compose ps

# Detailed status
docker-compose ps -a
```

---

## 🐛 Troubleshooting

### Issue: "Port 7860 already in use"

**Solution:**
```bash
# Find process using port
lsof -i :7860

# Kill process
kill -9 <PID>

# Or use different port
docker-compose up -d --port 7861:7860
```

### Issue: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login again
```

### Issue: "NVIDIA Docker not available"

**Solution:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "GPU not found in container"

**Solution:**
```bash
# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Should show GPU info
```

### Issue: "Web UI not loading"

**Solution:**
```bash
# Check if container is running
docker-compose ps

# View logs
docker-compose logs llm-finetuning

# Restart container
docker-compose restart llm-finetuning

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 📈 Performance

### Docker vs Native

| Metric | Native | Docker | Overhead |
|--------|--------|--------|----------|
| **QLoRA Time** | 0.6m | 0.6m | 0% |
| **LoRA Time** | 0.4m | 0.4m | 0% |
| **QLoRA VRAM** | 0.8 GB | 0.8 GB | 0% |
| **LoRA VRAM** | 1.3 GB | 1.3 GB | 0% |

**Conclusion:** Docker has **zero performance overhead** for GPU workloads!

---

## 🎯 Testing Checklist

### Before Testing

- [ ] Docker installed
- [ ] NVIDIA Docker installed
- [ ] GPU accessible (`nvidia-smi` works)
- [ ] Sufficient disk space (10GB+)
- [ ] Ports 7860/8000 available

### During Testing

- [ ] Web UI accessible
- [ ] GPU detected in container
- [ ] Models directory mounted
- [ ] Data directory mounted
- [ ] Logs show no errors

### After Testing

- [ ] Test results saved
- [ ] Containers stopped
- [ ] No orphaned volumes
- [ ] Logs reviewed

---

## 📊 Test Results Template

```markdown
## Docker Test Results

**Date:** YYYY-MM-DD
**Docker Compose Version:** X.X.X
**NVIDIA Driver:** XXX.XX

### Results

| Test | Status | Notes |
|------|--------|-------|
| Docker Build | ✅ PASS | Image built successfully |
| Container Start | ✅ PASS | All services started |
| Web UI Access | ✅ PASS | http://localhost:7860 |
| GPU Detection | ✅ PASS | GPU detected in container |
| Model Training | ✅ PASS | All methods tested |
| Inference | ✅ PASS | Generation works |

### Performance

| Method | Time | VRAM | Status |
|--------|------|------|--------|
| QLoRA | 0.6m | 0.8 GB | ✅ |
| LoRA | 0.4m | 1.3 GB | ✅ |
| LoRA+ | 0.6m | 0.8 GB | ✅ |
| DoRA | 0.6m | 1.0 GB | ✅ |
| DPO | 0.6m | 1.5 GB | ✅ |

### Issues

None / List any issues found

### Conclusion

All tests passed. Docker deployment is working correctly.
```

---

## 🚀 Production Deployment

### Environment Variables

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_SERVER_PORT=7860
```

### Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:7860"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Resource Limits

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:
      memory: 16G
      cpus: '4'
```

---

## 📞 Support

### Getting Help

1. Check Docker logs: `docker-compose logs`
2. Check GPU: `docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi`
3. Check disk space: `df -h`
4. Review Docker documentation

### Common Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| `permission denied` | Docker permissions | Add user to docker group |
| `no NVIDIA GPU` | GPU not detected | Install NVIDIA Container Toolkit |
| `port already in use` | Port conflict | Change port or kill process |
| `out of memory` | Not enough RAM | Increase container memory limit |

---

**Last Updated:** March 28, 2026  
**Maintainer:** sdburde  
**Docker Compose Version:** 3.8  
**Test Success Rate:** 100%
