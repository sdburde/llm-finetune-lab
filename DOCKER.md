# Docker Quick Start Guide

## Prerequisites

- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Docker Toolkit installed
- Docker and Docker Compose

## Installation

### 1. Install NVIDIA Docker

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Build Image

```bash
docker build -t llm-fine-tuning:latest .
```

### 3. Run with Docker Compose

```bash
# Start Web UI
docker-compose up -d

# Start API server only
docker-compose --profile api up -d
```

### 4. Access

- **Web UI**: http://localhost:7860
- **API**: http://localhost:8000

## Usage

### Run Training

```bash
# Via Web UI (recommended)
Open http://localhost:7860

# Via CLI
docker exec -it llm-finetuning python scripts/finetune.py \
    --method qlora \
    --vram 8 \
    --dataset ./data/custom_data.json
```

### Run Inference

```bash
docker exec -it llm-finetuning python scripts/infer.py \
    --model ./models/adapters/MODEL_NAME \
    --prompt "Hello!"
```

### View Logs

```bash
docker-compose logs -f
```

### Stop

```bash
docker-compose down
```

## Volume Mounts

| Container Path | Host Path | Purpose |
|----------------|-----------|---------|
| /app/models | ./models | Model storage |
| /app/data | ./data | Datasets |
| /app/configs | ./configs | Configurations |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| NVIDIA_VISIBLE_DEVICES | all | GPU selection |
| PYTHONUNBUFFERED | 1 | Unbuffered output |

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

### Permission denied

```bash
# Fix permissions
sudo chown -R $USER:$USER ./models ./data
```

### Out of memory

```bash
# Reduce VRAM limit in training
docker exec -it llm-finetuning python scripts/finetune.py --vram 6
```

## Production Deployment

### With specific GPU

```yaml
# docker-compose.prod.yml
services:
  llm-finetuning:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # Specific GPU
              capabilities: [gpu]
```

### With authentication

```bash
# Add nginx reverse proxy
# See nginx.conf example
```

### Scaling

```bash
# Multiple instances
docker-compose up -d --scale llm-finetuning=3
```
