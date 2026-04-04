# ================================
# Base: CUDA + cuDNN (dev needed for bitsandbytes)
# ================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ================================
# Environment
# ================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

# ================================
# System deps (minimal)
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Upgrade pip (important for resolver stability)
# ================================
RUN pip3 install --upgrade pip setuptools wheel

# ================================
# Core numerical stack
# ================================
RUN pip3 install --no-cache-dir "numpy<2.0"

# ================================
# Install PyTorch (GPU, pinned)
# ================================
RUN pip3 install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ================================
# ML stack (STRICT CONTROL)
# ================================
RUN pip3 install --no-cache-dir --no-deps \
    transformers==4.40.2 \
    peft==0.12.0 \
    datasets==2.18.0 \
    accelerate==0.27.2 \
    trl==0.8.6 \
    bitsandbytes==0.43.1 \
    sentencepiece==0.2.1 \
    protobuf==4.25.3 \
    scipy==1.11.4 \
    matplotlib==3.8.4 \
    pyyaml==6.0.1 \
    tqdm==4.67.1 \
    psutil==5.9.8

# ================================
# Core deps required by ML stack
# ================================
RUN pip3 install --no-cache-dir \
    packaging \
    regex \
    safetensors \
    huggingface-hub==0.23.4 \
    tokenizers==0.19.1 \
    filelock \
    requests \
    pandas \
    pyarrow \
    xxhash \
    multiprocess \
    httpx \
    dill \
    pyarrow-hotfix

# ================================
# Web/UI stack (LET PIP HANDLE DEPS)
# ================================
RUN pip3 install --no-cache-dir \
    gradio==4.44.1 \
    python-multipart \
    pydub \
    aiohttp \
    fastapi==0.111.0 \
    uvicorn \
    jinja2==3.1.4 \
    anyio \
    markupsafe==2.1.5 \
    starlette==0.37.2

# ================================
# Copy application code
# ================================
WORKDIR /app
COPY src/ /app/src/
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY data/ /app/data/

# ================================
# Environment
# ================================
ENV PYTHONPATH=/app