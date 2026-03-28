# LLM Fine-Tuning Toolkit - Docker Image
# Optimized for 8GB VRAM GPUs

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir gradio>=4.0.0

# Copy application code
COPY . .

# Install package
RUN pip3 install -e .

# Create directories for models, data, and test results
RUN mkdir -p /app/models /app/data /app/test_results

# Set volume mounts
VOLUME ["/app/models", "/app/data", "/app/test_results"]

# Expose ports
EXPOSE 7860 8000

# Default command - start Web UI
CMD ["python3", "app/gradio_app.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1
