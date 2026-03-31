# CUDA Version Compatibility Guide

## ⚠️ CUDA 13.x Issue

If you see this error:
```
Error 804: forward compatibility was attempted on non supported HW
```

**Cause:** CUDA 13.x is too new for your GPU.

**Affected GPUs:**
- RTX 30xx Laptop GPUs (3060, 3070, 3070 Ti, 3080)
- Some RTX 30xx Desktop GPUs
- Older GPUs (GTX 16xx, RTX 20xx)

**Solution:** Use CUDA 12.x instead.

---

## ✅ Recommended CUDA Versions

| GPU Series | Recommended CUDA | Status |
|------------|-----------------|--------|
| **RTX 30xx Laptop** | 12.1.1 | ✅ Works |
| **RTX 30xx Desktop** | 12.1.1 | ✅ Works |
| **RTX 40xx** | 12.1.1 or 13.x | ✅ Both work |
| **GTX 16xx** | 12.1.1 | ✅ Works |
| **RTX 20xx** | 12.1.1 | ✅ Works |

---

## 🔧 How to Fix

### 1. Update Dockerfile

Use CUDA 12.1.1 base image:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

**NOT:**
```dockerfile
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04  ❌
```

### 2. Rebuild Docker Image

```bash
# Stop existing container
docker compose down

# Rebuild with CUDA 12
docker compose build --no-cache

# Start container
docker compose up
```

### 3. Verify GPU Detection

In container logs, you should see:
```
✅ NVIDIA GeForce RTX 3070 Ti Laptop GPU | 8.2 GB total | bf16 ✓
```

**NOT:**
```
❌ No GPU detected — running on CPU (very slow)
```

---

## 🔍 Check Your GPU Compatibility

### Test CUDA 12.x:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi
```

### Test CUDA 13.x (will fail on RTX 30xx Laptop):
```bash
docker run --rm --gpus all nvidia/cuda:13.1.1-base nvidia-smi
```

If CUDA 13.x test fails with "forward compatibility" error, **use CUDA 12.x**.

---

## 📊 Why CUDA 13.x Fails

**Forward Compatibility Mode:**
- CUDA 13.x uses forward compatibility
- Requires newer GPU drivers
- RTX 30xx Laptop GPUs don't support this mode
- Results in "Error 804: forward compatibility was attempted on non supported HW"

**CUDA 12.x:**
- Uses standard compatibility mode
- Works with all CUDA-capable GPUs
- More stable for consumer GPUs

---

## 🚀 Quick Fix Commands

```bash
# 1. Stop container
docker compose down

# 2. Verify Dockerfile uses CUDA 12.1.1
grep "FROM nvidia" Dockerfile
# Should show: FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 3. Rebuild image
docker compose build --no-cache

# 4. Start container
docker compose up

# 5. Check logs for GPU detection
# Should see: ✅ NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

---

## 📝 Notes

- **Native PyTorch** (outside Docker) works fine with CUDA 13.x because it's compiled for your system
- **Docker containers** need matching CUDA versions
- **CUDA 12.1.1** is the safest choice for consumer GPUs
- **Laptop GPUs** often have compatibility issues with newest CUDA versions

---

**Last Updated:** March 31, 2026  
**Tested On:** RTX 3070 Ti Laptop  
**Working CUDA:** 12.1.1  
**Broken CUDA:** 13.1.1 (forward compatibility error)
