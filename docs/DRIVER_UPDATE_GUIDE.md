# 🚨 GPU Driver Too Old - Quick Fix

## Your Issue

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old 
(found version 12080). Please update your GPU driver
```

**Your driver version:** 12080 (CUDA 12.0)  
**Required for Docker:** CUDA 12.1+ driver

---

## ✅ Solution 1: Update NVIDIA Driver (Recommended)

### On Ubuntu:

```bash
# 1. Check current driver
nvidia-smi

# 2. Remove old drivers
sudo apt-get --purge remove "*nvidia*"

# 3. Add graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 4. Find recommended driver
ubuntu-drivers devices

# 5. Install recommended driver (usually nvidia-driver-535 or 550)
sudo apt install nvidia-driver-535

# 6. Reboot
sudo reboot

# 7. Verify after reboot
nvidia-smi
# Should show CUDA Version: 12.2 or higher
```

### Download from NVIDIA:

https://www.nvidia.com/Download/index.aspx

Select:
- Product Type: GeForce
- Product Series: GeForce 30 Series (Notebooks)
- Product: GeForce RTX 3070 Ti Laptop GPU
- OS: Your Linux version
- Download Type: Production Branch/Studio

---

## ✅ Solution 2: Use CUDA 12.0 (Temporary)

If you **cannot** update the driver, use CUDA 12.0:

```bash
# Stop container
docker compose down

# Build with CUDA 12.0 Dockerfile
docker build -f Dockerfile.cuda12.0 -t llm-fine-tuning:latest .

# Run container
docker run --rm -it --gpus all \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  llm-fine-tuning:latest
```

**Note:** This is a temporary workaround. Updating your driver is recommended.

---

## 🔍 Check Driver Compatibility

| Driver Version | CUDA Support | Status |
|---------------|--------------|--------|
| **12080** | CUDA 12.0 | ⚠️ Too old for Docker |
| **122xx** | CUDA 12.2+ | ✅ Works |
| **123xx+** | CUDA 12.3+ | ✅ Best |

---

## 🚀 Quick Test After Update

```bash
# 1. Verify driver update
nvidia-smi
# Should show CUDA Version: 12.2 or higher

# 2. Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi

# 3. Start container
docker compose up

# 4. Check logs
# Should see: ✅ NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

---

## 📝 Why This Happens

- **Docker containers** need GPU driver support on the host
- **CUDA 12.1+ containers** require driver version 122xx or higher
- **Your driver (12080)** is from CUDA 12.0 era
- **Solution:** Update driver OR use older CUDA container

---

## 🎯 Recommended Action

**Update your NVIDIA driver to version 535 or 550.**

This will:
- ✅ Fix Docker GPU detection
- ✅ Enable CUDA 12.1+ containers
- ✅ Improve GPU performance
- ✅ Fix compatibility issues

**Time required:** 10-15 minutes

---

**Last Updated:** March 31, 2026  
**Your Driver:** 12080 (too old)  
**Recommended:** 535.xx or 550.xx
