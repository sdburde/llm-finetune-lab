#!/usr/bin/env python3
import torch

def check_gpu() -> str:
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used  = torch.cuda.memory_allocated(0) / 1e9
        bf16  = "bf16 ✓" if torch.cuda.is_bf16_supported() else "fp16 only"
        return f"✅ {name}  |  {total:.1f} GB total  |  {used:.2f} GB used  |  {bf16}"
    return "❌ No GPU detected — running on CPU (very slow)"


check_gpu()