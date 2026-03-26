#!/usr/bin/env python3
"""
Environment Check Script for LLM Fine-Tuning
Checks: Python, CPU, RAM, GPU, VRAM, CUDA, and recommends best fine-tuning method

Usage:
    python scripts/check_env.py
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_python():
    """Check Python version and architecture."""
    print_section("PYTHON")
    print(f"Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Check required version
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 9:
        print("✅ Python version OK (3.9+)")
        return True
    else:
        print("❌ Python 3.9+ required")
        return False


def check_cpu():
    """Check CPU information."""
    print_section("CPU")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    # Count CPU cores
    try:
        import psutil
        cores = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        print(f"Cores: {physical} physical / {cores} logical")
    except ImportError:
        print("⚠️  Install psutil for detailed CPU info: pip install psutil")
    
    return True


def check_ram():
    """Check system RAM."""
    print_section("RAM (System Memory)")
    
    try:
        import psutil
        ram = psutil.virtual_memory()
        total_gb = ram.total / (1024 ** 3)
        available_gb = ram.available / (1024 ** 3)
        used_percent = ram.percent
        
        print(f"Total: {total_gb:.1f} GB")
        print(f"Available: {available_gb:.1f} GB")
        print(f"Used: {used_percent}%")
        
        # Recommendations
        if total_gb >= 32:
            print("✅ Excellent RAM (32GB+)")
            ram_score = 5
        elif total_gb >= 16:
            print("✅ Good RAM (16GB+)")
            ram_score = 4
        elif total_gb >= 8:
            print("⚠️  Minimum RAM (8GB) - may need to reduce batch sizes")
            ram_score = 3
        else:
            print("❌ Low RAM (<8GB) - consider upgrading or use cloud")
            ram_score = 1
        
        return ram_score, available_gb
    except ImportError:
        print("⚠️  Install psutil: pip install psutil")
        return 3, 8.0


def check_cuda():
    """Check CUDA availability."""
    print_section("CUDA")
    
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"CUDA available: Yes")
            # Parse version
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    print(f"Version: {line.strip()}")
            return True
        else:
            print("❌ CUDA not found (nvcc not in PATH)")
            return False
    except FileNotFoundError:
        print("❌ CUDA not installed or not in PATH")
        return False


def check_gpu():
    """Check GPU and VRAM using PyTorch."""
    print_section("GPU (PyTorch)")
    
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install: pip install torch torchvision torchaudio")
        return 0, False
    
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        compute_cap = torch.cuda.get_device_capability()
        bf16_support = torch.cuda.is_bf16_supported()
        
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_total:.1f} GB total")
        print(f"VRAM Used: {vram_used:.2f} GB")
        print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"BFloat16 Support: {'Yes' if bf16_support else 'No'}")
        
        # GPU Score and recommendations
        if vram_total >= 24:
            print("✅ Excellent GPU (24GB+) - Can run all methods")
            gpu_score = 5
        elif vram_total >= 16:
            print("✅ Good GPU (16GB+) - Can run most methods")
            gpu_score = 4
        elif vram_total >= 12:
            print("⚠️  Decent GPU (12GB) - QLoRA recommended for 7B+")
            gpu_score = 3
        elif vram_total >= 8:
            print("⚠️  Minimum GPU (8GB) - QLoRA only for 7B+")
            gpu_score = 2
        elif vram_total >= 6:
            print("⚠️  Low VRAM (6GB) - Use small models only")
            gpu_score = 1
        else:
            print("❌ Very low VRAM (<6GB) - Consider cloud GPU")
            gpu_score = 0
        
        return gpu_score, True
    else:
        print("❌ CUDA not available in PyTorch")
        print("\n   If you have an NVIDIA GPU:")
        print("   1. Install NVIDIA drivers")
        print("   2. Install CUDA toolkit")
        print("   3. Install PyTorch with CUDA:")
        print("      pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return 0, False


def check_packages():
    """Check required Python packages."""
    print_section("PACKAGES")
    
    required = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "trl",
    ]
    
    optional = [
        "psutil",
        "matplotlib",
        "scipy",
        "pyyaml",
    ]
    
    missing_required = []
    missing_optional = []
    
    for pkg in required:
        try:
            __import__(pkg)
            mod = sys.modules[pkg]
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {pkg}: {version}")
        except ImportError:
            print(f"❌ {pkg}: NOT INSTALLED")
            missing_required.append(pkg)
    
    print("\nOptional packages:")
    for pkg in optional:
        try:
            __import__(pkg)
            mod = sys.modules[pkg]
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {pkg}: {version}")
        except ImportError:
            print(f"⚠️  {pkg}: NOT INSTALLED")
            missing_optional.append(pkg)
    
    return missing_required, missing_optional


def check_disk_space():
    """Check available disk space."""
    print_section("DISK SPACE")
    
    try:
        import os
        stat = os.statvfs("/")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        
        print(f"Total: {total_gb:.1f} GB")
        print(f"Free: {free_gb:.1f} GB")
        
        if free_gb >= 50:
            print("✅ Plenty of space")
            return 5
        elif free_gb >= 20:
            print("✅ Enough space for most models")
            return 4
        elif free_gb >= 10:
            print("⚠️  Limited space - manage model downloads carefully")
            return 3
        else:
            print("❌ Low disk space")
            return 1
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return 3


def get_recommendations(gpu_score: int, ram_score: int, has_gpu: bool):
    """Get fine-tuning method recommendations."""
    print_section("RECOMMENDATIONS")
    
    print("\n📊 Fine-Tuning Method Compatibility:\n")
    
    methods = {
        "LoRA (8-bit)": {
            "min_gpu": 3,
            "vram_needed": "12GB+",
            "description": "Standard LoRA with 8-bit weights",
        },
        "QLoRA (4-bit)": {
            "min_gpu": 2,
            "vram_needed": "8GB+",
            "description": "Quantized LoRA - best for consumer GPUs ⭐",
        },
        "SFT (Full FT)": {
            "min_gpu": 4,
            "vram_needed": "16GB+",
            "description": "Full fine-tuning - best quality",
        },
        "DPO": {
            "min_gpu": 4,
            "vram_needed": "16GB+",
            "description": "Direct Preference Optimization",
        },
        "RLHF (GRPO)": {
            "min_gpu": 5,
            "vram_needed": "24GB+",
            "description": "Reinforcement Learning - most demanding",
        },
    }
    
    print(f"{'Method':<20} {'VRAM Needed':<15} {'Status':<10}")
    print("-" * 50)
    
    compatible = []
    for method, info in methods.items():
        status = "✅ Yes" if gpu_score >= info["min_gpu"] and has_gpu else "❌ No"
        if gpu_score >= info["min_gpu"] and has_gpu:
            compatible.append(method)
        print(f"{method:<20} {info['vram_needed']:<15} {status:<10}")
    
    print("\n" + "=" * 60)
    print("🎯 BEST METHOD FOR YOUR SYSTEM:")
    print("=" * 60)
    
    if not has_gpu:
        print("""
⚠️  NO GPU DETECTED!

Options:
1. Use CPU (very slow) - only for testing with tiny models
2. Use cloud GPU: Google Colab, Kaggle, Lambda Labs, RunPod
3. Install NVIDIA drivers and CUDA if you have an NVIDIA GPU

Recommended cloud options:
- Google Colab (Free T4): Good for QLoRA on 1.5B-3B models
- Colab Pro (A100): Can run 7B-13B models
- RunPod/Lambda Labs: Rent A100/H100 by the hour
""")
    elif gpu_score <= 1:
        print("""
⚠️  LOW VRAM DETECTED

Recommended:
1. Use QLoRA with small models (0.5B-1.5B parameters)
2. Reduce max_length to 128
3. Use batch_size=1 with gradient accumulation
4. Consider cloud GPU for larger models

Run: python scripts/train.py --method qlora --vram 4 --model Qwen/Qwen2.5-0.5B-Instruct
""")
    elif gpu_score == 2:
        print("""
✅ 8GB GPU DETECTED

Recommended: QLoRA (4-bit quantization)

Best models for your system:
- Qwen2.5-1.5B-Instruct (fits easily)
- Phi-3-mini-3.8B (fits with QLoRA)
- Mistral-7B (tight but possible)

Run: python scripts/train.py --method qlora --vram 8
""")
    elif gpu_score == 3:
        print("""
✅ 12GB GPU DETECTED

Recommended: LoRA or QLoRA

You can run:
- LoRA on 7B models
- QLoRA on 13B models
- Most fine-tuning methods

Run: python scripts/train.py --method lora --vram 12
""")
    elif gpu_score >= 4:
        print("""
✅ HIGH-END GPU (16GB+)

You can run ALL fine-tuning methods!

Recommended for production:
- LoRA for most tasks (best speed/quality balance)
- SFT for domain adaptation
- DPO for alignment
- RLHF for maximum control

Run: python scripts/train.py --method lora --vram 16
""")
    
    return compatible


def create_model_storage():
    """Create proper model storage directories."""
    print_section("MODEL STORAGE")
    
    base_dirs = [
        "models/downloads",      # Downloaded models
        "models/checkpoints",    # Training checkpoints
        "models/adapters",       # LoRA adapters
        "models/merged",         # Merged models
        "models/gguf",           # GGUF exports
        "models/ollama",         # Ollama models
    ]
    
    print("Creating model storage directories...")
    for dir_path in base_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}/")
    
    # Create .gitignore in models folder
    gitignore = Path("models/.gitignore")
    gitignore.write_text("""# Model files (too large for git)
*.bin
*.pt
*.pth
*.ckpt
*.gguf
*.safetensors
**/*
""")
    print(f"  ✅ models/.gitignore created")
    
    print("\nModel storage structure:")
    print("""
models/
├── downloads/      # HuggingFace cached models
├── checkpoints/    # Training checkpoints
├── adapters/       # LoRA adapter weights
├── merged/         # Merged (adapter + base) models
├── gguf/           # GGUF format for Ollama
└── ollama/         # Ollama model registry
""")
    
    return True


def main():
    """Run all checks."""
    print("""
╔══════════════════════════════════════════════════════════╗
║     LLM Fine-Tuning Environment Check                    ║
║     Checks your system for LLM fine-tuning readiness     ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run checks
    check_python()
    check_cpu()
    ram_score, ram_available = check_ram()
    has_cuda = check_cuda()
    gpu_score, has_gpu = check_gpu()
    missing_req, missing_opt = check_packages()
    disk_score = check_disk_space()
    
    # Create model storage
    create_model_storage()
    
    # Get recommendations
    compatible_methods = get_recommendations(gpu_score, ram_score, has_gpu)
    
    # Summary
    print_section("SUMMARY")
    
    overall_score = (gpu_score + ram_score + disk_score) / 3
    
    print(f"""
System Score: {overall_score:.1f}/5.0

GPU:  {gpu_score}/5 - {'Excellent' if gpu_score >= 4 else 'Good' if gpu_score >= 3 else 'Limited' if gpu_score >= 2 else 'Insufficient'}
RAM:  {ram_score}/5 - {'Excellent' if ram_score >= 4 else 'Good' if ram_score >= 3 else 'Limited'}
Disk: {disk_score}/5 - {'Excellent' if disk_score >= 4 else 'Good' if disk_score >= 3 else 'Limited'}
""")
    
    if missing_req:
        print(f"\n⚠️  MISSING REQUIRED PACKAGES: {', '.join(missing_req)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_req)}")
        print("\nOr install all:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    if has_gpu and gpu_score >= 2:
        print("""
1. Test QLoRA fine-tuning (recommended for your system):
   python ollama-finetuning/02_QLoRA_Finetuning.py

2. Or use CLI with auto-config:
   python scripts/train.py --method qlora --vram {vram}

3. Check all tutorials:
   ls ollama-finetuning/
""".format(vram=int(gpu_score * 4)))
    else:
        print("""
1. Install required packages:
   pip install -r requirements.txt

2. Consider using cloud GPU:
   - Google Colab (free T4)
   - Kaggle Notebooks
   - RunPod, Lambda Labs (paid)

3. For CPU-only testing (slow):
   python scripts/train.py --method qlora --model Qwen/Qwen2.5-0.5B-Instruct
""")
    
    print("""
📚 Documentation: docs/README.md
💻 Notebooks: notebooks/
⚙️  Configs: configs/
""")


if __name__ == "__main__":
    main()
