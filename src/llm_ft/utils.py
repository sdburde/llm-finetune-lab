"""
Utility functions for LLM fine-tuning.

This module provides common utilities for:
- Package installation
- GPU detection and memory management
- Model merging and export
- Ollama integration
- Training visualization
"""

import subprocess
import sys
import os
import gc
from typing import Optional, Tuple, List, Dict, Any
import torch


def pip_install(package: str) -> bool:
    """
    Install a Python package silently.
    
    Args:
        package: Package name with optional version specifier
        
    Returns:
        True if installation succeeded, False otherwise
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        capture_output=True,
        text=True
    )
    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} {package}")
    return result.returncode == 0


def install_all(extras: Optional[List[str]] = None) -> None:
    """
    Install all required dependencies for LLM fine-tuning.
    
    Args:
        extras: Optional list of additional packages to install
    """
    packages = [
        "transformers>=4.40.0",
        "peft>=0.12.0",
        "datasets",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",
        "trl>=0.8.6",
        "scipy",
        "matplotlib",
    ]
    if extras:
        packages.extend(extras)
    
    print("Installing dependencies...")
    for pkg in packages:
        pip_install(pkg)
    print("All dependencies installed.\n")


def detect_gpu() -> Tuple[str, float, bool, torch.dtype]:
    """
    Detect GPU availability and capabilities.
    
    Returns:
        Tuple of (device, vram_gb, bf16_supported, compute_dtype)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram = 0.0
    bf16 = False
    dtype = torch.float32
    
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
    
    print(f"Device        : {device}")
    if device == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"VRAM          : {vram:.1f} GB")
    print(f"Compute dtype : {'bfloat16' if bf16 else 'float16'}")
    
    return device, vram, bf16, dtype


def qlora_bnb_config(compute_dtype: torch.dtype) -> "BitsAndBytesConfig":
    """
    Create BitsAndBytesConfig for 4-bit QLoRA.
    
    Args:
        compute_dtype: Data type for computation (typically bfloat16 or float16)
        
    Returns:
        BitsAndBytesConfig object for 4-bit quantization
    """
    from transformers import BitsAndBytesConfig
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def prepare_for_training(model: torch.nn.Module) -> torch.nn.Module:
    """
    Prepare a model for training with gradient checkpointing.
    Replaces deprecated prepare_model_for_kbit_training.
    
    Args:
        model: The model to prepare
        
    Returns:
        The prepared model
    """
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    return model


def vram_snapshot(label: str = "") -> None:
    """
    Print current and peak VRAM usage.
    
    Args:
        label: Optional label for the snapshot
    """
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1e9
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  VRAM {label}: allocated={used:.2f}GB  peak={peak:.2f}GB")


def merge_and_save(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizer",
    out_dir: str,
    compute_dtype: torch.dtype
) -> None:
    """
    Merge LoRA adapter into base weights in-memory.
    Avoids the PeftModel.from_pretrained torch.distributed bug.
    
    Args:
        model: Trained PEFT model
        tokenizer: Model tokenizer
        out_dir: Directory to save merged model
        compute_dtype: Compute dtype used during training
    """
    print(f"\nMerging adapter in-memory → {out_dir}")
    model.eval()
    
    # Bake adapter weights into base model
    merged = model.merge_and_unload()
    merged = merged.to("cpu")  # Move off GPU
    merged.config.torch_dtype = torch.float16  # Safe metadata only
    
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    
    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Merged model saved → {out_dir}")


def convert_to_gguf(
    hf_dir: str,
    out_file: str,
    quant: str = "q4_k_m"
) -> bool:
    """
    Convert HuggingFace model to GGUF format using llama.cpp.
    
    Args:
        hf_dir: Path to HuggingFace model directory
        out_file: Output GGUF file path
        quant: Quantization type (q4_k_m, q5_k_m, q8_0, etc.)
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    # Clone llama.cpp if needed
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git",
             "--depth=1", "--quiet"],
            capture_output=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r",
             "llama.cpp/requirements.txt"],
            capture_output=True
        )
    
    print(f"Converting to GGUF ({quant})...")
    result = subprocess.run(
        [sys.executable, "llama.cpp/convert_hf_to_gguf.py",
         hf_dir, "--outfile", out_file, "--outtype", quant],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and os.path.exists(out_file):
        size = os.path.getsize(out_file) / 1e9
        print(f"GGUF created  : {out_file}  ({size:.2f} GB)")
        return True
    else:
        print("GGUF conversion failed:")
        print((result.stdout + result.stderr)[-2000:])
        return False


def write_modelfile(
    gguf_path: str,
    model_tag: str,
    system_prompt: str,
    stop_words: List[str],
    temperature: float = 0.7,
    num_ctx: int = 2048,
    filename: str = "Modelfile"
) -> str:
    """
    Write Ollama Modelfile for model registration.
    
    Args:
        gguf_path: Path to GGUF model file
        model_tag: Ollama model tag name
        system_prompt: System prompt for the model
        stop_words: List of stop sequences
        temperature: Sampling temperature
        num_ctx: Context window size
        filename: Output Modelfile path
        
    Returns:
        Path to created Modelfile
    """
    stops = "\n".join(f'PARAMETER stop "{s}"' for s in stop_words)
    content = f"""FROM {gguf_path}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p 0.9
PARAMETER num_ctx {num_ctx}
PARAMETER num_predict 512
{stops}
"""
    with open(filename, "w") as f:
        f.write(content)
    
    print(f"Modelfile written → {filename}")
    return filename


def register_ollama(modelfile: str, model_tag: str) -> None:
    """
    Register model with Ollama (starts server if needed).
    
    Args:
        modelfile: Path to Modelfile
        model_tag: Ollama model tag name
    """
    import time
    
    print(f"\nRegistering with Ollama as '{model_tag}'...")
    
    # Start ollama serve if not running
    try:
        import requests
        requests.get("http://localhost:11434", timeout=2)
    except Exception:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(6)
    
    result = subprocess.run(
        ["ollama", "create", model_tag, "-f", modelfile],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅  Model '{model_tag}' registered successfully")
    else:
        print(f"⚠️  ollama create failed — run manually:")
        print(f"    ollama create {model_tag} -f {modelfile}")
    
    print(f"""
── To run your model ─────────────────────────────
  ollama run {model_tag}
  # or via Python:
  import requests, json
  r = requests.post("http://localhost:11434/api/generate",
      json={{"model": "{model_tag}", "prompt": "Hello!", "stream": False}})
  print(r.json()["response"])
──────────────────────────────────────────────────""")


def plot_loss(
    trainer: Any,
    save_path: str,
    title: str = "Training Loss"
) -> None:
    """
    Plot training loss curve from trainer history.
    
    Args:
        trainer: Trainer object with state.log_history
        save_path: Path to save the plot
        title: Plot title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    hist = trainer.state.log_history
    steps = [x["step"] for x in hist if "loss" in x]
    losses = [x["loss"] for x in hist if "loss" in x]
    
    if not steps:
        print("No loss data to plot.")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker="o", linewidth=2, color="#5B8DD9")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    
    if len(losses) >= 2:
        pct = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Loss  {losses[0]:.4f} → {losses[-1]:.4f}  ({pct:.1f}% drop)")
        print(f"{'✅ Good' if pct > 20 else '⚠️  Modest' if pct > 5 else '❌ Low'}")
    
    print(f"Curve saved → {save_path}")
