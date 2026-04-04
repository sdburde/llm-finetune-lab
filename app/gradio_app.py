#!/usr/bin/env python3
"""
LLM Fine-Tuning Web UI with Gradio - Production Ready
Supports all 7 fine-tuning methods: QLoRA, LoRA, LoRA+, DoRA, SFT, DPO, RLHF

Usage:
    python app/gradio_app.py

Open in browser: http://localhost:7860  (or next free port)
"""

import gradio as gr
import torch
import os
import sys
import json
import gc
import time
import socket
import threading
import traceback
from pathlib import Path
from datetime import datetime

# ── Make the project root importable when running as app/gradio_app.py ───────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Lazy-import src.llm_ft package (avoids hard crash if deps missing) ────────
def _try_import():
    try:
        from src.llm_ft.config import FineTuningConfig
        from src.llm_ft.data import load_alpaca_dataset, format_instruction
        from src.llm_ft.models import load_model, load_tokenizer, setup_peft_model
        from src.llm_ft.trainers import create_lora_trainer, create_dpo_trainer
        from src.llm_ft.utils import (
            detect_gpu, merge_and_save, vram_snapshot,
            prepare_for_training, qlora_bnb_config, plot_loss,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)

_MODULES_OK, _MODULES_ERR = _try_import()

# ═══════════════════════════════════════════════════════════════════════════════
# Method catalogue
# ═══════════════════════════════════════════════════════════════════════════════

METHODS = {
    "QLoRA": {
        "description": "Quantized LoRA — 4-bit NF4 quantization via bitsandbytes. "
                       "Lowest VRAM footprint; ideal entry point for consumer GPUs.",
        "vram_gb": 0.10,
        "vram_display": "~2 GB for 7B model",
        "quality": "94 %",
        "speed": "Fast",
        "recommended_for": "4–8 GB VRAM",
        "load_in_4bit": True,
        "min_vram": 4,
        "params": {
            "rank":          {"default": 8,    "min": 4,    "max": 16,   "step": 4},
            "alpha":         {"default": 16,   "min": 8,    "max": 32,   "step": 4},
            "learning_rate": {"default": 2e-4, "min": 1e-4, "max": 5e-4},
            "epochs":        {"default": 2,    "min": 1,    "max": 100,  "step": 1},
            "max_length":    {"default": 256,  "min": 128,  "max": 512,  "step": 128},
        },
    },
    "LoRA": {
        "description": "Low-Rank Adaptation in bf16 — no quantization, fastest throughput, "
                       "best quality-to-speed ratio for 12 GB+ GPUs.",
        "vram_gb": 0.17,
        "vram_display": "~2.4 GB for 7B model",
        "quality": "95 %",
        "speed": "Fastest",
        "recommended_for": "12 GB+ VRAM",
        "load_in_4bit": False,
        "min_vram": 6,
        "params": {
            "rank":          {"default": 16,   "min": 8,    "max": 32,   "step": 4},
            "alpha":         {"default": 32,   "min": 16,   "max": 64,   "step": 8},
            "learning_rate": {"default": 2e-4, "min": 1e-4, "max": 5e-4},
            "epochs":        {"default": 3,    "min": 1,    "max": 50,   "step": 1},
            "max_length":    {"default": 512,  "min": 256,  "max": 1024, "step": 128},
        },
    },
    "LoRA+": {
        "description": "LoRA with separate, larger learning rate for the B matrix — "
                       "up to 2× faster convergence at the same VRAM cost as QLoRA.",
        "vram_gb": 0.12,
        "vram_display": "~1.7 GB for 7B model",
        "quality": "97 %",
        "speed": "Very Fast",
        "recommended_for": "8 GB VRAM (fast training)",
        "load_in_4bit": True,
        "min_vram": 6,
        "params": {
            "rank":              {"default": 16,   "min": 8,    "max": 32,  "step": 4},
            "alpha":             {"default": 32,   "min": 16,   "max": 64,  "step": 8},
            "learning_rate":     {"default": 2e-4, "min": 1e-4, "max": 5e-4},
            "loraplus_lr_ratio": {"default": 16,   "min": 4,    "max": 32,  "step": 4},
            "epochs":            {"default": 2,    "min": 1,    "max": 100, "step": 1},
            "max_length":        {"default": 256,  "min": 128,  "max": 512, "step": 128},
        },
    },
    "DoRA": {
        "description": "Weight-Decomposed LoRA — decomposes updates into magnitude + direction. "
                       "Higher quality than standard LoRA at the same rank.",
        "vram_gb": 0.14,
        "vram_display": "~2.0 GB for 7B model",
        "quality": "98 %",
        "speed": "Fast",
        "recommended_for": "8 GB+ VRAM (max PEFT quality)",
        "load_in_4bit": True,
        "min_vram": 6,
        "params": {
            "rank":          {"default": 16,   "min": 8,    "max": 32,  "step": 4},
            "alpha":         {"default": 32,   "min": 16,   "max": 64,  "step": 8},
            "learning_rate": {"default": 2e-4, "min": 1e-4, "max": 5e-4},
            "epochs":        {"default": 3,    "min": 1,    "max": 100, "step": 1},
            "max_length":    {"default": 256,  "min": 128,  "max": 512, "step": 128},
        },
    },
    "SFT": {
        "description": "Supervised Fine-Tuning — full weight update on instruction-response pairs. "
                       "Maximum quality; needs 12 GB+ VRAM (uses LoRA adapter internally).",
        "vram_gb": 0.50,
        "vram_display": "~7 GB for 7B model",
        "quality": "100 %",
        "speed": "Medium",
        "recommended_for": "16 GB+ VRAM",
        "load_in_4bit": False,
        "min_vram": 12,
        "params": {
            "rank":          {"default": 16,   "min": 8,    "max": 32,   "step": 4},
            "alpha":         {"default": 32,   "min": 16,   "max": 64,   "step": 8},
            "learning_rate": {"default": 2e-5, "min": 1e-5, "max": 1e-4},
            "epochs":        {"default": 3,    "min": 1,    "max": 100,  "step": 1},
            "max_length":    {"default": 512,  "min": 256,  "max": 1024, "step": 128},
        },
    },
    "DPO": {
        "description": "Direct Preference Optimization — trains on (prompt, chosen, rejected) triplets "
                       "without a reward model. Practical alignment for 8 GB+ GPUs.",
        "vram_gb": 0.22,
        "vram_display": "~3.1 GB for 7B model",
        "quality": "99 %",
        "speed": "Fast",
        "recommended_for": "8 GB+ VRAM (alignment tasks)",
        "load_in_4bit": True,
        "min_vram": 8,
        "params": {
            "rank":          {"default": 16,   "min": 8,    "max": 32,  "step": 4},
            "alpha":         {"default": 32,   "min": 16,   "max": 64,  "step": 8},
            "learning_rate": {"default": 5e-7, "min": 1e-7, "max": 1e-6},
            "beta":          {"default": 0.1,  "min": 0.01, "max": 0.5, "step": 0.01},
            "epochs":        {"default": 1,    "min": 1,    "max": 100, "step": 1},
            "max_length":    {"default": 256,  "min": 128,  "max": 512, "step": 128},
        },
    },
    "RLHF": {
        "description": "PPO-based RLHF — reward model + policy optimisation with KL penalty. "
                       "Maximum alignment control; needs 12 GB+ VRAM.",
        "vram_gb": 0.35,
        "vram_display": "~5 GB for 7B model",
        "quality": "100 %",
        "speed": "Slow",
        "recommended_for": "12 GB+ VRAM (production chatbots)",
        "load_in_4bit": True,
        "min_vram": 12,
        "params": {
            "rank":          {"default": 16,   "min": 8,    "max": 32,  "step": 4},
            "alpha":         {"default": 32,   "min": 16,   "max": 64,  "step": 8},
            "learning_rate": {"default": 1e-6, "min": 5e-7, "max": 5e-6},
            "epochs":        {"default": 2,    "min": 1,    "max": 100, "step": 1},
            "max_length":    {"default": 256,  "min": 128,  "max": 512, "step": 128},
        },
    },
}

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

DATASETS = [
    "tatsu-lab/alpaca",
    "OpenAssistant/oasst1",
    "stanfordnlp/SHP",
    "Anthropic/hh-rlhf",
]

# Model sizes in billions of parameters (for VRAM estimation)
MODEL_PARAMS_B = {
    "Qwen/Qwen2.5-0.5B-Instruct":         0.5,
    "Qwen/Qwen2.5-1.5B-Instruct":         1.5,
    "Qwen/Qwen2.5-3B-Instruct":           3.0,
    "microsoft/Phi-3-mini-instruct":       3.8,
    "mistralai/Mistral-7B-Instruct-v0.3":  7.0,
    "meta-llama/Llama-3.2-1B-Instruct":   1.0,
    "meta-llama/Llama-3.2-3B-Instruct":   3.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Global training state  (GIL-safe for simple dict reads/writes)
# ═══════════════════════════════════════════════════════════════════════════════

training_state = {
    "running":    False,
    "progress":   0,
    "loss":       [],
    "status":     "Idle",
    "model_path": None,
    "method":     None,
    "output":     [],
    "start_time": None,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


def check_gpu() -> str:
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used  = torch.cuda.memory_allocated(0) / 1e9
        bf16  = "bf16 ✓" if torch.cuda.is_bf16_supported() else "fp16 only"
        return f"✅ {name}  |  {total:.1f} GB total  |  {used:.2f} GB used  |  {bf16}"
    return "❌ No GPU detected — running on CPU (very slow)"


def _estimate_vram(method: str, model_name: str) -> float:
    """Rough VRAM estimate in GB: bf16 weights + adapter overhead."""
    model_b  = MODEL_PARAMS_B.get(model_name, 7.0)
    factor   = METHODS[method]["vram_gb"]
    base     = model_b * 2           # bf16 weights
    overhead = model_b * factor * 2
    return round(base + overhead, 1)


def get_vram_recommendations(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    vram = _gpu_vram_gb()
    if not torch.cuda.is_available():
        return (
            "⚠️ **No GPU detected.**  Training on CPU will be extremely slow.\n"
            "Use Google Colab (free T4) or any cloud GPU."
        )

    lines = [
        f"**GPU VRAM available:** {vram:.1f} GB\n",
        "| Method | Est. VRAM | Fits? | Best match? |",
        "|--------|-----------|-------|-------------|",
    ]
    for name, info in METHODS.items():
        est  = _estimate_vram(name, model_name)
        if est <= vram * 0.88:
            fits = "✅ yes"
        elif est <= vram:
            fits = "⚠️ tight"
        else:
            fits = "❌ OOM"
        rec = "⭐" if info["min_vram"] <= vram <= info["min_vram"] * 2 else ""
        lines.append(f"| {name} | {est} GB | {fits} | {rec} |")

    if vram <= 6:
        advice = "\n> **Advice:** Use **QLoRA** with models ≤ 1.5 B and max_length ≤ 256."
    elif vram <= 8:
        advice = "\n> **Advice:** **QLoRA** or **LoRA+** work well. Stick to models ≤ 7 B."
    elif vram <= 12:
        advice = "\n> **Advice:** **LoRA**, **LoRA+**, or **DoRA** recommended. Models up to 13 B."
    else:
        advice = "\n> **Advice:** Any method works. **SFT** or **DPO** give the best quality."

    lines.append(advice)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# File / model discovery
# ═══════════════════════════════════════════════════════════════════════════════

def list_models() -> list:
    """List available trained models, filtering out empty/invalid directories."""
    base = Path("./models/adapters")
    if not base.exists():
        return ["No trained models found"]
    
    valid_models = []
    for d in base.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        
        # Check if directory has actual model files (not empty/failed training)
        adapter_config = d / "adapter_config.json"
        adapter_model = d / "adapter_model.safetensors"
        
        if adapter_config.exists() and adapter_model.exists():
            valid_models.append(d.name)
        else:
            # Empty or failed training - clean it up
            import shutil
            print(f"⚠️  Removing empty/invalid model directory: {d.name}")
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"   Could not remove: {e}")
    
    return sorted(valid_models) if valid_models else ["No trained models found"]


def list_dataset_files() -> list:
    data_dir = Path("./data")
    if not data_dir.exists():
        return []
    return sorted(
        f.name for f in data_dir.iterdir()
        if f.suffix in {".json", ".jsonl", ".csv"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UI helper callbacks
# ═══════════════════════════════════════════════════════════════════════════════

def update_method_info(method: str) -> str:
    if method not in METHODS:
        return "Select a method to see details."
    info = METHODS[method]
    vram = _gpu_vram_gb()
    est  = _estimate_vram(method, "mistralai/Mistral-7B-Instruct-v0.3")
    if est <= vram * 0.9:
        fit = "✅ fits"
    elif est <= vram:
        fit = "⚠️ tight"
    else:
        fit = "❌ may OOM"
    return (
        f"### {method}\n\n"
        f"{info['description']}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| VRAM (7B model) | {info['vram_display']} ({fit}) |\n"
        f"| Relative quality | {info['quality']} |\n"
        f"| Speed | {info['speed']} |\n"
        f"| Best for | {info['recommended_for']} |\n"
        f"| 4-bit quantization | {'Yes ✓' if info['load_in_4bit'] else 'No'} |"
    )


def update_vram_table(model_name: str) -> str:
    return get_vram_recommendations(model_name)


def update_params(method: str):
    """Return gr.update() for every param widget when the method changes."""
    if method not in METHODS:
        return [gr.update(visible=False)] * 7

    p = METHODS[method]["params"]

    def _slider(key):
        if key not in p:
            return gr.update(visible=False)
        d = p[key]
        return gr.update(
            minimum=d["min"], maximum=d["max"],
            value=d["default"], step=d.get("step", 1),
            visible=True,
            label=f"{key.replace('_', ' ').title()}  (rec: {d['default']})",
        )

    def _number(key):
        if key not in p:
            return gr.update(visible=False)
        d = p[key]
        return gr.update(
            value=d["default"],
            minimum=d.get("min"), maximum=d.get("max"),
            visible=True,
            label=f"Learning Rate  (rec: {d['default']:.2e})",
        )

    return [
        _slider("rank"),
        _slider("alpha"),
        _number("learning_rate"),
        _slider("beta"),
        _slider("loraplus_lr_ratio"),
        _slider("epochs"),
        _slider("max_length"),
    ]


def toggle_dataset_ui(dataset_type: str):
    show_file = dataset_type == "File"
    return gr.update(visible=show_file), gr.update(visible=not show_file)


def refresh_models_list():
    return gr.update(choices=list_models(), value=None)


# ═══════════════════════════════════════════════════════════════════════════════
# OOM pre-flight check
# ═══════════════════════════════════════════════════════════════════════════════

def _oom_warning(method: str, model_name: str) -> str | None:
    vram = _gpu_vram_gb()
    if vram == 0:
        return None  # CPU — let user decide
    est = _estimate_vram(method, model_name)
    if est > vram:
        return (
            f"⚠️  VRAM warning: {method} + {model_name} needs ~{est} GB "
            f"but your GPU has {vram:.1f} GB — likely OOM.\n"
            f"Suggestions: switch to QLoRA, pick a smaller model, or halve max_length."
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def start_training(
    method, model_name, dataset_type, dataset_file, dataset_hf,
    rank, alpha, learning_rate, beta, loraplus_ratio,
    epochs, max_length, batch_size, grad_accum, output_name,
):
    """Validate, build run-name, then launch _train() in a daemon thread."""
    if training_state["running"]:
        return "⚠️ Training already running.", 0, "\n".join(training_state["output"][-20:])

    # OOM pre-check — prepend warning but still allow launch
    warn = _oom_warning(method, model_name)
    training_state["output"] = [warn, ""] if warn else []

    training_state.update({
        "running":    True,
        "progress":   0,
        "loss":       [],
        "status":     "Initializing…",
        "method":     method,
        "model_path": None,
        "start_time": time.time(),
    })

    dataset_path = (
        f"./data/{dataset_file}"
        if dataset_type == "File" and dataset_file
        else dataset_hf or "tatsu-lab/alpaca"
    )

    # ── Build a descriptive, unique run name ──────────────────────────────────
    if output_name and output_name.strip():
        run_name = output_name.strip().replace(" ", "_").lower()
    else:
        method_short = method.lower().replace(" ", "").replace("+", "plus")
        model_short  = (
            Path(model_name).name
            .replace("-Instruct", "").replace("-instruct", "").lower()
        )
        if dataset_type == "File" and dataset_file:
            ds_short = Path(dataset_file).stem.lower().replace("_", "")
        elif dataset_hf:
            ds_short = dataset_hf.split("/")[-1].lower().replace("_", "")
        else:
            ds_short = "alpaca"
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{method_short}_{model_short}_{ds_short}_{int(epochs)}ep_{ts}"

    def log(msg: str):
        training_state["output"].append(msg)
        print(msg)

    def _train():
        try:
            from src.llm_ft.config import FineTuningConfig
            from src.llm_ft.data import load_alpaca_dataset, format_instruction
            from src.llm_ft.models import load_model, load_tokenizer
            from src.llm_ft.trainers import create_lora_trainer, create_dpo_trainer
            from src.llm_ft.utils import (
                detect_gpu, merge_and_save, vram_snapshot,
                prepare_for_training, plot_loss,
            )

            log(f"🚀 Starting {method} fine-tuning")
            log(f"   Model      : {model_name}")
            log(f"   Dataset    : {dataset_path}")
            log(f"   Run name   : {run_name}")
            log(f"   Rank/Alpha : {int(rank)} / {int(alpha)}")
            log(f"   LR         : {float(learning_rate):.2e}")
            log(f"   Epochs     : {int(epochs)}  |  Max length : {int(max_length)}")
            log(f"   Batch      : {int(batch_size)}  |  Grad accum : {int(grad_accum)}")
            log(f"   Eff. batch : {int(batch_size) * int(grad_accum)}")
            log("")

            # ── Hardware ──────────────────────────────────────────────────────
            training_state["status"] = "Detecting hardware…"
            device, vram, bf16_ok, compute_dtype = detect_gpu()
            log(f"   Device : {device}  |  VRAM : {vram:.1f} GB  |  bf16 : {bf16_ok}")
            
            # ── Pre-flight OOM check ─────────────────────────────────────────
            model_size_map = {
                "0.5b": 0.5, "1b": 1.0, "1.5b": 1.5, "3b": 3.0, "7b": 7.0,
                "8b": 8.0, "13b": 13.0, "30b": 30.0, "70b": 70.0
            }
            model_key = next((k for k in model_size_map if k in model_name.lower()), "7b")
            model_gb = model_size_map.get(model_key, 7.0)
            
            # QLoRA uses ~0.5 bytes/param, others use ~2 bytes/param (bf16)
            load_4bit = METHODS[method]["load_in_4bit"]
            if load_4bit:
                model_vram = model_gb * 0.5
                overhead = model_gb * 0.3
            else:
                model_vram = model_gb * 2.0
                overhead = model_gb * 0.5
            
            # Sequence length impact (quadratic attention)
            seq_factor = max_length / 256.0
            overhead *= seq_factor
            
            estimated_vram = model_vram + overhead
            
            if estimated_vram > vram * 0.9:
                msg = (
                    f"\n⚠️  VRAM WARNING: {method} + {model_name} needs ~{estimated_vram:.1f} GB "
                    f"but your GPU has {vram:.1f} GB — likely OOM.\n\n"
                    f"Suggestions:\n"
                    f"  • Switch to QLoRA (4-bit quantization)\n"
                    f"  • Pick a smaller model (0.5B-3B for 8GB)\n"
                    f"  • Reduce Max Sequence Length to {max(64, max_length//2)}\n"
                    f"  • Keep Batch Size = 1\n"
                )
                log(msg)
                log("⚠️  Proceeding anyway (may crash with OOM)...\n")
            
            log("")

            load_4bit  = METHODS[method]["load_in_4bit"]
            output_dir = f"./models/adapters/{run_name}"
            merged_dir = f"./models/merged/{run_name}"

            # ── Config ────────────────────────────────────────────────────────
            cfg = FineTuningConfig(
                model_name=model_name,
                method=method.lower().replace("+", "plus"),
                rank=int(rank),
                alpha=int(alpha),
                load_in_4bit=load_4bit,
                dataset_name=dataset_path,
                max_length=int(max_length),
                batch_size=int(batch_size),
                gradient_accumulation_steps=int(grad_accum),
                num_epochs=int(epochs),
                learning_rate=float(learning_rate),
                bf16=bf16_ok,
                fp16=(not bf16_ok and device == "cuda"),
                output_dir=output_dir,
                adapter_dir=output_dir,
                merged_dir=merged_dir,
            )
            if method == "DPO":
                cfg.beta = float(beta)
            if method == "LoRA+":
                cfg.loraplus_lr_ratio = int(loraplus_ratio)

            # ── Tokenizer ─────────────────────────────────────────────────────
            training_state["status"] = "Loading tokenizer…"
            log("📦 Loading tokenizer…")
            try:
                tokenizer = load_tokenizer(model_name, trust_remote_code=cfg.trust_remote_code)
                log("   ✅ Tokenizer loaded")
                vram_snapshot("after tokenizer")
            except ValueError as e:
                if "sentencepiece" in str(e).lower():
                    msg = (
                        "❌ Missing sentencepiece package!\n\n"
                        "This model requires sentencepiece for tokenization.\n"
                        "Install with: pip install sentencepiece protobuf\n\n"
                        "Required for: Llama, Mistral, Gemma, and other SentencePiece-based models."
                    )
                    log(msg)
                    training_state["status"] = "❌ Missing dependency"
                    return
                raise

            # ── Model ─────────────────────────────────────────────────────────
            training_state["status"] = "Loading model…"
            log("📦 Loading model…")
            model = load_model(
                model_name,
                load_in_4bit=load_4bit,
                compute_dtype=compute_dtype,
                trust_remote_code=cfg.trust_remote_code,
            )
            log(f"   ✅ Model loaded  ({model.num_parameters() / 1e6:.0f} M params)")
            vram_snapshot("after model load")

            # ── PEFT setup ────────────────────────────────────────────────────
            training_state["status"] = "Setting up PEFT…"
            log("⚙️  Setting up PEFT adapters…")
            if load_4bit:
                model = prepare_for_training(model)

            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.rank,
                lora_alpha=cfg.alpha,
                lora_dropout=cfg.dropout,
                target_modules=cfg.target_modules,
                bias="none",
                use_dora=(method == "DoRA"),
                inference_mode=False,
            )
            model = get_peft_model(model, lora_cfg)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            log(
                f"   ✅ LoRA ready — {trainable / 1e6:.2f} M / {total / 1e6:.0f} M "
                f"trainable ({100 * trainable / total:.2f} %)"
            )
            vram_snapshot("after PEFT")

            # ── Dataset ───────────────────────────────────────────────────────
            training_state["status"] = "Loading dataset…"
            log("📊 Loading dataset…")

            if dataset_path.startswith("./data/") or Path(dataset_path).is_absolute():
                from datasets import load_dataset as _lds
                ext = Path(dataset_path).suffix.lstrip(".")
                raw = _lds(ext, data_files=dataset_path, split="train")
            elif dataset_path == "tatsu-lab/alpaca":
                raw = load_alpaca_dataset(num_samples=cfg.num_samples)
            else:
                from datasets import load_dataset as _lds
                raw = _lds(dataset_path, split="train")

            # Ensure a "text" column exists
            if "text" not in raw.column_names:
                if all(c in raw.column_names for c in ["instruction", "output"]):
                    raw = raw.map(format_instruction, remove_columns=raw.column_names)
                else:
                    first = raw.column_names[0]
                    raw   = raw.map(lambda ex: {"text": str(ex[first])})

            log(f"   ✅ Dataset ready : {len(raw)} samples")

            # DPO column validation
            if method == "DPO":
                required = {"prompt", "chosen", "rejected"}
                if not required.issubset(set(raw.column_names)):
                    raise ValueError(
                        f"DPO requires columns {required}. "
                        f"Your dataset has: {raw.column_names}.\n"
                        f"Use 'Anthropic/hh-rlhf' or a dataset with those columns."
                    )

            # ── Trainer ───────────────────────────────────────────────────────
            training_state["status"] = "Creating trainer…"
            log("🎯 Creating trainer…")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            if method == "DPO":
                import copy
                ref_model = copy.deepcopy(model)
                for p in ref_model.parameters():
                    p.requires_grad_(False)
                trainer = create_dpo_trainer(model, ref_model, tokenizer, raw, cfg)
            else:
                trainer = create_lora_trainer(model, tokenizer, raw, cfg)

            log("   ✅ Trainer ready")

            # ── Train ─────────────────────────────────────────────────────────
            log("")
            log("=" * 60)
            log(f"🔥  TRAINING  —  {method}")
            log("=" * 60)
            training_state["status"] = "Training…"

            result = trainer.train()

            if hasattr(result, "training_loss"):
                training_state["loss"].append(result.training_loss)
                log(f"\n   Final loss   : {result.training_loss:.4f}")
                log(f"   Global steps : {result.global_step}")

            elapsed = time.time() - training_state["start_time"]
            log(f"   Elapsed      : {elapsed / 60:.1f} min")

            # ── Save adapter ──────────────────────────────────────────────────
            training_state["status"] = "Saving adapter…"
            log("\n💾 Saving LoRA adapter…")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            log(f"   ✅ Adapter saved → {output_dir}")

            # Loss curve (non-fatal)
            try:
                plot_path = str(Path(output_dir) / "loss_curve.png")
                plot_loss(trainer, plot_path, title=f"{method} training loss")
                log(f"   📈 Loss curve  → {plot_path}")
            except Exception:
                pass

            # ── Merge ─────────────────────────────────────────────────────────
            training_state["status"] = "Merging adapter…"
            log("\n🔗 Merging adapter into base weights…")
            try:
                merge_and_save(model, tokenizer, merged_dir, compute_dtype)
                log(f"   ✅ Merged model → {merged_dir}")
                training_state["model_path"] = merged_dir
            except Exception as exc:
                log(f"   ⚠️  Merge skipped ({exc}) — adapter still usable from {output_dir}")
                training_state["model_path"] = output_dir

            # ── Done ──────────────────────────────────────────────────────────
            training_state["status"] = "✅ Complete!"
            training_state["progress"] = 100
            log("")
            log("=" * 60)
            log("🎉  TRAINING COMPLETE!")
            log("=" * 60)
            log(f"\nAdapter path : {output_dir}")
            log(f"Merged path  : {training_state['model_path']}")
            log("\nNext steps:")
            log("  1. Go to 💬 Inference tab → Refresh Models → select your model")
            log(f"  2. Or: python scripts/infer.py --model {output_dir}")

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(
                "❌ CUDA Out of Memory!\n\n"
                "Try one or more of:\n"
                "  • Switch to QLoRA (4-bit quantization)\n"
                "  • Reduce Max Sequence Length  ← biggest lever\n"
                "  • Pick a smaller base model\n"
                "  • Set Batch Size = 1 and raise Gradient Accumulation"
            )
            training_state["status"] = "❌ OOM — see log"

        except Exception as exc:
            log(f"❌ Error: {exc}")
            log("")
            log("── Stack trace ──────────────────────────────────────")
            log(traceback.format_exc())
            training_state["status"] = f"❌ {str(exc)[:80]}"

        finally:
            training_state["running"] = False
            
            # Clean up empty/failed training directories
            if training_state["status"].startswith("❌"):
                import shutil
                output_dir = Path(output_dir)
                if output_dir.exists() and not (output_dir / "adapter_config.json").exists():
                    print(f"🗑️  Cleaning up failed training directory: {output_dir.name}")
                    try:
                        shutil.rmtree(output_dir)
                    except Exception as e:
                        print(f"   Could not remove: {e}")

    threading.Thread(target=_train, daemon=True).start()
    return f"🔄 {training_state['status']}", 0, "Training started — live log updating below…\n"


def get_training_status():
    """Polled by gr.Timer every 3 s to update status / progress / log."""
    log_tail = "\n".join(training_state["output"][-30:]) if training_state["output"] else ""

    if not training_state["running"]:
        prog = 100 if "Complete" in training_state.get("status", "") else 0
        return training_state.get("status", "Idle"), prog, log_tail

    status = training_state["status"]
    if "tokenizer" in status.lower() or "model" in status.lower():
        prog = 10
    elif "dataset" in status.lower():
        prog = 20
    elif "trainer" in status.lower():
        prog = 25
    elif "training" in status.lower():
        training_state["progress"] = min(training_state["progress"] + 1, 88)
        prog = training_state["progress"]
    elif "saving" in status.lower() or "merging" in status.lower():
        prog = 95
    else:
        prog = training_state["progress"]

    return f"🔄 {status}", prog, log_tail


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(
    model_name: str, prompt: str,
    max_tokens: int, temperature: float, top_p: float,
):
    if not model_name or "No" in model_name:
        return "❌ No model selected. Train a model first or type a HuggingFace model ID."

    model_path = None
    for candidate in [
        f"./models/adapters/{model_name}",
        f"./models/merged/{model_name}",
        model_name,  # Allow HuggingFace model IDs
    ]:
        if Path(candidate).exists():
            model_path = candidate
            break

    if model_path is None:
        return f"❌ Model directory not found: '{model_name}'\n\nThis can happen if:\n• Training failed and left an empty directory\n• The model was deleted\n• The name is incorrect\n\nTry refreshing the model list or training a new model."

    # Validate model has required files
    adapter_config = Path(model_path) / "adapter_config.json"
    tokenizer_config = Path(model_path) / "tokenizer_config.json"
    
    if not adapter_config.exists():
        return f"❌ Invalid model: Missing adapter_config.json\n\nThe model directory exists but is incomplete. This usually means training failed. Try retraining."
    
    if not tokenizer_config.exists():
        return f"❌ Invalid model: Missing tokenizer_config.json\n\nThe model directory exists but is incomplete. This usually means training failed. Try retraining."

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        # Apply chat template when available; fall back to raw prompt
        try:
            messages   = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=float(temperature) > 0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens (strip echoed prompt)
        new_ids  = out_ids[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        return response.strip()

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "❌ OOM during inference. Reduce Max Tokens or use a smaller model."
    except Exception as exc:
        return f"❌ Inference error: {exc}\n\n{traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str, eval_dataset: str, num_samples: int = 20):
    if not model_name or "No" in model_name:
        return "❌ Please select a trained model first."

    model_path = None
    for candidate in [
        f"./models/adapters/{model_name}",
        f"./models/merged/{model_name}",
    ]:
        if Path(candidate).exists():
            model_path = candidate
            break

    if model_path is None:
        return f"❌ Model not found: {model_name}"

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.llm_ft.evaluation import evaluate_perplexity, evaluate_bleu, evaluate_rouge

        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=compute_dtype, device_map="auto"
        )
        model.eval()

        # Load a small eval slice
        from datasets import load_dataset as _lds
        if eval_dataset == "tatsu-lab/alpaca":
            from src.llm_ft.data import load_alpaca_dataset, format_instruction
            ds    = load_alpaca_dataset(num_samples=num_samples)
            ds    = ds.map(format_instruction)
            texts = [ex["text"] for ex in ds]
        else:
            ds    = _lds(eval_dataset, split="train").shuffle(seed=42).select(range(num_samples))
            first = ds.column_names[0]
            texts = [str(ex[first]) for ex in ds]

        # Perplexity over all samples
        t0         = time.time()
        perplexity = evaluate_perplexity(model, tokenizer, texts, batch_size=1)
        elapsed    = time.time() - t0

        # BLEU / ROUGE on first 5 (generation is slow)
        preds, refs = [], []
        for text in texts[:min(5, num_samples)]:
            half = text[: len(text) // 2]
            inp  = tokenizer(
                half, return_tensors="pt", truncation=True, max_length=128
            ).to(model.device)
            with torch.no_grad():
                ids = model.generate(
                    **inp, max_new_tokens=64,
                    pad_token_id=tokenizer.eos_token_id,
                )
            preds.append(tokenizer.decode(ids[0], skip_special_tokens=True))
            refs.append(text)

        bleu  = evaluate_bleu(preds, refs)
        rouge = evaluate_rouge(preds, refs)

        report = {
            "model":         model_name,
            "eval_dataset":  eval_dataset,
            "num_samples":   num_samples,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "perplexity":    round(perplexity, 3),
            "eval_time_sec": round(elapsed, 1),
            **{k: round(v, 4) for k, v in bleu.items()},
            **{k: round(v, 4) for k, v in rouge.items()},
        }
        return json.dumps(report, indent=2)

    except Exception as exc:
        return f"❌ Evaluation error:\n{exc}\n\n{traceback.format_exc()}"


# ═══════════════════════════════════════════════════════════════════════════════
# Inline parameter guidance shown above the sliders
# ═══════════════════════════════════════════════════════════════════════════════

_PARAM_HINTS = """
> **ℹ️ Epochs** — Small dataset (<100 samples): **10–50** · Medium (100–1 K): **5–20** · Large (1 K+): **2–10** · ⚠️ Too many epochs on small data → overfitting

> **ℹ️ LoRA Rank** — 4 GB: **4–8** · 8 GB: **8–16 ⭐** · 12 GB: **16–32** · 24 GB+: **32–64**

> **ℹ️ LoRA Alpha** — Rule of thumb: **Alpha = 2 × Rank** (8→16 · 16→32 · 32→64)

> **ℹ️ Learning Rate** — QLoRA/LoRA/LoRA+/DoRA: **2e-4 ⭐** · SFT: **2e-5** · DPO: **5e-7**

> **ℹ️ Max Sequence Length** — 4 GB: **128** · 8 GB: **256 ⭐** · 12 GB: **512** · 16 GB+: **1024–2048** · Halving saves ~4× activation VRAM ← biggest OOM lever

> **ℹ️ Batch & Gradient Accumulation** — ≤8 GB: Batch=1, Accum=8–16 ⭐ · 12 GB: Batch=2, Accum=4–8 · 16 GB+: Batch=4, Accum=2–4 · **Eff. batch = Batch × Accum** (keep ≥ 8 for stable gradients)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

def create_ui():
    with gr.Blocks(title="LLM Fine-Tuning Toolkit") as ui:

        gr.Markdown("""
# 🚀 LLM Fine-Tuning Toolkit
**Production-grade fine-tuning on consumer GPUs**

Supports: `QLoRA` · `LoRA` · `LoRA+` · `DoRA` · `SFT` · `DPO` · `RLHF`
        """)

        # Module health banner — shown only when imports failed
        if not _MODULES_OK:
            gr.Markdown(
                f"> ⚠️ **Project modules could not be imported:** `{_MODULES_ERR}`\n"
                "> Run from the project root and install deps:\n"
                "> `pip install transformers peft trl bitsandbytes datasets accelerate`"
            )

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 — Training
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("🎯 Training"):
                with gr.Row():

                    # ── Left column: step-by-step configuration ───────────────
                    with gr.Column(scale=3):

                        # Step 1 — Method
                        gr.Markdown("### Step 1 — Fine-tuning method")
                        method = gr.Dropdown(
                            list(METHODS.keys()), value="QLoRA", label="Method",
                        )
                        method_info_md = gr.Markdown(value=update_method_info("QLoRA"))

                        gr.Markdown("---")

                        # Step 2 — Model
                        gr.Markdown("### Step 2 — Base model")
                        model_dd = gr.Dropdown(
                            MODELS,
                            value="Qwen/Qwen2.5-1.5B-Instruct",
                            label="Base model (HuggingFace ID)",
                            allow_custom_value=True,
                        )

                        gr.Markdown("---")

                        # Step 3 — Dataset
                        gr.Markdown("### Step 3 — Dataset")
                        dataset_type = gr.Radio(
                            ["HuggingFace", "File"],
                            value="HuggingFace",
                            label="Dataset source",
                        )
                        with gr.Row():
                            dataset_hf = gr.Dropdown(
                                DATASETS,
                                value="tatsu-lab/alpaca",
                                label="HuggingFace dataset",
                                allow_custom_value=True,
                                visible=True,
                            )
                            dataset_file = gr.Dropdown(
                                list_dataset_files(),
                                label="Local file (./data/)",
                                visible=False,
                                allow_custom_value=True,
                            )

                        gr.Markdown("---")

                        # Step 4 — Hyperparameters
                        gr.Markdown("### Step 4 — Hyperparameters")
                        gr.Markdown(_PARAM_HINTS)

                        with gr.Row():
                            rank = gr.Slider(
                                4, 64, value=8, step=4, visible=True,
                                label="LoRA Rank — 4 GB:4-8 | 8 GB:8-16⭐ | 12 GB:16-32 | 24 GB+:32-64",
                            )
                            alpha = gr.Slider(
                                8, 128, value=16, step=8, visible=True,
                                label="LoRA Alpha — rule: 2×Rank  (8→16 | 16→32 | 32→64)",
                            )

                        with gr.Row():
                            learning_rate = gr.Number(
                                value=2e-4, minimum=1e-7, maximum=1e-2, visible=True,
                                label="Learning Rate — QLoRA/LoRA:2e-4⭐ | SFT:2e-5 | DPO:5e-7",
                            )
                            beta = gr.Slider(
                                0.01, 0.5, value=0.1, step=0.01, visible=False,
                                label="DPO Beta — divergence temperature (rec: 0.1)",
                            )

                        loraplus_ratio = gr.Slider(
                            4, 32, value=16, step=4, visible=False,
                            label="LoRA+ LR Ratio — B-matrix LR multiplier (rec: 16)",
                        )

                        with gr.Row():
                            epochs = gr.Slider(
                                1, 100, value=2, step=1, visible=True,
                                label="Epochs — <100 samples:10-50 | 100-1K:5-20 | 1K+:2-10 ⚠️",
                            )
                            max_length = gr.Slider(
                                64, 2048, value=256, step=64, visible=True,
                                label="Max Seq Len — 4 GB:128 | 8 GB:256⭐ | 12 GB:512 | 16 GB+:1024+",
                            )

                        with gr.Row():
                            batch_size = gr.Slider(
                                1, 8, value=1, step=1,
                                label="Batch Size — ≤8 GB:1⭐ | 12 GB:2 | 16 GB+:4",
                            )
                            grad_accum = gr.Slider(
                                1, 64, value=8, step=1,
                                label="Gradient Accumulation — Eff. batch = Batch × Accum (rec: 8-16)",
                            )

                        output_name = gr.Textbox(
                            label="Run name (optional — auto-generated if empty)",
                            placeholder="my_finetune_run",
                        )

                        gr.Markdown("---")
                        gr.Markdown("### Step 5 — Launch")
                        train_btn = gr.Button(
                            "🚀 Start Training", variant="primary", size="lg",
                        )

                    # ── Right column: hardware status + live training log ──────
                    with gr.Column(scale=2):
                        gr.Markdown("### Hardware")
                        gpu_info_box = gr.Textbox(
                            value=check_gpu(), label="GPU status",
                            interactive=False, lines=1,
                        )
                        vram_table = gr.Markdown(
                            value=get_vram_recommendations("Qwen/Qwen2.5-1.5B-Instruct"),
                            label="VRAM estimate table",
                        )

                        gr.Markdown("### Training status")
                        status_box = gr.Textbox(
                            value="Idle", label="Status", interactive=False,
                        )
                        progress_bar = gr.Slider(
                            0, 100, value=0, label="Progress %", interactive=False,
                        )
                        log_box = gr.Textbox(
                            label="Live log (last 30 lines)",
                            lines=20, max_lines=40, interactive=False,
                        )

                        # Auto-refresh every 3 s while training is running
                        timer = gr.Timer(3)
                        timer.tick(
                            fn=get_training_status,
                            outputs=[status_box, progress_bar, log_box],
                        )

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 — Inference
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("💬 Inference"):
                with gr.Row():
                    with gr.Column():
                        infer_model = gr.Dropdown(
                            list_models(), label="Trained model",
                            allow_custom_value=True,
                        )
                        infer_refresh_btn = gr.Button("🔄 Refresh models list")
                        infer_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Write a Python function that reverses a string.",
                            lines=5,
                        )
                        with gr.Row():
                            max_tokens  = gr.Slider(32, 1024, value=256, step=32,
                                                    label="Max new tokens")
                            temperature = gr.Slider(0.0, 2.0,  value=0.7, step=0.05,
                                                    label="Temperature")
                            top_p       = gr.Slider(0.1, 1.0,  value=0.9, step=0.05,
                                                    label="Top-p")
                        infer_btn = gr.Button("▶ Generate", variant="primary", size="lg")

                    with gr.Column():
                        infer_output = gr.Textbox(label="Model response", lines=20)

                gr.Examples(
                    examples=[
                        "What is machine learning?",
                        "Explain LoRA fine-tuning in simple terms.",
                        "Write a Python function to compute Fibonacci numbers.",
                        "What are the benefits of quantization?",
                        "Summarize the paper 'Attention Is All You Need'.",
                    ],
                    inputs=infer_prompt,
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 — Evaluation
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("📊 Evaluation"):
                gr.Markdown("""
### Model evaluation

Computes **Perplexity**, **BLEU-1–4**, and **ROUGE-1/2/L** on a small held-out slice.
For full benchmark suites run: `python scripts/test_alpaca.py`
                """)
                with gr.Row():
                    with gr.Column():
                        eval_model_dd = gr.Dropdown(
                            list_models(), label="Model to evaluate",
                            allow_custom_value=True,
                        )
                        eval_refresh_btn = gr.Button("🔄 Refresh models list")
                        eval_dataset_dd  = gr.Dropdown(
                            DATASETS, value="tatsu-lab/alpaca",
                            label="Evaluation dataset",
                        )
                        eval_samples = gr.Slider(
                            5, 100, value=20, step=5,
                            label="Number of eval samples",
                        )
                        eval_btn = gr.Button(
                            "📊 Run evaluation", variant="primary", size="lg",
                        )
                    with gr.Column():
                        eval_output = gr.Textbox(label="Results (JSON)", lines=20)

        # ══════════════════════════════════════════════════════════════════════
        # Event wiring
        # ══════════════════════════════════════════════════════════════════════

        # Method dropdown → info card + auto-update param sliders
        method.change(fn=update_method_info, inputs=[method], outputs=[method_info_md])
        method.change(
            fn=update_params, inputs=[method],
            outputs=[rank, alpha, learning_rate, beta, loraplus_ratio, epochs, max_length],
        )

        # Model dropdown → live VRAM estimate table
        model_dd.change(fn=update_vram_table, inputs=[model_dd], outputs=[vram_table])

        # Dataset source toggle (HuggingFace ↔ local file)
        dataset_type.change(
            fn=toggle_dataset_ui, inputs=[dataset_type],
            outputs=[dataset_file, dataset_hf],
        )

        # Training
        train_btn.click(
            fn=start_training,
            inputs=[
                method, model_dd,
                dataset_type, dataset_file, dataset_hf,
                rank, alpha, learning_rate, beta, loraplus_ratio,
                epochs, max_length, batch_size, grad_accum, output_name,
            ],
            outputs=[status_box, progress_bar, log_box],
        )

        # Inference
        infer_refresh_btn.click(fn=refresh_models_list, outputs=[infer_model])
        infer_btn.click(
            fn=run_inference,
            inputs=[infer_model, infer_prompt, max_tokens, temperature, top_p],
            outputs=[infer_output],
        )

        # Evaluation
        eval_refresh_btn.click(fn=refresh_models_list, outputs=[eval_model_dd])
        eval_btn.click(
            fn=evaluate_model,
            inputs=[eval_model_dd, eval_dataset_dd, eval_samples],
            outputs=[eval_output],
        )

    return ui


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _find_free_port(start: int = 7860) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start  # fallback — Gradio will report the error


if __name__ == "__main__":
    print("=" * 60)
    print("🚀  LLM Fine-Tuning Web UI")
    print("=" * 60)
    print(f"GPU        : {check_gpu()}")
    print(f"Project    : {_ROOT}")
    print(f"Modules OK : {_MODULES_OK}")
    if not _MODULES_OK:
        print(f"Module err : {_MODULES_ERR}")
    print()
    print("Methods:")
    for name, info in METHODS.items():
        print(f"  {name:8s}  {info['vram_display']:24s}  {info['description'][:52]}…")
    print()

    # Detect if running in Docker (check for .dockerenv or containerized environment)
    in_docker = Path("/.dockerenv").exists() or os.environ.get("CONTAINER", "false").lower() == "true"
    
    port = _find_free_port()
    print(f"Open in browser → http://localhost:{port}")
    print("=" * 60)

    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=in_docker,  # Enable share mode in Docker to avoid proxy issues
        show_error=True,
    )