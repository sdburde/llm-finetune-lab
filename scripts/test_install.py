#!/usr/bin/env python3
"""
Quick test script to verify installation and basic functionality.
Runs a minimal inference test without training.

Usage:
    python scripts/test_install.py
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "datasets": "Datasets",
        "accelerate": "Accelerate",
        "bitsandbytes": "BitsAndBytes",
        "trl": "TRL",
    }
    
    failed = []
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {name}: {version}")
        except ImportError as e:
            print(f"❌ {name}: NOT INSTALLED")
            failed.append(pkg)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages installed")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\n" + "="*60)
    print("TESTING CUDA")
    print("="*60)
    
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA available")
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {vram:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - CPU mode only")
        return False


def test_model_load():
    """Test loading a small model."""
    print("\n" + "="*60)
    print("TESTING MODEL LOAD (Qwen2.5-0.5B)")
    print("="*60)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Use CPU for test
        )
        
        # Test inference
        print("Testing inference...")
        prompt = "Hello! How are you?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model loaded and working!")
        print(f"   Input: {prompt}")
        print(f"   Output: {response[:100]}...")
        
        del model, tokenizer
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return False


def test_dataset_load():
    """Test loading a dataset."""
    print("\n" + "="*60)
    print("TESTING DATASET LOAD (Alpaca - 10 samples)")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("tatsu-lab/alpaca", split="train").select(range(10))
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"   Features: {dataset.features}")
        return True
        
    except Exception as e:
        print(f"❌ Dataset load failed: {e}")
        return False


def test_peft():
    """Test PEFT/LoRA setup."""
    print("\n" + "="*60)
    print("TESTING PEFT/LoRA SETUP")
    print("="*60)
    
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        # Load small model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"✅ PEFT setup successful")
        print(f"   Trainable: {trainable}/{total} ({100*trainable/total:.2f}%)")
        
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ PEFT setup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════╗
║     LLM Fine-Tuning - Installation Test                  ║
║     Verifies packages, GPU, model loading, and PEFT      ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Dataset": test_dataset_load(),
        "Model Load": test_model_load(),
        "PEFT/LoRA": test_peft(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nResult: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready for fine-tuning.")
        print("\nNext steps:")
        print("  1. Check your hardware: python scripts/check_env.py")
        print("  2. Run fine-tuning: python scripts/finetune.py --method qlora")
        print("  3. See tutorials: ls notebooks/")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install requirements: pip install -r requirements.txt")
        print("  - Install CUDA/PyTorch with GPU support")
        print("  - Check disk space: df -h")
        return 1


if __name__ == "__main__":
    sys.exit(main())
