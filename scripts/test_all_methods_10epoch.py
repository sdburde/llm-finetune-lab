#!/usr/bin/env python3
"""
Comprehensive Test: All Fine-Tuning Methods with 10 Epochs
Tests: QLoRA, LoRA, LoRA+, DoRA, SFT, DPO
GPU: Optimized for 8-10GB VRAM
"""

import torch
import time
from pathlib import Path
from datetime import datetime

# Import from src/llm_ft
from src.llm_ft.config import FineTuningConfig
from src.llm_ft.data import load_alpaca_dataset, format_instruction
from src.llm_ft.models import load_model, load_tokenizer
from src.llm_ft.trainers import create_lora_trainer
from src.llm_ft.utils import detect_gpu, merge_and_save, vram_snapshot, prepare_for_training

# Test configuration
TEST_CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "tatsu-lab/alpaca",
    "samples": 20,
    "max_length": 128,
    "batch_size": 1,
    "grad_accum": 4,
    "epochs": 10,  # TEST: 10 EPOCHS FOR ALL METHODS
}

# Methods to test
METHODS = {
    "QLoRA": {
        "rank": 8,
        "alpha": 16,
        "lr": 2e-4,
        "load_in_4bit": True,
        "bf16": True,
    },
    "LoRA": {
        "rank": 16,
        "alpha": 32,
        "lr": 2e-4,
        "load_in_4bit": False,
        "bf16": True,
    },
    "LoRA+": {
        "rank": 16,
        "alpha": 32,
        "lr": 2e-4,
        "load_in_4bit": True,
        "bf16": True,
        "loraplus_lr_ratio": 16,
    },
    "DoRA": {
        "rank": 16,
        "alpha": 32,
        "lr": 2e-4,
        "load_in_4bit": True,
        "bf16": True,
    },
    "SFT": {
        "rank": 16,
        "alpha": 32,
        "lr": 2e-5,
        "load_in_4bit": False,
        "bf16": True,
    },
    "DPO": {
        "rank": 16,
        "alpha": 32,
        "lr": 5e-7,
        "load_in_4bit": True,
        "bf16": True,
        "beta": 0.1,
    },
}


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def test_method(method_name, config):
    """Test a single fine-tuning method with 10 epochs."""
    print_header(f"Testing {method_name} (10 Epochs)")
    
    start_time = time.time()
    results = {
        "method": method_name,
        "status": "FAIL",
        "time": 0,
        "loss": None,
        "error": None,
    }
    
    try:
        # Detect hardware
        device, vram, bf16_ok, compute_dtype = detect_gpu()
        print(f"GPU: {device}, VRAM: {vram:.1f} GB, bf16: {bf16_ok}")
        
        # Load tokenizer
        print("\n📦 Loading tokenizer...")
        tokenizer = load_tokenizer(TEST_CONFIG["model"], trust_remote_code=True)
        print("✅ Tokenizer loaded")
        
        # Load model
        print("\n📦 Loading model...")
        model = load_model(
            TEST_CONFIG["model"],
            load_in_4bit=config["load_in_4bit"],
            compute_dtype=compute_dtype,
            trust_remote_code=True,
        )
        print(f"✅ Model loaded ({model.num_parameters()/1e6:.0f}M params)")
        vram_snapshot("after model load")
        
        # Setup PEFT
        print("\n⚙️  Setting up PEFT...")
        model = prepare_for_training(model)
        
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["rank"],
            lora_alpha=config["alpha"],
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            use_dora=(method_name == "DoRA"),
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"✅ {method_name} ready — {trainable/1e6:.2f}M trainable ({100*trainable/total:.2f}%)")
        vram_snapshot("after PEFT")
        
        # Load dataset
        print("\n📊 Loading dataset...")
        raw = load_alpaca_dataset(num_samples=TEST_CONFIG["samples"])
        raw = raw.map(format_instruction, remove_columns=["instruction", "input", "output", "text"])
        print(f"✅ Dataset: {len(raw)} samples")
        
        # Create config
        output_dir = f"./models/adapters/test_10epoch_{method_name.lower().replace('+', 'plus')}"
        merged_dir = f"./models/merged/test_10epoch_{method_name.lower().replace('+', 'plus')}"
        
        ft_config = FineTuningConfig(
            model_name=TEST_CONFIG["model"],
            method=method_name.lower().replace("+", "plus"),
            rank=config["rank"],
            alpha=config["alpha"],
            load_in_4bit=config["load_in_4bit"],
            dataset_name=TEST_CONFIG["dataset"],
            num_samples=TEST_CONFIG["samples"],
            max_length=TEST_CONFIG["max_length"],
            batch_size=TEST_CONFIG["batch_size"],
            gradient_accumulation_steps=TEST_CONFIG["grad_accum"],
            num_epochs=TEST_CONFIG["epochs"],  # 10 EPOCHS
            learning_rate=config["lr"],
            bf16=config.get("bf16", False),
            output_dir=output_dir,
            adapter_dir=output_dir,
            merged_dir=merged_dir,
        )
        
        if method_name == "LoRA+":
            ft_config.loraplus_lr_ratio = config.get("loraplus_lr_ratio", 16)
        if method_name == "DPO":
            ft_config.beta = config.get("beta", 0.1)
        
        # Create trainer
        print("\n🎯 Creating trainer...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer = create_lora_trainer(model, tokenizer, raw, ft_config)
        print("✅ Trainer ready")
        
        # Train
        print("\n" + "=" * 60)
        print(f"🔥  STARTING 10 EPOCH TRAINING — {method_name}")
        print("=" * 60)
        
        result = trainer.train()
        
        # Calculate metrics
        elapsed = time.time() - start_time
        final_loss = result.training_loss if hasattr(result, 'training_loss') else None
        
        print("\n" + "=" * 60)
        print(f"✅ {method_name} TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Steps: {result.global_step}")
        if final_loss:
            print(f"Final Loss: {final_loss:.4f}")
        
        # Save
        print("\n💾 Saving adapter...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Adapter saved → {output_dir}")
        
        print("\n🔗 Merging adapter...")
        merge_and_save(model, tokenizer, merged_dir, compute_dtype)
        print(f"✅ Merged model → {merged_dir}")
        
        # Record results
        results["status"] = "PASS"
        results["time"] = elapsed / 60
        results["loss"] = final_loss
        
        # Cleanup
        del model, tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        elapsed = time.time() - start_time
        results["status"] = "FAIL"
        results["time"] = elapsed / 60
        results["error"] = str(e)[:200]
        print(f"\n❌ {method_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    """Run all tests."""
    print_header("ALL METHODS - 10 EPOCH TEST SUITE")
    
    print(f"Model: {TEST_CONFIG['model']}")
    print(f"Dataset: {TEST_CONFIG['dataset']} ({TEST_CONFIG['samples']} samples)")
    print(f"Epochs: {TEST_CONFIG['epochs']}")
    print(f"Max Length: {TEST_CONFIG['max_length']}")
    print(f"Batch Size: {TEST_CONFIG['batch_size']}")
    print(f"Grad Accum: {TEST_CONFIG['grad_accum']}")
    print()
    
    # Check GPU
    device, vram, bf16_ok, _ = detect_gpu()
    print(f"GPU: {device}, VRAM: {vram:.1f} GB, bf16: {bf16_ok}")
    print()
    
    # Test each method
    all_results = []
    
    for method_name, config in METHODS.items():
        result = test_method(method_name, config)
        all_results.append(result)
        
        # Wait between tests
        print("\n⏳ Waiting 5 seconds before next test...")
        time.sleep(5)
    
    # Summary
    print_header("TEST SUMMARY - ALL METHODS (10 EPOCHS)")
    
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    total = len(all_results)
    
    print(f"\nTotal Methods: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {100 * passed / total:.1f}%")
    print()
    print("Detailed Results:")
    print("-" * 70)
    print(f"{'Method':<10} | {'Status':<6} | {'Time (min)':<10} | {'Loss':<10} | {'Error':<30}")
    print("-" * 70)
    
    for r in all_results:
        status = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL"
        time_str = f"{r['time']:.1f}" if r["time"] else "N/A"
        loss_str = f"{r['loss']:.4f}" if r["loss"] else "N/A"
        error_str = r["error"][:28] + "..." if r["error"] else "-"
        print(f"{r['method']:<10} | {status:<6} | {time_str:<10} | {loss_str:<10} | {error_str:<30}")
    
    # Save results
    from pathlib import Path
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": TEST_CONFIG,
        "gpu": {"name": device, "vram": vram},
        "results": all_results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": f"{100 * passed / total:.1f}%"
        }
    }
    
    report_path = Path("./test_results_and_analysis/all_methods_10epoch_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Full report saved to: {report_path}")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
