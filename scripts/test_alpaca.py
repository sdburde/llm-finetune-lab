#!/usr/bin/env python3
"""
Test and Evaluate All Methods on Alpaca Dataset

Usage:
    python scripts/test_alpaca.py
"""

import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test Configuration
TEST_CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "tatsu-lab/alpaca",
    "num_samples": 100,  # Small for quick testing
    "eval_samples": 20,
    "epochs": 1,
    "vram_limit": 8,
    "output_base": "./models/alpaca_test",
}

# Methods to Test
METHODS = {
    "QLoRA": {"method": "qlora", "rank": 8, "load_in_4bit": True, "lr": 2e-4},
    "LoRA": {"method": "lora", "rank": 8, "load_in_4bit": False, "lr": 2e-4},
    "LoRA+": {"method": "lora", "rank": 8, "load_in_4bit": True, "lr": 2e-4, "loraplus_ratio": 16},
    "DoRA": {"method": "dora", "rank": 8, "load_in_4bit": True, "lr": 2e-4},
    "SFT": {"method": "sft", "rank": 16, "load_in_4bit": False, "lr": 2e-5},
    "DPO": {"method": "dpo", "rank": 8, "load_in_4bit": True, "lr": 5e-7, "beta": 0.1},
}


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e9
    return 0


def test_and_evaluate(method_name, config):
    """Test a method and evaluate on Alpaca."""
    print(f"\n{'='*60}")
    print(f"Testing {method_name} on Alpaca Dataset")
    print(f"{'='*60}")
    
    start_time = time.time()
    vram_before = get_vram()
    
    try:
        from scripts.finetune import FineTuningEngine, FineTuningConfig
        
        # Create config
        ft_config = FineTuningConfig(
            method=config["method"],
            model_name=TEST_CONFIG["model"],
            dataset_name=TEST_CONFIG["dataset"],
            rank=config.get("rank", 8),
            alpha=config.get("rank", 8) * 2,
            batch_size=1,
            gradient_accumulation_steps=4,
            max_length=256,
            num_epochs=TEST_CONFIG["epochs"],
            learning_rate=config.get("lr", 2e-4),
            load_in_4bit=config.get("load_in_4bit", False),
            output_base=TEST_CONFIG["output_base"],
            run_name=f"alpaca_{method_name.lower().replace('+', 'plus')}",
            num_samples=TEST_CONFIG["num_samples"],
        )
        
        # Create engine
        engine = FineTuningEngine(ft_config)
        
        # Setup
        print("Setting up environment...")
        engine.setup_environment()
        
        print("Loading tokenizer...")
        engine.load_tokenizer()
        
        print("Loading model...")
        engine.load_model()
        
        print("Setting up PEFT...")
        engine.setup_peft()
        
        print("Loading Alpaca dataset...")
        dataset = engine.load_dataset()
        
        print("Creating trainer...")
        engine.create_trainer(dataset)
        
        # Train
        print(f"Training {method_name}...")
        result = engine.train()
        
        # Get metrics
        vram_used = get_vram() - vram_before
        time_taken = (time.time() - start_time) / 60
        training_loss = result.training_loss if hasattr(result, 'training_loss') else None
        
        # Save model
        output_dirs = ft_config.get_output_dirs()
        engine.trainer.save_model(output_dirs["adapter"])
        engine.tokenizer.save_pretrained(output_dirs["adapter"])
        
        # Evaluate
        print(f"\nEvaluating {method_name}...")
        eval_results = evaluate_model(engine.model, engine.tokenizer, TEST_CONFIG["eval_samples"])
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS: {method_name}")
        print(f"{'='*60}")
        print(f"Training Time: {time_taken:.2f} minutes")
        print(f"VRAM Used: {vram_used:.2f} GB")
        print(f"Training Loss: {training_loss:.4f}" if training_loss else "N/A")
        print(f"Eval Perplexity: {eval_results.get('perplexity', 'N/A'):.2f}")
        print(f"Output Dir: {output_dirs['adapter']}")
        
        # Cleanup
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "method": method_name,
            "status": "PASS",
            "time_minutes": time_taken,
            "vram_gb": vram_used,
            "training_loss": training_loss,
            "eval_perplexity": eval_results.get("perplexity"),
            "output_dir": output_dirs["adapter"],
        }
        
    except Exception as e:
        time_taken = (time.time() - start_time) / 60
        print(f"\n❌ {method_name} FAILED: {str(e)[:200]}")
        
        return {
            "method": method_name,
            "status": "FAIL",
            "error": str(e),
            "time_minutes": time_taken,
        }


def evaluate_model(model, tokenizer, num_samples=20):
    """Evaluate model on generation task."""
    import math
    
    model.eval()
    
    # Test prompts from Alpaca
    test_prompts = [
        "Give three tips for staying healthy.",
        "Explain what machine learning is.",
        "What is the capital of France?",
        "Write a short poem about nature.",
        "How do I make pancakes?",
    ]
    
    results = {
        "perplexity": 0.0,
        "generations": [],
    }
    
    total_loss = 0.0
    
    with torch.no_grad():
        for prompt in test_prompts[:num_samples]:
            # Format prompt
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
            )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results["generations"].append({
                "prompt": prompt,
                "response": response,
            })
            
            # Calculate perplexity (approximate)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()
    
    # Average perplexity
    if len(test_prompts) > 0:
        avg_loss = total_loss / len(test_prompts)
        results["perplexity"] = math.exp(avg_loss)
    
    return results


def generate_report(all_results):
    """Generate comprehensive test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": TEST_CONFIG,
        "results": all_results,
        "summary": {
            "total": len(all_results),
            "passed": sum(1 for r in all_results if r["status"] == "PASS"),
            "failed": sum(1 for r in all_results if r["status"] == "FAIL"),
        }
    }
    
    # Save report
    report_path = Path("./alpaca_test_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Full report saved to: {report_path}")
    
    return report


def main():
    """Run all tests."""
    print_header("LLM FINE-TUNING - ALPACA DATASET TEST")
    
    print(f"Model: {TEST_CONFIG['model']}")
    print(f"Dataset: {TEST_CONFIG['dataset']} ({TEST_CONFIG['num_samples']} samples)")
    print(f"Eval Samples: {TEST_CONFIG['eval_samples']}")
    print(f"Epochs: {TEST_CONFIG['epochs']}")
    print(f"VRAM Limit: {TEST_CONFIG['vram_limit']} GB")
    
    # Test each method
    all_results = []
    
    for method_name, config in METHODS.items():
        result = test_and_evaluate(method_name, config)
        all_results.append(result)
        
        # Wait between tests
        time.sleep(2)
    
    # Generate report
    print_header("TEST SUMMARY")
    report = generate_report(all_results)
    
    # Print summary
    print(f"\nTotal Tests: {report['summary']['total']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print("\nDetailed Results:")
    print("-" * 70)
    
    for result in all_results:
        method = result["method"]
        status = result["status"]
        time_m = result.get("time_minutes", "N/A")
        vram = result.get("vram_gb", "N/A")
        loss = result.get("training_loss", "N/A")
        perplexity = result.get("eval_perplexity", "N/A")
        
        if isinstance(time_m, float):
            time_m = f"{time_m:.1f}m"
        if isinstance(vram, float):
            vram = f"{vram:.1f}GB"
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        if isinstance(perplexity, float):
            perplexity = f"{perplexity:.2f}"
        
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {method:<10} | Time: {time_m:>6} | VRAM: {vram:>6} | Loss: {loss:>8} | PPL: {perplexity:>8}")
    
    # Show generations
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS")
    print("=" * 70)
    
    for result in all_results[:3]:  # Show first 3 methods
        if result["status"] == "PASS":
            print(f"\n{result['method']}:")
            print("-" * 60)
            
            # Load generations from output dir
            output_dir = result.get("output_dir", "")
            if output_dir:
                # Try to load and show a sample generation
                print(f"  Model saved: {output_dir}")
    
    return report["summary"]["passed"] == report["summary"]["total"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
