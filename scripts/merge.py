#!/usr/bin/env python3
"""
Merge LoRA adapter with base model.

Usage:
    python scripts/merge.py --adapter ./output/model_adapter --output ./merged_model
"""

import argparse
import gc
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, default="./merged", help="Output directory")
    parser.add_argument("--dtype", type=str, default="float16", help="Output dtype")
    return parser.parse_args()


def main():
    args = parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading base model and adapter from: {args.adapter}")
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.adapter,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Merge adapter
    print("Merging adapter weights...")
    merged_model = base_model.merge_and_unload()
    
    # Save
    print(f"Saving merged model to: {args.output}")
    merged_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    
    # Cleanup
    del merged_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"✅ Merged model saved: {args.output}")
    print(f"\nNext: Convert to GGUF")
    print(f"  python scripts/convert.py --model {args.output}")


if __name__ == "__main__":
    main()
