#!/usr/bin/env python3
"""
Convert HuggingFace model to GGUF format for Ollama.

Usage:
    python scripts/convert.py --model ./merged_model --output model.gguf --quant q4_k_m
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF")
    parser.add_argument("--model", type=str, required=True, help="HF model directory")
    parser.add_argument("--output", type=str, default="model.gguf", help="Output GGUF file")
    parser.add_argument("--quant", type=str, default="q4_k_m", 
                       choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                       help="Quantization type")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Clone llama.cpp if needed
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git",
            "--depth=1", "--quiet"
        ], check=True)
    
    # Convert
    print(f"Converting to GGUF ({args.quant})...")
    result = subprocess.run([
        sys.executable, "llama.cpp/convert_hf_to_gguf.py",
        args.model, "--outfile", args.output, "--outtype", args.quant
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(args.output):
        size_gb = os.path.getsize(args.output) / 1e9
        print(f"✅ GGUF created: {args.output} ({size_gb:.2f} GB)")
        
        # Create Modelfile
        model_name = os.path.basename(args.output).replace(".gguf", "")
        modelfile = f"""FROM {args.output}

SYSTEM \"\"\"You are a helpful AI assistant.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
PARAMETER num_predict 512
"""
        with open("Modelfile", "w") as f:
            f.write(modelfile)
        
        print(f"\nModelfile created. Register with Ollama:")
        print(f"  ollama create {model_name} -f Modelfile")
        print(f"  ollama run {model_name}")
    else:
        print("❌ Conversion failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
