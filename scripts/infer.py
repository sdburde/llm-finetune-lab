#!/usr/bin/env python3
"""
Inference script for fine-tuned models.

Usage:
    python scripts/infer.py --model ./merged_model --prompt "What is AI?"
    python scripts/infer.py --ollama my-model --prompt "Hello"
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned model")
    parser.add_argument("--model", type=str, help="Path to merged model directory")
    parser.add_argument("--ollama", type=str, help="Ollama model name")
    parser.add_argument("--prompt", type=str, default="Hello!", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    return parser.parse_args()


def infer_local(args):
    """Run inference on local model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Format prompt
    messages = [{"role": "user", "content": args.prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*60)
    print("RESPONSE")
    print("="*60)
    print(result)
    print("="*60)


def infer_ollama(args):
    """Run inference via Ollama API."""
    import requests
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": args.ollama,
        "prompt": args.prompt,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.max_tokens,
        }
    }
    
    print(f"Querying Ollama: {args.ollama}")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()["response"]
        print("\n" + "="*60)
        print("RESPONSE")
        print("="*60)
        print(result)
        print("="*60)
    else:
        print(f"Error: {response.status_code}")
        print("Make sure Ollama is running: ollama serve")


def main():
    args = parse_args()
    
    if args.ollama:
        infer_ollama(args)
    elif args.model:
        infer_local(args)
    else:
        print("Error: Specify --model or --ollama")
        print("Usage: python scripts/infer.py --model ./merged --prompt 'Hello'")
        print("   or: python scripts/infer.py --ollama my-model --prompt 'Hello'")


if __name__ == "__main__":
    main()
