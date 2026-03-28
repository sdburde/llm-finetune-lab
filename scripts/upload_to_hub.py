#!/usr/bin/env python3
"""
Upload fine-tuned models to HuggingFace Hub

Usage:
    python scripts/upload_to_hub.py --model ./models/adapters/RUN_NAME --repo-id sdburde/my-model
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def upload_to_hub(
    model_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = None
):
    """
    Upload fine-tuned model to HuggingFace Hub.
    
    Args:
        model_path: Path to model directory
        repo_id: Repository ID (username/model-name)
        token: HuggingFace API token
        private: Whether to make repo private
        commit_message: Commit message
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Please install huggingface_hub: pip install huggingface_hub")
        return
    
    # Initialize API
    api = HfApi()
    
    # Create repo if needed
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repo creation: {e}")
    
    # Upload model
    print(f"\nUploading model from: {model_path}")
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or "Upload fine-tuned model",
        )
        print(f"✅ Model uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return
    
    # Generate model card
    print("\nGenerating model card...")
    model_card = generate_model_card(model_path, repo_id)
    
    # Upload model card
    try:
        api.upload_file(
            path_or_fileobj=model_card,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            commit_message="Add model card",
        )
        print(f"✅ Model card uploaded!")
    except Exception as e:
        print(f"⚠️  Model card: {e}")
    
    print(f"\n🎉 Complete! View at: https://huggingface.co/{repo_id}")


def generate_model_card(model_path: str, repo_id: str) -> str:
    """Generate model card with metadata."""
    
    # Try to load training log
    log_path = Path(model_path) / ".." / ".." / "logs"
    log_file = list(log_path.glob("*.json"))[0] if list(log_path.glob("*.json")) else None
    
    metrics = {}
    if log_file:
        with open(log_file) as f:
            log_data = json.load(f)
            metrics = log_data.get("config", {})
    
    # Generate card
    card = f"""---
license: mit
language:
- en
tags:
- generated-from-llm-fine-tuning-toolkit
- qlora
- fine-tuned
---

# {repo_id.split('/')[-1]}

Fine-tuned model created with [LLM Fine-Tuning Toolkit](https://github.com/sdburde/llm-fine-tuning)

## Model Details

- **Base Model**: {metrics.get('model_name', 'Unknown')}
- **Method**: {metrics.get('method', 'QLoRA')}
- **Fine-tuned by**: sdburde
- **License**: MIT

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Method | {metrics.get('method', 'QLoRA')} |
| Rank | {metrics.get('rank', 8)} |
| Learning Rate | {metrics.get('learning_rate', 0.0002)} |
| Epochs | {metrics.get('num_epochs', 2)} |
| Batch Size | {metrics.get('batch_size', 1)} |
| Max Length | {metrics.get('max_length', 256)} |

## Usage

### Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("Hello!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Ollama
```bash
ollama run {repo_id.split('/')[-1]}
```

## Training Data

- **Dataset**: {metrics.get('dataset_name', 'Unknown')}
- **Samples**: {metrics.get('num_samples', 'Unknown')}

## Evaluation

See evaluation report in `evaluation_report.json`

## Limitations

This model is fine-tuned for specific tasks and may not generalize well to all domains.

## Citation

```bibtex
@software{{llm_fine_tuning_toolkit,
  title = {{LLM Fine-Tuning Toolkit}},
  author = {{sdburde}},
  year = {{2026}},
  url = {{https://github.com/sdburde/llm-fine-tuning}}
}}
```
"""
    
    return card.encode()


def parse_args():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--repo-id", type=str, required=True, help="Repo ID (user/model)")
    parser.add_argument("--token", type=str, help="HF API token (or set HF_TOKEN env)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--commit-message", type=str, help="Commit message")
    return parser.parse_args()


def main():
    args = parse_args()
    
    upload_to_hub(
        model_path=args.model,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
