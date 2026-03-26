"""Model loading and PEFT setup for LLM fine-tuning."""

import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """Load tokenizer for a given model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Loaded tokenizer: {model_name}")
    return tokenizer


def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """Load model with optional 4-bit quantization."""
    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            torch_dtype=compute_dtype,
        )
        print(f"Loaded model (4-bit): {model_name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            torch_dtype=compute_dtype,
        )
        print(f"Loaded model: {model_name}")
    
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")
    return model


def setup_peft_model(
    model,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> None:
    """Setup LoRA/QLoRA adapters on a model."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA setup: {trainable}/{total} parameters ({100*trainable/total:.2f}%)")
    return model
