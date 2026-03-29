"""Training utilities for different fine-tuning methods."""

from typing import Any, Optional
from .config import FineTuningConfig


def get_trainer_args(cfg: FineTuningConfig) -> dict:
    """Convert FineTuningConfig to TrainingArguments dict."""
    return {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_epochs,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "max_grad_norm": cfg.gradient_clip,
        "optim": cfg.optim,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_strategy": "no",
        "bf16": cfg.bf16 if cfg.bf16 is not None else False,
        "fp16": cfg.fp16 if cfg.fp16 is not None else False,
    }


def create_lora_trainer(model, tokenizer, dataset, cfg: FineTuningConfig):
    """Create SFTTrainer for LoRA/QLoRA training."""
    from trl import SFTTrainer, SFTConfig
    
    trainer_args = get_trainer_args(cfg)
    
    # TRL 0.29+ passes dataset params to SFTConfig, not SFTTrainer
    training_args = SFTConfig(
        **trainer_args,
        dataset_text_field="text",
        max_length=cfg.max_length,
        dataset_num_proc=2,
        packing=cfg.packing,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    return trainer


def create_dpo_trainer(model, ref_model, tokenizer, dataset, cfg: FineTuningConfig):
    """Create DPOTrainer for preference optimization."""
    from trl import DPOTrainer, DPOConfig
    
    trainer_args = get_trainer_args(cfg)
    trainer_args["beta"] = getattr(cfg, "beta", 0.1)
    
    training_args = DPOConfig(**trainer_args)
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    return trainer
