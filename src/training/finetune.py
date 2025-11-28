"""Fine-tuning pipeline for multimodal models using LoRA."""

from __future__ import annotations

from typing import Optional

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

from src.config import FinetuneConfig
from src.data.dataset import build_vision_language_collator, load_instruction_dataset


def _apply_lora_adapter(model, cfg: FinetuneConfig):
    """Wrap the base model with a LoRA adapter."""
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules or None,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, lora_config)


def fine_tune_model(
    model,
    processor,
    cfg: FinetuneConfig,
    dataset: Optional[Dataset] = None,
):
    """Run a LoRA fine-tuning loop on a multimodal dataset."""
    train_ds = dataset or load_instruction_dataset(cfg.dataset_path)

    collate_fn = build_vision_language_collator(
        processor=processor,
        image_column=cfg.image_column,
        text_column=cfg.text_column,
        label_column=cfg.label_column,
        max_length=cfg.max_length,
    )

    model = _apply_lora_adapter(model, cfg)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    return model
