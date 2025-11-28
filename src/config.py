"""Configuration objects for model loading, quantization, and fine-tuning."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """High-level model loading options."""

    model_name_or_path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = True
    attn_implementation: Optional[str] = "flash_attention_2"


@dataclass
class QuantizationConfig:
    """Quantization settings for bitsandbytes 4-bit loading."""

    load_in_4bit: bool = True
    bnb_compute_dtype: str = "float16"
    bnb_quant_type: str = "nf4"
    use_double_quant: bool = True


@dataclass
class FinetuneConfig:
    """Fine-tuning parameters for LoRA + Trainer."""

    output_dir: str = "outputs/multimodal-finetune"
    dataset_path: str = "data/train.jsonl"
    image_column: str = "image_path"
    text_column: str = "instruction"
    label_column: str = "output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_length: int = 512
    logging_steps: int = 10
    save_steps: int = 250
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    target_modules: List[str] = field(default_factory=list)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    bf16: bool = True
    fp16: bool = False
    packing: bool = False
