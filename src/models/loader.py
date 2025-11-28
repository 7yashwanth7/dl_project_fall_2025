"""Model loader for multimodal vision-language LLMs."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from src.config import ModelConfig, QuantizationConfig
from src.models.quantization import build_bnb_config


def _resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return dtype_str
    return getattr(torch, dtype_str)


def load_multimodal_model(
    model_cfg: ModelConfig,
    quant_cfg: Optional[QuantizationConfig] = None,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    """Load a vision-language model and its processor."""
    quantization_config = build_bnb_config(quant_cfg)
    torch_dtype = _resolve_dtype(model_cfg.torch_dtype)
    attn_implementation = (
        model_cfg.attn_implementation if model_cfg.use_flash_attention_2 else None
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )

    processor = AutoProcessor.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    return model, processor
