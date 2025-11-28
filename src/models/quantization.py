"""Utilities for quantizing and loading models."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import BitsAndBytesConfig

from src.config import QuantizationConfig


def build_bnb_config(cfg: Optional[QuantizationConfig]) -> Optional[BitsAndBytesConfig]:
    """Create a BitsAndBytesConfig from user-specified options."""
    if cfg is None or not cfg.load_in_4bit:
        return None

    # bitsandbytes 4-bit requires CUDA; fall back to full precision on CPU/MPS.
    if not torch.cuda.is_available():
        return None

    compute_dtype = getattr(torch, cfg.bnb_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def quantize_model(model: torch.nn.Module, cfg: Optional[QuantizationConfig]) -> torch.nn.Module:
    """Placeholder for post-loading quantization steps (e.g., GPTQ).

    With bitsandbytes, quantization happens at load time, so this function simply
    returns the model. Extend this to run offline quantization (e.g., GPTQ or AWQ)
    if desired.
    """
    _ = cfg  # kept for signature parity
    return model
