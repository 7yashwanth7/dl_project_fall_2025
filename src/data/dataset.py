"""Dataset helpers for multimodal fine-tuning."""

from __future__ import annotations

from typing import Callable

from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor


def load_instruction_dataset(path: str):
    """Load a JSON/JSONL dataset with fields: image_path, instruction, output."""
    return load_dataset("json", data_files=path)["train"]


def build_vision_language_collator(
    processor: AutoProcessor,
    image_column: str,
    text_column: str,
    label_column: str,
    max_length: int = 512,
) -> Callable:
    """Create a collator that packs images and text into model-ready tensors."""

    def collate_fn(examples):
        images = [Image.open(example[image_column]).convert("RGB") for example in examples]
        prompts = [example[text_column] for example in examples]
        labels = [example[label_column] for example in examples]

        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        label_inputs = processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs["labels"] = label_inputs["input_ids"]
        return inputs

    return collate_fn
