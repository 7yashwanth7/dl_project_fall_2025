"""Notebook-friendly helpers for loading and previewing radiology data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset
from PIL import Image


@dataclass
class PromptConfig:
    """Configurable system/user prompts for experiments."""

    system_message: str = "You are a radiologist who can understand the medical scan of images."
    user_prompt: str = (
        "Create a description based on the provided image and return the description of the "
        "image with details of the scan and what's the ability. The description should be SEO "
        "optimized and in medical terms."
    )


def format_sample(sample: dict, prompt_config: PromptConfig) -> dict:
    """Convert a raw dataset sample into OAI-style messages."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_config.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_config.user_prompt},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.get("caption", "")}],
            },
        ],
    }


def load_roco_samples(
    dataset_name: str = "eltorio/ROCOv2-radiology",
    split: str = "train",
    sample_size: Optional[int] = None,
    sample_indices: Optional[List[int]] = None,
    seed: int = 42,
    prompt_config: Optional[PromptConfig] = None,
    token: Optional[str] = None,
    **dataset_kwargs,
) -> list[dict]:
    """Load ROCOv2 samples, optionally limiting to a subset and formatting messages.

    `token` is forwarded for gated datasets; any extra `dataset_kwargs` are passed to `load_dataset`.
    """
    prompt_config = prompt_config or PromptConfig()
    # Handle token parameter name differences across datasets versions.
    try:
        ds = load_dataset(dataset_name, split=split, token=token, **dataset_kwargs)
    except TypeError:
        ds = load_dataset(dataset_name, split=split, use_auth_token=token, **dataset_kwargs)

    if sample_indices is not None:
        ds = ds.select(sample_indices)
    elif sample_size is not None:
        ds = ds.shuffle(seed=seed).select(range(sample_size))

    return [format_sample(sample, prompt_config) for sample in ds]


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    """Extract RGB images from a conversation-style message list."""
    image_inputs: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element.get("image", element)
                image_inputs.append(image.convert("RGB"))
    return image_inputs


def preview_sample(formatted_samples: list[dict], index: int = 0, display: bool = False) -> dict:
    """Return a single formatted sample and optionally render its image/text."""
    if not formatted_samples:
        raise ValueError("No samples available to preview.")
    if index < 0 or index >= len(formatted_samples):
        raise IndexError(f"Sample index {index} out of range for {len(formatted_samples)} samples.")

    sample = formatted_samples[index]
    if display:
        try:
            from IPython.display import display as ipy_display
        except Exception:  # pragma: no cover - only relevant in notebooks
            ipy_display = None

        user_content = sample["messages"][1]["content"]
        assistant_content = sample["messages"][2]["content"][0]["text"]

        for item in user_content:
            if isinstance(item, dict) and item.get("type") == "image" and ipy_display:
                ipy_display(item["image"])
            elif isinstance(item, dict) and item.get("type") == "text":
                print("User Prompt:")
                print(item["text"])

        print("\nAssistant Description:")
        print(assistant_content)
    return sample
