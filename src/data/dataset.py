"""Dataset helpers for multimodal fine-tuning."""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request
import zipfile
from typing import Callable, Dict

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


def maybe_download_imageclef_2025(data_dir: str, url_env_var: str = "IMAGECLEF_2025_URL") -> Path:
    """Prepare ImageCLEF 2025 data directory.

    ImageCLEF releases require credentials/registration, so we cannot hard-code a URL.
    If the expected files are missing, this checks an env var for a private URL and
    attempts to download. Otherwise, it instructs the user to download manually.
    """
    data_dir_path = Path(data_dir).expanduser().resolve()
    data_dir_path.mkdir(parents=True, exist_ok=True)

    # If files already exist, return immediately.
    expected_files = [
        data_dir_path / "captioning.jsonl",
        data_dir_path / "concept_detection.jsonl",
        data_dir_path / "explainability.jsonl",
    ]
    if all(f.exists() for f in expected_files):
        return data_dir_path

    url = os.environ.get(url_env_var)
    if not url:
        raise FileNotFoundError(
            "ImageCLEF 2025 data not found. Place captioning.jsonl, concept_detection.jsonl, "
            "and explainability.jsonl under the data directory, or set IMAGECLEF_2025_URL "
            "to an authenticated download link (e.g., a private ZIP) before running."
        )

    # Try to download a ZIP archive from the provided URL.
    zip_path = data_dir_path / "imageclef_2025.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:  # pragma: no cover - network env dependent
        raise FileNotFoundError(
            f"Failed to download ImageCLEF archive from {url}. "
            f"Ensure the URL is reachable and authenticated."
        ) from exc

    if zip_path.suffix == ".zip":
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir_path)
    else:
        raise FileNotFoundError(
            f"Downloaded file {zip_path} is not a ZIP. "
            f"Place the expected JSONL files manually in {data_dir_path}."
        )

    if not all(f.exists() for f in expected_files):
        raise FileNotFoundError(
            f"After extraction, expected files are still missing: {[str(f) for f in expected_files]}"
        )

    return data_dir_path


def load_imageclef_2025_splits(data_dir: str) -> Dict[str, object]:
    """Load ImageCLEF 2025 splits (captioning, concept detection, explainability).

    Expects JSONL files with columns: image_path, instruction, output.
    """
    data_root = maybe_download_imageclef_2025(data_dir)
    files = {
        "captioning": data_root / "captioning.jsonl",
        "concept_detection": data_root / "concept_detection.jsonl",
        "explainability": data_root / "explainability.jsonl",
    }

    return {
        split: load_instruction_dataset(str(path))
        for split, path in files.items()
    }
