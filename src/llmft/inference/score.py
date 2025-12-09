"""Batched scoring helpers for notebook and script usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm.auto import tqdm


def build_messages(sample: Dict[str, Any], system_message: str, user_prompt: str) -> List[Dict[str, Any]]:
    """Convert a sample with an image into chat messages."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image"},
            ],
        },
    ]


def prepare_batch(samples, processor, system_message, user_prompt):

    """Build model inputs for a batch of samples."""

    messages = [build_messages(s, system_message, user_prompt) for s in samples]
    chat_texts = [
        processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        for msg in messages
    ]
    images = [s["image"] for s in samples]
    inputs = processor(
        text=chat_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    return inputs


def decode_generations(
    generated_ids: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    tokenizer):
    
    """Decode only the generated tokens (strip the prompt)."""
    prompt_len = prompt_input_ids.shape[1]
    gen_only = generated_ids[:, prompt_len:]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)


def score_dataset(
    dataset: Any,
    model,
    processor,
    system_message: str,
    user_prompt: str,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate outputs for a dataset in batches."""
    gen_kwargs = gen_kwargs or {}
    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    results: List[Dict[str, Any]] = []
    model.eval()

    for start in tqdm(range(0, total, batch_size), desc="Scoring", leave=False):
        end = min(start + batch_size, total)
        batch = [dataset[i] for i in range(start, end)]

        inputs = prepare_batch(batch, processor, system_message, user_prompt)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        decoded = decode_generations(generated_ids, inputs["input_ids"], processor.tokenizer)

        for sample, output_text in zip(batch, decoded):
            results.append(
                {
                    "id": sample.get("id"),
                    "caption": sample.get("caption"),
                    "cui": sample.get("cui"),
                    "generation": output_text.strip(),
                }
            )

    return results


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def score_batch_dataset(
    datasets: Dict[str, Any],
    model,
    processor,
    system_message: str,
    user_prompt: str,
    output_dir: Path,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Score multiple datasets and save results per dataset."""
    paths: Dict[str, Path] = {}
    for name, ds in datasets.items():
        results = score_dataset(
            dataset=ds,
            model=model,
            processor=processor,
            system_message=system_message,
            user_prompt=user_prompt,
            batch_size=batch_size,
            max_samples=max_samples,
            gen_kwargs=gen_kwargs,
        )
        # save_jsonl(results, output_dir)
        # paths[name] = output_dir
    return results
