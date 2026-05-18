"""Preprocessing: tokenize Q&A pairs with Mistral chat-template masking.

Loads the raw JSONL dataset, applies the model's chat template, tokenizes,
masks the prompt portion of the labels (so the loss is computed only on
the assistant's response), and saves the processed dataset to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from src.config import BASE_MODEL_NAME, SFT_DATASET_JSONL, TOKENIZED_DATASET_DIR

logger = logging.getLogger(__name__)


def _build_chat_template_prompt(
    instruction: str,
    response: str,
    tokenizer: AutoTokenizer,
) -> tuple[str, str]:
    """Format a single turn and return (full_text, prompt_only_text).

    The full text includes both user and assistant messages. The prompt-only
    text includes the user message with a generation prompt so we can
    determine which tokens to mask during loss computation.
    """
    messages_full = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    messages_prompt = [{"role": "user", "content": instruction}]

    full = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )
    prompt = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    return full, prompt


def _preprocess_batch(
    examples: dict[str, list[str]],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> dict[str, list[list[int]]]:
    """Tokenize a batch and mask prompt tokens in the labels."""
    input_ids_batch: list[list[int]] = []
    attention_masks_batch: list[list[int]] = []
    labels_batch: list[list[int]] = []

    instructions = examples["instruction"]
    outputs = examples["output"]

    for instruction, output in zip(instructions, outputs):
        instruction = str(instruction or "").strip()
        output = str(output or "").strip()

        if not instruction and not output:
            continue

        try:
            full_text, prompt_text = _build_chat_template_prompt(
                instruction, output, tokenizer
            )
        except Exception:
            logger.warning(
                "Failed to apply chat template for instruction starting with "
                "'%s'. Skipping.", instruction[:60]
            )
            continue

        tokenized_full = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        tokenized_prompt = tokenizer(
            prompt_text,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = list(input_ids)

        prompt_len = min(len(tokenized_prompt["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        input_ids_batch.append(input_ids)
        attention_masks_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_masks_batch,
        "labels": labels_batch,
    }


def preprocess_dataset(
    model_name: str = BASE_MODEL_NAME,
    raw_path: str | Path = SFT_DATASET_JSONL,
    output_dir: str | Path = TOKENIZED_DATASET_DIR,
    max_length: int = 1024,
) -> Dataset:
    """Load, tokenize, and save the SFT dataset with label masking.

    Args:
        model_name: HuggingFace model identifier used for the tokenizer and
            chat template.
        raw_path: Path to the ``.jsonl`` file containing ``instruction`` and
            ``output`` fields.
        output_dir: Directory where the tokenized dataset will be saved via
            ``Dataset.save_to_disk``.
        max_length: Maximum token length for truncated sequences.

    Returns:
        The tokenized ``Dataset`` object.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    logger.info("Loading raw dataset from %s", raw_path)
    dataset = load_dataset("json", data_files=str(raw_path), split="train")
    logger.info("Loaded %d examples. Columns: %s", len(dataset), dataset.column_names)

    original_cols = list(dataset.column_names)
    tokenized = dataset.map(
        lambda batch: _preprocess_batch(batch, tokenizer, max_length),
        batched=True,
        remove_columns=original_cols,
    )

    logger.info("Saving tokenized dataset to %s", output_dir)
    tokenized.save_to_disk(str(output_dir))
    logger.info("Done. %d tokenized examples saved.", len(tokenized))

    return tokenized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_dataset()
