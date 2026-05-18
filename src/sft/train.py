"""SFT training entry point.

Fine-tunes Mistral-7B-Instruct-v0.3 on the preprocessed parking permit
Q&A dataset using LoRA. Supports three quantization modes:
``"none"`` (full-precision, suitable for MPS/CPU),
``"8bit"``, and ``"4bit"`` (QLoRA, suitable for CUDA GPUs).

Usage::

    python -m src.sft.train --quantization_mode 4bit
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BASE_MODEL_NAME,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_GRAD_ACCUMULATION,
    SFT_LEARNING_RATE,
    TOKENIZED_DATASET_DIR,
)
from src.utils.model_utils import (
    get_compute_dtype,
    get_device,
    load_base_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


def train_sft(
    quantization_mode: str = "none",
    model_name: str = BASE_MODEL_NAME,
    dataset_path: str | Path = TOKENIZED_DATASET_DIR,
    output_root: str | Path = "./sft_results",
    learning_rate: float = SFT_LEARNING_RATE,
    epochs: int = SFT_EPOCHS,
    batch_size: int = SFT_BATCH_SIZE,
    grad_accumulation: int = SFT_GRAD_ACCUMULATION,
    save_steps: int = 50,
    logging_steps: int = 10,
) -> None:
    """Run SFT training with LoRA.

    Args:
        quantization_mode: ``"none"``, ``"8bit"``, or ``"4bit"``.
        model_name: HuggingFace model ID.
        dataset_path: Directory containing the tokenized dataset produced
            by ``src.data_processing.preprocess``.
        output_root: Base directory for checkpoints and final adapters.
        learning_rate: Peak learning rate for AdamW.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        grad_accumulation: Gradient accumulation steps (effective batch =
            ``batch_size * grad_accumulation``).
        save_steps: Save a checkpoint every N steps.
        logging_steps: Log training metrics every N steps.
    """
    output_root = Path(output_root)
    device = get_device()
    logger.info("Device: %s | Quantization: %s", device, quantization_mode)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(model_name, padding_side="right")
    if tokenizer.pad_token != "[PAD]":
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found at {dataset_path}. "
            "Run 'python -m src.data_processing.preprocess' first."
        )
    train_data = load_from_disk(str(dataset_path))
    logger.info("Loaded %d training examples", len(train_data))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    base_model = load_base_model(model_name, device, quantization_mode)

    if tokenizer.pad_token == "[PAD]" and (
        base_model.get_input_embeddings().weight.size(0) != len(tokenizer)
    ):
        base_model.resize_token_embeddings(len(tokenizer))

    if quantization_mode != "none":
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )

    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    output_dir = output_root / f"mistral_lora_{quantization_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    enable_bf16 = False
    enable_fp16 = False
    if device.type == "cuda":
        if get_compute_dtype(device) == torch.bfloat16:
            enable_bf16 = True
        else:
            enable_fp16 = True

    is_quantized = quantization_mode != "none"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit" if is_quantized else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=enable_fp16,
        bf16=enable_bf16,
        remove_unused_columns=False,
        dataloader_num_workers=2 if device.type == "cuda" else 0,
        dataloader_pin_memory=device.type == "cuda",
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting SFT training (%s) ...", quantization_mode)
    trainer.train()

    # ------------------------------------------------------------------
    # Save adapters
    # ------------------------------------------------------------------
    final_path = output_dir / "final_model_adapters"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Adapters saved to %s", final_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="SFT training with LoRA")
    parser.add_argument(
        "--quantization_mode",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode (default: none).",
    )
    parser.add_argument(
        "--model_name", default=BASE_MODEL_NAME, help="Base model ID."
    )
    parser.add_argument(
        "--dataset_path",
        default=str(TOKENIZED_DATASET_DIR),
        help="Path to tokenized dataset.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=SFT_LEARNING_RATE
    )
    parser.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    args = parser.parse_args()

    train_sft(
        quantization_mode=args.quantization_mode,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
