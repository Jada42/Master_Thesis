"""Reward model training.

Trains a Bradley-Terry pairwise preference model on chosen/rejected response
pairs using a LoRA-tuned Mistral-7B backbone with a linear reward head.
Designed for CUDA (RunPod); uses full FP32 precision for numeric stability.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BASE_MODEL_NAME,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    RM_BATCH_SIZE,
    RM_EPOCHS,
    RM_GRAD_ACCUMULATION,
    RM_GRADIENT_CLIP,
    RM_LEARNING_RATE,
    RM_MAX_LENGTH,
    RM_WARMUP_RATIO,
    RM_WEIGHT_DECAY,
)
from src.rlhf.reward_model.dataset import PreferenceDataset
from src.rlhf.reward_model.model import RewardModel, pairwise_ranking_loss

logger = logging.getLogger(__name__)


def train_epoch(
    model: RewardModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_accum: int,
    grad_clip: float,
) -> float:
    """Run one training epoch.

    Returns average loss over the data loader.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        loss, _, _ = pairwise_ranking_loss(model, batch, device)
        loss = loss / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

    return total_loss / len(loader)


def evaluate(
    model: RewardModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on validation set.

    Returns ``(avg_loss, accuracy)``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            loss, chosen, rejected = pairwise_ranking_loss(model, batch, device)
            total_loss += loss.item()
            correct += (chosen > rejected).sum().item()
            total += chosen.size(0)

    return total_loss / len(loader), correct / total


def train_reward_model(
    preference_csv: str | Path,
    output_dir: str | Path = "./reward_model_output",
    base_model_name: str = BASE_MODEL_NAME,
    max_length: int = RM_MAX_LENGTH,
    batch_size: int = RM_BATCH_SIZE,
    grad_accum: int = RM_GRAD_ACCUMULATION,
    learning_rate: float = RM_LEARNING_RATE,
    epochs: int = RM_EPOCHS,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    warmup_ratio: float = RM_WARMUP_RATIO,
    weight_decay: float = RM_WEIGHT_DECAY,
    grad_clip: float = RM_GRADIENT_CLIP,
) -> None:
    """Train a reward model from preference pairs.

    Args:
        preference_csv: CSV with ``query``, ``chosen_response``,
            ``rejected_response`` columns.
        output_dir: Where to save checkpoints and adapters.
        base_model_name: HuggingFace model ID.
        max_length: Tokenization truncation length.
        batch_size: Per-GPU batch size.
        grad_accum: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        epochs: Number of training epochs.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        warmup_ratio: Fraction of steps used for linear warmup.
        weight_decay: AdamW weight decay.
        grad_clip: Max gradient norm.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Tokenizer & base model ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    base = get_peft_model(base, peft_config)
    base.print_trainable_parameters()

    model = RewardModel(base).to(device)

    # --- Data ---
    df = pd.read_csv(preference_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = PreferenceDataset(train_df, tokenizer, max_length)
    val_ds = PreferenceDataset(val_df, tokenizer, max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    train_losses, val_losses, val_accs = [], [], []
    best_acc = 0.0

    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, grad_accum, grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(
            "  Train loss: %.4f | Val loss: %.4f | Val acc: %.1f%%",
            train_loss, val_loss, val_acc * 100,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = output_dir / "best_reward_model"
            best_path.mkdir(exist_ok=True)
            model.base_model.save_pretrained(str(best_path))
            torch.save(model.reward_head.state_dict(), best_path / "reward_head.pt")
            logger.info("  New best model saved (acc=%.1f%%)", best_acc * 100)

    # --- Final model ---
    final_path = output_dir / "final_reward_model"
    final_path.mkdir(exist_ok=True)
    model.base_model.save_pretrained(str(final_path))
    torch.save(model.reward_head.state_dict(), final_path / "reward_head.pt")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, "o-", label="Train")
    ax1.plot(val_losses, "o-", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(val_accs, "o-", label="Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "training_results.png", dpi=150)
    plt.close(fig)

    logger.info("Training complete. Best val acc: %.1f%%", best_acc * 100)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a pairwise reward model.")
    parser.add_argument("--preference_csv", required=True, help="Combined preferences CSV.")
    parser.add_argument("--output_dir", default="./reward_model_output")
    parser.add_argument("--epochs", type=int, default=RM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=RM_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=RM_LEARNING_RATE)
    args = parser.parse_args()

    train_reward_model(
        preference_csv=args.preference_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
