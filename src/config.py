"""Shared configuration constants for the RLHF pipeline.

All paths, hyperparameters, and model identifiers are collected here
so that individual scripts can import them without duplication.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (relative to this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"

# ---------------------------------------------------------------------------
# LoRA defaults (shared across SFT and reward model training)
# ---------------------------------------------------------------------------
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ---------------------------------------------------------------------------
# SFT training defaults
# ---------------------------------------------------------------------------
SFT_BATCH_SIZE: int = 1
SFT_GRAD_ACCUMULATION: int = 8
SFT_LEARNING_RATE: float = 2e-4
SFT_EPOCHS: int = 1
SFT_MAX_LENGTH: int = 1024

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
INFERENCE_MAX_NEW_TOKENS: int = 512
INFERENCE_TEMPERATURE: float = 0.7
INFERENCE_TOP_P: float = 0.9

# ---------------------------------------------------------------------------
# RLHF / preference simulation
# ---------------------------------------------------------------------------
NUM_RESPONSES_PER_QUERY: int = 3
PREFERENCE_TEMPERATURE: float = 0.1
PREFERENCE_MAX_TOKENS: int = 800

# ---------------------------------------------------------------------------
# Reward model training
# ---------------------------------------------------------------------------
RM_BATCH_SIZE: int = 4
RM_GRAD_ACCUMULATION: int = 4
RM_LEARNING_RATE: float = 2e-5
RM_EPOCHS: int = 3
RM_MAX_LENGTH: int = 512
RM_WARMUP_RATIO: float = 0.1
RM_WEIGHT_DECAY: float = 0.01
RM_GRADIENT_CLIP: float = 1.0

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
SFT_DATASET_JSONL: Path = DATA_DIR / "parking_permit_sft_dataset.jsonl"
TOKENIZED_DATASET_DIR: Path = DATA_DIR / "tokenized_parking_permit_dataset"
VALIDATION_CSV: Path = DATA_DIR / "validation_qa.csv"
RLHF_QUERIES_JSONL: Path = DATA_DIR / "rlhf_queries_50.jsonl"
COMBINED_QUERIES_JSONL: Path = DATA_DIR / "combined_queries_100.jsonl"

# ---------------------------------------------------------------------------
# Evaluation dimensions
# ---------------------------------------------------------------------------
TRANSPARENCY_DIMENSIONS: list[str] = [
    "Explainability",
    "Completeness",
    "Accuracy",
    "Clarity",
    "Attribution",
    "Limitations",
]
