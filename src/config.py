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
# PPO / Constitutional AI training (Optuna-optimized, trial 43)
# ---------------------------------------------------------------------------
PPO_BATCH_SIZE: int = 6
PPO_MINI_BATCH_SIZE: int = 3
PPO_LEARNING_RATE: float = 9.391494524751205e-06
PPO_TARGET_KL: float = 0.021602267145796005
PPO_STEPS: int = 500
PPO_EPOCHS: int = 5
PPO_INIT_KL_COEF: float = 0.09327372581520726
PPO_GAMMA: float = 0.9789438458465525
PPO_VF_COEF: float = 0.36436016003108607

# Optimized reward-component weights
CONSTITUTIONAL_WEIGHT: float = 0.81   # Primary transparency driver
REWARD_MODEL_WEIGHT: float = 0.111    # Learned scorer (secondary)
COHERENCE_WEIGHT: float = 0.079       # Semantic coherence fine-tuning
CONSTITUTIONAL_LR: float = 1.7169751284913331e-06

# Constitutional iteration control
CONVERGENCE_THRESHOLD: float = 0.12701580628504341
MAX_CONSTITUTIONAL_ITERATIONS: int = 4

# Coherence module
COHERENCE_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
COHERENCE_HIDDEN_DIM: int = 512

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
