"""Reward model evaluation.

Loads a trained reward model checkpoint and runs qualitative tests on
hand-crafted query-response pairs to verify preference ordering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import BASE_MODEL_NAME
from src.rlhf.reward_model.model import RewardModel
from src.utils.model_utils import get_device

logger = logging.getLogger(__name__)


def load_reward_model(
    checkpoint_dir: str | Path,
    base_model_name: str = BASE_MODEL_NAME,
) -> RewardModel:
    """Load a saved reward model from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing LoRA adapter files
            (``adapter_config.json``, ``adapter_model.safetensors``) and
            ``reward_head.pt``.
        base_model_name: HuggingFace model ID.

    Returns:
        The reward model on the detected device, in evaluation mode.
    """
    checkpoint_dir = Path(checkpoint_dir)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    base.eval()

    peft_model = PeftModel.from_pretrained(
        base, str(checkpoint_dir), is_trainable=False
    )
    peft_model.eval()

    model = RewardModel(peft_model)

    head_path = checkpoint_dir / "reward_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"reward_head.pt not found in {checkpoint_dir}")
    model.reward_head.load_state_dict(torch.load(head_path, map_location="cpu"))

    model.to(device)
    model.eval()
    return model


def score_response(
    model: RewardModel,
    tokenizer: AutoTokenizer,
    query: str,
    response: str,
    max_length: int = 512,
) -> float:
    """Compute the reward score for a single query-response pair.

    Args:
        model: The loaded reward model.
        tokenizer: Matching tokenizer.
        query: Natural-language query.
        response: The model's answer.
        max_length: Tokenization truncation length.

    Returns:
        Scalar reward score (higher = more preferred).
    """
    formatted = f"Query: {query}\n\nResponse: {response}"
    inputs = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        score = model(
            inputs["input_ids"].to(device),
            inputs["attention_mask"].to(device),
        )
    return float(score.item())


def evaluate_checkpoint(checkpoint_dir: str | Path) -> None:
    """Run qualitative evaluation on a few hand-crafted test cases.

    Args:
        checkpoint_dir: Path to the reward model checkpoint directory.
    """
    model = load_reward_model(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    test_cases = [
        {
            "query": "What are the environmental rules for diesel cars for a parking permit?",
            "response_good": (
                "For diesel cars, the emission standard must be Euro 5 or higher. "
                "Reference: Municipality parking rules on amsterdam.nl/parking. "
                "If your car does not meet Euro 5, you are not eligible for a permit."
            ),
            "response_poor": "Diesel cars need Euro 5.",
        },
        {
            "query": "How do I apply by post for a parking permit?",
            "response_good": (
                "To apply by post, download the application form from the official "
                "website, fill it out, attach copies of required documents, and mail "
                "everything to the address listed on the form. The municipality aims "
                "to respond within 5 working days. (Source: amsterdam.nl/parking-postal)"
            ),
            "response_poor": "Send your documents by mail.",
        },
    ]

    logger.info("--- Qualitative Reward Model Evaluation ---")
    for i, case in enumerate(test_cases):
        good = score_response(model, tokenizer, case["query"], case["response_good"])
        poor = score_response(model, tokenizer, case["query"], case["response_poor"])
        correct = good > poor
        logger.info(
            "Case %d: good=%.4f  poor=%.4f  correct=%s", i + 1, good, poor, correct
        )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate a trained reward model.")
    parser.add_argument(
        "--checkpoint_dir", required=True, help="Path to reward model checkpoint."
    )
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint_dir)
