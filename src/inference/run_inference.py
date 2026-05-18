"""Inference with a LoRA-fine-tuned SFT model.

Loads the base Mistral model with LoRA adapters and generates answers
for each question in a validation CSV, saving the results as JSON.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import (
    BASE_MODEL_NAME,
    INFERENCE_MAX_NEW_TOKENS,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
)
from src.utils.model_utils import (
    format_chat_prompt,
    get_device,
    load_model_with_adapters,
)

logger = logging.getLogger(__name__)


def load_validation_queries(csv_path: str | Path) -> list[dict[str, str]]:
    """Read validation queries from a CSV with columns ID, Question, Answer, Source_Note_or_URL.

    Args:
        csv_path: Path to the validation CSV.

    Returns:
        A list of dicts with keys ``id``, ``question``, ``expected_answer``,
        and ``source_note``.
    """
    csv_path = Path(csv_path)
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "id": row["ID"],
                    "question": row["Question"],
                    "expected_answer": row["Answer"],
                    "source_note": row.get("Source_Note_or_URL", ""),
                }
            )
    logger.info("Loaded %d validation queries from %s", len(rows), csv_path)
    return rows


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS,
    temperature: float = INFERENCE_TEMPERATURE,
    top_p: float = INFERENCE_TOP_P,
) -> str:
    """Generate a single answer from a formatted prompt.

    Args:
        model: A PEFT-wrapped model in evaluation mode.
        tokenizer: The tokenizer matching the model.
        prompt: The formatted chat prompt string.
        max_new_tokens: Maximum tokens to generate beyond the input.
        temperature: Sampling temperature (higher = more diverse).
        top_p: Nucleus sampling threshold.

    Returns:
        The generated text with special tokens stripped.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_inference(
    base_model: str = BASE_MODEL_NAME,
    adapter_path: str | Path = "",
    validation_csv: str | Path = "",
    output_file: str | Path | None = None,
    max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS,
    temperature: float = INFERENCE_TEMPERATURE,
    top_p: float = INFERENCE_TOP_P,
) -> dict[str, Any]:
    """Run inference and return results dict (also saved to disk).

    Args:
        base_model: HuggingFace model identifier.
        adapter_path: Directory containing LoRA adapters.
        validation_csv: Path to the validation-queries CSV.
        output_file: Where to save results. Auto-generated if ``None``.
        max_new_tokens: Max tokens to generate per answer.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        A dictionary with ``metadata`` and ``results`` keys.
    """
    device = get_device()
    logger.info("Device: %s", device)

    model, tokenizer = load_model_with_adapters(
        base_model, adapter_path, device
    )
    queries = load_validation_queries(validation_csv)

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "base_model": base_model,
            "adapter_path": str(adapter_path),
            "num_queries": len(queries),
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        },
        "results": [],
    }

    for query in tqdm(queries, desc="Generating"):
        prompt = format_chat_prompt(query["question"], tokenizer)
        try:
            answer = generate_answer(
                model, tokenizer, prompt, max_new_tokens, temperature, top_p
            )
        except Exception as exc:
            logger.error("Error on query %s: %s", query["id"], exc)
            answer = f"ERROR: {exc}"

        results["results"].append(
            {
                "query_id": query["id"],
                "question": query["question"],
                "generated_answer": answer,
                "expected_answer": query["expected_answer"],
                "source_note": query["source_note"],
            }
        )

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"sft_inference_{ts}.json")

    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s (%d queries)", output_file, len(queries))
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run inference with a LoRA-tuned SFT model."
    )
    parser.add_argument(
        "--validation_csv",
        required=True,
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--base_model",
        default=BASE_MODEL_NAME,
        help="Base model ID.",
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output JSON path (auto-generated if omitted).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=INFERENCE_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--temperature", type=float, default=INFERENCE_TEMPERATURE
    )
    parser.add_argument("--top_p", type=float, default=INFERENCE_TOP_P)
    args = parser.parse_args()

    run_inference(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        validation_csv=args.validation_csv,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
