"""Generate diverse responses from an SFT model for RLHF preference data.

Loads a query file (one question per JSONL line), runs the SFT model
with varying temperatures to produce multiple responses per query, and
saves the results as a JSON array.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import (
    BASE_MODEL_NAME,
    INFERENCE_TOP_P,
    NUM_RESPONSES_PER_QUERY,
)
from src.utils.io_utils import load_jsonl
from src.utils.model_utils import (
    format_chat_prompt,
    get_device,
    load_model_with_adapters,
)

logger = logging.getLogger(__name__)


def generate_multiple(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    num_responses: int = NUM_RESPONSES_PER_QUERY,
    max_new_tokens: int = 300,
    base_temperature: float = 0.6,
    temp_variation: float = 0.2,
    top_p: float = INFERENCE_TOP_P,
) -> list[dict[str, Any]]:
    """Produce several responses for a single prompt by varying temperature.

    Args:
        model: PEFT model in eval mode.
        tokenizer: Matching tokenizer.
        device: torch device.
        prompt: Formatted chat prompt.
        num_responses: How many completions to generate.
        max_new_tokens: Max generation length per response.
        base_temperature: Starting sampling temperature.
        temp_variation: Increment added to temperature for each response.
        top_p: Nucleus sampling parameter.

    Returns:
        List of dicts with ``generation_run``, ``temperature_used``, and
        ``response_text`` keys.
    """
    responses: list[dict[str, Any]] = []
    for i in range(num_responses):
        temp = base_temperature + i * temp_variation
        temp = max(0.1, min(temp, 1.5))

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        responses.append(
            {
                "generation_run": i + 1,
                "temperature_used": round(temp, 2),
                "response_text": text,
            }
        )
    return responses


def generate_all_responses(
    query_file: str | Path,
    adapter_path: str | Path,
    base_model: str = BASE_MODEL_NAME,
    output_file: str | Path | None = None,
    num_responses: int = NUM_RESPONSES_PER_QUERY,
    max_new_tokens: int = 300,
    base_temperature: float = 0.6,
    temp_variation: float = 0.2,
) -> list[dict[str, Any]]:
    """Generate diverse SFT responses for all queries.

    Args:
        query_file: JSONL file with ``instruction`` field per line.
        adapter_path: LoRA adapter directory.
        base_model: HuggingFace model ID.
        output_file: Destination for the output JSON. Auto-generated if
            ``None``.
        num_responses: Completions to generate per query.
        max_new_tokens: Max tokens for each response.
        base_temperature: Starting temperature.
        temp_variation: Temperature increment per generation run.

    Returns:
        The full list of result dicts (also saved to disk).
    """
    device = get_device()
    model, tokenizer = load_model_with_adapters(
        base_model, adapter_path, device
    )

    queries = load_jsonl(query_file)
    logger.info("Loaded %d queries from %s", len(queries), query_file)

    all_results: list[dict[str, Any]] = []
    for item in tqdm(queries, desc="Generating per query"):
        instruction = item.get("instruction", "").strip()
        if not instruction:
            continue

        prompt = format_chat_prompt(instruction, tokenizer)
        try:
            responses = generate_multiple(
                model,
                tokenizer,
                device,
                prompt,
                num_responses=num_responses,
                max_new_tokens=max_new_tokens,
                base_temperature=base_temperature,
                temp_variation=temp_variation,
            )
        except Exception as exc:
            logger.error("Error for instruction '%s...': %s", instruction[:60], exc)
            responses = [{"error": str(exc)}]

        all_results.append(
            {
                "original_instruction": instruction,
                "formatted_prompt": prompt,
                "generated_responses": responses,
            }
        )

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"rlhf_responses_{ts}.json")

    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d query-response sets to %s", len(all_results), output_file)

    return all_results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate diverse SFT responses for RLHF."
    )
    parser.add_argument(
        "--query_file", required=True, help="Path to query JSONL."
    )
    parser.add_argument(
        "--adapter_path", required=True, help="LoRA adapter directory."
    )
    parser.add_argument(
        "--base_model", default=BASE_MODEL_NAME, help="Base model ID."
    )
    parser.add_argument("--output_file", default=None, help="Output JSON.")
    parser.add_argument(
        "--num_responses",
        type=int,
        default=NUM_RESPONSES_PER_QUERY,
    )
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--base_temperature", type=float, default=0.6)
    parser.add_argument("--temp_variation", type=float, default=0.2)
    args = parser.parse_args()

    generate_all_responses(
        query_file=args.query_file,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        output_file=args.output_file,
        num_responses=args.num_responses,
        max_new_tokens=args.max_new_tokens,
        base_temperature=args.base_temperature,
        temp_variation=args.temp_variation,
    )
