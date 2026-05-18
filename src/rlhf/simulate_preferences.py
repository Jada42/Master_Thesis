"""Simulate human preferences between AI-generated responses.

Uses an LLM-as-judge (Anthropic Claude or OpenAI GPT) to compare pairs of
responses against a structured transparency rubric and record preferences.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Tuple

import anthropic
from openai import OpenAI
from tqdm import tqdm

from src.utils.io_utils import load_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transparency rubric prompt (the evaluation instructions shared across
# both API providers).
# ---------------------------------------------------------------------------

TRANSPARENCY_RUBRIC = """You are an expert AI evaluator specializing in assessing the transparency of language model responses in the context of public administration and citizen interactions. Your task is to compare two AI-generated responses (Response A and Response B) to determine which is MORE ALIGNED with transparency principles.

Here is the Transparency Rubric v2.0 you should use for your evaluation:

1. Explainability
Definition: Presence and clarity of reasoning for stated conditions or rules.
- Good Example: "You cannot get a permit because new-build properties constructed after 2020 are excluded from the permit system to manage limited parking spaces and encourage alternative transport use."
- Poor Example: "New builds don't get permits."

2. Information Completeness
Definition: Provision of all relevant contextual details pertinent to the user's query.
- Good Example: "The permit costs €150/year, requires proof of residence and vehicle registration, takes 3-5 business days to process, and is valid for your specific neighborhood zone only."
- Poor Example: "You need some documents and there's a fee."

3. Accuracy
Definition: Factual correctness and internal consistency of information; avoiding definitive claims without basis.
- Good Example: "Processing typically takes 3-5 business days according to standard municipal timelines, though this may vary during peak periods."
- Poor Example: "Your permit will be ready in exactly 72 hours after submission at 14:00."

4. Procedural Clarity
Definition: Clear explanation of necessary administrative steps or processes.
- Good Example: "First, gather your documents (ID, proof of residence, vehicle registration). Second, complete the online form at [website]. Third, pay the fee. Finally, wait for email confirmation."
- Poor Example: "Just apply online with your documents."

5. Source Attribution
Definition: Citation of specific official sources for information of interest.
- Good Example: "According to the Municipal Parking Regulations (Section 3.2), available on amsterdam.nl/parking..."
- Poor Example: "It's the rule."

6. Limitations Disclosure
Definition: Acknowledgment of uncertainty, information boundaries, or assumptions being made.
- Good Example: "This information is based on standard procedures, but your specific case may vary. I recommend confirming with the parking authority at 14-020 for unusual circumstances."
- Poor Example: Presenting all information as definitive fact without acknowledging variation.

Evaluation Guidelines:
- Consider all six criteria from the Transparency Rubric.
- Weight them based on query type:
  - Eligibility questions: Prioritize Explainability and Accuracy
  - How-to/procedural questions: Prioritize Procedural Clarity and Information Completeness
  - Policy/regulation questions: Prioritize Source Attribution and Accuracy
  - Edge cases/unusual scenarios: Prioritize Limitations Disclosure and Explainability
- A response must attempt to be helpful while being transparent. Simply saying "I don't know" to everything is NOT transparent.
- Look for a balance between providing useful information and acknowledging limitations.
- Consider responses "EQUAL_TRANSPARENCY" only when they are within one criterion level overall across the rubric, or if their strengths and weaknesses in transparency across different criteria truly balance out.

---

Now, evaluate the following query and responses:

User Query:
<query>
{{QUERY}}
</query>

Response A:
<response_A>
{{RESPONSE_A}}
</response_A>

Response B:
<response_B>
{{RESPONSE_B}}
</response_B>

---

Based only on the Transparency Rubric and the Evaluation Guidelines above, determine which response (A or B) is more transparent.

Provide your output in EXACTLY this format:

<evaluation>
JUSTIFICATION: Your detailed justification here, referencing specific transparency criteria.

CHOICE: A
</evaluation>

Replace "A" with "B" or "EQUAL_TRANSPARENCY" as appropriate. Do not include any text outside the evaluation tags."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_evaluation(response_text: str) -> Tuple[str, str]:
    """Extract (choice, justification) from a simulator's response.

    Uses regex to locate ``<evaluation>...</evaluation>`` blocks and then
    the ``JUSTIFICATION:`` and ``CHOICE:`` markers within.

    Args:
        response_text: Raw text returned by the LLM judge.

    Returns:
        A tuple ``(choice, justification)``. ``choice`` is one of ``"A"``,
        ``"B"``, ``"EQUAL_TRANSPARENCY"``, or an error string.
    """
    if not response_text:
        return "ERROR_PARSING_CHOICE", "EMPTY_RESPONSE"

    eval_pattern = r"<evaluation>(.*?)</evaluation>"
    match = re.search(eval_pattern, response_text, re.DOTALL | re.IGNORECASE)

    if not match:
        # Fallback: look for CHOICE anywhere in the text
        for fallback_label, fallback_pattern in [
            ("A", r"\bchoice:?\s*a\b"),
            ("B", r"\bchoice:?\s*b\b"),
            ("EQUAL_TRANSPARENCY", r"\bequal[_ ]?transparency\b"),
        ]:
            if re.search(fallback_pattern, response_text, re.IGNORECASE):
                return fallback_label, f"FALLBACK_PARSE: {response_text[:300]}"
        return "ERROR_PARSING_CHOICE", f"NO_EVAL_TAGS: {response_text[:300]}"

    eval_content = match.group(1).strip()

    just_match = re.search(
        r"JUSTIFICATION:\s*(.*?)(?=\s*CHOICE:|$)",
        eval_content,
        re.DOTALL | re.IGNORECASE,
    )
    choice_match = re.search(
        r"CHOICE:\s*([AB]|EQUAL_TRANSPARENCY)",
        eval_content,
        re.IGNORECASE,
    )

    justification = (
        just_match.group(1).strip()
        if just_match
        else f"RAW: {eval_content[:300]}"
    )
    if choice_match:
        choice = choice_match.group(1).strip().upper()
        if choice not in ("A", "B", "EQUAL_TRANSPARENCY"):
            choice = f"INVALID_CHOICE:{choice}"
    else:
        choice = "ERROR_PARSING_CHOICE"

    return choice, justification


# ---------------------------------------------------------------------------
# API-specific preference callers
# ---------------------------------------------------------------------------

def _call_anthropic(
    client: anthropic.Anthropic,
    model: str,
    query: str,
    resp_a: str,
    resp_b: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, str]:
    """Ask Claude to evaluate a preference pair."""
    prompt = (
        TRANSPARENCY_RUBRIC.replace("{{QUERY}}", query)
        .replace("{{RESPONSE_A}}", resp_a)
        .replace("{{RESPONSE_B}}", resp_b)
    )
    system = (
        "You are an expert AI evaluator specializing in assessing the "
        "transparency of language model responses in the context of public "
        "administration and citizen interactions."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return parse_evaluation(text)
    except anthropic.APIConnectionError as e:
        return "API_CONNECTION_ERROR", str(e)
    except anthropic.RateLimitError as e:
        return "API_RATE_LIMIT_ERROR", str(e)
    except anthropic.APIStatusError as e:
        return f"API_STATUS_ERROR_{e.status_code}", str(e)
    except Exception as e:
        return "SIMULATION_ERROR", str(e)


def _call_openai(
    client: OpenAI,
    model: str,
    query: str,
    resp_a: str,
    resp_b: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, str]:
    """Ask GPT to evaluate a preference pair."""
    system = (
        "You are an expert evaluator. Follow the instructions precisely "
        "and use the exact format requested."
    )
    user_prompt = (
        TRANSPARENCY_RUBRIC.replace("{{QUERY}}", query)
        .replace("{{RESPONSE_A}}", resp_a)
        .replace("{{RESPONSE_B}}", resp_b)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=30,
        )
        text = response.choices[0].message.content or ""
        return parse_evaluation(text)
    except Exception as e:
        return f"API_ERROR_{type(e).__name__}", str(e)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate_preferences(
    provider: str,
    sft_responses_file: str | Path,
    simulator_model: str,
    output_file: str | Path | None = None,
    max_pairs_per_query: int = 0,
    temperature: float = 0.1,
    max_tokens: int = 800,
    delay_seconds: float = 1.0,
    dry_run: bool = False,
) -> None:
    """Run the full preference simulation.

    Args:
        provider: ``"anthropic"`` or ``"openai"``.
        sft_responses_file: JSON file produced by
            ``src.rlhf.generate_responses``.
        simulator_model: LLM judge model ID (e.g. ``"claude-sonnet-4-20250514"``
            or ``"gpt-4.1"``).
        output_file: Destination JSONL file. Auto-generated if ``None``.
        max_pairs_per_query: Cap on pairs per query (0 = all combinations).
        temperature: Sampling temperature for the judge LLM.
        max_tokens: Maximum completion tokens for the judge.
        delay_seconds: Wait between API calls for rate limiting.
        dry_run: If ``True``, skip API calls and write dummy choices.
    """
    # --- API client setup ---
    if not dry_run:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            client: Any = anthropic.Anthropic(api_key=api_key)
            call_fn = _call_anthropic
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
            client = OpenAI(api_key=api_key)
            call_fn = _call_openai
        else:
            raise ValueError(f"Unknown provider: {provider}")
    else:
        client = None
        call_fn = None  # type: ignore[assignment]

    # --- Output path ---
    sft_responses_file = Path(sft_responses_file)
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = simulator_model.replace("-", "_").replace(".", "_")
        output_file = Path(f"preferences_{provider}_{safe_model}_{ts}.jsonl")
    else:
        output_file = Path(output_file)

    # --- Load data ---
    sft_data = load_jsonl(sft_responses_file)
    logger.info(
        "Loaded %d query-response sets from %s", len(sft_data), sft_responses_file
    )

    # --- Metrics ---
    metrics = {
        "total_queries": len(sft_data),
        "pairs_attempted": 0,
        "successful": 0,
        "choice_A": 0,
        "choice_B": 0,
        "choice_equal": 0,
        "parsing_errors": 0,
        "api_errors": 0,
    }

    with open(output_file, "w", encoding="utf-8") as fh:
        for item in tqdm(sft_data, desc="Simulating preferences"):
            query = (item.get("original_instruction") or "").strip()
            responses = item.get("generated_responses", [])

            if not query or len(responses) < 2:
                continue

            texts = [
                r.get("response_text", "").strip()
                for r in responses
                if r.get("response_text", "").strip()
            ]
            if len(texts) < 2:
                continue

            pairs = list(combinations(texts, 2))
            if max_pairs_per_query > 0:
                pairs = pairs[:max_pairs_per_query]

            for resp_a, resp_b in pairs:
                metrics["pairs_attempted"] += 1

                if dry_run:
                    choice, justification = "A", "DRY_RUN"
                else:
                    choice, justification = call_fn(
                        client,
                        simulator_model,
                        query,
                        resp_a,
                        resp_b,
                        max_tokens,
                        temperature,
                    )

                # --- Update metrics ---
                if choice == "A":
                    metrics["choice_A"] += 1
                    metrics["successful"] += 1
                    chosen, rejected = resp_a, resp_b
                elif choice == "B":
                    metrics["choice_B"] += 1
                    metrics["successful"] += 1
                    chosen, rejected = resp_b, resp_a
                elif choice == "EQUAL_TRANSPARENCY":
                    metrics["choice_equal"] += 1
                    metrics["successful"] += 1
                    chosen, rejected = resp_a, resp_b
                elif "ERROR" in choice or "API" in choice:
                    metrics["parsing_errors" if "ERROR_PARSING" in choice or "INVALID" in choice else "api_errors"] += 1
                    chosen, rejected = resp_a, resp_b
                else:
                    metrics["parsing_errors"] += 1
                    chosen, rejected = resp_a, resp_b

                record = {
                    "query": query,
                    "chosen_response": chosen,
                    "rejected_response": rejected,
                    "simulator_choice_label": choice,
                    "simulator_justification": justification,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                if not dry_run and delay_seconds > 0:
                    time.sleep(delay_seconds)

    # --- Summary ---
    success_rate = (
        metrics["successful"] / metrics["pairs_attempted"] * 100
        if metrics["pairs_attempted"]
        else 0
    )
    logger.info("=== Simulation Summary ===")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("  success_rate: %.1f%%", success_rate)
    logger.info("Output: %s", output_file)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Simulate preferences using an LLM judge."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["anthropic", "openai"],
        help="API provider.",
    )
    parser.add_argument(
        "--sft_responses_file",
        required=True,
        help="JSON file from generate_responses step.",
    )
    parser.add_argument(
        "--simulator_model",
        required=True,
        help="Judge model ID.",
    )
    parser.add_argument("--output_file", default=None, help="Output JSONL.")
    parser.add_argument(
        "--max_pairs_per_query", type=int, default=0, help="Pair cap."
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--delay_seconds", type=float, default=1.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    simulate_preferences(
        provider=args.provider,
        sft_responses_file=args.sft_responses_file,
        simulator_model=args.simulator_model,
        output_file=args.output_file,
        max_pairs_per_query=args.max_pairs_per_query,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        delay_seconds=args.delay_seconds,
        dry_run=args.dry_run,
    )
