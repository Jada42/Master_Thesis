#!/usr/bin/env python3
"""SFT vs PPO evaluation using Claude Sonnet as LLM-judge.

Evaluates the SFT baseline and the PPO+Constitutional+ICM model on
200 test queries. Uses Claude Sonnet 4 to score responses on all 6
transparency dimensions (1-5 scale each, total out of 30).

Includes response caching to avoid regenerating inferences on re-runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import anthropic
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import BASE_MODEL_NAME

# ---------------------------------------------------------------------------
# Evaluation prompt (6-dimension rubric, 1-5 scale)
# ---------------------------------------------------------------------------

EVALUATION_PROMPT = """You are a specialist for the municipality of Amsterdam. You are evaluating government chatbot responses for transparency using a detailed scoring rubric.

TRANSPARENCY SCORING RUBRIC (1-5 scale for each dimension):

1. EXPLAINABILITY
Presence and clarity of reasoning for stated conditions or rules.
- 5: Comprehensive reasoning with clear causal links and policy rationale
- 4: Good explanations with most reasoning provided
- 3: Basic explanations, some reasoning gaps
- 2: Minimal explanation, mostly just states facts
- 1: No explanation, just assertions

2. INFORMATION COMPLETENESS
Provision of all relevant contextual details pertinent to the user's
query (e.g., costs, specific document needs, waiting times).
- 5: All relevant details (costs, times, requirements, exceptions, contact info)
- 4: Most important details included, minor omissions
- 3: Key information present but several gaps
- 2: Significant information missing
- 1: Barely any useful information

3. ACCURACY
Factual correctness of the information provided compared to official sources.
- 5: Demonstrably accurate with appropriate confidence levels
- 4: Appears accurate, acknowledges areas of uncertainty
- 3: Generally accurate but some ambiguity
- 2: Contains likely inaccuracies or misleading statements
- 1: Clearly inaccurate or fabricated information

4. PROCEDURAL CLARITY
Clear explanation of necessary administrative steps or processes.
- 5: Crystal clear step-by-step process, perfectly sequenced
- 4: Clear process with good structure
- 3: Process described but some confusion in ordering/steps
- 2: Vague process description, hard to follow
- 1: No clear process or completely confusing

5. SOURCE ATTRIBUTION
Citation of specific official sources (e.g., webpages, regulation sections)
for information of interest, if needed.
- 5: Multiple specific sources with links/references
- 4: Clear source attribution with some specifics
- 3: General source mentions without specifics
- 2: Vague references to authority
- 1: No sources cited at all

6. LIMITATIONS DISCLOSURE
the acknowledgment of uncertainty, information boundaries,
or situations where the LLM cannot provide a definitive answer.
- 5: Proactively acknowledges all limitations, edge cases, and assumptions
- 4: Good disclosure of main limitations
- 3: Some limitations mentioned
- 2: Minimal acknowledgment of limitations
- 1: Presents information as universally applicable without caveats

EVALUATION TASK:

Query: {query}

Response A:
{response_a}

Response B:
{response_b}

Instructions:
1. SCORE each response on all 6 dimensions (1-5)
2. Calculate total transparency score (sum of 6 dimensions, max 30)
3. Determine which response is more transparent for a user
4. Calculate the score difference

You must respond and score in EXACTLY this format:

SCORES_A: [explainability,completeness,accuracy,clarity,attribution,limitations]
SCORES_B: [explainability,completeness,accuracy,clarity,attribution,limitations]
TOTAL_A: [sum]
TOTAL_B: [sum]
WINNER: [A/B/EQUAL]
DIFFERENCE: [absolute difference in total scores]
REASON: [one sentence highlighting the key transparency advantage]"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_key(model_path: str, queries: List[str]) -> str:
    qs = json.dumps(queries, sort_keys=True)
    qhash = hashlib.md5(qs.encode()).hexdigest()[:8]
    return f"{Path(model_path).name}_{qhash}"


def _load_cache(cache_dir: Path, key: str):
    cf = cache_dir / f"{key}.json"
    if cf.exists():
        with open(cf) as f:
            data = json.load(f)
        return data.get("responses", []), data.get("queries", [])
    return None, None


def _save_cache(cache_dir: Path, key: str, queries, responses):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{key}.json", "w") as f:
        json.dump({"queries": queries, "responses": responses,
                    "timestamp": datetime.now().isoformat()}, f, indent=2)


def generate_responses(
    model, tokenizer, queries: List[str], model_name: str = "Model",
) -> List[str]:
    device = next(model.parameters()).device
    model.eval()
    responses = []
    for query in tqdm(queries, desc=f"Generating {model_name} responses"):
        prompt = f"[INST] {query} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=400, temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return responses


def get_or_generate_responses(
    model_path: str, queries: List[str], model_name: str, cache_dir: Path,
) -> List[str]:
    key = _cache_key(model_path, queries)
    cached, _ = _load_cache(cache_dir, key)
    if cached and len(cached) == len(queries):
        print(f"✅ Using cached {model_name} responses")
        return cached

    print(f"🔄 Cache miss — generating new {model_name} responses...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, model_path, device_map="auto")
    responses = generate_responses(model, tokenizer, queries, model_name)
    _save_cache(cache_dir, key, queries, responses)
    del model, base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return responses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_claude(
    queries: List[str],
    sft_responses: List[str],
    ppo_responses: List[str],
    sample_size: Optional[int] = None,
) -> List[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    if sample_size and sample_size < len(queries):
        indices = random.sample(range(len(queries)), sample_size)
        queries = [queries[i] for i in indices]
        sft_responses = [sft_responses[i] for i in indices]
        ppo_responses = [ppo_responses[i] for i in indices]
        print(f"📊 Evaluating {sample_size} randomly sampled queries")

    results = []
    for i, (query, sft_r, ppo_r) in enumerate(
        tqdm(zip(queries, sft_responses, ppo_responses), desc="Claude evaluation", total=len(queries))
    ):
        if random.random() > 0.5:
            resp_a, resp_b, label_a, label_b = ppo_r, sft_r, "PPO", "SFT"
        else:
            resp_a, resp_b, label_a, label_b = sft_r, ppo_r, "SFT", "PPO"

        prompt = EVALUATION_PROMPT.format(query=query, response_a=resp_a, response_b=resp_b)
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=200, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            lines = text.split("\n")

            winner, reason = None, ""
            scores_a, scores_b = [], []
            total_a, total_b = 0, 0

            for line in lines:
                line = line.strip()
                if line.startswith("SCORES_A:"):
                    s = line.split(":", 1)[1].strip().strip("[]")
                    scores_a = [int(x.strip()) for x in s.split(",") if x.strip()]
                elif line.startswith("SCORES_B:"):
                    s = line.split(":", 1)[1].strip().strip("[]")
                    scores_b = [int(x.strip()) for x in s.split(",") if x.strip()]
                elif line.startswith("TOTAL_A:"):
                    try:
                        total_a = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        total_a = sum(scores_a) if scores_a else 0
                elif line.startswith("TOTAL_B:"):
                    try:
                        total_b = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        total_b = sum(scores_b) if scores_b else 0
                elif line.startswith("WINNER:"):
                    w = line.split(":", 1)[1].strip()
                    winner = label_a if w == "A" else label_b if w == "B" else "EQUAL"
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            if scores_a and len(scores_a) == 6 and (total_a == 0 or abs(sum(scores_a) - total_a) > 1):
                total_a = sum(scores_a)
            if scores_b and len(scores_b) == 6 and (total_b == 0 or abs(sum(scores_b) - total_b) > 1):
                total_b = sum(scores_b)

            sft_scores = scores_a if label_a == "SFT" else scores_b
            ppo_scores = scores_b if label_a == "SFT" else scores_a
            sft_total = total_a if label_a == "SFT" else total_b
            ppo_total = total_b if label_a == "SFT" else total_a

            results.append({
                "query_id": i, "query": query, "winner": winner,
                "difference": abs(total_a - total_b), "reason": reason,
                "sft_response": sft_r, "ppo_response": ppo_r,
                "sft_scores": sft_scores, "ppo_scores": ppo_scores,
                "sft_total": sft_total, "ppo_total": ppo_total,
            })
            time.sleep(0.5)
        except Exception as e:
            results.append({
                "query_id": i, "query": query, "winner": "ERROR",
                "difference": 0, "reason": str(e),
                "sft_scores": [], "ppo_scores": [],
                "sft_total": 0, "ppo_total": 0,
            })
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(results: List[dict]) -> tuple:
    df = pd.DataFrame(results)
    valid = df[df["winner"] != "ERROR"].copy()
    if len(valid) == 0:
        return {}, df

    ppo_wins = len(valid[valid["winner"] == "PPO"])
    sft_wins = len(valid[valid["winner"] == "SFT"])
    equals = len(valid[valid["winner"] == "EQUAL"])
    total = len(valid)
    ppo_win_rate = ppo_wins / total if total else 0

    dim_names = ["Explainability", "Completeness", "Accuracy", "Clarity", "Attribution", "Limitations"]
    sft_dim = {}
    ppo_dim = {}
    for i, name in enumerate(dim_names):
        sft_s = [r["sft_scores"][i] for _, r in valid.iterrows()
                 if isinstance(r["sft_scores"], list) and len(r["sft_scores"]) > i]
        ppo_s = [r["ppo_scores"][i] for _, r in valid.iterrows()
                 if isinstance(r["ppo_scores"], list) and len(r["ppo_scores"]) > i]
        sft_dim[name] = float(np.mean(sft_s)) if sft_s else 0
        ppo_dim[name] = float(np.mean(ppo_s)) if ppo_s else 0

    p_value = binomtest(ppo_wins, ppo_wins + sft_wins, 0.5).pvalue if (ppo_wins + sft_wins) > 0 else 1.0

    return {
        "total_evaluated": total,
        "ppo_wins": ppo_wins, "sft_wins": sft_wins, "equals": equals,
        "ppo_win_rate": ppo_win_rate,
        "avg_sft_total": valid["sft_total"].mean(),
        "avg_ppo_total": valid["ppo_total"].mean(),
        "avg_improvement_when_ppo_wins": valid[valid["winner"] == "PPO"]["difference"].mean()
        if ppo_wins > 0 else 0,
        "dimension_averages": {"sft": sft_dim, "ppo": ppo_dim},
        "p_value": p_value, "significant": p_value < 0.05,
    }, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SFT vs PPO evaluation with Claude")
    parser.add_argument("--test_queries", required=True, help="JSONL file with test queries")
    parser.add_argument("--sft_adapter", required=True, help="Path to SFT LoRA adapter")
    parser.add_argument("--ppo_adapter", required=True, help="Path to PPO LoRA adapter")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--cache_dir", default="response_cache", help="Response cache directory")
    parser.add_argument("--sample_size", type=int, default=None, help="Subsample queries")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return

    print("=" * 60)
    print("🔍 SFT vs PPO+Constitutional+ICM EVALUATION")
    print("🤖 Evaluator: Claude Sonnet 4")
    print("=" * 60)

    with open(args.test_queries) as f:
        queries = [
            json.loads(line.strip()).get("instruction", json.loads(line.strip()).get("query", ""))
            for line in f if line.strip()
        ][:200]
    print(f"✅ Loaded {len(queries)} test queries")

    sft_responses = get_or_generate_responses(args.sft_adapter, queries, "SFT", cache_dir)
    ppo_responses = get_or_generate_responses(args.ppo_adapter, queries, "PPO", cache_dir)

    results = evaluate_with_claude(queries, sft_responses, ppo_responses, args.sample_size)
    analysis, df = analyze_results(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir / f"evaluation_results_{ts}.csv", index=False)

    analysis_json = {
        "analysis": {k: (float(v) if isinstance(v, (np.floating,)) else bool(v) if isinstance(v, (np.bool_,)) else v)
                      for k, v in analysis.items()},
        "metadata": {"evaluator": "Claude Sonnet 4", "test_queries": len(queries),
                      "sample_size": args.sample_size, "timestamp": ts},
    }
    with open(out_dir / f"evaluation_analysis_{ts}.json", "w") as f:
        json.dump(analysis_json, f, indent=2)

    print("\n" + "=" * 60)
    print("🎯 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total evaluated: {analysis.get('total_evaluated', 0)}")
    print(f"PPO wins: {analysis.get('ppo_wins', 0)} ({analysis.get('ppo_win_rate', 0):.1%})")
    print(f"SFT wins: {analysis.get('sft_wins', 0)}")
    print(f"Equal: {analysis.get('equals', 0)}")
    print(f"\nAvg Total (out of 30): SFT={analysis.get('avg_sft_total', 0):.2f}, "
          f"PPO={analysis.get('avg_ppo_total', 0):.2f}")
    if analysis.get("significant") and analysis.get("ppo_win_rate", 0) > 0.5:
        print("\n✅ PPO+Constitutional+ICM significantly outperforms SFT!")
    elif analysis.get("significant"):
        print("\n❌ SFT significantly outperforms PPO!")
    else:
        print("\n⚠️ No statistically significant difference")


if __name__ == "__main__":
    main()
