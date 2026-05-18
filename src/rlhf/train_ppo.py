#!/usr/bin/env python3
"""Enhanced PPO training with Constitutional AI alignment.

Combines three reward components (optimized via Optuna):
- Constitutional framework (81% weight) — rule-based transparency checks
  with iterative self-improvement
- Reward model (11% weight) — learned transparency scorer
- Intrinsic Coherence Module (8% weight) — semantic query-response match

Requires CUDA GPU with ≥24 GB VRAM (tested on RunPod A6000/A100).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from src.config import (
    BASE_MODEL_NAME,
    LORA_TARGET_MODULES,
)
from src.rlhf.coherence import AdvancedICM
from src.rlhf.constitution import (
    advanced_iterative_improvement,
    compute_enhanced_constitutional_reward,
    enhanced_constitutional_check,
)
from src.rlhf.stats_tracker import TrainingStatsTracker

# ---------------------------------------------------------------------------
# Optuna-optimized defaults (trial 43, score 0.789)
# ---------------------------------------------------------------------------

DEFAULT_PPO_CONFIG = PPOConfig(
    batch_size=6,
    mini_batch_size=3,
    learning_rate=9.391494524751205e-06,
    target_kl=0.021602267145796005,
    steps=500,
    ppo_epochs=5,
    init_kl_coef=0.09327372581520726,
    gamma=0.9789438458465525,
    vf_coef=0.36436016003108607,
)

DEFAULT_LORA_CONFIG = LoraConfig(
    r=28,
    lora_alpha=84,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=LORA_TARGET_MODULES,
)

# Optimized reward weights from Optuna (Notes.txt trial 43)
CONSTITUTIONAL_WEIGHT = 0.81
REWARD_MODEL_WEIGHT = 0.111
COHERENCE_WEIGHT = 0.079
CONSTITUTIONAL_LR = 1.7169751284913331e-06

CONVERGENCE_THRESHOLD = 0.12701580628504341
MAX_ITERATIONS = 4


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_enhanced_constitutional_reward(
    query: str,
    response: str,
    reward_model,
    tokenizer,
    constitutional_weight: float = CONSTITUTIONAL_WEIGHT,
    reward_model_weight: float = REWARD_MODEL_WEIGHT,
) -> Tuple[float, Dict[str, Any]]:
    """Combine constitutional score, reward model score, and coherence.

    Returns:
        Tuple of (total_reward, components_dict).
    """
    device = next(reward_model.parameters()).device

    # --- Base reward model score ---
    prompt = (
        f"Rate this Amsterdam government response for transparency "
        f"and helpfulness (0-5):\nQuery: {query}\nResponse: {response}\nRating:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits[0, -1, :]
        score_tokens = []
        for i in range(6):
            try:
                score_tokens.append(tokenizer.encode(str(i), add_special_tokens=False)[0])
            except Exception:
                score_tokens.append(tokenizer.unk_token_id)
        if len(score_tokens) == 6:
            score_probs = torch.softmax(logits[score_tokens], dim=0)
            base_reward = sum(i * score_probs[i].item() for i in range(6))
            base_reward = (base_reward - 2.5) / 2.5
        else:
            base_reward = 0.0

    # --- Constitutional analysis ---
    constitutional_analysis = enhanced_constitutional_check(response, query)
    constitutional_score = constitutional_analysis["net_adjustment"]

    total_reward = (
        constitutional_weight * constitutional_score
        + reward_model_weight * base_reward
    )
    return total_reward, {
        "base_reward": base_reward,
        "constitutional_score": constitutional_score,
        "total_reward": total_reward,
        "quality_tier": constitutional_analysis["quality_tier"],
        "issues_found": constitutional_analysis["issues"],
        "scores_by_dimension": constitutional_analysis["scores_by_dimension"],
    }


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_ppo(
    query_file: str | Path,
    output_dir: str | Path,
    base_model: str = BASE_MODEL_NAME,
    reward_model_path: str | None = None,
    ppo_config: PPOConfig | None = None,
    lora_config: LoraConfig | None = None,
    num_steps: int | None = None,
    constitutional_weight: float = CONSTITUTIONAL_WEIGHT,
    reward_model_weight: float = REWARD_MODEL_WEIGHT,
    coherence_weight: float = COHERENCE_WEIGHT,
    constitutional_lr: float = CONSTITUTIONAL_LR,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS,
) -> None:
    """Train a Mistral-7B model with Constitutional-AI-guided PPO.

    Args:
        query_file: JSONL file with queries (``instruction`` field per line).
        output_dir: Where to save LoRA adapters and training artifacts.
        base_model: HuggingFace model ID.
        reward_model_path: Path to trained reward model (uses base model if None).
        ppo_config: PPO hyperparameters (uses Optuna-optimized defaults).
        lora_config: LoRA hyperparameters (uses Optuna-optimized defaults).
        num_steps: Override PPO steps (default: from ppo_config).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_config = ppo_config or DEFAULT_PPO_CONFIG
    lora_config = lora_config or DEFAULT_LORA_CONFIG
    if num_steps is not None:
        ppo_config.steps = num_steps

    stats_tracker = TrainingStatsTracker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load queries ──────────────────────────────────────────────────
    query_file = Path(query_file)
    if query_file.exists():
        with open(query_file) as f:
            queries = [
                json.loads(line.strip()).get("instruction", "")
                for line in f if line.strip()
            ]
        print(f"✅ Loaded {len(queries)} queries")
    else:
        queries = [
            "How do I apply for a parking permit in Amsterdam?",
            "What are the costs for registering a new business?",
            "How long does it take to get a building permit?",
            "What documents do I need for a residence parking permit?",
            "I need some information about resident parking permits and blue zones.",
            "What are the requirements for applying for a DigiD?",
            "How can I get a copy of my birth certificate?",
            "What are the rules for waste disposal in Amsterdam?",
        ] * 7  # ~56 queries
        print(f"⚠️ Query file not found, using {len(queries)} sample queries")

    # ── Tokenizer & models ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_instance = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto",
    )
    peft_model = get_peft_model(base_model_instance, lora_config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

    # Reward model
    if reward_model_path and os.path.exists(reward_model_path):
        reward_model = AutoModelForCausalLM.from_pretrained(
            reward_model_path, torch_dtype=torch.float16, device_map="auto",
        ).eval()
        print("✅ Loaded reward model")
    else:
        print("⚠️ No reward model path — using base model as fallback")
        reward_model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto",
        ).eval()

    # ── Coherence module ──────────────────────────────────────────────
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    icm = AdvancedICM(embedding_dim).to(device)
    icm_optimizer = torch.optim.AdamW(icm.parameters(), lr=1e-4, weight_decay=1e-5)

    # ── Constitutional optimizer ──────────────────────────────────────
    constitutional_optimizer = torch.optim.AdamW(
        model.pretrained_model.parameters(),
        lr=constitutional_lr,
        weight_decay=1e-6,
    )

    # ── PPO trainer ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({"query": queries})
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: dict(
            (key, [d[key] for d in data]) for key in data[0]
        ),
    )

    generation_kwargs = {
        "max_new_tokens": 400,
        "temperature": 0.3,
        "do_sample": True,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # ── Logging ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iteration_log_file = output_dir / f"enhanced_iteration_log_{timestamp}.jsonl"
    constitutional_examples: List[Tuple[str, str]] = []

    # ── Training loop ─────────────────────────────────────────────────
    print("\n🔄 Starting enhanced iterative training loop...")

    for step, batch in enumerate(
        tqdm(ppo_trainer.dataloader, desc="Enhanced PPO Training")
    ):
        if step >= ppo_config.steps:
            break

        query_texts = batch["query"]
        query_tensors = [
            tokenizer(
                f"[INST] {q} [/INST]", return_tensors="pt", truncation=True,
            ).input_ids[0].to(device)
            for q in query_texts
        ]

        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs,
        )

        rewards = []
        batch_coherence_rewards = []

        for i, (query, response_tensor) in enumerate(
            zip(query_texts, response_tensors)
        ):
            response_text = tokenizer.decode(
                response_tensor, skip_special_tokens=True,
            )

            # Coherence reward
            with torch.no_grad():
                q_emb = embedding_model.encode(query, convert_to_tensor=True, device=device)
                r_emb = embedding_model.encode(response_text, convert_to_tensor=True, device=device)
            icm_optimizer.zero_grad()
            coherence_reward, icm_loss = icm.compute_reward_and_loss(q_emb, r_emb)
            icm_loss.backward()
            icm_optimizer.step()
            batch_coherence_rewards.append(coherence_reward)

            # Iterative constitutional improvement
            verbose = (step == 0 and i == 0) or (step % 25 == 0 and i == 0)
            improved_text, constitutional_score, iterations, history = (
                advanced_iterative_improvement(
                    response_text, query, model, tokenizer,
                    max_iterations=max_iterations,
                    convergence_threshold=convergence_threshold,
                    verbose=verbose,
                )
            )

            if iterations <= 1 and constitutional_score > 0.4:
                constitutional_examples.append((query, improved_text))

            final_reward, reward_components = compute_enhanced_constitutional_reward(
                query, improved_text, reward_model, tokenizer,
            )

            efficiency_bonus = max(0, (3 - iterations) * 0.15)
            total_reward = (
                final_reward + efficiency_bonus
                + (coherence_weight * coherence_reward)
            )
            rewards.append(torch.tensor(total_reward, dtype=torch.float32))

            stats_tracker.update_stats(
                iterations,
                reward_components["quality_tier"],
                total_reward,
                reward_components["constitutional_score"],
            )

            # Write log entry
            log_entry = {
                "step": step,
                "query": query,
                "original_response": response_text[:200],
                "final_response": improved_text[:200],
                "iterations": iterations,
                "history": history,
                "total_reward": total_reward,
                "constitutional_score": reward_components["constitutional_score"],
                "base_reward": reward_components["base_reward"],
                "coherence_reward": coherence_reward,
                "quality_tier": reward_components["quality_tier"],
                "issues_found": reward_components["issues_found"],
                "scores_by_dimension": reward_components["scores_by_dimension"],
            }
            with open(iteration_log_file, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")

        # PPO step
        ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Constitutional supervision (every step after first)
        if step > 0 and constitutional_examples:
            sample_size = min(len(constitutional_examples), 48)
            sampled = constitutional_examples[-sample_size:]
            model.pretrained_model.train()
            total_loss = 0.0
            for query, const_response in sampled:
                full = f"[INST] {query} [/INST] {const_response}"
                inputs = tokenizer(
                    full, return_tensors="pt", truncation=True, max_length=512,
                ).to(device)
                labels = inputs["input_ids"].clone()
                prompt_len = len(
                    tokenizer(
                        f"[INST] {query} [/INST]", return_tensors="pt",
                    )["input_ids"][0]
                )
                labels[:, :prompt_len] = -100
                constitutional_optimizer.zero_grad()
                outputs = model.pretrained_model(
                    input_ids=inputs["input_ids"], labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.pretrained_model.parameters(), max_norm=1.0,
                )
                constitutional_optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(sampled)
            stats_tracker.stats["supervised_updates"] += 1
            print(f"   ✅ Constitutional supervision — avg loss: {avg_loss:.4f}")
            model.pretrained_model.eval()
            constitutional_examples = constitutional_examples[-100:]

        # Progress
        if step % 10 == 0:
            metrics = stats_tracker.get_current_metrics()
            avg_coh = float(np.mean(batch_coherence_rewards)) if batch_coherence_rewards else 0
            print(f"\n📊 Step {step}: reward={torch.stack(rewards).mean().item():.3f}, "
                  f"const_scr={metrics.get('avg_constitutional_score', 0):.3f}, "
                  f"coh={avg_coh:.3f}, "
                  f"avg_it={metrics.get('avg_iterations', 0):.2f}, "
                  f"perfect={metrics.get('perfect_first_try_rate', 0)*100:.1f}%, "
                  f"excellent={metrics.get('excellent_rate', 0)*100:.1f}%")

    # ── Save model ────────────────────────────────────────────────────
    print(f"\n💾 Saving model to {output_dir}...")
    if hasattr(ppo_trainer.model, "pretrained_model"):
        save_target = ppo_trainer.model.pretrained_model
    else:
        save_target = ppo_trainer.model
    save_target.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save stats
    final_stats = stats_tracker.stats.copy()
    final_stats.update(stats_tracker.get_current_metrics())
    with open(output_dir / "enhanced_training_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2, default=str)
    stats_tracker.plot_training_progress(str(output_dir))

    # ── Validation ────────────────────────────────────────────────────
    print("\n🔍 Validation...")
    base_test = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto",
    )
    test_model = PeftModel.from_pretrained(base_test, str(output_dir))
    test_queries = [
        "What are the costs for a parking permit in Amsterdam?",
        "How do I register a new business?",
        "What documents do I need for a building permit?",
        "Help, I will move to Amsterdam, and I want a parking permit. Can I get one?",
    ]
    validation_results = []
    for test_query in test_queries:
        test_prompt = f"[INST] {test_query} [/INST]"
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            test_outputs = test_model.generate(
                **test_inputs, max_new_tokens=200, temperature=0.2,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        test_response = tokenizer.decode(
            test_outputs[0][len(test_inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        vc = enhanced_constitutional_check(test_response, test_query)
        validation_results.append({
            "query": test_query,
            "response": test_response[:150] + "...",
            "score": vc["net_adjustment"],
            "quality_tier": vc["quality_tier"],
            "issues": vc["issues"],
        })
        print(f"   {test_query[:50]}... → score={vc['net_adjustment']:.3f} ({vc['quality_tier']})")
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    avg_val = float(np.mean([r["score"] for r in validation_results]))
    print(f"\n📊 Validation avg score: {avg_val:.3f}")

    final_metrics = stats_tracker.get_current_metrics()
    print(f"\n🏆 Final: perfect_first_try={final_metrics.get('perfect_first_try_rate', 0)*100:.1f}%, "
          f"avg_iter={final_metrics.get('avg_iterations', 0):.2f}, "
          f"excellent={final_metrics.get('excellent_rate', 0)*100:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Constitutional AI PPO training")
    parser.add_argument("--query_file", required=True, help="JSONL query file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--reward_model_path", default=None)
    parser.add_argument("--base_model", default=BASE_MODEL_NAME)
    parser.add_argument("--num_steps", type=int, default=None)
    args = parser.parse_args()

    train_ppo(
        query_file=args.query_file,
        output_dir=args.output_dir,
        reward_model_path=args.reward_model_path,
        base_model=args.base_model,
        num_steps=args.num_steps,
    )
