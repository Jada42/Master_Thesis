#!/usr/bin/env python3
"""Ablation study for Constitutional AI PPO training.

Runs 6 configurations to isolate the contribution of each component:
1. Baseline — PPO without constitutional, reward model, or coherence
2. Constitutional only — rule-based transparency checks only
3. Reward model only — learned transparency scorer only
4. Constitutional + reward model — both scoring components
5. Constitutional + coherence — checks + semantic coherence
6. Full system — all three components

Requires ≥48 GB GPU VRAM (tested on RunPod A6000).
"""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from src.config import BASE_MODEL_NAME, LORA_TARGET_MODULES
from src.rlhf.coherence import IntrinsicCoherenceModule
from src.rlhf.constitution import (
    TransparencyConstitution,
    comprehensive_constitutional_check,
    advanced_iterative_improvement,
)

# ---------------------------------------------------------------------------
# Paths (override via environment or CLI)
# ---------------------------------------------------------------------------

BASE_PATH = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
QUERY_FILE = BASE_PATH / "rlhf_queries_first_50.jsonl"
REWARD_MODEL_PATH = BASE_PATH / "reward_model_output"
OUTPUT_BASE = BASE_PATH / "ablation_results_FINAL"

PPO_STEPS = 25


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "use_constitutional": False,
        "use_reward_model": False,
        "use_coherence": False,
        "description": "Baseline PPO without any constitutional AI",
    },
    "constitutional_only": {
        "use_constitutional": True,
        "use_reward_model": False,
        "use_coherence": False,
        "description": "Constitutional AI framework only",
    },
    "reward_model_only": {
        "use_constitutional": False,
        "use_reward_model": True,
        "use_coherence": False,
        "description": "Trained reward model only",
    },
    "constitutional_plus_rm": {
        "use_constitutional": True,
        "use_reward_model": True,
        "use_coherence": False,
        "description": "Constitutional AI + Reward Model",
    },
    "constitutional_plus_coherence": {
        "use_constitutional": True,
        "use_reward_model": False,
        "use_coherence": True,
        "description": "Constitutional AI + Coherence Module",
    },
    "full_system": {
        "use_constitutional": True,
        "use_reward_model": True,
        "use_coherence": True,
        "description": "Full system: Constitutional AI + Reward Model + Coherence",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_ppo_config() -> PPOConfig:
    return PPOConfig(
        batch_size=2,
        mini_batch_size=1,
        learning_rate=1e-6,
        target_kl=0.2,
        steps=PPO_STEPS,
        gradient_accumulation_steps=2,
        ppo_epochs=2,
        init_kl_coef=0.05,
    )


def compute_reward_model_score(
    query: str, response: str, reward_model, tokenizer,
) -> float:
    if reward_model is None:
        return 0.0
    try:
        prompt = f"Rate this response (0-5):\nQ: {query}\nA: {response}\nRating:"
        device = next(reward_model.parameters()).device
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True,
        ).to(device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=0)
        score_tokens = [
            tokenizer.encode(str(i), add_special_tokens=False)[0]
            for i in range(6)
        ]
        score_probs = [probs[st].item() for st in score_tokens]
        score = sum(i * p for i, p in enumerate(score_probs))
        return (score - 2.5) / 2.5
    except Exception as e:
        print(f"⚠️ Reward model scoring failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation_experiment(
    config_name: str,
    config: Dict[str, Any],
    queries: List[str],
    tokenizer,
    base_model_for_exp,
    reward_model_for_exp,
) -> Dict[str, Any] | None:
    print(f"\n{'='*60}\n🧪 {config_name.upper()}\n📝 {config['description']}\n{'='*60}")
    clear_memory()

    experiment_dir = OUTPUT_BASE / config_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    model_copy = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    if len(tokenizer) != model_copy.config.vocab_size:
        model_copy.resize_token_embeddings(len(tokenizer))

    peft_model_instance = get_peft_model(model_copy, lora_cfg)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model_instance).to("cuda")

    icm = None
    embedding_model = None
    if config["use_coherence"]:
        try:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
            icm = IntrinsicCoherenceModule(
                embedding_model.get_sentence_embedding_dimension(),
            ).to("cuda")
            icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-4)
        except Exception as e:
            print(f"⚠️ Coherence module failed to load: {e}")
            config["use_coherence"] = False

    subset_queries = queries[:20]
    dataset = Dataset.from_dict({"query": subset_queries})

    try:
        ppo_trainer = PPOTrainer(
            model=model, config=make_ppo_config(), dataset=dataset,
            tokenizer=tokenizer,
            data_collator=lambda d: dict((k, [i[k] for i in d]) for k in d[0]),
        )
    except Exception as e:
        print(f"❌ PPO Trainer creation failed: {e}")
        return None

    gen_kwargs = {
        "max_new_tokens": 200, "temperature": 0.4,
        "do_sample": True, "pad_token_id": tokenizer.eos_token_id,
    }
    stats: Dict[str, Any] = {
        "config_name": config_name, "reward_history": [],
        "constitutional_scores": [], "violations": 0, "iterations": [],
    }

    try:
        for batch_idx, batch in enumerate(
            tqdm(ppo_trainer.dataloader, desc=f"Training {config_name}")
        ):
            if batch_idx >= PPO_STEPS:
                break

            query_texts = batch["query"]
            query_tensors = [
                tokenizer(
                    f"[INST] {q} [/INST]", return_tensors="pt",
                ).input_ids[0].to("cuda")
                for q in query_texts
            ]

            try:
                response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
            except Exception as e:
                print(f"⚠️ Generation failed: {e}")
                continue

            rewards = []
            for query, resp_tensor in zip(query_texts, response_tensors):
                try:
                    response_text = tokenizer.decode(resp_tensor, skip_special_tokens=True)
                    if "[/INST]" in response_text:
                        response_text = response_text.split("[/INST]")[-1].strip()

                    c_score, rm_score, coh_score, iters = 0.0, 0.0, 0.0, 0

                    if config["use_constitutional"]:
                        response_text, c_score, iters, _ = advanced_iterative_improvement(
                            response_text, query, model, tokenizer,
                        )
                        stats["iterations"].append(iters)
                    elif comprehensive_constitutional_check(response_text, query)["needs_revision"]:
                        stats["violations"] += 1

                    if config["use_reward_model"]:
                        rm_score = compute_reward_model_score(
                            query, response_text, reward_model_for_exp, tokenizer,
                        )

                    if config["use_coherence"] and embedding_model is not None and icm is not None:
                        try:
                            q_emb = embedding_model.encode(query, convert_to_tensor=True)
                            r_emb = embedding_model.encode(response_text, convert_to_tensor=True)
                            coh_score = float(1 / (1 + torch.norm(q_emb - icm(r_emb))))
                        except Exception:
                            coh_score = 0.0

                    total_reward = 0.5 * c_score + 0.3 * rm_score + 0.2 * coh_score
                    if not any([
                        config["use_constitutional"],
                        config["use_reward_model"],
                        config["use_coherence"],
                    ]):
                        total_reward = 0.1

                    rewards.append(torch.tensor(total_reward, device="cuda"))
                    stats["constitutional_scores"].append(c_score)
                except Exception as e:
                    print(f"⚠️ Processing failed: {e}")
                    rewards.append(torch.tensor(0.0, device="cuda"))
                    stats["constitutional_scores"].append(0.0)

            if rewards:
                try:
                    ppo_trainer.step(query_tensors, response_tensors, rewards)
                    stats["reward_history"].append(
                        torch.tensor(rewards).mean().item(),
                    )
                except Exception as e:
                    print(f"⚠️ PPO step failed: {e}")

            if batch_idx % 1 == 0:
                clear_memory()

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()

    n_processed = len(subset_queries) * min(PPO_STEPS, len(ppo_trainer.dataloader))
    final_stats = {
        "config": config["description"],
        "avg_reward": float(np.mean(stats["reward_history"])) if stats["reward_history"] else 0.0,
        "avg_constitutional_score": float(np.mean(stats["constitutional_scores"]))
        if stats["constitutional_scores"] else 0.0,
        "violation_rate": (stats["violations"] / n_processed * 100)
        if not config["use_constitutional"] and stats["violations"] > 0 else 0,
        "avg_iterations": float(np.mean(stats["iterations"]))
        if config["use_constitutional"] and stats["iterations"] else 0,
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"📊 {config_name}: {final_stats}")

    del model, peft_model_instance, model_copy
    if embedding_model is not None:
        del embedding_model
    if icm is not None:
        del icm
    clear_memory()
    return final_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("🧪 COMPREHENSIVE CONSTITUTIONAL AI ABLATION STUDY")
    print("=" * 80)

    # Load queries
    if QUERY_FILE.exists():
        with open(QUERY_FILE) as f:
            queries = [
                json.loads(line.strip()).get("instruction", "")
                for line in f if line.strip()
            ]
        print(f"✅ Loaded {len(queries)} queries")
    else:
        queries = [
            "How do I apply for a parking permit in Amsterdam?",
            "What are the costs for registering a resident parking permit?",
            "Hey, I moved new to Amsterdam and have a Mercedes e-model, what can I do to park for free?",
            "What documents do I need for a DigiD?",
            "How do I register my scooter for a resident parking permit in Amsterdam?",
        ] * 4
        print(f"⚠️ Using {len(queries)} sample queries")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    needs_resize = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        needs_resize = True

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    if needs_resize:
        base_model.resize_token_embeddings(len(tokenizer))

    reward_model = None
    if REWARD_MODEL_PATH.exists():
        reward_model = AutoModelForCausalLM.from_pretrained(
            str(REWARD_MODEL_PATH), torch_dtype=torch.bfloat16, device_map={"": 0},
        )
        print("✅ Loaded reward model")
    else:
        print(f"⚠️ Reward model not found at {REWARD_MODEL_PATH}")

    if reward_model is None:
        for k in list(ABLATION_CONFIGS):
            if ABLATION_CONFIGS[k]["use_reward_model"]:
                print(f"   Skipping '{k}' — no reward model available")
                del ABLATION_CONFIGS[k]

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, cfg in ABLATION_CONFIGS.items():
        print(f"\n🔄 Starting experiment: {name}")
        try:
            result = run_ablation_experiment(
                name, cfg, queries, tokenizer, base_model, reward_model,
            )
            if result is not None:
                all_results[name] = result
        except Exception as e:
            print(f"\n❌ Experiment {name} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            clear_memory()

    if all_results:
        df = pd.DataFrame.from_dict(all_results, orient="index")
        print("\n--- FINAL RESULTS ---")
        print(df)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        df["avg_reward"].plot(kind="bar", ax=axes[0, 0], title="Average Reward")
        axes[0, 0].tick_params(axis="x", rotation=45)
        df["avg_constitutional_score"].plot(
            kind="bar", ax=axes[0, 1], title="Constitutional Score",
        )
        axes[0, 1].tick_params(axis="x", rotation=45)
        if "violation_rate" in df.columns:
            df["violation_rate"].plot(
                kind="bar", ax=axes[1, 0], title="Violation Rate (%)",
            )
            axes[1, 0].tick_params(axis="x", rotation=45)
        if "avg_iterations" in df.columns:
            df["avg_iterations"].plot(
                kind="bar", ax=axes[1, 1], title="Average Iterations",
            )
            axes[1, 1].tick_params(axis="x", rotation=45)
        plt.suptitle("Ablation Study Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            OUTPUT_BASE / "ablation_report.png", dpi=300, bbox_inches="tight",
        )
        with open(OUTPUT_BASE / "summary_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n🎯 ABLATION STUDY COMPLETE!")


if __name__ == "__main__":
    main()
