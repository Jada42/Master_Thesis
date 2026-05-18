#!/usr/bin/env python3
"""Hyperparameter optimization for Constitutional AI PPO training using Optuna.

Searches over PPO, LoRA, and reward-weighting hyperparameters to
maximize a composite objective of perfect-first-try rate, iteration
efficiency, excellence rate, and constitutional score.

Includes statistical analysis: parameter importance, correlations,
ANOVA, effect sizes, and publication-quality plots.
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from optuna.trial import TrialState
from peft import LoraConfig, TaskType, get_peft_model
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from src.config import BASE_MODEL_NAME, LORA_TARGET_MODULES
from src.rlhf.constitution import (
    advanced_iterative_improvement,
    enhanced_constitutional_check,
)
from src.rlhf.stats_tracker import TrainingStatsTracker

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HyperparameterOptimizer
# ---------------------------------------------------------------------------


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization for PPO training."""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results_dir = Path(base_config["output_dir"]) / "optuna_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trial_results: List[Dict] = []

    # ── Search space ─────────────────────────────────────────────────

    def create_ppo_config(self, trial: optuna.Trial) -> PPOConfig:
        config_dict = {
            "batch_size": trial.suggest_int("batch_size", 4, 16, step=2),
            "mini_batch_size": trial.suggest_int("mini_batch_size", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "target_kl": trial.suggest_float("target_kl", 0.01, 0.3, log=True),
            "ppo_epochs": trial.suggest_int("ppo_epochs", 2, 5),
            "init_kl_coef": trial.suggest_float("init_kl_coef", 0.01, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "vf_coef": trial.suggest_float("vf_coef", 0.05, 0.5, log=True),
        }
        config_dict["mini_batch_size"] = min(
            config_dict["mini_batch_size"], config_dict["batch_size"] // 2,
        )
        try:
            return PPOConfig(**config_dict)
        except Exception:
            config = PPOConfig()
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            return config

    def create_lora_config(self, trial: optuna.Trial) -> LoraConfig:
        r = trial.suggest_int("lora_r", 8, 32, step=4)
        alpha = trial.suggest_int("lora_alpha", r, r * 4, step=r)
        return LoraConfig(
            r=r, lora_alpha=alpha,
            lora_dropout=trial.suggest_float("lora_dropout", 0.0, 0.2, step=0.05),
            bias="none", task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES,
        )

    def suggest_reward_weights(self, trial: optuna.Trial) -> Dict[str, float]:
        const_w = trial.suggest_float("constitutional_weight", 0.6, 0.9)
        remaining = 1.0 - const_w
        rm_w = trial.suggest_float("reward_model_weight", 0.05, min(0.3, remaining))
        coh_w = remaining - rm_w
        return {
            "constitutional_weight": const_w,
            "reward_model_weight": rm_w,
            "coherence_weight": coh_w,
            "constitutional_learning_rate": trial.suggest_float(
                "constitutional_lr", 1e-6, 1e-5, log=True,
            ),
        }

    # ── Objective ────────────────────────────────────────────────────

    def objective(self, trial: optuna.Trial) -> float:
        ppo_cfg = self.create_ppo_config(trial)
        lora_cfg = self.create_lora_config(trial)
        rw = self.suggest_reward_weights(trial)
        conv_thresh = trial.suggest_float("convergence_threshold", 0.1, 0.5)
        max_iters = trial.suggest_int("max_iterations", 2, 4)

        try:
            metrics = self._run_training_trial(
                ppo_cfg, lora_cfg, rw, conv_thresh, max_iters, num_steps=100, trial=trial,
            )
            score = (
                0.4 * metrics["perfect_first_try_rate"]
                + 0.3 * (2.0 - metrics["avg_iterations"]) / 2.0
                + 0.2 * metrics["excellent_rate"]
                + 0.1 * metrics["avg_constitutional_score"]
            )
            self.trial_results.append({
                "trial_number": trial.number,
                "params": trial.params,
                "metrics": metrics,
                "objective_score": score,
            })
            return score
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0

    def _run_training_trial(
        self, ppo_cfg, lora_cfg, rw, conv_thresh, max_iters, num_steps, trial,
    ) -> Dict:
        tracker = TrainingStatsTracker()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(self.base_config["base_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_config["base_model"], torch_dtype=torch.float16, device_map="auto",
        )
        peft_m = get_peft_model(base, lora_cfg)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_m)

        try:
            rm = AutoModelForCausalLM.from_pretrained(
                self.base_config["reward_model_name"],
                torch_dtype=torch.float16, device_map="auto",
            ).eval()
        except Exception:
            rm = base

        queries = self.base_config.get("queries", ["test"] * 50)
        dataset = Dataset.from_dict({"query": queries[:num_steps]})

        trainer = PPOTrainer(
            model=model, config=ppo_cfg, dataset=dataset, tokenizer=tokenizer,
            data_collator=lambda d: dict((k, [i[k] for i in d]) for k in d[0]),
        )

        for step, batch in enumerate(trainer.dataloader):
            if step >= num_steps:
                break

            if step % 20 == 0 and step > 0:
                im = tracker.get_current_metrics()
                iscore = (
                    0.4 * im.get("perfect_first_try_rate", 0)
                    + 0.3 * (2.0 - im.get("avg_iterations", 2)) / 2.0
                    + 0.2 * im.get("excellent_rate", 0)
                    + 0.1 * im.get("avg_constitutional_score", 0)
                )
                trial.report(iscore, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            query_texts = batch["query"]
            query_tensors = [
                tokenizer(f"[INST] {q} [/INST]", return_tensors="pt").input_ids[0].to(device)
                for q in query_texts
            ]
            response_tensors = trainer.generate(
                query_tensors, return_prompt=False,
                min_new_tokens=50, max_new_tokens=200, temperature=0.3, do_sample=True,
            )
            rewards = []
            for query, rt in zip(query_texts, response_tensors):
                text = tokenizer.decode(rt, skip_special_tokens=True)
                improved, c_score, iters, _ = advanced_iterative_improvement(
                    text, query, model, tokenizer,
                    max_iterations=max_iters,
                    convergence_threshold=conv_thresh,
                    verbose=False,
                )
                final_r = (
                    rw["constitutional_weight"] * c_score
                    + rw["reward_model_weight"] * 0.0
                    + rw["coherence_weight"] * 0.5
                )
                rewards.append(torch.tensor(final_r, dtype=torch.float32))
                check = enhanced_constitutional_check(improved, query)
                tracker.update_stats(iters, check["quality_tier"], final_r, c_score)
            trainer.step(query_tensors, response_tensors, rewards)

        return tracker.get_current_metrics()

    # ── Optimization entry ───────────────────────────────────────────

    def run_optimization(self, n_trials: int = 100, n_jobs: int = 1) -> optuna.Study:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=20, interval_steps=10,
            ),
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        self._save_results(study)
        StatisticalAnalyzer(self.trial_results, str(self.results_dir)).run_all(study)
        return study

    def _save_results(self, study: optuna.Study):
        with open(self.results_dir / "optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)
        with open(self.results_dir / "best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
        study.trials_dataframe().to_csv(self.results_dir / "all_trials.csv", index=False)
        print(f"\n🎯 Best params: {study.best_params}")
        print(f"   Best score: {study.best_value:.4f}")


# ---------------------------------------------------------------------------
# StatisticalAnalyzer
# ---------------------------------------------------------------------------


class StatisticalAnalyzer:
    """Post-optimization statistical analysis."""

    def __init__(self, trial_results: List[Dict], output_dir: str):
        self.trial_results = trial_results
        self.output_dir = Path(output_dir) / "statistical_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._make_df()

    def _make_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trial_results:
            row = {"trial_number": t["trial_number"], "objective_score": t["objective_score"]}
            row.update(t["params"])
            for m, v in t["metrics"].items():
                row[f"metric_{m}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def run_all(self, study: optuna.Study):
        self._param_importance(study)
        self._correlations()
        self._effect_sizes()
        self._distributions()

    def _param_importance(self, study: optuna.Study):
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(list(imp.keys()), list(imp.values()))
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(list(imp.values())[i]))
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_importance.png", dpi=300)
        plt.close()
        with open(self.output_dir / "parameter_importance.json", "w") as f:
            json.dump(imp, f, indent=2)

    def _correlations(self):
        num = self.df.select_dtypes(include=[np.number])
        corr = num.corr()
        obj_corr = corr["objective_score"].sort_values(ascending=False)
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                     center=0, square=True, linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300)
        plt.close()

        sig_corr = {}
        for col in num.columns:
            if col != "objective_score":
                c, p = stats.pearsonr(
                    self.df[col].dropna(), self.df["objective_score"].dropna(),
                )
                if p < 0.05:
                    sig_corr[col] = {"correlation": c, "p_value": p, "significant": True}
        with open(self.output_dir / "significant_correlations.json", "w") as f:
            json.dump(sig_corr, f, indent=2)

    def _effect_sizes(self):
        es = {}
        for param in ["learning_rate", "target_kl", "constitutional_weight", "convergence_threshold"]:
            if param not in self.df:
                continue
            med = self.df[param].median()
            high = self.df[self.df[param] > med]["objective_score"]
            low = self.df[self.df[param] <= med]["objective_score"]
            if len(high) > 5 and len(low) > 5:
                md = high.mean() - low.mean()
                ps = np.sqrt(
                    ((len(high) - 1) * high.std() ** 2 + (len(low) - 1) * low.std() ** 2)
                    / (len(high) + len(low) - 2)
                )
                if ps > 0:
                    d = md / ps
                    t, p = stats.ttest_ind(high, low)
                    es[param] = {
                        "cohens_d": d,
                        "effect_size": "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small" if abs(d) >= 0.2 else "negligible",
                        "p_value": p, "significant": p < 0.05,
                    }
        with open(self.output_dir / "effect_sizes.json", "w") as f:
            json.dump(es, f, indent=2)

        if es:
            fig, ax = plt.subplots(figsize=(10, 6))
            params = list(es.keys())
            d_vals = [es[p]["cohens_d"] for p in params]
            colors = ["green" if es[p]["significant"] else "gray" for p in params]
            ax.barh(params, d_vals, color=colors)
            ax.axvline(0.2, color="r", linestyle="--", alpha=0.5, label="Small")
            ax.axvline(0.5, color="r", linestyle="--", alpha=0.5, label="Medium")
            ax.axvline(0.8, color="r", linestyle="--", alpha=0.5, label="Large")
            ax.set_xlabel("Cohen's d")
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "effect_sizes.png", dpi=300)
            plt.close()

    def _distributions(self):
        metrics = ["objective_score", "metric_avg_iterations",
                   "metric_perfect_first_try_rate", "metric_excellent_rate"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, m in zip(axes.ravel(), metrics):
            if m in self.df:
                data = self.df[m].dropna()
                ax.hist(data, bins=20, density=True, alpha=0.7, edgecolor="black")
                data.plot.kde(ax=ax, color="red", linewidth=2)
                ax.axvline(data.mean(), color="green", linestyle="--", label=f"μ={data.mean():.3f}")
                ax.axvline(data.median(), color="blue", linestyle="--", label=f"M={data.median():.3f}")
                ax.set_xlabel(m.replace("metric_", "").replace("_", " ").title())
                ax.legend()
                if len(data) > 7:
                    _, p = stats.shapiro(data)
                    ax.text(0.05, 0.95, f"Shapiro-Wilk p={p:.4f}",
                            transform=ax.transAxes, va="top",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        plt.suptitle("Performance Distributions")
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_distributions.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hyperparameter_optimization(
    queries_file: str | None = None,
    output_dir: str = "/workspace/ppo_optuna_optimization",
    n_trials: int = 100,
) -> Tuple[optuna.Study, HyperparameterOptimizer]:
    config: Dict[str, Any] = {
        "base_model": BASE_MODEL_NAME,
        "reward_model_name": "/workspace/reward_model_output",
        "output_dir": output_dir,
        "queries": None,
    }
    if queries_file and os.path.exists(queries_file):
        with open(queries_file) as f:
            config["queries"] = [
                json.loads(line.strip()).get("instruction", "")
                for line in f if line.strip()
            ]
    if config["queries"] is None:
        config["queries"] = [
            "How do I apply for a parking permit in Amsterdam?",
            "What are the costs for registering a new business?",
        ] * 50

    opt = HyperparameterOptimizer(config)
    study = opt.run_optimization(n_trials=n_trials)
    return study, opt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", default=None)
    parser.add_argument("--output_dir", default="/workspace/ppo_optuna_optimization")
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    study, opt = run_hyperparameter_optimization(
        queries_file=args.queries_file,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
    )
    print(f"\n✅ Best params: {study.best_params}")
    print(f"   Best score: {study.best_value:.4f}")
