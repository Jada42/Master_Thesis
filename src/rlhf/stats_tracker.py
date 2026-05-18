"""Training statistics tracker for PPO constitutional AI training.

Tracks iteration counts, quality tiers, reward history, and
constitutional scores across training steps. Generates publication-quality
plots at the end of training.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


class TrainingStatsTracker:
    """Collect and report per-step training statistics.

    Tracks:
    - Iteration distribution (how many revision rounds needed)
    - Quality tier distribution (excellent / good / needs_improvement)
    - Reward and constitutional score histories
    - Supervision update count
    """

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "total_iterations": 0,
            "perfect_first_try": 0,
            "failed_to_converge": 0,
            "responses_processed": 0,
            "iteration_distribution": {0: 0, 1: 0, 2: 0, 3: 0},
            "supervised_updates": 0,
            "quality_distribution": {
                "excellent": 0, "good": 0, "needs_improvement": 0,
            },
            "reward_history": [],
            "constitutional_score_history": [],
            "convergence_rate_history": [],
        }

    def update_stats(
        self,
        iterations: int,
        quality_tier: str,
        total_reward: float,
        constitutional_score: float,
    ):
        s = self.stats
        s["total_iterations"] += iterations
        s["responses_processed"] += 1
        s["iteration_distribution"][min(iterations, 3)] += 1
        s["quality_distribution"][quality_tier] += 1
        s["reward_history"].append(total_reward)
        s["constitutional_score_history"].append(constitutional_score)
        if iterations == 0:
            s["perfect_first_try"] += 1
        elif iterations == 3:
            s["failed_to_converge"] += 1

    def get_current_metrics(self) -> dict:
        s = self.stats
        n = s["responses_processed"]
        if n == 0:
            return {}
        return {
            "avg_iterations": s["total_iterations"] / n,
            "perfect_first_try_rate": s["perfect_first_try"] / n,
            "avg_reward": float(np.mean(s["reward_history"][-50:]))
            if s["reward_history"] else 0,
            "avg_constitutional_score": float(
                np.mean(s["constitutional_score_history"][-50:])
            ) if s["constitutional_score_history"] else 0,
            "excellent_rate": s["quality_distribution"]["excellent"] / n,
        }

    def plot_training_progress(self, output_dir: str):
        if len(self.stats["reward_history"]) < 10:
            return

        s = self.stats
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Enhanced PPO Training Progress", fontsize=16)

        # Reward progression
        axes[0, 0].plot(s["reward_history"], alpha=0.7, label="Total Reward")
        axes[0, 0].plot(
            s["constitutional_score_history"], alpha=0.7,
            label="Constitutional Score",
        )
        axes[0, 0].set_title("Reward Progression")
        axes[0, 0].set_xlabel("Training Step")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Iteration distribution
        iter_counts = list(s["iteration_distribution"].values())
        axes[0, 1].bar(
            ["0", "1", "2", "3+"], iter_counts,
            color=["green", "yellow", "orange", "red"],
        )
        axes[0, 1].set_title("Iteration Distribution")
        axes[0, 1].set_xlabel("Iterations Required")
        axes[0, 1].set_ylabel("Count")

        # Quality distribution
        quality_counts = list(s["quality_distribution"].values())
        axes[1, 0].pie(
            quality_counts,
            labels=["Excellent", "Good", "Needs Improvement"],
            autopct="%1.1f%%", colors=["green", "yellow", "red"],
        )
        axes[1, 0].set_title("Quality Distribution")

        # Moving averages
        window_size = min(20, len(s["reward_history"]) // 4)
        if window_size > 1:
            reward_ma = np.convolve(
                s["reward_history"],
                np.ones(window_size) / window_size, mode="valid",
            )
            const_ma = np.convolve(
                s["constitutional_score_history"],
                np.ones(window_size) / window_size, mode="valid",
            )
            axes[1, 1].plot(reward_ma, label=f"Reward (MA-{window_size})")
            axes[1, 1].plot(const_ma, label=f"Constitutional (MA-{window_size})")
            axes[1, 1].set_title("Moving Averages")
            axes[1, 1].set_xlabel("Training Step")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "training_progress.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()
