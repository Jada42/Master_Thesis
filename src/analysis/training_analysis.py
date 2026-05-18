"""Statistical analysis of PPO training iteration logs.

Generates publication-ready statistics, plots, and LaTeX tables
from the enhanced_iteration_log_*.jsonl files produced by train_ppo.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class TrainingResultsAnalyzer:
    """Analyze training iteration logs from Constitutional AI PPO training."""

    def __init__(self, log_file_path: str, output_dir: str = "results_analysis"):
        self.log_file_path = log_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_logs()
        print(f"✅ Loaded {len(self.data)} training samples")

    def _load_logs(self) -> List[Dict]:
        data = []
        with open(self.log_file_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    # ── Basic statistics ──────────────────────────────────────────────

    def calculate_basic_statistics(self) -> Dict:
        iterations = [d["iterations"] for d in self.data]
        constitutional_scores = [d["constitutional_score"] for d in self.data]
        total_rewards = [d["total_reward"] for d in self.data]
        quality_tiers = [d["quality_tier"] for d in self.data]
        n = len(self.data)

        return {
            "total_samples": n,
            "iterations": {
                "mean": float(np.mean(iterations)),
                "std": float(np.std(iterations)),
                "median": float(np.median(iterations)),
                "perfect_first_try": sum(1 for i in iterations if i == 0),
                "perfect_first_try_rate": sum(1 for i in iterations if i == 0) / n,
                "distribution": {
                    "0": sum(1 for i in iterations if i == 0),
                    "1": sum(1 for i in iterations if i == 1),
                    "2": sum(1 for i in iterations if i == 2),
                    "3+": sum(1 for i in iterations if i >= 3),
                },
            },
            "constitutional_scores": {
                "mean": float(np.mean(constitutional_scores)),
                "std": float(np.std(constitutional_scores)),
                "positive_rate": sum(1 for s in constitutional_scores if s > 0) / n,
            },
            "quality_tiers": {
                "excellent": quality_tiers.count("excellent"),
                "excellent_rate": quality_tiers.count("excellent") / n,
                "good": quality_tiers.count("good"),
                "good_rate": quality_tiers.count("good") / n,
                "needs_improvement": quality_tiers.count("needs_improvement"),
            },
            "total_rewards": {
                "mean": float(np.mean(total_rewards)),
                "std": float(np.std(total_rewards)),
            },
        }

    def analyze_dimension_scores(self):
        dim_data = []
        for entry in self.data:
            if "scores_by_dimension" in entry:
                s = entry["scores_by_dimension"]
                dim_data.append({
                    "repetition": s.get("repetition", 0),
                    "sources": s.get("sources", 0),
                    "procedures": s.get("procedures", 0),
                    "completeness": s.get("completeness", 0),
                    "limitations": s.get("limitations", 0),
                    "quality_tier": entry["quality_tier"],
                    "iterations": entry["iterations"],
                })
        df = pd.DataFrame(dim_data)
        dims = ["repetition", "sources", "procedures", "completeness", "limitations"]
        dim_stats = {}
        for dim in dims:
            if dim in df:
                dim_stats[dim] = {
                    "mean_score": float(df[dim].mean()),
                    "positive_rate": float((df[dim] > 0).mean()),
                }
        return df, dim_stats

    # ── Statistical tests ─────────────────────────────────────────────

    def run_statistical_tests(self) -> Dict:
        iterations = [d["iterations"] for d in self.data]
        steps = list(range(len(iterations)))
        n = len(self.data)

        # Training progress correlation
        corr, p_value = stats.pearsonr(steps, iterations)

        # Early vs late
        third = n // 3
        early = iterations[:third]
        late = iterations[-(n // 3):]
        t_stat, t_p = stats.ttest_ind(early, late)
        md = np.mean(early) - np.mean(late)
        ps = np.sqrt((np.std(early) ** 2 + np.std(late) ** 2) / 2)
        d = md / ps if ps > 0 else 0

        # Quality chi-square
        tiers = [d["quality_tier"] for d in self.data]
        early_q = tiers[:third]
        late_q = tiers[-(n // 3):]
        ct = pd.crosstab(
            pd.Series(["early"] * len(early_q) + ["late"] * len(late_q)),
            pd.Series(early_q + late_q),
        )
        chi2, chi_p, _, _ = stats.chi2_contingency(ct)

        return {
            "training_progress": {"correlation": corr, "p_value": p_value,
                                  "significant_improvement": p_value < 0.05 and corr < 0},
            "early_vs_late": {"early_mean": float(np.mean(early)), "late_mean": float(np.mean(late)),
                              "improvement": float(md), "t_statistic": t_stat, "p_value": t_p,
                              "cohens_d": d, "effect_size": self._cohens_label(d)},
            "quality_improvement": {"chi2": chi2, "p_value": chi_p, "significant": chi_p < 0.05,
                                    "early_excellent_rate": early_q.count("excellent") / len(early_q),
                                    "late_excellent_rate": late_q.count("excellent") / len(late_q)},
        }

    @staticmethod
    def _cohens_label(d: float) -> str:
        d = abs(d)
        return "large" if d >= 0.8 else "medium" if d >= 0.5 else "small" if d >= 0.2 else "negligible"

    # ── Plots ─────────────────────────────────────────────────────────

    def create_publication_plots(self):
        plt.style.use("seaborn-v0_8-paper")
        sns.set_context("paper", font_scale=1.2)

        iterations = [d["iterations"] for d in self.data]

        # Main results figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(iterations, bins=[0, 1, 2, 3, 4], align="left", rwidth=0.8,
                 color="steelblue", edgecolor="black")
        ax1.set_xlabel("Number of Iterations")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Iterations Required")
        ax1.set_xticks([0, 1, 2, 3])
        n = len(iterations)
        for i in range(4):
            c = sum(1 for it in iterations if it == i)
            ax1.text(i, c + 5, f"{c/n*100:.1f}%", ha="center")

        tiers = [d["quality_tier"] for d in self.data]
        qc = pd.Series(tiers).value_counts()
        colors = {"excellent": "#2ecc71", "good": "#f39c12", "needs_improvement": "#e74c3c"}
        qc.plot(kind="pie", ax=ax2, autopct="%1.1f%%",
                colors=[colors[k] for k in qc.index])
        ax2.set_ylabel("")
        ax2.set_title("Response Quality Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "main_results.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(self.output_dir / "main_results.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Learning progress
        fig, ax = plt.subplots(figsize=(8, 5))
        window = 50
        ma = pd.Series(iterations).rolling(window, min_periods=1).mean()
        ax.plot(ma, label="Moving Average", linewidth=2, color="darkblue")
        ax.scatter(range(len(iterations)), iterations, alpha=0.3, s=10, color="lightblue")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Iterations Required")
        ax.set_title("Learning Progress Over Time")
        ax.axhline(y=1.2, color="red", linestyle="--", alpha=0.7, label="Target (1.2)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_progress.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(self.output_dir / "learning_progress.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Dimension heatmap
        df, _ = self.analyze_dimension_scores()
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            dims = ["repetition", "sources", "procedures", "completeness", "limitations"]
            hm = df.groupby("quality_tier")[dims].mean()
            order = ["excellent", "good", "needs_improvement"]
            hm = hm.reindex([o for o in order if o in hm.index])
            sns.heatmap(hm.T, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                        cbar_kws={"label": "Average Score"}, ax=ax)
            ax.set_title("Average Dimension Scores by Quality Tier")
            plt.tight_layout()
            plt.savefig(self.output_dir / "dimension_heatmap.pdf", dpi=300, bbox_inches="tight")
            plt.savefig(self.output_dir / "dimension_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()

    # ── LaTeX tables ──────────────────────────────────────────────────

    def generate_latex_tables(self):
        s = self.calculate_basic_statistics()
        t = self.run_statistical_tests()

        t1 = (
            r"\begin{table}[h]" "\n"
            r"\centering" "\n"
            r"\caption{Constitutional AI Framework Performance Metrics}" "\n"
            r"\label{tab:main_results}" "\n"
            r"\begin{tabular}{lc}" "\n"
            r"\toprule" "\n"
            r"\textbf{Metric} & \textbf{Value} \\" "\n"
            r"\midrule" "\n"
            f"Total Samples & {s['total_samples']} \\\\\n"
            f"Perfect First-Try Rate & {s['iterations']['perfect_first_try_rate']*100:.1f}\\% \\\\\n"
            f"Average Iterations & {s['iterations']['mean']:.2f} $\\pm$ {s['iterations']['std']:.2f} \\\\\n"
            f"Excellent Response Rate & {s['quality_tiers']['excellent_rate']*100:.1f}\\% \\\\\n"
            f"Good Response Rate & {s['quality_tiers']['good_rate']*100:.1f}\\% \\\\\n"
            f"Mean Constitutional Score & {s['constitutional_scores']['mean']:.3f} $\\pm$ {s['constitutional_scores']['std']:.3f} \\\\\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}"
        )
        t2 = (
            r"\begin{table}[h]" "\n"
            r"\centering" "\n"
            r"\caption{Statistical Significance Tests}" "\n"
            r"\label{tab:statistical_tests}" "\n"
            r"\begin{tabular}{lccc}" "\n"
            r"\toprule" "\n"
            r"\textbf{Test} & \textbf{Statistic} & \textbf{p-value} & \textbf{Effect Size} \\" "\n"
            r"\midrule" "\n"
            f"Training Progress (Pearson r) & {t['training_progress']['correlation']:.3f} & {t['training_progress']['p_value']:.4f} & - \\\\\n"
            f"Early vs Late (t-test) & {t['early_vs_late']['t_statistic']:.2f} & {t['early_vs_late']['p_value']:.4f} & d={t['early_vs_late']['cohens_d']:.2f} ({t['early_vs_late']['effect_size']}) \\\\\n"
            f"Quality Distribution ($\\chi^2$) & {t['quality_improvement']['chi2']:.2f} & {t['quality_improvement']['p_value']:.4f} & - \\\\\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}"
        )

        with open(self.output_dir / "latex_tables.tex", "w") as f:
            f.write(t1 + "\n\n" + t2)
        return t1, t2

    # ── Run all ───────────────────────────────────────────────────────

    def run_complete_analysis(self):
        print("📊 Running Complete Results Analysis\n" + "=" * 50)

        stats = self.calculate_basic_statistics()
        print(f"   Perfect First-Try: {stats['iterations']['perfect_first_try_rate']*100:.1f}%")
        print(f"   Avg Iterations: {stats['iterations']['mean']:.2f}")
        print(f"   Excellence Rate: {stats['quality_tiers']['excellent_rate']*100:.1f}%")

        tests = self.run_statistical_tests()
        print(f"\n   Training progress p={tests['training_progress']['p_value']:.4f}")
        print(f"   Early vs Late: d={tests['early_vs_late']['cohens_d']:.2f} ({tests['early_vs_late']['effect_size']})")

        self.create_publication_plots()
        self.generate_latex_tables()

        all_results = {"basic_statistics": stats, "statistical_tests": tests,
                        "analysis_date": datetime.now().isoformat()}
        with open(self.output_dir / "all_statistics.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n🎉 Complete! Results saved to {self.output_dir}")
        return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python training_analysis.py <iteration_log.jsonl>")
        sys.exit(1)
    analyzer = TrainingResultsAnalyzer(sys.argv[1])
    analyzer.run_complete_analysis()
