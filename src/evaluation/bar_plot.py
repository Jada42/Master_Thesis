"""Publication-quality bar plot comparing SFT vs PPO across transparency dimensions.

Generates a bar chart with error bars (standard error) and significance stars
using paired t-tests with Bonferroni correction. Optionally produces a
histogram of per-query win-margin distributions.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

METHOD_LABELS: dict[str, str] = {
    "ttest_raw_p": "Paired t-test (raw p-values)",
    "ttest_bonf_p": "Paired t-test (Bonferroni-corrected)",
    "ttest_fdr_p": "Paired t-test (FDR-corrected)",
    "wilcoxon_raw_p": "Wilcoxon signed-rank (raw p-values)",
    "wilcoxon_bonf_p": "Wilcoxon signed-rank (Bonferroni-corrected)",
    "wilcoxon_fdr_p": "Wilcoxon signed-rank (FDR-corrected)",
}

TRANSPARENCY_DIMS = [
    "Explainability",
    "Completeness",
    "Accuracy",
    "Clarity",
    "Attribution",
    "Limitations",
]


def load_evaluation_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load an evaluation results CSV and parse score columns.

    Expects columns ``sft_scores`` and ``ppo_scores`` containing string
    representations of Python lists (e.g. ``"[3, 4, 2, ...]"``).

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with parsed ``sft_scores`` and ``ppo_scores`` columns.
    """
    df = pd.read_csv(csv_path)
    df["sft_scores"] = df["sft_scores"].apply(ast.literal_eval)
    df["ppo_scores"] = df["ppo_scores"].apply(ast.literal_eval)
    return df


def compute_statistics(
    df: pd.DataFrame,
    dims: list[str] = TRANSPARENCY_DIMS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute means, standard errors, and p-values per dimension.

    Args:
        df: DataFrame with ``sft_scores`` and ``ppo_scores``.
        dims: Ordered dimension names.

    Returns:
        ``(means, ses, pvals)`` DataFrames.
    """
    records = []
    for _, row in df.iterrows():
        for i, dim in enumerate(dims):
            records.append({"dim": dim, "model": "SFT", "score": row["sft_scores"][i]})
            records.append({"dim": dim, "model": "PPO", "score": row["ppo_scores"][i]})
    long = pd.DataFrame(records)

    stats = (
        long.groupby(["dim", "model"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    means = stats.pivot(index="dim", columns="model", values="mean").loc[dims]
    ses = stats.pivot(index="dim", columns="model", values="se").loc[dims]

    raw_t = [
        ttest_rel(
            long[(long["dim"] == d) & (long["model"] == "PPO")]["score"],
            long[(long["dim"] == d) & (long["model"] == "SFT")]["score"],
        ).pvalue
        for d in dims
    ]
    raw_w = [
        wilcoxon(
            long[(long["dim"] == d) & (long["model"] == "PPO")]["score"],
            long[(long["dim"] == d) & (long["model"] == "SFT")]["score"],
        ).pvalue
        for d in dims
    ]

    _, bonf_t, _, _ = multipletests(raw_t, alpha=0.05, method="bonferroni")
    _, fdr_t, _, _ = multipletests(raw_t, alpha=0.05, method="fdr_bh")
    _, bonf_w, _, _ = multipletests(raw_w, alpha=0.05, method="bonferroni")
    _, fdr_w, _, _ = multipletests(raw_w, alpha=0.05, method="fdr_bh")

    pvals = pd.DataFrame(
        {
            "dimension": dims,
            "ttest_raw_p": raw_t,
            "ttest_bonf_p": bonf_t,
            "ttest_fdr_p": fdr_t,
            "wilcoxon_raw_p": raw_w,
            "wilcoxon_bonf_p": bonf_w,
            "wilcoxon_fdr_p": fdr_w,
        }
    ).set_index("dimension")
    return means, ses, pvals


def _significance_stars(pvals: pd.DataFrame, method: str) -> dict[str, str]:
    """Map corrected p-values to conventional significance stars."""
    stars = {}
    for dim, p in pvals[method].items():
        if p < 0.001:
            stars[dim] = "***"
        elif p < 0.01:
            stars[dim] = "**"
        elif p < 0.05:
            stars[dim] = "*"
        else:
            stars[dim] = ""
    return stars


def _set_plotting_style() -> None:
    """Apply publication-ready matplotlib rcParams."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Palatino", "Georgia"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "figure.figsize": (6, 4),
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.constrained_layout.use": True,
        }
    )


def plot_bar_comparison(
    means: pd.DataFrame,
    ses: pd.DataFrame,
    sigs: dict[str, str],
    dims: list[str],
    output_path: str | Path,
    pvalue_method: str,
) -> None:
    """Render and save the SFT vs PPO bar chart.

    Args:
        means: Mean scores per dimension and model.
        ses: Standard errors.
        sigs: Significance stars per dimension.
        dims: Ordered dimension names.
        output_path: Destination file (PDF recommended).
        pvalue_method: Key into ``METHOD_LABELS`` for the subtitle.
    """
    x = np.arange(len(dims))
    w = 0.35
    fig, ax = plt.subplots()

    ax.bar(x - w / 2, means["SFT"], w, yerr=ses["SFT"], capsize=4, label="SFT")
    ax.bar(x + w / 2, means["PPO"], w, yerr=ses["PPO"], capsize=4, label="PPO")

    for i, dim in enumerate(dims):
        star = sigs.get(dim, "")
        if star:
            y_top = max(
                means.loc[dim, "SFT"] + ses.loc[dim, "SFT"],
                means.loc[dim, "PPO"] + ses.loc[dim, "PPO"],
            )
            ax.text(x[i], y_top + 0.03, star, ha="center", va="bottom", fontsize=12)

    subtitle = METHOD_LABELS.get(pvalue_method, "")
    ax.set_title(f"Dimension-wise Comparison: SFT vs PPO\n{subtitle}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=20, ha="right")
    ax.set_ylabel("Mean Score")
    ax.legend(frameon=False)

    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.1)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved bar plot to {output_path}")


def plot_win_margin_histogram(
    df: pd.DataFrame,
    output_path: str | Path,
    color: str = "steelblue",
) -> None:
    """Histogram of absolute per-query score differences (win margins).

    Args:
        df: Evaluation DataFrame.
        output_path: Destination file.
        color: Bar fill colour.
    """
    margins = [
        abs(np.mean(p) - np.mean(s))
        for s, p in zip(df["sft_scores"], df["ppo_scores"])
        if np.mean(p) != np.mean(s)
    ]
    if not margins:
        print("No non-zero margins to plot.")
        return

    fig, ax = plt.subplots()
    ax.hist(np.array(margins), bins=10, edgecolor="black", color=color)
    ax.set_xlabel("Win Margin (Mean Score Difference)")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Distribution of Win Margins Across Queries")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved histogram to {output_path}")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SFT vs PPO bar plot with significance testing."
    )
    parser.add_argument("csv", help="Path to evaluation_results.csv")
    parser.add_argument("--out", default="bar_plot.pdf", help="Output bar plot file.")
    parser.add_argument(
        "--pvalue-method",
        default="ttest_bonf_p",
        choices=list(METHOD_LABELS),
    )
    parser.add_argument(
        "--histogram-out",
        default=None,
        help="Optional histogram output file.",
    )
    parser.add_argument(
        "--histogram-color",
        default="tab:blue",
        help="Colour for histogram bars.",
    )
    args = parser.parse_args()

    df = load_evaluation_csv(args.csv)
    means, ses, pvals = compute_statistics(df)

    print("\nP-values and corrections:\n")
    print(pvals.to_string(float_format="%.4f"))

    _set_plotting_style()
    sigs = _significance_stars(pvals, args.pvalue_method)
    plot_bar_comparison(means, ses, sigs, TRANSPARENCY_DIMS, args.out, args.pvalue_method)

    if args.histogram_out:
        plot_win_margin_histogram(df, args.histogram_out, args.histogram_color)


if __name__ == "__main__":
    main()
