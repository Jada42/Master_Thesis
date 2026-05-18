"""Combine preference datasets from multiple LLM judges and compute agreement.

Loads JSONL preference files from different simulator models (e.g. GPT-4.1
and Sonnet-4), merges them into a single CSV, and reports cross-LLM
agreement (percent agreement and Cohen's kappa).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils.io_utils import load_jsonl, save_json

logger = logging.getLogger(__name__)


def _jsonl_to_dataframe(data: list[dict], source: str) -> pd.DataFrame:
    """Convert a list of preference dicts into a standardised DataFrame."""
    rows = []
    for item in data:
        rows.append(
            {
                "query": item.get("query", ""),
                "chosen_response": item.get("chosen_response", ""),
                "rejected_response": item.get("rejected_response", ""),
                "simulator_choice_label": item.get("simulator_choice_label", ""),
                "simulator_reasoning": item.get("simulator_justification", ""),
                "source_llm": source,
            }
        )
    return pd.DataFrame(rows)


def _compute_agreement(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> Optional[dict[str, Any]]:
    """Compute pairwise agreement between two judge DataFrames."""
    common = set(df_a["query"]) & set(df_b["query"])
    if not common:
        logger.warning("No common queries found between datasets.")
        return None

    a_common = df_a[df_a["query"].isin(common)]
    b_common = df_b[df_b["query"].isin(common)]

    merged = pd.merge(
        a_common[["query", "simulator_choice_label"]],
        b_common[["query", "simulator_choice_label"]],
        on="query",
        suffixes=("_a", "_b"),
    )
    if len(merged) == 0:
        return None

    agreements = (
        merged["simulator_choice_label_a"] == merged["simulator_choice_label_b"]
    ).sum()
    total = len(merged)
    rate = float(agreements) / total
    kappa = float(
        cohen_kappa_score(
            merged["simulator_choice_label_a"],
            merged["simulator_choice_label_b"],
        )
    )

    return {
        "agreement_rate": rate,
        "cohens_kappa": kappa,
        "sample_size": total,
        "agreements": int(agreements),
    }


def combine_datasets(
    files: dict[str, str | Path],
    output_csv: str | Path = "combined_preferences.csv",
    output_summary: str | Path | None = None,
) -> pd.DataFrame:
    """Merge multiple judge JSONL files and save a combined CSV.

    Args:
        files: Mapping ``{label: path}``, e.g.
            ``{"gpt4.1": "prefs_gpt.jsonl", "sonnet4": "prefs_sonnet.jsonl"}``.
        output_csv: Where to save the merged ``.csv``.
        output_summary: Optional path for a JSON summary of agreement stats.

    Returns:
        The combined ``DataFrame``.
    """
    dfs: list[pd.DataFrame] = []
    for label, path in files.items():
        data = load_jsonl(path)
        df = _jsonl_to_dataframe(data, label)
        logger.info("Loaded %d pairs from %s (%s)", len(df), path, label)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing essential fields and non-preference labels.
    essential = ["query", "chosen_response", "rejected_response", "simulator_choice_label"]
    combined = combined.dropna(subset=essential)
    combined = combined[combined["simulator_choice_label"].isin(["A", "B"])]

    logger.info("Combined dataset: %d pairs", len(combined))

    # Cross-LLM agreement
    agreement: Optional[dict[str, Any]] = None
    if len(dfs) >= 2:
        agreement = _compute_agreement(dfs[0], dfs[1])
        if agreement:
            logger.info(
                "Inter-judge agreement: %.1f%% (κ=%.3f, n=%d)",
                agreement["agreement_rate"] * 100,
                agreement["cohens_kappa"],
                agreement["sample_size"],
            )

    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    logger.info("Saved combined CSV to %s", output_csv)

    if output_summary:
        summary = {
            "total_pairs": int(len(combined)),
            "sources": {label: int(len(df)) for label, df in zip(files, dfs)},
            "unique_queries": int(combined["query"].nunique()),
            "inter_judge_agreement": agreement,
        }
        save_json(summary, output_summary)
        logger.info("Saved summary to %s", output_summary)

    return combined


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Combine preference datasets from multiple judges."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Pairs of LABEL=PATH, e.g. gpt4.1=gpt_prefs.jsonl sonnet4=sonnet_prefs.jsonl",
    )
    parser.add_argument(
        "--output_csv", default="combined_preferences.csv", help="Output CSV path."
    )
    parser.add_argument(
        "--output_summary", default=None, help="Optional JSON summary path."
    )
    args = parser.parse_args()

    file_map: dict[str, str] = {}
    for item in args.files:
        label, path = item.split("=", 1)
        file_map[label] = path

    combine_datasets(file_map, args.output_csv, args.output_summary)
