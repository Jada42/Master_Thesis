"""Reward model dataset.

Wraps a DataFrame of preference pairs (chosen vs. rejected responses) and
returns tokenized tensors suitable for pairwise ranking loss.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PreferenceDataset(Dataset):
    """PyTorch Dataset for RLHF preference pairs.

    Each item encodes a query-response pair where one response was labelled
    as preferred ("chosen") and the other as non-preferred ("rejected") by
    an LLM judge.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        """Initialise the dataset.

        Args:
            data: DataFrame with columns ``query``, ``chosen_response``,
                and ``rejected_response``.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Truncation length for tokenization.
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        prompt = f"### Question: {row['query']}\n### Answer: "

        chosen_enc = self.tokenizer(
            prompt + str(row["chosen_response"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            prompt + str(row["rejected_response"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }
