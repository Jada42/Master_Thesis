"""Reward model architecture.

Wraps a LoRA-tuned base model with a learned linear reward head that maps
last-token hidden states to scalar reward scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """Pairwise preference reward model.

    A base causal LM (e.g. Mistral-7B with LoRA) produces hidden states; the
    reward head extracts the representation of the last non-padding token and
    projects it to a scalar reward.
    """

    def __init__(self, base_model: nn.Module) -> None:
        """Initialise the reward model.

        Args:
            base_model: A ``transformers`` model (typically LoRA-wrapped)
                that returns ``hidden_states``.
        """
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar reward for each sequence in the batch.

        Args:
            input_ids: ``[batch, seq_len]`` token indices.
            attention_mask: ``[batch, seq_len]`` attention mask.

        Returns:
            ``[batch]`` reward scores.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        seq_lens = attention_mask.sum(dim=1) - 1  # last non-pad index
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, seq_lens]  # [batch, hidden]
        return self.reward_head(last_hidden).squeeze(-1)


def pairwise_ranking_loss(
    model: RewardModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Bradley-Terry pairwise loss.

    Args:
        model: The ``RewardModel``.
        batch: Dictionary with ``chosen_input_ids``, ``chosen_attention_mask``,
            ``rejected_input_ids``, ``rejected_attention_mask``.
        device: Target device.

    Returns:
        ``(loss, chosen_rewards, rejected_rewards)``.
    """
    chosen_rewards = model(
        batch["chosen_input_ids"].to(device),
        batch["chosen_attention_mask"].to(device),
    )
    rejected_rewards = model(
        batch["rejected_input_ids"].to(device),
        batch["rejected_attention_mask"].to(device),
    )
    loss = -torch.nn.functional.logsigmoid(
        chosen_rewards - rejected_rewards
    ).mean()
    return loss, chosen_rewards, rejected_rewards
