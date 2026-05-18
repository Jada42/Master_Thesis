"""Intrinsic Coherence Module (ICM) for RLHF/PPO alignment.

The ICM measures semantic coherence between a user query and the model's
response. A low coherence score penalizes off-topic or irrelevant answers,
encouraging the PPO policy to stay focused on the user's question.

Two variants are provided:
- ``IntrinsicCoherenceModule`` — simple linear projection (ablation study)
- ``AdvancedICM`` — 3-layer MLP with LayerNorm/ReLU/Dropout (production)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IntrinsicCoherenceModule(nn.Module):
    """Simple linear coherence projector (used in ablation study)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvancedICM(nn.Module):
    """Multi-layer coherence module with improved architecture.

    Architecture: Linear(embed_dim → 512) → LayerNorm → ReLU → Dropout(0.1)
    → Linear(512 → 256) → LayerNorm → ReLU → Linear(256 → embed_dim).

    The module learns to predict the ideal response embedding from the
    query embedding. The cosine similarity between the predicted and actual
    response embedding is used as the coherence reward.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Project query embedding toward response embedding space."""
        return self.network(query_embedding)

    def compute_reward_and_loss(
        self,
        query_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """Compute coherence reward (0–1) and MSE training loss.

        Returns:
            Tuple of (coherence_reward, loss). The reward is
            sigmoid(cosine_similarity) between the predicted and
            actual response embeddings.
        """
        predicted = self.forward(query_embedding)
        loss = self.loss_fn(predicted, response_embedding)
        cosine_sim = torch.cosine_similarity(predicted, response_embedding, dim=0)
        coherence_reward = float(torch.sigmoid(cosine_sim).item())
        return coherence_reward, loss
