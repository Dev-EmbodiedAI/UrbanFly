from __future__ import annotations

import torch
from torch import nn


class RiskHeads(nn.Module):
    def __init__(self, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.collision = nn.Linear(latent_dim, 1)
        self.clearance = nn.Linear(latent_dim, 1)
        self.progress = nn.Linear(latent_dim, 1)
        self.failure = nn.Linear(latent_dim, 1)

    def forward(self, latent_sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = latent_sequence.mean(dim=-2)
        hidden = self.shared(pooled)
        return {
            "collision_logits": self.collision(hidden).squeeze(-1),
            "minimum_clearance": torch.nn.functional.softplus(self.clearance(hidden).squeeze(-1)),
            "goal_progress": self.progress(hidden).squeeze(-1),
            "failure_logits": self.failure(hidden).squeeze(-1),
            "latent_states": latent_sequence,
            "uncertainty": torch.zeros_like(self.collision(hidden).squeeze(-1)),
        }
