from __future__ import annotations

import torch
from torch import nn

from .encoders import ContextEncoder
from .heads import RiskHeads


class GRUWorldModel(nn.Module):
    def __init__(self, state_dim: int = 13, trajectory_dim: int = 9, latent_dim: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.context = ContextEncoder(state_dim, latent_dim, dropout)
        self.action = nn.Linear(trajectory_dim, latent_dim)
        self.cell = nn.GRUCell(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = RiskHeads(latent_dim, dropout)

    def forward(self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, trajectories: torch.Tensor) -> dict[str, torch.Tensor]:
        if trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B, N, H, D]")
        batch, count, horizon, _ = trajectories.shape
        context = self.context(depth, state, goal)[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
        hidden = context
        sequence = []
        actions = self.action(trajectories.reshape(batch * count, horizon, -1))
        for step in range(horizon):
            hidden = self.cell(actions[:, step], hidden)
            sequence.append(self.dropout(hidden))
        latent = torch.stack(sequence, dim=1).reshape(batch, count, horizon, -1)
        return self.heads(latent)

