from __future__ import annotations

import torch
from torch import nn

from .encoders import ContextEncoder
from .heads import RiskHeads


class TransformerWorldModel(nn.Module):
    def __init__(
        self, state_dim: int = 13, trajectory_dim: int = 9, latent_dim: int = 64,
        layers: int = 2, heads: int = 4, dropout: float = 0.15, max_horizon: int = 64,
    ) -> None:
        super().__init__()
        self.context = ContextEncoder(state_dim, latent_dim, dropout)
        self.action = nn.Linear(trajectory_dim, latent_dim)
        self.position = nn.Parameter(torch.zeros(1, max_horizon + 1, latent_dim))
        layer = nn.TransformerEncoderLayer(latent_dim, heads, latent_dim * 4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.heads = RiskHeads(latent_dim, dropout)

    def forward(self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, trajectories: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, count, horizon, _ = trajectories.shape
        if horizon + 1 > self.position.shape[1]:
            raise ValueError("trajectory horizon exceeds configured max_horizon")
        context = self.context(depth, state, goal)[:, None, None, :].expand(-1, count, 1, -1)
        actions = self.action(trajectories)
        tokens = torch.cat([context, actions], dim=2).reshape(batch * count, horizon + 1, -1)
        tokens = tokens + self.position[:, : horizon + 1]
        mask = torch.triu(torch.ones(horizon + 1, horizon + 1, device=tokens.device, dtype=torch.bool), diagonal=1)
        output = self.transformer(tokens, mask=mask)[:, 1:].reshape(batch, count, horizon, -1)
        return self.heads(output)

