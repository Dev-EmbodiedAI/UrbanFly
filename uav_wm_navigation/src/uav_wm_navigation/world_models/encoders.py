from __future__ import annotations

import torch
from torch import nn


class DepthFrameEncoder(nn.Module):
    """Encode one normalized metric-depth frame into a compact feature."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(48 * 4, latent_dim), nn.ReLU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.network(depth)


class RGBFrameEncoder(nn.Module):
    """Encode one normalized RGB frame with the same latent width as depth."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(64 * 4, latent_dim), nn.SiLU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.network(rgb)


class DepthHistoryEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        # Keep the historical ``network.*`` state-dict keys compatible with
        # checkpoints produced before DepthFrameEncoder was introduced.
        self.network = DepthFrameEncoder(latent_dim).network
        self.temporal = nn.GRU(latent_dim, latent_dim, batch_first=True)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 5:
            raise ValueError("depth must have shape [B, K, 1, H, W]")
        batch, history = depth.shape[:2]
        encoded = self.network(depth.reshape(batch * history, *depth.shape[2:])).reshape(batch, history, -1)
        _, hidden = self.temporal(encoded)
        return hidden[-1]


class ContextEncoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.depth = DepthHistoryEncoder(latent_dim)
        self.state = nn.GRU(state_dim, latent_dim // 2, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2 + 3, latent_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        depth_latent = self.depth(depth)
        _, state_hidden = self.state(state)
        return self.fusion(torch.cat([depth_latent, state_hidden[-1], goal], dim=-1))
