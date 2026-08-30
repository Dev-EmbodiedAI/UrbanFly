from __future__ import annotations

import copy

import torch
from torch import nn

from .encoders import ContextEncoder
from .heads import RiskHeads


class ActionConditionedJEPAWorldModel(nn.Module):
    """V-JEPA-inspired action-conditioned latent predictor for YOPO candidates.

    The model predicts abstract future depth embeddings instead of pixels. Only
    the actually executed candidate receives a self-supervised future-latent
    target; supervised risk heads remain available for all candidates.
    """

    def __init__(
        self,
        state_dim: int = 13,
        trajectory_dim: int = 9,
        latent_dim: int = 64,
        dropout: float = 0.15,
        layers: int = 2,
        heads: int = 4,
        max_horizon: int = 64,
        ema_decay: float = 0.996,
        mask_ratio: float = 0.5,
        mask_patch_size: int = 12,
        future_start_index: int = 1,
    ) -> None:
        super().__init__()
        self.context = ContextEncoder(state_dim, latent_dim, dropout)
        self.target_encoder = copy.deepcopy(self.context.depth.network)
        self.target_encoder.requires_grad_(False)
        self.ema_decay = float(ema_decay)
        self.mask_ratio = float(mask_ratio)
        self.mask_patch_size = int(mask_patch_size)
        self.future_start_index = int(future_start_index)
        self.action = nn.Linear(trajectory_dim, latent_dim)
        self.position = nn.Parameter(torch.zeros(1, max_horizon + 1, latent_dim))
        layer = nn.TransformerEncoderLayer(latent_dim, heads, latent_dim * 4, dropout, batch_first=True)
        self.predictor = nn.TransformerEncoder(layer, layers)
        self.prediction_projection = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.heads = RiskHeads(latent_dim, dropout)

    def _mask_context(self, depth: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_ratio <= 0:
            return depth
        batch, history, _, height, width = depth.shape
        patch = max(self.mask_patch_size, 1)
        mask_h = max(1, (height + patch - 1) // patch)
        mask_w = max(1, (width + patch - 1) // patch)
        mask = torch.rand(batch * history, 1, mask_h, mask_w, device=depth.device) < self.mask_ratio
        mask = torch.nn.functional.interpolate(mask.float(), size=(height, width), mode="nearest").bool()
        flattened = depth.reshape(batch * history, 1, height, width)
        return flattened.masked_fill(mask, 0.0).reshape_as(depth)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for online, target in zip(self.context.depth.network.parameters(), self.target_encoder.parameters()):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)
        for online, target in zip(self.context.depth.network.buffers(), self.target_encoder.buffers()):
            target.copy_(online)

    def _jepa_target_loss(
        self,
        latent: torch.Tensor,
        future_depth: torch.Tensor,
        selected_index: torch.Tensor,
        future_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, future = future_depth.shape[:2]
        future = min(future, max(latent.shape[2] - self.future_start_index, 0))
        with torch.no_grad():
            target = self.target_encoder(
                future_depth[:, :future].reshape(batch * future, *future_depth.shape[2:])
            ).reshape(batch, future, -1)
            target = torch.nn.functional.layer_norm(target, (target.shape[-1],))
        selected = selected_index.long().clamp(min=0, max=latent.shape[1] - 1)
        predicted = latent[
            torch.arange(batch, device=latent.device), selected,
            self.future_start_index : self.future_start_index + future,
        ]
        predicted = self.prediction_projection(predicted)
        distance = 2.0 - 2.0 * torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        mask = torch.ones_like(distance) if future_valid_mask is None else future_valid_mask[:, :future].to(distance.dtype)
        mask = mask * (selected_index[:, None] >= 0).to(mask.dtype)
        cosine_loss = (distance * mask).sum() / mask.sum().clamp_min(1.0)
        valid_predicted = predicted.reshape(-1, predicted.shape[-1])
        standard_deviation = torch.sqrt(valid_predicted.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = torch.relu(1.0 - standard_deviation).mean()
        centered = valid_predicted - valid_predicted.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(valid_predicted.shape[0] - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        covariance_loss = off_diagonal.square().sum() / predicted.shape[-1]
        return cosine_loss, variance_loss, covariance_loss

    def forward(
        self,
        depth: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
        trajectories: torch.Tensor,
        future_depth: torch.Tensor | None = None,
        selected_index: torch.Tensor | None = None,
        future_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B, N, H, D]")
        batch, count, horizon, _ = trajectories.shape
        if horizon + 1 > self.position.shape[1]:
            raise ValueError("trajectory horizon exceeds configured max_horizon")
        context = self.context(self._mask_context(depth), state, goal)[:, None, None].expand(-1, count, 1, -1)
        actions = self.action(trajectories)
        tokens = torch.cat([context, actions], dim=2).reshape(batch * count, horizon + 1, -1)
        tokens = tokens + self.position[:, : horizon + 1]
        causal_mask = torch.triu(
            torch.ones(horizon + 1, horizon + 1, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        latent = self.predictor(tokens, mask=causal_mask)[:, 1:].reshape(batch, count, horizon, -1)
        output = self.heads(latent)
        if future_depth is not None and selected_index is not None:
            jepa_loss, variance_loss, covariance_loss = self._jepa_target_loss(
                latent, future_depth, selected_index, future_valid_mask
            )
        else:
            jepa_loss = latent.new_zeros(())
            variance_loss = latent.new_zeros(())
            covariance_loss = latent.new_zeros(())
        output["jepa_loss"] = jepa_loss
        output["variance_loss"] = variance_loss
        output["covariance_loss"] = covariance_loss
        output["auxiliary_loss"] = jepa_loss + 0.05 * variance_loss + 0.01 * covariance_loss
        return output
