from __future__ import annotations

import math

import torch
from torch import nn

from uav_wm_navigation.types import VoxelGridSpec

from .heads import RiskHeads


class DepthVoxelizer(nn.Module):
    def __init__(self, grid: VoxelGridSpec, depth_max_m: float = 20.0, fov_degrees: float = 90.0) -> None:
        super().__init__()
        self.grid = grid
        self.depth_max_m = float(depth_max_m)
        self.fov_degrees = float(fov_degrees)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 5:
            raise ValueError("depth must have shape [B,K,1,H,W]")
        batch, history, _, height, width = depth.shape
        sample = torch.nn.functional.interpolate(
            depth.reshape(batch * history, 1, height, width), size=(24, 40), mode="nearest"
        ).reshape(batch, history, 24, 40) * self.depth_max_m
        focal = 0.5 * 40.0 / math.tan(math.radians(self.fov_degrees) * 0.5)
        vertical, horizontal = torch.meshgrid(
            torch.arange(24, device=depth.device, dtype=depth.dtype),
            torch.arange(40, device=depth.device, dtype=depth.dtype), indexing="ij",
        )
        camera_x = (horizontal - 19.5) / focal * sample
        camera_y = (vertical - 11.5) / focal * sample
        points = torch.stack([sample, -camera_x, -camera_y], dim=-1)
        minimum = torch.tensor(self.grid.minimum_flu, device=depth.device, dtype=depth.dtype)
        indices = torch.floor((points - minimum) / self.grid.resolution_m).long()
        x_size, y_size, z_size = self.grid.shape_xyz
        valid = (sample > 0.05) & (sample < self.depth_max_m)
        valid &= (indices[..., 0] >= 0) & (indices[..., 0] < x_size)
        valid &= (indices[..., 1] >= 0) & (indices[..., 1] < y_size)
        valid &= (indices[..., 2] >= 0) & (indices[..., 2] < z_size)
        flat_index = indices[..., 2] * (y_size * x_size) + indices[..., 1] * x_size + indices[..., 0]
        flat_index = flat_index.clamp(0, x_size * y_size * z_size - 1)
        occupancy = depth.new_zeros(batch * history, x_size * y_size * z_size)
        occupancy.scatter_add_(
            1, flat_index.reshape(batch * history, -1), valid.reshape(batch * history, -1).to(depth.dtype)
        )
        occupancy.clamp_(0.0, 1.0)
        return occupancy.reshape(batch, history, z_size, y_size, x_size)


class OccFlowWorldModel(nn.Module):
    """Lightweight ego-centric 3-D occupancy/flow model for YOPO reranking."""

    def __init__(
        self, state_dim: int = 13, trajectory_dim: int = 9, latent_dim: int = 64,
        dropout: float = 0.15, history: int = 4, future_steps: int = 5,
        depth_max_m: float = 20.0, voxel_resolution_m: float = 0.5,
        voxel_minimum_flu: tuple[float, float, float] = (-4.0, -8.0, -3.0),
        voxel_maximum_flu: tuple[float, float, float] = (20.0, 8.0, 5.0),
    ) -> None:
        super().__init__()
        self.grid = VoxelGridSpec(voxel_minimum_flu, voxel_maximum_flu, voxel_resolution_m)
        self.future_steps = int(future_steps)
        self.voxelizer = DepthVoxelizer(self.grid, depth_max_m)
        self.encoder = nn.Sequential(
            nn.Conv3d(history, 12, 3, padding=1), nn.SiLU(),
            nn.Conv3d(12, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 24, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose3d(24, 16, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(16, self.future_steps * 6, 1),
        )
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32 + state_dim + 3, latent_dim), nn.SiLU()
        )
        self.action = nn.Linear(trajectory_dim, latent_dim)
        self.rollout = nn.GRUCell(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = RiskHeads(latent_dim, dropout)

    def _sample_path(self, occupancy_logits: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        probability = occupancy_logits.sigmoid()
        batch, candidates, horizon = trajectories.shape[:3]
        x_size, y_size, z_size = self.grid.shape_xyz
        minimum = trajectories.new_tensor(self.grid.minimum_flu)
        indices = torch.floor((trajectories[..., :3] - minimum) / self.grid.resolution_m).long()
        x = indices[..., 0].clamp(0, x_size - 1)
        y = indices[..., 1].clamp(0, y_size - 1)
        z = indices[..., 2].clamp(0, z_size - 1)
        time_index = torch.linspace(0, self.future_steps - 1, horizon, device=trajectories.device).round().long()
        flat = probability.reshape(batch, self.future_steps, 3, -1)
        voxel_index = z * (y_size * x_size) + y * x_size + x
        values = []
        for step in range(horizon):
            gather_index = voxel_index[:, :, step][:, None, :].expand(-1, 3, -1)
            values.append(torch.gather(flat[:, time_index[step]], 2, gather_index).transpose(1, 2))
        return torch.stack(values, dim=2)

    def forward(self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, trajectories: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        if trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B,N,H,D]")
        batch, count, horizon, _ = trajectories.shape
        voxels = self.voxelizer(depth)
        encoded = self.encoder(voxels)
        decoded = self.decoder(encoded)
        z_size, y_size, x_size = self.grid.shape_zyx
        decoded = decoded[..., :z_size, :y_size, :x_size]
        decoded = decoded.reshape(batch, self.future_steps, 6, z_size, y_size, x_size)
        occupancy_logits, flow = decoded[:, :, :3], decoded[:, :, 3:]
        context = self.context(torch.cat([
            encoded,
            state[:, -1, :, None, None, None].expand(-1, -1, *encoded.shape[-3:]),
            goal[:, :, None, None, None].expand(-1, -1, *encoded.shape[-3:]),
        ], dim=1))
        hidden = context[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        actions = self.action(trajectories).reshape(batch * count, horizon, -1)
        latent = []
        for step in range(horizon):
            hidden = self.rollout(actions[:, step], hidden)
            latent.append(self.dropout(hidden))
        latent_sequence = torch.stack(latent, dim=1).reshape(batch, count, horizon, -1)
        output = self.heads(latent_sequence)
        path_occupancy = self._sample_path(occupancy_logits, trajectories)
        explicit_risk = path_occupancy[..., :2].amax(dim=(-1, -2)).clamp(1e-4, 1 - 1e-4)
        output["collision_logits"] = output["collision_logits"] + torch.logit(explicit_risk)
        output["occupancy_logits"] = occupancy_logits
        output["flow"] = flow
        output["path_occupancy"] = path_occupancy
        output["auxiliary_loss"] = latent_sequence.new_zeros(())
        return output
