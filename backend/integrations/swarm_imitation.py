"""面向 2–8 架无人机的集中训练、共享执行 imitation baseline。

网络只消费公开的 ``cf_swarm_autopilot`` observation contract。训练标签可来自
Swarm privileged teacher、UrbanFly Helsinki Expert 或后续 DAgger 回放；不依赖
强化学习奖励即可训练。动态无人机数量通过共享单机编码器和 self-attention 支持。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True, slots=True)
class SwarmImitationConfig:
    depth_size: int = 64
    state_width: int = 190
    feature_width: int = 128
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.0


class _DepthEncoder(nn.Module):
    def __init__(self, output_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, output_width),
            nn.LayerNorm(output_width),
            nn.SiLU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.network(depth)


class SharedSwarmImitationPolicy(nn.Module):
    """每机共享编码，集中注意力，输出动态 ``[B,N,5]`` 动作。"""

    def __init__(self, config: SwarmImitationConfig | None = None) -> None:
        super().__init__()
        self.config = config or SwarmImitationConfig()
        width = self.config.feature_width
        self.depth_encoder = _DepthEncoder(width)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_width, width * 2),
            nn.LayerNorm(width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.config.attention_heads,
            dim_feedforward=width * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.team_encoder = nn.TransformerEncoder(layer, self.config.attention_layers)
        self.action_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 5),
        )
        self.collision_head = nn.Sequential(nn.Linear(width, 1), nn.Sigmoid())

    def forward(
        self,
        depth: torch.Tensor,
        state: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """接收 ``depth[B,N,H,W,1]`` 与 ``state[B,N,190]``。"""

        if depth.ndim != 5 or depth.shape[-1] != 1:
            raise ValueError("depth 必须是 [B,N,H,W,1]")
        if state.ndim != 3 or state.shape[:2] != depth.shape[:2]:
            raise ValueError("state 必须是与 depth 对齐的 [B,N,190]")
        if state.shape[-1] != self.config.state_width:
            raise ValueError(f"state 最后一维必须为 {self.config.state_width}")
        batch, drones = state.shape[:2]
        image = depth.permute(0, 1, 4, 2, 3).reshape(batch * drones, 1, depth.shape[2], depth.shape[3])
        depth_feature = self.depth_encoder(image).reshape(batch, drones, -1)
        state_feature = self.state_encoder(state)
        feature = self.fusion(torch.cat((depth_feature, state_feature), dim=-1))
        team_feature = self.team_encoder(feature, src_key_padding_mask=padding_mask)
        raw = self.action_head(team_feature)
        action = torch.cat(
            (
                torch.tanh(raw[..., 0:3]),
                torch.sigmoid(raw[..., 3:4]),
                torch.tanh(raw[..., 4:5]),
            ),
            dim=-1,
        )
        return {
            "action": action,
            "collision_probability": self.collision_head(team_feature).squeeze(-1),
            "feature": team_feature,
        }

    def imitation_loss(
        self,
        prediction: dict[str, torch.Tensor],
        teacher_action: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        collision_label: torch.Tensor | None = None,
        collision_weight: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        action = prediction["action"]
        if teacher_action.shape != action.shape:
            raise ValueError("teacher_action shape 必须与预测动作一致")
        mask = torch.ones(action.shape[:2], device=action.device, dtype=action.dtype)
        if valid_mask is not None:
            if valid_mask.shape != action.shape[:2]:
                raise ValueError("valid_mask 必须是 [B,N]")
            mask = valid_mask.to(dtype=action.dtype)
        denominator = mask.sum().clamp_min(1.0)
        direction = functional.smooth_l1_loss(
            action[..., 0:3], teacher_action[..., 0:3], reduction="none"
        ).mean(dim=-1)
        speed = functional.smooth_l1_loss(
            action[..., 3], teacher_action[..., 3], reduction="none"
        )
        yaw = functional.smooth_l1_loss(
            action[..., 4], teacher_action[..., 4], reduction="none"
        )
        action_loss = ((direction + speed + yaw) * mask).sum() / denominator
        collision_loss = action_loss.new_zeros(())
        if collision_label is not None:
            if collision_label.shape != action.shape[:2]:
                raise ValueError("collision_label 必须是 [B,N]")
            binary = functional.binary_cross_entropy(
                prediction["collision_probability"],
                collision_label.to(dtype=action.dtype),
                reduction="none",
            )
            collision_loss = (binary * mask).sum() / denominator
        total = action_loss + float(collision_weight) * collision_loss
        return {
            "loss": total,
            "action_loss": action_loss,
            "collision_loss": collision_loss,
        }

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "urbanfly-shared-swarm-imitation-v1",
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            target,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, map_location: str = "cpu") -> "SharedSwarmImitationPolicy":
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        if payload.get("schema") != "urbanfly-shared-swarm-imitation-v1":
            raise ValueError("不支持的 swarm imitation checkpoint schema")
        model = cls(SwarmImitationConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model
