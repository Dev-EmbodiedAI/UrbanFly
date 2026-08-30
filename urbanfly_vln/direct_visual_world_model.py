from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from .visual_world_model import ConvEncoder, VisualWorldModelConfig, action_from_delta


@dataclass(frozen=True)
class DirectWorldModelConfig:
    image_size: int = 64
    image_channels: int = 4
    state_dim: int = 12
    action_dim: int = 4
    base_channels: int = 64
    embed_dim: int = 1024
    hidden_dim: int = 1024
    bottom_crop_fraction: float = 1.0 / 3.0

    @classmethod
    def preset(cls, name: str) -> "DirectWorldModelConfig":
        presets = {
            "small": cls(base_channels=32, embed_dim=384, hidden_dim=512),
            "medium": cls(),
            "large": cls(base_channels=96, embed_dim=1536, hidden_dim=1536),
        }
        try:
            return presets[name]
        except KeyError as exc:
            raise ValueError(f"unknown direct world-model preset: {name}") from exc


class DirectVisualWorldModel(nn.Module):
    """One-step world model: RGB-D, state, action -> next state and reward."""

    def __init__(self, config: DirectWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        encoder_config = VisualWorldModelConfig(
            image_size=config.image_size,
            image_channels=config.image_channels,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            base_channels=config.base_channels,
            embed_dim=config.embed_dim,
            bottom_crop_fraction=config.bottom_crop_fraction,
        )
        self.encoder = ConvEncoder(encoder_config)
        input_dim = config.embed_dim + config.state_dim + config.action_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.state_delta_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.reward_head = nn.Linear(config.hidden_dim, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def predict_from_embedding(
        self,
        embedding: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([embedding, state, action], dim=-1))
        next_state = state + self.state_delta_head(hidden)
        reward = self.reward_head(hidden).squeeze(-1)
        return {"next_state": next_state, "reward": reward}

    def forward(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.predict_from_embedding(self.encode(observation), state, action)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self(batch["observations"], batch["states"], batch["actions"])
        state_loss = nn.functional.smooth_l1_loss(output["next_state"], batch["next_states"])
        reward_loss = nn.functional.smooth_l1_loss(output["reward"], batch["rewards"])
        total = state_loss + reward_loss
        return total, {
            "loss": total.detach(),
            "state": state_loss.detach(),
            "reward": reward_loss.detach(),
        }


class DirectWorldModelPlanner:
    DISTANCE_PENALTY_PER_M = 0.001

    def __init__(self, model: DirectVisualWorldModel, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.embedding: torch.Tensor | None = None

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device | None = None) -> "DirectWorldModelPlanner":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("format") != "urbanfly-direct-visual-world-model-v1":
            raise ValueError(f"unsupported direct world-model checkpoint: {payload.get('format')}")
        model = DirectVisualWorldModel(DirectWorldModelConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
        return cls(model, device)

    def observe(self, rgb: np.ndarray, depth_m: np.ndarray, depth_max_m: float = 20.0) -> None:
        config = self.model.config
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        rgb_height = max(1, round(rgb.shape[0] * (1.0 - config.bottom_crop_fraction)))
        depth_height = max(1, round(depth_m.shape[0] * (1.0 - config.bottom_crop_fraction)))
        rgb_image = Image.fromarray(rgb[:rgb_height]).resize((config.image_size, config.image_size))
        depth = np.clip(depth_m[:depth_height] / depth_max_m, 0.0, 1.0)
        depth_image = Image.fromarray((depth * 255).astype(np.uint8)).resize(
            (config.image_size, config.image_size)
        )
        rgb_array = np.asarray(rgb_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        depth_array = np.asarray(depth_image, dtype=np.float32)[None] / 255.0
        observation = torch.from_numpy(np.concatenate([rgb_array, depth_array])[None]).to(self.device)
        with torch.inference_mode():
            self.embedding = self.model.encode(observation)

    def select_delta(
        self,
        *,
        deltas: list[np.ndarray],
        velocity: np.ndarray,
        min_depth_m: float,
        p05_depth_m: float,
        final_goal_distance_m: float,
        candidate_indices: list[int],
        route_length: int,
        depth_max_m: float = 20.0,
    ) -> tuple[int, list[dict[str, float]]]:
        if self.embedding is None:
            raise RuntimeError("observe must be called before candidate selection")
        velocity = np.asarray(velocity, dtype=np.float32)
        actions = np.stack([action_from_delta(delta) for delta in deltas])
        states = []
        for delta, candidate_index in zip(deltas, candidate_indices):
            states.append(
                np.asarray(
                    [
                        *(velocity / 10.0),
                        np.linalg.norm(velocity) / 10.0,
                        p05_depth_m / depth_max_m,
                        min_depth_m / depth_max_m,
                        np.linalg.norm(delta) / 80.0,
                        final_goal_distance_m / 200.0,
                        candidate_index / max(route_length - 1, 1),
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
            )
        state_tensor = torch.from_numpy(np.stack(states)).to(self.device)
        action_tensor = torch.from_numpy(actions).to(self.device)
        embedding = self.embedding.repeat(len(deltas), 1)
        with torch.inference_mode():
            prediction = self.model.predict_from_embedding(embedding, state_tensor, action_tensor)
        next_state = prediction["next_state"].cpu().numpy()
        rewards = prediction["reward"].cpu().numpy()
        records = []
        for index in range(len(deltas)):
            predicted_p05 = float(next_state[index, 4] * depth_max_m)
            predicted_final_distance = float(next_state[index, 7] * 200.0)
            progress = final_goal_distance_m - predicted_final_distance
            distance = float(np.linalg.norm(deltas[index]))
            raw_model_score = float(rewards[index] + 0.1 * progress + 0.01 * predicted_p05)
            score = raw_model_score - self.DISTANCE_PENALTY_PER_M * distance
            records.append(
                {
                    "score": score,
                    "raw_model_score": raw_model_score,
                    "distance_penalty": self.DISTANCE_PENALTY_PER_M * distance,
                    "predicted_reward": float(rewards[index]),
                    "predicted_progress_m": float(progress),
                    "predicted_p05_depth_m": predicted_p05,
                    "predicted_final_distance_m": predicted_final_distance,
                }
            )
        selected = max(range(len(records)), key=lambda index: records[index]["score"])
        return selected, records


def save_direct_checkpoint(
    path: Path,
    model: DirectVisualWorldModel,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "urbanfly-direct-visual-world-model-v1",
            "config": asdict(model.config),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata or {},
        },
        path,
    )
