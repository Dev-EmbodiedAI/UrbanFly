from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisualWorldModelConfig:
    image_size: int = 64
    image_channels: int = 4
    action_dim: int = 4
    state_dim: int = 12
    base_channels: int = 64
    embed_dim: int = 1024
    deter_dim: int = 1024
    stoch_dim: int = 64
    hidden_dim: int = 1024
    min_std: float = 0.1
    bottom_crop_fraction: float = 1.0 / 3.0

    @classmethod
    def preset(cls, name: str) -> "VisualWorldModelConfig":
        presets = {
            "small": cls(base_channels=32, embed_dim=384, deter_dim=384, stoch_dim=32, hidden_dim=512),
            "medium": cls(base_channels=64, embed_dim=1024, deter_dim=1024, stoch_dim=64, hidden_dim=1024),
            "large": cls(base_channels=96, embed_dim=1536, deter_dim=1536, stoch_dim=96, hidden_dim=1536),
        }
        try:
            return presets[name]
        except KeyError as exc:
            raise ValueError(f"unknown visual world-model preset: {name}") from exc


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvEncoder(nn.Module):
    def __init__(self, config: VisualWorldModelConfig) -> None:
        super().__init__()
        b = config.base_channels
        channels = [config.image_channels, b, b * 2, b * 4, b * 8]
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
                    _norm(output_channels),
                    nn.SiLU(),
                ]
            )
        self.convs = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b * 8 * 4 * 4, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.SiLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convs(observation))


class ConvDecoder(nn.Module):
    def __init__(self, config: VisualWorldModelConfig) -> None:
        super().__init__()
        b = config.base_channels
        feature_dim = config.deter_dim + config.stoch_dim
        self.base_channels = b
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, b * 8 * 4 * 4),
            nn.SiLU(),
        )
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(b * 8, b * 4, 4, 2, 1),
            _norm(b * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(b * 4, b * 2, 4, 2, 1),
            _norm(b * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(b * 2, b, 4, 2, 1),
            _norm(b),
            nn.SiLU(),
            nn.ConvTranspose2d(b, config.image_channels, 4, 2, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        shape = feature.shape[:-1]
        flat = feature.reshape(-1, feature.shape[-1])
        hidden = self.projection(flat).reshape(-1, self.base_channels * 8, 4, 4)
        decoded = torch.sigmoid(self.deconvs(hidden))
        return decoded.reshape(*shape, *decoded.shape[-3:])


class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class VisualRSSM(nn.Module):
    """RGB-D recurrent state-space model for action-conditioned flight prediction."""

    def __init__(self, config: VisualWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(config)
        self.decoder = ConvDecoder(config)
        self.recurrent = nn.GRUCell(config.stoch_dim + config.action_dim, config.deter_dim)
        self.prior = PredictionHead(config.deter_dim, config.hidden_dim, config.stoch_dim * 2)
        self.posterior = PredictionHead(
            config.deter_dim + config.embed_dim,
            config.hidden_dim,
            config.stoch_dim * 2,
        )
        feature_dim = config.deter_dim + config.stoch_dim
        self.state_head = PredictionHead(feature_dim, config.hidden_dim, config.state_dim)
        self.reward_head = PredictionHead(feature_dim, config.hidden_dim, 1)
        self.risk_head = PredictionHead(feature_dim, config.hidden_dim, 1)
        self.continue_head = PredictionHead(feature_dim, config.hidden_dim, 1)
        self.value_head = PredictionHead(feature_dim, config.hidden_dim, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        device = device or next(self.parameters()).device
        return RSSMState(
            deter=torch.zeros(batch_size, self.config.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.config.stoch_dim, device=device),
        )

    def _distribution(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = params.chunk(2, dim=-1)
        mean = mean.clamp(-20.0, 20.0)
        std = F.softplus(raw_std.clamp(-5.0, 5.0)) + self.config.min_std
        return mean, std

    def transition(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        embedding: torch.Tensor | None = None,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, dict[str, torch.Tensor]]:
        deter = self.recurrent(torch.cat([previous.stoch, action], dim=-1), previous.deter)
        prior_mean, prior_std = self._distribution(self.prior(deter))
        if embedding is None:
            mean, std = prior_mean, prior_std
        else:
            mean, std = self._distribution(self.posterior(torch.cat([deter, embedding], dim=-1)))
        stoch = mean + std * torch.randn_like(std) if sample else mean
        state = RSSMState(deter=deter, stoch=stoch)
        return state, {
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "post_mean": mean,
            "post_std": std,
        }

    @staticmethod
    def feature(state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def predict(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "state": self.state_head(feature),
            "reward": self.reward_head(feature).squeeze(-1),
            "risk_logit": self.risk_head(feature).squeeze(-1),
            "continue_logit": self.continue_head(feature).squeeze(-1),
            "value": self.value_head(feature).squeeze(-1),
        }

    def observe(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        if observations.ndim != 5 or actions.ndim != 3:
            raise ValueError("observations must be BxTxCxHxW and actions must be BxTxA")
        batch_size, sequence_length = observations.shape[:2]
        embeddings = self.encoder(observations.flatten(0, 1)).reshape(batch_size, sequence_length, -1)
        state = self.initial(batch_size, observations.device)
        outputs: dict[str, list[torch.Tensor]] = {
            key: []
            for key in ("deter", "stoch", "prior_mean", "prior_std", "post_mean", "post_std")
        }
        zero_action = torch.zeros_like(actions[:, 0])
        for index in range(sequence_length):
            previous_action = zero_action if index == 0 else actions[:, index - 1]
            state, stats = self.transition(state, previous_action, embeddings[:, index], sample=sample)
            outputs["deter"].append(state.deter)
            outputs["stoch"].append(state.stoch)
            for key, value in stats.items():
                outputs[key].append(value)
        stacked = {key: torch.stack(value, dim=1) for key, value in outputs.items()}
        feature = torch.cat([stacked["deter"], stacked["stoch"]], dim=-1)
        return stacked | {"feature": feature} | self.predict(feature)

    def imagine(
        self,
        initial: RSSMState,
        actions: torch.Tensor,
        *,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = initial
        features: list[torch.Tensor] = []
        for index in range(actions.shape[1]):
            state, _ = self.transition(state, actions[:, index], sample=sample)
            features.append(self.feature(state))
        feature = torch.stack(features, dim=1)
        return {"feature": feature, "deter": state.deter, "stoch": state.stoch} | self.predict(feature)

    @staticmethod
    def _normal_kl(
        mean_q: torch.Tensor,
        std_q: torch.Tensor,
        mean_p: torch.Tensor,
        std_p: torch.Tensor,
    ) -> torch.Tensor:
        variance_ratio = (std_q / std_p).square()
        mean_term = ((mean_q - mean_p) / std_p).square()
        return 0.5 * (variance_ratio + mean_term - 1.0 + 2.0 * (std_p.log() - std_q.log()))

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        kl_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        state_weight: float = 1.0,
        reward_weight: float = 1.0,
        risk_weight: float = 2.0,
        continue_weight: float = 0.5,
        value_weight: float = 0.5,
        risk_positive_weight: float = 1.0,
        free_nats: float = 1.0,
        discount: float = 0.99,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.observe(batch["observations"], batch["actions"])
        reconstruction = self.decoder(output["feature"])
        rgb_loss = F.smooth_l1_loss(reconstruction[:, :, :3], batch["observations"][:, :, :3])
        depth_loss = F.smooth_l1_loss(reconstruction[:, :, 3:], batch["observations"][:, :, 3:])
        reconstruction_loss = rgb_loss + 2.0 * depth_loss
        state_loss = F.smooth_l1_loss(output["state"], batch["states"])

        zero = torch.zeros_like(batch["rewards"][:, :1])
        reward_target = torch.cat([zero, batch["rewards"][:, :-1]], dim=1)
        risk_target = torch.cat([zero, batch["risks"][:, :-1]], dim=1)
        continue_target = torch.cat([torch.ones_like(zero), batch["continues"][:, :-1]], dim=1)
        reward_loss = F.smooth_l1_loss(output["reward"], reward_target)
        risk_loss = F.binary_cross_entropy_with_logits(
            output["risk_logit"],
            risk_target,
            pos_weight=torch.as_tensor(risk_positive_weight, device=risk_target.device),
        )
        continue_loss = F.binary_cross_entropy_with_logits(output["continue_logit"], continue_target)

        returns = torch.zeros_like(reward_target)
        running = torch.zeros_like(reward_target[:, 0])
        for index in range(reward_target.shape[1] - 1, -1, -1):
            running = reward_target[:, index] + discount * continue_target[:, index] * running
            returns[:, index] = running
        value_loss = F.smooth_l1_loss(output["value"], returns.detach())

        dynamics_kl = self._normal_kl(
            output["post_mean"].detach(),
            output["post_std"].detach(),
            output["prior_mean"],
            output["prior_std"],
        ).sum(-1)
        representation_kl = self._normal_kl(
            output["post_mean"],
            output["post_std"],
            output["prior_mean"].detach(),
            output["prior_std"].detach(),
        ).sum(-1)
        kl_loss = 0.5 * (
            torch.clamp(dynamics_kl, min=free_nats).mean()
            + torch.clamp(representation_kl, min=free_nats).mean()
        )
        total = (
            reconstruction_weight * reconstruction_loss
            + state_weight * state_loss
            + reward_weight * reward_loss
            + risk_weight * risk_loss
            + continue_weight * continue_loss
            + value_weight * value_loss
            + kl_weight * kl_loss
        )
        metrics = {
            "loss": total.detach(),
            "reconstruction": reconstruction_loss.detach(),
            "rgb": rgb_loss.detach(),
            "depth": depth_loss.detach(),
            "state": state_loss.detach(),
            "reward": reward_loss.detach(),
            "risk": risk_loss.detach(),
            "continue": continue_loss.detach(),
            "value": value_loss.detach(),
            "kl": kl_loss.detach(),
        }
        return total, metrics


def action_from_delta(delta: np.ndarray, distance_scale_m: float = 80.0) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1e-6)
    return np.concatenate([direction, [min(distance / distance_scale_m, 1.0)]]).astype(np.float32)


class VisualWorldModelPlanner:
    def __init__(
        self,
        model: VisualRSSM,
        device: torch.device,
        *,
        horizon: int = 8,
        risk_weight: float = 8.0,
        discount: float = 0.97,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.horizon = int(horizon)
        self.risk_weight = float(risk_weight)
        self.discount = float(discount)
        self.belief = model.initial(1, device)
        self.previous_action = torch.zeros(1, model.config.action_dim, device=device)

    @classmethod
    def load(cls, checkpoint: Path, device: torch.device | None = None, **kwargs: Any) -> "VisualWorldModelPlanner":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("format") != "urbanfly-visual-rssm-v1":
            raise ValueError(f"unsupported visual world-model checkpoint: {payload.get('format')}")
        model = VisualRSSM(VisualWorldModelConfig(**payload["config"]))
        model.load_state_dict(payload["model"])
        return cls(model, device, **kwargs)

    def reset(self) -> None:
        self.belief = self.model.initial(1, self.device)
        self.previous_action.zero_()

    def observe(self, rgb: np.ndarray, depth_m: np.ndarray, depth_max_m: float = 20.0) -> None:
        from PIL import Image

        size = self.model.config.image_size
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        rgb_height = max(1, round(rgb.shape[0] * (1.0 - self.model.config.bottom_crop_fraction)))
        depth_height = max(1, round(depth_m.shape[0] * (1.0 - self.model.config.bottom_crop_fraction)))
        rgb_image = Image.fromarray(rgb[:rgb_height]).resize((size, size))
        depth = np.clip(depth_m[:depth_height] / depth_max_m, 0.0, 1.0)
        depth_image = Image.fromarray((depth * 255).astype(np.uint8)).resize((size, size))
        rgb_array = np.asarray(rgb_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        depth_array = np.asarray(depth_image, dtype=np.float32)[None] / 255.0
        observation = torch.from_numpy(np.concatenate([rgb_array, depth_array])[None]).to(self.device)
        with torch.inference_mode():
            embedding = self.model.encoder(observation)
            self.belief, _ = self.model.transition(
                self.belief,
                self.previous_action,
                embedding,
                sample=False,
            )

    def score_deltas(self, deltas: list[np.ndarray]) -> list[dict[str, float]]:
        actions_np = np.stack([action_from_delta(delta) for delta in deltas])
        actions = torch.from_numpy(actions_np).to(self.device)
        candidate_count = len(deltas)
        sequence = actions[:, None, :].repeat(1, self.horizon, 1)
        decay = torch.linspace(1.0, 0.35, self.horizon, device=self.device)
        sequence[:, :, 3] *= decay[None]
        initial = RSSMState(
            deter=self.belief.deter.repeat(candidate_count, 1),
            stoch=self.belief.stoch.repeat(candidate_count, 1),
        )
        with torch.inference_mode():
            imagined = self.model.imagine(initial, sequence, sample=False)
            risk = torch.sigmoid(imagined["risk_logit"])
            discounts = self.discount ** torch.arange(self.horizon, device=self.device)
            score = ((imagined["reward"] - self.risk_weight * risk) * discounts).sum(1)
            score += discounts[-1] * imagined["value"][:, -1]
        return [
            {
                "score": float(score[index].cpu()),
                "predicted_reward": float(imagined["reward"][index].sum().cpu()),
                "risk_probability": float(1.0 - torch.prod(1.0 - risk[index]).cpu()),
            }
            for index in range(candidate_count)
        ]

    def select_delta(
        self,
        deltas: list[np.ndarray],
        *,
        max_risk_probability: float = 1.0,
    ) -> tuple[int, list[dict[str, float]]]:
        records = self.score_deltas(deltas)
        safe = [
            index
            for index, record in enumerate(records)
            if record["risk_probability"] <= max_risk_probability
        ]
        selected = (
            max(safe, key=lambda index: records[index]["score"])
            if safe
            else min(range(len(records)), key=lambda index: records[index]["risk_probability"])
        )
        self.previous_action = torch.from_numpy(action_from_delta(deltas[selected]))[None].to(self.device)
        return selected, records


def save_visual_checkpoint(
    path: Path,
    model: VisualRSSM,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "urbanfly-visual-rssm-v1",
            "config": asdict(model.config),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "metadata": metadata or {},
        },
        path,
    )
