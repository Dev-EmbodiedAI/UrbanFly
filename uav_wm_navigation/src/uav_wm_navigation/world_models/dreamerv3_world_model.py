from __future__ import annotations

import torch
from torch import nn

from .encoders import DepthFrameEncoder, RGBFrameEncoder
from .heads import RiskHeads


def _categorical_sample(logits: torch.Tensor, training: bool) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    if training:
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        indices = torch.multinomial(flat, 1).reshape(*probabilities.shape[:-1])
    else:
        indices = probabilities.argmax(dim=-1)
    hard = torch.nn.functional.one_hot(indices, probabilities.shape[-1]).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def _categorical_kl(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    q = q_logits.softmax(dim=-1)
    return (q * (q_logits.log_softmax(dim=-1) - p_logits.log_softmax(dim=-1))).sum(dim=-1)


class DreamerV3WorldModel(nn.Module):
    """DreamerV3-style discrete RSSM used to imagine every YOPO primitive.

    This module intentionally implements the world-model portion only. YOPO is
    still the actor/candidate generator and the existing risk reranker is the
    decision layer, so this is not presented as a full DreamerV3 agent.
    """

    def __init__(
        self,
        state_dim: int = 13,
        trajectory_dim: int = 9,
        latent_dim: int = 64,
        dropout: float = 0.15,
        deterministic_dim: int = 96,
        stochastic_groups: int = 8,
        stochastic_classes: int = 8,
        free_nats: float = 1.0,
        kl_balance: float = 0.8,
    ) -> None:
        super().__init__()
        self.stochastic_groups = int(stochastic_groups)
        self.stochastic_classes = int(stochastic_classes)
        self.stochastic_dim = self.stochastic_groups * self.stochastic_classes
        self.free_nats = float(free_nats)
        self.kl_balance = float(kl_balance)
        self.observation = DepthFrameEncoder(latent_dim)
        self.rgb_observation = RGBFrameEncoder(latent_dim)
        self.rgb_fusion = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU())
        self.state = nn.Linear(state_dim, latent_dim)
        self.goal = nn.Linear(3, latent_dim)
        self.action = nn.Linear(trajectory_dim, latent_dim)
        # Previous factual/candidate action and current proprioception are both
        # transition inputs. This avoids the common one-step action shift bug.
        self.recurrent = nn.GRUCell(self.stochastic_dim + latent_dim * 2, deterministic_dim)
        self.prior = nn.Sequential(
            nn.Linear(deterministic_dim, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, self.stochastic_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(deterministic_dim + latent_dim * 2, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, self.stochastic_dim),
        )
        self.feature = nn.Sequential(
            nn.Linear(deterministic_dim + self.stochastic_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.heads = RiskHeads(latent_dim, dropout)
        self.depth_decoder = nn.Sequential(nn.Linear(latent_dim, 48 * 80), nn.Sigmoid())
        self.reward_head = nn.Linear(latent_dim, 1)
        self.continuation_head = nn.Linear(latent_dim, 1)

    def _logits(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.stochastic_groups, self.stochastic_classes)

    def _observe(
        self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor,
        actions: torch.Tensor | None = None, is_first: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, history = depth.shape[:2]
        observations = self.observation(depth.reshape(batch * history, *depth.shape[2:])).reshape(batch, history, -1)
        if rgb is not None:
            if rgb.shape[:2] != (batch, history):
                raise ValueError("rgb must align with depth as [B,K,3,H,W]")
            rgb_features = self.rgb_observation(rgb.reshape(batch * history, *rgb.shape[2:])).reshape(batch, history, -1)
            observations = observations + self.rgb_fusion(rgb_features)
        states = self.state(state)
        goal_feature = self.goal(goal)
        deterministic = depth.new_zeros(batch, self.recurrent.hidden_size)
        stochastic = depth.new_zeros(batch, self.stochastic_dim)
        zero_action = depth.new_zeros(batch, self.action.in_features)
        dynamics_terms, representation_terms = [], []
        for step in range(history):
            if is_first is not None:
                keep = (1.0 - is_first[:, step].to(depth.dtype))[:, None]
                deterministic, stochastic = deterministic * keep, stochastic * keep
            previous_action = zero_action if actions is None or step == 0 else actions[:, step - 1]
            action_feature = self.action(previous_action)
            deterministic = self.recurrent(
                torch.cat([stochastic, action_feature, states[:, step]], dim=-1), deterministic
            )
            prior_logits = self._logits(self.prior(deterministic))
            posterior_logits = self._logits(
                self.posterior(torch.cat([deterministic, observations[:, step], goal_feature], dim=-1))
            )
            stochastic = _categorical_sample(posterior_logits, self.training).flatten(start_dim=-2)
            dynamics_terms.append(_categorical_kl(posterior_logits.detach(), prior_logits))
            representation_terms.append(_categorical_kl(posterior_logits, prior_logits.detach()))
        dynamics_kl = torch.stack(dynamics_terms, dim=1).mean(dim=-1)
        representation_kl = torch.stack(representation_terms, dim=1).mean(dim=-1)
        dynamics_kl = dynamics_kl.clamp_min(self.free_nats).mean()
        representation_kl = representation_kl.clamp_min(self.free_nats).mean()
        balanced_kl = self.kl_balance * dynamics_kl + (1.0 - self.kl_balance) * representation_kl
        return deterministic, stochastic, balanced_kl

    def sequence_training_loss(
        self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor,
        reward: torch.Tensor, continuation: torch.Tensor, is_first: torch.Tensor,
        burn_in: int = 4,
        rgb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Train the RSSM on factual, contiguous episode transitions only."""
        batch, length = depth.shape[:2]
        observations = self.observation(depth.reshape(batch * length, *depth.shape[2:])).reshape(batch, length, -1)
        if rgb is not None:
            if rgb.shape[:2] != (batch, length):
                raise ValueError("rgb must align with depth as [B,T,3,H,W]")
            rgb_features = self.rgb_observation(rgb.reshape(batch * length, *rgb.shape[2:])).reshape(batch, length, -1)
            observations = observations + self.rgb_fusion(rgb_features)
        states, goals = self.state(state), self.goal(goal)
        deterministic = depth.new_zeros(batch, self.recurrent.hidden_size)
        stochastic = depth.new_zeros(batch, self.stochastic_dim)
        reconstruction_terms, reward_terms, continuation_terms, dynamics_terms, representation_terms = [], [], [], [], []
        for step in range(length):
            keep = (1.0 - is_first[:, step].to(depth.dtype))[:, None]
            deterministic, stochastic = deterministic * keep, stochastic * keep
            previous_action = depth.new_zeros(batch, actions.shape[-1]) if step == 0 else actions[:, step - 1]
            deterministic = self.recurrent(
                torch.cat([stochastic, self.action(previous_action), states[:, step]], dim=-1), deterministic
            )
            prior_logits = self._logits(self.prior(deterministic))
            posterior_logits = self._logits(
                self.posterior(torch.cat([deterministic, observations[:, step], goals[:, step]], dim=-1))
            )
            stochastic = _categorical_sample(posterior_logits, self.training).flatten(start_dim=-2)
            latent = self.feature(torch.cat([deterministic, stochastic], dim=-1))
            if step >= burn_in:
                prediction = self.depth_decoder(latent).reshape(batch, 1, 48, 80)
                target = torch.nn.functional.interpolate(depth[:, step], size=(48, 80), mode="nearest")
                reconstruction_terms.append(torch.nn.functional.smooth_l1_loss(prediction, target))
                reward_target = torch.sign(reward[:, step]) * torch.log1p(reward[:, step].abs())
                reward_terms.append(torch.nn.functional.smooth_l1_loss(self.reward_head(latent).squeeze(-1), reward_target))
                continuation_terms.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    self.continuation_head(latent).squeeze(-1), continuation[:, step]
                ))
                dynamics_terms.append(_categorical_kl(posterior_logits.detach(), prior_logits).mean(dim=-1))
                representation_terms.append(_categorical_kl(posterior_logits, prior_logits.detach()).mean(dim=-1))
        if not reconstruction_terms:
            raise ValueError("sequence length must be greater than burn_in")
        reconstruction_loss = torch.stack(reconstruction_terms).mean()
        reward_loss = torch.stack(reward_terms).mean()
        continuation_loss = torch.stack(continuation_terms).mean()
        dynamics_kl = torch.stack(dynamics_terms).clamp_min(self.free_nats).mean()
        representation_kl = torch.stack(representation_terms).clamp_min(self.free_nats).mean()
        kl_loss = self.kl_balance * dynamics_kl + (1.0 - self.kl_balance) * representation_kl
        total = reconstruction_loss + reward_loss + continuation_loss + kl_loss
        return total, {
            "sequence_reconstruction": reconstruction_loss, "sequence_reward": reward_loss,
            "sequence_continuation": continuation_loss, "sequence_kl": kl_loss,
        }

    def forward(
        self, depth: torch.Tensor, state: torch.Tensor, goal: torch.Tensor, trajectories: torch.Tensor,
        future_depth: torch.Tensor | None = None, selected_index: torch.Tensor | None = None,
        future_valid_mask: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if trajectories.ndim != 4:
            raise ValueError("trajectories must have shape [B, N, H, D]")
        batch, count, horizon, _ = trajectories.shape
        deterministic, stochastic, rssm_kl = self._observe(depth, state, goal, rgb=rgb)
        state_context = self.state(state[:, -1])
        deterministic = deterministic[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        stochastic = stochastic[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        state_context = state_context[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        actions = self.action(trajectories).reshape(batch * count, horizon, -1)
        imagined = []
        for step in range(horizon):
            deterministic = self.recurrent(
                torch.cat([stochastic, actions[:, step], state_context], dim=-1), deterministic
            )
            prior_logits = self._logits(self.prior(deterministic))
            stochastic = _categorical_sample(prior_logits, self.training).flatten(start_dim=-2)
            imagined.append(self.feature(torch.cat([deterministic, stochastic], dim=-1)))
        latent = torch.stack(imagined, dim=1).reshape(batch, count, horizon, -1)
        output = self.heads(latent)
        reconstruction_loss = latent.new_zeros(())
        if future_depth is not None and selected_index is not None:
            future_start = int(getattr(self, "future_start_index", 1))
            future = min(future_depth.shape[1], max(horizon - future_start, 0))
            selected = selected_index.long().clamp(0, count - 1)
            selected_latent = latent[
                torch.arange(batch, device=latent.device), selected, future_start : future_start + future
            ]
            predicted_depth = self.depth_decoder(selected_latent).reshape(batch, future, 1, 48, 80)
            target_depth = torch.nn.functional.interpolate(
                future_depth[:, :future].reshape(batch * future, 1, *future_depth.shape[-2:]),
                size=(48, 80), mode="nearest",
            ).reshape(batch, future, 1, 48, 80)
            error = torch.nn.functional.smooth_l1_loss(predicted_depth, target_depth, reduction="none").mean((-1, -2, -3))
            mask = torch.ones_like(error) if future_valid_mask is None else future_valid_mask[:, :future].to(error.dtype)
            reconstruction_loss = (error * mask).sum() / mask.sum().clamp_min(1.0)
            output["predicted_future_depth"] = predicted_depth
        output["rssm_kl"] = rssm_kl
        output["reconstruction_loss"] = reconstruction_loss
        output["auxiliary_loss"] = rssm_kl + reconstruction_loss
        return output
