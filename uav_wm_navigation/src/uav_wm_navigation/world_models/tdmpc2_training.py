from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from uav_wm_navigation.world_models.tdmpc2_continuous import TDMPC2Network


@dataclass(frozen=True, slots=True)
class TDMPC2Loss:
    total: float
    consistency: float
    reward: float
    q: float
    risk: float


class TDMPC2Trainer:
    """TD-MPC2-style joint latent, reward, value and risk optimization."""

    def __init__(
        self,
        model: TDMPC2Network,
        *,
        learning_rate: float = 3e-4,
        discount: float = 0.97,
        target_tau: float = 0.01,
        risk_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.target = deepcopy(model).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.discount = float(discount)
        self.target_tau = float(target_tau)
        self.risk_weight = float(risk_weight)
        self.steps = 0

    def train_step(self, batch: dict[str, torch.Tensor]) -> TDMPC2Loss:
        device = next(self.model.parameters()).device
        feature = batch["feature"].to(device)
        action = batch["action"].to(device).clamp(-1.0, 1.0)
        reward_target = batch["reward"].to(device)
        risk_target = batch["risk"].to(device).clamp(0.0, 1.0)
        next_feature = batch["next_feature"].to(device)
        continuation = batch["continuation"].to(device)
        latent = self.model.encode(feature)
        predicted_next = self.model.transition(latent, action)
        joined = torch.cat([latent, action], dim=-1)
        predicted_reward = self.model.reward(joined).squeeze(-1)
        predicted_q1 = self.model.q1(joined).squeeze(-1)
        predicted_q2 = self.model.q2(joined).squeeze(-1)
        predicted_risk_logits = self.model.risk(joined).squeeze(-1)
        with torch.no_grad():
            target_next_latent = self.target.encode(next_feature)
            next_action = self.target.policy(target_next_latent)
            target_joined = torch.cat([target_next_latent, next_action], dim=-1)
            target_q = reward_target + self.discount * continuation * torch.minimum(
                self.target.q1(target_joined).squeeze(-1),
                self.target.q2(target_joined).squeeze(-1),
            )
        consistency_loss = nn.functional.smooth_l1_loss(
            predicted_next, target_next_latent
        )
        reward_loss = nn.functional.mse_loss(predicted_reward, reward_target)
        q_loss = 0.5 * (
            nn.functional.mse_loss(predicted_q1, target_q)
            + nn.functional.mse_loss(predicted_q2, target_q)
        )
        risk_loss = nn.functional.binary_cross_entropy_with_logits(
            predicted_risk_logits, risk_target
        )
        total = consistency_loss + reward_loss + q_loss + self.risk_weight * risk_loss
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target.parameters(), self.model.parameters()
            ):
                target_parameter.lerp_(parameter, self.target_tau)
        self.steps += 1
        return TDMPC2Loss(
            total=float(total.detach()),
            consistency=float(consistency_loss.detach()),
            reward=float(reward_loss.detach()),
            q=float(q_loss.detach()),
            risk=float(risk_loss.detach()),
        )
