from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from .tdmpc2_visual import TDMPC2VisualNetwork


@dataclass(frozen=True, slots=True)
class VisualTDMPC2Loss:
    total: float
    consistency: float
    reward: float
    value: float
    risk: float
    clearance: float
    progress: float


class VisualTDMPC2Trainer:
    def __init__(self, model: TDMPC2VisualNetwork, learning_rate: float = 3e-4, discount: float = 0.97, target_tau: float = 0.01) -> None:
        self.model = model
        self.target = deepcopy(model).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.discount, self.target_tau, self.steps = float(discount), float(target_tau), 0

    def train_step(self, batch: dict[str, torch.Tensor]) -> VisualTDMPC2Loss:
        device = next(self.model.parameters()).device
        rgb, depth, valid, proprio = (batch[name].to(device) for name in ("rgb", "depth", "depth_valid", "proprio"))
        action = batch["action"][:, 0].to(device).clamp(-1, 1)
        reward_target = batch["reward"][:, 0].to(device)
        continuation = batch["continuation"][:, 0].to(device)
        collision = batch["collision"][:, 0].to(device)
        cpa = batch.get("cpa_risk", batch["collision"])[:, 0].to(device)
        risk_target = torch.maximum(collision, cpa).clamp(0, 1)
        clearance_target = batch["minimum_clearance"][:, 0].to(device).clamp(0, 120)
        goal_now, goal_next = proprio[:, 0, :3] * 120.0, proprio[:, 1, :3] * 120.0
        progress_target = torch.linalg.vector_norm(goal_now, dim=-1) - torch.linalg.vector_norm(goal_next, dim=-1)
        state_delta_target = goal_now - goal_next
        latent = self.model.encode(rgb[:, 0], depth[:, 0], valid[:, 0], proprio[:, 0])
        predicted_next = self.model.transition(latent, action)
        joined = torch.cat([latent, action], dim=-1)
        heads = self.model.predict_heads(latent, action)
        with torch.no_grad():
            target_next = self.target.encode(rgb[:, 1], depth[:, 1], valid[:, 1], proprio[:, 1])
            next_action = self.target.policy(target_next)
            target_joined = torch.cat([target_next, next_action], dim=-1)
            target_q = reward_target + self.discount * continuation * torch.minimum(
                self.target.q1(target_joined).squeeze(-1), self.target.q2(target_joined).squeeze(-1)
            )
        consistency_loss = nn.functional.smooth_l1_loss(predicted_next, target_next)
        reward_loss = nn.functional.smooth_l1_loss(heads["reward"], reward_target)
        q_loss = 0.5 * (
            nn.functional.mse_loss(self.model.q1(joined).squeeze(-1), target_q)
            + nn.functional.mse_loss(self.model.q2(joined).squeeze(-1), target_q)
        )
        risk_loss = nn.functional.binary_cross_entropy_with_logits(self.model.risk(joined).squeeze(-1), risk_target)
        clearance_loss = nn.functional.smooth_l1_loss(heads["clearance"] / 120.0, clearance_target / 120.0)
        progress_loss = nn.functional.smooth_l1_loss(heads["progress"], progress_target)
        continuation_loss = nn.functional.binary_cross_entropy_with_logits(self.model.continuation(joined).squeeze(-1), continuation)
        state_loss = nn.functional.smooth_l1_loss(self.model.state_delta(joined), state_delta_target)
        uncertainty_target = (predicted_next.detach() - target_next).square().mean(-1).clamp(0, 1)
        uncertainty_loss = nn.functional.smooth_l1_loss(heads["uncertainty"], uncertainty_target)
        total = consistency_loss + reward_loss + q_loss + risk_loss + clearance_loss + progress_loss + continuation_loss + 0.25 * state_loss + 0.1 * uncertainty_loss
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.model.parameters()):
                target.lerp_(online, self.target_tau)
        self.steps += 1
        return VisualTDMPC2Loss(*(float(value.detach()) for value in (total, consistency_loss, reward_loss, q_loss, risk_loss, clearance_loss, progress_loss)))
