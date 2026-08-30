from __future__ import annotations

import torch
from torch import nn


class WorldModelLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.weights = weights or {
            "collision": 1.0, "clearance": 1.0, "progress": 1.0,
            "failure": 1.0, "latent": 0.1, "dynamics": 1.0,
            "ranking": 0.5, "occupancy": 1.0, "flow": 0.2,
        }
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))

    def forward(self, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = target.get("label_valid_mask", torch.ones_like(target["collision"])).float()
        mask = mask * target.get("label_confidence", torch.ones_like(mask)).float()
        denominator = mask.sum().clamp(min=1.0)
        def masked_mean(values: torch.Tensor) -> torch.Tensor:
            return (values * mask).sum() / denominator
        collision = masked_mean(nn.functional.binary_cross_entropy_with_logits(
            prediction["collision_logits"], target["collision"].float(), pos_weight=self.positive_weight, reduction="none"
        ))
        failure = masked_mean(nn.functional.binary_cross_entropy_with_logits(
            prediction["failure_logits"], target["failure"].float(), pos_weight=self.positive_weight, reduction="none"
        ))
        clearance = masked_mean(nn.functional.huber_loss(
            prediction["minimum_clearance"], target["minimum_clearance"].float(), reduction="none"
        ))
        progress = masked_mean(nn.functional.huber_loss(
            prediction["goal_progress"], target["goal_progress"].float(), reduction="none"
        ))
        latent = prediction["latent_states"].new_zeros(())
        if "latent_states" in target:
            latent = nn.functional.mse_loss(prediction["latent_states"], target["latent_states"].float())
        dynamics = prediction.get("auxiliary_loss", prediction["latent_states"].new_zeros(()))
        predicted_score = (
            4.0 * torch.sigmoid(prediction["collision_logits"])
            + 2.0 * torch.sigmoid(prediction["failure_logits"])
            - prediction["minimum_clearance"] - prediction["goal_progress"]
        )
        target_score = (
            4.0 * target["collision"].float() + 2.0 * target["failure"].float()
            - target["minimum_clearance"].float() - target["goal_progress"].float()
        )
        predicted_delta = predicted_score[:, :, None] - predicted_score[:, None, :]
        target_preference = (target_score[:, :, None] > target_score[:, None, :]).float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        pair_mask = pair_mask * (torch.abs(target_score[:, :, None] - target_score[:, None, :]) > 1e-5)
        ranking = (
            nn.functional.binary_cross_entropy_with_logits(predicted_delta, target_preference, reduction="none")
            * pair_mask
        ).sum() / pair_mask.sum().clamp_min(1.0)
        occupancy = prediction["latent_states"].new_zeros(())
        if "occupancy_logits" in prediction and "occupancy" in target:
            occupancy_target = target["occupancy"].float()
            focal_bce = nn.functional.binary_cross_entropy_with_logits(
                prediction["occupancy_logits"], occupancy_target, reduction="none"
            )
            probability = prediction["occupancy_logits"].sigmoid()
            focal_bce = (focal_bce * (probability - occupancy_target).abs().square()).mean()
            intersection = (probability * occupancy_target).sum()
            dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + occupancy_target.sum() + 1.0)
            occupancy = focal_bce + dice
        flow = prediction["latent_states"].new_zeros(())
        if "flow" in prediction and "flow" in target:
            flow_mask = target.get("occupancy", torch.ones_like(prediction["flow"])[:, :, :1]).amax(dim=2, keepdim=True)
            flow = (nn.functional.smooth_l1_loss(prediction["flow"], target["flow"].float(), reduction="none") * flow_mask).sum() / flow_mask.sum().clamp_min(1.0)
        parts = {
            "collision": collision, "clearance": clearance, "progress": progress,
            "failure": failure, "latent": latent, "dynamics": dynamics,
            "ranking": ranking, "occupancy": occupancy, "flow": flow,
        }
        total = sum(self.weights.get(name, 0.0) * value for name, value in parts.items())
        return total, parts
