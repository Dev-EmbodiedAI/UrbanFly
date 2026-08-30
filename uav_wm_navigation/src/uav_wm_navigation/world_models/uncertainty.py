from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module, inputs: tuple[torch.Tensor, ...], samples: int = 5,
    calibration: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    if samples < 2:
        raise ValueError("MC dropout requires at least two samples")
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    # Activate dropout only. This avoids accidentally enabling JEPA context
    # masking or RSSM posterior sampling during MC Dropout inference.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    try:
        batch_size = int(inputs[0].shape[0])
        repeated_inputs = tuple(
            value.repeat((samples,) + (1,) * (value.ndim - 1)) for value in inputs
        )
        batched_output = model(*repeated_inputs)
        outputs = [
            {
                name: value.reshape(samples, batch_size, *value.shape[1:])[sample_index]
                for name, value in batched_output.items()
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == samples * batch_size
            }
            for sample_index in range(samples)
        ]
        calibration = calibration or {}
        collision_temperature = max(float(calibration.get("collision_temperature", 1.0)), 0.05)
        failure_temperature = max(float(calibration.get("failure_temperature", 1.0)), 0.05)
        collision = torch.stack([torch.sigmoid(item["collision_logits"] / collision_temperature) for item in outputs])
        failure = torch.stack([torch.sigmoid(item["failure_logits"] / failure_temperature) for item in outputs])
        return {
            "collision_probability": collision.mean(0),
            "failure_probability": failure.mean(0),
            "minimum_clearance": torch.stack([item["minimum_clearance"] for item in outputs]).mean(0),
            "goal_progress": torch.stack([item["goal_progress"] for item in outputs]).mean(0),
            "uncertainty": collision.var(0, unbiased=False) + failure.var(0, unbiased=False),
        }
    finally:
        for module, training in training_states.items():
            module.train(training)
