from __future__ import annotations

from typing import Mapping

import torch


REQUIRED_OUTPUTS = (
    "collision_logits", "failure_logits", "minimum_clearance", "goal_progress", "uncertainty"
)


def validate_world_model_output(output: Mapping[str, torch.Tensor], batch: int, candidates: int) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if name not in output]
    if missing:
        raise ValueError(f"world model output is missing {missing}")
    for name in REQUIRED_OUTPUTS:
        value = output[name]
        if value.shape != (batch, candidates):
            raise ValueError(f"{name} must have shape {(batch, candidates)}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
