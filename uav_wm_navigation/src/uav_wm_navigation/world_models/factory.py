from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from .dreamerv3_world_model import DreamerV3WorldModel
from .gru_world_model import GRUWorldModel
from .jepa_world_model import ActionConditionedJEPAWorldModel
from .occflow_world_model import OccFlowWorldModel
from .transformer_world_model import TransformerWorldModel


def build_world_model(config: Mapping[str, Any]) -> nn.Module:
    common = dict(
        state_dim=int(config["state_dim"]),
        trajectory_dim=int(config["trajectory_dim"]),
        latent_dim=int(config["latent_dim"]),
        dropout=float(config["dropout"]),
    )
    family = str(config["model"]).lower()
    if family == "gru":
        return GRUWorldModel(**common)
    if family == "transformer":
        return TransformerWorldModel(
            **common,
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            max_horizon=int(config["max_horizon"]),
        )
    if family == "dreamerv3":
        model = DreamerV3WorldModel(
            **common,
            deterministic_dim=int(config.get("deterministic_dim", 96)),
            stochastic_groups=int(config.get("stochastic_groups", 8)),
            stochastic_classes=int(config.get("stochastic_classes", 8)),
            free_nats=float(config.get("free_nats", 1.0)),
            kl_balance=float(config.get("kl_balance", 0.8)),
        )
        model.future_start_index = int(config.get("future_start_index", 1))
        return model
    if family == "jepa":
        return ActionConditionedJEPAWorldModel(
            **common,
            layers=int(config.get("layers", 2)),
            heads=int(config.get("heads", 4)),
            max_horizon=int(config.get("max_horizon", 64)),
            ema_decay=float(config.get("ema_decay", 0.996)),
            mask_ratio=float(config.get("mask_ratio", 0.5)),
            mask_patch_size=int(config.get("mask_patch_size", 12)),
            future_start_index=int(config.get("future_start_index", 1)),
        )
    if family == "occflow":
        return OccFlowWorldModel(
            **common,
            history=int(config.get("history", 4)),
            future_steps=int(config.get("future_steps", 5)),
            depth_max_m=float(config.get("depth_max_m", 20.0)),
            voxel_resolution_m=float(config.get("voxel_resolution_m", 0.5)),
            voxel_minimum_flu=tuple(config.get("voxel_minimum_flu", [-4.0, -8.0, -3.0])),
            voxel_maximum_flu=tuple(config.get("voxel_maximum_flu", [20.0, 8.0, 5.0])),
        )
    raise ValueError(f"unsupported world model family: {family}")
