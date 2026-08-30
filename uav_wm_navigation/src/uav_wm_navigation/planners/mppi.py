from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from uav_wm_navigation.world_models.base import WorldModelBase


@dataclass(frozen=True, slots=True)
class MPPICostWeights:
    goal: float = 1.0
    intermediate_goal: float = 0.15
    collision: float = 20.0
    smoothness: float = 0.1
    control: float = 0.01
    hard_collision: float = 1000.0
    hard_collision_probability: float = 0.8


@dataclass(slots=True)
class MPPIPlan:
    action_sequence: torch.Tensor
    first_action: torch.Tensor
    predicted_positions: torch.Tensor
    predicted_velocities: torch.Tensor
    predicted_collision_probability: torch.Tensor
    total_cost: float
    cost_components: dict[str, float]
    diagnostics: dict[str, Any]
    candidate_action_sequences: torch.Tensor | None = None
    candidate_positions: torch.Tensor | None = None
    candidate_costs: torch.Tensor | None = None
    predicted_latents: torch.Tensor | None = None


class MPPIPlanner:
    """Batched MPPI optimizer with warm start and finite-value fallback.

    All candidates are imagined in one world-model call per optimization
    iteration. The only Python loop is over the small iteration count; the
    recurrent world model itself may loop over horizon, never candidates.
    """

    def __init__(
        self,
        *,
        horizon: int = 15,
        dt: float = 0.1,
        num_samples: int = 256,
        num_iterations: int = 4,
        temperature: float = 1.0,
        noise_sigma: tuple[float, float, float, float] = (0.55, 0.55, 0.35, 0.35),
        action_min: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
        action_max: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        warm_start: bool = True,
        cost_weights: MPPICostWeights | None = None,
        save_debug: bool = False,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if horizon <= 0 or num_samples <= 0 or num_iterations <= 0 or dt <= 0.0:
            raise ValueError("horizon, samples, iterations and dt must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.num_samples = int(num_samples)
        self.num_iterations = int(num_iterations)
        self.temperature = float(temperature)
        self.warm_start = bool(warm_start)
        self.cost_weights = cost_weights or MPPICostWeights()
        self.save_debug = bool(save_debug)
        self.device = torch.device(device)
        self.noise_sigma = torch.tensor(noise_sigma, dtype=torch.float32, device=self.device)
        self.action_min = torch.tensor(action_min, dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(action_max, dtype=torch.float32, device=self.device)
        if not torch.all(self.action_min < self.action_max):
            raise ValueError("every action_min must be smaller than action_max")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self._mean = torch.zeros(self.horizon, 4, device=self.device)
        self._previous_action = torch.zeros(4, device=self.device)

    def reset(self) -> None:
        self._mean.zero_()
        self._previous_action.zero_()

    def _cost(
        self,
        rollout: dict[str, torch.Tensor],
        actions: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positions = rollout["position"]
        collision = rollout["collision_probability"].clamp(0.0, 1.0)
        goal_value = goal.reshape(1, 1, 3).to(positions)
        distance = torch.linalg.vector_norm(positions - goal_value, dim=-1)
        goal_cost = distance[:, -1]
        intermediate = distance.mean(dim=1)
        collision_cost = collision.sum(dim=1)
        hard_collision = (collision >= self.cost_weights.hard_collision_probability).any(dim=1).to(actions.dtype)
        previous = self._previous_action.reshape(1, 1, 4).expand(actions.shape[0], 1, 4)
        differences = torch.diff(torch.cat([previous, actions], dim=1), dim=1)
        smoothness = differences.square().sum(dim=(-1, -2))
        control = actions.square().sum(dim=(-1, -2))
        components = {
            "goal": goal_cost,
            "intermediate_goal": intermediate,
            "collision": collision_cost,
            "smoothness": smoothness,
            "control": control,
            "hard_collision": hard_collision,
        }
        weights = self.cost_weights
        total = (
            weights.goal * goal_cost
            + weights.intermediate_goal * intermediate
            + weights.collision * collision_cost
            + weights.smoothness * smoothness
            + weights.control * control
            + weights.hard_collision * hard_collision
        )
        return total, components

    @torch.inference_mode()
    def plan(
        self,
        world_model: WorldModelBase,
        current_latent: torch.Tensor,
        current_state: torch.Tensor,
        goal: torch.Tensor,
    ) -> MPPIPlan:
        started = time.perf_counter()
        if current_latent.ndim != 2 or current_latent.shape[0] != 1:
            raise ValueError("current_latent must have shape [1, latent_dim]")
        if goal.numel() != 3 or not torch.isfinite(goal).all():
            raise ValueError("goal must be a finite three-vector")
        if not torch.isfinite(current_latent).all() or not torch.isfinite(current_state).all():
            raise FloatingPointError("MPPI received a non-finite latent/state")
        mean = self._mean.clone()
        rollout_latency_ms = 0.0
        latest_actions = latest_costs = latest_rollout = latest_components = None
        for _ in range(self.num_iterations):
            noise = torch.randn(
                self.num_samples,
                self.horizon,
                4,
                generator=self.generator,
                device=self.device,
            ) * self.noise_sigma
            actions = torch.clamp(mean.unsqueeze(0) + noise, self.action_min, self.action_max)
            rollout_started = time.perf_counter()
            rollout = world_model.rollout(
                current_latent,
                current_state,
                actions,
                dt=self.dt,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            rollout_latency_ms += (time.perf_counter() - rollout_started) * 1000.0
            costs, components = self._cost(rollout, actions, goal)
            finite = torch.isfinite(costs)
            if not finite.any():
                raise FloatingPointError("MPPI produced no finite candidate")
            safe_costs = torch.where(finite, costs, torch.full_like(costs, torch.inf))
            minimum = safe_costs[finite].min()
            weights = torch.softmax(-(safe_costs - minimum) / self.temperature, dim=0)
            weights = torch.where(finite, weights, torch.zeros_like(weights))
            denominator = weights.sum().clamp_min(1e-8)
            mean = torch.einsum("n,nha->ha", weights / denominator, actions)
            mean = torch.clamp(mean, self.action_min, self.action_max)
            latest_actions, latest_costs = actions, safe_costs
            latest_rollout, latest_components = rollout, components
        assert latest_actions is not None and latest_costs is not None
        final_rollout_started = time.perf_counter()
        final_rollout = world_model.rollout(
            current_latent,
            current_state,
            mean.unsqueeze(0),
            dt=self.dt,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        rollout_latency_ms += (time.perf_counter() - final_rollout_started) * 1000.0
        final_cost, final_components = self._cost(final_rollout, mean.unsqueeze(0), goal)
        if not torch.isfinite(final_cost).all() or not torch.isfinite(mean).all():
            raise FloatingPointError("MPPI final trajectory is non-finite")
        first_action = mean[0].clone()
        self._previous_action = first_action
        if self.warm_start:
            self._mean = torch.cat([mean[1:], mean[-1:]], dim=0).detach()
        else:
            self._mean.zero_()
        optimization_latency_ms = (time.perf_counter() - started) * 1000.0
        best_index = int(torch.argmin(latest_costs).item())
        component_values = {name: float(value[0].item()) for name, value in final_components.items()}
        diagnostics = {
            "horizon": self.horizon,
            "num_samples": self.num_samples,
            "num_iterations": self.num_iterations,
            "rollout_latency_ms": rollout_latency_ms,
            "optimization_latency_ms": optimization_latency_ms,
            "best_sample_index": best_index,
            "warm_start": self.warm_start,
            "finite_candidate_fraction": float(torch.isfinite(latest_costs).float().mean().item()),
        }
        return MPPIPlan(
            action_sequence=mean.detach(),
            first_action=first_action.detach(),
            predicted_positions=final_rollout["position"][0].detach(),
            predicted_velocities=final_rollout["velocity"][0].detach(),
            predicted_collision_probability=final_rollout["collision_probability"][0].detach(),
            predicted_latents=final_rollout["latent"][0].detach(),
            total_cost=float(final_cost[0].item()),
            cost_components=component_values,
            diagnostics=diagnostics,
            candidate_action_sequences=latest_actions.detach() if self.save_debug else None,
            candidate_positions=latest_rollout["position"].detach() if self.save_debug else None,
            candidate_costs=latest_costs.detach() if self.save_debug else None,
        )
