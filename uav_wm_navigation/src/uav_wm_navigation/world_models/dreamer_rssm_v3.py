from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from uav_wm_navigation.control.reranker import CandidateRerankerV3
from uav_wm_navigation.types import CandidatePrediction, CandidateTrajectory, RiskPrediction, TimestampedSensorFrame

from .tdmpc2_candidate import candidate_actions_body_flu
from .tdmpc2_visual import RGBDEncoder, observation_visual_tensors


def _straight_through_categorical(logits: torch.Tensor, training: bool) -> torch.Tensor:
    probabilities = logits.softmax(-1)
    indices = torch.multinomial(probabilities.flatten(0, -2), 1).reshape(probabilities.shape[:-1]) if training else probabilities.argmax(-1)
    hard = nn.functional.one_hot(indices, probabilities.shape[-1]).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def _kl(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    q = q_logits.softmax(-1)
    return (q * (q_logits.log_softmax(-1) - p_logits.log_softmax(-1))).sum(-1).mean(-1)


class DreamerRSSMV3Network(nn.Module):
    """RGB-D categorical RSSM used only to imagine shared YOPO candidates."""

    def __init__(
        self, *, visual_dim: int = 384, deterministic_dim: int = 768,
        stochastic_groups: int = 32, stochastic_classes: int = 16,
        feature_dim: int = 512, action_dim: int = 4,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim); self.deterministic_dim = int(deterministic_dim)
        self.groups = int(stochastic_groups); self.classes = int(stochastic_classes)
        self.stochastic_dim = self.groups * self.classes
        self.feature_dim = int(feature_dim); self.action_dim = int(action_dim)
        self.visual = RGBDEncoder(self.visual_dim)
        self.proprio = nn.Sequential(nn.Linear(16, 192), nn.LayerNorm(192), nn.SiLU())
        self.observation = nn.Sequential(nn.Linear(self.visual_dim + 192, self.feature_dim), nn.LayerNorm(self.feature_dim), nn.SiLU())
        self.action = nn.Sequential(nn.Linear(action_dim, 192), nn.SiLU())
        self.recurrent = nn.GRUCell(self.stochastic_dim + 192, self.deterministic_dim)
        self.prior = nn.Sequential(nn.Linear(self.deterministic_dim, 1024), nn.SiLU(), nn.Linear(1024, self.stochastic_dim))
        self.posterior = nn.Sequential(nn.Linear(self.deterministic_dim + self.feature_dim, 1024), nn.SiLU(), nn.Linear(1024, self.stochastic_dim))
        self.feature = nn.Sequential(nn.Linear(self.deterministic_dim + self.stochastic_dim, self.feature_dim), nn.LayerNorm(self.feature_dim), nn.SiLU())
        joined = self.feature_dim + action_dim
        def head(output: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Linear(joined, 384), nn.SiLU(), nn.Linear(384, output))
        self.reward, self.risk, self.clearance, self.progress = head(), head(), head(), head()
        self.continuation, self.value, self.uncertainty, self.state_delta = head(), head(), head(), head(3)
        self.embedding_predictor = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.SiLU(), nn.Linear(self.feature_dim, self.feature_dim))

    def _reshape_logits(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.groups, self.classes)

    def encode_frames(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        batch, length = rgb.shape[:2]
        visual = self.visual(
            rgb.reshape(batch * length, *rgb.shape[2:]),
            depth.reshape(batch * length, *depth.shape[2:]),
            valid.reshape(batch * length, *valid.shape[2:]),
        ).reshape(batch, length, -1)
        return self.observation(torch.cat([visual, self.proprio(proprio)], dim=-1))

    def observe_sequence(
        self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor,
        proprio: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encode_frames(rgb, depth, valid, proprio)
        batch, length = embeddings.shape[:2]
        deterministic = embeddings.new_zeros(batch, self.deterministic_dim)
        stochastic = embeddings.new_zeros(batch, self.stochastic_dim)
        features, kl_terms = [], []
        for step in range(length):
            previous = actions.new_zeros(batch, self.action_dim) if step == 0 else actions[:, step - 1]
            deterministic = self.recurrent(torch.cat([stochastic, self.action(previous)], -1), deterministic)
            prior = self._reshape_logits(self.prior(deterministic))
            posterior = self._reshape_logits(self.posterior(torch.cat([deterministic, embeddings[:, step]], -1)))
            stochastic = _straight_through_categorical(posterior, self.training).flatten(-2)
            features.append(self.feature(torch.cat([deterministic, stochastic], -1)))
            dynamics_kl = _kl(posterior.detach(), prior)
            representation_kl = _kl(posterior, prior.detach())
            kl_terms.append(0.8 * dynamics_kl + 0.2 * representation_kl)
        return deterministic, stochastic, torch.stack(features, 1), torch.stack(kl_terms, 1)

    def heads(self, feature: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        joined = torch.cat([feature, action], -1)
        return {
            "reward": self.reward(joined).squeeze(-1),
            "risk": torch.sigmoid(self.risk(joined).squeeze(-1)),
            "risk_logits": self.risk(joined).squeeze(-1),
            "clearance": 120.0 * torch.sigmoid(self.clearance(joined).squeeze(-1)),
            "progress": self.progress(joined).squeeze(-1),
            "continuation": torch.sigmoid(self.continuation(joined).squeeze(-1)),
            "continuation_logits": self.continuation(joined).squeeze(-1),
            "value": self.value(joined).squeeze(-1),
            "uncertainty": torch.sigmoid(self.uncertainty(joined).squeeze(-1)),
            "state_delta": self.state_delta(joined),
        }

    def imagine(self, deterministic: torch.Tensor, stochastic: torch.Tensor, action_sequences: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, candidates, horizon, _ = action_sequences.shape
        deterministic = deterministic[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        stochastic = stochastic[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        actions = action_sequences.reshape(batch * candidates, horizon, -1)
        output: dict[str, list[torch.Tensor]] = {name: [] for name in ("reward", "risk", "clearance", "progress", "continuation", "value", "uncertainty")}
        position = action_sequences.new_zeros(batch * candidates, 3); positions = []
        for step in range(horizon):
            action = actions[:, step]
            deterministic = self.recurrent(torch.cat([stochastic, self.action(action)], -1), deterministic)
            prior = self._reshape_logits(self.prior(deterministic))
            stochastic = _straight_through_categorical(prior, False).flatten(-2)
            feature = self.feature(torch.cat([deterministic, stochastic], -1))
            heads = self.heads(feature, action)
            for name in output: output[name].append(heads[name])
            position = position + heads["state_delta"]; positions.append(position)
        result = {name: torch.stack(values, 1).reshape(batch, candidates, horizon) for name, values in output.items()}
        indices = [min(seconds * 5 - 1, horizon - 1) for seconds in (1, 2, 3)]
        result["predicted_state_1s_2s_3s"] = torch.stack([positions[index] for index in indices], 1).reshape(batch, candidates, 3, 3)
        return result


@dataclass(frozen=True, slots=True)
class DreamerRSSMLoss:
    total: float
    representation: float
    kl: float
    reward: float
    risk: float
    clearance: float
    progress: float


class DreamerRSSMV3Trainer:
    def __init__(self, model: DreamerRSSMV3Network, learning_rate: float = 2e-4) -> None:
        self.model = model; self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5); self.steps = 0

    def train_step(self, batch: dict[str, torch.Tensor]) -> DreamerRSSMLoss:
        device = next(self.model.parameters()).device
        rgb, depth, valid, proprio, actions = (batch[name].to(device) for name in ("rgb", "depth", "depth_valid", "proprio", "action"))
        _, _, features, kl = self.model.observe_sequence(rgb, depth, valid, proprio, actions)
        current_feature, current_action = features[:, :-1], actions[:, :-1]
        heads = self.model.heads(current_feature, current_action)
        with torch.no_grad(): target_embedding = self.model.encode_frames(rgb, depth, valid, proprio)[:, 1:]
        representation = 1.0 - nn.functional.cosine_similarity(self.model.embedding_predictor(current_feature), target_embedding, dim=-1).mean()
        reward = nn.functional.smooth_l1_loss(heads["reward"], batch["reward"][:, :-1].to(device))
        collision = batch["collision"][:, :-1].to(device); cpa = batch.get("cpa_risk", batch["collision"])[:, :-1].to(device)
        risk_target = torch.maximum(collision, cpa).clamp(0, 1)
        risk = nn.functional.binary_cross_entropy_with_logits(self.model.risk(torch.cat([current_feature, current_action], -1)).squeeze(-1), risk_target)
        clearance = nn.functional.smooth_l1_loss(heads["clearance"] / 120, batch["minimum_clearance"][:, :-1].to(device).clamp(0, 120) / 120)
        goal = proprio[..., :3] * 120; progress_target = torch.linalg.vector_norm(goal[:, :-1], dim=-1) - torch.linalg.vector_norm(goal[:, 1:], dim=-1)
        progress = nn.functional.smooth_l1_loss(heads["progress"], progress_target)
        continuation = nn.functional.binary_cross_entropy_with_logits(self.model.continuation(torch.cat([current_feature, current_action], -1)).squeeze(-1), batch["continuation"][:, :-1].to(device))
        kl_loss = kl.clamp_min(1.0).mean()
        total = representation + kl_loss + reward + risk + clearance + progress + continuation
        self.optimizer.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 100.0); self.optimizer.step(); self.steps += 1
        return DreamerRSSMLoss(*(float(item.detach()) for item in (total, representation, kl_loss, reward, risk, clearance, progress)))


class DreamerRSSMV3CandidateAssistant:
    def __init__(self, checkpoint: str | Path, *, device: str | None = None, history: int = 4, horizon: int = 15) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); payload = torch.load(Path(checkpoint).resolve(), map_location=self.device, weights_only=False)
        if payload.get("schema") != "urbanfly-world-model-v3" or payload.get("family") != "dreamer_rssm_v3" or int(payload.get("training_steps", 0)) <= 0: raise ValueError("checkpoint lacks trained Dreamer RSSM v3 provenance")
        self.model = DreamerRSSMV3Network(**payload.get("architecture", {})).to(self.device); self.model.load_state_dict(payload["model"]); self.model.eval()
        self.checkpoint_path = Path(checkpoint).resolve(); self.history = int(history); self.horizon = int(horizon); self.frames: deque[tuple[torch.Tensor, ...]] = deque(maxlen=self.history); self.actions: deque[np.ndarray] = deque(maxlen=self.history)

    def reset(self) -> None:
        self.frames.clear(); self.actions.clear()
        self.episode_id = "dreamer-rssm-v3"; self.step_id = 0
        self.previous_action = np.zeros(4, dtype=np.float32)

    def rank(self, frame: TimestampedSensorFrame, candidates: tuple[CandidateTrajectory, ...], goal_nwu: np.ndarray, reranker: CandidateRerankerV3, timeout_ms: float, max_risk: float):
        from uav_wm_navigation.world_models.tdmpc2_candidate import TDMPC2CandidateAssistant
        observation = TDMPC2CandidateAssistant._observation(self, frame.sensor, frame.state, goal_nwu)
        tensors = observation_visual_tensors(observation)
        self.frames.append(tuple(tensor.squeeze(0) for tensor in tensors)); self.actions.append(np.asarray(observation.previous_action, dtype=np.float32))
        while len(self.frames) < self.history: self.frames.appendleft(tuple(tensor.clone() for tensor in self.frames[0])); self.actions.appendleft(self.actions[0].copy())
        rgb = torch.stack([item[0] for item in self.frames])[None].to(self.device); depth = torch.stack([item[1] for item in self.frames])[None].to(self.device); valid = torch.stack([item[2] for item in self.frames])[None].to(self.device); proprio = torch.stack([item[3] for item in self.frames])[None].to(self.device); factual = torch.from_numpy(np.stack(self.actions)[None]).to(self.device)
        sequences = np.stack([candidate_actions_body_flu(candidate, frame.state, horizon_steps=self.horizon) for candidate in candidates])
        started = time.perf_counter()
        with torch.inference_mode():
            deterministic, stochastic, _, _ = self.model.observe_sequence(rgb, depth, valid, proprio, factual)
            output = self.model.imagine(deterministic, stochastic, torch.from_numpy(sequences[None]).to(self.device))
        latency = (time.perf_counter() - started) * 1000
        predictions = []
        for index in range(len(candidates)):
            predictions.append(CandidatePrediction(
                goal_progress=float(output["progress"][0, index].sum()), collision_probability=float(output["risk"][0, index].max()), minimum_clearance=float(output["clearance"][0, index].min()), cpa_risk=float(output["risk"][0, index].max()), terminal_value=float(output["value"][0, index, -1]), epistemic_uncertainty=float(output["uncertainty"][0, index].max()), failure_probability=float(1 - output["continuation"][0, index].min()), predicted_state_1s_2s_3s=output["predicted_state_1s_2s_3s"][0, index].cpu().numpy(),
            ))
        decision = reranker.rank(list(candidates), predictions, latency_ms=latency, timeout_ms=timeout_ms, max_risk=max_risk)
        selected = decision.selected_index if decision.selected_index >= 0 else int(np.argmin([candidate.yopo_cost for candidate in candidates]))
        self.previous_action = sequences[selected, 0].copy(); self.step_id += 1
        legacy = [RiskPrediction(item.collision_probability, item.minimum_clearance, item.goal_progress, item.failure_probability, item.epistemic_uncertainty, item.predicted_state_1s_2s_3s) for item in predictions]
        return decision, legacy, latency
