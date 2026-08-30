from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from uav_wm_navigation.control.reranker import CandidateRerankerV3
from uav_wm_navigation.types import CandidatePrediction, CandidateTrajectory, RiskPrediction, TimestampedSensorFrame
from .encoders import DepthFrameEncoder
from .tdmpc2_candidate import TDMPC2CandidateAssistant, candidate_actions_body_flu
from .tdmpc2_visual import observation_visual_tensors


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)
        self.scale = float(alpha) / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.scale * self.b(self.a(value))


def enable_lora_last_blocks(encoder: nn.Module, blocks: int = 4, rank: int = 8) -> int:
    sequence = getattr(encoder, "blocks", None)
    if sequence is None or len(sequence) < blocks:
        raise ValueError("official encoder does not expose enough transformer blocks for LoRA")
    replaced = 0
    for block in sequence[-blocks:]:
        for parent in block.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in {"qkv", "proj"}:
                    setattr(parent, name, LoRALinear(child, rank=rank))
                    replaced += 1
    if replaced == 0:
        raise ValueError("no qkv/proj linear layers found in the selected V-JEPA blocks")
    return replaced


def unfreeze_last_blocks(encoder: nn.Module, blocks: int = 4) -> int:
    sequence = getattr(encoder, "blocks", None)
    if sequence is None or len(sequence) < blocks:
        raise ValueError("official encoder does not expose enough transformer blocks")
    encoder.requires_grad_(False)
    for block in sequence[-blocks:]:
        block.requires_grad_(True)
    return sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)


class OfficialVJEPA21Encoder(nn.Module):
    """Strict adapter around Meta's official V-JEPA 2.1 ViT-L backbone.

    There is intentionally no random or lightweight fallback. Formal runs must
    provide the official checkpoint and retain its path/hash in provenance.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        repository: str = "facebookresearch/vjepa2",
        model_name: str = "vjepa2_1_vit_large_384",
        source: str = "github",
        history_frames: int = 16,
        train_mode: str = "frozen",
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        loaded = torch.hub.load(
            repository, model_name, source=source, pretrained=False,
            num_frames=int(history_frames), trust_repo=True,
        )
        self.encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("ema_encoder", payload.get("target_encoder", payload.get("encoder", payload)))
        cleaned = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
        }
        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        if len(unexpected) > 8 or len(missing) > 8:
            raise ValueError(f"official checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
        self.checkpoint = str(path)
        self.model_name = model_name
        self.train_mode = train_mode
        self.encoder.requires_grad_(False)
        if train_mode == "lora":
            self.lora_layers = enable_lora_last_blocks(self.encoder, blocks=4, rank=lora_rank)
        elif train_mode in {"frozen", "encoder_frozen"}:
            self.lora_layers = 0
            self.train_mode = "frozen"
        elif train_mode == "partial_unfreeze":
            self.lora_layers = 0
            self.partial_trainable_parameters = unfreeze_last_blocks(self.encoder, blocks=4)
        elif train_mode == "full_finetune":
            self.encoder.requires_grad_(True)
            self.lora_layers = 0
        else:
            raise ValueError("train_mode must be frozen, lora, partial_unfreeze, or full_finetune")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1), persistent=False)

    def forward(self, video_rgb: torch.Tensor) -> torch.Tensor:
        if video_rgb.ndim != 5 or video_rgb.shape[2] != 3:
            raise ValueError("video_rgb must have shape [B,T,3,H,W]")
        batch, frames = video_rgb.shape[:2]
        resized = torch.nn.functional.interpolate(
            video_rgb.reshape(batch * frames, 3, *video_rgb.shape[-2:]),
            size=(384, 384), mode="bilinear", align_corners=False,
        ).reshape(batch, frames, 3, 384, 384).permute(0, 2, 1, 3, 4)
        video = (resized - self.mean) / self.std
        context = torch.no_grad() if self.train_mode == "frozen" else torch.enable_grad()
        with context:
            output = self.encoder(video)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        if output.ndim < 2:
            raise ValueError("official V-JEPA encoder returned an invalid tensor")
        return output.reshape(batch, -1, output.shape[-1]).mean(dim=1)


class UAVActionConditionedJEPAPredictor(nn.Module):
    """UrbanFly action predictor; does not reuse the robot-arm AC predictor."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        *,
        proprio_dim: int = 16,
        latent_dim: int = 256,
        max_horizon: int = 25,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.depth = DepthFrameEncoder(96)
        self.proprio = nn.GRU(proprio_dim, 96, batch_first=True)
        self.context = nn.Sequential(nn.Linear(encoder_dim + 192, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU())
        self.action = nn.Linear(4, latent_dim)
        self.position = nn.Parameter(torch.zeros(1, max_horizon + 1, latent_dim))
        layer = nn.TransformerEncoderLayer(latent_dim, 8, latent_dim * 4, 0.1, batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(layer, 4)
        def head() -> nn.Sequential:
            return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, 1))
        self.collision, self.failure, self.clearance, self.progress = head(), head(), head(), head()
        self.cpa, self.reward, self.value, self.uncertainty = head(), head(), head(), head()
        self.state_delta = nn.Linear(latent_dim, 3)
        self.embedding_prediction = nn.Linear(latent_dim, encoder_dim)

    def forward(self, video_rgb: torch.Tensor, depth: torch.Tensor, proprio: torch.Tensor, action_sequences: torch.Tensor) -> dict[str, torch.Tensor]:
        if action_sequences.ndim != 4 or action_sequences.shape[-1] != 4:
            raise ValueError("action_sequences must have shape [B,N,H,4]")
        batch, candidates, horizon, _ = action_sequences.shape
        if horizon + 1 > self.position.shape[1]:
            raise ValueError("candidate horizon exceeds the configured JEPA maximum")
        visual = self.encoder(video_rgb)
        depth_feature = self.depth(depth[:, -1])
        _, state = self.proprio(proprio)
        context = self.context(torch.cat([visual, depth_feature, state[-1]], dim=-1))
        context = context[:, None, None].expand(-1, candidates, 1, -1)
        actions = self.action(action_sequences)
        tokens = torch.cat([context, actions], dim=2).reshape(batch * candidates, horizon + 1, -1)
        tokens = tokens + self.position[:, :horizon + 1]
        causal = torch.triu(torch.ones(horizon + 1, horizon + 1, dtype=torch.bool, device=tokens.device), diagonal=1)
        latent = self.predictor(tokens, mask=causal)[:, 1:].reshape(batch, candidates, horizon, -1)
        pooled = latent.mean(dim=2)
        state_delta = self.state_delta(latent)
        positions = torch.cumsum(state_delta, dim=2)
        indices = [min(seconds * 5 - 1, horizon - 1) for seconds in (1, 2, 3)]
        return {
            "collision_logits": self.collision(pooled).squeeze(-1),
            "failure_logits": self.failure(pooled).squeeze(-1),
            "minimum_clearance": 120.0 * torch.sigmoid(self.clearance(pooled).squeeze(-1)),
            "goal_progress": self.progress(pooled).squeeze(-1),
            "cpa_risk": torch.sigmoid(self.cpa(pooled).squeeze(-1)),
            "predicted_reward": self.reward(pooled).squeeze(-1),
            "terminal_value": self.value(pooled).squeeze(-1),
            "uncertainty": torch.sigmoid(self.uncertainty(pooled).squeeze(-1)),
            "latent_states": latent,
            "predicted_state_1s_2s_3s": torch.stack([positions[:, :, index] for index in indices], dim=2),
            "predicted_future_embedding": self.embedding_prediction(pooled),
        }


def vjepa_uav_training_loss(
    model: UAVActionConditionedJEPAPredictor,
    batch: dict[str, torch.Tensor],
    *,
    context_frames: int = 15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Action-conditioned JEPA objective on one factual future per context."""

    device = next(model.parameters()).device
    rgb, depth, proprio, actions = (batch[name].to(device) for name in ("rgb", "depth", "proprio", "action"))
    if rgb.shape[1] < context_frames + 15:
        raise ValueError("V-JEPA training requires at least 15 context + 15 future frames")
    context_rgb = rgb[:, :context_frames]
    future_rgb = rgb[:, context_frames:context_frames + 15]
    action_sequence = actions[:, context_frames - 1:context_frames - 1 + 15, None].transpose(1, 2)
    output = model(context_rgb, depth[:, :context_frames], proprio[:, :context_frames], action_sequence)
    with torch.no_grad(): target_embedding = model.encoder(future_rgb)
    embedding = 1.0 - nn.functional.cosine_similarity(output["predicted_future_embedding"][:, 0], target_embedding, dim=-1).mean()
    future_slice = slice(context_frames, context_frames + 15)
    collision_target = batch["collision"][:, future_slice].to(device).max(1).values
    cpa_target = batch.get("cpa_risk", batch["collision"])[:, future_slice].to(device).max(1).values
    continuation = batch["continuation"][:, future_slice].to(device)
    failure_target = 1.0 - continuation.min(1).values
    clearance_target = batch["minimum_clearance"][:, future_slice].to(device).min(1).values.clamp(0, 120)
    goal = proprio[..., :3] * 120.0
    progress_target = torch.linalg.vector_norm(goal[:, context_frames - 1], dim=-1) - torch.linalg.vector_norm(goal[:, context_frames + 14], dim=-1)
    reward_target = batch["reward"][:, future_slice].to(device).sum(1)
    collision = nn.functional.binary_cross_entropy_with_logits(output["collision_logits"][:, 0], collision_target)
    failure = nn.functional.binary_cross_entropy_with_logits(output["failure_logits"][:, 0], failure_target)
    clearance = nn.functional.smooth_l1_loss(output["minimum_clearance"][:, 0] / 120, clearance_target / 120)
    progress = nn.functional.smooth_l1_loss(output["goal_progress"][:, 0], progress_target)
    cpa = nn.functional.binary_cross_entropy(output["cpa_risk"][:, 0].clamp(1e-5, 1-1e-5), cpa_target)
    reward = nn.functional.smooth_l1_loss(output["predicted_reward"][:, 0], reward_target)
    future_offsets = goal[:, context_frames - 1, None] - goal[:, context_frames:context_frames + 15]
    indices = [4, 9, 14]
    state = nn.functional.smooth_l1_loss(output["predicted_state_1s_2s_3s"][:, 0], future_offsets[:, indices])
    total = embedding + collision + failure + clearance + progress + cpa + reward + 0.25 * state
    return total, {"embedding": embedding, "collision": collision, "failure": failure, "clearance": clearance, "progress": progress, "cpa": cpa, "reward": reward, "state": state}


class VJEPA21CandidateAssistant:
    """Checkpoint-backed official V-JEPA 2.1 encoder plus UAV predictor."""

    def __init__(self, checkpoint: str | Path, *, official_checkpoint: str | Path | None = None, device: str | None = None, horizon: int = 15) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(Path(checkpoint).resolve(), map_location="cpu", weights_only=False)
        if payload.get("schema") != "urbanfly-world-model-v3" or payload.get("family") != "vjepa2_1_uav" or int(payload.get("training_steps", 0)) <= 0:
            raise ValueError("checkpoint lacks trained V-JEPA 2.1 UAV provenance")
        official = Path(official_checkpoint or payload.get("official_checkpoint", "")).expanduser().resolve()
        if not official.is_file(): raise FileNotFoundError("official V-JEPA 2.1 checkpoint is required")
        mode = str(payload.get("train_mode", "frozen"))
        encoder = OfficialVJEPA21Encoder(official, train_mode=mode, history_frames=15)
        self.model = UAVActionConditionedJEPAPredictor(encoder, int(payload.get("encoder_dim", 1024)), latent_dim=int(payload.get("latent_dim", 256))).to(self.device)
        missing, unexpected = self.model.load_state_dict(payload["model"], strict=False)
        non_encoder_missing = [name for name in missing if not name.startswith("encoder.")]
        if unexpected or non_encoder_missing: raise ValueError(f"V-JEPA UAV checkpoint mismatch: missing={non_encoder_missing}, unexpected={unexpected}")
        self.model.eval(); self.checkpoint_path = Path(checkpoint).resolve(); self.horizon = int(horizon); self.frames: deque[tuple[torch.Tensor,...]] = deque(maxlen=15); self.previous_action=np.zeros(4,np.float32); self.step_id=0; self.episode_id="vjepa2-1-uav"

    def reset(self) -> None:
        self.frames.clear(); self.previous_action.fill(0); self.step_id=0

    def rank(self, frame: TimestampedSensorFrame, candidates: tuple[CandidateTrajectory,...], goal_nwu: np.ndarray, reranker: CandidateRerankerV3, timeout_ms: float, max_risk: float):
        observation = TDMPC2CandidateAssistant._observation(self, frame.sensor, frame.state, goal_nwu)
        tensors = observation_visual_tensors(observation); self.frames.append(tuple(item.squeeze(0) for item in tensors))
        while len(self.frames)<15: self.frames.appendleft(tuple(item.clone() for item in self.frames[0]))
        rgb=torch.stack([item[0] for item in self.frames])[None].to(self.device); depth=torch.stack([item[1] for item in self.frames])[None].to(self.device); proprio=torch.stack([item[3] for item in self.frames])[None].to(self.device)
        sequences=np.stack([candidate_actions_body_flu(candidate,frame.state,horizon_steps=self.horizon) for candidate in candidates]); started=time.perf_counter()
        with torch.inference_mode(): output=self.model(rgb,depth,proprio,torch.from_numpy(sequences[None]).to(self.device))
        latency=(time.perf_counter()-started)*1000; predictions=[]
        for index in range(len(candidates)):
            predictions.append(CandidatePrediction(goal_progress=float(output["goal_progress"][0,index]),collision_probability=float(torch.sigmoid(output["collision_logits"][0,index])),minimum_clearance=float(output["minimum_clearance"][0,index]),cpa_risk=float(output["cpa_risk"][0,index]),terminal_value=float(output["terminal_value"][0,index]),epistemic_uncertainty=float(output["uncertainty"][0,index]),failure_probability=float(torch.sigmoid(output["failure_logits"][0,index])),predicted_state_1s_2s_3s=output["predicted_state_1s_2s_3s"][0,index].cpu().numpy()))
        decision=reranker.rank(list(candidates),predictions,latency_ms=latency,timeout_ms=timeout_ms,max_risk=max_risk); selected=decision.selected_index if decision.selected_index>=0 else int(np.argmin([item.yopo_cost for item in candidates])); self.previous_action=sequences[selected,0].copy(); self.step_id+=1
        legacy=[RiskPrediction(item.collision_probability,item.minimum_clearance,item.goal_progress,item.failure_probability,item.epistemic_uncertainty,item.predicted_state_1s_2s_3s) for item in predictions]
        return decision,legacy,latency
