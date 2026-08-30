from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np

from uav_wm_navigation.control.reranker import CandidateRerankerV3
from uav_wm_navigation.types import CandidateTrajectory, RiskPrediction, TimestampedSensorFrame

from .tdmpc2_candidate import VisualTDMPC2CandidateAssistant
from .dreamer_rssm_v3 import DreamerRSSMV3CandidateAssistant
from .vjepa2_official import VJEPA21CandidateAssistant


class V3CandidateWorldModelRuntime:
    """Formal RGB-D candidate-only runtime for the shared YOPO comparison.

    This class deliberately has no trajectory generator.  It can only score
    the exact candidates supplied by the common YOPO planner.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        family: str,
        weights: dict[str, float] | None = None,
        timeout_ms: float = 150.0,
        max_risk: float = 0.75,
        device: str | None = None,
        horizon_steps: int = 15,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.family = str(family).lower()
        self.timeout_ms = float(timeout_ms)
        self.max_risk = float(max_risk)
        if self.family == "tdmpc2_visual":
            self.assistant = VisualTDMPC2CandidateAssistant(
                self.checkpoint_path,
                horizon_steps=horizon_steps,
                device=device,
            )
        elif self.family == "dreamer_rssm_v3":
            self.assistant = DreamerRSSMV3CandidateAssistant(
                self.checkpoint_path, device=device, horizon=horizon_steps,
            )
        elif self.family == "vjepa2_1_uav":
            self.assistant = VJEPA21CandidateAssistant(
                self.checkpoint_path, device=device, horizon=horizon_steps,
            )
        else:
            raise ValueError(
                f"formal v3 runtime for {self.family!r} is not registered; "
                "refusing to substitute a different model"
            )
        self.reranker = CandidateRerankerV3(weights)
        self._episode_id = ""
        self.reset()

    def reset(self) -> None:
        self._episode_id = f"urbanfly-v3-eval-{uuid.uuid4().hex[:12]}"
        if self.family == "tdmpc2_visual":
            self.assistant.reset(self._episode_id)
        else:
            self.assistant.reset()

    def rank(
        self,
        frame: TimestampedSensorFrame,
        candidates: tuple[CandidateTrajectory, ...],
        goal_nwu: np.ndarray,
    ):
        if self.family in {"dreamer_rssm_v3", "vjepa2_1_uav"}:
            return self.assistant.rank(
                frame, candidates, goal_nwu, self.reranker,
                self.timeout_ms, self.max_risk,
            )
        predictions, _, latency_ms = self.assistant.predict_candidates(candidates, frame.sensor, frame.state, goal_nwu)
        decision = self.reranker.rank(
            list(candidates), predictions, latency_ms=latency_ms,
            timeout_ms=self.timeout_ms, max_risk=self.max_risk,
        )
        legacy = [
            RiskPrediction(
                collision_probability=item.collision_probability,
                minimum_clearance=item.minimum_clearance,
                goal_progress=item.goal_progress,
                failure_probability=item.failure_probability,
                uncertainty=item.epistemic_uncertainty,
                latent_states=item.predicted_state_1s_2s_3s,
            )
            for item in predictions
        ]
        return decision, legacy, latency_ms
