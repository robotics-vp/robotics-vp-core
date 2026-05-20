"""Bounded neural seams for Phase 3.4 Embodiment / Actuation.

These are small, CPU-runnable modules intended to make the learned path real
without claiming benchmark promotion. Promotion remains governed by
``promotion.py`` and receipt/training evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .common import clip01, mapping, safe_float


@dataclass(frozen=True)
class SeamForwardResult:
    seam_id: str
    outputs: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seam_id": self.seam_id,
            "outputs": mapping(self.outputs),
            "metadata": mapping(self.metadata),
        }


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class _BaseEmbodimentSeam(nn.Module):
    seam_id: str
    input_dim: int

    def param_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def describe(self) -> dict[str, Any]:
        return {
            "seam_id": self.seam_id,
            "input_dim": self.input_dim,
            "param_count": self.param_count(),
            "promotion_required": True,
            "default_posture": "disabled_or_auto",
        }


class LocalContactDynamicsSeam(_BaseEmbodimentSeam):
    """Predict short-horizon contact risk/confidence from contact features."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.seam_id = "local_contact_dynamics"
        self.input_dim = input_dim
        self.net = _mlp(input_dim, hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(features.float())
        return {
            "next_contact_probability": torch.sigmoid(logits[..., 0]),
            "transition_risk": torch.sigmoid(logits[..., 1]),
            "forecast_confidence": torch.sigmoid(logits[..., 2]),
        }


class InverseRetargetingSeam(_BaseEmbodimentSeam):
    """Map source action/context features into target action-space proposals."""

    def __init__(self, input_dim: int = 32, target_action_dim: int = 12, hidden_dim: int = 64) -> None:
        super().__init__()
        self.seam_id = "inverse_retargeting"
        self.input_dim = input_dim
        self.target_action_dim = target_action_dim
        self.net = _mlp(input_dim, hidden_dim, target_action_dim + 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(features.float())
        return {
            "target_action": torch.tanh(raw[..., : self.target_action_dim]),
            "readiness_score": torch.sigmoid(raw[..., self.target_action_dim]),
        }

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["target_action_dim"] = self.target_action_dim
        return payload


class ActionProposalSeam(_BaseEmbodimentSeam):
    """Produce bounded action chunks and feasibility from embodiment context."""

    def __init__(self, input_dim: int = 32, action_dim: int = 12, chunk_len: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.seam_id = "action_proposal"
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.net = _mlp(input_dim, hidden_dim, action_dim * chunk_len + 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(features.float())
        action_values = torch.tanh(raw[..., : self.action_dim * self.chunk_len])
        return {
            "action_chunk": action_values.reshape(*features.shape[:-1], self.chunk_len, self.action_dim),
            "feasibility_score": torch.sigmoid(raw[..., -1]),
        }

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update({"action_dim": self.action_dim, "chunk_len": self.chunk_len})
        return payload


class DriftCalibrationSeam(_BaseEmbodimentSeam):
    """Estimate drift and calibration priority from receipt/state features."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 32) -> None:
        super().__init__()
        self.seam_id = "drift_calibration"
        self.input_dim = input_dim
        self.net = _mlp(input_dim, hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(features.float())
        return {
            "drift_score": torch.sigmoid(raw[..., 0]),
            "calibration_priority": torch.sigmoid(raw[..., 1]),
            "safety_margin_estimate": torch.sigmoid(raw[..., 2]),
        }


def encode_state_features(state: Any, input_dim: int = 32) -> torch.Tensor:
    """Encode an EmbodimentActuationWorldState-like object into fixed features."""
    values = [
        safe_float(getattr(getattr(state, "contact_state", None), "contact_event_count", 0.0)),
        safe_float(getattr(getattr(state, "contact_state", None), "contact_pair_count", 0.0)),
        safe_float(getattr(getattr(state, "contact_state", None), "impossible_contact_count", 0.0)),
        clip01(getattr(getattr(state, "contact_state", None), "contact_confidence_mean", 0.0)),
        clip01(getattr(getattr(state, "contact_state", None), "contact_coverage", 0.0)),
        clip01(getattr(getattr(state, "contact_affordance_graph", None), "scene_contact_feasibility", 0.0)),
        clip01(getattr(getattr(state, "contact_affordance_graph", None), "scene_affordance_coverage", 0.0)),
        clip01(getattr(getattr(state, "contact_affordance_graph", None), "scene_obstruction_severity", 0.0)),
        clip01(getattr(getattr(state, "inverse_retarget_trace", None), "readiness_score", 0.0)),
        clip01(getattr(getattr(state, "action_proposal_bundle", None), "action_feasibility_score", 0.0)),
        clip01(getattr(getattr(state, "drift_summary", None), "drift_score", 0.0)),
        clip01(getattr(getattr(state, "cost_vector", None), "risk_score", 0.0)),
        safe_float(getattr(getattr(state, "cost_vector", None), "latency_ms", 0.0)) / 1000.0,
        clip01(getattr(getattr(state, "safety_envelope", None), "margin_fraction", 0.0)),
        safe_float(getattr(getattr(state, "action_space", None), "dimension", 0.0)) / 64.0,
        safe_float(getattr(getattr(state, "observation_interface", None), "sample_hz", 0.0)) / 200.0,
    ]
    if len(values) < input_dim:
        values.extend([0.0] * (input_dim - len(values)))
    return torch.tensor(values[:input_dim], dtype=torch.float32)


def smoke_forward_all_seams(state: Any, *, action_dim: Optional[int] = None) -> dict[str, dict[str, Any]]:
    """Run CPU proof-of-life forward passes for all Phase 3.4 seam families."""
    inferred_action_dim = int(action_dim or max(1, getattr(getattr(state, "action_space", None), "dimension", 12)))
    features32 = encode_state_features(state, 32).unsqueeze(0)
    features20 = encode_state_features(state, 20).unsqueeze(0)
    features16 = encode_state_features(state, 16).unsqueeze(0)
    seams = {
        "local_contact_dynamics": LocalContactDynamicsSeam(input_dim=16),
        "inverse_retargeting": InverseRetargetingSeam(input_dim=32, target_action_dim=inferred_action_dim),
        "action_proposal": ActionProposalSeam(input_dim=32, action_dim=inferred_action_dim, chunk_len=4),
        "drift_calibration": DriftCalibrationSeam(input_dim=20),
    }
    inputs = {
        "local_contact_dynamics": features16,
        "inverse_retargeting": features32,
        "action_proposal": features32,
        "drift_calibration": features20,
    }
    outputs: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for seam_id, seam in seams.items():
            out = seam(inputs[seam_id])
            outputs[seam_id] = {
                "describe": seam.describe(),
                "output_shapes": {key: list(value.shape) for key, value in out.items()},
                "finite": all(bool(torch.isfinite(value).all().item()) for value in out.values()),
            }
    return outputs
