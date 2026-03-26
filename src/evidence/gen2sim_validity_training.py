"""Learned helper substrate for gen2sim validity/value admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.evidence.gen2sim_validity import (
    GEN2SIM_FEATURE_NAMES,
    GEN2SIM_OBJECTIVE_DIM,
    build_gen2sim_feature_vector,
)

try:
    import torch
    import torch.nn as nn

    class LearnedGen2SimValidityModel(nn.Module):
        """Small MLP that predicts bounded validity and value-support scores."""

        def __init__(
            self,
            *,
            input_dim: int = len(GEN2SIM_FEATURE_NAMES),
            hidden_dim: int = 64,
            objective_dim: int = GEN2SIM_OBJECTIVE_DIM,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.objective_dim = int(objective_dim)
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            self.validity_head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
            self.value_support_head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.net(x)
            return self.validity_head(hidden), self.value_support_head(hidden)

        def predict_features(self, feature_vector: Sequence[float]) -> Dict[str, Any]:
            x = torch.from_numpy(np.asarray(feature_vector, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                validity, value_support = self.forward(x)
            predicted_validity = float(validity.squeeze().item())
            predicted_value_support = float(value_support.squeeze().item())
            benchmark_gate = dict(getattr(self, "benchmark_gate", {}) or {})
            promotion_stage = (
                "promoted"
                if bool(benchmark_gate.get("ready", False))
                else str(getattr(self, "promotion_stage", "shadow_candidate") or "shadow_candidate")
            )
            return {
                "predicted_validity_score": predicted_validity,
                "predicted_value_support_score": predicted_value_support,
                "predicted_admission_score": float(
                    predicted_validity * (0.75 + (0.25 * predicted_value_support))
                ),
                "feature_names": GEN2SIM_FEATURE_NAMES[: self.input_dim],
                "feature_vector": [float(value) for value in feature_vector],
                "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
                "promotion_stage": promotion_stage,
            }

        def predict_context(self, *, context: Mapping[str, Any]) -> Dict[str, Any]:
            feature_vector = build_gen2sim_feature_vector(
                context,
                objective_dim=self.objective_dim,
            )
            return self.predict_features(feature_vector)

        @classmethod
        def from_checkpoint(cls, path: str) -> "LearnedGen2SimValidityModel":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = cls(
                input_dim=int(payload.get("input_dim", len(GEN2SIM_FEATURE_NAMES))),
                hidden_dim=int(payload.get("hidden_dim", 64)),
                objective_dim=int(payload.get("objective_dim", GEN2SIM_OBJECTIVE_DIM)),
            )
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            return model

    TORCH_AVAILABLE = True

except ImportError:  # pragma: no cover - explicit failure paths below
    TORCH_AVAILABLE = False
    torch = None

    class LearnedGen2SimValidityModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("LearnedGen2SimValidityModel requires torch")

        @classmethod
        def from_checkpoint(cls, path: str) -> "LearnedGen2SimValidityModel":
            raise ImportError("LearnedGen2SimValidityModel requires torch")


@dataclass(frozen=True)
class Gen2SimValidityTrainingRow:
    subject_id: str
    feature_vector: list[float]
    target_validity_score: float
    target_value_support_score: float
    promotion_stage: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def train_gen2sim_validity_model(
    rows: Sequence[Gen2SimValidityTrainingRow],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    save_path: Optional[str] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Train the learned helper from explicit assessment traces."""

    if not TORCH_AVAILABLE:
        raise ImportError("Training requires torch")
    if not rows:
        raise ValueError("No training rows provided")

    input_dim = len(rows[0].feature_vector)
    X = np.asarray([row.feature_vector for row in rows], dtype=np.float32)
    y_validity = np.asarray([row.target_validity_score for row in rows], dtype=np.float32)
    y_value_support = np.asarray(
        [row.target_value_support_score for row in rows],
        dtype=np.float32,
    )
    sample_weights = np.asarray(
        [
            1.0
            + (0.2 if row.promotion_stage == "shadow_candidate" else 0.0)
            + (0.35 if row.promotion_stage == "promoted" else 0.0)
            for row in rows
        ],
        dtype=np.float32,
    )

    X_tensor = torch.from_numpy(X)
    y_validity_tensor = torch.from_numpy(y_validity).unsqueeze(-1)
    y_value_support_tensor = torch.from_numpy(y_value_support).unsqueeze(-1)
    weight_tensor = torch.from_numpy(sample_weights).unsqueeze(-1)

    model = LearnedGen2SimValidityModel(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    history: Dict[str, Any] = {"loss": [], "validity_mse": [], "value_support_mse": []}
    model.train()
    for _ in range(int(epochs)):
        pred_validity, pred_value_support = model(X_tensor)
        validity_loss = loss_fn(pred_validity, y_validity_tensor)
        value_support_loss = loss_fn(pred_value_support, y_value_support_tensor)
        loss = ((validity_loss + value_support_loss) * weight_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["validity_mse"].append(float(validity_loss.mean().item()))
        history["value_support_mse"].append(float(value_support_loss.mean().item()))

    model.eval()
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": int(hidden_dim),
                "objective_dim": GEN2SIM_OBJECTIVE_DIM,
            },
            save_path,
        )
    return model, history


__all__ = [
    "GEN2SIM_FEATURE_NAMES",
    "GEN2SIM_OBJECTIVE_DIM",
    "Gen2SimValidityTrainingRow",
    "LearnedGen2SimValidityModel",
    "TORCH_AVAILABLE",
    "train_gen2sim_validity_model",
]
