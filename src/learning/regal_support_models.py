"""Learned anomaly support scores that augment typed regal rules."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.learning.pricing_models import ResidualMLP, ResidualPrediction, episode_feature_vector
from src.replay.schema import ReplayEpisodeRecord


class RegalSupportModel(ResidualMLP):
    """Predict anomaly/risk support, never replacing typed regality."""

    model_version = "shadow_regal_support_v1"


def regal_support_target(record: ReplayEpisodeRecord) -> float:
    status = str(record.regal_summary.get("overall_status", "pass"))
    base = {"pass": 0.05, "warn": 0.45, "fail": 0.90}.get(status, 0.25)
    hard_flags = sum(1 for flag in record.constraint_flags if str(flag.get("severity", "")) == "hard")
    pricing_confidence = float(record.pricing_summary.get("confidence", 0.0))
    return float(max(0.0, min(1.0, base + 0.05 * hard_flags + 0.15 * (1.0 - pricing_confidence))))


def train_regal_support_model(
    records: Sequence[ReplayEpisodeRecord],
    *,
    seed: int = 42,
    epochs: int = 12,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[RegalSupportModel, Dict[str, Any]]:
    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    features = torch.as_tensor([episode_feature_vector(record) for record in records], dtype=torch.float32, device=device)
    targets = torch.as_tensor([regal_support_target(record) for record in records], dtype=torch.float32, device=device)
    model = RegalSupportModel(input_dim=int(features.shape[-1]), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, confidence = model(features)
        prediction = torch.sigmoid(prediction)
        mse = F.mse_loss(prediction, targets)
        loss = mse + 0.1 * F.mse_loss(confidence, 1.0 - torch.abs(targets - 0.5))
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.item()), "mse": float(mse.item())})
    return model, {
        "epochs": epochs,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "final_mse": history[-1]["mse"] if history else 0.0,
        "history": history,
        "model_version": RegalSupportModel.model_version,
    }


def predict_regal_support(model: RegalSupportModel, record: ReplayEpisodeRecord, *, device: Optional[torch.device] = None) -> ResidualPrediction:
    device = device or next(model.parameters()).device
    features = torch.as_tensor(episode_feature_vector(record), dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        value, confidence = model(features)
    return ResidualPrediction(
        value=float(torch.sigmoid(value).squeeze(0).item()),
        confidence=float(confidence.squeeze(0).item()),
        metadata={"model_version": RegalSupportModel.model_version},
    )
