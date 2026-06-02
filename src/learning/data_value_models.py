"""Learned shadow datapack credit and marginal-value models."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.learning.pricing_models import ResidualMLP, ResidualPrediction, episode_feature_vector
from src.replay.schema import ReplayEpisodeRecord


class DataValueModel(ResidualMLP):
    """Predict datapack credit/value against the heuristic baseline."""

    model_version = "shadow_data_value_v1"


def data_value_target(record: ReplayEpisodeRecord) -> float:
    heuristic_credit = float(record.datapack_summary.get("data_share_credit", 0.0))
    frontier = float(record.datapack_summary.get("marginal_frontier_gain", 0.0))
    quality = float(record.datapack_summary.get("quality_score", 0.0))
    recommendation = str(record.regal_summary.get("datapack_recommendation", "keep"))
    review_penalty = {"keep": 0.0, "review": -0.5, "downweight": -1.0, "reward_credit": 0.5}.get(recommendation, 0.0)
    return float(max(-5.0, min(15.0, heuristic_credit + 8.0 * frontier * max(0.25, quality) + review_penalty)))


def train_data_value_model(
    records: Sequence[ReplayEpisodeRecord],
    *,
    seed: int = 42,
    epochs: int = 12,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[DataValueModel, Dict[str, Any]]:
    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    features = torch.as_tensor([episode_feature_vector(record) for record in records], dtype=torch.float32, device=device)
    targets = torch.as_tensor([data_value_target(record) for record in records], dtype=torch.float32, device=device)
    model = DataValueModel(input_dim=int(features.shape[-1]), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, confidence = model(features)
        mse = F.mse_loss(prediction, targets)
        confidence_target = torch.sigmoid(targets / 10.0)
        loss = mse + 0.1 * F.mse_loss(confidence, confidence_target)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.item()), "mse": float(mse.item())})
    return model, {
        "epochs": epochs,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "final_mse": history[-1]["mse"] if history else 0.0,
        "history": history,
        "model_version": DataValueModel.model_version,
    }


def predict_data_value(model: DataValueModel, record: ReplayEpisodeRecord, *, device: Optional[torch.device] = None) -> ResidualPrediction:
    device = device or next(model.parameters()).device
    features = torch.as_tensor(episode_feature_vector(record), dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        value, confidence = model(features)
    return ResidualPrediction(
        value=float(value.squeeze(0).item()),
        confidence=float(confidence.squeeze(0).item()),
        metadata={"model_version": DataValueModel.model_version},
    )
