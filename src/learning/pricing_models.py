"""Learned shadow pricing residual models and episode feature encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.replay.schema import ReplayEpisodeRecord
from src.utils.config_digest import sha256_json


def episode_feature_vector(record: ReplayEpisodeRecord) -> List[float]:
    """Deterministic episode feature surface for learned shadow advisors."""

    objective_axes = dict(record.objective_tensor_summary.get("axes", {}) or {})
    normalized_axes = dict(record.objective_tensor_summary.get("normalized_axes", {}) or {})
    econ_axes = dict(record.econ_tensor_summary.get("axes", {}) or {})
    pricing = dict(record.pricing_summary or {})
    hard_flags = sum(1 for flag in record.constraint_flags if str(flag.get("severity", "")) == "hard")
    soft_flags = sum(1 for flag in record.constraint_flags if str(flag.get("severity", "")) == "soft")
    deploy_map = {"allow_shadow": 1.0, "require_review": 0.5, "deny_shadow": 0.0}
    datapack_quality = float(record.datapack_summary.get("quality_score", 0.0))
    return [
        float(record.total_reward),
        float(record.total_steps),
        float(objective_axes.get("throughput", 0.0)),
        float(objective_axes.get("error", 0.0)),
        float(objective_axes.get("safety", 0.0)),
        float(objective_axes.get("energy", 0.0)),
        float(normalized_axes.get("throughput", 0.0)),
        float(normalized_axes.get("error", 0.0)),
        float(normalized_axes.get("safety", 0.0)),
        float(normalized_axes.get("energy", 0.0)),
        float(econ_axes.get("value_earned", 0.0)),
        float(econ_axes.get("price_tick", 0.0)),
        float(econ_axes.get("marginal_frontier_gain", 0.0)),
        float(econ_axes.get("constraint_penalty", 0.0)),
        float(econ_axes.get("uncertainty_discount", 0.0)),
        float(pricing.get("task_hour_price_tick", 0.0)),
        float(pricing.get("net_customer_rate", 0.0)),
        float(pricing.get("confidence", 0.0)),
        float(record.datapack_summary.get("data_share_credit", 0.0)),
        float(record.datapack_summary.get("marginal_frontier_gain", 0.0)),
        float(datapack_quality),
        float(hard_flags),
        float(soft_flags),
        float(deploy_map.get(str(record.regal_summary.get("deploy_recommendation", "allow_shadow")), 0.0)),
        _hash_to_unit(record.skill_mode),
        _hash_to_unit(record.source_domain),
    ]


@dataclass(frozen=True)
class ResidualPrediction:
    """Single shadow-model prediction with confidence."""

    value: float
    confidence: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": float(self.value),
            "confidence": float(self.confidence),
            "metadata": dict(self.metadata),
        }


class ResidualMLP(nn.Module):
    """Small deterministic MLP with residual/confidence heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        value = self.value(hidden).squeeze(-1)
        confidence = torch.sigmoid(self.confidence(hidden)).squeeze(-1)
        return value, confidence


class PricingDeltaModel(ResidualMLP):
    """Residual model that predicts delta over heuristic task-hour price."""

    model_version = "shadow_pricing_delta_v1"


def pricing_residual_target(record: ReplayEpisodeRecord) -> float:
    base_rate = float(record.pricing_summary.get("net_customer_rate", 0.0))
    quality = float(record.datapack_summary.get("quality_score", 0.0))
    frontier = float(record.datapack_summary.get("marginal_frontier_gain", 0.0))
    hard_flags = sum(1 for flag in record.constraint_flags if str(flag.get("severity", "")) == "hard")
    return float(max(-20.0, min(20.0, 12.0 * (quality - 0.5) + 6.0 * frontier - 3.5 * hard_flags + 0.05 * base_rate)))


def train_pricing_delta_model(
    records: Sequence[ReplayEpisodeRecord],
    *,
    seed: int = 42,
    epochs: int = 12,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[PricingDeltaModel, Dict[str, Any]]:
    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    features = torch.as_tensor([episode_feature_vector(record) for record in records], dtype=torch.float32, device=device)
    targets = torch.as_tensor([pricing_residual_target(record) for record in records], dtype=torch.float32, device=device)
    model = PricingDeltaModel(input_dim=int(features.shape[-1]), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction, confidence = model(features)
        mse = F.mse_loss(prediction, targets)
        confidence_target = torch.exp(-torch.abs(targets) / 10.0)
        confidence_loss = F.mse_loss(confidence, confidence_target)
        loss = mse + 0.1 * confidence_loss
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.item()), "mse": float(mse.item())})

    metrics = {
        "input_dim": int(features.shape[-1]),
        "epochs": epochs,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "final_mse": history[-1]["mse"] if history else 0.0,
        "history": history,
        "model_version": PricingDeltaModel.model_version,
    }
    return model, metrics


def predict_pricing_delta(model: PricingDeltaModel, record: ReplayEpisodeRecord, *, device: Optional[torch.device] = None) -> ResidualPrediction:
    device = device or next(model.parameters()).device
    features = torch.as_tensor(episode_feature_vector(record), dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        value, confidence = model(features)
    return ResidualPrediction(
        value=float(value.squeeze(0).item()),
        confidence=float(confidence.squeeze(0).item()),
        metadata={
            "model_version": PricingDeltaModel.model_version,
            "feature_digest": sha256_json(episode_feature_vector(record)),
        },
    )


def _hash_to_unit(value: str) -> float:
    digest = sha256_json({"value": value})
    return int(digest[:12], 16) / float(16 ** 12)
