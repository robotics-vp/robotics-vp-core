"""Learned trust/econ/readiness/correction overlays over coverage feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


FEATURE_NAMES = [
    "evidence_count",
    "economic_priority",
    "trust_priority",
    "promotion_readiness",
    "quality_score",
    "process_reward_delta",
    "policy_eval_delta",
    "cost_score",
    "backend_health_score",
    "wm_validation_pressure",
    "governance_blocked",
    "graph_mutation_pressure",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class FeedbackTopologyDataset:
    feature_names: List[str]
    features: List[List[float]]
    trust_targets: List[float]
    econ_targets: List[float]
    readiness_targets: List[float]
    correction_targets: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "features": list(self.features),
            "trust_targets": list(self.trust_targets),
            "econ_targets": list(self.econ_targets),
            "readiness_targets": list(self.readiness_targets),
            "correction_targets": list(self.correction_targets),
            "metadata": dict(self.metadata),
        }


def edge_feature_vector(edge: Any) -> np.ndarray:
    metadata = dict(getattr(edge, "metadata", {}) or {})
    vector = np.array(
        [
            np.log1p(_safe_float(getattr(edge, "evidence_count", 0.0))),
            _safe_float(getattr(edge, "economic_priority", 0.0)),
            _safe_float(getattr(edge, "trust_priority", 0.0)),
            _safe_float(getattr(edge, "promotion_readiness", 0.0)),
            _safe_float(metadata.get("quality_score", 0.0)),
            _safe_float(metadata.get("process_reward_delta", 0.0)),
            _safe_float(metadata.get("policy_eval_delta", 0.0)),
            _safe_float(metadata.get("cost_score", 0.0)),
            _safe_float(metadata.get("backend_health_score", 1.0)),
            _safe_float(metadata.get("wm_validation_pressure", 0.0)),
            1.0 if bool(metadata.get("governance_blocked", False)) else 0.0,
            min(_safe_float(metadata.get("graph_mutation_pressure", 0.0)) / 8.0, 1.0),
        ],
        dtype=np.float32,
    )
    return vector


def build_feedback_topology_dataset(coverage_graph: Any) -> FeedbackTopologyDataset:
    features: List[List[float]] = []
    trust_targets: List[float] = []
    econ_targets: List[float] = []
    readiness_targets: List[float] = []
    correction_targets: List[float] = []
    for edge in list(getattr(coverage_graph, "edges", []) or []):
        features.append(edge_feature_vector(edge).tolist())
        metadata = dict(getattr(edge, "metadata", {}) or {})
        trust_targets.append(_clip01(_safe_float(getattr(edge, "trust_priority", 0.0))))
        econ_targets.append(
            _clip01(_safe_float(getattr(edge, "economic_priority", 0.0)))
        )
        readiness_targets.append(
            _clip01(_safe_float(getattr(edge, "promotion_readiness", 0.0)))
        )
        correction_targets.append(
            _clip01(_safe_float(metadata.get("wm_validation_pressure", 0.0)))
        )
    return FeedbackTopologyDataset(
        feature_names=list(FEATURE_NAMES),
        features=features,
        trust_targets=trust_targets,
        econ_targets=econ_targets,
        readiness_targets=readiness_targets,
        correction_targets=correction_targets,
        metadata={"edge_count": len(features)},
    )


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True

    class _MultiHeadNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.trust_head = nn.Linear(hidden_dim, 1)
            self.econ_head = nn.Linear(hidden_dim, 1)
            self.readiness_head = nn.Linear(hidden_dim, 1)
            self.correction_head = nn.Linear(hidden_dim, 1)

        def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
            hidden = self.backbone(inputs)
            return {
                "trust": torch.sigmoid(self.trust_head(hidden)).squeeze(-1),
                "econ": torch.sigmoid(self.econ_head(hidden)).squeeze(-1),
                "readiness": torch.sigmoid(self.readiness_head(hidden)).squeeze(-1),
                "correction": torch.sigmoid(self.correction_head(hidden)).squeeze(-1),
            }

    @dataclass
    class SemanticFeedbackAdapterPackage:
        model: _MultiHeadNet
        feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
        metadata: Dict[str, Any] = field(default_factory=dict)

        def predict_edges(self, edges: Sequence[Any]) -> List[Dict[str, float]]:
            if not edges:
                return []
            features = torch.tensor(
                np.array(
                    [edge_feature_vector(edge) for edge in edges], dtype=np.float32
                )
            )
            with torch.no_grad():
                predictions = self.model(features)
            results: List[Dict[str, float]] = []
            for idx in range(len(edges)):
                results.append(
                    {
                        "trust_priority": float(predictions["trust"][idx].item()),
                        "economic_priority": float(predictions["econ"][idx].item()),
                        "promotion_readiness": float(
                            predictions["readiness"][idx].item()
                        ),
                        "wm_correction_pressure": float(
                            predictions["correction"][idx].item()
                        ),
                    }
                )
            return results

        def to_checkpoint(self) -> Dict[str, Any]:
            return {
                "feature_names": list(self.feature_names),
                "metadata": dict(self.metadata),
                "state_dict": self.model.state_dict(),
            }

        @classmethod
        def from_checkpoint(
            cls, payload: Mapping[str, Any]
        ) -> "SemanticFeedbackAdapterPackage":
            metadata = dict(payload.get("metadata", {}) or {})
            model = _MultiHeadNet(
                input_dim=len(payload.get("feature_names", FEATURE_NAMES)),
                hidden_dim=int(metadata.get("hidden_dim", 48)),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            return cls(
                model=model,
                feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
                metadata=metadata,
            )

    def train_semantic_feedback_adapter_package(
        dataset: FeedbackTopologyDataset,
        *,
        epochs: int = 24,
        learning_rate: float = 1e-3,
        hidden_dim: int = 48,
    ) -> SemanticFeedbackAdapterPackage:
        if not dataset.features:
            raise ValueError("no feedback topology samples available")
        inputs = torch.tensor(np.array(dataset.features, dtype=np.float32))
        trust_targets = torch.tensor(np.array(dataset.trust_targets, dtype=np.float32))
        econ_targets = torch.tensor(np.array(dataset.econ_targets, dtype=np.float32))
        readiness_targets = torch.tensor(
            np.array(dataset.readiness_targets, dtype=np.float32)
        )
        correction_targets = torch.tensor(
            np.array(dataset.correction_targets, dtype=np.float32)
        )
        model = _MultiHeadNet(
            input_dim=len(dataset.feature_names), hidden_dim=hidden_dim
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            predictions = model(inputs)
            loss = (
                loss_fn(predictions["trust"], trust_targets)
                + loss_fn(predictions["econ"], econ_targets)
                + loss_fn(predictions["readiness"], readiness_targets)
                + loss_fn(predictions["correction"], correction_targets)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        return SemanticFeedbackAdapterPackage(
            model=model,
            feature_names=list(dataset.feature_names),
            metadata={
                "hidden_dim": int(hidden_dim),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "dataset_metadata": dict(dataset.metadata),
            },
        )

    def shadow_fit_feedback_adapter_package(
        coverage_graph: Any,
        *,
        min_samples: int = 4,
    ) -> Optional[SemanticFeedbackAdapterPackage]:
        dataset = build_feedback_topology_dataset(coverage_graph)
        if len(dataset.features) < min_samples:
            return None
        try:
            return train_semantic_feedback_adapter_package(
                dataset, epochs=16, learning_rate=2e-3
            )
        except Exception:
            return None


except Exception:
    TORCH_AVAILABLE = False

    @dataclass
    class SemanticFeedbackAdapterPackage:  # type: ignore[no-redef]
        feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
        metadata: Dict[str, Any] = field(default_factory=dict)

        def predict_edges(self, edges: Sequence[Any]) -> List[Dict[str, float]]:
            return []

        def to_checkpoint(self) -> Dict[str, Any]:
            return {
                "feature_names": list(self.feature_names),
                "metadata": dict(self.metadata),
            }

    def train_semantic_feedback_adapter_package(  # type: ignore[misc,no-redef]
        *args: Any, **kwargs: Any
    ) -> SemanticFeedbackAdapterPackage:
        raise ImportError("train_semantic_feedback_adapter_package requires torch")

    def shadow_fit_feedback_adapter_package(  # type: ignore[misc,no-redef]
        *args: Any, **kwargs: Any
    ) -> Optional[SemanticFeedbackAdapterPackage]:
        return None


__all__ = [
    "FEATURE_NAMES",
    "FeedbackTopologyDataset",
    "SemanticFeedbackAdapterPackage",
    "build_feedback_topology_dataset",
    "edge_feature_vector",
    "shadow_fit_feedback_adapter_package",
    "train_semantic_feedback_adapter_package",
]
