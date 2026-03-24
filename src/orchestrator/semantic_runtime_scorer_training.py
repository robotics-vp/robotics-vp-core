"""Heavyweight training substrate for semantic runtime scorers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import json

from src.orchestrator.semantic_runtime_learning import SemanticRuntimeLearningCorpus, SemanticRuntimeLearningRow
from src.orchestrator.semantic_runtime_scorers import (
    AUTHORITY_FEATURE_NAMES,
    COUNTERFACTUAL_FEATURE_NAMES,
    META_ROUTE_FEATURE_NAMES,
    ORCHESTRATION_ROUTE_FEATURE_NAMES,
    _authority_feature_map,
    _counterfactual_feature_map,
    _meta_route_feature_map,
    _orchestration_route_feature_map,
    derive_authority_success_label,
)
from src.utils.json_safe import to_json_safe

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class SemanticRuntimeScorerTrainingDataset:
    meta_route_feature_names: list[str]
    meta_route_features: list[list[float]]
    meta_route_targets: list[float]
    orchestration_route_feature_names: list[str]
    orchestration_route_features: list[list[float]]
    orchestration_route_targets: list[float]
    authority_feature_names: list[str]
    authority_features: list[list[float]]
    authority_targets: list[float]
    counterfactual_feature_names: list[str]
    counterfactual_features: list[list[float]]
    counterfactual_targets: list[float]
    regret_targets: list[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta_route_feature_names": list(self.meta_route_feature_names),
            "meta_route_features": list(self.meta_route_features),
            "meta_route_targets": list(self.meta_route_targets),
            "orchestration_route_feature_names": list(self.orchestration_route_feature_names),
            "orchestration_route_features": list(self.orchestration_route_features),
            "orchestration_route_targets": list(self.orchestration_route_targets),
            "authority_feature_names": list(self.authority_feature_names),
            "authority_features": list(self.authority_features),
            "authority_targets": list(self.authority_targets),
            "counterfactual_feature_names": list(self.counterfactual_feature_names),
            "counterfactual_features": list(self.counterfactual_features),
            "counterfactual_targets": list(self.counterfactual_targets),
            "regret_targets": list(self.regret_targets),
            "metadata": dict(to_json_safe(dict(self.metadata))),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRuntimeScorerTrainingDataset":
        return cls(
            meta_route_feature_names=[str(item) for item in payload.get("meta_route_feature_names", []) or []],
            meta_route_features=[[float(value) for value in row] for row in payload.get("meta_route_features", []) or []],
            meta_route_targets=[float(value) for value in payload.get("meta_route_targets", []) or []],
            orchestration_route_feature_names=[
                str(item) for item in payload.get("orchestration_route_feature_names", []) or []
            ],
            orchestration_route_features=[
                [float(value) for value in row] for row in payload.get("orchestration_route_features", []) or []
            ],
            orchestration_route_targets=[float(value) for value in payload.get("orchestration_route_targets", []) or []],
            authority_feature_names=[str(item) for item in payload.get("authority_feature_names", []) or []],
            authority_features=[[float(value) for value in row] for row in payload.get("authority_features", []) or []],
            authority_targets=[float(value) for value in payload.get("authority_targets", []) or []],
            counterfactual_feature_names=[str(item) for item in payload.get("counterfactual_feature_names", []) or []],
            counterfactual_features=[
                [float(value) for value in row] for row in payload.get("counterfactual_features", []) or []
            ],
            counterfactual_targets=[float(value) for value in payload.get("counterfactual_targets", []) or []],
            regret_targets=[float(value) for value in payload.get("regret_targets", []) or []],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def _ordered_features(feature_map: Mapping[str, Any], feature_names: Sequence[str]) -> list[float]:
    return [_safe_float(feature_map.get(name, 0.0)) for name in feature_names]


def build_semantic_runtime_scorer_training_dataset(
    corpus: SemanticRuntimeLearningCorpus | Sequence[SemanticRuntimeLearningRow],
) -> SemanticRuntimeScorerTrainingDataset:
    rows = list(corpus.rows if isinstance(corpus, SemanticRuntimeLearningCorpus) else corpus)
    meta_route_features: list[list[float]] = []
    orchestration_route_features: list[list[float]] = []
    authority_features: list[list[float]] = []
    counterfactual_features: list[list[float]] = []
    meta_route_targets: list[float] = []
    orchestration_route_targets: list[float] = []
    authority_targets: list[float] = []
    counterfactual_targets: list[float] = []
    regret_targets: list[float] = []
    for row in rows:
        meta_route_features.append(_ordered_features(_meta_route_feature_map(row), META_ROUTE_FEATURE_NAMES))
        orchestration_route_features.append(
            _ordered_features(_orchestration_route_feature_map(row), ORCHESTRATION_ROUTE_FEATURE_NAMES)
        )
        authority_features.append(_ordered_features(_authority_feature_map(row), AUTHORITY_FEATURE_NAMES))
        meta_route_targets.append(_safe_float(row.inferential_summary.get("route_success_label", 0.0)))
        orchestration_route_targets.append(
            1.0
            if (
                row.inferential_summary.get("route_success_label", False)
                and bool(row.orchestration_transformer_target.get("tool_sequence", []))
            )
            else 0.0
        )
        authority_targets.append(1.0 if derive_authority_success_label(row) else 0.0)
        regret_targets.append(_safe_float(row.inferential_summary.get("estimated_regret", 0.0)))
        for counterfactual in row.counterfactuals:
            counterfactual_features.append(
                _ordered_features(_counterfactual_feature_map(row, counterfactual), COUNTERFACTUAL_FEATURE_NAMES)
            )
            counterfactual_targets.append(float(counterfactual.predicted_outcome_score))
    return SemanticRuntimeScorerTrainingDataset(
        meta_route_feature_names=list(META_ROUTE_FEATURE_NAMES),
        meta_route_features=meta_route_features,
        meta_route_targets=meta_route_targets,
        orchestration_route_feature_names=list(ORCHESTRATION_ROUTE_FEATURE_NAMES),
        orchestration_route_features=orchestration_route_features,
        orchestration_route_targets=orchestration_route_targets,
        authority_feature_names=list(AUTHORITY_FEATURE_NAMES),
        authority_features=authority_features,
        authority_targets=authority_targets,
        counterfactual_feature_names=list(COUNTERFACTUAL_FEATURE_NAMES),
        counterfactual_features=counterfactual_features,
        counterfactual_targets=counterfactual_targets,
        regret_targets=regret_targets,
        metadata={
            "row_count": len(rows),
            "counterfactual_count": len(counterfactual_targets),
            "corpus_summary": dict(getattr(corpus, "summary", {})),
        },
    )


def write_semantic_runtime_scorer_training_dataset(
    output_path: str | Path,
    dataset: SemanticRuntimeScorerTrainingDataset,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_semantic_runtime_scorer_training_dataset(
    path: str | Path,
) -> SemanticRuntimeScorerTrainingDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid semantic runtime scorer training dataset: {path}")
    return SemanticRuntimeScorerTrainingDataset.from_dict(payload)


if TORCH_AVAILABLE:

    class _HeadNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs).squeeze(-1)


    class SemanticRuntimeScorerNet(nn.Module):
        def __init__(
            self,
            meta_dim: int,
            orchestration_dim: int,
            authority_dim: int,
            counterfactual_dim: int,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__()
            self.meta_head = _HeadNet(meta_dim, hidden_dim)
            self.orchestration_head = _HeadNet(orchestration_dim, hidden_dim)
            self.authority_head = _HeadNet(authority_dim, hidden_dim)
            self.regret_head = _HeadNet(meta_dim, hidden_dim)
            self.counterfactual_head = _HeadNet(counterfactual_dim, hidden_dim)

        def forward(self, dataset: SemanticRuntimeScorerTrainingDataset) -> Dict[str, torch.Tensor]:
            return {
                "meta_route_logits": self.meta_head(torch.tensor(dataset.meta_route_features, dtype=torch.float32)),
                "orchestration_route_logits": self.orchestration_head(
                    torch.tensor(dataset.orchestration_route_features, dtype=torch.float32)
                ),
                "authority_logits": self.authority_head(torch.tensor(dataset.authority_features, dtype=torch.float32)),
                "regret_pred": self.regret_head(torch.tensor(dataset.meta_route_features, dtype=torch.float32)),
                "counterfactual_value_pred": self.counterfactual_head(
                    torch.tensor(dataset.counterfactual_features, dtype=torch.float32)
                )
                if dataset.counterfactual_features
                else torch.zeros(0, dtype=torch.float32),
            }


def train_semantic_runtime_scorer_net(
    dataset: SemanticRuntimeScorerTrainingDataset,
    *,
    epochs: int = 24,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        return {
            "torch_available": False,
            "trained": False,
            "reason": "torch_unavailable",
        }
    if not dataset.meta_route_features:
        return {
            "torch_available": True,
            "trained": False,
            "reason": "empty_dataset",
        }
    model = SemanticRuntimeScorerNet(
        meta_dim=len(dataset.meta_route_feature_names),
        orchestration_dim=len(dataset.orchestration_route_feature_names),
        authority_dim=len(dataset.authority_feature_names),
        counterfactual_dim=len(dataset.counterfactual_feature_names),
        hidden_dim=hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    meta_features = torch.tensor(dataset.meta_route_features, dtype=torch.float32)
    meta_targets = torch.tensor(dataset.meta_route_targets, dtype=torch.float32)
    orchestration_features = torch.tensor(dataset.orchestration_route_features, dtype=torch.float32)
    orchestration_targets = torch.tensor(dataset.orchestration_route_targets, dtype=torch.float32)
    authority_features = torch.tensor(dataset.authority_features, dtype=torch.float32)
    authority_targets = torch.tensor(dataset.authority_targets, dtype=torch.float32)
    regret_targets = torch.tensor(dataset.regret_targets, dtype=torch.float32)
    counterfactual_features = torch.tensor(dataset.counterfactual_features, dtype=torch.float32)
    counterfactual_targets = torch.tensor(dataset.counterfactual_targets, dtype=torch.float32)
    for _ in range(max(int(epochs), 1)):
        optimizer.zero_grad()
        meta_logits = model.meta_head(meta_features)
        orchestration_logits = model.orchestration_head(orchestration_features)
        authority_logits = model.authority_head(authority_features)
        regret_pred = model.regret_head(meta_features)
        loss = (
            bce(meta_logits, meta_targets)
            + bce(orchestration_logits, orchestration_targets)
            + bce(authority_logits, authority_targets)
            + mse(regret_pred, regret_targets)
        )
        if counterfactual_features.numel() > 0:
            counterfactual_pred = model.counterfactual_head(counterfactual_features)
            loss = loss + mse(counterfactual_pred, counterfactual_targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        meta_probs = torch.sigmoid(model.meta_head(meta_features))
        orchestration_probs = torch.sigmoid(model.orchestration_head(orchestration_features))
        authority_probs = torch.sigmoid(model.authority_head(authority_features))
        regret_pred = model.regret_head(meta_features)
        counterfactual_pred = model.counterfactual_head(counterfactual_features) if counterfactual_features.numel() > 0 else None
    summary = {
        "torch_available": True,
        "trained": True,
        "epochs": int(max(int(epochs), 1)),
        "learning_rate": float(learning_rate),
        "hidden_dim": int(hidden_dim),
        "meta_accuracy": float(torch.mean(((meta_probs >= 0.5) == (meta_targets >= 0.5)).float()).item()),
        "orchestration_accuracy": float(
            torch.mean(((orchestration_probs >= 0.5) == (orchestration_targets >= 0.5)).float()).item()
        ),
        "authority_accuracy": float(torch.mean(((authority_probs >= 0.5) == (authority_targets >= 0.5)).float()).item()),
        "regret_mae": float(torch.mean(torch.abs(regret_pred - regret_targets)).item()),
        "counterfactual_mae": float(torch.mean(torch.abs(counterfactual_pred - counterfactual_targets)).item())
        if counterfactual_pred is not None and counterfactual_targets.numel() > 0
        else 0.0,
    }
    return {
        "torch_available": True,
        "trained": True,
        "model": model,
        "summary": summary,
        "config": {
            "meta_dim": len(dataset.meta_route_feature_names),
            "orchestration_dim": len(dataset.orchestration_route_feature_names),
            "authority_dim": len(dataset.authority_feature_names),
            "counterfactual_dim": len(dataset.counterfactual_feature_names),
            "hidden_dim": int(hidden_dim),
        },
    }


def save_semantic_runtime_scorer_checkpoint(
    output_path: str | Path,
    training_result: Mapping[str, Any],
) -> Optional[str]:
    if not TORCH_AVAILABLE or not training_result.get("trained", False):
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = training_result.get("model")
    if model is None:
        return None
    torch.save(
        {
            "state_dict": model.state_dict(),
            "summary": dict(training_result.get("summary", {}) or {}),
            "config": dict(training_result.get("config", {}) or {}),
        },
        path,
    )
    return str(path)


__all__ = [
    "SemanticRuntimeScorerTrainingDataset",
    "TORCH_AVAILABLE",
    "build_semantic_runtime_scorer_training_dataset",
    "load_semantic_runtime_scorer_training_dataset",
    "save_semantic_runtime_scorer_checkpoint",
    "train_semantic_runtime_scorer_net",
    "write_semantic_runtime_scorer_training_dataset",
]
