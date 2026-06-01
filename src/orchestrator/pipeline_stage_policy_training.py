from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.orchestrator.pipeline_stage_policy import (
    PIPELINE_CONFIG_FLAG_KEYS,
    PIPELINE_STAGE_LABELS,
    PIPELINE_STAGE_POLICY_FEATURE_NAMES,
    extract_pipeline_stage_policy_target,
)
from src.utils.config_digest import sha256_json

torch: Any
nn: Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None


PIPELINE_STAGE_POLICY_MIN_ROWS = 48
PIPELINE_STAGE_POLICY_MIN_ACTIVATED_ROWS = 12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_pipeline_manager_states(paths: Sequence[str | Path]) -> list[Dict[str, Any]]:
    states: list[Dict[str, Any]] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"pipeline manager state not found: {candidate}")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            states.append(dict(payload))
        elif isinstance(payload, list):
            states.extend(dict(item) for item in payload if isinstance(item, Mapping))
    return states


@dataclass(frozen=True)
class PipelineStageTrainingExample:
    row_id: str
    feature_map: Dict[str, float]
    stage_distribution: Dict[str, float]
    config_flag_scores: Dict[str, float]
    activation_label: float
    target_source: str
    policy_source: str
    promotion_stage: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "feature_map": {
                str(key): float(value) for key, value in self.feature_map.items()
            },
            "stage_distribution": {
                str(key): float(value) for key, value in self.stage_distribution.items()
            },
            "config_flag_scores": {
                str(key): float(value) for key, value in self.config_flag_scores.items()
            },
            "activation_label": float(self.activation_label),
            "target_source": self.target_source,
            "policy_source": self.policy_source,
            "promotion_stage": self.promotion_stage,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PipelineStageTrainingDataset:
    examples: list[PipelineStageTrainingExample]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "pipeline_stage_policy_training_dataset_v1",
            "summary": dict(self.summary),
            "examples": [example.to_dict() for example in self.examples],
        }


def build_pipeline_stage_training_dataset(
    manager_states: Sequence[Mapping[str, Any]],
) -> PipelineStageTrainingDataset:
    from src.orchestrator.pipeline_manager import PipelineManager

    examples: list[PipelineStageTrainingExample] = []
    target_source_counts: Dict[str, int] = {}
    policy_source_counts: Dict[str, int] = {}
    promotion_stage_counts: Dict[str, int] = {}
    activated_rows = 0
    skipped_rows = 0
    for index, state in enumerate(manager_states):
        try:
            manager = PipelineManager.from_dict(dict(state))
        except Exception:
            skipped_rows += 1
            continue
        manager.config = dict(manager.config)
        manager.config["pipeline_stage_policy_helper_mode"] = "disabled"
        manager.config.pop("pipeline_stage_policy_package", None)
        manager.config.pop("pipeline_stage_policy_package_path", None)
        try:
            activation_plan = manager.build_iteration_activation_plan()
        except Exception:
            skipped_rows += 1
            continue
        trace = dict(activation_plan.get("stage_policy_trace", {}) or {})
        feature_map = {
            str(key): float(value)
            for key, value in dict(trace.get("feature_map", {}) or {}).items()
        }
        if not feature_map:
            skipped_rows += 1
            continue
        target = extract_pipeline_stage_policy_target(activation_plan)
        row_id = str(
            state.get("pipeline_id") or state.get("name") or f"pipeline_state_{index}"
        )
        target_source = "pipeline_manager_activation_receipt"
        policy_source = str(target.get("policy_source", "heuristic_fallback"))
        promotion_stage = str(target.get("promotion_stage", "heuristic_fallback"))
        target_source_counts[target_source] = (
            target_source_counts.get(target_source, 0) + 1
        )
        policy_source_counts[policy_source] = (
            policy_source_counts.get(policy_source, 0) + 1
        )
        promotion_stage_counts[promotion_stage] = (
            promotion_stage_counts.get(promotion_stage, 0) + 1
        )
        activated_rows += int(_safe_float(target.get("activation_label", 0.0)) > 0.5)
        examples.append(
            PipelineStageTrainingExample(
                row_id=row_id,
                feature_map=feature_map,
                stage_distribution={
                    label: _safe_float(
                        dict(target.get("stage_distribution", {}) or {}).get(label, 0.0)
                    )
                    for label in PIPELINE_STAGE_LABELS
                },
                config_flag_scores={
                    key: _safe_float(
                        dict(target.get("config_flag_scores", {}) or {}).get(key, 0.0)
                    )
                    for key in PIPELINE_CONFIG_FLAG_KEYS
                },
                activation_label=_safe_float(target.get("activation_label", 0.0)),
                target_source=target_source,
                policy_source=policy_source,
                promotion_stage=promotion_stage,
                metadata={
                    "pipeline_name": manager.name,
                    "iteration_count": len(manager.iterations),
                    "execution_mode": activation_plan.get("execution_mode", "advisory"),
                    "stage_order": [
                        str(row.get("stage", ""))
                        for row in list(
                            dict(
                                activation_plan.get("stage_activation_plan", {}) or {}
                            ).get("stages", [])
                        )
                        if isinstance(row, Mapping)
                    ],
                },
            )
        )

    summary = {
        "schema_version": "pipeline_stage_policy_training_summary_v1",
        "num_manager_states": len(manager_states),
        "num_examples": len(examples),
        "num_skipped_rows": skipped_rows,
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "policy_source_counts": dict(sorted(policy_source_counts.items())),
        "promotion_stage_counts": dict(sorted(promotion_stage_counts.items())),
        "activated_rows": activated_rows,
        "feature_names": list(PIPELINE_STAGE_POLICY_FEATURE_NAMES),
        "stage_labels": list(PIPELINE_STAGE_LABELS),
        "config_flag_keys": list(PIPELINE_CONFIG_FLAG_KEYS),
        "dataset_digest": sha256_json([example.to_dict() for example in examples]),
        "benchmark_gate": {
            "ready": len(examples) >= PIPELINE_STAGE_POLICY_MIN_ROWS
            and activated_rows >= PIPELINE_STAGE_POLICY_MIN_ACTIVATED_ROWS,
            "required_rows": PIPELINE_STAGE_POLICY_MIN_ROWS,
            "required_activated_rows": PIPELINE_STAGE_POLICY_MIN_ACTIVATED_ROWS,
            "observed_rows": len(examples),
            "observed_activated_rows": activated_rows,
        },
    }
    return PipelineStageTrainingDataset(examples=examples, summary=summary)


def save_pipeline_stage_training_dataset(
    dataset: PipelineStageTrainingDataset,
    path: str | Path,
) -> str:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return str(candidate)


def load_pipeline_stage_training_dataset(
    path: str | Path,
) -> PipelineStageTrainingDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [
        PipelineStageTrainingExample(
            row_id=str(example.get("row_id", "")),
            feature_map={
                str(key): float(value)
                for key, value in dict(example.get("feature_map", {}) or {}).items()
            },
            stage_distribution={
                str(key): float(value)
                for key, value in dict(
                    example.get("stage_distribution", {}) or {}
                ).items()
            },
            config_flag_scores={
                str(key): float(value)
                for key, value in dict(
                    example.get("config_flag_scores", {}) or {}
                ).items()
            },
            activation_label=float(example.get("activation_label", 0.0)),
            target_source=str(
                example.get("target_source", "pipeline_manager_activation_receipt")
            ),
            policy_source=str(example.get("policy_source", "heuristic_fallback")),
            promotion_stage=str(example.get("promotion_stage", "heuristic_fallback")),
            metadata=dict(example.get("metadata", {}) or {}),
        )
        for example in list(payload.get("examples", []) or [])
        if isinstance(example, Mapping)
    ]
    return PipelineStageTrainingDataset(
        examples=examples,
        summary=dict(payload.get("summary", {}) or {}),
    )


if TORCH_AVAILABLE:

    class PipelineStagePolicyNet(nn.Module):
        def __init__(
            self,
            input_dim: int = len(PIPELINE_STAGE_POLICY_FEATURE_NAMES),
            hidden_dim: int = 32,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.stage_head = nn.Linear(int(hidden_dim), len(PIPELINE_STAGE_LABELS))
            self.config_head = nn.Linear(
                int(hidden_dim), len(PIPELINE_CONFIG_FLAG_KEYS)
            )
            self.activation_head = nn.Linear(int(hidden_dim), 1)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.net(x)
            return (
                self.stage_head(hidden),
                self.config_head(hidden),
                self.activation_head(hidden),
            )


else:  # pragma: no cover

    class PipelineStagePolicyNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PipelineStagePolicyNet requires torch")


def train_pipeline_stage_policy_model(
    dataset: PipelineStageTrainingDataset,
    *,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
) -> tuple[Any, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the pipeline stage policy")
    if not dataset.examples:
        raise ValueError("pipeline stage policy training dataset is empty")

    X = np.asarray(
        [
            [
                float(example.feature_map.get(name, 0.0))
                for name in PIPELINE_STAGE_POLICY_FEATURE_NAMES
            ]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_stage = np.asarray(
        [
            [
                float(example.stage_distribution.get(label, 0.0))
                for label in PIPELINE_STAGE_LABELS
            ]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_config = np.asarray(
        [
            [
                float(example.config_flag_scores.get(key, 0.0))
                for key in PIPELINE_CONFIG_FLAG_KEYS
            ]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_activation = np.asarray(
        [[float(example.activation_label)] for example in dataset.examples],
        dtype=np.float32,
    )

    X_tensor = torch.from_numpy(X)
    y_stage_tensor = torch.from_numpy(y_stage)
    y_config_tensor = torch.from_numpy(y_config)
    y_activation_tensor = torch.from_numpy(y_activation)

    model = PipelineStagePolicyNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    history: Dict[str, list[float]] = {
        "loss": [],
        "stage_loss": [],
        "config_loss": [],
        "activation_loss": [],
    }

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        stage_logits, config_logits, activation_logits = model(X_tensor)
        stage_probs = torch.softmax(stage_logits, dim=-1)
        config_probs = torch.sigmoid(config_logits)
        stage_loss = mse_loss(stage_probs, y_stage_tensor)
        config_loss = mse_loss(config_probs, y_config_tensor)
        activation_loss = bce_loss(activation_logits, y_activation_tensor)
        loss = stage_loss + config_loss + activation_loss
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach().item()))
        history["stage_loss"].append(float(stage_loss.detach().item()))
        history["config_loss"].append(float(config_loss.detach().item()))
        history["activation_loss"].append(float(activation_loss.detach().item()))

    model.eval()
    checkpoint_path = None
    if save_path:
        checkpoint_path = Path(save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": X.shape[1],
                "hidden_dim": int(hidden_dim),
                "feature_names": list(PIPELINE_STAGE_POLICY_FEATURE_NAMES),
                "stage_labels": list(PIPELINE_STAGE_LABELS),
                "config_flag_keys": list(PIPELINE_CONFIG_FLAG_KEYS),
            },
            str(checkpoint_path),
        )

    return model, {
        "epochs": int(epochs),
        "lr": float(lr),
        "hidden_dim": int(hidden_dim),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "final_loss": history["loss"][-1],
        "history": history,
    }


__all__ = [
    "PIPELINE_STAGE_POLICY_MIN_ACTIVATED_ROWS",
    "PIPELINE_STAGE_POLICY_MIN_ROWS",
    "PipelineStagePolicyNet",
    "PipelineStageTrainingDataset",
    "PipelineStageTrainingExample",
    "TORCH_AVAILABLE",
    "build_pipeline_stage_training_dataset",
    "load_pipeline_manager_states",
    "load_pipeline_stage_training_dataset",
    "save_pipeline_stage_training_dataset",
    "train_pipeline_stage_policy_model",
]
