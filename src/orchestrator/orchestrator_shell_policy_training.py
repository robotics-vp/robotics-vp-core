from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.orchestrator.orchestrator_shell_policy import (
    SHELL_POLICY_FEATURE_NAMES,
    SHELL_POLICY_PRESET_LABELS,
    SHELL_POLICY_STRATEGY_KEYS,
    build_shell_policy_feature_map,
    extract_orchestrator_advisory_target,
)
from src.semantic.models import SemanticSnapshot
from src.utils.config_digest import sha256_json

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None


ORCHESTRATOR_SHELL_POLICY_MIN_ROWS = 64
ORCHESTRATOR_SHELL_POLICY_MIN_ACTIVATED_ROWS = 12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_ref(root_dir: Optional[str], ref: Any) -> Optional[Path]:
    if ref in (None, "", [], {}):
        return None
    path = Path(str(ref))
    if path.exists():
        return path
    if root_dir:
        candidate = Path(root_dir) / path
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class OrchestratorShellTrainingExample:
    row_id: str
    feature_map: Dict[str, float]
    preset_distribution: Dict[str, float]
    strategy_distribution: Dict[str, float]
    safety_emphasis: float
    activation_label: float
    target_source: str
    policy_source: str
    promotion_stage: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "feature_map": {str(key): float(value) for key, value in self.feature_map.items()},
            "preset_distribution": {
                str(key): float(value) for key, value in self.preset_distribution.items()
            },
            "strategy_distribution": {
                str(key): float(value) for key, value in self.strategy_distribution.items()
            },
            "safety_emphasis": float(self.safety_emphasis),
            "activation_label": float(self.activation_label),
            "target_source": self.target_source,
            "policy_source": self.policy_source,
            "promotion_stage": self.promotion_stage,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OrchestratorShellTrainingDataset:
    examples: list[OrchestratorShellTrainingExample]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "orchestrator_shell_training_dataset_v1",
            "summary": dict(self.summary),
            "examples": [example.to_dict() for example in self.examples],
        }


def load_runtime_rows(paths: Sequence[str | Path]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"semantic runtime rows not found: {candidate}")
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    return rows


def _load_snapshot_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        snapshot = metadata.get("semantic_snapshot")
        if isinstance(snapshot, Mapping):
            return dict(snapshot)
    artifact_refs = row.get("artifact_refs")
    root_dir = row.get("root_dir") or row.get("source_root")
    if not isinstance(artifact_refs, Mapping):
        return {}
    snapshot_path = _resolve_ref(
        root_dir,
        artifact_refs.get("semantic_snapshot_ref") or artifact_refs.get("semantic_snapshot_path"),
    )
    if snapshot_path is None:
        return {}
    return _load_json(snapshot_path)


def _load_advisory_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        advisory = metadata.get("orchestrator_advisory")
        if isinstance(advisory, Mapping):
            return dict(advisory)
    artifact_refs = row.get("artifact_refs")
    root_dir = row.get("root_dir") or row.get("source_root")
    if not isinstance(artifact_refs, Mapping):
        return {}
    advisory_path = _resolve_ref(
        root_dir,
        artifact_refs.get("orchestrator_advisory_ref") or artifact_refs.get("orchestrator_advisory_path"),
    )
    if advisory_path is None:
        return {}
    return _load_json(advisory_path)


def build_orchestrator_shell_training_dataset(
    runtime_rows: Sequence[Mapping[str, Any]],
) -> OrchestratorShellTrainingDataset:
    examples: list[OrchestratorShellTrainingExample] = []
    target_source_counts: Dict[str, int] = {}
    policy_source_counts: Dict[str, int] = {}
    promotion_stage_counts: Dict[str, int] = {}
    activated_rows = 0
    skipped_rows = 0
    for index, row in enumerate(runtime_rows):
        snapshot_payload = _load_snapshot_payload(row)
        advisory_payload = _load_advisory_payload(row)
        if not snapshot_payload or not advisory_payload:
            skipped_rows += 1
            continue
        try:
            snapshot = SemanticSnapshot.from_dict(snapshot_payload)
        except Exception:
            skipped_rows += 1
            continue
        target = extract_orchestrator_advisory_target(advisory_payload)
        feature_map = build_shell_policy_feature_map(snapshot)
        target_source = "orchestrator_advisory_receipt"
        policy_source = str(target.get("policy_source", "heuristic_fallback"))
        promotion_stage = str(target.get("promotion_stage", "heuristic_fallback"))
        target_source_counts[target_source] = target_source_counts.get(target_source, 0) + 1
        policy_source_counts[policy_source] = policy_source_counts.get(policy_source, 0) + 1
        promotion_stage_counts[promotion_stage] = promotion_stage_counts.get(promotion_stage, 0) + 1
        activated_rows += int(target.get("activation_label", 0.0) > 0.5)
        row_id = str(
            row.get("sample_id")
            or row.get("episode_id")
            or advisory_payload.get("task_id")
            or f"shell_row_{index}"
        )
        examples.append(
            OrchestratorShellTrainingExample(
                row_id=row_id,
                feature_map=feature_map,
                preset_distribution=dict(target.get("preset_distribution", {}) or {}),
                strategy_distribution=dict(target.get("sampler_strategy_overrides", {}) or {}),
                safety_emphasis=_safe_float(target.get("safety_emphasis", 0.0)),
                activation_label=_safe_float(target.get("activation_label", 0.0)),
                target_source=target_source,
                policy_source=policy_source,
                promotion_stage=promotion_stage,
                metadata={
                    "task_id": advisory_payload.get("task_id") or snapshot.task_id,
                    "execution_mode": target.get("execution_mode", "advisory"),
                    "activation_plan": dict(target.get("activation_plan", {}) or {}),
                },
            )
        )

    summary = {
        "schema_version": "orchestrator_shell_training_summary_v1",
        "num_runtime_rows": len(runtime_rows),
        "num_examples": len(examples),
        "num_skipped_rows": skipped_rows,
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "policy_source_counts": dict(sorted(policy_source_counts.items())),
        "promotion_stage_counts": dict(sorted(promotion_stage_counts.items())),
        "activated_rows": activated_rows,
        "feature_names": list(SHELL_POLICY_FEATURE_NAMES),
        "preset_labels": list(SHELL_POLICY_PRESET_LABELS),
        "strategy_keys": list(SHELL_POLICY_STRATEGY_KEYS),
        "dataset_digest": sha256_json([example.to_dict() for example in examples]),
        "benchmark_gate": {
            "ready": len(examples) >= ORCHESTRATOR_SHELL_POLICY_MIN_ROWS
            and activated_rows >= ORCHESTRATOR_SHELL_POLICY_MIN_ACTIVATED_ROWS,
            "required_rows": ORCHESTRATOR_SHELL_POLICY_MIN_ROWS,
            "required_activated_rows": ORCHESTRATOR_SHELL_POLICY_MIN_ACTIVATED_ROWS,
            "observed_rows": len(examples),
            "observed_activated_rows": activated_rows,
        },
    }
    return OrchestratorShellTrainingDataset(examples=examples, summary=summary)


def save_orchestrator_shell_training_dataset(
    dataset: OrchestratorShellTrainingDataset,
    path: str | Path,
) -> str:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(candidate)


def load_orchestrator_shell_training_dataset(path: str | Path) -> OrchestratorShellTrainingDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [
        OrchestratorShellTrainingExample(
            row_id=str(example.get("row_id", "")),
            feature_map={str(key): float(value) for key, value in dict(example.get("feature_map", {}) or {}).items()},
            preset_distribution={
                str(key): float(value)
                for key, value in dict(example.get("preset_distribution", {}) or {}).items()
            },
            strategy_distribution={
                str(key): float(value)
                for key, value in dict(example.get("strategy_distribution", {}) or {}).items()
            },
            safety_emphasis=float(example.get("safety_emphasis", 0.0)),
            activation_label=float(example.get("activation_label", 0.0)),
            target_source=str(example.get("target_source", "orchestrator_advisory_receipt")),
            policy_source=str(example.get("policy_source", "heuristic_fallback")),
            promotion_stage=str(example.get("promotion_stage", "heuristic_fallback")),
            metadata=dict(example.get("metadata", {}) or {}),
        )
        for example in list(payload.get("examples", []) or [])
        if isinstance(example, Mapping)
    ]
    return OrchestratorShellTrainingDataset(
        examples=examples,
        summary=dict(payload.get("summary", {}) or {}),
    )


if TORCH_AVAILABLE:

    class OrchestratorShellPolicyNet(nn.Module):
        def __init__(self, input_dim: int = len(SHELL_POLICY_FEATURE_NAMES), hidden_dim: int = 32) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.preset_head = nn.Linear(int(hidden_dim), len(SHELL_POLICY_PRESET_LABELS))
            self.strategy_head = nn.Linear(int(hidden_dim), len(SHELL_POLICY_STRATEGY_KEYS))
            self.safety_head = nn.Linear(int(hidden_dim), 1)
            self.activation_head = nn.Linear(int(hidden_dim), 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden = self.net(x)
            return (
                self.preset_head(hidden),
                self.strategy_head(hidden),
                self.safety_head(hidden),
                self.activation_head(hidden),
            )


else:  # pragma: no cover

    class OrchestratorShellPolicyNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("OrchestratorShellPolicyNet requires torch")


def train_orchestrator_shell_policy_model(
    dataset: OrchestratorShellTrainingDataset,
    *,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
) -> tuple[Any, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the orchestrator shell policy")
    if not dataset.examples:
        raise ValueError("orchestrator shell training dataset is empty")

    X = np.asarray(
        [
            [float(example.feature_map.get(name, 0.0)) for name in SHELL_POLICY_FEATURE_NAMES]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_preset = np.asarray(
        [
            [float(example.preset_distribution.get(label, 0.0)) for label in SHELL_POLICY_PRESET_LABELS]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_strategy = np.asarray(
        [
            [float(example.strategy_distribution.get(label, 0.0)) for label in SHELL_POLICY_STRATEGY_KEYS]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_safety = np.asarray([[float(example.safety_emphasis)] for example in dataset.examples], dtype=np.float32)
    y_activation = np.asarray([[float(example.activation_label)] for example in dataset.examples], dtype=np.float32)

    X_tensor = torch.from_numpy(X)
    y_preset_tensor = torch.from_numpy(y_preset)
    y_strategy_tensor = torch.from_numpy(y_strategy)
    y_safety_tensor = torch.from_numpy(y_safety)
    y_activation_tensor = torch.from_numpy(y_activation)

    model = OrchestratorShellPolicyNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    history: Dict[str, Any] = {
        "loss": [],
        "preset_loss": [],
        "strategy_loss": [],
        "safety_loss": [],
        "activation_loss": [],
    }

    model.train()
    for _ in range(int(epochs)):
        preset_logits, strategy_logits, safety_logits, activation_logits = model(X_tensor)
        preset_probs = torch.softmax(preset_logits, dim=-1)
        strategy_probs = torch.softmax(strategy_logits, dim=-1)
        safety_probs = torch.sigmoid(safety_logits)

        preset_loss = mse_loss(preset_probs, y_preset_tensor)
        strategy_loss = mse_loss(strategy_probs, y_strategy_tensor)
        safety_loss = mse_loss(safety_probs, y_safety_tensor)
        activation_loss = bce_loss(activation_logits, y_activation_tensor)
        loss = preset_loss + strategy_loss + safety_loss + activation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["preset_loss"].append(float(preset_loss.item()))
        history["strategy_loss"].append(float(strategy_loss.item()))
        history["safety_loss"].append(float(safety_loss.item()))
        history["activation_loss"].append(float(activation_loss.item()))

    model.eval()
    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": int(X.shape[1]),
                "hidden_dim": int(hidden_dim),
                "feature_names": list(SHELL_POLICY_FEATURE_NAMES),
                "preset_labels": list(SHELL_POLICY_PRESET_LABELS),
                "strategy_keys": list(SHELL_POLICY_STRATEGY_KEYS),
            },
            save_path,
        )
    return model, history


__all__ = [
    "ORCHESTRATOR_SHELL_POLICY_MIN_ACTIVATED_ROWS",
    "ORCHESTRATOR_SHELL_POLICY_MIN_ROWS",
    "OrchestratorShellPolicyNet",
    "OrchestratorShellTrainingDataset",
    "OrchestratorShellTrainingExample",
    "TORCH_AVAILABLE",
    "build_orchestrator_shell_training_dataset",
    "load_orchestrator_shell_training_dataset",
    "load_runtime_rows",
    "save_orchestrator_shell_training_dataset",
    "train_orchestrator_shell_policy_model",
]
