from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.orchestrator.queue_dispatch_policy import (
    QUEUE_DISPATCH_FEATURE_NAMES,
    build_queue_dispatch_feature_map,
    extract_queue_dispatch_target,
)
from src.utils.config_digest import sha256_json

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = None


QUEUE_DISPATCH_POLICY_MIN_ROWS = 64
QUEUE_DISPATCH_POLICY_MIN_RECEIPT_ROWS = 16


def _load_payload(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"queue payload not found: {path}")
    if path.suffix == ".jsonl":
        payloads: list[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    payloads.append(dict(payload))
        return payloads
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def load_queue_selection_payloads(paths: Sequence[str | Path]) -> list[Dict[str, Any]]:
    payloads: list[Dict[str, Any]] = []
    for path in paths:
        payloads.extend(_load_payload(Path(path)))
    return payloads


def _extract_entries(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if isinstance(payload.get("live_queue_selection"), Mapping):
        return [dict(row) for row in list(payload["live_queue_selection"].get("entries", []) or []) if isinstance(row, Mapping)]
    if isinstance(payload.get("entries"), list):
        return [dict(row) for row in list(payload.get("entries", []) or []) if isinstance(row, Mapping)]
    return []


@dataclass(frozen=True)
class QueueDispatchTrainingExample:
    row_id: str
    feature_map: Dict[str, float]
    dispatch_score: float
    target_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "feature_map": {str(key): float(value) for key, value in self.feature_map.items()},
            "dispatch_score": float(self.dispatch_score),
            "target_source": self.target_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class QueueDispatchTrainingDataset:
    examples: list[QueueDispatchTrainingExample]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "queue_dispatch_policy_training_dataset_v1",
            "summary": dict(self.summary),
            "examples": [example.to_dict() for example in self.examples],
        }


def build_queue_dispatch_training_dataset(
    payloads: Sequence[Mapping[str, Any]],
) -> QueueDispatchTrainingDataset:
    examples: list[QueueDispatchTrainingExample] = []
    target_source_counts: Dict[str, int] = {}
    receipt_rows = 0
    for payload_index, payload in enumerate(payloads):
        for entry_index, entry in enumerate(_extract_entries(payload)):
            feature_map = build_queue_dispatch_feature_map(entry)
            target = extract_queue_dispatch_target(entry)
            target_source = str(target.get("target_source", "heuristic_bootstrap"))
            target_source_counts[target_source] = target_source_counts.get(target_source, 0) + 1
            receipt_rows += int(target_source == "receipt_feedback")
            examples.append(
                QueueDispatchTrainingExample(
                    row_id=str(entry.get("episode_id") or f"queue_entry_{payload_index}_{entry_index}"),
                    feature_map=feature_map,
                    dispatch_score=float(target.get("dispatch_score", 0.0)),
                    target_source=target_source,
                    metadata={
                        "queue_name": payload.get("queue_name")
                        or dict(payload.get("live_queue_selection", {}) or {}).get("queue_name")
                        or "shadow_advisory_queue",
                        "replay_action": entry.get("replay_action", "holdout"),
                    },
                )
            )
    summary = {
        "schema_version": "queue_dispatch_policy_training_summary_v1",
        "num_payloads": len(payloads),
        "num_examples": len(examples),
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "receipt_feedback_rows": receipt_rows,
        "feature_names": list(QUEUE_DISPATCH_FEATURE_NAMES),
        "dataset_digest": sha256_json([example.to_dict() for example in examples]),
        "benchmark_gate": {
            "ready": len(examples) >= QUEUE_DISPATCH_POLICY_MIN_ROWS
            and receipt_rows >= QUEUE_DISPATCH_POLICY_MIN_RECEIPT_ROWS,
            "required_rows": QUEUE_DISPATCH_POLICY_MIN_ROWS,
            "required_receipt_rows": QUEUE_DISPATCH_POLICY_MIN_RECEIPT_ROWS,
            "observed_rows": len(examples),
            "observed_receipt_rows": receipt_rows,
        },
    }
    return QueueDispatchTrainingDataset(examples=examples, summary=summary)


def save_queue_dispatch_training_dataset(dataset: QueueDispatchTrainingDataset, path: str | Path) -> str:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(candidate)


def load_queue_dispatch_training_dataset(path: str | Path) -> QueueDispatchTrainingDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [
        QueueDispatchTrainingExample(
            row_id=str(example.get("row_id", "")),
            feature_map={str(key): float(value) for key, value in dict(example.get("feature_map", {}) or {}).items()},
            dispatch_score=float(example.get("dispatch_score", 0.0)),
            target_source=str(example.get("target_source", "heuristic_bootstrap")),
            metadata=dict(example.get("metadata", {}) or {}),
        )
        for example in list(payload.get("examples", []) or [])
        if isinstance(example, Mapping)
    ]
    return QueueDispatchTrainingDataset(
        examples=examples,
        summary=dict(payload.get("summary", {}) or {}),
    )


if TORCH_AVAILABLE:

    class QueueDispatchPolicyNet(nn.Module):
        def __init__(self, input_dim: int = len(QUEUE_DISPATCH_FEATURE_NAMES), hidden_dim: int = 32) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


else:  # pragma: no cover

    class QueueDispatchPolicyNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("QueueDispatchPolicyNet requires torch")


def train_queue_dispatch_policy_model(
    dataset: QueueDispatchTrainingDataset,
    *,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: str | None = None,
) -> tuple[Any, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the queue dispatch policy")
    if not dataset.examples:
        raise ValueError("queue dispatch policy training dataset is empty")

    X = np.asarray(
        [
            [float(example.feature_map.get(name, 0.0)) for name in QUEUE_DISPATCH_FEATURE_NAMES]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y = np.asarray([[float(example.dispatch_score)] for example in dataset.examples], dtype=np.float32)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    model = QueueDispatchPolicyNet(input_dim=X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    history: Dict[str, list[float]] = {"loss": []}
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        preds = torch.sigmoid(model(X_tensor))
        loss = mse_loss(preds, y_tensor)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach().item()))
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
                "feature_names": list(QUEUE_DISPATCH_FEATURE_NAMES),
            },
            str(checkpoint_path),
        )
    return model, {
        "epochs": int(epochs),
        "lr": float(lr),
        "hidden_dim": int(hidden_dim),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "history": history,
        "final_loss": history["loss"][-1],
    }


__all__ = [
    "QUEUE_DISPATCH_POLICY_MIN_RECEIPT_ROWS",
    "QUEUE_DISPATCH_POLICY_MIN_ROWS",
    "QueueDispatchPolicyNet",
    "QueueDispatchTrainingDataset",
    "QueueDispatchTrainingExample",
    "TORCH_AVAILABLE",
    "build_queue_dispatch_training_dataset",
    "load_queue_dispatch_training_dataset",
    "load_queue_selection_payloads",
    "save_queue_dispatch_training_dataset",
    "train_queue_dispatch_policy_model",
]
