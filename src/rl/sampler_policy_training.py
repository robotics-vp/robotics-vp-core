from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.rl.sampler_policy import (
    SAMPLER_EPISODE_FEATURE_NAMES,
    SAMPLER_PLAN_PARAMETER_NAMES,
    SAMPLER_POLICY_STRATEGIES,
    SAMPLER_POOL_FEATURE_NAMES,
)
from src.utils.config_digest import sha256_json

torch: Any
nn: Any
_torch: Any
_nn: Any
try:
    import torch as _torch
    import torch.nn as _nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    _torch = None
    _nn = None

torch = _torch
nn = _nn


SAMPLER_POLICY_MIN_POOL_ROWS = 32
SAMPLER_POLICY_MIN_RECEIPT_ROWS = 16


def _load_payload(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"sampler policy receipt not found: {path}")
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


def load_sampler_policy_receipts(paths: Sequence[str | Path]) -> list[Dict[str, Any]]:
    payloads: list[Dict[str, Any]] = []
    for path in paths:
        payloads.extend(_load_payload(Path(path)))
    receipts: list[Dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload.get("sampler_policy_receipt"), Mapping):
            receipts.append(dict(payload["sampler_policy_receipt"]))
        elif "pool_feature_map" in payload and "episode_entries" in payload:
            receipts.append(dict(payload))
    return receipts


@dataclass(frozen=True)
class SamplerPolicyPoolExample:
    row_id: str
    feature_map: Dict[str, float]
    strategy_targets: Dict[str, float]
    plan_targets: Dict[str, float]
    target_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "feature_map": {
                str(key): float(value) for key, value in self.feature_map.items()
            },
            "strategy_targets": {
                str(key): float(value) for key, value in self.strategy_targets.items()
            },
            "plan_targets": {
                str(key): float(value) for key, value in self.plan_targets.items()
            },
            "target_source": self.target_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SamplerPolicyEpisodeExample:
    row_id: str
    episode_id: str
    strategy: str
    feature_map: Dict[str, float]
    target_weight: float
    target_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "episode_id": self.episode_id,
            "strategy": self.strategy,
            "feature_map": {
                str(key): float(value) for key, value in self.feature_map.items()
            },
            "target_weight": float(self.target_weight),
            "target_source": self.target_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SamplerPolicyTrainingDataset:
    pool_examples: list[SamplerPolicyPoolExample]
    episode_examples: list[SamplerPolicyEpisodeExample]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "sampler_policy_training_dataset_v1",
            "summary": dict(self.summary),
            "pool_examples": [example.to_dict() for example in self.pool_examples],
            "episode_examples": [
                example.to_dict() for example in self.episode_examples
            ],
        }


def build_sampler_policy_training_dataset(
    receipts: Sequence[Mapping[str, Any]],
) -> SamplerPolicyTrainingDataset:
    pool_examples: list[SamplerPolicyPoolExample] = []
    episode_examples: list[SamplerPolicyEpisodeExample] = []
    target_source_counts: Dict[str, int] = {}
    receipt_rows = 0
    strategies_seen: set[str] = set()

    for receipt_index, receipt in enumerate(receipts):
        target_source = str(receipt.get("target_source", "heuristic_bootstrap"))
        target_source_counts[target_source] = (
            target_source_counts.get(target_source, 0) + 1
        )
        pool_examples.append(
            SamplerPolicyPoolExample(
                row_id=str(
                    receipt.get("receipt_id") or f"sampler_receipt_{receipt_index}"
                ),
                feature_map={
                    str(name): float(
                        dict(receipt.get("pool_feature_map", {}) or {}).get(name, 0.0)
                    )
                    for name in SAMPLER_POOL_FEATURE_NAMES
                },
                strategy_targets={
                    strategy: float(
                        dict(receipt.get("strategy_targets", {}) or {}).get(
                            strategy, 0.0
                        )
                    )
                    for strategy in SAMPLER_POLICY_STRATEGIES
                },
                plan_targets={
                    name: float(
                        dict(receipt.get("sampling_plan_targets", {}) or {}).get(
                            name, 0.0
                        )
                    )
                    for name in SAMPLER_PLAN_PARAMETER_NAMES
                },
                target_source=target_source,
                metadata={
                    "heuristic_selected_strategy": receipt.get(
                        "heuristic_selected_strategy"
                    ),
                    "final_strategy": receipt.get("final_strategy"),
                },
            )
        )
        for entry_index, entry in enumerate(
            list(receipt.get("episode_entries", []) or [])
        ):
            if not isinstance(entry, Mapping):
                continue
            episode_id = str(
                entry.get("episode_id") or f"episode_{receipt_index}_{entry_index}"
            )
            weight_targets = dict(entry.get("strategy_weight_targets", {}) or {})
            has_receipt_feedback = bool(
                dict(entry.get("metadata", {}) or {}).get("has_receipt_feedback", False)
            )
            receipt_rows += int(has_receipt_feedback)
            for strategy in SAMPLER_POLICY_STRATEGIES:
                if strategy not in weight_targets:
                    continue
                strategies_seen.add(strategy)
                episode_examples.append(
                    SamplerPolicyEpisodeExample(
                        row_id=f"{episode_id}:{strategy}",
                        episode_id=episode_id,
                        strategy=strategy,
                        feature_map={
                            str(name): float(
                                dict(entry.get("feature_map", {}) or {}).get(name, 0.0)
                            )
                            for name in SAMPLER_EPISODE_FEATURE_NAMES
                        },
                        target_weight=float(weight_targets.get(strategy, 0.0)),
                        target_source=str(entry.get("target_source", target_source)),
                        metadata={
                            "selected_in_batch": bool(
                                entry.get("selected_in_batch", False)
                            ),
                            "has_receipt_feedback": has_receipt_feedback,
                        },
                    )
                )

    summary = {
        "schema_version": "sampler_policy_training_summary_v1",
        "num_receipts": len(receipts),
        "num_pool_examples": len(pool_examples),
        "num_episode_examples": len(episode_examples),
        "receipt_feedback_rows": receipt_rows,
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "strategy_coverage": sorted(strategies_seen),
        "pool_feature_names": list(SAMPLER_POOL_FEATURE_NAMES),
        "episode_feature_names": list(SAMPLER_EPISODE_FEATURE_NAMES),
        "strategy_names": list(SAMPLER_POLICY_STRATEGIES),
        "plan_parameter_names": list(SAMPLER_PLAN_PARAMETER_NAMES),
        "dataset_digest": sha256_json(
            {
                "pool_examples": [example.to_dict() for example in pool_examples],
                "episode_examples": [example.to_dict() for example in episode_examples],
            }
        ),
        "benchmark_gate": {
            "ready": len(pool_examples) >= SAMPLER_POLICY_MIN_POOL_ROWS
            and receipt_rows >= SAMPLER_POLICY_MIN_RECEIPT_ROWS,
            "required_pool_rows": SAMPLER_POLICY_MIN_POOL_ROWS,
            "required_receipt_rows": SAMPLER_POLICY_MIN_RECEIPT_ROWS,
            "observed_pool_rows": len(pool_examples),
            "observed_receipt_rows": receipt_rows,
        },
    }
    return SamplerPolicyTrainingDataset(
        pool_examples=pool_examples,
        episode_examples=episode_examples,
        summary=summary,
    )


def save_sampler_policy_training_dataset(
    dataset: SamplerPolicyTrainingDataset, path: str | Path
) -> str:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return str(candidate)


def load_sampler_policy_training_dataset(
    path: str | Path,
) -> SamplerPolicyTrainingDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SamplerPolicyTrainingDataset(
        pool_examples=[
            SamplerPolicyPoolExample(
                row_id=str(example.get("row_id", "")),
                feature_map={
                    str(key): float(value)
                    for key, value in dict(example.get("feature_map", {}) or {}).items()
                },
                strategy_targets={
                    str(key): float(value)
                    for key, value in dict(
                        example.get("strategy_targets", {}) or {}
                    ).items()
                },
                plan_targets={
                    str(key): float(value)
                    for key, value in dict(
                        example.get("plan_targets", {}) or {}
                    ).items()
                },
                target_source=str(example.get("target_source", "heuristic_bootstrap")),
                metadata=dict(example.get("metadata", {}) or {}),
            )
            for example in list(payload.get("pool_examples", []) or [])
            if isinstance(example, Mapping)
        ],
        episode_examples=[
            SamplerPolicyEpisodeExample(
                row_id=str(example.get("row_id", "")),
                episode_id=str(example.get("episode_id", "")),
                strategy=str(example.get("strategy", "balanced")),
                feature_map={
                    str(key): float(value)
                    for key, value in dict(example.get("feature_map", {}) or {}).items()
                },
                target_weight=float(example.get("target_weight", 0.0)),
                target_source=str(example.get("target_source", "heuristic_bootstrap")),
                metadata=dict(example.get("metadata", {}) or {}),
            )
            for example in list(payload.get("episode_examples", []) or [])
            if isinstance(example, Mapping)
        ],
        summary=dict(payload.get("summary", {}) or {}),
    )


if TORCH_AVAILABLE:

    class SamplerPolicyPoolNet(nn.Module):
        def __init__(
            self,
            input_dim: int = len(SAMPLER_POOL_FEATURE_NAMES),
            hidden_dim: int = 32,
            strategy_count: int = len(SAMPLER_POLICY_STRATEGIES),
            plan_count: int = len(SAMPLER_PLAN_PARAMETER_NAMES),
        ) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.strategy_head = nn.Linear(int(hidden_dim), int(strategy_count))
            self.plan_head = nn.Linear(int(hidden_dim), int(plan_count))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.trunk(x)
            return self.strategy_head(hidden), self.plan_head(hidden)

    class SamplerPolicyEpisodeNet(nn.Module):
        def __init__(
            self,
            input_dim: int = len(SAMPLER_EPISODE_FEATURE_NAMES)
            + len(SAMPLER_POLICY_STRATEGIES),
            hidden_dim: int = 32,
        ) -> None:
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

    class SamplerPolicyPoolNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("SamplerPolicyPoolNet requires torch")

    class SamplerPolicyEpisodeNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("SamplerPolicyEpisodeNet requires torch")


def train_sampler_policy_models(
    dataset: SamplerPolicyTrainingDataset,
    *,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    save_path: str | None = None,
) -> tuple[Any, Any, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the sampler policy helper")
    if not dataset.pool_examples or not dataset.episode_examples:
        raise ValueError("sampler policy training dataset is empty")

    X_pool = np.asarray(
        [
            [
                float(example.feature_map.get(name, 0.0))
                for name in SAMPLER_POOL_FEATURE_NAMES
            ]
            for example in dataset.pool_examples
        ],
        dtype=np.float32,
    )
    y_strategy = np.asarray(
        [
            [
                float(example.strategy_targets.get(name, 0.0))
                for name in SAMPLER_POLICY_STRATEGIES
            ]
            for example in dataset.pool_examples
        ],
        dtype=np.float32,
    )
    y_plan = np.asarray(
        [
            [
                float(example.plan_targets.get(name, 0.0))
                for name in SAMPLER_PLAN_PARAMETER_NAMES
            ]
            for example in dataset.pool_examples
        ],
        dtype=np.float32,
    )
    X_episode = np.asarray(
        [
            [
                *[
                    float(example.feature_map.get(name, 0.0))
                    for name in SAMPLER_EPISODE_FEATURE_NAMES
                ],
                *[
                    1.0 if example.strategy == strategy else 0.0
                    for strategy in SAMPLER_POLICY_STRATEGIES
                ],
            ]
            for example in dataset.episode_examples
        ],
        dtype=np.float32,
    )
    y_episode = np.asarray(
        [[float(example.target_weight)] for example in dataset.episode_examples],
        dtype=np.float32,
    )

    pool_net = SamplerPolicyPoolNet(input_dim=X_pool.shape[1], hidden_dim=hidden_dim)
    episode_net = SamplerPolicyEpisodeNet(
        input_dim=X_episode.shape[1], hidden_dim=hidden_dim
    )
    optimizer = torch.optim.Adam(
        list(pool_net.parameters()) + list(episode_net.parameters()), lr=lr
    )
    mse_loss = nn.MSELoss()
    history: Dict[str, list[float]] = {"loss": [], "pool_loss": [], "episode_loss": []}

    X_pool_tensor = torch.from_numpy(X_pool)
    y_strategy_tensor = torch.from_numpy(y_strategy)
    y_plan_tensor = torch.from_numpy(y_plan)
    X_episode_tensor = torch.from_numpy(X_episode)
    y_episode_tensor = torch.from_numpy(y_episode)

    pool_net.train()
    episode_net.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        strategy_logits, plan_logits = pool_net(X_pool_tensor)
        strategy_probs = torch.softmax(strategy_logits, dim=-1)
        plan_probs = torch.sigmoid(plan_logits)
        episode_scores = torch.sigmoid(episode_net(X_episode_tensor))
        pool_loss = mse_loss(strategy_probs, y_strategy_tensor) + mse_loss(
            plan_probs, y_plan_tensor
        )
        episode_loss = mse_loss(episode_scores, y_episode_tensor)
        loss = pool_loss + episode_loss
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach().item()))
        history["pool_loss"].append(float(pool_loss.detach().item()))
        history["episode_loss"].append(float(episode_loss.detach().item()))

    pool_net.eval()
    episode_net.eval()
    checkpoint_path = None
    if save_path:
        checkpoint_path = Path(save_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pool_state_dict": pool_net.state_dict(),
                "episode_state_dict": episode_net.state_dict(),
                "pool_input_dim": X_pool.shape[1],
                "episode_input_dim": X_episode.shape[1],
                "hidden_dim": int(hidden_dim),
                "pool_feature_names": list(SAMPLER_POOL_FEATURE_NAMES),
                "episode_feature_names": list(SAMPLER_EPISODE_FEATURE_NAMES),
                "strategy_names": list(SAMPLER_POLICY_STRATEGIES),
                "plan_parameter_names": list(SAMPLER_PLAN_PARAMETER_NAMES),
            },
            str(checkpoint_path),
        )
    return (
        pool_net,
        episode_net,
        {
            "epochs": int(epochs),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "history": history,
            "final_loss": history["loss"][-1],
            "final_pool_loss": history["pool_loss"][-1],
            "final_episode_loss": history["episode_loss"][-1],
        },
    )


__all__ = [
    "SAMPLER_POLICY_MIN_POOL_ROWS",
    "SAMPLER_POLICY_MIN_RECEIPT_ROWS",
    "SamplerPolicyEpisodeExample",
    "SamplerPolicyEpisodeNet",
    "SamplerPolicyPoolExample",
    "SamplerPolicyPoolNet",
    "SamplerPolicyTrainingDataset",
    "TORCH_AVAILABLE",
    "build_sampler_policy_training_dataset",
    "load_sampler_policy_receipts",
    "load_sampler_policy_training_dataset",
    "save_sampler_policy_training_dataset",
    "train_sampler_policy_models",
]
