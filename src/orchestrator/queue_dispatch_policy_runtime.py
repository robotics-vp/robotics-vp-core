from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.orchestrator.queue_dispatch_policy import (
    QUEUE_DISPATCH_FEATURE_NAMES,
    build_queue_dispatch_feature_map,
)
from src.orchestrator.queue_dispatch_policy_training import (
    QueueDispatchPolicyNet,
    TORCH_AVAILABLE,
)

torch: Any

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover
    torch = None


def _clamp01(value: Any) -> float:
    try:
        candidate = float(value)
    except Exception:
        candidate = 0.0
    return max(0.0, min(1.0, candidate))


@dataclass(frozen=True)
class QueueDispatchPolicyRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_queue_dispatch_policy_runtime_package(
    path: str | Path,
) -> QueueDispatchPolicyRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(payload.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = (package_path.parent / checkpoint_path).resolve()
    return QueueDispatchPolicyRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(checkpoint_path),
        model_config=dict(payload.get("model_config", {}) or {}),
        benchmark_gate=dict(payload.get("benchmark_gate", {}) or {}),
        execution_preconditions=dict(payload.get("execution_preconditions", {}) or {}),
        inference_contract=dict(payload.get("inference_contract", {}) or {}),
        promotion_stage=str(
            payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"
        ),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


class LoadedQueueDispatchPolicyHelper:
    def __init__(self, package: QueueDispatchPolicyRuntimePackage) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to load the queue dispatch policy helper"
            )
        checkpoint_path = Path(package.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"queue dispatch policy checkpoint not found: {checkpoint_path}"
            )
        payload = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        input_dim = int(payload.get("input_dim", len(QUEUE_DISPATCH_FEATURE_NAMES)))
        hidden_dim = int(payload.get("hidden_dim", 32))
        self.package = package
        self.model = QueueDispatchPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.benchmark_gate = dict(package.benchmark_gate or {})
        self.inference_contract = dict(package.inference_contract or {})
        self.promotion_stage = str(package.promotion_stage or "shadow_candidate")

    def score_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        feature_map = build_queue_dispatch_feature_map(entry)
        vector = np.asarray(
            [
                float(feature_map.get(name, 0.0))
                for name in QUEUE_DISPATCH_FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        tensor = torch.from_numpy(vector).float().unsqueeze(0)
        with torch.no_grad():
            dispatch_score = float(torch.sigmoid(self.model(tensor)[0]).item())
        return {
            "dispatch_score": _clamp01(dispatch_score),
            "feature_map": feature_map,
        }


def resolve_queue_dispatch_policy_helper(
    *,
    helper_mode: str = "disabled",
    package: Optional[QueueDispatchPolicyRuntimePackage] = None,
    package_path: Optional[str | Path] = None,
) -> Optional[LoadedQueueDispatchPolicyHelper]:
    mode = str(helper_mode or "disabled")
    if mode == "disabled":
        return None
    if package is None and package_path is None:
        if mode == "required":
            raise ValueError("queue dispatch policy helper requires a package path")
        return None
    if package is None:
        assert package_path is not None
        package = load_queue_dispatch_policy_runtime_package(package_path)
    helper = LoadedQueueDispatchPolicyHelper(package)
    if mode == "required" and not bool(helper.benchmark_gate.get("ready", False)):
        raise ValueError(
            "queue dispatch policy helper requires a benchmark-gated package"
        )
    return helper


__all__ = [
    "LoadedQueueDispatchPolicyHelper",
    "QueueDispatchPolicyRuntimePackage",
    "load_queue_dispatch_policy_runtime_package",
    "resolve_queue_dispatch_policy_helper",
]
