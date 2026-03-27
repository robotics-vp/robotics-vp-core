"""Runtime package loading for semantic feedback adapter helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from src.world_model.feedback_topology_adapters import SemanticFeedbackAdapterPackage

try:  # pragma: no cover - explicit failure paths below
    import torch
except Exception:  # pragma: no cover
    torch = None


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


@dataclass(frozen=True)
class FeedbackTopologyRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_feedback_topology_runtime_package(path: str | Path) -> FeedbackTopologyRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(payload.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = (package_path.parent / checkpoint_path).resolve()
    return FeedbackTopologyRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(checkpoint_path),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        inference_contract=_mapping(payload.get("inference_contract")),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=_mapping(payload.get("metadata")),
    )


def resolve_feedback_adapter_helper(
    helper: Any,
    *,
    mode: Literal["disabled", "auto", "required"] = "auto",
) -> tuple[Optional[SemanticFeedbackAdapterPackage], Dict[str, Any]]:
    if mode not in {"disabled", "auto", "required"}:
        raise ValueError(f"Unsupported feedback-adapter mode: {mode}")
    if mode == "disabled":
        return None, {
            "mode": mode,
            "status": "disabled",
            "promotion_stage": "disabled",
            "benchmark_gate_ready": False,
        }
    if helper is None:
        if mode == "required":
            raise ValueError("feedback-adapter mode 'required' but no helper was provided")
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }
    if isinstance(helper, SemanticFeedbackAdapterPackage):
        benchmark_gate_ready = bool(getattr(helper, "benchmark_gate", {}).get("ready", False))
        if mode == "required" and not benchmark_gate_ready:
            raise ValueError("feedback-adapter mode 'required' requires a benchmark-gated package")
        return helper, {
            "mode": mode,
            "status": "loaded_direct",
            "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
            "benchmark_gate_ready": benchmark_gate_ready,
        }

    package: Optional[FeedbackTopologyRuntimePackage] = None
    checkpoint_path: Optional[Path] = None
    if isinstance(helper, (str, Path)):
        candidate = Path(helper)
        if candidate.suffix == ".json":
            package = load_feedback_topology_runtime_package(candidate)
            checkpoint_path = Path(package.checkpoint_path)
        else:
            checkpoint_path = candidate
    elif isinstance(helper, Mapping):
        if "checkpoint_path" in helper:
            package = FeedbackTopologyRuntimePackage(
                package_id=str(helper.get("package_id", "semantic_feedback_adapter")),
                package_path=str(helper.get("package_path", "")),
                checkpoint_path=str(helper.get("checkpoint_path", "")),
                benchmark_gate=_mapping(helper.get("benchmark_gate")),
                execution_preconditions=_mapping(helper.get("execution_preconditions")),
                inference_contract=_mapping(helper.get("inference_contract")),
                promotion_stage=str(helper.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
                metadata=_mapping(helper.get("metadata")),
            )
            checkpoint_path = Path(package.checkpoint_path)
        elif "state_dict" in helper:
            loaded = SemanticFeedbackAdapterPackage.from_checkpoint(helper)
            return loaded, {
                "mode": mode,
                "status": "loaded_checkpoint_mapping",
                "promotion_stage": "shadow_candidate",
                "benchmark_gate_ready": False,
            }

    if checkpoint_path is None or not checkpoint_path.exists():
        if mode == "required":
            raise ValueError("feedback-adapter mode 'required' but no loadable checkpoint/package was found")
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }
    if torch is None:
        raise ImportError("PyTorch is required to load the semantic feedback adapter helper")
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    loaded = SemanticFeedbackAdapterPackage.from_checkpoint(payload)
    benchmark_gate_ready = bool(package.benchmark_gate.get("ready", False)) if package is not None else False
    if package is not None:
        setattr(loaded, "benchmark_gate", dict(package.benchmark_gate))
        setattr(loaded, "execution_preconditions", dict(package.execution_preconditions))
        setattr(loaded, "inference_contract", dict(package.inference_contract))
        setattr(loaded, "promotion_stage", str(package.promotion_stage))
    if mode == "required" and not benchmark_gate_ready:
        raise ValueError("feedback-adapter mode 'required' requires a benchmark-gated package")
    return loaded, {
        "mode": mode,
        "status": "loaded",
        "package_id": package.package_id if package is not None else None,
        "package_path": package.package_path if package is not None else str(checkpoint_path),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "benchmark_gate_ready": benchmark_gate_ready,
        "unsatisfied_preconditions": list(
            package.execution_preconditions.get("unsatisfied_preconditions", [])
            if package is not None
            else []
        ),
    }


__all__ = [
    "FeedbackTopologyRuntimePackage",
    "load_feedback_topology_runtime_package",
    "resolve_feedback_adapter_helper",
]
