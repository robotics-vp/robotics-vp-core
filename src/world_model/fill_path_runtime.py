"""Runtime package loading for learned fill-path helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from src.world_model.fill_path_policy import LearnedFillPathPolicy


@dataclass(frozen=True)
class FillPathRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    metadata: Dict[str, Any]
    inference_contract: Dict[str, Any]


def _mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def load_fill_path_runtime_package(path: str | Path) -> FillPathRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    return FillPathRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(payload.get("checkpoint_path") or ""),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        metadata=_mapping(payload.get("metadata")),
        inference_contract=_mapping(payload.get("inference_contract")),
    )


def resolve_fill_path_helper(
    fill_path_policy: Any,
    *,
    mode: Literal["disabled", "auto", "required"] = "auto",
) -> tuple[Optional[Any], Dict[str, Any]]:
    if mode not in {"disabled", "auto", "required"}:
        raise ValueError(f"Unsupported fill-path policy mode: {mode}")
    if mode == "disabled":
        return None, {
            "mode": mode,
            "status": "disabled",
            "promotion_stage": "disabled",
            "benchmark_gate_ready": False,
        }
    if fill_path_policy is None:
        if mode == "required":
            raise ValueError("fill-path policy mode 'required' but no fill-path policy was provided")
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }
    if hasattr(fill_path_policy, "predict_batch_details") or hasattr(fill_path_policy, "predict_batch"):
        benchmark_gate_ready = bool(
            getattr(fill_path_policy, "benchmark_gate", {}).get("ready", False)
            if hasattr(fill_path_policy, "benchmark_gate")
            else False
        )
        if mode == "required" and not benchmark_gate_ready:
            raise ValueError("fill-path policy mode 'required' requires a benchmark-gated package")
        return fill_path_policy, {
            "mode": mode,
            "status": "loaded_direct",
            "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
            "benchmark_gate_ready": benchmark_gate_ready,
        }

    package: Optional[FillPathRuntimePackage] = None
    checkpoint_path: Optional[Path] = None
    if isinstance(fill_path_policy, (str, Path)):
        candidate = Path(fill_path_policy)
        if candidate.suffix == ".json":
            package = load_fill_path_runtime_package(candidate)
            checkpoint_path = Path(package.checkpoint_path)
        else:
            checkpoint_path = candidate
    elif isinstance(fill_path_policy, Mapping):
        if "checkpoint_path" in fill_path_policy:
            package = FillPathRuntimePackage(
                package_id=str(fill_path_policy.get("package_id", "fill_path_package")),
                package_path=str(fill_path_policy.get("package_path", "")),
                checkpoint_path=str(fill_path_policy.get("checkpoint_path", "")),
                benchmark_gate=_mapping(fill_path_policy.get("benchmark_gate")),
                execution_preconditions=_mapping(fill_path_policy.get("execution_preconditions")),
                metadata=_mapping(fill_path_policy.get("metadata")),
                inference_contract=_mapping(fill_path_policy.get("inference_contract")),
            )
            checkpoint_path = Path(package.checkpoint_path)

    if checkpoint_path is None or not checkpoint_path.exists():
        if mode == "required":
            raise ValueError(
                "fill-path policy mode 'required' but no loadable checkpoint/package was found"
            )
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }

    model = LearnedFillPathPolicy.from_checkpoint(str(checkpoint_path))
    benchmark_gate_ready = bool(package.benchmark_gate.get("ready", False)) if package is not None else False
    if package is not None:
        setattr(model, "benchmark_gate", dict(package.benchmark_gate))
        setattr(model, "execution_preconditions", dict(package.execution_preconditions))
        setattr(model, "inference_contract", dict(package.inference_contract))
    if mode == "required" and not benchmark_gate_ready:
        raise ValueError("fill-path policy mode 'required' requires a benchmark-gated package")
    return model, {
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
    "FillPathRuntimePackage",
    "load_fill_path_runtime_package",
    "resolve_fill_path_helper",
]
