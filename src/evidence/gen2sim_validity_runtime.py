"""Runtime package loading for learned gen2sim validity helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from src.evidence.gen2sim_validity_training import (
    LearnedGen2SimValidityModel,
    TORCH_AVAILABLE,
)


@dataclass(frozen=True)
class Gen2SimValidityRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def _mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def load_gen2sim_validity_runtime_package(path: str | Path) -> Gen2SimValidityRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    return Gen2SimValidityRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(payload.get("checkpoint_path") or ""),
        model_config=_mapping(payload.get("model_config")),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        inference_contract=_mapping(payload.get("inference_contract")),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=_mapping(payload.get("metadata")),
    )


def resolve_gen2sim_validity_helper(
    helper: Any,
    *,
    mode: Literal["disabled", "auto", "required"] = "auto",
) -> tuple[Optional[Any], Dict[str, Any]]:
    if mode not in {"disabled", "auto", "required"}:
        raise ValueError(f"Unsupported gen2sim validity mode: {mode}")
    if mode == "disabled":
        return None, {
            "mode": mode,
            "status": "disabled",
            "promotion_stage": "disabled",
            "benchmark_gate_ready": False,
        }
    if helper is None:
        if mode == "required":
            raise ValueError("gen2sim validity mode 'required' but no helper was provided")
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }
    if hasattr(helper, "predict_context") or hasattr(helper, "infer_context"):
        benchmark_gate_ready = bool(
            getattr(helper, "benchmark_gate", {}).get("ready", False)
            if hasattr(helper, "benchmark_gate")
            else False
        )
        if mode == "required" and not benchmark_gate_ready:
            raise ValueError("gen2sim validity mode 'required' requires a benchmark-gated package")
        return helper, {
            "mode": mode,
            "status": "loaded_direct",
            "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
            "benchmark_gate_ready": benchmark_gate_ready,
        }
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load a learned gen2sim validity helper")

    package: Optional[Gen2SimValidityRuntimePackage] = None
    checkpoint_path: Optional[Path] = None
    if isinstance(helper, (str, Path)):
        candidate = Path(helper)
        if candidate.suffix == ".json":
            package = load_gen2sim_validity_runtime_package(candidate)
            checkpoint_path = Path(package.checkpoint_path)
        else:
            checkpoint_path = candidate
    elif isinstance(helper, Mapping) and "checkpoint_path" in helper:
        package = Gen2SimValidityRuntimePackage(
            package_id=str(helper.get("package_id", "gen2sim_validity_package")),
            package_path=str(helper.get("package_path", "")),
            checkpoint_path=str(helper.get("checkpoint_path", "")),
            model_config=_mapping(helper.get("model_config")),
            benchmark_gate=_mapping(helper.get("benchmark_gate")),
            execution_preconditions=_mapping(helper.get("execution_preconditions")),
            inference_contract=_mapping(helper.get("inference_contract")),
            promotion_stage=str(helper.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
            metadata=_mapping(helper.get("metadata")),
        )
        checkpoint_path = Path(package.checkpoint_path)

    if checkpoint_path is None or not checkpoint_path.exists():
        if mode == "required":
            raise ValueError(
                "gen2sim validity mode 'required' but no loadable checkpoint/package was found"
            )
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }

    model = LearnedGen2SimValidityModel.from_checkpoint(str(checkpoint_path))
    benchmark_gate_ready = bool(package.benchmark_gate.get("ready", False)) if package else False
    if package is not None:
        setattr(model, "benchmark_gate", dict(package.benchmark_gate))
        setattr(model, "execution_preconditions", dict(package.execution_preconditions))
        setattr(model, "promotion_stage", str(package.promotion_stage or "shadow_candidate"))
        setattr(model, "inference_contract", dict(package.inference_contract))
    if mode == "required" and not benchmark_gate_ready:
        raise ValueError("gen2sim validity mode 'required' requires a benchmark-gated package")
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
    "Gen2SimValidityRuntimePackage",
    "load_gen2sim_validity_runtime_package",
    "resolve_gen2sim_validity_helper",
]
