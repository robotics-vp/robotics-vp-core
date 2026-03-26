"""Runtime package loading for the learned gap ranker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from src.world_model.gap_ranker import LearnedGapRanker


@dataclass(frozen=True)
class GapRankerRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    metadata: Dict[str, Any]


def _mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def load_gap_ranker_runtime_package(path: str | Path) -> GapRankerRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    return GapRankerRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(payload.get("checkpoint_path") or ""),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        metadata=_mapping(payload.get("metadata")),
    )


def resolve_gap_ranker_helper(
    gap_ranker: Any,
    *,
    mode: Literal["disabled", "auto", "required"] = "auto",
) -> tuple[Optional[Any], Dict[str, Any]]:
    if mode not in {"disabled", "auto", "required"}:
        raise ValueError(f"Unsupported gap-ranker mode: {mode}")
    if mode == "disabled":
        return None, {
            "mode": mode,
            "status": "disabled",
            "promotion_stage": "disabled",
            "benchmark_gate_ready": False,
        }
    if gap_ranker is None:
        if mode == "required":
            raise ValueError("gap-ranker mode 'required' but no gap ranker was provided")
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }
    if hasattr(gap_ranker, "rank_edges"):
        benchmark_gate_ready = bool(
            getattr(gap_ranker, "benchmark_gate", {}).get("ready", False)
            if hasattr(gap_ranker, "benchmark_gate")
            else False
        )
        if mode == "required" and not benchmark_gate_ready:
            raise ValueError("gap-ranker mode 'required' requires a benchmark-gated package")
        return gap_ranker, {
            "mode": mode,
            "status": "loaded_direct",
            "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
            "benchmark_gate_ready": benchmark_gate_ready,
        }

    package: Optional[GapRankerRuntimePackage] = None
    checkpoint_path: Optional[Path] = None
    if isinstance(gap_ranker, (str, Path)):
        candidate = Path(gap_ranker)
        if candidate.suffix == ".json":
            package = load_gap_ranker_runtime_package(candidate)
            checkpoint_path = Path(package.checkpoint_path)
        else:
            checkpoint_path = candidate
    elif isinstance(gap_ranker, Mapping):
        if "checkpoint_path" in gap_ranker:
            package = GapRankerRuntimePackage(
                package_id=str(gap_ranker.get("package_id", "gap_ranker_package")),
                package_path=str(gap_ranker.get("package_path", "")),
                checkpoint_path=str(gap_ranker.get("checkpoint_path", "")),
                benchmark_gate=_mapping(gap_ranker.get("benchmark_gate")),
                execution_preconditions=_mapping(gap_ranker.get("execution_preconditions")),
                metadata=_mapping(gap_ranker.get("metadata")),
            )
            checkpoint_path = Path(package.checkpoint_path)

    if checkpoint_path is None or not checkpoint_path.exists():
        if mode == "required":
            raise ValueError("gap-ranker mode 'required' but no loadable checkpoint/package was found")
        return None, {
            "mode": mode,
            "status": "package_missing",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }

    model = LearnedGapRanker.from_checkpoint(str(checkpoint_path))
    benchmark_gate_ready = bool(package.benchmark_gate.get("ready", False)) if package is not None else False
    if package is not None:
        setattr(model, "benchmark_gate", dict(package.benchmark_gate))
        setattr(model, "execution_preconditions", dict(package.execution_preconditions))
    if mode == "required" and not benchmark_gate_ready:
        raise ValueError("gap-ranker mode 'required' requires a benchmark-gated package")
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
    "GapRankerRuntimePackage",
    "load_gap_ranker_runtime_package",
    "resolve_gap_ranker_helper",
]
