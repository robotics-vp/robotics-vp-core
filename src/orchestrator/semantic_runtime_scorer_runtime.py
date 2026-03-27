"""Runtime package loading for semantic runtime scorers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.orchestrator.semantic_runtime_scorers import (
    SemanticRuntimeScorerPackage,
    load_semantic_runtime_scorer_package,
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


@dataclass(frozen=True)
class SemanticRuntimeScorerRuntimePackage:
    package_id: str
    package_path: str
    scorer_package_path: str
    torch_checkpoint_path: str
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_semantic_runtime_scorer_runtime_package(
    path: str | Path,
) -> SemanticRuntimeScorerRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    scorer_package_path = Path(str(payload.get("scorer_package_path") or ""))
    if scorer_package_path and not scorer_package_path.is_absolute():
        scorer_package_path = (package_path.parent / scorer_package_path).resolve()
    torch_checkpoint_path = Path(str(payload.get("torch_checkpoint_path") or ""))
    if torch_checkpoint_path and not torch_checkpoint_path.is_absolute():
        torch_checkpoint_path = (package_path.parent / torch_checkpoint_path).resolve()
    return SemanticRuntimeScorerRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        scorer_package_path=str(scorer_package_path),
        torch_checkpoint_path=str(torch_checkpoint_path),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        inference_contract=_mapping(payload.get("inference_contract")),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=_mapping(payload.get("metadata")),
    )


def load_semantic_runtime_scorer_from_runtime_package(
    path: str | Path,
) -> tuple[SemanticRuntimeScorerPackage, SemanticRuntimeScorerRuntimePackage]:
    package = load_semantic_runtime_scorer_runtime_package(path)
    scorer_package_path = Path(package.scorer_package_path)
    if not scorer_package_path.exists():
        raise FileNotFoundError(
            f"semantic runtime scorer package not found: {scorer_package_path}"
        )
    scorer_package = load_semantic_runtime_scorer_package(scorer_package_path)
    return scorer_package, package


__all__ = [
    "SemanticRuntimeScorerRuntimePackage",
    "load_semantic_runtime_scorer_from_runtime_package",
    "load_semantic_runtime_scorer_runtime_package",
]
