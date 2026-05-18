"""CPU-local batch execution facade for the sim/synth/physics WM."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .common import mapping, stable_id
from .runtime import (
    SimSynthPhysicsLoopResult,
    SimSynthPhysicsRuntime,
    SimSynthPhysicsRuntimeConfig,
)
from .state import SimSynthPhysicsWorldState


@dataclass(frozen=True)
class VectorizedSimBatchResult:
    """Batch summary over multiple sim/synth planning windows."""

    batch_id: str
    execution_mode: str
    results: list[SimSynthPhysicsLoopResult] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "vectorized_sim_batch_result_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "execution_mode": self.execution_mode,
            "results": [result.to_dict() for result in self.results],
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


class VectorizedSimRunner:
    """Honest local batch runner; vector-shaped now, parallel later."""

    def __init__(self, runtime: Optional[SimSynthPhysicsRuntime] = None) -> None:
        self.runtime = runtime or SimSynthPhysicsRuntime(SimSynthPhysicsRuntimeConfig())

    def execute_world_states(
        self,
        world_states: Sequence[SimSynthPhysicsWorldState],
        *,
        output_root: str | Path | None = None,
    ) -> VectorizedSimBatchResult:
        results: list[SimSynthPhysicsLoopResult] = []
        root = None if output_root is None else Path(output_root)
        for index, world_state in enumerate(world_states):
            output_dir = None
            if root is not None:
                output_dir = root / f"window_{index + 1:03d}_{world_state.state_id}"
            results.append(self.runtime.execute_world_state(world_state, output_dir=output_dir))
        route_status_counts = Counter(
            result.physics_execution_contract.route_status for result in results
        )
        payload = {
            "state_ids": [state.state_id for state in world_states],
            "execution_mode": "sequential_batch",
        }
        return VectorizedSimBatchResult(
            batch_id=stable_id("vectorized_sim_batch", payload),
            execution_mode="sequential_batch",
            results=results,
            metadata={
                "world_state_ids": [state.state_id for state in world_states],
                "result_count": len(results),
                "route_status_counts": dict(route_status_counts),
                "output_root": "" if root is None else str(root),
                "parallelism_posture": "cpu_local_sequential_now_parallelizable_later",
            },
        )


__all__ = ["VectorizedSimBatchResult", "VectorizedSimRunner"]
