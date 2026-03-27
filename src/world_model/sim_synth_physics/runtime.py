"""Runtime facade for the sim/synth/physics world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

from .compiler import compile_sim_synth_physics_world_state
from .diffusion_contracts import GapDrivenDiffusionPlan, compile_gap_driven_diffusion_plans
from .promotion import HelperMode
from .state import SimSynthPhysicsWorldState


@dataclass(frozen=True)
class SimSynthPhysicsRuntimeConfig:
    """Configuration for compiling sim/synth/physics WM state."""

    economic_weight: float = 1.0
    trust_weight: float = 1.0
    readiness_weight: float = 1.0
    agenda_limit: int = 10
    default_backend: str = "pybullet"
    default_objective: str = "balanced"
    gap_ranker_mode: Literal["disabled", "auto", "required"] = "auto"
    backend_selector_mode: HelperMode = "auto"
    branch_planner_mode: HelperMode = "auto"


class SimSynthPhysicsRuntime:
    """Compiler/runtime boundary for the sim/synth/physics WM."""

    def __init__(self, config: Optional[SimSynthPhysicsRuntimeConfig] = None) -> None:
        self.config = config or SimSynthPhysicsRuntimeConfig()

    def compile_world_state(
        self,
        coverage_graph: Any,
        *,
        semantic_context: Optional[Mapping[str, Any]] = None,
        economic_context: Optional[Mapping[str, Any]] = None,
        embodiment_context: Optional[Mapping[str, Any]] = None,
        benchmark_signals: Optional[Mapping[str, Any]] = None,
        gap_ranker: Any = None,
        backend_selector: Any = None,
        branch_planner: Any = None,
    ) -> SimSynthPhysicsWorldState:
        return compile_sim_synth_physics_world_state(
            coverage_graph,
            semantic_context=semantic_context,
            economic_context=economic_context,
            embodiment_context=embodiment_context,
            benchmark_signals=benchmark_signals,
            economic_weight=self.config.economic_weight,
            trust_weight=self.config.trust_weight,
            readiness_weight=self.config.readiness_weight,
            limit=self.config.agenda_limit,
            default_backend=self.config.default_backend,
            default_objective=self.config.default_objective,
            gap_ranker=gap_ranker,
            gap_ranker_mode=self.config.gap_ranker_mode,
            backend_selector=backend_selector,
            backend_selector_mode=self.config.backend_selector_mode,
            branch_planner=branch_planner,
            branch_planner_mode=self.config.branch_planner_mode,
        )

    def compile_diffusion_plans(
        self,
        world_state: SimSynthPhysicsWorldState,
        *,
        coverage_graph: Any = None,
        limit: Optional[int] = None,
    ) -> list[GapDrivenDiffusionPlan]:
        return compile_gap_driven_diffusion_plans(
            world_state,
            coverage_graph=coverage_graph,
            limit=limit,
        )
