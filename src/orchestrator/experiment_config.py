import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from src.orchestrator.context import OrchestratorContext, OrchestratorResult


@dataclass
class RunSpec:
    run_id: str
    env_name: str
    engine_type: str
    objective_preset: str
    energy_profile_mix: Dict[str, float]
    data_mix: Dict[str, float]
    notes: str
    pareto_profile_name: Optional[str] = None
    pareto_frontier_summary: Optional[Dict[str, Any]] = None
    rebate_pct: Optional[float] = None
    attributable_spread_capture: Optional[float] = None
    data_premium: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def orchestration_plan_to_run_specs(result: OrchestratorResult, ctx: OrchestratorContext) -> List[RunSpec]:
    """
    Convert an OrchestratorResult into one or more RunSpec instances.
    For now, emit a single run spec capturing the chosen settings.
    """
    notes = (
        f"Env={ctx.env_name}, engine={ctx.engine_type}, preset={result.objective_preset}, "
        f"energy_mix={result.energy_profile_weights}, data_mix={result.data_mix_weights}"
    )
    run = RunSpec(
        run_id=str(uuid.uuid4()),
        env_name=ctx.env_name,
        engine_type=result.chosen_backend,
        objective_preset=result.objective_preset,
        energy_profile_mix=result.energy_profile_weights,
        data_mix=result.data_mix_weights,
        notes=notes,
    )
    return [run]
