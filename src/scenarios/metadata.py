from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.motor_backend.datapacks import DatapackConfig
from src.objectives.economic_objective import EconomicObjectiveSpec


@dataclass(frozen=True)
class ScenarioMetadata:
    scenario_id: str
    task_id: str
    motor_backend: str
    objective_name: str
    objective_weights: Mapping[str, float]
    datapack_ids: Sequence[str]
    datapack_tags: Sequence[str]
    task_tags: Sequence[str]
    robot_families: Sequence[str]
    notes: str | None = None


def build_scenario_metadata(
    *,
    run_id: str,
    task_id: str,
    motor_backend: str,
    objective_name: str,
    objective: EconomicObjectiveSpec,
    datapacks: Sequence[DatapackConfig],
    notes: str | None = None,
) -> ScenarioMetadata:
    scenario_id = f"{motor_backend}:{task_id}:{objective_name}:{run_id}"

    datapack_ids = [d.id for d in datapacks]
    dp_tags: list[str] = []
    task_tags: list[str] = []
    robot_families: list[str] = []
    for d in datapacks:
        dp_tags.extend(d.tags)
        task_tags.extend(d.task_tags)
        robot_families.extend(d.robot_families)

    weights: dict[str, float] = {
        "mpl_weight": objective.mpl_weight,
        "energy_weight": objective.energy_weight,
        "error_weight": objective.error_weight,
        "novelty_weight": objective.novelty_weight,
        "risk_weight": objective.risk_weight,
    }
    if objective.extra_weights:
        weights.update(objective.extra_weights)

    return ScenarioMetadata(
        scenario_id=scenario_id,
        task_id=task_id,
        motor_backend=motor_backend,
        objective_name=objective_name,
        objective_weights=weights,
        datapack_ids=datapack_ids,
        datapack_tags=sorted(set(dp_tags)),
        task_tags=sorted(set(task_tags)),
        robot_families=sorted(set(robot_families)),
        notes=notes,
    )
