#!/usr/bin/env python3
"""
Smoke: orchestrator plan → robot run specs (dry-run).
"""
import json
import uuid
from src.orchestrator.context import build_orchestrator_context_from_datapacks
from src.orchestrator.orchestration_transformer import OrchestrationTransformer, propose_orchestrated_plan
from src.orchestrator.experiment_config import orchestration_plan_to_run_specs
from src.robot.backend import LocalSimRobotBackend, CloudRobotBackendStub, RobotRunSpec
from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv


def main():
    ctx = build_orchestrator_context_from_datapacks(
        base_dir="data/datapacks",
        env_name="drawer_vase_arm",
        engine_type="pybullet",
        task_type="fragility",
        customer_segment="industrial_high_wage",
        market_region="US_NE",
        interventions_path="data/energy_interventions.jsonl" if True else None,
    )
    model = OrchestrationTransformer()
    plan = propose_orchestrated_plan(model, ctx, "open drawer safely", steps=2)
    run_specs = orchestration_plan_to_run_specs(plan, ctx)
    robot_specs = [
        RobotRunSpec(
            run_id=rs.run_id + "-robot",
            env_name=rs.env_name,
            engine_type=rs.engine_type,
            skill_sequence=[],
            objective_profile={"preset": rs.objective_preset},
            energy_profile_mix=rs.energy_profile_mix,
            notes="dry-run robot spec",
        )
        for rs in run_specs
    ]
    local = LocalSimRobotBackend(DrawerVaseArmEnv)
    cloud = CloudRobotBackendStub()
    for spec in robot_specs[:2]:
        res_local = local.execute(spec)  # still runs sim but fine for smoke
        res_cloud = cloud.execute(spec)
        print("Spec:", json.dumps(spec.to_dict(), indent=2))
        print("Local result:", res_local.to_dict())
        print("Cloud stub result:", res_cloud.to_dict())


if __name__ == "__main__":
    main()
