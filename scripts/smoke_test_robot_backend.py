#!/usr/bin/env python3
"""
Smoke test for robot backend scaffolding.
"""
import uuid

from src.robot.backend import LocalSimRobotBackend, RobotRunSpec
from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv


def main():
    backend = LocalSimRobotBackend(DrawerVaseArmEnv)
    spec = RobotRunSpec(
        run_id=str(uuid.uuid4()),
        env_name="drawer_vase_arm",
        engine_type="pybullet",
        skill_sequence=[],
        objective_profile={"preset": "throughput"},
        energy_profile_mix={"BASE": 1.0},
        notes="stub robot run",
    )
    result = backend.execute(spec)
    print("Robot run result:", result.to_dict())


if __name__ == "__main__":
    main()
