from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RobotRunSpec:
    run_id: str
    env_name: str
    engine_type: str
    skill_sequence: List[Any]
    objective_profile: Optional[Dict[str, Any]] = None
    energy_profile_mix: Optional[Dict[str, float]] = None
    notes: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class RobotRunResult:
    run_id: str
    success: bool
    metrics: Dict[str, float]
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class RobotBackend(ABC):
    @abstractmethod
    def execute(self, spec: RobotRunSpec) -> RobotRunResult:
        ...


class LocalSimRobotBackend(RobotBackend):
    def __init__(self, env_ctor):
        self.env_ctor = env_ctor

    def execute(self, spec: RobotRunSpec) -> RobotRunResult:
        env = self.env_ctor()
        done = False
        obs, info = env.reset()
        steps = 0
        while not done and steps < getattr(env, "max_steps", 200):
            if spec.skill_sequence:
                action = spec.skill_sequence[min(steps, len(spec.skill_sequence) - 1)]
            elif hasattr(env, "action_space"):
                action = env.action_space.sample()
            else:
                import numpy as np
                n = len(getattr(env, "controlled_joint_ids", [])) or 3
                action = np.zeros(n, dtype=float)
            obs, _, done, truncated, info = env.step(action)
            if truncated:
                done = True
            steps += 1
        success = info.get("success", False) or info.get("completed", 0) > 0
        metrics = {
            "mpl": info.get("mpl_t", 0.0),
            "error": info.get("errors", 0.0),
            "energy_Wh": info.get("energy_Wh", 0.0),
        }
        return RobotRunResult(run_id=spec.run_id, success=success, metrics=metrics)


class CloudRobotBackendStub(RobotBackend):
    def execute(self, spec: RobotRunSpec) -> RobotRunResult:
        # Stub: log and return dummy result
        metrics = {"mpl": 0.0, "error": 0.0, "energy_Wh": 0.0}
        return RobotRunResult(run_id=spec.run_id, success=False, metrics=metrics, notes="Cloud backend stub")
