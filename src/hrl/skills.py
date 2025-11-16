"""
Skill definitions for HRL Drawer+Vase task.

Defines skill IDs and parameter structures.
"""

from dataclasses import dataclass, field
import numpy as np


class SkillID:
    """
    Enumeration of low-level skills for drawer+vase task.

    These IDs are fixed and should not change for Isaac Gym compatibility.
    """
    LOCATE_DRAWER = 0
    LOCATE_VASE = 1
    PLAN_SAFE_APPROACH = 2
    GRASP_HANDLE = 3
    OPEN_WITH_CLEARANCE = 4
    RETRACT_SAFE = 5

    NUM_SKILLS = 6

    @staticmethod
    def name(skill_id):
        """Get human-readable name for skill ID."""
        names = {
            0: "LOCATE_DRAWER",
            1: "LOCATE_VASE",
            2: "PLAN_SAFE_APPROACH",
            3: "GRASP_HANDLE",
            4: "OPEN_WITH_CLEARANCE",
            5: "RETRACT_SAFE"
        }
        return names.get(skill_id, "UNKNOWN")

    @staticmethod
    def description(skill_id):
        """Get description of skill."""
        descriptions = {
            0: "Locate and orient towards drawer handle",
            1: "Identify vase position and estimate collision risk",
            2: "Compute safe approach path avoiding fragile objects",
            3: "Move to and grasp the drawer handle",
            4: "Pull drawer open while maintaining clearance from vase",
            5: "Retract to safe home position"
        }
        return descriptions.get(skill_id, "Unknown skill")

    @staticmethod
    def all_ids():
        """Get list of all skill IDs."""
        return list(range(SkillID.NUM_SKILLS))


def skill_id_to_name(skill_id):
    """Convenience function for skill name lookup."""
    return SkillID.name(skill_id)


@dataclass
class SkillParams:
    """
    Parameters passed from π_H to π_L for skill execution.

    These parameters modulate skill behavior without changing the core skill.
    """
    # Safety parameters
    target_clearance: float = 0.15  # meters from vase
    min_clearance_threshold: float = 0.08  # emergency stop threshold

    # Motion parameters
    approach_speed: float = 0.8  # normalized [0, 1]
    pull_speed: float = 0.6      # normalized [0, 1]
    retract_speed: float = 0.5   # normalized [0, 1]

    # Grasp parameters
    grasp_force: float = 0.5     # normalized [0, 1]
    grasp_precision: float = 0.05  # meters tolerance

    # Timing
    timeout_steps: int = 100     # max steps for skill

    def to_array(self):
        """Convert to numpy array for policy input."""
        return np.array([
            self.target_clearance,
            self.approach_speed,
            self.grasp_force,
            self.retract_speed,
            self.timeout_steps / 100.0  # normalize
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr):
        """Create from numpy array."""
        return cls(
            target_clearance=float(arr[0]),
            approach_speed=float(arr[1]),
            grasp_force=float(arr[2]),
            retract_speed=float(arr[3]),
            timeout_steps=int(arr[4] * 100)
        )

    @classmethod
    def default_for_skill(cls, skill_id):
        """Get default parameters for specific skill."""
        if skill_id == SkillID.LOCATE_DRAWER:
            return cls(approach_speed=0.3, timeout_steps=20)
        elif skill_id == SkillID.LOCATE_VASE:
            return cls(approach_speed=0.2, timeout_steps=10)
        elif skill_id == SkillID.PLAN_SAFE_APPROACH:
            return cls(target_clearance=0.15, timeout_steps=5)
        elif skill_id == SkillID.GRASP_HANDLE:
            return cls(approach_speed=0.8, grasp_force=0.5, timeout_steps=50)
        elif skill_id == SkillID.OPEN_WITH_CLEARANCE:
            return cls(
                target_clearance=0.15,
                pull_speed=0.6,
                timeout_steps=150
            )
        elif skill_id == SkillID.RETRACT_SAFE:
            return cls(retract_speed=0.5, timeout_steps=50)
        else:
            return cls()


@dataclass
class SkillTrajectory:
    """
    Single skill execution trajectory for training.
    """
    skill_id: int
    skill_params: np.ndarray  # (5,)
    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray       # (T, action_dim)
    rewards: np.ndarray       # (T,)
    dones: np.ndarray         # (T,) bool
    next_observations: np.ndarray  # (T, obs_dim)
    success: bool = False
    total_reward: float = 0.0
    steps: int = 0

    @property
    def length(self):
        return len(self.observations)

    def to_dict(self):
        """Convert to dictionary for saving."""
        return {
            'skill_id': self.skill_id,
            'skill_params': self.skill_params,
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'next_observations': self.next_observations,
            'success': self.success,
            'total_reward': self.total_reward,
            'steps': self.steps,
        }

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(**d)


@dataclass
class SkillSequence:
    """
    Full task trajectory as sequence of skills.
    """
    skill_ids: list = field(default_factory=list)
    skill_params: list = field(default_factory=list)
    skill_trajectories: list = field(default_factory=list)
    task_success: bool = False
    total_steps: int = 0
    total_reward: float = 0.0

    def add_skill_trajectory(self, traj: SkillTrajectory):
        """Add a skill execution to the sequence."""
        self.skill_ids.append(traj.skill_id)
        self.skill_params.append(traj.skill_params)
        self.skill_trajectories.append(traj)
        self.total_steps += traj.steps
        self.total_reward += traj.total_reward

    def get_skill_sequence_str(self):
        """Get human-readable skill sequence."""
        return " -> ".join([SkillID.name(sid) for sid in self.skill_ids])
