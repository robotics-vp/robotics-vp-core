"""
PyBullet Dishwashing environment with an articulated arm (for energy economics/attribution).

This runs in parallel to the existing kinematic env (src/envs/dishwashing_env.py) and
is intended as an "energy bench" to exercise τ·ω-based energy metrics without touching
the frozen Phase B stack.
"""
import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
from typing import List, Dict, Any, Tuple

from src.config.econ_params import EconParams
from src.envs.dishwashing_env import EpisodeInfoSummary, summarize_episode_info

# Limb grouping (KUKA iiwa: 7 revolute joints; rough mapping)
LIMB_GROUPS = {
    "shoulder": [0, 1],
    "elbow": [2, 3],
    "wrist": [4, 5],
    "gripper": [6],
}


class DishwashingArmEnv:
    """
    Simplified dishwashing proxy with an articulated arm controlled in joint space.
    Uses torque/velocity from joint states to populate energy attribution.
    """

    def __init__(self, frames: int = 4, image_size: Tuple[int, int] = (64, 64),
                 max_steps: int = 120, headless: bool = True, econ_params: EconParams = None):
        self.frames = frames
        self.image_height, self.image_width = image_size
        self.max_steps = max_steps
        self.headless = headless
        self.econ_params = econ_params or EconParams(
            price_per_unit=0.3,
            damage_cost=1.0,
            energy_Wh_per_attempt=0.05,
            time_step_s=60.0,
            base_rate=2.0,
            p_min=0.02,
            k_err=0.12,
            q_speed=1.2,
            q_care=1.5,
            care_cost=0.25,
            max_steps=max_steps,
            max_catastrophic_errors=3,
            max_error_rate_sla=0.12,
            min_steps_for_sla=5,
            zero_throughput_patience=10,
            preset="toy",
        )

        self.physics_client = None
        self.robot_id = None
        self.controlled_joint_ids: List[int] = []

        # Episode state
        self.t = 0.0
        self.step_count = 0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.energy_Wh = 0.0
        self.limb_energy_Wh = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_energy_Wh = {}
        self.skill_energy_Wh = {}
        self.limb_power_sum_W = {k: 0.0 for k in LIMB_GROUPS}
        self.limb_peak_power_W = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_power_sum_W = {}
        self.joint_peak_power_W = {}
        self.joint_abs_vel_sum = {}
        self.joint_abs_tau_sum = {}
        self.joint_max_vel = {}
        self.joint_max_tau = {}
        self.joint_dir_counts = {}
        self.effector_energy_Wh = {"ee_main": 0.0}

        self.frame_buffer = deque(maxlen=frames)

    def reset(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        self.physics_client = p.connect(p.DIRECT if self.headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / 240.)

        # Plane
        p.loadURDF("plane.urdf")

        # Load articulated arm (KUKA iiwa)
        self.robot_id = p.loadURDF(
            pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # Controlled joints: all revolute joints
        self.controlled_joint_ids = []
        for j in range(p.getNumJoints(self.robot_id)):
            ji = p.getJointInfo(self.robot_id, j)
            if ji[2] == p.JOINT_REVOLUTE:
                self.controlled_joint_ids.append(j)

        # Disable default motors
        p.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.controlled_joint_ids)
        )

        # Reset episode stats
        self.t = 0.0
        self.step_count = 0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.energy_Wh = 0.0
        self.limb_energy_Wh = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_energy_Wh = {}
        self.skill_energy_Wh = {}
        self.limb_power_sum_W = {k: 0.0 for k in LIMB_GROUPS}
        self.limb_peak_power_W = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_power_sum_W = {}
        self.joint_peak_power_W = {}
        self.joint_abs_vel_sum = {}
        self.joint_abs_tau_sum = {}
        self.joint_max_vel = {}
        self.joint_max_tau = {}
        self.joint_dir_counts = {}
        self.effector_energy_Wh = {"ee_main": 0.0}
        self.frame_buffer.clear()

        return self._get_observation()

    def _get_observation(self):
        # Use joint positions as low-dim observation (no vision for now)
        joint_states = p.getJointStates(self.robot_id, self.controlled_joint_ids)
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        return joint_pos

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        # Map action to joint velocities
        action = np.clip(action, -1.0, 1.0)
        target_vel = action * 1.5  # rad/s
        p.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_vel.tolist(),
            forces=[20.0] * len(self.controlled_joint_ids)
        )

        # Step simulation
        p.stepSimulation()
        dt = 1.0 / 240.0
        self.t += dt
        self.step_count += 1

        # Joint states
        joint_states = p.getJointStates(self.robot_id, self.controlled_joint_ids)
        omega = np.array([s[1] for s in joint_states], dtype=np.float32)
        tau = np.array([s[3] for s in joint_states], dtype=np.float32)

        # Energy attribution per joint
        power = np.maximum(tau * omega, 0.0)  # clip regen to 0
        for j_idx, jid in enumerate(self.controlled_joint_ids):
            name = f"joint_{jid}"
            wh = float(power[j_idx] * dt / 3600.0)
            self.joint_energy_Wh[name] = self.joint_energy_Wh.get(name, 0.0) + wh
            self.joint_power_sum_W[name] = self.joint_power_sum_W.get(name, 0.0) + float(power[j_idx])
            self.joint_peak_power_W[name] = max(self.joint_peak_power_W.get(name, 0.0), float(power[j_idx]))
            self.joint_abs_vel_sum[name] = self.joint_abs_vel_sum.get(name, 0.0) + abs(float(omega[j_idx]))
            self.joint_abs_tau_sum[name] = self.joint_abs_tau_sum.get(name, 0.0) + abs(float(tau[j_idx]))
            self.joint_max_vel[name] = max(self.joint_max_vel.get(name, 0.0), abs(float(omega[j_idx])))
            self.joint_max_tau[name] = max(self.joint_max_tau.get(name, 0.0), abs(float(tau[j_idx])))
            dir_dict = self.joint_dir_counts.setdefault(name, {"pos": 0, "neg": 0})
            if omega[j_idx] >= 0:
                dir_dict["pos"] += 1
            else:
                dir_dict["neg"] += 1

        # Per-limb aggregate
        limb_energy_step = {k: 0.0 for k in LIMB_GROUPS}
        for limb, jids in LIMB_GROUPS.items():
            mask = [self.controlled_joint_ids.index(j) for j in jids if j in self.controlled_joint_ids]
            if mask:
                limb_power = float(np.sum(power[mask]))
            else:
                limb_power = 0.0
            self.limb_power_sum_W[limb] += limb_power
            self.limb_peak_power_W[limb] = max(self.limb_peak_power_W[limb], limb_power)
            limb_energy_step[limb] = limb_power * dt / 3600.0
            self.limb_energy_Wh[limb] += limb_energy_step[limb]

        # Effector
        total_energy_step = sum(limb_energy_step.values())
        self.effector_energy_Wh["ee_main"] += total_energy_step
        self.energy_Wh += total_energy_step

        # Simple task logic: attempts probabilistically, success linked to smooth motion
        attempt_prob = 0.2
        if np.random.rand() < attempt_prob:
            self.attempts += 1
            speed_mag = np.linalg.norm(target_vel)
            error_prob = 0.1 + 0.4 * max(0.0, speed_mag - 0.5)
            if np.random.rand() < error_prob:
                self.errors += 1
            else:
                self.completed += 1

        done = self.step_count >= self.max_steps

        # Build info for EpisodeInfoSummary compatibility
        info = {
            "t": self.t,
            "completed": self.completed,
            "attempts": self.attempts,
            "errors": self.errors,
            "energy_Wh": self.energy_Wh,
            "energy_Wh_per_unit": self.energy_Wh / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
            "units_done": self.completed,
            "limb_energy_Wh": self.limb_energy_Wh,
            "skill_energy_Wh": self.skill_energy_Wh,
            "energy_per_limb": {
                limb: {
                    "Wh": self.limb_energy_Wh[limb],
                    "Wh_per_unit": self.limb_energy_Wh[limb] / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                    "Wh_per_hour": self.limb_energy_Wh[limb] / max(self.t / 3600.0, 1e-6) if self.t > 0 else 0.0,
                    "power_sum_W": self.limb_power_sum_W.get(limb, 0.0),
                    "power_peak_W": self.limb_peak_power_W.get(limb, 0.0),
                }
                for limb in LIMB_GROUPS
            },
            "energy_per_skill": {},
            "energy_per_joint": {
                name: {
                    "Wh": self.joint_energy_Wh.get(name, 0.0),
                    "Wh_per_unit": self.joint_energy_Wh.get(name, 0.0) / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                    "Wh_per_hour": self.joint_energy_Wh.get(name, 0.0) / max(self.t / 3600.0, 1e-6) if self.t > 0 else 0.0,
                    "avg_power_W": self.joint_power_sum_W.get(name, 0.0) / max(self.step_count, 1),
                    "peak_power_W": self.joint_peak_power_W.get(name, 0.0),
                    "avg_abs_velocity": self.joint_abs_vel_sum.get(name, 0.0) / max(self.step_count, 1),
                    "max_abs_velocity": self.joint_max_vel.get(name, 0.0),
                    "avg_abs_torque": self.joint_abs_tau_sum.get(name, 0.0) / max(self.step_count, 1),
                    "max_abs_torque": self.joint_max_tau.get(name, 0.0),
                    "directionality": self.joint_dir_counts.get(name, {"pos": 0, "neg": 0}),
                }
                for name in self.joint_energy_Wh.keys()
            },
            "energy_per_effector": {
                "ee_main": {
                    "Wh": self.effector_energy_Wh["ee_main"],
                    "Wh_per_unit": self.effector_energy_Wh["ee_main"] / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                    "Wh_per_hour": self.effector_energy_Wh["ee_main"] / max(self.t / 3600.0, 1e-6) if self.t > 0 else 0.0,
                }
            },
            "coordination_metrics": {
                "mean_active_joints": float(np.mean(np.abs(target_vel) > 0.1)),
                "mean_joint_velocity_corr": 0.0,
            },
            "terminated_reason": "max_steps" if done else None,
            "mpl_t": (self.completed / (self.t / 3600.0)) if self.t > 0 else 0.0,
            "ep_t": (self.completed / self.energy_Wh) if self.energy_Wh > 0 else 0.0,
        }

        obs = self._get_observation()
        return obs, info, done

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
