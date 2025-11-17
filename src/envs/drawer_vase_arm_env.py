"""
Drawer+Vase articulated arm environment (energy bench).

Simplified version that uses an articulated arm (KUKA iiwa) and a goal geometry
instead of the full PyBullet drawer mechanics. Energy attribution uses real joint
τ·ω to populate metrics for analysis.
"""
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Any, Tuple

from src.config.econ_params import EconParams
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.envs.dishwashing_env import summarize_episode_info as summarize_dish  # reuse structure

LIMB_GROUPS = {
    "shoulder": [0, 1],
    "elbow": [2, 3],
    "wrist": [4, 5],
    "gripper": [6],
}


class DrawerVaseArmEnv:
    """
    Simplified drawer+vase task with an articulated arm.
    Success: move EE to target box; error: collide near vase region.
    """

    def __init__(self, max_steps: int = 200, headless: bool = True, econ_params: EconParams = None):
        self.max_steps = max_steps
        self.headless = headless
        # Provide a lightweight default so the env can be instantiated without a config.
        self.econ_params = econ_params or EconParams(
            price_per_unit=0.30,
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
        self.controlled_joint_ids = []
        self.ee_link_id = None

        # Episode state
        self.t = 0.0
        self.step_count = 0
        self.success = False
        self.vase_intact = True
        self.energy_Wh = 0.0
        self.limb_energy_Wh = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_energy_Wh = {}
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
        self.skill_energy_Wh = {}
        self.completed = 0
        self.attempts = 0
        self.errors = 0

    def reset(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.DIRECT if self.headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / 240.)

        p.loadURDF("plane.urdf")

        # Load arm
        self.robot_id = p.loadURDF(
            pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        self.controlled_joint_ids = []
        for j in range(p.getNumJoints(self.robot_id)):
            ji = p.getJointInfo(self.robot_id, j)
            if ji[2] == p.JOINT_REVOLUTE:
                self.controlled_joint_ids.append(j)
        self.ee_link_id = self.controlled_joint_ids[-1] if self.controlled_joint_ids else None
        p.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.controlled_joint_ids)
        )

        # Task geometry (vase sits close to the drawer target to induce risk)
        self.drawer_target = np.array([0.55, 0.0, 0.4])
        self.vase_pos = np.array([0.55, 0.0, 0.4])

        # Reset episode stats
        self.t = 0.0
        self.step_count = 0
        self.success = False
        self.vase_intact = True
        self.energy_Wh = 0.0
        self.limb_energy_Wh = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_energy_Wh = {}
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
        self.skill_energy_Wh = {}
        self.completed = 0
        self.attempts = 0
        self.errors = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, self.controlled_joint_ids)
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        return joint_pos

    def _ee_state(self):
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_id,
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        pos = np.array(link_state[0])
        vel = np.array(link_state[6]) if len(link_state) > 6 else np.zeros(3)
        return pos, vel

    def compute_ik_targets(self, target_pos, target_orn=None):
        """Compute IK joint targets for the end-effector."""
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([0, 0, 0])
        ik = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_id,
            target_pos,
            targetOrientation=target_orn,
        )
        return np.array(ik[: len(self.controlled_joint_ids)], dtype=np.float32)

    def _apply_position_control(self, joint_targets, speed_scale=1.0):
        """Position control helper used by scripted controller."""
        n = len(self.controlled_joint_ids)
        gains = [0.5 * speed_scale] * n
        forces = [200.0 * speed_scale] * n
        p.setJointMotorControlArray(
            self.robot_id,
            self.controlled_joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_targets.tolist(),
            positionGains=gains,
            forces=forces,
        )

    def step(self, action: Any):
        """
        Supports two modes:
        - velocity control: action is np.ndarray shaped (n_joints,)
        - position control: action is dict with 'target_pos' (xyz) and optional 'speed_scale'
        """
        if isinstance(action, dict) and "target_pos" in action:
            target_pos = np.asarray(action["target_pos"], dtype=np.float32)
            speed_scale = float(action.get("speed_scale", 1.0))
            joint_targets = self.compute_ik_targets(target_pos, action.get("target_orn", None))
            self._apply_position_control(joint_targets, speed_scale=speed_scale)
        else:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            target_vel = action * 1.5
            p.setJointMotorControlArray(
                self.robot_id,
                self.controlled_joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=target_vel.tolist(),
                forces=[20.0] * len(self.controlled_joint_ids)
            )
        p.stepSimulation()
        dt = 1.0 / 240.0
        self.t += dt
        self.step_count += 1

        # Energy attribution
        joint_states = p.getJointStates(self.robot_id, self.controlled_joint_ids)
        omega = np.array([s[1] for s in joint_states], dtype=np.float32)
        tau = np.array([s[3] for s in joint_states], dtype=np.float32)
        power = np.maximum(tau * omega, 0.0)
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

        limb_energy_step = {k: 0.0 for k in LIMB_GROUPS}
        for limb, jids in LIMB_GROUPS.items():
            mask = [self.controlled_joint_ids.index(j) for j in jids if j in self.controlled_joint_ids]
            limb_power = float(np.sum(power[mask])) if mask else 0.0
            self.limb_power_sum_W[limb] += limb_power
            self.limb_peak_power_W[limb] = max(self.limb_peak_power_W[limb], limb_power)
            limb_energy_step[limb] = limb_power * dt / 3600.0
            self.limb_energy_Wh[limb] += limb_energy_step[limb]
        total_energy_step = sum(limb_energy_step.values())
        self.effector_energy_Wh["ee_main"] += total_energy_step
        self.energy_Wh += total_energy_step

        # Task logic: success if EE reaches drawer target; error if near vase at high speed
        ee_pos, ee_vel = self._ee_state()
        dist_to_target = np.linalg.norm(ee_pos - self.drawer_target)
        if dist_to_target < 0.15 and not self.success:
            self.success = True
            self.completed += 1
        dist_to_vase = np.linalg.norm(ee_pos - self.vase_pos)
        speed = np.linalg.norm(ee_vel)
        if dist_to_vase < 0.12 and speed > 0.8:
            self.vase_intact = False
            self.errors += 1
        elif dist_to_vase < 0.15 and speed > 0.6:
            # Near-miss to make error rate non-trivial without breaking vase
            self.errors += 1
        elif dist_to_vase < 0.18:
            # Stochastic near-miss to create non-trivial error distribution
            prob = min(1.0, speed / 1.0) * 0.2
            if np.random.rand() < prob:
                self.errors += 1
                if speed > 0.9:
                    self.vase_intact = False

        self.attempts += 1

        done = self.step_count >= self.max_steps or self.success or not self.vase_intact

        info = self._get_info()
        obs = self._get_obs()
        return obs, 0.0, done, False, info

    def _get_info(self) -> Dict[str, Any]:
        time_hours = self.t / 3600.0
        mpl_t = (self.completed / time_hours) if time_hours > 0 else 0.0
        ep_t = (self.completed / self.energy_Wh) if self.energy_Wh > 0 else 0.0

        info = {
            "t": self.t,
            "completed": self.completed,
            "attempts": self.attempts,
            "errors": self.errors,
            "energy_Wh": self.energy_Wh,
            "energy_Wh_per_unit": self.energy_Wh / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
            "limb_energy_Wh": self.limb_energy_Wh,
            "skill_energy_Wh": self.skill_energy_Wh,
            "energy_per_limb": {
                limb: {
                    "Wh": self.limb_energy_Wh[limb],
                    "Wh_per_unit": self.limb_energy_Wh[limb] / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                    "Wh_per_hour": self.limb_energy_Wh[limb] / max(time_hours, 1e-6) if time_hours > 0 else 0.0,
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
                    "Wh_per_hour": self.joint_energy_Wh.get(name, 0.0) / max(time_hours, 1e-6) if time_hours > 0 else 0.0,
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
                    "Wh_per_hour": self.effector_energy_Wh["ee_main"] / max(time_hours, 1e-6) if time_hours > 0 else 0.0,
                }
            },
            "coordination_metrics": {
                "mean_active_joints": float(np.mean(np.abs([0.0]))),
                "mean_joint_velocity_corr": 0.0,
            },
            "terminated_reason": "success" if self.success else ("vase_broken" if not self.vase_intact else ("max_steps" if self.step_count >= self.max_steps else None)),
            "mpl_t": mpl_t,
            "ep_t": ep_t,
            "success": self.success,
            "vase_intact": self.vase_intact,
        }
        return info

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
