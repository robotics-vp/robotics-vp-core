import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.config.econ_params import EconParams

# Backward compatibility alias
DishwashingParams = EconParams

# Limb grouping placeholder (no explicit joints; treat all energy as shoulder/base)
LIMB_GROUPS = {
    "shoulder": [],
    "elbow": [],
    "wrist": [],
    "gripper": [],
}
# Placeholder joint names (non-physics env)
JOINT_NAMES = ["joint_0"]
class DishwashingEnv:
    """
    Dishwashing environment with 2D action space: (speed, care).

    Action:
        speed ∈ [0,1]: How fast to work (affects throughput)
        care ∈ [0,1]: How careful to be (reduces errors, costs throughput)

    Error Model:
        p_err = p_min + k * speed^q_s * (1-care)^q_c

    This makes 6% SLA feasible: speed≈0.6, care≈0.6 → MP≈90-110/hr, err≈5-7%
    """
    def __init__(self, params: EconParams):
        self.p = params
        self.reset()

    def reset(self):
        self.steps = 0
        self.t = 0.0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
        self.catastrophic_errors = 0
        self.energy_Wh = 0.0
        self.limb_energy_Wh = {
            "shoulder": 0.0,
            "elbow": 0.0,
            "wrist": 0.0,
            "gripper": 0.0,
        }
        self.limb_power_sum_W = {k: 0.0 for k in LIMB_GROUPS}
        self.limb_peak_power_W = {k: 0.0 for k in LIMB_GROUPS}
        self.joint_energy_Wh = {name: 0.0 for name in JOINT_NAMES}
        self.joint_power_sum_W = {name: 0.0 for name in JOINT_NAMES}
        self.joint_peak_power_W = {name: 0.0 for name in JOINT_NAMES}
        self.joint_abs_vel_sum = {name: 0.0 for name in JOINT_NAMES}
        self.joint_abs_tau_sum = {name: 0.0 for name in JOINT_NAMES}
        self.joint_max_vel = {name: 0.0 for name in JOINT_NAMES}
        self.joint_max_tau = {name: 0.0 for name in JOINT_NAMES}
        self.joint_dir_counts = {name: {"pos": 0, "neg": 0} for name in JOINT_NAMES}
        self.effector_energy_Wh = {"ee_main": 0.0}
        self.skill_energy_Wh = {}
        obs = self._obs()
        return obs

    def _obs(self):
        return {
            "t": self.t,
            "completed": self.completed,
            "attempts": self.attempts,
            "errors": self.errors
        }

    def step(self, action):
        """
        Step with 2D action: (speed, care).

        Args:
            action: array-like [speed, care], both in [0, 1]
        """
        prev_completed = self.completed
        prev_errors = self.errors
        prev_energy = self.energy_Wh

        # Parse 2D action
        if np.isscalar(action):
            # Fallback for 1D action (backward compat)
            speed, care = float(np.clip(action, 0.0, 1.0)), 0.5
        else:
            speed = float(np.clip(action[0], 0.0, 1.0))
            care = float(np.clip(action[1], 0.0, 1.0))

        # Attempts per minute (Poisson)
        rate_per_min = max(0.1, self.p.base_rate * (0.5 + 0.5 * speed))

        # Care reduces throughput (effort cost)
        rate_per_min *= (1.0 - self.p.care_cost * care)

        # Convert to per-step rate
        time_step_minutes = self.p.time_step_s / 60.0
        rate_per_step = rate_per_min * time_step_minutes

        attempts = max(1, int(np.random.poisson(rate_per_step)))
        self.attempts += attempts

        # Error probability with controllability
        # p_err = p_min + k * speed^q_s * (1-care)^q_c
        p_err = self.p.p_min + self.p.k_err * (speed ** self.p.q_speed) * ((1.0 - care) ** self.p.q_care)
        p_err = float(np.clip(p_err, 0.0, 0.5))

        errs = np.random.binomial(attempts, p_err)
        self.errors += errs

        # Successes
        succ = max(attempts - errs, 0)
        self.completed += succ

        # Catastrophic errors: all attempts failed in a step
        if attempts > 0 and errs == attempts:
            self.catastrophic_errors += 1

        # Energy accounting (simple proportional model)
        delta_energy_Wh = attempts * self.p.energy_Wh_per_attempt
        self.energy_Wh += delta_energy_Wh
        # Per-limb power/energy (no joint torques; attribute all to shoulder/base)
        dt = self.p.time_step_s
        power_total_W = delta_energy_Wh * 3600.0 / max(dt, 1e-6)
        limb_power_W = {limb: power_total_W if limb == "shoulder" else 0.0 for limb in LIMB_GROUPS}
        for limb, pwr in limb_power_W.items():
            self.limb_power_sum_W[limb] += pwr
            self.limb_peak_power_W[limb] = max(self.limb_peak_power_W[limb], pwr)
            self.limb_energy_Wh[limb] += pwr * dt / 3600.0
        # Skill-level energy (if HRL sets current_skill_id)
        if hasattr(self, "current_skill_id") and self.current_skill_id:
            skill_map = self.skill_energy_Wh.setdefault(self.current_skill_id, {k: 0.0 for k in LIMB_GROUPS})
            for limb, pwr in limb_power_W.items():
                skill_map[limb] += pwr * dt / 3600.0
        # Per-joint (placeholder zeros; no torques in this env)
        for jn in JOINT_NAMES:
            tau = 0.0
            omega = 0.0
            power = max(tau * omega, 0.0)
            self.joint_energy_Wh[jn] += power * dt / 3600.0
            self.joint_power_sum_W[jn] += power
            self.joint_peak_power_W[jn] = max(self.joint_peak_power_W[jn], power)
            self.joint_abs_vel_sum[jn] += abs(omega)
            self.joint_abs_tau_sum[jn] += abs(tau)
            self.joint_max_vel[jn] = max(self.joint_max_vel[jn], abs(omega))
            self.joint_max_tau[jn] = max(self.joint_max_tau[jn], abs(tau))
            if omega >= 0:
                self.joint_dir_counts[jn]["pos"] += 1
            else:
                self.joint_dir_counts[jn]["neg"] += 1
        # Effector energy (single effector)
        self.effector_energy_Wh["ee_main"] += delta_energy_Wh

        # Time and counters
        self.t += self.p.time_step_s
        self.steps += 1

        # Deltas and productivity metrics
        delta_units = self.completed - prev_completed
        delta_errors = self.errors - prev_errors
        dt_hours = self.p.time_step_s / 3600.0
        mpl_t = (delta_units / dt_hours) if dt_hours > 0 else 0.0
        if delta_energy_Wh > 0:
            ep_t = delta_units / delta_energy_Wh
        else:
            ep_t = 0.0

        # Early termination logic
        done = False
        terminated_reason = None

        if self.catastrophic_errors >= self.p.max_catastrophic_errors:
            done = True
            terminated_reason = "catastrophic_error"

        if (not done) and self.steps >= self.p.min_steps_for_sla and self.completed > 0:
            current_error_rate = self.errors / max(1, self.completed)
            if current_error_rate > self.p.max_error_rate_sla:
                done = True
                terminated_reason = "sla_violation"

        if (not done) and self.steps >= self.p.zero_throughput_patience and self.completed == 0:
            done = True
            terminated_reason = "zero_throughput"

        if self.steps >= self.p.max_steps:
            done = True
            if terminated_reason is None:
                terminated_reason = "max_steps"

        obs = self._obs()
        revenue_step = self.p.price_per_unit * delta_units
        error_cost_step = self.p.damage_cost * delta_errors
        profit_step = revenue_step - error_cost_step

        info = {
            "succ": succ,
            "errs": errs,
            "p_err": p_err,
            "speed": speed,
            "care": care,
            "rate_per_min": rate_per_min,
            "t": self.t,
            "delta_units": delta_units,
            "delta_energy_Wh": delta_energy_Wh,
            "delta_errors": delta_errors,
            "mpl_t": mpl_t,
            "ep_t": ep_t,
            "error_rate_t": self.errors / max(1, self.completed) if self.completed > 0 else 0.0,
            "units_done": self.completed,
            "errors": self.errors,
            "energy_Wh": self.energy_Wh,
            "energy_Wh_per_unit": delta_energy_Wh / max(delta_units, 1e-6) if delta_units > 0 else 0.0,
            "current_skill_id": getattr(self, "current_skill_id", None),
            "limb_energy_Wh": self.limb_energy_Wh,
            "skill_energy_Wh": self.skill_energy_Wh,
            "joint_energy_Wh": self.joint_energy_Wh,
            "terminated_reason": terminated_reason,
            "catastrophic_errors": self.catastrophic_errors,
            "profit_step": profit_step,
            "revenue_step": revenue_step,
            "error_cost_step": error_cost_step,
        }
        # JSON-safe per-limb summary
        energy_per_limb = {}
        for limb, wh in self.limb_energy_Wh.items():
            energy_per_limb[limb] = {
                "Wh": wh,
                "Wh_per_unit": wh / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                "Wh_per_hour": wh / max(self.t / 3600.0, 1e-6) if self.t > 0 else 0.0,
                "power_sum_W": self.limb_power_sum_W.get(limb, 0.0),
                "power_peak_W": self.limb_peak_power_W.get(limb, 0.0),
            }
        info["energy_per_limb"] = energy_per_limb
        # Skill summary
        energy_per_skill = {}
        for skill_id, limb_map in self.skill_energy_Wh.items():
            total_wh = sum(limb_map.values())
            energy_per_skill[skill_id] = {
                **{f"{limb}_Wh": limb_map.get(limb, 0.0) for limb in LIMB_GROUPS},
                "total_Wh": total_wh,
            }
        info["energy_per_skill"] = energy_per_skill
        # Per-joint summary
        energy_per_joint = {}
        for jn, wh in self.joint_energy_Wh.items():
            energy_per_joint[jn] = {
                "Wh": wh,
                "Wh_per_unit": wh / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                "Wh_per_hour": wh / max(self.t / 3600.0, 1e-6) if self.t > 0 else 0.0,
                "avg_power_W": self.joint_power_sum_W.get(jn, 0.0) / max(self.steps, 1),
                "peak_power_W": self.joint_peak_power_W.get(jn, 0.0),
                "avg_abs_velocity": self.joint_abs_vel_sum.get(jn, 0.0) / max(self.steps, 1),
                "max_abs_velocity": self.joint_max_vel.get(jn, 0.0),
                "avg_abs_torque": self.joint_abs_tau_sum.get(jn, 0.0) / max(self.steps, 1),
                "max_abs_torque": self.joint_max_tau.get(jn, 0.0),
                "directionality": {
                    "pos_steps": self.joint_dir_counts.get(jn, {}).get("pos", 0),
                    "neg_steps": self.joint_dir_counts.get(jn, {}).get("neg", 0),
                },
            }
        info["energy_per_joint"] = energy_per_joint

        # Effector summary (single effector)
        info["energy_per_effector"] = {
            "ee_main": {
                "Wh": self.effector_energy_Wh.get("ee_main", 0.0),
                "Wh_per_unit": self.effector_energy_Wh.get("ee_main", 0.0) / max(self.completed, 1e-6) if self.completed > 0 else 0.0,
                "Wh_per_hour": self.effector_energy_Wh.get("ee_main", 0.0) / max(self.t / 3600.0, 1e-6) if self.t > 0 else 0.0,
            }
        }
        info["coordination_metrics"] = {
            "mean_active_joints": 0.0,
            "mean_joint_velocity_corr": 0.0,
        }

        return obs, info, done


@dataclass
class EpisodeInfoSummary:
    """
    Canonical episode-level summary used by training and valuation layers.

    Fields:
        termination_reason: string enum (max_steps, sla_violation, catastrophic_error, zero_throughput, success, unknown)
        mpl_episode: units per hour over the episode
        ep_episode: units per Wh over the episode (energy productivity)
        error_rate_episode: errors per unit (fraction) over the episode
        throughput_units_per_hour: alias for mpl_episode
        energy_Wh: total energy consumed
        energy_Wh_per_unit: energy per completed unit (Wh/unit)
        energy_Wh_per_hour: energy per hour (Wh/hr)
        profit: total profit (revenue - error cost) aggregated over steps
        wage_parity: optional wage parity metric (robot wage / human wage) if available
    """
    termination_reason: str
    mpl_episode: float
    ep_episode: float
    error_rate_episode: float
    throughput_units_per_hour: float
    energy_Wh: float
    energy_Wh_per_unit: float
    energy_Wh_per_hour: float
    limb_energy_Wh: Dict[str, float]
    skill_energy_Wh: Dict[str, float]
    energy_per_limb: Dict[str, Dict[str, float]]
    energy_per_skill: Dict[str, Dict[str, float]]
    energy_per_joint: Dict[str, Dict[str, float]]
    energy_per_effector: Dict[str, Dict[str, float]]
    coordination_metrics: Dict[str, float]
    profit: float
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    media_refs: Dict[str, str] = field(default_factory=dict)
    wage_parity: Optional[float] = None


def summarize_episode_info(info_history: List[Dict[str, Any]]) -> EpisodeInfoSummary:
    """
    Aggregate per-step info dicts into a canonical EpisodeInfoSummary.
    """
    if not info_history:
        return EpisodeInfoSummary(
            termination_reason="unknown",
            mpl_episode=0.0,
            ep_episode=0.0,
            error_rate_episode=0.0,
            throughput_units_per_hour=0.0,
            energy_Wh=0.0,
            energy_Wh_per_unit=0.0,
            energy_Wh_per_hour=0.0,
            limb_energy_Wh={},
            skill_energy_Wh={},
            energy_per_limb={},
            energy_per_skill={},
            energy_per_joint={},
            energy_per_effector={},
            coordination_metrics={},
            profit=0.0,
            episode_id=str(uuid.uuid4()),
            media_refs={},
            wage_parity=None,
        )

    last = info_history[-1]
    total_energy = last.get("energy_Wh", 0.0)
    time_hours = last.get("t", 0.0) / 3600.0 if "t" in last else 0.0
    units_done = last.get("units_done", 0.0)
    errors = last.get("errors", 0.0)

    mpl_episode = (units_done / time_hours) if time_hours > 0 else 0.0
    ep_episode = (units_done / total_energy) if total_energy > 0 else 0.0
    error_rate_episode = errors / max(1.0, units_done)
    throughput_units_per_hour = mpl_episode
    energy_Wh_per_unit = total_energy / max(units_done, 1e-6) if total_energy > 0 else 0.0
    energy_Wh_per_hour = total_energy / max(time_hours, 1e-6) if total_energy > 0 else 0.0
    profit = sum(step.get("profit_step", 0.0) for step in info_history)
    limb_energy = last.get("limb_energy_Wh", {})
    skill_energy = last.get("skill_energy_Wh", {})
    energy_per_limb = last.get("energy_per_limb", {})
    energy_per_skill = last.get("energy_per_skill", {})
    energy_per_joint = last.get("energy_per_joint", {})
    energy_per_effector = last.get("energy_per_effector", {})
    coordination_metrics = last.get("coordination_metrics", {})

    return EpisodeInfoSummary(
        termination_reason=last.get("terminated_reason", "unknown") or "unknown",
        mpl_episode=mpl_episode,
        ep_episode=ep_episode,
        error_rate_episode=error_rate_episode,
        throughput_units_per_hour=throughput_units_per_hour,
        energy_Wh=total_energy,
        energy_Wh_per_unit=energy_Wh_per_unit,
        energy_Wh_per_hour=energy_Wh_per_hour,
        limb_energy_Wh=limb_energy,
        skill_energy_Wh=skill_energy,
        energy_per_limb=energy_per_limb,
        energy_per_skill=energy_per_skill,
        energy_per_joint=energy_per_joint,
        energy_per_effector=energy_per_effector,
        coordination_metrics=coordination_metrics,
        profit=profit,
        episode_id=last.get("episode_id", str(uuid.uuid4())),
        media_refs=last.get("media_refs", {"sim_trace": f"sim://{last.get('t',0.0):.3f}"}),
        wage_parity=None,
    )
