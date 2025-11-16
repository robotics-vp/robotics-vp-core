import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from src.config.econ_params import EconParams

# Backward compatibility alias
DishwashingParams = EconParams
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
            "terminated_reason": terminated_reason,
            "catastrophic_errors": self.catastrophic_errors,
            "profit_step": profit_step,
            "revenue_step": revenue_step,
            "error_cost_step": error_cost_step,
        }
        return obs, info, done


@dataclass
class EpisodeInfoSummary:
    """
    Canonical episode-level summary used by training and valuation layers.

    Fields:
        termination_reason: string enum for why the episode ended (e.g., max_steps, sla_violation, catastrophic_error, zero_throughput)
        mpl_episode: units per hour over the episode
        ep_episode: units per Wh over the episode
        error_rate_episode: errors per unit (fraction) over the episode
        throughput_units_per_hour: same as mpl_episode (explicit alias for clarity)
        energy_Wh: total energy consumed in the episode
        profit: total profit (revenue - error cost) aggregated over steps
        wage_parity: optional wage parity metric (robot wage / human wage) if available
    """
    termination_reason: str
    mpl_episode: float
    ep_episode: float
    error_rate_episode: float
    throughput_units_per_hour: float
    energy_Wh: float
    profit: float
    wage_parity: float = None


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
            profit=0.0,
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
    profit = sum(step.get("profit_step", 0.0) for step in info_history)

    return EpisodeInfoSummary(
        termination_reason=last.get("terminated_reason", "unknown") or "unknown",
        mpl_episode=mpl_episode,
        ep_episode=ep_episode,
        error_rate_episode=error_rate_episode,
        throughput_units_per_hour=throughput_units_per_hour,
        energy_Wh=total_energy,
        profit=profit,
        wage_parity=None,
    )
