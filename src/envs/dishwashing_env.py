from dataclasses import dataclass
import numpy as np

@dataclass
class DishwashingParams:
    price_per_unit: float = 0.30  # $/dish
    damage_cost: float = 1.0      # $ per broken dish
    time_step_s: float = 60.0     # 1 step = 1 minute
    base_rate: float = 2.0        # attempts/min at speed=1
    p_min: float = 0.02           # minimum error rate (2% floor)
    k_err: float = 0.12           # max additional error
    q_speed: float = 1.2          # speed curvature
    q_care: float = 1.5           # care curvature
    care_cost: float = 0.25       # throughput penalty for care

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
    def __init__(self, params: DishwashingParams):
        self.p = params
        self.reset()

    def reset(self):
        self.t = 0.0
        self.completed = 0
        self.attempts = 0
        self.errors = 0
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
        done = False

        # Successes
        succ = max(attempts - errs, 0)
        self.completed += succ

        self.t += self.p.time_step_s
        obs = self._obs()
        info = {
            "succ": succ,
            "errs": errs,
            "p_err": p_err,
            "speed": speed,
            "care": care,
            "rate_per_min": rate_per_min
        }
        return obs, info, done
