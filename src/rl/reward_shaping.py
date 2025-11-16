from typing import Dict, Tuple, Optional


def compute_econ_reward(
    mpl: float,
    ep: float,
    error_rate: float,
    wage_parity: Optional[float] = None,
    mode: str = "mpl_ep_error",
    alpha_mpl: float = 1.0,
    alpha_error: float = 1.0,
    alpha_ep: float = 1.0,
    alpha_wage: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Shared reward shaping for RL agents.

    Returns:
        reward: scalar used by RL
        components: dict of reward components for logging
    """
    if mode != "mpl_ep_error":
        raise ValueError(f"Unsupported reward mode: {mode}")

    mpl_component = alpha_mpl * mpl
    ep_component = alpha_ep * ep
    error_penalty = -alpha_error * error_rate

    wage_penalty = 0.0
    if wage_parity is not None and wage_parity > 1.0 and alpha_wage > 0:
        wage_penalty = -alpha_wage * max(0.0, wage_parity - 1.0)

    reward = mpl_component + ep_component + error_penalty + wage_penalty
    components = {
        "mpl_component": mpl_component,
        "ep_component": ep_component,
        "error_penalty": error_penalty,
        "wage_penalty": wage_penalty,
    }
    return reward, components
