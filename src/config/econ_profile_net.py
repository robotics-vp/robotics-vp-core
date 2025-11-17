"""
EconProfileNet: DL-ified economic hyperparameter adjustment layer.

Maps environment context (engine type, task type, physics stats) to small deltas
on top of base EconParams. This is scaffolding for future Isaac/UE5 portability.

No behavior change in training yet - just scaffolding and smoke test.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from .econ_params import EconParams


@dataclass
class EconProfileContext:
    """
    Context for adjusting economic parameters based on environment/engine/physics.

    This bridges to Isaac/UE5 later by capturing engine-specific characteristics.
    """
    env_name: str              # "dishwashing", "drawer_vase", "dishwashing_arm", etc.
    engine_type: str           # "pybullet", "isaac", "ue5"
    task_type: str             # "throughput", "fragility", "precision", etc.

    # Coarse physics / outcome descriptors (can be zero for now):
    mean_torque: float = 0.0
    max_torque: float = 0.0
    mean_velocity: float = 0.0
    mean_energy_Wh_per_unit: float = 0.0

    # Human baselines (can be zero or coming from config later):
    baseline_mpl_human: float = 0.0
    baseline_error_rate_human: float = 0.0


class EconProfileNet(nn.Module):
    """
    Map EconProfileContext (plus maybe a few scalar stats) to small deltas
    on EconParams. This is scaffolding only for now.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        # env_name, engine_type, task_type -> one-hot or simple embedding
        # For now, assume we hand-encode them as a small numeric vector.
        input_dim = 8  # e.g., 3 (engine one-hot) + 3 (task one-hot) + 2 phys scalars

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Outputs: [Δbase_rate, Δdamage_cost, Δcare_cost, Δenergy_Wh_per_attempt, Δmax_steps_scale]
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def context_to_tensor(ctx: EconProfileContext) -> torch.Tensor:
    """
    Convert EconProfileContext to input tensor for EconProfileNet.

    Simple encoding: engine_type one-hot, task_type one-hot, a couple of scalars.
    Can be extended later with more features.
    """
    # Engine type one-hot: [pybullet, isaac, ue5]
    engine = [0.0, 0.0, 0.0]
    if ctx.engine_type == "pybullet":
        engine[0] = 1.0
    elif ctx.engine_type == "isaac":
        engine[1] = 1.0
    elif ctx.engine_type == "ue5":
        engine[2] = 1.0

    # Task type one-hot: [throughput, fragility, precision]
    task = [0.0, 0.0, 0.0]
    if ctx.task_type == "throughput":
        task[0] = 1.0
    elif ctx.task_type == "fragility":
        task[1] = 1.0
    elif ctx.task_type == "precision":
        task[2] = 1.0

    # Scalar features (normalized or raw for now)
    scalars = [
        ctx.mean_energy_Wh_per_unit,
        ctx.baseline_mpl_human,
    ]

    vec = torch.tensor(engine + task + scalars, dtype=torch.float32)
    return vec


def build_econ_params_from_context(
    base: EconParams,
    ctx: EconProfileContext,
    profile_net: Optional[EconProfileNet] = None,
    clamp: bool = True,
) -> EconParams:
    """
    Apply DL-based adjustments to EconParams based on context.

    If profile_net is None -> return base unchanged (current behavior).
    Otherwise, apply small adjustments to selected fields.

    Args:
        base: Base EconParams from preset
        ctx: Environment/engine context
        profile_net: Neural network to compute deltas (optional)
        clamp: Whether to clamp values to reasonable ranges

    Returns:
        EconParams with adjustments applied
    """
    if profile_net is None:
        return base

    profile_net.eval()
    with torch.no_grad():
        x = context_to_tensor(ctx).unsqueeze(0)  # (1, D)
        deltas = profile_net(x)[0].cpu().numpy()

    d_base_rate, d_damage_cost, d_care_cost, d_energy_Wh_per_attempt, d_max_steps_scale = deltas

    # Create modified copy (don't mutate `base` in place)
    # EconParams doesn't have a clone() method, so recreate from dict
    params = EconParams(**{k: v for k, v in base.__dict__.items()})

    # Apply deltas
    params.base_rate = params.base_rate + float(d_base_rate)
    params.damage_cost = params.damage_cost + float(d_damage_cost)
    params.care_cost = params.care_cost + float(d_care_cost)
    params.energy_Wh_per_attempt = params.energy_Wh_per_attempt + float(d_energy_Wh_per_attempt)
    params.max_steps = int(params.max_steps * (1.0 + float(d_max_steps_scale)))

    if clamp:
        params.base_rate = max(params.base_rate, 0.0)
        params.damage_cost = max(params.damage_cost, 0.0)
        params.care_cost = max(params.care_cost, 0.0)
        params.energy_Wh_per_attempt = max(params.energy_Wh_per_attempt, 0.0)
        params.max_steps = max(1, min(params.max_steps, 10_000))

    return params


def econ_context_from_condition(
    condition,  # ConditionProfile - imported dynamically to avoid circular imports
    summary,    # EpisodeInfoSummary
    env_name: str,
    task_type: str,
) -> EconProfileContext:
    """
    Build an EconProfileContext from existing Phase C structures.

    This is the bridge to Isaac/UE5 later. Takes ConditionProfile and
    EpisodeInfoSummary to populate the context.

    Args:
        condition: ConditionProfile from datapack schema
        summary: EpisodeInfoSummary from episode
        env_name: Environment name (e.g., "drawer_vase")
        task_type: Task type (e.g., "fragility")

    Returns:
        EconProfileContext ready for EconProfileNet
    """
    # engine_type, lighting, friction etc already live in ConditionProfile
    engine_type = getattr(condition, "engine_type", "pybullet")

    ctx = EconProfileContext(
        env_name=env_name,
        engine_type=engine_type,
        task_type=task_type,
        mean_energy_Wh_per_unit=float(getattr(summary, "energy_Wh_per_unit", 0.0) or 0.0),
        baseline_mpl_human=0.0,           # placeholder, wired later
        baseline_error_rate_human=0.0,    # placeholder
        # mean_torque, max_torque, mean_velocity can be filled once arm envs / Isaac stats are available
    )
    return ctx
