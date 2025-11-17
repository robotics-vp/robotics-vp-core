from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass
class EconParams:
    """
    Centralized economic + SLA parameters for dishwashing-style tasks.

    This removes hardcoded reward/termination constants from environments so
    training scripts and evaluators can share a consistent configuration surface.
    """

    # Economics
    price_per_unit: float
    damage_cost: float
    energy_Wh_per_attempt: float

    # Throughput + error model
    time_step_s: float
    base_rate: float
    p_min: float
    k_err: float
    q_speed: float
    q_care: float
    care_cost: float

    # Horizons + safety
    max_steps: int
    max_catastrophic_errors: int
    max_error_rate_sla: float
    min_steps_for_sla: int
    zero_throughput_patience: int

    preset: str = "toy"

    # Drawer+Vase specific (optional, for backward compatibility)
    value_per_unit: float = 0.0  # Alias for price_per_unit
    vase_break_cost: float = 0.0
    electricity_price_kWh: float = 0.0
    other_costs_per_hr: float = 0.0
    allowable_risk_tolerance: float = 0.0
    fragility_penalty_coeff: float = 0.0


@dataclass
class DrawerVaseEconParams:
    """
    Economic parameters specific to Drawer+Vase task.

    Task: Open drawer while avoiding collision with fragile vase.
    """

    # Task value
    value_per_successful_drawer_open: float = 5.0  # Revenue for successfully opening drawer
    vase_break_cost: float = 50.0  # Cost of breaking vase (fragile, expensive)

    # Operating costs
    electricity_price_kWh: float = 0.12  # $/kWh
    other_costs_per_hr: float = 2.0  # Maintenance, etc.

    # Risk parameters
    allowable_risk_tolerance: float = 0.1  # Max acceptable collision probability
    fragility_penalty_coeff: float = 10.0  # Penalty multiplier for vase contact

    # SLA thresholds
    max_steps: int = 300
    max_high_risk_contacts: int = 5
    max_impulse_threshold: float = 5.0  # N·s before vase breaks

    preset: str = "drawer_vase"


def load_econ_params(
    profile: dict,
    preset: Optional[Union[Literal["toy", "realistic", "drawer_vase"], str]] = None
) -> Union[EconParams, DrawerVaseEconParams]:
    """
    Build EconParams from an internal profile with support for preset overrides.

    Args:
        profile: internal_profile.get_internal_experiment_profile(...)
        preset: optional preset selector; falls back to profile["econ_preset"] or "toy"
    """
    selected = preset or profile.get("econ_preset", "toy")

    # Baseline values from profile
    kwargs = dict(
        price_per_unit=profile.get("price_per_unit", 0.30),
        damage_cost=profile.get("damage_cost", 1.0),
        energy_Wh_per_attempt=profile.get("energy_Wh_per_attempt", 0.05),
        time_step_s=profile.get("time_step_s", 60.0),
        base_rate=profile.get("base_rate", 2.0),
        p_min=profile.get("p_min", 0.02),
        k_err=profile.get("k_err", 0.12),
        q_speed=profile.get("q_speed", 1.2),
        q_care=profile.get("q_care", 1.5),
        care_cost=profile.get("care_cost", 0.25),
        max_steps=profile.get("max_steps", 240),
        max_catastrophic_errors=profile.get("max_catastrophic_errors", 3),
        max_error_rate_sla=profile.get("max_error_rate_sla", 0.12),
        min_steps_for_sla=profile.get("min_steps_for_sla", 5),
        zero_throughput_patience=profile.get("zero_throughput_patience", 10),
    )

    # Preset tweaks (structure allows future data-driven calibration)
    if selected == "toy":
        pass  # keep profile defaults
    elif selected == "realistic":
        kwargs.update(
            price_per_unit=profile.get("price_per_unit_realistic", kwargs["price_per_unit"]),
            damage_cost=profile.get("damage_cost_realistic", kwargs["damage_cost"]),
            base_rate=profile.get("base_rate_realistic", kwargs["base_rate"]),
            max_error_rate_sla=profile.get("max_error_rate_sla_realistic", kwargs["max_error_rate_sla"]),
            energy_Wh_per_attempt=profile.get("energy_Wh_per_attempt_realistic", kwargs["energy_Wh_per_attempt"]),
        )
    elif selected == "drawer_vase":
        # Return DrawerVaseEconParams for drawer+vase task
        return DrawerVaseEconParams(
            value_per_successful_drawer_open=profile.get("value_per_drawer_open", 5.0),
            vase_break_cost=profile.get("vase_break_cost", 50.0),
            electricity_price_kWh=profile.get("electricity_price_kWh", 0.12),
            other_costs_per_hr=profile.get("other_costs_per_hr", 2.0),
            allowable_risk_tolerance=profile.get("allowable_risk_tolerance", 0.1),
            fragility_penalty_coeff=profile.get("fragility_penalty_coeff", 10.0),
            max_steps=profile.get("max_steps", 300),
            max_high_risk_contacts=profile.get("max_high_risk_contacts", 5),
            max_impulse_threshold=profile.get("max_impulse_threshold", 5.0),
            preset=selected,
        )
    else:
        raise ValueError(f"Unknown econ preset: {selected}")

    return EconParams(**kwargs, preset=selected)


def get_econ_params_with_profile(
    profile: dict,
    preset: Optional[str] = None,
    ctx=None,  # Optional[EconProfileContext]
    profile_net=None,  # Optional[EconProfileNet]
) -> Union[EconParams, DrawerVaseEconParams]:
    """
    Wrapper: current behavior if ctx/profile_net is None; otherwise
    apply DL adjustments.

    This is scaffolding for DL-ified economic hyperparameters.
    Does not change behavior when ctx/profile_net are None.

    Args:
        profile: Internal experiment profile dict
        preset: Optional preset selector
        ctx: Optional EconProfileContext for DL adjustments
        profile_net: Optional EconProfileNet for computing deltas

    Returns:
        EconParams (possibly adjusted by profile_net)
    """
    base = load_econ_params(profile, preset)

    if ctx is None or profile_net is None:
        return base

    # Late import to avoid circular dependency
    from .econ_profile_net import build_econ_params_from_context

    return build_econ_params_from_context(base, ctx, profile_net)
