from dataclasses import dataclass
from typing import Literal, Optional


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


def load_econ_params(profile: dict, preset: Optional[Literal["toy", "realistic"]] = None) -> EconParams:
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
    else:
        raise ValueError(f"Unknown econ preset: {selected}")

    return EconParams(**kwargs, preset=selected)
