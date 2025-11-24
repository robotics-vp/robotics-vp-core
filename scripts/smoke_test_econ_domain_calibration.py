#!/usr/bin/env python3
"""
Stage 5 smoke test for EconDomainAdapter calibration layer.

Validates:
- Deterministic calibration across runs
- Sign preservation and sane bounds
- Relative scaling matches config for sim_pybullet vs real_ros
- Raw components preserved with calibrated copies logged
"""
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.economics.domain_adapter import EconDomainAdapter
from src.ontology.models import EconVector


def _make_raw_econ() -> EconVector:
    return EconVector(
        episode_id="econ_calib_ep",
        mpl_units_per_hour=10.0,
        wage_parity=1.0,
        energy_cost=5.0,
        damage_cost=2.0,
        novelty_delta=0.5,
        reward_scalar_sum=100.0,
        mobility_penalty=0.1,
        precision_bonus=0.2,
        stability_risk_score=0.3,
        components={
            "energy_penalty": 5.0,
            "damage_penalty": 1.0,
            "mpl_component": 10.0,
        },
        source_domain="sim_pybullet",
    )


def _within_bounds(vector: EconVector, limit: float = 1e6) -> bool:
    vals = [
        vector.mpl_units_per_hour,
        vector.energy_cost,
        vector.damage_cost,
        vector.reward_scalar_sum,
        vector.mobility_penalty,
        vector.precision_bonus,
        vector.stability_risk_score,
    ]
    return all(abs(v) <= limit for v in vals if v is not None)


def main() -> int:
    raw = _make_raw_econ()

    sim_adapter = EconDomainAdapter(domain_name="sim_pybullet")
    real_adapter = EconDomainAdapter(domain_name="real_ros")

    sim_calibrated_a = sim_adapter.map_vector(raw)
    sim_calibrated_b = sim_adapter.map_vector(raw)
    real_calibrated = real_adapter.map_vector(raw)

    # Determinism
    assert asdict(sim_calibrated_a) == asdict(sim_calibrated_b), "sim calibration must be deterministic"

    # Sign preservation
    assert sim_calibrated_a.mpl_units_per_hour > 0 and real_calibrated.mpl_units_per_hour > 0, "mpl sign should be preserved"
    assert sim_calibrated_a.energy_cost > 0 and real_calibrated.energy_cost > 0, "energy sign should be preserved"

    # Relative scaling from config: real_ros scales energy up, mpl slightly down
    expected_energy_real = raw.energy_cost * 1.2  # config: scale_energy_wh=1.2
    expected_mpl_real = raw.mpl_units_per_hour * 0.95  # config: scale_mpl=0.95
    assert abs(real_calibrated.energy_cost - expected_energy_real) < 1e-6, "real_ros energy scaling mismatch"
    assert abs(real_calibrated.mpl_units_per_hour - expected_mpl_real) < 1e-6, "real_ros mpl scaling mismatch"
    assert real_calibrated.damage_cost > sim_calibrated_a.damage_cost, "damage scaling should follow config (real_ros > sim_pybullet)"

    # Raw components preserved, calibrated versions present
    assert sim_calibrated_a.metadata.get("raw_components") == raw.components
    assert "calibrated_components" in sim_calibrated_a.metadata
    assert sim_calibrated_a.metadata.get("calibrated_econ_snapshot") is not None

    # Bounded values
    assert _within_bounds(sim_calibrated_a) and _within_bounds(real_calibrated), "Calibration should keep econ values bounded"

    print("[smoke_test_econ_domain_calibration] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
