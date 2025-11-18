#!/usr/bin/env python3
"""
Smoke test for heuristic mobility policy and adjustments.
"""
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.physics.backends.mobility import MobilityContext
from src.physics.backends.mobility_heuristics import HeuristicMobilityPolicy


def _build_ctx(stability: float, drift_mm: float, slip_rate: float):
    return MobilityContext(
        task_id="task1",
        episode_id="ep1",
        env_name="pybullet",
        timestep=0,
        pose={"drift_mm": drift_mm},
        contacts={"slip_rate": slip_rate},
        target_precision_mm=5.0,
        stability_margin=stability,
        metadata={},
    )


def main():
    policy = HeuristicMobilityPolicy()
    ctx_precise = _build_ctx(stability=0.9, drift_mm=2.0, slip_rate=0.0)
    adj_precise = policy.compute_adjustment(ctx_precise)
    assert adj_precise.precision_gate_passed is True
    assert adj_precise.recovery_required is False

    ctx_unstable = _build_ctx(stability=0.1, drift_mm=12.0, slip_rate=0.5)
    adj_unstable = policy.compute_adjustment(ctx_unstable)
    assert adj_unstable.recovery_required is True
    assert adj_unstable.risk_level in {"MEDIUM", "HIGH"}

    # Determinism
    repeat = policy.compute_adjustment(ctx_unstable)
    assert adj_unstable.to_dict() == repeat.to_dict()
    print("[smoke_test_mobility_policies] PASS")


if __name__ == "__main__":
    main()
