#!/usr/bin/env python3
"""
Smoke test for EconCorrelator TrustMatrix semantics.
"""
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.analytics.econ_correlator_impl import EconCorrelator


def main():
    datapacks = []
    for i in range(2):
        # Recovery datapacks with successful outcomes to drive trust high
        datapacks.append(
            {
                "segments": [
                    {"metadata": {"failure_observed": True}},
                    {"metadata": {"recovery_observed": True}},
                ],
                "econ_vector": {"damage": 1.0, "mpl": 2.0, "energy_wh": 1.0, "success": True},
            }
        )
        # Risk datapacks with failures to drive trust via damage
        datapacks.append(
            {
                "segments": [
                    {"metadata": {"failure_observed": True}},
                ],
                "econ_vector": {"damage": 5.0, "mpl": 1.0, "energy_wh": 2.0, "success": False},
            }
        )

    correlator = EconCorrelator(config={"min_samples_for_trust": 2})
    trust_matrix = correlator.compute_correlations(datapacks)

    recovery_entry = trust_matrix.get("RecoveryTag")
    risk_entry = trust_matrix.get("RiskTag")
    assert recovery_entry, "Expected RecoveryTag in trust matrix"
    assert risk_entry, "Expected RiskTag in trust matrix"
    assert recovery_entry["trust_tier"] in {"trusted", "provisional"}
    assert risk_entry["sampling_multiplier"] in {1.5, 5.0, 1.0}
    assert recovery_entry["sampling_multiplier"] in {1.5, 5.0, 1.0}

    print("[smoke_test_econ_correlator_impl] PASS")


if __name__ == "__main__":
    main()
