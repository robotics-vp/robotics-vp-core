#!/usr/bin/env python3
"""
Stage 5 smoke test for EconCorrelator correlations and TrustMatrix output.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.econ_correlator_impl import EconCorrelator


def _make_datapacks() -> list:
    datapacks = []
    # High-risk episodes with large damage
    for _ in range(3):
        datapacks.append(
            {
                "segments": [{"metadata": {"failure_observed": True}}],
                "econ_vector": {"damage": 10.0, "mpl": 1.0, "energy_wh": 2.0, "success": False},
            }
        )
        datapacks.append(
            {
                "segments": [{"metadata": {"failure_observed": False}}],
                "econ_vector": {"damage": 1.0, "mpl": 2.5, "energy_wh": 1.0, "success": True},
            }
        )

    # Fragility indicators with flat damage to keep correlation weaker
    for _ in range(4):
        datapacks.append(
            {
                "segments": [{"metadata": {"fragile_interaction": True}}],
                "econ_vector": {"damage": 2.0, "mpl": 2.0, "energy_wh": 1.5, "success": True},
            }
        )
        datapacks.append(
            {
                "segments": [{"metadata": {"fragile_interaction": False}}],
                "econ_vector": {"damage": 2.0, "mpl": 2.0, "energy_wh": 1.5, "success": True},
            }
        )
    return datapacks


def main():
    datapacks = _make_datapacks()
    correlator = EconCorrelator(config={"min_samples_for_trust": 1})
    trust_matrix = correlator.compute_correlations(datapacks)
    assert trust_matrix, "TrustMatrix should not be empty"

    risk_score = trust_matrix["RiskTag"]["trust_score"]
    frag_score = trust_matrix["FragilityTag"]["trust_score"]
    assert risk_score > frag_score, "RiskTag should have higher trust than FragilityTag under stronger correlation"

    for entry in trust_matrix.values():
        assert 0.0 <= entry["trust_score"] <= 1.0, "trust_score must be bounded"

    # Determinism check
    trust_matrix_repeat = correlator.compute_correlations(datapacks)
    assert trust_matrix == trust_matrix_repeat, "Outputs must be deterministic"

    print("[smoke_test_econ_correlator_stage5] PASS")


if __name__ == "__main__":
    main()
