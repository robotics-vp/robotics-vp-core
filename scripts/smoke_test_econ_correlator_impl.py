import sys
import os
import json
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.econ_correlator import EconCorrelator, TrustEntry

def make_dummy_data():
    datapacks = [
        {"episode_id": "ep1", "semantic_tags": ["fragile", "grasp"]},
        {"episode_id": "ep2", "semantic_tags": ["fragile", "drop"]},
        {"episode_id": "ep3", "semantic_tags": ["fast", "move"]},
    ]
    econ_vectors = {
        "ep1": {"damage_cost": 5.0, "energy_cost": 1.0, "success": True},
        "ep2": {"damage_cost": 50.0, "energy_cost": 2.0, "success": False},
        "ep3": {"damage_cost": 0.0, "energy_cost": 10.0, "success": True},
    }
    return datapacks, econ_vectors

def test_computation():
    corre = EconCorrelator()
    datapacks, vectors = make_dummy_data()
    
    matrix = corre.compute_correlations(datapacks, vectors)
    
    # Check 'fragile'
    assert "fragile" in matrix
    entry = matrix["fragile"]
    # Total damage = 55, Count = 2 => Mean = 27.5
    assert abs(entry["mean_damage"] - 27.5) < 0.001
    # Total energy = 3, Count = 2 => Mean = 1.5
    assert abs(entry["mean_energy"] - 1.5) < 0.001
    # Successes = 1/2 => 0.5
    trust_penalty = min(1.0, 27.5 / 100.0) # 0.275
    expected_trust = 0.5 * (1.0 - 0.275) # 0.3625
    assert abs(entry["trust_score"] - expected_trust) < 0.001

    # Check 'fast'
    entry_fast = matrix["fast"]
    assert entry_fast["mean_damage"] == 0.0
    assert entry_fast["trust_score"] == 1.0

    print("[smoke_test_econ_correlator_impl] Computation Logic: PASS")

def main():
    test_computation()

if __name__ == "__main__":
    main()
