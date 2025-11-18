"""
Smoke test for Phase G policies and Econ Calibration.

Verifies:
1. DatapackAuditorPolicy (Heuristic) behavior and determinism.
2. EconDomainAdapter calibration logic.
3. RewardEngine integration with calibration.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.policies.registry import build_all_policies
from src.policies.datapack_auditor import HeuristicDatapackAuditor
from src.economics.domain_adapter import EconDomainAdapter, EconDomainAdapterConfig
from src.economics.reward_engine import RewardEngine
from src.ontology.models import Task, Robot, Episode, EpisodeEvent, EconVector

def test_datapack_auditor():
    print("\n--- Testing DatapackAuditorPolicy ---")
    auditor = HeuristicDatapackAuditor()
    
    # Case 1: High Value, Low Risk (AAA)
    safe_features = auditor.build_features(
        datapack={"episode_id": "ep_safe_001"},
        semantic_tags=[
            {"novelty_type": "edge_case", "novelty_score": 0.8},
            {"intervention_type": "failure_recovery"}, # Recovery reduces risk
        ],
        econ_slice={"expected_mpl_gain": 6.0, "novelty_score": 0.8}
    )
    result_safe = auditor.evaluate(safe_features)
    print(f"Safe Case Rating: {result_safe['rating']} (Expected AAA/AA)")
    assert result_safe['rating'] in ["AAA", "AA"], f"Expected high rating, got {result_safe['rating']}"
    
    # Case 2: High Risk, Low Value (JUNK)
    risky_features = auditor.build_features(
        datapack={"episode_id": "ep_risky_001"},
        semantic_tags=[
            {"risk_type": "collision", "severity": "critical"},
            {"fragility_level": "high"},
        ],
        econ_slice={"expected_mpl_gain": 1.0, "novelty_score": 0.1}
    )
    result_risky = auditor.evaluate(risky_features)
    print(f"Risky Case Rating: {result_risky['rating']} (Expected JUNK)")
    assert result_risky['rating'] == "JUNK", f"Expected JUNK, got {result_risky['rating']}"
    
    # Determinism Check
    result_safe_2 = auditor.evaluate(safe_features)
    assert json.dumps(result_safe, sort_keys=True) == json.dumps(result_safe_2, sort_keys=True), "Auditor not deterministic"
    print("Determinism check passed.")

def test_econ_domain_adapter():
    print("\n--- Testing EconDomainAdapter ---")
    config = EconDomainAdapterConfig(
        source_domain="pybullet",
        scaling={"energy_cost": 2.0, "damage_cost": 1.5}, # Sim underestimates energy/damage
        offsets={"wage_parity": -0.1}
    )
    adapter = EconDomainAdapter(config)
    
    raw_econ = EconVector(
        episode_id="ep_001",
        mpl_units_per_hour=10.0,
        wage_parity=1.0,
        energy_cost=5.0,
        damage_cost=10.0,
        novelty_delta=0.5,
        reward_scalar_sum=100.0,
        source_domain="pybullet"
    )
    
    calibrated = adapter.map_vector(raw_econ)
    
    print(f"Raw Energy: {raw_econ.energy_cost}, Calibrated: {calibrated.energy_cost}")
    print(f"Raw Damage: {raw_econ.damage_cost}, Calibrated: {calibrated.damage_cost}")
    
    assert calibrated.energy_cost == 10.0, "Energy scaling failed"
    assert calibrated.damage_cost == 15.0, "Damage scaling failed"
    assert calibrated.wage_parity == 0.9, "Wage offset failed"
    assert calibrated.source_domain == "pybullet", "Source domain lost"
    assert calibrated.metadata["is_calibrated"] is True, "Calibration flag missing"
    print("Calibration logic verified.")

def test_reward_engine_integration():
    print("\n--- Testing RewardEngine Integration ---")
    task = Task(task_id="t1", name="test_task")
    robot = Robot(robot_id="r1", name="test_robot")
    
    # Config with calibration
    config = {
        "source_domain": "pybullet",
        "econ_scaling": {"energy_cost": 2.0},
        "econ_offsets": {}
    }
    
    engine = RewardEngine(task, robot, config)
    
    # Fake episode events
    events = [
        EpisodeEvent(
            episode_id="ep_001", timestep=1, event_type="step", timestamp=None,
            reward_scalar=1.0,
            reward_components={"energy_penalty": 1.0, "mpl_component": 5.0}
        )
    ]
    episode = Episode(episode_id="ep_001", task_id="t1", robot_id="r1")
    
    econ = engine.compute_econ_vector(episode, events)
    
    print(f"Computed Econ Energy: {econ.energy_cost}")
    # Raw energy sum = 1.0. Scaled by 2.0 -> 2.0.
    assert econ.energy_cost == 2.0, f"RewardEngine calibration failed. Expected 2.0, got {econ.energy_cost}"
    assert econ.metadata["is_calibrated"] is True
    print("RewardEngine integration verified.")

if __name__ == "__main__":
    test_datapack_auditor()
    test_econ_domain_adapter()
    test_reward_engine_integration()
    print("\nAll Phase G smokes passed!")
