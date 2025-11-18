#!/usr/bin/env python3
"""
Smoke test for ontology adapters (datapack + episode) and store integration.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.datapack_adapters import datapack_from_stage1, datapack_from_stage2_enrichment
from src.ontology.episode_adapters import episode_from_descriptor
from src.ontology.store import OntologyStore


def main():
    root = Path("data/ontology/test_adapters")
    if root.exists():
        shutil.rmtree(root)

    store = OntologyStore(root_dir=str(root))
    task_id = "task_adapter"
    robot_id = "robot_adapter"

    # Fake Stage1 datapack
    stage1_raw = {
        "pack_id": "stage1_dp",
        "novelty_score": 0.3,
        "quality_score": 0.9,
        "semantic_tags": ["fragile", "drawer"],
        "timestamp": datetime(2024, 1, 1).timestamp(),
    }

    # Fake Stage2 enrichment
    stage2_enrichment = {
        "episode_id": "ep_enrich",
        "enrichment": {
            "novelty_tags": [{"novelty_score": 0.8, "expected_mpl_gain": 1.1, "comparison_basis": "smoke"}],
            "fragility_tags": [{"object_name": "vase", "fragility_level": "high", "damage_cost_usd": 50.0}],
            "coherence_score": 0.7,
            "validation_status": "passed",
        },
    }

    # Fake Stage3 descriptor
    descriptor = {
        "pack_id": "stage1_dp",
        "env_name": "drawer_vase",
        "backend": "pybullet",
        "engine_type": "pybullet",
        "objective_preset": "balanced",
        "objective_vector": [1, 1, 1, 1, 0],
        "tier": 1,
        "trust_score": 0.8,
        "sampling_metadata": {"strategy": "balanced"},
        "semantic_tags": ["fragile"],
    }

    # Convert via adapters
    dp1 = datapack_from_stage1(stage1_raw, task_id=task_id)
    dp2 = datapack_from_stage2_enrichment(stage2_enrichment, task_id=task_id)
    ep = episode_from_descriptor(descriptor, task_id=task_id, robot_id=robot_id)

    # Write to store
    store.append_datapacks([dp1, dp2])
    store.upsert_episode(ep)

    # Assertions
    again_dp1 = datapack_from_stage1(stage1_raw, task_id=task_id)
    assert dp1.datapack_id == again_dp1.datapack_id, "Datapack ID should be deterministic"
    dps = store.list_datapacks(task_id=task_id)
    assert len(dps) == 2, f"Expected 2 datapacks, got {len(dps)}"
    for d in dps:
        assert isinstance(d.tags, dict)
    ep_loaded = store.get_episode(ep.episode_id)
    assert ep_loaded is not None
    assert ep_loaded.task_id == task_id
    assert ep_loaded.robot_id == robot_id

    print("[smoke_test_ontology_adapters] All tests passed.")


if __name__ == "__main__":
    main()
