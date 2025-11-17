#!/usr/bin/env python3
"""
Smoke test for SIMA-2 semantic primitive extraction and ontology proposals.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sima2.semantic_primitive_extractor import (
    SemanticPrimitive,
    extract_primitives_from_rollout,
    primitive_to_ontology_update,
)


def _make_rollouts():
    return [
        {
            "task_type": "open_drawer",
            "events": [
                {"action": "grasp", "object": "drawer_handle", "tags": ["drawer", "grasp"], "success": True},
                {"action": "pull", "object": "drawer", "tags": ["drawer", "open"], "energy_intensity": 0.2},
            ],
            "metrics": {"steps": 2, "success": True, "energy_used": 0.4},
        },
        {
            "task_type": "move_fragile_vase",
            "events": [
                {"action": "lift", "object": "vase", "tags": ["vase", "fragile"], "success_rate": 0.85},
                {"action": "place", "object": "table", "tags": ["table", "place"], "energy_intensity": 0.1},
            ],
            "metrics": {"steps": 3, "success": True},
        },
        {
            "task_type": "push_box",
            "events": [
                {"action": "approach", "object": "box", "tags": ["box", "approach"], "success": True},
                {"action": "push", "object": "box", "tags": ["box", "forceful"], "success": True},
            ],
            "metrics": {"steps": 4, "collisions": 1, "energy_intensity": 6.0},
        },
    ]


def main():
    rollouts = _make_rollouts()
    for idx, rollout in enumerate(rollouts):
        primitives = extract_primitives_from_rollout(rollout)
        assert primitives, f"Expected primitives for rollout {idx}"
        for primitive in primitives:
            assert isinstance(primitive, SemanticPrimitive)
            assert primitive.tags, "Primitive tags should not be empty"
            # Risk rules: fragile -> high, collisions/energy bump -> medium, otherwise low
            if "fragile" in primitive.tags:
                assert primitive.risk_level == "high", f"Fragile primitive should be high risk, got {primitive.risk_level}"
            if rollout["metrics"].get("collisions"):
                assert primitive.risk_level in {"medium", "high"}, "Collisions should elevate risk"
            update = primitive_to_ontology_update(primitive)
            for key in ("action", "skill_name", "risk_level", "tags", "source"):
                assert key in update, f"Missing {key} in ontology update"
            assert update["action"] == "add_or_update_skill"
            assert isinstance(update["tags"], list)

    print("[smoke_test_sima2_semantic_extraction] All assertions passed.")


if __name__ == "__main__":
    main()
