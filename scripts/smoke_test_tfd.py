#!/usr/bin/env python3
"""
Smoke test for Text-Front-Door minimal implementation.
Validates canonical examples, determinism, and JSON-safe outputs.
"""
import json
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.tfd.compiler import TextFrontDoor
from src.tfd.intents import InstructionType


def _approx(a, b, tol=1e-6):
    return abs((a or 0.0) - (b or 0.0)) <= tol


def main():
    tfd = TextFrontDoor()
    examples = [
        ("Be very careful with the vase", InstructionType.MODIFY_RISK, {"risk_tolerance": 0.2, "skill_mode": "safety_critical"}),
        ("Don't worry about damage, just go fast", InstructionType.SPEED_MODE, {"risk_tolerance": 0.8, "skill_mode": "speed"}),
        ("Gentle movements only", InstructionType.PRECISION_MODE, {"skill_mode": "precision", "risk_tolerance": 0.3}),
        ("Minimize power consumption", InstructionType.ENERGY_CONSTRAINT, {"energy_budget_wh": 50.0, "skill_mode": "energy_efficient"}),
        ("Save energy but don't sacrifice quality", InstructionType.ENERGY_CONSTRAINT, {"energy_budget_wh": 75.0, "risk_tolerance": 0.4}),
        ("Use the dishwashing skill", InstructionType.DEPLOY_SKILL, {"skill_id": "dishwashing", "skill_mode": "precision"}),
        ("Activate exploration mode", InstructionType.EXPLORATION_MODE, {"skill_mode": "exploration", "curriculum_phase": "frontier"}),
        ("Increase productivity by 15%", InstructionType.TARGET_MPL, {"target_mpl_uplift": 9.0}),
        ("Work 20% faster", InstructionType.TARGET_MPL, {"target_mpl_uplift": 12.0}),
        ("Finish this task in 30 seconds", InstructionType.TIME_CONSTRAINT, {"time_budget_sec": 30.0, "skill_mode": "speed"}),
        ("Take your time, no rush", InstructionType.PRECISION_MODE, {"skill_mode": "precision"}),
        ("Explore new ways to open this drawer", InstructionType.EXPLORATION_MODE, {"skill_mode": "exploration"}),
        ("Try a different approach", InstructionType.PRIORITIZE_NOVELTY, {"novelty_tier": 2}),
        ("Figure out how to grip this weird object", InstructionType.EXPLORATION_MODE, {"objective_vector": {"weird_object": 1.0}}),
        ("Be quick but careful", None, {"risk_tolerance": 0.4, "time_budget_sec": 20.0}),
        ("Maximize output while staying under 100 Wh", InstructionType.ENERGY_CONSTRAINT, {"energy_budget_wh": 100.0}),
        ("Place the cup exactly in the center", InstructionType.PRECISION_MODE, {"objective_vector": {"center": 1.0}}),
        ("Double-check before placing", InstructionType.PRECISION_MODE, {"skill_mode": "precision"}),
        ("Sort the blue blocks, ignore the red ones", InstructionType.DEPLOY_SKILL, {"objective_vector": {"blue": 1.0, "red": 0.0}}),
        ("Focus on clearing the left side first", InstructionType.DEPLOY_SKILL, {"objective_vector": {"left_region": 1.0}}),
    ]

    for text, expected_intent, expected_fields in examples:
        r1 = tfd.process_instruction(text)
        r2 = tfd.process_instruction(text)
        assert json.loads(json.dumps(r1.to_dict())), "JSON roundtrip failed"
        assert r1.to_dict() == r2.to_dict(), "Determinism violated"

        if expected_intent is None:
            assert r1.status == "accepted", f"Expected acceptance for '{text}'"
        else:
            assert r1.parsed_intent is not None, f"Parsed intent missing for '{text}'"
            assert r1.parsed_intent.intent_type == expected_intent, f"Wrong intent for '{text}'"
            assert r1.status == "accepted", f"Unexpected status for '{text}': {r1.status}"

        cv = r1.condition_vector
        assert cv is not None, f"Condition vector missing for '{text}'"
        for key, val in expected_fields.items():
            actual = cv.to_dict().get(key)
            if isinstance(val, dict):
                for k, v in val.items():
                    assert actual and _approx(actual.get(k), v), f"Mismatch {key}.{k} for '{text}'"
            elif isinstance(val, float):
                assert _approx(actual, val), f"Mismatch {key} for '{text}': got {actual}, want {val}"
            else:
                assert actual == val, f"Mismatch {key} for '{text}': got {actual}, want {val}"

    print("[smoke_test_tfd] PASS")


if __name__ == "__main__":
    sys.exit(main())
