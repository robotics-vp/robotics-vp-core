#!/usr/bin/env python3
"""
Smoke test for Stage 1 → RL episode descriptor contract.

Validates that descriptors are normalized, reproducible, and structurally correct
without altering any RL behavior.
"""
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.episode_sampling import datapack_to_rl_episode_descriptor  # type: ignore
from src.rl.episode_descriptor_validator import (  # type: ignore
    normalize_episode_descriptor,
    validate_episode_descriptor,
    normalize_and_validate,
)
from src.valuation.datapack_schema import DataPackMeta  # type: ignore


def _assert_valid(descriptor):
    errors = validate_episode_descriptor(descriptor)
    assert not errors, f"Descriptor failed validation: {errors}"
    assert len(descriptor["objective_vector"]) == 5
    assert descriptor["env_name"]
    assert descriptor["engine_type"]
    assert descriptor["task_type"]


def test_datapack_path():
    """Ensure datapack → descriptor path yields a valid, normalized contract."""
    dp = DataPackMeta()
    descriptor = datapack_to_rl_episode_descriptor(dp)
    _assert_valid(descriptor)


def test_reproducible_normalization():
    """Normalization should be deterministic for missing fields."""
    raw_descriptor = {
        "objective_vector": [2.0, 1.0],
        "env_name": "drawer_vase",
        "tier": 2,
        "trust_score": 0.8,
    }
    first = normalize_episode_descriptor(raw_descriptor)
    second = normalize_episode_descriptor(deepcopy(raw_descriptor))
    assert first == second, "Normalization must be reproducible"
    _assert_valid(first)


def test_validation_flags_bad_inputs():
    """Validator should catch out-of-range fields."""
    bad_descriptor = {
        "objective_vector": [1, 2, 3, 4, 5],
        "env_name": "",
        "engine_type": "pybullet",
        "task_type": "drawer",
        "tier": -1,
        "trust_score": 1.5,
        "sampling_weight": -0.1,
    }
    direct_errors = validate_episode_descriptor(bad_descriptor)
    assert direct_errors, "Expected validation errors for intentionally bad descriptor"

    normalized, errors = normalize_and_validate(bad_descriptor)
    assert not errors, "Normalization should resolve basic structural issues"
    assert normalized["trust_score"] <= 1.0
    assert normalized["sampling_weight"] >= 0.0


def main():
    test_datapack_path()
    test_reproducible_normalization()
    test_validation_flags_bad_inputs()
    print("[smoke] Episode descriptor contract tests passed.")


if __name__ == "__main__":
    main()
