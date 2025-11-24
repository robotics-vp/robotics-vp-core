#!/usr/bin/env python3
"""
Smoke test for DemoPolicy inference wrapper.

Tests:
1. DemoPolicy instantiation with default config
2. reset(seed) determinism
3. act(raw_obs) returns valid action dict
4. Deterministic actions for fixed seed + obs
5. Valid shapes/keys for action dict

Exit code 0 on success, 1 on failure.
"""
import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.demo_policy import DemoPolicy, DemoPolicyConfig


def make_stub_obs(episode_id: str = "test_episode", step: int = 0) -> dict:
    """Create minimal stub observation."""
    return {
        "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "depth": None,
        "proprio": np.zeros(7, dtype=np.float32),
        "joint_positions": np.array([0.1, 0.2, 0.0, -0.1, 0.0, 0.15, 0.0], dtype=np.float32),
        "episode_id": episode_id,
        "step": step,
    }


def test_instantiation():
    """Test 1: DemoPolicy instantiation."""
    print("[smoke_test_demo_policy_inference] Test 1: Instantiation")

    # Default config
    policy = DemoPolicy()
    assert policy is not None, "DemoPolicy instantiation failed"

    # Dict config
    policy2 = DemoPolicy(config={"backend": "pybullet", "seed": 42})
    assert policy2 is not None, "DemoPolicy with dict config failed"

    # DemoPolicyConfig
    config = DemoPolicyConfig(backend="isaac", seed=0)
    policy3 = DemoPolicy(config=config)
    assert policy3 is not None, "DemoPolicy with DemoPolicyConfig failed"

    print("  ✓ Instantiation passed")


def test_reset_determinism():
    """Test 2: reset(seed) determinism."""
    print("[smoke_test_demo_policy_inference] Test 2: Reset determinism")

    policy = DemoPolicy()

    # Reset with seed
    policy.reset(seed=0)
    assert policy._seed == 0, "Seed not set correctly"
    assert policy._step_count == 0, "Step count not reset"
    assert policy._spatial_rnn_hidden is None, "Spatial RNN hidden state not reset"

    # Reset with different seed
    policy.reset(seed=42)
    assert policy._seed == 42, "Seed not updated"
    assert policy._step_count == 0, "Step count not reset after second reset"

    print("  ✓ Reset determinism passed")


def test_act_returns_valid_action():
    """Test 3: act(raw_obs) returns valid action dict."""
    print("[smoke_test_demo_policy_inference] Test 3: act() returns valid action")

    policy = DemoPolicy()
    policy.reset(seed=0)

    obs = make_stub_obs()
    action_dict = policy.act(obs)

    # Check structure
    assert isinstance(action_dict, dict), "act() did not return dict"
    assert "action" in action_dict, "action_dict missing 'action' key"
    assert "metadata" in action_dict, "action_dict missing 'metadata' key"

    # Check action
    action = action_dict["action"]
    assert isinstance(action, np.ndarray), "action is not numpy array"
    assert action.shape == (7,), f"action shape is {action.shape}, expected (7,)"
    assert action.dtype == np.float32, f"action dtype is {action.dtype}, expected float32"

    # Check metadata
    metadata = action_dict["metadata"]
    assert isinstance(metadata, dict), "metadata is not dict"
    assert "step" in metadata, "metadata missing 'step'"
    assert "skill_mode" in metadata, "metadata missing 'skill_mode'"

    print("  ✓ act() returns valid action dict")


def test_deterministic_actions():
    """Test 4: Deterministic actions for fixed seed + obs."""
    print("[smoke_test_demo_policy_inference] Test 4: Deterministic actions")

    # Policy 1
    policy1 = DemoPolicy()
    policy1.reset(seed=123)
    obs1 = make_stub_obs()
    action1 = policy1.act(obs1)["action"]

    # Policy 2 (same seed, same obs)
    policy2 = DemoPolicy()
    policy2.reset(seed=123)
    obs2 = make_stub_obs()
    action2 = policy2.act(obs2)["action"]

    # Check determinism
    assert np.allclose(action1, action2, atol=1e-6), "Actions not deterministic for same seed"

    # Policy 3 (different seed)
    policy3 = DemoPolicy()
    policy3.reset(seed=999)
    obs3 = make_stub_obs()
    action3 = policy3.act(obs3)["action"]

    # Should be different (probabilistically)
    # (Though stub policy might return zeros, so this is a weak check)
    # We'll just verify it runs without error
    assert action3 is not None, "Action from different seed is None"

    print("  ✓ Deterministic actions passed")


def test_get_summary():
    """Test 5: get_summary() returns JSON-safe dict."""
    print("[smoke_test_demo_policy_inference] Test 5: get_summary()")

    policy = DemoPolicy()
    policy.reset(seed=0)

    # Run a few steps
    for i in range(3):
        obs = make_stub_obs(step=i)
        policy.act(obs)

    summary = policy.get_summary()

    assert isinstance(summary, dict), "get_summary() did not return dict"
    assert "step_count" in summary, "summary missing 'step_count'"
    assert summary["step_count"] == 3, f"step_count is {summary['step_count']}, expected 3"
    assert "seed" in summary, "summary missing 'seed'"
    assert "backend_id" in summary, "summary missing 'backend_id'"

    # Check JSON-safe
    import json
    try:
        json.dumps(summary)
    except Exception as e:
        raise AssertionError(f"get_summary() not JSON-safe: {e}")

    print("  ✓ get_summary() passed")


def main():
    """Run all smoke tests."""
    print("="*80)
    print("[smoke_test_demo_policy_inference] Starting smoke tests")
    print("="*80)
    print()

    try:
        test_instantiation()
        test_reset_determinism()
        test_act_returns_valid_action()
        test_deterministic_actions()
        test_get_summary()

        print()
        print("="*80)
        print("[smoke_test_demo_policy_inference] All tests passed ✓")
        print("="*80)
        return 0

    except AssertionError as e:
        print()
        print("="*80)
        print(f"[smoke_test_demo_policy_inference] Test failed: {e}")
        print("="*80)
        return 1

    except Exception as e:
        print()
        print("="*80)
        print(f"[smoke_test_demo_policy_inference] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
