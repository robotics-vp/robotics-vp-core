"""
Smoke test for ConditionedVisionAdapter.

Tests:
- Three regimes produce different outputs
- Base representation (z_v) unchanged
- Flag-gating maintains backward compatibility
- Determinism
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np

from src.vision.interfaces import VisionFrame
from src.vision.conditioned_adapter import ConditionedVisionAdapter
from src.observation.condition_vector import ConditionVector


def create_test_frame() -> VisionFrame:
    """Create a test vision frame."""
    return VisionFrame(
        backend="test",
        task_id="test_task",
        episode_id="test_ep",
        timestep=0,
        width=64,
        height=64,
        channels=3,
    )


def test_regime_differences():
    """Test that three regimes produce different outputs."""
    print("Testing regime differences...")

    frame = create_test_frame()
    adapter = ConditionedVisionAdapter()

    # Safety regime
    cv_safety = ConditionVector(
        task_id="test",
        env_id="test",
        skill_mode="safety_critical",
        ood_risk_level=0.2,
    )

    # Exploration regime
    cv_explore = ConditionVector(
        task_id="test",
        env_id="test",
        skill_mode="frontier_exploration",
        novelty_tier=2,
    )

    # Efficiency regime
    cv_efficient = ConditionVector(
        task_id="test",
        env_id="test",
        skill_mode="efficiency_throughput",
        energy_budget_wh=50.0,
    )

    out_safety = adapter.forward(frame, cv_safety)
    out_explore = adapter.forward(frame, cv_explore)
    out_efficient = adapter.forward(frame, cv_efficient)

    # Risk maps should differ
    assert not np.allclose(out_safety["risk_map"], out_explore["risk_map"]), \
        "Safety and exploration risk maps should differ"

    # z_v should be identical (base representation invariant)
    for level in out_safety["z_v"]:
        assert np.allclose(out_safety["z_v"][level], out_explore["z_v"][level]), \
            f"z_v level {level} should be identical across regimes"
        assert np.allclose(out_safety["z_v"][level], out_efficient["z_v"][level]), \
            f"z_v level {level} should be identical across regimes"

    print("✓ Regime differences confirmed (risk maps differ, z_v unchanged)")


def test_flag_gating():
    """Test that conditioning can be disabled."""
    print("Testing flag gating...")

    frame = create_test_frame()

    # Adapter with conditioning enabled
    adapter_on = ConditionedVisionAdapter(config={"enable_conditioning": True})

    # Adapter with conditioning disabled
    adapter_off = ConditionedVisionAdapter(config={"enable_conditioning": False})

    cv = ConditionVector(
        task_id="test",
        env_id="test",
        skill_mode="safety_critical",
        ood_risk_level=0.2,
    )

    out_on = adapter_on.forward(frame, cv)
    out_off = adapter_off.forward(frame, cv)

    # With conditioning off, output should be baseline (no modulation)
    # Risk map should be default when conditioning is off
    assert np.allclose(out_off["risk_map"], adapter_off._default_risk_map()), \
        "Conditioning off should produce default risk map"

    # With conditioning on, risk map should be modulated
    risk_diff = np.abs(out_on["risk_map"] - out_off["risk_map"]).sum()
    assert risk_diff > 0, "Conditioning on should modulate risk map"

    print("✓ Flag gating works (on vs off produce different results)")


def test_determinism():
    """Test deterministic behavior."""
    print("Testing determinism...")

    frame = create_test_frame()
    adapter = ConditionedVisionAdapter()

    cv = ConditionVector(
        task_id="test",
        env_id="test",
        skill_mode="safety_critical",
        ood_risk_level=0.3,
    )

    # Run twice
    out1 = adapter.forward(frame, cv)
    out2 = adapter.forward(frame, cv)

    # Should produce identical outputs
    assert np.allclose(out1["risk_map"], out2["risk_map"]), \
        "Risk map should be deterministic"

    for level in out1["z_v"]:
        assert np.allclose(out1["z_v"][level], out2["z_v"][level]), \
            f"z_v level {level} should be deterministic"

    print("✓ Deterministic behavior confirmed")


def test_bounded_scales():
    """Test that all scales are bounded."""
    print("Testing bounded scales...")

    frame = create_test_frame()
    adapter = ConditionedVisionAdapter()

    # Extreme condition vector (should still be bounded)
    cv_extreme = ConditionVector(
        task_id="test",
        env_id="test",
        skill_mode="safety_critical",
        ood_risk_level=1.0,  # Maximum risk
    )

    out = adapter.forward(frame, cv_extreme)

    # Check that risk map values are bounded
    assert np.all(out["risk_map"] >= 0), "Risk map should be non-negative"
    assert np.all(out["risk_map"] <= 10.0), "Risk map should be bounded"

    # Check that modulated features are bounded
    for level, feat in out["features_modulated"].items():
        assert np.all(np.isfinite(feat)), f"Features at level {level} should be finite"
        assert not np.any(np.isnan(feat)), f"Features at level {level} should not be NaN"

    print("✓ All scales are properly bounded")


def test_z_v_invariance():
    """Test that z_v is invariant to condition vector."""
    print("Testing z_v invariance...")

    frame = create_test_frame()
    adapter = ConditionedVisionAdapter()

    # Different condition vectors
    cv1 = ConditionVector(task_id="test", env_id="test", skill_mode="safety_critical")
    cv2 = ConditionVector(task_id="test", env_id="test", skill_mode="frontier_exploration")
    cv3 = ConditionVector(task_id="test", env_id="test", skill_mode="efficiency_throughput")

    out1 = adapter.forward(frame, cv1)
    out2 = adapter.forward(frame, cv2)
    out3 = adapter.forward(frame, cv3)

    # z_v should be identical across all conditions
    for level in out1["z_v"]:
        assert np.allclose(out1["z_v"][level], out2["z_v"][level]), \
            f"z_v level {level} should be invariant (cv1 vs cv2)"
        assert np.allclose(out1["z_v"][level], out3["z_v"][level]), \
            f"z_v level {level} should be invariant (cv1 vs cv3)"

    print("✓ z_v invariance confirmed")


def main():
    print("=== ConditionedVisionAdapter Smoke Tests ===\n")

    try:
        test_regime_differences()
        test_flag_gating()
        test_determinism()
        test_bounded_scales()
        test_z_v_invariance()

        print("\n=== All Vision Adapter Tests Passed ===")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
