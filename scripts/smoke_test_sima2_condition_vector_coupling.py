"""
Smoke test for SIMA-2 OOD/Recovery Tags + TrustMatrix → ConditionVector coupling.

Tests that OOD and Recovery tags deterministically affect ConditionVector fields
(ood_risk_level and recovery_priority) in a bounded, documented way.

Run:
    python -m scripts.smoke_test_sima2_condition_vector_coupling
"""
import sys
from typing import Dict, Any

from src.observation.condition_vector_builder import ConditionVectorBuilder


def build_dummy_context() -> Dict[str, Any]:
    """Build minimal context for ConditionVector construction."""
    return {
        "episode_config": {"task_id": "test_task", "env_id": "test_env", "backend_id": "test_backend"},
        "econ_state": {"target_mpl": 60.0, "current_wage_parity": 0.8, "energy_budget_wh": 100.0},
        "curriculum_phase": "warmup",
        "sima2_trust": 0.7,
        "econ_slice": {"mpl": 60.0, "wage_parity": 0.8, "energy_wh": 100.0},
    }


def test_ood_tags_affect_ood_risk_level():
    """Test that OOD tags increase ood_risk_level."""
    print("[Test 1/6] OOD tags affect ood_risk_level...")

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Build without OOD tags (baseline)
    cv_baseline = builder.build(**context)
    baseline_ood = cv_baseline.ood_risk_level

    # Build with OOD tags
    context_with_ood = dict(context)
    context_with_ood["episode_metadata"] = {
        "ood_tags": [
            {"severity": 0.8, "source": "visual"},
            {"severity": 0.6, "source": "kinematic"},
        ]
    }
    cv_with_ood = builder.build(**context_with_ood)
    ood_level = cv_with_ood.ood_risk_level

    # OOD risk level should be higher with OOD tags
    assert ood_level > baseline_ood, \
        f"OOD risk level should increase with OOD tags: {ood_level} vs {baseline_ood}"

    # Should be bounded to [0, 1]
    assert 0.0 <= ood_level <= 1.0, \
        f"OOD risk level should be in [0, 1]: {ood_level}"

    # Should reflect the max severity (0.8)
    assert ood_level >= 0.7, \
        f"OOD risk level should be at least near max severity (0.8): {ood_level}"

    print(f"  ✓ OOD tags correctly increase ood_risk_level: {baseline_ood:.3f} → {ood_level:.3f}")


def test_recovery_tags_affect_recovery_priority():
    """Test that Recovery tags affect recovery_priority."""
    print("[Test 2/6] Recovery tags affect recovery_priority...")

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Build without recovery tags (baseline)
    cv_baseline = builder.build(**context)
    baseline_recovery = cv_baseline.recovery_priority

    # Build with high-value recovery tags
    context_with_recovery = dict(context)
    context_with_recovery["episode_metadata"] = {
        "recovery_tags": [
            {"value_add": "high", "correction_type": "re-grasp", "cost_wh": 5.0},
            {"value_add": "high", "correction_type": "pick_from_drop", "cost_wh": 8.0},
        ]
    }
    cv_with_recovery = builder.build(**context_with_recovery)
    recovery_priority = cv_with_recovery.recovery_priority

    # Recovery priority should be higher with recovery tags
    assert recovery_priority > baseline_recovery, \
        f"Recovery priority should increase with recovery tags: {recovery_priority} vs {baseline_recovery}"

    # Should be bounded to [0, 1]
    assert 0.0 <= recovery_priority <= 1.0, \
        f"Recovery priority should be in [0, 1]: {recovery_priority}"

    # High-value recoveries should result in high priority (>= 0.8)
    assert recovery_priority >= 0.8, \
        f"High-value recoveries should result in high priority: {recovery_priority}"

    print(f"  ✓ Recovery tags correctly increase recovery_priority: {baseline_recovery:.3f} → {recovery_priority:.3f}")


def test_ood_severity_ordering():
    """Test that higher OOD severity results in higher ood_risk_level."""
    print("[Test 3/6] OOD severity ordering...")

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Low severity OOD
    context_low = dict(context)
    context_low["episode_metadata"] = {
        "ood_tags": [{"severity": 0.3, "source": "visual"}]
    }
    cv_low = builder.build(**context_low)

    # Medium severity OOD
    context_medium = dict(context)
    context_medium["episode_metadata"] = {
        "ood_tags": [{"severity": 0.6, "source": "visual"}]
    }
    cv_medium = builder.build(**context_medium)

    # High severity OOD
    context_high = dict(context)
    context_high["episode_metadata"] = {
        "ood_tags": [{"severity": 0.9, "source": "visual"}]
    }
    cv_high = builder.build(**context_high)

    # Check ordering
    assert cv_low.ood_risk_level < cv_medium.ood_risk_level, \
        f"Low severity should be < medium: {cv_low.ood_risk_level} vs {cv_medium.ood_risk_level}"
    assert cv_medium.ood_risk_level < cv_high.ood_risk_level, \
        f"Medium severity should be < high: {cv_medium.ood_risk_level} vs {cv_high.ood_risk_level}"

    print(f"  ✓ OOD severity ordering correct: {cv_low.ood_risk_level:.3f} < {cv_medium.ood_risk_level:.3f} < {cv_high.ood_risk_level:.3f}")


def test_recovery_value_add_ordering():
    """Test that high-value recoveries result in higher priority than low-value."""
    print("[Test 4/6] Recovery value_add ordering...")

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Low-value recovery
    context_low = dict(context)
    context_low["episode_metadata"] = {
        "recovery_tags": [
            {"value_add": "low", "correction_type": "minor_adjustment", "cost_wh": 1.0}
        ]
    }
    cv_low = builder.build(**context_low)

    # High-value recovery
    context_high = dict(context)
    context_high["episode_metadata"] = {
        "recovery_tags": [
            {"value_add": "high", "correction_type": "re-grasp", "cost_wh": 5.0}
        ]
    }
    cv_high = builder.build(**context_high)

    # High-value should have higher priority
    assert cv_high.recovery_priority > cv_low.recovery_priority, \
        f"High-value recovery should have higher priority: {cv_high.recovery_priority} vs {cv_low.recovery_priority}"

    print(f"  ✓ Recovery value_add ordering correct: low={cv_low.recovery_priority:.3f} < high={cv_high.recovery_priority:.3f}")


def test_determinism():
    """Test that same inputs produce same ConditionVector."""
    print("[Test 5/6] SIMA-2 coupling determinism...")

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Add consistent OOD and recovery tags
    context["episode_metadata"] = {
        "ood_tags": [
            {"severity": 0.7, "source": "visual"},
            {"severity": 0.5, "source": "kinematic"},
        ],
        "recovery_tags": [
            {"value_add": "high", "correction_type": "re-grasp", "cost_wh": 5.0},
            {"value_add": "medium", "correction_type": "reposition", "cost_wh": 3.0},
        ]
    }

    # Build twice
    cv1 = builder.build(**context)
    cv2 = builder.build(**context)

    # Check determinism
    assert cv1.ood_risk_level == cv2.ood_risk_level, \
        f"OOD risk level should be deterministic: {cv1.ood_risk_level} vs {cv2.ood_risk_level}"
    assert cv1.recovery_priority == cv2.recovery_priority, \
        f"Recovery priority should be deterministic: {cv1.recovery_priority} vs {cv2.recovery_priority}"

    # Check vector determinism
    vec1 = cv1.to_vector()
    vec2 = cv2.to_vector()
    assert (vec1 == vec2).all(), "Vectors should be deterministic"

    print(f"  ✓ SIMA-2 coupling is deterministic")


def test_bounded_outputs():
    """Test that all computed values are bounded to [0, 1]."""
    print("[Test 6/6] Bounded outputs...")

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Test with extreme values
    extreme_contexts = [
        {
            **context,
            "episode_metadata": {
                "ood_tags": [{"severity": 10.0, "source": "visual"}]  # Extreme severity
            }
        },
        {
            **context,
            "episode_metadata": {
                "ood_tags": [
                    {"severity": 0.9, "source": "visual"},
                    {"severity": 0.8, "source": "kinematic"},
                    {"severity": 0.7, "source": "temporal"},
                    {"severity": 0.6, "source": "visual"},
                ]  # Many tags
            }
        },
        {
            **context,
            "episode_metadata": {
                "recovery_tags": [
                    {"value_add": "high", "correction_type": "re-grasp", "cost_wh": 1000.0}  # Extreme cost
                ]
            }
        },
    ]

    for i, ctx in enumerate(extreme_contexts):
        cv = builder.build(**ctx)

        # Check bounds
        assert 0.0 <= cv.ood_risk_level <= 1.0, \
            f"Context {i}: OOD risk level out of bounds: {cv.ood_risk_level}"
        assert 0.0 <= cv.recovery_priority <= 1.0, \
            f"Context {i}: Recovery priority out of bounds: {cv.recovery_priority}"

    print(f"  ✓ All outputs correctly bounded to [0, 1]")


def main():
    print("\n" + "="*60)
    print("SIMA-2 → ConditionVector Coupling Smoke Test")
    print("="*60 + "\n")

    try:
        test_ood_tags_affect_ood_risk_level()
        test_recovery_tags_affect_recovery_priority()
        test_ood_severity_ordering()
        test_recovery_value_add_ordering()
        test_determinism()
        test_bounded_outputs()

        print("\n" + "="*60)
        print("✅ All SIMA-2 coupling tests passed!")
        print("="*60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
