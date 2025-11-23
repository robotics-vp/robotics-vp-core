"""
Smoke test for TFD streaming/session runtime.

Tests that TFDSession handles:
- Sequential instructions
- Conflict resolution (safety beats speed)
- Deterministic canonical instruction
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.tfd.runtime import TFDSession


def test_basic_session():
    """Test basic session operations."""
    print("Testing basic TFD session...")

    session = TFDSession()

    # Add a safety instruction
    inst1 = session.add_instruction("Be very careful with the vase")
    assert inst1.status == "accepted", f"Expected accepted, got {inst1.status}"

    # Get canonical
    canonical = session.get_canonical_instruction()
    assert canonical is not None, "Expected canonical instruction"
    assert canonical.raw_text == "Be very careful with the vase"

    # Add another instruction
    inst2 = session.add_instruction("Work 20% faster")
    assert inst2.status == "accepted"

    # Both should be in history
    assert len(session.instructions) == 2

    print("✓ Basic session operations work")


def test_safety_first_conflict():
    """Test that safety instructions win over speed."""
    print("Testing safety-first conflict resolution...")

    session = TFDSession()

    # Add speed instruction first
    session.add_instruction("Go as fast as possible")

    # Then add safety instruction
    session.add_instruction("Be very careful")

    # Canonical should prioritize safety
    canonical = session.get_canonical_instruction()
    assert canonical is not None

    cv = canonical.condition_vector
    assert cv is not None
    # Should have low risk tolerance (safety)
    assert cv.risk_tolerance is not None
    assert cv.risk_tolerance < 0.5, f"Expected risk_tolerance < 0.5, got {cv.risk_tolerance}"

    print(f"✓ Safety instruction won (risk_tolerance={cv.risk_tolerance})")


def test_mixed_instruction_blending():
    """Test 'be quick but careful' blending."""
    print("Testing mixed instruction blending...")

    session = TFDSession()

    # Add safety first
    session.add_instruction("Be careful")

    # Then add speed
    session.add_instruction("Hurry")

    # Should blend: moderate risk, with time budget
    canonical = session.get_canonical_instruction()
    assert canonical is not None

    cv = canonical.condition_vector
    assert cv is not None

    # Should have moderate risk_tolerance (blended)
    assert cv.risk_tolerance is not None
    assert 0.3 < cv.risk_tolerance < 0.6, f"Expected blended risk_tolerance, got {cv.risk_tolerance}"

    # Should have time budget (acknowledge speed)
    assert cv.time_budget_sec is not None
    assert cv.time_budget_sec > 0

    print(f"✓ Blended instruction (risk={cv.risk_tolerance}, time={cv.time_budget_sec})")


def test_determinism():
    """Test that same sequence produces same canonical."""
    print("Testing determinism...")

    # Session 1
    session1 = TFDSession()
    session1.add_instruction("Be careful")
    session1.add_instruction("Go fast")

    canonical1 = session1.get_canonical_instruction()
    cv1 = canonical1.condition_vector if canonical1 else None

    # Session 2 (same sequence)
    session2 = TFDSession()
    session2.add_instruction("Be careful")
    session2.add_instruction("Go fast")

    canonical2 = session2.get_canonical_instruction()
    cv2 = canonical2.condition_vector if canonical2 else None

    # Should produce identical condition vectors
    assert cv1 is not None and cv2 is not None
    assert cv1.risk_tolerance == cv2.risk_tolerance
    assert cv1.skill_mode == cv2.skill_mode

    print("✓ Deterministic session behavior confirmed")


def test_session_summary():
    """Test session summary for logging."""
    print("Testing session summary...")

    session = TFDSession()
    session.add_instruction("Explore new approaches")

    summary = session.get_session_summary()
    assert summary["active"] is True
    assert summary["instruction_count"] == 1
    assert "canonical_text" in summary
    assert summary["skill_mode"] is not None

    print(f"✓ Session summary: {summary}")


def test_clear_session():
    """Test session clearing."""
    print("Testing session clear...")

    session = TFDSession()
    session.add_instruction("Be careful")
    session.add_instruction("Go fast")

    assert len(session.instructions) == 2

    session.clear()

    assert len(session.instructions) == 0
    assert session.get_canonical_instruction() is None

    print("✓ Session cleared successfully")


def main():
    print("=== TFD Streaming/Session Smoke Tests ===\n")

    try:
        test_basic_session()
        test_safety_first_conflict()
        test_mixed_instruction_blending()
        test_determinism()
        test_session_summary()
        test_clear_session()

        print("\n=== All TFD Session Tests Passed ===")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
