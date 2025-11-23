"""
Smoke test for TFD → ConditionVector integration.

Tests that TFDInstructions deterministically affect ConditionVector fields and
are properly logged in metadata.

Run:
    python -m scripts.smoke_test_tfd_condition_vector_integration
"""
import sys
from typing import Dict, Any

from src.tfd.compiler import TextFrontDoor
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.observation.adapter import ObservationAdapter
from src.logging.episode_logger import EpisodeLogger, _condition_summary
from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot


def build_dummy_context() -> Dict[str, Any]:
    """Build minimal context for ConditionVector construction."""
    return {
        "episode_config": {"task_id": "test_task", "env_id": "test_env", "backend_id": "test_backend"},
        "econ_state": {"target_mpl": 60.0, "current_wage_parity": 0.8, "energy_budget_wh": 100.0},
        "curriculum_phase": "warmup",
        "sima2_trust": 0.7,
        "econ_slice": {"mpl": 60.0, "wage_parity": 0.8, "energy_wh": 100.0},
    }


def test_tfd_instruction_parsing():
    """Test that TFD instructions parse correctly."""
    print("[Test 1/5] TFD instruction parsing...")

    tfd = TextFrontDoor()

    # Test speed mode instruction
    instruction = tfd.process_instruction("Go fast!")
    assert instruction.status == "accepted", f"Expected accepted, got {instruction.status}"
    assert instruction.condition_vector is not None, "Expected condition_vector"
    assert instruction.condition_vector.skill_mode == "speed", f"Expected speed mode, got {instruction.condition_vector.skill_mode}"

    # Test safety mode instruction
    instruction = tfd.process_instruction("Be very careful")
    assert instruction.status == "accepted"
    assert instruction.condition_vector.skill_mode == "safety_critical"
    assert instruction.condition_vector.safety_emphasis > 0.5

    # Test exploration mode instruction
    instruction = tfd.process_instruction("Explore weird objects")
    assert instruction.status == "accepted"
    assert instruction.condition_vector.skill_mode == "exploration"
    assert instruction.condition_vector.exploration_priority is not None and instruction.condition_vector.exploration_priority > 0.5

    print("  ✓ TFD parsing works correctly")


def test_tfd_integration_disabled_by_default():
    """Test that TFD integration is disabled by default (flag-gated)."""
    print("[Test 2/5] TFD integration disabled by default...")

    tfd = TextFrontDoor()
    instruction = tfd.process_instruction("Go fast!")
    instruction_dict = instruction.to_dict()

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Build WITHOUT enable_tfd_integration flag (default False)
    cv_without_tfd = builder.build(
        **context,
        tfd_instruction=instruction_dict,
        enable_tfd_integration=False,  # Explicitly False
    )

    # Build with NO tfd_instruction
    cv_baseline = builder.build(**context)

    # Skill mode should NOT be affected by TFD when integration is disabled
    # (both should use default skill mode from resolver)
    assert cv_without_tfd.skill_mode == cv_baseline.skill_mode, \
        f"Skill mode should be identical when TFD integration is disabled: {cv_without_tfd.skill_mode} vs {cv_baseline.skill_mode}"

    # TFD metadata should still be logged in metadata even when integration is disabled
    assert cv_without_tfd.metadata.get("tfd_metadata") is not None, \
        "TFD metadata should be logged even when integration is disabled"

    print("  ✓ TFD integration correctly disabled by default")


def test_tfd_integration_enabled():
    """Test that TFD integration affects ConditionVector when enabled."""
    print("[Test 3/5] TFD integration enabled...")

    tfd = TextFrontDoor()
    instruction = tfd.process_instruction("Go fast!")
    instruction_dict = instruction.to_dict()

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    # Build WITH enable_tfd_integration=True
    cv_with_tfd = builder.build(
        **context,
        tfd_instruction=instruction_dict,
        enable_tfd_integration=True,  # Enable TFD integration
    )

    # Skill mode SHOULD be affected by TFD instruction
    assert cv_with_tfd.skill_mode == "speed", \
        f"Expected skill_mode='speed' from TFD instruction, got '{cv_with_tfd.skill_mode}'"

    # TFD metadata should be logged
    assert cv_with_tfd.metadata.get("tfd_metadata") is not None, \
        "TFD metadata should be logged"

    tfd_metadata = cv_with_tfd.metadata["tfd_metadata"]
    assert tfd_metadata.get("status") == "accepted"
    assert tfd_metadata.get("raw_text") == "Go fast!"
    assert tfd_metadata.get("advisory_fields") is not None
    assert tfd_metadata["advisory_fields"].get("efficiency_preference") == 0.2

    print("  ✓ TFD integration correctly affects ConditionVector when enabled")


def test_tfd_determinism():
    """Test that same TFD instruction produces same ConditionVector."""
    print("[Test 4/5] TFD determinism...")

    tfd = TextFrontDoor()
    instructions = [
        "Go fast!",
        "Be very careful",
        "Explore weird objects",
        "Prioritize novelty",
    ]

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    for text in instructions:
        # Build condition vector twice with same instruction
        instruction1 = tfd.process_instruction(text)
        instruction2 = tfd.process_instruction(text)

        cv1 = builder.build(**context, tfd_instruction=instruction1.to_dict(), enable_tfd_integration=True)
        cv2 = builder.build(**context, tfd_instruction=instruction2.to_dict(), enable_tfd_integration=True)

        # Compare critical fields
        assert cv1.skill_mode == cv2.skill_mode, f"Skill mode mismatch for '{text}': {cv1.skill_mode} vs {cv2.skill_mode}"
        assert cv1.novelty_tier == cv2.novelty_tier, f"Novelty tier mismatch for '{text}'"
        assert cv1.curriculum_phase == cv2.curriculum_phase, f"Curriculum phase mismatch for '{text}'"

        # Compare vectors
        vec1 = cv1.to_vector()
        vec2 = cv2.to_vector()
        assert (vec1 == vec2).all(), f"Vector mismatch for '{text}'"

        # Compare metadata fingerprints
        meta1 = cv1.metadata.get("tfd_metadata", {})
        meta2 = cv2.metadata.get("tfd_metadata", {})
        assert meta1 == meta2, f"Metadata mismatch for '{text}'"

    print("  ✓ TFD integration is deterministic")


def test_tfd_logging_integration():
    """Test that TFD metadata flows through to episode logging."""
    print("[Test 5/5] TFD logging integration...")

    tfd = TextFrontDoor()
    instruction = tfd.process_instruction("Explore weird objects")
    instruction_dict = instruction.to_dict()

    builder = ConditionVectorBuilder()
    context = build_dummy_context()

    cv = builder.build(**context, tfd_instruction=instruction_dict, enable_tfd_integration=True)

    # Test condition_summary includes TFD metadata
    summary = _condition_summary(cv)
    assert summary is not None
    assert summary.get("tfd_metadata") is not None, "TFD metadata should be in condition summary"

    tfd_metadata = summary["tfd_metadata"]
    assert tfd_metadata.get("status") == "accepted"
    assert tfd_metadata.get("raw_text") == "Explore weird objects"
    assert tfd_metadata.get("intent_type") is not None
    assert tfd_metadata.get("advisory_fields") is not None
    assert "exploration_priority" in tfd_metadata["advisory_fields"]

    # Test that metadata is JSON-safe
    import json
    try:
        json.dumps(summary)
    except Exception as e:
        raise AssertionError(f"Condition summary is not JSON-safe: {e}")

    print("  ✓ TFD metadata correctly flows through logging")


def main():
    print("\n" + "="*60)
    print("TFD → ConditionVector Integration Smoke Test")
    print("="*60 + "\n")

    try:
        test_tfd_instruction_parsing()
        test_tfd_integration_disabled_by_default()
        test_tfd_integration_enabled()
        test_tfd_determinism()
        test_tfd_logging_integration()

        print("\n" + "="*60)
        print("✅ All TFD integration tests passed!")
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
