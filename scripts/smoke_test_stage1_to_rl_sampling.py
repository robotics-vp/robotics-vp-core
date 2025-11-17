#!/usr/bin/env python3
"""
Smoke test for Stage 1 → RL episode sampling handshake.

Validates:
- Datapacks can be loaded and parsed
- Episode descriptors are created correctly
- All required fields are present
- Sampling weights are computed correctly
"""

import sys
import json
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.valuation.datapack_schema import (
    DataPackMeta,
    ConditionProfile,
    ObjectiveProfile,
    GuidanceProfile,
    AttributionProfile,
)
from src.rl.episode_sampling import datapack_to_rl_episode_descriptor


def create_test_datapack() -> DataPackMeta:
    """Create a test datapack for validation."""
    condition = ConditionProfile(
        task_name="drawer_vase",
        engine_type="pybullet",
        world_id="test_world",
        objective_vector=[1.0, 1.0, 1.0],
    )

    objective_profile = ObjectiveProfile(
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="drawer_vase",
        customer_segment="balanced",
        market_region="US",
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
    )

    guidance_profile = GuidanceProfile(
        is_good=True,
        quality_label="high_value",
        env_name="drawer_vase",
        engine_type="pybullet",
        task_type="drawer_vase",
        customer_segment="balanced",
        objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
        main_driver="throughput_gain",
        delta_mpl=5.0,
        delta_error=-0.01,
        delta_energy_Wh=-0.5,
        delta_J=2.0,
        semantic_tags=["test", "smoke", "drawer"],
    )

    attribution_profile = AttributionProfile(
        env_name="drawer_vase",
        engine_type="pybullet",
        delta_mpl=5.0,
        delta_error=-0.01,
        delta_J=2.0,
        trust_score=0.8,
        w_econ=0.9,
        tier=2,
    )

    datapack = DataPackMeta(
        pack_id="test_datapack_001",
        task_name="drawer_vase",
        env_type="pybullet",
        schema_version="2.0-test",
        condition=condition,
        objective_profile=objective_profile,
        guidance_profile=guidance_profile,
        attribution=attribution_profile,
        semantic_tags=["test", "smoke"],
    )

    return datapack


def test_stage1_to_rl_sampling():
    """Test Stage 1 → RL episode sampling handshake."""
    print("=" * 70)
    print("Smoke Test: Stage 1 → RL Episode Sampling Handshake")
    print("=" * 70)

    try:
        # Test 1: Create test datapack
        print("\n[1/5] Creating test datapack...")
        datapack = create_test_datapack()
        print(f"✓ Created datapack: {datapack.pack_id}")

        # Test 2: Convert to episode descriptor
        print("\n[2/5] Converting to RL episode descriptor...")
        descriptor = datapack_to_rl_episode_descriptor(datapack)
        print(f"✓ Created episode descriptor")

        # Test 3: Validate required fields
        print("\n[3/5] Validating required fields...")
        required_fields = [
            "pack_id",
            "env_name",
            "backend",
            "objective_vector",
            "objective_preset",
            "tier",
            "trust_score",
            "sampling_weight",
            "semantic_tags",
            "focus_areas",
            "priority",
            "episode_length",
        ]

        for field in required_fields:
            assert field in descriptor, f"Missing required field: {field}"
            print(f"  ✓ {field}: {descriptor[field]}")

        print("✓ All required fields present")

        # Test 4: Validate field values
        print("\n[4/5] Validating field values...")

        # Check env_name
        assert descriptor['env_name'] == "drawer_vase", f"env_name mismatch: {descriptor['env_name']}"
        print("  ✓ env_name: drawer_vase")

        # Check backend
        assert descriptor['backend'] == "pybullet", f"backend mismatch: {descriptor['backend']}"
        print("  ✓ backend: pybullet")

        # Check objective_vector length
        assert len(descriptor['objective_vector']) >= 4, "objective_vector too short"
        print(f"  ✓ objective_vector: {descriptor['objective_vector']}")

        # Check tier
        assert descriptor['tier'] == 2, f"tier mismatch: {descriptor['tier']}"
        print("  ✓ tier: 2")

        # Check trust_score
        assert 0.0 <= descriptor['trust_score'] <= 1.0, f"trust_score out of range: {descriptor['trust_score']}"
        print(f"  ✓ trust_score: {descriptor['trust_score']:.3f}")

        # Check sampling_weight is computed
        assert descriptor['sampling_weight'] > 0, "sampling_weight should be positive"
        expected_weight = descriptor['trust_score'] * (1.0 + 0.5 * descriptor['tier'])
        assert abs(descriptor['sampling_weight'] - expected_weight) < 0.01, "sampling_weight calculation incorrect"
        print(f"  ✓ sampling_weight: {descriptor['sampling_weight']:.3f}")

        # Check semantic_tags
        assert isinstance(descriptor['semantic_tags'], list), "semantic_tags should be a list"
        assert len(descriptor['semantic_tags']) > 0, "semantic_tags should not be empty"
        print(f"  ✓ semantic_tags: {descriptor['semantic_tags']}")

        print("✓ All field values valid")

        # Test 5: Test with multiple datapacks
        print("\n[5/5] Testing batch conversion...")
        datapacks = [create_test_datapack() for _ in range(3)]
        descriptors = [datapack_to_rl_episode_descriptor(dp) for dp in datapacks]

        assert len(descriptors) == 3, f"Expected 3 descriptors, got {len(descriptors)}"
        print(f"✓ Batch conversion: {len(descriptors)} descriptors created")

        # Verify all have correct structure
        for i, desc in enumerate(descriptors):
            for field in required_fields:
                assert field in desc, f"Descriptor {i} missing field: {field}"

        print("✓ All batch descriptors valid")

        print("\n" + "=" * 70)
        print("Stage 1 → RL Sampling Handshake Smoke Test: PASSED")
        print("=" * 70)
        print("Summary:")
        print(f"  ✓ Datapack creation")
        print(f"  ✓ Episode descriptor conversion")
        print(f"  ✓ Required fields validation")
        print(f"  ✓ Field values validation")
        print(f"  ✓ Batch conversion")
        print("\nDescriptor keys:")
        for key in sorted(descriptor.keys()):
            print(f"  - {key}")

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("Stage 1 → RL Sampling Handshake Smoke Test: FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_stage1_to_rl_sampling()
    sys.exit(0 if success else 1)
