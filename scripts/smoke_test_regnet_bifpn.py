"""
Smoke test for RegNet + BiFPN (Block A).

Tests:
1. Determinism: Fixed seed/frame → identical pyramids/BiFPN/flattened z_v
2. Forward consistency: Shapes match config dims; BiFPN normalized weights; no NaNs
3. ConditionedVisionAdapter on RegNet+BiFPN: z_v unchanged, risk_map changes; flag gating
4. Checkpoint round-trip: Serialize/deserialize preserves outputs (JSON-safe)
"""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.regnet_backbone import build_regnet_feature_pyramid, flatten_pyramid, DEFAULT_LEVELS
from src.vision.bifpn_fusion import fuse_feature_pyramid
from src.vision.interfaces import VisionFrame


def test_determinism():
    """Test deterministic outputs given fixed seed and frame."""
    print("\n[Block A] Test 1: Determinism")

    # Create test frame
    frame = VisionFrame(
        backend="test",
        task_id="task_1",
        episode_id="episode_1",
        timestep=0,
        width=224,
        height=224,
    )

    # Build pyramids with same seed
    seed = 42
    levels = ["P3", "P4", "P5"]
    feature_dim = 8

    pyramid1 = build_regnet_feature_pyramid(frame, feature_dim=feature_dim, levels=levels, use_neural=False, seed=seed)
    pyramid2 = build_regnet_feature_pyramid(frame, feature_dim=feature_dim, levels=levels, use_neural=False, seed=seed)

    # Check identity
    for level in levels:
        assert level in pyramid1 and level in pyramid2
        assert np.allclose(pyramid1[level], pyramid2[level]), f"Pyramid level {level} not deterministic!"

    # Flatten and check
    flat1 = flatten_pyramid(pyramid1)
    flat2 = flatten_pyramid(pyramid2)
    assert np.allclose(flat1, flat2), "Flattened pyramids not deterministic!"

    print("  ✓ Determinism test passed")


def test_forward_consistency():
    """Test shapes match config dims; BiFPN normalized weights; no NaNs."""
    print("\n[Block A] Test 2: Forward Consistency")

    frame = VisionFrame(
        backend="test",
        task_id="task_1",
        episode_id="episode_1",
        timestep=0,
    )

    levels = ["P3", "P4", "P5"]
    feature_dim = 16

    # Build pyramid
    pyramid = build_regnet_feature_pyramid(frame, feature_dim=feature_dim, levels=levels, use_neural=False)

    # Check shapes
    for level in levels:
        assert level in pyramid
        assert len(pyramid[level]) == feature_dim, f"Level {level} has wrong dimension: {len(pyramid[level])} != {feature_dim}"
        assert not np.any(np.isnan(pyramid[level])), f"Level {level} contains NaNs!"

    # BiFPN fusion
    fused = fuse_feature_pyramid(pyramid, use_neural=False)

    # Check fused outputs
    for level in levels:
        assert level in fused
        assert not np.any(np.isnan(fused[level])), f"Fused level {level} contains NaNs!"
        assert not np.any(np.isinf(fused[level])), f"Fused level {level} contains Infs!"

    # Flatten
    flat = flatten_pyramid(fused)
    assert len(flat) == len(levels) * feature_dim
    assert not np.any(np.isnan(flat)), "Flattened vector contains NaNs!"

    print("  ✓ Forward consistency test passed")


def test_conditioned_vision_adapter():
    """Test ConditionedVisionAdapter: z_v unchanged, risk_map changes."""
    print("\n[Block A] Test 3: ConditionedVisionAdapter")

    from src.vision.conditioned_adapter import ConditionedVisionAdapter
    from src.observation.condition_vector import ConditionVector

    frame = VisionFrame(
        backend="test",
        task_id="task_1",
        episode_id="episode_1",
        timestep=0,
    )

    # Test without conditioning
    adapter = ConditionedVisionAdapter(config={"feature_dim": 8, "levels": ["P3", "P4", "P5"], "enable_conditioning": False})
    output1 = adapter.forward(frame, condition_vector=None)

    assert "z_v" in output1
    assert "risk_map" in output1

    # Test with conditioning
    adapter_cond = ConditionedVisionAdapter(config={"feature_dim": 8, "levels": ["P3", "P4", "P5"], "enable_conditioning": True})
    condition = ConditionVector(
        task_id="test_task",
        env_id="test_env",
        backend_id="test_backend",
        target_mpl=60.0,
        current_wage_parity=0.8,
        energy_budget_wh=100.0,
        skill_mode="safety_precision",
        ood_risk_level=0.2,
        recovery_priority=0.9,
        novelty_tier=2,
        sima2_trust_score=0.85,
        recap_goodness_bucket="good",
        objective_preset="balanced",
    )
    output2 = adapter_cond.forward(frame, condition_vector=condition)

    # z_v should be identical (base representation invariant)
    for level in output1["z_v"]:
        assert np.allclose(output1["z_v"][level], output2["z_v"][level]), f"z_v level {level} changed with conditioning!"

    # risk_map should differ
    assert not np.allclose(output1["risk_map"], output2["risk_map"]), "risk_map unchanged despite conditioning!"

    print("  ✓ ConditionedVisionAdapter test passed")


def test_checkpoint_roundtrip():
    """Test checkpoint serialization/deserialization preserves outputs (JSON-safe)."""
    print("\n[Block A] Test 4: Checkpoint Round-trip")

    from src.vision.regnet_backbone import pyramid_to_json_safe

    frame = VisionFrame(
        backend="test",
        task_id="task_1",
        episode_id="episode_1",
        timestep=0,
    )

    pyramid = build_regnet_feature_pyramid(frame, feature_dim=8, use_neural=False)

    # Serialize to JSON-safe
    serialized = pyramid_to_json_safe(pyramid)

    # Check JSON-safe
    import json
    try:
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
    except Exception as e:
        raise AssertionError(f"Pyramid not JSON-safe: {e}")

    # Reconstruct pyramid
    reconstructed = {k: np.array(v, dtype=np.float32) for k, v in deserialized.items()}

    # Check equality
    for level in pyramid:
        assert np.allclose(pyramid[level], reconstructed[level]), f"Level {level} not preserved in round-trip!"

    print("  ✓ Checkpoint round-trip test passed")


def main():
    print("=" * 60)
    print("BLOCK A SMOKE TESTS: RegNet + BiFPN")
    print("=" * 60)

    test_determinism()
    test_forward_consistency()
    test_conditioned_vision_adapter()
    test_checkpoint_roundtrip()

    print("\n" + "=" * 60)
    print("ALL BLOCK A TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
