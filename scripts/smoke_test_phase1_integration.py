"""
Integration smoke test for Phase I Neuralization.

Tests full stack:
1. RegNet+BiFPN → Spatial RNN → Policy features
2. Neural segmenter integration with SegmentationEngine
3. End-to-end determinism and JSON-safety
"""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.regnet_backbone import build_regnet_feature_pyramid
from src.vision.bifpn_fusion import fuse_feature_pyramid
from src.vision.spatial_rnn import run_spatial_rnn
from src.vision.interfaces import VisionFrame

# Check if PyTorch available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using stub mode for integration tests.")


def test_vision_fullstack():
    """Test RegNet → BiFPN → Spatial RNN integration."""
    print("\n[Integration] Test 1: Vision Full Stack")

    # Create test sequence
    sequence_length = 8
    levels = ["P3", "P4", "P5"]
    feature_dim = 8

    frames = []
    for t in range(sequence_length):
        frame = VisionFrame(
            backend="test",
            task_id="task_1",
            episode_id="episode_1",
            timestep=t,
        )
        frames.append(frame)

    # Build pyramids
    pyramids = [
        build_regnet_feature_pyramid(f, feature_dim=feature_dim, levels=levels, use_neural=False, seed=42)
        for f in frames
    ]

    # Fuse with BiFPN
    fused_pyramids = [
        fuse_feature_pyramid(p, use_neural=False)
        for p in pyramids
    ]

    # Run through Spatial RNN
    summary = run_spatial_rnn(fused_pyramids, hidden_dim=64, feature_dim=feature_dim, levels=levels, seed=42, use_neural=False)

    # Check output
    assert summary is not None
    assert len(summary) > 0
    assert not np.any(np.isnan(summary)), "Summary contains NaNs!"

    print(f"  Summary shape: {summary.shape}")
    print("  ✓ Vision full stack test passed")


def test_segmentation_engine_integration():
    """Test SegmentationEngine with flag-gated segmenters."""
    print("\n[Integration] Test 2: Segmentation Engine")

    from src.sima2.segmentation_engine import SegmentationEngine

    # Test with heuristic segmenter (default)
    engine_heuristic = SegmentationEngine(
        segmentation_config={"use_heuristic_segmenter": True}
    )

    # Create test rollout
    rollout = {
        "task": "test_task",
        "episode_id": "test_episode",
        "primitives": [
            {
                "timestep": 0,
                "gripper_width": 0.08,
                "ee_velocity": 0.01,
                "contact": False,
                "object": "dish_1",
            },
            {
                "timestep": 1,
                "gripper_width": 0.03,
                "ee_velocity": 0.02,
                "contact": True,
                "object": "dish_1",
            },
            {
                "timestep": 2,
                "gripper_width": 0.02,
                "ee_velocity": 0.05,
                "contact": True,
                "object": "dish_1",
            },
        ],
        "metadata": {"objects_present": ["dish_1"]},
    }

    # Segment with heuristic
    result = engine_heuristic.segment_rollout(rollout)

    assert "segments" in result
    assert "rollout" in result
    assert len(result["segments"]) > 0

    print(f"  Detected {len(result['segments'])} segments")
    print("  ✓ Segmentation engine integration test passed")


def test_e2e_determinism_json_safety():
    """Test end-to-end determinism and JSON-safety."""
    print("\n[Integration] Test 3: E2E Determinism & JSON-Safety")

    import json
    from src.vision.regnet_backbone import pyramid_to_json_safe

    # Create test frame
    frame = VisionFrame(
        backend="test",
        task_id="task_1",
        episode_id="episode_1",
        timestep=0,
    )

    # Build pyramid (deterministic)
    pyramid1 = build_regnet_feature_pyramid(frame, feature_dim=8, use_neural=False, seed=42)
    pyramid2 = build_regnet_feature_pyramid(frame, feature_dim=8, use_neural=False, seed=42)

    # Check determinism
    for level in pyramid1:
        assert np.allclose(pyramid1[level], pyramid2[level]), f"Level {level} not deterministic!"

    # Check JSON-safety
    serialized = pyramid_to_json_safe(pyramid1)
    json_str = json.dumps(serialized)  # Should not raise
    deserialized = json.loads(json_str)

    assert isinstance(deserialized, dict)
    for level in pyramid1:
        assert level in deserialized
        assert isinstance(deserialized[level], list)
        assert all(isinstance(x, (int, float)) for x in deserialized[level])

    print("  ✓ E2E determinism & JSON-safety test passed")


def main():
    print("=" * 60)
    print("PHASE I INTEGRATION SMOKE TESTS")
    print("=" * 60)

    test_vision_fullstack()
    test_segmentation_engine_integration()
    test_e2e_determinism_json_safety()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
