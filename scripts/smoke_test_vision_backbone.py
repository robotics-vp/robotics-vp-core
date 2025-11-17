#!/usr/bin/env python3
"""
Smoke tests for VisionBackbone abstraction and embedding utilities.

Tests:
1. DummyBackbone construction and encoding
2. MetaDINOBackbone soft-fail to DummyBackbone
3. Episode embedding integration with OpenVLAController
4. Embedding utilities (novelty, regime clustering, statistics)
5. DataPackMeta embedding serialization
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import numpy as np
from PIL import Image
import tempfile


def test_dummy_backbone():
    """Test DummyBackbone construction and basic encoding."""
    print("=" * 60)
    print("Test 1: DummyBackbone construction and encoding")
    print("=" * 60)

    from src.vla.backbones.dummy_backbone import DummyBackbone

    # Test construction
    backbone = DummyBackbone(embedding_dim=384)
    print(f"  Created DummyBackbone: {backbone.name}")
    print(f"  Embedding dimension: {backbone.embedding_dim}")
    assert backbone.embedding_dim == 384

    # Test single frame encoding
    img = Image.new("RGB", (128, 128), color="blue")
    embedding = backbone.encode_frame(img)
    print(f"  Single frame embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    assert embedding.shape == (384,)
    assert np.linalg.norm(embedding) > 0

    # Test determinism (same input should give same output)
    embedding2 = backbone.encode_frame(img)
    assert np.allclose(embedding, embedding2), "DummyBackbone should be deterministic"
    print("  Determinism check: PASSED")

    # Test sequence encoding
    frames = [
        Image.new("RGB", (128, 128), color="red"),
        Image.new("RGB", (128, 128), color="green"),
        Image.new("RGB", (128, 128), color="blue"),
    ]
    seq_embedding = backbone.encode_sequence(frames)
    print(f"  Sequence embedding (3 frames) shape: {seq_embedding.shape}")
    print(f"  Sequence embedding norm: {np.linalg.norm(seq_embedding):.4f}")
    assert seq_embedding.shape == (384,)

    print("  TEST 1 PASSED: DummyBackbone works correctly")
    print()


def test_meta_dino_soft_fail():
    """Test MetaDINOBackbone soft-fails to DummyBackbone when dependencies unavailable."""
    print("=" * 60)
    print("Test 2: MetaDINOBackbone soft-fail to DummyBackbone")
    print("=" * 60)

    from src.vla.backbones.meta_dino_backbone import MetaDINOBackbone

    # This should soft-fail to DummyBackbone if transformers/timm not available
    backbone = MetaDINOBackbone(
        model_name="facebook/dinov2-small",
        device="cpu",  # Use CPU to avoid CUDA issues
    )
    print(f"  Created backbone: {backbone.name}")
    print(f"  Embedding dimension: {backbone.embedding_dim}")
    print(f"  Real DINO available: {backbone.available}")

    # Should still work regardless of fallback
    img = Image.new("RGB", (224, 224), color="gray")
    embedding = backbone.encode_frame(img)
    print(f"  Frame embedding shape: {embedding.shape}")
    assert embedding.shape == (backbone.embedding_dim,)

    print("  TEST 2 PASSED: MetaDINOBackbone soft-fail works")
    print()


def test_openvla_controller_with_backbone():
    """Test OpenVLAController with vision backbone integration."""
    print("=" * 60)
    print("Test 3: OpenVLAController vision backbone integration")
    print("=" * 60)

    from src.vla.openvla_controller import OpenVLAController, OpenVLAConfig

    # Configure with vision backbone enabled
    cfg = OpenVLAConfig(
        use_vision_backbone=True,
        vision_backbone_type="dummy",
    )
    controller = OpenVLAController(cfg)
    print(f"  Created OpenVLAController with config:")
    print(f"    use_vision_backbone={cfg.use_vision_backbone}")
    print(f"    vision_backbone_type={cfg.vision_backbone_type}")

    # Load model (VLA model may not be available, but backbone should load)
    controller.load_model()
    print(f"  VLA model available: {controller.available}")
    print(f"  Vision backbone available: {controller.has_vision_backbone()}")
    assert controller.has_vision_backbone(), "Vision backbone should be loaded"

    # Start episode
    controller.start_episode()
    print("  Started episode (frame buffer reset)")

    # Simulate episode with multiple frames
    frames = [
        Image.new("RGB", (256, 256), color="red"),
        Image.new("RGB", (256, 256), color="green"),
        Image.new("RGB", (256, 256), color="blue"),
        Image.new("RGB", (256, 256), color="yellow"),
    ]

    for i, frame in enumerate(frames):
        action = controller.predict_action(frame, "open drawer safely")
        print(f"    Frame {i}: action dx={action['dx']:.2f}, dy={action['dy']:.2f}, gripper={action['gripper']:.2f}")

    # End episode and get embedding
    episode_embedding = controller.end_episode()
    print(f"  Episode embedding computed:")
    print(f"    Shape: {episode_embedding.shape}")
    print(f"    Norm: {np.linalg.norm(episode_embedding):.4f}")
    print(f"    Min: {np.min(episode_embedding):.4f}")
    print(f"    Max: {np.max(episode_embedding):.4f}")
    assert episode_embedding.shape == (384,)

    # Check embedding log
    emb_log = controller.get_embedding_log()
    print(f"  Embedding log entries: {len(emb_log)}")

    print("  TEST 3 PASSED: OpenVLAController backbone integration works")
    print()


def test_embedding_novelty():
    """Test embedding novelty computation."""
    print("=" * 60)
    print("Test 4: Embedding novelty computation")
    print("=" * 60)

    from src.valuation.embedding_utils import compute_embedding_novelty

    # Create reference embeddings (clustered around origin)
    np.random.seed(42)
    reference_embeddings = [np.random.randn(64) * 0.1 for _ in range(10)]

    # Novel embedding (far from cluster)
    novel_embedding = np.random.randn(64) * 2.0

    # Similar embedding (close to cluster)
    similar_embedding = np.random.randn(64) * 0.1

    novelty_high = compute_embedding_novelty(novel_embedding, reference_embeddings)
    novelty_low = compute_embedding_novelty(similar_embedding, reference_embeddings)

    print(f"  Reference embeddings: {len(reference_embeddings)} vectors")
    print(f"  Novel embedding novelty: {novelty_high:.4f}")
    print(f"  Similar embedding novelty: {novelty_low:.4f}")
    assert novelty_high > novelty_low, "Novel embedding should have higher novelty score"
    print("  Novelty ordering: CORRECT (novel > similar)")

    # Test edge case: empty reference set
    first_novelty = compute_embedding_novelty(novel_embedding, [])
    print(f"  First embedding novelty (empty ref): {first_novelty:.4f}")
    assert first_novelty == 1.0, "First embedding should be maximally novel"

    print("  TEST 4 PASSED: Embedding novelty computation works")
    print()


def test_regime_clustering():
    """Test regime clustering functionality."""
    print("=" * 60)
    print("Test 5: Regime clustering")
    print("=" * 60)

    from src.valuation.embedding_utils import (
        cluster_embeddings_kmeans,
        compute_regime_cluster,
        build_regime_centroids_from_embeddings,
    )

    # Create synthetic embeddings for 3 regimes
    np.random.seed(42)
    regime_a = [np.array([1.0, 0.0, 0.0]) + np.random.randn(3) * 0.1 for _ in range(5)]
    regime_b = [np.array([0.0, 1.0, 0.0]) + np.random.randn(3) * 0.1 for _ in range(5)]
    regime_c = [np.array([0.0, 0.0, 1.0]) + np.random.randn(3) * 0.1 for _ in range(5)]

    all_embeddings = regime_a + regime_b + regime_c
    labels = ["A"] * 5 + ["B"] * 5 + ["C"] * 5

    # Build centroids from labeled data
    centroids = build_regime_centroids_from_embeddings(all_embeddings, labels)
    print(f"  Built centroids for regimes: {list(centroids.keys())}")

    # Test regime assignment
    test_a = np.array([1.1, 0.1, -0.1])
    test_b = np.array([0.1, 1.1, 0.0])
    test_c = np.array([-0.1, 0.1, 1.2])

    regime_a_result, conf_a = compute_regime_cluster(test_a, centroids)
    regime_b_result, conf_b = compute_regime_cluster(test_b, centroids)
    regime_c_result, conf_c = compute_regime_cluster(test_c, centroids)

    print(f"  Test embedding near A: assigned to '{regime_a_result}' (conf={conf_a:.3f})")
    print(f"  Test embedding near B: assigned to '{regime_b_result}' (conf={conf_b:.3f})")
    print(f"  Test embedding near C: assigned to '{regime_c_result}' (conf={conf_c:.3f})")

    assert regime_a_result == "A", f"Expected A, got {regime_a_result}"
    assert regime_b_result == "B", f"Expected B, got {regime_b_result}"
    assert regime_c_result == "C", f"Expected C, got {regime_c_result}"

    # Test K-means clustering
    assignments, kmeans_centroids = cluster_embeddings_kmeans(all_embeddings, n_clusters=3)
    print(f"  K-means clustering: {len(set(assignments))} clusters found")
    print(f"  Cluster assignments: {assignments}")

    print("  TEST 5 PASSED: Regime clustering works")
    print()


def test_embedding_statistics():
    """Test embedding statistics computation."""
    print("=" * 60)
    print("Test 6: Embedding statistics")
    print("=" * 60)

    from src.valuation.embedding_utils import compute_embedding_statistics

    np.random.seed(42)
    embeddings = [np.random.randn(32) for _ in range(20)]

    stats = compute_embedding_statistics(embeddings)
    print(f"  Statistics for {len(embeddings)} embeddings (dim=32):")
    print(f"    Mean norm: {stats['mean_norm']:.4f}")
    print(f"    Std norm: {stats['std_norm']:.4f}")
    print(f"    Mean pairwise distance: {stats['mean_pairwise_distance']:.4f}")
    print(f"    Variance explained by PC1: {stats['variance_explained_pc1']:.4f}")

    assert stats['mean_norm'] > 0
    assert stats['mean_pairwise_distance'] > 0

    print("  TEST 6 PASSED: Embedding statistics computation works")
    print()


def test_datapack_embedding_serialization():
    """Test DataPackMeta with episode_embedding JSON serialization."""
    print("=" * 60)
    print("Test 7: DataPackMeta embedding serialization")
    print("=" * 60)

    from src.valuation.datapack_schema import DataPackMeta

    # Create datapack with embedding
    dp = DataPackMeta(
        task_name="drawer_vase",
        env_type="drawer_vase",
        semantic_tags=["fragile_glassware"],
        episode_embedding=[0.1, -0.2, 0.3, 0.4, -0.5],  # Small embedding for test
    )
    print(f"  Created DataPackMeta with embedding:")
    print(f"    pack_id: {dp.pack_id[:8]}...")
    print(f"    episode_embedding: {dp.episode_embedding}")

    # Serialize to dict
    d = dp.to_dict()
    print(f"  Serialized to dict: 'episode_embedding' in dict = {('episode_embedding' in d)}")
    assert "episode_embedding" in d
    assert d["episode_embedding"] == [0.1, -0.2, 0.3, 0.4, -0.5]

    # Serialize to JSON
    json_str = dp.to_json()
    print(f"  Serialized to JSON (length={len(json_str)} chars)")
    assert "episode_embedding" in json_str

    # Deserialize from dict
    dp_restored = DataPackMeta.from_dict(d)
    print(f"  Deserialized from dict:")
    print(f"    Restored embedding: {dp_restored.episode_embedding}")
    assert dp_restored.episode_embedding == [0.1, -0.2, 0.3, 0.4, -0.5]

    # Test with no embedding (None)
    dp_no_emb = DataPackMeta(task_name="test")
    d_no_emb = dp_no_emb.to_dict()
    assert d_no_emb["episode_embedding"] is None
    dp_no_emb_restored = DataPackMeta.from_dict(d_no_emb)
    assert dp_no_emb_restored.episode_embedding is None
    print("  None embedding serialization: PASSED")

    print("  TEST 7 PASSED: DataPackMeta embedding serialization works")
    print()


def test_end_to_end_pipeline():
    """End-to-end test: create episode, compute embedding, store in datapack."""
    print("=" * 60)
    print("Test 8: End-to-end embedding pipeline")
    print("=" * 60)

    from src.vla.openvla_controller import OpenVLAController, OpenVLAConfig
    from src.valuation.datapack_schema import DataPackMeta
    from src.valuation.embedding_utils import compute_embedding_novelty

    # Create controller with backbone
    cfg = OpenVLAConfig(use_vision_backbone=True, vision_backbone_type="dummy")
    controller = OpenVLAController(cfg)
    controller.load_model()

    # Simulate two episodes
    print("  Episode 1:")
    controller.start_episode()
    for _ in range(5):
        controller.predict_action(Image.new("RGB", (128, 128), color="red"), "grasp object")
    emb1 = controller.end_episode()
    print(f"    Embedding computed: norm={np.linalg.norm(emb1):.4f}")

    print("  Episode 2:")
    controller.start_episode()
    for _ in range(5):
        controller.predict_action(Image.new("RGB", (128, 128), color="blue"), "place object")
    emb2 = controller.end_episode()
    print(f"    Embedding computed: norm={np.linalg.norm(emb2):.4f}")

    # Create datapacks with embeddings
    dp1 = DataPackMeta(
        task_name="grasp_place",
        semantic_tags=["grasp"],
        episode_embedding=emb1.tolist(),
    )
    dp2 = DataPackMeta(
        task_name="grasp_place",
        semantic_tags=["place"],
        episode_embedding=emb2.tolist(),
    )
    print("  Created DataPackMeta objects with embeddings")

    # Compute novelty of episode 2 relative to episode 1
    novelty = compute_embedding_novelty(
        np.array(dp2.episode_embedding),
        [np.array(dp1.episode_embedding)]
    )
    print(f"  Novelty of episode 2 relative to episode 1: {novelty:.4f}")

    # Save and reload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dp1.to_dict(), f)
        temp_path = f.name

    try:
        with open(temp_path, 'r') as f:
            loaded_dict = json.load(f)
        dp_loaded = DataPackMeta.from_dict(loaded_dict)
        print(f"  Saved and reloaded datapack:")
        print(f"    Embedding preserved: {dp_loaded.episode_embedding == dp1.episode_embedding}")
        assert dp_loaded.episode_embedding == dp1.episode_embedding
    finally:
        os.unlink(temp_path)

    print("  TEST 8 PASSED: End-to-end embedding pipeline works")
    print()


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("VISION BACKBONE SMOKE TESTS")
    print("=" * 60 + "\n")

    test_dummy_backbone()
    test_meta_dino_soft_fail()
    test_openvla_controller_with_backbone()
    test_embedding_novelty()
    test_regime_clustering()
    test_embedding_statistics()
    test_datapack_embedding_serialization()
    test_end_to_end_pipeline()

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    print("\nSummary:")
    print("  - DummyBackbone: deterministic embeddings work")
    print("  - MetaDINOBackbone: soft-fail to dummy works")
    print("  - OpenVLAController: backbone integration (additive, logging only)")
    print("  - Embedding novelty: distance-based novelty scores work")
    print("  - Regime clustering: K-means and centroid-based assignment work")
    print("  - Embedding statistics: summary stats computed correctly")
    print("  - DataPackMeta: episode_embedding serialization complete")
    print("  - End-to-end: episode → embedding → datapack → JSON roundtrip")
    print()


if __name__ == "__main__":
    main()
