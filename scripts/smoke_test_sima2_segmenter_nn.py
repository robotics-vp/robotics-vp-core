"""
Smoke test for Neural SIMA-2 Segmenter (Block C).

Tests:
1. Deterministic forward: Fixed seed/input → same mask/logits
2. Serialization round-trip preserves outputs
3. Eval harness computes F1/precision/recall and asserts F1 >= 0.0 (relaxed for smoke test)
"""
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check if PyTorch available
try:
    import torch
    from src.sima2.segmenter_nn import (
        NeuralSegmenter,
        compute_segmentation_loss,
        compute_f1_score,
        save_checkpoint,
        load_checkpoint,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Cannot run Block C tests.")
    sys.exit(0)


def test_deterministic_forward():
    """Test deterministic forward pass."""
    print("\n[Block C] Test 1: Deterministic Forward")

    seed = 42
    batch_size = 2
    image_size = 224
    num_primitives = 10

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create test input
    test_input = torch.randn(batch_size, 3, image_size, image_size)

    # Build model
    model1 = NeuralSegmenter(in_channels=3, num_primitives=num_primitives, seed=seed)
    model1.eval()

    # Forward pass 1
    with torch.no_grad():
        output1 = model1(test_input)

    # Build another model with same seed
    model2 = NeuralSegmenter(in_channels=3, num_primitives=num_primitives, seed=seed)
    model2.eval()

    # Forward pass 2
    with torch.no_grad():
        output2 = model2(test_input)

    # Check outputs are identical
    assert torch.allclose(output1["boundary_logits"], output2["boundary_logits"], atol=1e-6), \
        "Boundary logits not deterministic!"
    assert torch.allclose(output1["primitive_logits"], output2["primitive_logits"], atol=1e-6), \
        "Primitive logits not deterministic!"

    # Check for NaNs
    assert not torch.any(torch.isnan(output1["boundary_logits"])), "Boundary logits contain NaNs!"
    assert not torch.any(torch.isnan(output1["primitive_logits"])), "Primitive logits contain NaNs!"

    print("  ✓ Deterministic forward test passed")


def test_serialization_roundtrip():
    """Test checkpoint serialization/deserialization preserves outputs."""
    print("\n[Block C] Test 2: Serialization Round-trip")

    import tempfile

    seed = 42
    num_primitives = 10
    image_size = 224

    # Build model
    model = NeuralSegmenter(in_channels=3, num_primitives=num_primitives, seed=seed)
    model.eval()

    # Create test input
    test_input = torch.randn(1, 3, image_size, image_size)

    # Forward pass before save
    with torch.no_grad():
        output_before = model(test_input)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        config = {"num_primitives": num_primitives, "seed": seed}
        metrics = {"f1": 0.9}
        save_checkpoint(model, epoch=1, config=config, metrics=metrics, checkpoint_path=checkpoint_path, seed=seed)

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)

        # Verify checkpoint contents
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
        assert "seed" in checkpoint
        assert checkpoint["seed"] == seed

        # Build new model and load state
        model_loaded = NeuralSegmenter(in_channels=3, num_primitives=num_primitives, seed=seed)
        model_loaded.load_state_dict(checkpoint["model_state_dict"])
        model_loaded.eval()

        # Forward pass after load
        with torch.no_grad():
            output_after = model_loaded(test_input)

        # Check outputs are identical
        assert torch.allclose(output_before["boundary_logits"], output_after["boundary_logits"], atol=1e-6), \
            "Boundary logits changed after round-trip!"
        assert torch.allclose(output_before["primitive_logits"], output_after["primitive_logits"], atol=1e-6), \
            "Primitive logits changed after round-trip!"

    print("  ✓ Serialization round-trip test passed")


def test_eval_harness():
    """Test eval harness computes F1/precision/recall."""
    print("\n[Block C] Test 3: Eval Harness")

    seed = 42
    num_primitives = 10
    image_size = 224
    batch_size = 4

    # Build model
    model = NeuralSegmenter(in_channels=3, num_primitives=num_primitives, seed=seed)
    model.eval()

    # Create synthetic data
    np.random.seed(seed)
    images = torch.randn(batch_size, 3, image_size, image_size)
    
    # Create synthetic ground truth (random masks)
    masks = torch.zeros(batch_size, 1, image_size, image_size)
    for i in range(batch_size):
        # Random blobs
        num_blobs = np.random.randint(1, 5)
        for _ in range(num_blobs):
            cx = np.random.randint(0, image_size)
            cy = np.random.randint(0, image_size)
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:image_size, :image_size]
            blob_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            masks[i, 0][blob_mask] = 1.0

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    # Compute F1 score
    precision, recall, f1 = compute_f1_score(
        outputs["boundary_logits"],
        masks,
        threshold=0.5,
    )

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    # For smoke test, just check values are in valid range
    assert 0.0 <= precision <= 1.0, f"Invalid precision: {precision}"
    assert 0.0 <= recall <= 1.0, f"Invalid recall: {recall}"
    assert 0.0 <= f1 <= 1.0, f"Invalid F1: {f1}"

    # Check loss computation
    primitive_targets = torch.randint(0, num_primitives, (batch_size,))
    losses = compute_segmentation_loss(
        boundary_logits=outputs["boundary_logits"],
        boundary_targets=masks,
        primitive_logits=outputs["primitive_logits"],
        primitive_targets=primitive_targets,
    )

    assert "total_loss" in losses
    assert losses["total_loss"].item() >= 0, "Loss should be non-negative!"
    assert not torch.isnan(losses["total_loss"]), "Loss contains NaNs!"

    print(f"  Total Loss: {losses['total_loss'].item():.6f}")
    print("  ✓ Eval harness test passed")


def main():
    print("=" * 60)
    print("BLOCK C SMOKE TESTS: Neural SIMA-2 Segmenter")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\nPyTorch not available. Skipping Block C tests.")
        return

    test_deterministic_forward()
    test_serialization_roundtrip()
    test_eval_harness()

    print("\n" + "=" * 60)
    print("ALL BLOCK C TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
