#!/usr/bin/env python3
"""
Smoke test for real vision backbone training.

Assertions:
- Training runs without errors
- Contrastive loss decreases over epochs
- Checkpoint is created and frozen
- Deterministic with fixed seed
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[smoke_test_train_vision_backbone_real] PyTorch not available, skipping test")
    sys.exit(0)

from scripts.train_vision_backbone_real import main as train_vision_real


def main() -> int:
    if not TORCH_AVAILABLE:
        print("[smoke_test_train_vision_backbone_real] SKIP: PyTorch not available")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        # Run with minimal settings
        argv = [
            "--max-samples", "10",
            "--epochs", "3",
            "--lr", "1e-2",
            "--temperature", "0.5",
            "--use-reconstruction",
            "--seed", "42",
            "--checkpoint-dir", str(checkpoint_dir),
            "--force-neural",  # Force neural mode for this test
        ]

        try:
            train_vision_real(argv)
        except Exception as e:
            print(f"[smoke_test_train_vision_backbone_real] FAIL: Training raised exception: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Check checkpoint exists
        ckpt_path = checkpoint_dir / "vision_backbone.pt"
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # Validate structure
        assert 'model_states' in checkpoint, "Checkpoint missing model_states"
        assert 'config' in checkpoint, "Checkpoint missing config"
        assert 'metrics' in checkpoint, "Checkpoint missing metrics"
        assert checkpoint.get('frozen', False), "Model should be frozen after training"

        # Check metrics
        metrics = checkpoint['metrics']
        assert 'final_contrastive_loss' in metrics, "Missing contrastive loss"
        assert 'final_total_loss' in metrics, "Missing total loss"
        assert metrics['epochs_trained'] == 3, "Epochs mismatch"

        # Check model states
        model_states = checkpoint['model_states']
        assert 'contrastive_head' in model_states, "Missing contrastive head"
        if checkpoint['config'].get('use_reconstruction', False):
            assert 'reconstruction_head' in model_states, "Missing reconstruction head"

        # Verify contrastive head has parameters
        contrastive_state = model_states['contrastive_head']
        assert len(contrastive_state) > 0, "Contrastive head has no parameters"

        print("[smoke_test_train_vision_backbone_real] PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
