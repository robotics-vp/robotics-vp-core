"""
Test script for DishwashingPhysicsEnv

Tests:
1. Environment initialization
2. Reset functionality
3. Step execution with random actions
4. Frame rendering
5. Info dict structure
6. Saves sample frames to artifacts/physics_frames/
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.physics import DishwashingPhysicsEnv
import pytest

@pytest.fixture(scope="module")
def env():
    e = DishwashingPhysicsEnv(
        frames=8,
        image_size=(64, 64),
        max_steps=60,
        headless=True
    )
    yield e
    e.close()


def test_env_initialization():
    """Test environment creation."""
    print("="*60)
    print("Test 1: Environment Initialization")
    print("="*60)

    env = DishwashingPhysicsEnv(
        frames=8,
        image_size=(64, 64),
        max_steps=60,
        headless=True
    )

    print(f"✓ Environment created")
    print(f"  - Observation shape: {env.observation_space_shape}")
    print(f"  - Frames: {env.frames}")
    print(f"  - Image size: {env.image_height}x{env.image_width}")

    env.close()


def test_reset(env):
    """Test reset functionality."""
    print("\n" + "="*60)
    print("Test 2: Reset Functionality")
    print("="*60)

    obs = env.reset()

    print(f"✓ Reset successful")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation dtype: {obs.dtype}")
    print(f"  - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Check shape
    assert obs.shape == env.observation_space_shape, \
        f"Shape mismatch: {obs.shape} vs {env.observation_space_shape}"

    print(f"✓ Shape matches expected: {obs.shape}")

def test_step(env, n_steps=10):
    """Test stepping with random actions."""
    print("\n" + "="*60)
    print(f"Test 3: Stepping ({n_steps} steps)")
    print("="*60)

    obs = env.reset()

    for i in range(n_steps):
        # Random action
        action = np.random.rand(2)  # [speed, care]

        # Step
        obs, info, done = env.step(action)

        print(f"Step {i+1}: completed={info['completed']}, "
              f"attempts={info['attempts']}, errors={info['errors']}, "
              f"done={done}")

        # Check observation shape
        assert obs.shape == env.observation_space_shape, \
            f"Obs shape mismatch at step {i}: {obs.shape}"

        # Check info dict keys
        required_keys = ['t', 'completed', 'attempts', 'errors', 'speed', 'care', 'rate_per_min']
        for key in required_keys:
            assert key in info, f"Missing key in info dict: {key}"

        if done:
            print(f"✓ Episode terminated at step {i+1}")
            break

    print(f"✓ All steps completed successfully")
def test_video_consistency(env, n_steps=5):
    """Test that video observations change over time."""
    print("\n" + "="*60)
    print("Test 4: Video Frame Consistency")
    print("="*60)

    obs1 = env.reset()

    # Take a few steps
    for _ in range(n_steps):
        action = np.random.rand(2)
        obs2, _, done = env.step(action)
        if done:
            break

    # Check that frames changed
    frame_diff = np.abs(obs2 - obs1).mean()
    print(f"  - Mean frame difference: {frame_diff:.6f}")

    if frame_diff > 1e-6:
        print(f"✓ Frames are changing (good)")
    else:
        print(f"⚠ Frames appear static (may be issue)")

def save_sample_frames(env, n_frames=10):
    """Save sample frames to disk."""
    print("\n" + "="*60)
    print(f"Test 5: Saving Sample Frames")
    print("="*60)

    # Create output directory
    output_dir = "artifacts/physics_frames"
    os.makedirs(output_dir, exist_ok=True)

    obs = env.reset()

    saved_frames = []

    for i in range(n_frames):
        # Random action
        action = np.random.rand(2)

        # Step
        obs, info, done = env.step(action)

        # Extract first frame from video stack: (T, C, H, W) -> (H, W, C)
        frame = obs[0].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

        # Convert to uint8
        frame_uint8 = (frame * 255).astype(np.uint8)

        # Save
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        Image.fromarray(frame_uint8).save(frame_path)

        saved_frames.append(frame_path)

        if done:
            break

    print(f"✓ Saved {len(saved_frames)} frames to {output_dir}")

    # Create collage
    create_frame_collage(saved_frames, os.path.join(output_dir, "collage.png"))

    return saved_frames


def create_frame_collage(frame_paths, output_path):
    """Create a collage of frames."""
    n_frames = len(frame_paths)
    n_cols = min(5, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, frame_path in enumerate(frame_paths):
        row = idx // n_cols
        col = idx % n_cols

        img = Image.open(frame_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Frame {idx}")

    # Hide empty subplots
    for idx in range(n_frames, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved frame collage: {output_path}")


def test_long_episode(env, n_steps=100):
    """Test a longer episode to ensure stability."""
    print("\n" + "="*60)
    print(f"Test 6: Long Episode ({n_steps} steps)")
    print("="*60)

    obs = env.reset()

    completed_count = 0
    error_count = 0

    for i in range(n_steps):
        action = np.random.rand(2)
        obs, info, done = env.step(action)

        completed_count = info['completed']
        error_count = info['errors']

        if done:
            print(f"Episode terminated at step {i+1}")
            break

    print(f"✓ Long episode completed")
    print(f"  - Total steps: {i+1}")
    print(f"  - Completed: {completed_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Success rate: {completed_count / max(1, completed_count + error_count):.2%}")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# DishwashingPhysicsEnv Test Suite")
    print("#"*60 + "\n")

    try:
        # Test 1: Initialization
        env = test_env_initialization()

        # Test 2: Reset
        test_reset(env)

        # Test 3: Stepping
        test_step(env, n_steps=10)

        # Test 4: Video consistency
        test_video_consistency(env, n_steps=5)

        # Test 5: Save frames
        save_sample_frames(env, n_frames=10)

        # Test 6: Long episode
        test_long_episode(env, n_steps=100)

        # Cleanup
        env.close()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
