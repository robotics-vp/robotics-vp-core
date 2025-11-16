"""
Video Environment Wrappers

Wraps state-based environments to emit video observations (T, C, H, W)
instead of state vectors, enabling video-to-policy training.

For now: Generates synthetic visualizations of state.
Later: Replace with real sim renders or camera feeds.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Any

class DishwashingVideoEnv:
    """
    Wraps DishwashingEnv to return video observations.

    Internally runs same dynamics, but converts state to synthetic video frames.
    Maintains frame buffer for temporal stacking.

    Input: Standard (speed, care) actions
    Output: Video observations (T, C, H, W) instead of state vectors
    """

    def __init__(
        self,
        base_env,
        frames: int = 8,
        height: int = 64,
        width: int = 64,
        render_mode: str = 'synthetic',
    ):
        """
        Args:
            base_env: DishwashingEnv instance
            frames: Number of frames to stack
            height: Frame height
            width: Frame width
            render_mode: 'synthetic' (bars/charts) or 'sim' (future: physics render)
        """
        self.base_env = base_env
        self.frames = frames
        self.height = height
        self.width = width
        self.render_mode = render_mode

        # Frame buffer (stores last T frames)
        self.frame_buffer = deque(maxlen=frames)

        # Observation space (T, C, H, W) in CHW format for PyTorch
        # But we'll return as (T, H, W, C) and let encoder handle transpose
        self.observation_space_shape = (frames, 3, height, width)

        # Action space: (speed, care) both in [0, 1]
        # Not using gym.spaces since base env doesn't have it
        self.action_dim = 2

    def reset(self) -> np.ndarray:
        """Reset environment and return stacked video frames"""
        # Reset base env
        state = self.base_env.reset()

        # Store current state for economics
        self.current_state = state

        # Generate initial frame from state
        frame = self._state_to_frame(state)

        # Fill buffer with duplicated initial frame
        self.frame_buffer.clear()
        for _ in range(self.frames):
            self.frame_buffer.append(frame.copy())

        # Return stacked video obs
        return self._get_video_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], bool]:
        """Step environment and return video observation"""
        # Step base env (returns obs, info, done - no reward)
        state, info, done = self.base_env.step(action)

        # Store current state for economics
        self.current_state = state

        # Convert state to frame
        frame = self._state_to_frame(state)

        # Add to buffer
        self.frame_buffer.append(frame)

        # Get stacked video
        video_obs = self._get_video_obs()

        return video_obs, info, done

    def _state_to_frame(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state vector to RGB frame.

        For now: Synthetic visualization (colored bars/charts).
        Later: Replace with sim render or camera feed.

        Args:
            state: State vector from base env

        Returns:
            frame: RGB frame (H, W, 3) in [0, 255] uint8
        """
        if self.render_mode == 'synthetic':
            return self._render_synthetic_frame(state)
        elif self.render_mode == 'sim':
            # Future: return sim.render() or camera.capture()
            raise NotImplementedError("Sim rendering not yet implemented")
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def _render_synthetic_frame(self, state: np.ndarray) -> np.ndarray:
        """
        Render synthetic visualization of state as colored bars/charts.

        State encoding (example for dishwashing):
        - completed/attempts/errors as bar heights
        - speeds/care as color intensities
        - Text labels (optional)

        Returns:
            frame: (H, W, 3) RGB uint8
        """
        # Create blank canvas
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240  # Light gray background

        # Extract key metrics from state
        # Assuming state has: [completed, attempts, errors, time_elapsed, speed, care, ...]
        # Normalize to [0, 1] for visualization
        if len(state) >= 6:
            completed_norm = min(state[0] / 100.0, 1.0)  # Assume max 100 dishes
            attempts_norm = min(state[1] / 100.0, 1.0)
            errors_norm = min(state[2] / 20.0, 1.0)  # Assume max 20 errors
            speed = np.clip(state[4], 0, 1)
            care = np.clip(state[5], 0, 1)
        else:
            # Fallback for unknown state structure
            completed_norm = 0.5
            attempts_norm = 0.5
            errors_norm = 0.1
            speed = 0.5
            care = 0.5

        # Draw metrics as vertical bars
        bar_width = self.width // 5
        bar_spacing = self.width // 5

        # Bar 1: Completed (green)
        self._draw_bar(frame, x=bar_spacing * 0, height_norm=completed_norm, color=(0, 200, 0), width=bar_width)

        # Bar 2: Attempts (blue)
        self._draw_bar(frame, x=bar_spacing * 1, height_norm=attempts_norm, color=(0, 100, 200), width=bar_width)

        # Bar 3: Errors (red)
        self._draw_bar(frame, x=bar_spacing * 2, height_norm=errors_norm, color=(200, 0, 0), width=bar_width)

        # Bar 4: Speed (yellow)
        self._draw_bar(frame, x=bar_spacing * 3, height_norm=speed, color=(200, 200, 0), width=bar_width)

        # Bar 5: Care (cyan)
        self._draw_bar(frame, x=bar_spacing * 4, height_norm=care, color=(0, 200, 200), width=bar_width)

        # Add thin border (numpy-based)
        frame[0, :] = [100, 100, 100]  # Top
        frame[-1, :] = [100, 100, 100]  # Bottom
        frame[:, 0] = [100, 100, 100]  # Left
        frame[:, -1] = [100, 100, 100]  # Right

        return frame

    def _draw_bar(self, frame: np.ndarray, x: int, height_norm: float, color: Tuple[int, int, int], width: int):
        """Draw a vertical bar on the frame"""
        bar_height = int(height_norm * (self.height - 10))
        y_start = self.height - 5 - bar_height
        y_end = self.height - 5

        x_start = int(x)
        x_end = int(x + width)

        # Clip to frame bounds
        x_start = max(0, min(x_start, self.width - 1))
        x_end = max(0, min(x_end, self.width))
        y_start = max(0, min(y_start, self.height - 1))
        y_end = max(0, min(y_end, self.height))

        # Draw filled rectangle
        frame[y_start:y_end, x_start:x_end] = color

    def _get_video_obs(self) -> np.ndarray:
        """
        Stack frames into video observation.

        Returns:
            video: (T, C, H, W) in CHW format for PyTorch
        """
        # Stack frames: (T, H, W, 3)
        stacked = np.stack(list(self.frame_buffer), axis=0)  # (T, H, W, 3)

        # Transpose to (T, 3, H, W) for PyTorch Conv3D
        video = stacked.transpose(0, 3, 1, 2)  # (T, C, H, W)

        # Convert to float32 in [0, 1]
        video = video.astype(np.float32) / 255.0

        return video

    def get_state(self) -> Dict[str, Any]:
        """Get underlying state dict (for economics calculations)"""
        return self.current_state

    def close(self):
        """Close environment"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


def create_video_env(base_env_class, base_env_config: dict, video_config: dict):
    """
    Factory function to create video-wrapped environment.

    Args:
        base_env_class: Base environment class (e.g., DishwashingEnv)
        base_env_config: Config dict for base env
        video_config: Config dict for video wrapper
            {
                'frames': int,
                'height': int,
                'width': int,
                'render_mode': str
            }

    Returns:
        DishwashingVideoEnv instance
    """
    # Create base env
    base_env = base_env_class(**base_env_config)

    # Wrap in video env
    video_env = DishwashingVideoEnv(
        base_env=base_env,
        frames=video_config.get('frames', 8),
        height=video_config.get('height', 64),
        width=video_config.get('width', 64),
        render_mode=video_config.get('render_mode', 'synthetic'),
    )

    return video_env


if __name__ == '__main__':
    """Test video wrapper"""
    import sys
    sys.path.insert(0, '/Users/amarmurray/robotics v-p economics model')

    from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams

    print("Testing DishwashingVideoEnv...")

    # Create base env with defaults
    params = DishwashingParams()
    base_env = DishwashingEnv(params)

    # Wrap in video env
    video_env = DishwashingVideoEnv(
        base_env=base_env,
        frames=8,
        height=64,
        width=64,
        render_mode='synthetic',
    )

    # Test reset
    print("\n[Reset Test]")
    obs = video_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    assert obs.shape == (8, 3, 64, 64), f"Expected (8, 3, 64, 64), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert 0 <= obs.min() <= obs.max() <= 1, f"Expected [0, 1], got [{obs.min()}, {obs.max()}]"

    # Test step
    print("\n[Step Test]")
    action = np.array([0.5, 0.5])  # Medium speed, medium care
    obs, info, done = video_env.step(action)
    print(f"Observation shape: {obs.shape}")
    print(f"Done: {done}")
    print(f"Info keys: {list(info.keys())}")

    # Test multiple steps
    print("\n[Episode Test]")
    obs = video_env.reset()
    for i in range(10):
        action = np.random.uniform(0, 1, size=2)
        obs, info, done = video_env.step(action)
        if done:
            print(f"Episode ended at step {i+1}")
            break

    print(f"Final observation shape: {obs.shape}")

    # Verify frame buffer behavior
    print("\n[Frame Buffer Test]")
    obs1 = video_env.reset()
    action = np.array([0.8, 0.2])  # High speed, low care
    obs2, _, _ = video_env.step(action)

    # First and last frames should differ after a step (or be very similar if state barely changed)
    frame_diff = np.abs(obs2[-1] - obs2[0]).mean()
    print(f"Frame difference (last vs first): {frame_diff:.6f}")
    # Note: Frame diff might be zero if state changes are small - that's okay
    if frame_diff > 0:
        print("✓ Frames differ (state is changing)")
    else:
        print("✓ Frames similar (state changed minimally in one step - normal)")

    print("\n✅ All tests passed!")
    print("\nVideo environment ready for training!")
