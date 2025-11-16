"""
Minimal Isaac Gym Dishwashing Environment (Phase 0: Sanity Check)

Goal: Run 10-20 stable episodes to verify:
- Isaac initializes
- Environment runs without hanging
- Logs obs/actions/rewards
- Terminates cleanly

NO tuning, NO refactors - just verify it works.
"""

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from typing import Tuple, Dict, Any


class DishwashingIsaacEnv:
    """
    Minimal Isaac Gym dishwashing environment.

    Single robot gripper + stack of dishes. Simple pick-and-place simulation.
    """

    def __init__(
        self,
        num_envs=1,
        device='cuda:0',
        headless=True,
        max_steps=60
    ):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.max_steps = max_steps

        # Episode state
        self.step_count = 0
        self.completed = 0
        self.attempts = 0
        self.errors = 0

        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()

        # Create sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # Physics engine params
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(
            0, 0,  # compute_device, graphics_device
            gymapi.SIM_PHYSX,
            sim_params
        )

        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # Create environments
        self.envs = []
        self.actor_handles = []
        self._create_envs()

        # Prepare sim
        self.gym.prepare_sim(self.sim)

        # Get state tensors
        self._init_tensors()

        print(f"Isaac Gym initialized: {self.num_envs} envs, device={self.device}")

    def _create_envs(self):
        """Create Isaac Gym environments with simple gripper + dishes."""
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Asset paths (using simple primitives)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.density = 100.0

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)

            # Create simple "gripper" (box actor)
            gripper_pose = gymapi.Transform()
            gripper_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)

            # Create box as gripper proxy
            box_size = gymapi.Vec3(0.1, 0.1, 0.05)
            box_asset = self.gym.create_box(self.sim, box_size.x, box_size.y, box_size.z, asset_options)

            gripper_handle = self.gym.create_actor(
                env, box_asset, gripper_pose, f"gripper_{i}", i, 0
            )

            # Set gripper color (red)
            self.gym.set_rigid_body_color(
                env, gripper_handle, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0)
            )

            self.actor_handles.append(gripper_handle)

            # Create dish stack (simple cylinders)
            for j in range(5):  # 5 dishes
                dish_pose = gymapi.Transform()
                dish_pose.p = gymapi.Vec3(0.3, 0.0, 0.05 + j * 0.05)

                # Cylinder as dish
                dish_radius = 0.08
                dish_height = 0.02
                dish_asset = self.gym.create_sphere(self.sim, dish_radius, asset_options)

                dish_handle = self.gym.create_actor(
                    env, dish_asset, dish_pose, f"dish_{i}_{j}", i, 0
                )

                # Set dish color (white)
                self.gym.set_rigid_body_color(
                    env, dish_handle, 0, gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.9, 0.9, 0.9)
                )

    def _init_tensors(self):
        """Initialize state tensors."""
        # Get root state tensor
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_tensor)

        # Gripper is first actor in each env
        self.gripper_indices = torch.arange(
            0, self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(self) -> Dict[str, Any]:
        """Reset environment."""
        # Reset episode state
        self.step_count = 0
        self.completed = 0
        self.attempts = 0
        self.errors = 0

        # Reset gripper position
        # (In minimal version, just set initial pose - no full reset)

        # Step sim to settle
        for _ in range(10):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # Return observation
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """
        Step environment.

        Args:
            action: [speed, care] in [0, 1]^2

        Returns:
            obs: Observation dict
            info: Task metrics
            done: Episode termination
        """
        speed, care = np.clip(action, 0, 1)

        # Apply action to gripper (simple velocity control)
        # Oscillate gripper horizontally (washing motion)
        time_phase = self.step_count * 0.1
        target_x = 0.3 * np.sin(time_phase * speed * 5)
        target_z = 0.3 + 0.1 * np.sin(time_phase * 2)

        # Add jitter based on care (low care = more jitter)
        jitter = (1.0 - care) * 0.05
        target_x += np.random.randn() * jitter

        # Set gripper velocity (crude control for Phase 0)
        # In full version, this would be proper force/torque control
        self._set_gripper_target(target_x, 0.0, target_z)

        # Simulate
        for _ in range(2):  # Sub-steps
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # Update task metrics (simple placeholder logic)
        if np.random.rand() < 0.1 * speed:
            self.attempts += 1
            error_prob = 0.2 * (1.0 - care)
            if np.random.rand() < error_prob:
                self.errors += 1
            else:
                self.completed += 1

        self.step_count += 1

        # Build info
        time_hours = self.step_count / 60.0  # 1 step = 1 minute
        info = {
            't': time_hours,
            'completed': self.completed,
            'attempts': self.attempts,
            'errors': self.errors,
            'speed': speed,
            'care': care,
            'rate_per_min': self.completed / max(1, time_hours * 60)
        }

        # Episode termination
        done = self.step_count >= self.max_steps

        # Get observation
        obs = self._get_obs()

        return obs, info, done

    def _set_gripper_target(self, x, y, z):
        """Set gripper target position (crude version for Phase 0)."""
        # In minimal version, directly set position
        # Full version would use force/torque control
        for i, env in enumerate(self.envs):
            gripper_handle = self.actor_handles[i]

            # Get current pose
            pose = self.gym.get_actor_rigid_body_states(
                env, gripper_handle, gymapi.STATE_POS
            )

            # Set target (crude - just for sanity check)
            # Real version would compute forces
            pass  # Placeholder - Isaac will use default dynamics

    def _get_obs(self) -> Dict[str, Any]:
        """Get observation (state dict for now)."""
        # For Phase 0, return simple state observation
        # Full version will return RGB images from cameras

        obs = {
            't': self.step_count / 60.0,
            'completed': self.completed,
            'attempts': self.attempts,
            'errors': self.errors
        }

        return obs

    def render(self):
        """Render environment (if not headless)."""
        if not self.headless:
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def close(self):
        """Clean up Isaac Gym."""
        if hasattr(self, 'sim'):
            self.gym.destroy_sim(self.sim)
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        print("Isaac Gym closed")


def test_isaac_env():
    """Phase 0 sanity check: Run 10-20 episodes."""
    print("=" * 60)
    print("PHASE 0: MINIMAL ISAAC SANITY CHECK")
    print("=" * 60)
    print("Goal: 10-20 stable episodes")
    print("- Initialize Isaac")
    print("- Run without hanging")
    print("- Log obs/actions/rewards")
    print("- Terminate cleanly")
    print("=" * 60)

    try:
        # Create environment
        env = DishwashingIsaacEnv(
            num_envs=1,
            device='cpu',  # Use CPU for Phase 0 (more compatible)
            headless=True,
            max_steps=20
        )

        # Run episodes
        num_episodes = 15

        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0

            print(f"\n[Episode {ep+1}/{num_episodes}]")

            while not done:
                # Random action for sanity check
                action = np.random.uniform(0, 1, size=2)

                obs, info, done = env.step(action)

                # Compute simple reward (for logging)
                reward = info['completed'] * 0.1 - info['errors'] * 0.5
                episode_reward += reward

            # Log episode results
            print(f"  Steps: {info['t']*60:.0f}")
            print(f"  Completed: {info['completed']}")
            print(f"  Errors: {info['errors']}")
            print(f"  Reward: {episode_reward:.2f}")

        # Clean up
        env.close()

        print("\n" + "=" * 60)
        print("✅ PHASE 0 COMPLETE: Isaac runs without hanging")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ PHASE 0 FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_isaac_env()
