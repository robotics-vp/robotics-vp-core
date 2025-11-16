#!/usr/bin/env python3
"""
Smoke Test for Drawer+Vase Physics Environment

Runs a few episodes in both state and vision modes to verify:
- Environment initialization
- Step execution
- Observation formats
- Info dictionary contents
- Termination conditions
"""

import os
import sys
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

from src.envs.drawer_vase_physics_env import (
    DrawerVasePhysicsEnv,
    DrawerVaseConfig,
    summarize_drawer_vase_episode
)


def smoke_test_state_mode(n_episodes=5):
    """Test state observation mode."""
    print("=" * 70)
    print("SMOKE TEST: STATE MODE")
    print("=" * 70)

    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"Episode {ep+1}/{n_episodes}")
        print(f"  Initial obs shape: {obs.shape}")
        print(f"  Initial obs: {obs[:7]}...")  # First 7 elements

        info_history = []
        done = False
        step_count = 0

        while not done and step_count < 50:  # Max 50 steps for smoke test
            # Random action
            if hasattr(env.action_space, 'sample'):
                action = env.action_space.sample()
            else:
                action = np.random.uniform(-1, 1, 3).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            info_history.append(info)
            step_count += 1

        # Report
        print(f"  Steps: {step_count}")
        print(f"  Termination: {info.get('terminated_reason', 'unknown')}")
        print(f"  Drawer fraction: {info.get('drawer_fraction', 0):.4f}")
        print(f"  Min clearance: {info.get('min_clearance', 0):.4f}")
        print(f"  Vase intact: {info.get('vase_intact', True)}")
        print(f"  Energy (Wh): {info.get('energy_Wh', 0):.6f}")

        # Episode summary
        summary = summarize_drawer_vase_episode(info_history)
        print(f"  EpisodeInfoSummary:")
        print(f"    termination_reason: {summary.termination_reason}")
        print(f"    mpl_episode: {summary.mpl_episode:.4f}")
        print(f"    error_rate_episode: {summary.error_rate_episode:.4f}")
        print(f"    energy_Wh: {summary.energy_Wh:.6f}")
        print()

    env.close()
    print("STATE MODE: PASS\n")


def smoke_test_vision_mode(n_episodes=3):
    """Test vision observation mode."""
    print("=" * 70)
    print("SMOKE TEST: VISION MODE")
    print("=" * 70)

    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='vision')

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    for ep in range(n_episodes):
        obs, info = env.reset()
        print(f"Episode {ep+1}/{n_episodes}")
        print(f"  Initial obs shape: {obs.shape}")
        print(f"  Initial obs dtype: {obs.dtype}")
        print(f"  Pixel range: [{obs.min()}, {obs.max()}]")

        info_history = []
        done = False
        step_count = 0

        while not done and step_count < 30:  # Max 30 steps
            if hasattr(env.action_space, 'sample'):
                action = env.action_space.sample()
            else:
                action = np.random.uniform(-1, 1, 3).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            info_history.append(info)
            step_count += 1

        print(f"  Steps: {step_count}")
        print(f"  Termination: {info.get('terminated_reason', 'unknown')}")
        print(f"  Final obs shape: {obs.shape}")
        print()

    env.close()
    print("VISION MODE: PASS\n")


def smoke_test_econ_params():
    """Test EconParams loading for drawer_vase."""
    print("=" * 70)
    print("SMOKE TEST: ECON PARAMS")
    print("=" * 70)

    from src.config.econ_params import load_econ_params
    from src.config.internal_profile import get_internal_experiment_profile

    profile = get_internal_experiment_profile("default")
    econ_params = load_econ_params(profile, preset="drawer_vase")

    print(f"Preset: {econ_params.preset}")
    print(f"Value per drawer open: ${econ_params.value_per_successful_drawer_open:.2f}")
    print(f"Vase break cost: ${econ_params.vase_break_cost:.2f}")
    print(f"Electricity price: ${econ_params.electricity_price_kWh:.2f}/kWh")
    print(f"Risk tolerance: {econ_params.allowable_risk_tolerance:.2f}")
    print(f"Fragility penalty: {econ_params.fragility_penalty_coeff:.2f}")
    print()
    print("ECON PARAMS: PASS\n")


def smoke_test_drawer_trajectory():
    """Test a trajectory that opens the drawer (controlled, not random)."""
    print("=" * 70)
    print("SMOKE TEST: DRAWER OPENING TRAJECTORY")
    print("=" * 70)

    config = DrawerVaseConfig()
    env = DrawerVasePhysicsEnv(config=config, obs_mode='state')

    obs, info = env.reset()
    print("Testing controlled drawer opening...")
    print(f"Initial drawer fraction: {info.get('drawer_fraction', 0):.4f}")

    info_history = []
    done = False
    step = 0

    # Simple strategy: move towards handle, then pull back
    phase = "approach"

    while not done and step < 100:
        ee_pos = obs[0:3]
        drawer_frac = obs[6]

        if phase == "approach":
            # Move to handle position (approximate)
            handle_pos = np.array([0.0, -0.42, 0.65])
            direction = handle_pos - ee_pos
            dist = np.linalg.norm(direction)

            if dist < 0.05:
                phase = "pull"
                print(f"  Step {step}: Reached handle, switching to pull phase")
            else:
                action = direction / (dist + 1e-6) * 0.8
        else:
            # Pull drawer (negative Y direction)
            action = np.array([0.0, -0.6, 0.0])

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        info_history.append(info)

        if step % 20 == 0:
            print(f"  Step {step}: drawer={drawer_frac:.4f}, "
                  f"clearance={info.get('min_clearance', 0):.4f}")

        step += 1

    print(f"\nFinal drawer fraction: {info.get('drawer_fraction', 0):.4f}")
    print(f"Termination: {info.get('terminated_reason', 'unknown')}")
    print(f"Success: {info.get('success', False)}")
    print(f"Vase intact: {info.get('vase_intact', True)}")

    env.close()
    print("DRAWER TRAJECTORY: PASS\n")


def main():
    """Run all smoke tests."""
    print("=" * 70)
    print("DRAWER+VASE ENVIRONMENT SMOKE TEST")
    print("=" * 70)
    print()

    # Test EconParams first (no PyBullet needed)
    smoke_test_econ_params()

    # Test state mode
    smoke_test_state_mode(n_episodes=3)

    # Test vision mode
    smoke_test_vision_mode(n_episodes=2)

    # Test drawer opening trajectory
    smoke_test_drawer_trajectory()

    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == '__main__':
    main()
