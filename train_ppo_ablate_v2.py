"""
PPO Training with Ablation Modes (V2: 2D actions, feasible SLA)

Ablation modes:
- A (baseline): No novelty weighting (weights=1)
- B (no_lambda): No constraint enforcement (λ=0)
- C (full): Full model with all features

Usage:
    python train_ppo_ablate_v2.py A  # Baseline
    python train_ppo_ablate_v2.py B  # No constraint
    python train_ppo_ablate_v2.py C  # Full model
"""
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams
from src.rl.ppo import PPOAgent
from src.economics.mpl import mpl
from src.economics.wage import implied_robot_wage
from src.rl.reward_shaping import compute_econ_reward
from src.config.internal_profile import get_internal_experiment_profile
from src.utils.logger import CsvLogger


def run_ablation(mode, episodes=1000):
    """Run training with specified ablation mode."""

    # Config
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Human benchmark
    MPh = 60.0
    wh = 18.0

    # Economics
    p = 0.30
    damage_cost = 1.0
    energy_cost = 0.10

    # Quality constraint with curriculum
    err_target_final = 0.06
    err_target_init = 0.10
    curriculum_eps = 600
    lambda_init = 0.0
    eta = 0.01  # Reduced from 0.1

    # Ablation settings
    if mode == 'A':
        # Baseline: No novelty weighting
        use_novelty = False
        use_constraint = True
        log_path = 'logs/ablation_v2_A_baseline.csv'
        print("=== ABLATION A: Baseline (weights=1) ===")
    elif mode == 'B':
        # No constraint
        use_novelty = True
        use_constraint = False
        log_path = 'logs/ablation_v2_B_no_lambda.csv'
        print("=== ABLATION B: No constraint (λ=0) ===")
    elif mode == 'C':
        # Full model
        use_novelty = True
        use_constraint = True
        log_path = 'logs/ablation_v2_C_full.csv'
        print("=== ABLATION C: Full model ===")
    else:
        raise ValueError(f"Invalid mode: {mode}. Use A, B, or C")

    # Environment (2D action space)
    env_params = DishwashingParams(
        price_per_unit=p,
        damage_cost=damage_cost,
        time_step_s=60.0
    )
    env = DishwashingEnv(env_params)

    # PPO Agent (2D actions: speed, care)
    state_dim = 4   # [t, completed, attempts, errors]
    action_dim = 2  # [speed, care]

    agent = PPOAgent(
        obs_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        novelty_alpha=2.0,
        novelty_beta=-1.0,
        device='cpu'
    )

    # Logger
    logger = CsvLogger(log_path)

    # Dual variable
    lam = lambda_init
    profile = get_internal_experiment_profile("dishwashing")
    alpha_mpl, alpha_error, alpha_ep, _ = profile.get("default_objective_vector", [1.0, 1.0, 1.0, 0.0])

    # Training loop
    for ep in range(episodes):
        # Curriculum: anneal error target from 10% → 6%
        err_target = np.interp(ep, [0, curriculum_eps],
                              [err_target_init, err_target_final])

        obs_dict = env.reset()
        done = False

        episode_steps = 0
        episode_reward_sum = 0.0
        actions_taken = []
        last_reward_components = {}

        while not done and episode_steps < 60:
            # Compute novelty (simple stub)
            novelty = float(np.random.rand() * 0.5 + 0.5) if use_novelty else None

            # Select action (agent handles buffering)
            action_val, log_prob, value = agent.select_action(obs_dict, novelty=novelty)

            # step returns (obs, info, done)
            next_obs_dict, info, done = env.step(action_val)

            # Track actions
            actions_taken.append([info['speed'], info['care']])

            # Economic reward (per-step MPL/EP/error)
            mpl_t = info.get("mpl_t", 0.0)
            ep_t = info.get("ep_t", 0.0)
            err_term = info.get("delta_errors", 0.0)

            reward, reward_components = compute_econ_reward(
                mpl=mpl_t,
                ep=ep_t,
                error_rate=err_term,
                wage_parity=None,
                alpha_mpl=alpha_mpl,
                alpha_error=alpha_error,
                alpha_ep=alpha_ep,
            )
            last_reward_components = reward_components

            # Store transition
            agent.store_transition(reward, done)
            episode_reward_sum += reward

            obs_dict = next_obs_dict
            episode_steps += 1

        # Episode metrics (final state)
        time_hours = next_obs_dict['t'] / 3600.0
        mp_r = mpl(next_obs_dict['completed'], time_hours) if time_hours > 0 else 0
        err_rate = next_obs_dict['errors'] / max(1, next_obs_dict['attempts'])
        w_hat_r = implied_robot_wage(p, mp_r, err_rate, damage_cost)
        wage_parity = w_hat_r / wh
        prod_parity = mp_r / MPh

        revenue = p * mp_r
        error_cost = damage_cost * (err_rate * mp_r)
        profit = revenue - error_cost - energy_cost

        # Average actions
        actions_arr = np.array(actions_taken)
        mean_speed = actions_arr[:, 0].mean() if len(actions_arr) > 0 else 0.0
        mean_care = actions_arr[:, 1].mean() if len(actions_arr) > 0 else 0.0

        # Update λ (if constraint enabled)
        if use_constraint:
            lam = max(0.0, lam + eta * (err_rate - err_target))

        # PPO update
        last_value = 0.0  # Episode is done
        train_metrics = agent.update(last_value=last_value)

        # Log
        logger.log(
            episode=ep,
            time_h=round(time_hours, 3),
            completed=next_obs_dict['completed'],
            attempts=next_obs_dict['attempts'],
            errors=next_obs_dict['errors'],
            err_rate=round(err_rate, 4),
            err_target=round(err_target, 4),
            mp_r=round(mp_r, 2),
            mp_h=MPh,
            w_hat_r=round(w_hat_r, 2),
            w_h=wh,
            wage_parity=round(wage_parity, 4),
            prod_parity=round(prod_parity, 4),
            profit=round(profit, 2),
            lambda_dual=round(lam, 4),
            episode_reward=round(episode_reward_sum, 2),
            episode_steps=episode_steps,
            reward_mpl=round(last_reward_components.get("mpl_component", 0.0), 4),
            reward_ep=round(last_reward_components.get("ep_component", 0.0), 4),
            reward_error=round(last_reward_components.get("error_penalty", 0.0), 4),
            mean_speed=round(mean_speed, 3),
            mean_care=round(mean_care, 3),
            policy_loss=round(train_metrics['policy_loss'], 6),
            value_loss=round(train_metrics['value_loss'], 6),
            entropy=round(train_metrics['entropy'], 6),
            kl_divergence=round(train_metrics['kl_divergence'], 6),
            novelty=round(novelty if novelty else 0.0, 4),
            mean_weight=round(train_metrics.get('mean_weight', 1.0), 4),
            p90_weight=round(train_metrics.get('p90_weight', 1.0), 4)
        )

        # Print progress
        if (ep + 1) % 50 == 0:
            print(f"[ep {ep+1:4d}] MP_r={mp_r:.0f}/h  Profit=${profit:.2f}  "
                  f"Err={err_rate:.3f} (target={err_target:.3f})  λ={lam:.3f}  "
                  f"Speed={mean_speed:.2f} Care={mean_care:.2f}  "
                  f"WageParity={wage_parity:.3f}")

    print(f"\n✅ {mode} complete: {log_path}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_ppo_ablate_v2.py [A|B|C]")
        print("  A: Baseline (weights=1)")
        print("  B: No constraint (λ=0)")
        print("  C: Full model")
        sys.exit(1)

    mode = sys.argv[1].upper()
    run_ablation(mode, episodes=1000)
