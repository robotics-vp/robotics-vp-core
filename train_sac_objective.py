"""
Experimental SAC training entrypoint with optional objective-conditioned rewards.

Does NOT alter the baseline train_sac.py. When --use-objective-reward is False,
behavior matches legacy reward usage. When True, RewardBuilder computes a
scalar reward while legacy reward is logged in parallel.
"""
import argparse
import numpy as np
import torch

from src.config.econ_params import load_econ_params
from src.config.internal_profile import get_internal_experiment_profile
from src.config.objective_profile import get_objective_presets
from src.envs.dishwashing_env import DishwashingEnv, summarize_episode_info
from src.rl.sac import SACAgent
from src.rl.econ_reward import compute_econ_reward
from src.valuation.reward_builder import build_reward_terms, combine_reward, default_objective_vector


def train_sac_objective(episodes=100, econ_preset="toy", use_objective=False, objective_preset="throughput"):
    profile = get_internal_experiment_profile(econ_preset)
    econ_params = load_econ_params(profile, preset=econ_preset)
    env = DishwashingEnv(econ_params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACAgent(
        encoder=None,
        latent_dim=0,
        action_dim=2,
        lr=3e-4,
        gamma=0.995,
        tau=5e-3,
        buffer_capacity=int(1e5),
        batch_size=256,
        target_entropy=-2.0,
        device=device,
    )

    obj_presets = get_objective_presets()
    obj_vec = obj_presets.get(objective_preset, default_objective_vector())

    for ep in range(episodes):
        obs = env.reset()
        done = False
        info_history = []
        legacy_reward_sum = 0.0
        objective_reward_sum = 0.0
        steps = 0
        while not done:
            action, _ = agent.select_action(obs, novelty=0.0)
            next_obs, reward, done = env.step(action)
            info_history.append(next_obs[1] if isinstance(next_obs, tuple) else {})
            # Compute legacy reward via econ_reward to mimic base
            mpl_t = info_history[-1].get("mpl_t", 0.0)
            ep_t = info_history[-1].get("ep_t", 0.0)
            err_term = info_history[-1].get("delta_errors", 0.0)
            legacy_reward, _ = compute_econ_reward(
                mpl=mpl_t,
                ep=ep_t,
                error_rate=err_term,
                wage_parity=None,
                mode="mpl_ep_error",
                alpha_mpl=1.0,
                alpha_error=1.0,
                alpha_ep=1.0,
                alpha_wage=0.0,
            )
            reward_to_use = legacy_reward
            if use_objective:
                # Build episode summary on the fly for objective reward
                summary = summarize_episode_info(info_history)
                terms = build_reward_terms(summary, econ_params)
                reward_to_use = combine_reward(obj_vec, terms)
            agent.store_transition(obs, action, reward_to_use, next_obs, done, novelty=0.0)
            agent.update()
            legacy_reward_sum += legacy_reward
            objective_reward_sum += reward_to_use
            obs = next_obs
            steps += 1
            if steps > 500:
                break

        print(f"[Episode {ep+1}] steps={steps} legacy_reward={legacy_reward_sum:.3f} objective_reward={objective_reward_sum:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--econ-preset", type=str, default="toy")
    parser.add_argument("--use-objective-reward", action="store_true", default=False)
    parser.add_argument("--objective-preset", type=str, default="throughput",
                        choices=["throughput", "energy_saver", "balanced", "safety_first", "custom"])
    args = parser.parse_args()
    train_sac_objective(
        episodes=args.episodes,
        econ_preset=args.econ_preset,
        use_objective=args.use_objective_reward,
        objective_preset=args.objective_preset,
    )
