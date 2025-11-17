"""
Experimental SAC training entrypoint with optional objective-conditioned rewards.

Does NOT alter the baseline train_sac.py. When --use-objective-reward is False,
behavior matches legacy reward usage. When True, RewardBuilder computes a
scalar reward while legacy reward is logged in parallel.
"""
import argparse
import json
import os
import time
import numpy as np
import torch

from src.config.econ_params import load_econ_params
from src.config.internal_profile import get_internal_experiment_profile
from src.config.objective_profile import ObjectiveVector
from src.envs.dishwashing_env import DishwashingEnv, summarize_episode_info
from src.rl.sac import SACAgent
from src.rl.reward_shaping import compute_econ_reward
from src.valuation.reward_builder import build_reward_terms, combine_reward, default_objective_vector
from src.utils.experimental_flags import assert_experimental_flag_acknowledged


def train_sac_objective(
    episodes=100,
    econ_preset="toy",
    use_objective=False,
    objective_preset="throughput",
    save_dir="results/sac_objective_training",
    log_every=10,
):
    assert_experimental_flag_acknowledged("use_objective_reward", use_objective)
    profile = get_internal_experiment_profile(econ_preset)
    econ_params = load_econ_params(profile, preset=econ_preset)
    env = DishwashingEnv(econ_params)

    # NOTE: SAC agent creation is deferred due to encoder requirements.
    # For experimental objective-conditioned training, we run episodes with random policy
    # to demonstrate the reward builder integration. Full SAC training requires encoder setup.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = None  # Placeholder - use random actions for now

    # Get objective vector from preset
    try:
        obj_vec_obj = ObjectiveVector.from_preset(objective_preset)
        obj_vec = obj_vec_obj.to_list()
    except ValueError:
        obj_vec = default_objective_vector()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    run_id = f"{int(time.time())}_{objective_preset}_{'objective' if use_objective else 'legacy'}"
    run_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save run config
    run_config = {
        "episodes": episodes,
        "econ_preset": econ_preset,
        "use_objective_reward": use_objective,
        "objective_preset": objective_preset,
        "objective_vector": obj_vec,
        "start_time": time.time(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Training history
    history = []

    print(f"Starting training: {run_id}")
    print(f"  Objective-based reward: {use_objective}")
    print(f"  Objective preset: {objective_preset}")
    print(f"  Objective vector: {obj_vec}")
    print(f"  Save dir: {run_dir}")

    for ep in range(episodes):
        obs = env.reset()
        done = False
        info_history = []
        legacy_reward_sum = 0.0
        objective_reward_sum = 0.0
        steps = 0

        # Track decomposed reward terms per episode
        episode_reward_terms = {
            "r_mpl_total": 0.0,
            "r_error_total": 0.0,
            "r_energy_total": 0.0,
            "r_safety_total": 0.0,
            "r_novelty_total": 0.0,
        }

        while not done:
            # Use random actions for experimental path (no encoder setup)
            action = np.random.randn(2) * 0.5
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

                # Track decomposed reward heads
                episode_reward_terms["r_mpl_total"] += terms["r_mpl"]
                episode_reward_terms["r_error_total"] += terms["r_error"]
                episode_reward_terms["r_energy_total"] += terms["r_energy"]
                episode_reward_terms["r_safety_total"] += terms["r_safety"]
                episode_reward_terms["r_novelty_total"] += terms["r_novelty"]

            # NOTE: Agent training disabled for experimental path
            # In full implementation, agent.store_transition + agent.update() here
            legacy_reward_sum += legacy_reward
            objective_reward_sum += reward_to_use
            obs = next_obs
            steps += 1
            if steps > 500:
                break

        # Get episode summary metrics
        episode_summary = summarize_episode_info(info_history)
        mpl_episode = getattr(episode_summary, "mpl_episode", 0.0)
        error_rate = getattr(episode_summary, "error_rate_episode", 0.0)
        energy_Wh = getattr(episode_summary, "energy_Wh", 0.0)

        # Log episode
        episode_log = {
            "episode": ep + 1,
            "steps": steps,
            "legacy_reward": legacy_reward_sum,
            "objective_reward": objective_reward_sum,
            "mpl_episode": mpl_episode,
            "error_rate": error_rate,
            "energy_Wh": energy_Wh,
            "objective_vector": obj_vec,
        }

        # Add decomposed reward heads if using objective reward
        if use_objective:
            episode_log.update(episode_reward_terms)

        history.append(episode_log)

        if (ep + 1) % log_every == 0 or ep == 0:
            print(
                f"[Episode {ep+1}/{episodes}] steps={steps} "
                f"legacy={legacy_reward_sum:.3f} objective={objective_reward_sum:.3f} "
                f"MPL={mpl_episode:.2f} err={error_rate:.4f} energy={energy_Wh:.2f}"
            )

    # Save training history
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")

    # Save summary stats
    summary_stats = {
        "total_episodes": episodes,
        "final_mpl": history[-1]["mpl_episode"],
        "final_error_rate": history[-1]["error_rate"],
        "final_energy_Wh": history[-1]["energy_Wh"],
        "mean_legacy_reward": np.mean([h["legacy_reward"] for h in history]),
        "mean_objective_reward": np.mean([h["objective_reward"] for h in history]),
        "end_time": time.time(),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Training complete. Results saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experimental SAC training with optional objective-conditioned rewards"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--econ-preset", type=str, default="toy", help="Econ preset to use")
    parser.add_argument("--use-objective-reward", action="store_true", default=False,
                        help="Use objective-conditioned reward instead of legacy")
    parser.add_argument("--objective-preset", type=str, default="throughput",
                        choices=["throughput", "energy_saver", "balanced", "safety_first", "custom"],
                        help="Objective preset for multi-objective reward")
    parser.add_argument("--save-dir", type=str, default="results/sac_objective_training",
                        help="Directory to save training results")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N episodes")
    args = parser.parse_args()
    train_sac_objective(
        episodes=args.episodes,
        econ_preset=args.econ_preset,
        use_objective=args.use_objective_reward,
        objective_preset=args.objective_preset,
        save_dir=args.save_dir,
        log_every=args.log_every,
    )
