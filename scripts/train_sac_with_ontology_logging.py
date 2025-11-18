#!/usr/bin/env python3
"""
Optional SAC training wrapper with ontology logging.

Does not alter reward math; uses EpisodeLogger + RewardEngine to log episodes,
events, and econ vectors into the ontology store.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.envs.dishwashing_env import DishwashingEnv
from src.rl.sac import SACAgent
from src.encoders.mlp_encoder import EncoderWithAuxiliaries
from src.economics.reward_engine import RewardEngine
from src.logging.episode_logger import EpisodeLogger
from src.ontology.models import Task, Robot
from src.ontology.store import OntologyStore
from src.config.econ_params import EconParams


def main():
    parser = argparse.ArgumentParser(description="SAC training with ontology logging (optional)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, default="task_dishwashing")
    parser.add_argument("--robot-id", type=str, default="robot_sac")
    parser.add_argument("--use-mobility-policy", action="store_true", help="Enable advisory mobility micro-policy (stub)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    econ_params = EconParams(
        price_per_unit=0.3,
        damage_cost=1.0,
        energy_Wh_per_attempt=0.05,
        time_step_s=60.0,
        base_rate=2.0,
        p_min=0.02,
        k_err=0.12,
        q_speed=1.2,
        q_care=1.5,
        care_cost=0.25,
        max_steps=240,
        max_catastrophic_errors=3,
        max_error_rate_sla=0.12,
        min_steps_for_sla=5,
        zero_throughput_patience=10,
        preset="toy",
    )
    env = DishwashingEnv(econ_params)
    obs_dim = 4
    latent_dim = 128
    encoder = EncoderWithAuxiliaries(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=256,
        use_consistency=True,
        use_contrastive=True
    )
    agent = SACAgent(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=2,
        lr=3e-4,
        gamma=0.995,
        tau=5e-3,
        buffer_capacity=int(1e6),
        batch_size=64,
        target_entropy=-2.0,
        device='cpu',
    )

    store = OntologyStore(root_dir=args.ontology_root)
    task = Task(
        task_id=args.task_id,
        name="Dishwashing",
        description="Stub task for SAC logging",
        environment_id="dishwashing_env",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(robot_id=args.robot_id, name="DishwasherBot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    reward_engine = RewardEngine(task, robot, config={})
    logger = EpisodeLogger(store=store, task=task, robot=robot)

    for ep_idx in range(args.episodes):
        episode = logger.start_episode()
        obs = env.reset()
        done = False
        timestep = 0
        while not done:
            action, _ = agent.select_action(obs, novelty=0.5)
            next_obs, info, done = env.step(action)
            raw_reward = info.get("reward", 0.0) if isinstance(info, dict) else 0.0
            scalar_reward, components = reward_engine.step_reward(raw_reward, info if isinstance(info, dict) else {})
            logger.log_step(
                timestep=timestep,
                reward_scalar=scalar_reward,
                reward_components=components,
                state_summary={"obs": obs},
            )
            agent.store_transition(obs, action, scalar_reward, next_obs, done, novelty=0.5)
            obs = next_obs
            timestep += 1
        econ = reward_engine.compute_econ_vector(episode, logger._events)  # Using buffered events
        logger.mark_outcome(status="success")
        logger.finalize(econ_vector=econ)

    print(f"[train_sac_with_ontology_logging] Completed {args.episodes} episodes with ontology logging at {args.ontology_root}")


if __name__ == "__main__":
    main()
