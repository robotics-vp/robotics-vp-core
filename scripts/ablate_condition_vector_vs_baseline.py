#!/usr/bin/env python3
"""
Tiny ablation scaffold comparing baseline vs policy-conditioned SAC runs.

Runs two short SAC rollouts on a stub environment:
- baseline: condition vectors built/logged but not fed to policy conditioning
- conditioned: policy conditioning flag enabled (zero-init keeps outputs stable)

Logs per-episode MPL, energy, damage, reward, curriculum phase, and skill_mode.
"""
import random
from typing import Dict, List

import numpy as np
import torch
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.encoders.mlp_encoder import EncoderWithAuxiliaries
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.rl.sac import SACAgent
from src.rl.trunk_net import TrunkNet


class TinyEnv:
    """Deterministic stub environment emitting SAC-compatible observations."""

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.t = 0
        self.completed = 0.0
        self.attempts = 0
        self.errors = 0
        return self._obs()

    def _obs(self):
        return {
            "t": self.t,
            "completed": self.completed,
            "attempts": self.attempts,
            "errors": self.errors,
        }

    def step(self, action: np.ndarray):
        speed = float(action[0]) if len(action) > 0 else 0.5
        care = float(action[1]) if len(action) > 1 else 0.5
        self.t += 1
        self.attempts += 1
        # Completed units grow with speed; errors drop with care
        self.completed += max(speed - 0.1, 0.0)
        self.errors += 1 if care < 0.2 and random.random() < 0.5 else 0

        mpl_t = self.completed / max(1.0, self.t / 3600.0)
        energy_Wh = 0.05 * speed * self.t
        info = {
            "mpl_t": mpl_t,
            "ep_t": self.completed,
            "delta_errors": float(self.errors),
            "energy_Wh": energy_Wh,
            "terminated_reason": "time" if self.t >= self.max_steps else "",
        }
        done = self.t >= self.max_steps
        return self._obs(), info, done


def _run_short_sac(use_condition_vector_for_policy: bool, episodes: int = 10, seed: int = 0) -> List[Dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = TinyEnv(max_steps=6)
    obs_dim = 4
    latent_dim = 16
    encoder = EncoderWithAuxiliaries(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=64,
        use_consistency=False,
        use_contrastive=False,
    )
    agent = SACAgent(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=2,
        lr=1e-3,
        gamma=0.99,
        tau=5e-3,
        buffer_capacity=500,
        batch_size=32,
        target_entropy=-2.0,
        device="cpu",
    )
    condition_builder = ConditionVectorBuilder()
    policy_conditioner = None
    if use_condition_vector_for_policy:
        policy_conditioner = TrunkNet(
            vision_dim=1,
            state_dim=1,
            condition_dim=32,
            hidden_dim=latent_dim,
            use_condition_film=False,
            use_condition_vector=True,
            use_condition_vector_for_policy=True,
            condition_fusion_mode="film",
            condition_film_hidden_dim=latent_dim,
            condition_context_dim=latent_dim,
        )

    results: List[Dict] = []
    for ep in range(episodes):
        obs = env.reset()
        condition_vector = condition_builder.build(
            episode_config={
                "task_id": "ablate_stub",
                "env_id": "stub_env",
                "backend": "stub_backend",
                "objective_preset": "balanced",
            },
            econ_state={"target_mpl": 0.0, "current_wage_parity": 1.0, "energy_budget_wh": 0.0},
            curriculum_phase="warmup" if ep < episodes // 2 else "frontier",
            sima2_trust=None,
            datapack_metadata={"tags": ["ablate_condition_vector"]},
            episode_step=0,
            episode_metadata={"episode_id": f"ablate_ep_{ep}"},
        )

        done = False
        episode_reward = 0.0
        episode_energy = 0.0
        episode_damage = 0.0

        while not done:
            novelty = 0.5
            if use_condition_vector_for_policy and policy_conditioner is not None:
                with torch.no_grad():
                    obs_tensor = agent._obs_to_tensor(obs)
                    latent = agent.encoder.encode(obs_tensor)
                    conditioned_latent = policy_conditioner.condition_policy_features(latent, condition_vector) or latent
                    action_tensor, _ = agent.actor.sample(conditioned_latent, deterministic=False, return_log_prob=False)
                    action = action_tensor.cpu().numpy()[0]
            else:
                action, _ = agent.select_action(obs, novelty=novelty)

            next_obs, info, done = env.step(action)
            reward = info.get("mpl_t", 0.0) - 0.1 * info.get("delta_errors", 0.0)
            agent.store_transition(obs, action, reward, next_obs, done, novelty)
            episode_reward += reward
            episode_energy += info.get("energy_Wh", 0.0)
            episode_damage += info.get("delta_errors", 0.0)
            obs = next_obs

        # One small training sweep for stability
        for _ in range(3):
            _ = agent.update()

        mpl_episode = env.completed / max(1e-6, env.t / 3600.0)
        results.append(
            {
                "episode": ep,
                "mpl": mpl_episode,
                "energy": episode_energy,
                "damage": episode_damage,
                "reward": episode_reward,
                "curriculum_phase": "warmup" if ep < episodes // 2 else "frontier",
                "skill_mode": getattr(condition_vector, "skill_mode", ""),
            }
        )
    return results


def main():
    baseline = _run_short_sac(use_condition_vector_for_policy=False, episodes=12, seed=7)
    conditioned = _run_short_sac(use_condition_vector_for_policy=True, episodes=12, seed=7)

    print("[Ablation] Baseline vs policy-conditioned (zero-init) runs completed.")
    for label, records in [("baseline", baseline), ("conditioned", conditioned)]:
        rewards = [r["reward"] for r in records]
        phases = {r["curriculum_phase"] for r in records}
        modes = {r["skill_mode"] for r in records}
        print(f"  {label}: episodes={len(records)} reward_mean={np.mean(rewards):.3f} phases={sorted(phases)} skill_modes={sorted(modes)}")


if __name__ == "__main__":
    main()
