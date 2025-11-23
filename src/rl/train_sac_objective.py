"""
Objective-Conditioned SAC Training Script (Stub).

This script is a placeholder for future objective-conditioned training.
It marks the exact location where RewardBuilder will be integrated.

Current Status: NO BEHAVIOR CHANGE - just logging structure.

Future Status: Will replace fixed reward with objective-conditioned reward
shaped by ObjectiveVector + EconContext + EnergyResponseNet.

See: src/valuation/REWARD_INTEGRATION_DESIGN.md for full integration plan.
"""

import argparse
from typing import Dict, Any, Optional

# Standard imports
import numpy as np

# Project imports
from src.config.econ_params import EconParams
from src.config.objective_profile import ObjectiveVector
from src.envs.physics.backend_factory import make_backend
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.valuation.reward_builder import (
    build_reward_terms,
    combine_reward,
    default_objective_vector,
)
from src.observation.condition_vector_builder import ConditionVectorBuilder


def extract_step_summary(info: Dict[str, Any]) -> EpisodeInfoSummary:
    """
    Convert step-level info dict to EpisodeInfoSummary-like structure.

    This is a bridge function for step-level reward shaping.
    In practice, we may need a StepInfo dataclass instead.

    Args:
        info: Step info dict from env.step()

    Returns:
        EpisodeInfoSummary (or similar) with step-level metrics
    """
    # TODO: Define step-level summary structure
    # For now, treat step info as if it were episode summary
    return EpisodeInfoSummary(
        termination_reason=info.get("terminated_reason", ""),
        mpl_episode=info.get("mpl", 0.0),
        ep_episode=info.get("ep", 0.0),
        error_rate_episode=info.get("error_rate", 0.0),
        throughput_units_per_hour=info.get("mpl", 0.0),
        energy_Wh=info.get("energy_Wh", 0.0),
        energy_Wh_per_unit=info.get("energy_Wh_per_unit", 0.0),
        energy_Wh_per_hour=info.get("energy_Wh_per_hour", 0.0),
        limb_energy_Wh=info.get("limb_energy_Wh", {}),
        skill_energy_Wh=info.get("skill_energy_Wh", {}),
        energy_per_limb=info.get("energy_per_limb", {}),
        energy_per_skill=info.get("energy_per_skill", {}),
        energy_per_joint=info.get("energy_per_joint", {}),
        energy_per_effector=info.get("energy_per_effector", {}),
        coordination_metrics=info.get("coordination_metrics", {}),
        profit=info.get("profit", 0.0),
        wage_parity=info.get("wage_parity"),
    )


def train_episode_with_objective(
    backend,
    policy,
    econ_params: EconParams,
    objective_vector: list,
    use_objective_reward: bool = False,
    condition_builder: Optional[ConditionVectorBuilder] = None,
) -> Dict[str, Any]:
    """
    Run single training episode with objective-conditioned reward.

    Args:
        backend: Physics backend (PyBullet or Isaac)
        policy: RL policy (SAC, PPO, etc.)
        econ_params: Economic parameters
        objective_vector: 5-element objective weights [mpl, error, energy, safety, novelty]
        use_objective_reward: If True, use RewardBuilder; else use raw reward

    Returns:
        Episode metrics dict
    """
    obs = backend.reset()
    done = False
    step_count = 0
    total_raw_reward = 0.0
    total_objective_reward = 0.0

    # Track both reward types for comparison
    raw_rewards = []
    objective_rewards = []

    while not done:
        # Get action from policy
        # TODO: Pass objective_vector to policy for conditioning
        action = policy.select_action(obs)

        # Step environment
        obs, raw_reward, done, info = backend.step(action)

        # ===============================================================
        # TODO: PLUG RewardBuilder HERE (future integration point)
        # ===============================================================
        # Build objective-conditioned reward
        step_summary = extract_step_summary(info)
        reward_terms = build_reward_terms(step_summary, econ_params)
        objective_reward = combine_reward(objective_vector, reward_terms)

        # Choose which reward to use for training
        if use_objective_reward:
            # FUTURE: Enable this when ready
            shaped_reward = objective_reward
        else:
            # CURRENT: Use raw reward (no behavior change)
            shaped_reward = raw_reward

        # Add to replay buffer (stub)
        # policy.buffer.add(obs, action, shaped_reward, next_obs, done)

        # Track metrics
        raw_rewards.append(raw_reward)
        objective_rewards.append(objective_reward)
        total_raw_reward += raw_reward
        total_objective_reward += objective_reward
        step_count += 1

    # Episode summary
    episode_summary = backend.get_episode_info()
    episode_reward_terms = build_reward_terms(episode_summary, econ_params)
    episode_J = combine_reward(objective_vector, episode_reward_terms)

    condition_vector = None
    if condition_builder is not None:
        condition_vector = condition_builder.build(
            episode_config={
                "task_id": getattr(backend, "task_id", "sac_objective_stub"),
                "env_id": getattr(backend, "env_id", "objective_env"),
                "backend": getattr(backend, "engine_type", "stub"),
                "objective_preset": "custom",
                "objective_vector": objective_vector,
            },
            econ_state={
                "target_mpl": episode_summary.mpl_episode,
                "current_wage_parity": episode_summary.wage_parity if episode_summary.wage_parity is not None else 1.0,
                "energy_budget_wh": episode_summary.energy_Wh,
            },
            curriculum_phase="warmup",
            sima2_trust=None,
            datapack_metadata={"tags": ["objective_stub"]},
            episode_step=step_count,
            episode_metadata={"episode_id": getattr(backend, "episode_id", "stub_episode")},
        )

    return {
        "steps": step_count,
        "total_raw_reward": total_raw_reward,
        "total_objective_reward": total_objective_reward,
        "episode_J": episode_J,
        "reward_terms": episode_reward_terms,
        "objective_vector": objective_vector,
        "mpl_episode": episode_summary.mpl_episode,
        "error_rate_episode": episode_summary.error_rate_episode,
        "energy_Wh": episode_summary.energy_Wh,
        "wage_parity": episode_summary.wage_parity,
        "termination_reason": episode_summary.termination_reason,
        "condition_vector": condition_vector.to_dict() if condition_builder and condition_vector else None,
        # Correlation between reward types
        "reward_correlation": np.corrcoef(raw_rewards, objective_rewards)[0, 1]
        if len(raw_rewards) > 1
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Objective-Conditioned SAC Training")
    parser.add_argument("--env", type=str, default="drawer_vase")
    parser.add_argument("--engine-type", type=str, default="pybullet")
    parser.add_argument(
        "--objective-preset",
        type=str,
        default=None,
        help="Objective preset: throughput, safety, energy_saver, balanced",
    )
    parser.add_argument(
        "--objective-vector",
        type=float,
        nargs=5,
        default=None,
        help="Custom objective vector: [w_mpl, w_error, w_energy, w_safety, w_novelty]",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--use-objective-reward",
        action="store_true",
        help="Enable objective-conditioned reward (default: use raw reward)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-condition-vector",
        action="store_true",
        help="Build a stub ConditionVector for determinism checks (no behavior change)",
    )
    args = parser.parse_args()

    # Load objective vector
    if args.objective_vector:
        objective_vector = args.objective_vector
        print(f"Using custom objective vector: {objective_vector}")
    elif args.objective_preset:
        objective_vector = ObjectiveVector.from_preset(args.objective_preset).to_list()
        print(f"Using preset '{args.objective_preset}': {objective_vector}")
    else:
        objective_vector = default_objective_vector()
        print(f"Using default objective vector: {objective_vector}")

    # Load econ params (stub)
    econ_params = EconParams(
        price_per_unit=0.30,
        mpl_human=60.0,
        wage_human=18.0,
        vase_break_cost=5.0,
        electricity_price_kWh=0.12,
    )

    print("\n" + "=" * 60)
    print("OBJECTIVE-CONDITIONED TRAINING (STUB)")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Engine: {args.engine_type}")
    print(f"Objective: {objective_vector}")
    print(f"Use objective reward: {args.use_objective_reward}")
    print("=" * 60)

    # NOTE: This is a stub. In real training:
    # 1. Create environment
    # 2. Create backend wrapper
    # 3. Create policy (SAC, PPO, etc.)
    # 4. Run training loop with train_episode_with_objective()
    # 5. Log metrics and datapacks

    print("\nStub training loop would run here...")
    print(f"  Episodes: {args.episodes}")
    print(f"  Objective vector: {objective_vector}")

    if args.use_objective_reward:
        print("\n  [!] OBJECTIVE REWARD ENABLED")
        print("      RewardBuilder will shape rewards based on objective_vector")
    else:
        print("\n  [i] OBJECTIVE REWARD DISABLED (default)")
        print("      Using raw environment reward (no behavior change)")

    if args.use_condition_vector:
        cv_builder = ConditionVectorBuilder()
        stub_cv = cv_builder.build(
            episode_config={
                "task_id": args.env,
                "env_id": args.env,
                "backend": args.engine_type,
                "objective_preset": args.objective_preset or "balanced",
                "objective_vector": objective_vector,
            },
            econ_state={"target_mpl": 0.0, "current_wage_parity": 1.0, "energy_budget_wh": 0.0},
            curriculum_phase="warmup",
            sima2_trust=None,
            datapack_metadata={"tags": ["objective_stub"]},
            episode_step=0,
            episode_metadata={"episode_id": "stub_episode", "sampler_strategy": "balanced"},
        )
        print("\n[ConditionVector] Stub build (JSON-safe):")
        print(stub_cv.to_dict())

    print("\nTo enable objective-conditioned reward:")
    print("  python -m src.rl.train_sac_objective --use-objective-reward")
    print("\nTo use a preset:")
    print("  python -m src.rl.train_sac_objective --objective-preset throughput")
    print("\nTo use custom weights:")
    print("  python -m src.rl.train_sac_objective --objective-vector 2.0 1.0 0.5 1.0 0.0")


if __name__ == "__main__":
    main()
