"""
SAC training with learned encoder, auxiliary losses, and Lagrangian constraint.

End-to-end deep learning:
- Encoder f_ψ learns latent representation
- Policy π_θ and critics Q_ϕ operate on latents
- Novelty weighting from diffusion
- Economic reward + Lagrangian constraint
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from src.envs.dishwashing_env import DishwashingEnv, summarize_episode_info
from src.rl.sac import SACAgent
from src.encoders.mlp_encoder import EncoderWithAuxiliaries
from src.economics.wage import implied_robot_wage
from src.economics.spread_allocation import compute_spread_allocation
from src.economics.data_value import OnlineDataValueEstimator
from src.economics.wage_indexer import WageIndexer, WageIndexConfig
from src.economics.pricing import compute_customer_cost_per_hour, compute_consumer_surplus
from src.utils.logger import CsvLogger
from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params
from src.rl.reward_shaping import compute_econ_reward
from src.physics.backends.factory import make_backend
from src.rl.episode_sampling import (
    DataPackRLSampler,
    load_episode_descriptors_from_jsonl,
    load_enrichments_from_jsonl,
)
from src.rl.curriculum import DataPackCurriculum
from src.valuation.datapack_schema import DataPackMeta
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.rl.trunk_net import TrunkNet


def _load_datapacks(path: Path):
    """Load datapacks from JSON/JSONL; returns [] if missing."""
    datapacks = []
    if not path or not path.exists():
        return datapacks
    if path.suffix == ".jsonl":
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                datapacks.append(DataPackMeta.from_dict(json.loads(line)))
    else:
        with path.open("r") as f:
            data = json.load(f)
            for dp in data:
                datapacks.append(DataPackMeta.from_dict(dp))
    return datapacks


def _build_stage3_components(
    use_datapack_curriculum: bool,
    sampler_mode: str,
    episodes: int,
    seed: int,
    datapacks_path: str,
    enrichments_path: str,
    descriptors_path: str,
    curriculum_total_steps: int,
    curriculum_config_path: str,
    use_condition_vector: bool = False,
):
    """Optionally construct sampler/curriculum; returns (sampler, curriculum) or (None, None)."""
    if not use_datapack_curriculum:
        return None, None

    datapacks = _load_datapacks(Path(datapacks_path)) if datapacks_path else []
    enrichments_path_obj = Path(enrichments_path) if enrichments_path else None
    descriptors_path_obj = Path(descriptors_path) if descriptors_path else None
    enrichments = load_enrichments_from_jsonl(enrichments_path_obj) if enrichments_path_obj and enrichments_path_obj.exists() else []
    descriptors = load_episode_descriptors_from_jsonl(descriptors_path_obj) if descriptors_path_obj and descriptors_path_obj.exists() else []

    if not datapacks and not descriptors:
        print("[stage3] No datapacks/descriptors found; datapack curriculum disabled.")
        return None, None

    sampler = DataPackRLSampler(
        datapacks=datapacks or None,
        enrichments=enrichments or None,
        existing_descriptors=descriptors or None,
        default_strategy=sampler_mode or "balanced",
        use_condition_vector=use_condition_vector,
    )

    curriculum = None
    if sampler_mode is None:
        config = {}
        if curriculum_config_path:
            try:
                with open(curriculum_config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"[stage3] Failed to load curriculum config ({e}); using defaults.")
        config["base_seed"] = seed
        config["use_condition_vector"] = use_condition_vector
        curriculum = DataPackCurriculum(
            sampler=sampler,
            total_steps=curriculum_total_steps or episodes,
            config=config,
        )

    print(f"[stage3] Datapack sampler initialized: {sampler.pool_summary()}")
    if curriculum:
        print("[stage3] Curriculum enabled.")
    elif sampler_mode:
        print(f"[stage3] Static sampler strategy: {sampler_mode}")
    return sampler, curriculum


def train_sac(
    episodes=1000,
    seed=42,
    econ_preset="toy",
    use_datapack_curriculum: bool = False,
    sampler_mode: str = None,
    curriculum_total_steps: int = None,
    datapacks_path: str = "",
    enrichments_path: str = "",
    descriptors_path: str = "",
    curriculum_config_path: str = "",
    log_path: str = "logs/sac_train.csv",
    checkpoint_path: str = "checkpoints/sac_final.pt",
    sampler: DataPackRLSampler = None,
    curriculum: DataPackCurriculum = None,
    physics_backend: str = "pybullet",
    use_mobility_policy: bool = False,
    use_condition_vector: bool = False,
    use_condition_vector_for_policy: bool = False,
):
    """Train SAC agent with economic objectives."""

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    profile = get_internal_experiment_profile("dishwashing")
    econ_params = load_econ_params(profile, preset=econ_preset)

    # Config
    MPh = 60.0
    wh_initial = 18.0  # Initial human wage (will be indexed)
    p = 0.30
    damage_cost = 1.0
    energy_cost = 0.10

    # Wage indexer config
    episodes_per_update = 100  # Update wage index every N episodes
    sector_inflation_annual = 0.02  # 2% annual inflation (stub)
    # Assume 4 quarters per year, so per-episode inflation rate
    episodes_per_year = 400  # Rough estimate for simulation
    sector_inflation_per_episode = sector_inflation_annual / episodes_per_year

    # Curriculum for error target
    err_target_final = 0.06
    err_target_init = 0.10
    curriculum_eps = 600

    # Lagrangian
    lambda_init = 0.0
    eta = 0.01

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Environment / physics backend
    backend = make_backend(physics_backend, {"econ_preset": econ_preset, "use_mobility_policy": use_mobility_policy})
    env = backend.env if hasattr(backend, "env") else DishwashingEnv(econ_params)

    # Encoder (with auxiliary heads)
    obs_dim = 4
    latent_dim = 128
    encoder = EncoderWithAuxiliaries(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dim=256,
        use_consistency=True,
        use_contrastive=True
    )

    # SAC Agent
    agent = SACAgent(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=2,  # [speed, care]
        lr=3e-4,
        gamma=0.995,
        tau=5e-3,
        buffer_capacity=int(1e6),
        batch_size=1024,
        target_entropy=-2.0,
        device=device
    )
    policy_conditioner = None
    if use_condition_vector and use_condition_vector_for_policy:
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
        ).to(device)

    # Logger
    logger = CsvLogger(log_path)

    # Dual variable
    lam = lambda_init

    # Track previous MPL for delta computation
    prev_mp_r = None
    objective_vector = profile.get("default_objective_vector", [1.0, 0.7, 0.5, 0.8])
    alpha_mpl, alpha_error, alpha_ep, alpha_safety = objective_vector

    # Online data value estimator (novelty → ΔMPL)
    data_value_estimator = OnlineDataValueEstimator(
        lookback_window=100,
        min_samples=10
    )

    # Wage indexer (dynamic human wage benchmark)
    # NOTE: This is for economic logging only, does NOT affect RL rewards
    wage_indexer = WageIndexer(
        initial_wage=wh_initial,
        config=WageIndexConfig(
            alpha=0.1,
            inflation_adj=True,
            min_update_interval=1
        )
    )
    wh = wage_indexer.current()  # Current indexed wage

    print("=== SAC Training with Learned Encoder ===")
    print(f"Episodes: {episodes}")
    print(f"Latent dim: {latent_dim}")
    print(f"Device: {device}")
    print(f"Initial human wage: ${wh:.2f}/hr\n")

    # Stage 3 sampling/curriculum (advisory-only, flag-gated)
    sampler_obj, curriculum_obj = sampler, curriculum
    if sampler_obj is None and curriculum_obj is None:
        sampler_obj, curriculum_obj = _build_stage3_components(
            use_datapack_curriculum=use_datapack_curriculum,
            sampler_mode=sampler_mode,
            episodes=episodes,
            seed=seed,
            datapacks_path=datapacks_path,
            enrichments_path=enrichments_path,
            descriptors_path=descriptors_path,
            curriculum_total_steps=curriculum_total_steps or episodes,
            curriculum_config_path=curriculum_config_path,
            use_condition_vector=use_condition_vector,
        )
    condition_builder = ConditionVectorBuilder() if use_condition_vector else None

    # Training loop
    for ep in range(episodes):
        # Update wage index periodically (deterministic stub for reproducibility)
        if ep > 0 and ep % episodes_per_update == 0:
            # Stub market wage: slight drift upward (deterministic)
            market_wage_stub = wh_initial * (1.0 + 0.005 * (ep // episodes_per_update))
            # Stub inflation per update period
            inflation_stub = sector_inflation_per_episode * episodes_per_update
            # Update wage indexer
            wh = wage_indexer.update(market_wage_stub, inflation_stub)

        # Curriculum
        err_target = np.interp(ep, [0, curriculum_eps],
                              [err_target_init, err_target_final])

        obs_dict = env.reset()
        done = False

        episode_steps = 0
        episode_reward_sum = 0.0
        actions_taken = []
        terminated_reason = None
        info_history = []
        descriptor = None
        sampler_strategy = ""
        curriculum_phase = ""
        condition_vector = None
        policy_condition_norm = 0.0

        if curriculum_obj:
            batch = curriculum_obj.sample_batch(step=ep, batch_size=1)
            if batch:
                descriptor = batch[0]
                meta = descriptor.get("sampling_metadata", {})
                sampler_strategy = meta.get("strategy", "")
                curriculum_phase = meta.get("phase", "")
        elif sampler_obj:
            batch = sampler_obj.sample_batch(
                batch_size=1,
                seed=seed + ep,
                strategy=sampler_mode or sampler_obj.default_strategy,
            )
            if batch:
                descriptor = batch[0]
                meta = descriptor.get("sampling_metadata", {})
                sampler_strategy = meta.get("strategy", sampler_mode or sampler_obj.default_strategy)
                curriculum_phase = meta.get("phase", "")

        # Collect episode
        while not done:
            # Compute novelty (stub for now)
            novelty = float(np.random.rand() * 0.5 + 0.5)

            # Select action
            action, _ = agent.select_action(obs_dict, novelty=novelty)

            # Environment step
            next_obs_dict, info, done = env.step(action)
            terminated_reason = info.get("terminated_reason", terminated_reason)
            info_history.append(info)

            # Track actions
            actions_taken.append([info['speed'], info['care']])

            # MPL/EP/error-based reward
            mpl_t = info.get("mpl_t", 0.0)
            ep_t = info.get("ep_t", 0.0)
            err_term = info.get("delta_errors", 0.0)
            wage_parity_step = None  # SAC dishwashing currently skip wage penalty

            reward, reward_components = compute_econ_reward(
                mpl=mpl_t,
                ep=ep_t,
                error_rate=err_term,
                wage_parity=wage_parity_step,
                mode="mpl_ep_error",
                alpha_mpl=alpha_mpl,
                alpha_error=alpha_error,
                alpha_ep=alpha_ep,
                alpha_wage=0.0,
            )

            last_reward_components = reward_components

            # Store transition
            agent.store_transition(obs_dict, action, reward, next_obs_dict, done, novelty)

            episode_reward_sum += reward
            obs_dict = next_obs_dict
            episode_steps += 1

        # Episode metrics
        summary = summarize_episode_info(info_history)
        time_hours = next_obs_dict['t'] / 3600.0
        mp_r = summary.mpl_episode
        err_rate = summary.error_rate_episode
        w_hat_r = implied_robot_wage(p, mp_r, err_rate, damage_cost)
        wage_parity = w_hat_r / wh
        prod_parity = mp_r / MPh

        revenue = p * mp_r
        error_cost = damage_cost * (err_rate * mp_r)
        profit = revenue - error_cost - energy_cost
        ep_episode = summary.ep_episode

        # Average actions
        actions_arr = np.array(actions_taken)
        mean_speed = actions_arr[:, 0].mean() if len(actions_arr) > 0 else 0.0
        mean_care = actions_arr[:, 1].mean() if len(actions_arr) > 0 else 0.0

        # Update λ
        lam = max(0.0, lam + eta * (err_rate - err_target))

        # SAC updates (multiple per episode after warmup)
        train_metrics = {}
        if ep > 10:  # Warmup
            for _ in range(episode_steps):  # One update per step
                metrics = agent.update(aux_loss_weight={
                    'consistency': 0.1,
                    'contrastive': 0.1
                })
                if metrics:
                    train_metrics = metrics  # Keep last

        # Compute ΔMPL for spread allocation
        if prev_mp_r is None:
            delta_mpl_total = 0.0
        else:
            delta_mpl_total = mp_r - prev_mp_r

        # Get novelty features (from SAC training metrics)
        novelty_raw = float(train_metrics.get('mean_novelty', 0.5))

        # Predict customer ΔMPL from novelty using online estimator
        delta_mpl_cust_pred = data_value_estimator.predict(novelty_raw)

        # Update estimator with actual ΔMPL (online learning)
        data_value_estimator.update(novelty_raw, delta_mpl_total)

        # Use predicted ΔMPL_cust for spread allocation
        delta_mpl_cust = delta_mpl_cust_pred

        # Update prev_mp_r for next episode
        prev_mp_r = mp_r

        # Compute spread allocation (mechanistic split based on ΔMPL contributions)
        spread_info = compute_spread_allocation(
            w_robot=w_hat_r,
            w_human=wh,
            hours=time_hours,
            delta_mpl_cust=delta_mpl_cust,
            delta_mpl_total=delta_mpl_total,
            eps_parity=0.05
        )

        # Customer pricing with consumer surplus guarantee
        # NOTE: This is for economic accounting only, does NOT affect RL
        customer_cost = compute_customer_cost_per_hour(
            w_robot=w_hat_r,
            w_human=wh,
            rebate=spread_info['rebate'] / time_hours if time_hours > 0 else 0.0,  # Convert to $/hr
            base_fee=0.0,
            floor_margin=0.0
        )
        consumer_surplus = compute_consumer_surplus(
            w_human=wh,
            customer_cost=customer_cost
        )

        if condition_builder is not None:
            semantic_tags = descriptor.get("semantic_tags", {}) if descriptor else {}
            advisory_context = descriptor.get("sampling_metadata", {}) if descriptor else {"strategy": sampler_strategy}
            episode_md = {
                "episode_id": descriptor.get("episode_id", descriptor.get("pack_id", f"ep_{ep}")) if descriptor else f"ep_{ep}",
                "sampler_strategy": sampler_strategy,
                "curriculum_phase": curriculum_phase or "warmup",
                "timestep": next_obs_dict.get("t"),
            }
            econ_slice = {
                "mpl": mp_r,
                "wage_parity": wage_parity,
                "energy_wh": summary.energy_Wh if hasattr(summary, "energy_Wh") else 0.0,
                "damage_cost": error_cost,
            }
            condition_vector = condition_builder.build(
                episode_config=descriptor or {"task_id": "dishwashing", "env_id": "dishwashing_env", "backend": physics_backend, "objective_preset": "balanced"},
                econ_state={"target_mpl": mp_r, "current_wage_parity": wage_parity, "energy_budget_wh": summary.energy_Wh if hasattr(summary, "energy_Wh") else 0.0},
                curriculum_phase=curriculum_phase or "warmup",
                sima2_trust=None,
                datapack_metadata=descriptor.get("metadata", {}) if descriptor else {},
                episode_step=episode_steps,
                econ_slice=econ_slice,
                semantic_tags=semantic_tags if isinstance(semantic_tags, dict) else {str(t): 1.0 for t in semantic_tags},
                recap_scores=None,
                trust_summary=None,
                episode_metadata=episode_md,
                advisory_context=advisory_context,
            )
            if policy_conditioner is not None:
                with torch.no_grad():
                    base_latent = torch.zeros(1, agent.latent_dim, device=device)
                    conditioned_latent = policy_conditioner.condition_policy_features(base_latent, condition_vector)
                    if conditioned_latent is not None:
                        policy_condition_norm = float(conditioned_latent.abs().sum().item())

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
            mpl_episode=round(mp_r, 2),
            ep_episode=round(ep_episode, 4),
            econ_preset=econ_params.preset,
            w_hat_r=round(w_hat_r, 2),
            w_h=wh,
            wage_parity=round(wage_parity, 4),
            prod_parity=round(prod_parity, 4),
            profit=round(profit, 2),
            lambda_dual=round(lam, 4),
            episode_reward=round(episode_reward_sum, 2),
            episode_steps=episode_steps,
            reward_mpl=round(last_reward_components.get("mpl_component", 0.0), 4) if 'last_reward_components' in locals() else 0.0,
            reward_ep=round(last_reward_components.get("ep_component", 0.0), 4) if 'last_reward_components' in locals() else 0.0,
            reward_error=round(last_reward_components.get("error_penalty", 0.0), 4) if 'last_reward_components' in locals() else 0.0,
            error_rate_episode=round(err_rate, 4),
            energy_Wh=round(summary.energy_Wh, 4),
            terminated_reason=terminated_reason or "",
            mean_speed=round(mean_speed, 3),
            mean_care=round(mean_care, 3),
            buffer_size=len(agent.replay_buffer),
            critic_loss=round(train_metrics.get('critic_loss', 0.0), 6),
            actor_loss=round(train_metrics.get('actor_loss', 0.0), 6),
            alpha=round(train_metrics.get('alpha', 0.0), 4),
            consistency_loss=round(train_metrics.get('consistency_loss', 0.0), 6),
            contrastive_loss=round(train_metrics.get('contrastive_loss', 0.0), 6),
            mean_novelty=round(train_metrics.get('mean_novelty', 0.0), 4),
            mean_weight=round(train_metrics.get('mean_weight', 1.0), 4),
            q_mean=round(train_metrics.get('q_mean', 0.0), 2),
            # Data value estimation
            novelty_raw=round(novelty_raw, 4),
            delta_mpl_cust_pred=round(delta_mpl_cust_pred, 4),
            # Spread allocation (mechanistic)
            delta_mpl_total=round(delta_mpl_total, 4),
            delta_mpl_cust=round(delta_mpl_cust, 4),
            spread=round(spread_info['spread'], 4),
            spread_value=round(spread_info['spread_value'], 4),
            s_cust=round(spread_info['s_cust'], 4),
            s_plat=round(spread_info['s_plat'], 4),
            rebate=round(spread_info['rebate'], 4),
            captured_spread=round(spread_info['captured'], 4),
            # Wage indexing and customer pricing
            w_h_indexed=round(wh, 4),
            customer_cost=round(customer_cost, 4),
            consumer_surplus=round(consumer_surplus, 4),
            # Stage 3 advisory metadata
            sampler_strategy=sampler_strategy,
            curriculum_phase=curriculum_phase,
            pack_id=descriptor.get("pack_id", "") if descriptor else "",
            objective_preset=descriptor.get("objective_preset", "") if descriptor else "",
            condition_skill_mode=getattr(condition_vector, "skill_mode", ""),
            condition_phase=getattr(condition_vector, "curriculum_phase", ""),
            policy_condition_norm=round(policy_condition_norm, 6),
        )

        # Print progress
        if (ep + 1) % 50 == 0:
            print(f"[ep {ep+1:4d}] MP={mp_r:.0f}/h  Profit=${profit:.2f}  "
                  f"Err={err_rate:.3f}(tgt={err_target:.3f})  λ={lam:.3f}  "
                  f"Speed={mean_speed:.2f} Care={mean_care:.2f}  "
                  f"WageParity={wage_parity:.3f}  "
                  f"Buffer={len(agent.replay_buffer)}  "
                  f"α={train_metrics.get('alpha', 0.0):.3f}")

    # Save final model
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    print(f"\n✅ Training complete: {log_path}")
    print(f"✅ Model saved: {checkpoint_path}\n")


if __name__ == "__main__":
    import os
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--econ-preset", type=str, default="toy", choices=["toy", "realistic"])
    parser.add_argument("--engine-type", type=str, default="pybullet")
    parser.add_argument("--objective-preset", type=str, default="throughput",
                        choices=["throughput", "energy_saver", "balanced", "safety_first", "custom"])
    parser.add_argument("--constraint-mpl-human", type=float, default=None)
    parser.add_argument("--constraint-energy-budget", type=float, default=None)
    parser.add_argument("--constraint-error-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Stage 3 curriculum flags (advisory-only)
    parser.add_argument("--use-datapack-curriculum", action="store_true", help="Enable Stage 3 datapack sampler/curriculum (advisory-only)")
    parser.add_argument("--sampler-mode", type=str, default=None, choices=["balanced", "frontier_prioritized", "econ_urgency"], help="Static sampler mode when curriculum is not used (requires --use-datapack-curriculum)")
    parser.add_argument("--datapacks-path", type=str, default="results/stage1_pipeline/datapacks.json", help="Path to Stage 1 datapacks JSON/JSONL")
    parser.add_argument("--enrichments-path", type=str, default="results/stage2_semantic/semantic_enrichments.jsonl", help="Path to Stage 2 semantic enrichments JSONL")
    parser.add_argument("--descriptors-path", type=str, default="", help="Optional JSONL of existing RL episode descriptors")
    parser.add_argument("--curriculum-total-steps", type=int, default=None, help="Override total steps for curriculum phase boundaries")
    parser.add_argument("--curriculum-config", type=str, default="", help="Optional JSON file overriding curriculum boundaries/mix")
    parser.add_argument("--physics-backend", type=str, default="pybullet", choices=["pybullet", "isaac_stub", "isaac"], help="Physics backend selection (pybullet default)")
    parser.add_argument("--use-mobility-policy", action="store_true", help="Enable advisory mobility micro-policy in physics backends")
    parser.add_argument("--log-path", type=str, default="logs/sac_train.csv")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/sac_final.pt")
    parser.add_argument("--use-condition-vector", action="store_true", help="Build/log ConditionVector without affecting SAC behavior")
    parser.add_argument(
        "--use-condition-vector-for-policy",
        action="store_true",
        help="Feed ConditionVector through the policy conditioning block (default off; zero-init keeps outputs unchanged).",
    )
    args = parser.parse_args()

    if args.engine_type != "pybullet":
        raise NotImplementedError("Only pybullet engine supported in this script.")
    train_sac(
        episodes=args.episodes,
        seed=args.seed,
        econ_preset=args.econ_preset,
        use_datapack_curriculum=args.use_datapack_curriculum,
        sampler_mode=args.sampler_mode,
        curriculum_total_steps=args.curriculum_total_steps,
        datapacks_path=args.datapacks_path,
        enrichments_path=args.enrichments_path,
        descriptors_path=args.descriptors_path,
        curriculum_config_path=args.curriculum_config,
        log_path=args.log_path,
        checkpoint_path=args.checkpoint_path,
        physics_backend=args.physics_backend,
        use_mobility_policy=args.use_mobility_policy,
        use_condition_vector=args.use_condition_vector,
        use_condition_vector_for_policy=args.use_condition_vector_for_policy,
    )
