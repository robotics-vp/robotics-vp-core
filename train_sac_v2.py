"""
SAC training with learned encoder - CONFIG-DRIVEN (State or Video mode)

Supports both:
- State mode: MLP encoder on state vectors
- Video mode: Video encoder on (T, C, H, W) observations

SAC, economics, and pricing logic UNCHANGED - only observation modality differs.
"""
import sys
import os
import argparse
import yaml
import numpy as np
import torch

from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams
from src.envs.video_wrappers import DishwashingVideoEnv
from src.envs.physics import DishwashingPhysicsEnv, create_physics_env
from src.rl.sac import SACAgent
from src.encoders.mlp_encoder import EncoderWithAuxiliaries
from src.encoders.builder import build_encoder
from src.economics.mpl import mpl
from src.economics.wage import implied_robot_wage
from src.economics.reward import econ_lagrangian_reward
from src.economics.spread_allocation import compute_spread_allocation
from src.economics.data_value import OnlineDataValueEstimator
from src.economics.wage_indexer import WageIndexer, WageIndexConfig
from src.economics.pricing import compute_customer_cost_per_hour, compute_consumer_surplus
from src.data_value.novelty_diffusion import DiffusionNoveltyEstimator
from src.utils.logger import CsvLogger


def make_env(cfg):
    """
    Environment factory - creates state, video, or physics environment based on config.

    Args:
        cfg: Config dict

    Returns:
        Environment instance
    """
    env_config = cfg.get('env', {})
    env_type = env_config.get('type', 'dishwashing')

    # Create base environment params
    env_params = DishwashingParams()

    if env_type == 'dishwashing':
        # State mode: standard env
        return DishwashingEnv(env_params)

    elif env_type == 'dishwashing_video':
        # Video mode: wrap with video env (synthetic frames)
        base_env = DishwashingEnv(env_params)

        video_config = env_config.get('video', {})
        video_env = DishwashingVideoEnv(
            base_env=base_env,
            frames=video_config.get('frames', 8),
            height=video_config.get('height', 64),
            width=video_config.get('width', 64),
            render_mode=video_config.get('render_mode', 'synthetic')
        )
        return video_env

    elif env_type == 'dishwashing_physics':
        # Physics mode: PyBullet simulation with real rendered frames
        # Phase A: Pass all calibration parameters
        physics_env = DishwashingPhysicsEnv(
            frames=env_config.get('frames', 8),
            image_size=tuple(env_config.get('image_size', [64, 64])),
            max_steps=env_config.get('max_steps', 60),
            headless=env_config.get('headless', True),
            camera_config=env_config.get('physics', {}).get('camera', None),
            # Phase A: Stochastic realism
            randomize_dishes=env_config.get('randomize_dishes', True),
            camera_jitter=env_config.get('camera_jitter', 0.02),
            lighting_variation=env_config.get('lighting_variation', 0.1),
            # Phase A: Better error model
            slip_probability=env_config.get('slip_probability', 0.05),
            gripper_failure_rate=env_config.get('gripper_failure_rate', 0.02),
            # Phase A: Human-ish throughput caps
            max_speed_multiplier=env_config.get('max_speed_multiplier', 2.0),
            max_acceleration=env_config.get('max_acceleration', 1.0)
        )
        return physics_env

    else:
        raise ValueError(f"Unknown env type: {env_type}")


def train_sac(config_path, episodes=None, seed=42):
    """
    Train SAC agent with economic objectives.

    Args:
        config_path: Path to YAML config file
        episodes: Number of episodes (overrides config if provided)
        seed: Random seed
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract config values
    MPh = cfg['economics']['human_mp']
    wh_initial = cfg['wage_indexer']['initial_wage']
    p = cfg['economics']['price_per_unit']
    damage_cost = cfg['economics']['damage_cost']
    energy_cost = cfg['economics']['energy_cost']
    # Phase A: Wage parity penalty coefficient
    beta_parity = cfg['economics'].get('beta_parity', 0.5)  # Default 0.5

    # Wage indexer config
    episodes_per_update = cfg['wage_indexer']['episodes_per_update']
    sector_inflation_annual = cfg['wage_indexer']['sector_inflation_annual']
    episodes_per_year = 400  # Rough estimate
    sector_inflation_per_episode = sector_inflation_annual / episodes_per_year

    # Curriculum for error target
    err_target_final = cfg['constraint']['err_target_final']
    err_target_init = cfg['constraint']['err_target_init']
    curriculum_eps = cfg['constraint']['curriculum_episodes']

    # Lagrangian
    lambda_init = cfg['constraint']['lambda_init']
    eta = cfg['constraint']['eta']

    # Training config
    if episodes is None:
        episodes = cfg['training']['max_episodes']

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Environment (state or video)
    env_type = cfg.get('env', {}).get('type', 'dishwashing')
    env = make_env(cfg)
    print(f"Environment: {env_type}")

    # Get observation dimensions
    if env_type == 'dishwashing':
        # State mode
        obs_dim = 4
        obs_shape = None
    elif env_type in ['dishwashing_video', 'dishwashing_physics']:
        # Video mode (synthetic or physics-rendered)
        obs_dim = None
        obs_shape = env.observation_space_shape  # (T, C, H, W)
    else:
        raise ValueError(f"Unknown env type: {env_type}")

    # Encoder (MLP or Video)
    encoder_type = cfg['encoder']['type']
    latent_dim = cfg['encoder']['latent_dim']

    if encoder_type == 'mlp':
        # Use EncoderWithAuxiliaries for state mode (has consistency/contrastive losses)
        encoder = EncoderWithAuxiliaries(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=cfg['encoder']['mlp'].get('hidden_dim', 256),
            use_consistency=True,
            use_contrastive=True
        )
        print(f"Encoder: MLP (obs_dim={obs_dim} -> latent_dim={latent_dim})")

    elif encoder_type == 'video':
        # Use video encoder from builder
        encoder = build_encoder(cfg['encoder'], video_shape=obs_shape, device=device)
        print(f"Encoder: Video (arch={cfg['encoder']['video']['arch']}, latent_dim={latent_dim})")

        # Note: Video encoder doesn't have auxiliary losses yet
        # We'll just use it for encoding
    elif encoder_type == 'aligned':
        # Use pretrained aligned encoder (z_V)
        from src.encoders.student_video_encoder import AlignedVideoEncoder
        aligned_cfg = cfg['encoder'].get('aligned', {})
        encoder = AlignedVideoEncoder(
            latent_dim=latent_dim,
            arch=aligned_cfg.get('arch', 'simple2dcnn'),
            input_channels=aligned_cfg.get('input_channels', 3),
            projection_dim=aligned_cfg.get('projection_dim', None),
            alignment_type=aligned_cfg.get('alignment_type', 'mse'),
            temperature=aligned_cfg.get('temperature', 0.1),
        )

        # Load pretrained checkpoint if provided
        checkpoint_path = cfg['encoder'].get('checkpoint', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'student_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['student_state_dict'])
                print(f"Encoder: Aligned (loaded from {checkpoint_path})")
                print(f"  Final cosine similarity: {checkpoint.get('final_cos_sim', 'N/A')}")
            else:
                encoder.load_state_dict(checkpoint)
                print(f"Encoder: Aligned (loaded weights from {checkpoint_path})")
        else:
            print(f"Encoder: Aligned (untrained)")

        # Freeze encoder for now (canonical visual backbone)
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"  Encoder frozen: True")
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # SAC Agent
    sac_cfg = cfg['sac']
    agent = SACAgent(
        encoder=encoder,
        latent_dim=latent_dim,
        action_dim=2,  # [speed, care]
        lr=sac_cfg.get('lr_actor', 3e-4),
        gamma=sac_cfg.get('gamma', 0.995),
        tau=sac_cfg.get('tau', 5e-3),
        buffer_capacity=sac_cfg.get('buffer_size', int(1e6)),
        batch_size=sac_cfg.get('batch_size', 1024),
        target_entropy=-2.0,
        device=device
    )

    # Logger
    log_config = cfg.get('logging', {})
    log_prefix = log_config.get('prefix', 'sac')
    log_path = f"logs/{log_prefix}_train.csv"
    logger = CsvLogger(log_path)

    # Dual variable
    lam = lambda_init

    # Track previous MPL for delta computation
    prev_mp_r = None

    # Online data value estimator (novelty → ΔMPL)
    data_value_estimator = OnlineDataValueEstimator(
        lookback_window=100,
        min_samples=10
    )

    # Diffusion-based novelty estimator (operates on latents)
    diffusion_novelty = DiffusionNoveltyEstimator(
        latent_dim=latent_dim,
        device=device
    )

    # Wage indexer
    wage_indexer = WageIndexer(
        initial_wage=wh_initial,
        config=WageIndexConfig(
            alpha=cfg['wage_indexer']['alpha'],
            inflation_adj=cfg['wage_indexer']['inflation_adj'],
            min_update_interval=1
        )
    )
    wh = wage_indexer.current()

    print("=== SAC Training ===")
    print(f"Config: {config_path}")
    print(f"Episodes: {episodes}")
    print(f"Latent dim: {latent_dim}")
    print(f"Environment: {env_type}")
    print(f"Encoder: {encoder_type}")
    print(f"Initial human wage: ${wh:.2f}/hr\n")

    # Training loop
    for ep in range(episodes):
        # Update wage index periodically
        if ep > 0 and ep % episodes_per_update == 0:
            market_wage_stub = wh_initial * (1.0 + 0.005 * (ep // episodes_per_update))
            inflation_stub = sector_inflation_per_episode * episodes_per_update
            wh = wage_indexer.update(market_wage_stub, inflation_stub)

        # Curriculum
        err_target = np.interp(ep, [0, curriculum_eps],
                              [err_target_init, err_target_final])

        # Reset environment
        if env_type == 'dishwashing':
            # State mode: returns dict
            obs_dict = env.reset()
        else:
            # Video mode: returns (T, C, H, W) array
            obs_video = env.reset()
            # Get underlying state for economic computation
            obs_dict = env.get_state()

        done = False
        episode_steps = 0
        episode_reward_sum = 0.0
        actions_taken = []

        # Collect episode
        while not done and episode_steps < 60:
            # Use dummy novelty during episode (compute properly once at end)
            novelty = 0.5

            # Select action
            if env_type == 'dishwashing':
                # State mode: pass dict
                action, _ = agent.select_action(obs_dict, novelty=novelty)
            else:
                # Video mode: pass video observation directly
                # Agent will convert to tensor and encode
                action, _ = agent.select_action(obs_video, novelty=novelty)

            # Environment step
            if env_type == 'dishwashing':
                next_obs_dict, info, done = env.step(action)
            else:
                # Video mode
                next_obs_video, info, done = env.step(action)
                # Get underlying state for economic computation
                next_obs_dict = env.get_state()

            # Track actions
            actions_taken.append([info['speed'], info['care']])

            # Economic reward (same for both modes)
            time_hours = next_obs_dict['t'] / 3600.0
            mp_r = mpl(next_obs_dict['completed'], time_hours) if time_hours > 0 else 0
            err_rate = next_obs_dict['errors'] / max(1, next_obs_dict['attempts'])

            reward = econ_lagrangian_reward(
                mp_r, err_rate, p, damage_cost,
                lam, err_target, energy_cost
            )

            # Phase A: Add wage parity penalty
            # Pull wage_parity down toward 1.0 (human benchmark)
            w_hat_r_step = implied_robot_wage(p, mp_r, err_rate, damage_cost)
            wage_parity_step = w_hat_r_step / wh if wh > 0 else 1.0
            over_parity = max(0.0, wage_parity_step - 1.0)

            # Apply penalty for exceeding human wage parity
            reward -= beta_parity * over_parity

            # Store transition
            if env_type == 'dishwashing':
                agent.store_transition(obs_dict, action, reward, next_obs_dict, done, novelty)
            else:
                # Video mode: store video observations directly
                agent.store_transition(obs_video, action, reward, next_obs_video, done, novelty)

            episode_reward_sum += reward

            # Update obs
            if env_type == 'dishwashing':
                obs_dict = next_obs_dict
            else:
                obs_video = next_obs_video
                obs_dict = next_obs_dict
                obs_for_agent = {'obs': obs_video}

            episode_steps += 1

        # Episode metrics (same for both modes)
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

        # Update λ
        lam = max(0.0, lam + eta * (err_rate - err_target))

        # SAC updates (fixed count per episode, not per step!)
        train_metrics = {}
        updates_per_episode = 1  # Fixed O(1) updates per episode
        if ep > 10:  # Warmup
            for _ in range(updates_per_episode):
                if encoder_type == 'mlp':
                    # MLP encoder has auxiliary losses
                    metrics = agent.update(aux_loss_weight={
                        'consistency': 0.1,
                        'contrastive': 0.1
                    })
                else:
                    # Video encoder: no auxiliary losses yet
                    metrics = agent.update(aux_loss_weight={})

                if metrics:
                    train_metrics = metrics

        # Compute ΔMPL for spread allocation
        if prev_mp_r is None:
            delta_mpl_total = 0.0
        else:
            delta_mpl_total = mp_r - prev_mp_r

        # Compute novelty ONCE per episode (not per step!)
        with torch.no_grad():
            if env_type == 'dishwashing':
                # State mode: convert final state to tensor
                obs_tensor = torch.FloatTensor([
                    next_obs_dict['t'], next_obs_dict['completed'],
                    next_obs_dict['attempts'], next_obs_dict['errors']
                ]).unsqueeze(0).to(device)
            else:
                # Video mode: use final video observation
                obs_tensor = torch.FloatTensor(next_obs_video).unsqueeze(0).to(device)

            # Encode to latent
            z = encoder.encode(obs_tensor)  # (1, latent_dim)

            # Compute novelty on latent
            novelty_tensor = diffusion_novelty.compute(z)  # (1,) tensor
            novelty_raw = float(novelty_tensor[0].item()) if isinstance(novelty_tensor, torch.Tensor) else float(novelty_tensor)

        # Predict customer ΔMPL from novelty
        delta_mpl_cust_pred = data_value_estimator.predict(novelty_raw)

        # Update estimator
        data_value_estimator.update(novelty_raw, delta_mpl_total)

        # Use predicted ΔMPL_cust
        delta_mpl_cust = delta_mpl_cust_pred

        # Update prev_mp_r
        prev_mp_r = mp_r

        # Spread allocation (same for both modes)
        spread_info = compute_spread_allocation(
            w_robot=w_hat_r,
            w_human=wh,
            hours=time_hours,
            delta_mpl_cust=delta_mpl_cust,
            delta_mpl_total=delta_mpl_total,
            eps_parity=0.05
        )

        # Customer pricing (same for both modes)
        customer_cost = compute_customer_cost_per_hour(
            w_robot=w_hat_r,
            w_human=wh,
            rebate=spread_info['rebate'] / time_hours if time_hours > 0 else 0.0,
            base_fee=0.0,
            floor_margin=0.0
        )
        consumer_surplus = compute_consumer_surplus(
            w_human=wh,
            customer_cost=customer_cost
        )

        # Log (same for both modes)
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
            # Data value
            novelty_raw=round(novelty_raw, 4),
            delta_mpl_cust_pred=round(delta_mpl_cust_pred, 4),
            # Spread allocation
            delta_mpl_total=round(delta_mpl_total, 4),
            delta_mpl_cust=round(delta_mpl_cust, 4),
            spread=round(spread_info['spread'], 4),
            spread_value=round(spread_info['spread_value'], 4),
            s_cust=round(spread_info['s_cust'], 4),
            s_plat=round(spread_info['s_plat'], 4),
            rebate=round(spread_info['rebate'], 4),
            captured_spread=round(spread_info['captured'], 4),
            # Wage indexing and pricing
            w_h_indexed=round(wh, 4),
            customer_cost=round(customer_cost, 4),
            consumer_surplus=round(consumer_surplus, 4)
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
    model_path = f"checkpoints/{log_prefix}_final.pt"
    agent.save(model_path)
    print(f"\n✅ Training complete: {log_path}")
    print(f"✅ Model saved: {model_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC with economic objectives')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Train
    train_sac(args.config, episodes=args.episodes, seed=args.seed)
