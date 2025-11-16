"""
SAC training with learned encoder, auxiliary losses, and Lagrangian constraint.

End-to-end deep learning:
- Encoder f_ψ learns latent representation
- Policy π_θ and critics Q_ϕ operate on latents
- Novelty weighting from diffusion
- Economic reward + Lagrangian constraint
"""
import sys
import numpy as np
import torch

from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams
from src.rl.sac import SACAgent
from src.encoders.mlp_encoder import EncoderWithAuxiliaries
from src.economics.mpl import mpl
from src.economics.wage import implied_robot_wage
from src.economics.reward import econ_lagrangian_reward
from src.economics.spread_allocation import compute_spread_allocation
from src.economics.data_value import OnlineDataValueEstimator
from src.economics.wage_indexer import WageIndexer, WageIndexConfig
from src.economics.pricing import compute_customer_cost_per_hour, compute_consumer_surplus
from src.utils.logger import CsvLogger


def train_sac(episodes=1000, seed=42):
    """Train SAC agent with economic objectives."""

    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # Environment
    env_params = DishwashingParams()
    env = DishwashingEnv(env_params)

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

    # Logger
    logger = CsvLogger('logs/sac_train.csv')

    # Dual variable
    lam = lambda_init

    # Track previous MPL for delta computation
    prev_mp_r = None

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

        # Collect episode
        while not done and episode_steps < 60:
            # Compute novelty (stub for now)
            novelty = float(np.random.rand() * 0.5 + 0.5)

            # Select action
            action, _ = agent.select_action(obs_dict, novelty=novelty)

            # Environment step
            next_obs_dict, info, done = env.step(action)

            # Track actions
            actions_taken.append([info['speed'], info['care']])

            # Economic reward
            time_hours = next_obs_dict['t'] / 3600.0
            mp_r = mpl(next_obs_dict['completed'], time_hours) if time_hours > 0 else 0
            err_rate = next_obs_dict['errors'] / max(1, next_obs_dict['attempts'])

            reward = econ_lagrangian_reward(
                mp_r, err_rate, p, damage_cost,
                lam, err_target, energy_cost
            )

            # Store transition
            agent.store_transition(obs_dict, action, reward, next_obs_dict, done, novelty)

            episode_reward_sum += reward
            obs_dict = next_obs_dict
            episode_steps += 1

        # Episode metrics
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
    agent.save('checkpoints/sac_final.pt')
    print(f"\n✅ Training complete: logs/sac_train.csv")
    print(f"✅ Model saved: checkpoints/sac_final.pt\n")


if __name__ == "__main__":
    import os
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episodes = 1000
    if len(sys.argv) > 1:
        episodes = int(sys.argv[1])

    train_sac(episodes=episodes)
