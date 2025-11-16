# train_ppo.py
"""
PPO training with diffusion-based novelty and economic valuation.

Integrates:
- PPO policy learning
- Diffusion novelty signals
- Economic metrics (MPL, profit, wage parity)
- Data valuation (ΔMPL regression → pricing)
"""
import os, yaml, random
import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams
from src.rl.ppo import PPOAgent
from src.economics.mpl import mpl
from src.economics.wage import implied_robot_wage
from src.rl.reward_shaping import compute_econ_reward
from src.config.internal_profile import get_internal_experiment_profile
from src.utils.logger import CsvLogger
from src.data_value.novelty_diffusion import (
    DiffusionNoveltyTracker,
    StubDenoiser,
    StubShortDenoise,
    mse_noise_gap,
    recon_gap,
    gaussian_noise_sampler
)


def load_config(path="src/configs/dishwashing.yaml"):
    import sys
    # Allow override via command line
    if len(sys.argv) > 1:
        path = sys.argv[1]
    with open(path, "r") as f:
        return yaml.safe_load(f)


def obs_to_latent(obs):
    """
    Convert observation dict to latent vector for novelty computation.

    In real V2P: this would be video encoder output.
    For now: simple feature extraction.
    """
    features = np.array([
        obs['t'] / 3600.0,
        obs['completed'] / 200.0,
        obs['attempts'] / 300.0,
        obs['errors'] / 50.0
    ], dtype=np.float32)
    return torch.FloatTensor(features).unsqueeze(0)


class DataValueRegressor:
    """
    Online regression: Novelty → ΔMPL

    Estimates how much MPL improvement each episode contributes
    based on its novelty score. Used for economic data valuation.
    """

    def __init__(self, window=200):
        self.window = window
        self.X_history = []  # novelty scores
        self.y_history = []  # ΔMPL values
        self.model = Ridge(alpha=1.0)
        self.fitted = False

    def add_sample(self, novelty, delta_mpl):
        """Add new (novelty, ΔMPL) pair."""
        self.X_history.append(novelty)
        self.y_history.append(delta_mpl)

        # Keep only recent window
        if len(self.X_history) > self.window:
            self.X_history = self.X_history[-self.window:]
            self.y_history = self.y_history[-self.window:]

        # Fit model if enough samples
        if len(self.X_history) >= 20:
            X = np.array(self.X_history).reshape(-1, 1)
            y = np.array(self.y_history)
            self.model.fit(X, y)
            self.fitted = True

    def predict(self, novelty):
        """Predict ΔMPL from novelty score."""
        if not self.fitted:
            return 0.0

        return self.model.predict([[novelty]])[0]

    def get_baseline(self):
        """Get baseline ΔMPL (recent average)."""
        if len(self.y_history) < 10:
            return 0.0
        return np.mean(self.y_history[-50:])


def run():
    # Config
    cfg = load_config()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Human benchmarks
    mp_h = float(cfg["human"]["mpl_units_per_hour"])
    w_h = float(cfg["human"]["wage_per_hour"])

    # Economics
    p = float(cfg["economics"]["price_per_unit"])
    c_d = float(cfg["economics"]["damage_cost"])
    energy_cost = float(cfg["economics"]["energy_cost_per_hour"])

    # Quality constraint
    use_lagrangian = cfg["quality"]["lagrangian"]["enabled"]
    e_star = float(cfg["quality"]["error_target"])
    lam = float(cfg["quality"]["lagrangian"]["lambda_init"])
    eta = float(cfg["quality"]["lagrangian"]["step_eta"])

    # Environment
    env = DishwashingEnv(DishwashingParams(
        price_per_unit=p,
        damage_cost=c_d,
        time_step_s=60.0
    ))

    # PPO Agent
    obs_dim = 4  # [t, completed, attempts, errors]
    action_dim = 1  # speed control
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=64,
        novelty_alpha=1.0,
        novelty_beta=0.0
    )

    # Diffusion novelty tracker (using stub models for now)
    latent_dim = obs_dim
    denoiser = StubDenoiser(latent_dim)
    short_denoise = StubShortDenoise()
    novelty_tracker = DiffusionNoveltyTracker(
        ema_decay=0.99,
        alpha=1.0,
        beta=1.0
    )

    # Data value regressor
    data_regressor = DataValueRegressor(window=200)

    # Logging
    episodes = int(cfg["train"]["episodes"])
    eval_every = int(cfg["train"]["eval_every"])
    max_secs = int(cfg["train"]["max_seconds_per_episode"])
    logger = CsvLogger("logs/ppo_training.csv")

    # Training loop
    mp_r_prev = 1e-8

    for ep in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        # Compute initial novelty
        latent = obs_to_latent(obs)
        with torch.no_grad():
            n_mse = mse_noise_gap(
                denoiser, latent,
                ts=[0.1, 0.5, 0.9],
                noise_sampler=gaussian_noise_sampler,
                reps=2
            ).item()
            n_recon = recon_gap(short_denoise, latent).item()

        novelty = novelty_tracker.compute_novelty(
            torch.tensor(n_mse),
            torch.tensor(n_recon),
            update_stats=True
        ).item()

    profile = get_internal_experiment_profile("dishwashing")
    alpha_mpl, alpha_error, alpha_ep, _ = profile.get("default_objective_vector", [1.0, 1.0, 1.0, 0.0])

        last_reward_components = {}
        # Episode rollout
    while env.t < max_secs:
        # Select action with novelty signal
        action, logprob, value = agent.select_action(obs, novelty=novelty)

            # Step environment
            obs_next, info, done = env.step(action[0])

        # Compute economic reward (per-step MPL/EP/error)
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
            episode_reward += reward
            episode_steps += 1

            # Update novelty for next state
            latent = obs_to_latent(obs_next)
            with torch.no_grad():
                n_mse = mse_noise_gap(
                    denoiser, latent,
                    ts=[0.1, 0.5, 0.9],
                    noise_sampler=gaussian_noise_sampler,
                    reps=2
                ).item()
                n_recon = recon_gap(short_denoise, latent).item()

            novelty = novelty_tracker.compute_novelty(
                torch.tensor(n_mse),
                torch.tensor(n_recon),
                update_stats=True
            ).item()

            obs = obs_next

            if done:
                break

        # End of episode metrics
        time_h = env.t / 3600.0
        attempts = env.attempts
        errors = env.errors
        completed = env.completed
        err_rate = errors / max(attempts, 1)
        mp_r = mpl(units_completed=completed, time_hours=max(time_h, 1e-8))
        w_hat_r = implied_robot_wage(
            price_per_unit=p, mp_r=mp_r, error_rate=err_rate, damage_cost=c_d
        )

        # Update dual variable (λ)
        if use_lagrangian:
            lam = max(0.0, lam + eta * (err_rate - e_star))

        # Economic metrics
        wage_parity = w_hat_r / w_h if w_h > 0 else 0.0
        prod_parity = mp_r / mp_h if mp_h > 0 else 0.0
        profit = p * mp_r - c_d * (err_rate * mp_r) - energy_cost

        # Data valuation
        delta_mpl = mp_r - mp_r_prev
        data_regressor.add_sample(novelty, delta_mpl)

        # Predicted ΔMPL from novelty
        predicted_delta_mpl = data_regressor.predict(novelty)
        baseline_delta_mpl = data_regressor.get_baseline()

        # Economic value of this datapoint
        data_value = p * max(0.0, predicted_delta_mpl - baseline_delta_mpl)

        # Non-sharing premium (pricing for data withholding)
        kappa = cfg["data_value"]["kappa_confidence"]
        hours_horizon = cfg["data_value"]["pricing_horizon_hours"]
        scale_mult = cfg["data_value"]["deployment_scale"]

        nonshare_premium = (
            p * kappa * max(0.0, predicted_delta_mpl) * hours_horizon * scale_mult
        )

        # PPO update
        last_value = agent.ac(obs_to_latent(obs).to(agent.device))[2].item()
        train_metrics = agent.update(last_value=last_value)

        # Logging
        logger.log(
            episode=ep,
            time_h=round(time_h, 6),
            completed=int(completed),
            attempts=int(attempts),
            errors=int(errors),
            err_rate=round(err_rate, 6),
            mp_r=round(mp_r, 6),
            mp_h=mp_h,
            w_hat_r=round(w_hat_r, 6),
            w_h=w_h,
            wage_parity=round(wage_parity, 6),
            prod_parity=round(prod_parity, 6),
            profit=round(profit, 6),
            lambda_dual=round(lam, 6),
            err_target=e_star,
            episode_reward=round(episode_reward, 6),
            episode_steps=episode_steps,
            reward_mpl=round(last_reward_components.get("mpl_component", 0.0), 6),
            reward_ep=round(last_reward_components.get("ep_component", 0.0), 6),
            reward_error=round(last_reward_components.get("error_penalty", 0.0), 6),
            # PPO metrics
            policy_loss=round(train_metrics['policy_loss'], 6),
            value_loss=round(train_metrics['value_loss'], 6),
            entropy=round(train_metrics['entropy'], 6),
            kl_divergence=round(train_metrics['kl_divergence'], 6),
            # Novelty & data valuation
            novelty=round(novelty, 6),
            mean_weight=round(train_metrics['mean_weight'], 6),
            p90_weight=round(train_metrics['p90_weight'], 6),
            mean_valuation=round(train_metrics['mean_valuation'], 6),
            delta_mpl=round(delta_mpl, 6),
            predicted_delta_mpl=round(predicted_delta_mpl, 6),
            data_value=round(data_value, 6),
            nonshare_premium=round(nonshare_premium, 6)
        )

        mp_r_prev = mp_r

        # Print progress
        if (ep + 1) % max(1, eval_every) == 0:
            print(f"[ep {ep+1}] MP_r={mp_r:.2f}/h  Profit=${profit:.2f}  "
                  f"Err={err_rate:.3f}  λ={lam:.3f}  "
                  f"Nov={novelty:.3f}  Wt={train_metrics['mean_weight']:.2f}(p90={train_metrics['p90_weight']:.2f})  "
                  f"KL={train_metrics['kl_divergence']:.4f}  Premium=${nonshare_premium:.2f}  "
                  f"PolLoss={train_metrics['policy_loss']:.3f}")

    print(f"\n✅ Training complete! Logs saved to logs/ppo_training.csv")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/ppo_final.pt")
    print(f"✅ Model saved to checkpoints/ppo_final.pt")


if __name__ == "__main__":
    run()
