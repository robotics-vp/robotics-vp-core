# train.py
import os, math, yaml, random, numpy as np
from pathlib import Path

from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams
from src.economics.mpl import mpl
from src.economics.wage import implied_robot_wage
from src.economics.reward import wage_parity_reward, econ_lagrangian_reward
from src.utils.logger import CsvLogger
from src.data_value.tracker import DataValueTracker

class HeuristicAgent:
    """
    Simple controller:
      - if error rate too high, slow down
      - else speed up modestly
    """
    def __init__(self, target_err=0.06, step=0.05):
        self.speed = 0.3
        self.target_err = target_err
        self.step = step

    def act(self, obs, info_history):
        # try to infer recent error rate from history
        if info_history:
            errs = sum(i["errs"] for i in info_history[-10:])
            atts = sum(i["succ"] + i["errs"] for i in info_history[-10:])
            err_rate = (errs / max(atts, 1))
        else:
            err_rate = 0.05
        if err_rate > self.target_err:
            self.speed = max(0.0, self.speed - self.step)
        else:
            self.speed = min(1.0, self.speed + self.step/2)
        return self.speed

def load_config(path="src/configs/dishwashing.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def target_schedule(cfg, t_episode):
    sch = cfg["reward"]["target_schedule"]
    if sch["type"] == "constant":
        return float(sch["value"])
    else:
        # placeholder for future ramps
        return float(sch.get("value", 1.0))

def run():
    cfg = load_config()
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    # Human benchmarks
    mp_h = float(cfg["human"]["mpl_units_per_hour"])
    w_h = float(cfg["human"]["wage_per_hour"])

    # Econ
    p = float(cfg["economics"]["price_per_unit"])
    c_d = float(cfg["economics"]["damage_cost"])
    energy_cost = float(cfg["economics"]["energy_cost_per_hour"])

    # Quality constraint (Lagrangian)
    use_lagrangian = cfg["quality"]["lagrangian"]["enabled"]
    e_star = float(cfg["quality"]["error_target"])
    lam = float(cfg["quality"]["lagrangian"]["lambda_init"])
    eta = float(cfg["quality"]["lagrangian"]["step_eta"])

    # Legacy reward params (for comparison if needed)
    rwd = cfg["reward"]
    alpha, beta, gamma = float(rwd["alpha"]), float(rwd["beta"]), float(rwd["gamma"])
    lam_down = float(rwd["asymmetry"]["lambda_down"])
    lam_up = float(rwd["asymmetry"]["lambda_up"])

    # Env
    env = DishwashingEnv(DishwashingParams(
        price_per_unit=p,
        damage_cost=c_d,
        time_step_s=60.0   # 1 step = 1 minute
    ))

    episodes = int(cfg["train"]["episodes"])
    eval_every = int(cfg["train"]["eval_every"])
    max_secs = int(cfg["train"]["max_seconds_per_episode"])

    logger = CsvLogger(cfg["log"]["csv_path"])
    agent = HeuristicAgent()

    # Data value tracker
    data_tracker = DataValueTracker(
        price_per_unit=p,
        window_size=100
    )

    for ep in range(episodes):
        obs = env.reset()
        info_hist = []
        last_completed = 0
        last_time_s = 0.0
        mp_r_prev = 1e-8  # avoid div by zero in first reward

        # episode roll
        while env.t < max_secs:
            a = agent.act(obs, info_hist)
            obs, info, done = env.step(a)
            info_hist.append(info)
            if done:
                break

        # metrics
        time_h = env.t / 3600.0
        attempts = env.attempts
        errors = env.errors
        completed = env.completed
        err_rate = errors / max(attempts, 1)
        mp_r = mpl(units_completed=completed, time_hours=max(time_h, 1e-8))
        w_hat_r = implied_robot_wage(price_per_unit=p, mp_r=mp_r, error_rate=err_rate, damage_cost=c_d)

        # Compute reward using Lagrangian approach
        if use_lagrangian:
            reward_ep = econ_lagrangian_reward(
                mp_r=mp_r,
                err_rate=err_rate,
                price_per_unit=p,
                damage_cost=c_d,
                lam=lam,
                err_target=e_star,
                energy_cost_per_hour=energy_cost
            )
            # Dual ascent: update λ to enforce constraint
            lam = max(0.0, lam + eta * (err_rate - e_star))
        else:
            # Legacy wage parity reward
            s_t = target_schedule(cfg, ep)
            reward_ep = wage_parity_reward(
                mp_r=mp_r, mp_r_prev=mp_r_prev, mp_h=mp_h,
                w_hat_r=w_hat_r, w_h=w_h, err_rate=err_rate,
                alpha=alpha, beta=beta, gamma=gamma, target=s_t,
                lam_down=lam_down, lam_up=lam_up
            )

        # Compute economic metrics (for logging/KPIs)
        wage_parity = w_hat_r / w_h if w_h > 0 else 0.0
        prod_parity = mp_r / mp_h if mp_h > 0 else 0.0
        profit = p * mp_r - c_d * (err_rate * mp_r) - energy_cost

        # Data valuation metrics
        episode_features = {
            'mp_r': mp_r,
            'err_rate': err_rate,
            'speed': agent.speed  # Current agent speed
        }
        data_value_info = data_tracker.update(mp_r, mp_r_prev, episode_features)

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
            reward=round(reward_ep, 6),
            # Data valuation metrics
            novelty=round(data_value_info['novelty'], 6),
            causal_gain=round(data_value_info['causal_gain'], 6),
            data_value=round(data_value_info['economic_value'], 6),
            cumulative_data_value=round(data_value_info['cumulative_value'], 6)
        )

        mp_r_prev = mp_r

        if (ep + 1) % max(1, eval_every) == 0:
            print(f"[ep {ep+1}] MP_r={mp_r:.2f}/h  Profit=${profit:.2f}  "
                  f"Err={err_rate:.3f} (target={e_star:.3f})  λ={lam:.3f}  R={reward_ep:.3f}")

if __name__ == "__main__":
    run()
