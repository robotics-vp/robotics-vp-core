# demo.py
from src.envs.dishwashing_env import DishwashingEnv, DishwashingParams
from src.agents.random_policy import RandomPolicy

def main():
    env = DishwashingEnv(DishwashingParams())
    pol = RandomPolicy()

    obs = env.reset()
    info_hist = []
    for _ in range(2):  # 2 minutes total now that step=1 minute
        a = pol.act(obs, info_hist)
        obs, info, done = env.step(a)
        info_hist.append(info)

    attempts = env.attempts
    print(f"Completed={env.completed}  Attempts={attempts}  Errors={env.errors}")

if __name__ == "__main__":
    main()
