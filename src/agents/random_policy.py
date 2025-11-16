# src/agents/random_policy.py
import numpy as np

class RandomPolicy:
    def act(self, obs, info_history):
        # action is speed in [0,1]
        return float(np.random.rand())
