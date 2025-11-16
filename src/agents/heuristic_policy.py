# src/agents/heuristic_policy.py
class HeuristicPolicy:
    def __init__(self, target_err=0.06, step=0.05, init_speed=0.3):
        self.speed = init_speed
        self.target_err = target_err
        self.step = step

    def act(self, obs, info_history):
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
