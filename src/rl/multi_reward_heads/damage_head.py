from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


class DamageRewardHead:
    def compute(self, summary: EpisodeInfoSummary, econ_params: EconParams) -> float:
        # Penalize errors/catastrophic events
        return -summary.error_rate_episode
