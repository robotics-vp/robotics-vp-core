from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


class NoveltyRewardHead:
    def compute(self, summary: EpisodeInfoSummary, econ_params: EconParams) -> float:
        # Placeholder novelty (none available in summary)
        return 0.0
