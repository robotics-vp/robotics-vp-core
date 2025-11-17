from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


class MPLRewardHead:
    def compute(self, summary: EpisodeInfoSummary, econ_params: EconParams) -> float:
        return summary.mpl_episode
