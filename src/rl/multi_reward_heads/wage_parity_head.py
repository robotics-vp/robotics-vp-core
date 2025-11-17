from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


class WageParityRewardHead:
    def compute(self, summary: EpisodeInfoSummary, econ_params: EconParams) -> float:
        wp = summary.wage_parity or 1.0 if hasattr(summary, "wage_parity") else 1.0
        return wp
