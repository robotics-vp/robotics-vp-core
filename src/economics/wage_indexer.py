"""
Dynamic wage indexer for tracking human wage benchmarks.

Smooths toward market wages and optionally adjusts for sector inflation.
This is an accounting layer only - does NOT affect RL rewards or training.
"""
from dataclasses import dataclass


@dataclass
class WageIndexConfig:
    """Configuration for wage indexer."""
    alpha: float = 0.1              # Smoothing factor (0 = no update, 1 = instant)
    inflation_adj: bool = True      # Whether to adjust for sector inflation
    min_update_interval: int = 1    # Episodes between updates (stub for now)


class WageIndexer:
    """
    Tracks human wage benchmark with smoothing and inflation adjustment.

    This is used purely for:
    - Computing wage parity (ŵᵣ / wₕ)
    - Spread allocation (ŵᵣ - wₕ)
    - Customer pricing (capped at wₕ)

    NOT used for RL rewards or policy training.
    """

    def __init__(self, initial_wage: float, config: WageIndexConfig):
        """
        Args:
            initial_wage: Starting human wage benchmark ($/hr)
            config: WageIndexConfig with smoothing and inflation settings
        """
        self.w_h_prev = initial_wage
        self.config = config

    def update(self, market_wage: float, sector_inflation: float = 0.0) -> float:
        """
        Update benchmark wage with smoothing and inflation adjustment.

        Args:
            market_wage: Current market wage observation ($/hr)
            sector_inflation: Sector inflation rate (fraction, e.g., 0.02 = 2%)

        Returns:
            Updated benchmark wage wₕ ($/hr)

        Implementation:
            1. Smooth toward market_wage: w_smooth = α·market + (1-α)·w_prev
            2. Adjust for inflation: w_real = w_smooth / (1 + inflation)
            3. Store and return w_real
        """
        alpha = self.config.alpha

        # Exponential smoothing toward market wage
        w_smoothed = alpha * market_wage + (1.0 - alpha) * self.w_h_prev

        # Optional inflation adjustment (convert nominal → real)
        if self.config.inflation_adj:
            w_real = w_smoothed / (1.0 + sector_inflation)
        else:
            w_real = w_smoothed

        # Update internal state
        self.w_h_prev = w_real
        return w_real

    def current(self) -> float:
        """
        Get current benchmark wage.

        Returns:
            Current wₕ ($/hr)
        """
        return self.w_h_prev
