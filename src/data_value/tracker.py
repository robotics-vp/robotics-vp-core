"""
Lightweight data value tracker for economic valuation of training episodes.
Tracks novelty, causal gain, and economic value of each datapoint.
"""
import numpy as np
from collections import deque


class DataValueTracker:
    """
    Tracks the economic value of training data via:
    1. Novelty: How different is this episode from recent history
    2. Causal Gain: ΔMPL relative to rolling baseline
    3. Economic Value: p · (ΔMPLᵢ - ΔMPL_bar)
    """

    def __init__(self, price_per_unit, window_size=100):
        """
        Args:
            price_per_unit: Revenue per unit ($/unit)
            window_size: Rolling window for baseline ΔMPL calculation
        """
        self.price_per_unit = price_per_unit
        self.window_size = window_size

        # History tracking
        self.mpl_history = deque(maxlen=window_size)
        self.feature_history = []  # For novelty computation

        # Statistics
        self.total_value = 0.0
        self.cumulative_novelty = 0.0

    def compute_novelty(self, episode_features):
        """
        Compute novelty score based on distance from recent episodes.

        For now, uses simple feature-based similarity.
        In practice, could use:
        - Embedding distance (video features)
        - State-action coverage
        - Policy output variance

        Args:
            episode_features: dict with keys like 'mp_r', 'err_rate', 'speed', etc.

        Returns:
            novelty_score: float in [0, 1], higher = more novel
        """
        if len(self.feature_history) == 0:
            # First episode is maximally novel
            return 1.0

        # Simple novelty: normalized distance from recent mean
        # Feature vector: [mp_r, err_rate] (normalized)
        current = np.array([
            episode_features['mp_r'] / 200.0,  # normalize by ~2x human max
            episode_features['err_rate']
        ])

        # Compute mean of recent history
        recent = np.array([
            [f['mp_r'] / 200.0, f['err_rate']]
            for f in self.feature_history[-min(50, len(self.feature_history)):]
        ])
        mean_recent = recent.mean(axis=0)

        # Euclidean distance as novelty proxy
        dist = np.linalg.norm(current - mean_recent)

        # Normalize to [0, 1] range (0.5 is typical distance)
        novelty = np.clip(dist / 0.5, 0.0, 1.0)

        return float(novelty)

    def compute_causal_gain(self, mp_r_current, mp_r_prev):
        """
        Compute causal gain: how much this episode improved MPL
        relative to rolling baseline improvement.

        Args:
            mp_r_current: Current episode MPL
            mp_r_prev: Previous episode MPL

        Returns:
            causal_gain: ΔMPL - ΔMPL_baseline (units/hr)
        """
        # Current improvement
        delta_mpl = mp_r_current - mp_r_prev

        # Baseline: rolling average of recent improvements
        if len(self.mpl_history) < 2:
            # Not enough history; return raw delta
            return delta_mpl

        # Compute rolling baseline ΔMPL
        recent_mpls = list(self.mpl_history)
        delta_mpl_baseline = np.mean([
            recent_mpls[i] - recent_mpls[i-1]
            for i in range(1, len(recent_mpls))
        ])

        # Causal gain = how much better than baseline
        causal_gain = delta_mpl - delta_mpl_baseline

        return float(causal_gain)

    def compute_economic_value(self, causal_gain):
        """
        Compute economic value of this datapoint.

        Value = p · max(0, causal_gain)

        Only positive contributions (above baseline) have value.

        Args:
            causal_gain: ΔMPL - ΔMPL_baseline

        Returns:
            value: Economic value ($/hr improvement)
        """
        value = self.price_per_unit * max(0.0, causal_gain)
        return float(value)

    def update(self, mp_r_current, mp_r_prev, episode_features):
        """
        Update tracker with new episode data.

        Args:
            mp_r_current: Current episode MPL
            mp_r_prev: Previous episode MPL
            episode_features: dict with episode info

        Returns:
            dict with novelty, causal_gain, economic_value
        """
        # Compute metrics
        novelty = self.compute_novelty(episode_features)
        causal_gain = self.compute_causal_gain(mp_r_current, mp_r_prev)
        economic_value = self.compute_economic_value(causal_gain)

        # Update history
        self.mpl_history.append(mp_r_current)
        self.feature_history.append(episode_features)

        # Update statistics
        self.total_value += economic_value
        self.cumulative_novelty += novelty

        return {
            'novelty': novelty,
            'causal_gain': causal_gain,
            'economic_value': economic_value,
            'cumulative_value': self.total_value
        }
