"""
Online data value estimator.

Predicts E[ΔMPL_i] from novelty features using incremental learning.
This replaces the novelty proxy with honest "data → ΔMPL → price" path.
"""
import numpy as np
from sklearn.linear_model import SGDRegressor
from collections import deque


class OnlineDataValueEstimator:
    """
    Online regression to predict ΔMPL from novelty features.

    Uses SGDRegressor for incremental learning (handles distribution shift).
    Maintains rolling history for stability.
    """

    def __init__(self, lookback_window=100, min_samples=10):
        """
        Args:
            lookback_window: Keep only recent N observations
            min_samples: Minimum samples before making predictions
        """
        self.lookback_window = lookback_window
        self.min_samples = min_samples

        # SGD regressor with adaptive learning rate
        self.model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,  # Sufficient for initial fit convergence
            tol=1e-3,
            warm_start=True,  # Incremental learning
            random_state=42
        )

        # History: [(features, ΔMPL_actual), ...]
        self.history = deque(maxlen=lookback_window)
        self.is_fitted = False

    def predict(self, novelty_features):
        """
        Predict E[ΔMPL] from novelty features.

        Args:
            novelty_features: Single feature value (novelty score)
                             Can be extended to multi-dimensional

        Returns:
            predicted_delta_mpl: Expected ΔMPL (units/hr)
        """
        if not self.is_fitted or len(self.history) < self.min_samples:
            # Not enough data yet, return neutral estimate
            return 0.0

        # Reshape for sklearn
        features = np.array([[novelty_features]])

        try:
            prediction = self.model.predict(features)[0]
            # Clip to reasonable range (no negative ΔMPL predictions)
            return float(np.clip(prediction, -50, 50))
        except Exception:
            return 0.0

    def update(self, novelty_features, actual_delta_mpl):
        """
        Incremental update after observing actual ΔMPL.

        Args:
            novelty_features: Single feature value (novelty score)
            actual_delta_mpl: Observed ΔMPL (units/hr)
        """
        # Add to history
        self.history.append((novelty_features, actual_delta_mpl))

        # Skip update if not enough data
        if len(self.history) < self.min_samples:
            return

        # Reshape for sklearn
        X = np.array([[novelty_features]])
        y = np.array([actual_delta_mpl])

        # Incremental fit
        if not self.is_fitted:
            # First fit (need multiple samples for stability)
            if len(self.history) >= self.min_samples:
                X_hist = np.array([[h[0]] for h in self.history])
                y_hist = np.array([h[1] for h in self.history])
                self.model.fit(X_hist, y_hist)
                self.is_fitted = True
        else:
            # Incremental update
            self.model.partial_fit(X, y)

    def get_statistics(self):
        """
        Get current model statistics.

        Returns:
            dict with model info and prediction quality
        """
        if not self.is_fitted:
            return {
                'fitted': False,
                'n_samples': len(self.history),
                'min_samples_needed': self.min_samples
            }

        # Compute prediction error on history
        if len(self.history) >= self.min_samples:
            X_hist = np.array([[h[0]] for h in self.history])
            y_hist = np.array([h[1] for h in self.history])

            predictions = self.model.predict(X_hist)
            errors = predictions - y_hist
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors ** 2).mean())

            return {
                'fitted': True,
                'n_samples': len(self.history),
                'mae': float(mae),
                'rmse': float(rmse),
                'mean_actual_delta_mpl': float(y_hist.mean()),
                'std_actual_delta_mpl': float(y_hist.std()),
                'model_coef': float(self.model.coef_[0]) if hasattr(self.model, 'coef_') else None,
                'model_intercept': float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None,
            }
        else:
            return {
                'fitted': True,
                'n_samples': len(self.history),
                'min_samples_needed': self.min_samples
            }
