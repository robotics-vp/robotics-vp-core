"""
ConditionedVisionAdapter: Vision modulation based on ConditionVector.

Per CONDITIONED_VISION_ADAPTER_SEMANTICS.md:
- FiLM modulation of visual features
- Attention reweighting in BiFPN fusion
- RNN gating for temporal smoothing
- Risk/affordance map scaling

Deterministic, bounded, flag-gated.
"""
from typing import Any, Dict, Optional

import numpy as np

from src.vision.interfaces import VisionFrame
from src.vision.regnet_backbone import build_regnet_feature_pyramid, DEFAULT_LEVELS
from src.vision.bifpn_fusion import fuse_feature_pyramid
from src.observation.condition_vector import ConditionVector


class ConditionedVisionAdapter:
    """
    Vision encoder modulated by ConditionVector.

    Architecture:
    1. RegNet backbone → multi-scale features (z_v: base representation, unchanged)
    2. FiLM conditioning → feature modulation
    3. BiFPN → multi-scale fusion (condition-weighted)
    4. Risk/Affordance heads → semantic maps (condition-scaled)

    Invariants:
    - z_v (base representation) identical for same input, regardless of ConditionVector
    - Deterministic: same (image, ConditionVector) → same output
    - Bounded: all scales clamped to [0.1, 10.0]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize conditioned vision adapter.

        Args:
            config: Optional config dict with:
                - feature_dim: Dimension of feature vectors (default 8)
                - levels: Pyramid levels (default ["P3", "P4", "P5"])
                - enable_conditioning: Flag to enable/disable conditioning (default True)
        """
        self.config = config or {}
        self.feature_dim = self.config.get("feature_dim", 8)
        self.levels = list(self.config.get("levels", DEFAULT_LEVELS))
        self.enable_conditioning = bool(self.config.get("enable_conditioning", True))

        # Bounds for safety
        self.min_scale = 0.1
        self.max_scale = 10.0

    def forward(
        self,
        frame: VisionFrame,
        condition_vector: Optional[ConditionVector] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass with optional conditioning.

        Args:
            frame: Input vision frame
            condition_vector: Optional ConditionVector for modulation

        Returns:
            Dict with:
                - z_v: Base representation (unchanged)
                - features_modulated: FiLM-modulated features
                - fused_features: BiFPN-fused features
                - risk_map: Risk saliency map (condition-scaled)
                - affordance_map: Affordance map
        """
        # 1. Base encode (unchanged by condition)
        z_v = build_regnet_feature_pyramid(frame, feature_dim=self.feature_dim, levels=self.levels)

        # If conditioning disabled or no condition vector, return base features
        if not self.enable_conditioning or condition_vector is None:
            fused = fuse_feature_pyramid(z_v)
            return {
                "z_v": z_v,
                "features_modulated": z_v,
                "fused_features": fused,
                "risk_map": self._default_risk_map(),
                "affordance_map": self._default_affordance_map(),
            }

        # 2. Apply FiLM modulation
        features_modulated = self._apply_film_modulation(z_v, condition_vector)

        # 3. Compute condition-dependent fusion weights
        fusion_weights = self._compute_fusion_weights(condition_vector)

        # 4. BiFPN fusion
        fused_features = fuse_feature_pyramid(features_modulated, weights=fusion_weights)

        # 5. Semantic heads with condition-scaled outputs
        risk_map = self._modulate_risk_map(self._default_risk_map(), condition_vector)
        affordance_map = self._default_affordance_map()

        return {
            "z_v": z_v,  # Base representation (invariant)
            "features_modulated": features_modulated,
            "fused_features": fused_features,
            "risk_map": risk_map,
            "affordance_map": affordance_map,
        }

    def _apply_film_modulation(
        self,
        features: Dict[str, np.ndarray],
        condition_vector: ConditionVector,
    ) -> Dict[str, np.ndarray]:
        """
        Apply FiLM (Feature-wise Linear Modulation) to features.

        gamma (scale) and beta (shift) derived from condition vector.
        """
        modulated = {}

        # Derive FiLM parameters from condition vector
        gamma, beta = self._compute_film_params(condition_vector)

        for level, feat in features.items():
            # Apply: features_out = gamma * features + beta
            # Clamp gamma to prevent explosion/collapse
            gamma_clamped = np.clip(gamma, self.min_scale, self.max_scale)

            modulated_feat = gamma_clamped * feat + beta
            modulated[level] = modulated_feat.astype(np.float32)

        return modulated

    def _compute_film_params(self, condition_vector: ConditionVector) -> tuple[float, float]:
        """
        Compute FiLM gamma (scale) and beta (shift) from ConditionVector.

        Logic:
        - Safety mode: gamma ~ 1.0 (preserve features), beta ~ 0
        - Exploration mode: gamma > 1.0 (amplify), beta ~ 0
        - Efficiency mode: gamma < 1.0 (dampen), beta ~ 0
        """
        skill_mode = condition_vector.skill_mode or "efficiency_throughput"

        gamma = 1.0
        beta = 0.0

        if "safety" in skill_mode.lower():
            gamma = 1.3  # Amplify for detail
        elif "exploration" in skill_mode.lower() or "frontier" in skill_mode.lower():
            gamma = 1.5  # Amplify for novelty
        elif "efficiency" in skill_mode.lower():
            gamma = 0.7  # Dampen to save compute

        return gamma, beta

    def _compute_fusion_weights(self, condition_vector: ConditionVector) -> Dict[str, float]:
        """
        Compute BiFPN fusion weights based on ConditionVector.

        Logic:
        - Precision mode: Boost high-resolution (fine) features
        - Speed mode: Boost low-resolution (coarse) features
        - Balanced otherwise
        """
        skill_mode = condition_vector.skill_mode or "efficiency_throughput"

        weights = {level: 1.0 for level in self.levels}

        if "precision" in skill_mode.lower() or "safety" in skill_mode.lower():
            # Amplify finest level (highest resolution)
            if len(self.levels) > 0:
                weights[self.levels[-1]] = 2.0
        elif "speed" in skill_mode.lower() or "efficiency" in skill_mode.lower():
            # Amplify coarsest level (lowest resolution)
            if len(self.levels) > 0:
                weights[self.levels[0]] = 2.0

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _modulate_risk_map(self, base_risk: np.ndarray, condition_vector: ConditionVector) -> np.ndarray:
        """
        Modulate risk map based on ConditionVector.

        Logic:
        - Low risk_tolerance → amplify risks
        - High risk_tolerance → dampen risks
        """
        # Default: no modulation
        risk_tolerance = getattr(condition_vector, "ood_risk_level", None)
        if risk_tolerance is None:
            # Fallback: use skill_mode heuristic
            skill_mode = condition_vector.skill_mode or ""
            if "safety" in skill_mode.lower():
                risk_tolerance = 0.2  # Low tolerance (amplify risks)
            else:
                risk_tolerance = 0.5  # Neutral

        # Compute scale
        scale = 1.0 - float(risk_tolerance) + 0.5
        scale = np.clip(scale, self.min_scale, self.max_scale)

        return (base_risk * scale).astype(np.float32)

    def _default_risk_map(self) -> np.ndarray:
        """Return default risk map (stub)."""
        return np.ones(self.feature_dim, dtype=np.float32) * 0.5

    def _default_affordance_map(self) -> np.ndarray:
        """Return default affordance map (stub)."""
        return np.ones(self.feature_dim, dtype=np.float32) * 0.5


def build_conditioned_vision_features(
    frame: VisionFrame,
    condition_vector: Optional[ConditionVector] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to build conditioned vision features.

    Args:
        frame: VisionFrame input
        condition_vector: Optional ConditionVector for modulation
        config: Optional adapter config

    Returns:
        Dict with z_v, modulated features, fused features, and semantic maps
    """
    adapter = ConditionedVisionAdapter(config=config)
    return adapter.forward(frame, condition_vector)
