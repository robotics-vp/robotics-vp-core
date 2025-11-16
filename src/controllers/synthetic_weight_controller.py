import numpy as np
from typing import Dict, Optional


class SyntheticWeightController:
    """
    Centralized synthetic weighting logic:
    - Quality gate: trust × w_econ (or trust-only)
    - Budget control: scale to hit λ/target share with safety caps
    """

    def __init__(
        self,
        max_synth_share: float = 0.4,
        econ_weight_cap: float = 1.0,
        trust_floor: float = 0.0,
        default_lambda: float = 0.2,
    ):
        self.max_synth_share = max_synth_share
        self.econ_weight_cap = econ_weight_cap
        self.trust_floor = trust_floor
        self.default_lambda = default_lambda

    def compute_weights(
        self,
        trust: np.ndarray,
        econ: Optional[np.ndarray],
        n_real: int,
        mode: str = "trust_econ_lambda",
        lambda_target: Optional[float] = None,
    ) -> Dict[str, object]:
        trust = np.asarray(trust, dtype=np.float32)
        econ = np.ones_like(trust, dtype=np.float32) if econ is None else np.asarray(econ, dtype=np.float32)
        econ = np.clip(econ, 0.0, self.econ_weight_cap)

        # Build quality term
        if mode == "baseline":
            quality = np.zeros_like(trust)
        elif mode == "trust_only":
            quality = trust.copy()
        elif mode in ("trust_econ", "trust_econ_lambda"):
            quality = trust * econ
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Trust floor
        quality = np.where(trust >= self.trust_floor, quality, 0.0)

        # Budget target
        target = lambda_target if lambda_target is not None else self.default_lambda
        target = float(max(0.0, min(target, self.max_synth_share)))

        if n_real <= 0 or quality.sum() <= 0 or target <= 0:
            scaled = quality
            scale = 1.0
        else:
            desired_synth_total = n_real * target / (1 - target + 1e-6)
            scale = desired_synth_total / (quality.sum() + 1e-6)
            scaled = quality * scale

        real_weight_sum = float(n_real)
        synth_weight_sum = float(scaled.sum())
        effective_share = synth_weight_sum / (real_weight_sum + synth_weight_sum + 1e-6)

        debug = {
            "mode": mode,
            "lambda_target": target,
            "scale_factor": scale,
            "quality_mean": float(quality.mean()) if quality.size else 0.0,
            "quality_max": float(quality.max()) if quality.size else 0.0,
            "effective_synth_share": effective_share,
            "max_synth_share": self.max_synth_share,
        }

        return {
            "weights": scaled,
            "quality": quality,
            "trust": trust,
            "econ": econ,
            "debug": debug,
        }
